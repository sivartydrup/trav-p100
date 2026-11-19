//! Typed launchers for the sm_60 kernels in `cu/kernels.cu`.

use crate::driver::{CUdeviceptr, CuError};
use crate::{launch, Buf, Device, Function, LaunchArgs, Module, Stream, CUDA_SUCCESS};
use crate::driver::Driver;

pub const QK_K: usize = 256;

/// Repack a GGUF Q6_K tensor (210-byte blocks) into the 224-byte layout the
/// kernel consumes:
///
/// ```text
///   0  : f16 d
///   4  : i8  sc2[16]    sc2[2t], sc2[2t+1] = scales of lane t's two sub-blocks
///   32 : u8  ql[128]    identical to GGUF
///   160: u8  hi[64]     8 runs x 8 bytes, four 2-bit high parts per byte
/// ```
///
/// The high-bit plane is the only permutation: the kernel needs 16 values of
/// 2-bit fields as one 32-bit word so `spread2` can raise them to bytes with a
/// multiply and a mask instead of 16 shifts (shifts are half rate on GP100).
pub fn repack_q6k(src: &[u8]) -> Vec<u8> {
    assert_eq!(src.len() % 210, 0, "Q6_K tensor size");
    let nb = src.len() / 210;
    let mut out = vec![0u8; nb * 224];
    for b in 0..nb {
        let s = &src[b * 210..b * 210 + 210];
        let o = &mut out[b * 224..b * 224 + 224];
        o[0..2].copy_from_slice(&s[208..210]); // d
        // ql
        o[32..160].copy_from_slice(&s[0..128]);
        // high-bit plane: run r covers values [32r, 32r+32)
        for r in 0..8usize {
            let n = r / 4;
            let rr = r % 4;
            let ql_off = 64 * n + 32 * (rr & 1);
            let high = rr >= 2;
            let qh_off = 32 * n;
            let bits = 2 * rr;
            for j in 0..32usize {
                let nib = if high {
                    s[ql_off + j] >> 4
                } else {
                    s[ql_off + j] & 0xF
                };
                let hb = (s[128 + qh_off + j] >> bits) & 3;
                let q = nib | (hb << 4);
                o[160 + r * 8 + j / 4] |= (q >> 4) << (2 * (j % 4));
            }
        }
        // per-lane scale pairs
        for t in 0..8usize {
            let jj = t / 2;
            let hf = t % 2;
            let s_lo = 2 * jj + 4 * (jj / 2) + hf;
            o[4 + 2 * t] = s[192 + s_lo];
            o[4 + 2 * t + 1] = s[192 + s_lo + 4];
        }
    }
    out
}

pub struct Kernels {
    _module: Module,
    pub prep_act: Function,
    pub rms_norm: Function,
    pub add_rms_norm: Function,
    pub gemv_q4k: [Function; 5],
    pub gemv_q6k: [Function; 5],
    pub rope: Function,
    pub write_kv: Function,
    pub attn_decode: Function,
    pub attn_combine: Function,
    pub attn_prefill: Function,
    pub silu_mul: Function,
    pub add_inplace: Function,
    pub scale: Function,
    pub embed_q4k: Function,
    pub test_add: Function,
    pub bw_read: Function,
}

fn need(f: Result<Function, CuError>) -> Function {
    f.expect("kernel missing from cubin")
}

impl Kernels {
    pub fn load(dev: &Device) -> crate::Result<Kernels> {
        let module = dev.module(crate::cubin())?;
        let k = |n: &str| need(module.kernel(n));
        Ok(Kernels {
            prep_act: k("prep_act"),
            rms_norm: k("rms_norm"),
            add_rms_norm: k("add_rms_norm"),
            gemv_q4k: [
                k("gemv_q4k_c1"),
                k("gemv_q4k_c2"),
                k("gemv_q4k_c4"),
                k("gemv_q4k_c8"),
                k("gemv_q4k_r2c8"),
            ],
            gemv_q6k: [
                k("gemv_q6k_c1"),
                k("gemv_q6k_c2"),
                k("gemv_q6k_c4"),
                k("gemv_q6k_c8"),
                k("gemv_q6k_r2c8"),
            ],
            rope: k("rope_neox"),
            write_kv: k("write_kv"),
            attn_decode: k("attn_decode"),
            attn_combine: k("attn_combine"),
            attn_prefill: k("attn_prefill"),
            silu_mul: k("silu_mul"),
            add_inplace: k("add_inplace"),
            scale: k("scale_f32"),
            embed_q4k: k("embed_q4k"),
            test_add: k("k_test_add"),
            bw_read: k("k_bw_read"),
            _module: module,
        })
    }

    fn cols_idx(cols: usize) -> usize {
        match cols {
            1 => 0,
            2 => 1,
            4 => 2,
            8 => 3,
            _ => panic!("unsupported column tile {cols} (1,2,4,8)"),
        }
    }

    /// `out[c][row] = sum_k W[row][k] * Y[c][k] + bias[row]`
    #[allow(clippy::too_many_arguments)]
    pub fn gemv(
        &self,
        q6: bool,
        cols: usize,
        dev: &Device,
        w: &Buf,
        row_stride: usize,
        y: &Buf,
        y_stride: usize,
        ys: &Buf,
        ys_stride: usize,
        bias: Option<&Buf>,
        out: &Buf,
        out_stride: usize,
        nrows: usize,
        k: usize,
        stream: Option<&Stream>,
    ) -> crate::Result<()> {
        // Eight columns with two rows per warp: activation staging and shared
        // bandwidth are amortised over twice the MACs, which is what limits the
        // prefill path.
        let (f, rows_per_warp): (&Function, usize) = if cols == 8 && nrows >= 16 {
            (if q6 { &self.gemv_q6k[4] } else { &self.gemv_q4k[4] }, 2)
        } else if q6 {
            (&self.gemv_q6k[Self::cols_idx(cols)], 1)
        } else {
            (&self.gemv_q4k[Self::cols_idx(cols)], 1)
        };
        let block = 256u32;
        let warps = 8usize;
        let per_block = warps * rows_per_warp;
        let grid = (((nrows + per_block - 1) / per_block) as u32, 1, 1);
        let mut a_w = w.as_ptr();
        let mut a_rs = row_stride as i64;
        let mut a_y = y.as_ptr();
        let mut a_ysd = y_stride as i64;
        let mut a_ys = ys.as_ptr();
        let mut a_yss = ys_stride as i64;
        let mut a_b = bias.map(|b| b.as_ptr()).unwrap_or(0);
        let mut a_o = out.as_ptr();
        let mut a_ost = out_stride as i64;
        let mut a_n = nrows as i32;
        let mut a_k = k as i32;
        let mut args = LaunchArgs::new();
        args.push(&mut a_w)
            .push(&mut a_rs)
            .push(&mut a_y)
            .push(&mut a_ysd)
            .push(&mut a_ys)
            .push(&mut a_yss)
            .push(&mut a_b)
            .push(&mut a_o)
            .push(&mut a_ost)
            .push(&mut a_n)
            .push(&mut a_k);
        // Must match CSB in the kernel: 8 super-blocks for <=2 columns, else 4.
        let csb = if cols * rows_per_warp <= 4 { 8 } else { 4 };
        let smem = (cols * csb * 256 * 2 + cols * csb * 16 * 8) as u32;
        dev.set_current();
        launch(f, grid, (block, 1, 1), smem, stream, &mut args)
    }

    pub fn prep(
        &self,
        dev: &Device,
        x: &Buf,
        y: &Buf,
        ys: &Buf,
        ncols: usize,
        k: usize,
        stream: Option<&Stream>,
    ) -> crate::Result<()> {
        let warps_total = ncols * (k / 32);
        let mut a_x = x.as_ptr();
        let mut a_y = y.as_ptr();
        let mut a_ys = ys.as_ptr();
        let mut a_c = ncols as i32;
        let mut a_k = k as i32;
        let mut args = LaunchArgs::new();
        args.push(&mut a_x).push(&mut a_y).push(&mut a_ys).push(&mut a_c).push(&mut a_k);
        dev.set_current();
        let grid = ((warps_total as u32 + 7) / 8, 1, 1);
        launch(&self.prep_act, grid, (256, 1, 1), 0, stream, &mut args)
    }

    pub fn rms_norm(
        &self,
        dev: &Device,
        x: &Buf,
        w: &Buf,
        out: &Buf,
        rows: usize,
        n: usize,
        eps: f32,
        stream: Option<&Stream>,
    ) -> crate::Result<()> {
        let mut a_x = x.as_ptr();
        let mut a_w = w.as_ptr();
        let mut a_o = out.as_ptr();
        let mut a_n = n as i32;
        let mut a_e = eps;
        let mut args = LaunchArgs::new();
        args.push(&mut a_x).push(&mut a_w).push(&mut a_o).push(&mut a_n).push(&mut a_e);
        dev.set_current();
        launch(&self.rms_norm, (rows as u32, 1, 1), (256, 1, 1), 0, stream, &mut args)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_rms_norm(
        &self,
        dev: &Device,
        x: &Buf,
        delta: &Buf,
        w: &Buf,
        out: &Buf,
        rows: usize,
        n: usize,
        eps: f32,
        stream: Option<&Stream>,
    ) -> crate::Result<()> {
        let mut a_x = x.as_ptr();
        let mut a_d = delta.as_ptr();
        let mut a_w = w.as_ptr();
        let mut a_o = out.as_ptr();
        let mut a_n = n as i32;
        let mut a_e = eps;
        let mut args = LaunchArgs::new();
        args.push(&mut a_x).push(&mut a_d).push(&mut a_w).push(&mut a_o).push(&mut a_n).push(&mut a_e);
        dev.set_current();
        launch(&self.add_rms_norm, (rows as u32, 1, 1), (256, 1, 1), 0, stream, &mut args)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        dev: &Device,
        q: &Buf,
        k: &Buf,
        pos: &Buf,
        inv_freq: &Buf,
        n_q_heads: usize,
        n_k_heads: usize,
        head_dim: usize,
        n_q: usize,
        k_off: usize,
        stream: Option<&Stream>,
    ) -> crate::Result<()> {
        let total = n_q * (n_q_heads * head_dim / 2 + n_k_heads * head_dim / 2);
        let mut a_q = q.as_ptr();
        let mut a_k = k.as_ptr();
        let mut a_p = pos.as_ptr();
        let mut a_if = inv_freq.as_ptr();
        let mut a_nq = n_q_heads as i32;
        let mut a_nk = n_k_heads as i32;
        let mut a_hd = head_dim as i32;
        let mut a_nt = n_q as i32;
        let mut a_off = k_off as i32;
        let mut args = LaunchArgs::new();
        args.push(&mut a_q)
            .push(&mut a_k)
            .push(&mut a_p)
            .push(&mut a_if)
            .push(&mut a_nq)
            .push(&mut a_nk)
            .push(&mut a_hd)
            .push(&mut a_nt)
            .push(&mut a_off);
        dev.set_current();
        let grid = ((total as u32 + 255) / 256, 1, 1);
        launch(&self.rope, grid, (256, 1, 1), 0, stream, &mut args)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn write_kv(
        &self,
        dev: &Device,
        kc: &Buf,
        vc: &Buf,
        k: &Buf,
        v: &Buf,
        n_kv: usize,
        head_dim: usize,
        start_pos: usize,
        n: usize,
        stream: Option<&Stream>,
    ) -> crate::Result<()> {
        let total = n * n_kv * head_dim;
        let mut a_kc = kc.as_ptr();
        let mut a_vc = vc.as_ptr();
        let mut a_k = k.as_ptr();
        let mut a_v = v.as_ptr();
        let mut a_nkv = n_kv as i32;
        let mut a_hd = head_dim as i32;
        let mut a_sp = start_pos as i32;
        let mut a_n = n as i32;
        let mut args = LaunchArgs::new();
        args.push(&mut a_kc)
            .push(&mut a_vc)
            .push(&mut a_k)
            .push(&mut a_v)
            .push(&mut a_nkv)
            .push(&mut a_hd)
            .push(&mut a_sp)
            .push(&mut a_n);
        dev.set_current();
        let grid = ((total as u32 + 255) / 256, 1, 1);
        launch(&self.write_kv, grid, (256, 1, 1), 0, stream, &mut args)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode(
        &self,
        dev: &Device,
        kc: &Buf,
        vc: &Buf,
        q: &Buf,
        out: &Buf,
        part_m: &Buf,
        part_l: &Buf,
        part_acc: &Buf,
        n_heads: usize,
        n_kv: usize,
        head_dim: usize,
        n_ctx: usize,
        n_split: usize,
        qpk: usize,
        stream: Option<&Stream>,
    ) -> crate::Result<()> {
        assert!(qpk <= 7, "decode attention block covers at most 7 query heads");
        let mut a_kc = kc.as_ptr();
        let mut a_vc = vc.as_ptr();
        let mut a_q = q.as_ptr();
        let mut a_o = out.as_ptr();
        let mut a_pm = part_m.as_ptr();
        let mut a_pl = part_l.as_ptr();
        let mut a_pa = part_acc.as_ptr();
        let mut a_nh = n_heads as i32;
        let mut a_nkv = n_kv as i32;
        let mut a_hd = head_dim as i32;
        let mut a_nc = n_ctx as i32;
        let mut a_ns = n_split as i32;
        let mut a_qpk = qpk as i32;
        let mut args = LaunchArgs::new();
        args.push(&mut a_kc)
            .push(&mut a_vc)
            .push(&mut a_q)
            .push(&mut a_o)
            .push(&mut a_pm)
            .push(&mut a_pl)
            .push(&mut a_pa)
            .push(&mut a_nh)
            .push(&mut a_nkv)
            .push(&mut a_hd)
            .push(&mut a_nc)
            .push(&mut a_ns)
            .push(&mut a_qpk);
        dev.set_current();
        launch(
            &self.attn_decode,
            (n_kv as u32, n_split as u32, 1),
            (256, 1, 1),
            0,
            stream,
            &mut args,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attn_combine(
        &self,
        dev: &Device,
        part_m: &Buf,
        part_l: &Buf,
        part_acc: &Buf,
        out: &Buf,
        n_heads: usize,
        head_dim: usize,
        n_split: usize,
        stream: Option<&Stream>,
    ) -> crate::Result<()> {
        let mut a_pm = part_m.as_ptr();
        let mut a_pl = part_l.as_ptr();
        let mut a_pa = part_acc.as_ptr();
        let mut a_o = out.as_ptr();
        let mut a_nh = n_heads as i32;
        let mut a_hd = head_dim as i32;
        let mut a_ns = n_split as i32;
        let mut args = LaunchArgs::new();
        args.push(&mut a_pm)
            .push(&mut a_pl)
            .push(&mut a_pa)
            .push(&mut a_o)
            .push(&mut a_nh)
            .push(&mut a_hd)
            .push(&mut a_ns);
        dev.set_current();
        launch(&self.attn_combine, (n_heads as u32, 1, 1), (128, 1, 1), 0, stream, &mut args)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill(
        &self,
        dev: &Device,
        kc: &Buf,
        vc: &Buf,
        q: &Buf,
        out: &Buf,
        n_heads: usize,
        n_kv: usize,
        head_dim: usize,
        n_ctx: usize,
        qpk: usize,
        q_pos0: usize,
        nq: usize,
        stream: Option<&Stream>,
    ) -> crate::Result<()> {
        assert!(nq <= 8, "prefill attention handles at most 8 queries per block");
        let mut a_kc = kc.as_ptr();
        let mut a_vc = vc.as_ptr();
        let mut a_q = q.as_ptr();
        let mut a_o = out.as_ptr();
        let mut a_nh = n_heads as i32;
        let mut a_nkv = n_kv as i32;
        let mut a_hd = head_dim as i32;
        let mut a_nc = n_ctx as i32;
        let mut a_qpk = qpk as i32;
        let mut a_p0 = q_pos0 as i32;
        let mut a_nq = nq as i32;
        let mut args = LaunchArgs::new();
        args.push(&mut a_kc)
            .push(&mut a_vc)
            .push(&mut a_q)
            .push(&mut a_o)
            .push(&mut a_nh)
            .push(&mut a_nkv)
            .push(&mut a_hd)
            .push(&mut a_nc)
            .push(&mut a_qpk)
            .push(&mut a_p0)
            .push(&mut a_nq);
        dev.set_current();
        launch(&self.attn_prefill, (n_heads as u32, 1, 1), (256, 1, 1), 0, stream, &mut args)
    }

    pub fn silu_mul(
        &self,
        dev: &Device,
        out: &Buf,
        gate: &Buf,
        up: &Buf,
        n: usize,
        stream: Option<&Stream>,
    ) -> crate::Result<()> {
        let mut a_o = out.as_ptr();
        let mut a_g = gate.as_ptr();
        let mut a_u = up.as_ptr();
        let mut a_n = n as i32;
        let mut args = LaunchArgs::new();
        args.push(&mut a_o).push(&mut a_g).push(&mut a_u).push(&mut a_n);
        dev.set_current();
        launch(&self.silu_mul, (((n as u32) + 255) / 256, 1, 1), (256, 1, 1), 0, stream, &mut args)
    }

    pub fn add_inplace(
        &self,
        dev: &Device,
        x: &Buf,
        y: &Buf,
        n: usize,
        stream: Option<&Stream>,
    ) -> crate::Result<()> {
        let mut a_x = x.as_ptr();
        let mut a_y = y.as_ptr();
        let mut a_n = n as i32;
        let mut args = LaunchArgs::new();
        args.push(&mut a_x).push(&mut a_y).push(&mut a_n);
        dev.set_current();
        launch(&self.add_inplace, (((n as u32) + 255) / 256, 1, 1), (256, 1, 1), 0, stream, &mut args)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn embed(
        &self,
        dev: &Device,
        w: &Buf,
        row_stride: usize,
        ids: &Buf,
        out: &Buf,
        n: usize,
        k: usize,
        stream: Option<&Stream>,
    ) -> crate::Result<()> {
        let nsb = k / 256;
        let warps = n * nsb;
        let mut a_w = w.as_ptr();
        let mut a_rs = row_stride as i64;
        let mut a_i = ids.as_ptr();
        let mut a_o = out.as_ptr();
        let mut a_n = n as i32;
        let mut a_k = k as i32;
        let mut args = LaunchArgs::new();
        args.push(&mut a_w).push(&mut a_rs).push(&mut a_i).push(&mut a_o).push(&mut a_n).push(&mut a_k);
        dev.set_current();
        let grid = ((warps as u32 + 7) / 8, 1, 1);
        launch(&self.embed_q4k, grid, (256, 1, 1), 0, stream, &mut args)
    }

    pub fn test_add(
        &self,
        dev: &Device,
        a: &Buf,
        b: &Buf,
        c: &Buf,
        n: usize,
    ) -> crate::Result<()> {
        let mut pa = a.as_ptr();
        let mut pb = b.as_ptr();
        let mut pc = c.as_ptr();
        let mut pn = n as i32;
        let mut args = LaunchArgs::new();
        args.push(&mut pa).push(&mut pb).push(&mut pc).push(&mut pn);
        dev.set_current();
        launch(&self.test_add, (((n as u32) + 255) / 256, 1, 1), (256, 1, 1), 0, None, &mut args)
    }

    pub fn bw_read(
        &self,
        dev: &Device,
        p: &Buf,
        n4: i64,
        out: &Buf,
        blocks: u32,
    ) -> crate::Result<()> {
        let mut a_p = p.as_ptr();
        let mut a_n = n4;
        let mut a_o = out.as_ptr();
        let mut args = LaunchArgs::new();
        args.push(&mut a_p).push(&mut a_n).push(&mut a_o);
        dev.set_current();
        launch(&self.bw_read, (blocks, 1, 1), (256, 1, 1), 0, None, &mut args)
    }
}

/// Check the last launch on this context.
pub fn last_error(dev: &Device) -> crate::Result<()> {
    let drv = Driver::get()?;
    dev.set_current();
    let rc = unsafe { (drv.ctx_synchronize)() };
    if rc == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(drv.error(rc))
    }
}

/// Raw pointer helper used by the engine for partition buffers.
pub fn ptr_of(b: &Buf) -> CUdeviceptr {
    b.as_ptr()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spread2_cpu(v: u32) -> u32 {
        let lo = ((v & 0xF).wrapping_mul(0x41)) & 0x0303;
        let hi = ((((v >> 4).wrapping_mul(0x41)) & 0x0303) << 16) & 0xFFFF_FFFF;
        lo | hi
    }

    #[test]
    fn repack_q6k_layout_matches_reference() {
        // deterministic pseudo-random block, valid: d small, scales in [-32,31]
        let mut s = vec![0u8; 210];
        let mut x: u64 = 0x1234_5678_9abc_def0;
        for b in s.iter_mut() {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (x >> 33) as u8;
        }
        s[208..210].copy_from_slice(&half::f16::from_f32(0.01).to_bits().to_le_bytes());
        for i in 0..16 {
            s[192 + i] = ((x >> (i % 60)) & 0x3F) as u8;
        }
        let rep = repack_q6k(&s);
        let mut want = vec![0f32; 256];
        trav_gguf::quant::dequantize_row(trav_gguf::quant::GgmlType::Q6K, &s, &mut want);
        let d = half::f16::from_bits(u16::from_le_bytes([rep[0], rep[1]])).to_f32();

        let mut bad = 0;
        for t in 0..8usize {
            let jj = t / 2;
            let hf = t % 2;
            let off_lo = 32 * jj + 64 * (jj / 2) + 16 * hf;
            let run = jj + 2 * (jj / 2);
            let sc_lo = rep[4 + 2 * t] as i8 as f32;
            let sc_hi = rep[4 + 2 * t + 1] as i8 as f32;
            let rd = |o: usize| u32::from_le_bytes(rep[o..o + 4].try_into().unwrap());
            let h0 = rd(160 + run * 8 + 4 * hf);
            let h1 = rd(160 + (run + 2) * 8 + 4 * hf);
            for wi in 0..4 {
                let w = rd(32 + 32 * jj + 16 * hf + 4 * wi);
                let lo = w & 0x0F0F_0F0F;
                let hi = (w >> 4) & 0x0F0F_0F0F;
                let hbl = spread2_cpu((h0 >> (8 * wi)) & 0xFF);
                let hbh = spread2_cpu((h1 >> (8 * wi)) & 0xFF);
                let vl = lo | (hbl << 4);
                let vh = hi | (hbh << 4);
                for b in 0..4usize {
                    let ql = ((vl >> (8 * b)) & 0xFF) as f32;
                    let qh = ((vh >> (8 * b)) & 0xFF) as f32;
                    let gl = d * sc_lo * (ql - 32.0);
                    let gh = d * sc_hi * (qh - 32.0);
                    let i = off_lo + 4 * wi + b;
                    if (gl - want[i]).abs() > 1e-4 || (gh - want[i + 64]).abs() > 1e-4 {
                        if bad < 8 {
                            println!("t={t} off_lo={off_lo} i={i} gl={gl} want={} ql={ql} sc_lo={sc_lo} | gh={gh} want32={} qh={qh} sc_hi={sc_hi}",
                                want[i], want[i + 32]);
                        }
                        bad += 1;
                    }
                }
            }
        }
        assert_eq!(bad, 0, "{bad} mismatched values");
    }
}

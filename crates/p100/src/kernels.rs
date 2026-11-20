//! Typed launchers for the sm_60 kernels in `cu/kernels.cu`.

use crate::driver::Driver;
use crate::driver::{CUdeviceptr, CuError};
use crate::{launch, Buf, Device, Function, LaunchArgs, Module, Stream, CUDA_SUCCESS};

pub const QK_K: usize = 256;

/// Repack a GGUF Q6_K tensor (210-byte blocks) into the 224-byte layout the
/// kernel consumes:
///
/// ```text
///   0  : f16 d
///   4  : i8  sc2[16]    sc2[2t], sc2[2t+1] = scales of lane t's two sub-blocks
///   32 : u8  ql[128]    identical to GGUF
///   160: u8  hi[64]     4 pairs x 16 bytes: a lane's two runs, halves
///                       interleaved as [lo.hf0, hi.hf0, lo.hf1, hi.hf1]
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
        // high-bit plane: run r covers values [32r, 32r+32). A lane owns runs
        // (r, r+2) and reads only one half of each, so the four halves of that
        // pair are interleaved as [lo.hf0, hi.hf0, lo.hf1, hi.hf1]: one 8-byte
        // load then fetches both of the lane's runs instead of two gathers.
        for r in 0..8usize {
            let pair = ((r >> 2) << 1) | (r & 1);
            let is_hi = (r >> 1) & 1;
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
                let byte = 160 + pair * 16 + (j / 16) * 8 + is_hi * 4 + (j % 16) / 4;
                o[byte] |= (q >> 4) << (2 * (j % 4));
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

/// Split-K choice for the prefill GEMM. Splitting k costs a zeroed output and
/// atomic accumulates, but it is the only way a short row count (k_proj is 512
/// rows) or an awkward block count fills all 112 resident slots. Pick the
/// smallest split that gets close to a full last wave.
fn pick_split(blocks: usize, slots: usize, max_split: usize) -> usize {
    let mut best = (1usize, f64::MAX);
    for ns in 1..=max_split.clamp(1, 64) {
        let total = blocks * ns;
        if total > slots * 8 {
            break;
        }
        let waves = total.div_ceil(slots) as f64;
        let eff = total as f64 / (waves * slots as f64);
        let cost = (1.0 + 0.02 * (ns - 1) as f64) / eff;
        if cost < best.1 {
            best = (ns, cost);
        }
    }
    best.0
}

pub struct Kernels {
    _module: Module,
    pub prep_act: Function,
    pub rms_norm: Function,
    pub add_rms_norm: Function,
    pub gemv_q4k: [Function; 8],
    pub gemv_q6k: [Function; 8],
    /// Fused dequantise+GEMM, used where a whole-prompt fp16 mirror does not
    /// pay (small matrices, split-K shapes).
    pub gemm_q4k_n64t128: Function,
    pub gemm_q4k_n32t128: Function,
    pub gemm_q6k_n32t128: Function,
    pub dequant_q4k_f16: Function,
    pub dequant_q6k_f16: Function,
    pub gemm_f16: Function,
    pub prep_col: Function,
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
    pub bw_chunk: Function,
    pub bw_rows4: Function,
    pub hfma2: Function,
}

fn need(f: Result<Function, CuError>) -> Function {
    f.expect("kernel missing from cubin")
}

impl Kernels {
    pub fn load(dev: &Device) -> crate::Result<Kernels> {
        let module = dev.module(crate::cubin())?;
        let k = |n: &str| need(module.kernel(n));
        let gemm_q4k_n64t128 = k("gemm_q4k_n64t128");
        let gemm_q4k_n32t128 = k("gemm_q4k_n32t128");
        let gemm_q6k_n32t128 = k("gemm_q6k_n32t128");
        let dequant_q4k_f16 = k("dequant_q4k_f16");
        let dequant_q6k_f16 = k("dequant_q6k_f16");
        let gemm_f16 = k("gemm_f16_128");
        gemm_f16.set_carveout(100).ok();
        // Two GEMM blocks (29.7 / 31.7 KiB) should stay resident per SM; ask
        // for the full 64 KiB shared carveout and the 48 KiB dynamic limit.
        for f in [&gemm_q4k_n64t128, &gemm_q4k_n32t128, &gemm_q6k_n32t128] {
            module.set_max_smem(f, 48 * 1024).ok();
            f.set_carveout(100).ok();
        }
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
                k("gemv_q4k_r2c1"),
                k("gemv_q4k_r4c1"),
                k("gemv_q4k_r8c1"),
            ],
            gemv_q6k: [
                k("gemv_q6k_c1"),
                k("gemv_q6k_c2"),
                k("gemv_q6k_c4"),
                k("gemv_q6k_c8"),
                k("gemv_q6k_r2c8"),
                k("gemv_q6k_r2c1"),
                k("gemv_q6k_r4c1"),
                k("gemv_q6k_r8c1"),
            ],
            gemm_q4k_n64t128,
            gemm_q4k_n32t128,
            gemm_q6k_n32t128,
            dequant_q4k_f16,
            dequant_q6k_f16,
            gemm_f16,
            prep_col: k("prep_act_col"),
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
            bw_chunk: k("k_bw_chunk"),
            bw_rows4: k("k_bw_rows4"),
            hfma2: k("k_hfma2"),
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
        // One column dominates decode: each block stages the whole activation
        // row but covers only `8 * rows` weight rows, so more rows per warp
        // amortises that staging over more weight bytes. Eight columns keep the
        // two-row kernel because there the MACs, not the staging, are the limit.
        // A block stages its whole activation row from global, so few rows per
        // warp means that staging is re-read once per 8 or 16 weight rows. More
        // rows per warp amortises it but leaves the card short of blocks, which
        // is what split-K fixes: splitting the k range doubles the block count
        // at the cost of an atomic accumulate.
        let rows_per_warp = if cols == 1 {
            Self::rows_for(nrows, q6)
        } else if cols == 8 && nrows >= 16 {
            2
        } else {
            1
        };
        // Split-K is not used here: the GEMV runs one block per SM at eight
        // warps, so 56 blocks already fill the card, and the zeroed output plus
        // the atomics cost more than the staging they save. Measured on the
        // model's shapes — 3584 rows, q4_k: rows 2 = 319 GB/s, rows 2 + 2
        // splits = 220; 18944 rows: rows 4 = 400, rows 4 + 2 splits = 360.
        // `gemv_r` keeps the parameter so the sweep can pin that.
        self.gemv_r(
            q6, cols, rows_per_warp, 1, dev, w, row_stride, y, y_stride, ys, ys_stride,
            bias, out, out_stride, nrows, k, stream,
        )
    }

    /// Rows per warp for the single-column path. Each block stages one
    /// activation row, so the staging overhead is `k * 2 / (rows * row_bytes)`,
    /// but more rows per warp means fewer blocks and fewer warps per SM.
    ///
    /// Measured with `gemv_split_bandwidth` on the shapes the model launches
    /// (q4_k unless marked): 3584 rows 2 = 319 GB/s, 4 = 298, 8 = 266; 18944
    /// rows 4 = 400, 8 = 376; 512 rows 2 = 115, 1 = 109; q6_k down_proj
    /// (3584 rows, 3136 B rows) 4 = 377, 2 = 313; q6_k lm_head 4 = 395.
    fn rows_for(nrows: usize, q6: bool) -> usize {
        if q6 {
            if nrows >= 1024 {
                4
            } else {
                2
            }
        } else if nrows >= 16384 {
            4
        } else {
            2
        }
    }

    /// [`gemv`] with an explicit row-per-warp choice; the tests sweep it.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_r(
        &self,
        q6: bool,
        cols: usize,
        rows_per_warp: usize,
        nsplit: usize,
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
        let arr = if q6 { &self.gemv_q6k } else { &self.gemv_q4k };
        let f: &Function = match (cols, rows_per_warp) {
            (1, 2) => &arr[5],
            (1, 4) => &arr[6],
            (1, 8) => &arr[7],
            (8, 2) => &arr[4],
            _ => &arr[Self::cols_idx(cols)],
        };
        let block = 256u32;
        let warps = 8usize;
        let per_block = warps * rows_per_warp;
        if nsplit > 1 {
            out.zero()?;
        }
        let grid = (
            nrows.div_ceil(per_block) as u32,
            1,
            nsplit.max(1) as u32,
        );
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
        x_off: usize,
        y: &Buf,
        ys: &Buf,
        ncols: usize,
        k: usize,
        stream: Option<&Stream>,
    ) -> crate::Result<()> {
        let warps_total = ncols * (k / 32);
        let mut a_x = x.as_ptr() + (x_off * 4) as u64;
        let mut a_y = y.as_ptr();
        let mut a_ys = ys.as_ptr();
        let mut a_c = ncols as i32;
        let mut a_k = k as i32;
        let mut args = LaunchArgs::new();
        args.push(&mut a_x)
            .push(&mut a_y)
            .push(&mut a_ys)
            .push(&mut a_c)
            .push(&mut a_k);
        dev.set_current();
        let grid = ((warps_total as u32 + 7) / 8, 1, 1);
        launch(&self.prep_act, grid, (256, 1, 1), 0, stream, &mut args)
    }

    /// Per-column activation preparation for the GEMM: one power-of-two scale
    /// per column, `ys[c][k/16] = (2^-kk, sum of 16 scaled values)`.
    pub fn prep_col(
        &self,
        dev: &Device,
        x: &Buf,
        x_off: usize,
        y: &Buf,
        ys: &Buf,
        ncols: usize,
        k: usize,
        stream: Option<&Stream>,
    ) -> crate::Result<()> {
        assert_eq!(k % 16, 0, "GEMM k must be a multiple of 16");
        let mut a_x = x.as_ptr() + (x_off * 4) as u64;
        let mut a_y = y.as_ptr();
        let mut a_ys = ys.as_ptr();
        let mut a_c = ncols as i32;
        let mut a_k = k as i32;
        let mut args = LaunchArgs::new();
        args.push(&mut a_x)
            .push(&mut a_y)
            .push(&mut a_ys)
            .push(&mut a_c)
            .push(&mut a_k);
        dev.set_current();
        // One block per column: the kernel spreads each row over all 8 warps.
        let grid = (ncols as u32, 1, 1);
        launch(&self.prep_col, grid, (256, 1, 1), 0, stream, &mut args)
    }

    /// Production prefill GEMM: picks the instantiated tile for the weight
    /// format and a split-K that keeps the last wave full, then launches.
    ///
    /// Tiles measured on the P100 over the shapes Qwen2.5-7B launches (see
    /// docs/DESIGN.md): q4_k runs a 64x64 block with a 8x4 register tile and
    /// 64-value stages (3 blocks/SM); q6_k needs 128-value stages and keeps the
    /// 64x32 block, which is the fastest of its variants.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_auto(
        &self,
        q6: bool,
        sm_count: usize,
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
        ncols: usize,
        stream: Option<&Stream>,
    ) -> crate::Result<()> {
        let (f, tm, tn, tk, tt) = if q6 {
            (&self.gemm_q6k_n32t128, 64usize, 32usize, 128usize, 128usize)
        } else {
            (&self.gemm_q4k_n64t128, 64, 64, 64, 128)
        };
        let blocks = nrows.div_ceil(tm) * ncols.div_ceil(tn);
        let nsplit = pick_split(blocks, sm_count * 2, k / tk);
        self.gemm_tiled(f, tm, tn, tk, tt, dev, w, row_stride, y, y_stride, ys, ys_stride,
            bias, out, out_stride, nrows, k, ncols, nsplit, stream)
    }

    /// [`Kernels::gemm`] against an explicit tile shape, for the variants that
    /// are instantiated from `gemm_tiled` in kernels.cu. `tm`/`tn`/`tk`/`tt`
    /// must match the template arguments the function was instantiated with.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_tiled_pad(
        &self,
        f: &Function,
        tm: usize,
        tn: usize,
        tk: usize,
        tt: usize,
        pad: usize,
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
        ncols: usize,
        nsplit: usize,
        stream: Option<&Stream>,
    ) -> crate::Result<()> {
        assert_eq!(k % tk, 0, "gemm k must be a multiple of the stage");
        assert!(ncols > 0);
        if nsplit > 1 {
            out.zero()?;
        }
        let smem = ((tk / 2) * (tm + tn + pad) * 4) as u32;
        let mut a_w = w.as_ptr();
        let mut a_rs = row_stride as i64;
        let mut a_y = y.as_ptr();
        let mut a_yd = y_stride as i64;
        let mut a_ys = ys.as_ptr();
        let mut a_yss = ys_stride as i64;
        let mut a_b = bias.map(|b| b.as_ptr()).unwrap_or(0);
        let mut a_o = out.as_ptr();
        let mut a_ost = out_stride as i64;
        let mut a_n = nrows as i32;
        let mut a_k = k as i32;
        let mut a_c = ncols as i32;
        let mut a_ns = nsplit as i32;
        let mut args = LaunchArgs::new();
        args.push(&mut a_w)
            .push(&mut a_rs)
            .push(&mut a_y)
            .push(&mut a_yd)
            .push(&mut a_ys)
            .push(&mut a_yss)
            .push(&mut a_b)
            .push(&mut a_o)
            .push(&mut a_ost)
            .push(&mut a_n)
            .push(&mut a_k)
            .push(&mut a_c)
            .push(&mut a_ns);
        dev.set_current();
        let grid = (
            nrows.div_ceil(tm) as u32,
            ncols.div_ceil(tn) as u32,
            nsplit as u32,
        );
        launch(f, grid, (tt as u32, 1, 1), smem, stream, &mut args)
    }

    /// Dequantise a packed weight matrix to an fp16 workspace, row-major.
    #[allow(clippy::too_many_arguments)]
    pub fn dequant_f16(
        &self,
        q6: bool,
        dev: &Device,
        w: &Buf,
        row_stride: usize,
        out: &Buf,
        nrows: usize,
        k: usize,
        stream: Option<&Stream>,
    ) -> crate::Result<()> {
        let f = if q6 { &self.dequant_q6k_f16 } else { &self.dequant_q4k_f16 };
        let mut a_w = w.as_ptr();
        let mut a_rs = row_stride as i64;
        let mut a_o = out.as_ptr();
        let mut a_n = nrows as i32;
        let mut a_k = k as i32;
        let mut args = LaunchArgs::new();
        args.push(&mut a_w).push(&mut a_rs).push(&mut a_o).push(&mut a_n).push(&mut a_k);
        dev.set_current();
        let tasks = (nrows * k / 64) * 2;
        let grid = ((tasks as u32 + 255) / 256, 1, 1);
        launch(f, grid, (256, 1, 1), 0, stream, &mut args)
    }

    /// fp16-weight prefill GEMM (see `gemm_f16` in kernels.cu): `W` is the
    /// dequantised copy, `row_stride` is in halves. The block is 128x128 with
    /// no split-K, so `nrows / 128 * ncols / 128` should fill the card.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f16(
        &self,
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
        ncols: usize,
        stream: Option<&Stream>,
    ) -> crate::Result<()> {
        // 128x128 with an 8x8 register tile; 64x128/8x4 and 128x128/4x8 measure
        // within 1% of it, so the wider fragment stays.
        self.gemm_f16_v(&self.gemm_f16, 128, 128, dev, w, row_stride, y, y_stride,
            ys, ys_stride, bias, out, out_stride, nrows, k, ncols, stream)
    }

    /// [`Kernels::gemm_f16`] against an explicit instantiation.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f16_v(
        &self,
        f: &Function,
        tm: usize,
        tn: usize,
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
        ncols: usize,
        stream: Option<&Stream>,
    ) -> crate::Result<()> {
        assert_eq!(k % 32, 0, "fp16 gemm k must be a multiple of 32");
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
        let mut a_c = ncols as i32;
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
            .push(&mut a_k)
            .push(&mut a_c);
        dev.set_current();
        let grid = (nrows.div_ceil(tm) as u32, ncols.div_ceil(tn) as u32, 1);
        launch(f, grid, (256, 1, 1), 0, stream, &mut args)
    }

    /// [`Kernels::gemm_tiled_pad`] with the default four-half2 activation pad.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_tiled(
        &self,
        f: &Function,
        tm: usize,
        tn: usize,
        tk: usize,
        tt: usize,
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
        ncols: usize,
        nsplit: usize,
        stream: Option<&Stream>,
    ) -> crate::Result<()> {
        self.gemm_tiled_pad(f, tm, tn, tk, tt, 4, dev, w, row_stride, y, y_stride, ys,
            ys_stride, bias, out, out_stride, nrows, k, ncols, nsplit, stream)
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
        args.push(&mut a_x)
            .push(&mut a_w)
            .push(&mut a_o)
            .push(&mut a_n)
            .push(&mut a_e);
        dev.set_current();
        // Decode is one row: 256 threads leaves 8 warps on the whole card, and
        // the row reduction is block-wide, so widen the block instead of
        // adding blocks. 24-32 regs/thread still fits 1024.
        let block = if rows == 1 { 1024 } else { 256 };
        launch(
            &self.rms_norm,
            (rows as u32, 1, 1),
            (block, 1, 1),
            0,
            stream,
            &mut args,
        )
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
        args.push(&mut a_x)
            .push(&mut a_d)
            .push(&mut a_w)
            .push(&mut a_o)
            .push(&mut a_n)
            .push(&mut a_e);
        dev.set_current();
        // Decode is one row: 256 threads leaves 8 warps on the whole card, and
        // the row reduction is block-wide, so widen the block instead of
        // adding blocks. 24-32 regs/thread still fits 1024.
        let block = if rows == 1 { 1024 } else { 256 };
        launch(
            &self.add_rms_norm,
            (rows as u32, 1, 1),
            (block, 1, 1),
            0,
            stream,
            &mut args,
        )
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
        ctl: &Buf,
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
        let mut a_sp = ctl.as_ptr();
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
        ctl: &Buf,
        n_split: usize,
        qpk: usize,
        stream: Option<&Stream>,
    ) -> crate::Result<()> {
        assert!(
            qpk <= 7,
            "decode attention block covers at most 7 query heads"
        );
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
        let mut a_nc = ctl.as_ptr();
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
        launch(
            &self.attn_combine,
            (n_heads as u32, 1, 1),
            (128, 1, 1),
            0,
            stream,
            &mut args,
        )
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
        assert!(nq > 0, "prefill attention needs at least one query");
        let _ = n_ctx;
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
        launch(
            &self.attn_prefill,
            (n_heads as u32, nq.div_ceil(32) as u32, 1),
            (256, 1, 1),
            0,
            stream,
            &mut args,
        )
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
        args.push(&mut a_o)
            .push(&mut a_g)
            .push(&mut a_u)
            .push(&mut a_n);
        dev.set_current();
        launch(
            &self.silu_mul,
            (((n as u32) + 255) / 256, 1, 1),
            (256, 1, 1),
            0,
            stream,
            &mut args,
        )
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
        launch(
            &self.add_inplace,
            (((n as u32) + 255) / 256, 1, 1),
            (256, 1, 1),
            0,
            stream,
            &mut args,
        )
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
        args.push(&mut a_w)
            .push(&mut a_rs)
            .push(&mut a_i)
            .push(&mut a_o)
            .push(&mut a_n)
            .push(&mut a_k);
        dev.set_current();
        let grid = ((warps as u32 + 7) / 8, 1, 1);
        launch(&self.embed_q4k, grid, (256, 1, 1), 0, stream, &mut args)
    }

    pub fn test_add(&self, dev: &Device, a: &Buf, b: &Buf, c: &Buf, n: usize) -> crate::Result<()> {
        let mut pa = a.as_ptr();
        let mut pb = b.as_ptr();
        let mut pc = c.as_ptr();
        let mut pn = n as i32;
        let mut args = LaunchArgs::new();
        args.push(&mut pa).push(&mut pb).push(&mut pc).push(&mut pn);
        dev.set_current();
        launch(
            &self.test_add,
            (((n as u32) + 255) / 256, 1, 1),
            (256, 1, 1),
            0,
            None,
            &mut args,
        )
    }

    /// `n` iterations of 16 independent HFMA2 per thread, no memory traffic.
    pub fn hfma2_bench(
        &self,
        dev: &Device,
        out: &Buf,
        n: i32,
        blocks: u32,
        block: u32,
    ) -> crate::Result<()> {
        let mut a_o = out.as_ptr();
        let mut a_n = n;
        let mut args = LaunchArgs::new();
        args.push(&mut a_o).push(&mut a_n);
        dev.set_current();
        launch(
            &self.hfma2,
            (blocks, 1, 1),
            (block, 1, 1),
            0,
            None,
            &mut args,
        )
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
        launch(
            &self.bw_read,
            (blocks, 1, 1),
            (256, 1, 1),
            0,
            None,
            &mut args,
        )
    }

    /// Block walk with all 32 lanes issuing (four lane-groups, four rows).
    pub fn bw_rows4(
        &self,
        dev: &Device,
        p: &Buf,
        nrows: i32,
        row_u4: i64,
        out: &Buf,
        blocks: u32,
    ) -> crate::Result<()> {
        let mut a_p = p.as_ptr();
        let mut a_r = nrows;
        let mut a_s = row_u4;
        let mut a_o = out.as_ptr();
        let mut args = LaunchArgs::new();
        args.push(&mut a_p)
            .push(&mut a_r)
            .push(&mut a_s)
            .push(&mut a_o);
        dev.set_current();
        launch(
            &self.bw_rows4,
            (blocks, 1, 1),
            (256, 1, 1),
            0,
            None,
            &mut args,
        )
    }

    /// Read weights in `chunk`-uint4 contiguous spans, with no arithmetic.
    /// `nrows` must be a multiple of 8 so every warp covers whole rows.
    pub fn bw_chunk(
        &self,
        dev: &Device,
        p: &Buf,
        nrows: i32,
        chunk: i32,
        row_u4: i64,
        out: &Buf,
        blocks: u32,
    ) -> crate::Result<()> {
        assert_eq!(nrows % 8, 0, "bw_chunk wants whole rows per warp");
        assert!(chunk > 0, "chunk must be positive");
        let mut a_p = p.as_ptr();
        let mut a_r = nrows;
        let mut a_b = chunk;
        let mut a_s = row_u4;
        let mut a_o = out.as_ptr();
        let mut args = LaunchArgs::new();
        args.push(&mut a_p)
            .push(&mut a_r)
            .push(&mut a_b)
            .push(&mut a_s)
            .push(&mut a_o);
        dev.set_current();
        launch(
            &self.bw_chunk,
            (blocks, 1, 1),
            (256, 1, 1),
            0,
            None,
            &mut args,
        )
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
            let run = jj;
            let sc_lo = rep[4 + 2 * t] as i8 as f32;
            let sc_hi = rep[4 + 2 * t + 1] as i8 as f32;
            let rd = |o: usize| u32::from_le_bytes(rep[o..o + 4].try_into().unwrap());
            let h0 = rd(160 + run * 16 + 8 * hf);
            let h1 = rd(160 + run * 16 + 8 * hf + 4);
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

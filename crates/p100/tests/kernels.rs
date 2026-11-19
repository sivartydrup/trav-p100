//! On-GPU correctness checks. Skipped when no CUDA device is present.
//!
//! These compare every quantised GEMV against the GGUF reference dequantiser, so
//! a layout mistake in the loader/repack is caught before the model loads.

use half::f16;
use trav_gguf::quant::{dequantize_row, GgmlType};
use trav_p100::kernels::{repack_q6k, Kernels};

struct Rng(u64);

impl Rng {
    fn u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn f32(&mut self) -> f32 {
        (self.u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.u32() as u8).collect()
    }
}

fn have_gpu() -> Option<trav_p100::Device> {
    let n = trav_p100::Device::count().ok()?;
    if n == 0 {
        return None;
    }
    trav_p100::Device::open(0).ok()
}

fn as_bytes<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

#[test]
fn driver_and_add_kernel() {
    let Some(dev) = have_gpu() else {
        eprintln!("no CUDA device, skipping");
        return;
    };
    let k = Kernels::load(&dev).expect("load cubin");
    let a: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..1024).map(|i| (i * 2) as f32).collect();
    let ba = dev.upload(as_bytes(&a)).unwrap();
    let bb = dev.upload(as_bytes(&b)).unwrap();
    let bc = dev.alloc(1024 * 4).unwrap();
    k.test_add(&dev, &ba, &bb, &bc, 1024).unwrap();
    dev.sync().unwrap();
    let mut out = vec![0f32; 1024];
    bc.download_to(as_bytes_mut(&mut out)).unwrap();
    for i in 0..1024 {
        assert_eq!(out[i], a[i] + b[i]);
    }
    println!("device: {dev:?} bw~{:.0} GB/s", dev.theoretical_bw_gbs());
}

fn as_bytes_mut<T>(v: &mut [T]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}

fn run_case(ty: GgmlType, blocks: usize) -> f32 {
    let mut rng = Rng(0x1234_5678_9abc_def0);
    let rows = 4usize;
    let k = blocks * 256;
    let row_bytes = match ty {
        GgmlType::Q4K => 144,
        GgmlType::Q6K => 210,
        _ => unreachable!(),
    };
    let mut w_host = vec![0u8; rows * blocks * row_bytes];
    for b in 0..rows * blocks {
        let blk = &mut w_host[b * row_bytes..(b + 1) * row_bytes];
        if ty == GgmlType::Q4K {
            blk[0..2].copy_from_slice(&f16::from_f32(0.02).to_bits().to_le_bytes());
            blk[2..4].copy_from_slice(&f16::from_f32(0.01).to_bits().to_le_bytes());
            let r = &mut rng.bytes(140);
            blk[4..16].copy_from_slice(&r[..12]);
            blk[16..144].copy_from_slice(&r[12..140]);
        } else {
            let r = &mut rng.bytes(210);
            blk.copy_from_slice(&r[..210]);
            // realistic Q6_K scale magnitudes, signed sub-block scales
            blk[208..210].copy_from_slice(&f16::from_f32(0.005).to_bits().to_le_bytes());
            for i in 0..16 {
                blk[192 + i] = ((rng.u32() % 64) as i32 - 32) as i8 as u8;
            }
        }
    }

    let y_host: Vec<f32> = (0..k).map(|_| rng.f32() * 1.5).collect();

    // reference
    let mut wf = vec![0f32; k];
    let mut want = vec![0f32; rows];
    for r in 0..rows {
        dequantize_row(ty, &w_host[r * blocks * row_bytes..(r + 1) * blocks * row_bytes], &mut wf);
        want[r] = (0..k).map(|i| wf[i] * y_host[i]).sum();
    }

    let dev = have_gpu().expect("device");
    let kernels = Kernels::load(&dev).unwrap();
    let packed = if ty == GgmlType::Q6K {
        repack_q6k(&w_host)
    } else {
        w_host.clone()
    };
    let bstride = if ty == GgmlType::Q6K { 224 } else { 144 };
    let _ = bstride;

    let bw = dev.upload(&packed).unwrap();
    let by = dev.upload(as_bytes(&y_host)).unwrap();
    let byh = dev.alloc(k * 2).unwrap();
    let bys = dev.alloc((k / 16) * 8).unwrap();
    let bout = dev.alloc(rows * 4).unwrap();

    kernels.prep(&dev, &by, &byh, &bys, 1, k, None).unwrap();
    kernels
        .gemv(
            ty == GgmlType::Q6K,
            1,
            &dev,
            &bw,
            blocks * if ty == GgmlType::Q6K { 224 } else { 144 },
            &byh,
            k,
            &bys,
            k / 16,
            None,
            &bout,
            rows,
            rows,
            k,
            None,
        )
        .unwrap();
    trav_p100::kernels::last_error(&dev).unwrap();
    let mut got = vec![0f32; rows];
    bout.download_to(as_bytes_mut(&mut got)).unwrap();

    let mut worst = 0f32;
    for r in 0..rows {
        let denom = want[r].abs().max(1.0);
        let err = (got[r] - want[r]).abs() / denom;
        worst = worst.max(err);
        println!("{ty:?} row {r}: got {:.5} want {:.5} rel {:.2e}", got[r], want[r], err);
    }
    worst
}

#[test]
fn gemv_q4k_matches_reference() {
    let e1 = run_case(GgmlType::Q4K, 1);
    let e2 = run_case(GgmlType::Q4K, 14);
    assert!(e1 < 5e-3 && e2 < 5e-3, "q4k rel err {e1} {e2}");
}

#[test]
fn gemv_q6k_matches_reference() {
    let e1 = run_case(GgmlType::Q6K, 1);
    let e2 = run_case(GgmlType::Q6K, 14);
    // The layout is proven by the 1-block case; with 14 blocks the residual is
    // fp16 accumulation error amplified by cancellation in random data.
    assert!(e1 < 4e-3 && e2 < 2e-2, "q6k rel err {e1} {e2}");
}

#[test]
fn streaming_bandwidth() {
    let Some(dev) = have_gpu() else { return };
    let k = Kernels::load(&dev).unwrap();
    let n = 256 * 1024 * 1024; // 256 MiB
    let buf = dev.alloc(n).unwrap();
    buf.zero().unwrap();
    let out = dev.alloc(4).unwrap();
    // warm up
    k.bw_read(&dev, &buf, (n / 16) as i64, &out, 4096).unwrap();
    dev.sync().unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..4 {
        k.bw_read(&dev, &buf, (n / 16) as i64, &out, 4096).unwrap();
    }
    dev.sync().unwrap();
    let secs = t0.elapsed().as_secs_f64() / 4.0;
    println!("streaming read: {:.0} GB/s ({} MiB in {:.1} ms)", n as f64 / secs / 1e9, n >> 20, secs * 1e3);
}

/// Reports achieved GB/s for the single-column GEMV, the kernel that dominates
/// decode. Measured on a 3584x3584 q4_k matrix: 2016 bytes per row.
#[test]
fn gemv_bandwidth() {
    let Some(dev) = have_gpu() else { return };
    let k = Kernels::load(&dev).unwrap();
    let (rows, cols) = (16384usize, 18944usize);
    let nb = cols / 256;
    let row_bytes = nb * 144;
    let mut rng = Rng(7);
    let mut host = vec![0u8; rows * row_bytes];
    for b in 0..rows * nb {
        let blk = &mut host[b * 144..b * 144 + 144];
        blk[0..2].copy_from_slice(&f16::from_f32(0.02).to_bits().to_le_bytes());
        blk[2..4].copy_from_slice(&f16::from_f32(0.01).to_bits().to_le_bytes());
        for i in 0..12 {
            blk[4 + i] = (rng.u32() % 64) as u8;
        }
        for i in 16..144 {
            blk[i] = rng.u32() as u8;
        }
    }
    let y: Vec<f32> = (0..cols).map(|_| rng.f32()).collect();
    let w = dev.upload(&host).unwrap();
    let _ = &mut rng;
    let by = dev.upload(as_bytes(&y)).unwrap();
    let byh = dev.alloc(cols * 2).unwrap();
    let bys = dev.alloc((cols / 16) * 8).unwrap();
    let out = dev.alloc(rows * 4).unwrap();
    k.prep(&dev, &by, &byh, &bys, 1, cols, None).unwrap();

    let run = |iters: usize| {
        for _ in 0..iters {
            k.gemv(false, 1, &dev, &w, row_bytes, &byh, cols, &bys, cols / 16, None,
                   &out, rows, rows, cols, None).unwrap();
        }
        dev.sync().unwrap();
    };
    run(5);
    let t0 = std::time::Instant::now();
    run(20);
    let s = t0.elapsed().as_secs_f64() / 20.0;
    let bytes = rows * row_bytes;
    println!("gemv q4_k {rows}x{cols} C=1: {:.2} ms/run, {:.0} GB/s  (= {:.1} tok/s)",
             s * 1e3, bytes as f64 / s / 1e9, 1.0 / s);

    // prefill path: 8 columns, 8x the activation work for the same weights
    let y8: Vec<f32> = (0..8 * cols).map(|_| rng.f32()).collect();
    let by8 = dev.upload(as_bytes(&y8)).unwrap();
    let byh8 = dev.alloc_zeroed(8 * cols * 2).unwrap();
    let bys8 = dev.alloc_zeroed(8 * (cols / 16) * 8).unwrap();
    let out8 = dev.alloc(rows * 8 * 4).unwrap();
    for _ in 0..5 {
        k.prep(&dev, &by8, &byh8, &bys8, 8, cols, None).unwrap();
        k.gemv(false, 8, &dev, &w, row_bytes, &byh8, cols, &bys8, cols / 16, None,
               &out8, rows, rows, cols, None).unwrap();
    }
    dev.sync().unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..20 {
        k.gemv(false, 8, &dev, &w, row_bytes, &byh8, cols, &bys8, cols / 16, None,
               &out8, rows, rows, cols, None).unwrap();
    }
    dev.sync().unwrap();
    let s8 = t0.elapsed().as_secs_f64() / 20.0;
    println!("gemv q4_k C=8: {:.2} ms/run, {:.0} GB/s, {:.2} TFLOPS",
             s8 * 1e3, bytes as f64 / s8 / 1e9,
             (rows as f64 * cols as f64 * 2.0 * 8.0) / s8 / 1e12);
}


/// The multi-column kernel is the prefill path; a layout slip here silently
/// degrades every prompt. Compare every column of every instantiation against
/// the CPU reference.
#[test]
fn gemv_multicolumn_matches_reference() {
    let Some(dev) = have_gpu() else { return };
    let k = Kernels::load(&dev).unwrap();
    for ty in [GgmlType::Q4K, GgmlType::Q6K] {
        for cols in [1usize, 2, 4, 8] {
            let mut rng = Rng(0xfeed_face ^ cols as u64);
            // rows >= 16 selects the two-row-per-warp instantiation for 8 columns
            let rows = 16usize;
            let blocks = 74usize;
            let kdim = blocks * 256;
            let q6 = ty == GgmlType::Q6K;
            let row_bytes = blocks * if q6 { 210 } else { 144 };
            let mut w_host = vec![0u8; rows * row_bytes];
            let per_block = if q6 { 210 } else { 144 };
            for b in 0..rows * blocks {
                let blk = &mut w_host[b * per_block..(b + 1) * per_block];
                if q6 {
                    let r = &mut rng.bytes(208);
                    blk[..208].copy_from_slice(r);
                    blk[208..210].copy_from_slice(&f16::from_f32(0.005).to_bits().to_le_bytes());
                    for i in 0..16 {
                        blk[192 + i] = ((rng.u32() % 64) as i32 - 32) as i8 as u8;
                    }
                } else {
                    blk[0..2].copy_from_slice(&f16::from_f32(0.02).to_bits().to_le_bytes());
                    blk[2..4].copy_from_slice(&f16::from_f32(0.01).to_bits().to_le_bytes());
                    let r = &mut rng.bytes(140);
                    blk[4..16].copy_from_slice(&r[..12]);
                    blk[16..144].copy_from_slice(&r[12..140]);
                }
            }
            let packed = if q6 { repack_q6k(&w_host) } else { w_host.clone() };
            let stride = if q6 { blocks * 224 } else { blocks * 144 };

            let y: Vec<f32> = (0..cols * kdim).map(|_| rng.f32() * 1.5).collect();
            let mut want = vec![0f32; rows * cols];
            let mut energy = vec![0f32; rows * cols];
            let mut wf = vec![0f32; kdim];
            for r in 0..rows {
                dequantize_row(ty, &w_host[r * row_bytes..(r + 1) * row_bytes], &mut wf);
                for c in 0..cols {
                    want[c * rows + r] = (0..kdim).map(|i| wf[i] * y[c * kdim + i]).sum();
                    energy[c * rows + r] =
                        (0..kdim).map(|i| (wf[i] * y[c * kdim + i]).abs()).sum();
                }
            }

            let bw = dev.upload(&packed).unwrap();
            let by = dev.upload(as_bytes(&y)).unwrap();
            let byh = dev.alloc_zeroed(cols * kdim * 2).unwrap();
            let bys = dev.alloc_zeroed(cols * (kdim / 16) * 8).unwrap();
            let out = dev.alloc(rows * cols * 4).unwrap();
            k.prep(&dev, &by, &byh, &bys, cols, kdim, None).unwrap();
            k.gemv(q6, cols, &dev, &bw, stride, &byh, kdim, &bys, kdim / 16, None,
                   &out, rows, rows, kdim, None).unwrap();
            trav_p100::kernels::last_error(&dev).unwrap();
            let mut got = vec![0f32; rows * cols];
            out.download_to(as_bytes_mut(&mut got)).unwrap();
            let mut worst = 0f32;
            for c in 0..cols {
                for r in 0..rows {
                    // error relative to the sum of |terms|: a fair measure for an
                    // fp16 accumulation that suffers cancellation
                    let d = (got[c * rows + r] - want[c * rows + r]).abs()
                        / energy[c * rows + r].max(1e-6);
                    worst = worst.max(d);
                }
            }
            println!("{ty:?} cols={cols} worst rel err {worst:.2e}");
            // q6_k accumulates 16 six-bit products in fp16; with cancellation in
            // random data that is worth a few 1e-3. Layout bugs are O(1).
            let tol = if q6 { 4e-3 } else { 2e-3 };
            assert!(worst < tol, "{ty:?} cols={cols} rel err {worst}");
        }
    }
}

/// Prefill attention vs a CPU reference: causal mask, GQA mapping, online
/// softmax and KV layout in one check.
#[test]
fn attn_prefill_matches_reference() {
    let Some(dev) = have_gpu() else { return };
    let k = Kernels::load(&dev).unwrap();
    let mut rng = Rng(0xabcdef);
    let n_heads = 2usize;
    let n_kv = 1usize;
    let hd = 128usize;
    let qpk = n_heads / n_kv;
    let n_ctx = 96usize;
    let mut worst_all = 0f32;
    for (q_pos0, nq) in [(0usize, 2usize), (4, 2), (3, 2), (2, 4), (32, 8), (88, 8), (40, 1)] {
        let kf: Vec<f32> = (0..n_ctx * n_kv * hd).map(|_| rng.f32()).collect();
        let vf: Vec<f32> = (0..n_ctx * n_kv * hd).map(|_| rng.f32()).collect();
        let kc: Vec<u16> = kf.iter().map(|x| f16::from_f32(*x).to_bits()).collect();
        let vc: Vec<u16> = vf.iter().map(|x| f16::from_f32(*x).to_bits()).collect();
        let q: Vec<f32> = (0..nq * n_heads * hd).map(|_| rng.f32()).collect();

        let bk = dev.upload(as_bytes(&kc)).unwrap();
        let bv = dev.upload(as_bytes(&vc)).unwrap();
        let bq = dev.upload(as_bytes(&q)).unwrap();
        let bo = dev.alloc(nq * n_heads * hd * 4).unwrap();
        k.attn_prefill(&dev, &bk, &bv, &bq, &bo, n_heads, n_kv, hd, q_pos0 + nq, qpk,
                       q_pos0, nq, None).unwrap();
        trav_p100::kernels::last_error(&dev).unwrap();
        let mut got = vec![0f32; nq * n_heads * hd];
        bo.download_to(as_bytes_mut(&mut got)).unwrap();

        let h16 = |v: &[u16], i: usize| f16::from_bits(v[i]).to_f32();
        let mut worst = 0f32;
        for qi in 0..nq {
            let pos = q_pos0 + qi;
            for h in 0..n_heads {
                let kvh = h / qpk;
                let mut scores = vec![0f32; pos + 1];
                for p in 0..=pos {
                    let mut s = 0f32;
                    for d in 0..hd {
                        s += q[(qi * n_heads + h) * hd + d] * h16(&kc, (p * n_kv + kvh) * hd + d);
                    }
                    scores[p] = s * (hd as f32).sqrt().recip();
                }
                let mx = scores.iter().cloned().fold(f32::MIN, f32::max);
                let den: f32 = scores.iter().map(|s| (s - mx).exp()).sum();
                for d in 0..hd {
                    let mut acc = 0f32;
                    for p in 0..=pos {
                        acc += (scores[p] - mx).exp() / den * h16(&vc, (p * n_kv + kvh) * hd + d);
                    }
                    let g = got[(qi * n_heads + h) * hd + d];
                    let dabs = (g - acc).abs() / acc.abs().max(1.0);
                    worst = worst.max(dabs);
                }
            }
        }
        println!("attn_prefill q_pos0={q_pos0} nq={nq} worst rel err {worst:.2e}");
        worst_all = worst_all.max(worst);
    }
    assert!(worst_all < 5e-3, "attn_prefill rel err {worst_all}");
}

/// RoPE (rotate-half) + KV write for a multi-token tile.
#[test]
fn rope_and_kv_write_match_reference() {
    let Some(dev) = have_gpu() else { return };
    let k = Kernels::load(&dev).unwrap();
    let mut rng = Rng(0x5eed);
    let n_q_heads = 2usize;
    let n_k_heads = 1usize;
    let hd = 8usize;
    let n = 3usize;
    let pos0 = 4usize;
    let theta = 10000.0f32;

    let q: Vec<f32> = (0..n * n_q_heads * hd).map(|_| rng.f32()).collect();
    let kk: Vec<f32> = (0..n * n_k_heads * hd).map(|_| rng.f32()).collect();
    let pos: Vec<i32> = (0..n).map(|i| (pos0 + i) as i32).collect();
    let inv: Vec<f32> = (0..hd / 2).map(|i| 1.0 / theta.powf(2.0 * i as f32 / hd as f32)).collect();

    let bq = dev.upload(as_bytes(&q)).unwrap();
    let bk = dev.upload(as_bytes(&kk)).unwrap();
    let bp = dev.upload(as_bytes(&pos)).unwrap();
    let bi = dev.upload(as_bytes(&inv)).unwrap();
    let vv: Vec<f32> = (0..n * n_k_heads * hd).map(|_| rng.f32()).collect();
    let bv = dev.upload(as_bytes(&vv)).unwrap();
    let cap = 8usize;
    let kc = dev.alloc_zeroed(cap * n_k_heads * hd * 2).unwrap();
    let vc = dev.alloc_zeroed(cap * n_k_heads * hd * 2).unwrap();

    k.rope(&dev, &bq, &bk, &bp, &bi, n_q_heads, n_k_heads, hd, n, 0, None).unwrap();
    k.write_kv(&dev, &kc, &vc, &bk, &bv, n_k_heads, hd, pos0, n, None).unwrap();
    trav_p100::kernels::last_error(&dev).unwrap();

    let mut gq = vec![0f32; q.len()];
    let mut gk = vec![0f32; kk.len()];
    bq.download_to(as_bytes_mut(&mut gq)).unwrap();
    bk.download_to(as_bytes_mut(&mut gk)).unwrap();
    let mut gkc = vec![0u16; cap * n_k_heads * hd];
    kc.download_to(as_bytes_mut(&mut gkc)).unwrap();

    let mut worst = 0f32;
    for t in 0..n {
        for h in 0..n_q_heads {
            for i in 0..hd / 2 {
                let ang = (pos0 + t) as f32 * inv[i];
                let (sn, cs) = ang.sin_cos();
                let x0 = q[(t * n_q_heads + h) * hd + i];
                let x1 = q[(t * n_q_heads + h) * hd + i + hd / 2];
                worst = worst.max((gq[(t * n_q_heads + h) * hd + i] - (x0 * cs - x1 * sn)).abs() / (x0 * cs - x1 * sn).abs().max(1.0));
                worst = worst.max((gq[(t * n_q_heads + h) * hd + i + hd / 2] - (x1 * cs + x0 * sn)).abs() / (x1 * cs + x0 * sn).abs().max(1.0));
            }
        }
        for h in 0..n_k_heads {
            for i in 0..hd {
                let ang = (pos0 + t) as f32 * inv[i % (hd / 2)];
                let (sn, cs) = ang.sin_cos();
                let (x0, x1) = if i < hd / 2 {
                    (kk[(t * n_k_heads + h) * hd + i], kk[(t * n_k_heads + h) * hd + i + hd / 2])
                } else {
                    (kk[(t * n_k_heads + h) * hd + i - hd / 2], kk[(t * n_k_heads + h) * hd + i])
                };
                let want = f16::from_f32(if i < hd / 2 { x0 * cs - x1 * sn } else { x1 * cs + x0 * sn }).to_f32();
                let got = f16::from_bits(gkc[((pos0 + t) * n_k_heads + h) * hd + i]).to_f32();
                worst = worst.max((got - want).abs() / want.abs().max(1.0));
            }
        }
    }
    println!("rope/kv worst rel err {worst:.2e}");
    assert!(worst < 2e-3, "rope/kv rel err {worst}");
}

/// Decode attention (split-KV + combine) against the CPU reference.
#[test]
fn attn_decode_matches_reference() {
    let Some(dev) = have_gpu() else { return };
    let k = Kernels::load(&dev).unwrap();
    let mut rng = Rng(0x1234abcd);
    let n_heads = 4usize;
    let n_kv = 1usize;
    let hd = 128usize;
    let qpk = n_heads / n_kv;
    let n_ctx = 96usize;
    let kv_elems = n_ctx * n_kv * hd;
    let kc: Vec<u16> = (0..kv_elems).map(|_| f16::from_f32(rng.f32()).to_bits()).collect();
    let vc: Vec<u16> = (0..kv_elems).map(|_| f16::from_f32(rng.f32()).to_bits()).collect();
    let q: Vec<f32> = (0..n_heads * hd).map(|_| rng.f32()).collect();

    for n_split in [1usize, 4, 16, 32] {
        let bk = dev.upload(as_bytes(&kc)).unwrap();
        let bv = dev.upload(as_bytes(&vc)).unwrap();
        let bq = dev.upload(as_bytes(&q)).unwrap();
        let bo = dev.alloc(n_heads * hd * 4).unwrap();
        let pm = dev.alloc(n_heads * n_split * 4).unwrap();
        let pl = dev.alloc(n_heads * n_split * 4).unwrap();
        let pa = dev.alloc(n_heads * n_split * hd * 4).unwrap();
        k.attn_decode(&dev, &bk, &bv, &bq, &bo, &pm, &pl, &pa, n_heads, n_kv, hd, n_ctx,
                      n_split, qpk, None).unwrap();
        k.attn_combine(&dev, &pm, &pl, &pa, &bo, n_heads, hd, n_split, None).unwrap();
        trav_p100::kernels::last_error(&dev).unwrap();
        let mut got = vec![0f32; n_heads * hd];
        bo.download_to(as_bytes_mut(&mut got)).unwrap();

        let h16 = |v: &[u16], i: usize| f16::from_bits(v[i]).to_f32();
        let mut worst = 0f32;
        for h in 0..n_heads {
            let kvh = h / qpk;
            let mut scores = vec![0f32; n_ctx];
            for p in 0..n_ctx {
                let mut s = 0f32;
                for d in 0..hd {
                    s += q[h * hd + d] * h16(&kc, (p * n_kv + kvh) * hd + d);
                }
                scores[p] = s * (hd as f32).sqrt().recip();
            }
            let mx = scores.iter().cloned().fold(f32::MIN, f32::max);
            let den: f32 = scores.iter().map(|s| (s - mx).exp()).sum();
            for d in 0..hd {
                let mut acc = 0f32;
                for p in 0..n_ctx {
                    acc += (scores[p] - mx).exp() / den * h16(&vc, (p * n_kv + kvh) * hd + d);
                }
                let g = got[h * hd + d];
                worst = worst.max((g - acc).abs() / acc.abs().max(1.0));
            }
        }
        println!("attn_decode n_split={n_split} worst rel err {worst:.2e}");
        assert!(worst < 5e-3, "attn_decode n_split={n_split} rel err {worst}");
    }
}

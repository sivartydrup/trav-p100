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
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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

fn run_case(ty: GgmlType, blocks: usize, rows: usize, rows_per_warp: usize) -> f32 {
    run_case_split(ty, blocks, rows, rows_per_warp, 1)
}

fn run_case_split(ty: GgmlType, blocks: usize, rows: usize, rows_per_warp: usize, nsplit: usize) -> f32 {
    let mut rng = Rng(0x1234_5678_9abc_def0);
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
        dequantize_row(
            ty,
            &w_host[r * blocks * row_bytes..(r + 1) * blocks * row_bytes],
            &mut wf,
        );
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

    kernels.prep(&dev, &by, 0, &byh, &bys, 1, k, None).unwrap();
    kernels
        .gemv_r(
            ty == GgmlType::Q6K,
            1,
            rows_per_warp,
            nsplit,
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
        println!(
            "{ty:?} row {r}: got {:.5} want {:.5} rel {:.2e}",
            got[r], want[r], err
        );
    }
    worst
}

#[test]
fn gemv_q4k_matches_reference() {
    let e1 = run_case(GgmlType::Q4K, 1, 4, 1);
    let e2 = run_case(GgmlType::Q4K, 14, 4, 1);
    // The decode dispatch now uses 8 rows per warp for wide projections, so
    // that shape needs its own correctness pass.
    let e3 = run_case(GgmlType::Q4K, 14, 8, 8);
    let e4 = run_case(GgmlType::Q4K, 14, 4, 2);
    let e5 = run_case_split(GgmlType::Q4K, 14, 4, 2, 3);
    let e6 = run_case_split(GgmlType::Q4K, 14, 8, 8, 2);
    assert!(
        e1 < 5e-3 && e2 < 5e-3 && e3 < 5e-3 && e4 < 5e-3 && e5 < 5e-3 && e6 < 5e-3,
        "q4k rel err {e1} {e2} {e3} {e4} {e5} {e6}"
    );
}

#[test]
fn gemv_q6k_matches_reference() {
    let e1 = run_case(GgmlType::Q6K, 1, 4, 1);
    let e2 = run_case(GgmlType::Q6K, 14, 4, 1);
    let e3 = run_case(GgmlType::Q6K, 14, 8, 8);
    let e5 = run_case_split(GgmlType::Q6K, 14, 4, 2, 3);
    // The layout is proven by the 1-block case; with 14 blocks the residual is
    // fp16 accumulation error amplified by cancellation in random data.
    assert!(
        e1 < 4e-3 && e2 < 2e-2 && e3 < 2e-2 && e5 < 2e-2,
        "q6k rel err {e1} {e2} {e3} {e5}"
    );
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
    println!(
        "streaming read: {:.0} GB/s ({} MiB in {:.1} ms)",
        n as f64 / secs / 1e9,
        n >> 20,
        secs * 1e3
    );
}

/// The GEMV reads q4_k weights as 144-byte blocks (9 x uint4), not as a flat
/// stream. This measures that pattern with no dequantise, so the gap between
/// it and the 594 GB/s streaming ceiling is the access pattern's cost and the
/// remaining gap to `gemv_q4k_r8c1` is the arithmetic's.
#[test]
fn weight_read_granularity() {
    let Some(dev) = have_gpu() else { return };
    let k = Kernels::load(&dev).unwrap();
    let rows = 18944usize; // gate_proj / up_proj
    let nblk = 14usize; // 3584 values / 256 per super-block
    let row_u4 = nblk * 9;
    let bytes = rows * row_u4 * 16;
    let buf = dev.alloc(bytes).unwrap();
    buf.zero().unwrap();
    let out = dev.alloc(4).unwrap();
    // Same bytes as a flat uint4 stream: the difference from the block walk
    // below is the cost of 144-byte (4.5-sector) block granularity.
    k.bw_read(&dev, &buf, (bytes / 16) as i64, &out, 4096).unwrap();
    dev.sync().unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..10 {
        k.bw_read(&dev, &buf, (bytes / 16) as i64, &out, 4096).unwrap();
    }
    dev.sync().unwrap();
    let s = t0.elapsed().as_secs_f64() / 10.0;
    println!(
        "flat stream  {rows} rows: {:.2} ms, {:.0} GB/s",
        s * 1e3,
        bytes as f64 / s / 1e9
    );

    // 144 B (one q4_k block) up to a whole 2016 B row. The GEMV launches
    // 8 warps x 8 rows = 64 rows per block => 296 blocks.
    for chunk in [9i32, 18, 36, 126] {
        k.bw_chunk(&dev, &buf, rows as i32, chunk, row_u4 as i64, &out, 296)
            .unwrap();
        dev.sync().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..10 {
            k.bw_chunk(&dev, &buf, rows as i32, chunk, row_u4 as i64, &out, 296)
                .unwrap();
        }
        dev.sync().unwrap();
        let s = t0.elapsed().as_secs_f64() / 10.0;
        println!(
            "span {:>4} B: {:.2} ms, {:.0} GB/s",
            chunk as usize * 16,
            s * 1e3,
            bytes as f64 / s / 1e9
        );
    }

    k.bw_rows4(&dev, &buf, rows as i32, row_u4 as i64, &out, 296)
        .unwrap();
    dev.sync().unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..10 {
        k.bw_rows4(&dev, &buf, rows as i32, row_u4 as i64, &out, 296)
            .unwrap();
    }
    dev.sync().unwrap();
    let s = t0.elapsed().as_secs_f64() / 10.0;
    println!(
        "4 rows/instr, 32 lanes: {:.2} ms, {:.0} GB/s",
        s * 1e3,
        bytes as f64 / s / 1e9
    );
}

/// Reports achieved GB/s for the single-column GEMV, the kernel that dominates
/// decode, over the projection shapes Qwen2.5-7B actually launches. A shape
/// whose row count underfills the card shows up as a low GB/s here.
#[test]
fn gemv_bandwidth() {
    let Some(dev) = have_gpu() else { return };
    let k = Kernels::load(&dev).unwrap();
    for (rows, cols) in [
        (3584usize, 3584usize),   // q_proj / o_proj
        (512, 3584),              // k_proj / v_proj (GQA 4 KV heads x 128)
        (18944, 3584),            // gate_proj / up_proj
        (3584, 18944),            // down_proj
        (152064, 3584),           // lm_head
    ] {
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
    k.prep(&dev, &by, 0, &byh, &bys, 1, cols, None).unwrap();

    let run = |rows_per_warp: usize, iters: usize| {
        for _ in 0..iters {
            k.gemv_r(
                false,
                1,
                rows_per_warp,
                1,
                &dev,
                &w,
                row_bytes,
                &byh,
                cols,
                &bys,
                cols / 16,
                None,
                &out,
                rows,
                rows,
                cols,
                None,
            )
            .unwrap();
        }
        dev.sync().unwrap();
    };
    let bytes = rows * row_bytes;
    for rows_per_warp in [1usize, 2, 4, 8] {
        run(rows_per_warp, 5);
        let t0 = std::time::Instant::now();
        run(rows_per_warp, 20);
        let s = t0.elapsed().as_secs_f64() / 20.0;
        println!(
            "gemv q4_k {rows}x{cols} C=1 rows={rows_per_warp}: {:.2} ms/run, {:.0} GB/s \
             (= {:.1} tok/s)",
            s * 1e3,
            bytes as f64 / s / 1e9,
            1.0 / s
        );
    }

    // prefill path: 8 columns, 8x the activation work for the same weights
    let y8: Vec<f32> = (0..8 * cols).map(|_| rng.f32()).collect();
    let by8 = dev.upload(as_bytes(&y8)).unwrap();
    let byh8 = dev.alloc_zeroed(8 * cols * 2).unwrap();
    let bys8 = dev.alloc_zeroed(8 * (cols / 16) * 8).unwrap();
    let out8 = dev.alloc(rows * 8 * 4).unwrap();
    for _ in 0..5 {
        k.prep(&dev, &by8, 0, &byh8, &bys8, 8, cols, None).unwrap();
        k.gemv(
            false,
            8,
            &dev,
            &w,
            row_bytes,
            &byh8,
            cols,
            &bys8,
            cols / 16,
            None,
            &out8,
            rows,
            rows,
            cols,
            None,
        )
        .unwrap();
    }
    dev.sync().unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..20 {
        k.gemv(
            false,
            8,
            &dev,
            &w,
            row_bytes,
            &byh8,
            cols,
            &bys8,
            cols / 16,
            None,
            &out8,
            rows,
            rows,
            cols,
            None,
        )
        .unwrap();
    }
    dev.sync().unwrap();
    let s8 = t0.elapsed().as_secs_f64() / 20.0;
    println!(
        "gemv q4_k {rows}x{cols} C=8: {:.2} ms/run, {:.0} GB/s, {:.2} TFLOPS",
        s8 * 1e3,
        bytes as f64 / s8 / 1e9,
        (rows as f64 * cols as f64 * 2.0 * 8.0) / s8 / 1e12
    );
    }
}

/// The multi-column kernel is the prefill path; a layout slip here silently
/// degrades every prompt. Compare every column of every instantiation against
/// the CPU reference.
#[test]
fn gemv_multicolumn_matches_reference() {
    let Some(dev) = have_gpu() else { return };
    let k = Kernels::load(&dev).unwrap();
    for ty in [GgmlType::Q4K, GgmlType::Q6K] {
        // (columns, rows per warp): the single-column decode path uses 2 or 4
        // rows per warp for long projections, so every instantiation is checked.
        for (cols, rows_per_warp) in [(1usize, 1usize), (1, 2), (1, 4), (2, 1), (4, 1), (8, 1), (8, 2)] {
            let mut rng = Rng(0xfeed_face ^ (cols * 8 + rows_per_warp) as u64);
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
            let packed = if q6 {
                repack_q6k(&w_host)
            } else {
                w_host.clone()
            };
            let stride = if q6 { blocks * 224 } else { blocks * 144 };

            let y: Vec<f32> = (0..cols * kdim).map(|_| rng.f32() * 1.5).collect();
            let mut want = vec![0f32; rows * cols];
            let mut energy = vec![0f32; rows * cols];
            let mut wf = vec![0f32; kdim];
            for r in 0..rows {
                dequantize_row(ty, &w_host[r * row_bytes..(r + 1) * row_bytes], &mut wf);
                for c in 0..cols {
                    want[c * rows + r] = (0..kdim).map(|i| wf[i] * y[c * kdim + i]).sum();
                    energy[c * rows + r] = (0..kdim).map(|i| (wf[i] * y[c * kdim + i]).abs()).sum();
                }
            }

            let bw = dev.upload(&packed).unwrap();
            let by = dev.upload(as_bytes(&y)).unwrap();
            let byh = dev.alloc_zeroed(cols * kdim * 2).unwrap();
            let bys = dev.alloc_zeroed(cols * (kdim / 16) * 8).unwrap();
            let out = dev.alloc(rows * cols * 4).unwrap();
            k.prep(&dev, &by, 0, &byh, &bys, cols, kdim, None).unwrap();
            k.gemv_r(
                q6,
                cols,
                rows_per_warp,
                1,
                &dev,
                &bw,
                stride,
                &byh,
                kdim,
                &bys,
                kdim / 16,
                None,
                &out,
                rows,
                rows,
                kdim,
                None,
            )
            .unwrap();
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
            println!("{ty:?} cols={cols} rows={rows_per_warp} worst rel err {worst:.2e}");
            // q6_k accumulates 16 six-bit products in fp16; with cancellation in
            // random data that is worth a few 1e-3. Layout bugs are O(1).
            let tol = if q6 { 4e-3 } else { 2e-3 };
            assert!(worst < tol, "{ty:?} cols={cols} rows={rows_per_warp} rel err {worst}");
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
    let n_ctx = 192usize;
    let mut worst_all = 0f32;
    for (q_pos0, nq) in [
        (0usize, 2usize),
        (4, 2),
        (3, 2),
        (2, 4),
        (32, 8),
        (88, 8),
        (40, 1),
        (32, 13),
        (32, 32),
        (64, 13),
        (0, 48),
        (0, 64),
        (0, 96),
        (64, 35),
        (96, 33),
    ] {
        let kf: Vec<f32> = (0..n_ctx * n_kv * hd).map(|_| rng.f32()).collect();
        let vf: Vec<f32> = (0..n_ctx * n_kv * hd).map(|_| rng.f32()).collect();
        let kc: Vec<u16> = kf.iter().map(|x| f16::from_f32(*x).to_bits()).collect();
        let vc: Vec<u16> = vf.iter().map(|x| f16::from_f32(*x).to_bits()).collect();
        let q: Vec<f32> = (0..nq * n_heads * hd).map(|_| rng.f32()).collect();

        let bk = dev.upload(as_bytes(&kc)).unwrap();
        let bv = dev.upload(as_bytes(&vc)).unwrap();
        let bq = dev.upload(as_bytes(&q)).unwrap();
        let bo = dev.alloc(nq * n_heads * hd * 4).unwrap();
        k.attn_prefill(
            &dev,
            &bk,
            &bv,
            &bq,
            &bo,
            n_heads,
            n_kv,
            hd,
            q_pos0 + nq,
            qpk,
            q_pos0,
            nq,
            None,
        )
        .unwrap();
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
    let inv: Vec<f32> = (0..hd / 2)
        .map(|i| 1.0 / theta.powf(2.0 * i as f32 / hd as f32))
        .collect();

    let bq = dev.upload(as_bytes(&q)).unwrap();
    let bk = dev.upload(as_bytes(&kk)).unwrap();
    let bp = dev.upload(as_bytes(&pos)).unwrap();
    let bi = dev.upload(as_bytes(&inv)).unwrap();
    let vv: Vec<f32> = (0..n * n_k_heads * hd).map(|_| rng.f32()).collect();
    let bv = dev.upload(as_bytes(&vv)).unwrap();
    let cap = 8usize;
    let kc = dev.alloc_zeroed(cap * n_k_heads * hd * 2).unwrap();
    let vc = dev.alloc_zeroed(cap * n_k_heads * hd * 2).unwrap();

    k.rope(
        &dev, &bq, &bk, &bp, &bi, n_q_heads, n_k_heads, hd, n, 0, None,
    )
    .unwrap();
    let ctl = dev
        .upload(as_bytes(&[pos0 as i32, (pos0 + n) as i32]))
        .unwrap();
    k.write_kv(&dev, &kc, &vc, &bk, &bv, n_k_heads, hd, &ctl, n, None)
        .unwrap();
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
                worst = worst.max(
                    (gq[(t * n_q_heads + h) * hd + i] - (x0 * cs - x1 * sn)).abs()
                        / (x0 * cs - x1 * sn).abs().max(1.0),
                );
                worst = worst.max(
                    (gq[(t * n_q_heads + h) * hd + i + hd / 2] - (x1 * cs + x0 * sn)).abs()
                        / (x1 * cs + x0 * sn).abs().max(1.0),
                );
            }
        }
        for h in 0..n_k_heads {
            for i in 0..hd {
                let ang = (pos0 + t) as f32 * inv[i % (hd / 2)];
                let (sn, cs) = ang.sin_cos();
                let (x0, x1) = if i < hd / 2 {
                    (
                        kk[(t * n_k_heads + h) * hd + i],
                        kk[(t * n_k_heads + h) * hd + i + hd / 2],
                    )
                } else {
                    (
                        kk[(t * n_k_heads + h) * hd + i - hd / 2],
                        kk[(t * n_k_heads + h) * hd + i],
                    )
                };
                let want = f16::from_f32(if i < hd / 2 {
                    x0 * cs - x1 * sn
                } else {
                    x1 * cs + x0 * sn
                })
                .to_f32();
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
    let n_ctx = 192usize;
    let kv_elems = n_ctx * n_kv * hd;
    let kc: Vec<u16> = (0..kv_elems)
        .map(|_| f16::from_f32(rng.f32()).to_bits())
        .collect();
    let vc: Vec<u16> = (0..kv_elems)
        .map(|_| f16::from_f32(rng.f32()).to_bits())
        .collect();
    let q: Vec<f32> = (0..n_heads * hd).map(|_| rng.f32()).collect();

    for n_split in [1usize, 4, 16, 32] {
        let bk = dev.upload(as_bytes(&kc)).unwrap();
        let bv = dev.upload(as_bytes(&vc)).unwrap();
        let bq = dev.upload(as_bytes(&q)).unwrap();
        let bo = dev.alloc(n_heads * hd * 4).unwrap();
        let pm = dev.alloc(n_heads * n_split * 4).unwrap();
        let pl = dev.alloc(n_heads * n_split * 4).unwrap();
        let pa = dev.alloc(n_heads * n_split * hd * 4).unwrap();
        let ctl = dev.upload(as_bytes(&[0i32, n_ctx as i32])).unwrap();
        k.attn_decode(
            &dev, &bk, &bv, &bq, &bo, &pm, &pl, &pa, n_heads, n_kv, hd, &ctl, n_split, qpk, None,
        )
        .unwrap();
        k.attn_combine(&dev, &pm, &pl, &pa, &bo, n_heads, hd, n_split, None)
            .unwrap();
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
        assert!(
            worst < 5e-3,
            "attn_decode n_split={n_split} rel err {worst}"
        );
    }
}

/// The prefill GEMM dequantises each weight once into shared memory and reuses
/// it across 32 token columns. A layout or scale slip would still produce
/// fluent text, so every column (including a ragged tail) is compared against
/// the CPU dequantiser.
#[test]
fn gemm_matches_reference() {
    let Some(dev) = have_gpu() else { return };
    let k = Kernels::load(&dev).unwrap();
    for ty in [GgmlType::Q4K, GgmlType::Q6K] {
        for cols in [32usize, 7] {
            let mut rng = Rng(0x900d_5eed ^ cols as u64 ^ (ty == GgmlType::Q6K) as u64);
            let rows = 70usize; // 64 + tail
            let blocks = 74usize;
            let kdim = blocks * 256;
            let q6 = ty == GgmlType::Q6K;
            let per_block = if q6 { 210 } else { 144 };
            let mut w_host = vec![0u8; rows * blocks * per_block];
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
            let packed = if q6 {
                repack_q6k(&w_host)
            } else {
                w_host.clone()
            };
            let stride = blocks * if q6 { 224 } else { 144 };
            let row_bytes = blocks * per_block;

            let x: Vec<f32> = (0..cols * kdim).map(|_| rng.f32() * 1.5).collect();
            let bias: Vec<f32> = (0..rows).map(|_| rng.f32() * 0.1).collect();

            let mut wf = vec![0f32; kdim];
            let mut want = vec![0f32; rows * cols];
            let mut energy = vec![0f32; rows * cols];
            for r in 0..rows {
                dequantize_row(ty, &w_host[r * row_bytes..(r + 1) * row_bytes], &mut wf);
                for c in 0..cols {
                    let mut acc = 0f32;
                    let mut en = 0f32;
                    for i in 0..kdim {
                        let t = wf[i] * x[c * kdim + i];
                        acc += t;
                        en += t.abs();
                    }
                    want[c * rows + r] = acc + bias[r];
                    energy[c * rows + r] = en;
                }
            }

            let bw = dev.upload(&packed).unwrap();
            let bx = dev.upload(as_bytes(&x)).unwrap();
            let bb = dev.upload(as_bytes(&bias)).unwrap();
            let byh = dev.alloc_zeroed(cols * kdim * 2).unwrap();
            let bys = dev.alloc_zeroed(cols * (kdim / 16) * 8).unwrap();
            let out = dev.alloc(rows * cols * 4).unwrap();
            k.prep_col(&dev, &bx, 0, &byh, &bys, cols, kdim, None)
                .unwrap();
            let (f, tm, tn, tk, tt) = if q6 {
                (&k.gemm_q6k_n32t128, 64usize, 32usize, 128usize, 128usize)
            } else {
                (&k.gemm_q4k_n64t128, 64, 64, 64, 128)
            };
            for nsplit in [1usize, 2, 3] {
                k.gemm_tiled_pad(f, tm, tn, tk, tt, 4, &dev, &bw, stride, &byh, kdim, &bys,
                    kdim / 16, Some(&bb), &out, rows, rows, kdim, cols, nsplit, None)
                    .unwrap();
                trav_p100::kernels::last_error(&dev).unwrap();
                let mut got = vec![0f32; rows * cols];
                out.download_to(as_bytes_mut(&mut got)).unwrap();

                let mut worst = 0f32;
                for c in 0..cols {
                    for r in 0..rows {
                        let d = (got[c * rows + r] - want[c * rows + r]).abs()
                            / energy[c * rows + r].max(1e-6);
                        worst = worst.max(d);
                    }
                }
                println!("gemm {ty:?} cols={cols} split={nsplit} worst rel err {worst:.2e}");
                assert!(
                    worst < 2e-2,
                    "gemm {ty:?} cols={cols} split={nsplit} rel err {worst}"
                );
            }
        }
    }
}

/// Throughput of the prefill path, printed as TFLOPS/GB/s for regression checks.
#[test]
fn gemm_throughput() {
    let Some(dev) = have_gpu() else { return };
    let k = Kernels::load(&dev).unwrap();
    let (rows, cols) = (16384usize, 18944usize);
    let blocks = cols / 256;
    let row_bytes = blocks * 144;
    let mut rng = Rng(11);
    let mut host = vec![0u8; rows * row_bytes];
    for b in 0..rows * blocks {
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
    let w = dev.upload(&host).unwrap();
    let bn = 32usize;
    let x: Vec<f32> = (0..bn * cols).map(|_| rng.f32()).collect();
    let bx = dev.upload(as_bytes(&x)).unwrap();
    let byh = dev.alloc_zeroed(bn * cols * 2).unwrap();
    let bys = dev.alloc_zeroed(bn * (cols / 16) * 8).unwrap();
    let out = dev.alloc(rows * bn * 4).unwrap();

    println!(
        "gemm occupancy: {} blocks/SM, {} regs",
        k.gemm_q4k_n64t128
            .occupancy_blocks(128, (32 * (64 + 68)) * 4)
            .unwrap_or(-1),
        k.gemm_q4k_n64t128.num_regs().unwrap_or(-1)
    );
    let run = |iters: usize| {
        for _ in 0..iters {
            k.prep_col(&dev, &bx, 0, &byh, &bys, bn, cols, None)
                .unwrap();
            k.gemm_tiled(&k.gemm_q4k_n64t128, 64, 64, 64, 128, &dev, &w, row_bytes, &byh,
                cols, &bys, cols / 16, None, &out, rows, rows, cols, bn, 1, None)
                .unwrap();
        }
        dev.sync().unwrap();
    };
    run(3);
    let t0 = std::time::Instant::now();
    run(10);
    let s = t0.elapsed().as_secs_f64() / 10.0;
    println!(
        "gemm q4_k {rows}x{cols} N=32: {:.2} ms/run, {:.0} GB/s, {:.2} TFLOPS ({:.0} tok/s)",
        s * 1e3,
        rows as f64 * row_bytes as f64 / s / 1e9,
        rows as f64 * cols as f64 * 2.0 * bn as f64 / s / 1e12,
        bn as f64 / s
    );
}

/// Calibration: how fast can this card actually issue HFMA2 on its own?
#[test]
fn hfma2_peak() {
    let Some(dev) = have_gpu() else { return };
    let k = Kernels::load(&dev).unwrap();
    let out = dev.alloc(4).unwrap();
    for (blocks, threads) in [(56u32 * 4, 256u32), (56, 128), (56, 64), (56 * 2, 128)] {
        let iters = 200_000i32;
        k.hfma2_bench(&dev, &out, 1000, blocks, threads).unwrap();
        dev.sync().unwrap();
        let t0 = std::time::Instant::now();
        k.hfma2_bench(&dev, &out, iters, blocks, threads).unwrap();
        dev.sync().unwrap();
        let s = t0.elapsed().as_secs_f64();
        let flops = blocks as f64 * threads as f64 * iters as f64 * 16.0 * 4.0;
        println!(
            "HFMA2 {blocks} blocks x {threads}: {:.2} TFLOPS",
            flops / s / 1e12
        );
    }
}

/// Token embedding rows share the q4_k dequant path with the GEMV but write
/// values straight into the residual stream; a nibble-order slip here would
/// change every layer while still producing fluent text.
#[test]
fn embed_q4k_matches_reference() {
    let Some(dev) = have_gpu() else { return };
    let k = Kernels::load(&dev).unwrap();
    let mut rng = Rng(0xe0b3d);
    let rows = 8usize;
    let blocks = 2usize;
    let kdim = blocks * 256;
    let mut w_host = vec![0u8; rows * blocks * 144];
    for b in 0..rows * blocks {
        let blk = &mut w_host[b * 144..(b + 1) * 144];
        blk[0..2].copy_from_slice(&f16::from_f32(0.02).to_bits().to_le_bytes());
        blk[2..4].copy_from_slice(&f16::from_f32(0.01).to_bits().to_le_bytes());
        blk[4..16].copy_from_slice(&rng.bytes(12));
        blk[16..144].copy_from_slice(&rng.bytes(128));
    }
    let ids: Vec<i32> = vec![7, 3, 0];
    let mut wf = vec![0f32; kdim];
    let mut want = vec![0f32; ids.len() * kdim];
    for (t, &id) in ids.iter().enumerate() {
        let r = id as usize;
        dequantize_row(
            GgmlType::Q4K,
            &w_host[r * blocks * 144..(r + 1) * blocks * 144],
            &mut wf,
        );
        want[t * kdim..(t + 1) * kdim].copy_from_slice(&wf);
    }
    let bw = dev.upload(&w_host).unwrap();
    let bi = dev.upload(as_bytes(&ids)).unwrap();
    let bo = dev.alloc(ids.len() * kdim * 4).unwrap();
    k.embed(&dev, &bw, blocks * 144, &bi, &bo, ids.len(), kdim, None)
        .unwrap();
    trav_p100::kernels::last_error(&dev).unwrap();
    let mut got = vec![0f32; ids.len() * kdim];
    bo.download_to(as_bytes_mut(&mut got)).unwrap();
    let mut worst = 0f32;
    for i in 0..got.len() {
        worst = worst.max((got[i] - want[i]).abs() / want[i].abs().max(1.0));
    }
    println!("embed_q4k worst rel err {worst:.2e}");
    assert!(worst < 1e-5, "embed rel err {worst}");
}

/// Per-launch cost: time a trivial kernel launch in a tight loop. Decode runs
/// ~536 launches per token, so this number decides whether launch overhead or
/// weight streaming dominates.
#[test]
fn launch_overhead() {
    let Some(dev) = have_gpu() else { return };
    let k = Kernels::load(&dev).unwrap();
    // Small enough that the kernel itself is ~0: 256 elements, one block.
    let n = 256usize;
    let ah = vec![0f32; n];
    let a = dev.upload(as_bytes(&ah)).unwrap();
    let b = dev.alloc(n * 4).unwrap();
    let run = |m: usize| {
        for _ in 0..m {
            k.add_inplace(&dev, &a, &b, n, None).unwrap();
        }
        dev.sync().unwrap();
    };
    run(20);
    let t = std::time::Instant::now();
    run(2000);
    let per = t.elapsed().as_secs_f64() / 2000.0;
    println!("launch overhead {:.3} us/launch", per * 1e6);
}

/// The fp16 prefill path materialises weights in a scratch buffer; the mirror
/// must be the same numbers the GGUF dequantiser produces (fp16 rounding only).
#[test]
fn dequant_f16_matches_gguf() {
    let Some(dev) = have_gpu() else { return };
    let k = Kernels::load(&dev).unwrap();
    for ty in [GgmlType::Q4K, GgmlType::Q6K] {
        let q6 = ty == GgmlType::Q6K;
        let (rows, blocks) = (3usize, 14usize);
        let per_block = if q6 { 210 } else { 144 };
        let kdim = blocks * 256;
        let mut rng = Rng(0xd0d0 + q6 as u64);
        let mut w_host = vec![0u8; rows * blocks * per_block];
        for b in 0..rows * blocks {
            let blk = &mut w_host[b * per_block..(b + 1) * per_block];
            if q6 {
                for i in 0..208 {
                    blk[i] = rng.u32() as u8;
                }
                blk[208..210].copy_from_slice(&f16::from_f32(0.005).to_bits().to_le_bytes());
                for i in 0..16 {
                    blk[192 + i] = ((rng.u32() % 64) as i32 - 32) as i8 as u8;
                }
            } else {
                blk[0..2].copy_from_slice(&f16::from_f32(0.02).to_bits().to_le_bytes());
                blk[2..4].copy_from_slice(&f16::from_f32(0.01).to_bits().to_le_bytes());
                for i in 4..144 {
                    blk[i] = rng.u32() as u8;
                }
            }
        }
        let packed = if q6 { repack_q6k(&w_host) } else { w_host.clone() };
        let stride = blocks * if q6 { 224 } else { 144 };
        let bw = dev.upload(&packed).unwrap();
        let w16 = dev.alloc_zeroed(rows * kdim * 2).unwrap();
        k.dequant_f16(q6, &dev, &bw, stride, &w16, rows, kdim, None).unwrap();
        trav_p100::kernels::last_error(&dev).unwrap();
        let mut hw = vec![0u8; rows * kdim * 2];
        w16.download_to(as_bytes_mut(&mut hw)).unwrap();
        let mut wf = vec![0f32; kdim];
        let mut worst = 0f32;
        let mut worst_at = (0usize, 0usize);
        for r in 0..rows {
            dequantize_row(
                ty,
                &w_host[r * blocks * per_block..(r + 1) * blocks * per_block],
                &mut wf,
            );
            for i in 0..kdim {
                let got = f16::from_bits(u16::from_le_bytes([
                    hw[(r * kdim + i) * 2],
                    hw[(r * kdim + i) * 2 + 1],
                ]))
                .to_f32();
                let d = (got - wf[i]).abs();
                if d > worst {
                    worst = d;
                    worst_at = (r, i);
                }
            }
        }
        println!("dequant {ty:?}: worst |diff| {worst:.2e} at {worst_at:?}");
        assert!(worst < 2e-2, "dequant {ty:?} off by {worst}");
    }
}

/// fp16 GEMM against a dense fp64 dot of the same mirrored weights, for the two
/// production shapes the prefill path sends through it.
#[test]
fn gemm_f16_matches_dense() {
    let Some(dev) = have_gpu() else { return };
    let k = Kernels::load(&dev).unwrap();
    for &(rows, kdim, q6, cols) in &[(3584usize, 3584usize, false, 128usize), (3584, 18944, true, 128)] {
        let blocks = kdim / 256;
        let per_block = if q6 { 210 } else { 144 };
        let stride = blocks * if q6 { 224 } else { 144 };
        let mut rng = Rng(0x1f16 + rows as u64 * 7 + q6 as u64);
        let mut w_host = vec![0u8; rows * blocks * per_block];
        for b in 0..rows * blocks {
            let blk = &mut w_host[b * per_block..(b + 1) * per_block];
            if q6 {
                for i in 0..208 {
                    blk[i] = rng.u32() as u8;
                }
                blk[208..210].copy_from_slice(&f16::from_f32(0.005).to_bits().to_le_bytes());
                for i in 0..16 {
                    blk[192 + i] = ((rng.u32() % 64) as i32 - 32) as i8 as u8;
                }
            } else {
                blk[0..2].copy_from_slice(&f16::from_f32(0.02).to_bits().to_le_bytes());
                blk[2..4].copy_from_slice(&f16::from_f32(0.01).to_bits().to_le_bytes());
                for i in 4..144 {
                    blk[i] = rng.u32() as u8;
                }
            }
        }
        let packed = if q6 { repack_q6k(&w_host) } else { w_host.clone() };
        let bw = dev.upload(&packed).unwrap();
        let x: Vec<f32> = (0..cols * kdim).map(|_| rng.f32() * 1.5).collect();
        let bx = dev.upload(as_bytes(&x)).unwrap();
        let byh = dev.alloc_zeroed(cols * kdim * 2).unwrap();
        let bys = dev.alloc_zeroed(cols * (kdim / 16) * 8).unwrap();
        let out = dev.alloc_zeroed(rows * cols * 4).unwrap();
        let w16 = dev.alloc_zeroed(rows * kdim * 2).unwrap();
        k.prep_col(&dev, &bx, 0, &byh, &bys, cols, kdim, None).unwrap();
        k.dequant_f16(q6, &dev, &bw, stride, &w16, rows, kdim, None).unwrap();
        k.gemm_f16(&dev, &w16, kdim, &byh, kdim, &bys, kdim / 16, None, &out, rows, rows,
            kdim, cols, None).unwrap();
        trav_p100::kernels::last_error(&dev).unwrap();
        let mut got = vec![0f32; rows * cols];
        out.download_to(as_bytes_mut(&mut got)).unwrap();
        let mut hw = vec![0u8; rows * kdim * 2];
        w16.download_to(as_bytes_mut(&mut hw)).unwrap();
        // every 37th row and a few columns is enough to catch layout slips
        // The fp16 input rounding and the 128-value fp16 chains show up as a
        // fraction of the term energy, which is how the other GEMM tests and
        // the design notes measure accuracy.
        let mut worst = 0f32;
        for c in [0usize, cols / 3, cols - 1] {
            for r in (0..rows).step_by(37) {
                let mut acc = 0f64;
                let mut en = 0f32;
                for i in 0..kdim {
                    let wv = f16::from_bits(u16::from_le_bytes([
                        hw[(r * kdim + i) * 2],
                        hw[(r * kdim + i) * 2 + 1],
                    ]))
                    .to_f32();
                    let t = wv * x[c * kdim + i];
                    acc += t as f64;
                    en += t.abs();
                }
                worst = worst.max((got[c * rows + r] - acc as f32).abs() / en.max(1e-6));
            }
        }
        println!("gemm_f16 {rows}x{kdim} q6={q6}: worst rel err vs term energy {worst:.2e}");
        assert!(worst < 2e-2, "gemm_f16 {rows}x{kdim} rel err {worst}");
    }
}

/// Throughput of the fp16 prefill GEMM on the gate shape — the one that dominates
/// prefill — as a single number to compare tile/loop variants against.
#[test]
fn gemm_f16_rate() {
    let Some(dev) = have_gpu() else { return };
    let k = Kernels::load(&dev).unwrap();
    let (rows, kdim, cols) = (18944usize, 3584usize, 512usize);
    let blocks = kdim / 256;
    let mut rng = Rng(0x9a9a);
    let mut w_host = vec![0u8; rows * blocks * 144];
    for b in 0..rows * blocks {
        let blk = &mut w_host[b * 144..(b + 1) * 144];
        blk[0..2].copy_from_slice(&f16::from_f32(0.02).to_bits().to_le_bytes());
        blk[2..4].copy_from_slice(&f16::from_f32(0.01).to_bits().to_le_bytes());
        for i in 4..144 {
            blk[i] = rng.u32() as u8;
        }
    }
    let bw = dev.upload(&w_host).unwrap();
    let x: Vec<f32> = (0..cols * kdim).map(|_| rng.f32() * 1.5).collect();
    let bx = dev.upload(as_bytes(&x)).unwrap();
    let byh = dev.alloc_zeroed(cols * kdim * 2).unwrap();
    let bys = dev.alloc_zeroed(cols * (kdim / 16) * 8).unwrap();
    let w16 = dev.alloc_zeroed(rows * kdim * 2).unwrap();
    let out = dev.alloc_zeroed(rows * cols * 4).unwrap();
    k.prep_col(&dev, &bx, 0, &byh, &bys, cols, kdim, None).unwrap();
    k.dequant_f16(false, &dev, &bw, blocks * 144, &w16, rows, kdim, None).unwrap();
    let mut best = f64::MAX;
    for _ in 0..10 {
        let t0 = std::time::Instant::now();
        k.gemm_f16(&dev, &w16, kdim, &byh, kdim, &bys, kdim / 16, None, &out, rows, rows,
            kdim, cols, None).unwrap();
        dev.sync().unwrap();
        best = best.min(t0.elapsed().as_secs_f64());
    }
    println!(
        "gemm_f16 gate {rows}x{kdim} cols={cols}: {:.2} ms, {:.1} TFLOPS",
        best * 1e3,
        2.0 * rows as f64 * kdim as f64 * cols as f64 / best / 1e12
    );
}

/// Single-column GEMV sweep over rows-per-warp and split-K, on the shapes the
/// model launches (q4_k attention/ffn plus q6_k down and lm_head). Split-K is
/// included because it looks like the fix for the 3584-row shapes; it is not.
#[test]
fn gemv_split_bandwidth() {
    let Some(dev) = have_gpu() else { return };
    let k = Kernels::load(&dev).unwrap();
    let shapes = [
        ("q/o   ", 3584usize, 3584usize, false),
        ("k/v   ", 512, 3584, false),
        ("gate  ", 18944, 3584, false),
        ("down  ", 3584, 18944, true),
        ("lm_head", 152064, 3584, true),
    ];
    let run = |rows: usize, cols: usize, q6: bool, rpw: usize, ns: usize, iters: usize,
               w: &trav_p100::Buf, byh: &trav_p100::Buf, bys: &trav_p100::Buf,
               out: &trav_p100::Buf, stride: usize| {
        for _ in 0..iters {
            k.gemv_r(q6, 1, rpw, ns, &dev, w, stride, byh, cols, bys, cols / 16, None, out,
                rows, rows, cols, None).unwrap();
        }
    };
    // warm-up: 100 ms of the 18944-row shape so clocks are up before timing
    {
        let (rows, cols) = (18944usize, 3584usize);
        let w = dev.alloc(rows * (cols / 256) * 144).unwrap();
        let byh = dev.alloc(cols * 2).unwrap();
        let bys = dev.alloc((cols / 16) * 8).unwrap();
        let out = dev.alloc(rows * 4).unwrap();
        run(rows, cols, false, 8, 1, 800, &w, &byh, &bys, &out, (cols / 256) * 144);
        dev.sync().unwrap();
    }
    for &(name, rows, cols, q6) in &shapes {
        let blocks_k = cols / 256;
        let per_block = if q6 { 210 } else { 144 };
        let stride = blocks_k * if q6 { 224 } else { 144 };
        let mut rng = Rng(0x5151 + rows as u64);
        let mut host = vec![0u8; rows * blocks_k * per_block];
        for b in 0..rows * blocks_k {
            let blk = &mut host[b * per_block..(b + 1) * per_block];
            if q6 {
                for i in 0..208 {
                    blk[i] = rng.u32() as u8;
                }
                blk[208..210].copy_from_slice(&f16::from_f32(0.005).to_bits().to_le_bytes());
                for i in 0..16 {
                    blk[192 + i] = ((rng.u32() % 64) as i32 - 32) as i8 as u8;
                }
            } else {
                blk[0..2].copy_from_slice(&f16::from_f32(0.02).to_bits().to_le_bytes());
                blk[2..4].copy_from_slice(&f16::from_f32(0.01).to_bits().to_le_bytes());
                for i in 4..144 {
                    blk[i] = rng.u32() as u8;
                }
            }
        }
        let packed = if q6 { repack_q6k(&host) } else { host.clone() };
        drop(host);
        let y: Vec<f32> = (0..cols).map(|_| rng.f32()).collect();
        let w = dev.upload(&packed).unwrap();
        drop(packed);
        let by = dev.upload(as_bytes(&y)).unwrap();
        let byh = dev.alloc(cols * 2).unwrap();
        let bys = dev.alloc((cols / 16) * 8).unwrap();
        let out = dev.alloc(rows * 4).unwrap();
        k.prep(&dev, &by, 0, &byh, &bys, 1, cols, None).unwrap();
        let bytes = rows * blocks_k * per_block;
        let runs = 20usize;
        let mut line = format!("{name} {rows:>6}x{cols} q6={}  ", q6 as u8);
        for &(rpw, ns) in &[
            (1usize, 1usize),
            (2, 1),
            (4, 1),
            (8, 1),
            (2, 2),
            (4, 2),
            (8, 2),
            (2, 4),
        ] {
            if rows.div_ceil(8 * rpw) * ns > 8192 {
                continue;
            }
            run(rows, cols, q6, rpw, ns, 4, &w, &byh, &bys, &out, stride);
            dev.sync().unwrap();
            let t0 = std::time::Instant::now();
            run(rows, cols, q6, rpw, ns, runs, &w, &byh, &bys, &out, stride);
            dev.sync().unwrap();
            let s = t0.elapsed().as_secs_f64() / runs as f64;
            line += &format!("r{rpw}s{ns}:{:.0} ", bytes as f64 / s / 1e9);
        }
        println!("{line}");
    }
}

#[test]
fn greedy_argmax_matches_host() {
    let Some(dev) = have_gpu() else { return };
    let kernels = Kernels::load(&dev).unwrap();
    use trav_p100::kernels::{ARGMAX_TOKEN_OFFSET, GREEDY_SCRATCH_BYTES};
    let scratch = dev.alloc(GREEDY_SCRATCH_BYTES).unwrap();
    let vocab = 152_064;
    let mut logits = vec![-10.0f32; vocab + 1];
    for (first, value, expected) in [
        (-10.0, 9.0, vocab / 2),
        (-10.0, -0.0, vocab / 2),
        (f32::NAN, f32::INFINITY, 0),
        (f32::NEG_INFINITY, f32::INFINITY, vocab / 2),
    ] {
        logits[0] = f32::INFINITY; // outside the slice passed to the kernel
        logits[1] = first;
        logits[2] = f32::NAN;
        logits[1 + vocab / 2] = value;
        logits[vocab] = if value == 0.0 { 0.0 } else { value };
        let device_logits = dev.upload(as_bytes(&logits)).unwrap();
        kernels.argmax(&device_logits, 4, vocab, &scratch).unwrap();
        let mut token = [0u32];
        scratch.download_at(ARGMAX_TOKEN_OFFSET, as_bytes_mut(&mut token)).unwrap();
        assert_eq!(token[0] as usize, expected);
    }
}

#[test]
fn greedy_repetition_penalty_matches_host() {
    let Some(dev) = have_gpu() else { return };
    let kernels = Kernels::load(&dev).unwrap();
    let scratch = dev.alloc(trav_p100::kernels::GREEDY_SCRATCH_BYTES).unwrap();
    let logits = [9.0f32, 4.0, -3.0, 8.0, 0.0];
    let tokens = [0u32, 2, 2, 0, 0, 4, 99, 1];
    let penalty = 1.05;
    let mut expected = logits;
    for id in tokens {
        if let Some(v) = expected.get_mut(id as usize) {
            *v = if *v > 0.0 { *v / penalty } else { *v * penalty };
        }
    }
    let device_logits = dev.upload(as_bytes(&logits)).unwrap();
    kernels.penalize_recent(&device_logits, 0, logits.len(), &tokens, penalty, &scratch).unwrap();
    let mut got = [0.0f32; 5];
    device_logits.download_to(as_bytes_mut(&mut got)).unwrap();
    assert_eq!(got.map(f32::to_bits), expected.map(f32::to_bits));
}

#[test]
fn gemm_f16_handles_partial_tiles_and_unpaired_stage() {
    let Some(dev) = have_gpu() else { return };
    let kernels = Kernels::load(&dev).unwrap();
    let (rows, cols, k) = (529usize, 145usize, 96usize);
    let stride = rows.next_multiple_of(4);
    let weight = |row: usize| (row % 7) as f32 - 3.0;
    let activation = |col: usize| (col % 5) as f32 - 2.0;
    let w: Vec<f16> = (0..rows * k).map(|i| f16::from_f32(weight(i / k))).collect();
    let y: Vec<f16> = (0..cols * k).map(|i| f16::from_f32(activation(i / k))).collect();
    let ys_stride = k / 16;
    let ys = vec![0.5f32; cols * ys_stride * 2];
    let bias: Vec<f32> = (0..rows).map(|row| row as f32 / 8.0).collect();
    let bw = dev.upload(as_bytes(&w)).unwrap();
    let by = dev.upload(as_bytes(&y)).unwrap();
    let bys = dev.upload(as_bytes(&ys)).unwrap();
    let bb = dev.upload(as_bytes(&bias)).unwrap();
    let out = dev.alloc(cols * stride * 4).unwrap();
    kernels.gemm_f16(&dev, &bw, k, &by, k, &bys, ys_stride, Some(&bb), &out, stride,
                     rows, k, cols, None).unwrap();
    let mut got = vec![0.0f32; cols * stride];
    out.download_to(as_bytes_mut(&mut got)).unwrap();
    for col in 0..cols {
        for row in 0..rows {
            let expected = weight(row) * activation(col) * k as f32 * 0.5 + bias[row];
            assert_eq!(got[col * stride + row], expected, "row {row}, column {col}");
        }
    }
}

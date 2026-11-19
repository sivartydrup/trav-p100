//! GGML tensor types and the reference (CPU) dequantisers.
//!
//! The block layouts here are byte-for-byte the ones written by `gguf`/`ggml`; the
//! GPU loader and the CUDA kernels consume the same bytes, so this module doubles
//! as the executable specification the kernels are tested against.

use half::{bf16, f16};

/// GGML tensor type ids as stored in GGUF (`ggml_type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    IQ2XXS,
    IQ2XS,
    IQ3XXS,
    IQ1S,
    IQ4NL,
    IQ3S,
    IQ2S,
    IQ4XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    BF16,
    Unknown(u32),
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Self {
        use GgmlType::*;
        match v {
            0 => F32,
            1 => F16,
            2 => Q4_0,
            3 => Q4_1,
            6 => Q5_0,
            7 => Q5_1,
            8 => Q8_0,
            9 => Q8_1,
            10 => Q2K,
            11 => Q3K,
            12 => Q4K,
            13 => Q5K,
            14 => Q6K,
            15 => Q8K,
            16 => IQ2XXS,
            17 => IQ2XS,
            18 => IQ3XXS,
            19 => IQ1S,
            20 => IQ4NL,
            21 => IQ3S,
            22 => IQ2S,
            23 => IQ4XS,
            24 => I8,
            25 => I16,
            26 => I32,
            27 => I64,
            28 => F64,
            30 => BF16,
            other => Unknown(other),
        }
    }

    pub fn as_u32(self) -> u32 {
        use GgmlType::*;
        match self {
            F32 => 0,
            F16 => 1,
            Q4_0 => 2,
            Q4_1 => 3,
            Q5_0 => 6,
            Q5_1 => 7,
            Q8_0 => 8,
            Q8_1 => 9,
            Q2K => 10,
            Q3K => 11,
            Q4K => 12,
            Q5K => 13,
            Q6K => 14,
            Q8K => 15,
            IQ2XXS => 16,
            IQ2XS => 17,
            IQ3XXS => 18,
            IQ1S => 19,
            IQ4NL => 20,
            IQ3S => 21,
            IQ2S => 22,
            IQ4XS => 23,
            I8 => 24,
            I16 => 25,
            I32 => 26,
            I64 => 27,
            F64 => 28,
            BF16 => 30,
            Unknown(v) => v,
        }
    }

    pub fn name(self) -> &'static str {
        use GgmlType::*;
        match self {
            F32 => "f32",
            F16 => "f16",
            Q4_0 => "q4_0",
            Q4_1 => "q4_1",
            Q5_0 => "q5_0",
            Q5_1 => "q5_1",
            Q8_0 => "q8_0",
            Q8_1 => "q8_1",
            Q2K => "q2_k",
            Q3K => "q3_k",
            Q4K => "q4_k",
            Q5K => "q5_k",
            Q6K => "q6_k",
            Q8K => "q8_k",
            IQ2XXS => "iq2_xxs",
            IQ2XS => "iq2_xs",
            IQ3XXS => "iq3_xxs",
            IQ1S => "iq1_s",
            IQ4NL => "iq4_nl",
            IQ3S => "iq3_s",
            IQ2S => "iq2_s",
            IQ4XS => "iq4_xs",
            I8 => "i8",
            I16 => "i16",
            I32 => "i32",
            I64 => "i64",
            F64 => "f64",
            BF16 => "bf16",
            Unknown(_) => "unknown",
        }
    }

    /// Values per block (`1` for unquantised types).
    pub fn block_size(self) -> usize {
        use GgmlType::*;
        match self {
            F32 | F16 | BF16 | I8 | I16 | I32 | I64 | F64 => 1,
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_1 | IQ4NL => 32,
            Q2K | Q3K | Q4K | Q5K | Q6K | Q8K | IQ2XXS | IQ2XS | IQ3XXS | IQ1S | IQ3S
            | IQ2S | IQ4XS => 256,
            Unknown(_) => 1,
        }
    }

    /// Bytes per block (`1` for unquantised types is wrong on purpose: use
    /// [`GgmlType::type_size`] only for block types).
    pub fn type_size(self) -> usize {
        use GgmlType::*;
        match self {
            F32 | I32 => 4,
            F16 | BF16 | I16 => 2,
            F64 | I64 => 8,
            I8 => 1,
            Q4_0 | IQ4NL => 18,
            Q4_1 => 20,
            Q5_0 => 22,
            Q5_1 => 24,
            Q8_0 => 34,
            Q8_1 => 36,
            Q2K => 84,
            Q3K => 110,
            Q4K => 144,
            Q5K => 176,
            Q6K => 210,
            Q8K => 292,
            IQ2XXS => 66,
            IQ2XS => 74,
            IQ3XXS => 98,
            IQ1S => 50,
            IQ3S => 110,
            IQ2S => 82,
            IQ4XS => 136,
            Unknown(_) => 1,
        }
    }

    /// Bytes required to store `n` values of this type.
    pub fn row_bytes(self, n: usize) -> usize {
        let bs = self.block_size();
        assert!(
            n % bs == 0,
            "{}: {n} values is not a multiple of the block size {bs}",
            self.name()
        );
        n / bs * self.type_size()
    }

    pub fn is_quantized(self) -> bool {
        self.block_size() > 1
    }
}

/// `get_scale_min_k4` from ggml: unpack the 6-bit scale and min of sub-block `j`.
#[inline]
pub fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    debug_assert!(j < 8 && q.len() >= 12);
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

#[inline]
fn h(v: &[u8], off: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([v[off], v[off + 1]])).to_f32()
}

/// Dequantise one row of `ty` (exactly `out.len()` values) into `out`.
///
/// # Panics
/// If `bytes` is too short or `out.len()` is not a multiple of the block size.
pub fn dequantize_row(ty: GgmlType, bytes: &[u8], out: &mut [f32]) {
    match ty {
        GgmlType::F32 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
            }
        }
        GgmlType::F16 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = h(bytes, i * 2);
            }
        }
        GgmlType::BF16 => {
            for (i, o) in out.iter_mut().enumerate() {
                let b = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                *o = bf16::from_bits(b).to_f32();
            }
        }
        GgmlType::Q8_0 => {
            for (b, chunk) in out.chunks_exact_mut(32).enumerate() {
                let blk = &bytes[b * 34..b * 34 + 34];
                let d = h(blk, 0);
                for (i, o) in chunk.iter_mut().enumerate() {
                    *o = d * blk[2 + i] as i8 as f32;
                }
            }
        }
        GgmlType::Q4_0 => {
            for (b, chunk) in out.chunks_exact_mut(32).enumerate() {
                let blk = &bytes[b * 18..b * 18 + 18];
                let d = h(blk, 0);
                for i in 0..16 {
                    let q = blk[2 + i];
                    chunk[i] = d * ((q & 0xF) as i32 - 8) as f32;
                    chunk[i + 16] = d * ((q >> 4) as i32 - 8) as f32;
                }
            }
        }
        GgmlType::Q5_0 => {
            for (b, chunk) in out.chunks_exact_mut(32).enumerate() {
                let blk = &bytes[b * 22..b * 22 + 22];
                let d = h(blk, 0);
                let qh = u32::from_le_bytes(blk[2..6].try_into().unwrap());
                for i in 0..16 {
                    let q = blk[6 + i];
                    let x0 = ((q & 0xF) as u32 | ((qh >> i) & 1) << 4) as i32 - 16;
                    let x1 = ((q >> 4) as u32 | ((qh >> (i + 16)) & 1) << 4) as i32 - 16;
                    chunk[i] = d * x0 as f32;
                    chunk[i + 16] = d * x1 as f32;
                }
            }
        }
        GgmlType::Q4K => {
            for (b, chunk) in out.chunks_exact_mut(256).enumerate() {
                let blk = &bytes[b * 144..b * 144 + 144];
                let d = h(blk, 0);
                let min = h(blk, 2);
                let scales = &blk[4..16];
                let qs = &blk[16..144];
                let mut is = 0usize;
                let mut qoff = 0usize;
                for c in 0..4 {
                    let (sc0, m0) = get_scale_min_k4(is, scales);
                    let (sc1, m1) = get_scale_min_k4(is + 1, scales);
                    let d1 = d * sc0 as f32;
                    let m1f = min * m0 as f32;
                    let d2 = d * sc1 as f32;
                    let m2f = min * m1 as f32;
                    let base = c * 64;
                    for l in 0..32 {
                        let q = qs[qoff + l];
                        chunk[base + l] = d1 * (q & 0xF) as f32 - m1f;
                        chunk[base + 32 + l] = d2 * (q >> 4) as f32 - m2f;
                    }
                    qoff += 32;
                    is += 2;
                }
            }
        }
        GgmlType::Q5K => {
            for (b, chunk) in out.chunks_exact_mut(256).enumerate() {
                let blk = &bytes[b * 176..b * 176 + 176];
                let d = h(blk, 0);
                let min = h(blk, 2);
                let scales = &blk[4..16];
                let qh = &blk[16..48];
                let qs = &blk[48..176];
                let mut is = 0usize;
                let mut qoff = 0usize;
                let mut u1: u8 = 1;
                let mut u2: u8 = 2;
                for c in 0..4 {
                    let (sc0, m0) = get_scale_min_k4(is, scales);
                    let (sc1, m1) = get_scale_min_k4(is + 1, scales);
                    let d1 = d * sc0 as f32;
                    let m1f = min * m0 as f32;
                    let d2 = d * sc1 as f32;
                    let m2f = min * m1 as f32;
                    let base = c * 64;
                    for l in 0..32 {
                        let q = qs[qoff + l];
                        let hi0 = if qh[l] & u1 != 0 { 16 } else { 0 };
                        let hi1 = if qh[l] & u2 != 0 { 16 } else { 0 };
                        chunk[base + l] = d1 * ((q & 0xF) + hi0) as f32 - m1f;
                        chunk[base + 32 + l] = d2 * ((q >> 4) + hi1) as f32 - m2f;
                    }
                    qoff += 32;
                    is += 2;
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
        }
        GgmlType::Q6K => {
            for (b, chunk) in out.chunks_exact_mut(256).enumerate() {
                let blk = &bytes[b * 210..b * 210 + 210];
                let ql = &blk[0..128];
                let qh = &blk[128..192];
                let sc = &blk[192..208];
                let d = h(blk, 208);
                let mut y = 0usize;
                let (mut qlo, mut qho, mut sco) = (0usize, 0usize, 0usize);
                for _n in 0..2 {
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[qlo + l] & 0xF) | (((qh[qho + l] >> 0) & 3) << 4)) as i8 - 32;
                        let q2 = ((ql[qlo + l + 32] & 0xF) | (((qh[qho + l] >> 2) & 3) << 4)) as i8 - 32;
                        let q3 = ((ql[qlo + l] >> 4) | (((qh[qho + l] >> 4) & 3) << 4)) as i8 - 32;
                        let q4 = ((ql[qlo + l + 32] >> 4) | (((qh[qho + l] >> 6) & 3) << 4)) as i8 - 32;
                        chunk[y + l] = d * sc[sco + is] as i8 as f32 * q1 as f32;
                        chunk[y + l + 32] = d * sc[sco + is + 2] as i8 as f32 * q2 as f32;
                        chunk[y + l + 64] = d * sc[sco + is + 4] as i8 as f32 * q3 as f32;
                        chunk[y + l + 96] = d * sc[sco + is + 6] as i8 as f32 * q4 as f32;
                    }
                    y += 128;
                    qlo += 64;
                    qho += 32;
                    sco += 8;
                }
            }
        }
        GgmlType::Q8K => {
            for (b, chunk) in out.chunks_exact_mut(256).enumerate() {
                let blk = &bytes[b * 292..b * 292 + 292];
                let d = f32::from_le_bytes(blk[0..4].try_into().unwrap());
                for i in 0..256 {
                    chunk[i] = d * blk[4 + i] as i8 as f32;
                }
            }
        }
        other => panic!("dequantize_row: unsupported type {}", other.name()),
    }
}

/// Dequantise a tensor of `ty` with `n` values into `out` (length `n`).
pub fn dequantize(ty: GgmlType, bytes: &[u8], out: &mut [f32]) {
    assert!(out.len() % ty.block_size() == 0);
    dequantize_row(ty, bytes, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip_f16(v: f32) -> u16 {
        f16::from_f32(v).to_bits()
    }

    #[test]
    fn q4k_single_block_matches_reference() {
        // d = 1/64, dmin = 1/128, all scales/mins distinct, nibbles 0..15.
        let mut blk = vec![0u8; 144];
        blk[0..2].copy_from_slice(&roundtrip_f16(1.0 / 64.0).to_le_bytes());
        blk[2..4].copy_from_slice(&roundtrip_f16(1.0 / 128.0).to_le_bytes());
        for i in 0..12 {
            blk[4 + i] = 0x21 * (i as u8 + 1) & 0x3F;
        }
        for i in 0..128 {
            blk[16 + i] = ((i as u8) & 0xF) | (((i as u8 + 1) & 0xF) << 4);
        }
        let mut out = vec![0.0f32; 256];
        dequantize_row(GgmlType::Q4K, &blk, &mut out);
        let d = 1.0 / 64.0;
        let min = 1.0 / 128.0;
        for c in 0..4 {
            let (sc0, m0) = get_scale_min_k4(2 * c, &blk[4..16]);
            let (sc1, m1) = get_scale_min_k4(2 * c + 1, &blk[4..16]);
            for l in 0..32 {
                let q = blk[16 + c * 32 + l];
                let want0 = d * sc0 as f32 * (q & 0xF) as f32 - min * m0 as f32;
                let want1 = d * sc1 as f32 * (q >> 4) as f32 - min * m1 as f32;
                assert!((out[c * 64 + l] - want0).abs() < 1e-5, "{c} {l}");
                assert!((out[c * 64 + 32 + l] - want1).abs() < 1e-5, "{c} {l}");
            }
        }
    }

    #[test]
    fn q6k_dequant_is_finite() {
        let mut blk = vec![0u8; 210];
        // q6_k block: ql[128] | qh[64] | scales[16] | d
        blk[208..210].copy_from_slice(&roundtrip_f16(0.01).to_le_bytes());
        for i in 0..16 {
            blk[192 + i] = (i as i8 - 8) as u8;
        }
        for i in 0..128 {
            blk[i] = i as u8;
        }
        let mut out = vec![0.0f32; 256];
        dequantize_row(GgmlType::Q6K, &blk, &mut out);
        assert!(out.iter().all(|v| v.is_finite()));
        let d = f16::from_bits(u16::from_le_bytes([blk[208], blk[209]])).to_f32();
        assert!((out[0] - d * -8.0 * -32.0).abs() < 1e-6);
    }
}

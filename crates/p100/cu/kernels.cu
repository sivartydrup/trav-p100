// trav-p100 kernels, built for sm_60 (Tesla P100 / GP100) only.
//
// Ground rules taken from measurement on this exact card:
//   * GP100 has no DP4A. int8 dots cost four scalar multiplies, so every weight
//     kernel here dequantises 4/5/6-bit weights into *subnormal* half bit
//     patterns (q * 2^-24, exact by construction) and accumulates with HFMA2,
//     which runs at full rate on GP100. Activations are pre-scaled per 32-block
//     by a power of two so no product can overflow fp16.
//   * Decode is load-issue bound. Each lane therefore fetches one 16-byte load
//     per 16 weight bytes, and the per-block scale/min correction is applied in
//     fp32 once per 16 values (never per value).
//   * Every weight probe is 16-byte aligned: Q4_K blocks are 144 B (9 x uint4)
//     and Q6_K is repacked by the loader to 224 B with ql/hi at fixed offsets.

#include <cuda_fp16.h>
#include <cstdint>

#define WARP 32

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

static __device__ __forceinline__ uint32_t bperm(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}

// byte `sel` of w as an exact float: put it in the mantissa of 2^23, subtract.
static __device__ __forceinline__ float byte_f(uint32_t w, int sel) {
    return __int_as_float(bperm(w, 0x4B000000u, (uint32_t)(0x7650 | sel))) - 8388608.0f;
}

static __device__ __forceinline__ half2 h2_bits(uint32_t u) {
    union { uint32_t u; half2 h; } c;
    c.u = u;
    return c.h;
}

union H8 { uint4 u; half2 h[4]; };

// four 2-bit fields of a byte -> four bytes (Q6_K high-bit plane).
// A single multiply by 0x41041 would be shorter but carries between copies
// corrupt it whenever a field is 2 or 3; two nibble multiplies are exact.
static __device__ __forceinline__ uint32_t spread2(uint32_t v) {
    const uint32_t lo = ((v & 0xFu) * 0x41u) & 0x0303u;
    const uint32_t hi = (((v >> 4) * 0x41u) & 0x0303u) << 16;
    return lo | hi;
}

// ---------------------------------------------------------------------------
// activation preparation: f32 -> f16 with a per-32-block power-of-two prescale,
// plus the exact fp32 16-value sums used for quant offset terms.
// ys[s] = { 2^(24-k), sum of the 16 f32 values of sub-block s }, s = k/16.
// ---------------------------------------------------------------------------

extern "C" __global__ void prep_act(const float* __restrict__ x,
                                    __half* __restrict__ y,
                                    float2* __restrict__ ys,
                                    int ncols, int k) {
    const int lane = threadIdx.x & (WARP - 1);
    const int nb = k >> 5;
    const int wid = blockIdx.x * (blockDim.x / WARP) + (threadIdx.x / WARP);
    const int total = ncols * nb;
    if (wid >= total) return;
    const int c = wid / nb;
    const int b = wid - c * nb;
    const int64_t off = (int64_t)c * k + b * 32 + lane;
    const float v = x[off];

    float s = v;
    float a = fabsf(v);
    #pragma unroll
    for (int o = 1; o < 16; o <<= 1) {
        s += __shfl_xor_sync(0xFFFFFFFFu, s, o);
        a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, o));
    }
    const float s16 = s;
    a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, 16));

    int e = 0;
    if (a > 0.0f) frexpf(a, &e);       // a in [2^(e-1), 2^e)
    const int kk = 15 - e;
    y[off] = __float2half_rn(ldexpf(v, kk));
    if ((lane & 15) == 0) {
        // per-column stride is k/16 float2 entries
        ys[c * nb * 2 + 2 * b + (lane >> 4)] = make_float2(ldexpf(1.0f, 24 - kk), s16);
    }
}

// ---------------------------------------------------------------------------
// RMS norm (fp32 residual stream, fp16 weights), one block per row
// ---------------------------------------------------------------------------

extern "C" __global__ void __launch_bounds__(1024)
rms_norm(const float* __restrict__ x, const __half* __restrict__ w,
         float* __restrict__ out, int n, float eps) {
    const int row = blockIdx.x;
    const float* xr = x + (int64_t)row * n;
    float* orow = out + (int64_t)row * n;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;

    float acc = 0.0f;
    for (int i = tid; i < n; i += nt) {
        const float v = xr[i];
        acc += v * v;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, o);
    __shared__ float red[32];
    if ((tid & 31) == 0) red[tid >> 5] = acc;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (int i = 0; i < (int)(nt / 32); i++) t += red[i];
        red[0] = rsqrtf(t / (float)n + eps);
    }
    __syncthreads();
    const float scale = red[0];
    for (int i = tid; i < n; i += nt) {
        orow[i] = xr[i] * scale * __half2float(w[i]);
    }
}

// fused residual add + rms norm: out = rms_norm(x + delta) * w, x += delta
extern "C" __global__ void __launch_bounds__(1024)
add_rms_norm(float* __restrict__ x, const float* __restrict__ delta,
             const __half* __restrict__ w, float* __restrict__ out,
             int n, float eps) {
    const int row = blockIdx.x;
    float* xr = x + (int64_t)row * n;
    const float* dr = delta + (int64_t)row * n;
    float* orow = out + (int64_t)row * n;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;

    float acc = 0.0f;
    for (int i = tid; i < n; i += nt) {
        const float v = xr[i] + dr[i];
        xr[i] = v;
        acc += v * v;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, o);
    __shared__ float red[32];
    if ((tid & 31) == 0) red[tid >> 5] = acc;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (int i = 0; i < (int)(nt / 32); i++) t += red[i];
        red[0] = rsqrtf(t / (float)n + eps);
    }
    __syncthreads();
    const float scale = red[0];
    for (int i = tid; i < n; i += nt) {
        orow[i] = xr[i] * scale * __half2float(w[i]);
    }
}

// ---------------------------------------------------------------------------
// quantised matvec / matmul: out[c][row] = sum_k W[row][k] * Y[c][k] (+ bias)
//
// One warp computes one row. Eight lanes cover one 256-value super-block; lane
// t owns two 16-value sub-blocks (low nibbles + high nibbles of the same 16
// bytes). COLS columns reuse each dequantised weight word.
// ---------------------------------------------------------------------------

// One super-block's worth of quantised weights for a lane.
struct WBlk {
    uint4 hd;
    uint4 qs;
    uint32_t h0, h1;
    uint32_t sc2;
};

template <bool Q6>
static __device__ __forceinline__ void load_wblk(WBlk& b, const char* __restrict__ prow,
                                                 int sb, int jj, int hf, int t) {
    if constexpr (Q6) {
        const char* __restrict__ p = prow + (int64_t)sb * 224;
        // Only d (2 bytes) is used from this word; a 16-byte load here costs a
        // third of the kernel's per-lane requests for nothing.
        b.hd = make_uint4(*(const uint32_t*)(p), 0, 0, 0);
        b.qs = *(const uint4*)(p + 32 + 32 * jj + 16 * hf);
        // The repack interleaves a lane's two high-bit runs half by half, so
        // this is one 8-byte load instead of two 4-byte gathers.
        const uint2 hi = *(const uint2*)(p + 160 + 16 * jj + 8 * hf);
        b.h0 = hi.x;
        b.h1 = hi.y;
        b.sc2 = *(const uint16_t*)(p + 4 + 2 * t);
    } else {
        const char* __restrict__ p = prow + (int64_t)sb * 144;
        b.hd = *(const uint4*)(p);
        b.qs = *(const uint4*)(p + 16 + 32 * jj + 16 * hf);
    }
}

// Dequantise one lane's 32 values of super-block `b` into eight half2 words
// (subnormal `q * 2^-24`) and the four fp32 scale/offset terms.
template <bool Q6>
static __device__ __forceinline__ void decode_wblk(const WBlk& b, int jj, int t,
                                                   half2* qlo, half2* qhi,
                                                   float& dsc_lo, float& dsc_hi,
                                                   float& dm_lo, float& dm_hi) {
    if constexpr (Q6) {
        const uint32_t sc2 = b.sc2;
        const float d = __half2float(__ushort_as_half((unsigned short)(b.hd.x & 0xFFFFu)));
        const float sc_lo = (float)(int8_t)(sc2 & 0xFF);
        const float sc_hi = (float)(int8_t)((sc2 >> 8) & 0xFF);
        dsc_lo = d * sc_lo;
        dsc_hi = d * sc_hi;
        dm_lo = dsc_lo * 32.0f;
        dm_hi = dsc_hi * 32.0f;
        const uint32_t hw[2] = {b.h0, b.h1};
        const uint32_t w[4] = {b.qs.x, b.qs.y, b.qs.z, b.qs.w};
        #pragma unroll
        for (int wi = 0; wi < 4; wi++) {
            const uint32_t lo = w[wi] & 0x0F0F0F0Fu;
            const uint32_t hi = (w[wi] >> 4) & 0x0F0F0F0Fu;
            const uint32_t hbl = spread2((hw[0] >> (8 * wi)) & 0xFFu);
            const uint32_t hbh = spread2((hw[1] >> (8 * wi)) & 0xFFu);
            const uint32_t vl = lo | (hbl << 4);
            const uint32_t vh = hi | (hbh << 4);
            qlo[2 * wi + 0] = h2_bits(bperm(vl, 0, 0x4140));
            qlo[2 * wi + 1] = h2_bits(bperm(vl, 0, 0x4342));
            qhi[2 * wi + 0] = h2_bits(bperm(vh, 0, 0x4140));
            qhi[2 * wi + 1] = h2_bits(bperm(vh, 0, 0x4342));
        }
    } else {
        const uint32_t A = b.hd.y, B = b.hd.z, C = b.hd.w;
        const bool hi_sc = jj >= 2;
        uint32_t scw, mw;
        if (hi_sc) {
            scw = (C & 0x0F0F0F0Fu) | ((A >> 2) & 0x30303030u);
            mw = ((C >> 4) & 0x0F0F0F0Fu) | ((B >> 2) & 0x30303030u);
        } else {
            scw = A & 0x3F3F3F3Fu;
            mw = B & 0x3F3F3F3Fu;
        }
        const int bsel = 2 * (jj & 1);
        const float d = __half2float(__ushort_as_half((unsigned short)(b.hd.x & 0xFFFFu)));
        const float dmin = __half2float(__ushort_as_half((unsigned short)(b.hd.x >> 16)));
        dsc_lo = d * byte_f(scw, bsel);
        dsc_hi = d * byte_f(scw, bsel + 1);
        dm_lo = dmin * byte_f(mw, bsel);
        dm_hi = dmin * byte_f(mw, bsel + 1);
        const uint32_t w[4] = {b.qs.x, b.qs.y, b.qs.z, b.qs.w};
        #pragma unroll
        for (int wi = 0; wi < 4; wi++) {
            const uint32_t vl = w[wi] & 0x0F0F0F0Fu;
            const uint32_t vh = (w[wi] >> 4) & 0x0F0F0F0Fu;
            qlo[2 * wi + 0] = h2_bits(bperm(vl, 0, 0x4140));
            qlo[2 * wi + 1] = h2_bits(bperm(vl, 0, 0x4342));
            qhi[2 * wi + 0] = h2_bits(bperm(vh, 0, 0x4140));
            qhi[2 * wi + 1] = h2_bits(bperm(vh, 0, 0x4342));
        }
    }
}

template <bool Q6, int ROWS, int COLS>
static __device__ __forceinline__ void
gemv_body(const char* __restrict__ W, int64_t row_stride,
          const __half* __restrict__ Y, int64_t y_stride,
          const float2* __restrict__ YS, int64_t ys_stride,
          const float* __restrict__ bias,
          float* __restrict__ out, int64_t out_stride,
          int nrows, int k, void* __restrict__ smem) {
    const int lane = threadIdx.x & (WARP - 1);
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;

    const int g = lane >> 3;
    const int t = lane & 7;
    const int jj = t >> 1;
    const int hf = t & 1;

    const int row0 = (blockIdx.x * warps + warp) * ROWS;
    const int nsb = k / 256;
    // Split-K: blockIdx.z owns a contiguous slice of staging passes, so a
    // short row count can still fill the card (or afford more rows per warp).
    // Partials are fp32 atomics into a zeroed output; only split 0 adds bias.
    const int nsplit = gridDim.z;

    // Shared staging: 8 super-blocks per pass for narrow kernels, 4 when eight
    // columns are in flight (two blocks stay resident per SM either way).
    constexpr int CSB = (COLS * ROWS <= 4) ? 8 : 4;
    __half* sy = (__half*)smem;                       // [COLS][CSB*256]
    float2* sys = (float2*)(sy + COLS * CSB * 256);   // [COLS][CSB*16]

    int off_lo, off_hi;
    if constexpr (Q6) {
        off_lo = 32 * jj + 64 * (jj >> 1) + 16 * hf;
        off_hi = off_lo + 64;
    } else {
        off_lo = 64 * jj + 16 * hf;
        off_hi = off_lo + 32;
    }
    const int s_lo = off_lo >> 4;
    const int s_hi = off_hi >> 4;

    const char* __restrict__ prow[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const int row = min(row0 + r, nrows - 1);
        prow[r] = W + (int64_t)row * row_stride;
    }

    float sum[ROWS][COLS];
    #pragma unroll
    for (int r = 0; r < ROWS; r++)
        #pragma unroll
        for (int c = 0; c < COLS; c++) sum[r][c] = 0.0f;

    const int npass = (nsb + CSB - 1) / CSB;
    const int pfirst = (int)((int64_t)npass * blockIdx.z / nsplit);
    const int plast = (int)((int64_t)npass * (blockIdx.z + 1) / nsplit);
    for (int p = pfirst; p < plast; p++) {
        const int sb0 = p * CSB;
        const int rem = min(CSB, nsb - sb0);
        for (int i = threadIdx.x; i < COLS * CSB * 32; i += blockDim.x) {
            const int c = i / (CSB * 32);
            const int j = i - c * (CSB * 32);
            if (j * 8 < rem * 256) {
                *(uint4*)(sy + c * CSB * 256 + j * 8) =
                    *(const uint4*)(Y + (int64_t)c * y_stride + sb0 * 256 + j * 8);
            }
        }
        for (int i = threadIdx.x; i < COLS * CSB * 16; i += blockDim.x) {
            const int c = i / (CSB * 16);
            const int j = i - c * (CSB * 16);
            if (j < rem * 16) {
                sys[i] = YS[(int64_t)c * ys_stride + sb0 * 16 + j];
            }
        }
        __syncthreads();

        for (int sb = sb0 + g; sb < sb0 + rem; sb += 4) {
            half2 qlo[ROWS][8], qhi[ROWS][8];
            float dsc[ROWS][4];
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                WBlk b;
                load_wblk<Q6>(b, prow[r], sb, jj, hf, t);
                decode_wblk<Q6>(b, jj, t, qlo[r], qhi[r],
                                dsc[r][0], dsc[r][1], dsc[r][2], dsc[r][3]);
            }
            const int rel = sb - sb0;
            #pragma unroll
            for (int c = 0; c < COLS; c++) {
                const __half* base = sy + c * CSB * 256 + rel * 256;
                const float2 ys_lo = sys[c * CSB * 16 + rel * 16 + s_lo];
                const float2 ys_hi = sys[c * CSB * 16 + rel * 16 + s_hi];
                H8 yl0, yl1, yh0, yh1;
                yl0.u = *(const uint4*)(base + off_lo);
                yl1.u = *(const uint4*)(base + off_lo + 8);
                yh0.u = *(const uint4*)(base + off_hi);
                yh1.u = *(const uint4*)(base + off_hi + 8);
                const float2 yl = ys_lo, yh = ys_hi;
                #pragma unroll
                for (int r = 0; r < ROWS; r++) {
                    half2 al = __float2half2_rn(0.0f);
                    half2 ah = al;
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        al = __hfma2(qlo[r][i], yl0.h[i], al);
                        ah = __hfma2(qhi[r][i], yh0.h[i], ah);
                    }
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        al = __hfma2(qlo[r][4 + i], yl1.h[i], al);
                        ah = __hfma2(qhi[r][4 + i], yh1.h[i], ah);
                    }
                    const float sl = __half2float(__hadd(__low2half(al), __high2half(al)));
                    const float sh = __half2float(__hadd(__low2half(ah), __high2half(ah)));
                    sum[r][c] += dsc[r][0] * yl.x * sl - dsc[r][2] * yl.y
                               + dsc[r][1] * yh.x * sh - dsc[r][3] * yh.y;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        #pragma unroll
        for (int c = 0; c < COLS; c++) {
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) sum[r][c] += __shfl_xor_sync(0xFFFFFFFFu, sum[r][c], o);
            const int row = row0 + r;
            if (lane == 0 && row < nrows) {
                float v = sum[r][c];
                if (nsplit > 1) {
                    if (blockIdx.z == 0 && bias) v += bias[row];
                    atomicAdd(&out[c * out_stride + row], v);
                } else {
                    if (bias) v += bias[row];
                    out[c * out_stride + row] = v;
                }
            }
        }
    }
}

// Plain C symbols: the Rust launcher looks kernels up by name, so no mangling.
#define GEMV_ARGS                                                              \
    const char* __restrict__ W, int64_t row_stride,                            \
    const __half* __restrict__ Y, int64_t y_stride,                            \
    const float2* __restrict__ YS, int64_t ys_stride,                          \
    const float* __restrict__ bias,                                            \
    float* __restrict__ out, int64_t out_stride,                               \
    int nrows, int k

#define DEF_GEMV(NAME, Q6V, ROWSV, COLSV)                                      \
    extern "C" __global__ void __launch_bounds__(256) NAME(GEMV_ARGS) {        \
        extern __shared__ __align__(16) uint8_t gemv_smem[];                    \
        gemv_body<Q6V, ROWSV, COLSV>(W, row_stride, Y, y_stride, YS, ys_stride,  \
                                     bias, out, out_stride, nrows, k, gemv_smem);\
    }

DEF_GEMV(gemv_q4k_c1, false, 1, 1)
DEF_GEMV(gemv_q4k_c2, false, 1, 2)
DEF_GEMV(gemv_q4k_c4, false, 1, 4)
DEF_GEMV(gemv_q4k_c8, false, 1, 8)
DEF_GEMV(gemv_q4k_r2c8, false, 2, 8)
DEF_GEMV(gemv_q4k_r2c1, false, 2, 1)
DEF_GEMV(gemv_q4k_r4c1, false, 4, 1)
DEF_GEMV(gemv_q4k_r8c1, false, 8, 1)
DEF_GEMV(gemv_q6k_c1, true, 1, 1)
DEF_GEMV(gemv_q6k_c2, true, 1, 2)
DEF_GEMV(gemv_q6k_c4, true, 1, 4)
DEF_GEMV(gemv_q6k_c8, true, 1, 8)
DEF_GEMV(gemv_q6k_r2c8, true, 2, 8)
DEF_GEMV(gemv_q6k_r2c1, true, 2, 1)
DEF_GEMV(gemv_q6k_r4c1, true, 4, 1)
DEF_GEMV(gemv_q6k_r8c1, true, 8, 1)

// ---------------------------------------------------------------------------
// fused dequantise + tiled GEMM: the prefill path.
//
// One block owns 64 weight rows x 32 token columns. K is staged 128 values at a
// time; the compressed weights are dequantised into shared memory *once* per
// stage (still as exact `q * 2^-24` subnormals) and every one of the 32 columns
// reuses them. Both tiles are stored as half2 indexed by k-pair, so one 16-byte
// load fetches a thread's whole 4-row weight fragment or 4-column activation
// fragment: two shared-memory instructions feed sixteen HFMA2.
//
// Each thread holds a 4-row x 4-column register tile and accumulates one scale
// block's products in fp16 half2 lanes, then folds the q4_k/q6_k scale and min
// into fp32 -- the same numeric contract as the GEMV, but with the weight
// stream read once per 32 tokens instead of once per 8.
//
// Activations are prepared by `prep_act_col`, which uses a single power-of-two
// scale per column, so that scale is applied once in the epilogue rather than
// once per k-block.
// ---------------------------------------------------------------------------

#define GM 64            // weight rows per block
#define GN 32            // token columns per block
#define GK 128           // k values staged at a time
#define GKP (GK / 2)     // k-pairs per stage
#define GT 128           // threads per block
#define GSW GM           // half2 per k-pair in the weight tile
#define GSY (GN + 4)     // half2 per k-pair in the activation tile

#undef GM
#undef GN
#undef GK
#undef GKP
#undef GT
#undef GSW
#undef GSY

// ---------------------------------------------------------------------------
// The same GEMM parameterised over its block tile, so tile and occupancy can be
// swept without touching the arithmetic:
//   TM x TN  weight rows x token columns per block
//   TK       k values staged per step (multiple of 64; q6_k needs 128 because a
//            chunk's low and high nibbles sit 64 values apart)
//   TT       threads per block, TT == (TM/RR)*(TN/CC)
//   RR x CC  per-thread register tile
// Both tiles stay half2 keyed by k-pair, so a thread's whole RR-row or CC-column
// fragment is a 16-byte load per four entries.
// ---------------------------------------------------------------------------
template <bool Q6, int TM, int TN, int TK, int TT, int RR, int CC, int PAD = 4>
static __device__ __forceinline__ void
gemm_tiled(const char* __restrict__ W, int64_t row_stride,
           const __half* __restrict__ Y, int64_t y_stride,
           const float2* __restrict__ YS, int64_t ys_stride,
           const float* __restrict__ bias,
           float* __restrict__ out, int64_t out_stride,
           int nrows, int k, int ncols, int nsplit, void* __restrict__ smem) {
    constexpr int TKP = TK / 2;           // k-pairs staged
    constexpr int GSW = TM;               // half2 per k-pair, weight tile
    constexpr int GSY = TN + PAD;         // half2 per k-pair, activation tile
    constexpr int NCH = TK / 64;          // 64-value chunks per stage
    constexpr int RG = TM / RR;
    constexpr int CG = TN / CC;
    constexpr int NST = TT / TKP;         // column strides per activation stage
    static_assert(RG * CG == TT, "threads must exactly cover the register tile");
    static_assert(TT % TKP == 0 && TN % NST == 0, "activation staging needs a clean split");
    static_assert(!Q6 || TK >= 128, "q6_k needs 128-value stages");
    static_assert(RR % 4 == 0 && CC % 4 == 0, "fragments load as uint4");

    const half2 K12 = __float2half2_rn(4096.0f);
    const int tid = threadIdx.x;
    const int row0 = blockIdx.x * TM;
    const int col0 = blockIdx.y * TN;

    half2* sw = (half2*)smem;             // [TKP][GSW]
    half2* sy = sw + TKP * GSW;           // [TKP][GSY]

    float sum[RR][CC];
    #pragma unroll
    for (int i = 0; i < RR; i++)
        #pragma unroll
        for (int c = 0; c < CC; c++) sum[i][c] = 0.0f;

    const int rg = tid / CG;
    const int cg = tid - rg * CG;
    const int lr0 = rg * RR;
    const int c0 = cg * CC;

    const int akp = tid % TKP;
    const int acb = tid / TKP;
    const bool full_cols = col0 + TN <= ncols;

    const int total = k / TK;
    const int kfirst = (int)((int64_t)total * blockIdx.z / nsplit);
    const int klast = (int)((int64_t)total * (blockIdx.z + 1) / nsplit);
    const bool accum = nsplit > 1;
    const int ntask = TM * NCH * 2;

    for (int st = kfirst; st < klast; st++) {
        const int k0 = st * TK;
        const int sb = k0 >> 8;
        const int jjbase = (k0 & 255) >> 6;

        // ---- dequantise this stage's weights into shared memory -----------
        for (int task = tid; task < ntask; task += TT) {
            const int row = task / (2 * NCH);
            const int q = task - row * 2 * NCH;
            const int hf = q & 1;
            const int chunk = q >> 1;
            const int jj = jjbase + chunk;
            const char* prow = W + (int64_t)min(row0 + row, nrows - 1) * row_stride;
            const int tt = Q6 ? 2 * jj + hf : 0;
            WBlk b;
            load_wblk<Q6>(b, prow, sb, jj, hf, tt);
            half2 qlo[8], qhi[8];
            float dsl, dsh, dml, dmh;
            decode_wblk<Q6>(b, jj, tt, qlo, qhi, dsl, dsh, dml, dmh);
            const int kp = (Q6 ? 16 : 32) * chunk + 8 * hf;
            const int kh = kp + (Q6 ? 32 : 16);
            const half2 slo = __float2half2_rn(dsl * 4096.0f);
            const half2 mlo = __float2half2_rn(-dml);
            const half2 shi = __float2half2_rn(dsh * 4096.0f);
            const half2 mhi = __float2half2_rn(-dmh);
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                sw[(kp + i) * GSW + row] = __hfma2(__hmul2(qlo[i], K12), slo, mlo);
                sw[(kh + i) * GSW + row] = __hfma2(__hmul2(qhi[i], K12), shi, mhi);
            }
        }

        // ---- stage activations --------------------------------------------
        if (full_cols) {
            const __half* src = Y + (int64_t)(col0 + acb) * y_stride + k0 + 2 * akp;
            half2* dst = sy + akp * GSY + acb;
            #pragma unroll
            for (int n = 0; n < TN / NST; n++) {
                *dst = *(const half2*)src;
                src += NST * y_stride;
                dst += NST;
            }
        } else {
            half2* dst = sy + akp * GSY + acb;
            #pragma unroll
            for (int n = 0; n < TN / NST; n++) {
                const int cs = min(col0 + acb + n * NST, ncols - 1);
                *dst = *(const half2*)(Y + (int64_t)cs * y_stride + k0 + 2 * akp);
                dst += NST;
            }
        }
        __syncthreads();

        // ---- register-blocked GEMM over the stage -------------------------
        {
            half2 acc[RR][CC];
            #pragma unroll
            for (int i = 0; i < RR; i++)
                #pragma unroll
                for (int c = 0; c < CC; c++) acc[i][c] = __float2half2_rn(0.0f);

            #pragma unroll
            for (int p = 0; p < TKP; p++) {
                H8 wv[RR / 4];
                #pragma unroll
                for (int r4 = 0; r4 < RR / 4; r4++)
                    wv[r4].u = *(const uint4*)(sw + p * GSW + lr0 + 4 * r4);
                H8 yv[CC / 4];
                #pragma unroll
                for (int c4 = 0; c4 < CC / 4; c4++)
                    yv[c4].u = *(const uint4*)(sy + p * GSY + c0 + 4 * c4);
                #pragma unroll
                for (int i = 0; i < RR; i++)
                    #pragma unroll
                    for (int c = 0; c < CC; c++)
                        acc[i][c] = __hfma2(wv[i / 4].h[i % 4], yv[c / 4].h[c % 4], acc[i][c]);
            }

            #pragma unroll
            for (int i = 0; i < RR; i++)
                #pragma unroll
                for (int c = 0; c < CC; c++)
                    sum[i][c] += __half2float(__hadd(__low2half(acc[i][c]),
                                                     __high2half(acc[i][c])));
        }
        __syncthreads();
    }

    // ---- epilogue: one power-of-two scale per column, then bias -----------
    float down[CC];
    #pragma unroll
    for (int c = 0; c < CC; c++) {
        const int cs = min(col0 + c0 + c, ncols - 1);
        down[c] = YS[(int64_t)cs * ys_stride].x;
    }
    #pragma unroll
    for (int i = 0; i < RR; i++) {
        const int orow = row0 + lr0 + i;
        if (orow >= nrows) continue;
        const float b = bias ? bias[orow] : 0.0f;
        #pragma unroll
        for (int c = 0; c < CC; c++) {
            const int ocol = col0 + c0 + c;
            if (ocol >= ncols) continue;
            float* dst = &out[(int64_t)ocol * out_stride + orow];
            if (accum) {
                atomicAdd(dst, sum[i][c] * down[c] + (kfirst == 0 ? b : 0.0f));
            } else {
                *dst = sum[i][c] * down[c] + b;
            }
        }
    }
}

#define GEMM_ARGS                                                              \
    const char* __restrict__ W, int64_t row_stride,                            \
    const __half* __restrict__ Y, int64_t y_stride,                            \
    const float2* __restrict__ YS, int64_t ys_stride,                          \
    const float* __restrict__ bias,                                            \
    float* __restrict__ out, int64_t out_stride,                               \
    int nrows, int k, int ncols, int nsplit

#define DEF_GEMM_TILED(NAME, Q6V, TMr, TNr, TKr, TTr, RRr, CCr)                \
    extern "C" __global__ void __launch_bounds__(TTr) NAME(GEMM_ARGS) {        \
        extern __shared__ __align__(16) uint8_t gemm_smem[];                    \
        gemm_tiled<Q6V, TMr, TNr, TKr, TTr, RRr, CCr>(                          \
            W, row_stride, Y, y_stride, YS, ys_stride, bias, out, out_stride,   \
            nrows, k, ncols, nsplit, gemm_smem);                                \
    }

// 64 columns per block: halves both the weight stream and the dequantise work
// per token. Three candidate occupancies, all sharing the same arithmetic.
DEF_GEMM_TILED(gemm_q4k_n64t128, false, 64, 64, 64, 128, 8, 4)   // 3 blocks/SM
DEF_GEMM_TILED(gemm_q4k_n32t128, false, 64, 32, 128, 128, 4, 4)  // reference
DEF_GEMM_TILED(gemm_q6k_n32t128, true, 64, 32, 128, 128, 4, 4)   // reference
// 128x128 block tile: 64 HFMA2 per four shared loads, and 8x8 independent
// accumulator chains per thread (what gemm_fold_kernel_u2 does).
// ---------------------------------------------------------------------------
// Prefill path with the dequantise hoisted out of the GEMM.
//
// A prefill tile runs every layer before the next tile starts, so a
// whole-prompt tile touches each weight matrix once. Materialising it as fp16 in
// a workspace therefore costs one read of the packed weights plus a write and a
// read of the fp16 copy, and lets the GEMM stage by transpose-copy instead of
// dequantising in the inner loop. That buys the 8x8 register tile: 64 HFMA2 per
// four LDS.128, 64 independent chains, and a fold to fp32 every 256 k values.
// ---------------------------------------------------------------------------

template <bool Q6>
static __device__ __forceinline__ void
dequant_f16_body(const char* __restrict__ W, int64_t row_stride,
                 __half* __restrict__ out, int nrows, int k) {
    // One thread per 32 values: the two 16-value runs of one 64-value chunk,
    // exactly the lane task the packed kernels already use.
    const int tasks = (k >> 6) * 2;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = (int64_t)nrows * tasks;
    if (idx >= total) return;
    const int row = (int)(idx / tasks);
    const int q = (int)(idx - (int64_t)row * tasks);
    const int hf = q & 1;
    const int chunk = q >> 1;
    const int sb = chunk >> 2;
    const int jjl = chunk & 3;
    const int tt = Q6 ? 2 * jjl + hf : 0;
    WBlk b;
    load_wblk<Q6>(b, W + (int64_t)row * row_stride, sb, jjl, hf, tt);
    half2 qlo[8], qhi[8];
    float dsl, dsh, dml, dmh;
    decode_wblk<Q6>(b, jjl, tt, qlo, qhi, dsl, dsh, dml, dmh);
    // Value offsets inside the super-block. q4_k packs its four 64-value
    // chunks linearly; q6_k's second pair of chunks lives 64 values further in
    // (its ql bytes hold values 64 apart), which is why the GEMM can only use
    // 128-value stages for it.
    const int base0 = sb * 256
                    + (Q6 ? 32 * jjl + 64 * (jjl >> 1) + 16 * hf
                          : 64 * jjl + 16 * hf);
    const int base_hi = base0 + (Q6 ? 64 : 32);
    const float sl = ldexpf(dsl, 24), sh = ldexpf(dsh, 24);
    __half* o = out + (int64_t)row * k;
    // 16 values per run, written as two 16-byte stores: 2-byte stores would put
    // this kernel at 50 GB/s.
    union { uint4 u; __half h[8]; } vr[4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const float2 lo = __half22float2(qlo[i]);
        const float2 hi = __half22float2(qhi[i]);
        const int q1 = 2 * i, q2 = 16 + 2 * i;
        vr[q1 / 8].h[q1 % 8] = __float2half_rn(fmaf(sl, lo.x, -dml));
        vr[q1 / 8].h[q1 % 8 + 1] = __float2half_rn(fmaf(sl, lo.y, -dml));
        vr[q2 / 8].h[q2 % 8] = __float2half_rn(fmaf(sh, hi.x, -dmh));
        vr[q2 / 8].h[q2 % 8 + 1] = __float2half_rn(fmaf(sh, hi.y, -dmh));
    }
    *(uint4*)(o + base0) = vr[0].u;
    *(uint4*)(o + base0 + 8) = vr[1].u;
    *(uint4*)(o + base_hi) = vr[2].u;
    *(uint4*)(o + base_hi + 8) = vr[3].u;
}

extern "C" __global__ void dequant_q4k_f16(const char* __restrict__ W, int64_t row_stride,
                                           __half* __restrict__ out, int nrows, int k) {
    dequant_f16_body<false>(W, row_stride, out, nrows, k);
}

extern "C" __global__ void dequant_q6k_f16(const char* __restrict__ W, int64_t row_stride,
                                           __half* __restrict__ out, int nrows, int k) {
    dequant_f16_body<true>(W, row_stride, out, nrows, k);
}

// ---------------------------------------------------------------------------
// fp16 weight GEMM, parameterised over the block tile:
//   TM x TN  weight rows x token columns per block
//   BK2      k-pairs staged at a time (16 = 32 k values)
//   TT       threads, TT == (TM/RR)*(TN/CC)
//   RR x CC  per-thread register tile
// Stages are double buffered through registers so the next stage's global loads
// overlap this stage's compute. Rows and columns are owned as two 16-byte
// fragments 64 entries apart, which keeps the 16 chunks a warp reads spread
// over the 32 banks twice instead of four times.
// ---------------------------------------------------------------------------

union U4w { uint4 u; uint32_t w[4]; };

template <int TM, int TN, int BK2, int TT, int RR, int CC>
static __device__ __forceinline__ void
gemm_f16_body(const __half* __restrict__ W, int64_t row_stride,
              const __half* __restrict__ Y, int64_t y_stride,
              const float2* __restrict__ YS, int64_t ys_stride,
              const float* __restrict__ bias,
              float* __restrict__ out, int64_t out_stride,
              int nrows, int k, int ncols) {
    constexpr int RG = TM / RR, CG = TN / CC;
    constexpr int NH = TM / 2, NQ = TN / 2;     // second fragment offset
    constexpr int NCHA = TM * BK2 / 4;          // uint4 chunks per stage
    constexpr int NCHB = TN * BK2 / 4;
    static_assert(RR == 4 || RR == 8, "row fragments are one or two uint4");
    static_assert(CC == 4 || CC == 8, "column fragments are one or two uint4");
    constexpr int NPA = (NCHA + TT - 1) / TT;   // chunks per thread
    constexpr int NPB = (NCHB + TT - 1) / TT;
    static_assert(RG * CG == TT, "threads must cover the register tile");

    __shared__ __align__(16) uint32_t As[2][BK2][TM];
    __shared__ __align__(16) uint32_t Bs[2][BK2][TN];

    const int t = threadIdx.x;
    const int tx = t % CG;
    const int ty = t / CG;
    const int mb = blockIdx.x * TM;
    const int nb = blockIdx.y * TN;

    // fp32 partials, then the fp16 chain; the chain is indexed [column][row] so
    // that, for the column held fixed in the inner loop, consecutive rows land
    // on consecutive register banks — the bank of a[i] in the same FMA follows
    // the row index too, so the two stop colliding.
    float acc[RR][CC];
    half2 h[CC][RR];
    #pragma unroll
    for (int i = 0; i < RR; i++)
        #pragma unroll
        for (int j = 0; j < CC; j++) {
            acc[i][j] = 0.0f;
            h[j][i] = __float2half2_rn(0.0f);
        }

    // Staging: rows of the weight tile and columns of the activation tile are
    // counted separately — they are the same only when TM == TN.
    int rowA[NPA], chA[NPA], rowB[NPB], chB[NPB];
    bool vldA[NPA], vldB[NPB];
    const uint4* pa[NPA];
    const uint4* pb[NPB];
    #pragma unroll
    for (int s = 0; s < NPA; s++) {
        const int l = t + s * TT;
        vldA[s] = l < NCHA;
        rowA[s] = vldA[s] ? l / (BK2 / 4) : 0;
        chA[s] = vldA[s] ? l % (BK2 / 4) : 0;
        const int mr = mb + rowA[s] < nrows ? mb + rowA[s] : 0;
        pa[s] = (const uint4*)(W + (int64_t)mr * row_stride + chA[s] * 8);
    }
    #pragma unroll
    for (int s = 0; s < NPB; s++) {
        const int l = t + s * TT;
        vldB[s] = l < NCHB;
        rowB[s] = vldB[s] ? l / (BK2 / 4) : 0;
        chB[s] = vldB[s] ? l % (BK2 / 4) : 0;
        const int mc = nb + rowB[s] < ncols ? nb + rowB[s] : 0;
        pb[s] = (const uint4*)(Y + (int64_t)mc * y_stride + chB[s] * 8);
    }

    U4w sa[NPA], sb[NPB];
    auto gload = [&](int st) {
        #pragma unroll
        for (int s = 0; s < NPA; s++) {
            if (vldA[s]) sa[s].u = pa[s][st * (BK2 / 4)];
        }
        #pragma unroll
        for (int s = 0; s < NPB; s++) {
            if (vldB[s]) sb[s].u = pb[s][st * (BK2 / 4)];
        }
    };
    auto sstore = [&](int buf) {
        #pragma unroll
        for (int s = 0; s < NPA; s++) {
            if (!vldA[s]) continue;
            uint32_t* a = &As[buf][chA[s] * 4][rowA[s]];
            #pragma unroll
            for (int j = 0; j < 4; j++) a[j * TM] = sa[s].w[j];
        }
        #pragma unroll
        for (int s = 0; s < NPB; s++) {
            if (!vldB[s]) continue;
            uint32_t* b = &Bs[buf][chB[s] * 4][rowB[s]];
            #pragma unroll
            for (int j = 0; j < 4; j++) b[j * TN] = sb[s].w[j];
        }
    };
    auto tile = [&](int buf) {
        #pragma unroll
        for (int k2 = 0; k2 < BK2; k2++) {
            uint32_t a[RR], b[CC];
            const uint4 a0 = *(const uint4*)(&As[buf][k2][ty * 4]);
            const uint4 b0 = *(const uint4*)(&Bs[buf][k2][tx * 4]);
            a[0] = a0.x; a[1] = a0.y; a[2] = a0.z; a[3] = a0.w;
            b[0] = b0.x; b[1] = b0.y; b[2] = b0.z; b[3] = b0.w;
            if (RR == 8) {
                const uint4 a1 = *(const uint4*)(&As[buf][k2][ty * 4 + NH]);
                a[4] = a1.x; a[5] = a1.y; a[6] = a1.z; a[7] = a1.w;
            }
            if (CC == 8) {
                const uint4 b1 = *(const uint4*)(&Bs[buf][k2][tx * 4 + NQ]);
                b[4] = b1.x; b[5] = b1.y; b[6] = b1.z; b[7] = b1.w;
            }
            #pragma unroll
            for (int j = 0; j < CC; j++)
                #pragma unroll
                for (int i = 0; i < RR; i++)
                    h[j][i] = __hfma2(*(const half2*)&b[j], *(const half2*)&a[i], h[j][i]);
        }
    };
    auto fold = [&]() {
        #pragma unroll
        for (int i = 0; i < RR; i++)
            #pragma unroll
            for (int j = 0; j < CC; j++) {
                acc[i][j] += __half2float(__hadd(__low2half(h[j][i]),
                                                 __high2half(h[j][i])));
                h[j][i] = __float2half2_rn(0.0f);
            }
    };

    const int nt = k / (2 * BK2);
    gload(0);
    sstore(0);
    __syncthreads();
    for (int st = 0; st < nt; st++) {
        const int buf = st & 1;
        if (st + 1 < nt) gload(st + 1);
        tile(buf);
        if ((st & 7) == 7 || st + 1 == nt) fold();
        if (st + 1 < nt) {
            __syncthreads();
            sstore(buf ^ 1);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < CC; j++) {
        const int n = nb + (j < 4 ? tx * 4 + j : NQ + tx * 4 + j - 4);
        if (n >= ncols) continue;
        const float s = YS[(int64_t)n * ys_stride].x;
        #pragma unroll
        for (int ih = 0; ih < RR / 4; ih++) {
            const int m0 = mb + ih * NH + ty * 4;
            if (m0 >= nrows) continue;
            float4 v = make_float4(acc[ih * 4 + 0][j] * s, acc[ih * 4 + 1][j] * s,
                                   acc[ih * 4 + 2][j] * s, acc[ih * 4 + 3][j] * s);
            if (bias) {
                v.x += bias[m0];
                v.y += m0 + 1 < nrows ? bias[m0 + 1] : 0.0f;
                v.z += m0 + 2 < nrows ? bias[m0 + 2] : 0.0f;
                v.w += m0 + 3 < nrows ? bias[m0 + 3] : 0.0f;
            }
            float* dst = out + (int64_t)n * out_stride + m0;
            if (m0 + 4 <= nrows) {
                *(float4*)dst = v;
            } else {
                for (int i = 0; i < 4; i++)
                    if (m0 + i < nrows) dst[i] = (&v.x)[i];
            }
        }
    }
}

#define DEF_GEMM_F16(NAME, TMr, TNr, BK2r, TTr, RRr, CCr)                       \
    extern "C" __global__ void __launch_bounds__(TTr) NAME(                    \
        const __half* __restrict__ W, int64_t row_stride,                      \
        const __half* __restrict__ Y, int64_t y_stride,                        \
        const float2* __restrict__ YS, int64_t ys_stride,                      \
        const float* __restrict__ bias,                                        \
        float* __restrict__ out, int64_t out_stride,                           \
        int nrows, int k, int ncols) {                                         \
        gemm_f16_body<TMr, TNr, BK2r, TTr, RRr, CCr>(W, row_stride, Y, y_stride, \
            YS, ys_stride, bias, out, out_stride, nrows, k, ncols);            \
    }

// 128x128, 8x8: 64 HFMA2 per four LDS.128, one block (8 warps) per SM.
DEF_GEMM_F16(gemm_f16_128, 128, 128, 16, 256, 8, 8)


// ---------------------------------------------------------------------------
// per-column activation preparation for the GEMM: one power-of-two scale for
// the whole row. The GEMM applies 2^-kk once in its epilogue, after the fp16
// accumulation; ys[c][j].x holds it (the .y slot is unused by this path).
// ---------------------------------------------------------------------------

extern "C" __global__ void prep_act_col(const float* __restrict__ x,
                                        __half* __restrict__ y,
                                        float2* __restrict__ ys,
                                        int ncols, int k) {
    // One block per column. The old layout gave each column a single warp that
    // walked the whole row serially, so a 32-column prefill tile used 4 blocks
    // and every warp sat in a 112-deep dependent max chain. Spreading the row
    // over the block's 8 warps cuts that chain to k/(8*32) and fills 32 SMs.
    const int col = blockIdx.x;
    if (col >= ncols) return;
    const int tid = threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int warp = tid / WARP;
    const int nwarps = blockDim.x / WARP;
    const int64_t base = (int64_t)col * k;

    // 4 floats per load: the old scalar walk of the row ran at ~20 GB/s and
    // this kernel is 3% of prefill.
    const float4* x4 = (const float4*)(x + base);
    const int k4 = k >> 2;
    float a = 0.0f;
    for (int i = warp * WARP + lane; i < k4; i += nwarps * WARP) {
        const float4 v = x4[i];
        a = fmaxf(a, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, o));

    __shared__ float smax[32];
    if (lane == 0) smax[warp] = a;
    __syncthreads();
    float m = smax[0];
    for (int w = 1; w < nwarps; w++) m = fmaxf(m, smax[w]);

    int e = 0;
    if (m > 0.0f) frexpf(m, &e);
    // Target max|y| ~ 2^6: products stay far from fp16 overflow while the
    // accumulator keeps ~128 values per stage, and the power of two is exact.
    // Target max|y| ~ 2^3, not 2^6: the fp16 chains of the GEMM then stay far
    // from overflow, which lets them run 256 k values between folds.
    const int kk = 3 - e;
    const float up = ldexpf(1.0f, kk);
    const float down = ldexpf(1.0f, -kk);
    const int ns = k >> 4;
    for (int s = warp * WARP + lane; s < ns; s += nwarps * WARP) {
        const int64_t off = base + (int64_t)s * 16;
        half2* dst = (half2*)(y + off);
        const float4* src = (const float4*)(x + off);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const float4 v = src[j];
            dst[2 * j + 0] = __floats2half2_rn(v.x * up, v.y * up);
            dst[2 * j + 1] = __floats2half2_rn(v.z * up, v.w * up);
        }
        ys[(int64_t)col * ns + s] = make_float2(down, 0.0f);
    }
}

// ---------------------------------------------------------------------------
// RoPE (Qwen2 rotate-half / NeoX layout: pairs (i, i + head_dim/2))
// ---------------------------------------------------------------------------

extern "C" __global__ void rope_neox(float* __restrict__ q, float* __restrict__ k,
                                     const int* __restrict__ pos,
                                     const float* __restrict__ inv_freq,
                                     int n_q_heads, int n_k_heads, int head_dim,
                                     int n_q, int k_off) {
    const int hd2 = head_dim >> 1;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int per_q = n_q_heads * hd2;
    const int total = n_q * (per_q + n_k_heads * hd2);
    if (idx >= total) return;

    const bool is_q = idx < n_q * per_q;
    const int local = is_q ? idx : idx - n_q * per_q;
    const int nheads = is_q ? n_q_heads : n_k_heads;
    const int per_head = hd2;
    const int token = local / (nheads * per_head);
    const int rem = local - token * (nheads * per_head);
    const int head = rem / per_head;
    const int i = rem - head * per_head;

    const int p = pos[token];
    const float ang = (float)p * inv_freq[i];
    float sn, cs;
    sincosf(ang, &sn, &cs);

    float* base = is_q ? q : k;
    const int stride = nheads * head_dim;
    float* v = base + (int64_t)token * stride + head * head_dim;
    const float x0 = v[i];
    const float x1 = v[i + hd2];
    v[i] = x0 * cs - x1 * sn;
    v[i + hd2] = x1 * cs + x0 * sn;
}

// ---------------------------------------------------------------------------
// write K/V (f32) into the f16 cache at [pos][kv_head][head_dim]
// ---------------------------------------------------------------------------

// `ctl` carries the two per-token scalars from device memory: ctl[0] is the
// KV write position, ctl[1] the decode attention length. Reading them from a
// buffer instead of a kernel argument is what makes the decode step capturable
// into a CUDA graph — the arguments of a graph node are frozen at instantiate.
extern "C" __global__ void write_kv(__half* __restrict__ kc, __half* __restrict__ vc,
                                    const float* __restrict__ k, const float* __restrict__ v,
                                    int n_kv, int head_dim,
                                    const int* __restrict__ ctl, int n) {
    const int start_pos = ctl[0];
    const int64_t per = (int64_t)n_kv * head_dim;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)n * per) return;
    const int t = idx / per;
    const int64_t o = idx - t * per;
    const int64_t dst = ((int64_t)start_pos + t) * per + o;
    kc[dst] = __float2half_rn(k[idx]);
    vc[dst] = __float2half_rn(v[idx]);
}

// ---------------------------------------------------------------------------
// decode attention: one block per (kv head, kv split); 7 warps cover the 7 query
// heads that share the kv head, so K/V are fetched once. Online softmax.
// ---------------------------------------------------------------------------

#define KV_TILE 32

extern "C" __global__ void __launch_bounds__(256)
attn_decode(const __half* __restrict__ kc, const __half* __restrict__ vc,
            const float* __restrict__ q, float* __restrict__ o,
            float* __restrict__ part_m, float* __restrict__ part_l,
            float* __restrict__ part_acc,
            int n_heads, int n_kv, int head_dim, const int* __restrict__ ctl,
            int n_split, int qpk) {
    const int n_ctx = ctl[1];
    const int h = blockIdx.x;
    const int split = blockIdx.y;
    const int lane = threadIdx.x & (WARP - 1);
    const int warp = threadIdx.x >> 5;
    const int chunk = (n_ctx + n_split - 1) / n_split;
    const int begin = split * chunk;
    const int end = min(n_ctx, begin + chunk);
    const int hd4 = head_dim >> 2;

    __shared__ __half ks[KV_TILE][128];
    __shared__ __half vs[KV_TILE][128];

    const int64_t per = (int64_t)n_kv * head_dim;
    float m = -INFINITY, l = 0.0f;
    float acc[4] = {0, 0, 0, 0};
    const int qh = h * qpk + warp;
    const bool active = warp < qpk;

    // Qwen attention scales q @ k^T by head_dim^-0.5 before the softmax.
    const float qscale = rsqrtf((float)head_dim);
    float qv[4] = {0, 0, 0, 0};
    if (active) {
        const float* qp = q + (int64_t)qh * head_dim;
        #pragma unroll
        for (int i = 0; i < 4; i++) qv[i] = qp[lane * 4 + i] * qscale;
    }

    for (int base = begin; base < end; base += KV_TILE) {
        const int cnt = min(KV_TILE, end - base);
        // cooperative staging: 128 halves per position, K and V
        for (int r = threadIdx.x; r < cnt * (head_dim / 8); r += blockDim.x) {
            const int p = r / (head_dim / 8);
            const int c8 = r - p * (head_dim / 8);
            const uint4 kk = *(const uint4*)(kc + (int64_t)(base + p) * per + h * head_dim + c8 * 8);
            const uint4 vv = *(const uint4*)(vc + (int64_t)(base + p) * per + h * head_dim + c8 * 8);
            *(uint4*)&ks[p][c8 * 8] = kk;
            *(uint4*)&vs[p][c8 * 8] = vv;
        }
        __syncthreads();
        if (active) {
            for (int p = 0; p < cnt; p++) {
                const __half* kp = ks[p];
                float s = 0.0f;
                #pragma unroll
                for (int i = 0; i < 4; i++) s += qv[i] * __half2float(kp[lane * 4 + i]);
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, o);
                const float mn = fmaxf(m, s);
                const float corr = __expf(m - mn);
                const float w = __expf(s - mn);
                l = l * corr + w;
                m = mn;
                const __half* vp = vs[p];
                #pragma unroll
                for (int i = 0; i < 4; i++) acc[i] = acc[i] * corr + w * __half2float(vp[lane * 4 + i]);
            }
        }
        __syncthreads();
    }

    if (!active) return;
    const int64_t idx = (int64_t)qh * n_split + split;
    if (lane == 0) {
        part_m[idx] = m;
        part_l[idx] = l;
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) part_acc[idx * head_dim + lane * 4 + i] = acc[i];
}

extern "C" __global__ void __launch_bounds__(128)
attn_combine(const float* __restrict__ part_m, const float* __restrict__ part_l,
             const float* __restrict__ part_acc, float* __restrict__ o,
             int n_heads, int head_dim, int n_split) {
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    if (d >= head_dim) return;
    float mx = -INFINITY;
    for (int s = 0; s < n_split; s++) mx = fmaxf(mx, part_m[h * n_split + s]);
    float den = 0.0f, num = 0.0f;
    for (int s = 0; s < n_split; s++) {
        const int64_t idx = (int64_t)h * n_split + s;
        const float w = __expf(part_m[idx] - mx);
        den += part_l[idx] * w;
        num += part_acc[idx * head_dim + d] * w;
    }
    o[(int64_t)h * head_dim + d] = num / den;
}

// ---------------------------------------------------------------------------
// prefill attention: one block per q head, one warp per query, KV tile in smem,
// causal mask, online softmax. `nq` queries starting at `q_pos0`.
// ---------------------------------------------------------------------------

extern "C" __global__ void __launch_bounds__(256)
attn_prefill(const __half* __restrict__ kc, const __half* __restrict__ vc,
             const float* __restrict__ q, float* __restrict__ o,
             int n_heads, int n_kv, int head_dim, int n_ctx, int qpk,
             int q_pos0, int nq) {
    const int h = blockIdx.x;
    const int qg = blockIdx.y;              // query group, 32 per block
    const int q0 = qg * 32;
    const int nqf = min(32, nq - q0);       // queries in this group
    const int lane = threadIdx.x & (WARP - 1);
    const int warp = threadIdx.x >> 5;
    const int hd4 = head_dim >> 2;

    __shared__ __half ks[KV_TILE][128];
    __shared__ __half vs[KV_TILE][128];

    const int64_t per = (int64_t)n_kv * head_dim;
    const int kvh = h / qpk;
    const int p_end = q_pos0 + q0 + nqf;   // exclusive, this group's last key

    // One warp takes up to four queries that are 8 apart, so a 32-token
    // prefill tile stages K/V once instead of once per 8-token sub-tile.
    constexpr int QS = 4;
    float qv[QS][4];
    float acc[QS][4];
    float m[QS], l[QS];
    int nslot = 0;
    const float qscale = rsqrtf((float)head_dim);
    #pragma unroll
    for (int s = 0; s < QS; s++) {
        const int qi = q0 + warp + s * 8;
        if (qi < q0 + nqf) {
            const float* qp = q + (int64_t)qi * n_heads * head_dim + (int64_t)h * head_dim;
            #pragma unroll
            for (int i = 0; i < 4; i++) qv[s][i] = qp[lane * 4 + i] * qscale;
            #pragma unroll
            for (int i = 0; i < 4; i++) acc[s][i] = 0.0f;
            m[s] = -INFINITY;
            l[s] = 0.0f;
            nslot = s + 1;
        }
    }
    // Idle warps must not return: every thread takes part in the cooperative
    // K/V staging and in the block barriers. They simply carry no query.

    for (int base = 0; base < p_end; base += KV_TILE) {
        const int cnt = min(KV_TILE, p_end - base);
        for (int r = threadIdx.x; r < cnt * (head_dim / 8); r += blockDim.x) {
            const int p = r / (head_dim / 8);
            const int c8 = r - p * (head_dim / 8);
            const uint4 kk = *(const uint4*)(kc + (int64_t)(base + p) * per + kvh * head_dim + c8 * 8);
            const uint4 vv = *(const uint4*)(vc + (int64_t)(base + p) * per + kvh * head_dim + c8 * 8);
            *(uint4*)&ks[p][c8 * 8] = kk;
            *(uint4*)&vs[p][c8 * 8] = vv;
        }
        __syncthreads();
        #pragma unroll
        for (int s = 0; s < QS; s++) {
            if (s >= nslot) break;
            const int p_me = q_pos0 + q0 + warp + s * 8;
            const int lim = min(cnt, p_me + 1 - base);
            for (int p = 0; p < lim; p++) {
                const __half* kp = ks[p];
                float sc = 0.0f;
                #pragma unroll
                for (int i = 0; i < 4; i++) sc += qv[s][i] * __half2float(kp[lane * 4 + i]);
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) sc += __shfl_xor_sync(0xFFFFFFFFu, sc, o);
                const float mn = fmaxf(m[s], sc);
                const float corr = __expf(m[s] - mn);
                const float w = __expf(sc - mn);
                l[s] = l[s] * corr + w;
                m[s] = mn;
                const __half* vp = vs[p];
                #pragma unroll
                for (int i = 0; i < 4; i++)
                    acc[s][i] = acc[s][i] * corr + w * __half2float(vp[lane * 4 + i]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int s = 0; s < QS; s++) {
        if (s >= nslot) break;
        const int qi = q0 + warp + s * 8;
        float* op = o + (int64_t)qi * n_heads * head_dim + (int64_t)h * head_dim;
        #pragma unroll
        for (int i = 0; i < 4; i++) op[lane * 4 + i] = acc[s][i] / l[s];
    }
}

// ---------------------------------------------------------------------------
// elementwise
// ---------------------------------------------------------------------------

extern "C" __global__ void silu_mul(float* __restrict__ out, const float* __restrict__ gate,
                                    const float* __restrict__ up, int n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)n) return;
    const float g = gate[i];
    out[i] = (g / (1.0f + __expf(-g))) * up[i];
}

extern "C" __global__ void add_inplace(float* __restrict__ x, const float* __restrict__ y, int n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)n) return;
    x[i] += y[i];
}

extern "C" __global__ void scale_f32(float* __restrict__ x, float s, int n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)n) return;
    x[i] *= s;
}

// ---------------------------------------------------------------------------
// token embedding rows (Q4_K) -> f32 residual stream
// ---------------------------------------------------------------------------

extern "C" __global__ void __launch_bounds__(256)
embed_q4k(const char* __restrict__ W, int64_t row_stride,
          const int* __restrict__ ids, float* __restrict__ out, int n, int k) {
    const int lane = threadIdx.x & (WARP - 1);
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const int nsb = k / 256;
    const int total = n * nsb;
    const int idx = blockIdx.x * warps + warp;
    if (idx >= total) return;
    const int t = idx / nsb;
    const int sb = idx - t * nsb;

    const int g = lane >> 3, tt = lane & 7;
    const int jj = tt >> 1, hf = tt & 1;
    const char* p = W + (int64_t)ids[t] * row_stride + (int64_t)sb * 144;

    const uint4 hd = *(const uint4*)(p);
    const uint4 qs = *(const uint4*)(p + 16 + 32 * jj + 16 * hf);
    const uint32_t A = hd.y, B = hd.z, C = hd.w;
    const bool hi_sc = jj >= 2;
    uint32_t scw, mw;
    if (hi_sc) {
        scw = (C & 0x0F0F0F0Fu) | ((A >> 2) & 0x30303030u);
        mw = ((C >> 4) & 0x0F0F0F0Fu) | ((B >> 2) & 0x30303030u);
    } else {
        scw = A & 0x3F3F3F3Fu;
        mw = B & 0x3F3F3F3Fu;
    }
    const int bsel = 2 * (jj & 1);
    const float2 dm = __half22float2(*(const half2*)&hd.x);
    const float d = dm.x, dmin = dm.y;
    const float scl[2] = {d * byte_f(scw, bsel), d * byte_f(scw, bsel + 1)};
    const float mn[2] = {dmin * byte_f(mw, bsel), dmin * byte_f(mw, bsel + 1)};

    const int off_lo = 64 * jj + 16 * hf;
    const uint32_t w[4] = {qs.x, qs.y, qs.z, qs.w};
    float* dst = out + (int64_t)t * k + sb * 256;
    #pragma unroll
    for (int wi = 0; wi < 4; wi++) {
        const uint32_t vl = w[wi] & 0x0F0F0F0Fu;
        const uint32_t vh = (w[wi] >> 4) & 0x0F0F0F0Fu;
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            dst[off_lo + 4 * wi + b] = scl[0] * (float)((vl >> (8 * b)) & 0xF) - mn[0];
            dst[off_lo + 32 + 4 * wi + b] = scl[1] * (float)((vh >> (8 * b)) & 0xF) - mn[1];
        }
    }
}

// ---------------------------------------------------------------------------
// self-test kernel
// ---------------------------------------------------------------------------

extern "C" __global__ void k_test_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// fp16x2 issue-rate probe: 16 independent accumulator chains, no memory.
extern "C" __global__ void k_hfma2(float* out, int n) {
    const half2 a = __float2half2_rn(1.0f);
    const half2 b = __float2half2_rn(1.0001f);
    half2 acc[16];
    #pragma unroll
    for (int j = 0; j < 16; j++) acc[j] = __float2half2_rn(0.0f);
    for (int i = 0; i < n; i++) {
        #pragma unroll
        for (int j = 0; j < 16; j++) acc[j] = __hfma2(a, b, acc[j]);
    }
    half2 s = __float2half2_rn(0.0f);
    #pragma unroll
    for (int j = 0; j < 16; j++) s = __hadd2(s, acc[j]);
    if (__low2float(s) == 123.0f) out[0] = 1.0f;
}

// Weight-read granularity probe. `chunk` is the number of uint4 fetched
// contiguously before the warp advances, so a sweep of {9, 18, 36, 126} covers
// 144 B (one q4_k block), 288 B, 576 B and a whole 2016 B row. A q4_k block is
// 4.5 sectors, so per-block fetches waste sector traffic; this measures how
// much contiguous span is needed to recover the flat-stream rate.
extern "C" __global__ void k_bw_chunk(const uint4* __restrict__ p, int nrows,
                                     int chunk, int64_t row_u4, float* out) {
    const int lane = threadIdx.x & (WARP - 1);
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    uint32_t acc = 0;
    const int step = gridDim.x * warps * 8;
    for (int row = (blockIdx.x * warps + warp) * 8; row + 8 <= nrows; row += step) {
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            const uint4* rp = p + (int64_t)(row + r) * row_u4;
            for (int c = 0; c < (int)row_u4; c += chunk) {
                const int span = min(chunk, (int)row_u4 - c);
                for (int i = lane; i < span; i += WARP) {
                    uint4 v = __ldg(rp + c + i);
                    acc += v.x + v.y + v.z + v.w;
                }
            }
        }
    }
    if (acc == 0xDEADBEEFu) out[0] = 1.0f;
}

// Same q4_k block walk, but each of the four 8-lane groups owns a different
// row, so all 32 lanes issue instead of 8 with three replicas. This is the
// layout a staged rewrite would use.
extern "C" __global__ void k_bw_rows4(const uint4* __restrict__ p, int nrows,
                                      int64_t row_u4, float* out) {
    const int lane = threadIdx.x & (WARP - 1);
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const int g = lane >> 3;
    const int t = lane & 7;
    uint32_t acc = 0;
    const int nblk = (int)(row_u4 / 9);
    const int step = gridDim.x * warps * 32;
    for (int r0 = (blockIdx.x * warps + warp) * 32; r0 + 32 <= nrows; r0 += step) {
        #pragma unroll
        for (int rr = 0; rr < 8; rr++) {
            const uint4* rp = p + (int64_t)(r0 + rr * 4 + g) * row_u4;
            for (int sb = 0; sb < nblk; sb++) {
                uint4 hd = __ldg(rp + sb * 9);
                uint4 qs = __ldg(rp + sb * 9 + 1 + t);
                acc += hd.x + qs.x + qs.y + qs.z + qs.w;
            }
        }
    }
    if (acc == 0xDEADBEEFu) out[0] = 1.0f;
}

// global memory read bandwidth probe (uint4 streaming)
extern "C" __global__ void k_bw_read(const uint4* __restrict__ p, int64_t n4, float* out) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    uint32_t acc = 0;
    for (; i < n4; i += stride) {
        uint4 v = __ldg(p + i);
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 0xDEADBEEFu) out[0] = 1.0f;
}

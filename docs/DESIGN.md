# Design notes and measurements

Everything here was measured on the machine this repository was written on:
2 × Tesla P100-PCIE-16GB (`sm_60`, 56 SMs, 1.328 GHz, ECC on, PHB topology),
CUDA 12.8, Ryzen-class host. Model: `qwen2.5-7b-instruct-q4_k_m`
(28 layers, 28 heads, 4 KV heads, head_dim 128, 3584 embd, 18944 ffn,
152064 vocab, 4.43 GiB of weights after q6_k repacking).

## 1. What the hardware forces

**No DP4A.** `sm_60` predates the integer dot product present from `sm_61`.
Every int8 quantised dot is four scalar multiplies plus sign handling. The
llama.cpp fork on this machine works around it with a separate fp16 kernel and
measures 2.0× on 8B-class models.

**Full-rate `HFMA2`.** GP100 performs two fp16 FMAs per FP32 issue slot
(19 TFLOPS fp16 vs 9.5 TFLOPS fp32). That is the whole reason this engine's
weight path is fp16 rather than int8.

**No tensor cores.** Prefill cannot fall back on MMA, so the only way to reach
the compute roof is register blocking and shared-memory reuse.

**Memory.** The datasheet says 732 GB/s; a simple streaming read over 256 MiB
reaches **594 GB/s** here (ECC on), which is the real ceiling every bandwidth
number below is measured against. The `streaming_bandwidth` test reproduces it.

## 2. The weight kernels

### 2.1 Subnormal dequantisation

A 4-bit value `q ∈ [0,15]` written as a raw fp16 bit pattern is the *subnormal*
`q · 2^-24` — exact, and free: `PRMT` assembles two nibble bytes into two half
words with no conversion instruction. The same trick covers q6_k
(`q ∈ [0,63]` → `q · 2^-24`) and q5_k.

The weight is then `d · sc · q`, so the scale must carry the compensating
`2^24`. Putting `2^24` on the weight would overflow fp16 for realistic `d·sc`
(≥ 2^11), so instead the **activation** carries it: `prep_act` scales each
32-value block by a power of two so that `max|x| < 2^15`, and the kernel
multiplies the fp16 dot by `d·sc·2^(24-k)` in fp32. The power-of-two scale is
exact, so this costs no accuracy.

The offset (`-dmin·m` for q4_k/q5_k, the implicit `-32` for q6_k) is applied
once per 16 values in fp32, using sums that `prep_act` computes exactly
(`ys[s] = { 2^(24-k), Σ x over the 16 values }`). Only 16 products are ever
accumulated in fp16, which keeps the error at the same level as a
dequantise-to-fp16-then-dot implementation while running twice as fast.

### 2.2 Layouts

| type | block | on-device layout | why |
|---|---|---|---|
| q4_k | 256 values | 144 B native (`d, dmin, scales[12], qs[128]`) | 144 = 9 × 16, so every `uint4` probe is aligned |
| q6_k | 256 values | **repacked** to 224 B (`d@0, sc2[16]@4, ql[128]@32, hi[64]@160`) | native blocks are 210 B, i.e. 2 mod 16; every 16-byte load would be misaligned |

The q6_k repack also transposes the high bits into four 2-bit fields per byte,
so `spread2()` can raise a byte's four fields to four bytes with two multiplies
and two masks instead of sixteen shifts (shifts and `PRMT` are half rate on
GP100). `sc2` is permuted per lane so a lane reads its two sub-block scales with
one 16-bit load.

### 2.3 Decode GEMV

One warp computes one row. Eight lanes cover a 256-value super-block; each lane
owns two 16-value sub-blocks — the low and high nibbles of the same 16 bytes.
That is one 16-byte load per 32 weight values, which is the widest probe the
formats allow. Per iteration a warp touches four super-blocks (four lane
groups), and the four groups' `qs` reads are contiguous.

`prep_act` output is staged into shared memory eight super-blocks at a time
(4 KiB of `y` + 1 KiB of sums for one column).

Measured on a 16384 × 18944 q4_k matrix (174 MB):

| version | time | bandwidth |
|---|---:|---:|
| activation loads from global, 16 B probes | — | 126 GB/s |
| + stage activations in shared memory | — | 231 GB/s |
| + 16-byte staging loads instead of 2-byte | — | **385 GB/s** |
| + two rows per warp, 8-column tile (prefill) | — | 97 GB/s, 2.7 TFLOPS |

The 2-byte staging detail is worth calling out: the first staging loop loaded
one `__half` per iteration per thread, which is a 64-byte transaction per warp
instruction. Vectorising it to one `uint4` per thread cut the staging
instruction count 8× and moved the kernel from 231 to 385 GB/s.

### 2.4 Prefill

The same kernel is instantiated for 1/2/4/8 columns, so one weight fetch serves
up to 8 tokens. For 8 columns the launcher selects a two-rows-per-warp variant
(halving the activation traffic per MAC) and a 4-super-block staging window so
two blocks stay resident per SM.

This is the known weak point: 2.7 TFLOPS of the 19 TFLOPS roof. The per-MAC
cost is dominated by shared-memory reads (four 16-byte loads per column per
sub-block) and by the fact that a warp only ever holds *one* row's weights.
The fix is a fused dequantise + GEMM: dequantise a 64 × 128 weight tile into
shared memory once, then run a register-blocked 64 × 32 GEMM over a 32-token
tile, which amortises both the dequantisation and the shared-memory traffic over
32 columns instead of 8. That is the next piece of work; decode does not depend
on it.

## 3. Attention

Qwen2.5-7B is GQA 7:1 (28 query heads, 4 KV heads). Two kernels:

**Decode** (`attn_decode` + `attn_combine`). One block per (KV head, KV split);
the 7 warps that share a KV head each own one query head, so K/V is fetched
once per position instead of 7 times. 32 positions at a time are staged into
shared memory cooperatively; each warp runs an online-softmax loop and writes
partial `(m, l, acc)`. A second kernel combines the splits. Without the GQA fold
the KV re-reads alone are ~7× the unique traffic, which is fatal at long
context.

At the time of writing, `attn_splits = 12` is best for both a 256-token and a
512-token benchmark: 4 leaves SMs idle, 32 gives each block too few positions to
amortise the combine.

**Prefill** (`attn_prefill`). One block per query head, one warp per query,
KV tiles of 32 positions staged in shared memory, causal mask per query, same
online softmax. 8 queries per block.

Both are verified against a CPU reference (see `crates/p100/tests/kernels.rs`):
prefill to 3e-6, decode to 2e-6 relative error, for 1/4/16 splits and four
causal configurations.

## 4. Multi-GPU

Two P100s over PCIe with a host bridge (`nvidia-smi topo -m` reports `PHB`) and
no NVLink. Tensor parallelism would need a per-layer all-reduce; at batch 1 that
is latency, not bandwidth, and it costs more than the extra SMs return. So the
engine uses **layer split**: layer `l` lives on device `l·ndev/nlayer`, each
device keeps its own layers' KV cache, and the only cross-device traffic is the
residual row (14 KiB) at each boundary, staged through pinned host memory.

Verified: the same prompt on one and two GPUs produces coherent text; a single
stream gets no speedup (there is nothing to pipeline), which is expected.

## 5. Numerical error budget

| source | measured |
|---|---|
| q4_k GEMV vs CPU dot | 1e-4 of the term energy |
| q6_k GEMV vs CPU dot | 3e-4 (16 six-bit products accumulate in fp16) |
| prefill attention vs CPU | 3e-6 |
| decode attention vs CPU | 2e-6 |
| batched prefill vs token-at-a-time decoding | same argmax, 3.6-3.9 % logit RMS spread |
| end-to-end output | coherent on all prompts tried |

The batched-vs-streaming spread is fp16 KV rounding amplified over 28 layers,
not a layout bug: layer 0's K/V is bit-identical between the two paths, and
layer 1's differs by one fp16 ULP.

## 6. Validation against llama.cpp

Two engines that implement the same arithmetic should agree on perplexity, and
nothing else catches a kernel that produces fluent-but-wrong output. Same file,
same 512-token chunks, same weights:

| corpus | llama-perplexity | trav-p100 |
|---|---:|---:|
| `README.md` + `docs/DESIGN.md`, `-c 512` | 15.7617 | 15.7830 |
| three repeated sentences, `-c 256` | 1.0048 | 1.2278 |
| token ids for a mixed prose/number string | 35 ids | identical 35 ids |

The markdown corpus is the meaningful one: a 0.13% difference over 2044 scored
tokens is the fp16-accumulation and reduction-order budget, not a behavioural
difference.

The first version of this engine was missing the `head_dim^-0.5` scaling of
`q @ k^T`. Output was still fluent and answers to easy questions were correct,
which is exactly why output inspection is not a test: perplexity was 3.3 where
llama.cpp measured 1.0, and the batched-vs-streaming logit spread was ~10%
instead of ~0.1%. Both symptoms disappeared with the scale in place.

## 7. Deliverables and how to reproduce

```console
cargo test -p trav-p100 --release -- --nocapture      # kernel correctness + bandwidth
./target/release/trav-p100 self-test --model ... --gpus 0 --prefill-cols 8
./target/release/trav-p100 perplexity --model ... --file README.md --ctx 512 --chunks 4
./target/release/trav-p100 bench --model ... --profile configs/qwen2.5-7b-instruct-q4_k_m.toml
```

The `gemv_bandwidth` test prints the achieved GB/s for the 1-column and
8-column kernels so a regression in the memory path is obvious without running
the model.

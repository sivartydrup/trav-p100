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

The q6_k high-bit plane is also reordered: a lane owns two 32-value runs 64
values apart, and it needs one 16-value half of each. Storing those four halves
as `[lo.hf0, hi.hf0, lo.hf1, hi.hf1]` makes the pair one 8-byte load instead of
two 4-byte gathers, which took the q6_k GEMV from 318 to 341 GB/s and lm_head
from 394 to 430 GB/s (decode 67.8 → 68.9 tok/s).

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

The single-column GEMV over the projection shapes Qwen2.5-7B actually launches
(`gemv_bandwidth`, one P100):

| projection | rows × k | GB/s | ms |
|---|---:|---:|---:|
| q_proj / o_proj | 3584 × 3584 | 276 | 0.03 |
| k_proj / v_proj | 512 × 3584 | 107 | 0.01 |
| gate_proj / up_proj | 18944 × 3584 | 342 | 0.11 |
| down_proj | 3584 × 18944 | 349 | 0.11 |
| lm_head | 152064 × 3584 | 366 | 0.84 |

Summed over 28 layers plus lm_head that is 12.3 ms/token; the full untraced
step now measures 13.4 ms, so launch gaps and the kernels the sweep does not
cover add ~1 ms. Before the step was captured as a graph, the sixteen launches
per layer (448 per token) at a few microseconds each were most of that gap. The
kernel itself sits at 46–62% of the ~594 GB/s streaming rate, so the decode
headroom is split between the access pattern (one warp per row reads 144-byte
blocks, not 128-byte sectors) and launch count, not activation staging.

`wblk_bandwidth` now splits that access-pattern share from the arithmetic. It
reads the same 38.2 MB as the gate/up GEMV with no dequantise, varying only the
contiguous span fetched per row:

| fetch span per row | GB/s |
|---|---:|
| flat `uint4` stream, same bytes | 572 |
| whole 2016 B row, contiguous | 505 |
| 576 B (4 super-blocks), contiguous | 485 |
| **4 lane-groups x 4 rows, `hd`+`qs` split (what `gemv_body` does)** | **466** |
| 288 B (2 super-blocks), contiguous | 450 |
| 144 B (1 super-block), 9 of 32 lanes | 256 |

The 144 B row is the wrong comparison for this kernel: `sb0 + g` makes the four
lane-groups work on four consecutive super-blocks, so all 32 lanes issue and
576 B of one row are in flight, rather than one block with 8 lanes and three
replicas. The apples-to-apples figure is 466 GB/s — the same block walk with the
same `hd` (4 x 16 B) + `qs` (4 x 128 B) request split and no arithmetic.

So the real kernel's 399 GB/s is 14% below its own read-only ceiling, and that
14% is the dequantise plus HFMA2, not the fetch. A perfectly contiguous fetch
would reach only 485. That rules out shared-memory weight staging: it buys
485/466 = 1.04x on the read side while turning one 16-byte memory instruction
into three (LDG + STS + LDS), in a kernel whose design budget is one load per
16 weight bytes. The remaining decode headroom was the ~0.9 ms/token of launch
and host gap at the time; the CUDA graph and GPU-side greedy selection have cut
that to ~0.3 ms, so the weight fetch is the larger lever now.

Two kernels were grid-starved rather than resource-starved. `rms_norm` and
`add_rms_norm` take `blockIdx.x` as the row, so the decode call (one row) ran 8
warps on a single SM although 24 and 32 regs/thread would allow 8 blocks.
Raising `__launch_bounds__` to 1024 and using 1024 threads for the single-row
case puts 32 warps on that SM with the existing block-wide reduction (`red[32]`
already covers 32 warps); multi-row prefill keeps 256. Over a 136-token decode:
`add_rms_norm` 15.0 → 8.0 us, `rms_norm` 10.3 → 6.7 us, decode 66.4 → 67.7
tok/s. Both now sit at 6-8 GB/s, which is most of one SM's rate, so the next
step there would need a cross-block reduction and a second launch.

Folding `prep_act` into the producer kernels was tried and reverted. `rms_norm`
and `add_rms_norm` are one block per row by construction (the RMS reduction is
block-wide), so the prep work moves off the 14 blocks `prep_act` naturally
launches and onto 32 warps of a single SM: `rms_norm`+prep 9.2 → 10.3 us and
`add_rms_norm`+prep 10.5 → 11.3 us per layer. Only `silu_mul`, which is already
grid-parallel, got cheaper (4.9 → 4.1 us). End to end decode did not move
(67.7 tok/s against 67.7), so the 112 prep launches per token buy less than the
0.9 ms/token gap over 537 launches suggests — the queued launches overlap with
GPU work.

#### Rows per warp, and why split-K is not used here

Each GEMV block stages its whole activation row into shared memory, so the
staging traffic is `k·2 B` per block against `rows_per_warp · 8 · row_bytes` of
weights: more rows per warp amortises the staging, but fewer blocks means fewer
warps per SM. Both sides were measured on the shapes the model launches, after a
100 ms warm-up (short runs sit at the idle clock and mislead, which is what the
older `gemv_bandwidth` numbers did), by `gemv_split_bandwidth`:

| shape | best | GB/s |
|---|---|---:|
| q/o 3584×3584 q4_k | rows 2 | **319** |
| k/v 512×3584 q4_k | rows 2 | **115** |
| gate/up 18944×3584 q4_k | rows 4 | **400** |
| down 3584×18944 q6_k | rows 4 | **377** |
| lm_head 152064×3584 q6_k | rows 4 | **395** |

Split-K does not help any of them. The GEMV runs one block per SM at eight
warps, so 56 blocks are already the whole card; splitting k only adds a zeroed
output, an atomic accumulate and a second launch. 3584 rows q4_k: rows 2 =
319 GB/s, rows 2 with 2 splits = 220, rows 4 with 2 splits = 182. `gemv_r` keeps
the `nsplit` parameter so the sweep keeps pinning it, and the kernel's atomic
path is covered by the reference tests (`nsplit` 2 and 3).

Getting the rule right was worth 68.9 → 71.9 tok/s at the time (73.8 tok/s
today): the old table asked for 8 rows on the 18944-row projections and 2 on
q6_k, where 4 rows and 4 rows are best.


#### The decode step as a CUDA graph

A decode token is ~424 dependent kernel launches, so the step is captured once
per sequence on its own stream and replayed for the rest of the generation
(`Seq::graph`). Three things had to change for the capture to be legal and the
replay to be correct:

* **The two position-dependent scalars moved to device memory.** `write_kv`'s
  start position and `attn_decode`'s cache length are now read from a small
  buffer (`ctl`, 8 bytes, re-uploaded with the token id and rope position before
  each step). A graph node's arguments are frozen at instantiate, so a scalar
  argument would pin the graph to one position.
* **The decode attention split count is fixed** at `attn_splits` instead of
  `min(attn_splits, n_ctx)`. A split whose range is empty writes `l = 0` and
  drops out in the combine, so the clamp was only ever a saving.
* **The logits copy back to the host left `run_tile`.** A blocking DtoH cannot be
  captured; the copy now happens after the graph launch (or after the plain
  launches), from the byte offset `run_tile` returns.

Not captured: prefill tiles, `TRAV_DUMP` runs, and multi-device layer splits —
the residual hop between devices is a blocking copy and would have to become
async first. `TRAV_NO_GRAPH=1` forces the plain path for A/B comparison.

Measured with that switch, three reps each: **73.62 tok/s with graphs, 73.43
without** — about +0.3%. The reason is that the launch gap this was meant to
remove is not where the time went. Summing kernel durations from an `nsys` trace
gives 97.6% GPU-busy and ~0.3 ms per token between kernels, but node-level graph
tracing serialises the replay and makes the profiled step ~7% slower than an
untraced one (14.4 ms of kernels against a 14.7 ms wall; untraced the step is
13.4 ms), so read the trace as a share measurement. What is left in that gap is
the GPU-side hand-off between ~424 dependent kernels, which a graph does not
remove — fewer, larger kernels would.

Within the trace, 86% of the step is weight GEMV (12.3 ms: 5.6 ms on the
3584-row q4_k projections, 3.3 ms on the q6_k down projection, 3.1 ms on the
512-row k/v), 4.8% is attention (0.7 ms) and 9.5% is norm, activation prep,
SiLU, RoPE and KV write (1.4 ms).

### 2.4 Prefill

**Whole-prompt tiles.** A tile is pushed through all 28 layers before the next
one starts, so a 512-token prompt is a single tile (`MAX_PREFILL_COLS = 512`)
and every weight matrix is read once per prompt rather than once per 32 tokens.
That is llama.cpp's layer-major batching, reached by making the tile the prompt.

**Two weight paths.** Matrices that can fill the card with 128-row blocks (q, o,
gate, up, down) are dequantised once per prompt into an fp16 scratch — one
matrix at a time, 136 MiB for gate/up, 1.09 GiB when lm_head runs through it —
and then multiplied by a plain fp16 GEMM. The 512-row k/v projections stay on
the fused quantised kernel, where a 128-row tile would leave half the SMs idle
and split-K across `blockIdx.z` fills it instead.

**fp16 GEMM** (`gemm_f16`). 128 × 128 block, 256 threads, 8 × 8 register tile
(64 independent `half2` chains), 32-value stages double buffered *through
registers* so stage `s+1`'s global loads overlap stage `s`'s compute, and a fold
to fp32 every 256 k values. Each k-pair costs four `LDS.128` for 64 `HFMA2`.
Fragments are two 16-byte loads 64 entries apart rather than one 32-byte load:
a 16-byte stride across the warp spreads its 16 chunks over the 32 banks twice
instead of four times. Measured with 512 columns on the gate/up shape
(18944 × 3584, q4_k): **13.6 TFLOPS** (`gemm_f16_rate`). The 8 × 8 tile needs
~222 registers, so one block (8 warps) is resident per SM; an
ablation probe on the pre-barrier kernel (since removed) measured the same loop at **15.6 TFLOPS** with
the shared loads and the staging deleted, and 12.9 with the loads but no
staging. Narrower register tiles that admit two blocks per SM (64 × 128 with
8 × 4, or 128 × 128 with 4 × 8) measure within 1% of the wide one, so the
wider fragment stays.

**Dequantise pass** (`dequant_q4k_f16`, `dequant_q6k_f16`). One thread per 32
values, the same lane task the packed kernels use, so the values are written as
four 16-byte stores rather than 32 two-byte ones. It reads the packed weights
once and writes the fp16 copy (12.5 GiB per prefill for q4_k + q6_k) at
~250 GB/s. This is 9% of prefill; llama.cpp's equivalent step is 14%.

**Why not fuse it.** The fused dequantise+GEMM is still in the tree and still
runs k/v, but its staging (dequantise + activation copy + two barriers per
stage) costs 23% of the loop time: the best fused tile, 64 rows × 64 columns
with an 8 × 4 register tile and 64-value stages, measured 10.5 TFLOPS at 512
columns (7.3 at 32 columns). Taking the dequantise out of the inner loop is what
pays for the 8 × 8 tile.

Both activation paths are prepared per column by `prep_act_col`, which now loads
four floats at a time and targets `max|x| ≈ 2^3` (it was 2^6). The smaller
target keeps the fp16 chains far from overflow, which is what allows a fold
every 256 k values instead of every 128.

**Measured** (one P100, pp512): **823 tok/s**, 1.22 ms per token. That is
0.93 ms of fp16 GEMM, 0.11 ms of dequantise, 0.09 ms of prefill attention,
0.03 ms of activation prep, 0.03 ms of fused k/v GEMM and 0.03 ms of SiLU,
norms, RoPE, embedding and lm_head. The fused path still handles k/v and any
tile shorter than 24 tokens (8-column GEMV).

Against llama.cpp's prefill (443 ms GEMM + 87 ms dequantise + 25 ms attention
for pp512), the dequantise step is cheaper here and attention is 2× more
expensive; the difference that remains is the GEMM itself, 13.6 TFLOPS
against 14.2.

llama.cpp ships a SASS-patched variant of its kernel that renames registers to
remove the same-bank source pairs ptxas leaves behind. The headroom was measured
before doing that work, with two ablations of `gemm_f16` on the gate shape (both
compute nonsense, only the timing counts), against a real kernel that measured
12.0 TFLOPS at the time and 13.6 after the later barrier removal:

| variant (pre-barrier kernel) | TFLOPS |
|---|---:|
| real kernel | 12.0 |
| LDS reads from a fixed address (no bank spread) | 12.6 |
| no LDS in the inner loop at all | 12.5 |

So the whole load path — bank conflicts included — costs 4–6% here, and the
bank-conflict count (28% of FMAs, with 16-byte fragments whose operand
registers necessarily span all four banks) does not translate into stalls: with
two warps per scheduler they overlap. A SASS renaming pass would need an
assembler (CuAssembler) that is not installable on this machine, so it is not
where the next 10% lives. What is left is issue efficiency with eight warps and
222 registers per thread: one block per SM, and no tile shape measured (64×128
with 8×4, 128×128 with 4×8) does better than 1%.

## 3. Attention

Qwen2.5-7B is GQA 7:1 (28 query heads, 4 KV heads). Two kernels:

**Decode** (`attn_decode` + `attn_combine`). One block per (KV head, KV split);
the 7 warps that share a KV head each own one query head, so K/V is fetched
once per position instead of 7 times. 32 positions at a time are staged into
shared memory cooperatively; each warp runs an online-softmax loop and writes
partial `(m, l, acc)`. A second kernel combines the splits. Without the GQA fold
the KV re-reads alone are ~7× the unique traffic, which is fatal at long
context.

`attn_splits` is flat between 8 and 32 on the 512-token benchmark
(73.3–73.8 tok/s); 4 leaves SMs idle and loses ~3%. The tuner's current pick
for this model is 8, resolved on prefill.

**Prefill** (`attn_prefill`). One block per query head, one warp per query,
KV tiles of 32 positions staged in shared memory, causal mask per query, same
online softmax. 8 queries per block.

Both are verified against a CPU reference (see `crates/p100/tests/kernels.rs`):
prefill to 3e-6, decode to 2e-6 relative error, for 1/4/16/32 splits and four
causal configurations.

## 4. Multi-GPU

Two P100s over PCIe with a host bridge (`nvidia-smi topo -m` reports `PHB`) and
no NVLink. Tensor parallelism would need a per-layer all-reduce; at batch 1 that
is latency, not bandwidth, and it costs more than the extra SMs return. So the
engine uses **layer split**: layer `l` lives on device `l·ndev/nlayer`, each
device keeps its own layers' KV cache, and the only cross-device traffic is the
residual row (14 KiB) at each boundary, staged through pinned host memory.

Verified: the same prompt on one and two GPUs produces coherent text. A single
stream gets no meaningful speedup — 832 tok/s prefill / 74.7 decode on two cards
against 823 / 73.8 on one — because there is nothing to pipeline.

## 5. Numerical error budget

| source | measured |
|---|---|
| q4_k GEMV vs CPU dot | 1e-4 of the term energy |
| q6_k GEMV vs CPU dot | 3e-4 (16 six-bit products accumulate in fp16) |
| q4_k GEMM vs CPU dot | 3e-4 (fp16 weight value, 128 products per stage) |
| q6_k GEMM vs CPU dot | 6e-5 |
| prefill attention vs CPU | 3e-6 |
| decode attention vs CPU | 2e-6 |
| batched prefill vs token-at-a-time decoding | same argmax, 0.05-0.1 % logit RMS spread |
| end-to-end output | coherent on all prompts tried |

The batched-vs-streaming spread is fp16 KV rounding and the GEMM's different
reduction order, amplified over 28 layers, not a layout bug; it grows from
0.05% to 0.1% once a tile longer than 24 tokens runs the fp16 GEMM instead of
the 8-column GEMV.

## 6. Validation against llama.cpp

Two engines that implement the same arithmetic should agree on perplexity, and
nothing else catches a kernel that produces fluent-but-wrong output. Same file,
same 512-token chunks, same weights, measured on this checkout:

| check | llama.cpp | trav-p100 |
|---|---:|---:|
| 4 x 512-token chunks of `README.md` + `docs/DESIGN.md` | 11.9095 | 11.9101 |
| token ids for the 11k-token corpus | equal | equal (last token is a trailing EOS in `llama-tokenize`) |

The scoring window matters more than any kernel detail. `llama-perplexity`
requests logits from position `n_ctx/2` and scores `n_ctx/2 - 1` targets per
chunk (257..511 of a 512-token window); the first half is context only.
`trav-p100 perplexity` now defaults to the same convention (`--ppl-skip`
defaults to `ctx/2 + 1`). On the current docs the two engines score 11.9095 and
11.9101 over 1020 scored tokens, a 0.01% difference; trav-p100's per-chunk
values are `[11.3502, 17.0166, 14.4477, 7.2108]`.

Scoring all 511 positions instead (`--ppl-skip 0`) gives 15.96: the 256-token
cold start is included, so that number is not comparable to `llama-perplexity`.
Comparing the two conventions produced a false "44% accuracy gap" during this
work; the like-for-like number is the 0.01% above.

The first version of this engine was missing the `head_dim^-0.5` scaling of
`q @ k^T`. Output was still fluent and answers to easy questions were correct,
which is exactly why output inspection is not a test: perplexity was 3.3 where
llama.cpp measured 1.0, and the batched-vs-streaming logit spread was ~10%
instead of ~0.1%. Both symptoms disappeared with the scale in place.

## 7. Deliverables and how to reproduce

```console
cargo test -p trav-p100 --release -- --nocapture      # kernel correctness + issue/memory ceilings
./target/release/trav-p100 self-test --model ... --gpus 0 --profile configs/qwen2.5-7b-instruct-q4_k_m.toml
./target/release/trav-p100 perplexity --model ... --file /tmp/corpus.md --ctx 512 --chunks 4
./target/release/trav-p100 bench --model ... --profile configs/qwen2.5-7b-instruct-q4_k_m.toml
```

The `gemv_bandwidth`, `gemm_throughput`, `hfma2_peak`, `dequant_f16_matches_gguf`
and `gemm_f16_matches_dense` tests print the achieved GB/s, TFLOPS, the pure-HFMA2
ceiling and the fp16 prefill path's accuracy, so a regression in the memory or
issue paths is obvious without running the model. `self-test` must report the
same argmax and at most a 0.1% logit spread; run it on a prompt longer than 24
tokens to exercise the GEMM path.

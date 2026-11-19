# trav-p100

An LLM inference engine written from scratch in Rust + CUDA, built for one
specific GPU: the **NVIDIA Tesla P100 (GP100, `sm_60`)**. No llama.cpp, no
ggml, no cuBLAS — the GGUF reader, the quantised kernels, the KV cache, the
scheduler, the sampler and the HTTP server are all in this repository.

The point is not to reimplement a generic engine. It is that `sm_60` is a very
peculiar chip, and generic kernels leave a lot on the table there:

| property | consequence |
|---|---|
| no DP4A / INT8 dot product | every int8 quantised dot costs four scalar multiplies |
| full-rate `HFMA2` (2× FP32) | fp16 maths is *cheaper* than the usual int8 emulation |
| no tensor cores | prefill is CUDA-core bound, so data reuse in registers/smem is everything |
| 732 GB/s HBM2, ~594 GB/s measured | decode is bandwidth bound; wasted sectors are wasted tokens/s |
| no NVLink on PCIe cards | multi-GPU must avoid per-layer all-reduce |

Every kernel here is shaped by those five facts. The details, including the
measurements that drove each decision, are in
[`docs/DESIGN.md`](docs/DESIGN.md).

## Status and measured performance

One P100, `qwen2.5-7b-instruct-q4_k_m` (4.43 GiB of weights), greedy, 8192-token
KV cache. `llama-bench` numbers are from the P100-tuned llama.cpp fork already
on this machine (upstream `b10660` + the GP100 patch set).

| engine | prefill (pp512) | decode (tg128) | perplexity |
|---|---:|---:|---:|
| llama.cpp, P100-patched | 781 tok/s | 66.5 tok/s | 15.762 |
| **trav-p100** | **180 tok/s** | **65.1 tok/s** | **15.783** |

The perplexity column is the same 512-token-chunk evaluation of the same file
(`llama-perplexity -c 512` vs `trav-p100 perplexity --ctx 512`). A 0.13%
difference is what two correct implementations of the same arithmetic produce
under different reduction orders; it is the check that matters most, because
it catches bugs that still generate fluent text.

Decode is at parity; prefill is the known weak spot (see
[Limitations](#limitations)). Two effects matter for decode: the engine streams
**423 GB/s** from HBM in the single-column GEMV (71% of this card's measured
594 GB/s ceiling), and the lm_head + attention + launch overhead add ~4 ms on
top of the ~10.5 ms of weight traffic.

Two P100s with layer split, same model: 69.2 tok/s decode, 180 tok/s prefill.
Layer split does not help a single stream (there is no pipelining across
requests yet) — it is there so that a model that does not fit on one card still
runs, which is the 27B case.

## What works

* **GGUF v3** reader with split shards (`.gguf` + `-00001-of-00002`).
* **q4_k and q6_k** weights. `qwen2` / `qwen3` architectures with GQA,
  QKV biases, RoPE (rotate-half) and RMSNorm.
* **Qwen2/3 byte-level BPE** tokenizer built straight from the GGUF vocab
  (llama.cpp's `qwen2` pre-tokeniser regex, including the look-ahead) and the
  ChatML template that ships inside the model.
* Batched prefill (8 columns per GEMV), split-KV flash decoding, online
  softmax, fp16 KV cache.
* **Layer-split across any number of P100s**, chosen with `--gpus 0,1`.
* Sampling: temperature, top-k, top-p, min-p, repetition penalty, seeds.
* OpenAI-compatible HTTP API with SSE streaming.
* A **tuner** that sweeps the runtime knobs and writes a per-model TOML profile.

## Quick start

Requirements: Rust 1.82+, CUDA 12.x with `nvcc` (kernels are compiled to a
`sm_60` cubin at build time and embedded in the binary), and a P100. Building
does *not* need the CUDA development package: the driver API is loaded at runtime
with `dlopen`.

```console
cargo build --release

# what is in the box
./target/release/trav-p100 info
./target/release/trav-p100 inspect /models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf

# one prompt
./target/release/trav-p100 run \
  --model /models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  --gpus 0 --profile configs/qwen2.5-7b-instruct-q4_k_m.toml \
  --prompt "Explain what a KV cache is, in two sentences."

# interactive
./target/release/trav-p100 chat --model ... --gpus 0

# throughput
./target/release/trav-p100 bench --model ... --gpus 0 --prompt-tokens 512 --n-predict 128

# OpenAI-compatible API
./target/release/trav-p100 serve --model ... --gpus 0,1 --addr 127.0.0.1:8080

curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "one short sentence about Paris"}],
  "max_tokens": 64, "stream": true }'
```

`--gpus` takes a comma-separated list of CUDA device indices. `--ctx` overrides
the KV capacity from the profile; `--profile` loads a tuned TOML.

## Tuning for a model

```console
./target/release/trav-p100 tune --model ... --gpus 0 --ctx 8192 \
    --out configs/qwen2.5-7b-instruct-q4_k_m.toml
```

The tuner measures prefill and decode for each candidate and writes the winner:

```toml
ctx = 8192
prefill_cols = 8   # tokens per prefill tile = columns of the fused GEMV
attn_splits = 12   # flash-decoding KV splits (SM coverage vs combine cost)
```

`prefill_cols = 8` is worth ~1.7× over `4` on this card; `attn_splits` above 16
starts to hurt because each split gets too few positions to amortise the
cross-split combine.

## Verification

Correctness is checked at three levels, not by eyeballing output:

```console
cargo test -p trav-p100            # kernels vs the GGUF reference dequantiser
./target/release/trav-p100 self-test --model ... --gpus 0 --prefill-cols 8
```

* `cargo test -p trav-p100` compares every GEMV instantiation (q4_k / q6_k ×
  1/2/4/8 columns, including the two-rows-per-warp variant) against a CPU
  dequantise-and-dot, and does the same for prefill attention (four causal
  configurations), split-KV decode attention (1/4/16 splits) and RoPE + KV
  write.
* `self-test` runs the same prompt through the batched prefill path and through
  token-at-a-time decoding, then compares the logits. They must pick the same
  argmax; the residual spread is fp16 KV rounding (≈4% of the logit scale).
* `perplexity` on a text file is checked against `llama-perplexity` on the same
  file: 15.783 vs 15.762 (0.13%) on `README.md + docs/DESIGN.md` at `--ctx 512`,
  and 1.228 vs 1.005 on a highly repetitive corpus. `tokenize` prints ids that
  can be diffed against `llama-tokenize`.
* `bench` doubles as a smoke test that the numbers are where the design says
  they should be.

## Design in one page

```mermaid
flowchart LR
  A[tokens] --> B[embed q4_k] --> C[rmsnorm]
  C --> D[prep_act<br/>f32 → f16, per-32-block scale + 16-value sums]
  D --> E[fused dequant GEMV q/k/v<br/>HFMA2 on subnormal half]
  E --> F[RoPE] --> G[KV cache f16]
  G --> H{one token?}
  H -- yes --> I[split-KV flash decode]
  H -- no --> J[tiled causal prefill attention]
  I --> K[o_proj]
  J --> K
  K --> L[residual + rmsnorm]
  L --> M[gate/up GEMV] --> N[SwiGLU] --> O[down GEMV]
  O --> P[residual] --> Q[lm_head q6_k] --> R[sampler]
```

Four decisions carry most of the performance:

1. **Dequantise into subnormal halves.** A 4-bit value `q` maps exactly onto the
   fp16 subnormal bit pattern `q * 2^-24`, so one `PRMT` turns two nibbles into
   two half words with no conversion instruction. Weights are then multiplied by
   `d·sc·2^24` and accumulated with `HFMA2`; the offset/min terms are applied
   once per 16 values in fp32, using exact activation sums computed in the
   preparation kernel. This is what makes decode competitive on a chip without
   DP4A.
2. **16-byte loads, everywhere.** q4_k blocks are 144 B (9 × `uint4`), so a lane
   fetches exactly one `uint4` per 32 weight values. q6_k's 210-byte blocks are
   *not* aligned, so the loader repacks them to 224 B with `ql` and the 2-bit
   high plane at fixed 16-byte offsets (and the high bits pre-packed four
   values per byte, so a multiply-and-mask raises them to bytes).
3. **Stage activations in shared memory.** Streaming weights through L1 evicts
   the 7 KiB activation row, and the re-reads then cost ~4× the weight traffic.
   Staging eight super-blocks at a time was worth 1.84× on the decode kernel
   (126 → 231 GB/s), and vectorising the staging loads another 1.67×
   (231 → 385 GB/s).
4. **Layer split for multiple GPUs.** PCIe P100s have no NVLink, so a per-layer
   all-reduce would dominate. Each device owns a contiguous layer range and only
   the residual row (14 KiB) crosses the bus between ranges.

## Limitations

* **Prefill is 4× slower than llama.cpp** (183 vs 781 tok/s). The multi-column
  GEMV re-reads the weight stream once per 8 tokens and its activation staging
  is not amortised enough; a fused dequantise + tiled GEMM (weights dequantised
  into shared memory once, reused across a 32-token tile) is the fix and is the
  next piece of work. Decode does not depend on it.
* Only `qwen2` / `qwen3` and only `q4_k` / `q6_k` weights. Other architectures or
  quant types fail loudly at load time rather than producing wrong output.
* One request at a time. The column tile is a prefill tile, not a continuous
  batch across requests. `--parallel` does not exist yet.
* KV cache is fp16 only; there is no quantised KV.
* Context is limited by memory and by the kernel's `int` indexing (well above
  100k tokens for this model), not by the training context.

## Repository map

```
crates/gguf      GGUF v2/v3 reader, split shards, quant block layouts + CPU reference
crates/p100      CUDA driver bindings (dlopen), device/buffer/stream layer, kernel launchers
crates/p100/cu   kernels.cu — every kernel, sm_60 only
crates/engine    model loader/repacker, KV cache, forward pass, sampler, tokenizer, HTTP API
crates/cli       the trav-p100 binary
configs/         per-model tuning profiles
docs/DESIGN.md   measurements, dead ends, and the reasoning behind the kernels
```

## License

Apache-2.0. See [`LICENSE`](LICENSE).

//! `trav-p100` command line: inspect models, run prompts, chat, benchmark and
//! tune a per-model profile.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Result};
use clap::{Args, Parser, Subcommand};
use trav_engine::engine::{Engine, Seq, Tuning};
use trav_engine::sampler::SamplingParams;
use trav_engine::tokenizer::Role;

#[derive(Parser)]
#[command(
    name = "trav-p100",
    version,
    about = "LLM inference engine specialised for the Tesla P100 (sm_60)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// List the CUDA devices this build can use.
    Info,
    /// Dump GGUF metadata, tensor types and sizes.
    Inspect {
        model: String,
        /// Print every metadata key (used to check rope scaling / norms).
        #[arg(long)]
        meta: bool,
    },
    /// Run one prompt to completion.
    Run(RunArgs),
    /// Interactive chat (ChatML/Qwen template).
    Chat(ChatArgs),
    /// Measure prefill and decode throughput.
    Bench(BenchArgs),
    /// Sweep runtime knobs and write a tuning profile.
    Tune(TuneArgs),
    /// Serve an OpenAI-compatible HTTP API.
    Serve(ServeArgs),
    /// Cross-check the batched prefill path against token-at-a-time decoding.
    SelfTest(SelfTestArgs),
    /// Perplexity over a text file, for comparison with `llama-perplexity`.
    Perplexity(PerplexityArgs),
    /// Print token ids for a piece of text (compare with `llama-tokenize`).
    Tokenize { model: String, text: String },
}

#[derive(Args, Clone)]
struct ModelArgs {
    /// GGUF model path (split shards are found automatically).
    #[arg(short, long)]
    model: String,
    /// Comma separated CUDA device indices, e.g. `0` or `0,1`.
    #[arg(long, default_value = "0", value_delimiter = ',')]
    gpus: Vec<usize>,
    /// KV capacity per sequence (overrides the profile).
    #[arg(long)]
    ctx: Option<usize>,
    /// Tuning profile TOML.
    #[arg(long)]
    profile: Option<String>,
    /// Write the effective profile here.
    #[arg(long)]
    save_profile: Option<String>,
    /// Override the prefill column tile (1,2,4,8).
    #[arg(long)]
    prefill_cols: Option<usize>,
    /// Override the flash-decoding KV split count.
    #[arg(long)]
    attn_splits: Option<usize>,
}

impl ModelArgs {
    fn tuning(&self) -> Result<Tuning> {
        let mut t = match &self.profile {
            Some(p) => Tuning::load(Path::new(p))?,
            None => Tuning::default(),
        };
        if let Some(c) = self.ctx {
            t.ctx = c;
        }
        if let Some(c) = self.prefill_cols {
            t.prefill_cols = c;
        }
        if let Some(s) = self.attn_splits {
            t.attn_splits = s;
        }
        Ok(t)
    }

    fn engine(&self, tuning: Tuning) -> Result<Engine> {
        Engine::new(&self.model, &self.gpus, tuning)
    }
}

#[derive(Args, Clone)]
struct SampleArgs {
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,
    #[arg(long, default_value_t = 20)]
    top_k: usize,
    #[arg(long, default_value_t = 0.0)]
    min_p: f32,
    #[arg(long, default_value_t = 1.05)]
    repeat_penalty: f32,
    #[arg(long, default_value_t = 0x9e3779b97f4a7c15)]
    seed: u64,
    /// 0 = greedy
    #[arg(long, default_value_t = 0)]
    greedy: u8,
}

impl SampleArgs {
    fn params(&self) -> SamplingParams {
        SamplingParams {
            temperature: if self.greedy != 0 {
                0.0
            } else {
                self.temperature
            },
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            repeat_penalty: self.repeat_penalty,
            repeat_last_n: 256,
            seed: self.seed,
        }
    }
}

#[derive(Args)]
struct RunArgs {
    #[command(flatten)]
    m: ModelArgs,
    #[command(flatten)]
    s: SampleArgs,
    #[arg(long, default_value = "Hello, who are you?")]
    prompt: String,
    #[arg(long)]
    system: Option<String>,
    #[arg(long, default_value_t = 256)]
    max_new: usize,
    /// Print token ids instead of text.
    #[arg(long)]
    ids: bool,
    /// Throwaway hello-world: print the next 8 tokens of the raw prompt with no
    /// chat template (used by the smoke test).
    #[arg(long)]
    raw: bool,
}

#[derive(Args)]
struct ChatArgs {
    #[command(flatten)]
    m: ModelArgs,
    #[command(flatten)]
    s: SampleArgs,
    #[arg(long)]
    system: Option<String>,
    #[arg(long, default_value_t = 512)]
    max_new: usize,
}

#[derive(Args)]
struct BenchArgs {
    #[command(flatten)]
    m: ModelArgs,
    #[arg(long, default_value_t = 512)]
    prompt_tokens: usize,
    #[arg(long, default_value_t = 128)]
    n_predict: usize,
    #[arg(long, default_value_t = 2)]
    reps: usize,
    #[arg(long, default_value_t = 0)]
    greedy: u8,
}

#[derive(Args)]
struct SelfTestArgs {
    #[command(flatten)]
    m: ModelArgs,
    #[arg(
        long,
        default_value = "The quick brown fox jumps over the lazy dog, and then it"
    )]
    prompt: String,
}

#[derive(Args)]
struct PerplexityArgs {
    #[command(flatten)]
    m: ModelArgs,
    #[arg(long)]
    file: String,
    /// Chunks of `ctx` tokens to score.
    #[arg(long, default_value_t = 4)]
    chunks: usize,
    /// Leading tokens of each chunk used as context but not scored.
    /// Defaults to `ctx/2 + 1`, which is what llama-perplexity scores (the
    /// first half of the window is context only). Use 0 for all positions.
    #[arg(long)]
    ppl_skip: Option<usize>,
}

#[derive(Args)]
struct ServeArgs {
    #[command(flatten)]
    m: ModelArgs,
    /// Bind address.
    #[arg(long, default_value = "127.0.0.1:8080")]
    addr: String,
}

#[derive(Args)]
struct TuneArgs {
    #[command(flatten)]
    m: ModelArgs,
    #[arg(long, default_value_t = 512)]
    prompt_tokens: usize,
    #[arg(long, default_value_t = 128)]
    n_predict: usize,
    #[arg(long, default_value_t = 2)]
    reps: usize,
    /// Where to write the winning profile.
    #[arg(long)]
    out: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Info => info(),
        Cmd::Inspect { model, meta } => inspect(&model, meta),
        Cmd::Run(a) => run(a),
        Cmd::Chat(a) => chat(a),
        Cmd::Bench(a) => bench(a),
        Cmd::Tune(a) => tune(a),
        Cmd::Serve(a) => serve(a),
        Cmd::SelfTest(a) => selftest(a),
        Cmd::Perplexity(a) => perplexity(a),
        Cmd::Tokenize { model, text } => {
            let g = trav_gguf::Gguf::open(&model)?;
            let t = trav_engine::Tokenizer::from_gguf(&g)?;
            let ids = t.encode(&text);
            println!("{} tokens", ids.len());
            println!("{:?}", ids);
            println!("round-trip: {:?}", t.decode(&ids));
            Ok(())
        }
    }
}

fn info() -> Result<()> {
    let n = trav_p100::Device::count()?;
    println!(
        "{n} CUDA device(s), kernels built for {}",
        trav_p100::arch_string()
    );
    for i in 0..n {
        let d = trav_p100::Device::open(i)?;
        let (free, total) = d.mem_info()?;
        println!(
            "  [{i}] {}  cc {}.{}  {} SMs  {:.1} GiB free / {:.1} GiB  ~{:.0} GB/s  {} MHz",
            d.name,
            d.cc.0,
            d.cc.1,
            d.sm_count,
            free as f64 / 2f64.powi(30),
            total as f64 / 2f64.powi(30),
            d.theoretical_bw_gbs(),
            d.clock_khz / 1000
        );
        if i > 0 {
            let p = trav_p100::Device::open(0)?;
            println!(
                "      peer access from GPU 0: {}",
                if p.can_access_peer(&d) {
                    "yes"
                } else {
                    "no (host bridge)"
                }
            );
        }
    }
    Ok(())
}

fn inspect(path: &str, meta: bool) -> Result<()> {
    let g = trav_gguf::Gguf::open(path)?;
    if meta {
        for (k, v) in &g.meta {
            println!("{k} = {}", fmt_meta(v));
        }
        return Ok(());
    }
    println!("file      {}", g.path.display());
    println!("arch      {} ({})", g.architecture(), g.name());
    println!("alignment {}", g.alignment);
    let hp = trav_engine::model::HParams::from_gguf(&g)?;
    println!(
        "layers {}  heads {}  kv heads {}  embd {}  ffn {}  vocab {}  ctx {}",
        hp.n_layer, hp.n_head, hp.n_head_kv, hp.n_embd, hp.n_ff, hp.vocab, hp.n_ctx_train
    );
    let mut hist: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for t in &g.tensors {
        let e = hist.entry(t.ty.name().to_string()).or_default();
        e.0 += 1;
        e.1 += t.nbytes();
    }
    println!("\nquant   tensors      bytes");
    for (k, (c, b)) in hist {
        println!("{k:<8} {c:>6} {:>10}", fmt_bytes(b));
    }
    println!(
        "\ntotal   weights {}",
        fmt_bytes(trav_engine::model::HParams::weight_bytes(&g))
    );
    Ok(())
}

fn fmt_meta(v: &trav_gguf::Value) -> String {
    use trav_gguf::Value;
    match v {
        Value::String(s) if s.len() > 120 => format!("{s:?}..."),
        Value::String(s) => format!("{s:?}"),
        Value::Array(a) if a.len() > 8 => {
            format!("[{} items; first {:?}]", a.len(), &a[..4])
        }
        Value::Array(a) => format!("{a:?}"),
        other => format!("{other:?}"),
    }
}

fn fmt_bytes(b: usize) -> String {
    if b >= 1 << 30 {
        format!("{:.2} GiB", b as f64 / 2f64.powi(30))
    } else {
        format!("{:.1} MiB", b as f64 / 2f64.powi(20))
    }
}

/// Prefill `ids` in tiles, returning the last-token logits and the elapsed time.
fn prefill(eng: &mut Engine, seq: &mut Seq, ids: &[u32]) -> Result<(Vec<f32>, f64)> {
    let cols = eng.tuning.prefill_cols.clamp(1, 512);
    let t0 = Instant::now();
    let mut logits = Vec::new();
    let mut i = 0;
    while i < ids.len() {
        let n = trav_engine::engine::prefill_tile(cols, ids.len() - i);
        let last = i + n == ids.len();
        logits = eng.step(seq, &ids[i..i + n], last)?;
        i += n;
    }
    Ok((logits, t0.elapsed().as_secs_f64()))
}

fn run(a: RunArgs) -> Result<()> {
    let tuning = a.m.tuning()?;
    let mut eng = a.m.engine(tuning.clone())?;
    if let Some(p) = &a.m.save_profile {
        tuning.save(Path::new(p))?;
        eprintln!("wrote profile {p}");
    }
    let tok_ids = if a.raw {
        eng.model.tokenizer.encode(&a.prompt)
    } else {
        eng.model.tokenizer.render_chat(
            a.system.as_deref(),
            &[(Role::User, a.prompt.clone())],
            true,
        )
    };
    let mut seq = eng.new_seq(tuning.ctx, a.s.params())?;
    let (mut logits, secs) = prefill(&mut eng, &mut seq, &tok_ids)?;
    eprintln!(
        "prefill {} tokens in {:.0} ms = {:.0} tok/s",
        tok_ids.len(),
        secs * 1e3,
        tok_ids.len() as f64 / secs
    );

    let t0 = Instant::now();
    let mut n = 0usize;
    let mut dec = trav_engine::tokenizer::StreamDecoder::default();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    while n < a.max_new {
        let next = seq.sample(&logits);
        if seq.stop.contains(&next) {
            break;
        }
        if a.ids {
            print!("{next} ");
        } else {
            let s = dec.push(&eng.model.tokenizer, next);
            write!(out, "{s}")?;
            out.flush()?;
        }
        n += 1;
        if seq.n_past + 1 > seq.kv.cap {
            break;
        }
        logits = eng.step(&mut seq, &[next], true)?;
    }
    if !a.ids {
        let tail = dec.flush();
        if !tail.is_empty() {
            write!(out, "{tail}")?;
        }
        writeln!(out)?;
    }
    let el = t0.elapsed().as_secs_f64();
    eprintln!(
        "\ndecode {n} tokens in {:.0} ms = {:.2} tok/s",
        el * 1e3,
        if n > 1 { (n - 1) as f64 / el } else { 0.0 }
    );
    Ok(())
}

fn chat(a: ChatArgs) -> Result<()> {
    let tuning = a.m.tuning()?;
    let mut eng = a.m.engine(tuning.clone())?;
    let mut turns: Vec<(Role, String)> = Vec::new();
    println!("trav-p100 chat. Ctrl-D to exit.");
    loop {
        print!("\n> ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line)? == 0 {
            println!();
            return Ok(());
        }
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        turns.push((Role::User, line));
        let ids = eng
            .model
            .tokenizer
            .render_chat(a.system.as_deref(), &turns, true);
        let mut seq = eng.new_seq(tuning.ctx, a.s.params())?;
        let (mut logits, secs) = prefill(&mut eng, &mut seq, &ids)?;
        eprintln!(
            "[{} prompt tokens, {:.0} ms, {:.0} tok/s]",
            ids.len(),
            secs * 1e3,
            ids.len() as f64 / secs
        );
        let mut reply = String::new();
        let mut dec = trav_engine::tokenizer::StreamDecoder::default();
        let t0 = Instant::now();
        let mut n = 0usize;
        while n < a.max_new {
            let next = seq.sample(&logits);
            if seq.stop.contains(&next) {
                break;
            }
            let s = dec.push(&eng.model.tokenizer, next);
            print!("{s}");
            std::io::stdout().flush()?;
            reply.push_str(&s);
            n += 1;
            if seq.n_past + 1 > seq.kv.cap {
                break;
            }
            logits = eng.step(&mut seq, &[next], true)?;
        }
        let tail = dec.flush();
        print!("{tail}");
        reply.push_str(&tail);
        let el = t0.elapsed().as_secs_f64();
        eprintln!(
            "\n[{n} tokens, {:.2} tok/s]",
            (n.saturating_sub(1)) as f64 / el
        );
        turns.push((Role::Assistant, reply));
    }
}

fn perplexity(a: PerplexityArgs) -> Result<()> {
    let tuning = a.m.tuning()?;
    let mut eng = a.m.engine(tuning.clone())?;
    let text = std::fs::read_to_string(&a.file)?;
    let ids = eng.model.tokenizer.encode(&text);
    let ctx = tuning.ctx;
    if ids.len() < ctx + 1 {
        bail!("corpus has {} tokens, need at least {}", ids.len(), ctx + 1);
    }
    let chunks = a.chunks.min(ids.len() / ctx);
    let vocab = eng.model.hp.vocab;
    let cols = tuning.prefill_cols.clamp(1, 512);
    let skip = a.ppl_skip.unwrap_or(ctx / 2 + 1).min(ctx.saturating_sub(1));
    println!(
        "scoring tokens {skip}..{} of each {ctx}-token chunk{}",
        ctx - 1,
        if skip == 0 {
            ""
        } else {
            " (llama-perplexity convention)"
        }
    );
    let mut ppls = Vec::new();
    for c in 0..chunks {
        let chunk = &ids[c * ctx..(c + 1) * ctx];
        let mut seq = eng.new_seq(ctx + 8, SamplingParams::default())?;
        let mut nll = 0f64;
        let mut n = 0usize;
        let mut i = 0;
        while i < chunk.len() {
            let tc = trav_engine::engine::prefill_tile(cols, chunk.len() - i);
            let logits = eng.step_all_logits(&mut seq, &chunk[i..i + tc])?;
            for j in 0..tc {
                let target = i + j + 1;
                if target >= chunk.len() || target < skip {
                    continue;
                }
                let row = &logits[j * vocab..(j + 1) * vocab];
                let max = row.iter().cloned().fold(f32::MIN, f32::max) as f64;
                let sum: f64 = row.iter().map(|v| ((*v as f64) - max).exp()).sum();
                let lse = max + sum.ln();
                let lp = lse - row[chunk[target] as usize] as f64;
                nll += lp;
                n += 1;
                if std::env::var_os("TRAV_PPL_DEBUG").is_some() && lp > 2.0 {
                    let mut best = 0usize;
                    for k in 1..vocab {
                        if row[k] > row[best] {
                            best = k;
                        }
                    }
                    eprintln!(
                        "pos {target} target {} lp {lp:.3} argmax {} ({:.1})",
                        chunk[target], best, row[best]
                    );
                }
            }
            i += tc;
        }
        let ppl = (nll / n as f64).exp();
        ppls.push(ppl);
        println!("[{c}] {ppl:.4}  ({n} predictions)");
    }
    let mean: f64 = ppls.iter().map(|p| p.ln()).sum::<f64>() / ppls.len() as f64;
    println!("perplexity: {:.4}", mean.exp());
    Ok(())
}

fn serve(a: ServeArgs) -> Result<()> {
    let tuning = a.m.tuning()?;
    let ctx = tuning.ctx;
    let eng = a.m.engine(tuning)?;
    let name = trav_gguf::Gguf::open(&a.m.model)
        .map(|g| g.name().to_string())
        .unwrap_or_else(|_| a.m.model.clone());
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(trav_engine::server::serve(
        eng,
        name,
        trav_engine::server::ServeConfig { addr: a.addr, ctx },
    ))
}

/// The batched (multi-column) prefill and the one-token decode path must agree.
/// They exercise different kernels (prefill attention with a causal mask over a
/// tile, vs split-KV decode attention), so matching logits is a strong check on
/// both the KV cache and the mask.
fn selftest(a: SelfTestArgs) -> Result<()> {
    let tuning = a.m.tuning()?;
    let mut eng = a.m.engine(tuning.clone())?;
    let ids = eng.model.tokenizer.encode(&a.prompt);
    if ids.len() < 4 {
        bail!("prompt too short");
    }
    eprintln!("prompt: {} tokens", ids.len());

    let mut seq = eng.new_seq(
        tuning.ctx,
        SamplingParams {
            temperature: 0.0,
            ..Default::default()
        },
    )?;
    let (batched, pp_s) = prefill(&mut eng, &mut seq, &ids)?;

    let mut seq2 = eng.new_seq(
        tuning.ctx,
        SamplingParams {
            temperature: 0.0,
            ..Default::default()
        },
    )?;
    let mut single = Vec::new();
    let t0 = Instant::now();
    for (i, &t) in ids.iter().enumerate() {
        single = eng.step(&mut seq2, &[t], i + 1 == ids.len())?;
    }
    let per_s = t0.elapsed().as_secs_f64();

    // The KV cache must match token for token; if it does, any logit gap is the
    // attention reduction order.
    let nkvd = eng.model.hp.n_head_kv * eng.model.hp.head_dim;
    let kv_bytes = ids.len() * nkvd * 2;
    let mut kv_worst = 0f32;
    let mut kv_diff = 0usize;
    for l in 0..eng.model.hp.n_layer {
        let mut a = vec![0u8; kv_bytes];
        let mut b = vec![0u8; kv_bytes];
        seq.kv.k[l].download_to(&mut a)?;
        seq2.kv.k[l].download_to(&mut b)?;
        let mut layer_diff = 0usize;
        for i in 0..kv_bytes / 2 {
            let x = half::f16::from_bits(u16::from_le_bytes([a[2 * i], a[2 * i + 1]])).to_f32();
            let y = half::f16::from_bits(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])).to_f32();
            if x != y {
                kv_diff += 1;
                layer_diff += 1;
                kv_worst = kv_worst.max((x - y).abs());
                if layer_diff <= 3 {
                    let pos = i / nkvd;
                    let rem = i - pos * nkvd;
                    eprintln!(
                        "    l{l} pos{pos} head{} d{}: {} vs {}",
                        rem / eng.model.hp.head_dim,
                        rem % eng.model.hp.head_dim,
                        x,
                        y
                    );
                }
            }
        }
        if l < 4 || layer_diff > 0 {
            eprintln!("  layer {l}: {layer_diff} differing halves");
        }
    }
    println!("KV cache: {kv_diff} differing halves, max |diff| {kv_worst:.4e}");

    let n = batched.len().min(single.len());
    let mut max_abs = 0f32;
    let mut sum_sq = 0f64;
    let mut max_i = 0usize;
    for i in 0..n {
        let d = (batched[i] - single[i]).abs();
        if d > max_abs {
            max_abs = d;
            max_i = i;
        }
        sum_sq += (d as f64) * (d as f64);
    }
    let rms = (sum_sq / n as f64).sqrt();
    let scale = batched
        .iter()
        .cloned()
        .fold(f32::MIN, f32::max)
        .abs()
        .max(1.0);

    let top = |v: &[f32]| -> Vec<(u32, f32)> {
        let mut idx: Vec<(f32, u32)> = v
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, x)| (x, i as u32))
            .collect();
        idx.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        idx.truncate(5);
        idx.into_iter().map(|(x, i)| (i, x)).collect()
    };
    println!(
        "prefill {:.1} ms ({:.0} tok/s), one-at-a-time {:.1} ms",
        pp_s * 1e3,
        ids.len() as f64 / pp_s,
        per_s * 1e3
    );
    println!("logit max |diff| {max_abs:.5e} at {max_i}, rms {rms:.3e}, |max logit| {scale:.1}");
    println!("batched top5: {:?}", top(&batched));
    println!("single  top5: {:?}", top(&single));
    println!("text: {}", eng.model.tokenizer.decode(&ids));
    // The two paths reduce in different orders (tile attention vs split-KV plus
    // fp16 KV rounding), so exact equality is not expected; the argmax must
    // still agree, and the spread must stay small relative to the logit scale.
    let top1 = |v: &[f32]| -> u32 {
        let mut best = 0usize;
        for i in 1..v.len() {
            if v[i] > v[best] {
                best = i;
            }
        }
        best as u32
    };
    let (t1, t2) = (top1(&batched), top1(&single));
    if t1 != t2 {
        bail!("argmax differs between batched ({t1}) and streaming ({t2}) paths");
    }
    if rms / scale as f64 > 0.25 {
        bail!("logit spread {rms:.3} is large relative to the scale {scale:.1}");
    }
    println!(
        "OK: same argmax (token {t1}), spread {:.1}% of logit scale",
        100.0 * rms / scale as f64
    );
    Ok(())
}

/// A deterministic prompt of roughly `n` tokens.
fn bench_prompt(eng: &Engine, n: usize) -> Vec<u32> {
    let base = "The quick brown fox jumps over the lazy dog while the engine \
                streams quantised weights through the Tesla P100 memory bus. ";
    let mut text = String::with_capacity(n * 6);
    while eng.model.tokenizer.encode(&text).len() < n {
        text.push_str(base);
    }
    let mut ids = eng.model.tokenizer.encode(&text);
    ids.truncate(n);
    ids
}

fn measure(
    eng: &mut Engine,
    ids: &[u32],
    n_predict: usize,
    greedy: bool,
) -> Result<(f64, f64, usize)> {
    let mut params = SamplingParams::default();
    if greedy {
        params.temperature = 0.0;
    }
    let mut seq = eng.new_seq(eng.tuning.ctx.max(ids.len() + n_predict + 8), params)?;
    let (mut logits, pre_s) = prefill(eng, &mut seq, ids)?;
    let t0 = Instant::now();
    let mut n = 0usize;
    while n < n_predict {
        let next = seq.sample(&logits);
        n += 1;
        if seq.n_past + 1 > seq.kv.cap {
            break;
        }
        logits = eng.step(&mut seq, &[next], true)?;
    }
    let dec_s = t0.elapsed().as_secs_f64();
    Ok((ids.len() as f64 / pre_s, (n - 1) as f64 / dec_s, n))
}

fn bench(a: BenchArgs) -> Result<()> {
    let tuning = a.m.tuning()?;
    let mut eng = a.m.engine(tuning.clone())?;
    let ids = bench_prompt(&eng, a.prompt_tokens);
    // warm up: first launch pays module load and page faults
    let _ = measure(&mut eng, &ids, 8, true)?;
    let mut best_pp = 0f64;
    let mut best_tg = 0f64;
    for _ in 0..a.reps.max(1) {
        let (pp, tg, n) = measure(&mut eng, &ids, a.n_predict, a.greedy != 0)?;
        best_pp = best_pp.max(pp);
        best_tg = best_tg.max(tg);
        println!("pp{} {:.0} tok/s   tg{} {:.1} tok/s", ids.len(), pp, n, tg);
    }
    println!(
        "\nbest: prefill {:.0} tok/s ({} tokens), decode {:.2} tok/s",
        best_pp,
        ids.len(),
        best_tg
    );
    Ok(())
}

fn tune(a: TuneArgs) -> Result<()> {
    let mut tuning = a.m.tuning()?;
    let eng0 = a.m.engine(tuning.clone())?;
    let ids = bench_prompt(&eng0, a.prompt_tokens);
    drop(eng0);
    let mut eng = a.m.engine(tuning.clone())?;
    let _ = measure(&mut eng, &ids, 8, true)?;

    let mut best = (tuning.attn_splits, tuning.prefill_cols, 0f64, 0f64);
    println!("splits  prefill_cols      pp tok/s    tg tok/s");
    for &splits in &[4usize, 8, 12, 16, 24, 32] {
        for &cols in &[8usize, 64, 128, 256, 512] {
            eng.tuning.attn_splits = splits;
            eng.tuning.prefill_cols = cols;
            let mut pp_best = 0f64;
            let mut tg_best = 0f64;
            for _ in 0..a.reps.max(1) {
                let (pp, tg, _) = measure(&mut eng, &ids, a.n_predict, true)?;
                pp_best = pp_best.max(pp);
                tg_best = tg_best.max(tg);
            }
            println!("{splits:>6}  {cols:>11}  {pp_best:>11.0}  {tg_best:>11.2}");
            // Decode dominates, but a sub-1% difference is noise: prefer the
            // configuration with the better prefill in that case.
            let better = best.3 == 0.0
                || tg_best > best.3 * 1.02
                || (tg_best >= best.3 * 0.98 && pp_best > best.2);
            if better {
                best = (splits, cols, pp_best, tg_best);
            }
        }
    }
    tuning.attn_splits = best.0;
    tuning.prefill_cols = best.1;
    let out = a.out.unwrap_or_else(|| {
        let stem = Path::new(&a.m.model)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "model".into());
        format!("configs/{stem}.toml")
    });
    tuning.save(Path::new(&out))?;
    println!(
        "\nbest: attn_splits={} prefill_cols={} (pp {:.0}, tg {:.2}); wrote {out}",
        best.0, best.1, best.2, best.3
    );
    Ok(())
}

//! Per-model tuning profile, KV cache, scratch buffers and the forward pass.
//!
//! One code path serves prefill and decode: a tile of `tc` tokens is pushed
//! through the same layer loop, with `tc == 1` selecting the split-KV decode
//! attention and `tc > 1` the causal tiled prefill attention. The GEMV kernel is
//! templated on the number of columns, so a prefill tile of 8 reuses every
//! dequantised weight word 8 times.

use std::path::Path;

use anyhow::{bail, Context, Result};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use trav_p100::{Buf, PinnedBuf};

use crate::model::{Model, QTensor};
use crate::sampler::{sample, SamplingParams};

/// Widest prefill tile. A tile is pushed through every layer before the next
/// one starts, so a whole-prompt tile is the layer-major batch: each weight
/// matrix is read and dequantised once per prompt instead of once per tile.
pub const MAX_PREFILL_COLS: usize = 512;

/// Largest flash-decoding split count the launcher accepts. The split buffers
/// are sized for this rather than for the profile's value, because the tuner
/// rewrites `attn_splits` on a live engine (`tune` sweeps it before writing the
/// profile).
pub const MAX_ATTN_SPLITS: usize = 64;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Tuning {
    /// KV capacity per sequence.
    pub ctx: usize,
    /// Tokens per prefill tile (columns of the fused GEMV).
    pub prefill_cols: usize,
    /// Flash-decoding KV splits; more blocks means better SM coverage.
    pub attn_splits: usize,
}

impl Default for Tuning {
    fn default() -> Self {
        Tuning {
            ctx: 8192,
            prefill_cols: MAX_PREFILL_COLS,
            attn_splits: 12,
        }
    }
}

impl Tuning {
    pub fn load(path: &Path) -> Result<Tuning> {
        let s = std::fs::read_to_string(path)
            .with_context(|| format!("reading tuning profile {}", path.display()))?;
        let t: Tuning = toml::from_str(&s)?;
        if t.prefill_cols > MAX_PREFILL_COLS {
            bail!(
                "prefill tiles above {MAX_PREFILL_COLS} tokens are not instantiated"
            );
        }
        if t.ctx == 0 {
            bail!("ctx must be > 0");
        }
        Ok(t)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p).ok();
        }
        std::fs::write(path, toml::to_string_pretty(self)?)?;
        Ok(())
    }
}

/// Tokens for the next prefill tile. Full tiles are `prefill_cols` (128 by
/// default), which amortises the weight stream over the most columns; the GEMM
/// computes a 64-column block, so a tail shorter than 24 tokens is cheaper as an
/// 8-column GEMV tile (measured 171 vs 179 tok/s at 16).
pub fn prefill_tile(prefill_cols: usize, remaining: usize) -> usize {
    let n = remaining.min(prefill_cols.clamp(1, MAX_PREFILL_COLS));
    if n < 24 {
        n.min(8)
    } else {
        n
    }
}

pub struct Kv {
    pub k: Vec<Buf>,
    pub v: Vec<Buf>,
    pub cap: usize,
}

pub struct Seq {
    pub tokens: Vec<u32>,
    pub n_past: usize,
    pub kv: Kv,
    pub params: SamplingParams,
    pub rng: ChaCha8Rng,
    pub stop: Vec<u32>,
    sample_scratch: Vec<(f32, u32)>,
    /// Captured decode step for this sequence's KV buffers; built on the first
    /// single-token step and replayed for the rest.
    graph: Option<trav_p100::GraphExec>,
}

impl Engine {
    fn tokenizer(&self) -> &crate::tokenizer::Tokenizer {
        &self.model.tokenizer
    }
}

impl Seq {
    /// Sample the next token from `logits` (updates the per-sequence RNG).
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        sample(
            logits,
            &self.params,
            &self.tokens,
            &mut self.rng,
            &mut self.sample_scratch,
        )
    }
}

struct Scratch {
    x: Buf,
    xn: Buf,
    yh: Buf,
    ys: Buf,
    yh1: Buf,
    ys1: Buf,
    q: Buf,
    kk: Buf,
    vv: Buf,
    attn: Buf,
    o: Buf,
    gate: Buf,
    up: Buf,
    act: Buf,
    ayh: Buf,
    ays: Buf,
    logits: Buf,
    argmax: Buf,
    ids: Buf,
    pos: Buf,
    inv_freq: Buf,
    pm: Buf,
    pl: Buf,
    pacc: Buf,
    /// fp16 mirror of one weight matrix, for the prefill GEMM path.
    w16: Buf,
    /// Per-token scalars read by the kernels: `[kv write position, attn length]`.
    ctl: Buf,
}

impl Scratch {
    fn alloc(m: &Model, d: usize, tc: usize) -> Result<Scratch> {
        let hp = &m.hp;
        let dev = &m.devices[d];
        let k = hp.n_embd;
        let ff = hp.n_ff;
        let hd = hp.head_dim;
        let nkvd = hp.n_head_kv * hd;
        // Sized for the launcher's ceiling, not the current profile: `tune`
        // mutates `attn_splits` on a constructed engine.
        let ns = MAX_ATTN_SPLITS;
        let inv: Vec<f32> = (0..hd / 2)
            .map(|i| 1.0 / hp.rope_theta.powf(2.0 * i as f32 / hd as f32))
            .collect();
        let inv_buf = dev.upload(as_bytes(&inv))?;
        Ok(Scratch {
            x: dev.alloc_zeroed(tc * k * 4)?,
            xn: dev.alloc(tc * k * 4)?,
            yh: dev.alloc_zeroed(tc * k * 2)?,
            ys: dev.alloc_zeroed(tc * (k / 16) * 8)?,
            yh1: dev.alloc_zeroed(k * 2)?,
            ys1: dev.alloc_zeroed((k / 16) * 8)?,
            q: dev.alloc(tc * k * 4)?,
            kk: dev.alloc(tc * nkvd * 4)?,
            vv: dev.alloc(tc * nkvd * 4)?,
            attn: dev.alloc(tc * k * 4)?,
            o: dev.alloc(tc * k * 4)?,
            gate: dev.alloc(tc * ff * 4)?,
            up: dev.alloc(tc * ff * 4)?,
            act: dev.alloc(tc * ff * 4)?,
            ayh: dev.alloc_zeroed(tc * ff * 2)?,
            ays: dev.alloc_zeroed(tc * (ff / 16) * 8)?,
            logits: dev.alloc(tc * hp.vocab * 4)?,
            argmax: dev.alloc(trav_p100::kernels::GREEDY_SCRATCH_BYTES)?,
            ids: dev.alloc(tc * 4)?,
            pos: dev.alloc(tc * 4)?,
            inv_freq: inv_buf,
            pm: dev.alloc(hp.n_head * ns * 4)?,
            pl: dev.alloc(hp.n_head * ns * 4)?,
            pacc: dev.alloc(hp.n_head * ns * hd * 4)?,
            w16: dev.alloc_zeroed(f16_ws_bytes(hp))?,
            ctl: dev.alloc_zeroed(8)?,
        })
    }
}

pub struct Engine {
    pub model: Model,
    pub tuning: Tuning,
    scratch: Vec<Scratch>,
    stage: PinnedBuf,
    logits_host: PinnedBuf,
    /// Stream the decode step is captured on and replayed through.
    stream: Option<trav_p100::Stream>,
}

fn as_bytes<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn as_bytes_mut<T>(v: &mut [T]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}

/// Bytes needed to hold the largest weight matrix as fp16 (the prefill path
/// dequantises one matrix at a time into the scratch).
fn f16_ws_bytes(hp: &crate::model::HParams) -> usize {
    let big = (hp.n_embd * hp.n_ff).max(hp.n_ff * hp.n_embd);
    big.max(hp.vocab * hp.n_embd) * 2
}

impl Engine {
    pub fn new(model_path: &str, gpus: &[usize], tuning: Tuning) -> Result<Engine> {
        let model = crate::model::load(model_path, gpus)?;
        // Buffers are sized for the widest prefill tile (`MAX_PREFILL_COLS`) and
        // the largest split count (`MAX_ATTN_SPLITS`) so the tuner can widen
        // either knob after construction; a short tail tile runs the smaller
        // GEMV instantiations against the same buffers.
        let tc = MAX_PREFILL_COLS;
        let hp = &model.hp;
        let nkvd = hp.n_head_kv * hp.head_dim;

        // KV cache is the dominant allocation after the weights; check up front.
        let kv_per_seq: usize = tuning.ctx * nkvd * 2 * 2 * hp.n_layer;
        for d in &model.devices {
            let (free, _) = d.mem_info()?;
            if kv_per_seq + (64 << 20) > free {
                bail!(
                    "GPU {}: {} MiB free, KV cache for {} tokens needs {} MiB",
                    d.id,
                    free >> 20,
                    tuning.ctx,
                    kv_per_seq >> 20
                );
            }
        }

        let mut scratch = Vec::new();
        for d in 0..model.devices.len() {
            scratch.push(Scratch::alloc(&model, d, tc)?);
        }
        let stage = model.devices[0].pinned(tc * hp.n_embd * 4)?;
        let logits_host = model.devices[model.out_dev].pinned(hp.vocab * 4)?;

        // A CUDA graph needs a non-default stream to be captured on.
        let stream = model
            .devices
            .first()
            .and_then(|d| d.stream().ok());
        Ok(Engine {
            model,
            tuning,
            scratch,
            stage,
            logits_host,
            stream,
        })
    }

    pub fn new_seq(&self, cap: usize, params: SamplingParams) -> Result<Seq> {
        let m = &self.model;
        let nkvd = m.hp.n_head_kv * m.hp.head_dim;
        let bytes = cap * nkvd * 2;
        let mut k = Vec::with_capacity(m.hp.n_layer);
        let mut v = Vec::with_capacity(m.hp.n_layer);
        for l in 0..m.hp.n_layer {
            let d = &m.devices[m.layer_dev[l]];
            k.push(d.alloc_zeroed(bytes)?);
            v.push(d.alloc_zeroed(bytes)?);
        }
        let mut stop = vec![m.tokenizer.eos];
        if m.tokenizer.im_end != m.tokenizer.eos {
            stop.push(m.tokenizer.im_end);
        }
        Ok(Seq {
            tokens: Vec::new(),
            n_past: 0,
            kv: Kv { k, v, cap },
            rng: {
                use rand::SeedableRng;
                ChaCha8Rng::seed_from_u64(params.seed)
            },
            params,
            stop,
            sample_scratch: Vec::with_capacity(m.hp.vocab),
            graph: None,
        })
    }

    /// Run `toks` and return the logits of *every* token (for perplexity).
    pub fn step_all_logits(&mut self, s: &mut Seq, toks: &[u32]) -> Result<Vec<f32>> {
        self.step_mode(s, toks, 2)
    }

    /// Run `toks` (1..=32) through the model at `pos0 = seq.n_past`, appending to
    /// the KV cache, and return the logits of the last token.
    pub fn step(&mut self, s: &mut Seq, toks: &[u32], want_logits: bool) -> Result<Vec<f32>> {
        self.step_mode(s, toks, if want_logits { 1 } else { 0 })
    }

    /// Advance the sequence and sample from the last token's logits.
    pub fn step_sample(&mut self, s: &mut Seq, toks: &[u32]) -> Result<u32> {
        let mut recent = 0;
        if s.params.repeat_penalty != 1.0 {
            recent = s.params.repeat_last_n.min(s.tokens.len() + toks.len());
        }
        if s.params.temperature <= 0.0
            && recent <= trav_p100::kernels::GREEDY_MAX_RECENT
            && std::env::var_os("TRAV_CPU_GREEDY").is_none()
        {
            let off = self.forward(s, toks, 1)?;
            let d = self.model.out_dev;
            let sc = &self.scratch[d];
            let kf = &self.model.kernels[d];
            let vocab = self.model.hp.vocab;
            if recent != 0 {
                kf.penalize_recent(&sc.logits, off, vocab, &s.tokens[s.tokens.len() - recent..],
                                   s.params.repeat_penalty, &sc.argmax)?;
            }
            kf.argmax(&sc.logits, off, vocab, &sc.argmax)?;
            let mut token = [0u32];
            sc.argmax.download_at(trav_p100::kernels::ARGMAX_TOKEN_OFFSET,
                                  as_bytes_mut(&mut token))?;
            return Ok(token[0]);
        }
        let logits = self.step(s, toks, true)?;
        Ok(s.sample(&logits))
    }

    fn step_mode(&mut self, s: &mut Seq, toks: &[u32], mode: u8) -> Result<Vec<f32>> {
        let off = self.forward(s, toks, mode)?;
        let tc = toks.len();
        let vocab = self.model.hp.vocab;
        match mode {
            1 => {
                let d = self.model.out_dev;
                self.scratch[d].logits.download_at(
                    off,
                    &mut self.logits_host.as_mut_slice()[..vocab * 4],
                )?;
                let mut out = vec![0f32; vocab];
                out.copy_from_slice(&self.logits_host.as_f32(vocab)[..vocab]);
                Ok(out)
            }
            2 => {
                let mut out = vec![0f32; tc * vocab];
                let d = self.model.out_dev;
                self.scratch[d].logits.download_at(0, as_bytes_mut(&mut out))?;
                Ok(out)
            }
            _ => Ok(Vec::new()),
        }
    }

    /// Run the tile and return the byte offset of the requested logits.
    /// `mode`: 0 = no logits, 1 = last column, 2 = all columns.
    fn forward(&mut self, s: &mut Seq, toks: &[u32], mode: u8) -> Result<usize> {
        if toks.is_empty() {
            bail!("empty tile");
        }
        let tc = toks.len();
        if tc > self.tuning.prefill_cols.clamp(1, MAX_PREFILL_COLS) {
            bail!("tile of {tc} tokens exceeds the prefill tile");
        }
        if s.n_past + tc > s.kv.cap {
            bail!("context full: {} + {} > {}", s.n_past, tc, s.kv.cap);
        }
        let pos0 = s.n_past;

        // Per-step inputs: token ids, rope positions, and the two scalars the
        // kernels read from device memory (KV write position, attention
        // length). Uploading them before the step is what lets a captured
        // decode graph be replayed with new values.
        let ids: Vec<i32> = toks.iter().map(|&t| t as i32).collect();
        let pos: Vec<i32> = (0..tc).map(|i| (pos0 + i) as i32).collect();
        let ctl: [i32; 2] = [pos0 as i32, (pos0 + tc) as i32];
        for d in 0..self.model.devices.len() {
            let sc = &mut self.scratch[d];
            sc.ids.upload_from(as_bytes(&ids))?;
            sc.pos.upload_from(as_bytes(&pos))?;
            sc.ctl.upload_from(as_bytes(&ctl))?;
        }

        // The single-token step is the one shape worth capturing: with the
        // scalars on the device every launch in it is position independent, so
        // one capture per sequence covers the whole generation. Prefill tiles,
        // multi-device layer splits (the residual hop is a blocking copy) and
        // TRAV_DUMP keep the plain path.
        let graphable = tc == 1
            && mode == 1
            && self.model.devices.len() == 1
            && !dump_on()
            && std::env::var_os("TRAV_NO_GRAPH").is_none();
        let off = if graphable {
            let stream = self.stream.as_ref().expect("decode stream");
            if s.graph.is_some() {
                let g = s.graph.as_ref().expect("decode graph");
                g.launch(stream)?;
                stream.sync()?;
                0
            } else {
                stream.begin_capture()?;
                let res = run_tile(
                    &self.model,
                    &self.tuning,
                    &mut self.scratch,
                    &mut self.stage,
                    s,
                    toks,
                    pos0,
                    mode,
                    Some(stream),
                );
                // Always close the capture, or the stream stays unusable.
                let cap = stream.end_capture();
                let off = res?;
                let exec = cap?.instantiate()?;
                exec.launch(stream)?;
                stream.sync()?;
                s.graph = Some(exec);
                off
            }
        } else {
            run_tile(
                &self.model,
                &self.tuning,
                &mut self.scratch,
                &mut self.stage,
                s,
                toks,
                pos0,
                mode,
                None,
            )?
        };
        s.n_past += tc;
        s.tokens.extend_from_slice(toks);
        Ok(off)
    }

    /// Prefill a prompt in tiles, then decode up to `max_new` tokens. Returns the
    /// number of tokens actually generated.
    /// `on_text` receives decoded text as it becomes valid UTF-8; returning
    /// `false` stops generation.
    pub fn generate(
        &mut self,
        s: &mut Seq,
        prompt: &[u32],
        max_new: usize,
        mut on_text: impl FnMut(&str) -> bool,
    ) -> Result<usize> {
        if prompt.is_empty() {
            bail!("empty prompt");
        }
        if prompt.len() + max_new > s.kv.cap {
            bail!(
                "prompt {} + {} new tokens exceeds the {} token KV cache",
                prompt.len(),
                max_new,
                s.kv.cap
            );
        }
        s.tokens.clear();
        s.n_past = 0;
        let cols = self.tuning.prefill_cols.clamp(1, MAX_PREFILL_COLS);
        let mut i = 0;
        let mut next = loop {
            let n = prefill_tile(cols, prompt.len() - i);
            if i + n == prompt.len() {
                if max_new == 0 {
                    self.step(s, &prompt[i..], true)?;
                    return Ok(0);
                }
                break self.step_sample(s, &prompt[i..])?;
            }
            self.step(s, &prompt[i..i + n], false)?;
            i += n;
        };

        let mut generated = 0usize;
        let mut dec = crate::tokenizer::StreamDecoder::default();
        while generated < max_new {
            if s.stop.contains(&next) {
                break;
            }
            generated += 1;
            let text = dec.push(self.tokenizer(), next);
            if !text.is_empty() && !on_text(&text) {
                break;
            }
            if generated >= max_new || s.n_past + 1 > s.kv.cap {
                break;
            }
            next = self.step_sample(s, &[next])?;
        }
        let tail = dec.flush();
        if !tail.is_empty() {
            on_text(&tail);
        }
        Ok(generated)
    }
}

/// Activation preparation: the GEMM path needs one power-of-two scale per
/// column; the GEMV path keeps its per-32-block scale.
fn prep_input(
    kf: &trav_p100::kernels::Kernels,
    dev: &trav_p100::Device,
    x: &Buf,
    y: &Buf,
    ys: &Buf,
    cols: usize,
    k: usize,
    stream: Option<&trav_p100::Stream>,
) -> Result<()> {
    if cols > 8 {
        kf.prep_col(dev, x, 0, y, ys, cols, k, stream)?;
    } else {
        kf.prep(dev, x, 0, y, ys, cols, k, stream)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn run_tile(
    m: &Model,
    tuning: &Tuning,
    scratch: &mut [Scratch],
    stage: &mut PinnedBuf,
    s: &mut Seq,
    toks: &[u32],
    pos0: usize,
    mode: u8,
    stream: Option<&trav_p100::Stream>,
) -> Result<usize> {
    let hp = &m.hp;
    let tc = toks.len();
    let k = hp.n_embd;
    let ff = hp.n_ff;
    let hd = hp.head_dim;
    let nkv = hp.n_head_kv;
    let nkvd = nkv * hd;
    let qpk = hp.q_per_kv();
    let eps = hp.eps;

    let do_dump = dump_on();
    let mut dbg: Vec<f32> = Vec::new();

    let mut cur = m.embd_dev;
    {
        let sc = &mut scratch[m.embd_dev];
        let dev = &m.devices[m.embd_dev];
        m.kernels[m.embd_dev].embed(
            dev,
            &m.embd.buf,
            m.embd.row_stride,
            &sc.ids,
            &sc.x,
            tc,
            k,
            stream,
        )?;
    }

    for l in 0..hp.n_layer {
        let d = m.layer_dev[l];
        if d != cur {
            let bytes = tc * k * 4;
            scratch[cur]
                .x
                .download_to(&mut stage.as_mut_slice()[..bytes])?;
            scratch[d].x.upload_from(&stage.as_slice()[..bytes])?;
            cur = d;
        }
        let dev = &m.devices[d];
        let kf = &m.kernels[d];
        let ly = &m.layers[l];
        let sc = &mut scratch[d];
        let kv = &s.kv;

        kf.rms_norm(dev, &sc.x, &ly.attn_norm, &sc.xn, tc, k, eps, stream)?;
        prep_input(kf, dev, &sc.xn, &sc.yh, &sc.ys, tc, k, stream)?;
        gemv(
            kf,
            dev,
            &ly.wq,
            tc,
            &sc.yh,
            k,
            &sc.ys,
            k / 16,
            Some(&ly.bq),
            &sc.q,
            k,
            Some(&sc.w16),
            stream,
        )?;
        gemv(
            kf,
            dev,
            &ly.wk,
            tc,
            &sc.yh,
            k,
            &sc.ys,
            k / 16,
            Some(&ly.bk),
            &sc.kk,
            nkvd,
            Some(&sc.w16),
            stream,
        )?;
        gemv(
            kf,
            dev,
            &ly.wv,
            tc,
            &sc.yh,
            k,
            &sc.ys,
            k / 16,
            Some(&ly.bv),
            &sc.vv,
            nkvd,
            Some(&sc.w16),
            stream,
        )?;

        kf.rope(
            dev,
            &sc.q,
            &sc.kk,
            &sc.pos,
            &sc.inv_freq,
            hp.n_head,
            nkv,
            hd,
            tc,
            0,
            stream,
        )?;
        kf.write_kv(
            dev, &kv.k[l], &kv.v[l], &sc.kk, &sc.vv, nkv, hd, &sc.ctl, tc,
            stream,
        )?;

        if tc == 1 {
            // The split count is fixed so the decode step can be captured; a
            // split whose range is empty writes l = 0 and drops out in the
            // combine. The cache length itself is read from `ctl`.
            let ns = tuning.attn_splits.clamp(1, MAX_ATTN_SPLITS);
            kf.attn_decode(
                dev, &kv.k[l], &kv.v[l], &sc.q, &sc.attn, &sc.pm, &sc.pl, &sc.pacc, hp.n_head, nkv,
                hd, &sc.ctl, ns, qpk, stream,
            )?;
            kf.attn_combine(
                dev, &sc.pm, &sc.pl, &sc.pacc, &sc.attn, hp.n_head, hd, ns, stream,
            )?;
        } else {
            kf.attn_prefill(
                dev,
                &kv.k[l],
                &kv.v[l],
                &sc.q,
                &sc.attn,
                hp.n_head,
                nkv,
                hd,
                pos0 + tc,
                qpk,
                pos0,
                tc,
                stream,
            )?;
        }

        if do_dump && l == 0 {
            eprintln!("  layer 0 tc={tc}:");
            dump("attn[0]", &sc.attn, tc, k, &mut dbg);
        }
        prep_input(kf, dev, &sc.attn, &sc.yh, &sc.ys, tc, k, stream)?;
        gemv(
            kf,
            dev,
            &ly.wo,
            tc,
            &sc.yh,
            k,
            &sc.ys,
            k / 16,
            None,
            &sc.o,
            k,
            Some(&sc.w16),
            stream,
        )?;
        if do_dump && l == 0 {
            dump("o[0]", &sc.o, tc, k, &mut dbg);
        }
        kf.add_rms_norm(dev, &sc.x, &sc.o, &ly.ffn_norm, &sc.xn, tc, k, eps, stream)?;
        prep_input(kf, dev, &sc.xn, &sc.yh, &sc.ys, tc, k, stream)?;
        gemv(
            kf,
            dev,
            &ly.w_gate,
            tc,
            &sc.yh,
            k,
            &sc.ys,
            k / 16,
            None,
            &sc.gate,
            ff,
            Some(&sc.w16),
            stream,
        )?;
        gemv(
            kf,
            dev,
            &ly.w_up,
            tc,
            &sc.yh,
            k,
            &sc.ys,
            k / 16,
            None,
            &sc.up,
            ff,
            Some(&sc.w16),
            stream,
        )?;
        kf.silu_mul(dev, &sc.act, &sc.gate, &sc.up, tc * ff, stream)?;
        if do_dump && l == 0 {
            dump("act[0]", &sc.act, tc, ff, &mut dbg);
        }
        prep_input(kf, dev, &sc.act, &sc.ayh, &sc.ays, tc, ff, stream)?;
        gemv(
            kf,
            dev,
            &ly.w_down,
            tc,
            &sc.ayh,
            ff,
            &sc.ays,
            ff / 16,
            None,
            &sc.o,
            k,
            Some(&sc.w16),
            stream,
        )?;
        if do_dump && l == 0 {
            dump("down[0]", &sc.o, tc, k, &mut dbg);
        }
        kf.add_inplace(dev, &sc.x, &sc.o, tc * k, stream)?;
        if do_dump && l == 0 {
            dump("x[0]", &sc.x, tc, k, &mut dbg);
        }
    }

    if mode == 0 {
        return Ok(0);
    }
    let d = m.out_dev;
    debug_assert_eq!(m.layer_dev[hp.n_layer - 1], d);
    let dev = &m.devices[d];
    let kf = &m.kernels[d];
    let sc = &mut scratch[d];
    kf.rms_norm(dev, &sc.x, &m.out_norm, &sc.xn, tc, k, eps, stream)?;
    // Byte offset of the last column's logits in `sc.logits`; the copy back to
    // the host happens outside (it must not be part of a captured graph).
    let mut off = (tc - 1) * hp.vocab * 4;
    if mode == 1 && tc > 8 {
        // Generation only reads the last column's logits; running lm_head over
        // a single column instead of the whole tile is 1/32 of its traffic.
        kf.prep(dev, &sc.xn, (tc - 1) * k, &sc.yh1, &sc.ys1, 1, k, stream)?;
        gemv(
            kf,
            dev,
            &m.lm_head,
            1,
            &sc.yh1,
            k,
            &sc.ys1,
            k / 16,
            None,
            &sc.logits,
            hp.vocab,
            Some(&sc.w16),
            stream,
        )?;
        off = 0;
    } else {
        prep_input(kf, dev, &sc.xn, &sc.yh, &sc.ys, tc, k, stream)?;
        gemv(
            kf,
            dev,
            &m.lm_head,
            tc,
            &sc.yh,
            k,
            &sc.ys,
            k / 16,
            None,
            &sc.logits,
            hp.vocab,
            Some(&sc.w16),
            stream,
        )?;
    }
    Ok(off)
}

fn dump_on() -> bool {
    std::env::var_os("TRAV_DUMP").is_some()
}

fn dump(tag: &str, buf: &Buf, rows: usize, n: usize, host: &mut Vec<f32>) {
    host.resize(rows * n, 0.0);
    buf.download_to(as_bytes_mut(host)).expect("dump");
    for r in 0..rows {
        let row = &host[r * n..(r + 1) * n];
        let sum: f32 = row.iter().sum();
        let l2: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        eprintln!(
            "    {tag} row{r}: sum {sum:.6} l2 {l2:.6} first {:?}",
            &row[..3]
        );
    }
}

fn gemv(
    kf: &trav_p100::kernels::Kernels,
    dev: &trav_p100::Device,
    w: &QTensor,
    cols: usize,
    y: &Buf,
    y_stride: usize,
    ys: &Buf,
    ys_stride: usize,
    bias: Option<&Buf>,
    out: &Buf,
    out_stride: usize,
    w16: Option<&Buf>,
    stream: Option<&trav_p100::Stream>,
) -> Result<()> {
    if cols > 8 {
        // Large matrices go through the fp16 path: dequantise once into the
        // scratch (a whole-prompt tile touches each matrix once) and run the
        // 128x128 block GEMM, whose staging is a copy. Small matrices stay on
        // the fused kernel, where split-K fills the card.
        let need = w.nrows * w.k * 2;
        let blocks16 = w.nrows.div_ceil(128) * cols.div_ceil(128);
        if let Some(ws) = w16.filter(|b| need <= b.bytes) {
            if cols >= 64 && blocks16 >= 112 {
                kf.dequant_f16(w.q6, dev, &w.buf, w.row_stride, ws, w.nrows, w.k, stream)?;
                kf.gemm_f16(dev, ws, w.k, y, y_stride, ys, ys_stride, bias, out,
                    out_stride, w.nrows, w.k, cols, stream)?;
                return Ok(());
            }
        }
        // 64-column blocks, 3 blocks/SM, split-K when the row count cannot fill
        // the card (see `Kernels::gemm_auto`).
        kf.gemm_auto(
            w.q6,
            dev.sm_count as usize,
            dev,
            &w.buf,
            w.row_stride,
            y,
            y_stride,
            ys,
            ys_stride,
            bias,
            out,
            out_stride,
            w.nrows,
            w.k,
            cols,
            stream,
        )?;
        return Ok(());
    }
    let is_pow2 = matches!(cols, 1 | 2 | 4 | 8);
    if is_pow2 {
        kf.gemv(
            w.q6,
            cols,
            dev,
            &w.buf,
            w.row_stride,
            y,
            y_stride,
            ys,
            ys_stride,
            bias,
            out,
            out_stride,
            w.nrows,
            w.k,
            stream,
        )?;
    } else {
        // 3,5,6,7 columns: run the largest instantiation and mask the tail by
        // repeating the last column (only the first `cols` outputs are read).
        let c = if cols <= 2 {
            2
        } else if cols <= 4 {
            4
        } else {
            8
        };
        kf.gemv(
            w.q6,
            c,
            dev,
            &w.buf,
            w.row_stride,
            y,
            y_stride,
            ys,
            ys_stride,
            bias,
            out,
            out_stride,
            w.nrows,
            w.k,
            stream,
        )?;
    }
    Ok(())
}

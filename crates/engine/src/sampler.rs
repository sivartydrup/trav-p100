//! Sampling: temperature, top-k, top-p, min-p and repetition penalty.
//!
//! The vocabulary here is 152 064 entries, so the candidate list is reduced with
//! a linear-time selection instead of a full sort, reusing one buffer per
//! sequence.

use rand::Rng;
use rand_chacha::ChaCha8Rng;

#[derive(Clone, Debug)]
pub struct SamplingParams {
    /// `0` means greedy (argmax).
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            temperature: 0.7,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.0,
            repeat_penalty: 1.05,
            repeat_last_n: 256,
            seed: 0x9e37_79b9_7f4a_7c15,
        }
    }
}

/// `recent` is the full token history (prompt + generated).
pub fn sample(
    logits: &[f32],
    p: &SamplingParams,
    recent: &[u32],
    rng: &mut ChaCha8Rng,
    scratch: &mut Vec<(f32, u32)>,
) -> u32 {
    scratch.clear();
    scratch.extend(logits.iter().enumerate().map(|(i, &v)| (v, i as u32)));

    if p.repeat_penalty != 1.0 && p.repeat_last_n > 0 {
        let from = recent.len().saturating_sub(p.repeat_last_n);
        for &t in &recent[from..] {
            if let Some(e) = scratch.get_mut(t as usize) {
                e.0 = if e.0 > 0.0 {
                    e.0 / p.repeat_penalty
                } else {
                    e.0 * p.repeat_penalty
                };
            }
        }
    }

    if p.temperature <= 0.0 {
        let mut best = 0usize;
        for i in 1..scratch.len() {
            if scratch[i].0 > scratch[best].0 {
                best = i;
            }
        }
        return scratch[best].1;
    }

    let k = if p.top_k == 0 {
        scratch.len()
    } else {
        p.top_k.min(scratch.len())
    };
    if k < scratch.len() {
        scratch.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
        scratch.truncate(k);
    }
    scratch.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

    let max = scratch[0].0;
    let inv_t = 1.0 / p.temperature;
    let mut sum = 0f64;
    for e in scratch.iter_mut() {
        e.0 = ((e.0 - max) * inv_t).exp();
        sum += e.0 as f64;
    }
    let inv_sum = 1.0 / sum;

    // min_p is relative to the most likely token
    let min_p = if p.min_p > 0.0 { p.min_p } else { 0.0 };
    let mut cum = 0f64;
    let mut keep = 0usize;
    for (i, e) in scratch.iter().enumerate() {
        let pr = e.0 as f64 * inv_sum;
        cum += pr;
        keep = i + 1;
        if pr < min_p as f64 {
            keep = i; // this one and the rest are below min_p
            break;
        }
        if cum >= p.top_p as f64 {
            break;
        }
    }
    let keep = keep.max(1);
    let total: f64 = scratch[..keep].iter().map(|e| e.0 as f64).sum();

    let mut r = rng.gen::<f64>() * total;
    for e in &scratch[..keep] {
        r -= e.0 as f64;
        if r <= 0.0 {
            return e.1;
        }
    }
    scratch[keep - 1].1
}

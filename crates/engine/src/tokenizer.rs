//! Qwen2/Qwen3 byte-level BPE, read straight from the GGUF vocabulary.
//!
//! The pre-tokeniser is llama.cpp's `qwen2` split regex. It needs one
//! look-ahead (`\s+(?!\S)`), which the `regex` crate rejects, hence
//! `fancy-regex`. Pieces are byte-encoded with the GPT-2 table before merging,
//! exactly like the original tokenizer.

use std::collections::HashMap;

use anyhow::{anyhow, Context, Result};
use fancy_regex::Regex;
use trav_gguf::Gguf;

pub const DEFAULT_SYSTEM: &str =
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

/// llama.cpp `LLAMA_VOCAB_PRE_TYPE_QWEN2`, in alternative order.
const QWEN2_PATTERN: &str = concat!(
    r"\p{N}{1,3}",
    r"|[\u{4e00}-\u{9fa5}\u{3040}-\u{30ff}]+",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+",
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*",
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?",
    r"| ?[\p{P}\p{S}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

pub struct Tokenizer {
    tokens: Vec<String>,
    token_types: Vec<u32>,
    byte_id: [u32; 256],
    unicode_to_byte: HashMap<char, u8>,
    merges: HashMap<(u32, u32), (u32, u32)>,
    re: Regex,
    pub eos: u32,
    pub im_start: u32,
    pub im_end: u32,
    pub pad: u32,
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Tokenizer> {
        let toks = g
            .meta_array("tokenizer.ggml.tokens")
            .ok_or_else(|| anyhow!("GGUF has no tokenizer.ggml.tokens"))?;
        let mut tokens = Vec::with_capacity(toks.len());
        for v in toks {
            tokens.push(
                v.as_str()
                    .ok_or_else(|| anyhow!("tokenizer.ggml.tokens is not a string array"))?
                    .to_string(),
            );
        }
        let token_types = g
            .meta_array("tokenizer.ggml.token_type")
            .map(|a| a.iter().map(|v| v.as_u32().unwrap_or(0)).collect())
            .unwrap_or_else(|| vec![1u32; tokens.len()]);

        let mut id_of = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            id_of.insert(t.clone(), i as u32);
        }

        let (byte_to_uni, unicode_to_byte) = gpt2_byte_table();
        let mut byte_id = [u32::MAX; 256];
        for b in 0..256usize {
            let s: String = byte_to_uni[b].to_string();
            byte_id[b] = *id_of
                .get(&s)
                .ok_or_else(|| anyhow!("vocabulary has no byte token for 0x{b:02x}"))?;
        }

        let merges_raw = g
            .meta_array("tokenizer.ggml.merges")
            .ok_or_else(|| anyhow!("GGUF has no tokenizer.ggml.merges"))?;
        let mut merges = HashMap::with_capacity(merges_raw.len());
        for (rank, m) in merges_raw.iter().enumerate() {
            let s = m.as_str().ok_or_else(|| anyhow!("merge is not a string"))?;
            let (a, b) = s
                .split_once(' ')
                .ok_or_else(|| anyhow!("malformed merge {s:?}"))?;
            let (Some(&ia), Some(&ib)) = (id_of.get(a), id_of.get(b)) else {
                continue;
            };
            let joined = format!("{a}{b}");
            let Some(&im) = id_of.get(&joined) else {
                continue;
            };
            merges.insert((ia, ib), (rank as u32, im));
        }

        let special = |name: &str| -> Result<u32> {
            id_of
                .get(name)
                .copied()
                .ok_or_else(|| anyhow!("vocabulary has no {name} token"))
        };
        let pre = QWEN2_PATTERN;
        let re = Regex::new(pre).with_context(|| format!("pre-tokeniser regex: {pre}"))?;

        Ok(Tokenizer {
            eos: g
                .meta_u32("tokenizer.ggml.eos_token_id")
                .unwrap_or_else(|| special("<|im_end|>").unwrap_or(0)),
            pad: g.meta_u32("tokenizer.ggml.padding_token_id").unwrap_or(0),
            im_start: special("<|im_start|>")
                .or_else(|_| special("<|endoftext|>"))
                .unwrap_or(0),
            im_end: special("<|im_end|>")
                .or_else(|_| special("<|endoftext|>"))
                .unwrap_or(0),
            tokens,
            token_types,
            byte_id,
            unicode_to_byte,
            merges,
            re,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_control(&self, id: u32) -> bool {
        matches!(self.token_types.get(id as usize), Some(&3) | Some(&4))
    }

    pub fn token(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(|s| s.as_str())
    }

    fn bpe(&self, piece: &str) -> Vec<u32> {
        let mut syms: Vec<u32> = piece.bytes().map(|b| self.byte_id[b as usize]).collect();
        while syms.len() > 1 {
            let mut best: Option<(usize, u32, u32)> = None; // (pos, rank, merged)
            for i in 0..syms.len() - 1 {
                if let Some(&(rank, merged)) = self.merges.get(&(syms[i], syms[i + 1])) {
                    if best.is_none_or(|(_, r, _)| rank < r) {
                        best = Some((i, rank, merged));
                    }
                }
            }
            let Some((_, _, merged)) = best else { break };
            let mut out = Vec::with_capacity(syms.len());
            let mut i = 0;
            while i < syms.len() {
                if i + 1 < syms.len()
                    && self
                        .merges
                        .get(&(syms[i], syms[i + 1]))
                        .is_some_and(|&(_, m)| m == merged)
                {
                    out.push(merged);
                    i += 2;
                } else {
                    out.push(syms[i]);
                    i += 1;
                }
            }
            syms = out;
        }
        syms
    }

    /// Encode plain text (special tokens are not recognised, as in the
    /// tokenizer's `add_special_tokens: false` configuration).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3 + 4);
        for m in self.re.find_iter(text) {
            let Ok(m) = m else { continue };
            out.extend(self.bpe(m.as_str()));
        }
        out
    }

    /// Raw bytes for one token id.
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        let s = match self.tokens.get(id as usize) {
            Some(s) => s,
            None => return Vec::new(),
        };
        let mut out = Vec::with_capacity(s.len());
        for ch in s.chars() {
            if let Some(&b) = self.unicode_to_byte.get(&ch) {
                out.push(b);
            } else {
                let mut b = [0u8; 4];
                out.extend_from_slice(ch.encode_utf8(&mut b).as_bytes());
            }
        }
        out
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if self.is_control(id) {
                continue;
            }
            bytes.extend_from_slice(&self.token_bytes(id));
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// ChatML, as embedded in this model's `tokenizer.chat_template`.
    pub fn render_chat(
        &self,
        system: Option<&str>,
        turns: &[(Role, String)],
        add_generation_prompt: bool,
    ) -> Vec<u32> {
        let mut out = Vec::new();
        let push = |out: &mut Vec<u32>, text: &str| out.extend(self.encode(text));
        let sys = system.unwrap_or(DEFAULT_SYSTEM);
        out.push(self.im_start);
        push(&mut out, &format!("system\n{sys}"));
        out.push(self.im_end);
        push(&mut out, "\n");
        for (role, content) in turns {
            out.push(self.im_start);
            push(&mut out, &format!("{}\n{content}", role.as_str()));
            out.push(self.im_end);
            push(&mut out, "\n");
        }
        if add_generation_prompt {
            out.push(self.im_start);
            push(&mut out, "assistant\n");
        }
        out
    }
}

/// Incremental detokeniser that holds back incomplete UTF-8 sequences.
#[derive(Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    pub fn push(&mut self, tok: &Tokenizer, id: u32) -> String {
        if tok.is_control(id) {
            return String::new();
        }
        self.pending.extend_from_slice(&tok.token_bytes(id));
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let s = s.to_string();
                self.pending.clear();
                s
            }
            Err(e) => {
                let valid = e.valid_up_to();
                if valid == 0 {
                    return String::new();
                }
                let s = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
                self.pending.drain(..valid);
                s
            }
        }
    }

    pub fn flush(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}

/// GPT-2's reversible byte <-> printable-unicode table.
fn gpt2_byte_table() -> ([char; 256], HashMap<char, u8>) {
    let mut b2u = ['\0'; 256];
    let mut u2b = HashMap::with_capacity(256);
    let mut n = 0u32;
    for b in 0u32..256 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || b >= 174;
        let ch = if printable {
            char::from_u32(b).unwrap()
        } else {
            let c = char::from_u32(256 + n).unwrap();
            n += 1;
            c
        };
        b2u[b as usize] = ch;
        u2b.insert(ch, b as u8);
    }
    (b2u, u2b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_table_roundtrips() {
        let (b2u, u2b) = gpt2_byte_table();
        for b in 0..256usize {
            assert_eq!(u2b[&b2u[b]], b as u8);
        }
        // the three well-known anchors of the GPT-2 table
        assert_eq!(b2u[0x20], 'Ġ');
        assert_eq!(b2u[0x0a], 'Ċ');
        assert_eq!(b2u[b'a' as usize], 'a');
    }
}

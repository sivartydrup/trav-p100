//! Model description, weight loading / repacking, and the device plan.
//!
//! Layer split is the multi-GPU mode: there is no NVLink between PCIe P100s
//! (the two cards here sit behind a host bridge), so a per-layer all-reduce
//! would cost more than it buys. Each device owns a contiguous layer range and
//! only the 14 KiB residual row crosses the bus, once per layer boundary.

use anyhow::{anyhow, bail, Context, Result};
use trav_gguf::{GgmlType, Gguf, TensorInfo};
use trav_p100::kernels::{repack_q6k, Kernels};
use trav_p100::{Buf, Device};

use crate::tokenizer::Tokenizer;

#[derive(Clone, Debug)]
pub struct HParams {
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub eps: f32,
    pub n_ctx_train: usize,
    pub vocab: usize,
    pub n_vocab_padded: usize,
}

impl HParams {
    pub fn from_gguf(g: &Gguf) -> Result<HParams> {
        let arch = g.architecture().to_string();
        if arch != "qwen2" && arch != "qwen3" {
            bail!("unsupported architecture {arch:?}: this build implements qwen2/qwen3");
        }
        let k = |n: &str| -> Result<usize> {
            g.meta_usize(&format!("{arch}.{n}"))
                .ok_or_else(|| anyhow!("missing metadata {arch}.{n}"))
        };
        let n_embd = k("embedding_length")?;
        let n_head = k("attention.head_count")?;
        let n_head_kv = k("attention.head_count_kv")?;
        let head_dim = g
            .meta_usize(&format!("{arch}.attention.key_length"))
            .unwrap_or(n_embd / n_head);
        let vocab = g
            .meta_array("tokenizer.ggml.tokens")
            .map(|a| a.len())
            .ok_or_else(|| anyhow!("no vocabulary"))?;
        Ok(HParams {
            n_layer: k("block_count")?,
            n_head,
            n_head_kv,
            n_embd,
            n_ff: k("feed_forward_length")?,
            head_dim,
            rope_theta: g
                .meta_f32(&format!("{arch}.rope.freq_base"))
                .unwrap_or(10_000.0),
            eps: g
                .meta_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))
                .unwrap_or(1e-6),
            n_ctx_train: k("context_length").unwrap_or(4096),
            vocab,
            n_vocab_padded: vocab,
        })
    }

    pub fn q_per_kv(&self) -> usize {
        self.n_head / self.n_head_kv
    }

    /// Approximate resident bytes for weights, for the pre-flight memory check.
    pub fn weight_bytes(g: &Gguf) -> usize {
        g.tensors
            .iter()
            .map(|t| {
                if t.ty == GgmlType::Q6K {
                    (t.nbytes() / 210) * 224
                } else {
                    t.nbytes()
                }
            })
            .sum()
    }
}

/// A quantised weight matrix as the kernels see it.
pub struct QTensor {
    pub buf: Buf,
    pub q6: bool,
    pub k: usize,
    pub nrows: usize,
    pub row_stride: usize,
}

impl QTensor {
    fn load(dev: &Device, g: &Gguf, name: &str) -> Result<QTensor> {
        let t = g
            .tensor(name)
            .ok_or_else(|| anyhow!("tensor {name} missing"))?;
        Self::from_tensor(dev, g, t)
    }

    fn from_tensor(dev: &Device, g: &Gguf, t: &TensorInfo) -> Result<QTensor> {
        if t.dims.len() != 2 {
            bail!("{}: expected 2 dims, got {:?}", t.name, t.dims);
        }
        let k = t.dims[0];
        let nrows = t.dims[1];
        let (q6, bytes) = match t.ty {
            GgmlType::Q4K => (false, g.data(t).to_vec()),
            GgmlType::Q6K => (true, repack_q6k(g.data(t))),
            other => bail!(
                "{}: unsupported weight type {} (this engine handles q4_k and q6_k)",
                t.name,
                other.name()
            ),
        };
        let per_block = if q6 { 224 } else { 144 };
        let row_stride = (k / 256) * per_block;
        debug_assert_eq!(bytes.len(), row_stride * nrows, "{}", t.name);
        let buf = dev.upload(&bytes).with_context(|| t.name.clone())?;
        Ok(QTensor {
            buf,
            q6,
            k,
            nrows,
            row_stride,
        })
    }

    pub fn bytes(&self) -> usize {
        self.row_stride * self.nrows
    }
}

pub struct Layer {
    pub dev: usize,
    pub attn_norm: Buf,
    pub wq: QTensor,
    pub bq: Buf,
    pub wk: QTensor,
    pub bk: Buf,
    pub wv: QTensor,
    pub bv: Buf,
    pub wo: QTensor,
    pub ffn_norm: Buf,
    pub w_gate: QTensor,
    pub w_up: QTensor,
    pub w_down: QTensor,
}

pub struct Model {
    pub hp: HParams,
    pub tokenizer: Tokenizer,
    pub devices: Vec<Device>,
    pub kernels: Vec<Kernels>,
    pub layer_dev: Vec<usize>,
    pub first_layer: Vec<usize>,
    pub layers: Vec<Layer>,
    pub embd: QTensor,
    pub embd_dev: usize,
    pub out_norm: Buf,
    pub lm_head: QTensor,
    pub out_dev: usize,
    pub weight_bytes: usize,
}


/// Which device owns layer `l` for `ndev` devices (contiguous ranges).
pub fn layer_owner(l: usize, n_layer: usize, ndev: usize) -> usize {
    (l * ndev) / n_layer.max(1)
}

pub fn load(model_path: &str, gpu_ids: &[usize]) -> Result<Model> {
    let g = Gguf::open(model_path).with_context(|| format!("opening {model_path}"))?;
    let hp = HParams::from_gguf(&g)?;
    let tokenizer = Tokenizer::from_gguf(&g)?;

    let count = Device::count()?;
    if gpu_ids.is_empty() {
        bail!("no GPUs selected");
    }
    for &id in gpu_ids {
        if id >= count {
            bail!("requested GPU {id} but only {count} are present");
        }
    }
    let mut devices = Vec::new();
    for &id in gpu_ids {
        devices.push(Device::open(id).with_context(|| format!("opening GPU {id}"))?);
    }
    let ndev = devices.len();
    if ndev > hp.n_layer {
        bail!("{} GPUs for {} layers", ndev, hp.n_layer);
    }

    let mut kernels = Vec::new();
    for d in &devices {
        kernels.push(Kernels::load(d)?);
    }

    let weight_bytes = HParams::weight_bytes(&g);
    // Refuse to start if the plan cannot fit; the error is more useful than an
    // out-of-memory 200 ms into the load.
    for (di, d) in devices.iter().enumerate() {
        let mine: usize = g
            .tensors
            .iter()
            .filter(|t| {
                let owner = tensor_owner(t, &hp, ndev);
                owner == di
            })
            .map(|t| {
                if t.ty == GgmlType::Q6K {
                    t.nbytes() / 210 * 224
                } else {
                    t.nbytes()
                }
            })
            .sum();
        let (free, _) = d.mem_info()?;
        if mine + (64 << 20) > free {
            bail!(
                "GPU {}: need {:.2} GiB for weights, {:.2} GiB free",
                d.id,
                mine as f64 / 2f64.powi(30),
                free as f64 / 2f64.powi(30)
            );
        }
    }

    let mut layer_dev = vec![0usize; hp.n_layer];
    let mut first_layer = vec![usize::MAX; ndev];
    for l in 0..hp.n_layer {
        let d = layer_owner(l, hp.n_layer, ndev);
        layer_dev[l] = d;
        first_layer[d] = first_layer[d].min(l);
    }
    for (d, f) in first_layer.iter_mut().enumerate() {
        if *f == usize::MAX {
            *f = if d == 0 { 0 } else { hp.n_layer };
        }
    }

    let embd_dev = 0;
    let out_dev = ndev - 1;
    let embd = QTensor::load(&devices[embd_dev], &g, "token_embd.weight")?;
    let out_norm = upload_f16(&devices[out_dev], &g, "output_norm.weight")?;
    let lm_head = QTensor::load(&devices[out_dev], &g, "output.weight")?;

    let mut layers = Vec::with_capacity(hp.n_layer);
    for l in 0..hp.n_layer {
        let d = &devices[layer_dev[l]];
        let p = format!("blk.{l}");
        layers.push(Layer {
            dev: layer_dev[l],
            attn_norm: upload_f16(d, &g, &format!("{p}.attn_norm.weight"))?,
            wq: QTensor::load(d, &g, &format!("{p}.attn_q.weight"))?,
            bq: upload_f32(d, &g, &format!("{p}.attn_q.bias"))?,
            wk: QTensor::load(d, &g, &format!("{p}.attn_k.weight"))?,
            bk: upload_f32(d, &g, &format!("{p}.attn_k.bias"))?,
            wv: QTensor::load(d, &g, &format!("{p}.attn_v.weight"))?,
            bv: upload_f32(d, &g, &format!("{p}.attn_v.bias"))?,
            wo: QTensor::load(d, &g, &format!("{p}.attn_output.weight"))?,
            ffn_norm: upload_f16(d, &g, &format!("{p}.ffn_norm.weight"))?,
            w_gate: QTensor::load(d, &g, &format!("{p}.ffn_gate.weight"))?,
            w_up: QTensor::load(d, &g, &format!("{p}.ffn_up.weight"))?,
            w_down: QTensor::load(d, &g, &format!("{p}.ffn_down.weight"))?,
        });
    }

    Ok(Model {
        hp,
        tokenizer,
        devices,
        kernels,
        layer_dev,
        first_layer,
        layers,
        embd,
        embd_dev,
        out_norm,
        lm_head,
        out_dev,
        weight_bytes,
    })
}

fn tensor_owner(t: &TensorInfo, hp: &HParams, ndev: usize) -> usize {
    if let Some(rest) = t.name.strip_prefix("blk.") {
        if let Some((idx, _)) = rest.split_once('.') {
            if let Ok(l) = idx.parse::<usize>() {
                return layer_owner(l, hp.n_layer, ndev);
            }
        }
    }
    if t.name == "token_embd.weight" {
        0
    } else {
        ndev - 1
    }
}

fn upload_f16(dev: &Device, g: &Gguf, name: &str) -> Result<Buf> {
    let t = g.tensor(name).ok_or_else(|| anyhow!("tensor {name} missing"))?;
    let f = g.tensor_f32(name)?;
    let h: Vec<u16> = f
        .iter()
        .map(|v| half::f16::from_f32(*v).to_bits())
        .collect();
    let bytes = unsafe {
        std::slice::from_raw_parts(h.as_ptr() as *const u8, h.len() * 2)
    };
    let _ = t;
    Ok(dev.upload(bytes)?)
}

fn upload_f32(dev: &Device, g: &Gguf, name: &str) -> Result<Buf> {
    let f = g.tensor_f32(name)?;
    let bytes = unsafe { std::slice::from_raw_parts(f.as_ptr() as *const u8, f.len() * 4) };
    Ok(dev.upload(bytes)?)
}

//! A dependency-light GGUF v2/v3 reader.
//!
//! Tensors are memory-mapped in place: the loader hands the raw quantised bytes to
//! the GPU without a copy on the host, and split (`-00001-of-00002`) files are
//! stitched into one logical tensor map.

pub mod quant;

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use thiserror::Error;

pub use quant::{dequantize, dequantize_row, GgmlType};

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("bad magic (not a GGUF file)")]
    Magic,
    #[error("unsupported GGUF version {0}")]
    Version(u32),
    #[error("truncated or malformed GGUF: {0}")]
    Format(String),
    #[error("unknown metadata value type {0}")]
    ValueType(u32),
    #[error("shard {0} is missing tensor {1}")]
    MissingShard(usize, String),
    #[error("duplicate tensor {0}")]
    Duplicate(String),
    #[error("tensor {0}: {1}")]
    Tensor(String, String),
}

pub type Result<T, E = GgufError> = std::result::Result<T, E>;

/// A GGUF metadata value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_u32(&self) -> Option<u32> {
        match *self {
            Value::U8(v) => Some(v as u32),
            Value::U16(v) => Some(v as u32),
            Value::U32(v) => Some(v),
            Value::I32(v) if v >= 0 => Some(v as u32),
            Value::U64(v) if v <= u32::MAX as u64 => Some(v as u32),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Value::F32(v) => Some(v),
            Value::F64(v) => Some(v as f32),
            Value::U32(v) => Some(v as f32),
            Value::I32(v) => Some(v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

/// One tensor entry in the GGUF tensor directory.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// Extents, fastest axis first (`dims[0]` is the row length).
    pub dims: Vec<usize>,
    pub ty: GgmlType,
    /// Offset from the start of this shard's data section.
    pub offset: u64,
    /// Index into [`Gguf::shard_paths`].
    pub shard: usize,
}

impl TensorInfo {
    pub fn nelements(&self) -> usize {
        self.dims.iter().product()
    }

    /// Row length (fastest axis).
    pub fn row_len(&self) -> usize {
        self.dims[0]
    }

    pub fn nrows(&self) -> usize {
        self.dims[1..].iter().product()
    }

    pub fn nbytes(&self) -> usize {
        self.ty.row_bytes(self.row_len()) * self.nrows()
    }
}

struct Shard {
    mmap: Mmap,
    data_start: usize,
}

pub struct Gguf {
    pub path: PathBuf,
    pub meta: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    pub alignment: usize,
    shards: Vec<Shard>,
    index: BTreeMap<String, usize>,
}

impl fmt::Debug for Gguf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Gguf")
            .field("path", &self.path)
            .field("shards", &self.shards.len())
            .field("tensors", &self.tensors.len())
            .finish()
    }
}

impl Gguf {
    /// Open `path`, following GGUF split shards automatically.
    pub fn open(path: impl AsRef<Path>) -> Result<Gguf> {
        let path = path.as_ref().to_path_buf();
        let shard_paths = discover_shards(&path)?;
        let mut meta: BTreeMap<String, Value> = BTreeMap::new();
        let mut tensors: Vec<TensorInfo> = Vec::new();
        let mut shards = Vec::new();
        let mut alignment = 32usize;

        for (si, sp) in shard_paths.iter().enumerate() {
            let file = std::fs::File::open(sp)?;
            let mmap = unsafe { Mmap::map(&file)? };
            let (sm, st, data_start, align) = parse_shard(&mmap, sp, si)?;
            if si == 0 {
                meta = sm;
                alignment = align;
            }
            tensors.extend(st);
            shards.push(Shard { mmap, data_start });
        }

        // Split files each carry the full tensor count; validate we got them all.
        if let Some(Value::U32(total)) = meta.get("split.tensors.count") {
            if tensors.len() != *total as usize {
                return Err(GgufError::Format(format!(
                    "expected {} tensors across shards, found {}",
                    total,
                    tensors.len()
                )));
            }
        }

        let mut index = BTreeMap::new();
        for (i, t) in tensors.iter().enumerate() {
            if index.insert(t.name.clone(), i).is_some() {
                return Err(GgufError::Duplicate(t.name.clone()));
            }
        }
        // Sorting by name makes layer iteration predictable for debugging.
        tensors.sort_by(|a, b| a.name.cmp(&b.name));
        index.clear();
        for (i, t) in tensors.iter().enumerate() {
            index.insert(t.name.clone(), i);
        }

        Ok(Gguf {
            path,
            meta,
            tensors,
            alignment,
            shards,
            index,
        })
    }

    pub fn shard_paths(&self) -> impl Iterator<Item = &Path> {
        // Shard paths are only needed for diagnostics; the mmap holds the data.
        std::iter::empty()
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    /// Raw bytes of a tensor, borrowing from the mmap.
    pub fn data(&self, t: &TensorInfo) -> &[u8] {
        let shard = &self.shards[t.shard];
        let start = shard.data_start + t.offset as usize;
        let end = start + t.nbytes();
        &shard.mmap[start..end]
    }

    pub fn data_of(&self, name: &str) -> Result<&[u8]> {
        let t = self
            .tensor(name)
            .ok_or_else(|| GgufError::Tensor(name.to_string(), "not found".into()))?;
        Ok(self.data(t))
    }

    /// Dequantise a whole tensor to f32 on the host (small tensors only:
    /// norms, biases, embeddings for reference checks).
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let t = self
            .tensor(name)
            .ok_or_else(|| GgufError::Tensor(name.to_string(), "not found".into()))?;
        let mut out = vec![0f32; t.nelements()];
        dequantize(t.ty, self.data(t), &mut out);
        Ok(out)
    }

    pub fn meta_u32(&self, k: &str) -> Option<u32> {
        self.meta.get(k).and_then(Value::as_u32)
    }

    pub fn meta_usize(&self, k: &str) -> Option<usize> {
        self.meta_u32(k).map(|v| v as usize)
    }

    pub fn meta_f32(&self, k: &str) -> Option<f32> {
        self.meta.get(k).and_then(Value::as_f32)
    }

    pub fn meta_str(&self, k: &str) -> Option<&str> {
        self.meta.get(k).and_then(Value::as_str)
    }

    pub fn meta_bool(&self, k: &str) -> Option<bool> {
        self.meta.get(k).and_then(Value::as_bool)
    }

    pub fn meta_array(&self, k: &str) -> Option<&[Value]> {
        self.meta.get(k).and_then(Value::as_array)
    }

    pub fn architecture(&self) -> &str {
        self.meta_str("general.architecture").unwrap_or("unknown")
    }

    pub fn name(&self) -> &str {
        self.meta_str("general.name").unwrap_or("unnamed")
    }
}

fn discover_shards(path: &Path) -> Result<Vec<PathBuf>> {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| GgufError::Format("non-UTF-8 file name".into()))?;
    let stem = name.strip_suffix(".gguf").unwrap_or(name);
    let Some((left, right)) = stem.rsplit_once("-of-") else {
        return Ok(vec![path.to_path_buf()]);
    };
    let Some((prefix, idx)) = left.rsplit_once('-') else {
        return Ok(vec![path.to_path_buf()]);
    };
    let (Ok(total), Ok(first)) = (right.parse::<usize>(), idx.parse::<usize>()) else {
        return Ok(vec![path.to_path_buf()]);
    };
    if total <= 1 {
        return Ok(vec![path.to_path_buf()]);
    }
    let width = idx.len();
    let dir = path.parent().unwrap_or(Path::new("."));
    let mut out = Vec::with_capacity(total);
    for i in 1..=total {
        let p = dir.join(format!("{prefix}-{i:0width$}-of-{right:0width$}.gguf"));
        // `right` may have its own width; rebuild it verbatim.
        let p = if p.exists() {
            p
        } else {
            dir.join(format!("{prefix}-{i:0width$}-of-{right}.gguf"))
        };
        if !p.exists() {
            return Err(GgufError::MissingShard(i, p.display().to_string()));
        }
        out.push(p);
    }
    let _ = first;
    Ok(out)
}

const MAX_META_ITEMS: u64 = 1 << 24;

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Cursor { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            return Err(GgufError::Format(format!(
                "read of {n} bytes at {} past end ({})",
                self.pos,
                self.buf.len()
            )));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn i8(&mut self) -> Result<i8> {
        Ok(self.u8()? as i8)
    }

    fn i16(&mut self) -> Result<i16> {
        Ok(self.u16()? as i16)
    }

    fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()? as i32)
    }

    fn i64(&mut self) -> Result<i64> {
        Ok(self.u64()? as i64)
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }

    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        let b = self.take(n)?;
        String::from_utf8(b.to_vec())
            .map_err(|_| GgufError::Format("metadata string is not UTF-8".into()))
    }

    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.i8()?),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.i16()?),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.i32()?),
            6 => Value::F32(self.f32()?),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::String(self.string()?),
            9 => {
                let at = self.u32()?;
                let n = self.u64()?;
                if n > MAX_META_ITEMS {
                    return Err(GgufError::Format(format!("array of {n} items")));
                }
                let mut v = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    v.push(self.value(at)?);
                }
                Value::Array(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.i64()?),
            12 => Value::F64(self.f64()?),
            other => return Err(GgufError::ValueType(other)),
        })
    }
}

fn parse_shard(
    buf: &[u8],
    path: &Path,
    shard_idx: usize,
) -> Result<(BTreeMap<String, Value>, Vec<TensorInfo>, usize, usize)> {
    let mut c = Cursor::new(buf);
    if c.take(4)? != b"GGUF" {
        return Err(GgufError::Magic);
    }
    let version = c.u32()?;
    if version != 2 && version != 3 {
        return Err(GgufError::Version(version));
    }
    let n_tensors = c.u64()?;
    let n_kv = c.u64()?;
    if n_tensors > MAX_META_ITEMS || n_kv > MAX_META_ITEMS {
        return Err(GgufError::Format("implausible tensor/metadata count".into()));
    }

    let mut meta = BTreeMap::new();
    for _ in 0..n_kv {
        let key = c.string()?;
        let ty = c.u32()?;
        let val = c.value(ty)?;
        meta.insert(key, val);
    }

    let align = meta
        .get("general.alignment")
        .and_then(Value::as_u32)
        .unwrap_or(32) as usize;
    if align == 0 || !align.is_power_of_two() {
        return Err(GgufError::Format(format!("bad alignment {align}")));
    }

    let mut tensors = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = c.string()?;
        let nd = c.u32()? as usize;
        if nd > 4 {
            return Err(GgufError::Format(format!("{name}: {nd} dims")));
        }
        let mut dims = Vec::with_capacity(nd);
        for _ in 0..nd {
            dims.push(c.u64()? as usize);
        }
        let ty = GgmlType::from_u32(c.u32()?);
        let offset = c.u64()?;
        tensors.push(TensorInfo {
            name,
            dims,
            ty,
            offset,
            shard: shard_idx,
        });
    }

    let data_start = c.pos.div_ceil(align) * align;
    if data_start > buf.len() {
        return Err(GgufError::Format("data section past end of file".into()));
    }

    // Bounds-check every tensor against this shard's mapping.
    for t in &tensors {
        let end = t.offset as usize + t.nbytes();
        if data_start + end > buf.len() {
            return Err(GgufError::Tensor(
                t.name.clone(),
                format!(
                    "{} bytes at {} exceeds shard size {} ({})",
                    t.nbytes(),
                    data_start + t.offset as usize,
                    buf.len(),
                    path.display()
                ),
            ));
        }
        if t.ty == GgmlType::Q4K || t.ty == GgmlType::Q6K {
            if t.row_len() % 256 != 0 {
                return Err(GgufError::Tensor(
                    t.name.clone(),
                    format!("row length {} is not a multiple of 256", t.row_len()),
                ));
            }
        }
    }

    Ok((meta, tensors, data_start, align))
}

//! GGUF v3 model format — minimal mmap'd parser for BitNet GGUFs.
//!
//! Enough of the format to load any public BitNet b1.58 GGUF (notably
//! `microsoft/bitnet-b1.58-2B-4t-gguf`) without requantizing to our native
//! `.h1b`. The top-level [`GgufFile`] remains *parse only*: tensor bytes
//! are handed back as raw `&[u8]` slices. Actual bit-unpacking from
//! llama.cpp ternary formats into halo's 2-bit packed layout lives in
//! [`unpack`]; see [`unpack::iq2_s_to_halo_v2`] for the BitNet-compatible
//! IQ2_S decoder.
//!
//! Layout (little-endian everywhere, offsets in bytes from start of file):
//!
//! ```text
//!   0x00  4   magic             = b"GGUF" (0x46554747 LE)
//!   0x04  4   version           (u32 ; we require >= 3)
//!   0x08  8   tensor_count      (u64)
//!   0x10  8   metadata_kv_count (u64)
//!   0x18  .   metadata_kvs      (kv_count × { gguf_string key; u32 type; <value> })
//!   ....  .   tensor_infos      (tensor_count × { gguf_string name; u32 n_dims;
//!                                                 u64[n_dims] shape; u32 dtype;
//!                                                 u64 offset })
//!   ....  .   alignment pad to `general.alignment` (default 32)
//!   ....  .   tensor data (absolute_offset = data_start + tensor.offset)
//! ```
//!
//! `gguf_string` = `{ u64 len; bytes[len] }`.
//!
//! Reference
//! ---------
//! Structure of this module was inspired by the Rust GGUF loader in
//! `Wavegoodvybe2929/bitnet-rust` (MIT / Apache-2.0), particularly the
//! `GgufValueType` and `GgufTensorType` enum layout and the top-level
//! parsing flow. No block of >~20 lines was lifted verbatim; the bulk of the
//! decoding was re-written against [`crate::h1b::Mapped`] so everything is
//! mmap-backed and zero-copy, which the upstream loader does not do.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use byteorder::{ByteOrder, LittleEndian};

use crate::error::HaloError;
use crate::h1b::Mapped;

mod iq2s_grid;
pub mod unpack;

pub const GGUF_MAGIC: [u8; 4] = *b"GGUF"; // 0x47475546 LE
pub const GGUF_MIN_VERSION: u32 = 3;

/// Default alignment for tensor data when no `general.alignment` key is set.
/// Matches llama.cpp's `GGUF_DEFAULT_ALIGNMENT`.
const DEFAULT_ALIGNMENT: u64 = 32;

// --------------------------------------------------------------------------
// GGUF scalar / value types
// --------------------------------------------------------------------------

/// Metadata value type tag (first u32 of a KV pair's value).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufValueType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl GgufValueType {
    fn try_from_u32(v: u32) -> Result<Self, HaloError> {
        Ok(match v {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            _ => return Err(HaloError::InvalidConfig("unknown GGUF value type")),
        })
    }
}

/// Tensor data type (the GGML GGUFTensorType enum). Only the variants we
/// can either consume or skip past are listed explicitly; everything else
/// falls into [`GgufTensorType::Unknown`] — parsing continues, but callers
/// won't be able to decode the payload.
///
/// The variant names match the canonical GGML names (`Q4_K`, `IQ2_S`, …)
/// rather than Rust's `UpperCamel` preference; using the upstream names
/// verbatim keeps grep-ability with llama.cpp / ggml-quants.h.
#[allow(non_camel_case_types)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufTensorType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    /// Catch-all. Raw `u32` preserved so callers can still route on it.
    Unknown(u32),
}

impl GgufTensorType {
    fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            16 => Self::IQ2_XXS,
            17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS,
            19 => Self::IQ1_S,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            23 => Self::IQ4_XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1_M,
            30 => Self::BF16,
            34 => Self::TQ1_0,
            35 => Self::TQ2_0,
            other => Self::Unknown(other),
        }
    }

    /// Raw u32 tag, primarily for debug / logging / round-trip.
    pub fn as_u32(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2_K => 10,
            Self::Q3_K => 11,
            Self::Q4_K => 12,
            Self::Q5_K => 13,
            Self::Q6_K => 14,
            Self::Q8_K => 15,
            Self::IQ2_XXS => 16,
            Self::IQ2_XS => 17,
            Self::IQ3_XXS => 18,
            Self::IQ1_S => 19,
            Self::IQ4_NL => 20,
            Self::IQ3_S => 21,
            Self::IQ2_S => 22,
            Self::IQ4_XS => 23,
            Self::I8 => 24,
            Self::I16 => 25,
            Self::I32 => 26,
            Self::I64 => 27,
            Self::F64 => 28,
            Self::IQ1_M => 29,
            Self::BF16 => 30,
            Self::TQ1_0 => 34,
            Self::TQ2_0 => 35,
            Self::Unknown(v) => v,
        }
    }
}

/// A parsed metadata value. Arrays carry a typed list of children. Strings
/// own their bytes (small and finite — there's no point mmap-aliasing them).
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(GgufArray),
    U64(u64),
    I64(i64),
    F64(f64),
}

/// Typed homogeneous array. `elem_type` is preserved so callers can tell
/// `[]u32` from `[]string` without pattern-matching on the first element.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufArray {
    pub elem_type: u32,
    pub values: Vec<GgufValue>,
}

impl GgufValue {
    /// Helper: interpret as string, if it is one. Does *not* coerce
    /// numerics into strings.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Helper: interpret small scalar as u32. Covers the cases where
    /// a GGUF producer chose U32/I32/U64/I64 for the same logical field
    /// (common for model dims).
    pub fn as_u32(&self) -> Option<u32> {
        match *self {
            GgufValue::U8(v) => Some(v as u32),
            GgufValue::U16(v) => Some(v as u32),
            GgufValue::U32(v) => Some(v),
            GgufValue::I8(v) if v >= 0 => Some(v as u32),
            GgufValue::I16(v) if v >= 0 => Some(v as u32),
            GgufValue::I32(v) if v >= 0 => Some(v as u32),
            GgufValue::U64(v) if v <= u32::MAX as u64 => Some(v as u32),
            GgufValue::I64(v) if v >= 0 && v <= u32::MAX as i64 => Some(v as u32),
            _ => None,
        }
    }

    /// Helper: interpret any numeric / bool as f32.
    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            GgufValue::F32(v) => Some(v),
            GgufValue::F64(v) => Some(v as f32),
            GgufValue::U8(v) => Some(v as f32),
            GgufValue::U16(v) => Some(v as f32),
            GgufValue::U32(v) => Some(v as f32),
            GgufValue::U64(v) => Some(v as f32),
            GgufValue::I8(v) => Some(v as f32),
            GgufValue::I16(v) => Some(v as f32),
            GgufValue::I32(v) => Some(v as f32),
            GgufValue::I64(v) => Some(v as f32),
            _ => None,
        }
    }
}

/// Tensor directory entry. `offset` is *relative to the aligned tensor
/// data start*, not absolute in the file. Use [`GgufFile::tensor`] to get
/// the byte slice.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: GgufTensorType,
    /// Offset from the start of the tensor-data region (not from file start).
    pub offset: u64,
    /// Size in bytes of this tensor's raw payload (computed from dtype +
    /// shape + block size). `None` for [`GgufTensorType::Unknown`].
    pub size_bytes: Option<u64>,
}

// --------------------------------------------------------------------------
// GgufFile
// --------------------------------------------------------------------------

/// A mmap-backed parsed GGUF file. Metadata KVs + tensor directory live on
/// the Rust heap (bounded, ~kilobytes); tensor payloads stay in the mmap.
pub struct GgufFile {
    #[allow(dead_code)]
    path: PathBuf,
    mmap: Mapped,
    version: u32,
    tensor_data_start: u64,
    metadata: BTreeMap<String, GgufValue>,
    tensors: Vec<GgufTensorInfo>,
    /// Name → index into `tensors`, populated during parse for O(log n) lookup.
    tensor_index: BTreeMap<String, usize>,
}

impl std::fmt::Debug for GgufFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufFile")
            .field("path", &self.path)
            .field("bytes", &self.mmap.as_slice().len())
            .field("version", &self.version)
            .field("tensor_count", &self.tensors.len())
            .field("metadata_count", &self.metadata.len())
            .field("tensor_data_start", &self.tensor_data_start)
            .finish()
    }
}

impl GgufFile {
    /// Mmap and parse.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, HaloError> {
        let p = path.as_ref();
        let file = File::open(p).map_err(|e| HaloError::io_at(p, e))?;
        let mmap = Mapped::from_file_public(&file).map_err(|e| HaloError::io_at(p, e))?;
        Self::parse(p.to_path_buf(), mmap)
    }

    /// Parse an already-loaded buffer. Used by tests; in production, prefer
    /// [`Self::open`] which uses mmap.
    pub fn parse_bytes(path: impl Into<PathBuf>, bytes: Vec<u8>) -> Result<Self, HaloError> {
        Self::parse(path.into(), Mapped::Owned(bytes))
    }

    fn parse(path: PathBuf, mmap: Mapped) -> Result<Self, HaloError> {
        let buf = mmap.as_slice();
        // Header is 4 + 4 + 8 + 8 = 24 bytes minimum.
        if buf.len() < 24 {
            return Err(HaloError::Truncated {
                offset: 0,
                needed: 24,
                have: buf.len(),
            });
        }

        let mut magic = [0u8; 4];
        magic.copy_from_slice(&buf[0..4]);
        if magic != GGUF_MAGIC {
            return Err(HaloError::BadMagic {
                expected: GGUF_MAGIC,
                got: magic,
            });
        }

        let version = LittleEndian::read_u32(&buf[4..8]);
        if version < GGUF_MIN_VERSION {
            return Err(HaloError::UnsupportedVersion {
                version: version as i32,
                min: GGUF_MIN_VERSION as i32,
                max: i32::MAX,
            });
        }

        let tensor_count = LittleEndian::read_u64(&buf[8..16]);
        let metadata_kv_count = LittleEndian::read_u64(&buf[16..24]);

        let mut cur = Cursor::new(buf, 24);

        // ---- Metadata KVs ----
        let mut metadata: BTreeMap<String, GgufValue> = BTreeMap::new();
        for _ in 0..metadata_kv_count {
            let key = cur.read_gguf_string()?;
            let value_type_raw = cur.read_u32()?;
            let value_type = GgufValueType::try_from_u32(value_type_raw)?;
            let value = cur.read_value(value_type)?;
            metadata.insert(key, value);
        }

        // ---- Tensor directory ----
        let mut tensors: Vec<GgufTensorInfo> = Vec::with_capacity(tensor_count as usize);
        let mut tensor_index = BTreeMap::new();
        for _ in 0..tensor_count {
            let name = cur.read_gguf_string()?;
            let n_dims = cur.read_u32()?;
            if n_dims > 8 {
                // GGUF spec says "at least 4"; 8 is already a huge red flag.
                return Err(HaloError::InvalidConfig(
                    "tensor n_dims > 8, file looks corrupt",
                ));
            }
            let mut shape = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                shape.push(cur.read_u64()?);
            }
            let dtype_raw = cur.read_u32()?;
            let dtype = GgufTensorType::from_u32(dtype_raw);
            let offset = cur.read_u64()?;
            let size_bytes = tensor_size_bytes(dtype, &shape);
            let idx = tensors.len();
            tensor_index.insert(name.clone(), idx);
            tensors.push(GgufTensorInfo {
                name,
                shape,
                dtype,
                offset,
                size_bytes,
            });
        }

        // ---- Alignment to tensor-data region ----
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .map(|a| a as u64)
            .unwrap_or(DEFAULT_ALIGNMENT);
        let hdr_end = cur.pos as u64;
        let pad = (alignment - (hdr_end % alignment)) % alignment;
        let tensor_data_start = hdr_end + pad;

        Ok(Self {
            path,
            mmap,
            version,
            tensor_data_start,
            metadata,
            tensors,
            tensor_index,
        })
    }

    // ---- Accessors ----

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn tensors(&self) -> &[GgufTensorInfo] {
        &self.tensors
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn metadata(&self) -> &BTreeMap<String, GgufValue> {
        &self.metadata
    }

    pub fn tensor_data_start(&self) -> u64 {
        self.tensor_data_start
    }

    /// Raw backing bytes (whole file).
    pub fn bytes(&self) -> &[u8] {
        self.mmap.as_slice()
    }

    /// Fetch a metadata value by key.
    pub fn kv(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    /// Fetch a tensor's metadata entry by name.
    pub fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensor_index
            .get(name)
            .and_then(|i| self.tensors.get(*i))
    }

    /// Fetch a tensor's raw payload bytes. Returns `None` if the tensor is
    /// unknown, an unknown dtype (so we can't compute its byte size), or
    /// if the recorded offset+size runs off the end of the mmap.
    pub fn tensor(&self, name: &str) -> Option<&[u8]> {
        let info = self.tensor_info(name)?;
        let sz = info.size_bytes? as usize;
        let abs = self.tensor_data_start.checked_add(info.offset)? as usize;
        let end = abs.checked_add(sz)?;
        let buf = self.mmap.as_slice();
        if end > buf.len() {
            return None;
        }
        Some(&buf[abs..end])
    }

    /// Convenience: extract the BitNet-relevant header fields into a
    /// struct shaped the same as [`crate::h1b::H1bConfig`] (for the fields
    /// it has in common). See [`BitnetHeader`] for the full shape.
    pub fn read_bitnet_metadata(&self) -> Result<BitnetHeader, HaloError> {
        BitnetHeader::from_gguf(self)
    }
}

// --------------------------------------------------------------------------
// BitnetHeader — bridge between GGUF KV store and our H1bConfig shape.
// --------------------------------------------------------------------------

/// A BitNet-shaped view of a GGUF file's metadata.
///
/// The GGUF metadata is namespaced by architecture (`llama.*` for
/// `general.architecture=="llama"`, `bitnet.*` for the less-common native
/// bitnet arch, etc.). We read `general.architecture` first and then pick
/// the right namespace. For fields that don't appear, we fall back to the
/// same defaults our [`crate::h1b`] parser uses (`DEFAULT_ROPE_THETA`,
/// `DEFAULT_RMS_NORM_EPS`) so downstream code can treat `.h1b` and
/// `.gguf` identically at the router level.
///
/// `tokens` and `merges` are lifted out of their arrays so callers don't
/// have to re-pattern-match on [`GgufValue::Array`]. Both are potentially
/// large (50k+ entries) but we only store the strings, not the array
/// wrapper — so this is still well under a megabyte.
#[derive(Debug, Clone)]
pub struct BitnetHeader {
    pub architecture: String,
    pub block_count: u32,
    pub embedding_length: u32,
    pub feed_forward_length: u32,
    pub attention_head_count: u32,
    pub attention_head_count_kv: u32,
    pub rope_freq_base: f32,
    pub rms_norm_eps: f32,

    pub tokenizer_model: String,
    pub tokens: Vec<String>,
    pub merges: Vec<String>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
}

impl BitnetHeader {
    pub fn from_gguf(g: &GgufFile) -> Result<Self, HaloError> {
        // 1) Architecture — dictates which namespace the model-dim keys live in.
        let architecture = g
            .kv("general.architecture")
            .and_then(|v| v.as_str())
            .ok_or(HaloError::InvalidConfig(
                "GGUF missing general.architecture",
            ))?
            .to_string();

        // BitNet b1.58 2B4T ships as arch=="bitnet" in some exports, "llama"
        // in others (Microsoft's official release is "bitnet"). Try the
        // arch-specific namespace first, then fall back to llama.* — llama
        // is the de-facto lingua franca for these keys.
        let ns_primary = architecture.as_str();
        let ns_fallback = "llama";

        let pick_u32 = |key_tail: &str| -> Option<u32> {
            g.kv(&format!("{}.{}", ns_primary, key_tail))
                .and_then(|v| v.as_u32())
                .or_else(|| {
                    g.kv(&format!("{}.{}", ns_fallback, key_tail))
                        .and_then(|v| v.as_u32())
                })
        };
        let pick_f32 = |key_tail: &str| -> Option<f32> {
            g.kv(&format!("{}.{}", ns_primary, key_tail))
                .and_then(|v| v.as_f32())
                .or_else(|| {
                    g.kv(&format!("{}.{}", ns_fallback, key_tail))
                        .and_then(|v| v.as_f32())
                })
        };

        let block_count = pick_u32("block_count")
            .ok_or(HaloError::InvalidConfig("GGUF missing <arch>.block_count"))?;
        let embedding_length = pick_u32("embedding_length").ok_or(HaloError::InvalidConfig(
            "GGUF missing <arch>.embedding_length",
        ))?;
        let feed_forward_length = pick_u32("feed_forward_length").ok_or(
            HaloError::InvalidConfig("GGUF missing <arch>.feed_forward_length"),
        )?;
        let attention_head_count = pick_u32("attention.head_count").ok_or(
            HaloError::InvalidConfig("GGUF missing <arch>.attention.head_count"),
        )?;
        // head_count_kv defaults to head_count when absent (MHA, not GQA).
        let attention_head_count_kv =
            pick_u32("attention.head_count_kv").unwrap_or(attention_head_count);
        let rope_freq_base = pick_f32("rope.freq_base").unwrap_or(crate::types::DEFAULT_ROPE_THETA);
        let rms_norm_eps = pick_f32("attention.layer_norm_rms_epsilon")
            .unwrap_or(crate::types::DEFAULT_RMS_NORM_EPS);

        // 2) Tokenizer block. Everything under `tokenizer.ggml.*`.
        let tokenizer_model = g
            .kv("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let tokens = g
            .kv("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                GgufValue::Array(a) => Some(
                    a.values
                        .iter()
                        .filter_map(|x| x.as_str().map(|s| s.to_string()))
                        .collect::<Vec<_>>(),
                ),
                _ => None,
            })
            .unwrap_or_default();

        let merges = g
            .kv("tokenizer.ggml.merges")
            .and_then(|v| match v {
                GgufValue::Array(a) => Some(
                    a.values
                        .iter()
                        .filter_map(|x| x.as_str().map(|s| s.to_string()))
                        .collect::<Vec<_>>(),
                ),
                _ => None,
            })
            .unwrap_or_default();

        let bos_token_id = g.kv("tokenizer.ggml.bos_token_id").and_then(|v| v.as_u32());
        let eos_token_id = g.kv("tokenizer.ggml.eos_token_id").and_then(|v| v.as_u32());

        Ok(Self {
            architecture,
            block_count,
            embedding_length,
            feed_forward_length,
            attention_head_count,
            attention_head_count_kv,
            rope_freq_base,
            rms_norm_eps,
            tokenizer_model,
            tokens,
            merges,
            bos_token_id,
            eos_token_id,
        })
    }
}

// --------------------------------------------------------------------------
// Cursor — bounds-checked reader that threads a byte position through the
// file. Private; no callers outside this module.
// --------------------------------------------------------------------------

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8], pos: usize) -> Self {
        Self { buf, pos }
    }

    fn ensure(&self, n: usize) -> Result<(), HaloError> {
        if self.pos + n > self.buf.len() {
            Err(HaloError::Truncated {
                offset: self.pos,
                needed: n,
                have: self.buf.len().saturating_sub(self.pos),
            })
        } else {
            Ok(())
        }
    }

    fn read_u8(&mut self) -> Result<u8, HaloError> {
        self.ensure(1)?;
        let v = self.buf[self.pos];
        self.pos += 1;
        Ok(v)
    }
    fn read_i8(&mut self) -> Result<i8, HaloError> {
        Ok(self.read_u8()? as i8)
    }
    fn read_u16(&mut self) -> Result<u16, HaloError> {
        self.ensure(2)?;
        let v = LittleEndian::read_u16(&self.buf[self.pos..self.pos + 2]);
        self.pos += 2;
        Ok(v)
    }
    fn read_i16(&mut self) -> Result<i16, HaloError> {
        self.ensure(2)?;
        let v = LittleEndian::read_i16(&self.buf[self.pos..self.pos + 2]);
        self.pos += 2;
        Ok(v)
    }
    fn read_u32(&mut self) -> Result<u32, HaloError> {
        self.ensure(4)?;
        let v = LittleEndian::read_u32(&self.buf[self.pos..self.pos + 4]);
        self.pos += 4;
        Ok(v)
    }
    fn read_i32(&mut self) -> Result<i32, HaloError> {
        self.ensure(4)?;
        let v = LittleEndian::read_i32(&self.buf[self.pos..self.pos + 4]);
        self.pos += 4;
        Ok(v)
    }
    fn read_f32(&mut self) -> Result<f32, HaloError> {
        self.ensure(4)?;
        let v = LittleEndian::read_f32(&self.buf[self.pos..self.pos + 4]);
        self.pos += 4;
        Ok(v)
    }
    fn read_u64(&mut self) -> Result<u64, HaloError> {
        self.ensure(8)?;
        let v = LittleEndian::read_u64(&self.buf[self.pos..self.pos + 8]);
        self.pos += 8;
        Ok(v)
    }
    fn read_i64(&mut self) -> Result<i64, HaloError> {
        self.ensure(8)?;
        let v = LittleEndian::read_i64(&self.buf[self.pos..self.pos + 8]);
        self.pos += 8;
        Ok(v)
    }
    fn read_f64(&mut self) -> Result<f64, HaloError> {
        self.ensure(8)?;
        let v = LittleEndian::read_f64(&self.buf[self.pos..self.pos + 8]);
        self.pos += 8;
        Ok(v)
    }
    fn read_bool(&mut self) -> Result<bool, HaloError> {
        Ok(self.read_u8()? != 0)
    }

    /// GGUF string = `{ u64 len ; bytes[len] }`. Non-UTF-8 falls back to
    /// lossy decoding so a pathological tokenizer blob can't kill parsing.
    fn read_gguf_string(&mut self) -> Result<String, HaloError> {
        let len = self.read_u64()? as usize;
        self.ensure(len)?;
        let slice = &self.buf[self.pos..self.pos + len];
        self.pos += len;
        Ok(match std::str::from_utf8(slice) {
            Ok(s) => s.to_string(),
            Err(_) => String::from_utf8_lossy(slice).into_owned(),
        })
    }

    fn read_value(&mut self, ty: GgufValueType) -> Result<GgufValue, HaloError> {
        Ok(match ty {
            GgufValueType::U8 => GgufValue::U8(self.read_u8()?),
            GgufValueType::I8 => GgufValue::I8(self.read_i8()?),
            GgufValueType::U16 => GgufValue::U16(self.read_u16()?),
            GgufValueType::I16 => GgufValue::I16(self.read_i16()?),
            GgufValueType::U32 => GgufValue::U32(self.read_u32()?),
            GgufValueType::I32 => GgufValue::I32(self.read_i32()?),
            GgufValueType::F32 => GgufValue::F32(self.read_f32()?),
            GgufValueType::Bool => GgufValue::Bool(self.read_bool()?),
            GgufValueType::String => GgufValue::String(self.read_gguf_string()?),
            GgufValueType::U64 => GgufValue::U64(self.read_u64()?),
            GgufValueType::I64 => GgufValue::I64(self.read_i64()?),
            GgufValueType::F64 => GgufValue::F64(self.read_f64()?),
            GgufValueType::Array => {
                let elem_type_raw = self.read_u32()?;
                let elem_type = GgufValueType::try_from_u32(elem_type_raw)?;
                let n = self.read_u64()? as usize;
                let mut values = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    values.push(self.read_value(elem_type)?);
                }
                GgufValue::Array(GgufArray {
                    elem_type: elem_type_raw,
                    values,
                })
            }
        })
    }
}

// --------------------------------------------------------------------------
// Tensor byte-size math (per GGML block layout).
// --------------------------------------------------------------------------

/// Return payload size in bytes for a tensor with dtype `t` and shape
/// `shape`, or `None` if the dtype is `Unknown` / we don't know its block
/// packing.
fn tensor_size_bytes(t: GgufTensorType, shape: &[u64]) -> Option<u64> {
    let elems: u64 = shape.iter().copied().product();
    // (block_size_elems, bytes_per_block). For non-block types, block_size=1
    // and bytes_per_block is just the element width.
    let (bs, bpb): (u64, u64) = match t {
        GgufTensorType::F32 => (1, 4),
        GgufTensorType::F16 | GgufTensorType::BF16 => (1, 2),
        GgufTensorType::F64 => (1, 8),
        GgufTensorType::I8 => (1, 1),
        GgufTensorType::I16 => (1, 2),
        GgufTensorType::I32 => (1, 4),
        GgufTensorType::I64 => (1, 8),

        // Standard quants: 32-elem blocks in llama.cpp, sizes per ggml.h.
        GgufTensorType::Q4_0 => (32, 2 + 16),
        GgufTensorType::Q4_1 => (32, 2 + 2 + 16),
        GgufTensorType::Q5_0 => (32, 2 + 4 + 16),
        GgufTensorType::Q5_1 => (32, 2 + 2 + 4 + 16),
        GgufTensorType::Q8_0 => (32, 2 + 32),
        GgufTensorType::Q8_1 => (32, 4 + 4 + 32),

        // K-quants: 256-elem super-blocks.
        GgufTensorType::Q2_K => (256, 256 / 16 + 256 / 4 + 2 + 2),
        GgufTensorType::Q3_K => (256, 256 / 8 + 256 / 4 + 12 + 2),
        GgufTensorType::Q4_K => (256, 2 + 2 + 12 + 256 / 2),
        GgufTensorType::Q5_K => (256, 2 + 2 + 12 + 256 / 8 + 256 / 2),
        GgufTensorType::Q6_K => (256, 256 / 2 + 256 / 4 + 256 / 16 + 2),
        GgufTensorType::Q8_K => (256, 4 + 256 + 2 * 256 / 16),

        // IQ-quants — block sizes per upstream ggml.h / ggml-quants.h.
        GgufTensorType::IQ2_XXS => (256, 2 + 256 / 4),
        GgufTensorType::IQ2_XS => (256, 2 + 256 / 4 + 256 / 32),
        GgufTensorType::IQ2_S => (256, 2 + 256 / 4 + 256 / 16),
        GgufTensorType::IQ3_XXS => (256, 2 + 256 / 4 + 256 / 8),
        GgufTensorType::IQ3_S => (256, 2 + 256 / 4 + 256 / 8 + 256 / 32 + 4),
        GgufTensorType::IQ1_S => (256, 2 + 256 / 8 + 256 / 16),
        GgufTensorType::IQ1_M => (256, 256 / 8 + 256 / 16 + 256 / 32),
        GgufTensorType::IQ4_NL => (32, 2 + 16),
        GgufTensorType::IQ4_XS => (256, 2 + 2 + 256 / 64 + 256 / 2),

        // Ternary base-3 (BitNet native in llama.cpp TQ1_0 / TQ2_0).
        GgufTensorType::TQ1_0 => (256, 2 + 4 * 13 + 128 / 64),
        GgufTensorType::TQ2_0 => (256, 2 + 64),

        GgufTensorType::Unknown(_) => return None,
    };
    if bs == 0 {
        return None;
    }
    // Tensors are laid out as blocks of `bs` elements. Element count must
    // be a multiple of bs; if not, we can't safely size it and bail.
    if !elems.is_multiple_of(bs) {
        return None;
    }
    Some((elems / bs) * bpb)
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Test buffer builders -----------------------------------------------

    fn put_u32(buf: &mut Vec<u8>, v: u32) {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fn put_u64(buf: &mut Vec<u8>, v: u64) {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fn put_f32(buf: &mut Vec<u8>, v: f32) {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fn put_gguf_str(buf: &mut Vec<u8>, s: &str) {
        put_u64(buf, s.len() as u64);
        buf.extend_from_slice(s.as_bytes());
    }

    /// Push a metadata KV: key, u32 type tag, inline value payload.
    fn put_kv_string(buf: &mut Vec<u8>, key: &str, val: &str) {
        put_gguf_str(buf, key);
        put_u32(buf, GgufValueType::String as u32);
        put_gguf_str(buf, val);
    }
    fn put_kv_u32(buf: &mut Vec<u8>, key: &str, val: u32) {
        put_gguf_str(buf, key);
        put_u32(buf, GgufValueType::U32 as u32);
        put_u32(buf, val);
    }
    fn put_kv_f32(buf: &mut Vec<u8>, key: &str, val: f32) {
        put_gguf_str(buf, key);
        put_u32(buf, GgufValueType::F32 as u32);
        put_f32(buf, val);
    }
    fn put_kv_array_str(buf: &mut Vec<u8>, key: &str, items: &[&str]) {
        put_gguf_str(buf, key);
        put_u32(buf, GgufValueType::Array as u32);
        put_u32(buf, GgufValueType::String as u32);
        put_u64(buf, items.len() as u64);
        for s in items {
            put_gguf_str(buf, s);
        }
    }
    fn put_kv_array_u32(buf: &mut Vec<u8>, key: &str, items: &[u32]) {
        put_gguf_str(buf, key);
        put_u32(buf, GgufValueType::Array as u32);
        put_u32(buf, GgufValueType::U32 as u32);
        put_u64(buf, items.len() as u64);
        for v in items {
            put_u32(buf, *v);
        }
    }

    /// Start a GGUF buffer with a correct header prefix. Returns the buf
    /// with magic + version + (tensor_count, kv_count) filled in — the
    /// caller is responsible for actually pushing KVs / tensor entries
    /// that match those counts.
    fn new_gguf(version: u32, tensor_count: u64, kv_count: u64) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC);
        put_u32(&mut buf, version);
        put_u64(&mut buf, tensor_count);
        put_u64(&mut buf, kv_count);
        buf
    }

    // -- Tests --------------------------------------------------------------

    #[test]
    fn magic_mismatch() {
        let mut bytes = vec![0u8; 32];
        bytes[0..4].copy_from_slice(b"XXXX");
        let err = GgufFile::parse_bytes("x", bytes).unwrap_err();
        assert!(
            matches!(err, HaloError::BadMagic { .. }),
            "expected BadMagic, got {err:?}"
        );
    }

    #[test]
    fn version_too_old_rejected() {
        let bytes = new_gguf(2, 0, 0);
        let err = GgufFile::parse_bytes("x", bytes).unwrap_err();
        match err {
            HaloError::UnsupportedVersion { version, .. } => assert_eq!(version, 2),
            other => panic!("expected UnsupportedVersion, got {other:?}"),
        }
        // Error message must mention the version number for humans.
        let bytes = new_gguf(1, 0, 0);
        let err_s = format!("{}", GgufFile::parse_bytes("x", bytes).unwrap_err());
        assert!(
            err_s.contains("version"),
            "error message should mention version: {err_s}"
        );
    }

    #[test]
    fn kv_roundtrip_string_u32_array() {
        // Three KVs: one string, one u32, one array-of-u32.
        let mut buf = new_gguf(3, 0, 3);
        put_kv_string(&mut buf, "general.architecture", "llama");
        put_kv_u32(&mut buf, "llama.block_count", 30);
        put_kv_array_u32(&mut buf, "demo.nums", &[11, 22, 33]);

        let g = GgufFile::parse_bytes("t", buf).unwrap();
        assert_eq!(
            g.kv("general.architecture").and_then(|v| v.as_str()),
            Some("llama")
        );
        assert_eq!(g.kv("llama.block_count").and_then(|v| v.as_u32()), Some(30));
        match g.kv("demo.nums") {
            Some(GgufValue::Array(a)) => {
                assert_eq!(a.elem_type, GgufValueType::U32 as u32);
                assert_eq!(a.values.len(), 3);
                assert_eq!(a.values[0].as_u32(), Some(11));
                assert_eq!(a.values[2].as_u32(), Some(33));
            }
            other => panic!("expected array, got {other:?}"),
        }
    }

    #[test]
    fn one_tensor_entry_parses() {
        // A file with one F32 tensor of shape [2,3], no metadata, no payload.
        let mut buf = new_gguf(3, 1, 0);
        put_gguf_str(&mut buf, "my.tensor");
        put_u32(&mut buf, 2); // n_dims
        put_u64(&mut buf, 2);
        put_u64(&mut buf, 3);
        put_u32(&mut buf, GgufTensorType::F32.as_u32());
        put_u64(&mut buf, 0); // offset=0 into tensor-data region

        // Pad so tensor-data starts at an aligned offset.
        let cur_len = buf.len();
        let pad = (DEFAULT_ALIGNMENT - (cur_len as u64 % DEFAULT_ALIGNMENT)) % DEFAULT_ALIGNMENT;
        buf.extend(std::iter::repeat_n(0, pad as usize));
        // 2*3*f32 = 24 bytes of payload. Just zeros is fine.
        buf.extend(std::iter::repeat_n(0xAA, 6 * 4));

        assert!(buf.len() >= 24, "buffer too tiny: {}", buf.len());
        let g = GgufFile::parse_bytes("t", buf).unwrap();
        assert_eq!(g.tensor_count(), 1);
        let t = g
            .tensor_info("my.tensor")
            .expect("tensor should be indexed");
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.dtype, GgufTensorType::F32);
        assert_eq!(t.size_bytes, Some(24));

        let bytes = g.tensor("my.tensor").expect("slice should exist");
        assert_eq!(bytes.len(), 24);
        assert!(bytes.iter().all(|b| *b == 0xAA));
    }

    #[test]
    fn bitnet_header_extraction() {
        // Build a "BitNet-looking" GGUF: arch=llama, typical BitNet 2B4T dims.
        let kvs: u64 = 13;
        let mut buf = new_gguf(3, 0, kvs);
        put_kv_string(&mut buf, "general.architecture", "llama");
        put_kv_u32(&mut buf, "llama.block_count", 30);
        put_kv_u32(&mut buf, "llama.embedding_length", 2560);
        put_kv_u32(&mut buf, "llama.feed_forward_length", 6912);
        put_kv_u32(&mut buf, "llama.attention.head_count", 20);
        put_kv_u32(&mut buf, "llama.attention.head_count_kv", 5);
        put_kv_f32(&mut buf, "llama.rope.freq_base", 500_000.0);
        put_kv_f32(&mut buf, "llama.attention.layer_norm_rms_epsilon", 1e-5);
        put_kv_string(&mut buf, "tokenizer.ggml.model", "llama");
        put_kv_array_str(&mut buf, "tokenizer.ggml.tokens", &["<s>", "</s>", "a"]);
        put_kv_array_str(&mut buf, "tokenizer.ggml.merges", &["a b"]);
        put_kv_u32(&mut buf, "tokenizer.ggml.bos_token_id", 1);
        put_kv_u32(&mut buf, "tokenizer.ggml.eos_token_id", 2);

        let g = GgufFile::parse_bytes("t", buf).unwrap();
        let h = g.read_bitnet_metadata().unwrap();
        assert_eq!(h.architecture, "llama");
        assert_eq!(h.block_count, 30);
        assert_eq!(h.embedding_length, 2560);
        assert_eq!(h.feed_forward_length, 6912);
        assert_eq!(h.attention_head_count, 20);
        assert_eq!(h.attention_head_count_kv, 5);
        assert!((h.rope_freq_base - 500_000.0).abs() < 1e-3);
        assert!((h.rms_norm_eps - 1e-5).abs() < 1e-9);
        assert_eq!(h.tokenizer_model, "llama");
        assert_eq!(h.tokens.len(), 3);
        assert_eq!(h.tokens[0], "<s>");
        assert_eq!(h.merges, vec!["a b".to_string()]);
        assert_eq!(h.bos_token_id, Some(1));
        assert_eq!(h.eos_token_id, Some(2));
    }

    #[test]
    fn bitnet_header_default_kv_heads_and_rope() {
        // head_count_kv absent → defaults to head_count. rope/eps absent →
        // defaults to DEFAULT_ROPE_THETA / DEFAULT_RMS_NORM_EPS.
        let mut buf = new_gguf(3, 0, 5);
        put_kv_string(&mut buf, "general.architecture", "llama");
        put_kv_u32(&mut buf, "llama.block_count", 2);
        put_kv_u32(&mut buf, "llama.embedding_length", 16);
        put_kv_u32(&mut buf, "llama.feed_forward_length", 32);
        put_kv_u32(&mut buf, "llama.attention.head_count", 4);

        let g = GgufFile::parse_bytes("t", buf).unwrap();
        let h = g.read_bitnet_metadata().unwrap();
        assert_eq!(h.attention_head_count_kv, 4);
        assert_eq!(h.rope_freq_base, crate::types::DEFAULT_ROPE_THETA);
        assert_eq!(h.rms_norm_eps, crate::types::DEFAULT_RMS_NORM_EPS);
    }

    #[test]
    fn tensor_size_math_spot_checks() {
        // F32 [4,8] = 128 bytes
        assert_eq!(tensor_size_bytes(GgufTensorType::F32, &[4, 8]), Some(128));
        // F16 [32] = 64 bytes
        assert_eq!(tensor_size_bytes(GgufTensorType::F16, &[32]), Some(64));
        // Q8_0 [32] = 1 block × 34 bytes
        assert_eq!(tensor_size_bytes(GgufTensorType::Q8_0, &[32]), Some(34));
        // Q4_0 [64] = 2 blocks × 18 bytes = 36
        assert_eq!(tensor_size_bytes(GgufTensorType::Q4_0, &[64]), Some(36));
        // IQ2_S [256] = 1 super-block × (2 + 64 + 16) = 82
        assert_eq!(tensor_size_bytes(GgufTensorType::IQ2_S, &[256]), Some(82));
        // Unknown dtype → None.
        assert_eq!(
            tensor_size_bytes(GgufTensorType::Unknown(9999), &[256]),
            None
        );
        // Non-multiple of block-size → None (caller should see this as
        // "can't slice", not "zero-length tensor").
        assert_eq!(tensor_size_bytes(GgufTensorType::Q8_0, &[31]), None);
    }

    /// Gated integration test: if a real GGUF lives at a known path on
    /// this box, open it and assert it looks like a llama / bitnet model.
    /// `#[ignore]` so it never runs on CI; run manually with `--ignored`.
    #[test]
    #[ignore]
    fn live_open_bitnet_gguf() {
        let candidates = [
            "/home/bcloud/1bit systems/models/bitnet-b1.58-2B-4t.gguf",
            "/home/bcloud/1bit systems/models/bitnet.gguf",
        ];
        for p in candidates {
            if std::path::Path::new(p).exists() {
                let g = GgufFile::open(p).unwrap();
                let h = g.read_bitnet_metadata().unwrap();
                assert!(
                    h.architecture == "llama" || h.architecture == "bitnet",
                    "unexpected arch: {}",
                    h.architecture
                );
                return;
            }
        }
        eprintln!("no BitNet GGUF on this box; skipping live test");
    }
}

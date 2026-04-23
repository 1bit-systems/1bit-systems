//! `gguf-to-h1b` — frame PrismML Bonsai GGUF files (`Q1_0_g128` /
//! `TQ2_0_g128`) into halo-ai's `.h1b` container with the appropriate
//! Bonsai flag bit set.
//!
//! # Scope
//!
//! Bonsai's architecture is Qwen3, which has a different layer layout
//! from halo-ai's native BitNet b1.58 shape (extra `attn_q_norm` +
//! `attn_k_norm`, no `attn_sub_norm`, no split ffn sub-norm). Wiring the
//! Bonsai forward pass into `bitnet_decode` is a **separate** job — this
//! tool is the file-format bridge only. It reads the GGUF tensor
//! directory, validates every `blk.*.{attn_q,attn_k,attn_v,attn_output,
//! ffn_gate,ffn_up,ffn_down}.weight` tensor is a Bonsai ternary dtype
//! (39/40/41/42 — see [`BonsaiDtype`]), and emits a `.h1b` whose ternary
//! tensor payloads are **verbatim byte-for-byte** the GGUF payloads. No
//! repacking, no reordering, no quantization change.
//!
//! The FP32 norm slots in `.h1b` are filled with zeros — the Bonsai
//! runtime (not this tool) must look up the matching `*_norm.weight`
//! tensors in the GGUF directly via name, because they don't map 1:1 to
//! the BitNet per-layer schema. Embeddings + final norm are similarly
//! passed through as placeholder zeros; real wiring comes in the follow-up
//! pass that teaches the loader to special-case `is_bonsai_tq2()`.
//!
//! # Non-goals
//!
//! - No fine-tuning, no re-quantization.
//! - No tokenizer-side work (keep GGUF's tokenizer arrays; downstream can
//!   read them back through the existing `onebit_core::GgufFile` parser).
//! - No HIP kernel wiring.
//!
//! # Why frame it this way
//!
//! Keeping the Bonsai payloads in `.h1b` (not reading raw GGUF at runtime)
//! means the existing `bitnet_decode` loader only grows **one** code path
//! ("is this `.h1b` a Bonsai file?") rather than two ("is this `.gguf` a
//! Bonsai file? is this `.h1b` a Bonsai file?"). That's option (A) from
//! the task spec: extend the `.h1b` format, bypass GGUF at runtime.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use byteorder::{ByteOrder, LittleEndian};

use onebit_core::gguf::{GgufFile, GgufTensorInfo, GgufTensorType};
use onebit_core::h1b::{H1B_FLAG_BONSAI_Q1, H1B_FLAG_BONSAI_TQ2, H1B_MAGIC, H1bWeightFormat};

pub mod htok_export;
pub use htok_export::{HtokExportError, HtokStats, build_htok_from_gguf, export_htok_sidecar};

/// Bonsai-specific GGUF dtype tags. PrismML's converter assigns its own
/// numbers distinct from ggml's canonical TQ1_0 (34) / TQ2_0 (35) — both
/// because the block layout differs (g128 not g256) and because PrismML's
/// fork landed before ggml-quants upstreamed the -g128 variant. See
/// `docs/wiki/Bonsai-Kernel-Spec.md` for the full byte layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BonsaiDtype {
    /// `Q1_0_g128` — 1-bit, 18-byte blocks, 128 weights per block.
    /// Observed in `prism-ml/Bonsai-1.7B-gguf/Bonsai-1.7B-Q1_0.gguf`.
    Q1G128 = 41,
    /// `TQ2_0_g128` — ternary 2-bit, 34-byte blocks, 128 weights per block.
    /// Observed in `prism-ml/Ternary-Bonsai-1.7B-gguf/Ternary-Bonsai-1.7B-Q2_0.gguf`.
    TQ2G128 = 42,
}

impl BonsaiDtype {
    /// Recognize a GGUF tensor dtype tag as a Bonsai packed format.
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            41 => Some(Self::Q1G128),
            42 => Some(Self::TQ2G128),
            _ => None,
        }
    }

    /// Bytes per 128-weight block.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::Q1G128 => 18,
            Self::TQ2G128 => 34,
        }
    }

    /// Which `.h1b` flag bit signals this format.
    pub fn h1b_flag(self) -> i32 {
        match self {
            Self::Q1G128 => H1B_FLAG_BONSAI_Q1,
            Self::TQ2G128 => H1B_FLAG_BONSAI_TQ2,
        }
    }

    /// `H1bWeightFormat` variant with group size filled in.
    pub fn as_h1b_format(self) -> H1bWeightFormat {
        match self {
            Self::Q1G128 => H1bWeightFormat::BonsaiQ1 { group_size: 128 },
            Self::TQ2G128 => H1bWeightFormat::BonsaiTQ2 { group_size: 128 },
        }
    }
}

/// Fixed group size — 128 weights per block for both Bonsai formats.
pub const BONSAI_GROUP_SIZE: usize = 128;

/// Errors surfaced by the converter. Thin wrapper over `anyhow::Error` at
/// the CLI layer; callable APIs use this richer enum so tests can pattern
/// match without string-sniffing.
#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    #[error("failed to open GGUF: {0}")]
    GgufOpen(#[from] onebit_core::HaloError),

    #[error("input is not a Bonsai GGUF: architecture = {0:?}")]
    NotQwen3Architecture(Option<String>),

    #[error(
        "expected every ternary tensor to use the same Bonsai dtype; found \
        {0:?} and {1:?}"
    )]
    MixedBonsaiDtype(BonsaiDtype, BonsaiDtype),

    #[error(
        "tensor {name} has dtype {dtype:?} which is not a Bonsai packed format \
        (expected 41=Q1_0_g128 or 42=TQ2_0_g128)"
    )]
    NotBonsaiDtype { name: String, dtype: GgufTensorType },

    #[error("tensor {name} has {ndim}D shape; expected 2D (rows, cols)")]
    Not2D { name: String, ndim: usize },

    #[error("tensor {name}: cols={cols} is not a multiple of the g128 group size")]
    BadCols { name: String, cols: u64 },

    #[error(
        "tensor {name}: payload size {got} bytes; expected {want} \
        (rows={rows} × cols={cols}, {block}-byte blocks)"
    )]
    BadPayloadSize {
        name: String,
        got: usize,
        want: usize,
        rows: u64,
        cols: u64,
        block: usize,
    },

    #[error("no ternary tensors found in input GGUF — is this really a Bonsai file?")]
    NoTernaryTensors,

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Post-conversion summary. Printed by the CLI, consumed by the integration
/// tests as an authoritative shape probe (saves re-parsing the output file
/// just to assert a layer count).
#[derive(Debug, Clone)]
pub struct ConvertStats {
    pub dtype: BonsaiDtype,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub vocab_size: u32,
    pub context_length: u32,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    /// Total bytes of ternary payload copied through to the `.h1b` output.
    pub ternary_bytes_carried: u64,
    /// Final on-disk size of the output `.h1b`.
    pub output_bytes: u64,
    pub output_path: PathBuf,
    /// Reserved flag word encoded into the `.h1b` header's `cfg[8]`.
    pub h1b_reserved_flags: i32,
}

/// Per-layer ternary tensor names in the GGUF. Matches PrismML's Bonsai
/// GGUF export (`blk.{L}.{name}.weight`).
const PER_LAYER_TERNARY: &[&str] = &[
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
];

/// Read the Bonsai GGUF + write the framed `.h1b`. Returns a summary that
/// the CLI renders to the user.
///
/// Contract: the output file is created fresh (truncated on existing
/// paths), and is only fully valid on `Ok`. On `Err` the output may be
/// partially written — callers that care about atomicity should write to
/// a temp file and rename on success.
pub fn convert_file(input: &Path, output: &Path) -> Result<ConvertStats, ConvertError> {
    let gguf = GgufFile::open(input)?;
    let header = gguf.read_bitnet_metadata()?;
    if header.architecture != "qwen3" {
        return Err(ConvertError::NotQwen3Architecture(Some(
            header.architecture,
        )));
    }
    let n_layers = header.block_count;
    let hs = header.embedding_length;
    let is_ = header.feed_forward_length;
    let nh = header.attention_head_count;
    let nkv = header.attention_head_count_kv;
    // Prefer explicit attention.key_length if present (Bonsai Qwen3 sets it
    // to 128 even though hidden/heads would imply 128 anyway). Defaults
    // gracefully to hidden/heads if the KV dims are irregular.
    let hd = gguf
        .kv("qwen3.attention.key_length")
        .and_then(|v| v.as_u32())
        .unwrap_or_else(|| hs.checked_div(nh).unwrap_or(0));
    let vocab = gguf
        .kv("qwen3.vocab_size")
        .and_then(|v| v.as_u32())
        .unwrap_or(header.tokens.len() as u32);
    let ctx_len = gguf
        .kv("qwen3.context_length")
        .and_then(|v| v.as_u32())
        .unwrap_or(0);

    // Detect whether this GGUF is Q1 or TQ2 by inspecting the first
    // per-layer ternary tensor's dtype. Every other ternary tensor must
    // match (mixed Q1 + TQ2 in one checkpoint is not something PrismML
    // ships, and the downstream dispatcher can't represent it anyway).
    let mut detected: Option<BonsaiDtype> = None;
    let mut ternary_tensors: Vec<(String, &GgufTensorInfo, BonsaiDtype)> = Vec::new();
    for l in 0..n_layers {
        for t in PER_LAYER_TERNARY {
            let name = format!("blk.{}.{}.weight", l, t);
            let info = gguf
                .tensor_info(&name)
                .ok_or_else(|| ConvertError::NotBonsaiDtype {
                    name: name.clone(),
                    dtype: GgufTensorType::Unknown(u32::MAX),
                })?;
            let dtype = BonsaiDtype::from_u32(info.dtype.as_u32()).ok_or_else(|| {
                ConvertError::NotBonsaiDtype {
                    name: name.clone(),
                    dtype: info.dtype,
                }
            })?;
            match detected {
                None => detected = Some(dtype),
                Some(d) if d != dtype => {
                    return Err(ConvertError::MixedBonsaiDtype(d, dtype));
                }
                _ => {}
            }
            if info.shape.len() != 2 {
                return Err(ConvertError::Not2D {
                    name,
                    ndim: info.shape.len(),
                });
            }
            // GGUF convention: shape is (cols, rows) for matmul tensors —
            // outer dim of a `linear` weight is the input (K), inner is
            // the output (N). We treat `shape[0]` as cols (K) and
            // `shape[1]` as rows (N) to match the existing `.h1b`
            // convention (`ternary(rows, cols)`).
            let cols = info.shape[0];
            let _rows = info.shape[1];
            if cols % BONSAI_GROUP_SIZE as u64 != 0 {
                return Err(ConvertError::BadCols { name, cols });
            }
            ternary_tensors.push((name, info, dtype));
        }
    }
    let dtype = detected.ok_or(ConvertError::NoTernaryTensors)?;

    // ── Emit the .h1b ──────────────────────────────────────────────────
    let mut out = Vec::with_capacity(64 * 1024);
    out.extend_from_slice(&H1B_MAGIC);
    let version: i32 = 2; // v2 header carries rope_theta + rms_norm_eps; Bonsai flag overrides format.
    out.extend_from_slice(&version.to_le_bytes());

    // Config (9 × i32).
    let reserved: i32 = dtype.h1b_flag();
    for v in [
        hs as i32,
        is_ as i32,
        n_layers as i32,
        nh as i32,
        nkv as i32,
        vocab as i32,
        ctx_len as i32,
        0i32, // tie_embeddings — unused in Bonsai framing
        reserved,
    ] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    // v2 extras.
    out.extend_from_slice(&header.rope_freq_base.to_le_bytes());
    out.extend_from_slice(&header.rms_norm_eps.to_le_bytes());

    // Model-level tensors (embedding + final norm) — filled with zeros in
    // this framing pass. Real wiring reads them out of the GGUF at load
    // time via the `GgufFile` pass-through the loader will grow. Keeping
    // zeros here means we don't inflate the `.h1b` with a huge
    // embedding duplicate, AND the downstream sanity check "embedding
    // size == vocab × hidden" still passes.
    let embedding_bytes = (vocab as u64) * (hs as u64) * 4;
    let final_norm_bytes = (hs as u64) * 4;
    let mut file = File::create(output)?;
    file.write_all(&out)?;
    let mut output_bytes: u64 = out.len() as u64;
    output_bytes += write_zeros(&mut file, embedding_bytes)?;
    output_bytes += write_zeros(&mut file, final_norm_bytes)?;

    // Per-layer tensors.
    //
    //   [input_norm hs][post_attn_norm hs][attn_sub_norm hs × 4 slots]
    //   [ffn_sub_norm_hs × 2 slots][ffn_sub_norm is]
    //   (q, k, v, o, gate, up, down — packed + (maybe) scales)
    //
    // Norm bytes are filled with zeros (Bonsai Qwen3 has different norm
    // layout; the runtime loader will look up the real norms by name).
    let fmt = dtype.as_h1b_format();
    let fmt_inline_scales = fmt.has_inline_block_scales();
    let mut ternary_bytes_carried: u64 = 0;
    for l in 0..n_layers {
        output_bytes += write_zeros(&mut file, (hs as u64) * 4)?; // input_norm
        output_bytes += write_zeros(&mut file, (hs as u64) * 4)?; // post_attn_norm
        for _ in 0..4 {
            output_bytes += write_zeros(&mut file, (hs as u64) * 4)?; // attn_sub_norm × 4
        }
        for _ in 0..2 {
            output_bytes += write_zeros(&mut file, (hs as u64) * 4)?; // trunc ffn slots
        }
        output_bytes += write_zeros(&mut file, (is_ as u64) * 4)?; // ffn_sub_norm

        // Seven ternary tensors in the canonical .h1b order.
        let q_shape = ternary_shape(&gguf, l, "attn_q")?;
        let k_shape = ternary_shape(&gguf, l, "attn_k")?;
        let v_shape = ternary_shape(&gguf, l, "attn_v")?;
        let o_shape = ternary_shape(&gguf, l, "attn_output")?;
        let gate_shape = ternary_shape(&gguf, l, "ffn_gate")?;
        let up_shape = ternary_shape(&gguf, l, "ffn_up")?;
        let down_shape = ternary_shape(&gguf, l, "ffn_down")?;

        for (name_tail, (rows, cols)) in [
            ("attn_q", q_shape),
            ("attn_k", k_shape),
            ("attn_v", v_shape),
            ("attn_output", o_shape),
            ("ffn_gate", gate_shape),
            ("ffn_up", up_shape),
            ("ffn_down", down_shape),
        ] {
            let name = format!("blk.{}.{}.weight", l, name_tail);
            // Onebit-core's `tensor()` returns None for Bonsai-specific
            // dtypes (41/42) because it can't compute their size — the
            // parser folds them into `GgufTensorType::Unknown`. We
            // compute the payload size ourselves (rows × blocks × 18|34)
            // and slice the mmap directly.
            let info = gguf
                .tensor_info(&name)
                .ok_or_else(|| ConvertError::NotBonsaiDtype {
                    name: name.clone(),
                    dtype: GgufTensorType::Unknown(u32::MAX),
                })?;
            let expected = rows
                .checked_mul(cols / BONSAI_GROUP_SIZE as u64)
                .and_then(|b| b.checked_mul(dtype.block_bytes() as u64))
                .ok_or_else(|| ConvertError::BadPayloadSize {
                    name: name.clone(),
                    got: 0,
                    want: 0,
                    rows,
                    cols,
                    block: dtype.block_bytes(),
                })?;
            let abs = gguf.tensor_data_start().saturating_add(info.offset) as usize;
            let end =
                abs.checked_add(expected as usize)
                    .ok_or_else(|| ConvertError::BadPayloadSize {
                        name: name.clone(),
                        got: 0,
                        want: expected as usize,
                        rows,
                        cols,
                        block: dtype.block_bytes(),
                    })?;
            let buf = gguf.bytes();
            if end > buf.len() {
                return Err(ConvertError::BadPayloadSize {
                    name,
                    got: buf.len().saturating_sub(abs),
                    want: expected as usize,
                    rows,
                    cols,
                    block: dtype.block_bytes(),
                });
            }
            let payload = &buf[abs..end];
            file.write_all(payload)?;
            output_bytes += payload.len() as u64;
            ternary_bytes_carried += payload.len() as u64;
            if !fmt_inline_scales {
                // Halo/Sherry/TQ1 formats follow each packed tensor with
                // `[rows] f32` scales. Bonsai formats embed scales inline
                // and emit zero trailing scale bytes — so this arm is
                // unreachable for today's code path, but we keep the
                // structural mirror of `h1b::serialize` for clarity.
                output_bytes += write_zeros(&mut file, rows * 4)?;
            }
        }
    }

    file.flush()?;
    file.sync_all()?;
    drop(file);

    Ok(ConvertStats {
        dtype,
        hidden_size: hs,
        intermediate_size: is_,
        num_layers: n_layers,
        num_heads: nh,
        num_kv_heads: nkv,
        head_dim: hd,
        vocab_size: vocab,
        context_length: ctx_len,
        rope_theta: header.rope_freq_base,
        rms_norm_eps: header.rms_norm_eps,
        ternary_bytes_carried,
        output_bytes,
        output_path: output.to_path_buf(),
        h1b_reserved_flags: reserved,
    })
}

fn ternary_shape(gguf: &GgufFile, layer: u32, name_tail: &str) -> Result<(u64, u64), ConvertError> {
    let name = format!("blk.{}.{}.weight", layer, name_tail);
    let info = gguf
        .tensor_info(&name)
        .ok_or_else(|| ConvertError::NotBonsaiDtype {
            name: name.clone(),
            dtype: GgufTensorType::Unknown(u32::MAX),
        })?;
    if info.shape.len() != 2 {
        return Err(ConvertError::Not2D {
            name,
            ndim: info.shape.len(),
        });
    }
    // GGUF stores `shape = [cols, rows]` for matmul weights.
    let cols = info.shape[0];
    let rows = info.shape[1];
    Ok((rows, cols))
}

/// Stream zero bytes to the output file in 64 KiB chunks. Avoids allocating
/// a giant zero buffer for the embedding tensor (vocab × hidden × 4 can
/// reach 1+ GiB).
fn write_zeros(file: &mut File, count: u64) -> std::io::Result<u64> {
    if count == 0 {
        return Ok(0);
    }
    let chunk = [0u8; 64 * 1024];
    let mut remaining = count;
    while remaining > 0 {
        let n = remaining.min(chunk.len() as u64) as usize;
        file.write_all(&chunk[..n])?;
        remaining -= n as u64;
    }
    Ok(count)
}

/// Cheap standalone check — read only the first 4 bytes of the file and
/// confirm the magic. Used by callers that want to quick-reject a
/// non-GGUF without paying for the full parser.
pub fn is_gguf_magic(path: &Path) -> Result<bool, ConvertError> {
    use std::io::Read;
    let mut f = File::open(path)?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    Ok(&magic == b"GGUF")
}

/// Expose the GGUF version this file claims (byte offset 4..8). Mostly for
/// diagnostics; the converter itself delegates the version check to
/// `onebit_core::GgufFile::parse`.
pub fn read_gguf_version(path: &Path) -> Result<u32, ConvertError> {
    use std::io::Read;
    let mut f = File::open(path)?;
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    if &buf[0..4] != b"GGUF" {
        return Ok(0);
    }
    Ok(LittleEndian::read_u32(&buf[4..8]))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Magic-check helper rejects non-GGUF files without touching the full
    /// parser.
    #[test]
    fn gguf_magic_check_rejects_non_gguf() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("not-gguf.bin");
        std::fs::write(&path, b"GGML\x03\x00\x00\x00").unwrap();
        assert!(!is_gguf_magic(&path).unwrap(), "GGML is not GGUF");

        let path2 = td.path().join("looks-gguf.bin");
        std::fs::write(&path2, b"GGUF\x03\x00\x00\x00").unwrap();
        assert!(is_gguf_magic(&path2).unwrap());
        assert_eq!(read_gguf_version(&path2).unwrap(), 3);
    }

    /// Bonsai dtype tags 41 / 42 are the only recognized Bonsai dtypes,
    /// and their block-size math lines up with the `H1bWeightFormat`
    /// row_bytes contract exposed by onebit-core.
    #[test]
    fn bonsai_dtype_tag_and_row_bytes_roundtrip() {
        let q1 = BonsaiDtype::from_u32(41).unwrap();
        assert_eq!(q1, BonsaiDtype::Q1G128);
        assert_eq!(q1.block_bytes(), 18);
        assert_eq!(q1.h1b_flag(), H1B_FLAG_BONSAI_Q1);
        assert_eq!(
            q1.as_h1b_format().row_bytes(128).unwrap(),
            18,
            "row_bytes(128) must match one 18-byte block"
        );
        assert_eq!(q1.as_h1b_format().row_bytes(2048).unwrap(), 16 * 18);

        let tq2 = BonsaiDtype::from_u32(42).unwrap();
        assert_eq!(tq2, BonsaiDtype::TQ2G128);
        assert_eq!(tq2.block_bytes(), 34);
        assert_eq!(tq2.h1b_flag(), H1B_FLAG_BONSAI_TQ2);
        assert_eq!(tq2.as_h1b_format().row_bytes(128).unwrap(), 34);
        assert_eq!(tq2.as_h1b_format().row_bytes(2048).unwrap(), 16 * 34);

        // Unknown tags reject.
        assert!(BonsaiDtype::from_u32(35).is_none()); // canonical ggml TQ2_0 (g256)
        assert!(BonsaiDtype::from_u32(0).is_none());
    }

    /// Build a minimal single-tensor GGUF in memory, round-trip it through
    /// `GgufFile::open` via a temp file, and confirm the tensor payload
    /// we get back byte-matches what we wrote. This is the load-bearing
    /// "can we read any tensor" test — if this breaks, nothing else in
    /// the converter will work.
    ///
    /// We pick a 2-element FP32 tensor (trivial dtype, known size) rather
    /// than a Bonsai dtype because onebit-core's parser has zero size
    /// accounting for dtype tag 42 (Bonsai-specific). Building a fully
    /// Bonsai-flavoured GGUF here would re-implement PrismML's exporter
    /// and exceed the scaffold's scope.
    #[test]
    fn gguf_parse_single_fp32_tensor_round_trip() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("tiny.gguf");
        std::fs::write(&path, build_tiny_gguf()).unwrap();

        let g = GgufFile::open(&path).expect("tiny GGUF must parse");
        assert_eq!(g.version(), 3);
        let info = g.tensor_info("w").expect("tensor w must exist");
        assert_eq!(info.shape, vec![2]);
        assert_eq!(info.dtype.as_u32(), 0); // F32
        let payload = g.tensor("w").expect("tensor payload");
        let mut v = [0f32; 2];
        v[0] = f32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
        v[1] = f32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]);
        assert_eq!(v, [1.5f32, -2.25f32]);
    }

    /// Build a tiny GGUF v3 in memory with one F32 tensor and one arch KV.
    /// Only the path actually exercised by `gguf_parse_single_fp32_tensor_round_trip`.
    fn build_tiny_gguf() -> Vec<u8> {
        fn put_str(out: &mut Vec<u8>, s: &str) {
            out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        b.extend_from_slice(&1u64.to_le_bytes()); // kv_count
        // KV: general.architecture = "tiny"
        put_str(&mut b, "general.architecture");
        b.extend_from_slice(&8u32.to_le_bytes()); // string type
        put_str(&mut b, "tiny");
        // Tensor info: name "w", 1D [2], dtype F32 (0), offset 0.
        put_str(&mut b, "w");
        b.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        b.extend_from_slice(&2u64.to_le_bytes()); // shape[0]
        b.extend_from_slice(&0u32.to_le_bytes()); // dtype F32
        b.extend_from_slice(&0u64.to_le_bytes()); // offset (from data start)
        // Pad to 32-byte alignment (default GGUF alignment).
        while b.len() % 32 != 0 {
            b.push(0);
        }
        // Payload: 2 × FP32.
        b.extend_from_slice(&1.5f32.to_le_bytes());
        b.extend_from_slice((-2.25f32).to_le_bytes().as_ref());
        b
    }
}

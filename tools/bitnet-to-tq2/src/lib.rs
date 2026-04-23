//! `bitnet-to-tq2` — repack Microsoft's `microsoft/bitnet-b1.58-2B-4T-bf16`
//! master weights into a `.h1b` **v4** file using PrismML's
//! `TQ2_0_g128` block layout.
//!
//! # What this tool does
//!
//! 1. Memory-maps the HF safetensors archive produced by
//!    `hf download microsoft/bitnet-b1.58-2B-4T-bf16`.
//! 2. Walks the 7-per-layer ternary linear tensors
//!    (`self_attn.{q,k,v,o}_proj.weight`, `mlp.{gate,up,down}_proj.weight`),
//!    decodes the bf16 `{-1, 0, +1} × per-tensor scale` master weights, and
//!    requantizes each K-axis group of 128 weights into a 34-byte
//!    `TQ2_0_g128` block (32 B of 2-bit codes + fp16 block scale AFTER the
//!    codes — see `docs/wiki/Bonsai-Kernel-Spec.md` §TQ2\_0\_g128).
//! 3. Emits a `.h1b` v4 container with the `H1B_FLAG_BONSAI_TQ2 = 0x8` bit
//!    set in `H1bConfig::reserved`. fp16/fp32 norm + embedding tensors are
//!    passed through as raw fp32 blobs (no quantization).
//!
//! # Quantization rule
//!
//! BitNet b1.58 was trained with `absmean` as the *per-tensor* scale
//! (see MS `modeling_bitnet.py`:
//! `scale = 1.0 / w.abs().mean().clamp(min=1e-5); w_q = (w *
//! scale).round().clamp(-1, 1)`). The TQ2\_0\_g128 block layout on the
//! other hand stores one fp16 scale **per 128-weight block**. Two scale
//! policies are supported:
//!
//! * **Per-tensor scale** (default, `--per-tensor-scale`) — compute one
//!   `s_tensor = mean(|w|)` over ALL weights in the tensor and store
//!   `d = s_tensor` into EVERY block's fp16 slot. Matches native BitNet
//!   training scale exactly. Wastes 2 B × N\_blocks but preserves ~0.5
//!   bits of scale precision that per-block absmean throws away on a
//!   noisy 128-lane absmean estimate. Output is coherent English.
//!
//! * **Per-block scale** (opt-in, `--per-block-scale`) — compute
//!   `d = mean(|w_block|)` independently per 128-weight block. This is
//!   the TQ2\_0\_g128 native policy, useful for A/B testing. On
//!   MS-BitNet-2B-4T's bf16 master the per-block absmean collapses to a
//!   noisy approximation of the per-tensor scale and the output degrades
//!   to "the the the..." degeneracy; see the 2026-04-20 writeup in
//!   `project_bitnet_live_bench.md`.
//!
//! Codes are sign-extracted identically in both modes:
//!   `code[i] = 0b10 if w_i >= t ; 0b00 if w_i <= -t ; 0b01 otherwise`,
//! where `t = 0.5 * d`. Threshold computed from whichever `d` is used.
//!
//! # Arch mismatch warning
//!
//! MS BitNet-b1.58-2B-4T is a **BitNet** architecture: squared-ReLU GLU
//! feed-forward (`relu2`), `attn_sub_norm`, `ffn_sub_norm`, tied
//! embeddings, 128 000-token LLaMA-3 vocab. The current `bitnet_decode`
//! Bonsai dispatch branch hard-codes **Qwen3** (RMS-only, SwiGLU). The
//! `.h1b` flag we set here (`H1B_FLAG_BONSAI_TQ2`) only signals
//! "Bonsai-format weights" — it does NOT (and cannot, at this layer)
//! toggle between BitNet and Qwen3 forward passes. Wiring a new dispatch
//! branch that says "BonsaiTQ2 weights + BitNet arch" is a follow-up
//! lane.
//!
//! See `tools/bitnet-to-tq2/README.md` for the full note.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use byteorder::{ByteOrder, LittleEndian};
use half::{bf16, f16};
use safetensors::SafeTensors;
use safetensors::tensor::{Dtype, TensorView};

use onebit_core::h1b::{H1B_FLAG_BONSAI_TQ2, H1B_MAGIC};

// ---------------------------------------------------------------------------
// Public constants
// ---------------------------------------------------------------------------

/// TQ2\_0\_g128 fixed group size — 128 weights per block.
pub const TQ2_GROUP_SIZE: usize = 128;

/// TQ2\_0\_g128 fixed block size — 32 B of codes + 2 B of fp16 scale.
pub const TQ2_BLOCK_BYTES: usize = 34;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    #[error("failed to open input safetensors: {0}")]
    Io(#[from] std::io::Error),

    #[error("safetensors parse error: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),

    #[error("tensor {name}: K-axis dim {cols} not a multiple of TQ2 group size ({group})")]
    GroupMisaligned {
        name: String,
        cols: usize,
        group: usize,
    },

    #[error("tensor {name}: expected dtype {expected:?}, found {found:?}")]
    DtypeMismatch {
        name: String,
        expected: &'static str,
        found: Dtype,
    },

    #[error("tensor {name}: expected shape {expected:?}, found {found:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        found: Vec<usize>,
    },

    #[error("tensor {name}: not found in safetensors archive")]
    MissingTensor { name: String },

    #[error("input dir missing expected file: {path}")]
    MissingFile { path: PathBuf },

    #[error("malformed config.json: {0}")]
    BadConfig(String),
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Scalar model config we need to emit into the `.h1b` header. Populated
/// from the HF `config.json` (which lives next to `model.safetensors`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelConfig {
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub vocab_size: u32,
    pub max_position_embeddings: u32,
    pub tie_word_embeddings: bool,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
}

impl ModelConfig {
    /// Parse the minimal set of fields we need from HF `config.json`.
    pub fn parse_json(bytes: &[u8]) -> Result<Self, ConvertError> {
        let v: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|e| ConvertError::BadConfig(format!("json: {e}")))?;
        fn req_u32(v: &serde_json::Value, k: &str) -> Result<u32, ConvertError> {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as u32)
                .ok_or_else(|| ConvertError::BadConfig(format!("missing {k}")))
        }
        fn req_f32(v: &serde_json::Value, k: &str) -> Result<f32, ConvertError> {
            v.get(k)
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .ok_or_else(|| ConvertError::BadConfig(format!("missing {k}")))
        }
        Ok(ModelConfig {
            hidden_size: req_u32(&v, "hidden_size")?,
            intermediate_size: req_u32(&v, "intermediate_size")?,
            num_hidden_layers: req_u32(&v, "num_hidden_layers")?,
            num_attention_heads: req_u32(&v, "num_attention_heads")?,
            num_key_value_heads: req_u32(&v, "num_key_value_heads")?,
            vocab_size: req_u32(&v, "vocab_size")?,
            max_position_embeddings: req_u32(&v, "max_position_embeddings")?,
            tie_word_embeddings: v
                .get("tie_word_embeddings")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            rope_theta: req_f32(&v, "rope_theta")?,
            rms_norm_eps: req_f32(&v, "rms_norm_eps").unwrap_or(1e-5),
        })
    }
}

// ---------------------------------------------------------------------------
// Quantization core — one 128-weight group → one 34-byte TQ2 block
// ---------------------------------------------------------------------------

/// Which scale policy to use when repacking a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleMode {
    /// One `s_tensor = mean(|w|)` over ALL weights in the tensor, then
    /// duplicated into every block's fp16 slot. Matches BitNet b1.58's
    /// native per-tensor training scale. Default as of 2026-04-20 — the
    /// per-block absmean fix revealed that the 128-lane noise eats enough
    /// scale precision to push MS-BitNet-2B-4T output into "the the the…"
    /// degeneracy.
    PerTensor,
    /// Independent `d = mean(|w_block|)` per 128-weight block — the TQ2
    /// native policy. Retained for A/B testing; known-degenerate on
    /// MS-BitNet weights.
    PerBlock,
}

/// Quantize a 128-element fp32 group into a `TQ2_0_g128` block using a
/// caller-supplied scale `d_override`. Used by the per-tensor path.
///
/// Returns the 34-byte block: `[fp16 d : 2 B][32 B codes LSB-first, 4/byte]`.
/// Code map: `00→-d, 01→0, 10→+d, 11→0 (unused)`.
///
/// Layout note: d-FIRST, not d-last — matches the `bonsai_tq2_gemv.hip`
/// kernel's `kScaleOffset = 0` / `kCodesOffset = 2`.
pub fn quantize_group_with_scale(
    weights: &[f32; TQ2_GROUP_SIZE],
    d_override: f32,
) -> [u8; TQ2_BLOCK_BYTES] {
    let mut block = [0u8; TQ2_BLOCK_BYTES];

    if d_override == 0.0 {
        // All-zero scale → force decoder-zero path (code 0b01 per lane).
        for b in &mut block[2..] {
            *b = 0x55;
        }
        return block;
    }

    let threshold = 0.5 * d_override;
    for (j, &w) in weights.iter().enumerate() {
        let code: u8 = if w >= threshold {
            0b10 // +d
        } else if w <= -threshold {
            0b00 // -d
        } else {
            0b01 //  0
        };
        let byte_idx = 2 + j / 4;
        let shift = (j % 4) * 2;
        block[byte_idx] |= code << shift;
    }
    let d = f16::from_f32(d_override);
    let d_bits = d.to_bits();
    block[0] = (d_bits & 0xff) as u8;
    block[1] = (d_bits >> 8) as u8;
    block
}

/// Quantize a 128-element fp32 group using the per-block absmean policy.
/// Computes `d = mean(|w|)` from the group itself.
pub fn quantize_group_tq2(weights: &[f32; TQ2_GROUP_SIZE]) -> [u8; TQ2_BLOCK_BYTES] {
    let absmean = weights.iter().copied().map(|x| x.abs()).sum::<f32>() / TQ2_GROUP_SIZE as f32;
    quantize_group_with_scale(weights, absmean)
}

/// Inverse of [`quantize_group_tq2`] — used by round-trip tests. Produces
/// one 128-element fp32 group from a 34-byte block.
pub fn dequantize_group_tq2(block: &[u8; TQ2_BLOCK_BYTES]) -> [f32; TQ2_GROUP_SIZE] {
    let mut out = [0.0f32; TQ2_GROUP_SIZE];
    let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
    for (j, slot) in out.iter_mut().enumerate() {
        let byte_idx = 2 + j / 4;
        let shift = (j % 4) * 2;
        let code = (block[byte_idx] >> shift) & 0b11;
        *slot = match code {
            0b00 => -d,
            0b10 => d,
            _ => 0.0,
        };
    }
    out
}

/// Quantize a full row (length must be a multiple of 128) into a
/// `(K/128) * 34`-byte packed buffer using the per-block absmean policy.
/// Provided for compatibility; prefer [`quantize_row_with_scale`] when you
/// already know the per-tensor scale.
pub fn quantize_row_tq2(row: &[f32]) -> Result<Vec<u8>, ConvertError> {
    if row.len() % TQ2_GROUP_SIZE != 0 {
        return Err(ConvertError::GroupMisaligned {
            name: "<row>".into(),
            cols: row.len(),
            group: TQ2_GROUP_SIZE,
        });
    }
    let n_blocks = row.len() / TQ2_GROUP_SIZE;
    let mut out = Vec::with_capacity(n_blocks * TQ2_BLOCK_BYTES);
    let mut scratch = [0.0f32; TQ2_GROUP_SIZE];
    for b in 0..n_blocks {
        let base = b * TQ2_GROUP_SIZE;
        scratch.copy_from_slice(&row[base..base + TQ2_GROUP_SIZE]);
        out.extend_from_slice(&quantize_group_tq2(&scratch));
    }
    Ok(out)
}

/// Quantize a full row using a caller-supplied scale `d` duplicated into
/// every block. This is the per-tensor-scale path used by the default
/// `convert` mode.
pub fn quantize_row_with_scale(row: &[f32], d: f32) -> Result<Vec<u8>, ConvertError> {
    if row.len() % TQ2_GROUP_SIZE != 0 {
        return Err(ConvertError::GroupMisaligned {
            name: "<row>".into(),
            cols: row.len(),
            group: TQ2_GROUP_SIZE,
        });
    }
    let n_blocks = row.len() / TQ2_GROUP_SIZE;
    let mut out = Vec::with_capacity(n_blocks * TQ2_BLOCK_BYTES);
    let mut scratch = [0.0f32; TQ2_GROUP_SIZE];
    for b in 0..n_blocks {
        let base = b * TQ2_GROUP_SIZE;
        scratch.copy_from_slice(&row[base..base + TQ2_GROUP_SIZE]);
        out.extend_from_slice(&quantize_group_with_scale(&scratch, d));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// bf16 → fp32 helpers
// ---------------------------------------------------------------------------

/// Decode a bf16 tensor view (little-endian raw bytes) into a fp32 `Vec`.
/// Safetensors stores bf16 as two little-endian bytes per element.
fn bf16_view_to_f32(view: &TensorView<'_>, name: &str) -> Result<Vec<f32>, ConvertError> {
    match view.dtype() {
        Dtype::BF16 => {}
        other => {
            return Err(ConvertError::DtypeMismatch {
                name: name.into(),
                expected: "BF16",
                found: other,
            });
        }
    }
    let raw = view.data();
    let n = raw.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let lo = raw[2 * i];
        let hi = raw[2 * i + 1];
        let bits = u16::from_le_bytes([lo, hi]);
        out.push(bf16::from_bits(bits).to_f32());
    }
    Ok(out)
}

/// Decode a bf16 tensor view into a fp32 `Vec` of raw-little-endian bytes
/// (4 B/elem). This mirrors what `.h1b`'s norm/embedding slots expect.
fn bf16_view_to_f32_bytes(view: &TensorView<'_>, name: &str) -> Result<Vec<u8>, ConvertError> {
    let fp32 = bf16_view_to_f32(view, name)?;
    let mut bytes = Vec::with_capacity(fp32.len() * 4);
    for x in fp32 {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    Ok(bytes)
}

// ---------------------------------------------------------------------------
// Tensor-name mapping — HF BitNet → `.h1b` layer-tensor order
// ---------------------------------------------------------------------------

/// The seven ternary tensors per layer, in the canonical `.h1b` order.
const PER_LAYER_TERNARY: &[(&str, &str)] = &[
    ("self_attn.q_proj", "attn_q"),
    ("self_attn.k_proj", "attn_k"),
    ("self_attn.v_proj", "attn_v"),
    ("self_attn.o_proj", "attn_output"),
    ("mlp.gate_proj", "ffn_gate"),
    ("mlp.up_proj", "ffn_up"),
    ("mlp.down_proj", "ffn_down"),
];

// ---------------------------------------------------------------------------
// Top-level converter
// ---------------------------------------------------------------------------

/// Per-tensor outcome, reported in [`ConvertStats`].
#[derive(Debug, Clone)]
pub struct TensorReport {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub block_count: usize,
    pub packed_bytes: usize,
}

/// Post-conversion summary. Printed by the CLI.
#[derive(Debug, Clone)]
pub struct ConvertStats {
    pub config: ModelConfig,
    pub per_layer: Vec<Vec<TensorReport>>,
    pub embedding_bytes: u64,
    pub final_norm_bytes: u64,
    pub packed_ternary_bytes: u64,
    pub output_bytes: u64,
    pub output_path: PathBuf,
    pub h1b_reserved_flags: i32,
    /// Tensors we passed through as raw fp32 (norms / embeddings).
    pub fp32_passthrough_names: Vec<String>,
    /// Tensors we noticed in the input that didn't fit any expected slot.
    pub unmatched_tensors: Vec<String>,
}

/// Read a HF BitNet-bf16 checkpoint directory and write a `.h1b` v4 file
/// with the Bonsai TQ2 flag set, using the default per-tensor scale
/// policy. Thin wrapper over [`convert_with_mode`].
pub fn convert(input_dir: &Path, output_path: &Path) -> Result<ConvertStats, ConvertError> {
    convert_with_mode(input_dir, output_path, ScaleMode::PerTensor)
}

/// Full converter — explicit scale-mode selection. See [`ScaleMode`].
pub fn convert_with_mode(
    input_dir: &Path,
    output_path: &Path,
    scale_mode: ScaleMode,
) -> Result<ConvertStats, ConvertError> {
    // -- Load config.json -------------------------------------------------
    let cfg_path = input_dir.join("config.json");
    if !cfg_path.exists() {
        return Err(ConvertError::MissingFile { path: cfg_path });
    }
    let cfg_bytes = std::fs::read(&cfg_path)?;
    let cfg = ModelConfig::parse_json(&cfg_bytes)?;

    // -- mmap safetensors -------------------------------------------------
    let st_path = input_dir.join("model.safetensors");
    if !st_path.exists() {
        return Err(ConvertError::MissingFile { path: st_path });
    }
    let file = File::open(&st_path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;

    // -- Header -----------------------------------------------------------
    let reserved: i32 = H1B_FLAG_BONSAI_TQ2;
    let version: i32 = 4;
    let head_dim = (cfg.hidden_size / cfg.num_attention_heads) as usize;
    let tie_embeddings_i32: i32 = if cfg.tie_word_embeddings { 1 } else { 0 };

    let mut out_file = File::create(output_path)?;
    let mut out_bytes: u64 = 0;

    // Magic + version.
    out_file.write_all(&H1B_MAGIC)?;
    out_file.write_all(&version.to_le_bytes())?;
    out_bytes += 4 + 4;
    // cfg[0..9].
    for v in [
        cfg.hidden_size as i32,
        cfg.intermediate_size as i32,
        cfg.num_hidden_layers as i32,
        cfg.num_attention_heads as i32,
        cfg.num_key_value_heads as i32,
        cfg.vocab_size as i32,
        cfg.max_position_embeddings as i32,
        tie_embeddings_i32,
        reserved,
    ] {
        out_file.write_all(&v.to_le_bytes())?;
        out_bytes += 4;
    }
    // v2 extras: rope_theta, rms_norm_eps.
    out_file.write_all(&cfg.rope_theta.to_le_bytes())?;
    out_file.write_all(&cfg.rms_norm_eps.to_le_bytes())?;
    out_bytes += 8;

    // -- Model-level tensors ---------------------------------------------
    //
    //   [embedding: fp32, vocab × hs]
    //   [final_norm: fp32, hs]
    //
    // We decode bf16 → fp32 and write it raw. These are small relative to
    // the ternary payload so we don't bother mmapping-through.
    let mut fp32_passthrough_names: Vec<String> = Vec::new();
    let embedding_name = "model.embed_tokens.weight";
    let embedding_view = st
        .tensor(embedding_name)
        .map_err(|_| ConvertError::MissingTensor {
            name: embedding_name.into(),
        })?;
    let embedding_bytes = bf16_view_to_f32_bytes(&embedding_view, embedding_name)?;
    expect_shape_2d(
        &embedding_view,
        embedding_name,
        cfg.vocab_size as usize,
        cfg.hidden_size as usize,
    )?;
    out_file.write_all(&embedding_bytes)?;
    out_bytes += embedding_bytes.len() as u64;
    fp32_passthrough_names.push(embedding_name.into());

    let final_norm_name = "model.norm.weight";
    let final_norm_view = st
        .tensor(final_norm_name)
        .map_err(|_| ConvertError::MissingTensor {
            name: final_norm_name.into(),
        })?;
    let final_norm_bytes = bf16_view_to_f32_bytes(&final_norm_view, final_norm_name)?;
    expect_shape_1d(&final_norm_view, final_norm_name, cfg.hidden_size as usize)?;
    out_file.write_all(&final_norm_bytes)?;
    out_bytes += final_norm_bytes.len() as u64;
    fp32_passthrough_names.push(final_norm_name.into());

    // -- Per-layer payloads -----------------------------------------------
    let mut packed_ternary_bytes: u64 = 0;
    let mut per_layer_reports: Vec<Vec<TensorReport>> =
        Vec::with_capacity(cfg.num_hidden_layers as usize);
    for l in 0..cfg.num_hidden_layers {
        let input_norm = pass_through_fp32_1d(
            &st,
            &format!("model.layers.{l}.input_layernorm.weight"),
            cfg.hidden_size as usize,
            &mut fp32_passthrough_names,
        )?;
        let post_attn_norm = pass_through_fp32_1d(
            &st,
            &format!("model.layers.{l}.post_attention_layernorm.weight"),
            cfg.hidden_size as usize,
            &mut fp32_passthrough_names,
        )?;
        let attn_sub_norm = pass_through_fp32_1d(
            &st,
            &format!("model.layers.{l}.self_attn.attn_sub_norm.weight"),
            cfg.hidden_size as usize,
            &mut fp32_passthrough_names,
        )?;
        let ffn_sub_norm = pass_through_fp32_1d(
            &st,
            &format!("model.layers.{l}.mlp.ffn_sub_norm.weight"),
            cfg.intermediate_size as usize,
            &mut fp32_passthrough_names,
        )?;

        // Norm layout — mirrors `onebit_core::h1b::serialize` exactly so a
        // future dispatcher can re-use the existing offset math.
        //   [input_norm hs][post_attn_norm hs][attn_sub_norm hs × 4 slots]
        //   [trunc_ffn_sub hs × 2 slots][ffn_sub_norm is]
        out_file.write_all(&input_norm)?;
        out_file.write_all(&post_attn_norm)?;
        for _ in 0..4 {
            out_file.write_all(&attn_sub_norm)?;
        }
        let trunc = &ffn_sub_norm[..(cfg.hidden_size as usize) * 4];
        out_file.write_all(trunc)?;
        out_file.write_all(trunc)?;
        out_file.write_all(&ffn_sub_norm)?;
        out_bytes += (cfg.hidden_size as u64) * 4 // input
            + (cfg.hidden_size as u64) * 4 // post_attn
            + (cfg.hidden_size as u64) * 4 * 4 // attn_sub × 4
            + (cfg.hidden_size as u64) * 4 * 2 // trunc ffn × 2
            + (cfg.intermediate_size as u64) * 4; // ffn_sub

        // Seven ternary tensors.
        let nh = cfg.num_attention_heads as usize;
        let nkv = cfg.num_key_value_heads as usize;
        let hs = cfg.hidden_size as usize;
        let is_ = cfg.intermediate_size as usize;
        let shapes: [(&str, &str, usize, usize); 7] = [
            ("self_attn.q_proj", "attn_q", nh * head_dim, hs),
            ("self_attn.k_proj", "attn_k", nkv * head_dim, hs),
            ("self_attn.v_proj", "attn_v", nkv * head_dim, hs),
            ("self_attn.o_proj", "attn_output", hs, nh * head_dim),
            ("mlp.gate_proj", "ffn_gate", is_, hs),
            ("mlp.up_proj", "ffn_up", is_, hs),
            ("mlp.down_proj", "ffn_down", hs, is_),
        ];
        assert_eq!(shapes.len(), PER_LAYER_TERNARY.len());
        let mut layer_reports: Vec<TensorReport> = Vec::with_capacity(7);
        for (hf_infix, _h1b_tail, rows, cols) in shapes {
            let name = format!("model.layers.{l}.{hf_infix}.weight");
            let view = st
                .tensor(&name)
                .map_err(|_| ConvertError::MissingTensor { name: name.clone() })?;
            // HF bf16 master: shape = [rows, cols] bf16. Ternary values are
            // exactly `{-1, 0, +1} × per-tensor α` (pre-multiplied — the
            // `.weight_scale` tensor is absent from the -bf16 variant).
            expect_shape_2d(&view, &name, rows, cols)?;
            if cols % TQ2_GROUP_SIZE != 0 {
                return Err(ConvertError::GroupMisaligned {
                    name,
                    cols,
                    group: TQ2_GROUP_SIZE,
                });
            }
            let fp32 = bf16_view_to_f32(&view, &name)?;
            debug_assert_eq!(fp32.len(), rows * cols);

            // Per-tensor scale (if applicable): mean(|w|) over the WHOLE
            // tensor, computed once and reused in every block. Matches the
            // BitNet b1.58 training scale exactly.
            let s_tensor = match scale_mode {
                ScaleMode::PerTensor => {
                    let sum: f64 = fp32.iter().map(|&x| x.abs() as f64).sum();
                    if fp32.is_empty() {
                        0.0f32
                    } else {
                        (sum / fp32.len() as f64) as f32
                    }
                }
                ScaleMode::PerBlock => 0.0, // unused in the PerBlock arm below
            };

            // Per-row quantization. 34 B × (cols/128) per row.
            let n_blocks_per_row = cols / TQ2_GROUP_SIZE;
            let row_packed = n_blocks_per_row * TQ2_BLOCK_BYTES;
            let mut packed = Vec::with_capacity(rows * row_packed);
            for r in 0..rows {
                let row = &fp32[r * cols..(r + 1) * cols];
                for b in 0..n_blocks_per_row {
                    let start = b * TQ2_GROUP_SIZE;
                    let group: &[f32; TQ2_GROUP_SIZE] = row[start..start + TQ2_GROUP_SIZE]
                        .try_into()
                        .expect("group slice matches TQ2_GROUP_SIZE");
                    let block = match scale_mode {
                        ScaleMode::PerTensor => quantize_group_with_scale(group, s_tensor),
                        ScaleMode::PerBlock => quantize_group_tq2(group),
                    };
                    packed.extend_from_slice(&block);
                }
            }
            out_file.write_all(&packed)?;
            out_bytes += packed.len() as u64;
            packed_ternary_bytes += packed.len() as u64;
            layer_reports.push(TensorReport {
                name: name.clone(),
                rows,
                cols,
                block_count: rows * n_blocks_per_row,
                packed_bytes: packed.len(),
            });
        }
        per_layer_reports.push(layer_reports);
    }

    out_file.flush()?;
    out_file.sync_all()?;
    drop(out_file);

    // Discover any tensors we didn't consume (diagnostic only).
    let mut consumed: std::collections::HashSet<&str> = std::collections::HashSet::new();
    consumed.insert("model.embed_tokens.weight");
    consumed.insert("model.norm.weight");
    // Plus every norm + ternary name we walked.
    for n in &fp32_passthrough_names {
        consumed.insert(n.as_str());
    }
    let mut ternary_names: Vec<String> = Vec::new();
    for l in 0..cfg.num_hidden_layers {
        for (hf_infix, _) in PER_LAYER_TERNARY {
            ternary_names.push(format!("model.layers.{l}.{hf_infix}.weight"));
        }
    }
    for n in &ternary_names {
        consumed.insert(n.as_str());
    }
    let mut unmatched: Vec<String> = Vec::new();
    for name in st.names() {
        // `name` is `&str` from safetensors::SafeTensors::names().
        if !consumed.contains(name) {
            // lm_head is tied in BitNet; surface it for visibility but don't
            // treat it as an error.
            unmatched.push(name.to_string());
        }
    }
    unmatched.sort();

    Ok(ConvertStats {
        config: cfg,
        per_layer: per_layer_reports,
        embedding_bytes: embedding_bytes.len() as u64,
        final_norm_bytes: final_norm_bytes.len() as u64,
        packed_ternary_bytes,
        output_bytes: out_bytes,
        output_path: output_path.to_path_buf(),
        h1b_reserved_flags: reserved,
        fp32_passthrough_names,
        unmatched_tensors: unmatched,
    })
}

fn pass_through_fp32_1d(
    st: &SafeTensors<'_>,
    name: &str,
    expected_len: usize,
    recorder: &mut Vec<String>,
) -> Result<Vec<u8>, ConvertError> {
    let view = st
        .tensor(name)
        .map_err(|_| ConvertError::MissingTensor { name: name.into() })?;
    expect_shape_1d(&view, name, expected_len)?;
    let bytes = bf16_view_to_f32_bytes(&view, name)?;
    recorder.push(name.to_string());
    Ok(bytes)
}

fn expect_shape_1d(view: &TensorView<'_>, name: &str, expected: usize) -> Result<(), ConvertError> {
    if view.shape() != [expected] {
        return Err(ConvertError::ShapeMismatch {
            name: name.into(),
            expected: vec![expected],
            found: view.shape().to_vec(),
        });
    }
    Ok(())
}

fn expect_shape_2d(
    view: &TensorView<'_>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<(), ConvertError> {
    if view.shape() != [rows, cols] {
        return Err(ConvertError::ShapeMismatch {
            name: name.into(),
            expected: vec![rows, cols],
            found: view.shape().to_vec(),
        });
    }
    Ok(())
}

/// Read back the header flag word from a `.h1b` we just wrote, for use in
/// integration tests that want to assert the Bonsai TQ2 bit is set without
/// re-parsing the whole file through `onebit_core`.
pub fn read_reserved_flags(path: &Path) -> Result<i32, ConvertError> {
    use std::io::Read;
    let mut f = File::open(path)?;
    let mut buf = [0u8; 48];
    f.read_exact(&mut buf)?;
    if buf[0..4] != H1B_MAGIC {
        return Err(ConvertError::BadConfig("not a .h1b file".into()));
    }
    // magic(4) + version(4) + cfg[0..9] as i32 → reserved is cfg[8].
    let reserved_offset = 4 + 4 + 8 * 4;
    Ok(LittleEndian::read_i32(
        &buf[reserved_offset..reserved_offset + 4],
    ))
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Synth a 128-weight group of `{-1, 0, +1} × 0.3`, repack through the
    /// TQ2 quantizer, unpack via the scalar reference decoder, and assert
    /// (a) every non-zero input preserves its sign in the output and
    /// (b) the stored scale `d` matches the BitNet absmean formula.
    ///
    /// On a ternary `{-0.3, 0, +0.3}` pattern with 1/3 of each level:
    ///   absmean = (43 × 0.3 + 43 × 0.3 + 42 × 0) / 128 ≈ 0.2015625
    /// so `d = absmean ≈ 0.2`, not 0.3. Round-trip reconstructs values at
    /// `±d` — the magnitude shrinks, but signs and zero positions match.
    #[test]
    fn quantize_group_absmean_roundtrip() {
        // Build a deterministic pattern spanning all three ternary levels.
        let mut w = [0.0f32; 128];
        for (i, cell) in w.iter_mut().enumerate() {
            *cell = match i % 3 {
                0 => 0.3,
                1 => -0.3,
                _ => 0.0,
            };
        }
        let block = quantize_group_tq2(&w);
        // Scale d = absmean = (86 × 0.3) / 128 ≈ 0.2015625, within one fp16
        // ULP. Layout is d-first (bytes 0..2), codes-second (bytes 2..34)
        // — matches the PrismML GGUF and `bonsai_tq2_gemv.hip` kernel.
        let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let absmean_expected: f32 = w.iter().map(|x| x.abs()).sum::<f32>() / 128.0;
        assert!(
            (d - absmean_expected).abs() < 1e-3,
            "d={d}, expected ~{absmean_expected}"
        );

        let out = dequantize_group_tq2(&block);

        // Sign-preservation: every lane that was non-zero on input must
        // still be non-zero with the same sign on output (magnitude
        // shrinks to ±d since d < |w|; that's the expected absmean
        // rounding, not an error).
        for i in 0..128 {
            if w[i] > 0.0 {
                assert!(out[i] > 0.0, "lane {i} lost + sign");
            } else if w[i] < 0.0 {
                assert!(out[i] < 0.0, "lane {i} lost - sign");
            } else {
                assert_eq!(out[i], 0.0, "lane {i} should be zero");
            }
        }

        // Every non-zero output must equal exactly ±d (within fp16 round).
        for i in 0..128 {
            if w[i] != 0.0 {
                assert!(
                    (out[i].abs() - d).abs() < 1e-3,
                    "lane {i}: out.abs()={} expected d={d}",
                    out[i].abs()
                );
            }
        }
    }

    /// Confirm the `H1B_FLAG_BONSAI_TQ2` bit is set in the `reserved` cfg
    /// slot of the emitted header, other reserved bits are zero, and the
    /// version field is exactly 4. We build a minimal in-memory safetensors
    /// blob with a single layer's worth of tensors and round-trip it.
    #[test]
    fn header_flag_composition() {
        let td = tempfile::tempdir().unwrap();
        let input_dir = td.path().join("in");
        std::fs::create_dir_all(&input_dir).unwrap();

        // 1-layer BitNet-ish config — shapes trimmed for a fast test.
        let cfg = r#"{
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "vocab_size": 32,
            "max_position_embeddings": 64,
            "tie_word_embeddings": true,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5
        }"#;
        std::fs::write(input_dir.join("config.json"), cfg).unwrap();
        build_tiny_safetensors(&input_dir.join("model.safetensors"));

        let out_path = td.path().join("out.h1b");
        let stats = convert(&input_dir, &out_path).expect("convert must succeed");
        assert_eq!(stats.h1b_reserved_flags, H1B_FLAG_BONSAI_TQ2);
        let read_flags = read_reserved_flags(&out_path).unwrap();
        assert_eq!(
            read_flags, H1B_FLAG_BONSAI_TQ2,
            "disk flag != in-memory flag"
        );

        // version field sits at bytes [4..8].
        let disk = std::fs::read(&out_path).unwrap();
        let version = i32::from_le_bytes(disk[4..8].try_into().unwrap());
        assert_eq!(version, 4, "version must be 4");

        // Other bits in `reserved` are zero.
        let nontq2 = read_flags & !H1B_FLAG_BONSAI_TQ2;
        assert_eq!(nontq2, 0, "unexpected non-TQ2 reserved bits: {nontq2:#x}");

        // Packed ternary payload must be non-empty.
        assert!(stats.packed_ternary_bytes > 0);
    }

    /// Validates `clap` parsing for the two required args. We invoke the
    /// same parser the CLI uses so regressions here surface before `main`.
    #[test]
    fn cli_arg_parsing() {
        use clap::Parser;
        let args = crate::cli::Args::try_parse_from([
            "bitnet-to-tq2",
            "--in",
            "/tmp/in",
            "--out",
            "/tmp/out.h1b",
        ])
        .expect("valid args should parse");
        assert_eq!(args.input.to_str().unwrap(), "/tmp/in");
        assert_eq!(args.output.to_str().unwrap(), "/tmp/out.h1b");
        // Default scale mode is per-tensor (the bug fix default).
        assert_eq!(args.scale_mode(), ScaleMode::PerTensor);

        // Explicit --per-block-scale flips to PerBlock.
        let pb = crate::cli::Args::try_parse_from([
            "bitnet-to-tq2",
            "--in",
            "/tmp/in",
            "--out",
            "/tmp/out.h1b",
            "--per-block-scale",
        ])
        .expect("valid args should parse");
        assert_eq!(pb.scale_mode(), ScaleMode::PerBlock);

        // --per-tensor-scale + --per-block-scale → mutually exclusive.
        let err = crate::cli::Args::try_parse_from([
            "bitnet-to-tq2",
            "--in",
            "/tmp/in",
            "--out",
            "/tmp/out.h1b",
            "--per-tensor-scale",
            "--per-block-scale",
        ])
        .unwrap_err();
        assert!(
            err.to_string().contains("cannot be used with") || err.to_string().contains("conflict")
        );

        // Missing --in → error.
        let err =
            crate::cli::Args::try_parse_from(["bitnet-to-tq2", "--out", "/tmp/x"]).unwrap_err();
        assert!(err.to_string().contains("--in") || err.to_string().contains("in"));
    }

    /// With `ScaleMode::PerTensor`, every block across every ternary
    /// tensor must share the SAME fp16 `d` value (the per-tensor
    /// absmean). Per-block mode must produce at least some variation.
    #[test]
    fn per_tensor_scale_duplicated_across_blocks() {
        let td = tempfile::tempdir().unwrap();
        let input_dir = td.path().join("in");
        std::fs::create_dir_all(&input_dir).unwrap();

        // 1-layer config with ≥2 blocks per row so we can assert
        // "all blocks equal".
        let cfg = r#"{
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "vocab_size": 32,
            "max_position_embeddings": 64,
            "tie_word_embeddings": true,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5
        }"#;
        std::fs::write(input_dir.join("config.json"), cfg).unwrap();
        build_tiny_safetensors_256(&input_dir.join("model.safetensors"));

        let out_pt = td.path().join("pt.h1b");
        let stats_pt =
            convert_with_mode(&input_dir, &out_pt, ScaleMode::PerTensor).expect("pt convert");
        let out_pb = td.path().join("pb.h1b");
        let stats_pb =
            convert_with_mode(&input_dir, &out_pb, ScaleMode::PerBlock).expect("pb convert");

        // File sizes must match exactly — layout is identical, only the
        // block `d` values change.
        assert_eq!(
            std::fs::metadata(&out_pt).unwrap().len(),
            std::fs::metadata(&out_pb).unwrap().len(),
            "per-tensor vs per-block file sizes must match"
        );
        assert_eq!(stats_pt.packed_ternary_bytes, stats_pb.packed_ternary_bytes);

        // Extract every block `d` from the first ternary tensor of layer 0
        // (q_proj) in both files. Easier to work off the ConvertStats +
        // known header size than to reparse — just read both full files
        // and scan for TQ2 blocks in the ternary payload region.
        let pt_bytes = std::fs::read(&out_pt).unwrap();
        let pb_bytes = std::fs::read(&out_pb).unwrap();

        // Header = 8 (magic+version) + 9*4 (cfg) + 8 (rope+eps) = 52 B.
        let header = 52usize;
        // Embedding fp32: vocab × hidden_size = 32 × 256 × 4 = 32 768 B.
        let embed = 32 * 256 * 4;
        // Final norm fp32: 256 × 4 = 1 024 B.
        let final_norm = 256 * 4;
        // Layer norms (per the writer): input(256*4) + post_attn(256*4)
        //   + attn_sub×4(256*4*4) + trunc_ffn×2(256*4*2) + ffn_sub(512*4)
        let norms = 256 * 4 + 256 * 4 + 256 * 4 * 4 + 256 * 4 * 2 + 512 * 4;
        let ternary_start = header + embed + final_norm + norms;

        // q_proj: rows = nh*head_dim = 4 * (256/4) = 256, cols = 256.
        let cols = 256usize;
        let n_blocks_per_row = cols / TQ2_GROUP_SIZE; // = 2
        let q_blocks = 256 * n_blocks_per_row; // 512 blocks

        let mut pt_ds: Vec<u16> = Vec::with_capacity(q_blocks);
        let mut pb_ds: Vec<u16> = Vec::with_capacity(q_blocks);
        for b in 0..q_blocks {
            let off = ternary_start + b * TQ2_BLOCK_BYTES;
            pt_ds.push(u16::from_le_bytes([pt_bytes[off], pt_bytes[off + 1]]));
            pb_ds.push(u16::from_le_bytes([pb_bytes[off], pb_bytes[off + 1]]));
        }

        // Per-tensor: all block scales identical.
        let first = pt_ds[0];
        for (i, &d) in pt_ds.iter().enumerate() {
            assert_eq!(
                d, first,
                "per-tensor block {i}: d={d:#x} != first={first:#x}"
            );
        }
        // And that single value must equal fp16(mean(|w|)) of the whole
        // tensor. Our synthesized q_proj is ternary × 0.3 so the
        // per-tensor absmean is ~0.2 (2/3 non-zero × 0.3).
        let d_f32 = f16::from_bits(first).to_f32();
        assert!(
            d_f32 > 0.15 && d_f32 < 0.25,
            "per-tensor d out of expected range: {d_f32}"
        );

        // Per-block: at least two distinct values (noise across blocks).
        let any_diff = pb_ds.iter().any(|&d| d != pb_ds[0]);
        assert!(
            any_diff,
            "per-block scales were all identical — seeded RNG may be degenerate"
        );
    }

    /// 256-hidden variant of the fixture so we get ≥2 blocks per row.
    fn build_tiny_safetensors_256(path: &Path) {
        use safetensors::tensor::{Dtype, TensorView};
        let hs: usize = 256;
        let is_: usize = 512;
        let nh: usize = 4;
        let nkv: usize = 1;
        let vocab: usize = 32;
        let head_dim = hs / nh;
        fn fill_bf16(n: usize, v: f32) -> Vec<u8> {
            let bits = bf16::from_f32(v).to_bits().to_le_bytes();
            let mut out = Vec::with_capacity(n * 2);
            for _ in 0..n {
                out.extend_from_slice(&bits);
            }
            out
        }
        let embed = fill_bf16(vocab * hs, 0.0);
        let hs_ones = fill_bf16(hs, 1.0);
        let is_ones = fill_bf16(is_, 1.0);
        fn ternary_tensor(rows: usize, cols: usize, seed: u32) -> Vec<u8> {
            let mut out = Vec::with_capacity(rows * cols * 2);
            let mut s = seed;
            for _ in 0..(rows * cols) {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let v = match s % 3 {
                    0 => 0.3f32,
                    1 => -0.3f32,
                    _ => 0.0f32,
                };
                out.extend_from_slice(&bf16::from_f32(v).to_bits().to_le_bytes());
            }
            out
        }
        let q_rows = nh * head_dim;
        let kv_rows = nkv * head_dim;
        let q = ternary_tensor(q_rows, hs, 1);
        let k = ternary_tensor(kv_rows, hs, 2);
        let v = ternary_tensor(kv_rows, hs, 3);
        let o = ternary_tensor(hs, nh * head_dim, 4);
        let gate = ternary_tensor(is_, hs, 5);
        let up = ternary_tensor(is_, hs, 6);
        let down = ternary_tensor(hs, is_, 7);
        let tensors: Vec<(String, TensorView)> = vec![
            (
                "model.embed_tokens.weight".into(),
                TensorView::new(Dtype::BF16, vec![vocab, hs], &embed).unwrap(),
            ),
            (
                "model.norm.weight".into(),
                TensorView::new(Dtype::BF16, vec![hs], &hs_ones).unwrap(),
            ),
            (
                "model.layers.0.input_layernorm.weight".into(),
                TensorView::new(Dtype::BF16, vec![hs], &hs_ones).unwrap(),
            ),
            (
                "model.layers.0.post_attention_layernorm.weight".into(),
                TensorView::new(Dtype::BF16, vec![hs], &hs_ones).unwrap(),
            ),
            (
                "model.layers.0.self_attn.attn_sub_norm.weight".into(),
                TensorView::new(Dtype::BF16, vec![hs], &hs_ones).unwrap(),
            ),
            (
                "model.layers.0.mlp.ffn_sub_norm.weight".into(),
                TensorView::new(Dtype::BF16, vec![is_], &is_ones).unwrap(),
            ),
            (
                "model.layers.0.self_attn.q_proj.weight".into(),
                TensorView::new(Dtype::BF16, vec![q_rows, hs], &q).unwrap(),
            ),
            (
                "model.layers.0.self_attn.k_proj.weight".into(),
                TensorView::new(Dtype::BF16, vec![kv_rows, hs], &k).unwrap(),
            ),
            (
                "model.layers.0.self_attn.v_proj.weight".into(),
                TensorView::new(Dtype::BF16, vec![kv_rows, hs], &v).unwrap(),
            ),
            (
                "model.layers.0.self_attn.o_proj.weight".into(),
                TensorView::new(Dtype::BF16, vec![hs, nh * head_dim], &o).unwrap(),
            ),
            (
                "model.layers.0.mlp.gate_proj.weight".into(),
                TensorView::new(Dtype::BF16, vec![is_, hs], &gate).unwrap(),
            ),
            (
                "model.layers.0.mlp.up_proj.weight".into(),
                TensorView::new(Dtype::BF16, vec![is_, hs], &up).unwrap(),
            ),
            (
                "model.layers.0.mlp.down_proj.weight".into(),
                TensorView::new(Dtype::BF16, vec![hs, is_], &down).unwrap(),
            ),
        ];
        safetensors::serialize_to_file(tensors.iter().map(|(n, v)| (n.as_str(), v)), None, path)
            .expect("serialize tiny safetensors");
    }

    // ---- helpers ---------------------------------------------------------

    /// Build a minimal safetensors file with the tensor names the converter
    /// needs: embedding, norms, final norm, one layer × seven proj. bf16
    /// everywhere. Values are `{-1, 0, +1}` signed ternary for the proj.
    /// tensors (scaled by 0.3) and uniform 1.0 for norms.
    fn build_tiny_safetensors(path: &Path) {
        use safetensors::tensor::{Dtype, TensorView};
        let hs: usize = 128;
        let is_: usize = 256;
        let nh: usize = 4;
        let nkv: usize = 1;
        let vocab: usize = 32;
        let head_dim = hs / nh;
        // Allocate the raw bf16 buffers up-front so the TensorView borrows
        // stay live through serialize. serialize_to_file is safer than
        // hand-rolling the header.
        fn fill_bf16(n: usize, v: f32) -> Vec<u8> {
            let bits = bf16::from_f32(v).to_bits().to_le_bytes();
            let mut out = Vec::with_capacity(n * 2);
            for _ in 0..n {
                out.extend_from_slice(&bits);
            }
            out
        }
        let embed = fill_bf16(vocab * hs, 0.0);
        let hs_ones = fill_bf16(hs, 1.0);
        let is_ones = fill_bf16(is_, 1.0);

        // Signed ternary pattern at 0.3 per weight.
        fn ternary_tensor(rows: usize, cols: usize, seed: u32) -> Vec<u8> {
            let mut out = Vec::with_capacity(rows * cols * 2);
            let mut s = seed;
            for _ in 0..(rows * cols) {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let v = match s % 3 {
                    0 => 0.3f32,
                    1 => -0.3f32,
                    _ => 0.0f32,
                };
                out.extend_from_slice(&bf16::from_f32(v).to_bits().to_le_bytes());
            }
            out
        }

        let q_rows = nh * head_dim;
        let kv_rows = nkv * head_dim;
        let q = ternary_tensor(q_rows, hs, 1);
        let k = ternary_tensor(kv_rows, hs, 2);
        let v = ternary_tensor(kv_rows, hs, 3);
        let o = ternary_tensor(hs, nh * head_dim, 4);
        let gate = ternary_tensor(is_, hs, 5);
        let up = ternary_tensor(is_, hs, 6);
        let down = ternary_tensor(hs, is_, 7);

        let tensors: Vec<(String, TensorView)> = vec![
            (
                "model.embed_tokens.weight".into(),
                TensorView::new(Dtype::BF16, vec![vocab, hs], &embed).unwrap(),
            ),
            (
                "model.norm.weight".into(),
                TensorView::new(Dtype::BF16, vec![hs], &hs_ones).unwrap(),
            ),
            (
                "model.layers.0.input_layernorm.weight".into(),
                TensorView::new(Dtype::BF16, vec![hs], &hs_ones).unwrap(),
            ),
            (
                "model.layers.0.post_attention_layernorm.weight".into(),
                TensorView::new(Dtype::BF16, vec![hs], &hs_ones).unwrap(),
            ),
            (
                "model.layers.0.self_attn.attn_sub_norm.weight".into(),
                TensorView::new(Dtype::BF16, vec![hs], &hs_ones).unwrap(),
            ),
            (
                "model.layers.0.mlp.ffn_sub_norm.weight".into(),
                TensorView::new(Dtype::BF16, vec![is_], &is_ones).unwrap(),
            ),
            (
                "model.layers.0.self_attn.q_proj.weight".into(),
                TensorView::new(Dtype::BF16, vec![q_rows, hs], &q).unwrap(),
            ),
            (
                "model.layers.0.self_attn.k_proj.weight".into(),
                TensorView::new(Dtype::BF16, vec![kv_rows, hs], &k).unwrap(),
            ),
            (
                "model.layers.0.self_attn.v_proj.weight".into(),
                TensorView::new(Dtype::BF16, vec![kv_rows, hs], &v).unwrap(),
            ),
            (
                "model.layers.0.self_attn.o_proj.weight".into(),
                TensorView::new(Dtype::BF16, vec![hs, nh * head_dim], &o).unwrap(),
            ),
            (
                "model.layers.0.mlp.gate_proj.weight".into(),
                TensorView::new(Dtype::BF16, vec![is_, hs], &gate).unwrap(),
            ),
            (
                "model.layers.0.mlp.up_proj.weight".into(),
                TensorView::new(Dtype::BF16, vec![is_, hs], &up).unwrap(),
            ),
            (
                "model.layers.0.mlp.down_proj.weight".into(),
                TensorView::new(Dtype::BF16, vec![hs, is_], &down).unwrap(),
            ),
        ];
        safetensors::serialize_to_file(tensors.iter().map(|(n, v)| (n.as_str(), v)), None, path)
            .expect("serialize tiny safetensors");
    }
}

/// CLI argument definitions, shared between `main.rs` and the cli-arg
/// parsing test in this crate.
pub mod cli {
    use super::ScaleMode;
    use clap::Parser;
    use std::path::PathBuf;

    #[derive(Parser, Debug)]
    #[command(
        name = "bitnet-to-tq2",
        version,
        about = "Repack microsoft/bitnet-b1.58-2B-4T-bf16 → .h1b v4 TQ2_0_g128",
        long_about = None
    )]
    pub struct Args {
        /// Input directory holding `model.safetensors` + `config.json`
        /// from the HF -bf16 variant.
        #[arg(long = "in", value_name = "DIR")]
        pub input: PathBuf,

        /// Output `.h1b` file. Overwrites if it exists.
        #[arg(long = "out", value_name = "FILE")]
        pub output: PathBuf,

        /// Use per-tensor absmean for the block `d` (BitNet training
        /// scale, duplicated into every block's fp16 slot). Default ON.
        #[arg(
            long = "per-tensor-scale",
            default_value_t = true,
            conflicts_with = "per_block_scale"
        )]
        pub per_tensor_scale: bool,

        /// Use per-128-block absmean for the block `d` (TQ2 native
        /// policy). Opt-in for A/B testing — known to produce degenerate
        /// output on MS-BitNet-2B-4T.
        #[arg(long = "per-block-scale", default_value_t = false)]
        pub per_block_scale: bool,
    }

    impl Args {
        /// Resolve the CLI flags to a concrete [`ScaleMode`]. `clap`
        /// already enforces mutual exclusion via `conflicts_with`.
        pub fn scale_mode(&self) -> ScaleMode {
            if self.per_block_scale {
                ScaleMode::PerBlock
            } else {
                ScaleMode::PerTensor
            }
        }
    }
}

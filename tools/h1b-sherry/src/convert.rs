//! File-level converter: `.h1b` (TQ1 v4) -> `.h1b` (Sherry v3, fp16 flag set).

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use onebit_core::h1b::{
    H1B_FLAG_HADAMARD_ROTATED, H1B_FLAG_SHERRY_FP16, H1bConfig, H1bFile, H1bLayerOffsets,
    H1bWeightFormat, LayerTensors, ModelTensors, Span, TernaryTensor, serialize,
};

use crate::sherry_pack::{PackRowStats, pack_sherry_row, sherry_row_bytes};
use crate::tq1_unpack::{tq1_row_bytes, unpack_tq1_row};

/// Aggregate counters returned by [`convert_file`]. Mostly used by CLI
/// output and by integration tests to assert on the structured-sparsity
/// enforcement.
#[derive(Debug, Default, Clone, Copy)]
pub struct ConvertStats {
    pub groups_total: u64,
    pub forced_zero_flips: u64,
    pub rows_total: u64,
    pub layers_processed: u32,
    pub hadamard_preserved: bool,
}

impl ConvertStats {
    pub fn flip_fraction(&self) -> f64 {
        if self.groups_total == 0 {
            0.0
        } else {
            self.forced_zero_flips as f64 / self.groups_total as f64
        }
    }
}

/// Read `in_path`, requant every ternary tensor from TQ1 v4 -> Sherry v3,
/// write the result to `out_path` with `H1B_FLAG_SHERRY_FP16` set in the
/// `reserved` cfg slot. Preserves `H1B_FLAG_HADAMARD_ROTATED` if it was
/// set upstream.
pub fn convert_file(in_path: &Path, out_path: &Path) -> Result<ConvertStats> {
    let src =
        H1bFile::open(in_path).with_context(|| format!("opening input {}", in_path.display()))?;

    let cfg = *src.config();
    let fmt = cfg
        .weight_format()
        .map_err(|e| anyhow!("input weight format: {e}"))?;
    if fmt != H1bWeightFormat::TQ1V4 {
        bail!(
            "h1b-sherry expects a v4 (TQ1) input file; got format {:?} (version {}, reserved 0x{:x})",
            fmt,
            cfg.version,
            cfg.reserved,
        );
    }

    // Build the output config: v3 + SHERRY_FP16, preserving HADAMARD bit.
    let preserved = cfg.reserved & H1B_FLAG_HADAMARD_ROTATED;
    let out_cfg = H1bConfig {
        version: 3,
        reserved: H1B_FLAG_SHERRY_FP16 | preserved,
        ..cfg
    };

    // We need owned byte buffers for every tensor the output serializer
    // consumes. Norms + scales copy verbatim; packed weights are repacked.
    let hs = cfg.hidden_size as usize;
    let is_ = cfg.intermediate_size as usize;
    let nh = cfg.num_heads as usize;
    let nkv = cfg.num_kv_heads as usize;
    let hd = cfg.head_dim().map_err(|e| anyhow!("head_dim: {e}"))? as usize;
    let vocab = cfg.vocab_size as usize;
    let n_layers = cfg.num_layers as usize;

    // Model-level tensors — fp32, copy verbatim.
    let m = *src.model_offsets();
    let embedding_owned = src.tensor_bytes(m.embedding).to_vec();
    let final_norm_owned = src.tensor_bytes(m.final_norm).to_vec();
    if embedding_owned.len() != vocab * hs * 4 {
        bail!("embedding size mismatch");
    }

    // Per-layer conversion.
    let mut layer_blobs: Vec<LayerOwned> = Vec::with_capacity(n_layers);
    let mut stats = ConvertStats {
        hadamard_preserved: preserved != 0,
        layers_processed: n_layers as u32,
        ..Default::default()
    };

    for (li, lo) in src.layer_offsets().iter().enumerate() {
        let owned = build_layer_owned(&src, lo, hs, is_, nh, nkv, hd, &mut stats)
            .with_context(|| format!("layer {li}"))?;
        layer_blobs.push(owned);
    }

    // Build serializer-facing views.
    let model_view = ModelTensors {
        embedding_fp32: &embedding_owned,
        final_norm_fp32: &final_norm_owned,
    };
    let layer_views: Vec<LayerTensors<'_>> = layer_blobs.iter().map(LayerOwned::as_view).collect();

    let bytes =
        serialize(&out_cfg, &model_view, &layer_views).map_err(|e| anyhow!("serialize: {e}"))?;

    std::fs::write(out_path, &bytes).with_context(|| format!("writing {}", out_path.display()))?;
    Ok(stats)
}

struct LayerOwned {
    input_norm: Vec<u8>,
    post_attn_norm: Vec<u8>,
    attn_sub_norm: Vec<u8>,
    ffn_sub_norm: Vec<u8>,
    q_packed: Vec<u8>,
    q_scales: Vec<u8>,
    k_packed: Vec<u8>,
    k_scales: Vec<u8>,
    v_packed: Vec<u8>,
    v_scales: Vec<u8>,
    o_packed: Vec<u8>,
    o_scales: Vec<u8>,
    gate_packed: Vec<u8>,
    gate_scales: Vec<u8>,
    up_packed: Vec<u8>,
    up_scales: Vec<u8>,
    down_packed: Vec<u8>,
    down_scales: Vec<u8>,
}

impl LayerOwned {
    fn as_view(&self) -> LayerTensors<'_> {
        LayerTensors {
            input_norm_fp32: &self.input_norm,
            post_attn_norm_fp32: &self.post_attn_norm,
            attn_sub_norm_fp32: &self.attn_sub_norm,
            ffn_sub_norm_fp32: &self.ffn_sub_norm,
            q: TernaryTensor {
                packed: &self.q_packed,
                scales: &self.q_scales,
            },
            k: TernaryTensor {
                packed: &self.k_packed,
                scales: &self.k_scales,
            },
            v: TernaryTensor {
                packed: &self.v_packed,
                scales: &self.v_scales,
            },
            o: TernaryTensor {
                packed: &self.o_packed,
                scales: &self.o_scales,
            },
            gate: TernaryTensor {
                packed: &self.gate_packed,
                scales: &self.gate_scales,
            },
            up: TernaryTensor {
                packed: &self.up_packed,
                scales: &self.up_scales,
            },
            down: TernaryTensor {
                packed: &self.down_packed,
                scales: &self.down_scales,
            },
        }
    }
}

// TODO(gap-p2): fold `(hs, is_, nh, nkv, hd)` into a `LayerDims` struct; the
// current 8-arg signature predates the shape struct. Not in scope for the
// clippy gate flip — every call site would need updating.
#[allow(clippy::too_many_arguments)]
fn build_layer_owned(
    src: &H1bFile,
    lo: &H1bLayerOffsets,
    hs: usize,
    is_: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    stats: &mut ConvertStats,
) -> Result<LayerOwned> {
    let norm = |s: Span| src.tensor_bytes(s).to_vec();

    // Each ternary tensor: (rows, cols). Sherry expects cols % 32 == 0.
    let tensors: [(&'static str, Span, Span, usize, usize); 7] = [
        ("q", lo.q_packed, lo.q_scales, nh * hd, hs),
        ("k", lo.k_packed, lo.k_scales, nkv * hd, hs),
        ("v", lo.v_packed, lo.v_scales, nkv * hd, hs),
        ("o", lo.o_packed, lo.o_scales, hs, nh * hd),
        ("gate", lo.gate_packed, lo.gate_scales, is_, hs),
        ("up", lo.up_packed, lo.up_scales, is_, hs),
        ("down", lo.down_packed, lo.down_scales, hs, is_),
    ];

    let mut packed_slots: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(7);
    for (name, packed_span, scales_span, rows, cols) in tensors {
        if !cols.is_multiple_of(32) {
            bail!("tensor {name}: cols={cols} not divisible by 32 (Sherry requirement)",);
        }
        let tq1_src = src.tensor_bytes(packed_span);
        let row_in_bytes = tq1_row_bytes(cols);
        if tq1_src.len() != rows * row_in_bytes {
            bail!(
                "tensor {name}: TQ1 packed bytes mismatch (got {}, expected {rows}*{row_in_bytes})",
                tq1_src.len(),
            );
        }

        let row_out_bytes = sherry_row_bytes(cols);
        let mut sherry_dst = vec![0u8; rows * row_out_bytes];
        let mut ternary_buf = vec![0i8; cols];

        for r in 0..rows {
            let in_start = r * row_in_bytes;
            let out_start = r * row_out_bytes;
            unpack_tq1_row(
                &tq1_src[in_start..in_start + row_in_bytes],
                &mut ternary_buf,
                cols,
            );
            let row_stats: PackRowStats = pack_sherry_row(
                &ternary_buf,
                &mut sherry_dst[out_start..out_start + row_out_bytes],
                cols,
            );
            stats.groups_total += (cols / 4) as u64;
            stats.forced_zero_flips += row_stats.forced_zero_flips as u64;
            stats.rows_total += 1;
        }

        let scales = src.tensor_bytes(scales_span).to_vec();
        if scales.len() != rows * 4 {
            bail!("tensor {name}: scales bytes mismatch");
        }
        packed_slots.push((sherry_dst, scales));
    }

    let mut iter = packed_slots.into_iter();
    let mut next = || iter.next().expect("7 slots populated");
    let (q_packed, q_scales) = next();
    let (k_packed, k_scales) = next();
    let (v_packed, v_scales) = next();
    let (o_packed, o_scales) = next();
    let (gate_packed, gate_scales) = next();
    let (up_packed, up_scales) = next();
    let (down_packed, down_scales) = next();

    Ok(LayerOwned {
        input_norm: norm(lo.input_norm),
        post_attn_norm: norm(lo.post_attn_norm),
        attn_sub_norm: norm(lo.attn_sub_norm),
        ffn_sub_norm: norm(lo.ffn_sub_norm),
        q_packed,
        q_scales,
        k_packed,
        k_scales,
        v_packed,
        v_scales,
        o_packed,
        o_scales,
        gate_packed,
        gate_scales,
        up_packed,
        up_scales,
        down_packed,
        down_scales,
    })
}

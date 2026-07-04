#!/usr/bin/env python3
"""
Single transformer layer forward pass from Qwen3-0.6B using dequantized weights.

Dumps ALL intermediate values for debugging.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from q4nx_reference import (
    dequantize_weight,
    get_header,
    read_bf16,
)

# ── model dimensions (from config.json) ─────────────────────────────────
HIDDEN = 1024
NH = 16          # num_attention_heads
NKV = 8          # num_key_value_heads
HD = 128         # head_dim
GQA = NH // NKV  # 2
INTERMEDIATE = 3072
RPE_THETA = 1_000_000.0
EPS = 1e-6
NUM_LAYERS = 28


# ── helpers ─────────────────────────────────────────────────────────────

def rms_norm(x, weight):
    """RMSNorm: x / sqrt(mean(x²) + eps) * weight.  Returns float32."""
    x_f32 = x.astype(np.float32, copy=False)
    rms = np.sqrt(np.mean(x_f32 ** 2) + EPS)
    return (x_f32 / rms) * weight


def silu(x):
    """SiLU activation: x * sigmoid(x)."""
    return x * (1.0 / (1.0 + np.exp(-x)))


def rotate_half(x):
    """HF rotate_half convention for a vector of length HD."""
    hd2 = x.shape[-1] // 2
    x1 = x[..., :hd2]
    x2 = x[..., hd2:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rope(x, cos, sin):
    """
    Apply rotary position embedding to a [..., HD] tensor.

    Uses HuggingFace rotate_half convention where each pair (i, i+HD/2)
    shares the same (cos, sin) from the [HD/2] tables.
    """
    hd2 = x.shape[-1] // 2
    x1 = x[..., :hd2]
    x2 = x[..., hd2:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def precompute_rope(seq_len, head_dim=HD, theta=RPE_THETA):
    """
    Precompute cos/sin for RoPE at positions 0..seq_len-1.

    Returns arrays of shape [seq_len, HD/2] for use with apply_rope()'s
    rotate_half convention — each pair (i, i+HD/2) shares one (cos, sin).
    """
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    positions = np.arange(seq_len, dtype=np.float64)
    angles = np.outer(positions, inv_freq)  # [seq_len, HD//2]
    cos = np.cos(angles).astype(np.float32)  # [seq_len, HD//2]
    sin = np.sin(angles).astype(np.float32)
    return cos, sin  # [seq_len, HD/2] — no repeat for rotate_half


# ── tensor stats ────────────────────────────────────────────────────────

def describe(name, tensor):
    """Print name, shape, first 10 elements, min, max, mean for debugging."""
    flat = tensor.ravel()
    print(f"\n{'='*70}")
    print(f"{name}  shape={tensor.shape}  dtype={tensor.dtype}")
    print(f"  first 10: {flat[:10]}")
    print(f"  min={flat.min():.8f}  max={flat.max():.8f}  mean={flat.mean():.8f}")
    print(f"  norm={np.linalg.norm(flat):.8f}")
    return tensor  # pass-through for saving


# ── weight loader ───────────────────────────────────────────────────────

class LayerWeights:
    """Load and cache all weights for one transformer layer."""

    def __init__(self, layer_idx):
        h = get_header()
        pre = f"model.layers.{layer_idx}"

        # BF16 weights
        self.ln1_weight = read_bf16(h[f"{pre}.input_layernorm.weight"]["data_offsets"][0], HIDDEN)
        self.ln2_weight = read_bf16(h[f"{pre}.post_attention_layernorm.weight"]["data_offsets"][0], HIDDEN)
        self.q_norm = read_bf16(h[f"{pre}.self_attn.q_norm.weight"]["data_offsets"][0], HD)
        self.k_norm = read_bf16(h[f"{pre}.self_attn.k_norm.weight"]["data_offsets"][0], HD)

        # I8 weights — dequantize with logical in_features
        def _dq(name, out_f, in_f):
            info = h[name]
            off = info["data_offsets"][0]
            i8_rows = info["shape"][0]
            return dequantize_weight(off, i8_rows, in_f)

        self.q_w = _dq(f"{pre}.self_attn.q_proj.weight", NH * HD, HIDDEN)    # [2048, 1024]
        self.k_w = _dq(f"{pre}.self_attn.k_proj.weight", NKV * HD, HIDDEN)    # [1024, 1024]
        self.v_w = _dq(f"{pre}.self_attn.v_proj.weight", NKV * HD, HIDDEN)    # [1024, 1024]
        self.o_w = _dq(f"{pre}.self_attn.o_proj.weight", HIDDEN, NH * HD)     # [1024, 2048]

        self.gate_w = _dq(f"{pre}.mlp.gate_proj.weight", INTERMEDIATE, HIDDEN)  # [3072, 1024]
        self.up_w = _dq(f"{pre}.mlp.up_proj.weight", INTERMEDIATE, HIDDEN)      # [3072, 1024]
        self.down_w = _dq(f"{pre}.mlp.down_proj.weight", HIDDEN, INTERMEDIATE)  # [1024, 3072]


# ── single-layer forward ────────────────────────────────────────────────

def forward_layer(h, layer_idx, params):
    """
    Run one full transformer layer on hidden state h.
    
    h: [HIDDEN] float32 array.
    Returns the updated hidden state and a dict of all intermediates.
    """
    intermediates = {}
    w = LayerWeights(layer_idx)
    cos_full, sin_full = params["rope_cos"], params["rope_sin"]
    seq_pos = params.get("seq_pos", 0)

    # ── 1. RMSNorm before attention ──
    h_ln1 = rms_norm(h, w.ln1_weight)
    describe("after_rmsnorm_1 (input to attn)", h_ln1)
    intermediates["h_ln1"] = h_ln1

    # ── 2. QKV projections ──
    # W is [out_f, in_f]; output = W @ h  (h is [HIDDEN])
    q = w.q_w @ h   # [2048]
    k = w.k_w @ h   # [1024]
    v = w.v_w @ h   # [1024]
    describe("q_proj (flat, before reshape)", q)
    describe("k_proj (flat, before reshape)", k)
    describe("v_proj (flat, before reshape)", v)
    intermediates["q_flat"] = q
    intermediates["k_flat"] = k
    intermediates["v_flat"] = v

    # ── 3. Reshape into heads ──
    q = q.reshape(NH, HD)    # [16, 128]
    k = k.reshape(NKV, HD)   # [8, 128]
    v = v.reshape(NKV, HD)   # [8, 128]
    describe("q_heads (after reshape)", q)
    describe("k_heads (after reshape)", k)
    describe("v_heads (after reshape)", v)
    intermediates["q_heads"] = q
    intermediates["k_heads"] = k
    intermediates["v_heads"] = v

    # ── 4. QK norm (element-wise per head) ──
    for hh in range(NH):
        q[hh] = q[hh] * w.q_norm
    for kh in range(NKV):
        k[kh] = k[kh] * w.k_norm
    describe("q_heads (after QK norm)", q)
    describe("k_heads (after QK norm)", k)
    intermediates["q_normed"] = q.copy()
    intermediates["k_normed"] = k.copy()

    # ── 5. RoPE on q and k ──
    cos_p = cos_full[seq_pos]   # [HD/2]
    sin_p = sin_full[seq_pos]   # [HD/2]
    for hh in range(NH):
        q[hh] = apply_rope(q[hh], cos_p, sin_p)
    for kh in range(NKV):
        k[kh] = apply_rope(k[kh], cos_p, sin_p)
    describe("q_heads (after RoPE)", q)
    describe("k_heads (after RoPE)", k)
    intermediates["q_rope"] = q.copy()
    intermediates["k_rope"] = k.copy()

    # ── 6. Attention with GQA ──
    # GQA factor = NH // NKV = 2
    # For a single token (no cached past), we attend to just this position.
    # Softmax over 1 position is always 1.0, so attn_vec = v[kvh] * 1.0
    attn_out = np.zeros(NH * HD, dtype=np.float32)
    for hh in range(NH):
        kvh = hh // GQA
        # Single token: softmax over 1 element is always 1.0
        attn_vec = v[kvh]
        attn_out[hh * HD:(hh + 1) * HD] = attn_vec

    describe("attn_out (flat, before O proj)", attn_out)
    intermediates["attn_out_flat"] = attn_out

    # ── 7. Output projection ──
    # o_w is [H=1024, NH*HD=2048]; attn_out is [2048]
    # output[j] = sum_k o_w[j,k] * attn_out[k]  = o_w @ attn_out
    attn_proj = w.o_w @ attn_out   # [1024]
    describe("attn_proj (after O projection)", attn_proj)
    intermediates["attn_proj"] = attn_proj

    # ── 8. Residual ──
    h_after_attn = h + attn_proj
    describe("h_after_attn (first residual)", h_after_attn)
    intermediates["h_after_attn"] = h_after_attn

    # ── 9. Post-attention RMSNorm ──
    h_ln2 = rms_norm(h_after_attn, w.ln2_weight)
    describe("after_rmsnorm_2 (input to FFN)", h_ln2)
    intermediates["h_ln2"] = h_ln2

    # ── 10. SwiGLU FFN ──
    gate = w.gate_w @ h_ln2   # [3072]
    up = w.up_w @ h_ln2       # [3072]
    describe("ffn_gate (before activation)", gate)
    describe("ffn_up", up)
    intermediates["ffn_gate"] = gate
    intermediates["ffn_up"] = up

    hidden = silu(gate) * up
    describe("ffn_hidden (after silu(*)*)", hidden)
    intermediates["ffn_hidden"] = hidden

    ffn_out = w.down_w @ hidden   # [1024]
    describe("ffn_out (after down_proj)", ffn_out)
    intermediates["ffn_out"] = ffn_out

    # ── 11. Residual ──
    h_out = h_after_attn + ffn_out
    describe("h_out (final)", h_out)
    intermediates["h_out"] = h_out

    return h_out, intermediates


# ── main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Qwen3-0.6B Layer 0 Single-Token Forward Trace")
    print("=" * 70)

    # Load embedding table and generate test input
    h = get_header()
    embed_info = h["model.embed_tokens.weight"]
    embeds = read_bf16(embed_info["data_offsets"][0], 151936 * HIDDEN).reshape(151936, HIDDEN)
    describe("embed_tokens (whole table)", embeds)

    # Use token ID 100 (a common token) as test input
    token_id = 100
    h_test = embeds[token_id].copy()
    print(f"\nUsing token_id={token_id} embedding")
    describe("input_embedding", h_test)

    # Precompute RoPE tables
    seq_len = 1  # single token
    cos_full, sin_full = precompute_rope(seq_len)
    describe("rope_cos (pos 0)", cos_full[0])
    describe("rope_sin (pos 0)", sin_full[0])

    params = {
        "rope_cos": cos_full,
        "rope_sin": sin_full,
        "seq_pos": 0,
    }

    # Run layer 0 forward
    h_out, intermediates = forward_layer(h_test, layer_idx=0, params=params)

    # ── save all intermediates ──
    npz_path = "/tmp/layer_trace_outputs.npz"
    np.savez(npz_path, **intermediates)
    print(f"\n{'='*70}")
    print(f"All intermediates saved to: {npz_path}")
    print(f"Keys: {list(intermediates.keys())}")
    print("Done.")


if __name__ == "__main__":
    main()

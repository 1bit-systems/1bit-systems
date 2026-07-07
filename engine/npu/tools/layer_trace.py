#!/usr/bin/env python3
"""
Layer-by-layer numerical trace for NPU engine v12 correctness debug.

Dequantizes ONE layer's weights from model.q4nx, runs the forward pass on CPU
(with numpy), and compares intermediates against what the C++ engine produces.

Usage:
  ./layer_trace.py                    # Run trace, dump layer 0 intermediates
  ./layer_trace.py --layer 5         # Trace a different layer
  ./layer_trace.py --dump-all        # Dump all intermediates to files for diff
"""

import json, struct, sys, os, numpy as np
from pathlib import Path

MODEL_PATH = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx"
TOKENIZER_PATH = MODEL_PATH + "/../tokenizer.json"

# Model config (Qwen3-0.6B)
H = 1024      # hidden size
NC = 28       # num layers
NH = 16       # num q heads
NKV = 8       # num kv heads
HD = 128      # head dim
IM = 3072     # intermediate size
NV = 151936   # vocab size
GQA = NH // NKV  # 2
ROPE_THETA = 1000000.0

# Test prompt tokens (same as C++ engine)
TEST_TOKENS = [151643, 872, 198, 11852, 151644, 198, 151643, 77091, 198]


def bf16_to_f32(bf16_val):
    """Convert BF16 uint16 to float32."""
    bits = np.uint32(bf16_val) << 16
    return np.frombuffer(bits.tobytes(), dtype=np.float32)[0]


def dequant_weight(data, i8_rows, in_features):
    """Dequantize I8 weight tensor → [out_features, in_features] float matrix.
    Vectorized numpy version for speed.
    """
    tile_cols = in_features // 256
    tile_rows = i8_rows // tile_cols
    out_rows = tile_rows * 32
    out_cols = tile_cols * 256
    
    # Pre-allocate result
    result = np.zeros((out_rows, out_cols), dtype=np.float32)
    
    # Process all I8 rows as tiles
    for ir in range(i8_rows):
        rd = data[ir * 5120 : (ir + 1) * 5120]
        tile_row = ir // tile_cols
        tile_col = ir % tile_cols
        
        # Parse scales and zeros as BF16 → F32
        scales_u16 = np.frombuffer(rd[0:512], dtype=np.uint16).reshape(8, 32)
        zeros_u16 = np.frombuffer(rd[512:1024], dtype=np.uint16).reshape(8, 32)
        
        # BF16 → F32 conversion
        scales_bits = scales_u16.astype(np.uint32) << 16
        scales_f32 = np.frombuffer(scales_bits.tobytes(), dtype=np.float32).reshape(8, 32)
        zeros_bits = zeros_u16.astype(np.uint32) << 16
        zeros_f32 = np.frombuffer(zeros_bits.tobytes(), dtype=np.float32).reshape(8, 32)
        
        packed = np.frombuffer(rd[1024:], dtype=np.uint8).reshape(2, 256, 8)
        
        # Process 2 lanes (rows 0-15, rows 16-31)
        for lane in range(2):
            lane_packed = packed[lane]  # [256, 8]
            for row_in_lane in range(16):
                lr = lane * 16 + row_in_lane
                byte_idx = row_in_lane // 2
                nibble_sel = row_in_lane % 2
                
                byte_vals = lane_packed[:, byte_idx]  # [256]
                if nibble_sel == 0:
                    vals = (byte_vals & 0x0F).astype(np.int16)
                else:
                    vals = ((byte_vals >> 4) & 0x0F).astype(np.int16)
                vals = np.where(vals >= 8, vals - 16, vals).astype(np.int8)
                
                # Per-group scale/zero: group = col // 32
                for g in range(8):
                    col_start = g * 32
                    col_end = (g + 1) * 32
                    s = float(scales_f32[g, lr])
                    z = float(zeros_f32[g, lr])
                    row_idx = tile_row * 32 + lr
                    col_slice = tile_col * 256 + col_start
                    col_data = vals[col_start:col_end].astype(np.float32) * s + z
                    # Q4NX: dequant = val * scale + zp (zero_point is a general bias term)
                    result[row_idx, col_slice:col_slice+32] = col_data
    
    return result


def load_model():
    """Load the Q4NX model and return header + raw data."""
    print(f"Loading model from {MODEL_PATH}...")
    with open(MODEL_PATH, 'rb') as f:
        raw_data = f.read()
    
    hdr_size = struct.unpack('<Q', raw_data[0:8])[0]
    header = json.loads(raw_data[8:8+hdr_size])
    weight_data = raw_data[8+hdr_size:]
    return header, weight_data


def get_weight(header, weight_data, name, in_features=H):
    """Get a dequantized weight tensor by name."""
    info = header.get(name)
    if info is None:
        raise KeyError(f"Weight {name} not found")
    
    start, end = info['data_offsets']
    raw = weight_data[start:end]
    
    if info['dtype'] == 'BF16':
        arr = np.frombuffer(raw, dtype=np.uint16)
        bits = arr.astype(np.uint32) << 16
        return np.frombuffer(bits.tobytes(), dtype=np.float32).reshape(info['shape']).copy()
    
    # I8 format: dequantize
    i8_rows = info['shape'][0]
    return dequant_weight(raw, i8_rows, in_features)


def rms_norm(x, weight, eps=1e-6):
    """RMSNorm: x = x / sqrt(mean(x²) + eps) * weight"""
    ss = np.mean(x.astype(np.float64)**2)
    inv_rms = 1.0 / np.sqrt(float(ss) + eps)
    return (x * inv_rms * weight).astype(np.float32)


def precompute_rope(hd, theta, max_pos=4096):
    """Precompute RoPE cos/sin for all positions."""
    rc = np.zeros((max_pos, hd), dtype=np.float32)
    rs = np.zeros((max_pos, hd), dtype=np.float32)
    hd2 = hd // 2
    for p in range(max_pos):
        for d in range(hd2):
            f = 1.0 / (theta ** (2.0 * d / hd))
            angle = p * f
            rc[p, d] = np.cos(angle)
            rs[p, d] = np.sin(angle)
            rc[p, d + hd2] = np.cos(angle)
            rs[p, d + hd2] = np.sin(angle)
    return rc, rs


def apply_rope(x, hd, pos, rc, rs):
    """Apply RoPE rotation (interleaved pairs convention)."""
    hd2 = hd // 2
    result = x.copy()
    for d in range(hd2):
        a, b = x[d], x[d + hd2]
        c, s = rc[pos, d], rs[pos, d]
        result[d] = a * c - b * s
        result[d + hd2] = b * c + a * s
    return result


def silu(x):
    """SiLU activation: x * sigmoid(x)"""
    return x / (1.0 + np.exp(-x))


def softmax(x):
    """Stable softmax."""
    mx = np.max(x)
    e = np.exp(x.astype(np.float64) - mx)
    return (e / np.sum(e)).astype(np.float32)


def forward_layer(header, weight_data, hidden_states, layer_idx, pos, kv_cache, rc, rs):
    """
    Run one transformer layer forward.
    Returns (output_hidden, updated_kv_cache, debug_info)
    """
    prefix = f"model.layers.{layer_idx}"
    H_DIM = hidden_states.shape[0]
    
    debug = {}
    
    # === Input RMSNorm ===
    in_weight = get_weight(header, weight_data, f"{prefix}.input_layernorm.weight")
    h_normed = rms_norm(hidden_states, in_weight)
    debug['01_pre_rmsnorm'] = hidden_states.copy()
    debug['02_in_rmsnorm'] = h_normed.copy()
    
    # === QKV projection ===
    q_w = get_weight(header, weight_data, f"{prefix}.self_attn.q_proj.weight", H_DIM)   # [2048, 1024]
    k_w = get_weight(header, weight_data, f"{prefix}.self_attn.k_proj.weight", H_DIM)   # [1024, 1024]
    v_w = get_weight(header, weight_data, f"{prefix}.self_attn.v_proj.weight", H_DIM)   # [1024, 1024]
    
    q = q_w @ h_normed   # [2048]  (Q = W_Q @ h)
    k = k_w @ h_normed   # [1024]
    v = v_w @ h_normed   # [1024]
    
    debug['03_q_raw'] = q.copy()
    debug['04_k_raw'] = k.copy()
    debug['05_v_raw'] = v.copy()
    
    # === Q/K norm ===
    qn_w = get_weight(header, weight_data, f"{prefix}.self_attn.q_norm.weight", HD) if f"{prefix}.self_attn.q_norm.weight" in header else None
    kn_w = get_weight(header, weight_data, f"{prefix}.self_attn.k_norm.weight", HD) if f"{prefix}.self_attn.k_norm.weight" in header else None
    
    if qn_w is not None:
        q_heads = q.reshape(NH, HD)
        for hh in range(NH):
            ss = np.sum(q_heads[hh].astype(np.float64)**2)  # sum of squares (matches C++)
            iq = 1.0 / np.sqrt(float(ss / HD) + 1e-6)
            q_heads[hh] = q_heads[hh] * iq * qn_w
        q = q_heads.reshape(-1)
        debug['06_q_normed'] = q.copy()
    
    if kn_w is not None:
        kv_heads = k.reshape(NKV, HD)
        for kh in range(NKV):
            ss = np.sum(kv_heads[kh].astype(np.float64)**2)  # sum of squares (matches C++)
            ik = 1.0 / np.sqrt(float(ss / HD) + 1e-6)
            kv_heads[kh] = kv_heads[kh] * ik * kn_w
        k = kv_heads.reshape(-1)
        debug['07_k_normed'] = k.copy()
    
    # === RoPE ===
    q_heads = q.reshape(NH, HD)
    for hh in range(NH):
        q_heads[hh] = apply_rope(q_heads[hh], HD, pos, rc, rs)
    q = q_heads.reshape(-1)
    
    kv_heads = k.reshape(NKV, HD)
    for kh in range(NKV):
        kv_heads[kh] = apply_rope(kv_heads[kh], HD, pos, rc, rs)
    k = kv_heads.reshape(-1)
    
    debug['08_q_rope'] = q.copy()
    debug['09_k_rope'] = k.copy()
    # v stays as-is (no norm/RoPE for V)
    
    # === Store in KV cache ===
    if layer_idx not in kv_cache:
        kv_cache[layer_idx] = {'k': {}, 'v': {}}
    
    k_heads = k.reshape(NKV, HD)
    v_heads = v.reshape(NKV, HD)
    for kh in range(NKV):
        kv_cache[layer_idx]['k'][pos] = k_heads[kh].copy()
        kv_cache[layer_idx]['v'][pos] = v_heads[kh].copy()
    
    # === Attention ===
    cl = pos + 1  # context length
    attn_out = np.zeros(NH * HD, dtype=np.float32)
    q_heads = q.reshape(NH, HD)
    
    for hh in range(NH):
        kvh = hh // GQA
        scores = np.zeros(cl, dtype=np.float32)
        for p in range(cl):
            k_cached = kv_cache[layer_idx]['k'].get(p, np.zeros(HD))
            scores[p] = np.dot(q_heads[hh].astype(np.float64), 
                               k_cached.astype(np.float64)) / np.sqrt(HD)
        
        scores = softmax(scores)
        
        for d in range(HD):
            val = 0.0
            for p in range(cl):
                v_cached = kv_cache[layer_idx]['v'].get(p, np.zeros(HD))
                val += float(scores[p]) * float(v_cached[d])
            attn_out[hh * HD + d] = val
    
    debug['10_attention'] = attn_out.copy()
    
    # === O projection ===
    o_w = get_weight(header, weight_data, f"{prefix}.self_attn.o_proj.weight", NH * HD)  # [1024, 2048]
    o_out = o_w @ attn_out  # [1024]
    debug['11_o_proj'] = o_out.copy()
    
    # === Residual + Post RMSNorm ===
    h_residual = hidden_states + o_out
    debug['12_residual_attn'] = h_residual.copy()
    
    pa_weight = get_weight(header, weight_data, f"{prefix}.post_attention_layernorm.weight")
    h_mlp_in = rms_norm(h_residual, pa_weight)
    debug['13_post_rmsnorm'] = h_mlp_in.copy()
    
    # === MLP: Gate, Up, SiLU, Down ===
    gate_w = get_weight(header, weight_data, f"{prefix}.mlp.gate_proj.weight", H_DIM)  # [3072, 1024]
    up_w = get_weight(header, weight_data, f"{prefix}.mlp.up_proj.weight", H_DIM)     # [3072, 1024]
    down_w = get_weight(header, weight_data, f"{prefix}.mlp.down_proj.weight", IM)    # [1024, 3072]
    
    gate = gate_w @ h_mlp_in  # [3072]
    up = up_w @ h_mlp_in      # [3072]
    debug['14_gate'] = gate.copy()
    debug['15_up'] = up.copy()
    
    silu_out = silu(gate) * up  # [3072]
    debug['16_silu'] = silu_out.copy()
    
    down = down_w @ silu_out  # [1024]
    debug['17_down'] = down.copy()
    
    # === Final residual ===
    output = h_residual + down
    debug['18_output'] = output.copy()
    
    return output, kv_cache, debug


def run_trace(layer_idx=0, npt=1):
    """Run forward trace for one layer and dump intermediates."""
    print(f"Loading model...")
    header, weight_data = load_model()
    
    # Get embedding table
    w_emb = get_weight(header, weight_data, "model.embed_tokens.weight", H)
    print(f"Embedding shape: {w_emb.shape}")
    
    # Get lm_head weights
    lm_w = get_weight(header, weight_data, "lm_head.weight", H)
    print(f"LM head shape: {lm_w.shape}")
    
    # Get final norm
    fn_w = get_weight(header, weight_data, "model.norm.weight", H)
    print(f"Final norm shape: {fn_w.shape}")
    
    # Precompute RoPE
    rc, rs = precompute_rope(HD, ROPE_THETA, 4096)
    
    # Get input embedding for first token
    tok = TEST_TOKENS[0]
    h = w_emb[tok].copy()
    print(f"\nInput token: {tok}")
    print(f"Embedding norm: {np.mean(np.abs(h)):.6f}")
    print(f"Embedding first 10: {h[:10]}")
    
    # Run layer forward
    kv_cache = {}
    pos = 0
    h = h.astype(np.float32)
    
    # Run through ALL layers up to layer_idx (to get proper KV cache state)
    for l in range(layer_idx):
        h, kv_cache, _ = forward_layer(header, weight_data, h, l, pos, kv_cache, rc, rs)
    
    # Run the target layer
    print(f"\n=== Running layer {layer_idx} ===")
    h, kv_cache, debug = forward_layer(header, weight_data, h, layer_idx, pos, kv_cache, rc, rs)
    
    # Print debug info for each intermediate
    print(f"\n--- Layer {layer_idx} Intermediates ---")
    for key, val in sorted(debug.items()):
        print(f"  {key}: shape={val.shape}, "
              f"min={val.min():.6f}, max={val.max():.6f}, "
              f"mean={np.mean(np.abs(val)):.6f}, "
              f"first5={val.flatten()[:5].tolist()}")
    
    # Run final norm + LM head
    final_normed = rms_norm(h, fn_w)
    logits = lm_w @ final_normed  # [NV]
    top5 = np.argsort(-logits)[:5]
    print(f"\n--- LM Head Output ---")
    print(f"  Final hidden mean_abs: {np.mean(np.abs(h)):.6f}")
    print(f"  Final normed mean_abs: {np.mean(np.abs(final_normed)):.6f}")
    print(f"  Top-5 tokens: {top5.tolist()}")
    print(f"  Top-5 logits: {logits[top5].tolist()}")
    
    # Also dump to file for comparison with C++ engine
    dump_dir = f"/tmp/layer_trace_l{layer_idx}"
    os.makedirs(dump_dir, exist_ok=True)
    for key, val in debug.items():
        val.astype(np.float32).tofile(f"{dump_dir}/{key}.f32")
    print(f"\nIntermediates dumped to {dump_dir}/")
    
    return debug


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--dump-all', action='store_true')
    args = parser.parse_args()
    
    debug = run_trace(layer_idx=args.layer)
    
    if args.dump_all:
        import json
        info = {k: {'shape': list(v.shape), 'min': float(v.min()), 'max': float(v.max())} 
                for k, v in debug.items()}
        with open(f'/tmp/layer_trace_l{args.layer}/info.json', 'w') as f:
            json.dump(info, f, indent=2)
        print(f"Info dumped")

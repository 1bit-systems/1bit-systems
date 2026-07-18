#!/usr/bin/env python3
"""
Full 28-layer reference forward pass from Q4NX dequantized weights.

Dumps ALL key intermediates (hidden states, QKV outputs, attention outputs,
gate/up/down projections, final norms) as f32 binary files for direct
comparison against NPU engine trace dumps.

Usage:
    python3 tools/full_ref_forward.py [--model /path/to/model.q4nx]
                                      [--tokens 151644,872,198,13048,151645]
                                      [--output /tmp/trace_ref/]
                                      [--layers 0,1,2,13,26,27  # default: all 28]
"""
import argparse, json, mmap, os, struct, sys
from pathlib import Path
import numpy as np

# ── model dimensions (Qwen3-0.6B) ─────────────────────────────────────
H = 1024
NH = 16
NKV = 8
HD = 128
GQA = NH // NKV        # 2
IM = 3072
NV = 151936
NUM_LAYERS = 28
EPS = 1e-6
RPE_THETA = 1_000_000.0
TILE_BYTES = 5120
COLS_PER_TILE = 256
ROWS_PER_TILE = 32
GROUP_SIZE = 32
LANE_BYTES = 2048

# ── helpers ────────────────────────────────────────────────────────────

def bf16_to_f32(bits):
    if isinstance(bits, np.ndarray):
        i32 = bits.astype(np.uint32) << np.uint32(16)
        nan_mask = (bits & 0x7F80) == 0x7F80
        out = i32.view(np.float32)
        out[nan_mask] = 0.0
        return out
    if bits & 0x7F80 == 0x7F80:
        return 0.0
    u32 = np.uint32(bits) << np.uint32(16)
    return struct.unpack("<f", struct.pack("<I", int(u32)))[0]

def rms_norm(x, weight):
    x_f32 = x.astype(np.float32, copy=False)
    rms = np.sqrt(np.mean(x_f32 ** 2) + EPS)
    return (x_f32 / rms) * weight

def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))

def rotate_half(x):
    hd2 = x.shape[-1] // 2
    x1, x2 = x[..., :hd2], x[..., hd2:]
    return np.concatenate([-x2, x1], axis=-1)

def apply_rope(x, cos, sin):
    hd2 = x.shape[-1] // 2
    x1, x2 = x[..., :hd2], x[..., hd2:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

def precompute_rope(seq_len, head_dim=HD, theta=RPE_THETA):
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    positions = np.arange(seq_len, dtype=np.float64)
    angles = np.outer(positions, inv_freq)
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    return cos, sin

# ── Q4NX loader ───────────────────────────────────────────────────────

class Q4NXModel:
    """Load model.q4nx and provide dequantized weights."""
    
    def __init__(self, model_path):
        self.path = model_path
        self._file = open(model_path, "rb")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        hsz = struct.unpack_from("<Q", self._mm, 0)[0]
        self._data_start = 8 + hsz
        self.header = json.loads(self._mm[8:8+hsz].decode("utf-8"))
        print(f"Loaded: {model_path} ({len(self.header)} tensors)")
    
    def _raw_off(self, data_offset):
        return self._data_start + data_offset
    
    def read_bf16(self, offset, n):
        raw = np.frombuffer(self._mm, dtype=np.uint16, count=n, offset=self._raw_off(offset))
        return bf16_to_f32(raw)
    
    def dequantize_weight(self, name, in_features):
        """Dequantize a named I8 tensor. Returns f32 [out_features, in_features]."""
        info = self.header[name]
        offset = info["data_offsets"][0]
        i8_rows = info["shape"][0]
        n_tile_cols = in_features // COLS_PER_TILE
        out_features = (i8_rows // n_tile_cols) * ROWS_PER_TILE
        
        result = np.zeros((out_features, in_features), dtype=np.float32)
        for tile_idx in range(i8_rows):
            tile_off = self._raw_off(offset) + tile_idx * TILE_BYTES
            tile_row = tile_idx // n_tile_cols
            tile_col = tile_idx % n_tile_cols
            
            scales_raw = np.frombuffer(self._mm, dtype=np.uint16, count=256, offset=tile_off)
            scales = bf16_to_f32(scales_raw)
            zps_raw = np.frombuffer(self._mm, dtype=np.uint16, count=256, offset=tile_off + 512)
            zps = bf16_to_f32(zps_raw)
            int4_raw = np.frombuffer(self._mm, dtype=np.uint8, count=4096, offset=tile_off + 1024)
            
            for lane in range(2):
                for lr_in_lane in range(16):
                    local_row = lane * 16 + lr_in_lane
                    global_row = tile_row * ROWS_PER_TILE + local_row
                    byte_idx = lr_in_lane // 2
                    is_lo = (lr_in_lane % 2) == 0
                    
                    for local_col in range(COLS_PER_TILE):
                        addr = lane * LANE_BYTES + local_col * 8 + byte_idx
                        byte_val = int(int4_raw[addr])
                        raw_nibble = byte_val & 0x0F if is_lo else (byte_val >> 4) & 0x0F
                        int4_val = raw_nibble if raw_nibble < 8 else raw_nibble - 16
                        global_col = tile_col * COLS_PER_TILE + local_col
                        group = local_col // GROUP_SIZE
                        s = scales[group * ROWS_PER_TILE + local_row]
                        zp = zps[group * ROWS_PER_TILE + local_row]
                        result[global_row, global_col] = float(int4_val) * s + zp
        return result

# ── single-token forward through one layer ────────────────────────────

def forward_layer(model, h, layer_idx, w_cache, rope_cos, rope_sin, seq_pos):
    """
    Run one transformer layer on hidden state h (f32 [H]).
    Returns (h_out, intermediates_dict).
    """
    inter = {}
    
    # Load or get cached weights
    if layer_idx not in w_cache:
        w = {}
        pre = f"model.layers.{layer_idx}"
        w['ln1'] = np.clip(model.read_bf16(model.header[f"{pre}.input_layernorm.weight"]["data_offsets"][0], H), -2.0, 2.0)
        w['ln2'] = np.clip(model.read_bf16(model.header[f"{pre}.post_attention_layernorm.weight"]["data_offsets"][0], H), -2.0, 2.0)
        w['q_norm'] = model.read_bf16(model.header[f"{pre}.self_attn.q_norm.weight"]["data_offsets"][0], HD)
        w['k_norm'] = model.read_bf16(model.header[f"{pre}.self_attn.k_norm.weight"]["data_offsets"][0], HD)
        w['q'] = model.dequantize_weight(f"{pre}.self_attn.q_proj.weight", H)
        w['k'] = model.dequantize_weight(f"{pre}.self_attn.k_proj.weight", H)
        w['v'] = model.dequantize_weight(f"{pre}.self_attn.v_proj.weight", H)
        w['o'] = model.dequantize_weight(f"{pre}.self_attn.o_proj.weight", NH * HD)
        w['gate'] = model.dequantize_weight(f"{pre}.mlp.gate_proj.weight", H)
        w['up'] = model.dequantize_weight(f"{pre}.mlp.up_proj.weight", H)
        w['down'] = model.dequantize_weight(f"{pre}.mlp.down_proj.weight", IM)
        w_cache[layer_idx] = w
    else:
        w = w_cache[layer_idx]
    
    # 1. Pre-attention RMSNorm
    h_ln1 = rms_norm(h, w['ln1'])
    inter['h_ln1'] = h_ln1
    
    # 2. QKV projections
    q_flat = w['q'] @ h       # [NH*HD]
    k_flat = w['k'] @ h       # [NKV*HD]
    v_flat = w['v'] @ h       # [NKV*HD]
    inter['q_flat'] = q_flat
    inter['k_flat'] = k_flat
    inter['v_flat'] = v_flat
    
    # 3. Reshape into heads
    q = q_flat.copy().reshape(NH, HD)
    k = k_flat.copy().reshape(NKV, HD)
    v = v_flat.copy().reshape(NKV, HD)
    inter['q_heads'] = q.copy()
    inter['k_heads'] = k.copy()
    inter['v_heads'] = v.copy()
    
    # 4. QK norm
    q_normed = q * w['q_norm']  # broadcasts
    k_normed = k * w['k_norm']
    inter['q_normed'] = q_normed.copy()
    inter['k_normed'] = k_normed.copy()
    
    # 5. RoPE
    cos_p = rope_cos[seq_pos]  # [HD/2]
    sin_p = rope_sin[seq_pos]
    for hh in range(NH):
        q_normed[hh] = apply_rope(q_normed[hh], cos_p, sin_p)
    for kh in range(NKV):
        k_normed[kh] = apply_rope(k_normed[kh], cos_p, sin_p)
    inter['q_rope'] = q_normed.copy()
    inter['k_rope'] = k_normed.copy()
    
    # 6. Attention (single token — softmax over 1 = 1.0)
    attn_flat = np.zeros(NH * HD, dtype=np.float32)
    for hh in range(NH):
        kvh = hh // GQA
        attn_flat[hh*HD:(hh+1)*HD] = v[kvh]
    inter['attn_flat'] = attn_flat
    
    # 7. Output projection
    attn_proj = w['o'] @ attn_flat  # [H]
    inter['attn_proj'] = attn_proj
    
    # 8. Residual
    h_after_attn = h + attn_proj
    inter['h_after_attn'] = h_after_attn
    
    # 9. Post-attention RMSNorm
    h_ln2 = rms_norm(h_after_attn, w['ln2'])
    inter['h_ln2'] = h_ln2
    
    # 10. SwiGLU FFN
    gate = w['gate'] @ h_ln2  # [IM]
    up = w['up'] @ h_ln2      # [IM]
    inter['ffn_gate'] = gate
    inter['ffn_up'] = up
    
    hidden = silu(gate) * up
    inter['ffn_hidden'] = hidden
    
    ffn_out = w['down'] @ hidden  # [H]
    inter['ffn_out'] = ffn_out
    
    # 11. Residual
    h_out = h_after_attn + ffn_out
    inter['h_out'] = h_out
    
    return h_out, inter


# ── main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Full 28-layer reference forward pass from Q4NX")
    ap.add_argument("--model", default=os.path.expanduser("~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx"))
    ap.add_argument("--tokens", default="151644,872,198,13048,151645,198,151644,77091,198",
                    help="Comma-separated token IDs (default: '<|im_start|>user\\nHi<|im_end|>\\n<|im_start|>assistant\\n')")
    ap.add_argument("--output", default="/tmp/trace_ref/", help="Output directory for trace binaries")
    ap.add_argument("--layers", default=None, help="Comma-separated layer indices (default: all 28)")
    args = ap.parse_args()
    
    model = Q4NXModel(args.model)
    
    # Parse tokens
    token_ids = [int(t.strip()) for t in args.tokens.split(",")]
    npt = len(token_ids)
    print(f"Tokens ({npt}): {token_ids}")
    
    # Parse layers
    if args.layers:
        layer_list = [int(x) for x in args.layers.split(",")]
    else:
        layer_list = list(range(NUM_LAYERS))
    
    # Embedding table
    emb_info = model.header["model.embed_tokens.weight"]
    embeds = model.read_bf16(emb_info["data_offsets"][0], NV * H).reshape(NV, H)
    
    # Final norm
    final_norm = np.clip(model.read_bf16(model.header["model.norm.weight"]["data_offsets"][0], H), -2.0, 2.0)
    
    # Precompute RoPE
    rope_cos, rope_sin = precompute_rope(4096)
    
    # Output dir
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Weight cache across layers
    w_cache = {}
    
    # Run prefill (only last token matters for decode comparison)
    h = embeds[token_ids[0]].copy()
    print(f"\n─── Full {NUM_LAYERS}-layer forward ───")
    
    for l in range(NUM_LAYERS):
        h_out, inter = forward_layer(model, h, l, w_cache, rope_cos, rope_sin, seq_pos=0)
        
        if l in layer_list:
            layer_dir = out_dir / f"layer_{l:02d}"
            layer_dir.mkdir(exist_ok=True)
            
            for name, tensor in inter.items():
                path = layer_dir / f"{name}.f32"
                tensor.astype(np.float32).tofile(str(path))
            
            # Also save input and output
            h.astype(np.float32).tofile(str(layer_dir / "h_in.f32"))
            h_out.astype(np.float32).tofile(str(layer_dir / "h_out.f32"))
        
        # Compute stats for progress
        h_norm = np.linalg.norm(h_out)
        h_max = np.max(np.abs(h_out))
        print(f"  Layer {l:2d}: |h|={h_norm:.2f}  max|h|={h_max:.2f}")
        h = h_out
    
    # Final norm + LM head
    h_final = rms_norm(h, final_norm)
    h_final.astype(np.float32).tofile(str(out_dir / "final_norm_output.f32"))
    print(f"\n  Final norm: |h|={np.linalg.norm(h_final):.2f}")
    
    # LM head (tied embeddings)
    logits = embeds @ h_final  # [NV]
    logits.astype(np.float32).tofile(str(out_dir / "logits.f32"))
    
    top5 = np.argsort(logits)[-5:][::-1]
    print(f"\n  Top-5 tokens: {top5.tolist()}")
    print(f"  Top-5 logits: {logits[top5].tolist()}")
    
    # ── also dump the quantized weights as used by the NPU engine ──
    # (dequantized I8 → host-side f32 before INT8 quant)
    print(f"\n─── Saving weight references ───")
    wdir = out_dir / "weights"
    wdir.mkdir(exist_ok=True)
    for l in range(NUM_LAYERS):
        pre = f"model.layers.{l}"
        for name in ['q', 'k', 'v', 'o', 'gate', 'up', 'down']:
            try:
                w_np = model.dequantize_weight(f"{pre}.self_attn.{name}.weight" if name in 'qkvo' 
                                               else f"{pre}.mlp.{name}.weight", 
                                               H if name != 'o' else NH*HD if name != 'down' else IM)
                if name == 'down':
                    w_np = model.dequantize_weight(f"{pre}.mlp.{name}.weight", IM)
                w_np.astype(np.float32).tofile(str(wdir / f"layer{l:02d}_{name}.f32"))
            except:
                pass
        # Norms
        for nname in ['input_layernorm', 'post_attention_layernorm']:
            w_np = model.read_bf16(model.header[f"{pre}.{nname}.weight"]["data_offsets"][0], H)
            w_np.astype(np.float32).tofile(str(wdir / f"layer{l:02d}_{nname}.f32"))
    
    print(f"  All references saved to {out_dir}")
    print(f"\n  To compare: npu_engine_universalistic --trace /tmp/trace_npu/")
    print(f"  Then: python3 tools/cb_trace_diff.py /tmp/trace_ref/ /tmp/trace_npu/")

if __name__ == "__main__":
    main()

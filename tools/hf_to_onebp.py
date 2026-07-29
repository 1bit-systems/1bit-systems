#!/usr/bin/env python3
"""
hf_to_onebp.py — Convert HuggingFace safetensors models to 1BP format.

Supports Moonshot AI models (Kimi K3, Moonlight, Kimi-VL) and general
transformer architectures. Handles MXFP4, BF16, FP16, and INT8 quantization.

Usage:
  # Convert a local safetensors directory to 1BP
  python3 tools/hf_to_onebp.py --input ./models/Moonlight-16B-A3B --output models/Moonlight-16B-A3B.1bp

  # Download from HuggingFace and convert
  python3 tools/hf_to_onebp.py --repo moonshotai/Moonlight-16B-A3B --output models/Moonlight-16B-A3B.1bp

  # Specify architecture and quantization
  python3 tools/hf_to_onebp.py --input ./model --output model.1bp --arch kimi_vl --quant TQ2

Requires: torch, numpy, safetensors, huggingface_hub
"""
import argparse
import json
import math
import os
import struct
import sys
import time
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# 1BP Format Constants (must match include/onebp_format.h)
# ═══════════════════════════════════════════════════════════════════════════

ONEBP_MAGIC     = 0x00504231  # "1BP\0"
ONEBP_VERSION   = 1

# Quantization types
ONEBP_Q4NX = 0   # 4-bit block quant, bf16 scales
ONEBP_I8   = 1   # INT8 per-tensor quant
ONEBP_TQ1  = 2   # Ternary TQ1 (1.58-bit)
ONEBP_TQ2  = 3   # Ternary TQ2 (2-bit)
ONEBP_F16  = 4   # Float16
ONEBP_F32  = 5   # Float32
ONEBP_MXFP4 = 6  # MXFP4 microscaling (E2M1)

# Architecture types
ONEBP_DENSE     = 0   # Dense transformer
ONEBP_MOE       = 1   # Mixture of Experts
ONEBP_VISION    = 2   # Vision-language
ONEBP_TERNARY   = 4   # Ternary/1-bit
ONEBP_MAMBA     = 5   # Mamba SSM
ONEBP_KIMI_K3   = 50  # Kimi K3 (KDA + MLA + LatentMoE)
ONEBP_MOONLIGHT = 51  # Moonlight (Gated MLA MoE)
ONEBP_KIMI_VL   = 52  # Kimi-VL (Moonlight + MoonViT)

# Architecture string → enum mapping
ARCH_MAP = {
    "kimi_k3":   ONEBP_KIMI_K3,
    "moonlight": ONEBP_MOONLIGHT,
    "kimi_vl":   ONEBP_KIMI_VL,
    "kimi":      ONEBP_KIMI_K3,
}

QUANT_MAP = {
    "Q4NX":  ONEBP_Q4NX,
    "I8":    ONEBP_I8,
    "TQ1":   ONEBP_TQ1,
    "TQ2":   ONEBP_TQ2,
    "F16":   ONEBP_F16,
    "F32":   ONEBP_F32,
    "MXFP4": ONEBP_MXFP4,
}

# ═══════════════════════════════════════════════════════════════════════════
# MXFP4 Dequant/Quant Helpers
# ═══════════════════════════════════════════════════════════════════════════

# MXFP4 E2M1 lookup
MXFP4_LUT = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
   -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=np.float32)

def mxfp4_dequant_block(nibbles, scale_e8m0, n=32):
    """Dequantize one MXFP4 block to float32 array."""
    nibbles = np.asarray(nibbles, dtype=np.uint8)
    # Expand nibbles: each byte → 2 nibbles (low first)
    vals = np.zeros(n, dtype=np.uint8)
    vals[0::2] = nibbles & 0x0F
    vals[1::2] = (nibbles >> 4) & 0x0F
    # Scale: E8M0 is a base-2 exponent with bias = 0
    scale = 2.0 ** (int(scale_e8m0) - 127)
    return MXFP4_LUT[vals] * scale

def quant_to_mxfp4(data, block_size=32):
    """Quantize float32 array to MXFP4 blocks.
    Returns (nibbles_bytes, scales_bytes) as arrays.
    """
    n = data.shape[0]
    n_blocks = (n + block_size - 1) // block_size
    nibbles_list = []
    scales_list = []

    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, n)
        block = data[start:end]
        # Pad with zeros if needed
        if len(block) < block_size:
            block = np.pad(block, (0, block_size - len(block)))

        # Find the max absolute value in the block
        max_abs = np.max(np.abs(block))
        if max_abs < 1e-10:
            scale_e8m0 = 0  # zero block
            nibbles = np.zeros(block_size // 2, dtype=np.uint8)
        else:
            # Find optimal scale (E8M0) that fits the max value in MXFP4 range
            # MXFP4 max representable value = 6.0 * 2^(E8M0 - 127)
            # We want: 6.0 * 2^(s-127) >= max_abs
            target_scale = max_abs / 6.0
            s = min(255, max(0, int(np.ceil(np.log2(target_scale))) + 127))
            scale_val = 2.0 ** (s - 127)

            # Quantize to nearest MXFP4 value
            scaled = block / scale_val
            # Find nearest value in MXFP4 LUT
            indices = np.argmin(np.abs(MXFP4_LUT[np.newaxis, :] - scaled[:, np.newaxis]), axis=1)
            indices = indices.astype(np.uint8)

            # Pack nibbles: 2 per byte, low first
            nibbles = (indices[1::2] << 4) | indices[0::2]
            scale_e8m0 = s

        nibbles_list.append(nibbles)
        scales_list.append(np.uint8(scale_e8m0))

    return np.concatenate(nibbles_list), np.array(scales_list, dtype=np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
# 1BP File Writer
# ═══════════════════════════════════════════════════════════════════════════

def f32b(v):
    """float32 → upper 16 bits (BF16-like for Q4NX scales)."""
    return np.float32(v).view(np.uint32) >> 16

def quant_tile_q4nx(data, tr=32, tc=256, gs=32):
    """Q4NX tile quantization: asymmetric 4-bit with bf16 scales.
    Returns raw bytes for one tile.
    """
    r, c = data.shape
    pr, pc = tr, tc
    grps = pc // gs
    padded = np.zeros((pr, pc), dtype=np.float32)
    padded[:r, :c] = data

    grouped = padded.reshape(pr, grps, gs)
    mn = grouped.min(axis=2)
    mx = grouped.max(axis=2)
    rng = mx - mn
    flat_range = rng < 1e-10
    scale = np.where(flat_range, 0.0, rng / 15.0)
    zp_mn = np.where(flat_range, 0.0, mn)
    flat_scale = scale < 1e-10
    scale = np.where(flat_scale, 1.0, scale).astype(np.float32)
    zp_mn = np.where(flat_scale, 0.0, zp_mn).astype(np.float32)

    sc = f32b(scale).astype(np.uint16)
    zp = f32b(zp_mn).astype(np.uint16)
    inv = 1.0 / scale
    qi = np.clip(np.round((grouped - zp_mn[:, :, None]) * inv[:, :, None]), 0, 15).astype(np.uint8)
    qi_flat = qi.reshape(pr, pc)
    pk = (qi_flat[:, 1::2] << 4) | qi_flat[:, 0::2]
    return sc.tobytes() + zp.tobytes() + pk.tobytes()

def quant_tile_tq2(data, tr=32, tc=256, gs=32):
    """TQ2 symmetric ternary: -scale, 0, +scale, packed 2-bit."""
    r, c = data.shape
    pr, pc = tr, tc
    grps = pc // gs
    padded = np.zeros((pr, pc), dtype=np.float32)
    padded[:r, :c] = data
    grouped = padded.reshape(pr, grps, gs)
    mx = np.abs(grouped).max(axis=2)
    scale = np.where(mx < 1e-10, 1.0, mx).astype(np.float32)
    sc = f32b(scale).astype(np.uint16)
    inv = 1.0 / scale
    signed = np.clip(np.round(grouped * inv[:, :, None]), -1, 1).astype(np.int8)
    code = (signed + 1).astype(np.uint8).reshape(pr, pc)
    c0, c1, c2, c3 = code[:, 0::4], code[:, 1::4], code[:, 2::4], code[:, 3::4]
    pk = (c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)).astype(np.uint8)
    return sc.tobytes() + pk.tobytes()

def quant_tile_mxfp4(data, tr=32, tc=256, gs=32):
    """MXFP4 tile: block-scaled FP4 (E2M1)."""
    r, c = data.shape
    pr, pc = tr, tc
    padded = np.zeros((pr, pc), dtype=np.float32)
    padded[:r, :c] = data
    # Process per row
    result = b""
    for row in range(pr):
        row_data = padded[row, :]
        nibbles, scales = quant_to_mxfp4(row_data, gs)
        result += nibbles.tobytes() + scales.tobytes()
    return result

def quant_tile_f16(data, tr=32, tc=256):
    """F16 tile: no quantization, just store as float16."""
    r, c = data.shape
    pr, pc = tr, tc
    padded = np.zeros((pr, pc), dtype=np.float32)
    padded[:r, :c] = data
    return padded.astype(np.float16).tobytes()

def tiled_size(rows, cols, quant, tr=32, tc=256, gs=32):
    """Compute tiled size in bytes for a given quantization."""
    ntr = (rows + tr - 1) // tr
    ntc = (cols + tc - 1) // tc
    if quant == ONEBP_Q4NX:
        # scales + zero_points + packed int4
        tile_b = tr * (tc // gs) * 4 + tr * tc // 2
    elif quant == ONEBP_TQ2:
        # scales + packed 2-bit
        tile_b = tr * (tc // gs) * 2 + tr * tc // 4
    elif quant == ONEBP_TQ1:
        tq1_grps = (tc + 4) // 5
        tile_b = tr * tq1_grps * 2 + tr * tq1_grps
    elif quant == ONEBP_MXFP4:
        # MXFP4: 16 nibble bytes + 1 scale byte per group
        n_grps = tc // gs
        tile_b = tr * (n_grps * 16 + n_grps * 1)
    elif quant == ONEBP_F16:
        tile_b = tr * tc * 2
    else:  # F32 default
        tile_b = tr * tc * 4
    return ntr * ntc * tile_b

def quant_tile(data, quant):
    """Dispatch to correct quantizer based on quant type."""
    if quant == ONEBP_Q4NX:
        return quant_tile_q4nx(data)
    elif quant == ONEBP_TQ2:
        return quant_tile_tq2(data)
    elif quant == ONEBP_MXFP4:
        return quant_tile_mxfp4(data)
    elif quant == ONEBP_F16:
        return quant_tile_f16(data)
    else:
        return data.astype(np.float32).tobytes()


# ═══════════════════════════════════════════════════════════════════════════
# Safetensors Loader
# ═══════════════════════════════════════════════════════════════════════════

def load_safetensors(path_or_dir):
    """Load safetensors files from a path or directory.
    Returns dict of tensor_name → numpy array.
    """
    import safetensors
    from safetensors import safe_open

    if os.path.isdir(path_or_dir):
        # Load index file if it exists
        index_path = os.path.join(path_or_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            # Group by file
            file_tensors = {}
            for name, filename in weight_map.items():
                filepath = os.path.join(path_or_dir, filename)
                if filepath not in file_tensors:
                    file_tensors[filepath] = []
                file_tensors[filepath].append(name)

            # Load all files
            tensors = {}
            for filepath, names in file_tensors.items():
                if os.path.exists(filepath):
                    with safe_open(filepath, framework="np") as f:
                        for name in names:
                            tensors[name] = f.get_tensor(name)
            return tensors
        else:
            # Load all .safetensors files in the directory
            safetensor_files = sorted([
                os.path.join(path_or_dir, f)
                for f in os.listdir(path_or_dir)
                if f.endswith(".safetensors")
            ])
            tensors = {}
            for sf in safetensor_files:
                with safe_open(sf, framework="np") as f:
                    for key in f.keys():
                        tensors[key] = f.get_tensor(key)
            return tensors
    elif os.path.isfile(path_or_dir):
        tensors = {}
        with safe_open(path_or_dir, framework="np") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        return tensors
    else:
        raise FileNotFoundError(f"Not found: {path_or_dir}")

def load_config(path_or_dir):
    """Load config.json from model directory."""
    config_path = os.path.join(path_or_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {}


# ═══════════════════════════════════════════════════════════════════════════
# Architecture Detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_architecture(config, tensors):
    """Determine architecture type from config and tensor names."""
    arch_str = config.get("architectures", [""])[0] if isinstance(config.get("architectures"), list) else config.get("architectures", "")
    model_type = config.get("model_type", "")

    # Check tensor names for clues
    tensor_names = list(tensors.keys())
    has_kda = any("kda" in k.lower() for k in tensor_names)
    has_mla = any("mla" in k.lower() or "kv_a" in k.lower() for k in tensor_names)
    has_moe = any("expert" in k.lower() or "moe" in k.lower() or "router" in k.lower() for k in tensor_names)
    has_vision = any("vision" in k.lower() or "vit" in k.lower() or "patch_embed" in k.lower() for k in tensor_names)
    has_moonvit = any("moonvit" in k.lower() or "moon_vit" in k.lower() for k in tensor_names)
    has_attnres = any("attnres" in k.lower() or "attn_res" in k.lower() for k in tensor_names)
    total_params = sum(t.size for t in tensors.values())

    # Detect Kimi K3 (2.8T)
    if has_kda and has_attnres and total_params > 1e12:
        return "kimi_k3"

    # Detect Moonlight (Moonlight-16B-A3B / Kimi-VL base)
    if has_mla and has_moe and "moonlight" in model_type.lower() or "kimi" in model_type.lower() and not has_kda:
        return "moonlight"

    # Detect Kimi-VL variant
    if has_moonvit:
        return "kimi_vl"

    # Fallback: check config
    if "kimi" in model_type.lower() or "kimi" in arch_str.lower():
        if "vl" in model_type.lower():
            return "kimi_vl"
        return "kimi_k3" if "k3" in model_type.lower() else "moonlight"

    return config.get("model_type", "unknown")


# ═══════════════════════════════════════════════════════════════════════════
# 1BP Header Builder
# ═══════════════════════════════════════════════════════════════════════════

def build_header(config, tensors, arch_str, quant_type, tile_rows=32, tile_cols=256, group_size=32):
    """Build a 1BP header from config and architecture info."""
    hdr = bytearray(256)
    struct.pack_into("<I", hdr, 0, ONEBP_MAGIC)
    struct.pack_into("<I", hdr, 4, ONEBP_VERSION)

    arch_enum = ARCH_MAP.get(arch_str, ONEBP_DENSE)
    struct.pack_into("<I", hdr, 8, arch_enum)
    struct.pack_into("<I", hdr, 12, QUANT_MAP.get(quant_type, ONEBP_Q4NX))

    # Read dimensions from config with fallbacks
    hs = config.get("hidden_size", config.get("d_model", 0))
    nl = config.get("num_hidden_layers", config.get("num_layers", 0))
    nh = config.get("num_attention_heads", config.get("num_heads", 0))
    nkv = config.get("num_key_value_heads", config.get("num_kv_heads", nh))
    hd = config.get("head_dim", 0)
    im = config.get("intermediate_size", config.get("intermediate_size", 0))
    vs = config.get("vocab_size", 0)
    msl = config.get("max_position_embeddings", config.get("max_seq_len", 2048))

    # If head_dim not in config, compute from hidden_size / num_heads
    if hd == 0 and nh > 0 and hs > 0:
        hd = hs // nh

    # Detect from tensors if config is empty
    if hs == 0:
        for name, tensor in tensors.items():
            if "embed_tokens" in name or "embedding" in name:
                if len(tensor.shape) >= 2:
                    hs = tensor.shape[-1]
                    vs = tensor.shape[-2]
                    break
    if nh == 0:
        for name, tensor in tensors.items():
            if "q_proj" in name and "weight" in name and len(tensor.shape) >= 2:
                # Q projection: [H, NH * HD] typically
                q_dim = tensor.shape[0]
                for k_name, k_tensor in tensors.items():
                    if "k_proj" in name.replace("q", "k") and len(k_tensor.shape) >= 2:
                        kv_dim = k_tensor.shape[0]
                        if hs > 0 and q_dim > 0:
                            nh = q_dim // (q_dim // (kv_dim // (nkv if nkv > 0 else 1)) if hs > 0 else 1)
                        break
                break

    struct.pack_into("<i", hdr, 16, hs)
    struct.pack_into("<i", hdr, 20, nl)
    struct.pack_into("<i", hdr, 24, nh if nh > 0 else 1)
    struct.pack_into("<i", hdr, 28, nkv if nkv > 0 else nh)
    struct.pack_into("<i", hdr, 32, hd if hd > 0 else (hs // nh if nh > 0 else 128))
    struct.pack_into("<i", hdr, 36, im)
    struct.pack_into("<i", hdr, 40, vs)
    struct.pack_into("<i", hdr, 44, msl)
    struct.pack_into("<I", hdr, 48, tile_rows)
    struct.pack_into("<I", hdr, 52, tile_cols)
    struct.pack_into("<I", hdr, 56, group_size)

    # Quantization flags
    has_q = 1 if config.get("q_norm", config.get("has_q_norm", False)) else 0
    has_k = 1 if config.get("k_norm", config.get("has_k_norm", False)) else 0
    has_bias = 1 if config.get("bias", config.get("has_bias", False)) else 0
    struct.pack_into("<I", hdr, 60, has_q)
    struct.pack_into("<I", hdr, 64, has_k)
    struct.pack_into("<I", hdr, 68, has_bias)

    # RoPE theta (stored as fixed-point * 1000)
    rope_theta = config.get("rope_theta", config.get("rope.theta", 10000.0))
    struct.pack_into("<I", hdr, 72, int(rope_theta * 1000))

    # Token IDs
    struct.pack_into("<i", hdr, 76, config.get("bos_token_id", 1))
    struct.pack_into("<i", hdr, 80, config.get("eos_token_id", 2))

    # ─── Architecture-specific fields ───────────────────────────
    if arch_str in ("kimi_k3", "moonlight", "kimi_vl"):
        # MoE config
        n_exp = config.get("num_experts", config.get("moe_num_experts", 256))
        n_used = config.get("num_experts_per_tok", config.get("top_k", 8))
        n_ff_exp = config.get("intermediate_size", im)
        n_ff_shexp = config.get("shared_expert_intermediate_size", n_ff_exp)

        struct.pack_into("<I", hdr, 84, n_exp)
        struct.pack_into("<I", hdr, 88, n_used)
        struct.pack_into("<I", hdr, 92, n_ff_exp)
        struct.pack_into("<I", hdr, 96, n_ff_shexp)

        # K3 specific: KDA config
        if arch_str == "kimi_k3":
            n_kda = config.get("num_kda_layers", 69)
            n_mla = config.get("num_mla_layers", 24)
            kda_dim = config.get("kda_latent_dim", 3584)
            struct.pack_into("<i", hdr, 100, n_kda)     # reserved[0]
            struct.pack_into("<i", hdr, 104, n_mla)     # reserved[1]
            # KDA/VL specific in reserved fields
            struct.pack_into("<i", hdr, 108, kda_dim)   # reserved[2]

    # Vision config
    if arch_str == "kimi_vl":
        v_hs = config.get("vision_hidden_size", 1024)
        struct.pack_into("<i", hdr, 112, v_hs)

    # Model tag
    tag = config.get("_name_or_path", config.get("model_type", arch_str))
    tag_bytes = tag.encode("utf-8")[:63]
    hdr[192:192+len(tag_bytes)] = tag_bytes

    return hdr


# ═══════════════════════════════════════════════════════════════════════════
# Tensor Name Mapping (Moonshot → 1BP conventions)
# ═══════════════════════════════════════════════════════════════════════════

def map_tensor_name(hf_name):
    """Map HuggingFace/safetensors tensor names to 1BP canonical names.
    Returns (name, skip_weight) where skip_weight=True for non-weight tensors.
    """
    # Skip non-weight tensors
    if any(s in hf_name for s in ["grad", "adam", "momentum", "optimizer"]):
        return None, True

    # Moonshot/Kimi naming conventions
    replacements = [
        ("model.layers.", "blk."),
        ("model.", ""),
        ("self_attn.", "attn."),
        ("mlp.", "ffn."),
        ("input_layernorm", "rms_attn_w"),
        ("post_attention_layernorm", "rms_ffn_w"),
        ("q_proj.weight", "attn_q.weight"),
        ("k_proj.weight", "attn_k.weight"),
        ("v_proj.weight", "attn_v.weight"),
        ("o_proj.weight", "attn_o.weight"),
        ("gate_proj.weight", "ffn_gate.weight"),
        ("up_proj.weight", "ffn_up.weight"),
        ("down_proj.weight", "ffn_down.weight"),
        ("embed_tokens.weight", "token_embd.weight"),
        ("norm.weight", "output_norm.weight"),
        ("lm_head.weight", "output.weight"),
    ]

    name = hf_name
    for old, new in replacements:
        if old in name:
            name = name.replace(old, new)
            break

    return name, (".weight" not in hf_name and "_proj" not in hf_name and
                  "embed" not in hf_name and "norm" not in hf_name and
                  "lm_head" not in hf_name)


# ═══════════════════════════════════════════════════════════════════════════
# Main Conversion
# ═══════════════════════════════════════════════════════════════════════════

def convert(input_path, output_path, arch_str=None, quant_str="Q4NX", verbose=True):
    """Convert a HuggingFace model to 1BP format."""
    t_start = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  HuggingFace → 1BP Converter")
        print(f"{'='*60}")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")

    # Load config
    config = load_config(input_path)
    if verbose and config:
        mt = config.get("model_type", "unknown")
        print(f"  Model type: {mt}")

    # Load tensors
    if verbose:
        print(f"  Loading tensors...")
    tensors = load_safetensors(input_path)
    if verbose:
        print(f"  Loaded {len(tensors)} tensors ({sum(t.size for t in tensors.values()):,} params)")

    # Detect architecture
    if arch_str is None:
        arch_str = detect_architecture(config, tensors)
        if verbose:
            print(f"  Detected architecture: {arch_str}")
    else:
        if verbose:
            print(f"  Using architecture: {arch_str}")

    # Determine quantization
    quant_type = QUANT_MAP.get(quant_str, ONEBP_Q4NX)

    # Build header
    hdr = build_header(config, tensors, arch_str, quant_str)

    # Compute tensor index
    tile_rows, tile_cols, group_size = 32, 256, 32
    tensor_entries = []
    data_offset = 256  # start after header

    valid_names = {}
    for hf_name, tensor in tensors.items():
        mapped_name, skip = map_tensor_name(hf_name)
        if skip or mapped_name is None:
            continue
        if len(tensor.shape) < 2:
            if verbose >= 2:
                print(f"  Skipping 1D tensor: {hf_name} ({tensor.shape})")
            continue
        rows, cols = tensor.shape[-2], tensor.shape[-1]
        tsize = tiled_size(rows, cols, quant_type, tile_rows, tile_cols, group_size)
        tensor_entries.append((mapped_name, rows, cols, data_offset, tsize, tensor))
        valid_names[mapped_name] = hf_name
        data_offset += tsize

    # Count unique layers
    layer_ids = set()
    for name in valid_names:
        import re
        m = re.search(r'blk\.(\d+)', name)
        if m:
            layer_ids.add(int(m.group(1)))
    n_layers = max(layer_ids) + 1 if layer_ids else config.get("num_hidden_layers", 0)

    if verbose:
        print(f"  Tensors to convert: {len(tensor_entries)}")
        print(f"  Layers detected: {n_layers}")
        print(f"  Data size: {data_offset / (1024*1024):.1f} MB")
        print(f"  Quantization: {quant_str}")

    # Write output file
    with open(output_path, "wb") as f:
        # Write header (placeholder)
        f.write(hdr)

        # Write tensor index
        for name, rows, cols, offset, tsize, _ in tensor_entries:
            name_bytes = name.encode("utf-8")
            name_len = min(len(name_bytes), 63)
            f.write(struct.pack("<I", name_len))
            f.write(name_bytes[:name_len])
            f.write(b"\0")
            f.write(struct.pack("<I", 2))  # ndim
            f.write(struct.pack("<II", rows, cols))
            f.write(struct.pack("<Q", offset))
            f.write(struct.pack("<Q", tsize))

        print(f"  Index written, tensor_count={len(tensor_entries)}")

        # Update tensor count in header
        hdr_updated = bytearray(hdr)
        struct.pack_into("<I", hdr_updated, 84, len(tensor_entries))

        # Write quantized tile data
        quant_start = time.time()
        for idx, (name, rows, cols, offset, tsize, tensor) in enumerate(tensor_entries):
            t0 = time.time()

            # Convert to float32 (handles BF16, FP16, INT8, etc.)
            data = tensor.astype(np.float32)

            # Tile and quantize
            ntr = (rows + tile_rows - 1) // tile_rows
            ntc = (cols + tile_cols - 1) // tile_cols

            for tr_idx in range(ntr):
                for tc_idx in range(ntc):
                    r0 = tr_idx * tile_rows
                    c0 = tc_idx * tile_cols
                    rh = min(tile_rows, rows - r0)
                    cw = min(tile_cols, cols - c0)
                    tile_data = data[r0:r0+rh, c0:c0+cw]
                    tile_bytes = quant_tile(tile_data, quant_type)
                    # Pad to full tile size if needed
                    expected = tiled_size(rh, cw, quant_type, tile_rows, tile_cols, group_size)
                    if len(tile_bytes) < expected:
                        tile_bytes += b"\0" * (expected - len(tile_bytes))
                    f.write(tile_bytes)

            dt = time.time() - t0
            if verbose:
                mb = tsize / (1024 * 1024)
                print(f"  [{idx+1}/{len(tensor_entries)}] {name:50s} {rows:5d}×{cols:<5d} → {tsize/1024:8.1f} KB ({dt:.2f}s)")

        quant_time = time.time() - quant_start

    # Rewrite header with correct tensor count
    with open(output_path, "r+b") as f:
        f.write(hdr_updated)

    total = time.time() - t_start
    fsize = os.path.getsize(output_path)
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Conversion complete!")
        print(f"  Output: {output_path} ({fsize / (1024*1024):.1f} MB)")
        print(f"  Quantization: {quant_time:.1f}s")
        print(f"  Total time: {total:.1f}s")
        print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace safetensors models to 1BP format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tools/hf_to_onebp.py --repo moonshotai/Moonlight-16B-A3B -o models/Moonlight.1bp
  python3 tools/hf_to_onebp.py -i ./models/Moonlight-16B-A3B -o models/Moonlight.1bp --arch moonlight
  python3 tools/hf_to_onebp.py -i ./models/Kimi-VL -o models/Kimi-VL.1bp --quant TQ2 --arch kimi_vl
        """
    )
    parser.add_argument("-i", "--input", help="Input path (local directory or file)")
    parser.add_argument("--repo", help="HuggingFace repo name (downloads first)")
    parser.add_argument("-o", "--output", required=True, help="Output .1bp file path")
    parser.add_argument("--arch", choices=list(ARCH_MAP.keys()) + [None],
                        default=None, help="Architecture type (auto-detect if omitted)")
    parser.add_argument("--quant", choices=list(QUANT_MAP.keys()), default="Q4NX",
                        help="Quantization type (default: Q4NX)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")
    args = parser.parse_args()

    if args.repo:
        from huggingface_hub import snapshot_download
        print(f"Downloading {args.repo}...")
        local_path = snapshot_download(args.repo, resume_download=True)
        print(f"Downloaded to {local_path}")
        input_path = local_path
    elif args.input:
        input_path = args.input
    else:
        parser.print_help()
        sys.exit(1)

    convert(input_path, args.output, args.arch, args.quant, verbose=not args.quiet)


if __name__ == "__main__":
    main()

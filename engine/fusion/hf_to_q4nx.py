#!/usr/bin/env python3
"""
HF → Q4NX Converter

Reads any HuggingFace model from a directory (config.json + *.safetensors),
auto-detects architecture, quantizes projection weights to 4-bit block format,
and writes a model.q4nx file suitable for the 1bit fused inference engine.

Usage:
    python3 hf_to_q4nx.py /path/to/hf/model/dir [--output model.q4nx] [--verbose]
"""

# ── Fix Python path: remove fusion dir so tokenize.py doesn't shadow stdlib ──
import sys
import os as _os
_fusion_dir = _os.path.dirname(_os.path.abspath(__file__))
if _fusion_dir in sys.path:
    sys.path.remove(_fusion_dir)
# Re-add fusion dir at the end so sub-modules can still be found
sys.path.append(_fusion_dir)

import argparse
import glob
import json
import math
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np

_SAFETENSORS_AVAILABLE = False
safe_open = None
st_save_file = None

try:
    from safetensors import safe_open as _so
    safe_open = _so
    _SAFETENSORS_AVAILABLE = True
except ImportError:
    pass

# safetensors.torch may fail if torch is broken; that's OK, we don't need it
try:
    from safetensors.torch import save_file as _sf
    st_save_file = _sf
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════
# Architecture detection
# ═══════════════════════════════════════════════════════════════

ARCH_CONFIGS = {
    "qwen3_0_6b": {
        "H": 1536, "NC": 28, "NH": 12, "NKV": 2, "HD": 128,
        "IM": 4096, "NV": 151936, "max_seq_len": 4096,
        "arch": "dense", "activation": "silu", "norm": "rmsnorm",
        "pos_encoding": "rope", "has_qk_norm": True,
        "has_shared_experts": False, "n_experts": 0, "top_k": 0,
        "expert_inter_size": 0,
    },
    "qwen3_1_5b": {
        "H": 2048, "NC": 28, "NH": 16, "NKV": 2, "HD": 128,
        "IM": 8192, "NV": 151936, "max_seq_len": 4096,
        "arch": "dense", "activation": "silu", "norm": "rmsnorm",
        "pos_encoding": "rope", "has_qk_norm": True,
        "has_shared_experts": False, "n_experts": 0, "top_k": 0,
        "expert_inter_size": 0,
    },
    "qwen3_7b": {
        "H": 4096, "NC": 32, "NH": 32, "NKV": 8, "HD": 128,
        "IM": 16384, "NV": 151936, "max_seq_len": 8192,
        "arch": "dense", "activation": "silu", "norm": "rmsnorm",
        "pos_encoding": "rope", "has_qk_norm": True,
        "has_shared_experts": False, "n_experts": 0, "top_k": 0,
        "expert_inter_size": 0,
    },
    "qwen3_14b": {
        "H": 5120, "NC": 40, "NH": 40, "NKV": 8, "HD": 128,
        "IM": 20480, "NV": 152064, "max_seq_len": 8192,
        "arch": "dense", "activation": "silu", "norm": "rmsnorm",
        "pos_encoding": "rope", "has_qk_norm": True,
        "has_shared_experts": False, "n_experts": 0, "top_k": 0,
        "expert_inter_size": 0,
    },
    "qwen2_5_7b": {
        "H": 4096, "NC": 28, "NH": 32, "NKV": 8, "HD": 128,
        "IM": 11008, "NV": 152064, "max_seq_len": 4096,
        "arch": "dense", "activation": "silu", "norm": "rmsnorm",
        "pos_encoding": "rope", "has_qk_norm": False,
        "has_shared_experts": False, "n_experts": 0, "top_k": 0,
        "expert_inter_size": 0,
    },
    "qwen2_5_32b": {
        "H": 5120, "NC": 64, "NH": 40, "NKV": 8, "HD": 128,
        "IM": 20480, "NV": 152064, "max_seq_len": 8192,
        "arch": "dense", "activation": "silu", "norm": "rmsnorm",
        "pos_encoding": "rope", "has_qk_norm": False,
        "has_shared_experts": False, "n_experts": 0, "top_k": 0,
        "expert_inter_size": 0,
    },
    "llama3_1_8b": {
        "H": 4096, "NC": 32, "NH": 32, "NKV": 8, "HD": 128,
        "IM": 14336, "NV": 128256, "max_seq_len": 8192,
        "arch": "dense", "activation": "silu", "norm": "rmsnorm",
        "pos_encoding": "rope", "has_qk_norm": False,
        "has_shared_experts": False, "n_experts": 0, "top_k": 0,
        "expert_inter_size": 0,
    },
    "llama3_2_1b": {
        "H": 2048, "NC": 16, "NH": 16, "NKV": 8, "HD": 64,
        "IM": 8192, "NV": 128256, "max_seq_len": 8192,
        "arch": "dense", "activation": "silu", "norm": "rmsnorm",
        "pos_encoding": "rope", "has_qk_norm": False,
        "has_shared_experts": False, "n_experts": 0, "top_k": 0,
        "expert_inter_size": 0,
    },
    "llama3_2_3b": {
        "H": 3072, "NC": 28, "NH": 24, "NKV": 8, "HD": 128,
        "IM": 8192, "NV": 128256, "max_seq_len": 8192,
        "arch": "dense", "activation": "silu", "norm": "rmsnorm",
        "pos_encoding": "rope", "has_qk_norm": False,
        "has_shared_experts": False, "n_experts": 0, "top_k": 0,
        "expert_inter_size": 0,
    },
    "gemma2_2b": {
        "H": 2304, "NC": 26, "NH": 18, "NKV": 2, "HD": 128,
        "IM": 9216, "NV": 256128, "max_seq_len": 8192,
        "arch": "dense", "activation": "gelu", "norm": "rmsnorm",
        "pos_encoding": "rope", "has_qk_norm": True,
        "has_shared_experts": False, "n_experts": 0, "top_k": 0,
        "expert_inter_size": 0,
    },
    "gemma2_9b": {
        "H": 3584, "NC": 42, "NH": 16, "NKV": 8, "HD": 256,
        "IM": 14336, "NV": 256128, "max_seq_len": 8192,
        "arch": "dense", "activation": "gelu", "norm": "rmsnorm",
        "pos_encoding": "rope", "has_qk_norm": True,
        "has_shared_experts": False, "n_experts": 0, "top_k": 0,
        "expert_inter_size": 0,
    },
    "deepseek_v2_lite": {
        "H": 2048, "NC": 27, "NH": 16, "NKV": 2, "HD": 128,
        "IM": 1536, "NV": 102400, "max_seq_len": 4096,
        "arch": "deepseek_moe", "activation": "silu", "norm": "rmsnorm",
        "pos_encoding": "rope", "has_qk_norm": False,
        "has_shared_experts": True, "n_experts": 64, "top_k": 6,
        "expert_inter_size": 1536,
    },
    "deepseek_v3": {
        "H": 7168, "NC": 61, "NH": 56, "NKV": 8, "HD": 128,
        "IM": 2048, "NV": 129280, "max_seq_len": 8192,
        "arch": "deepseek_moe", "activation": "silu", "norm": "rmsnorm",
        "pos_encoding": "rope", "has_qk_norm": False,
        "has_shared_experts": True, "n_experts": 256, "top_k": 8,
        "expert_inter_size": 2048,
    },
    "mixtral_8x7b": {
        "H": 4096, "NC": 32, "NH": 32, "NKV": 8, "HD": 128,
        "IM": 14336, "NV": 32000, "max_seq_len": 32768,
        "arch": "moe", "activation": "silu", "norm": "rmsnorm",
        "pos_encoding": "rope", "has_qk_norm": False,
        "has_shared_experts": False, "n_experts": 8, "top_k": 2,
        "expert_inter_size": 14336,
    },
    "zaya1_8b": {
        "H": 4096, "NC": 40, "NH": 32, "NKV": 8, "HD": 128,
        "IM": 14336, "NV": 151936, "max_seq_len": 4096,
        "arch": "moe", "activation": "silu", "norm": "rmsnorm",
        "pos_encoding": "rope", "has_qk_norm": False,
        "has_shared_experts": False, "n_experts": 8, "top_k": 2,
        "expert_inter_size": 14336,
    },
}


def detect_arch(config: dict) -> str:
    """Detect the model tag from config.json contents."""
    archs = config.get("architectures", [])
    arch_type = archs[0] if archs else ""

    h = config.get("hidden_size", config.get("d_model", 0))
    nc = config.get("num_hidden_layers", config.get("num_layers", 0))
    nh = config.get("num_attention_heads", 0)
    nkv = config.get("num_key_value_heads", nh)
    hd = config.get("head_dim", 0)
    if hd == 0:
        hd = h // nh if nh > 0 else 128
    im = config.get("intermediate_size", 0)
    nv = config.get("vocab_size", 0)
    ne = config.get("num_local_experts", config.get("num_experts", 0))

    # ── Qwen3 ──
    if arch_type == "Qwen3ForCausalLM":
        # Standard HF Qwen3-0.6B has H=1024, IM=3072; a separate "NPU
        # variant" config used elsewhere in this repo has H=1536, IM=4096.
        # Both must be recognized -- this previously only checked for the
        # NPU variant (or H=1024 paired with IM=4096, which no real config
        # has), so converting an unmodified HF Qwen3-0.6B checkout fell
        # through every branch to the "qwen3_7b" catch-all at the bottom.
        if h == 1536 or (h == 1024 and im == 3072) or (h == 1024 and im == 4096):
            return "qwen3_0_6b"
        elif h == 2048 or (h == 1024 and im == 8192):
            return "qwen3_1_5b"
        elif h == 4096:
            if im == 16384:
                return "qwen3_7b"
            elif im == 11008:
                return "qwen2_5_7b"
            return "qwen3_7b"
        elif h == 5120:
            return "qwen3_14b"
        # Fallback dimension match
        for tag, cfg in ARCH_CONFIGS.items():
            if tag.startswith("qwen3") and cfg["H"] == h and cfg["NC"] == nc:
                return tag
        return "qwen3_7b"

    # ── Qwen2.5 ──
    if arch_type == "Qwen2.5ForCausalLM":
        if nv == 152064 and h == 5120 and nc >= 40:
            return "qwen2_5_32b"
        return "qwen2_5_7b"

    # ── Qwen2 (fallback to Qwen2.5) ──
    if arch_type == "Qwen2ForCausalLM":
        if ne > 0:
            return "deepseek_v2_lite"
        return "qwen2_5_7b"

    # ── Llama ──
    if arch_type == "LlamaForCausalLM":
        if hd == 64:
            return "llama3_2_1b"
        if h == 3072:
            return "llama3_2_3b"
        return "llama3_1_8b"

    # ── Gemma2 ──
    if arch_type == "Gemma2ForCausalLM":
        if h == 2304:
            return "gemma2_2b"
        elif h == 3584:
            return "gemma2_9b"
        return "gemma2_2b"

    # ── DeepSeek ──
    if arch_type in ("DeepseekV2ForCausalLM", "DeepseekForCausalLM"):
        if h <= 2048 and ne <= 64:
            return "deepseek_v2_lite"
        return "deepseek_v3"

    # ── Mixtral ──
    if arch_type == "MixtralForCausalLM":
        return "mixtral_8x7b"

    # ── model_type fallback ──
    mt = config.get("model_type", "")
    if mt == "gemma2":
        if h == 2304:
            return "gemma2_2b"
        elif h == 3584:
            return "gemma2_9b"
        return "gemma2_2b"

    # Generic: match by dimensions
    best = None
    best_score = -1
    for tag, cfg in ARCH_CONFIGS.items():
        score = 0
        if cfg["H"] == h: score += 1
        if cfg["NC"] == nc: score += 1
        if cfg["NH"] == nh: score += 1
        if cfg["NKV"] == nkv: score += 1
        if cfg["HD"] == hd: score += 1
        if cfg["IM"] == im: score += 1
        if ne and cfg.get("n_experts", 0) == ne: score += 2
        if score > best_score:
            best_score = score
            best = tag

    return best if best else "unknown"


# ═══════════════════════════════════════════════════════════════
# BF16 ↔ f32 conversion
# ═══════════════════════════════════════════════════════════════

def f32_to_bf16(f: float) -> int:
    """Convert float32 to bfloat16 (uint16), rounding to nearest even."""
    bits = struct.pack("f", f)
    i32 = struct.unpack("I", bits)[0]
    rounding_bias = ((i32 >> 16) & 1) + 0x7FFF
    return (i32 + rounding_bias) >> 16


def bf16_to_f32(v: int) -> float:
    """Convert bfloat16 (uint16) to float32."""
    bits = struct.pack("I", v << 16)
    return struct.unpack("f", bits)[0]


# ═══════════════════════════════════════════════════════════════
# I8 tile quantization
# ═══════════════════════════════════════════════════════════════

TILE_R = 32   # tile rows
TILE_C = 256  # tile columns
TILE_BYTES = 5120  # total bytes per tile


def quantize_block(block_f32: np.ndarray) -> bytes:
    """
    Quantize a 32×256 block of f32 values to I8 tile format.

    Returns 5120 bytes with:
      [0..512)     = 256 BF16 scales (uint16), indexed [g*32 + lr]
      [512..1024)  = 256 BF16 zero-points, indexed [g*32 + lr]
      [1024..5120) = 4096 bytes packed I4 data

    The block is divided into 8 groups of 32 columns each.
    Per group, each row has its own scale and zero-point.
    """
    assert block_f32.shape == (32, 256), f"Expected (32,256), got {block_f32.shape}"

    scales = [0] * 256
    zps = [0] * 256
    packed = bytearray(4096)

    for g in range(8):  # 8 groups of 32 columns
        g_start = g * 32
        g_end = g_start + 32
        group = block_f32[:, g_start:g_end]  # (32, 32)

        for lr in range(32):  # per-row min-max calibration
            row_vals = group[lr, :]  # 32 values

            min_val = float(np.min(row_vals))
            max_val = float(np.max(row_vals))

            # Symmetric quantization to 4-bit (0..15)
            if max_val - min_val < 1e-10:
                scale = 1.0
                zero_point = min_val
                qvals = [0] * 32
            else:
                scale = (max_val - min_val) / 15.0
                zero_point = min_val
                qvals = np.round((row_vals - zero_point) / scale).astype(np.uint8)
                qvals = np.clip(qvals, 0, 15).tolist()

            scales[g * 32 + lr] = f32_to_bf16(scale)
            zps[g * 32 + lr] = f32_to_bf16(zero_point)

            # Pack 4-bit values into bytes
            lane = lr // 16
            lr2 = lr % 16
            bi = lr2 // 2
            ns = lr % 2

            for c in range(32):
                idx = lane * 2048 + c * 8 + bi
                nibble = int(qvals[c])
                if ns == 0:
                    packed[idx] = (packed[idx] & 0xF0) | nibble
                else:
                    packed[idx] = (packed[idx] & 0x0F) | (nibble << 4)

    # Build output: 5120 bytes
    out = bytearray(5120)
    # Scales as uint16 (512 bytes)
    out[0:512] = struct.pack('<256H', *scales)
    # Zero-points as uint16 (512 bytes)
    out[512:1024] = struct.pack('<256H', *zps)
    # Packed data (4096 bytes)
    out[1024:5120] = bytes(packed)

    return bytes(out)


def quantize_i8_tiled(weight_f32: np.ndarray,
                      out_rows: int,
                      out_cols: int) -> bytes:
    """
    Quantize a weight matrix to I8 tiled format.

    Args:
        weight_f32: Full f32 weight matrix of shape [out_rows, out_cols]
        out_rows: Expected output dimension
        out_cols: Expected input dimension

    Returns:
        Concatenated 5120-byte tiles
    """
    ntc = (out_cols + TILE_C - 1) // TILE_C
    ntr = (out_rows + TILE_R - 1) // TILE_R

    all_tiles = bytearray()

    for tr in range(ntr):
        for tc in range(ntc):
            r_start = tr * TILE_R
            r_end = min(r_start + TILE_R, out_rows)
            c_start = tc * TILE_C
            c_end = min(c_start + TILE_C, out_cols)

            # Extract tile block (pad with zeros if partial)
            tile_block = np.zeros((TILE_R, TILE_C), dtype=np.float32)
            tile_block[:r_end - r_start, :c_end - c_start] = \
                weight_f32[r_start:r_end, c_start:c_end]

            tile_bytes = quantize_block(tile_block)
            all_tiles.extend(tile_bytes)

    return bytes(all_tiles)


# ═══════════════════════════════════════════════════════════════
# Q4NX file writer
# ═══════════════════════════════════════════════════════════════

def is_projection_weight(name: str) -> bool:
    """Check if a tensor name corresponds to a projection weight."""
    projection_patterns = [
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
        ".self_attn.o_proj.weight",
        ".mlp.gate_proj.weight",
        ".mlp.up_proj.weight",
        ".mlp.down_proj.weight",
    ]
    if "mlp.experts" in name or "moe" in name:
        return True
    for pat in projection_patterns:
        if name.endswith(pat):
            return True
    return False


def is_norm_or_embedding(name: str) -> bool:
    """Check if a tensor should be stored as BF16."""
    if name in (
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
        "model.final_norm.weight",
    ):
        return True
    if name.endswith("input_layernorm.weight"):
        return True
    if name.endswith("post_attention_layernorm.weight"):
        return True
    if name.endswith("self_attn.q_norm.weight"):
        return True
    if name.endswith("self_attn.k_norm.weight"):
        return True
    if "norm" in name and ".weight" in name:
        return True
    return False


def bf16_encode(np_tensor: np.ndarray) -> bytes:
    """Convert a numpy float32 tensor to raw BF16 bytes.

    Vectorized equivalent of `f32_to_bf16` applied elementwise (verified
    bit-identical against the scalar version, including inf/nan/extreme
    values) -- needed once --precision bf16 started routing full projection
    weight matrices (hundreds of millions of elements) through here instead
    of just the much smaller norm/embedding tensors.
    """
    arr = np.ascontiguousarray(np_tensor, dtype=np.float32).reshape(-1)
    i32 = arr.view(np.uint32).astype(np.uint64)
    rounding_bias = ((i32 >> np.uint64(16)) & np.uint64(1)) + np.uint64(0x7FFF)
    bf16 = ((i32 + rounding_bias) >> np.uint64(16)).astype(np.uint16)
    return bf16.tobytes()


def _read_safetensors_manual(path: str) -> dict:
    """
    Read a safetensors file manually, converting bfloat16 to float32.

    Returns dict of {name: (numpy_f32_array, dtype_str)}.
    """
    with open(path, "rb") as f:
        # 8 bytes: header size (u64 LE)
        hdr_size_bytes = f.read(8)
        if len(hdr_size_bytes) < 8:
            raise ValueError(f"{path}: too short")
        hdr_size = struct.unpack("<Q", hdr_size_bytes)[0]

        # Read JSON header
        hdr_json = f.read(hdr_size)
        if len(hdr_json) < hdr_size:
            raise ValueError(f"{path}: header truncated")

        # Parse header metadata
        try:
            metadata = json.loads(hdr_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}: invalid JSON header: {e}")

        result = {}
        data_start = 8 + hdr_size

        for key, info in metadata.items():
            if "dtype" not in info or "shape" not in info or "data_offsets" not in info:
                # Skip metadata-only entries
                continue

            dtype = info["dtype"]
            shape = tuple(info["shape"])
            offsets = info["data_offsets"]
            begin = offsets[0]
            end = offsets[1]
            nbytes = end - begin

            # Seek to data
            f.seek(data_start + begin)
            raw_bytes = f.read(nbytes)
            if len(raw_bytes) < nbytes:
                raise ValueError(f"{path}: tensor '{key}' truncated")

            if dtype == "BF16" or dtype == "bfloat16":
                # Convert bfloat16 (uint16) to float32. Vectorized (bf16->f32
                # is exact -- just left-shift into the high 16 bits of a u32
                # and reinterpret as f32, no rounding involved -- verified
                # against the scalar bf16_to_f32 across the full u16 range).
                # Needed once large weight tensors started flowing through
                # this reader too, not just small norm/embedding tensors.
                bf16_u16 = np.frombuffer(raw_bytes, dtype="<u2")
                np_tensor = (bf16_u16.astype(np.uint32) << np.uint32(16)).view(np.float32).reshape(shape).copy()
                result[key] = (np_tensor, "BF16")

            elif dtype == "F32" or dtype == "float32":
                n = nbytes // 4
                f32_vals = struct.unpack(f"<{n}f", raw_bytes)
                np_tensor = np.array(f32_vals, dtype=np.float32).reshape(shape)
                result[key] = (np_tensor, "F32")

            elif dtype == "F16" or dtype == "float16":
                # Convert float16 to float32
                n = nbytes // 2
                f16_bytes = np.frombuffer(raw_bytes, dtype=np.float16)
                f32_vals = f16_bytes.astype(np.float32)
                np_tensor = f32_vals.reshape(shape)
                result[key] = (np_tensor, "F16")

            elif dtype == "I8" or dtype == "int8":
                np_tensor = np.frombuffer(raw_bytes, dtype=np.int8).reshape(shape)
                result[key] = (np_tensor, "I8")

            elif dtype == "I16" or dtype == "int16":
                np_tensor = np.frombuffer(raw_bytes, dtype=np.int16).reshape(shape)
                result[key] = (np_tensor, "I16")

            elif dtype == "I32" or dtype == "int32":
                np_tensor = np.frombuffer(raw_bytes, dtype=np.int32).reshape(shape)
                result[key] = (np_tensor, "I32")

            elif dtype == "I64" or dtype == "int64":
                np_tensor = np.frombuffer(raw_bytes, dtype=np.int64).reshape(shape)
                result[key] = (np_tensor, "I64")

            else:
                # Unknown dtype: pass through as raw uint8
                np_tensor = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(shape)
                result[key] = (np_tensor, dtype)

        return result


def convert_hf_to_q4nx(
    model_dir: str,
    output_path: str,
    verbose: bool = False,
    precision: str = "i8",
) -> dict:
    """
    Convert a HuggingFace model directory to Q4NX format.

    precision: "i8" (default) quantizes projection weights (q/k/v/o_proj,
    gate/up/down_proj) to the 4-bit Q4NX tile format -- small, lossy.
    "bf16" stores them at the same near-lossless precision already used for
    norms/embeddings instead of quantizing them at all. Confirmed via
    tools/debug/reference_forward.py that "i8" introduces real, compounding
    error (embedding output matches an f32 HuggingFace reference exactly;
    by the last of 28 layers a Qwen3-0.6B model.q4nx's hidden-state RMS was
    ~30x the reference's) -- "bf16" trades ~4x file size for eliminating
    that as a source of divergence.
    """
    if verbose:
        print(f"📂 Reading model from: {model_dir}")

    # ── 1. Load config.json ──
    config_path = _os.path.join(model_dir, "config.json")
    if not _os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {model_dir}")

    with open(config_path, "r") as f:
        config = json.load(f)

    model_type = config.get("model_type", config.get("architectures", [""])[0])
    if verbose:
        print(f"📋 Model type: {model_type}")

    # ── 2. Detect architecture ──
    tag = detect_arch(config)
    if tag == "unknown":
        if verbose:
            print(f"⚠️  Unknown architecture, converting with generic settings")
        arch_cfg = None
    else:
        arch_cfg = ARCH_CONFIGS.get(tag)
        if verbose and arch_cfg:
            print(f"🏷️  Detected: {tag} "
                  f"(H={arch_cfg['H']} NC={arch_cfg['NC']} NH={arch_cfg['NH']} "
                  f"NKV={arch_cfg['NKV']} HD={arch_cfg['HD']} IM={arch_cfg['IM']})")

    # ── 3. Find all weight files ──
    safetensor_files = sorted(glob.glob(_os.path.join(model_dir, "*.safetensors")))
    bin_files = sorted(glob.glob(_os.path.join(model_dir, "*.bin")))

    use_safetensors = len(safetensor_files) > 0

    # Tensors are always read via _read_safetensors_manual() below (numpy has
    # no native bfloat16 dtype, so the safetensors package's own np/pt
    # loaders can't be used directly regardless) -- that reader is pure
    # struct/numpy and needs nothing from the `safetensors` package. The
    # `safe_open`/`_SAFETENSORS_AVAILABLE` import at the top of this file is
    # otherwise unused; gating weight-file selection on it here previously
    # made a broken or missing `safetensors` install silently discard all
    # *.safetensors files in favor of (usually nonexistent) *.bin ones.
    weight_files = safetensor_files if use_safetensors else bin_files

    if not weight_files:
        raise FileNotFoundError(
            f"No weight files found in {model_dir}. "
            f"Expected *.safetensors or *.bin"
        )

    if verbose:
        print(f"📦 Weight files ({len(weight_files)}):")
        for wf in weight_files:
            size_mb = _os.path.getsize(wf) / (1024 * 1024)
            print(f"   {_os.path.basename(wf)} ({size_mb:.1f} MB)")

    # ── 4. Load all tensors ──
    # Read safetensors manually since the library doesn't support
    # bfloat16 with framework='np' (numpy has no native bfloat16 dtype).
    all_tensors = {}
    tensor_data = {}

    if use_safetensors:
        for sf_path in weight_files:
            tensors = _read_safetensors_manual(sf_path)
            for key, (np_tensor, dtype_str) in tensors.items():
                all_tensors[key] = (dtype_str, np_tensor.shape)
                tensor_data[key] = np_tensor
    else:
        try:
            import torch
        except ImportError:
            raise ImportError(
                "Need PyTorch to load .bin files. Install: pip install torch"
            )

        for bin_path in weight_files:
            state = torch.load(bin_path, map_location="cpu", weights_only=True)
            for key, tensor in state.items():
                np_tensor = tensor.numpy()
                all_tensors[key] = (str(np_tensor.dtype), np_tensor.shape)
                tensor_data[key] = np_tensor

    if verbose:
        print(f"📊 Loaded {len(all_tensors)} tensors")
        norm_tensors = [n for n in all_tensors if is_norm_or_embedding(n)]
        proj_tensors = [n for n in all_tensors if is_projection_weight(n)]
        other = [n for n in all_tensors
                 if not is_norm_or_embedding(n) and not is_projection_weight(n)]
        print(f"   Norms/emb: {len(norm_tensors)}")
        print(f"   Projections: {len(proj_tensors)}")
        if other:
            print(f"   Other (skipped): {len(other)}")

    # ── 5. Build Q4NX data ──
    q4nx_tensors = {}
    current_offset = 0

    def tensor_sort_key(name):
        if is_norm_or_embedding(name):
            return (0, name)
        elif is_projection_weight(name):
            return (1, name)
        else:
            return (2, name)

    sorted_names = sorted(all_tensors.keys(), key=tensor_sort_key)

    for name in sorted_names:
        np_tensor = tensor_data[name]

        if is_norm_or_embedding(name):
            raw_bytes = bf16_encode(np_tensor)
            dtype = "BF16"
            shape = list(np_tensor.shape)
            data_size = len(raw_bytes)
            if verbose:
                print(f"   ✓ BF16: {name} {np_tensor.shape}")

        elif is_projection_weight(name):
            if np_tensor.ndim != 2:
                if verbose:
                    print(f"   ⚠️  Skipping non-2D: {name} {np_tensor.shape}")
                continue

            out_rows, out_cols = np_tensor.shape

            if precision == "bf16":
                raw_bytes = bf16_encode(np_tensor)
                dtype = "BF16"
                shape = list(np_tensor.shape)
                data_size = len(raw_bytes)
                if verbose:
                    print(f"   ✓ BF16: {name} [{out_rows}×{out_cols}] → {data_size} bytes")
            else:
                raw_bytes = quantize_i8_tiled(np_tensor, out_rows, out_cols)
                ntc = (out_cols + TILE_C - 1) // TILE_C
                ntr = (out_rows + TILE_R - 1) // TILE_R
                n_blocks = ntr * ntc
                dtype = "I8"
                shape = [n_blocks, TILE_BYTES]
                data_size = len(raw_bytes)

                if verbose:
                    print(f"   ✓ I8:   {name} [{out_rows}×{out_cols}] → "
                          f"{n_blocks} tiles x {TILE_BYTES}B = {data_size} bytes")

        else:
            if verbose:
                print(f"   ⚠️  Skipping: {name} {np_tensor.shape} (unknown type)")
            continue

        q4nx_tensors[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [current_offset, current_offset + data_size],
            "raw_bytes": raw_bytes,
        }
        current_offset += data_size

    # ── 6. Build JSON header ──
    header_dict = {}
    for name, info in q4nx_tensors.items():
        header_dict[name] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": info["data_offsets"],
        }

    header_json = json.dumps(header_dict, separators=(",", ":"))
    header_bytes = header_json.encode("utf-8")
    header_size = len(header_bytes)

    # ── 7. Write output file ──
    total = 8 + header_size + current_offset

    if verbose:
        print(f"\n💾 Writing: {output_path}")
        print(f"   Header: {header_size} bytes ({len(q4nx_tensors)} tensors)")
        print(f"   Data: {current_offset} bytes")
        print(f"   Total: {total} bytes ({total / (1024*1024):.1f} MB)")

    # File layout expected by engine/fusion/model_data.zig's loadModel():
    #   [4 bytes] magic 0x18 0x87 0x00 0x00
    #   [4 bytes] reserved flags (u32, currently unused -- 0)
    #   [n bytes] JSON header, starting at byte 8, length determined by
    #             brace-matching (no separate length-prefix field is read)
    #   [data]    tensor payloads, offsets in the JSON relative to the end
    #             of the JSON header
    # This previously wrote a `<Q` (8-byte LE) header_size field instead of
    # the magic+flags pair, which the loader would misread as `{ 0x15, 0x87,
    # 0, 0 }` (the low bytes of the real header_size) and reject with
    # InvalidMagic -- these two pieces of tooling had diverged.
    Q4NX_MAGIC = bytes([0x18, 0x87, 0x00, 0x00])
    with open(output_path, "wb") as f:
        f.write(Q4NX_MAGIC)
        f.write(struct.pack("<I", 0))  # reserved flags
        f.write(header_bytes)
        for name in sorted_names:
            if name in q4nx_tensors:
                f.write(q4nx_tensors[name]["raw_bytes"])

    stats = {
        "tag": tag,
        "num_tensors": len(q4nx_tensors),
        "header_bytes": header_size,
        "data_bytes": current_offset,
        "total_bytes": total,
    }

    if verbose:
        print(f"\n✅ Conversion complete! Tag: {tag}, "
              f"Tensors: {stats['num_tensors']}, "
              f"Size: {stats['total_bytes'] / (1024*1024):.1f} MB")

    return stats


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace models to Q4NX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/Qwen3-0.6B
  %(prog)s /path/to/Llama-3.2-1B --output model.q4nx --verbose
  %(prog)s /path/to/DeepSeek-V2-Lite --verbose

Supported architectures:
  Qwen3-0.6B, Qwen3-1.5B, Qwen3-7B, Qwen3-14B
  Qwen2.5-7B, Qwen2.5-32B
  Llama-3.1-8B, Llama-3.2-1B, Llama-3.2-3B
  Gemma-2-2B, Gemma-2-9B
  DeepSeek-V2-Lite, DeepSeek-V3
  Mixtral-8x7B, Zaya1-8B
        """,
    )
    parser.add_argument(
        "model_dir", type=str, nargs="?", default=None,
        help="Path to HuggingFace model directory (config.json + *.safetensors)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="model.q4nx",
        help="Output .q4nx file path (default: model.q4nx)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed progress information",
    )
    parser.add_argument(
        "--list-archs", action="store_true",
        help="List all supported architectures and exit",
    )
    parser.add_argument(
        "--precision", choices=["i8", "bf16"], default="i8",
        help="Projection weight (q/k/v/o_proj, gate/up/down_proj) precision: "
             "'i8' quantizes to the 4-bit Q4NX tile format (default, small, "
             "lossy); 'bf16' stores them near-losslessly instead (same "
             "precision as norms/embeddings already use, ~4x larger file, "
             "eliminates quantization error as a source of divergence from "
             "the source model)",
    )

    args = parser.parse_args()

    if args.list_archs:
        print("Supported architectures:")
        print(f"  {'Tag':<20} {'H':<6} {'NC':<4} {'NH':<4} "
              f"{'NKV':<4} {'HD':<4} {'IM':<6} {'NV':<8} {'Type':<16}")
        print("  " + "-" * 85)
        for tag, cfg in sorted(ARCH_CONFIGS.items()):
            print(f"  {tag:<20} {cfg['H']:<6} {cfg['NC']:<4} {cfg['NH']:<4} "
                  f"{cfg['NKV']:<4} {cfg['HD']:<4} {cfg['IM']:<6} "
                  f"{cfg['NV']:<8} {cfg['arch']:<16}")
        return

    if args.model_dir is None:
        parser.print_help()
        sys.exit(1)

    if not _os.path.isdir(args.model_dir):
        print(f"❌ Error: {args.model_dir} is not a directory")
        sys.exit(1)

    if not _os.path.exists(_os.path.join(args.model_dir, "config.json")):
        print(f"❌ Error: {args.model_dir} does not contain config.json")
        sys.exit(1)

    try:
        convert_hf_to_q4nx(
            model_dir=args.model_dir,
            output_path=args.output,
            verbose=args.verbose,
            precision=args.precision,
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

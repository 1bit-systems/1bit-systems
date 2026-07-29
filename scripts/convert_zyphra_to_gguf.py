#!/usr/bin/env python3
"""GGUF converter for Zyphra Zamba-7B-v1 and BlackMamba models.

Converts HuggingFace PyTorch checkpoints to GGUF format compatible with
llama.cpp's Mamba2/Mamba architecture support.

Usage:
  python3 convert_zyphra_to_gguf.py Zyphra/Zamba-7B-v1 ./zamba-7b-v1.gguf
  python3 convert_zyphra_to_gguf.py Zyphra/BlackMamba-2.8B ./blackmamba-2.8b.gguf
"""

import struct
import json
import sys
import os
import torch
import numpy as np
from pathlib import Path


def validate_gguf_file(path: str, min_size_mb: int = 100):
    """Validate a GGUF output file for integrity.

    Checks:
    - File exists and is not 0 bytes
    - Size meets minimum threshold
    - Starts with GGUF magic bytes

    Raises RuntimeError on critical failures, prints warnings otherwise.
    """
    if not os.path.exists(path):
        raise RuntimeError(f"Output file {path} does not exist!")
    if os.path.getsize(path) == 0:
        raise RuntimeError(f"Output file {path} is 0 bytes! Conversion failed.")
    min_size = min_size_mb * 1024 * 1024
    actual_size = os.path.getsize(path)
    if actual_size < min_size:
        print(f"WARNING: Output file size ({actual_size} bytes / {actual_size/1024/1024:.1f} MB) "
              f"is below {min_size_mb} MB minimum — file may be truncated or corrupt.")
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            print(f"WARNING: File does not start with GGUF magic bytes (got {magic.hex()})")
        else:
            print(f"GGUF magic bytes verified: {magic}")
    print(f"Validation passed: {path} ({actual_size} bytes)")

# ── GGUF constants ──
GGUF_MAGIC = b'GGUF'
GGUF_VERSION = 3

# Tensor types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 7
GGML_TYPE_Q4_K = 11
GGML_TYPE_Q6_K = 13

# KV keys
KV_GENERAL_ARCHITECTURE = "general.architecture"
KV_SSM_CONV_KERNEL = "ssm.conv_kernel"
KV_SSM_INNER_SIZE = "ssm.inner_size"
KV_SSM_STATE_SIZE = "ssm.state_size"
KV_SSM_TIME_STEP_RANK = "ssm.dt_rank"
KV_SSM_GROUP_COUNT = "ssm.group_count"
KV_ATTN_LAYERNORM_RMS_EPS = "attention.layer_norm_rms_epsilon"

def gguf_quantize_q4_0(data: np.ndarray) -> bytes:
    """Quantize FP32 data to Q4_0 blocks."""
    assert data.ndim == 1
    n = len(data)
    out = bytearray()
    for i in range(0, n, 32):
        block = data[i:i+32]
        amax = np.max(np.abs(block))
        if amax == 0:
            scale = 0.0
        else:
            scale = amax / 7.0
        # FP16 scale
        scale_f16 = struct.pack('<e', scale)
        out.extend(scale_f16)
        # Pack 32 4-bit values
        for j in range(0, 32, 2):
            v0 = block[j] / scale if scale > 0 else 0
            v1 = block[j+1] / scale if j+1 < len(block) and scale > 0 else 0
            q0 = max(-8, min(7, int(round(v0)))) & 0x0F
            q1 = max(-8, min(7, int(round(v1)))) & 0x0F if j+1 < len(block) else 0
            out.append((q0 << 4) | q1)
    return bytes(out)

class GgufWriter:
    """Minimal GGUF file writer."""
    
    def __init__(self, path: str, arch: str):
        self.f = open(path, 'wb')
        self.arch = arch
        self.kv_data = bytearray()
        self.tensor_info = []
        self.tensor_data = bytearray()
        self.n_kv = 0
        self.n_tensors = 0
        
        # Write header placeholder
        self.f.write(GGUF_MAGIC)
        self.f.write(struct.pack('<I', GGUF_VERSION))
        self.f.write(struct.pack('<Q', 0))  # n_tensors placeholder
        self.f.write(struct.pack('<Q', 0))  # n_kv placeholder
        
        # Write architecture
        self.add_string(KV_GENERAL_ARCHITECTURE, arch)
    
    def add_uint32(self, key: str, val: int):
        self._write_kv(key, 4, struct.pack('<I', val))
    
    def add_float32(self, key: str, val: float):
        self._write_kv(key, 6, struct.pack('<f', val))
    
    def add_string(self, key: str, val: str):
        self._write_kv(key, 8, struct.pack('<Q', len(val)) + val.encode())
    
    def _write_kv(self, key: str, val_type: int, val_bytes: bytes):
        key_bytes = key.encode()
        self.kv_data.extend(struct.pack('<Q', len(key_bytes)))
        self.kv_data.extend(key_bytes)
        self.kv_data.extend(struct.pack('<I', val_type))
        self.kv_data.extend(val_bytes)
        self.n_kv += 1
    
    def add_tensor(self, name: str, data: np.ndarray, dtype: int = GGML_TYPE_F32):
        """Register a tensor. Data is kept for later serialization."""
        assert data.ndim <= 4, f"Tensor {name} has {data.ndim} dims"
        self.tensor_info.append((name, data.shape, dtype))
        self.n_tensors += 1
        
        if dtype == GGML_TYPE_Q4_0:
            quantized = gguf_quantize_q4_0(data.flatten())
            self.tensor_data.extend(quantized)
        elif dtype == GGML_TYPE_F16:
            self.tensor_data.extend(data.astype(np.float16).tobytes())
        else:
            self.tensor_data.extend(data.astype(np.float32).tobytes())
    
    def close(self):
        """Finalize and write the GGUF file."""
        # Update headers
        self.f.seek(8)
        self.f.write(struct.pack('<Q', self.n_tensors))
        self.f.write(struct.pack('<Q', self.n_kv))
        
        # Write KV pairs
        self.f.write(bytes(self.kv_data))
        
        # Align to 32 bytes
        pos = self.f.tell()
        align = 32
        rem = pos % align
        if rem:
            self.f.write(b'\x00' * (align - rem))
        
        tensor_data_offset = self.f.tell()
        
        # Write tensor info
        offset = 0
        for name, shape, dtype in self.tensor_info:
            name_bytes = name.encode()
            self.f.write(struct.pack('<Q', len(name_bytes)))
            self.f.write(name_bytes)
            self.f.write(struct.pack('<I', len(shape)))
            for s in shape:
                self.f.write(struct.pack('<Q', s))
            self.f.write(struct.pack('<I', dtype))
            self.f.write(struct.pack('<Q', offset))
            
            # Compute size for offset tracking
            if dtype == GGML_TYPE_Q4_0:
                n_blocks = (np.prod(shape) + 31) // 32
                elem_size = n_blocks * 18
            elif dtype == GGML_TYPE_F16:
                elem_size = np.prod(shape) * 2
            else:
                elem_size = np.prod(shape) * 4
            offset += elem_size
        
        # Align tensor data to 32 bytes from tensor_data_offset
        pos = self.f.tell()
        rem = pos % align
        if rem:
            self.f.write(b'\x00' * (align - rem))
        
        # Write tensor data
        self.f.write(bytes(self.tensor_data))
        
        self.f.close()
        print(f"Wrote {self.n_tensors} tensors to {self.f.name}")


def convert_zamba_v1(model_id: str, output: str):
    """Convert Zamba-7B-v1 to GGUF."""
    from transformers import AutoModelForCausalLM, AutoConfig
    
    print(f"Loading {model_id}...")
    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    
    H = config.hidden_size
    n_layers = config.num_hidden_layers
    d_state = 16  # Zamba1 uses d_state=16
    d_conv = 4
    d_inner = config.intermediate_size  # FFN hidden
    vocab = config.vocab_size
    
    print(f"Zamba-7B-v1: H={H} L={n_layers} V={vocab}")
    
    writer = GgufWriter(output, "zamba")
    writer.add_uint32("block_count", n_layers)
    writer.add_uint32("embedding_length", H)
    writer.add_uint32("feed_forward_length", d_inner)
    writer.add_uint32("attention.head_count", config.num_attention_heads)
    writer.add_uint32("attention.head_count_kv", config.num_key_value_heads)
    writer.add_uint32("context_length", config.max_position_embeddings)
    writer.add_float32("attention.layer_norm_rms_epsilon", config.rms_norm_eps)
    writer.add_uint32("ssm.conv_kernel", d_conv)
    writer.add_uint32("ssm.inner_size", H * 2)  # Mamba expand=2
    writer.add_uint32("ssm.state_size", d_state)
    writer.add_uint32("ssm.dt_rank", d_state)
    writer.add_uint32("ssm.group_count", 1)
    writer.add_uint32("vocab_size", vocab)
    
    state_dict = model.state_dict()
    
    # Embedding
    emb = state_dict["model.embed_tokens.weight"].numpy()
    writer.add_tensor("token_embd.weight", emb, GGML_TYPE_F16)
    
    # Output norm
    fn = state_dict["model.final_layernorm.weight"].numpy()
    writer.add_tensor("output_norm.weight", fn)
    
    # Determine hybrid layers from config
    is_hybrid = [t == "hybrid" for t in config.layers_block_type]
    
    for l in range(n_layers):
        prefix = f"model.layers.{l}"
        hyb = is_hybrid[l]
        
        # Input norm (always present)
        in_n = state_dict[f"{prefix}.input_layernorm.weight"].numpy()
        writer.add_tensor(f"blk.{l}.attn_norm.weight", in_n)
        
        if hyb:
            # Shared attention
            sa_prefix = f"{prefix}.shared_attention"
            q = state_dict[f"{sa_prefix}.q_proj.weight"].numpy()
            k = state_dict[f"{sa_prefix}.k_proj.weight"].numpy()
            v = state_dict[f"{sa_prefix}.v_proj.weight"].numpy()
            o = state_dict[f"{sa_prefix}.o_proj.weight"].numpy()
            writer.add_tensor(f"blk.{l}.attn_q.weight", q.T, GGML_TYPE_Q4_0)
            writer.add_tensor(f"blk.{l}.attn_k.weight", k.T, GGML_TYPE_Q4_0)
            writer.add_tensor(f"blk.{l}.attn_v.weight", v.T, GGML_TYPE_Q4_0)
            writer.add_tensor(f"blk.{l}.attn_output.weight", o.T, GGML_TYPE_Q4_0)
            
            # Post-attention norm
            pa_n = state_dict[f"{sa_prefix}.post_attention_layernorm.weight"].numpy()
            writer.add_tensor(f"blk.{l}.post_attention_norm.weight", pa_n)
            
            # Shared MLP
            sm_prefix = f"{prefix}.shared_mlp"
            gu = state_dict[f"{sm_prefix}.gate_proj.weight"].numpy()
            up = state_dict[f"{sm_prefix}.up_proj.weight"].numpy()
            down = state_dict[f"{sm_prefix}.down_proj.weight"].numpy()
            writer.add_tensor(f"blk.{l}.ffn_gate.weight", gu.T, GGML_TYPE_Q4_0)
            writer.add_tensor(f"blk.{l}.ffn_up.weight", up.T, GGML_TYPE_Q4_0)
            writer.add_tensor(f"blk.{l}.ffn_down.weight", down.T, GGML_TYPE_Q4_0)
            
            # FFN norm
            ff_n = state_dict[f"{sm_prefix}.layernorm.weight"].numpy()
            writer.add_tensor(f"blk.{l}.ffn_norm.weight", ff_n)
        
        # Mamba1 block
        mb_prefix = f"{prefix}.mamba"
        ip = state_dict[f"{mb_prefix}.in_proj.weight"].numpy()
        c1w = state_dict[f"{mb_prefix}.conv1d.weight"].numpy()
        c1b = state_dict[f"{mb_prefix}.conv1d.bias"].numpy()
        dt = state_dict[f"{mb_prefix}.dt.weight"].numpy().flatten()
        A_log = state_dict[f"{mb_prefix}.A_log"].numpy().flatten()
        D = state_dict[f"{mb_prefix}.D"].numpy().flatten()
        x_proj = state_dict[f"{mb_prefix}.x_proj.weight"].numpy()
        dt_proj = state_dict[f"{mb_prefix}.dt_proj.weight"].numpy()
        out = state_dict[f"{mb_prefix}.out_proj.weight"].numpy()
        
        writer.add_tensor(f"blk.{l}.ssm_in.weight", ip.T, GGML_TYPE_Q4_0)
        writer.add_tensor(f"blk.{l}.ssm_conv1d.weight", c1w.T)
        writer.add_tensor(f"blk.{l}.ssm_conv1d.bias", c1b)
        writer.add_tensor(f"blk.{l}.ssm_dt.bias", dt)
        writer.add_tensor(f"blk.{l}.ssm_a", A_log)
        writer.add_tensor(f"blk.{l}.ssm_d", D)
        writer.add_tensor(f"blk.{l}.ssm_out.weight", out.T, GGML_TYPE_Q4_0)
    
    # LM head (tied)
    writer.add_tensor("output.weight", emb, GGML_TYPE_Q4_0)
    
    writer.close()
    
    # ── Validate output file ──
    validate_gguf_file(output)


def convert_blackmamba(model_id: str, output: str):
    """Convert BlackMamba to GGUF."""
    print(f"Loading {model_id}...")
    
    # BlackMamba uses a custom config format
    import json
    import requests
    
    config_url = f"https://huggingface.co/{model_id}/raw/main/config.json"
    config = json.loads(requests.get(config_url).text)
    
    H = config["hidden_size"]
    n_layers = config["num_layers"]
    d_state = config["state_size"]
    d_conv = config["conv_dimension"]
    vocab = config["vocab_size"]
    
    print(f"BlackMamba: H={H} L={n_layers} d_state={d_state} V={vocab}")
    
    # Load weights
    from huggingface_hub import hf_hub_download
    import safetensors.torch
    
    # Try safetensors first, then bin
    try:
        model_index = requests.get(
            f"https://huggingface.co/{model_id}/raw/main/model.safetensors.index.json"
        ).json()
        shard_files = set(model_index["weight_map"].values())
    except:
        shard_files = ["pytorch_model.bin"]
    
    state_dict = {}
    for sf in shard_files:
        path = hf_hub_download(model_id, sf)
        if sf.endswith('.safetensors'):
            sd = safetensors.torch.load_file(path)
        else:
            sd = torch.load(path, map_location='cpu', weights_only=True)
        state_dict.update(sd)
    
    writer = GgufWriter(output, "mamba")  # Use "mamba" arch (Mamba1)
    writer.add_uint32("block_count", n_layers)
    writer.add_uint32("embedding_length", H)
    writer.add_uint32("feed_forward_length", H * 4)  # FFN hidden
    writer.add_uint32("context_length", 2048)
    writer.add_float32("attention.layer_norm_rms_epsilon", 1e-5)
    writer.add_uint32("ssm.conv_kernel", d_conv)
    writer.add_uint32("ssm.inner_size", H * 2)
    writer.add_uint32("ssm.state_size", d_state)
    writer.add_uint32("ssm.dt_rank", d_state)
    writer.add_uint32("ssm.group_count", 1)
    writer.add_uint32("vocab_size", vocab)
    
    # Embedding
    emb = state_dict["model.embed_tokens.weight"].numpy()
    writer.add_tensor("token_embd.weight", emb, GGML_TYPE_F16)
    
    # Final norm
    fn = state_dict["model.final_layernorm.weight"].numpy()
    writer.add_tensor("output_norm.weight", fn)
    
    for l in range(n_layers):
        prefix = f"model.layers.{l}"
        
        in_n = state_dict[f"{prefix}.input_layernorm.weight"].numpy()
        writer.add_tensor(f"blk.{l}.attn_norm.weight", in_n)
        
        mb_prefix = f"{prefix}.mamba"
        ip = state_dict[f"{mb_prefix}.in_proj.weight"].numpy()
        c1w = state_dict[f"{mb_prefix}.conv1d.weight"].numpy()
        c1b = state_dict[f"{mb_prefix}.conv1d.bias"].numpy()
        
        # Mamba1 specific
        x_proj = state_dict[f"{mb_prefix}.x_proj.weight"].numpy()
        dt_proj_w = state_dict[f"{mb_prefix}.dt_proj.weight"].numpy()
        dt_proj_b = state_dict[f"{mb_prefix}.dt_proj.bias"].numpy()
        A = state_dict[f"{mb_prefix}.A"].numpy().flatten()
        D = state_dict[f"{mb_prefix}.D"].numpy().flatten()
        out = state_dict[f"{mb_prefix}.out_proj.weight"].numpy()
        
        writer.add_tensor(f"blk.{l}.ssm_in.weight", ip.T, GGML_TYPE_Q4_0)
        writer.add_tensor(f"blk.{l}.ssm_conv1d.weight", c1w.T)
        writer.add_tensor(f"blk.{l}.ssm_conv1d.bias", c1b)
        writer.add_tensor(f"blk.{l}.ssm_x.weight", x_proj.T, GGML_TYPE_Q4_0)
        writer.add_tensor(f"blk.{l}.ssm_dt.weight", dt_proj_w.T)
        writer.add_tensor(f"blk.{l}.ssm_dt.bias", dt_proj_b)
        writer.add_tensor(f"blk.{l}.ssm_a", A)
        writer.add_tensor(f"blk.{l}.ssm_d", D)
        writer.add_tensor(f"blk.{l}.ssm_out.weight", out.T, GGML_TYPE_Q4_0)
        
        # MoE weights if applicable
        moe_layers = config.get("mamba_moe_layers", [])
        if l < len(moe_layers) and moe_layers[l] != "r":
            n_experts = int(moe_layers[l])  # "8" → 8
            moe_prefix = f"{prefix}.moe"
            
            gate = state_dict[f"{moe_prefix}.gate.weight"].numpy()
            writer.add_tensor(f"blk.{l}.ffn_gate.weight", gate.T, GGML_TYPE_Q4_0)
            
            for e in range(n_experts):
                wi1 = state_dict[f"{moe_prefix}.experts.{e}.wi_1.weight"].numpy()
                wo = state_dict[f"{moe_prefix}.experts.{e}.wo_1.weight"].numpy()
                writer.add_tensor(f"blk.{l}.ffn_expert.{e}.weight_1", wi1.T, GGML_TYPE_Q4_0)
                writer.add_tensor(f"blk.{l}.ffn_expert.{e}.weight_2", wo.T, GGML_TYPE_Q4_0)
    
    output_w = state_dict.get("model.lm_head.weight", emb)
    writer.add_tensor("output.weight", output_w.numpy(), GGML_TYPE_Q4_0)
    
    writer.close()
    
    # ── Validate output file ──
    validate_gguf_file(output)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python3 convert_zyphra_to_gguf.py Zyphra/Zamba-7B-v1 ./zamba-7b-v1.gguf")
        print("  python3 convert_zyphra_to_gguf.py Zyphra/BlackMamba-2.8B ./blackmamba-2.8b.gguf")
        sys.exit(1)
    
    model_id = sys.argv[1]
    output = sys.argv[2]
    
    if "Zamba" in model_id and "v1" in model_id:
        convert_zamba_v1(model_id, output)
    elif "BlackMamba" in model_id:
        convert_blackmamba(model_id, output)
    else:
        print(f"Unknown model: {model_id}")
        sys.exit(1)

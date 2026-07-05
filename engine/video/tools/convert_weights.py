#!/usr/bin/env python3
"""Convert HuggingFace Wan2.2 DiT weights to 1bit video .bin format.

Usage:
  python3 tools/convert_weights.py \
      --model Wan-AI/Wan2.1-T2V-1.3B \
      --output wan13b.bin \
      --dtype float32

  python3 tools/convert_weights.py \
      --model alibaba-pai/Wan2.2-Fun-1.3B \
      --output wan22_13b.bin \
      --dtype float16
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import torch
import numpy as np

# Match C struct VideoWeightHeader (64 bytes)
#   char magic[8] = "VIDWEIGH"
#   uint32_t version
#   uint32_t num_layers
#   uint32_t hidden_size
#   uint32_t num_heads
#   uint32_t head_dim
#   uint32_t mlp_hidden
#   uint32_t text_hidden
#   uint64_t weight_offset
#   uint8_t padding[20]

HEADER_FORMAT = '<8sIIIIIIIQ20x'  # 8+4*7+8+20 = 64 bytes
HEADER_SIZE = 64
MAGIC = b'VIDWEIGH'


def extract_dit_params(model) -> dict:
    """Extract DiT architecture params from HuggingFace model config."""
    cfg = model.config if hasattr(model, 'config') else model.module.config
    
    params = {
        'hidden_size': getattr(cfg, 'hidden_size', 1024),
        'num_hidden_layers': getattr(cfg, 'num_hidden_layers', 
                              getattr(cfg, 'num_layers', 30)),
        'num_attention_heads': getattr(cfg, 'num_attention_heads', 16),
        'head_dim': getattr(cfg, 'head_dim', 
                    getattr(cfg, 'hidden_size', 1024) // 
                    getattr(cfg, 'num_attention_heads', 16)),
        'intermediate_size': getattr(cfg, 'intermediate_size', 4096),
        'text_hidden_size': getattr(cfg, 'text_hidden_size', 4096),
        'patch_size': getattr(cfg, 'patch_size', 2),
        'in_channels': getattr(cfg, 'in_channels', 16),
    }
    return params


def convert_weights(model, output_path: Path, dtype: str = 'float32'):
    """Convert model weights to binary format."""
    print(f"[Convert] Loading model weights...")
    
    params = extract_dit_params(model)
    H = params['hidden_size']
    NC = params['num_hidden_layers']
    NH = params['num_attention_heads']
    HD = params['head_dim']
    IM = params['intermediate_size']
    TH = params['text_hidden_size']
    
    print(f"  Architecture: H={H} NC={NC} NH={NH} HD={HD} IM={IM} TH={TH}")
    
    np_dtype = np.float32 if dtype == 'float32' else np.float16
    
    # Collect all weights in order
    weights = []
    tensors_exported = 0
    
    state_dict = model.state_dict() if hasattr(model, 'state_dict') else model
    
    # Patch embedding
    for name in ['patch_embedding.weight', 'x_embedder.weight', 'pos_embed.weight']:
        if name in state_dict:
            w = state_dict[name].cpu().numpy().astype(np_dtype).ravel()
            weights.append(w.tobytes())
            tensors_exported += 1
            print(f"  {name}: {w.shape}")

    # Per-layer weights
    for layer_idx in range(NC):
        prefix = f'model.layers.{layer_idx}.'
        alt_prefixes = [
            f'transformer_blocks.{layer_idx}.',
            f'layers.{layer_idx}.',
            f'dit.layers.{layer_idx}.',
        ]
        
        for prefix_candidate in [prefix] + alt_prefixes:
            # Self-attention
            for proj in ['self_attn.q_proj', 'self_attn.k_proj', 
                        'self_attn.v_proj', 'self_attn.o_proj',
                        'attention.q_proj', 'attention.k_proj',
                        'attention.v_proj', 'attention.o_proj',
                        'attn.q_proj', 'attn.k_proj', 'attn.v_proj', 'attn.o_proj']:
                for suf in ['weight', 'bias']:
                    key = f'{prefix_candidate}{proj}.{suf}'
                    if key in state_dict:
                        w = state_dict[key].cpu().numpy().astype(np_dtype).ravel()
                        weights.append(w.tobytes())
                        tensors_exported += 1
            
            # Cross-attention (text conditioning)
            for proj in ['cross_attn.q_proj', 'cross_attn.k_proj',
                        'cross_attn.v_proj', 'cross_attn.o_proj',
                        'cross_attention.q_proj', 'cross_attention.k_proj',
                        'cross_attention.v_proj', 'cross_attention.o_proj']:
                for suf in ['weight', 'bias']:
                    key = f'{prefix_candidate}{proj}.{suf}'
                    if key in state_dict:
                        w = state_dict[key].cpu().numpy().astype(np_dtype).ravel()
                        weights.append(w.tobytes())
                        tensors_exported += 1
            
            # MLP
            for proj in ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj',
                        'ffn.gate_proj', 'ffn.up_proj', 'ffn.down_proj',
                        'mlp.fc1', 'mlp.fc2', 'mlp.gate', 'mlp.up', 'mlp.down']:
                for suf in ['weight', 'bias']:
                    key = f'{prefix_candidate}{proj}.{suf}'
                    if key in state_dict:
                        w = state_dict[key].cpu().numpy().astype(np_dtype).ravel()
                        weights.append(w.tobytes())
                        tensors_exported += 1
            
            # Layer norms
            for norm in ['self_attn_layer_norm', 'input_layernorm',
                        'cross_attn_layer_norm', 'post_attention_layernorm',
                        'ffn_layer_norm', 'final_layer_norm']:
                for suf in ['weight', 'bias']:
                    key = f'{prefix_candidate}{norm}.{suf}'
                    if key in state_dict:
                        w = state_dict[key].cpu().numpy().astype(np_dtype).ravel()
                        weights.append(w.tobytes())
                        tensors_exported += 1
                        
            # AdaLN modulation
            for name in ['adaLN_modulation.weight', 'adaLN_modulation.bias',
                        'adaLN_modulation.1.weight', 'adaLN_modulation.1.bias']:
                key = f'{prefix_candidate}{name}'
                if key in state_dict:
                    w = state_dict[key].cpu().numpy().astype(np_dtype).ravel()
                    weights.append(w.tobytes())
                    tensors_exported += 1

    # Final layer norm + projection
    for name in ['final_layer_norm.weight', 'final_layer_norm.bias',
                'final_proj.weight', 'final_proj.bias',
                'pred_head.weight', 'pred_head.bias']:
        if name in state_dict:
            w = state_dict[name].cpu().numpy().astype(np_dtype).ravel()
            weights.append(w.tobytes())
            tensors_exported += 1

    # Timestep embedding
    for name in ['t_embedder.weight', 't_embedder.bias',
                'time_embed.weight', 'time_embed.bias']:
        if name in state_dict:
            w = state_dict[name].cpu().numpy().astype(np_dtype).ravel()
            weights.append(w.tobytes())
            tensors_exported += 1

    print(f"\n  Total tensors exported: {tensors_exported}")

    # Write binary file
    weight_data = b''.join(weights)
    
    header = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        1,          # version
        NC,         # num_layers
        H,          # hidden_size
        NH,         # num_heads
        HD,         # head_dim
        IM,         # mlp_hidden
        TH,         # text_hidden
        HEADER_SIZE  # weight_offset
    )
    
    assert len(header) == HEADER_SIZE, f"Header size mismatch: {len(header)} != {HEADER_SIZE}"
    
    output_path.write_bytes(header + weight_data)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n[Convert] Written {output_path}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Data size: {len(weight_data) / (1024*1024):.1f} MB")
    print(f"  Header: {HEADER_SIZE} bytes")


def main():
    parser = argparse.ArgumentParser(description="Convert Wan2.2 weights to .bin format")
    parser.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                        help="HF model ID or local path")
    parser.add_argument("--output", "-o", required=True, type=Path,
                        help="Output .bin file path")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                        help="Output precision")
    args = parser.parse_args()

    print(f"[Convert] Loading model from {args.model}...")
    
    # Try loading as diffusers pipeline first, then raw model
    try:
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if args.dtype == "float16" else torch.float32,
            device_map="auto",
        )
        model = pipe.transformer if hasattr(pipe, 'transformer') else pipe
    except Exception as e:
        print(f"  Diffusers load failed: {e}")
        print(f"  Trying raw model...")
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                args.model,
                torch_dtype=torch.float16 if args.dtype == "float16" else torch.float32,
            )
        except Exception as e2:
            print(f"  Transformers load failed: {e2}")
            sys.exit(1)
    
    convert_weights(model, args.output, args.dtype)


if __name__ == "__main__":
    main()

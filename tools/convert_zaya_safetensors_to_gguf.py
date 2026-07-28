#!/usr/bin/env python3
"""Convert ZAYA safetensors to GGUF format (F16).

Works for any ZayaForCausalLM model (ZAYA1-8B, ZAYA1-base, ZAYA1-reasoning-base).

Usage: python3 tools/convert_zaya_safetensors_to_gguf.py <model_dir> <output.gguf>
"""
import sys, os, json, time
import numpy as np
from safetensors import safe_open
from gguf import GGUFWriter, GGMLQuantizationType

# ── Config ──────────────────────────────────────────────────────────────────
# These are read from config.json, but we need them for tensor shape validation
EXPECTED_ARCH = "zaya"

# Tensor name mapping: safetensors → GGUF
GLOBAL_MAP = {
    'model.embed_tokens.weight':             'token_embd.weight',
    'model.final_norm.weight':                'output_norm.weight',
    'model.res_scale.hidden_states_bias':     'input_hidden_states_scale.bias',
    'model.res_scale.hidden_states_scale':    'input_hidden_states_scale.weight',
    'model.res_scale.residual_bias':          'input_residual_scale.bias',
    'model.res_scale.residual_scale':         'input_residual_scale.weight',
}

LAYER_MAP = {
    'input_norm.weight':                     'attn_norm.weight',
    'self_attn.o_proj.weight':               'attn_output.weight',
    'self_attn.qkv.linear_q.weight':         'attn_q.weight',
    'self_attn.qkv.linear_k.weight':         'attn_k.weight',
    'self_attn.qkv.val_proj1.weight':        'cca_val_proj1.weight',
    'self_attn.qkv.val_proj2.weight':        'cca_val_proj2.weight',
    'self_attn.qkv.conv_qk.0.weight':       'cca_conv_grp.weight',
    'self_attn.qkv.conv_qk.0.bias':         'cca_conv_grp.bias',
    'self_attn.qkv.conv_qk.1.weight':       'cca_conv_grp_2.weight',
    'self_attn.qkv.conv_qk.1.bias':         'cca_conv_grp_2.bias',
    'self_attn.qkv.temp':                   'cca_k_scale.weight',
    'res_scale.hidden_states_bias':         'res_scale_hs.bias',
    'res_scale.hidden_states_scale':        'res_scale_hs.weight',
    'res_scale.residual_bias':              'res_scale_res.bias',
    'res_scale.residual_scale':             'res_scale_res.weight',
}

ZAYA_BLOCK_MAP = {
    'zaya_block.router.down_proj.weight':           'ffn_gate_inp.weight',
    'zaya_block.router.down_proj.bias':             'ffn_gate_inp.bias',
    'zaya_block.router.router_mlp.0.weight':        'ffn_gate.weight',
    'zaya_block.router.router_mlp.0.bias':          'ffn_gate.bias',
    'zaya_block.router.router_mlp.2.weight':        'zaya_router_mlp2.weight',
    'zaya_block.router.router_mlp.2.bias':          'zaya_router_mlp2.bias',
    'zaya_block.router.router_mlp.4.weight':        'zaya_router_mlp4.weight',
    'zaya_block.router.balancing_biases':            'zaya_router_biases.weight',
    'zaya_block.router.rmsnorm_eda.weight':         'zaya_router_eda.weight',
    'zaya_block.router.router_states_scale':        'zaya_router_states_scale',
    'zaya_block.experts.local_experts.{e}.linear_fc1.weight': 'ffn_down_exps.weight',
    'zaya_block.experts.local_experts.{e}.linear_fc2.weight': 'ffn_gate_up_exps.weight',
}


def load_tensor(model_dir, shard, name):
    path = os.path.join(model_dir, shard)
    try:
        with safe_open(path, framework='np') as f:
            return f.get_tensor(name)
    except TypeError:
        # BF16 not supported by numpy, use torch
        import torch
        with safe_open(path, framework='pt') as f:
            t = f.get_tensor(name)  # Returns torch.tensor
            return t.to(torch.float16).cpu().numpy()


def to_f16(t):
    import torch
    if isinstance(t, np.ndarray):
        if t.dtype == np.float32:
            return t.astype(np.float16)
        if t.dtype == np.float16:
            return t
        if t.dtype == np.dtype('bfloat16'):
            return torch.from_numpy(t).to(torch.float16).numpy()
        return t.astype(np.float32).astype(np.float16)
    # torch tensor
    return t.to(torch.float16).numpy()


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_dir> <output.gguf>")
        sys.exit(1)

    model_dir = sys.argv[1]
    output_path = sys.argv[2]

    with open(os.path.join(model_dir, 'config.json')) as f:
        cfg = json.load(f)
    with open(os.path.join(model_dir, 'model.safetensors.index.json')) as f:
        idx = json.load(f)

    wm = idx['weight_map']
    shards = sorted(set(wm.values()))

    n_layers = int(cfg.get('num_hidden_layers', 80))
    hidden = int(cfg.get('hidden_size', 2048))
    n_heads = int(cfg.get('num_attention_heads', 16))
    n_kv = int(cfg.get('num_key_value_heads', 2))
    head_dim = int(cfg.get('head_dim', 128))
    vocab = int(cfg.get('vocab_size', 262272))
    max_seq = int(cfg.get('max_position_embeddings', 32768))
    norm_eps = float(cfg.get('norm_epsilon', 1e-5))
    rope_base = float(cfg.get('rotary_base', 1000000.0))
    use_parallel = bool(cfg.get('parallel_residual', False))

    # Detect zaya_block pattern
    moe_layers = set()
    for k in wm:
        if 'zaya_block' in k:
            parts = k.split('.')
            if len(parts) >= 3 and parts[1] == 'layers':
                moe_layers.add(int(parts[2]))

    n_experts = 16  # ZAYA fixed

    print(f"  Model: {cfg.get('_name_or_path', os.path.basename(model_dir))}")
    print(f"  Architecture: zaya, Layers: {n_layers}, Hidden: {hidden}")
    print(f"  Heads: {n_heads}, KV: {n_kv}, Vocab: {vocab}")
    print(f"  MoE layers: {sorted(moe_layers) if moe_layers else 'N/A (dense)'}")
    print(f"  Shards: {shards}")
    print(f"  Output: {output_path}")
    print()

    writer = GGUFWriter(output_path, 'zaya')

    # Metadata
    writer.add_block_count(n_layers)
    writer.add_context_length(max_seq)
    writer.add_embedding_length(hidden)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv)
    writer.add_layer_norm_rms_eps(norm_eps)
    writer.add_vocab_size(vocab)
    writer.add_rope_dimension_count(head_dim)
    writer.add_rope_freq_base(rope_base)
    writer.add_expert_count(n_experts)
    writer.add_expert_used_count(1)
    ffn_hidden = int(cfg.get('ffn_hidden_size', 0)) or (hidden * 4)
    writer.add_feed_forward_length(ffn_hidden)
    writer.add_file_type(1)  # F32 (we write F16 but without raw_dtype it auto-detects)
    writer.add_bool('zaya.use_parallel_residual', use_parallel)

    t0 = time.time()
    total_tensors = 0

    def add_t(name, tensor):
        nonlocal total_tensors
        f16 = to_f16(tensor)
        writer.add_tensor(name, f16)
        total_tensors += 1
        if total_tensors % 100 == 0:
            elapsed = time.time() - t0
            print(f"    {total_tensors} tensors ({elapsed:.0f}s)...")

    # Global tensors
    print("  Global tensors...")
    for st_name, gguf_name in GLOBAL_MAP.items():
        if st_name in wm:
            t = load_tensor(model_dir, wm[st_name], st_name)
            add_t(gguf_name, t)

    # Per-layer
    print(f"  Layers (0..{n_layers-1})...")
    for li in range(n_layers):
        is_moe = li in moe_layers

        # Base layer tensors
        for st_suf, gguf_suf in LAYER_MAP.items():
            st_name = f"model.layers.{li}.{st_suf}"
            if st_name in wm:
                t = load_tensor(model_dir, wm[st_name], st_name)
                add_t(f"blk.{li}.{gguf_suf}", t)

        # zaya_block (MoE) tensors if present
        if is_moe:
            for st_suf, gguf_suf in ZAYA_BLOCK_MAP.items():
                if '{e}' in st_suf:
                    # Stack expert tensors: load all, stack into [rows, cols, n_experts]
                    expert_tensors = []
                    for ei in range(n_experts):
                        e_st_name = f"model.layers.{li}.{st_suf.replace('{e}', str(ei))}"
                        if e_st_name in wm:
                            t = load_tensor(model_dir, wm[e_st_name], e_st_name)
                            t_f16 = to_f16(t)
                            expert_tensors.append(t_f16)
                    if expert_tensors:
                        stacked = np.stack(expert_tensors, axis=-1)
                        add_t(f"blk.{li}.{gguf_suf}", stacked)
                else:
                    st_name = f"model.layers.{li}.{st_suf}"
                    if st_name in wm:
                        t = load_tensor(model_dir, wm[st_name], st_name)
                        add_t(f"blk.{li}.{gguf_suf}", t)

    # Write
    elapsed = time.time() - t0
    print(f"\n  Writing {total_tensors} tensors ({elapsed:.0f}s)...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size = os.path.getsize(output_path)
    print(f"  ✅ Done: {output_path} ({size/1e9:.2f} GB)")
    print(f"  Total tensors: {total_tensors}")


if __name__ == '__main__':
    main()

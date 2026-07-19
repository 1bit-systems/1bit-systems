#!/usr/bin/env python3
"""
q4nx_to_gguf.py v3 — Convert FLM Q4NX models to GGUF for GPU inference.

Handles:
  - I8 weights: convert to F16 (universal GPU support)
  - BF16 weights: convert to F16
  - Tokenizer: copy from model directory into GGUF metadata
  - NPU tiling: reverse the column-major strided layout

Usage:
  python q4nx_to_gguf.py /path/to/model.q4nx -o model.gguf
"""

import json, os, sys, struct
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q4nx_reader import Q4NXReader

try:
    import gguf
except ImportError:
    print("Install gguf: pip install gguf"); sys.exit(1)

# ── Tiling constants ──
ROW_BLOCK = 32; COL_BLOCK = 256; PARALLEL = 16

# ── Tensor name mapping ──
GGUF_LAYER = {
    "input_layernorm.weight":           "blk.{}.attn_norm.weight",
    "post_attention_layernorm.weight":  "blk.{}.ffn_norm.weight",
    "self_attn.q_proj.weight":          "blk.{}.attn_q.weight",
    "self_attn.k_proj.weight":          "blk.{}.attn_k.weight",
    "self_attn.v_proj.weight":          "blk.{}.attn_v.weight",
    "self_attn.o_proj.weight":          "blk.{}.attn_output.weight",
    "self_attn.q_norm.weight":          "blk.{}.attn_q_norm.weight",
    "self_attn.k_norm.weight":          "blk.{}.attn_k_norm.weight",
    "self_attn.gate_proj.weight":       "blk.{}.attn_output_gate.weight",
    "mlp.gate_proj.weight":             "blk.{}.ffn_gate.weight",
    "mlp.down_proj.weight":             "blk.{}.ffn_down.weight",
    "mlp.up_proj.weight":               "blk.{}.ffn_up.weight",
    "mlp.gate_exps_proj.weight":        "blk.{}.ffn_gate_exps.weight",
    "mlp.up_exps_proj.weight":          "blk.{}.ffn_up_exps.weight",
    "mlp.down_exps_proj.weight":        "blk.{}.ffn_down_exps.weight",
    "mlp.share_gate_exps_proj.weight":  "blk.{}.ffn_gate_shared_exps.weight",
    "mlp.share_up_exps_proj.weight":    "blk.{}.ffn_up_shared_exps.weight",
    "mlp.share_down_exps_proj.weight":  "blk.{}.ffn_down_shared_exps.weight",
    "moe_router.weight":                "blk.{}.moe_router.weight",
    "shared_expert_gate.weight":        "blk.{}.shared_expert_gate.weight",
    "linear_attn.qkv_proj.weight":      "blk.{}.linear_attn.qkv_proj.weight",
    "linear_attn.ssm_out_proj.weight":  "blk.{}.linear_attn.ssm_out_proj.weight",
    "linear_attn.ssm_conv1d.weight":    "blk.{}.linear_attn.ssm_conv1d.weight",
    "linear_attn.ssm_alpha_proj.weight":"blk.{}.linear_attn.ssm_alpha_proj.weight",
    "linear_attn.ssm_beta_proj.weight": "blk.{}.linear_attn.ssm_beta_proj.weight",
    "linear_attn.ssm_norm.weight":      "blk.{}.linear_attn.ssm_norm.weight",
    "linear_attn.ssm_a":                "blk.{}.linear_attn.ssm_a",
    "linear_attn.ssm_dt.bias":          "blk.{}.linear_attn.ssm_dt.bias",
}


def to_gguf_name(name: str) -> str:
    if name == "model.embed_tokens.weight": return "token_embd.weight"
    if name == "model.norm.weight": return "output_norm.weight"
    if name == "lm_head.weight": return "output.weight"
    if name.startswith("model.layer."):  # MoE naming
        p = name.split(".")
        return to_gguf_name(f"model.layers.{p[2]}.{'.'.join(p[3:])}")
    if name.startswith("model.layers."):
        p = name.split(".")
        rest = ".".join(p[3:])
        if rest in GGUF_LAYER:
            return GGUF_LAYER[rest].format(p[2])
    return name


def untile_i8(arr: np.ndarray) -> np.ndarray:
    """Reverse NPU column-major strided tiling."""
    R, C = arr.shape
    nr, nc = R // ROW_BLOCK, C // COL_BLOCK
    rp = ROW_BLOCK // PARALLEL
    blocked = arr.reshape(nr, ROW_BLOCK, nc, COL_BLOCK)
    result = np.zeros_like(blocked)
    for ri in range(nr):
        for ci in range(nc):
            blk = blocked[ri, :, ci, :]
            # Stored as [r//p, c, p] → reverse to [r//p, p, c] → [r, c]
            shaped = blk.reshape(rp, COL_BLOCK, PARALLEL)
            unshaped = shaped.transpose(0, 2, 1)
            result[ri, :, ci, :] = unshaped.reshape(ROW_BLOCK, COL_BLOCK)
    return result.transpose(0, 2, 1, 3).reshape(R, C)


def add_tokenizer(writer, model_dir: str, config: dict):
    """Add tokenizer metadata from HuggingFace tokenizer files."""
    tok_path = os.path.join(model_dir, "tokenizer.json")
    tok_cfg_path = os.path.join(model_dir, "tokenizer_config.json")
    
    if not os.path.exists(tok_path):
        print("  [WARN] tokenizer.json not found, skipping tokenizer metadata")
        return
    
    import json as j
    with open(tok_path) as f:
        tok_data = j.load(f)
    
    # Determine model type
    model_type = tok_data.get("model", {}).get("type", "BPE")
    if model_type == "BPE":
        writer.add_tokenizer_model("gpt2")
    elif model_type == "Unigram":
        writer.add_tokenizer_model("llama")
    elif "SentencePiece" in model_type or "sentencepiece" in model_type.lower():
        writer.add_tokenizer_model("llama")
    else:
        writer.add_tokenizer_model("gpt2")
    
    # Pre-tokenizer type from config
    tok_cfg = {}
    if os.path.exists(tok_cfg_path):
        with open(tok_cfg_path) as f:
            tok_cfg = j.load(f)
    
    add_bos = tok_cfg.get("add_bos_token", True)
    add_eos = tok_cfg.get("add_eos_token", False)
    chat_template = tok_cfg.get("chat_template", "")
    
    # Determine pre-tokenizer
    model_arch = config.get("model_type", "").lower()
    if "qwen" in model_arch:
        writer.add_tokenizer_pre("qwen2")
    elif "gemma" in model_arch:
        writer.add_tokenizer_pre("gemma")
    elif "llama" in model_arch:
        writer.add_tokenizer_pre("llama-bpe")
    else:
        writer.add_tokenizer_pre("qwen2")
    
    # Extract tokens
    vocab = tok_data.get("model", {}).get("vocab", {})
    if not vocab:
        added = tok_data.get("added_tokens", [])
        for t in added:
            vocab[t["content"]] = t["id"]
    
    sorted_tokens = sorted(vocab.items(), key=lambda x: x[1])
    tokens = [t[0] for t in sorted_tokens]
    
    if not tokens:
        # Maybe scores-based format (sentencepiece)
        model_data = tok_data.get("model", {})
        tok_ids = model_data.get("vocab", {})
        if isinstance(tok_ids, dict):
            sorted_tokens = sorted(tok_ids.items(), key=lambda x: x[1])
            tokens = [t[0] for t in sorted_tokens]
    
    writer.add_token_list(tokens)
    
    # Scores
    scores_data = tok_data.get("model", {}).get("scores", [])
    if scores_data and len(scores_data) == len(tokens):
        writer.add_token_scores(scores_data)
    
    # Type hints
    ty = []
    for i, t in enumerate(tokens):
        if t in ("<unk>", "<s>", "</s>", "<pad>", "<|endoftext|>", "<|im_end|>", "<|im_start|>"):
            ty.append(3)
        elif t.startswith("<0x"):
            ty.append(6)
        else:
            ty.append(1)
    writer.add_token_types(ty)
    
    merges = tok_data.get("model", {}).get("merges", [])
    if merges:
        # Convert from [["a","b"], ...] format to ["a b", ...] format
        merge_strs = []
        for m in merges:
            if isinstance(m, list):
                merge_strs.append(" ".join(m))
            elif isinstance(m, str):
                merge_strs.append(m)
        writer.add_token_merges(merge_strs)
    
    bos = tok_cfg.get("bos_token_id", config.get("bos_token_id"))
    eos = tok_cfg.get("eos_token_id", config.get("eos_token_id"))
    # Handle list values (e.g., [151645]) and None
    if isinstance(bos, list): bos = bos[0] if bos else None
    if isinstance(eos, list): eos = eos[0] if eos else None
    if isinstance(bos, (int,)):
        writer.add_bos_token_id(bos)
    if isinstance(eos, (int,)):
        writer.add_eos_token_id(eos)
    
    if chat_template:
        writer.add_chat_template(chat_template)
    
    # These sometimes fail with type errors in different gguf versions
    try:
        writer.add_add_bos_token(bool(add_bos))
    except: pass
    try:
        writer.add_add_eos_token(bool(add_eos))
    except: pass
    
    print(f"  Tokenizer: {len(tokens)} tokens, model={model_type}")


def convert(q4nx_path: str, output_path: str, arch_override: str = None):
    model_dir = os.path.dirname(os.path.abspath(q4nx_path))
    config_path = os.path.join(model_dir, "config.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found at {config_path}")
    
    print(f"Reading: {q4nx_path}")
    reader = Q4NXReader(q4nx_path)
    
    with open(config_path) as f:
        config = json.load(f)
    
    arch = arch_override or config.get("model_type", "qwen3").replace("_", "-")
    hidden = config.get("hidden_size", 0)
    
    writer = gguf.GGUFWriter(output_path, arch)
    
    # ── Metadata ──
    n_layers = config.get("num_hidden_layers", 0) or len(config.get("layer_types", []))
    writer.add_block_count(n_layers)
    writer.add_embedding_length(hidden)
    ff = config.get("intermediate_size") or config.get("moe_intermediate_size", 0)
    writer.add_feed_forward_length(ff)
    writer.add_head_count(config.get("num_attention_heads", 0))
    writer.add_head_count_kv(config.get("num_key_value_heads", 0))
    writer.add_layer_norm_rms_eps(config.get("rms_norm_eps", 1e-6))
    writer.add_file_type(gguf.GGMLQuantizationType.F16)
    if "rope_theta" in config:
        writer.add_rope_freq_base(config["rope_theta"])
    if config.get("max_position_embeddings"):
        writer.add_context_length(config["max_position_embeddings"])
    
    # Vocab — tell llama.cpp the real size so embedding dims match
    if config.get("vocab_size"):
        writer.add_vocab_size(config["vocab_size"])
    
    # MoE
    if "num_experts" in config:
        writer.add_expert_count(config["num_experts"])
        if config.get("num_experts_per_tok"):
            writer.add_expert_used_count(config["num_experts_per_tok"])
    if config.get("shared_expert_intermediate_size"):
        writer.add_shared_expert_feed_forward_length(config["shared_expert_intermediate_size"])
    
    # ── Tokenizer ──
    # Get token count BEFORE writing tensors (needed for embedding truncation)
    tok_path = os.path.join(model_dir, "tokenizer.json")
    actual_vocab = config.get("vocab_size", 0)
    token_count = 0
    if os.path.exists(tok_path):
        import json as _j
        with open(tok_path) as f:
            _tok_data = _j.load(f)
        _v = _tok_data.get("model", {}).get("vocab", {})
        token_count = len(_v)
        if token_count == 0:
            _ids = _tok_data.get("model", {}).get("vocab", {})
            if isinstance(_ids, dict):
                token_count = len(_ids)
    
    add_tokenizer(writer, model_dir, config)
    
    # ── Write tensors ──
    count = 0
    skipped_names = set()
    
    for t in sorted(reader.tensors, key=lambda x: x['offset']):
        name = t['name']
        if "vision" in name or "audio" in name:
            skipped_names.add(name)
            continue
        if "chat_template" in name:
            continue
        
        gguf_name = to_gguf_name(name)
        # Skip lm_head when weights are tied (llama.cpp uses token_embd for output)
        if gguf_name == 'output.weight' and config.get('tie_word_embeddings', False):
            continue
        raw = reader.get_tensor_raw(name)
        dtype = t['dtype']
        shape = t['shape']
        
        if dtype == 'BF16':
            # BF16 → F16
            u16 = np.frombuffer(raw, dtype=np.uint16).copy()
            f32 = (u16.astype(np.uint32) << 16).view(np.float32)
            arr = f32.astype(np.float16)
            if len(shape) >= 2:
                arr = arr.reshape(shape)
            # Truncate embedding to match tokenizer vocab (llama.cpp checks this)
            if gguf_name == 'token_embd.weight' and token_count > 0:
                if arr.shape[0] > token_count:
                    print(f"  Truncated token_embd from {arr.shape[0]} to {token_count}")
                    arr = arr[:token_count, :]
            writer.add_tensor(gguf_name, arr)
            count += 1
        
        elif dtype == 'I8':
            arr = np.frombuffer(raw, dtype=np.int8)
            if len(shape) == 2:
                arr = arr.reshape(shape[0], shape[1])
                if arr.shape[0] % ROW_BLOCK == 0 and arr.shape[1] % COL_BLOCK == 0:
                    arr = untile_i8(arr)
                # The NPU pads columns to 5120. Strip padding by taking first N cols.
                # For attention: expected cols = hidden_size or head_dim * n_heads, etc.
                # We know the expected shape from llama.cpp. Let's infer from config:
                exp_rows = shape[1]  # NPU col dim becomes HF row dim? 
                # Actually, we should look at the GGUF expected shapes
            elif len(shape) == 3:
                arr = arr.reshape(shape[0] * shape[1], shape[2])
                if arr.shape[0] % ROW_BLOCK == 0 and arr.shape[1] % COL_BLOCK == 0:
                    arr = untile_i8(arr)
            arr_f16 = arr.astype(np.float16)
            
            # Infer expected shape from config for I8 weight matrices
            # Q4NX stores [NPU_rows, 5120] but HF expects [out_dim, in_dim]
            # The NPU column dimension (5120) is always padded from hidden_size
            if gguf_name.startswith('blk.') and arr_f16.shape[1] == 5120 and '_norm' not in gguf_name:
                # Strip NPU padding: keep only the first `hidden` columns
                # The NPU stores 5120 = N * hidden_size columns. Take first hidden.
                arr_f16 = arr_f16[:, :hidden]
            
            writer.add_tensor(gguf_name, arr_f16)
            count += 1
        
        elif dtype == 'F32':
            arr = np.frombuffer(raw, dtype=np.float32)
            if len(shape) > 0:
                arr = arr.reshape(shape)
            writer.add_tensor(gguf_name, arr)
            count += 1
    
    if skipped_names:
        print(f"  Skipped {len(skipped_names)} vision/audio tensors")
    
    print(f"Writing {count} tensors to {output_path}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    size = os.path.getsize(output_path)
    print(f"Done: {output_path} ({size/1024/1024/1024:.2f} GB)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Q4NX → GGUF converter v3")
    parser.add_argument("q4nx_path")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--arch", default=None)
    args = parser.parse_args()
    
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.q4nx_path), "model.gguf")
    
    convert(args.q4nx_path, args.output, args.arch)
    
    print(f"\nRun on GPU:")
    print(f"  LD_LIBRARY_PATH={os.path.expanduser('~/.cache/lemonade/bin/llamacpp/vulkan')} \\")
    print(f"  {os.path.expanduser('~/.cache/lemonade/bin/llamacpp/vulkan/llama-server')} \\")
    print(f"    -m {args.output} --port 13305 --host 127.0.0.1 -ngl 99")


if __name__ == "__main__":
    main()

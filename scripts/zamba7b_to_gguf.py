#!/usr/bin/env python3
"""Zamba-7B-v1 → GGUF converter.

Zamba-7B-v1 architecture:
- 76 layers: pure Mamba1 SSM (64) + hybrid SSM+attention (12)
- Shared transformer weights (attention + FFN) stored in first hybrid layer (layer 2)
- Each hybrid layer has a linear mixing projection + mamba_decoder
- No MoE, d_state=16, d_conv=4, hidden=3712, d_inner=7424, dt_rank=232
- Vocab: 32000, RoPE, RMS norm
"""
import struct, sys, json, torch, numpy as np
from huggingface_hub import hf_hub_download

GGML_F16 = 1
GGML_F32 = 0

class Writer:
    def __init__(self, path, arch):
        self.f = open(path, 'wb')
        self.arch = arch
        self.kv = bytearray()
        self.ti = []
        self.td = bytearray()
        self.nk = 0
        self.nt = 0
        self.f.write(b'GGUF' + struct.pack('<II', 3, 0))

    def add_uint32(self, k, v):
        kb = k.encode()
        self.kv.extend(struct.pack('<Q', len(kb)) + kb + struct.pack('<I', 4) + struct.pack('<I', v))
        self.nk += 1

    def add_float32(self, k, v):
        kb = k.encode()
        self.kv.extend(struct.pack('<Q', len(kb)) + kb + struct.pack('<I', 6) + struct.pack('<f', v))
        self.nk += 1

    def add_string(self, k, v):
        kb = k.encode(); vb = v.encode()
        self.kv.extend(struct.pack('<Q', len(kb)) + kb + struct.pack('<I', 8) + struct.pack('<Q', len(vb)) + vb)
        self.nk += 1

    def add_string_array(self, k, arr):
        kb = k.encode()
        self.kv.extend(struct.pack('<Q', len(kb)) + kb + struct.pack('<I', 9))
        # GGUF array: element_type (8=string), then length, then elements
        self.kv.extend(struct.pack('<I', 8))
        self.kv.extend(struct.pack('<Q', len(arr)))
        for s in arr:
            sb = s.encode()
            self.kv.extend(struct.pack('<Q', len(sb)) + sb)
        self.nk += 1

    def add_tensor(self, name, data, dtype=GGML_F32):
        d = data.flatten()
        if dtype == GGML_F16:
            raw = d.astype(np.float16).tobytes()
        else:
            raw = d.astype(np.float32).tobytes()
        self.ti.append((name, data.shape, dtype))
        self.td.extend(raw)
        self.nt += 1

    def close(self):
        self.f.seek(8)
        self.f.write(struct.pack('<QQ', self.nt, self.nk))
        self.f.write(bytes(self.kv))
        align = 32
        offset = 0
        for name, shape, dtype in self.ti:
            nb = name.encode()
            self.f.write(struct.pack('<Q', len(nb)) + nb + struct.pack('<I', len(shape)))
            for s in shape:
                self.f.write(struct.pack('<Q', s))
            self.f.write(struct.pack('<I', dtype) + struct.pack('<Q', offset))
            n_elems = int(np.prod(shape))
            es = n_elems * (2 if dtype == GGML_F16 else 4)
            offset += es
        pos = self.f.tell()
        if pos % align:
            self.f.write(b'\x00' * (align - pos % align))
        self.f.write(bytes(self.td))
        self.f.close()
        print(f"Written {self.nt} tensors to {self.f.name}")


def convert(output):
    # Download model index and config
    idx_path = hf_hub_download('Zyphra/Zamba-7B-v1', 'model.safetensors.index.json')
    cfg_path = hf_hub_download('Zyphra/Zamba-7B-v1', 'config.json')
    tok_path = hf_hub_download('Zyphra/Zamba-7B-v1', 'tokenizer.json')

    idx = json.load(open(idx_path))
    cfg = json.load(open(cfg_path))
    weight_map = idx['weight_map']

    H = cfg['hidden_size']          # 3712
    nl = cfg['num_hidden_layers']   # 76
    V = cfg['vocab_size']           # 32000
    di = H * cfg['mamba_expand']    # 7424
    ds = cfg['mamba_d_state']       # 16
    dc = cfg['mamba_d_conv']        # 4
    dt_rank = cfg['mamba_dt_rank']  # 232

    print(f"Zamba-7B-v1: H={H} L={nl} d_state={ds} d_conv={dc} V={V} dt_rank={dt_rank}")

    # Load state dict shards
    shards = set(weight_map.values())
    sd = {}
    for shard in shards:
        print(f"  Loading {shard}...")
        path = hf_hub_download('Zyphra/Zamba-7B-v1', shard)
        shard_sd = torch.load(path, map_location='cpu', weights_only=True)
        sd.update(shard_sd)
        del shard_sd

    w = Writer(output, "zamba")

    # Metadata
    w.add_string("general.architecture", "zamba")
    w.add_uint32("zamba.block_count", nl)
    w.add_uint32("zamba.embedding_length", H)
    w.add_uint32("zamba.feed_forward_length", cfg['intermediate_size'])
    w.add_uint32("zamba.context_length", cfg['max_position_embeddings'])
    w.add_float32("zamba.attention.layer_norm_rms_epsilon", cfg['rms_norm_eps'])
    w.add_uint32("zamba.attention.head_count", cfg['num_attention_heads'])
    w.add_uint32("zamba.attention.head_count_kv", cfg['num_key_value_heads'])
    w.add_float32("zamba.rope.freq_base", cfg.get('rope_theta', 10000.0))
    w.add_uint32("zamba.ssm.conv_kernel", dc)
    w.add_uint32("zamba.ssm.inner_size", di)
    w.add_uint32("zamba.ssm.state_size", ds)
    w.add_uint32("zamba.ssm.dt_rank", dt_rank)
    w.add_uint32("zamba.vocab_size", V)
    w.add_uint32("zamba.attn_layer_offset", cfg['attn_layer_offset'])
    w.add_uint32("zamba.attn_layer_period", cfg['attn_layer_period'])

    # Tokenizer
    import tokenizers
    tok = tokenizers.Tokenizer.from_file(tok_path)
    vocab = tok.get_vocab()
    tokens = [''] * V
    for word, id in vocab.items():
        if id < V: tokens[id] = word
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_string_array("tokenizer.ggml.tokens", tokens)
    w.add_uint32("tokenizer.ggml.bos_token_id", cfg['bos_token_id'])
    w.add_uint32("tokenizer.ggml.eos_token_id", cfg['eos_token_id'])

    # Embedding + output norm
    emb = sd['model.embed_tokens.weight'].to(torch.float32).numpy()
    w.add_tensor("token_embd.weight", emb, GGML_F16)
    fn = sd['model.final_layernorm.weight'].to(torch.float32).numpy()
    w.add_tensor("output_norm.weight", fn)
    # LM head (tied embeddings)
    w.add_tensor("output.weight", emb, GGML_F16)

    # Shared transformer weights (from first hybrid layer)
    shared_prefix = "model.layers.2.shared_transf"
    shared_attn = {
        "self_attn.q_proj": sd[f"{shared_prefix}.self_attn.q_proj.weight"],
        "self_attn.k_proj": sd[f"{shared_prefix}.self_attn.k_proj.weight"],
        "self_attn.v_proj": sd[f"{shared_prefix}.self_attn.v_proj.weight"],
        "self_attn.o_proj": sd[f"{shared_prefix}.self_attn.o_proj.weight"],
        "feed_forward.gate_proj": sd[f"{shared_prefix}.feed_forward.gate_proj.weight"],
        "feed_forward.up_proj": sd[f"{shared_prefix}.feed_forward.up_proj.weight"],
        "feed_forward.down_proj": sd[f"{shared_prefix}.feed_forward.down_proj.weight"],
    }
    shared_norms = {
        "input_layernorm": sd[f"{shared_prefix}.input_layernorm.weight"],
        "pre_ff_layernorm": sd[f"{shared_prefix}.pre_ff_layernorm.weight"],
    }

    # Per-layer loop
    hybrid_indices = [i for i, t in enumerate(cfg['layers_block_type']) if t == 'hybrid']

    for l in range(nl):
        prefix = f"model.layers.{l}"
        is_hybrid = cfg['layers_block_type'][l] == 'hybrid'

        if is_hybrid:
            # Hybrid layer: has mamba_decoder + linear mixing
            md = f"{prefix}.mamba_decoder"
            in_n = sd[f"{md}.input_layernorm.weight"]
            in_proj = sd[f"{md}.mamba.in_proj.weight"]
            c1w = sd[f"{md}.mamba.conv1d.weight"].to(torch.float32).numpy().reshape(di, dc)
            c1b = sd[f"{md}.mamba.conv1d.bias"]
            xp = sd[f"{md}.mamba.x_proj_weight"]
            dpw = sd[f"{md}.mamba.dt_proj_weight"]
            dpb = sd[f"{md}.mamba.dt_proj_bias"]
            A_log = sd[f"{md}.mamba.A_log"]
            Dv = sd[f"{md}.mamba.D"]
            out_proj = sd[f"{md}.mamba.out_proj.weight"]
            linear_w = sd[f"{prefix}.linear.weight"]

            w.add_tensor(f"blk.{l}.attn_norm.weight", in_n.to(torch.float32).numpy())
            w.add_tensor(f"blk.{l}.ssm_in.weight", in_proj.to(torch.float32).numpy().T, GGML_F16)
            w.add_tensor(f"blk.{l}.ssm_conv1d.weight", c1w.astype(np.float32), GGML_F32)
            w.add_tensor(f"blk.{l}.ssm_conv1d.bias", c1b.to(torch.float32).numpy())
            w.add_tensor(f"blk.{l}.ssm_x.weight", xp.to(torch.float32).numpy().T, GGML_F16)
            w.add_tensor(f"blk.{l}.ssm_dt.weight", dpw.to(torch.float32).numpy().T, GGML_F32)
            w.add_tensor(f"blk.{l}.ssm_dt.bias", dpb.to(torch.float32).numpy())
            w.add_tensor(f"blk.{l}.ssm_a", A_log.to(torch.float32).numpy().T.flatten(), GGML_F32)
            w.add_tensor(f"blk.{l}.ssm_d", Dv.to(torch.float32).numpy(), GGML_F32)
            w.add_tensor(f"blk.{l}.ssm_out.weight", out_proj.to(torch.float32).numpy().T, GGML_F16)
            w.add_tensor(f"blk.{l}.linear.weight", linear_w.to(torch.float32).numpy().T, GGML_F16)

        else:
            # Pure Mamba layer
            in_n = sd[f"{prefix}.input_layernorm.weight"]
            in_proj = sd[f"{prefix}.mamba.in_proj.weight"]
            c1w = sd[f"{prefix}.mamba.conv1d.weight"].to(torch.float32).numpy().reshape(di, dc)
            c1b = sd[f"{prefix}.mamba.conv1d.bias"]
            xp = sd[f"{prefix}.mamba.x_proj_weight"]
            dpw = sd[f"{prefix}.mamba.dt_proj_weight"]
            dpb = sd[f"{prefix}.mamba.dt_proj_bias"]
            A_log = sd[f"{prefix}.mamba.A_log"]
            Dv = sd[f"{prefix}.mamba.D"]
            out_proj = sd[f"{prefix}.mamba.out_proj.weight"]

            w.add_tensor(f"blk.{l}.attn_norm.weight", in_n.to(torch.float32).numpy())
            w.add_tensor(f"blk.{l}.ssm_in.weight", in_proj.to(torch.float32).numpy().T, GGML_F16)
            w.add_tensor(f"blk.{l}.ssm_conv1d.weight", c1w.astype(np.float32), GGML_F32)
            w.add_tensor(f"blk.{l}.ssm_conv1d.bias", c1b.to(torch.float32).numpy())
            w.add_tensor(f"blk.{l}.ssm_x.weight", xp.to(torch.float32).numpy().T, GGML_F16)
            w.add_tensor(f"blk.{l}.ssm_dt.weight", dpw.to(torch.float32).numpy().T, GGML_F32)
            w.add_tensor(f"blk.{l}.ssm_dt.bias", dpb.to(torch.float32).numpy())
            w.add_tensor(f"blk.{l}.ssm_a", A_log.to(torch.float32).numpy().T.flatten(), GGML_F32)
            w.add_tensor(f"blk.{l}.ssm_d", Dv.to(torch.float32).numpy(), GGML_F32)
            w.add_tensor(f"blk.{l}.ssm_out.weight", out_proj.to(torch.float32).numpy().T, GGML_F16)

        if l == 2:  # First hybrid layer — write shared transformer weights
            for name, tensor in shared_attn.items():
                t = tensor.to(torch.float32).numpy()
                w.add_tensor(f"blk.{l}.shared_transf.{name}.weight", t.T, GGML_F16)
            for name, tensor in shared_norms.items():
                w.add_tensor(f"blk.{l}.shared_transf.{name}.weight",
                             tensor.to(torch.float32).numpy())

        print(f"  Layer {l}/{nl} {'[hybrid]' if is_hybrid else '[mamba]'} done")

    w.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} output.gguf")
        sys.exit(1)
    convert(sys.argv[1])

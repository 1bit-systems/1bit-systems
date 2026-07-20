#!/usr/bin/env python3
"""Standalone BlackMamba → GGUF converter."""
import struct, sys, json, torch, numpy as np
from huggingface_hub import hf_hub_download

GGML_Q4_0 = 2
GGML_F16 = 1
GGML_F32 = 0

def quant_q4_0(data):
    n = len(data)
    out = bytearray()
    for i in range(0, n, 32):
        blk = data[i:i+32]
        amax = np.max(np.abs(blk))
        scale = amax / 7.0 if amax > 0 else 1e-10
        out.extend(struct.pack('<e', np.float16(scale)))
        for j in range(0, 32, 2):
            v0 = blk[j] / scale if j < len(blk) else 0
            v1 = blk[j+1] / scale if j+1 < len(blk) else 0
            q0 = max(-8, min(7, int(round(v0)))) & 0x0F
            q1 = max(-8, min(7, int(round(v1)))) & 0x0F if j+1 < len(blk) else 0
            out.append((q0 << 4) | q1)
    return bytes(out)

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
    def add_tensor(self, name, data, dtype=GGML_F32):
        d = data.flatten()
        if dtype == GGML_Q4_0:
            raw = quant_q4_0(d)
        elif dtype == GGML_F16:
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
        # No alignment padding here -- GGUF's tensor-info section immediately
        # follows KV metadata with no gap. Padding belongs only right before
        # the tensor *data* section below. A stray pad here was previously
        # dormant only because the old (unprefixed) metadata keys happened to
        # land the KV section on a 32-byte boundary by coincidence; prefixing
        # keys with the architecture name changed that byte count and exposed
        # a real corrupt-file bug (readers found garbage tensor names/n_dims).
        align = 32
        offset = 0
        for name, shape, dtype in self.ti:
            nb = name.encode()
            self.f.write(struct.pack('<Q', len(nb)) + nb + struct.pack('<I', len(shape)))
            for s in shape:
                self.f.write(struct.pack('<Q', s))
            self.f.write(struct.pack('<I', dtype) + struct.pack('<Q', offset))
            n_elems = int(np.prod(shape))
            if dtype == GGML_Q4_0:
                es = ((n_elems + 31) // 32) * 18
            elif dtype == GGML_F16:
                es = n_elems * 2
            else:
                es = n_elems * 4
            offset += es
        pos = self.f.tell()
        if pos % align:
            self.f.write(b'\x00' * (align - pos % align))
        self.f.write(bytes(self.td))
        self.f.close()
        print(f"Written {self.nt} tensors to {self.f.name}")

def convert(model_id, output):
    path = hf_hub_download(model_id, "pytorch_model.bin")
    sd = torch.load(path, map_location='cpu')
    sd = sd['model'] if 'model' in sd else sd
    
    cfg = json.loads(open(hf_hub_download(model_id, "config.json")).read())
    H = cfg["hidden_size"]
    nl = cfg["num_layers"]
    ds = cfg["state_size"]
    dc = cfg["conv_dimension"]
    V = cfg["vocab_size"]
    di = H * 2
    dt_rank = cfg.get("dt_rank", ds)
    
    print(f"BlackMamba: H={H} L={nl} d_state={ds} d_conv={dc} V={V}")
    
    w = Writer(output, "mamba")
    w.add_string("general.architecture", "mamba")
    # GGUF convention (and every reader in this repo, e.g. model_discovery.cpp's
    # `ends_with(key, ".block_count")` suffix matching) expects these keys
    # prefixed with the architecture name — bare "block_count" etc. matched
    # neither the HF-style nor the architecture-prefixed lookup in any reader,
    # so BlackMamba's H/L/IM always came back 0 (config silently unreadable).
    w.add_uint32("mamba.block_count", nl)
    w.add_uint32("mamba.embedding_length", H)
    w.add_uint32("mamba.feed_forward_length", di)
    w.add_uint32("mamba.context_length", 2048)
    w.add_float32("mamba.attention.layer_norm_rms_epsilon", 1e-5)
    w.add_uint32("mamba.ssm.conv_kernel", dc)
    w.add_uint32("mamba.ssm.inner_size", di)
    w.add_uint32("mamba.ssm.state_size", ds)
    w.add_uint32("mamba.ssm.dt_rank", dt_rank)
    w.add_uint32("mamba.ssm.group_count", 1)
    w.add_uint32("mamba.vocab_size", V)
    
    emb = sd["embedding.word_embeddings.weight"].to(torch.float32).numpy()
    w.add_tensor("token_embd.weight", emb, GGML_F16)
    
    fn = sd["decoder.final_layernorm.weight"].to(torch.float32).numpy()
    w.add_tensor("output_norm.weight", fn)
    
    for l in range(nl):
        print(f"  Layer {l}/{nl}...", end=' ', flush=True)
        in_n = sd.get(f"decoder.layers.{l}.norm.weight")
        if in_n is not None:
            in_n = in_n.to(torch.float32).numpy()
            w.add_tensor(f"blk.{l}.attn_norm.weight", in_n)
        
        mb = f"decoder.layers.{l}.mixer"
        
        if f"{mb}.in_proj.weight" in sd:
            ip = sd[f"{mb}.in_proj.weight"].to(torch.float32).numpy()
            c1w = sd[f"{mb}.conv1d.weight"].to(torch.float32).numpy().reshape(dc, di)
            c1b = sd[f"{mb}.conv1d.bias"].to(torch.float32).numpy()
            xp = sd[f"{mb}.x_proj.weight"].to(torch.float32).numpy()
            dpw = sd[f"{mb}.dt_proj.weight"].to(torch.float32).numpy()
            dpb = sd[f"{mb}.dt_proj.bias"].to(torch.float32).numpy()
            A_log = sd[f"{mb}.A_log"].to(torch.float32).numpy()
            Dv = sd[f"{mb}.D"].to(torch.float32).numpy()
            out = sd[f"{mb}.out_proj.weight"].to(torch.float32).numpy()
            
            w.add_tensor(f"blk.{l}.ssm_in.weight", ip.T, GGML_Q4_0)
            w.add_tensor(f"blk.{l}.ssm_conv1d.weight", c1w, GGML_F32)
            w.add_tensor(f"blk.{l}.ssm_conv1d.bias", c1b)
            w.add_tensor(f"blk.{l}.ssm_x.weight", xp.T, GGML_Q4_0)
            w.add_tensor(f"blk.{l}.ssm_dt.weight", dpw.T, GGML_F32)
            w.add_tensor(f"blk.{l}.ssm_dt.bias", dpb)
            w.add_tensor(f"blk.{l}.ssm_a", A_log.T.flatten())
            w.add_tensor(f"blk.{l}.ssm_d", Dv)
            w.add_tensor(f"blk.{l}.ssm_out.weight", out.T, GGML_Q4_0)
        
        # MoE
        router = sd.get(f"{mb}.router.weight")
        if router is not None:
            w.add_tensor(f"blk.{l}.ffn_gate.weight", router.to(torch.float32).numpy().T, GGML_Q4_0)
            n_exp = router.shape[0]
            for e in range(n_exp):
                fc1 = sd[f"{mb}.local_experts.{e}.linear_fc1.weight"].to(torch.float32).numpy()
                fc2 = sd[f"{mb}.local_experts.{e}.linear_fc2.weight"].to(torch.float32).numpy()
                w.add_tensor(f"blk.{l}.ffn_expert.{e}.weight_1", fc1.T, GGML_Q4_0)
                w.add_tensor(f"blk.{l}.ffn_expert.{e}.weight_2", fc2.T, GGML_Q4_0)
        print("done")
    
    # LM head
    w.add_tensor("output.weight", emb, GGML_Q4_0)
    w.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 blackmamba_to_gguf.py Zyphra/BlackMamba-1.5B ./blackmamba.gguf")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])

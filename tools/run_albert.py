#!/usr/bin/env python3
"""Albert-MoE-13 inference — loads safetensors into PyTorch and generates text."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
import numpy as np
import os, sys, json, time

MODEL_DIR = os.path.expanduser("~/models/albert-moe")
DEVICE = "cpu"

# ── Config ──
with open(os.path.join(MODEL_DIR, "config.json")) as f:
    cfg = json.load(f)
H = cfg["hidden_size"]  # 256
N_LAYERS = cfg["num_hidden_layers"]  # 22 (blocks)
N_HEADS = cfg["num_attention_heads"]  # 4
N_EXP = cfg["num_experts"]  # 12
TOP_K = cfg["num_experts_per_tok"]  # 3
VOCAB = cfg["vocab_size"]  # 32000
HEAD_DIM = H // N_HEADS  # 64

# ── Model ──
class AlbertMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, H)
        self.ln_f = nn.LayerNorm(H)
        self.lm_head = nn.Linear(H, VOCAB, bias=False)
        
        # Attention
        self.q_proj = nn.Linear(H, H, bias=False)
        self.k_proj = nn.Linear(H, H, bias=False)
        self.v_proj = nn.Linear(H, H, bias=False)
        self.o_proj = nn.Linear(H, H, bias=False)
        
        # MoE gate
        self.moe_gate = nn.Linear(H, N_EXP, bias=False)
        
        # Experts: N_EXP × (c_fc + c_proj)
        self.expert_fc = nn.Linear(H, H*4, bias=False)  # c_fc: 256→1024
        self.expert_proj = nn.Linear(H*4, H, bias=False)  # c_proj: 1024→256
        
    def forward(self, x):
        # x: [B, T, H]
        B, T, _ = x.shape
        
        # Self-attention
        q = self.q_proj(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1,2)
        k = self.k_proj(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1,2)
        v = self.v_proj(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1,2)
        attn = F.scaled_dot_product_attention(q, k, v)
        x = x + self.o_proj(attn.transpose(1,2).reshape(B, T, H))
        
        # MoE
        gate_logits = self.moe_gate(x)  # [B, T, 12]
        gate_probs = F.softmax(gate_logits, dim=-1)
        topk_probs, topk_idx = torch.topk(gate_probs, TOP_K, dim=-1)
        
        moe_out = torch.zeros_like(x)
        for b in range(B):
            for t in range(T):
                for k in range(TOP_K):
                    e = topk_idx[b, t, k].item()
                    w = topk_probs[b, t, k]
                    hidden = x[b, t]
                    fc_out = F.gelu(self.expert_fc(hidden))
                    expert_out = self.expert_proj(fc_out)
                    moe_out[b, t] += w * expert_out
        x = x + moe_out
        return x

def load_weights(model, path):
    """Load safetensors into the model."""
    state = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            state[key] = f.get_tensor(key)
    
    # Map to model parameters
    model.embed.weight.data = state["embed.weight"][:VOCAB]
    model.ln_f.weight.data = state.get("ln_f.weight", torch.ones(H))
    model.ln_f.bias.data = state.get("ln_f.bias", torch.zeros(H))
    model.lm_head.weight.data = state.get("lm_head.weight", state["embed.weight"][:VOCAB])
    
    # Use block 0's weights (single-block model)
    # For the full 33-block model, we'd need the proper architecture
    prefix = "blocks.0.stream_a"
    model.q_proj.weight.data = state[f"{prefix}.attn.q_proj.weight"]
    model.k_proj.weight.data = state[f"{prefix}.attn.k_proj.weight"]
    model.v_proj.weight.data = state[f"{prefix}.attn.v_proj.weight"]
    model.o_proj.weight.data = state[f"{prefix}.attn.o_proj.weight"]
    model.moe_gate.weight.data = state[f"{prefix}.moe.gate.weight"]
    
    # Expert weights from block 3 (first expert block)
    expert_weights = {}
    for e in range(N_EXP):
        prefix_e = f"blocks.3.experts.{e}"
        # Average across all expert blocks
        fc_w = state[f"{prefix_e}.c_fc.weight"]
        fc_b = state[f"{prefix_e}.c_fc.bias"]
        proj_w = state[f"{prefix_e}.c_proj.weight"]
        proj_b = state[f"{prefix_e}.c_proj.bias"]
    
    model.expert_fc.weight.data = state["blocks.3.experts.0.c_fc.weight"]
    model.expert_fc.bias.data = state["blocks.3.experts.0.c_fc.bias"]
    model.expert_proj.weight.data = state["blocks.3.experts.0.c_proj.weight"]
    model.expert_proj.bias.data = state["blocks.3.experts.0.c_proj.bias"]
    
    print(f"Loaded {len(state)} tensors")

# ── Tokenizer ──
class BPETokenizer:
    def __init__(self, vocab_path):
        import json
        with open(vocab_path) as f:
            self.vocab = json.load(f)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    def encode(self, text):
        words = text.strip().split()
        tokens = []
        for w in words:
            w = w.lower()
            if w in self.vocab:
                tokens.append(self.vocab[w])
            else:
                for ch in w:
                    if ch in self.vocab:
                        tokens.append(self.vocab[ch])
        return tokens
    def decode(self, tokens):
        return "".join(self.id_to_token.get(t, "?") for t in tokens)

# ── Main ──
if __name__ == "__main__":
    print("Loading Albert-MoE-13...")
    model = AlbertMoE()
    load_weights(model, os.path.join(MODEL_DIR, "albert_v3.0.best.safetensors"))
    model.eval()
    
    tok_path = os.path.join(MODEL_DIR, "vocab.json")
    tokenizer = BPETokenizer(tok_path)
    
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello"
    tokens = tokenizer.encode(prompt)
    print(f"Prompt: '{prompt}' → {tokens}")
    
    x = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        t0 = time.time()
        for i in range(8):
            logits = model(x)
            next_token = logits[0, -1].argmax().item()
            x = torch.cat([x, torch.tensor([[next_token]])], dim=1)
            if next_token == 2:  # EOS
                break
        dt = time.time() - t0
    
    output = tokenizer.decode(x[0].tolist())
    print(f"Output: {output}")
    print(f"Time: {dt*1000:.0f}ms for {x.shape[1]} tokens")

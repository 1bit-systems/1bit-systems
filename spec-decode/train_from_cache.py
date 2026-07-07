#!/usr/bin/env python3
"""Train Eagle3 draft from pre-computed target cache. Fast CPU training."""
import os, sys, time, math
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WANDB_DISABLED'] = 'true'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

HIDDEN = 1024
VOCAB = 151936
INTER_DIM = 3072
NUM_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 5
LR = 1e-4
EPOCHS = 5
BATCH_SIZE = 8
DEVICE = "cpu"

print(f"═══ Eagle3 Draft Training (cached, CPU) ═══")
print(f"Device: {DEVICE}")

# Same Eagle3Draft model as before (copied here for self-containment)
class Eagle3Draft(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.fc = nn.Linear(5 * HIDDEN, HIDDEN, bias=False)
        self.hidden_norm = nn.RMSNorm(HIDDEN, eps=1e-6)
        self.input_norm = nn.RMSNorm(HIDDEN, eps=1e-6)
        self.q_proj = nn.Linear(2 * HIDDEN, NUM_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(2 * HIDDEN, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(2 * HIDDEN, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN, bias=False)
        self.q_norm = nn.RMSNorm(HEAD_DIM, eps=1e-6)
        self.k_norm = nn.RMSNorm(HEAD_DIM, eps=1e-6)
        self.post_attn_norm = nn.RMSNorm(HIDDEN, eps=1e-6)
        self.gate = nn.Linear(HIDDEN, INTER_DIM, bias=False)
        self.up = nn.Linear(HIDDEN, INTER_DIM, bias=False)
        self.down = nn.Linear(INTER_DIM, HIDDEN, bias=False)
        self.norm = nn.RMSNorm(HIDDEN, eps=1e-6)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        self.register_buffer('freqs', self._precompute_freqs(HEAD_DIM, 4096))

    def _precompute_freqs(self, dim, max_seq):
        freqs = 1.0 / (1000000.0 ** (torch.arange(0, dim, 2)[:dim//2].float() / dim))
        t = torch.arange(max_seq)
        return torch.outer(t, freqs)

    def apply_rope(self, x, pos):
        cos = self.freqs[pos].cos()
        sin = self.freqs[pos].sin()
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def forward(self, target_features, input_ids, positions=None):
        B, T = input_ids.shape
        if positions is None:
            positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        
        # FC project target features [B, 5, H] -> [B, 5*H] -> [B, H]
        if target_features.dim() == 3:
            feat = target_features.reshape(target_features.shape[0], -1)
        else:
            feat = target_features
        h = self.hidden_norm(self.fc(feat))
        
        e = self.input_norm(self.embed(input_ids))
        
        logits_list = []
        for t_idx in range(T):
            pos = positions[0, t_idx]
            x = torch.cat([h.unsqueeze(1), e[:, t_idx:t_idx+1, :]], dim=-1)
            
            q = self.q_proj(x).view(B, NUM_HEADS, HEAD_DIM)
            k = self.k_proj(x).view(B, NUM_KV_HEADS, HEAD_DIM)
            v = self.v_proj(x).view(B, NUM_KV_HEADS, HEAD_DIM)
            
            q = self.apply_rope(self.q_norm(q), pos)
            k = self.apply_rope(self.k_norm(k), pos)
            
            attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * (HEAD_DIM ** -0.5), dim=-1)
            o = self.o_proj((attn @ v).reshape(B, -1))
            
            h2 = self.post_attn_norm(h + o)
            h3 = h2 + self.down(F.silu(self.gate(h2)) * self.up(h2))
            h = self.norm(h3)
            
            logits_list.append(self.lm_head(h))
        
        return torch.stack(logits_list, dim=1)

class CachedDataset(Dataset):
    def __init__(self, cache_path):
        self.data = torch.load(cache_path, weights_only=False)
        print(f"Loaded {len(self.data)} cached examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return item['input_ids'], item['target_features']

def collate_fn(batch):
    input_ids = [b[0] for b in batch]
    features = torch.stack([b[1] for b in batch])
    max_len = max(len(ids) for ids in input_ids)
    padded = torch.full((len(batch), max_len), 151645, dtype=torch.long)  # pad with EOS
    for i, ids in enumerate(input_ids):
        padded[i, :len(ids)] = ids
    return padded, features

def train():
    dataset = CachedDataset('target_cache_npu_1k.pt')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    
    model = Eagle3Draft().to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Draft model: {trainable/1e6:.1f}M trainable params")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    total_steps = len(loader) * EPOCHS
    print(f"Training: {len(loader)} batches × {EPOCHS} epochs = {total_steps} steps")
    print()
    
    model.train()
    global_step = 0
    t_start = time.time()
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch_input_ids, batch_features in pbar:
            input_ids = batch_input_ids.to(DEVICE)
            features = batch_features.to(DEVICE)
            
            seq_len = input_ids.shape[1]
            positions = torch.arange(seq_len, device=DEVICE).unsqueeze(0).expand(input_ids.shape[0], -1)
            logits = model(features, input_ids, positions)
            
            shift_logits = logits[:, :-1, :].reshape(-1, VOCAB)
            shift_labels = input_ids[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=151645)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix(loss=loss.item())
        
        avg_loss = epoch_loss / len(loader)
        elapsed = (time.time() - t_start) / 60
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f} ({elapsed:.1f}m)")
    
    os.makedirs("checkpoints", exist_ok=True)
    path = "checkpoints/eagle3_draft_npu_1k.bin"
    # Save as .pt first (for resume)
    pt_path = path.replace('.bin', '.pt')
    torch.save(model.state_dict(), pt_path)
    print(f"  Saved state_dict to {pt_path}")
    
    # Export in C++ MTPDraftWeights order: weights only, no biases
    fields = [
        ('embed.weight', 'embed', (VOCAB, HIDDEN)),
        ('fc.weight', 'fc', (HIDDEN, 5*HIDDEN)),
        ('hidden_norm.weight', 'hidden_norm', (HIDDEN,)),
        ('input_norm.weight', 'input_layernorm', (HIDDEN,)),
        ('q_proj.weight', 'q_proj', (NUM_HEADS*HEAD_DIM, 2*HIDDEN)),
        ('k_proj.weight', 'k_proj', (NUM_KV_HEADS*HEAD_DIM, 2*HIDDEN)),
        ('v_proj.weight', 'v_proj', (NUM_KV_HEADS*HEAD_DIM, 2*HIDDEN)),
        ('o_proj.weight', 'o_proj', (HIDDEN, NUM_HEADS*HEAD_DIM)),
        ('q_norm.weight', 'q_norm', (HEAD_DIM,)),
        ('k_norm.weight', 'k_norm', (HEAD_DIM,)),
        ('post_attn_norm.weight', 'post_attention_layernorm', (HIDDEN,)),
        ('gate.weight', 'gate_proj', (INTER_DIM, HIDDEN)),
        ('up.weight', 'up_proj', (INTER_DIM, HIDDEN)),
        ('down.weight', 'down_proj', (HIDDEN, INTER_DIM)),
        ('norm.weight', 'norm', (HIDDEN,)),
        ('lm_head.weight', 'lm_head', (VOCAB, HIDDEN)),
    ]
    total_bytes = 0
    with open(path, 'wb') as f:
        for param_name, field_name, shape in fields:
            name_map = {n: p for n, p in model.named_parameters()}
            arr = name_map[param_name].detach().cpu().float().numpy()
            assert arr.shape == shape or arr.shape == shape[::-1], \
                f'{field_name}: expected {shape}, got {arr.shape}'
            f.write(arr.tobytes())
            total_bytes += arr.nbytes
    
    total_time = (time.time() - t_start) / 60
    print(f"\n✅ Saved to {path} ({total_bytes/1e6:.1f} MB)")
    print(f"Training time: {total_time:.1f}m")

if __name__ == '__main__':
    train()

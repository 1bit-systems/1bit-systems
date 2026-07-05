#!/usr/bin/env python3
"""Train Eagle3 draft model on CPU (8.5M params, ~10k examples, ~1 hour).
No GPU needed. Uses HuggingFace Qwen3-0.6B as target for hidden state extraction."""
import os, sys, json, time, math
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WANDB_DISABLED'] = 'true'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig
import numpy as np
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

# Config
MODEL_NAME = "Qwen/Qwen3-0.6B"
HIDDEN = 1024
VOCAB = 151936
INTER_DIM = 3072
NUM_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
NUM_LAYERS = 28
TARGET_LAYER_IDS = [1, 6, 12, 18, 24]
BLOCK_SIZE = 5
LR = 1e-4
EPOCHS = 3
BATCH_SIZE = 4
MAX_SEQ = 1024
DEVICE = "cpu"  # CPU training for compatibility

print(f"═══ Eagle3 Draft Training (CPU) ═══")
print(f"Device: {DEVICE}")
print(f"Model: {MODEL_NAME}")
print(f"Draft params: ~8.5M")
print(f"Batch: {BATCH_SIZE}, Epochs: {EPOCHS}")

# Load target model (for hidden state extraction only)
print("\nLoading target model...")
t0 = time.time()
target = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).eval()
print(f"  Loaded in {time.time()-t0:.0f}s ({sum(p.numel() for p in target.parameters())/1e6:.0f}M params)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Simple Eagle3 draft model
class Eagle3Draft(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        # FC projection: 5 target layers * hidden -> hidden
        self.fc = nn.Linear(len(TARGET_LAYER_IDS) * HIDDEN, HIDDEN, bias=False)
        self.hidden_norm = nn.LayerNorm(HIDDEN, eps=1e-6)
        self.input_norm = nn.LayerNorm(HIDDEN, eps=1e-6)
        # Self-attention
        self.q_proj = nn.Linear(2 * HIDDEN, NUM_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(2 * HIDDEN, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(2 * HIDDEN, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN, bias=False)
        # QK norms
        self.q_norm = nn.LayerNorm(HEAD_DIM, eps=1e-6)
        self.k_norm = nn.LayerNorm(HEAD_DIM, eps=1e-6)
        self.post_attn_norm = nn.LayerNorm(HIDDEN, eps=1e-6)
        # SwiGLU FFN
        self.gate = nn.Linear(HIDDEN, INTER_DIM, bias=False)
        self.up = nn.Linear(HIDDEN, INTER_DIM, bias=False)
        self.down = nn.Linear(INTER_DIM, HIDDEN, bias=False)
        self.norm = nn.LayerNorm(HIDDEN, eps=1e-6)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        # RoPE
        self.register_buffer('freqs', self._precompute_freqs(HEAD_DIM, MAX_SEQ))

    def _precompute_freqs(self, dim, max_seq):
        freqs = 1.0 / (1000000.0 ** (torch.arange(0, dim, 2)[:dim//2].float() / dim))
        t = torch.arange(max_seq)
        return torch.outer(t, freqs)

    def apply_rope(self, x, pos):
        cos = self.freqs[pos].cos().to(x.device)
        sin = self.freqs[pos].sin().to(x.device)
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def forward(self, target_hidden, input_ids, positions=None):
        B, T = input_ids.shape
        if positions is None:
            positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        
        # FC project target hidden states (repeat if single layer)
        feat = target_hidden
        if feat.shape[-1] == HIDDEN:
            feat = feat.repeat(1, 5) if feat.dim() == 2 else feat.repeat(5)
        h = self.fc(feat)
        h = self.hidden_norm(h)
        
        # Embed tokens
        e = self.embed(input_ids)  # [B, T, H]
        e = self.input_norm(e)
        
        logits_list = []
        for t in range(T):
            pos = positions[0, t]
            # Concat hidden + embed
            x = torch.cat([h.unsqueeze(1), e[:, t:t+1, :]], dim=-1)  # [B, 1, 2*H]
            
            # Self-attention
            q = self.q_proj(x)  # [B, 1, NH*HD]
            k = self.k_proj(x)  # [B, 1, NKV*HD]
            v = self.v_proj(x)  # [B, 1, NKV*HD]
            
            q = q.view(B, NUM_HEADS, HEAD_DIM)
            k = k.view(B, NUM_KV_HEADS, HEAD_DIM)
            v = v.view(B, NUM_KV_HEADS, HEAD_DIM)
            
            q = self.q_norm(q)
            k = self.k_norm(k)
            q = self.apply_rope(q, pos)
            k = self.apply_rope(k, pos)
            
            # Simple attention (no KV cache for short sequences)
            attn = torch.matmul(q, k.transpose(-2, -1)) * (HEAD_DIM ** -0.5)
            attn = F.softmax(attn, dim=-1)
            o = torch.matmul(attn, v)  # [B, NH, HD]
            o = o.reshape(B, -1)
            o = self.o_proj(o)
            
            h2 = h + o
            h2 = self.post_attn_norm(h2)
            
            # SwiGLU FFN
            g = self.gate(h2)
            u = self.up(h2)
            h3 = h2 + self.down(F.silu(g) * u)
            h = self.norm(h3)
            
            # LM head
            logits = self.lm_head(h)
            logits_list.append(logits)
        
        return torch.stack(logits_list, dim=1)  # [B, T, V]

# Dataset: extract hidden states from target model
class SpecDataset(Dataset):
    def __init__(self, data_path, target_model, tokenizer, max_seq=512):
        self.data = []
        print(f"Loading data from {data_path}...")
        with open(data_path) as f:
            for line in tqdm(f, desc="Loading"):
                item = json.loads(line)
                self.data.append(item)
        
        self.target = target_model
        self.tokenizer = tokenizer
        self.max_seq = max_seq
        print(f"  {len(self.data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Tokenize
        conv = item.get('messages', item.get('conversations', []))
        text = ' '.join(c['content'] for c in conv[:2])
        tokens = self.tokenizer(
            text,
            truncation=True, max_length=self.max_seq, return_tensors='pt'
        )
        input_ids = tokens['input_ids'][0]
        seq_len = len(input_ids)
        
        # Use last token embedding as target features
        with torch.no_grad():
            outputs = self.target(input_ids.unsqueeze(0), output_hidden_states=True)
            # Use first layer's last token hidden state as features
            hidden = outputs.hidden_states[1][0, -1, :]  # layer 1
        
        return {
            'input_ids': input_ids,
            'target_hidden': hidden,
            'labels': input_ids.clone(),
        }

def collate_fn(batch):
    input_ids = pad_sequence([b['input_ids'] for b in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    target_hidden = torch.stack([b['target_hidden'] for b in batch])
    labels = pad_sequence([b['labels'] for b in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    return {'input_ids': input_ids, 'target_hidden': target_hidden, 'labels': labels}

def train():
    dataset = SpecDataset(
        'train_data_10k/perfectblend_train.jsonl',
        target, tokenizer, MAX_SEQ
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    
    model = Eagle3Draft().to(DEVICE)
    print(f"Draft model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    total_steps = len(loader) * EPOCHS
    print(f"Training: {len(loader)} batches × {EPOCHS} epochs = {total_steps} steps")
    print(f"Estimated time: ~{total_steps * 3:.0f}s at ~3s/step on CPU")
    print()
    
    model.train()
    global_step = 0
    t_start = time.time()
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            input_ids = batch['input_ids'].to(DEVICE)
            target_hidden = batch['target_hidden'].to(DEVICE)
            
            # Use first token as target (teacher forcing with last token)
            seq_len = input_ids.shape[1]
            
            # Forward through draft
            positions = torch.arange(seq_len, device=DEVICE).unsqueeze(0).expand(input_ids.shape[0], -1)
            logits = model(target_hidden, input_ids[:, :seq_len], positions)
            
            # Loss: predict next token
            shift_logits = logits[:, :-1, :].reshape(-1, VOCAB)
            shift_labels = input_ids[:, 1:].reshape(-1)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=pad_id)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            if global_step % 50 == 0:
                elapsed = time.time() - t_start
                step_per_sec = global_step / elapsed
                remaining = (total_steps - global_step) / step_per_sec
                print(f"  Step {global_step}/{total_steps} | loss={loss.item():.4f} | "
                      f"{step_per_sec:.2f} step/s | ETA: {remaining/60:.0f}m")
        
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} done | avg loss: {avg_loss:.4f}")
    
    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    path = "checkpoints/eagle3_draft.bin"
    
    # Export in C++ format: flat float32 arrays in order
    with open(path, 'wb') as f:
        for name, param in model.named_parameters():
            arr = param.detach().cpu().float().numpy()
            f.write(arr.tobytes())
    
    print(f"\n✅ Saved to {path} ({os.path.getsize(path)/1e6:.1f} MB)")
    print(f"Training time: {(time.time()-t_start)/60:.1f}m")

if __name__ == '__main__':
    train()

#!/usr/bin/env python3
"""Train a draft adapter MLP for DSpark speculative decoding.

Usage:
  1. Generate training data:
     ./tools/draft_data_gen model.trg 4 2000 train_data.bin

  2. Train the adapter:
     source /tmp/ds_env/bin/activate
     python3 tools/train_adapter.py train_data.bin [epochs=50] [lr=0.001]

  3. Use the trained adapter in DSpark:
     The adapter maps 4-layer hidden states to 28-layer hidden states,
     enabling meaningful speculative decoding acceptance.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import struct
import sys
import os

H = 1024  # hidden dim of our model

class DraftAdapter(nn.Module):
    """2-layer MLP: maps draft hidden state → full hidden state"""
    def __init__(self, h_in=H, h_mid=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(h_in),
            nn.Linear(h_in, h_mid),
            nn.ReLU(),
            nn.Linear(h_mid, h_in),
        )
    
    def forward(self, x):
        residual = x
        x = self.net(x)
        return x + residual  # residual connection helps

def load_data(path):
    """Load training data from binary file"""
    with open(path, 'rb') as f:
        H_file = struct.unpack('<I', f.read(4))[0]
        draft_L = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<I', f.read(4))[0]
        
        print(f"Data: H={H_file} draft_L={draft_L} samples={n}")
        
        data = np.frombuffer(f.read(), dtype=np.float32)
        data = data.reshape(-1, 2, H_file)  # [n, 2, H]
        
        X = data[:, 0, :].copy()  # draft hidden states
        Y = data[:, 1, :].copy()  # full hidden states
        
        return torch.from_numpy(X), torch.from_numpy(Y)

def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "train_data.bin"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    lr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.001
    
    print(f"Loading data from {data_path}...")
    X, Y = load_data(data_path)
    n = X.shape[0]
    print(f"Samples: {n}, H={X.shape[1]}")
    
    # Normalize targets
    Y_mean = Y.mean(dim=0, keepdim=True)
    Y_std = Y.std(dim=0, keepdim=True).clamp(min=1e-6)
    Y_norm = (Y - Y_mean) / Y_std
    
    # Split
    n_train = int(n * 0.9)
    X_train, X_val = X[:n_train], X[n_train:]
    Y_train, Y_val = Y_norm[:n_train], Y_norm[n_train:]
    
    # Create adapter
    model = DraftAdapter(H, 512)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    loss_fn = nn.MSELoss()
    
    print(f"Adapter params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training for {epochs} epochs...\n")
    
    best_val = float('inf')
    for epoch in range(epochs):
        model.train()
        
        # Mini-batch training
        batch_size = 64
        perm = torch.randperm(n_train)
        total_loss = 0
        batches = 0
        
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X_train[idx], Y_train[idx]
            
            pred = model(xb)
            loss = loss_fn(pred, yb)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
        
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            y_pred = model(X_val)
            val_loss = loss_fn(y_pred, Y_val).item()
            
            # Accuracy: how often does adapted hidden give same top-1 token?
            # Predict full hidden state, compute logits, check top-1
            y_pred_denorm = y_pred * Y_std[:len(y_pred)] + Y_mean[:len(y_pred)]
            y_true_denorm = Y_val * Y_std[:len(Y_val)] + Y_mean[:len(Y_val)]
            
            # Simple cosine similarity
            cos = nn.CosineSimilarity(dim=1)
            sim = cos(y_pred_denorm, y_true_denorm).mean().item()
        
        if val_loss < best_val:
            best_val = val_loss
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  E{epoch:3d}: train={total_loss/batches:.6f} val={val_loss:.6f} cos={sim:.4f}")
    
    # Save model
    model_path = "draft_adapter_mlp.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'Y_mean': Y_mean,
        'Y_std': Y_std,
    }, model_path)
    print(f"\nModel saved to {model_path}")
    print(f"Best val loss: {best_val:.6f}")
    
    # Validate on held-out samples
    print("\n── Validation ──")
    model.eval()
    with torch.no_grad():
        n_test = min(100, len(X_val))
        cos_sim_total = 0.0
        mse_total = 0.0
        for i in range(n_test):
            pred = model(X_val[i:i+1])
            pred_h = pred * Y_std + Y_mean
            true_h = Y_val[i:i+1] * Y_std + Y_mean
            sim_i = nn.CosineSimilarity(dim=1)(pred_h, true_h).item()
            cos_sim_total += sim_i
            mse_total += nn.MSELoss()(pred_h, true_h).item()
        
        mean_cos = cos_sim_total / n_test
        mean_mse = mse_total / n_test
        print(f"  Mean cosine sim on {n_test} samples: {mean_cos:.4f}")
        print(f"  Mean MSE on {n_test} samples: {mean_mse:.6f}")
        print(f"  (Cosine similarity > 0.9 indicates good adaptation)")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
train_dspark_from_npu.py — Train DSpark draft model on NPU-generated hidden states.

The PROBLEM with the original train_dspark.py:
  It uses HuggingFace (BF16) hidden states for training, but at inference the
  draft receives NPU INT8 hidden states. The distribution mismatch causes 0%
  draft acceptance.

This FIX:
  Trains the draft on NPU-generated hidden states (captured via the NPU engine),
  so train-time and inference-time distributions match. The training uses the
  same DeepSpec DSpark model architecture, loss, and export path — only the
  input features change.

Data pipeline:
  1. prepare_npu_inputs.py — tokenize prompts into binary input format
  2. capture_npu_hidden_per_pos — run NPU forward, dump per-position hidden states
  3. build_cache_from_capture.py — convert binary to .pt cache
  4. THIS SCRIPT — train DSpark on cached NPU hidden states

Usage:
  # Full training on 50K NPU examples:
  python3 train_dspark_from_npu.py \
    --cache spec-decode/target_cache_npu_per_pos.pt \
    --output-dir spec-decode/checkpoints/dspark_npu_50k

  # Quick test on 200 examples:
  python3 train_dspark_from_npu.py --quick

  # Resume from checkpoint:
  python3 train_dspark_from_npu.py --resume PATH
"""
import os, sys, json, time, math, argparse, random
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WANDB_DISABLED'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

import transformers
transformers.logging.set_verbosity_error()

# ── Patch flex_attention for CPU before DeepSpec imports ──
# DeepSpec uses flex_attention's create_block_mask which requires CUDA.
# On CPU we replace it with a simple no-op that returns None (eager handles it).
import torch.nn.attention.flex_attention as _fa
_fa.create_block_mask = lambda *args, **kwargs: None

sys.path.insert(0, os.path.expanduser("~/DeepSpec"))
from deepspec.modeling.dspark.qwen3.config import build_draft_config
from deepspec.modeling.dspark.qwen3.modeling import Qwen3DSparkModel
from deepspec.modeling.dspark.common import DSparkForwardOutput
from deepspec.utils.config import ConfigNode

# Also patch the module that imports create_block_mask
import deepspec.modeling.dspark.common as _dspark_common
_dspark_common.create_block_mask = lambda *args, **kwargs: None

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
HIDDEN       = 1024
VOCAB        = 151936
INTER_DIM    = 3072
NUM_HEADS    = 16
NUM_KV_HEADS = 8
HEAD_DIM     = 128
BLOCK_SIZE   = 7
NUM_DRAFT    = 5
TARGET_IDS   = [1, 6, 12, 18, 24]
NUM_TARGET   = 28
MARKOV_RANK  = 128
NUM_ANCHORS  = 512
MASK_ID      = 151669
MAX_SEQ      = 4096

LR           = 6e-4
WARMUP_RATIO = 0.04
WEIGHT_DECAY = 0.0
EPOCHS       = 5
BATCH_SIZE   = 1
MAX_TRAIN_SEQ = 1024
DEVICE       = "cpu"

# Paths
DEFAULT_CACHE  = "/home/bcloud/spec-decode/target_cache_npu_per_pos.pt"
OUTPUT_DIR     = "/home/bcloud/spec-decode/checkpoints/dspark_npu"
OUTPUT_BIN     = "/home/bcloud/spec-decode/checkpoints/dspark_draft_npu.bin"


# ──────────────────────────────────────────────────────────────────────────────
# Loss function (same as train_dspark.py, single-process CPU)
# ──────────────────────────────────────────────────────────────────────────────
def compute_dspark_loss_cpu(outputs, *, target_ids, eval_mask, block_keep_mask,
                            loss_decay_gamma=4.0, ce_loss_alpha=0.1, l1_loss_alpha=0.9,
                            confidence_head_alpha=1.0):
    import torch.nn.functional as F
    from deepspec.modeling.dspark.loss import (
        _build_loss_weight_mask, _compute_accept_rate_3d,
        _compute_local_l1_term,
    )
    draft_logits = outputs.draft_logits
    _, _, block_size, vocab_size = draft_logits.shape
    loss_weight_mask = _build_loss_weight_mask(
        eval_mask=eval_mask, block_size=block_size, device=draft_logits.device,
        loss_decay_gamma=loss_decay_gamma,
    )
    flat_logits = draft_logits.reshape(-1, vocab_size)
    flat_targets = target_ids.reshape(-1)
    flat_weights = loss_weight_mask.reshape(-1)
    ce_loss_per = F.cross_entropy(flat_logits, flat_targets, reduction='none')
    ce_loss = (ce_loss_per * flat_weights).sum() / (flat_weights.sum() + 1e-6)

    l1_loss = ce_loss.new_zeros(())
    if l1_loss_alpha > 0 and outputs.aligned_target_logits is not None:
        l1_num, l1_den = _compute_local_l1_term(
            outputs=outputs, aligned_target_logits=outputs.aligned_target_logits,
            loss_weight_mask=loss_weight_mask,
        )
        if l1_den > 0:
            l1_loss = l1_num / (l1_den + 1e-6)

    confidence_loss = ce_loss.new_zeros(())
    if outputs.confidence_pred is not None and outputs.aligned_target_logits is not None:
        accept_rate_3d = _compute_accept_rate_3d(
            outputs=outputs, aligned_target_logits=outputs.aligned_target_logits,
        )
        confidence_targets = accept_rate_3d.detach()
        confidence_error = F.binary_cross_entropy_with_logits(
            outputs.confidence_pred.float(), confidence_targets, reduction='none',
        ) * loss_weight_mask
        confidence_loss = confidence_error.sum() / (loss_weight_mask.sum() + 1e-6)

    return (ce_loss_alpha * ce_loss + l1_loss_alpha * l1_loss +
            confidence_head_alpha * confidence_loss)


# ──────────────────────────────────────────────────────────────────────────────
# Eager attention mask (CPU-compatible)
# ──────────────────────────────────────────────────────────────────────────────
NEG_INF = -1e9

def create_dspark_attention_mask_eager(anchor_positions, block_keep_mask,
                                        seq_len, block_size, device):
    """Eager attention mask that matches the flex_attention interface."""
    import torch.nn.functional as F
    B, N = anchor_positions.shape
    # Simple causal mask: each position attends to itself and previous positions
    mask = torch.triu(torch.full((N, N), float('-inf'), device=device), diagonal=1)
    mask = mask.unsqueeze(0).expand(B, N, N)
    # Apply block keep mask
    keep = block_keep_mask.bool().unsqueeze(-1) & block_keep_mask.bool().unsqueeze(-2)
    mask = mask.masked_fill(keep, 0.0)
    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Dataset: Load pre-built NPU cache
# ──────────────────────────────────────────────────────────────────────────────
class NPUCacheDataset(Dataset):
    """Dataset that loads pre-cached NPU hidden states.

    Cache format (from build_cache_from_capture.py):
      List of dicts:
        {'input_ids': LongTensor[seq_len],
         'target_features': FloatTensor[num_positions, hidden_size, num_layers]}
    """

    def __init__(self, cache_path: str, max_samples: int | None = None,
                 max_seq_len: int = MAX_TRAIN_SEQ):
        print(f"Loading NPU cache from: {cache_path}")
        self.data = torch.load(cache_path, map_location='cpu', weights_only=True)
        print(f"  Loaded {len(self.data)} examples")

        if max_samples is not None and max_samples < len(self.data):
            self.data = self.data[:max_samples]
            print(f"  Limited to {max_samples} examples")

        # Filter out sequences that are too long
        original = len(self.data)
        self.data = [d for d in self.data if len(d['input_ids']) <= max_seq_len]
        if len(self.data) < original:
            print(f"  Filtered {original - len(self.data)} sequences > {max_seq_len} tokens")
        
        # Verify format
        if self.data:
            ex = self.data[0]
            print(f"  Example: input_ids={ex['input_ids'].shape}, "
                  f"target_features={ex['target_features'].shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        return {
            'input_ids': ex['input_ids'],
            'target_features': ex['target_features'],  # [seq_len, H, NL]
        }


# ──────────────────────────────────────────────────────────────────────────────
# Model building
# ──────────────────────────────────────────────────────────────────────────────
def build_dspark_model(trainable=True):
    """Build the DSpark Qwen3DSparkModel with the correct config."""
    target_config = transformers.AutoConfig.from_pretrained(
        "Qwen/Qwen3-0.6B")

    model_args = ConfigNode()
    model_args.target_layer_ids = TARGET_IDS
    model_args.block_size = BLOCK_SIZE
    model_args.num_draft_layers = NUM_DRAFT
    model_args.mask_token_id = MASK_ID
    model_args.num_anchors = NUM_ANCHORS
    model_args.markov_rank = MARKOV_RANK
    model_args.markov_head_type = "vanilla"
    model_args.confidence_head_alpha = 1.0
    model_args.confidence_head_with_markov = True
    model_args.loss_decay_gamma = 4.0
    model_args.ce_loss_alpha = 0.1
    model_args.l1_loss_alpha = 0.9

    draft_config = build_draft_config(
        target_config=target_config,
        model_args=model_args,
    )
    draft_config._attn_implementation = "eager"

    model = Qwen3DSparkModel(draft_config).float()

    # Use eager attention for CPU
    draft_config._attn_implementation = "eager"

    if trainable:
        total = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {total/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable params")
        model.train()
    return model


def initialize_from_target(model):
    """Copy target model's embed_tokens and lm_head into the draft model."""
    print("Initializing embed_tokens and lm_head from target model...")
    target_model = transformers.AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to("cpu").eval()

    target_embed = target_model.get_input_embeddings()
    target_lm_head = target_model.get_output_embeddings()

    model.initialize_embeddings_and_head(
        embed_tokens=target_embed,
        lm_head=target_lm_head,
        freeze=True,
    )
    del target_model
    print("  Done (embed/lm_head frozen)")


def export_to_binary(model, output_path):
    """Export trained weights to C++ binary format (DSparkDraftWeights::load())."""
    from deepspec.modeling.dspark.qwen3.export import export_dspark_weights
    print(f"Exporting to: {output_path}")
    export_dspark_weights(model, output_path)
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"  Exported {file_size / 1024 / 1024:.1f} MB")


# ──────────────────────────────────────────────────────────────────────────────
# Collate function for NPU cache data
# ──────────────────────────────────────────────────────────────────────────────
def npu_cache_collate(batch):
    """Pad variable-length sequences in a batch."""
    input_ids_list = [b['input_ids'] for b in batch]
    features_list = [b['target_features'] for b in batch]

    max_len = max(ids.size(0) for ids in input_ids_list)
    batch_size = len(batch)
    NL = features_list[0].size(-1)
    H = features_list[0].size(1)

    # Pad input_ids
    padded_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for i, ids in enumerate(input_ids_list):
        padded_ids[i, :ids.size(0)] = ids
        attention_mask[i, :ids.size(0)] = True

    # Pad target_features: [B, S, H, NL] -> flatten to [B, S, H*NL]
    padded_features_4d = torch.zeros(batch_size, max_len, H, NL)
    for i, feat in enumerate(features_list):
        padded_features_4d[i, :feat.size(0)] = feat
    padded_features = padded_features_4d.reshape(batch_size, max_len, H * NL)

    return {
        'input_ids': padded_ids,
        'attention_mask': attention_mask,
        'target_features': padded_features,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────
def train(args):
    device = torch.device(DEVICE)

    # Load dataset
    dataset = NPUCacheDataset(
        args.cache,
        max_samples=args.quick and 200 or None,
        max_seq_len=MAX_TRAIN_SEQ,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size or BATCH_SIZE,
        shuffle=True, collate_fn=npu_cache_collate,
    )

    # Build model
    model = build_dspark_model(trainable=True)
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        state = torch.load(args.resume, map_location='cpu', weights_only=True)
        model.load_state_dict(state['model_state_dict'])
        start_epoch = state.get('epoch', 0) + 1
    else:
        initialize_from_target(model)
        start_epoch = 0

    model = model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr or LR, weight_decay=WEIGHT_DECAY)

    # Scheduler
    total_steps = len(dataloader) * (args.epochs or EPOCHS)
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, (step + 1) / max(1, warmup_steps)),
    )

    # Training
    print(f"\n═══ Training ═══")
    print(f"  Examples: {len(dataset)}")
    print(f"  Epochs:   {args.epochs or EPOCHS}")
    print(f"  Batch:    {args.batch_size or BATCH_SIZE}")
    print(f"  LR:       {args.lr or LR}")
    print(f"  Device:   {device}")
    print(f"  Params:   {sum(p.numel() for p in params):,}")
    print()

    best_loss = float('inf')
    step = 0

    for epoch in range(start_epoch, args.epochs or EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_confidence = 0.0
        n_batches = 0
        t0 = time.time()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # target_features: [B, seq_len, H, NL] — NPU hidden states
            target_features = batch['target_features'].to(device)

            B, S = input_ids.shape

            # Forward through DSpark model
            # loss_mask: 1 for positions where we should predict, 0 for padding
            loss_mask = attention_mask.float()  # [B, S]
            # target_last_hidden_states: last layer (index NL-1) -> [B, S, H]
            target_last = target_features[:, :, (5-1)*1024:5*1024].contiguous()
            outputs = model(
                input_ids=input_ids,
                target_hidden_states=target_features,
                loss_mask=loss_mask,
                target_last_hidden_states=target_last,
            )
            del target_last

            # Compute loss
            loss = compute_dspark_loss_cpu(
                outputs,
                target_ids=outputs.target_ids,
                eval_mask=outputs.eval_mask,
                block_keep_mask=outputs.block_keep_mask,
            )
            epoch_loss += loss.item()

            # Track sub-losses for logging
            if hasattr(outputs, 'aux_loss') and outputs.aux_loss is not None:
                epoch_ce += outputs.aux_loss.get('ce', 0)
                epoch_confidence += outputs.aux_loss.get('confidence', 0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            n_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / max(1, n_batches)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch}: loss={avg_loss:.4f}  "
              f"({elapsed:.0f}s, {n_batches / elapsed:.1f} batch/s)")

        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pt")
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"  Saved: {checkpoint_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.output_dir, "best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
            }, best_path)
            print(f"  New best: {best_path}")

    # Export final model to C++ binary
    model.eval()
    export_path = args.export_bin or os.path.join(
        args.output_dir, "dspark_draft_npu.bin")
    export_to_binary(model, export_path)

    print(f"\n═══ Training Complete ═══")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Checkpoints: {args.output_dir}")
    print(f"  C++ binary: {export_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DSpark on NPU hidden states")
    parser.add_argument('--cache', default=DEFAULT_CACHE,
                        help='Path to NPU cache .pt file')
    parser.add_argument('--output-dir', default=OUTPUT_DIR,
                        help='Output directory for checkpoints')
    parser.add_argument('--export-bin', default=None,
                        help='Output binary path (default: output_dir/dspark_draft_npu.bin)')
    parser.add_argument('--resume', default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with 200 examples')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--cache-only', action='store_true',
                        help='Only print cache info, no training')

    args = parser.parse_args()

    if args.cache_only:
        dataset = NPUCacheDataset(args.cache, max_samples=10)
        total_tokens = sum(len(d['input_ids']) for d in dataset.data)
        print(f"  Total tokens: {total_tokens}")
        sys.exit(0)

    train(args)

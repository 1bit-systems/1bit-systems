#!/usr/bin/env python3
"""DSpark 0.6B CPU Training — 5-layer draft with Markov + confidence heads.

Trains the DSpark draft model (Qwen3DSparkModel) on CPU using the target model
(Qwen3-0.6B) for on-the-fly hidden state extraction. Exports trained weights
to the C++ binary format that DSparkDraftWeights::load() expects.

Usage:
    python3 train_dspark.py                          # Full training (10k samples)
    python3 train_dspark.py --quick                   # Quick test (200 samples)
    python3 train_dspark.py --resume PATH             # Resume from checkpoint
    python3 train_dspark.py --cache-only              # Only build target cache
"""
import os, sys, json, time, math, re, argparse
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

# ── Silence HF warnings before importing anything that uses them ──
import transformers
transformers.logging.set_verbosity_error()

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
HIDDEN       = 1024       # hidden_size
VOCAB        = 151936     # vocab_size
INTER_DIM    = 3072       # intermediate_size (FFN)
NUM_HEADS    = 16         # num_attention_heads
NUM_KV_HEADS = 8          # num_key_value_heads
HEAD_DIM     = 128        # head_dim
BLOCK_SIZE   = 7          # draft tokens per forward
NUM_DRAFT    = 5          # number of draft layers
TARGET_IDS   = [1, 6, 12, 18, 24]  # target layer ids (5 layers)
NUM_TARGET   = 28         # total target layers
MARKOV_RANK  = 128        # Markov head rank
NUM_ANCHORS  = 512        # max anchors
MASK_ID      = 151669     # mask_token_id
MAX_SEQ      = 4096       # max position embeddings

LR           = 6e-4
WARMUP_RATIO = 0.04
WEIGHT_DECAY = 0.0
EPOCHS       = 3
BATCH_SIZE   = 1          # CPU — keep small
MAX_TRAIN_SEQ = 1024      # truncate long sequences for speed
DEVICE       = "cpu"
TARGET_MODEL = "Qwen/Qwen3-0.6B"

# Paths
DATA_PATH      = "/home/bcloud/spec-decode/train_data_10k/perfectblend_train.jsonl"
CACHE_PATH     = "/home/bcloud/spec-decode/target_cache_dspark.pt"
OUTPUT_DIR     = "/home/bcloud/spec-decode/checkpoints/dspark_qwen3_0.6b"
OUTPUT_BIN     = "/home/bcloud/spec-decode/checkpoints/dspark_draft.bin"


# ──────────────────────────────────────────────────────────────────────────────
# DeepSpec imports (must come after env vars)
# ──────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.expanduser("~/DeepSpec"))

from deepspec.utils.config import ConfigNode
from deepspec.modeling.dspark.qwen3.config import build_draft_config
from deepspec.modeling.dspark.qwen3.modeling import Qwen3DSparkModel
from deepspec.modeling.dspark.common import DSparkForwardOutput


# ──────────────────────────────────────────────────────────────────────────────
# Simplified DSpark loss (no distributed ops — for CPU single-process training)
# ──────────────────────────────────────────────────────────────────────────────
def compute_dspark_loss_cpu(
    *,
    outputs: DSparkForwardOutput,
    loss_decay_gamma: float = 4.0,
    ce_loss_alpha: float = 0.1,
    l1_loss_alpha: float = 0.9,
    confidence_head_alpha: float = 1.0,
) -> torch.Tensor:
    """DSpark loss without distributed all-reduce. Single-process CPU version."""
    import torch.nn.functional as F
    from deepspec.modeling.dspark.loss import (
        _build_loss_weight_mask,
        _compute_accept_rate_3d,
        _compute_local_l1_term,
        _compute_local_probabilistic_stats,
    )
    from deepspec.utils.metrics import add_metric

    draft_logits = outputs.draft_logits
    target_ids = outputs.target_ids
    eval_mask = outputs.eval_mask
    block_keep_mask = outputs.block_keep_mask
    _, _, block_size, vocab_size = draft_logits.shape
    device = draft_logits.device

    loss_weight_mask = _build_loss_weight_mask(
        eval_mask=eval_mask,
        block_size=block_size,
        device=device,
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
            outputs=outputs,
            aligned_target_logits=outputs.aligned_target_logits,
            loss_weight_mask=loss_weight_mask,
        )
        if l1_den > 0:
            l1_loss = l1_num / (l1_den + 1e-6)

    confidence_loss = ce_loss.new_zeros(())
    if outputs.confidence_pred is not None and outputs.aligned_target_logits is not None:
        accept_rate_3d = _compute_accept_rate_3d(
            outputs=outputs,
            aligned_target_logits=outputs.aligned_target_logits,
        )
        confidence_targets = accept_rate_3d.detach()
        confidence_error = F.binary_cross_entropy_with_logits(
            outputs.confidence_pred.float(),
            confidence_targets,
            reduction='none',
        ) * loss_weight_mask
        confidence_loss = confidence_error.sum() / (loss_weight_mask.sum() + 1e-6)

    return (
        ce_loss_alpha * ce_loss
        + l1_loss_alpha * l1_loss
        + confidence_head_alpha * confidence_loss
    )


# ──────────────────────────────────────────────────────────────────────────────
# Eager attention mask (replaces flex_attention BlockMask for CPU)
# ──────────────────────────────────────────────────────────────────────────────
NEG_INF = -1e9  # not -inf to avoid softmax NaN on fully-masked rows

def create_dspark_attention_mask_eager(
    *,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    seq_len: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Vectorized eager attention mask (no Python loops).

    Returns: [B, 1, Q_len, KV_len] where 0.0 = allowed, NEG_INF = masked.
    """
    B, num_blocks = anchor_positions.shape
    Q_len = num_blocks * block_size
    KV_len = seq_len + Q_len

    mask = torch.full((B, 1, Q_len, KV_len), NEG_INF, dtype=torch.float32, device=device)

    # Reshape to [B, 1, num_blocks, block_size, KV_len] for block-level ops
    mask_blocks = mask.view(B, 1, num_blocks, block_size, KV_len)

    # Expand keep_mask to [B, 1, num_blocks, 1, 1]
    keep = block_keep_mask[:, None, :, None, None]  # [B, 1, num_blocks, 1, 1]

    # 1) Context positions: mask[b, :, qb, :, :anchor_pos[b,qb]] = 0.0
    #    Create range indices [0, 1, ..., seq_len-1] and compare to anchor_pos
    ctx_range = torch.arange(seq_len, device=device)  # [seq_len]
    # anchor_positions[b, qb] → [B, 1, num_blocks, 1, 1]
    anchor = anchor_positions[:, None, :, None, None]  # [B, 1, num_blocks, 1, 1]
    ctx_mask = (ctx_range[None, None, None, None, :] < anchor)  # [B, 1, num_blocks, 1, seq_len]
    mask_blocks[..., :seq_len] = torch.where(
        ctx_mask & keep,
        torch.tensor(0.0, device=device),
        mask_blocks[..., :seq_len],
    )

    # 2) Draft positions: each block can attend to its own draft KV entries
    #    mask[b, :, qb, :, seq_len + qb*block_size : seq_len + (qb+1)*block_size] = 0.0
    block_idx = torch.arange(num_blocks, device=device)  # [num_blocks]
    kv_start = seq_len + block_idx * block_size  # [num_blocks]
    kv_end = kv_start + block_size  # [num_blocks]

    for qb in range(num_blocks):
        if not bool(block_keep_mask[0, qb]):  # B=1 in practice
            continue
        s, e = int(kv_start[qb]), int(kv_end[qb])
        mask_blocks[0, 0, qb, :, s:e] = 0.0

    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Helper: extract target hidden states via forward hooks
# ──────────────────────────────────────────────────────────────────────────────
class TargetFeatureExtractor:
    """Runs Qwen3-0.6B on CPU and captures per-position hidden states
    from the specified target layers."""

    def __init__(self, model_name: str, target_layer_ids: list[int], device: str = "cpu"):
        self.device = device
        self.target_layer_ids = target_layer_ids

        print(f"Loading target model {model_name} on {device}...")
        t0 = time.time()
        self.model = transformers.AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to(device).eval()
        print(f"  Loaded in {time.time()-t0:.1f}s ({sum(p.numel() for p in self.model.parameters())/1e6:.0f}M params)")

        self.hidden_size = self.model.config.hidden_size
        self.backbone = self._get_backbone()
        self.layer_modules = self.backbone.layers

    def _get_backbone(self):
        """Get the transformer backbone (model.model for Qwen3)."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model
        return self.model

    @torch.no_grad()
    def extract(self, input_ids: torch.Tensor, attention_mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run target model and extract hidden states from target layers.

        Returns:
            target_hidden_states: [batch, seq_len, num_target_layers * hidden_size]
            target_last_hidden_states: [batch, seq_len, hidden_size]
        """
        captured = {}
        handles = []

        def make_hook(layer_id: int):
            def hook(_mod, _inp, out):
                if isinstance(out, torch.Tensor):
                    captured[layer_id] = out.detach().to(dtype=torch.bfloat16)
                elif isinstance(out, (tuple, list)) and len(out) > 0:
                    captured[layer_id] = out[0].detach().to(dtype=torch.bfloat16)
            return hook

        for lid in self.target_layer_ids:
            handles.append(self.layer_modules[lid].register_forward_hook(make_hook(lid)))

        try:
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
            )
            target_last_hidden = output.last_hidden_state.detach().to(dtype=torch.bfloat16)

            # Concatenate features from all target layers: [B, S, NTL * H]
            per_layer = []
            for lid in self.target_layer_ids:
                h = captured[lid]  # [B, S, H]
                per_layer.append(h)
            target_hidden = torch.cat(per_layer, dim=-1)  # [B, S, NTL*H]
        finally:
            for h in handles:
                h.remove()

        return target_hidden, target_last_hidden


# ──────────────────────────────────────────────────────────────────────────────
# Dataset: stream JSONL and extract features on-the-fly
# ──────────────────────────────────────────────────────────────────────────────
def build_conversation_collator(tokenizer, max_length=MAX_TRAIN_SEQ):
    """Create a collator that processes conversation records into model inputs."""
    from transformers import AutoTokenizer
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def collate_fn(batch):
        """batch: list of dict with 'id', 'conversations'"""
        all_input_ids = []
        all_loss_masks = []

        for record in batch:
            conversations = record["conversations"]
            # Build chat template for loss masking: only assistant tokens get loss
            # We tokenize each turn separately
            input_ids = []
            loss_mask = []
            for turn in conversations:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role == "user":
                    # Format as chat template
                    tokens = tokenizer.encode(
                        f"<|im_start|>user\n{content}<|im_end|>\n",
                        add_special_tokens=False,
                    )
                    input_ids.extend(tokens)
                    loss_mask.extend([0] * len(tokens))
                elif role == "assistant":
                    tokens = tokenizer.encode(
                        f"<|im_start|>assistant\n{content}<|im_end|>\n",
                        add_special_tokens=False,
                    )
                    input_ids.extend(tokens)
                    loss_mask.extend([1] * len(tokens))

            # Truncate to max_length
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                loss_mask = loss_mask[:max_length]

            if len(input_ids) < 10:  # Skip very short sequences
                continue

            all_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            all_loss_masks.append(torch.tensor(loss_mask, dtype=torch.uint8))

        if not all_input_ids:
            return None

        return {
            "input_ids": all_input_ids,
            "loss_mask": all_loss_masks,
        }

    return collate_fn


def load_jsonl_dataset(path: str, max_samples: int | None = None) -> list[dict]:
    """Load JSONL file, return list of dicts."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if max_samples is not None and len(data) >= max_samples:
                break
            data.append(json.loads(line))
    print(f"Loaded {len(data)} samples from {path}")
    return data


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────
def build_dspark_model(trainable=True):
    """Build the DSpark Qwen3DSparkModel with the correct config."""
    target_config = transformers.AutoConfig.from_pretrained(TARGET_MODEL)

    # Build ConfigNode identical to our config file
    # (Can't load the config file directly because it imports tensorboard.)
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
    # Use eager attention on CPU
    draft_config._attn_implementation = "eager"

    model = Qwen3DSparkModel(draft_config).to(DEVICE).float()

    # Patch: replace flex_attention mask with eager tensor mask in the module
    import deepspec.modeling.dspark.qwen3.modeling as dspark_mod
    dspark_mod.create_dspark_attention_mask = create_dspark_attention_mask_eager

    if trainable:
        total = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {total/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable params")
    return model


def initialize_from_target(model: Qwen3DSparkModel):
    """Copy target model's embed_tokens and lm_head into the draft model."""
    print("Initializing embed_tokens and lm_head from target model...")
    target_model = transformers.AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL,
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


def pad_collate(batch):
    """Collate function for cached data (with padding)."""
    if batch is None:
        return None
    if len(batch) == 0:
        return None

    input_ids = [b["input_ids"] for b in batch]
    loss_masks = [b["loss_mask"] for b in batch]
    target_hidden = [b["target_hidden_states"] for b in batch]
    target_last = [b["target_last_hidden_states"] for b in batch]

    max_len = max(ids.shape[0] for ids in input_ids)

    padded_ids = torch.full((len(batch), max_len), fill_value=0, dtype=torch.long)
    padded_mask = torch.zeros((len(batch), max_len), dtype=torch.uint8)
    padded_hidden = torch.zeros((len(batch), max_len, target_hidden[0].shape[-1]),
                                 dtype=torch.bfloat16)
    padded_last = torch.zeros((len(batch), max_len, target_last[0].shape[-1]),
                               dtype=torch.bfloat16)

    for i, (ids, mask) in enumerate(zip(input_ids, loss_masks)):
        sl = ids.shape[0]
        padded_ids[i, :sl] = ids
        padded_mask[i, :sl] = mask
        padded_hidden[i, :sl] = target_hidden[i]
        padded_last[i, :sl] = target_last[i]

    return {
        "input_ids": padded_ids,
        "loss_mask": padded_mask,
        "target_hidden_states": padded_hidden,
        "target_last_hidden_states": padded_last,
    }


def prepare_target_cache(extractor: TargetFeatureExtractor,
                          dataset: list[dict],
                          cache_path: str,
                          tokenizer,
                          max_samples: int | None = None):
    """Build target cache (torch .pt file) from JSONL data and target model."""
    print(f"\n{'='*60}")
    print(f"Building target cache → {cache_path}")
    print(f"{'='*60}")

    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset[:max_samples]

    collator = build_conversation_collator(tokenizer)
    cache = []

    # Process in mini-batches (batch_size=1 for CPU)
    t0 = time.time()
    for i, record in enumerate(tqdm(dataset, desc="Extracting features")):
        batch = collator([record])
        if batch is None:
            continue

        input_ids = batch["input_ids"][0].unsqueeze(0)  # [1, S]
        seq_len = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        target_hidden, target_last = extractor.extract(input_ids, attention_mask)

        cache.append({
            "input_ids": input_ids[0].to(torch.long),  # [S]
            "loss_mask": batch["loss_mask"][0].to(torch.uint8),  # [S]
            "target_hidden_states": target_hidden[0].to(torch.bfloat16),  # [S, NTL*H]
            "target_last_hidden_states": target_last[0].to(torch.bfloat16),  # [S, H]
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(dataset) - (i + 1)) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(dataset)}] {rate:.1f} samples/s, ETA {eta:.0f}s")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(cache, cache_path)
    print(f"Saved {len(cache)} cached samples to {cache_path}")
    return cache


def train_from_cache(cache, args):
    """Train DSpark model from pre-computed target cache."""
    print(f"\n{'='*60}")
    print(f"Training DSpark 0.6B from cache ({len(cache)} samples)")
    print(f"{'='*60}")

    model = build_dspark_model(trainable=True)
    initialize_from_target(model)

    # Plan to unfreeze: FC projection, all 5 layers, Markov head, confidence head
    # Embed/lm_head remain frozen
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    class CachedDataset(Dataset):
        def __init__(self, cache):
            self.data = cache

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    train_dataset = CachedDataset(cache)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=pad_collate,
        num_workers=0,
    )

    total_steps = len(train_loader) * args.epochs

    print(f"Training: {len(train_loader)} batches × {args.epochs} epochs = {total_steps} steps")
    print(f"Device: {DEVICE}")
    print()

    model.train()
    global_step = 0
    t_start = time.time()
    best_loss = float("inf")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_ce = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            if batch is None:
                continue

            input_ids = batch["input_ids"].to(DEVICE)
            loss_mask = batch["loss_mask"].to(DEVICE)
            target_hidden = batch["target_hidden_states"].to(DEVICE).float()
            target_last = batch["target_last_hidden_states"].to(DEVICE).float()

            outputs = model(
                input_ids=input_ids,
                target_hidden_states=target_hidden,
                loss_mask=loss_mask,
                target_last_hidden_states=target_last,
            )

            loss = compute_dspark_loss_cpu(
                outputs=outputs,
                loss_decay_gamma=4.0,
                ce_loss_alpha=0.1,
                l1_loss_alpha=0.9,
                confidence_head_alpha=1.0,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                1.0,
            )
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        elapsed = (time.time() - t_start) / 60
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f} ({elapsed:.1f}m)")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, os.path.join(OUTPUT_DIR, "best_model.pt"))
            print(f"  ✓ New best model saved")

    total_time = (time.time() - t_start) / 60
    print(f"\n{'='*60}")
    print(f"Training complete: {total_time:.1f}m, best loss: {best_loss:.4f}")
    print(f"{'='*60}")

    return model


def train_on_the_fly(args):
    """Train DSpark model with on-the-fly feature extraction (no pre-caching)."""
    print(f"\n{'='*60}")
    print(f"Training DSpark 0.6B on-the-fly")
    print(f"{'='*60}")

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(TARGET_MODEL)

    # Build model
    model = build_dspark_model(trainable=True)
    initialize_from_target(model)

    # Load feature extractor (target model for getting hidden states)
    extractor = TargetFeatureExtractor(TARGET_MODEL, TARGET_IDS, DEVICE)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Load dataset
    dataset = load_jsonl_dataset(DATA_PATH, args.max_samples)
    collator = build_conversation_collator(tokenizer)

    total_steps = len(dataset) * args.epochs
    print(f"Training: {len(dataset)} samples × {args.epochs} epochs = {total_steps} steps")
    print()

    model.train()
    global_step = 0
    t_start = time.time()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataset, desc=f"Epoch {epoch+1}/{args.epochs}")

        for record in pbar:
            batch = collator([record])
            if batch is None:
                continue

            input_ids = batch["input_ids"][0].unsqueeze(0).to(DEVICE)  # [1, S]
            loss_mask = batch["loss_mask"][0].unsqueeze(0).to(DEVICE)  # [1, S]
            attention_mask = torch.ones_like(input_ids)

            # Extract target features
            target_hidden, target_last = extractor.extract(input_ids, attention_mask)

            # Forward through DSpark model
            outputs = model(
                input_ids=input_ids,
                target_hidden_states=target_hidden.float(),
                loss_mask=loss_mask,
                target_last_hidden_states=target_last.float(),
            )

            loss = compute_dspark_loss_cpu(
                outputs=outputs,
                loss_decay_gamma=4.0,
                ce_loss_alpha=0.1,
                l1_loss_alpha=0.9,
                confidence_head_alpha=1.0,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                1.0,
            )
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataset)
        elapsed = (time.time() - t_start) / 60
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f} ({elapsed:.1f}m)")

    total_time = (time.time() - t_start) / 60
    print(f"\nTraining complete: {total_time:.1f}m")
    return model


def export_to_binary(model: Qwen3DSparkModel, output_path: str, markov_rank: int = 128):
    """Export DSpark weights to flat C++ binary format.

    Binary layout matches DSparkDraftWeights::load() in dspark_draft.h.
    """
    print(f"\n{'='*60}")
    print(f"Exporting weights to C++ binary → {output_path}")
    print(f"{'='*60}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def write_tensor(f_out, t, name):
        """Write weight tensor as float32 to binary file."""
        arr = t.detach().cpu().float().contiguous().numpy().astype(np.float32)
        f_out.write(arr.tobytes())
        sz_mb = arr.nbytes / 1e6
        print(f"  {name:40s} {str(list(t.shape)):30s} {sz_mb:7.2f} MB")
        return arr.nbytes

    model = model.cpu().float()
    NL = NUM_DRAFT  # num_draft_layers
    H = HIDDEN
    V = VOCAB
    NH = NUM_HEADS
    NKV = NUM_KV_HEADS
    D = HEAD_DIM
    IM = INTER_DIM
    R = markov_rank
    NTL = len(TARGET_IDS)

    total_bytes = 0

    with open(output_path, "wb") as f:
        # ── 1. Shared tensors (first 3: embed_tokens, fc, hidden_norm) ──
        total_bytes += write_tensor(f, model.embed_tokens.weight, "embed_tokens")
        total_bytes += write_tensor(f, model.fc.weight, "fc")
        total_bytes += write_tensor(f, model.hidden_norm.weight, "hidden_norm")

        # ── 2. Per-layer fields (all layers of each field contiguously) ──
        for name in [
            "input_layernorm",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "self_attn.q_norm",
            "self_attn.k_norm",
            "post_attention_layernorm",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]:
            field_name = name.replace("self_attn.", "").replace("mlp.", "")
            for l in range(NL):
                # Get the tensor from the model
                if name == "input_layernorm":
                    t = model.layers[l].input_layernorm.weight
                elif name == "self_attn.q_proj":
                    t = model.layers[l].self_attn.q_proj.weight
                elif name == "self_attn.k_proj":
                    t = model.layers[l].self_attn.k_proj.weight
                elif name == "self_attn.v_proj":
                    t = model.layers[l].self_attn.v_proj.weight
                elif name == "self_attn.o_proj":
                    t = model.layers[l].self_attn.o_proj.weight
                elif name == "self_attn.q_norm":
                    t = model.layers[l].self_attn.q_norm.weight
                elif name == "self_attn.k_norm":
                    t = model.layers[l].self_attn.k_norm.weight
                elif name == "post_attention_layernorm":
                    t = model.layers[l].post_attention_layernorm.weight
                elif name == "mlp.gate_proj":
                    t = model.layers[l].mlp.gate_proj.weight
                elif name == "mlp.up_proj":
                    t = model.layers[l].mlp.up_proj.weight
                elif name == "mlp.down_proj":
                    t = model.layers[l].mlp.down_proj.weight
                else:
                    continue

                label = field_name if l == 0 else ""
                total_bytes += write_tensor(f, t, f"  {label}")

        # ── 3. Final norm + output head ──
        total_bytes += write_tensor(f, model.norm.weight, "norm")
        total_bytes += write_tensor(f, model.lm_head.weight, "lm_head")

        # ── 4. Markov head ──
        total_bytes += write_tensor(f, model.markov_head.markov_w1.weight, "markov_w1")
        # W2 is nn.Linear: weight shape [V, R]
        total_bytes += write_tensor(f, model.markov_head.markov_w2.weight, "markov_w2")

        # ── 5. Confidence head ──
        # weight: [1, H+R] → flatten to 1D
        cw = model.confidence_head.proj.weight  # [1, H+R]
        total_bytes += write_tensor(f, cw.reshape(-1), "confidence_weight")

        # bias: [1]
        cb = model.confidence_head.proj.bias  # [1]
        total_bytes += write_tensor(f, cb.reshape(-1), "confidence_bias")

    print(f"\n✅ Exported {output_path} ({total_bytes/1e6:.1f} MB, {total_bytes/1e9:.3f} GB)")


def save_checkpoint(model, path: str):
    """Save model state dict as checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "draft_config": {
            "hidden_size": HIDDEN,
            "num_heads": NUM_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "vocab_size": VOCAB,
            "inter_dim": INTER_DIM,
            "block_size": BLOCK_SIZE,
            "num_draft_layers": NUM_DRAFT,
            "target_layer_ids": TARGET_IDS,
            "markov_rank": MARKOV_RANK,
            "num_anchors": NUM_ANCHORS,
        },
    }, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, path: str):
    """Load model state dict from checkpoint."""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from {path}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DSpark 0.6B CPU Training")
    parser.add_argument("--quick", action="store_true", help="Quick test (200 samples)")
    parser.add_argument("--cache-only", action="store_true", help="Only build cache, don't train")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--export-only", type=str, default=None, help="Export .pt checkpoint to binary")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--max-samples", type=int, default=None, help="Max training samples")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="Weight decay")
    parser.add_argument("--on-the-fly", action="store_true",
                        help="Train with on-the-fly extraction (no cache)")
    args = parser.parse_args()

    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║      DSpark 0.6B Training — CPU                             ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    print(f"Target: Qwen3-0.6B (H={HIDDEN}, V={VOCAB}, layers={NUM_DRAFT})")
    print(f"Markov rank={MARKOV_RANK}, block_size={BLOCK_SIZE}, anchors={NUM_ANCHORS}")
    print(f"LR={args.lr}, epochs={args.epochs}, batch={args.batch_size}")
    print()

    # ── Export-only mode ──
    if args.export_only:
        print("Export mode: loading checkpoint and exporting to binary...")
        model = build_dspark_model(trainable=False)
        model = load_checkpoint(model, args.export_only)
        export_to_binary(model, OUTPUT_BIN, MARKOV_RANK)
        return

    # ── Determine sample count ──
    n_samples = args.max_samples or (200 if args.quick else 10_000_000)
    train_data = DATA_PATH

    # ── On-the-fly training (no pre-caching) ──
    if args.on_the_fly:
        model = train_on_the_fly(args)
    else:
        # Build or load cache, then train
        if os.path.exists(CACHE_PATH) and not args.quick:
            print(f"Loading pre-built cache from {CACHE_PATH}")
            cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(TARGET_MODEL)
            dataset = load_jsonl_dataset(train_data, n_samples)
            extractor = TargetFeatureExtractor(TARGET_MODEL, TARGET_IDS, DEVICE)
            cache = prepare_target_cache(extractor, dataset, CACHE_PATH, tokenizer, n_samples)

        if args.cache_only:
            print("Cache built. Exiting (--cache-only).")
            return

        model = train_from_cache(cache, args)

    # ── Export ──
    export_to_binary(model, OUTPUT_BIN, MARKOV_RANK)

    # ── Save checkpoint ──
    save_checkpoint(model, os.path.join(OUTPUT_DIR, "final_model.pt"))
    print(f"\n✅ Done! Binary at: {OUTPUT_BIN}")
    print(f"   Checkpoint at: {os.path.join(OUTPUT_DIR, 'final_model.pt')}")


if __name__ == "__main__":
    main()

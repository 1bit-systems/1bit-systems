#! /usr/bin/env python3
"""Training harness for the Text→Codec adapter model.

Trains a ``TextToCodecModel`` to map byte-level text tokens to RVQ-VAE
codec indices, using a frozen codec encoder to produce ground-truth
codec tokens from audio waveforms.

Usage
-----
.. code-block:: bash

    # Basic training
    python -m zaya_audio.train_adapter \\
        --codec_checkpoint models/codec_final.pt \\
        --audio_dir data/voice_samples/ \\
        --text_file data/transcripts.jsonl \\
        --output_dir models/adapter/ \\
        --epochs 50 --batch_size 16 --lr 1e-3

    # Resume from checkpoint
    python -m zaya_audio.train_adapter \\
        --resume models/adapter/step_1000.pt
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from .codec import RVQVAE
from .config import AudioCodecConfig, DEFAULT_CONFIG
from .text_to_codec_model import (
    TextToCodecConfig,
    TextToCodecModel,
    tokenize_text,
    tokenize_text_batch,
)
from .utils import load_audio, save_audio

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("train_adapter")

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


def detect_device(device_arg: Optional[str] = None) -> torch.device:
    """Auto-detect the best available device."""
    if device_arg is not None:
        return torch.device(device_arg)

    if torch.cuda.is_available():
        is_rocm = torch.version.hip is not None
        backend = "ROCm" if is_rocm else "CUDA"
        log.info("Detected %s device: %s", backend, torch.cuda.get_device_name(0))
        return torch.device("cuda")

    log.info("No GPU detected — falling back to CPU (training may be slow)")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class AdapterDataset(Dataset):
    """Dataset that pairs audio files with text transcripts.

    For each sample:
    1. Loads audio → encodes with frozen codec → gets VQ indices
    2. Tokenizes text to byte-level tokens
    3. Returns (text_tokens, codec_indices, text_len, codec_len)

    Parameters
    ----------
    codec_encoder : nn.Module
        Frozen RVQVAE encoder (``encode_audio`` method).
    audio_dir : str or Path
        Directory containing WAV files.
    text_file : str or Path
        Path to JSONL file with ``{"audio": ..., "text": ...}`` entries.
    text_max_len : int
        Maximum text token sequence length.
    codec_config : AudioCodecConfig
        Codec configuration (for latent frame computation).
    """

    def __init__(
        self,
        codec_encoder: nn.Module,
        audio_dir: str,
        text_file: str,
        text_max_len: int = 512,
        codec_config: Optional[AudioCodecConfig] = None,
    ):
        super().__init__()
        self.codec_encoder = codec_encoder
        self.audio_dir = Path(audio_dir).expanduser().resolve()
        self.text_max_len = text_max_len
        self.codec_config = codec_config or DEFAULT_CONFIG

        # Load transcripts
        text_path = Path(text_file).expanduser().resolve()
        if not text_path.exists():
            raise FileNotFoundError(f"Text file not found: {text_path}")

        self.samples: List[Dict[str, str]] = []
        with open(text_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    self.samples.append(entry)

        log.info("Loaded %d transcript entries from %s", len(self.samples), text_path)

        # Pre-compute latent frames per segment
        self.total_downsample = self.codec_config.total_downsample_factor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """Get a single training sample.

        Returns
        -------
        text_tokens : Tensor
            ``(L,)`` byte-level token ids.
        codec_indices : Tensor
            ``(T_latent, n_codebooks)`` codec indices from the encoder.
        text_len : int
            Actual text length.
        codec_len : int
            Actual codec frame count.
        """
        sample = self.samples[idx]

        # ── Load audio ──
        audio_path = self.audio_dir / sample["audio"]
        if not audio_path.exists():
            # Try relative to audio_dir
            audio_path = Path(sample["audio"])
            if not audio_path.is_absolute():
                audio_path = self.audio_dir / sample["audio"]

        waveform = load_audio(str(audio_path), sr=self.codec_config.sample_rate)

        # Trim or pad to max_seq_len
        target_len = self.codec_config.max_seq_len
        if waveform.shape[-1] >= target_len:
            # Random crop (for training variety)
            import random

            start = random.randint(0, waveform.shape[-1] - target_len)
            waveform = waveform[:, start : start + target_len]
        else:
            # Pad
            pad_len = target_len - waveform.shape[-1]
            waveform = F.pad(waveform, (0, pad_len))

        waveform = waveform.unsqueeze(0)  # (1, 1, T)

        # ── Encode with frozen codec ──
        device = next(self.codec_encoder.parameters()).device
        waveform = waveform.to(device)

        with torch.no_grad():
            _, indices_list = self.codec_encoder.encode_audio(waveform)
            # indices_list: list of 8 tensors, each (1, T_latent)
            # Stack to (1, T_latent, n_codebooks)
            codec_indices = torch.stack(indices_list, dim=-1).squeeze(0)
            # (T_latent, n_codebooks)

        # ── Tokenize text ──
        text_tokens = tokenize_text(sample["text"], max_len=self.text_max_len)
        # (L,)

        return (
            text_tokens,
            codec_indices.cpu(),
            text_tokens.size(0),
            codec_indices.size(0),
        )


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def collate_adapter(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for ``DataLoader``.

    Pads text tokens and codec indices to the max length in the batch.

    Returns
    -------
    text_tokens : Tensor
        ``(B, max_text_L)``, padded with 0.
    codec_indices : Tensor
        ``(B, max_codec_T, n_codebooks)``, padded with -1.
    text_lengths : Tensor
        ``(B,)`` actual text lengths.
    codec_lengths : Tensor
        ``(B,)`` actual codec frame counts.
    """
    text_tokens_list, codec_indices_list, text_lens, codec_lens = zip(*batch)

    # Pad text tokens
    max_text_len = max(text_lens)
    padded_text = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    for i, tokens in enumerate(text_tokens_list):
        padded_text[i, : tokens.size(0)] = tokens

    # Pad codec indices (-1 for invalid positions, masked in loss)
    max_codec_len = max(codec_lens)
    n_codebooks = codec_indices_list[0].size(-1)
    padded_codec = torch.full(
        (len(batch), max_codec_len, n_codebooks),
        -1,
        dtype=torch.long,
    )
    for i, indices in enumerate(codec_indices_list):
        padded_codec[i, : indices.size(0), :] = indices

    return (
        padded_text,
        padded_codec,
        torch.tensor(text_lens, dtype=torch.long),
        torch.tensor(codec_lens, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@torch.no_grad()
def validate_adapter(
    model: TextToCodecModel,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """Run validation pass.

    Returns
    -------
    avg_loss : float
    avg_codec_loss : float
    avg_duration_loss : float
    avg_token_accuracy : float
        Percentage of correctly predicted codec tokens.
    """
    model.eval()
    total_loss = 0.0
    total_codec = 0.0
    total_dur = 0.0
    total_correct = 0
    total_tokens = 0
    n_batches = 0

    for batch in val_loader:
        text_tokens, codec_indices, text_lengths, codec_lengths = [
            t.to(device) if isinstance(t, torch.Tensor) else t for t in batch
        ]

        loss, codec_loss, dur_loss = model.compute_loss(
            text_tokens, text_lengths, codec_indices, codec_lengths
        )

        total_loss += loss.item()
        total_codec += codec_loss.item()
        total_dur += dur_loss.item()
        n_batches += 1

        # Token accuracy
        logits, _ = model(text_tokens, text_lengths=text_lengths)
        # logits: (B, L', n_codebooks, codebook_size)
        pred = logits.argmax(dim=-1)  # (B, L', n_codebooks)
        for cb in range(pred.size(-1)):
            target = codec_indices[:, : pred.size(1), cb]
            valid = target >= 0
            total_correct += ((pred[:, :, cb] == target) & valid).sum().item()
            total_tokens += valid.sum().item()

    avg_loss = total_loss / max(n_batches, 1)
    avg_codec = total_codec / max(n_batches, 1)
    avg_dur = total_dur / max(n_batches, 1)
    accuracy = total_correct / max(total_tokens, 1) * 100.0

    model.train()
    return avg_loss, avg_codec, avg_dur, accuracy


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    step: int,
    loss: float,
    path: Path,
) -> None:
    """Save a training checkpoint."""
    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "text_config": model.config.__dict__,
    }
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    log.info("Checkpoint saved: %s (step=%d, loss=%.6f)", path, step, loss)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    device: torch.device = torch.device("cpu"),
) -> int:
    """Load a training checkpoint.

    Returns
    -------
    start_step : int
        The step number to resume from.
    """
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in state:
        scaler.load_state_dict(state["scaler_state_dict"])
    start_step = state.get("step", 0)
    loss = state.get("loss", float("inf"))
    log.info("Loaded checkpoint: %s (step=%d, loss=%.6f)", path, start_step, loss)
    return start_step


# ---------------------------------------------------------------------------
# Warmup + Cosine scheduler
# ---------------------------------------------------------------------------


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        eta_min: float = 1e-6,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            scale = (step + 1) / max(1, self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        return [
            self.eta_min
            + 0.5 * (base_lr - self.eta_min) * (1.0 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]


# ---------------------------------------------------------------------------
# Log sample generations to TensorBoard
# ---------------------------------------------------------------------------


@torch.no_grad()
def log_generations(
    writer: SummaryWriter,
    model: TextToCodecModel,
    codec_decoder: nn.Module,
    val_batch: Tuple,
    step: int,
    codec_config: AudioCodecConfig,
    device: torch.device,
    n_samples: int = 2,
) -> None:
    """Generate audio from text and log to TensorBoard.

    Takes the first ``n_samples`` from the validation batch, runs the
    full text → codec indices → audio pipeline, and logs the resulting
    audio to TensorBoard.
    """
    model.eval()
    text_tokens, _, text_lengths, codec_lengths = [
        t.to(device) if isinstance(t, torch.Tensor) else t
        for t in val_batch
    ]

    for i in range(min(n_samples, text_tokens.size(0))):
        tokens = text_tokens[i : i + 1, : text_lengths[i]]
        lengths = text_lengths[i : i + 1]

        # Generate codec indices
        codec_indices = model.generate(tokens, text_lengths=lengths)
        # codec_indices: (1, L', n_codebooks)

        # Decode to audio via codec decoder
        # First, reconstruct the quantised latent from indices
        # We need the codec's decoder, which takes (B, latent_dim, T_latent)
        # So we need to look up the codebook embeddings and sum them.
        # For simplicity, use the full codec's reconstruct path:
        # Encode → indices → decode. But we already have indices.
        # The decoder needs z_q (quantized latent), not indices.
        # Let's create z_q by looking up embeddings then summing.

        z_q = _indices_to_zq(
            codec_indices,
            codec_decoder.res_vq,
            device,
        )  # (1, latent_dim, T_latent)

        # Create a dummy speaker embedding
        speaker_emb = torch.zeros(1, codec_config.speaker_dim, device=device)

        audio = codec_decoder.decode(z_q, speaker_emb)  # (1, 1, T)

        writer.add_audio(
            f"generations/sample_{i}",
            audio[0].cpu().clamp(-1.0, 1.0),
            global_step=step,
            sample_rate=codec_config.sample_rate,
        )

    model.train()


@torch.no_grad()
def _indices_to_zq(
    indices: torch.Tensor,
    res_vq: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Convert codec indices to quantised latent ``z_q``.

    Looks up each codebook embedding and sums across codebooks.

    Parameters
    ----------
    indices : Tensor
        ``(B, T, n_codebooks)``.
    res_vq : ResidualVQ
        The RVQ module with ``embed`` buffer ``(n_codebooks, codebook_size, code_dim)``.
    device : torch.device

    Returns
    -------
    z_q : Tensor
        ``(B, code_dim, T)``, shape compatible with ``post_vq`` and the decoder.
    """
    B, T, n_cb = indices.shape
    code_dim = res_vq.code_dim
    z_q = torch.zeros(B, code_dim, T, device=device)

    for cb in range(n_cb):
        embed = res_vq.embed[cb]  # (K, code_dim)
        idx = indices[:, :, cb].clamp(min=0, max=res_vq.codebook_size - 1)  # (B, T)
        z_q_cb = F.embedding(idx, embed)  # (B, T, code_dim)
        z_q = z_q + z_q_cb.permute(0, 2, 1)  # (B, code_dim, T)

    return z_q


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Text→Codec adapter model for voice cloning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    parser.add_argument(
        "--codec_checkpoint",
        type=str,
        required=True,
        help="Path to trained codec checkpoint (codec_final.pt)",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory containing WAV audio files (segments/)",
    )
    parser.add_argument(
        "--text_file",
        type=str,
        required=True,
        help="Path to transcripts.jsonl (JSONL with audio/text fields)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and tensorboard logs",
    )

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)

    # Adapter model config
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=768)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--duration_loss_weight", type=float, default=1.0
    )

    # Checkpoint / resume
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to adapter checkpoint"
    )

    # Validation
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--val_every", type=int, default=200)

    # Logging
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--gen_interval", type=int, default=500)

    # Device
    parser.add_argument("--device", type=str, default=None)

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Resolve paths
    audio_dir = Path(args.audio_dir).expanduser().resolve()
    text_file = Path(args.text_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save training config
    with open(output_dir / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Device
    device = detect_device(args.device)
    use_amp = device.type == "cuda"
    log.info("Using device: %s  (mixed precision: %s)", device, use_amp)

    # ─── Load frozen codec ───────────────────────────────────────
    log.info("Loading codec from: %s", args.codec_checkpoint)
    codec_config = DEFAULT_CONFIG
    codec = RVQVAE(codec_config).to(device)

    ckpt = torch.load(
        Path(args.codec_checkpoint).expanduser().resolve(),
        map_location=device,
        weights_only=True,
    )
    state_dict = ckpt.get("model_state_dict", ckpt)
    codec.load_state_dict(state_dict, strict=False)  # allow partial match
    codec.eval()
    for p in codec.parameters():
        p.requires_grad_(False)
    log.info("Codec loaded (frozen) — %s", codec)

    # ─── Build adapter model ────────────────────────────────────
    text_config = TextToCodecConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        duration_loss_weight=args.duration_loss_weight,
    )
    model = TextToCodecModel(text_config).to(device)
    log.info(
        "Adapter model: %s  (%.2fM params)",
        model.__class__.__name__,
        model.parameter_count() / 1e6,
    )

    # ─── Data ───────────────────────────────────────────────────
    log.info("Loading training data from: %s", audio_dir)
    full_dataset = AdapterDataset(
        codec_encoder=codec,
        audio_dir=str(audio_dir),
        text_file=str(text_file),
        text_max_len=text_config.max_seq_len,
        codec_config=codec_config,
    )

    val_size = max(1, int(len(full_dataset) * args.val_split))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    log.info("Dataset split: %d train / %d validation", train_size, val_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_adapter,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_adapter,
        drop_last=False,
    )

    # ─── Optimiser ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )

    # Steps per epoch estimate
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = min(200, total_steps // 20)

    scheduler = WarmupCosineLR(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        eta_min=1e-6,
    )

    # ─── Mixed precision ───────────────────────────────────────
    scaler = GradScaler(enabled=use_amp)

    # ─── TensorBoard ───────────────────────────────────────────
    tb_dir = output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tb_dir))

    # ─── Resume ────────────────────────────────────────────────
    start_step = 0
    start_epoch = 0
    if args.resume:
        ckpt_path = Path(args.resume).expanduser().resolve()
        start_step = load_checkpoint(ckpt_path, model, optimizer, scaler, device)
        start_epoch = start_step // max(steps_per_epoch, 1)

    # ─── Training state ────────────────────────────────────────
    best_val_loss = float("inf")
    best_ckpt_path = output_dir / "best.pt"
    global_step = start_step

    # ─── Training loop ─────────────────────────────────────────
    log.info("=" * 60)
    log.info(
        "Starting adapter training: %d epochs, %d steps/epoch",
        args.epochs,
        steps_per_epoch,
    )
    log.info("=" * 60)

    model.train()
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        epoch_batches = 0

        for batch in train_loader:
            text_tokens, codec_indices, text_lengths, codec_lengths = batch
            text_tokens = text_tokens.to(device, non_blocking=True)
            codec_indices = codec_indices.to(device, non_blocking=True)
            text_lengths = text_lengths.to(device, non_blocking=True)
            codec_lengths = codec_lengths.to(device, non_blocking=True)

            # ── Forward ──
            with autocast(enabled=use_amp):
                loss, codec_loss, dur_loss = model.compute_loss(
                    text_tokens, text_lengths, codec_indices, codec_lengths
                )

            # ── Backward ──
            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()

            global_step += 1
            epoch_loss += loss.item()
            epoch_batches += 1

            # ── Logging ──
            if global_step % args.log_interval == 0:
                lr_current = optimizer.param_groups[0]["lr"]
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/codec_loss", codec_loss.item(), global_step)
                writer.add_scalar("train/duration_loss", dur_loss.item(), global_step)
                writer.add_scalar("train/grad_norm", grad_norm, global_step)
                writer.add_scalar("train/lr", lr_current, global_step)

                log.info(
                    "E%03d | step %6d | loss %.5f | codec %.5f | dur %.5f "
                    "| grad %.4f | lr %.2e",
                    epoch + 1,
                    global_step,
                    loss.item(),
                    codec_loss.item(),
                    dur_loss.item(),
                    grad_norm,
                    lr_current,
                )

            # ── Validation ──
            if global_step % args.val_every == 0:
                val_loss, val_codec, val_dur, val_acc = validate_adapter(
                    model, val_loader, device
                )
                writer.add_scalar("val/loss", val_loss, global_step)
                writer.add_scalar("val/codec_loss", val_codec, global_step)
                writer.add_scalar("val/duration_loss", val_dur, global_step)
                writer.add_scalar("val/token_accuracy", val_acc, global_step)
                log.info(
                    "═══ VALIDATION ═══ step %6d | loss %.5f | codec %.5f "
                    "| dur %.5f | acc %.2f%%",
                    global_step,
                    val_loss,
                    val_codec,
                    val_dur,
                    val_acc,
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model, optimizer, scaler, global_step, val_loss, best_ckpt_path
                    )
                    log.info("★ New best model (val_loss=%.6f)", val_loss)

            # ── Generate samples ──
            if global_step % args.gen_interval == 0:
                val_batch = next(iter(val_loader))
                log_generations(
                    writer,
                    model,
                    codec,
                    val_batch,
                    global_step,
                    codec_config,
                    device,
                    n_samples=2,
                )

            # ── Regular checkpoint ──
            if global_step % (args.val_every * 2) == 0:
                ckpt_name = f"step_{global_step}_{loss.item():.4f}.pt"
                ckpt_path = output_dir / ckpt_name
                save_checkpoint(
                    model, optimizer, scaler, global_step, loss.item(), ckpt_path
                )

        # ── End of epoch ──
        scheduler.step()
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
        log.info(
            "─── Epoch %3d complete | avg_loss %.5f | time %ds ───",
            epoch + 1,
            avg_epoch_loss,
            int(epoch_time),
        )

    # ── Final save ──
    final_path = output_dir / "adapter_final.pt"
    save_checkpoint(
        model, optimizer, scaler, global_step, avg_epoch_loss, final_path
    )
    log.info("Training complete! Final model saved: %s", final_path)
    if best_ckpt_path.exists():
        log.info("Best checkpoint: %s (val_loss=%.6f)", best_ckpt_path, best_val_loss)

    writer.close()


if __name__ == "__main__":
    main()

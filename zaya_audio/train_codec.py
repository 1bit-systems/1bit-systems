#! /usr/bin/env python3
"""Training harness for the RVQ-VAE audio codec (zaya_audio).

Trains the ``RVQVAE`` model from ``zaya_audio.codec`` on a directory of WAV
files using multi-scale STFT loss + VQ commitment loss.  Supports mixed-precision
training (ROCm / CUDA / CPU), checkpoint resume, TensorBoard logging, and
periodic validation.

Usage
-----
.. code-block:: bash

    python -m zaya_audio.train_codec \\
        --data_dir ./voice_samples/my_voice/segments \\
        --output_dir ./training/my_codec \\
        --batch_size 8 --epochs 100

    # Resume from checkpoint
    python -m zaya_audio.train_codec \\
        --data_dir ./voice_samples/my_voice/segments \\
        --output_dir ./training/my_codec \\
        --resume ./training/my_codec/step_1234_0.0234.pt
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split

from .codec import RVQVAE, MultiScaleSTFTLoss
from .config import AudioCodecConfig, DEFAULT_CONFIG
from .dataset import AudioDataset, collate_audio

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("train_codec")

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def detect_device(device_arg: Optional[str] = None) -> torch.device:
    """Auto-detect the best available device."""
    if device_arg is not None:
        return torch.device(device_arg)

    if torch.cuda.is_available():
        # Check if ROCm or CUDA
        is_rocm = torch.version.hip is not None
        backend = "ROCm" if is_rocm else "CUDA"
        log.info("Detected %s device: %s", backend, torch.cuda.get_device_name(0))
        return torch.device("cuda")

    log.info("No GPU detected — falling back to CPU (training may be slow)")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Warmup + Cosine scheduler
# ---------------------------------------------------------------------------

class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with linear warmup.

    Linearly ramps the learning rate from ``0`` to ``base_lr`` over
    ``warmup_epochs``, then follows a cosine decay to ``eta_min`` over
    the remaining epochs.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    warmup_epochs : int
        Number of warmup epochs (linear ramp).
    total_epochs : int
        Total number of training epochs.
    eta_min : float
        Minimum learning rate after cosine decay.
    """

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 eta_min: float = 1e-6):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            # Linear warmup
            scale = (epoch + 1) / self.warmup_epochs
            return [base_lr * scale for base_lr in self.base_lrs]
        # Cosine decay after warmup
        progress = (epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs
        )
        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) * (1.0 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model: nn.Module, val_loader: DataLoader, config: AudioCodecConfig,
             device: torch.device) -> Tuple[float, float, float]:
    """Run a full validation pass.

    Returns
    -------
    avg_loss : float
        Average total loss over the validation set.
    avg_commit : float
        Average commitment loss.
    avg_diversity : float
        Average diversity loss.
    """
    model.eval()
    total_loss = 0.0
    total_commit = 0.0
    total_diversity = 0.0
    n_batches = 0

    for batch in val_loader:
        waveforms, speaker_ids, _ = batch
        waveforms = waveforms.to(device)
        speaker_ids = speaker_ids.to(device)

        # Forward
        x_hat, vq_indices, commit_loss, diversity_loss = model(
            waveforms, speaker_ids=speaker_ids
        )
        loss = model.compute_loss(waveforms, x_hat, commit_loss, diversity_loss)

        total_loss += loss.item()
        total_commit += commit_loss.item()
        total_diversity += diversity_loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_commit = total_commit / max(n_batches, 1)
    avg_diversity = total_diversity / max(n_batches, 1)

    model.train()
    return avg_loss, avg_commit, avg_diversity


def compute_codebook_usage(indices: torch.Tensor, n_codebooks: int,
                           codebook_size: int) -> torch.Tensor:
    """Compute per-codebook usage percentage.

    Parameters
    ----------
    indices : Tensor
        Codebook indices for the first codebook, ``(B, T)``.
    n_codebooks : int
        Total number of codebooks.
    codebook_size : int
        Number of entries per codebook.

    Returns
    -------
    usage_pct : Tensor
        Percentage of codebook entries used, ``(n_codebooks,)``.
    """
    # For simplicity we log the first codebook usage.
    # A more complete implementation would track all codebooks.
    flat = indices.flatten()
    usage = torch.bincount(flat, minlength=codebook_size)
    used = (usage > 0).sum().item()
    return torch.tensor([used / codebook_size * 100.0] * n_codebooks)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    scaler: Optional[GradScaler], step: int, loss: float,
                    path: Path) -> None:
    """Save a training checkpoint."""
    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": model.config.__dict__,
    }
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    log.info("Checkpoint saved: %s (step=%d, loss=%.6f)", path, step, loss)


def load_checkpoint(path: Path, model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scaler: Optional[GradScaler] = None,
                    device: torch.device = torch.device("cpu")) -> int:
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
# Audio logging to TensorBoard
# ---------------------------------------------------------------------------

def log_audio_samples(writer: SummaryWriter, model: nn.Module,
                      batch: Tuple[torch.Tensor, torch.Tensor, list],
                      step: int, sr: int, tag: str = "audio/recon") -> None:
    """Log original and reconstructed audio to TensorBoard."""
    model.eval()
    waveforms, speaker_ids, _ = batch
    device = next(model.parameters()).device
    waveforms = waveforms.to(device)
    speaker_ids = speaker_ids.to(device)

    with torch.no_grad():
        x_hat, _, _, _ = model(waveforms, speaker_ids=speaker_ids)

    # Log first 4 samples
    n_samples = min(4, waveforms.shape[0])
    for i in range(n_samples):
        writer.add_audio(
            f"{tag}/original_{i}",
            waveforms[i].cpu(),
            global_step=step,
            sample_rate=sr,
        )
        writer.add_audio(
            f"{tag}/reconstructed_{i}",
            x_hat[i].cpu().clamp(-1.0, 1.0),
            global_step=step,
            sample_rate=sr,
        )
    model.train()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the RVQ-VAE audio codec for voice cloning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing WAV files for training")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for checkpoints and logs")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size per step")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Peak learning rate")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Total number of training epochs")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes")

    # Checkpoint / resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Device
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (e.g. 'cuda:0', 'cpu')")

    # Validation
    parser.add_argument("--val_split", type=float, default=0.05,
                        help="Fraction of data held out for validation")
    parser.add_argument("--val_every", type=int, default=500,
                        help="Run validation every N training steps")

    # Logging
    parser.add_argument("--log_interval", type=int, default=20,
                        help="Log loss every N steps")
    parser.add_argument("--audio_log_interval", type=int, default=2000,
                        help="Log audio samples every N steps")

    return parser


# ---------------------------------------------------------------------------
# Lazy SummaryWriter (avoids hard tensorboard dep at import time)
# ---------------------------------------------------------------------------

def _get_summary_writer(*args, **kwargs):
    """Lazily import tensorboard when first needed."""
    from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(*args, **kwargs)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Resolve paths
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = detect_device(args.device)
    use_amp = (device.type == "cuda")
    log.info("Using device: %s  (mixed precision: %s)", device, use_amp)

    # ---- Data ----
    config = DEFAULT_CONFIG
    log.info("Loading audio dataset from: %s", data_dir)
    full_dataset = AudioDataset(
        root_dir=str(data_dir),
        config=config,
        augment=True,
    )

    # Split train / validation
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
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
        collate_fn=collate_audio,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_audio,
        drop_last=False,
    )

    # ---- Model ----
    log.info("Building RVQVAE model...")
    model = RVQVAE(config).to(device)
    log.info("Model: %s  (%.2fM params)",
             model, model.parameter_count() / 1e6)

    # ---- Optimiser ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )

    # ---- Scheduler (warmup + cosine) ----
    scheduler = WarmupCosineLR(
        optimizer,
        warmup_epochs=min(5, args.epochs),
        total_epochs=args.epochs,
        eta_min=1e-6,
    )

    # ---- Mixed precision ----
    scaler = GradScaler(enabled=use_amp)

    # ---- TensorBoard ----
    tb_dir = output_dir / "tensorboard"
    writer = _get_summary_writer(log_dir=str(tb_dir))
    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.__dict__, f, indent=2)
    log.info("Config saved: %s", config_path)

    # ---- Resume ----
    start_step = 0
    start_epoch = 0
    if args.resume:
        ckpt_path = Path(args.resume).expanduser().resolve()
        start_step = load_checkpoint(ckpt_path, model, optimizer, scaler, device)
        start_epoch = start_step // max(len(train_loader), 1)

    # ---- Training state ----
    best_val_loss = float("inf")
    best_ckpt_path = output_dir / "model_best.pt"
    global_step = start_step

    # ---- Training loop ----
    log.info("=" * 60)
    log.info("Starting training: %d epochs, %d batches/epoch",
             args.epochs, len(train_loader))
    log.info("=" * 60)

    model.train()
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        epoch_batches = 0

        for batch in train_loader:
            waveforms, speaker_ids, _ = batch
            waveforms = waveforms.to(device, non_blocking=True)
            speaker_ids = speaker_ids.to(device, non_blocking=True)

            # ---- Forward (mixed precision) ----
            with autocast(enabled=use_amp):
                x_hat, vq_indices, commit_loss, diversity_loss = model(
                    waveforms, speaker_ids=speaker_ids
                )
                loss = model.compute_loss(
                    waveforms, x_hat, commit_loss, diversity_loss
                )

            # ---- Backward ----
            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                # Gradient clipping (unscale first)
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

            # ---- Logging (loss, grad norm, codebook usage) ----
            if global_step % args.log_interval == 0:
                lr_current = optimizer.param_groups[0]["lr"]
                usage_pct = compute_codebook_usage(
                    vq_indices, config.n_codebooks, config.codebook_size
                ).mean().item()

                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/commit_loss", commit_loss.item(), global_step)
                writer.add_scalar("train/diversity_loss", diversity_loss.item(), global_step)
                writer.add_scalar("train/grad_norm", grad_norm, global_step)
                writer.add_scalar("train/lr", lr_current, global_step)
                writer.add_scalar("train/codebook_usage_pct", usage_pct, global_step)

                log.info(
                    "E%03d | step %6d | loss %.5f | commit %.5f | div %.5f "
                    "| grad %.4f | lr %.2e | cb_usage %.1f%%",
                    epoch + 1, global_step, loss.item(),
                    commit_loss.item(), diversity_loss.item(),
                    grad_norm, lr_current, usage_pct,
                )

            # ---- Validation ----
            if global_step % args.val_every == 0:
                val_loss, val_commit, val_div = validate(
                    model, val_loader, config, device
                )
                writer.add_scalar("val/loss", val_loss, global_step)
                writer.add_scalar("val/commit_loss", val_commit, global_step)
                writer.add_scalar("val/diversity_loss", val_div, global_step)
                log.info(
                    "══════════ VALIDATION ══════════ "
                    "step %6d | loss %.5f | commit %.5f | div %.5f",
                    global_step, val_loss, val_commit, val_div,
                )

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, scaler, global_step,
                                    val_loss, best_ckpt_path)
                    log.info("★ New best model (val_loss=%.6f)", val_loss)

            # ---- Audio logging ----
            if global_step % args.audio_log_interval == 0:
                # Get a batch from the validation set for audio logging
                val_batch = next(iter(val_loader))
                log_audio_samples(writer, model, val_batch, global_step,
                                  config.sample_rate)

            # ---- Regular checkpoint ----
            if global_step % (args.val_every * 2) == 0:
                ckpt_name = f"step_{global_step}_{loss.item():.4f}.pt"
                ckpt_path = output_dir / ckpt_name
                save_checkpoint(model, optimizer, scaler, global_step,
                                loss.item(), ckpt_path)

        # ---- End of epoch ----
        scheduler.step()
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
        log.info(
            "─── Epoch %3d complete | avg_loss %.5f | time %ds ───",
            epoch + 1, avg_epoch_loss, int(epoch_time),
        )

    # ---- Final save ----
    final_path = output_dir / "codec_final.pt"
    save_checkpoint(model, optimizer, scaler, global_step,
                    avg_epoch_loss, final_path)
    log.info("Training complete! Final model saved: %s", final_path)
    log.info("Best checkpoint: %s (val_loss=%.6f)", best_ckpt_path, best_val_loss)

    writer.close()


if __name__ == "__main__":
    main()

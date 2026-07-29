#! /usr/bin/env python3
"""One-command pipeline orchestrator for zaya_audio.

Orchestrates the full voice cloning pipeline end-to-end:

    record (optional) → train_codec → extract_embeddings → train_adapter → export

Each stage can be run independently or as part of the full pipeline.
The pipeline supports resume from any stage (completed stages are
skipped on re-run).

Usage
-----
.. code-block:: bash

    # Run the full pipeline (record, train, embed, adapt, export)
    python -m zaya_audio.pipeline --mode all --voice-name my_voice

    # Run only specific stages
    python -m zaya_audio.pipeline --mode train_codec --voice-name my_voice

    # Resume from training (skip recording)
    python -m zaya_audio.pipeline --mode all --voice-name my_voice --resume
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(pipeline)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Stage state tracking
# ---------------------------------------------------------------------------

PIPELINE_STAGES = [
    "record",
    "train_codec",
    "extract_embeddings",
    "train_adapter",
    "export",
]


@dataclass
class PipelineState:
    """Tracks which pipeline stages have been completed."""
    stages: Dict[str, bool] = field(default_factory=dict)

    def __post_init__(self):
        for stage in PIPELINE_STAGES:
            self.stages.setdefault(stage, False)

    def mark_complete(self, stage: str) -> None:
        """Mark a stage as complete."""
        self.stages[stage] = True

    def is_complete(self, stage: str) -> bool:
        """Check if a stage is complete."""
        return self.stages.get(stage, False)

    def save(self, path: Path) -> None:
        """Save pipeline state to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"stages": self.stages}, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "PipelineState":
        """Load pipeline state from JSON."""
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return cls(stages=data.get("stages", {}))
        return cls()


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Configuration for a full pipeline run."""
    voice_name: str
    base_dir: Path
    data_dir: Optional[Path] = None
    codec_output_dir: Optional[Path] = None
    speaker_emb_path: Optional[Path] = None
    adapter_output_dir: Optional[Path] = None
    export_path: Optional[Path] = None

    # Hyperparameters
    record_duration: int = 1800
    record_sample_rate: int = 24000
    codec_batch_size: int = 8
    codec_lr: float = 3e-4
    codec_epochs: int = 100
    codec_num_workers: int = 4
    embedding_dim: int = 512
    adapter_rank: int = 16
    adapter_epochs: int = 50

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PipelineConfig":
        """Build config from parsed CLI args."""
        base_dir = Path(args.output_dir).expanduser().resolve() / args.voice_name
        cfg = cls(
            voice_name=args.voice_name,
            base_dir=base_dir,
            record_duration=args.record_duration,
            record_sample_rate=args.record_sample_rate,
            codec_batch_size=args.codec_batch_size,
            codec_lr=args.codec_lr,
            codec_epochs=args.codec_epochs,
            codec_num_workers=args.codec_num_workers,
            embedding_dim=args.embedding_dim,
            adapter_rank=args.adapter_rank,
            adapter_epochs=args.adapter_epochs,
        )

        # Derived paths
        cfg.data_dir = base_dir / "voice_samples" / args.voice_name / "segments"
        cfg.codec_output_dir = base_dir / "codec"
        cfg.speaker_emb_path = base_dir / "speaker_embedding.pt"
        cfg.adapter_output_dir = base_dir / "adapter"
        cfg.export_path = base_dir / f"{args.voice_name}.voice"

        return cfg


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_record(cfg: PipelineConfig) -> bool:
    """Run the recording stage."""
    log.info("Stage: record ── Recording voice samples")

    cmd = [
        sys.executable, "-m", "zaya_audio.record",
        "--duration", str(cfg.record_duration),
        "--sample_rate", str(cfg.record_sample_rate),
        "--output_dir", str(cfg.base_dir / "voice_samples"),
        "--speaker_name", cfg.voice_name,
    ]

    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=_project_root())
    if result.returncode != 0:
        log.error("Recording stage failed with code %d", result.returncode)
        return False

    log.info("Recording complete: %s", cfg.data_dir)
    return True


def run_train_codec(cfg: PipelineConfig, resume: bool = False) -> bool:
    """Run the codec training stage."""
    data_dir = cfg.data_dir
    if data_dir is None or not data_dir.exists():
        log.error("Data directory not found: %s. Run 'record' stage first.", data_dir)
        return False

    log.info("Stage: train_codec ── Training RVQ-VAE codec")

    cmd = [
        sys.executable, "-m", "zaya_audio.train_codec",
        "--data_dir", str(data_dir),
        "--output_dir", str(cfg.codec_output_dir),
        "--batch_size", str(cfg.codec_batch_size),
        "--lr", str(cfg.codec_lr),
        "--epochs", str(cfg.codec_epochs),
        "--num_workers", str(cfg.codec_num_workers),
    ]

    if resume:
        # Find latest checkpoint
        ckpt = _find_latest_checkpoint(cfg.codec_output_dir)
        if ckpt is not None:
            cmd.extend(["--resume", str(ckpt)])
            log.info("Resuming from checkpoint: %s", ckpt)

    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=_project_root())
    if result.returncode != 0:
        log.error("Codec training stage failed with code %d", result.returncode)
        return False

    log.info("Codec training complete: %s", cfg.codec_output_dir)
    return True


def run_extract_embeddings(cfg: PipelineConfig) -> bool:
    """Extract speaker embeddings from trained codec.

    Uses the trained encoder to compute a speaker embedding for the
    voice.  Currently a placeholder that produces a zero embedding
    to demonstrate the pipeline shape.

    TODO: Implement full embedding extraction from codec encoder +
    pooling over segments.
    """
    log.info("Stage: extract_embeddings ── Extracting speaker embedding")

    final_ckpt = cfg.codec_output_dir / "codec_final.pt" if cfg.codec_output_dir else None
    if final_ckpt is None or not final_ckpt.exists():
        log.warning(
            "codec_final.pt not found at %s. "
            "Will extract from the latest available checkpoint.",
            final_ckpt,
        )
        # Try best checkpoint
        best_ckpt = cfg.codec_output_dir / "model_best.pt" if cfg.codec_output_dir else None
        if best_ckpt is not None and best_ckpt.exists():
            final_ckpt = best_ckpt
        else:
            log.error("No trained codec found. Run 'train_codec' stage first.")
            return False

    # ── Extract speaker embedding ──────────────────────────────────
    # In a full implementation, we would:
    #   1. Load the trained RVQVAE model.
    #   2. Encode all voice segments.
    #   3. Average the latent representations or use a learnable
    #      embedding layer to produce a single 512-d vector.
    #
    # For now, we generate a default embedding to keep the pipeline
    # end-to-end functional.
    import torch

    log.info("Loading checkpoint: %s", final_ckpt)
    state = torch.load(final_ckpt, map_location="cpu", weights_only=True)

    # Create a default speaker embedding (512-dim, unit-norm)
    speaker_emb = torch.randn(1, cfg.embedding_dim)
    speaker_emb = speaker_emb / speaker_emb.norm(dim=1, keepdim=True)

    # Save
    if cfg.speaker_emb_path is not None:
        cfg.speaker_emb_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(speaker_emb, cfg.speaker_emb_path)
        log.info("Speaker embedding saved: %s (shape=%s)",
                 cfg.speaker_emb_path, list(speaker_emb.shape))

    return True


def run_train_adapter(cfg: PipelineConfig) -> bool:
    """Train LoRA adapter for voice adaptation.

    Placeholder: demonstrates the adapter training entry point.
    The actual adapter training loop should use the trained codec +
    speaker embedding + voice segments to fine-tune low-rank adapters
    on the decoder's FiLM layers.

    TODO: Implement full LoRA adapter training loop.
    """
    log.info("Stage: train_adapter ── Training LoRA voice adapter (placeholder)")

    if cfg.adapter_output_dir is not None:
        cfg.adapter_output_dir.mkdir(parents=True, exist_ok=True)

    # In a full implementation, this stage would:
    #   1. Load the trained codec decoder + speaker embedding.
    #   2. Insert LoRA adapters into decoder FiLM projections.
    #   3. Train adapters on the voice segments (freeze base model).
    #   4. Save adapter weights to adapter_output_dir / "adapter_final.pt".
    #
    # For now, we create an empty placeholder.
    import torch

    # Dummy adapter state (rank-16 LoRA on a single linear layer)
    dummy_adapter = {
        "dummy_lora_a": torch.randn(cfg.embedding_dim, cfg.adapter_rank),
        "dummy_lora_b": torch.randn(cfg.adapter_rank, 1024),
    }
    adapter_path = cfg.adapter_output_dir / "adapter_final.pt"
    torch.save(dummy_adapter, adapter_path)
    log.info("Adapter saved (placeholder): %s", adapter_path)

    return True


def run_export(cfg: PipelineConfig) -> bool:
    """Export the trained voice as a .voice pack."""
    log.info("Stage: export ── Exporting .voice pack")

    # Paths
    if cfg.codec_output_dir is None:
        log.error("codec_output_dir not set.")
        return False

    final_ckpt = cfg.codec_output_dir / "codec_final.pt"
    if not final_ckpt.exists():
        final_ckpt = cfg.codec_output_dir / "model_best.pt"

    speaker_emb_path = cfg.speaker_emb_path
    adapter_path = cfg.adapter_output_dir / "adapter_final.pt" if cfg.adapter_output_dir else None

    if not final_ckpt.exists():
        log.error("No codec checkpoint found in %s. Run 'train_codec' stage first.",
                  cfg.codec_output_dir)
        return False

    if speaker_emb_path is None or not speaker_emb_path.exists():
        log.error("Speaker embedding not found at %s. Run 'extract_embeddings' stage first.",
                  speaker_emb_path)
        return False

    import torch
    from .voice_pack import save_voice_pack, VoicePackMetadata
    from .config import DEFAULT_CONFIG

    # Load decoder state from checkpoint
    ckpt = torch.load(final_ckpt, map_location="cpu", weights_only=True)
    model_state = ckpt.get("model_state_dict", ckpt)

    # Extract only decoder weights (decoder.X.*, post_conv.*, dec_proj.*)
    decoder_state = {
        k: v for k, v in model_state.items()
        if k.startswith("decoder.") or k.startswith("post_conv.") or k.startswith("dec_proj.")
    }
    log.info("Extracted decoder state: %d tensors", len(decoder_state))

    # Load speaker embedding
    speaker_emb = torch.load(speaker_emb_path, map_location="cpu", weights_only=True)

    # Load adapter (optional)
    adapter_state = None
    if adapter_path is not None and adapter_path.exists():
        adapter_state = torch.load(adapter_path, map_location="cpu", weights_only=True)
        log.info("Loaded adapter: %s", adapter_path)

    # Build metadata
    config_dict = DEFAULT_CONFIG.__dict__.copy()
    metadata = VoicePackMetadata(
        name=cfg.voice_name,
        speaker_name=cfg.voice_name,
        language="en",
        sample_rate=cfg.record_sample_rate,
        created_at=datetime.utcnow().isoformat() + "Z",
        model_version="0.1.0",
        codec_params=config_dict,
        mos_score=0.0,
    )

    # Save
    if cfg.export_path is not None:
        save_voice_pack(
            str(cfg.export_path),
            decoder_state,
            speaker_emb,
            adapter_state=adapter_state,
            metadata=metadata,
            config=config_dict,
        )
        log.info("Voice pack exported: %s", cfg.export_path)

    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """Return the project root directory (parent of zaya_audio)."""
    return Path(__file__).resolve().parent.parent


def _find_latest_checkpoint(output_dir: Optional[Path]) -> Optional[Path]:
    """Find the latest training checkpoint in the output directory."""
    if output_dir is None or not output_dir.exists():
        return None
    checkpoints = sorted(output_dir.glob("step_*.pt"))
    if checkpoints:
        return checkpoints[-1]  # lexicographically sorted → latest
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zaya Audio — One-command voice cloning pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["all"] + PIPELINE_STAGES,
        help="Pipeline stage to run (or 'all' for end-to-end)",
    )

    # Voice identity
    parser.add_argument("--voice-name", type=str, required=True,
                        help="Name for the voice being cloned")

    # Output directory
    parser.add_argument("--output-dir", type=str, default="./pipeline_output",
                        help="Base output directory for all pipeline results")

    # Resume
    parser.add_argument("--resume", action="store_true",
                        help="Skip completed stages and resume from the last "
                             "incomplete stage")

    # Recording
    parser.add_argument("--record-duration", type=int, default=1800,
                        help="Recording duration in seconds")
    parser.add_argument("--record-sample-rate", type=int, default=24000,
                        help="Recording sample rate")

    # Codec training
    parser.add_argument("--codec-batch-size", type=int, default=8,
                        help="Codec training batch size")
    parser.add_argument("--codec-lr", type=float, default=3e-4,
                        help="Codec training learning rate")
    parser.add_argument("--codec-epochs", type=int, default=100,
                        help="Codec training epochs")
    parser.add_argument("--codec-num-workers", type=int, default=4,
                        help="Codec training data loader workers")

    # Embedding / adapter
    parser.add_argument("--embedding-dim", type=int, default=512,
                        help="Speaker embedding dimensionality")
    parser.add_argument("--adapter-rank", type=int, default=16,
                        help="LoRA adapter rank")
    parser.add_argument("--adapter-epochs", type=int, default=50,
                        help="LoRA adapter training epochs")

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Build config
    cfg = PipelineConfig.from_args(args)
    cfg.base_dir.mkdir(parents=True, exist_ok=True)

    # Pipeline state
    state_path = cfg.base_dir / ".pipeline_state.json"
    state = PipelineState.load(state_path) if args.resume else PipelineState()

    # Determine stages to run
    if args.mode == "all":
        stages_to_run = PIPELINE_STAGES
    else:
        stages_to_run = [args.mode]

    # Filter out completed stages if resuming
    if args.resume:
        skipped = [s for s in stages_to_run if state.is_complete(s)]
        stages_to_run = [s for s in stages_to_run if not state.is_complete(s)]
        if skipped:
            log.info("Resume: skipping %d completed stage(s): %s",
                     len(skipped), ", ".join(skipped))

    if not stages_to_run:
        log.info("All requested stages already complete. Nothing to do.")
        log.info("To re-run, remove %s or use --mode to specify a stage.",
                 state_path)
        return

    # ── Run stages ────────────────────────────────────────────────
    log.info("═" * 56)
    log.info("Zaya Audio Pipeline  v0.1.0")
    log.info("Voice:  %s", cfg.voice_name)
    log.info("Mode:   %s", args.mode)
    log.info("Output: %s", cfg.base_dir)
    log.info("Stages: %s", ", ".join(stages_to_run))
    log.info("═" * 56)

    stage_runners = {
        "record": run_record,
        "train_codec": lambda c: run_train_codec(c, resume=args.resume),
        "extract_embeddings": run_extract_embeddings,
        "train_adapter": run_train_adapter,
        "export": run_export,
    }

    pipeline_ok = True
    for stage in stages_to_run:
        if stage not in stage_runners:
            log.error("Unknown stage: %s", stage)
            pipeline_ok = False
            break

        log.info("")
        log.info("▸▸▸ Stage: %s ▸▸▸", stage)
        start_time = time.time()

        success = stage_runners[stage](cfg)
        elapsed = time.time() - start_time

        if success:
            state.mark_complete(stage)
            state.save(state_path)
            log.info("✓ Stage '%s' complete in %ds", stage, int(elapsed))
        else:
            log.error("✗ Stage '%s' FAILED after %ds", stage, int(elapsed))
            pipeline_ok = False
            break

    # ── Summary ───────────────────────────────────────────────────
    print()
    print(f"[pipeline] {'═' * 48}")
    if pipeline_ok:
        print(f"[pipeline] Pipeline complete ✓")
        if cfg.export_path and cfg.export_path.exists():
            size_mb = cfg.export_path.stat().st_size / (1024 * 1024)
            print(f"[pipeline] Voice pack: {cfg.export_path} ({size_mb:.1f} MB)")
    else:
        print(f"[pipeline] Pipeline FAILED ✗ — check logs above")
    print(f"[pipeline] Output: {cfg.base_dir}")
    print(f"[pipeline] {'═' * 48}")


if __name__ == "__main__":
    main()

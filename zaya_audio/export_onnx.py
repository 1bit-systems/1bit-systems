#! /usr/bin/env python3
"""ONNX export for the zaya_audio voice cloning pipeline.

Exports the codec decoder and the text→codec adapter as standalone
ONNX models, then packs them into a ``.voice`` voice pack.

Usage
-----
.. code-block:: bash

    python -m zaya_audio.export_onnx \\
        --codec_checkpoint models/codec_final.pt \\
        --adapter_checkpoint models/adapter/best.pt \\
        --output_dir models/onnx/ \\
        --voice_name my_voice

The exported ONNX models:
    - ``codec_decoder.onnx`` — Decodes codec tokens + speaker emb → audio
    - ``adapter.onnx`` — Converts text tokens → codec token logits
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .codec import RVQVAE
from .config import AudioCodecConfig, DEFAULT_CONFIG
from .text_to_codec_model import TextToCodecConfig, TextToCodecModel
from .voice_pack import (
    VoicePackMetadata,
    save_voice_pack,
    load_voice_pack,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("export_onnx")

# ---------------------------------------------------------------------------
# ONNX export helpers
# ---------------------------------------------------------------------------


def _make_axes_dynamic(axes: Dict[str, int]) -> Dict[str, Dict[int, str]]:
    """Build ``dynamic_axes`` dict for ``torch.onnx.export``.

    Parameters
    ----------
    axes : dict of name → index
        E.g. ``{"seq_len": 0, "audio_samples": 1}``.

    Returns
    -------
    dict
        ``{name: {index: name}}`` for use with ``dynamic_axes``.
    """
    return {name: {idx: name} for name, idx in axes.items()}


# ═══════════════════════════════════════════════════════════════════
# 1. Export codec decoder
# ═══════════════════════════════════════════════════════════════════


def _export_codec_decoder(
    codec: RVQVAE,
    output_path: Path,
    verbose: bool = False,
) -> None:
    """Export the codec decoder portion as an ONNX model.

    The exported model takes ``(codec_tokens, speaker_emb)`` and
    outputs ``audio``.
    """
    codec.eval()
    device = next(codec.parameters()).device

    # ── Build the decoder graph manually ─────────────────────────
    # Inputs:
    #   codec_tokens: (n_codebooks, T_latent)  <- int32
    #   speaker_emb:  (1, speaker_dim)         <- float32
    #
    # Internal steps:
    #   1. Lookup codebook embeddings per codebook, sum -> z_q (1, code_dim, T)
    #   2. post_vq: (1, code_dim, T) -> (1, latent_dim, T)
    #   3. dec_proj: (1, latent_dim, T) -> (1, dec_in_ch, T)
    #   4. decoder blocks (FiLM + upsampling) -> (1, last_ch, T_audio)
    #   5. post_conv: (1, last_ch, T_audio) -> (1, 1, T_audio)

    class DecoderONNX(nn.Module):
        """ONNX-compatible decoder: codec tokens + speaker emb → audio."""

        def __init__(self, codec: RVQVAE):
            super().__init__()
            config = codec.config
            self.n_codebooks = config.n_codebooks
            self.codebook_size = config.codebook_size
            self.code_dim = config.code_dim
            self.latent_dim = config.latent_dim
            self.speaker_dim = config.speaker_dim

            # Copy codebook embeddings
            self.register_buffer("codebook_embed", codec.res_vq.embed.clone())

            # Post-VQ projection
            self.post_vq = codec.post_vq

            # Decoder projection and blocks
            self.dec_proj = codec.dec_proj
            self.decoder = codec.decoder
            self.post_conv = codec.post_conv

        def forward(
            self,
            codec_tokens: torch.Tensor,
            speaker_emb: torch.Tensor,
        ) -> torch.Tensor:
            """Forward pass.

            Parameters
            ----------
            codec_tokens : LongTensor
                ``(n_codebooks, T_latent)`` — int64 or int32.
            speaker_emb : Tensor
                ``(1, speaker_dim)``.

            Returns
            -------
            audio : Tensor
                ``(1, T_audio)``.
            """
            # Reshape tokens to (1, T, n_codebooks)
            tokens = codec_tokens.unsqueeze(0).permute(0, 2, 1).long()
            # tokens: (1, T_latent, n_codebooks)

            B, T, n_cb = tokens.shape
            device = tokens.device
            dtype = self.dec_proj.weight.dtype

            # ── Step 1: Lookup and sum codebook embeddings ──
            z_q = torch.zeros(B, self.code_dim, T, device=device, dtype=dtype)
            for cb in range(self.n_codebooks):
                embed = self.codebook_embed[cb]  # (K, code_dim)
                idx = tokens[:, :, cb].clamp(min=0, max=self.codebook_size - 1)
                z_q_cb = torch.nn.functional.embedding(idx, embed)  # (B, T, code_dim)
                z_q = z_q + z_q_cb.permute(0, 2, 1).to(dtype)

            # ── Step 2: Post-VQ projection ──
            z = self.post_vq(z_q)  # (B, latent_dim, T)

            # ── Step 3: Decoder ──
            x = self.dec_proj(z)  # (B, dec_in_ch, T)
            for block in self.decoder:
                x = block(x, speaker_emb)
            audio = self.post_conv(x)  # (B, 1, T_audio)

            return audio

    decoder_onnx = DecoderONNX(codec).to(device)
    decoder_onnx.eval()

    # ── Create dummy inputs ──
    config = codec.config
    T_latent = config.latent_frames_per_segment

    dummy_tokens = torch.randint(
        0, config.codebook_size,
        (config.n_codebooks, T_latent),
        dtype=torch.long,
        device=device,
    )
    dummy_emb = torch.randn(1, config.speaker_dim, device=device)

    # ── Export ──
    log.info("Exporting codec decoder to ONNX...")

    # Run once to verify shapes
    with torch.no_grad():
        audio_ref = decoder_onnx(dummy_tokens, dummy_emb)

    torch.onnx.export(
        decoder_onnx,
        (dummy_tokens, dummy_emb),
        str(output_path),
        input_names=["codec_tokens", "speaker_emb"],
        output_names=["audio"],
        dynamic_axes={
            "codec_tokens": {1: "seq_len"},  # dynamic T_latent
            "speaker_emb": {0: "batch"},
            "audio": {1: "audio_samples"},
        },
        opset_version=17,
        do_constant_folding=True,
        verbose=verbose,
    )

    log.info("Codec decoder ONNX exported: %s  (%.2f MB)", output_path,
             output_path.stat().st_size / 1e6)

    # ── Verify with onnxruntime ──
    _verify_onnx_decoder(
        str(output_path),
        decoder_onnx,
        dummy_tokens.cpu(),
        dummy_emb.cpu(),
        audio_ref.cpu(),
    )


def _verify_onnx_decoder(
    onnx_path: str,
    pytorch_model: nn.Module,
    tokens: torch.Tensor,
    speaker_emb: torch.Tensor,
    audio_ref: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> None:
    """Verify ONNX decoder output matches PyTorch."""
    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("onnxruntime not installed — skipping verification")
        return

    log.info("Verifying codec decoder ONNX against PyTorch...")

    # Create ONNX runtime session
    sess = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )

    # Run inference
    onnx_inputs = {
        "codec_tokens": tokens.numpy().astype(np.int64),
        "speaker_emb": speaker_emb.numpy().astype(np.float32),
    }
    onnx_outputs = sess.run(None, onnx_inputs)
    audio_onnx = torch.from_numpy(onnx_outputs[0])

    # Compare
    max_diff = (audio_onnx - audio_ref).abs().max().item()
    log.info("Max absolute difference: %.6f  (threshold: %.6f)", max_diff, atol)

    if max_diff > atol:
        log.warning(
            "ONNX verification WARNING: max diff %.6f exceeds threshold %.6f",
            max_diff, atol,
        )
    else:
        log.info("ONNX verification PASSED ✓")


# ═══════════════════════════════════════════════════════════════════
# 2. Export adapter (TextToCodecModel)
# ═══════════════════════════════════════════════════════════════════


def _export_adapter(
    model: TextToCodecModel,
    output_path: Path,
    verbose: bool = False,
) -> None:
    """Export the text→codec adapter as an ONNX model.

    The exported model takes ``(text_tokens, text_length)`` and
    outputs ``codec_logits``.
    """
    model.eval()
    device = next(model.parameters()).device

    # ── Create dummy inputs ──
    L = 32  # short sequence for export tracing
    dummy_tokens = torch.randint(0, 256, (1, L), dtype=torch.long, device=device)
    dummy_length = torch.tensor([L], dtype=torch.long, device=device)

    # ── Export ──
    log.info("Exporting adapter to ONNX...")

    # Run once to verify
    with torch.no_grad():
        logits_ref, _ = model(dummy_tokens, text_lengths=dummy_length)

    torch.onnx.export(
        model,
        (dummy_tokens, dummy_length),
        str(output_path),
        input_names=["text_tokens", "text_length"],
        output_names=["codec_logits"],
        dynamic_axes={
            "text_tokens": {1: "seq_len"},
            "codec_logits": {1: "expanded_len"},
        },
        opset_version=17,
        do_constant_folding=True,
        verbose=verbose,
    )

    log.info("Adapter ONNX exported: %s  (%.2f MB)", output_path,
             output_path.stat().st_size / 1e6)

    # ── Verify with onnxruntime ──
    _verify_onnx_adapter(
        str(output_path),
        model,
        dummy_tokens.cpu(),
        dummy_length.cpu(),
        logits_ref.cpu(),
    )


def _verify_onnx_adapter(
    onnx_path: str,
    pytorch_model: nn.Module,
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    logits_ref: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> None:
    """Verify ONNX adapter output matches PyTorch."""
    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("onnxruntime not installed — skipping verification")
        return

    log.info("Verifying adapter ONNX against PyTorch...")

    sess = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )

    onnx_inputs = {
        "text_tokens": tokens.numpy().astype(np.int64),
        "text_length": lengths.numpy().astype(np.int64),
    }
    onnx_outputs = sess.run(None, onnx_inputs)
    logits_onnx = torch.from_numpy(onnx_outputs[0])

    # Compare (align shapes — they may differ due to dynamic expansion)
    L_min = min(logits_ref.shape[1], logits_onnx.shape[1])
    max_diff = (logits_onnx[:, :L_min] - logits_ref[:, :L_min]).abs().max().item()

    log.info("Max absolute difference: %.6f  (threshold: %.6f)", max_diff, atol)

    if max_diff > atol:
        log.warning(
            "ONNX verification WARNING: max diff %.6f exceeds threshold %.6f",
            max_diff, atol,
        )
    else:
        log.info("ONNX verification PASSED ✓")


# ═══════════════════════════════════════════════════════════════════
# 3. Build .voice pack
# ═══════════════════════════════════════════════════════════════════


def _build_voice_pack(
    codec: RVQVAE,
    adapter: TextToCodecModel,
    output_dir: Path,
    voice_name: str,
) -> Path:
    """Pack the exported model components into a ``.voice`` archive.

    Parameters
    ----------
    codec : RVQVAE
        Full codec model (used to extract decoder state + config).
    adapter : TextToCodecModel
        Trained adapter model.
    output_dir : Path
        Directory to write the ``.voice`` file.
    voice_name : str
        Name for the voice.

    Returns
    -------
    voice_path : Path
        Path to the created ``.voice`` file.
    """
    codec_config = codec.config

    # Extract decoder state dict
    decoder_state = {
        k: v.cpu()
        for k, v in codec.state_dict().items()
        if k.startswith("decoder.")
        or k.startswith("post_conv.")
        or k.startswith("dec_proj.")
        or k.startswith("post_vq.")
        or k.startswith("res_vq.")  # needed for codebook lookups
    }

    # Create a default speaker embedding (512-dim, unit-norm)
    speaker_emb = torch.randn(1, codec_config.speaker_dim)
    speaker_emb = speaker_emb / speaker_emb.norm(dim=1, keepdim=True)

    # Extract adapter state dict
    adapter_state = {
        k: v.cpu() for k, v in adapter.state_dict().items()
    }

    # Build metadata
    config_dict = {
        "sample_rate": codec_config.sample_rate,
        "n_codebooks": codec_config.n_codebooks,
        "codebook_size": codec_config.codebook_size,
        "code_dim": codec_config.code_dim,
        "latent_dim": codec_config.latent_dim,
        "speaker_dim": codec_config.speaker_dim,
        "total_downsample_factor": codec_config.total_downsample_factor,
        "max_seq_len": codec_config.max_seq_len,
    }

    metadata = VoicePackMetadata(
        name=voice_name,
        speaker_name=voice_name,
        language="en",
        sample_rate=codec_config.sample_rate,
        model_version="0.1.0",
        codec_params=config_dict,
        mos_score=0.0,
    )

    voice_path = output_dir / f"{voice_name}.voice"
    save_voice_pack(
        str(voice_path),
        decoder_state,
        speaker_emb,
        adapter_state=adapter_state,
        metadata=metadata,
        config=config_dict,
    )

    return voice_path


# ═══════════════════════════════════════════════════════════════════
# Argument parser
# ═══════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export ONNX models and build .voice pack for zaya_audio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--codec_checkpoint",
        type=str,
        required=True,
        help="Path to trained codec checkpoint (codec_final.pt)",
    )
    parser.add_argument(
        "--adapter_checkpoint",
        type=str,
        default=None,
        help="Path to trained adapter checkpoint (best.pt or adapter_final.pt). "
             "If omitted, only the codec decoder is exported.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/onnx",
        help="Output directory for ONNX files and .voice pack",
    )
    parser.add_argument(
        "--voice_name",
        type=str,
        default="my_voice",
        help="Name for the voice pack",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose ONNX export logging",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (e.g. 'cpu')",
    )

    return parser


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ──
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log.info("Using device: %s", device)

    # ── Load codec ──
    log.info("Loading codec checkpoint: %s", args.codec_checkpoint)
    codec_config = DEFAULT_CONFIG
    codec = RVQVAE(codec_config).to(device)

    ckpt_path = Path(args.codec_checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        log.error("Codec checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    codec.load_state_dict(state_dict, strict=False)
    codec.eval()
    log.info("Codec loaded: %s", codec)

    # ── Export codec decoder ──
    decoder_onnx_path = output_dir / "codec_decoder.onnx"
    _export_codec_decoder(codec, decoder_onnx_path, verbose=args.verbose)

    # ── Load adapter (if given) ──
    adapter = None
    if args.adapter_checkpoint:
        adapter_ckpt_path = Path(args.adapter_checkpoint).expanduser().resolve()
        if not adapter_ckpt_path.exists():
            log.error("Adapter checkpoint not found: %s", adapter_ckpt_path)
            sys.exit(1)

        log.info("Loading adapter checkpoint: %s", adapter_ckpt_path)
        ckpt = torch.load(adapter_ckpt_path, map_location=device, weights_only=True)
        state_dict = ckpt.get("model_state_dict", ckpt)

        # Infer config from checkpoint or use defaults
        text_config = TextToCodecConfig()
        adapter = TextToCodecModel(text_config).to(device)
        adapter.load_state_dict(state_dict)
        adapter.eval()
        log.info(
            "Adapter loaded: %.2fM params",
            adapter.parameter_count() / 1e6,
        )

        # Export adapter ONNX
        adapter_onnx_path = output_dir / "adapter.onnx"
        _export_adapter(adapter, adapter_onnx_path, verbose=args.verbose)

    # ── Build .voice pack ──
    log.info("Building .voice pack...")
    voice_path = _build_voice_pack(
        codec,
        adapter or TextToCodecModel(TextToCodecConfig()).to("cpu"),
        output_dir,
        args.voice_name,
    )
    log.info("Voice pack created: %s", voice_path)

    # Summary
    print()
    print(f"[export_onnx] {'=' * 48}")
    print(f"[export_onnx] Output directory: {output_dir}")
    print(f"[export_onnx]   - codec_decoder.onnx")
    if adapter:
        print(f"[export_onnx]   - adapter.onnx")
    print(f"[export_onnx]   - {voice_path.name}")
    print(f"[export_onnx] {'=' * 48}")


if __name__ == "__main__":
    main()

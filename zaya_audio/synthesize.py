#! /usr/bin/env python3
"""Voice synthesis CLI and API for zaya_audio.

Supports two execution paths:

1. **PyTorch path** (default, debugging) — loads the codec decoder and
   adapter from a ``.voice`` pack and runs inference with PyTorch.
2. **ONNX path** (fast, production) — runs the codec decoder and adapter
   through ONNX Runtime for maximum performance.

Usage
-----
.. code-block:: bash

    # CLI
    python -m zaya_audio.synthesize \\
        --voice models/onnx/my_voice.voice \\
        --text "Hello, this is my cloned voice."

    # Python API
    from zaya_audio.synthesize import VoiceSynthesizer

    synth = VoiceSynthesizer("my_voice.voice")
    audio = synth.synthesize("Hello, this is my cloned voice.")
    save_audio(audio, "output.wav")
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .text_to_codec_model import (
    TextToCodecConfig,
    TextToCodecModel,
    tokenize_text,
    tokenize_text_batch,
)
from .utils import save_audio
from .voice_pack import load_voice_pack

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("synthesize")


# ═══════════════════════════════════════════════════════════════════
# VoiceSynthConfig
# ═══════════════════════════════════════════════════════════════════


@dataclass
class VoiceSynthConfig:
    """Configuration for the voice synthesizer.

    Attributes
    ----------
    sample_rate : int
        Audio sample rate (from voice pack).
    n_codebooks : int
        Number of RVQ codebooks (from voice pack).
    codebook_size : int
        Entries per codebook (from voice pack).
    code_dim : int
        Codebook embedding dimension (from voice pack).
    latent_dim : int
        Latent dimension (from voice pack).
    speaker_dim : int
        Speaker embedding dimension (from voice pack).
    total_downsample_factor : int
        Total time-dim reduction of the codec encoder.
    max_seq_len : int
        Maximum audio segment length in samples.
    text_max_len : int
        Maximum text token sequence length.
    use_onnx : bool
        If ``True``, use ONNX Runtime for inference.
    onnx_decoder_path : str, optional
        Path to the ONNX codec decoder model.
    onnx_adapter_path : str, optional
        Path to the ONNX adapter model.
    """

    sample_rate: int = 24000
    n_codebooks: int = 8
    codebook_size: int = 1024
    code_dim: int = 64
    latent_dim: int = 256
    speaker_dim: int = 512
    total_downsample_factor: int = 1280
    max_seq_len: int = 72000
    text_max_len: int = 512
    use_onnx: bool = False
    onnx_decoder_path: Optional[str] = None
    onnx_adapter_path: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════
# VoiceSynthesizer
# ═══════════════════════════════════════════════════════════════════


class VoiceSynthesizer:
    """End-to-end text-to-speech synthesizer for zaya_audio.

    Loads a voice pack (codec decoder + speaker embedding + adapter)
    and provides a ``synthesize(text)`` method that returns audio.

    Parameters
    ----------
    voice_path : str or Path
        Path to a ``.voice`` voice pack.
    use_onnx : bool
        If ``True``, prefer ONNX Runtime for inference.
    device : str, optional
        Device for PyTorch inference (e.g. ``"cpu"``, ``"cuda"``).
    """

    def __init__(
        self,
        voice_path: Union[str, Path],
        use_onnx: bool = False,
        device: Optional[str] = None,
    ):
        self.voice_path = Path(voice_path).expanduser().resolve()
        if not self.voice_path.exists():
            raise FileNotFoundError(f"Voice pack not found: {self.voice_path}")

        log.info("Loading voice pack: %s", self.voice_path)
        self.pack = load_voice_pack(str(self.voice_path))

        # Extract config
        config_dict = self.pack.get("config", {})
        self.config = VoiceSynthConfig(
            sample_rate=config_dict.get("sample_rate", 24000),
            n_codebooks=config_dict.get("n_codebooks", 8),
            codebook_size=config_dict.get("codebook_size", 1024),
            code_dim=config_dict.get("code_dim", 64),
            latent_dim=config_dict.get("latent_dim", 256),
            speaker_dim=config_dict.get("speaker_dim", 512),
            total_downsample_factor=config_dict.get(
                "total_downsample_factor", 1280
            ),
            max_seq_len=config_dict.get("max_seq_len", 72000),
            use_onnx=use_onnx,
        )

        # Determine device
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Load components
        self.speaker_emb = self.pack["speaker_embedding"].to(self.device)
        if self.speaker_emb.dim() == 1:
            self.speaker_emb = self.speaker_emb.unsqueeze(0)

        self._load_decoder()
        self._load_adapter()

        log.info(
            "VoiceSynthesizer ready — "
            "decoder: %s, adapter: %s, device: %s, onnx: %s",
            "loaded" if self.decoder is not None else "N/A",
            "loaded" if self.adapter is not None else "N/A (using direct codec tokens)",
            self.device,
            use_onnx,
        )

    def _load_decoder(self) -> None:
        """Load the codec decoder from the voice pack."""
        self.decoder = None
        self._decoder_onnx_sess = None
        decoder_state = self.pack.get("decoder")
        if decoder_state is None:
            log.warning("No decoder found in voice pack")
            return

        if self.config.use_onnx:
            # Try to load ONNX model from alongside the voice pack
            onnx_path = self.voice_path.parent / "codec_decoder.onnx"
            if onnx_path.exists():
                self.config.onnx_decoder_path = str(onnx_path)
                try:
                    import onnxruntime as ort

                    self._decoder_onnx_sess = ort.InferenceSession(
                        str(onnx_path),
                        providers=["CPUExecutionProvider"],
                    )
                    log.info("Loaded ONNX decoder: %s", onnx_path)
                    return
                except Exception as e:
                    log.warning(
                        "Failed to load ONNX decoder, falling back to PyTorch: %s", e
                    )
            else:
                log.info("ONNX decoder not found at %s, using PyTorch", onnx_path)

        # PyTorch path: build a lightweight decoder from the state dict
        self.decoder = _build_decoder_from_state(decoder_state, self.config).to(
            self.device
        )
        self.decoder.eval()

    def _load_adapter(self) -> None:
        """Load the text→codec adapter from the voice pack."""
        self.adapter = None
        self._adapter_onnx_sess = None
        adapter_state = self.pack.get("adapter")
        if adapter_state is None:
            log.info("No adapter found — synthesis requires direct codec tokens")
            return

        if self.config.use_onnx:
            onnx_path = self.voice_path.parent / "adapter.onnx"
            if onnx_path.exists():
                self.config.onnx_adapter_path = str(onnx_path)
                try:
                    import onnxruntime as ort

                    self._adapter_onnx_sess = ort.InferenceSession(
                        str(onnx_path),
                        providers=["CPUExecutionProvider"],
                    )
                    log.info("Loaded ONNX adapter: %s", onnx_path)
                    return
                except Exception as e:
                    log.warning(
                        "Failed to load ONNX adapter, falling back to PyTorch: %s", e
                    )
            else:
                log.info("ONNX adapter not found at %s, using PyTorch", onnx_path)

        # PyTorch path
        text_config = TextToCodecConfig()
        self.adapter = TextToCodecModel(text_config)
        self.adapter.load_state_dict(adapter_state)
        self.adapter = self.adapter.to(self.device)
        self.adapter.eval()

    # ─── Public API ─────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        return_tokens: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Synthesize audio from text.

        Parameters
        ----------
        text : str
            Input text to speak.
        return_tokens : bool
            If ``True``, also return the generated codec tokens.

        Returns
        -------
        audio : Tensor
            ``(1, T_audio)`` mono waveform, values in ``[-1, 1]``.
        codec_tokens : Tensor, optional
            Only if ``return_tokens=True``.  ``(n_codebooks, T_latent)``
            codec token indices.
        """
        # ── Tokenize text ──
        text_tokens, text_lengths = tokenize_text_batch(
            [text], max_len=self.config.text_max_len
        )  # (1, L), (1,)
        text_tokens = text_tokens.to(self.device)
        text_lengths = text_lengths.to(self.device)

        # ── Generate codec tokens ──
        if self.adapter is not None:
            codec_indices = self._run_adapter(text_tokens, text_lengths)
            # codec_indices: (1, L', n_codebooks)
        else:
            log.error("No adapter loaded — cannot generate codec tokens from text")
            raise RuntimeError("Adapter not loaded — voice pack missing adapter weights")

        # ── Decode to audio ──
        audio = self._decode_to_audio(codec_indices)  # (1, T_audio)

        # ── Reshape codec tokens to (n_codebooks, T_latent) ──
        tokens_out = codec_indices.squeeze(0).permute(1, 0)  # (n_books, L')

        if return_tokens:
            return audio, tokens_out
        return audio

    def synthesize_batch(
        self, texts: list
    ) -> list:
        """Synthesize audio for multiple texts sequentially.

        Parameters
        ----------
        texts : list of str
            Input texts.

        Returns
        -------
        audios : list of Tensor
            Each ``(1, T_audio)`` mono waveform.
        """
        return [self.synthesize(t) for t in texts]

    # ─── Internal: adapter inference ────────────────────────────

    @torch.no_grad()
    def _run_adapter(
        self, text_tokens: torch.Tensor, text_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Run text→codec adapter inference.

        Returns codec indices as ``(1, L', n_codebooks)``.
        """
        if self._adapter_onnx_sess is not None:
            return self._run_adapter_onnx(text_tokens, text_lengths)
        return self.adapter.generate(text_tokens, text_lengths=text_lengths)

    def _run_adapter_onnx(
        self, text_tokens: torch.Tensor, text_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Run adapter via ONNX Runtime."""
        onnx_inputs = {
            "text_tokens": text_tokens.cpu().numpy().astype(np.int64),
            "text_length": text_lengths.cpu().numpy().astype(np.int64),
        }
        logits = self._adapter_onnx_sess.run(None, onnx_inputs)[0]
        logits_tensor = torch.from_numpy(logits)
        # Argmax over codebook dimension
        return logits_tensor.argmax(dim=-1, keepdim=False)  # (1, L', n_books)

    # ─── Internal: decoder inference ────────────────────────────

    @torch.no_grad()
    def _decode_to_audio(self, codec_indices: torch.Tensor) -> torch.Tensor:
        """Decode codec indices to audio waveform.

        Parameters
        ----------
        codec_indices : Tensor
            ``(B, L', n_codebooks)`` — dtype long, values in [0, codebook_size).

        Returns
        -------
        audio : Tensor
            ``(B, 1, T_audio)``.
        """
        if self._decoder_onnx_sess is not None:
            return self._decode_to_audio_onnx(codec_indices)

        if self.decoder is None:
            raise RuntimeError("Codec decoder not loaded")

        return self.decoder(codec_indices, self.speaker_emb)

    def _decode_to_audio_onnx(self, codec_indices: torch.Tensor) -> torch.Tensor:
        """Decode to audio via ONNX Runtime.

        The ONNX decoder expects ``codec_tokens`` of shape
        ``(n_codebooks, T_latent)``.
        """
        # Transpose: (B, T, n_books) -> (n_books, T)
        tokens = codec_indices.squeeze(0).permute(1, 0).cpu().numpy().astype(np.int64)
        emb = self.speaker_emb.cpu().numpy().astype(np.float32)

        onnx_inputs = {
            "codec_tokens": tokens,
            "speaker_emb": emb,
        }
        audio = self._decoder_onnx_sess.run(None, onnx_inputs)[0]
        return torch.from_numpy(audio)


# ═══════════════════════════════════════════════════════════════════
# Decoder builder (PyTorch)
# ═══════════════════════════════════════════════════════════════════


class _DecoderONNXStyle(nn.Module):
    """Lightweight codec decoder that takes indices + speaker emb → audio.

    Matches the ONNX decoder's interface for the PyTorch fallback path.
    Loaded from voice pack decoder weights.
    """

    def __init__(self, config: VoiceSynthConfig):
        super().__init__()
        self.n_codebooks = config.n_codebooks
        self.codebook_size = config.codebook_size
        self.code_dim = config.code_dim
        self.latent_dim = config.latent_dim

        # Will be populated from state dict
        self.register_buffer("codebook_embed", torch.zeros(
            config.n_codebooks, config.codebook_size, config.code_dim
        ))
        self.post_vq = nn.Conv1d(config.code_dim, config.latent_dim, kernel_size=1)
        self.dec_proj = nn.Conv1d(config.latent_dim, config.latent_dim, kernel_size=3, padding=1)
        self.decoder = nn.ModuleList()  # DecoderBlocks loaded from state
        self.post_conv = nn.Conv1d(config.latent_dim, 1, kernel_size=7, padding=3)

    def forward(
        self,
        codec_indices: torch.Tensor,
        speaker_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Decode codec indices + speaker emb to audio.

        Parameters
        ----------
        codec_indices : Tensor
            ``(B, T_latent, n_codebooks)`` — long.
        speaker_emb : Tensor
            ``(B, speaker_dim)``.

        Returns
        -------
        audio : Tensor
            ``(B, 1, T_audio)``.
        """
        B, T, n_cb = codec_indices.shape
        device = codec_indices.device
        dtype = self.dec_proj.weight.dtype

        # Lookup and sum codebook embeddings
        z_q = torch.zeros(B, self.code_dim, T, device=device, dtype=dtype)
        for cb in range(self.n_codebooks):
            embed = self.codebook_embed[cb]
            idx = codec_indices[:, :, cb].clamp(min=0, max=self.codebook_size - 1)
            z_q_cb = F.embedding(idx, embed)
            z_q = z_q + z_q_cb.permute(0, 2, 1).to(dtype)

        # Decode
        z = self.post_vq(z_q)
        x = self.dec_proj(z)
        for block in self.decoder:
            x = block(x, speaker_emb)
        audio = self.post_conv(x)
        return audio


def _build_decoder_from_state(
    state_dict: Dict[str, torch.Tensor],
    config: VoiceSynthConfig,
) -> nn.Module:
    """Build a lightweight decoder from a voice pack state dict.

    The state dict contains all tensors needed: codebook embeddings,
    post_vq weights, decoder block weights, post_conv weights.

    Parameters
    ----------
    state_dict : dict
        Decoder weights from the voice pack.
    config : VoiceSynthConfig
        Configuration.

    Returns
    -------
    decoder : _DecoderONNXStyle
        Initialised decoder module.
    """
    from .codec import RVQVAE
    from .config import DEFAULT_CONFIG

    # Build a full RVQVAE and load state to get the decoder architecture
    codec_config = DEFAULT_CONFIG
    codec = RVQVAE(codec_config)
    codec.load_state_dict(state_dict, strict=False)

    # Create lightweight decoder
    decoder = _DecoderONNXStyle(config)

    # Copy codebook embeddings
    decoder.codebook_embed = codec.res_vq.embed.clone()

    # Copy post_vq
    decoder.post_vq.weight.data = codec.post_vq.weight.data.clone()
    decoder.post_vq.bias.data = codec.post_vq.bias.data.clone()

    # Copy dec_proj
    decoder.dec_proj.weight.data = codec.dec_proj.weight.data.clone()
    if codec.dec_proj.bias is not None:
        decoder.dec_proj.bias.data = codec.dec_proj.bias.data.clone()

    # Copy decoder blocks
    decoder.decoder = codec.decoder

    # Copy post_conv
    decoder.post_conv.weight.data = codec.post_conv.weight.data.clone()
    if codec.post_conv.bias is not None:
        decoder.post_conv.bias.data = codec.post_conv.bias.data.clone()

    return decoder


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zaya Audio — Text-to-speech synthesis CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--voice", type=str, required=True,
        help="Path to .voice pack",
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="Text to synthesize (omit for interactive mode)",
    )
    parser.add_argument(
        "--output", type=str, default="output.wav",
        help="Output audio file path",
    )
    parser.add_argument(
        "--onnx", action="store_true",
        help="Use ONNX Runtime for inference",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for PyTorch inference (e.g. 'cpu', 'cuda')",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print voice pack info and exit",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    voice_path = Path(args.voice).expanduser().resolve()
    if not voice_path.exists():
        print(f"Error: Voice pack not found: {voice_path}", file=sys.stderr)
        sys.exit(1)

    # ── Print info mode ──
    if args.list:
        from .voice_pack import get_voice_pack_info

        info = get_voice_pack_info(str(voice_path))
        print(json.dumps({
            "name": info.name,
            "speaker_name": info.speaker_name,
            "language": info.language,
            "sample_rate": info.sample_rate,
            "model_version": info.model_version,
            "codec_params": info.codec_params,
        }, indent=2))
        return

    # ── Build synthesizer ──
    try:
        synth = VoiceSynthesizer(
            voice_path,
            use_onnx=args.onnx,
            device=args.device,
        )
    except Exception as e:
        print(f"Error loading voice pack: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Interactive mode ──
    if args.text is None:
        print("Zaya Audio — Interactive TTS mode (Ctrl+D to exit)")
        print(f"Voice: {voice_path.name}")
        print()
        try:
            while True:
                text = input("Text > ").strip()
                if not text:
                    continue

                print("  Synthesizing... ", end="", flush=True)
                start = time.time()
                audio = synth.synthesize(text)
                elapsed = time.time() - start
                audio_len = audio.shape[-1] / synth.config.sample_rate
                print(
                    f"done ({elapsed:.2f}s, {audio_len:.1f}s audio, "
                    f"RTF={elapsed / max(audio_len, 0.01):.2f}x)"
                )
        except EOFError:
            print()
        return

    # ── Single text mode ──
    log.info("Synthesizing: %s", args.text)
    start = time.time()
    audio = synth.synthesize(args.text)
    elapsed = time.time() - start

    # Save
    output_path = Path(args.output).expanduser().resolve()
    save_audio(audio, str(output_path), sr=synth.config.sample_rate)

    audio_len = audio.shape[-1] / synth.config.sample_rate
    log.info(
        "Generated %.1fs audio in %.2fs (RTF=%.2f) → %s",
        audio_len,
        elapsed,
        elapsed / max(audio_len, 0.01),
        output_path,
    )


if __name__ == "__main__":
    main()

#! /usr/bin/env python3
""".voice pack format — portable voice snapshot for the Zaya Co-Host platform.

A ``.voice`` pack is a zip archive containing the decoder weights, speaker
embedding, optional LoRA adapter weights, codec configuration, and metadata.
It captures a complete voice identity that can be loaded at inference time
without access to the original training data or full codec model.

The module is intentionally **importable without PyTorch** (``torch`` is
only imported inside functions that need it), so callers can read metadata
with a lightweight dependency footprint.

Structure
---------
A ``.voice`` archive contains::

    decoder.pt         — Codec decoder ``state_dict`` (~22 MB)
    speaker_emb.pt     — 512-dim float32 speaker embedding (~2 KB)
    adapter.pt         — Optional LoRA adapter weights (~3 MB)
    config.json        — Codec hyperparameters (``AudioCodecConfig``)
    metadata.json      — ``VoicePackMetadata``

Usage
-----
.. code-block:: python

    # Save
    save_voice_pack("my_voice.voice", decoder_state, speaker_emb, metadata=meta)

    # Load
    pack = load_voice_pack("my_voice.voice")
    x_hat = decoder(pack["decoder"], pack["speaker_embedding"])

    # Quick info (no torch required)
    info = get_voice_pack_info("my_voice.voice")
    print(info.name, info.speaker_name, info.mos_score)
"""

import json
import os
import tempfile
import zipfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

# ─── Metadata ────────────────────────────────────────────────────────────────


@dataclass
class VoicePackMetadata:
    """Metadata for a ``.voice`` voice pack.

    Attributes
    ----------
    name : str
        Human-readable name for this voice pack (e.g. "My Voice v2").
    speaker_name : str
        Name of the speaker whose voice was cloned.
    language : str
        ISO 639-1 language code (default ``"en"``).
    sample_rate : int
        Expected audio sample rate in Hz (default ``24000``).
    created_at : str
        ISO 8601 timestamp of creation.  Filled automatically if empty.
    model_version : str
        Version of the zaya_audio model used to create this pack
        (default ``"0.1.0"``).
    codec_params : dict
        Codec hyperparameters stored as a flat dict (``n_codebooks``,
        ``codebook_size``, ``code_dim``, ``latent_dim``, etc.).
    mos_score : float
        Mean Opinion Score (0.0–5.0) if evaluated.  Default ``0.0``.
    """

    name: str
    speaker_name: str
    language: str = "en"
    sample_rate: int = 24000
    created_at: str = ""
    model_version: str = "0.1.0"
    codec_params: dict = field(default_factory=dict)
    mos_score: float = 0.0


# ─── Save ────────────────────────────────────────────────────────────────────


def save_voice_pack(
    path: str,
    decoder_state: "torch.Tensor",
    speaker_embedding: "torch.Tensor",
    adapter_state: Optional[Dict[str, "torch.Tensor"]] = None,
    metadata: Optional[VoicePackMetadata] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a ``.voice`` voice pack as a zip archive.

    Parameters
    ----------
    path : str
        Output file path (``.voice`` extension recommended).
    decoder_state : dict
        ``state_dict`` of the codec decoder (``RVQVAE.decoder`` +
        ``post_conv``).  Must be serialisable via ``torch.save``.
    speaker_embedding : Tensor
        512-dim float32 speaker embedding vector, shape ``(1, 512)``
        or ``(512,)``.
    adapter_state : dict, optional
        LoRA adapter ``state_dict`` (e.g. from ``peft`` or a hand-rolled
        adapter module).
    metadata : VoicePackMetadata, optional
        Voice pack metadata.  Fields like ``created_at`` and ``sample_rate``
        are auto-filled from context if omitted.
    config : dict, optional
        Codec hyperparameter dict.  If omitted, a minimal config is built
        from decoder weight shapes.
    """
    import torch

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    # ── Normalise speaker embedding ──
    if speaker_embedding.dim() == 1:
        speaker_embedding = speaker_embedding.unsqueeze(0)  # (1, D)

    # ── Default metadata ──
    if metadata is None:
        metadata = VoicePackMetadata(
            name=path.stem,
            speaker_name="unknown",
        )
    if not metadata.created_at:
        metadata.created_at = datetime.now(timezone.utc).isoformat()
    metadata.sample_rate = metadata.sample_rate or 24000

    # ── Default config from decoder state ──
    if config is None:
        config = _infer_config(decoder_state)
    metadata.codec_params = config

    # ── Build archive ──
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            # Decoder weights
            decoder_bytes = _serialize_torch(decoder_state)
            zf.writestr("decoder.pt", decoder_bytes)

            # Speaker embedding
            emb_bytes = _serialize_torch(speaker_embedding)
            zf.writestr("speaker_emb.pt", emb_bytes)

            # Adapter weights (optional)
            if adapter_state is not None:
                adapter_bytes = _serialize_torch(adapter_state)
                zf.writestr("adapter.pt", adapter_bytes)

            # Config
            zf.writestr("config.json", json.dumps(config, indent=2, default=str))

            # Metadata
            zf.writestr(
                "metadata.json",
                json.dumps(asdict(metadata), indent=2, default=str),
            )

        # Atomically move temp file to destination
        os.replace(tmp.name, path)
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"[voice_pack] Saved: {path} ({size_mb:.1f} MB)")


# ─── Load ────────────────────────────────────────────────────────────────────


def load_voice_pack(path: str) -> Dict[str, Any]:
    """Load a ``.voice`` voice pack from a zip archive.

    Returns a dictionary with keys:
        - ``decoder`` — decoder ``state_dict`` (``OrderedDict``)
        - ``speaker_embedding`` — speaker embedding tensor ``(1, D)``
        - ``adapter`` — LoRA adapter ``state_dict`` (optional)
        - ``config`` — codec hyperparameter dict
        - ``metadata`` — ``VoicePackMetadata``

    Parameters
    ----------
    path : str
        Path to the ``.voice`` zip archive.
    """
    import torch

    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Voice pack not found: {path}")

    result: Dict[str, Any] = {}

    with zipfile.ZipFile(str(path), "r") as zf:
        # Validate required members
        required = ["decoder.pt", "speaker_emb.pt", "config.json", "metadata.json"]
        for member in required:
            if member not in zf.namelist():
                raise ValueError(
                    f"Invalid voice pack: missing required member '{member}' "
                    f"in {path}"
                )

        # Load tensors / state dicts
        result["decoder"] = torch.load(
            zf.extract("decoder.pt"), map_location="cpu", weights_only=True
        )
        result["speaker_embedding"] = torch.load(
            zf.extract("speaker_emb.pt"), map_location="cpu", weights_only=True
        )

        # Adapter (optional)
        if "adapter.pt" in zf.namelist():
            result["adapter"] = torch.load(
                zf.extract("adapter.pt"), map_location="cpu", weights_only=True
            )

        # Config
        result["config"] = json.loads(zf.read("config.json"))

        # Metadata
        result["metadata"] = VoicePackMetadata(
            **json.loads(zf.read("metadata.json"))
        )

    return result


# ─── Quick info (no torch) ───────────────────────────────────────────────────


def get_voice_pack_info(path: str) -> VoicePackMetadata:
    """Read metadata from a ``.voice`` pack without loading model weights.

    This function only reads ``metadata.json`` and ``config.json`` from
    the zip archive.  It does **not** import PyTorch, so it's safe for
    lightweight inspection.

    Parameters
    ----------
    path : str
        Path to the ``.voice`` zip archive.

    Returns
    -------
    VoicePackMetadata
        Voice pack metadata (name, speaker, language, sample_rate, etc.).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the archive is missing required members.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Voice pack not found: {path}")

    with zipfile.ZipFile(str(path), "r") as zf:
        if "metadata.json" not in zf.namelist():
            raise ValueError(f"Missing 'metadata.json' in voice pack: {path}")

        metadata_dict = json.loads(zf.read("metadata.json"))

        # Fill codec_params from config.json if present
        if "config.json" in zf.namelist():
            config_dict = json.loads(zf.read("config.json"))
            metadata_dict.setdefault("codec_params", config_dict)

        return VoicePackMetadata(**metadata_dict)


# ─── Internals ───────────────────────────────────────────────────────────────


def _infer_config(decoder_state: Dict[str, "torch.Tensor"]) -> Dict[str, Any]:
    """Infer codec hyperparameters from decoder weight shapes."""
    import torch

    config: Dict[str, Any] = {}

    # Count decoder blocks (ModuleList entries: decoder.0, decoder.1, ...)
    n_blocks = sum(
        1 for k in decoder_state if k.startswith("decoder.") and k.endswith(".film.proj.weight")
    )
    config["n_decoder_blocks"] = n_blocks

    # Speaker dimension from FiLM projection weight shape
    film_key = next(
        (k for k in decoder_state if k.endswith(".film.proj.weight")), None
    )
    if film_key is not None:
        config["speaker_dim"] = decoder_state[film_key].shape[1]

    # Latent dimension from dec_proj weight
    if "dec_proj.weight" in decoder_state:
        config["latent_dim"] = decoder_state["dec_proj.weight"].shape[1]

    # Sample rate (default if not inferable)
    config.setdefault("sample_rate", 24000)

    return config


def _serialize_torch(obj: Any) -> bytes:
    """Serialize an object with ``torch.save`` and return bytes."""
    import torch
    import io

    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


# ─── CLI quick info ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path.voice>", file=sys.stderr)
        sys.exit(1)

    info = get_voice_pack_info(sys.argv[1])
    print(json.dumps(asdict(info), indent=2, default=str))

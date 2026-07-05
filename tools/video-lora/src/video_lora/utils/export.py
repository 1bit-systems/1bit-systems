"""Audio export utilities."""

from pathlib import Path
from typing import Union

import numpy as np
import torch


def export_to_wav(
    audio: Union[torch.Tensor, np.ndarray],
    path: Union[str, Path],
    sample_rate: int = 44100,
) -> Path:
    """Export audio tensor to WAV file.

    Args:
        audio: Audio tensor of shape ``(channels, samples)`` or ``(samples,)``.
        path: Output file path.
        sample_rate: Sample rate in Hz.

    Returns:
        Path to the saved WAV file.
    """
    import soundfile as sf

    path = Path(path)
    if path.suffix == "":
        path = path.with_suffix(".wav")

    # Convert to numpy if tensor
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().float().numpy()

    # Ensure 2D (channels, samples)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]

    # Transpose to (samples, channels) for soundfile
    audio = audio.T

    # Normalize to [-1, 1] if needed
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val

    sf.write(str(path), audio, sample_rate)
    return path

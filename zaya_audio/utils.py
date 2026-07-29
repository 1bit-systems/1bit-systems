"""Utility functions for zaya_audio: audio I/O, mel spectrograms,
parameter counting, and dynamic-range compression/decompression.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torchaudio
import torchaudio.functional as Faudio


# ═══════════════════════════════════════════════════════════════════
# Parameter counting
# ═══════════════════════════════════════════════════════════════════

def count_params(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count parameters in a PyTorch model.

    Parameters
    ----------
    model : nn.Module
        The model to count.
    trainable_only : bool, default=True
        If ``True``, only count parameters with ``requires_grad=True``.

    Returns
    -------
    int
        Total parameter count.

    Example
    -------
    >>> from zaya_audio.codec import RVQVAE
    >>> from zaya_audio.config import DEFAULT_CONFIG
    >>> model = RVQVAE(DEFAULT_CONFIG)
    >>> count_params(model)
    5866437
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# ═══════════════════════════════════════════════════════════════════
# Audio I/O
# ═══════════════════════════════════════════════════════════════════

def load_audio(path: Union[str, Path], sr: int = 24_000) -> torch.Tensor:
    """Load and resample audio to the target sample rate.

    Automatically mixes multi-channel audio to mono.

    Parameters
    ----------
    path : str or Path
        Path to an audio file (any format supported by ``torchaudio``).
    sr : int
        Target sample rate in Hz.

    Returns
    -------
    waveform : Tensor
        Mono waveform, shape ``(1, T)``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    RuntimeError
        If decoding fails.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, orig_sr = torchaudio.load(str(path))

    # Mix to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if orig_sr != sr:
        waveform = Faudio.resample(waveform, orig_freq=orig_sr, new_freq=sr)

    return waveform


def save_audio(tensor: torch.Tensor, path: Union[str, Path],
               sr: int = 24_000) -> None:
    """Save a waveform tensor to a WAV file.

    Parameters
    ----------
    tensor : Tensor
        Waveform, shape ``(1, T)``.  Values should be in ``[-1, 1]``.
    path : str or Path
        Output file path (``.wav`` extension recommended).
    sr : int
        Sample rate to embed in the file header.

    Raises
    ------
    ValueError
        If the tensor is not 2D ``(1, T)``.
    """
    if tensor.dim() != 2 or tensor.shape[0] != 1:
        raise ValueError(
            f"Expected shape (1, T), got {tuple(tensor.shape)}"
        )
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), tensor.clamp(-1.0, 1.0).cpu(), sample_rate=sr)


# ═══════════════════════════════════════════════════════════════════
# Mel spectrogram
# ═══════════════════════════════════════════════════════════════════

def compute_melspec(
    audio: torch.Tensor,
    n_fft: int = 1024,
    n_mels: int = 80,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
    sr: int = 24_000,
    power: float = 2.0,
    center: bool = True,
    pad_mode: str = "reflect",
) -> torch.Tensor:
    """Compute mel spectrogram from a waveform tensor.

    Parameters
    ----------
    audio : Tensor
        Waveform, ``(B, 1, T)`` or ``(1, T)``.
    n_fft : int
        FFT size.
    n_mels : int
        Number of mel bands.
    hop_length : int, optional
        Hop length (default: ``n_fft // 4``).
    win_length : int, optional
        Window length (default: ``n_fft``).
    f_min : float
        Minimum frequency (Hz).
    f_max : float, optional
        Maximum frequency (Hz, default: ``sr / 2``).
    sr : int
        Sample rate.
    power : float
        Exponent of the magnitude spectrogram (2.0 = power spectrum).
    center : bool
        Centre-pad the waveform.
    pad_mode : str
        Padding mode.

    Returns
    -------
    melspec : Tensor
        Mel spectrogram, ``(B, n_mels, T_frames)``.
    """
    if audio.dim() == 2:
        audio = audio.unsqueeze(0)  # (1, 1, T)

    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    # Compute STFT
    stft = torch.stft(
        audio.squeeze(1),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length, device=audio.device),
        center=center,
        pad_mode=pad_mode,
        return_complex=True,
    )  # (B, F, T_frames)

    # Magnitude spectrogram
    mag = stft.abs().pow(power)  # (B, F, T_frames)

    # Mel filterbank
    mel_basis = _mel_filterbank(
        sr, n_fft, n_mels, f_min, f_max or (sr / 2)
    ).to(audio.device)  # (n_mels, F)

    melspec = mel_basis @ mag  # (B, n_mels, T_frames)
    return melspec


def _mel_filterbank(
    sr: int, n_fft: int, n_mels: int,
    f_min: float, f_max: float,
) -> torch.Tensor:
    """Build a mel filterbank matrix (PyTorch-native, no librosa)."""
    # Mel-scale frequencies for each filter centre
    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    # FFT bin frequencies
    fft_bins = torch.linspace(0, sr / 2, n_fft // 2 + 1)

    # Triangular filters
    fb = torch.zeros(n_mels, n_fft // 2 + 1)
    for i in range(n_mels):
        left = hz_points[i]
        centre = hz_points[i + 1]
        right = hz_points[i + 2]
        # Rising edge
        idx_rise = (fft_bins >= left) & (fft_bins <= centre)
        fb[i, idx_rise] = (fft_bins[idx_rise] - left) / (centre - left)
        # Falling edge
        idx_fall = (fft_bins >= centre) & (fft_bins <= right)
        fb[i, idx_fall] = (right - fft_bins[idx_fall]) / (right - centre)

    # Normalise each filter to unit area
    fb = fb / fb.sum(dim=1, keepdim=True).clamp(min=1e-10)
    return fb


def _hz_to_mel(hz: float) -> float:
    """Convert Hz to mel (HTK formula)."""
    return 2595.0 * (torch.as_tensor(1.0 + hz / 700.0).log10().item())


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    """Convert mel to Hz (HTK formula)."""
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


# ═══════════════════════════════════════════════════════════════════
# Dynamic range compression / decompression
# ═══════════════════════════════════════════════════════════════════

def dynamic_range_compression(
    x: torch.Tensor, C: float = 1.0, clip_val: float = 1e-5
) -> torch.Tensor:
    """Logarithmic dynamic range compression (log ||mel-spectrogram||).

    ``y = log(max(C * x, clip_val))``

    This is commonly applied before feeding mel spectrograms into
    neural vocoders (WaveGlow, HiFi-GAN).

    Parameters
    ----------
    x : Tensor
        Magnitude or mel-spectrogram, arbitrary shape.
    C : float
        Scaling factor (typically 1.0).
    clip_val : float
        Minimum value before log to avoid ``log(0)``.

    Returns
    -------
    Tensor
        Compressed representation, same shape as ``x``.
    """
    return torch.log(torch.clamp(C * x, min=clip_val))


def dynamic_range_decompression(
    x: torch.Tensor, C: float = 1.0
) -> torch.Tensor:
    """Inverse of ``dynamic_range_compression``.

    ``y = exp(x) / C``

    Parameters
    ----------
    x : Tensor
        Compressed representation.
    C : float
        Scaling factor (same value used during compression).

    Returns
    -------
    Tensor
        Linear-scale estimate, same shape as ``x``.
    """
    return torch.exp(x) / C


# ═══════════════════════════════════════════════════════════════════
# Convenience: print model summary
# ═══════════════════════════════════════════════════════════════════

def model_summary(model: torch.nn.Module) -> None:
    """Print a table of per-module parameter counts.

    Parameters
    ----------
    model : nn.Module
        The model to analyse.
    """
    total = 0
    print(f"{'Module':<40} {'Params':>10}")
    print("-" * 52)
    for name, param in model.named_parameters():
        if param.requires_grad:
            n = param.numel()
            total += n
            # Truncate long names for readability
            short_name = name if len(name) < 40 else name[:37] + "..."
            print(f"{short_name:<40} {n:>10,}")
    print("-" * 52)
    print(f"{'Total trainable':<40} {total:>10,}")

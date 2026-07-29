"""Audio dataset and augmentation for zaya_audio training.

Provides random-cropped waveform segments, optional on-the-fly
augmentation (pitch shift, volume jitter, additive noise), and
a PyTorch ``Dataset`` that walks a directory tree of WAV files.
"""

import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as Faudio
from torch.utils.data import Dataset

from .config import DEFAULT_CONFIG, AudioCodecConfig


class AudioDataset(Dataset):
    """Audio dataset that loads and crops WAV segments on the fly.

    Walks ``root_dir`` recursively and indexes every ``.wav`` file.
    On ``__getitem__``, loads the full file, resamples to the target
    sample rate, applies optional augmentation, and returns a random
    crop of length ``max_seq_len``.

    Parameters
    ----------
    root_dir : str or Path
        Directory tree containing ``.wav`` files.
    config : AudioCodecConfig, optional
        Hyperparameter config  (uses ``DEFAULT_CONFIG`` if omitted).
    augment : bool
        Enable on-the-fly augmentation  (pitch shift, volume jitter,
        additive noise).
    speaker_map : dict[str, int], optional
        Mapping from filename stem to speaker ID.  If given,
        ``__getitem__`` returns ``speaker_id`` as an integer.
        Otherwise ``speaker_id = -1``.
    extension : str
        File extension to scan for (default ``.wav``).
    """

    def __init__(
        self,
        root_dir: str,
        config: Optional[AudioCodecConfig] = None,
        augment: bool = False,
        speaker_map: Optional[Dict[str, int]] = None,
        extension: str = ".wav",
    ):
        super().__init__()
        self.config = config or DEFAULT_CONFIG
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.augment = augment
        self.speaker_map = speaker_map or {}

        # Index all WAV files
        self.files: List[Path] = sorted(self.root_dir.rglob(f"*{extension}"))
        if not self.files:
            raise FileNotFoundError(
                f"No {extension} files found under {self.root_dir}"
            )
        print(f"[AudioDataset] Found {len(self.files)} files under {self.root_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """Load a random segment from the *idx*-th audio file.

        Returns
        -------
        waveform : Tensor
            Mono waveform, shape ``(1, max_seq_len)``.  If the file
            is shorter than the segment length, it's zero-padded.
        speaker_id : int
            Speaker ID from ``speaker_map`` or ``-1`` if unknown.
        filename : str
            Absolute path to the source WAV file.
        """
        path = self.files[idx]
        filename = str(path)

        # Load audio (always mono, resample to target SR)
        waveform, sr = torchaudio.load(filename)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # mix to mono
        if sr != self.config.sample_rate:
            waveform = Faudio.resample(
                waveform, orig_freq=sr,
                new_freq=self.config.sample_rate
            )

        target_len = self.config.max_seq_len

        # Random crop or zero-pad
        if waveform.shape[-1] >= target_len:
            start = random.randint(0, waveform.shape[-1] - target_len)
            segment = waveform[:, start:start + target_len]
        else:
            # Pad to target length
            pad_len = target_len - waveform.shape[-1]
            segment = F.pad(waveform, (0, pad_len))

        # On-the-fly augmentation
        if self.augment:
            segment = self._apply_augmentation(segment)

        # Speaker ID
        stem = path.stem
        speaker_id = self.speaker_map.get(stem, -1)

        return segment, speaker_id, filename

    def _apply_augmentation(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random pitch shift, volume jitter, and additive noise.

        All augmentations are applied with 50 % probability each and
        configured via ``self.config.augment_*`` fields.

        Parameters
        ----------
        waveform : Tensor
            Mono waveform, ``(1, T)``.

        Returns
        -------
        Tensor
            Augmented waveform, ``(1, T)``.
        """
        sr = self.config.sample_rate

        # ── Pitch shift (±2 semitones) ────────────────────────────
        if random.random() < 0.5:
            n_semitones = random.uniform(
                -self.config.augment_pitch_shift_semitones,
                self.config.augment_pitch_shift_semitones,
            )
            # torchaudio's pitch shift uses F0 + resampling
            n_steps = int(round(n_semitones))
            waveform = Faudio.pitch_shift(
                waveform, sample_rate=sr, n_steps=n_steps
            )
            # Ensure same output length (pitch shift may change it slightly)
            if waveform.shape[-1] != self.config.max_seq_len:
                target_len = self.config.max_seq_len
                if waveform.shape[-1] > target_len:
                    waveform = waveform[:, :target_len]
                else:
                    pad_len = target_len - waveform.shape[-1]
                    waveform = F.pad(waveform, (0, pad_len))

        # ── Volume jitter (±3 dB) ─────────────────────────────────
        if random.random() < 0.5:
            db_change = random.uniform(
                -self.config.augment_volume_db,
                self.config.augment_volume_db,
            )
            gain = 10 ** (db_change / 20.0)
            waveform = waveform * gain

        # ── Additive noise ────────────────────────────────────────
        if random.random() < 0.5:
            # White noise at given SNR
            snr_db = self.config.augment_noise_floor_snr
            signal_power = waveform.pow(2).mean()
            noise_power = signal_power / (10 ** (snr_db / 10.0))
            noise = torch.randn_like(waveform) * math.sqrt(noise_power)
            waveform = waveform + noise

        # Prevent clipping
        waveform = torch.clamp(waveform, -1.0, 1.0)

        return waveform


# ═══════════════════════════════════════════════════════════════════
# Collation helper
# ═══════════════════════════════════════════════════════════════════

def collate_audio(
    batch: List[Tuple[torch.Tensor, int, str]],
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Collate function for ``DataLoader``.

    Parameters
    ----------
    batch : list of (waveform, speaker_id, filename)
        Output of ``AudioDataset.__getitem__``.

    Returns
    -------
    waveforms : Tensor
        Stacked waveforms, ``(B, 1, T)``.
    speaker_ids : Tensor
        Speaker IDs, ``(B,)``.
    filenames : list[str]
        Source filenames.
    """
    waveforms, speaker_ids, filenames = zip(*batch)
    return (
        torch.stack(waveforms, dim=0),       # (B, 1, T)
        torch.tensor(speaker_ids, dtype=torch.long),
        list(filenames),
    )

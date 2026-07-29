"""Hyperparameter dataclass for the zaya_audio voice cloning pipeline.

All core knobs live here — audio codec topology, VQ codebook sizing,
training loss weights, and segment/max-length constraints.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class AudioCodecConfig:
    """Immutable hyperparameter container for the RVQ-VAE codec.

    Attributes
    ----------
    sample_rate : int
        Target audio sample rate (Hz). 24 kHz is standard for
        speech codecs — enough bandwidth for voiced formants
        without wasting capacity on ultrasonic content.
    n_codebooks : int
        Number of Residual VQ codebooks stacked in series. Each
        subsequent codebook quantises the residual of the previous.
        8 codebooks × 1024 entries = 80 bits per frame (10 B/frame).
    codebook_size : int
        Entries per VQ codebook. 1024 = 10-bit codebook.
    code_dim : int
        Dimensionality of each codebook vector. Projected from
        ``latent_dim`` before quantisation and back after.
    speaker_dim : int
        Dimensionality of the speaker embedding vector that
        conditions the decoder via FiLM (Feature-wise Linear
        Modulation).
    latent_dim : int
        Dimensionality of the encoder output / decoder input
        latent space (before VQ projection).
    encoder_strides : list[int]
        Convolution stride for each of the 5 encoder blocks.
        Total downsampling factor = product = 1280×.
    decoder_strides : list[int]
        Convolution stride for each of the 5 decoder blocks.
        Reversed order of ``encoder_strides`` so the decoder
        mirrors the encoder.
    hidden_dim : int
        Base channel count in encoder/decoder blocks. Actual
        channels per block grow from ``hidden_dim`` toward
        4× ``hidden_dim`` as resolution decreases.
    n_res_blocks : int
        Number of residual blocks inside each encoder/decoder
        block (all share the same count for simplicity).
    target_params : int
        Design target for total trainable parameters. The model
        is verified against this at construction time via
        ``parameter_count()``.
    segment_duration_secs : float
        Duration of audio segments used for training (random
        crops). Shorter segments fit larger batches.
    max_seq_len : int
        Maximum number of audio samples per segment after
        resampling (= segment_duration_secs × sample_rate).
    """

    # --- Audio ---
    sample_rate: int = 24_000
    segment_duration_secs: float = 3.0
    max_seq_len: int = 72_000  # 3s × 24 kHz

    # --- Codec topology ---
    n_codebooks: int = 8
    codebook_size: int = 1024
    code_dim: int = 64

    # --- Embedding / latent spaces ---
    speaker_dim: int = 512
    latent_dim: int = 256

    # --- Encoder / decoder ---
    encoder_strides: Tuple[int, ...] = (4, 5, 4, 4, 4)
    decoder_strides: Tuple[int, ...] = (4, 4, 4, 5, 4)  # reversed
    hidden_dim: int = 64
    n_res_blocks: int = 4

    # --- Channel multiplier per block ---
    # Each EncoderBlock / DecoderBlock operates at a channel count
    # that scales with depth: [1×, 1×, 1.5×, 2×, 2×] relative to
    # hidden_dim.  This gives ~5.87M total params.
    channel_scales: Tuple[float, ...] = (1.0, 1.0, 1.5, 2.27, 3.0)

    # --- VQ training ---
    vq_commitment_beta: float = 0.25
    diversity_alpha: float = 0.1
    ema_decay: float = 0.99

    # --- STFT loss ---
    stft_fft_sizes: Tuple[int, ...] = (512, 1024, 2048)
    stft_hop_sizes: Tuple[int, ...] = (128, 256, 512)
    stft_win_lengths: Tuple[int, ...] = (512, 1024, 2048)

    # --- Parameter target (verification) ---
    target_params: int = 5_870_000

    # --- Dataset ---
    augment_pitch_shift_semitones: float = 2.0
    augment_volume_db: float = 3.0
    augment_noise_floor_snr: float = 40.0

    @property
    def total_downsample_factor(self) -> int:
        """Product of all encoder strides = total time-dim reduction."""
        factor = 1
        for s in self.encoder_strides:
            factor *= s
        return factor

    @property
    def latent_frames_per_segment(self) -> int:
        """Number of time frames in the latent space for one segment."""
        return self.max_seq_len // self.total_downsample_factor

    def channels_for_block(self, block_idx: int) -> int:
        """Channel count for the *block_idx*-th encoder/decoder block."""
        return int(self.hidden_dim * self.channel_scales[block_idx])


# Single global config instance — override fields before constructing
# the model if you need different topology.
DEFAULT_CONFIG = AudioCodecConfig()

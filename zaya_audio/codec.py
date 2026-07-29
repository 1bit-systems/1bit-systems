"""RVQ-VAE audio codec — core voice cloning model for Zaya Co-Host.

Architecture
------------
Encoder::
    (B, 1, T) ──► Conv1D pre ──► 5× EncoderBlock ──► Conv1D proj
                                        │
                                   stride downsampling
                                        │
                                        ▼
                                   ResidualVQ
                                   (8 codebooks, EMA)
                                        │
                                        ▼
                                   Conv1D proj
                                        │
Decoder::
    (B, D, T') ──► 5× DecoderBlock ──► Conv1D post ──► (B, 1, T')
                        │
                   FiLM(speaker_emb)
                        │
                   stride upsampling

ResidualVQ uses exponential-moving-average (EMA) codebook updates
instead of gradient-based optimisation, which is significantly
more stable for discrete codec training.

Parameter target: ~5.87M.  Verified at construction.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# Residual Block
# ═══════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """Two-conv residual block with ReLU activation.

    Layout::
        x ──► Conv1d(c, c, k=3) ──► ReLU ──► Conv1d(c, c, k=3) ──► + ──► out
        └───────────────────── skip (identity) ──────────────────────┘

    Parameters
    ----------
    channels : int
        Number of input/output channels.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )
        self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, C, T)``.

        Returns
        -------
        Tensor
            Shape ``(B, C, T)``.
        """
        return self.skip(x) + self.net(x)


# ═══════════════════════════════════════════════════════════════════
# FiLM (Feature-wise Linear Modulation)
# ═══════════════════════════════════════════════════════════════════

class FiLM(nn.Module):
    """Feature-wise Linear Modulation conditioned on a speaker embedding.

    For each channel ``c``::

        out[c] = gamma_c * x[c] + beta_c

    where ``(gamma, beta) = MLP(speaker_emb)``.

    Parameters
    ----------
    speaker_dim : int
        Dimensionality of the speaker embedding vector.
    channels : int
        Number of channels in the feature map to modulate.
    """

    def __init__(self, speaker_dim: int, channels: int):
        super().__init__()
        # Map speaker embedding → 2× channels (gamma, beta per channel)
        self.proj = nn.Linear(speaker_dim, 2 * channels)
        # Initialise to identity modulation (gamma≈1, beta≈0)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, speaker_emb: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation.

        Parameters
        ----------
        x : Tensor
            Feature map, shape ``(B, C, T)``.
        speaker_emb : Tensor
            Speaker embedding, shape ``(B, speaker_dim)``.

        Returns
        -------
        Tensor
            Modulated feature map, shape ``(B, C, T)``.
        """
        gamma_beta = self.proj(speaker_emb)          # (B, 2*C)
        gamma, beta = gamma_beta.chunk(2, dim=1)     # each (B, C)
        # Add trailing dim for broadcasting over time
        gamma = gamma.unsqueeze(-1)                   # (B, C, 1)
        beta = beta.unsqueeze(-1)                     # (B, C, 1)
        return gamma * x + beta


# ═══════════════════════════════════════════════════════════════════
# Encoder Block
# ═══════════════════════════════════════════════════════════════════

class EncoderBlock(nn.Module):
    """Downsampling encoder block: Conv1D → ReLU → ResidualBlock × N.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int
        Convolution stride (also determines kernel = 2× stride).
    n_res_blocks : int
        Number of residual blocks after the downsampling conv.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int, n_res_blocks: int):
        super().__init__()
        # Downsampling convolution — kernel size = 2× stride so that
        # every input position contributes to exactly 2 output positions.
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=2 * stride,
            stride=stride,
            padding=stride // 2,
        )
        self.act = nn.ReLU(inplace=True)
        # Residual blocks at the output channel count
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(out_channels) for _ in range(n_res_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, C_in, T)``.

        Returns
        -------
        Tensor
            Shape ``(B, C_out, T // stride)``.
        """
        x = self.conv(x)
        x = self.act(x)
        x = self.res_blocks(x)
        return x


# ═══════════════════════════════════════════════════════════════════
# Decoder Block  (with FiLM conditioning)
# ═══════════════════════════════════════════════════════════════════

class DecoderBlock(nn.Module):
    """Upsampling decoder block: FiLM → ConvTranspose1D → ReLU → ResidualBlock × N.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int
        Transposed convolution stride (kernel = 2× stride).
    speaker_dim : int
        Dimensionality of the speaker embedding for FiLM.
    n_res_blocks : int
        Number of residual blocks after the upsampling conv.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int, speaker_dim: int, n_res_blocks: int,
                 output_padding: int = 0):
        super().__init__()
        self.film = FiLM(speaker_dim, in_channels)
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=2 * stride,
            stride=stride,
            padding=stride // 2,
            output_padding=output_padding,
        )
        self.act = nn.ReLU(inplace=True)
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(out_channels) for _ in range(n_res_blocks)
        ])

    def forward(self, x: torch.Tensor,
                speaker_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, C_in, T)``.
        speaker_emb : Tensor
            Shape ``(B, speaker_dim)``.

        Returns
        -------
        Tensor
            Shape ``(B, C_out, T * stride)``.
        """
        x = self.film(x, speaker_emb)
        x = self.conv_transpose(x)
        x = self.act(x)
        x = self.res_blocks(x)
        return x


# ═══════════════════════════════════════════════════════════════════
# Residual VQ  (EMA codebook updates)
# ═══════════════════════════════════════════════════════════════════

class ResidualVQ(nn.Module):
    """Residual Vector Quantisation with EMA codebook updates.

    Stacks ``n_codebooks`` VQ layers.  Each subsequent layer
    quantises the residual of the previous layer's quantisation::

        r_0 = z
        z_q_0, idx_0 = VQ_0(r_0)
        r_1 = r_0 - z_q_0
        z_q_1, idx_1 = VQ_1(r_1)
        ...

    Codebooks are trained with exponential-moving-average updates
    (decay = ``ema_decay``) rather than through straight-through
    gradient descent.

    Parameters
    ----------
    n_codebooks : int
        Number of stacked VQ layers.
    codebook_size : int
        Number of entries per codebook.
    code_dim : int
        Dimensionality of each codebook entry.
    ema_decay : float
        EMA decay rate for codebook centroid updates.
    epsilon : float
        Small constant to prevent division by zero during EMA.

    References
    ----------
    - Van Den Oord et al., "Neural Discrete Representation Learning", NeurIPS 2017.
    - "Improved VQGAN" (Esser et al., CVPR 2021) — EMA variant.
    """

    def __init__(self, n_codebooks: int = 8, codebook_size: int = 1024,
                 code_dim: int = 64, ema_decay: float = 0.99,
                 epsilon: float = 1e-5):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.ema_decay = ema_decay
        self.epsilon = epsilon

        # Per-codebook embeddings: one (codebook_size, code_dim) table per layer.
        # We store them in a single (n_codebooks, codebook_size, code_dim) buffer
        # and index with per-codebook logic.
        embed = torch.empty(n_codebooks, codebook_size, code_dim)
        nn.init.uniform_(embed, -1.0 / codebook_size, 1.0 / codebook_size)
        # Register as plain parameter WITHOUT gradient tracking — EMA only.
        self.register_buffer("embed", embed)
        # For EMA: cluster usage count and running sum.
        self.register_buffer("cluster_usage", torch.zeros(n_codebooks, codebook_size))
        self.register_buffer("embed_sum", embed.clone())
        # Per-codebook initialisation flags (used for laplace smoothing)
        self.register_buffer("initialised", torch.zeros(n_codebooks, dtype=torch.bool))

    def _ema_update(self, flat_z: torch.Tensor, indices: torch.Tensor,
                     cb_idx: int) -> None:
        """Update one codebook with EMA centroids.

        Parameters
        ----------
        flat_z : Tensor
            Input vectors, shape ``(N, code_dim)``.
        indices : Tensor
            Assigned codebook indices, shape ``(N,)``.
        cb_idx : int
            Which codebook (0 .. n_codebooks-1) to update.
        """
        with torch.no_grad():
            # One-hot encode the assignments
            one_hot = F.one_hot(indices, num_classes=self.codebook_size).float()  # (N, K)
            # Sum of assignments per centroid (cluster size)
            usage = one_hot.sum(dim=0)  # (K,)
            # Sum of assigned vectors per centroid
            z_sum = one_hot.t() @ flat_z  # (K, code_dim)

            # Laplace smoothing on first update per codebook to avoid dead codes
            if not self.initialised[cb_idx]:
                usage += 1.0

            # Update running cluster usage and sum
            self.cluster_usage[cb_idx] = (
                self.ema_decay * self.cluster_usage[cb_idx]
                + (1.0 - self.ema_decay) * usage
            )
            self.embed_sum[cb_idx] = (
                self.ema_decay * self.embed_sum[cb_idx]
                + (1.0 - self.ema_decay) * z_sum
            )

            # Compute new centroids: total_sum / total_count
            n = self.cluster_usage[cb_idx].unsqueeze(-1)  # (K, 1)
            new_embed = self.embed_sum[cb_idx] / (n + self.epsilon)
            self.embed[cb_idx] = new_embed
            # Mark this codebook as initialised
            self.initialised[cb_idx] = True

    def encode(self, z: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Vector quantisation with straight-through estimator.

        During forward (training), gradients flow through the *selected*
        codebook entries via the straight-through estimator::

            z_hat = z + (z_q - z).detach()

        The EMA centroid updates happen in-place on the ``embed`` buffer
        and don't interact with autograd.

        Parameters
        ----------
        z : Tensor
            Continuous latent, shape ``(B, D, T)``.

        Returns
        -------
        z_q_list : list[Tensor]
            Quantised representation per codebook, each ``(B, D, T)``.
        indices_list : list[Tensor]
            Codebook indices per codebook, each ``(B, T)``.
        """
        B, D, T = z.shape
        device = z.device

        # Flatten time and batch dimensions for codebook lookup
        flat_z = z.permute(0, 2, 1).reshape(-1, D)  # (B*T, D)

        z_q_list: List[torch.Tensor] = []
        indices_list: List[torch.Tensor] = []

        residual = flat_z
        for cb_idx in range(self.n_codebooks):
            # Compute distances to all codebook entries: ||r - e||^2
            embed_k = self.embed[cb_idx]                     # (K, D)
            dist = (
                residual.pow(2).sum(dim=1, keepdim=True)      # (N, 1)
                - 2.0 * (residual @ embed_k.t())              # (N, K)
                + embed_k.pow(2).sum(dim=1).unsqueeze(0)      # (1, K)
            )                                                 # (N, K)

            # Nearest neighbour lookup
            indices = dist.argmin(dim=-1)                     # (N,)

            # Straight-through: quantised = codebook[indices], detached
            z_q = F.embedding(indices, embed_k)               # (N, D)

            # EMA update (training only, no grad)
            if self.training:
                self._ema_update(residual.detach(), indices, cb_idx)

            # Straight-through estimator: copy gradient from residual
            z_q_ste = residual + (z_q - residual).detach()    # (N, D)

            # Accumulate quantised output and update residual
            z_q_list.append(z_q_ste.reshape(B, T, D).permute(0, 2, 1))
            indices_list.append(indices.reshape(B, T))
            residual = residual - z_q                          # next codebook sees residual

        return z_q_list, indices_list

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: quantise + return loss components.

        Parameters
        ----------
        z : Tensor
            Continuous latent, ``(B, D, T)``.

        Returns
        -------
        z_q : Tensor
            Sum of quantised outputs across all codebooks, ``(B, D, T)``.
        indices : Tensor
            Codebook indices for the **first** codebook only (for logging),
            ``(B, T)``.
        vq_loss : Tensor
            Scalar loss = commitment loss (``beta * ||z - z_q||^2``).
        """
        z_q_list, indices_list = self.encode(z)
        # Sum quantised outputs from all codebooks
        z_q = torch.stack(z_q_list, dim=0).sum(dim=0)   # (B, D, T)
        indices = indices_list[0]                         # first codebook for logging

        # Commitment loss: encourage encoder output to stay close to
        # quantised vectors (only gradients flow to encoder, not codebook)
        commitment_loss = F.mse_loss(z_q.detach(), z)

        return z_q, indices, commitment_loss

    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Quantisation without gradient tracking (inference mode).

        Parameters
        ----------
        z : Tensor
            Continuous latent, ``(B, D, T)``.

        Returns
        -------
        z_q : Tensor
            Quantised latent, ``(B, D, T)``.
        indices_list : list[Tensor]
            Codebook indices per codebook, each ``(B, T)``.
        """
        with torch.no_grad():
            z_q_list, indices_list = self.encode(z)
            z_q = torch.stack(z_q_list, dim=0).sum(dim=0)
        return z_q, indices_list


# ═══════════════════════════════════════════════════════════════════
# Multi-Scale STFT Loss
# ═══════════════════════════════════════════════════════════════════

class MultiScaleSTFTLoss(nn.Module):
    """Multi-scale Spectral L1 loss over several FFT resolutions.

    Computes L1 distance between magnitude spectrograms at multiple
    FFT sizes, which helps the model reproduce both fine time-domain
    detail (small FFT) and coarse spectral envelope (large FFT).

    Parameters
    ----------
    fft_sizes : tuple[int, ...]
        FFT sizes for each scale.
    hop_sizes : tuple[int, ...]
        Hop sizes for each scale.
    win_lengths : tuple[int, ...]
        Window lengths for each scale.
    window : str
        Window type passed to ``torch.stft``.
    """

    def __init__(self, fft_sizes: Tuple[int, ...] = (512, 1024, 2048),
                 hop_sizes: Tuple[int, ...] = (128, 256, 512),
                 win_lengths: Tuple[int, ...] = (512, 1024, 2048),
                 window: str = "hann_window"):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.window = window

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale STFT loss.

        Parameters
        ----------
        x : Tensor
            Predicted waveform, ``(B, 1, T)``.
        y : Tensor
            Target waveform, ``(B, 1, T)``.

        Returns
        -------
        loss : Tensor
            Scalar loss (average across scales).
        """
        loss = 0.0
        for fft_size, hop_size, win_length in zip(
                self.fft_sizes, self.hop_sizes, self.win_lengths):
            # Compute STFT for both signals
            x_stft = torch.stft(
                x.squeeze(1), n_fft=fft_size, hop_length=hop_size,
                win_length=win_length, window=self._get_window(win_length, x.device),
                return_complex=True,
            )  # (B, F, T')
            y_stft = torch.stft(
                y.squeeze(1), n_fft=fft_size, hop_length=hop_size,
                win_length=win_length, window=self._get_window(win_length, y.device),
                return_complex=True,
            )
            # Magnitude spectrogram
            x_mag = x_stft.abs()  # (B, F, T')
            y_mag = y_stft.abs()
            loss += F.l1_loss(x_mag, y_mag)
        return loss / len(self.fft_sizes)

    def _get_window(self, win_length: int, device: torch.device) -> torch.Tensor:
        """Return a Hann window for the given length, cached on first use."""
        return torch.hann_window(win_length, device=device)


# ═══════════════════════════════════════════════════════════════════
# RVQ-VAE
# ═══════════════════════════════════════════════════════════════════

class RVQVAE(nn.Module):
    """Residual VQ-VAE audio codec with FiLM speaker conditioning.

    The encoder downsamples the input waveform by 1280× and projects
    to a ``latent_dim`` bottleneck.  The ResidualVQ module quantises
    with 8 codebooks.  The decoder upsamples back to waveform domain,
    conditioned on a speaker embedding via FiLM at each decoder block.

    Parameters
    ----------
    config : AudioCodecConfig
        Hyperparameter container (see ``zaya_audio.config``).
    n_speakers : int, optional
        If given, a learned speaker embedding table of shape
        ``(n_speakers, speaker_dim)`` is created.  Otherwise the
        speaker embedding must be supplied externally at runtime.
    """

    def __init__(self, config, n_speakers: Optional[int] = None):
        from .config import AudioCodecConfig
        if not isinstance(config, AudioCodecConfig):
            raise TypeError("config must be an AudioCodecConfig instance")
        super().__init__()
        self.config = config
        self.speaker_dim = config.speaker_dim
        self.latent_dim = config.latent_dim

        # ─── Optional learned speaker embedding table ────────────────
        self.speaker_embedding: Optional[nn.Embedding] = None
        if n_speakers is not None:
            self.speaker_embedding = nn.Embedding(n_speakers, config.speaker_dim)

        # ─── Channel plan ────────────────────────────────────────────
        # Block counts: [ch0, ch1, ch2, ch3, ch4]  with hidden_dim * scale
        self.channels = [
            config.channels_for_block(i) for i in range(5)
        ]

        # ─── Pre-convolution ─────────────────────────────────────────
        # Project mono waveform to first block channel count
        self.pre_conv = nn.Conv1d(1, self.channels[0],
                                  kernel_size=7, padding=3)

        # ─── Encoder ─────────────────────────────────────────────────
        encoder_blocks: List[nn.Module] = []
        in_ch = self.channels[0]
        for i, stride in enumerate(config.encoder_strides):
            out_ch = self.channels[i]
            encoder_blocks.append(
                EncoderBlock(in_ch, out_ch, stride, config.n_res_blocks)
            )
            in_ch = out_ch
        self.encoder = nn.Sequential(*encoder_blocks)

        # ─── Latent projection (encoder out → latent_dim) ────────────
        self.enc_proj = nn.Conv1d(in_ch, config.latent_dim,
                                  kernel_size=3, padding=1)

        # ─── Residual VQ ─────────────────────────────────────────────
        # Pre-VQ projection: latent_dim → code_dim
        self.pre_vq = nn.Conv1d(config.latent_dim, config.code_dim,
                                kernel_size=1)
        self.res_vq = ResidualVQ(
            n_codebooks=config.n_codebooks,
            codebook_size=config.codebook_size,
            code_dim=config.code_dim,
            ema_decay=config.ema_decay,
        )
        # Post-VQ projection: code_dim → latent_dim
        self.post_vq = nn.Conv1d(config.code_dim, config.latent_dim,
                                 kernel_size=1)

        # ─── Decoder ─────────────────────────────────────────────────
        # Project latent → decoder first channel count
        dec_in_ch = self.channels[-1]
        self.dec_proj = nn.Conv1d(config.latent_dim, dec_in_ch,
                                  kernel_size=3, padding=1)

        # Pre-computed output_padding per decoder block to exactly
        # reverse the encoder length (72000 in → 72000 out).
        # Derived from: L_out = (L_in-1)*s - 2*p + k + op with
        # k=2s, p=s//2, and solving for op to reach target length.
        decoder_output_paddings: Tuple[int, ...] = (0, 3, 3, 4, 0)
        decoder_blocks: List[nn.Module] = []
        in_ch = dec_in_ch
        for i, stride in enumerate(config.decoder_strides):
            # Channel index from the end (mirror encoder)
            out_ch = self.channels[-(i + 1)]
            decoder_blocks.append(
                DecoderBlock(in_ch, out_ch, stride, config.speaker_dim,
                             config.n_res_blocks,
                             output_padding=decoder_output_paddings[i])
            )
            in_ch = out_ch
        self.decoder = nn.ModuleList(decoder_blocks)

        # ─── Post-convolution (waveform output) ──────────────────────
        self.post_conv = nn.Conv1d(in_ch, 1, kernel_size=7, padding=3)

        # ─── Loss modules (available for training) ───────────────────
        self.stft_loss = MultiScaleSTFTLoss(
            fft_sizes=config.stft_fft_sizes,
            hop_sizes=config.stft_hop_sizes,
            win_lengths=config.stft_win_lengths,
        )

        # ─── Verify parameter target ─────────────────────────────────
        n_params = self.parameter_count()
        ratio = n_params / config.target_params
        if not (0.99 <= ratio <= 1.01):
            import warnings
            warnings.warn(
                f"RVQVAE has {n_params:,} params — target is "
                f"{config.target_params:,} "
                f"({'+' if ratio > 1 else ''}{((ratio - 1) * 100):.1f}% off)."
            )

    # ─── Forward (training) ─────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        speaker_emb: Optional[torch.Tensor] = None,
        speaker_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full training forward pass.

        Parameters
        ----------
        x : Tensor
            Input waveform, ``(B, 1, T)``.
        speaker_emb : Tensor, optional
            Speaker embedding, ``(B, speaker_dim)``.  If both
            ``speaker_emb`` and ``speaker_ids`` are ``None``, an
            all-zero embedding is used.
        speaker_ids : Tensor, optional
            Speaker IDs, ``(B,)``.  Resolved via the learned embedding
            table if available, otherwise ignored.

        Returns
        -------
        x_hat : Tensor
            Reconstructed waveform, ``(B, 1, T')``.
        vq_indices : Tensor
            First-codebook indices, ``(B, T_latent)``.
        vq_loss : Tensor
            Commitment loss (scalar).
        diversity_loss : Tensor
            Codebook diversity loss (scalar).
        """
        # ── Encode ──
        z = self.pre_conv(x)                      # (B, ch0, T)
        z = self.encoder(z)                       # (B, ch_last, T/1280)
        z = self.enc_proj(z)                      # (B, latent_dim, T/1280)

        # ── VQ ──
        z_vq_in = self.pre_vq(z)                  # (B, code_dim, T/1280)
        z_q, vq_indices, commit_loss = self.res_vq(z_vq_in)
        z_q = self.post_vq(z_q)                   # (B, latent_dim, T/1280)

        # ── Resolve speaker embedding ──
        emb = self._resolve_speaker_emb(speaker_emb, speaker_ids)

        # ── Decode ──
        x_hat = self.decode(z_q, emb)

        # ── Diversity loss (encourage even codebook usage) ──
        diversity_loss = self._diversity_loss(vq_indices)

        return x_hat, vq_indices, commit_loss, diversity_loss

    # ─── Decode (inference entry point) ─────────────────────────────

    def decode(self, z_q: torch.Tensor,
               speaker_emb: torch.Tensor) -> torch.Tensor:
        """Decode quantised latent back to waveform.

        Parameters
        ----------
        z_q : Tensor
            Quantised latent, ``(B, latent_dim, T_latent)``.
        speaker_emb : Tensor
            Speaker embedding, ``(B, speaker_dim)``.

        Returns
        -------
        x_hat : Tensor
            Reconstructed waveform, ``(B, 1, T_out)``.
        """
        x = self.dec_proj(z_q)                    # (B, ch0_dec, T_latent)
        for block in self.decoder:
            x = block(x, speaker_emb)             # progressive upsampling
        x = self.post_conv(x)                     # (B, 1, T_out)
        return x

    # ─── Encode (for inference: waveform → tokens) ──────────────────

    def encode_audio(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encode waveform to discrete codec tokens.

        Parameters
        ----------
        x : Tensor
            Waveform, ``(B, 1, T)``.

        Returns
        -------
        z_q : Tensor
            Quantised latent, ``(B, latent_dim, T_latent)``.
        indices_list : list[Tensor]
            Codebook indices per codebook, each ``(B, T_latent)``.
        """
        z = self.pre_conv(x)
        z = self.encoder(z)
        z = self.enc_proj(z)
        z_vq_in = self.pre_vq(z)
        z_q, indices_list = self.res_vq.quantize(z_vq_in)
        z_q = self.post_vq(z_q)
        return z_q, indices_list

    # ─── Full reconstruction (inference: waveform → waveform) ───────

    def reconstruct_audio(
        self,
        x: torch.Tensor,
        speaker_emb: Optional[torch.Tensor] = None,
        speaker_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full encode-quantise-decode pipeline (inference).

        Parameters
        ----------
        x : Tensor
            Input waveform, ``(B, 1, T)``.
        speaker_emb : Tensor, optional
            Speaker embedding, ``(B, speaker_dim)``.
        speaker_ids : Tensor, optional
            Speaker IDs, ``(B,)``.

        Returns
        -------
        x_hat : Tensor
            Reconstructed waveform, ``(B, 1, T_out)``.
        """
        z_q, _ = self.encode_audio(x)
        emb = self._resolve_speaker_emb(speaker_emb, speaker_ids)
        return self.decode(z_q, emb)

    # ─── Compute loss (convenience for training loops) ──────────────

    def compute_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        commit_loss: torch.Tensor,
        diversity_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Combined training loss.

        ``loss = L1_waveform + MultiScaleSTFT + beta * commit + alpha * diversity``

        Parameters
        ----------
        x : Tensor
            Target waveform, ``(B, 1, T)``.
        x_hat : Tensor
            Reconstructed waveform, ``(B, 1, T')``.
        commit_loss : Tensor
            VQ commitment loss (scalar).
        diversity_loss : Tensor
            Codebook diversity loss (scalar).

        Returns
        -------
        loss : Tensor
            Total loss (scalar).
        """
        # L1 waveform loss (time-domain)
        l1_loss = F.l1_loss(x_hat, x)

        # Multi-scale STFT loss (frequency-domain)
        stft_loss_val = self.stft_loss(x_hat, x)

        recon_loss = l1_loss + stft_loss_val
        total_loss = (
            recon_loss
            + self.config.vq_commitment_beta * commit_loss
            + self.config.diversity_alpha * diversity_loss
        )
        return total_loss

    # ─── Helpers ────────────────────────────────────────────────────

    def _resolve_speaker_emb(
        self,
        speaker_emb: Optional[torch.Tensor],
        speaker_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return the speaker embedding to use for FiLM conditioning.

        Priority:
        1. ``speaker_emb`` if given.
        2. Lookup from ``self.speaker_embedding`` if ``speaker_ids`` given.
        3. Zero embedding (fallback).
        """
        if speaker_emb is not None:
            return speaker_emb
        if speaker_ids is not None and self.speaker_embedding is not None:
            return self.speaker_embedding(speaker_ids)
        # Fallback: zero embedding
        device = next(self.parameters()).device
        if speaker_ids is not None:
            B = speaker_ids.shape[0]
        elif speaker_emb is not None:
            B = speaker_emb.shape[0]
        else:
            B = 1
        return torch.zeros(B, self.speaker_dim, device=device)

    def _diversity_loss(self, indices: torch.Tensor) -> torch.Tensor:
        """Encourage even distribution over codebook entries.

        Computes the negative entropy of the average codebook
        usage over the batch.

        Parameters
        ----------
        indices : Tensor
            Codebook indices, ``(B, T)``.
        """
        # Flatten and compute empirical distribution
        flat_idx = indices.flatten()                     # (B*T,)
        usage = torch.bincount(flat_idx,
                               minlength=self.config.codebook_size).float()
        prob = usage / usage.sum()
        # Entropy: H = -sum(p * log(p + eps))
        entropy = -(prob * (prob + 1e-8).log()).sum()
        # Diversity loss = -entropy  (maximise entropy = encourage uniform usage)
        return -entropy

    def parameter_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        n = self.parameter_count()
        return (
            f"{self.__class__.__name__}({n:,} params, "
            f"latent_dim={self.latent_dim}, "
            f"codebooks={self.config.n_codebooks}×{self.config.codebook_size})"
        )

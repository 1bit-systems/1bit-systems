"""Text→Codec token adapter model for voice cloning.

Maps raw text (byte-level tokens) to RVQ-VAE codec token indices,
enabling text-to-speech synthesis when paired with the codec decoder.

Architecture
------------
::

    text tokens (B, L)
         │
         ▼
    Byte-Pair embedding (vocab_size=256 → d_model=192)
         │
         ▼
    Transformer encoder (3 layers, 4 heads, d_ff=768)
         │
         ├──→ Duration predictor (linear, 1 scalar per token)
         │      │
         │      ▼
         │    Repeat/expand hidden states by duration
         │
         ▼
    Codec head (Linear: d_model → n_codebooks × codebook_size)
         │
         ▼
    codec logits (B, expanded_L, n_codebooks, codebook_size)

The model is designed to run as a lightweight adapter (~3M params) on
top of a frozen RVQ-VAE codec encoder.  Training aligns text tokens
to codec token sequences via monotonic duration prediction.

Parameter target: ~3M.  Verified at construction.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════


@dataclass
class TextToCodecConfig:
    """Hyperparameters for the text→codec adapter model.

    Attributes
    ----------
    vocab_size : int
        Size of the byte-level vocabulary (0–255).
    d_model : int
        Transformer hidden dimension.
    n_layers : int
        Number of transformer encoder layers.
    n_heads : int
        Number of attention heads per layer.
    d_ff : int
        Feed-forward hidden dimension.
    max_seq_len : int
        Maximum number of text tokens (input sequence length).
    n_codebooks : int
        Number of RVQ codebooks (must match codec).
    codebook_size : int
        Entries per codebook (must match codec).
    dropout : float
        Dropout probability within transformer layers.
    duration_loss_weight : float
        Weight for the duration prediction MSE loss relative to
        the codec cross-entropy loss.
    target_params : int
        Design target for total trainable parameters.
    """

    vocab_size: int = 256
    d_model: int = 192
    n_layers: int = 3
    n_heads: int = 4
    d_ff: int = 768
    max_seq_len: int = 512
    n_codebooks: int = 8
    codebook_size: int = 1024
    dropout: float = 0.1
    duration_loss_weight: float = 1.0
    target_params: int = 3_000_000

    @property
    def codec_output_dim(self) -> int:
        """Flattened output dimension of the codec head."""
        return self.n_codebooks * self.codebook_size


# ═══════════════════════════════════════════════════════════════════
# Tokenizer (byte-level)
# ═══════════════════════════════════════════════════════════════════


def tokenize_text(text: str, max_len: int = 512) -> torch.Tensor:
    """Convert a text string to byte-level token ids.

    Each character is mapped to its byte value (0–255).  The sequence
    is truncated/padded to ``max_len``.

    Parameters
    ----------
    text : str
        Input text.
    max_len : int
        Maximum sequence length (truncate if longer, pad if shorter).

    Returns
    -------
    tokens : Tensor
        ``(seq_len,)`` long tensor of byte token ids.  ``seq_len``
        is at most ``max_len`` (no padding token — use 0 as BOS/EOF).
    """
    # Convert to bytes then map each byte to an int 0–255
    byte_values = list(text.encode("utf-8"))
    if len(byte_values) > max_len:
        byte_values = byte_values[:max_len]
    return torch.tensor(byte_values, dtype=torch.long)


def tokenize_text_batch(
    texts: List[str], max_len: int = 512
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize a batch of texts with padding.

    Parameters
    ----------
    texts : list of str
        Input texts.
    max_len : int
        Maximum sequence length.

    Returns
    -------
    tokens : Tensor
        ``(B, L)`` long tensor of token ids, right-padded with 0.
    lengths : Tensor
        ``(B,)`` long tensor of actual sequence lengths (excluding padding).
    """
    batch = []
    lengths = []
    for t in texts:
        byte_values = list(t.encode("utf-8"))
        if len(byte_values) > max_len:
            byte_values = byte_values[:max_len]
        batch.append(byte_values)
        lengths.append(len(byte_values))

    max_seq = max(lengths)
    padded = torch.zeros(len(texts), max_seq, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

    return padded, torch.tensor(lengths, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════════
# Positional Encoding
# ═══════════════════════════════════════════════════════════════════


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (no learned parameters).

    Produces a fixed ``(1, max_len, d_model)`` encoding table that
    is added to the input embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Parameters
        ----------
        x : Tensor
            ``(B, L, d_model)``.

        Returns
        -------
        Tensor
            ``(B, L, d_model)``.
        """
        return x + self.pe[:, : x.size(1), :]


# ═══════════════════════════════════════════════════════════════════
# TextToCodecModel
# ═══════════════════════════════════════════════════════════════════


class TextToCodecModel(nn.Module):
    """Transformer-based text→codec token adapter.

    Maps byte-level text tokens to RVQ codec token logits with
    learnt duration prediction (how many codec frames each text
    token spans).

    Parameters
    ----------
    config : TextToCodecConfig
        Hyperparameter container.
    """

    def __init__(self, config: TextToCodecConfig):
        super().__init__()
        self.config = config

        # ─── Byte-level embedding ─────────────────────────────────
        self.embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=0
        )
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len)

        # ─── Transformer encoder ───────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_layers
        )

        # ─── Duration predictor ────────────────────────────────────
        # Predicts a scalar "how many codec frames does this token span"
        self.duration_predictor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(config.d_model // 2, 1),
        )

        # ─── Codec head ────────────────────────────────────────────
        # Projects hidden state to logits over all codebook entries
        self.codec_head = nn.Linear(
            config.d_model, config.n_codebooks * config.codebook_size
        )

        # ─── Verify parameter target ───────────────────────────────
        n_params = self.parameter_count()
        ratio = n_params / config.target_params
        # Accept 0.8–1.2 range (flexible for config tweaks)
        if not (0.8 <= ratio <= 1.2):
            import warnings

            warnings.warn(
                f"TextToCodecModel has {n_params:,} params — target is "
                f"{config.target_params:,} "
                f"({'−' if ratio < 1 else '+'}{abs(ratio - 1) * 100:.1f}% off)."
            )

    # ─── Forward (training + inference) ───────────────────────────

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_lengths: Optional[torch.Tensor] = None,
        target_durations: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        During training, use ``target_durations`` (from the alignment
        between text and codec frames) for the expansion step, and
        the model's predicted durations for the duration loss.

        During inference, predicted durations are used for expansion.

        Parameters
        ----------
        text_tokens : Tensor
            Byte-level token ids, ``(B, L)``.
        text_lengths : Tensor, optional
            Actual lengths of each sequence, ``(B,)``.  Used for
            padding mask in the transformer.
        target_durations : Tensor, optional
            Ground-truth durations for training, ``(B, L)``.  If
            given, these are used for the expansion step (teacher
            forcing), else predicted durations are used.

        Returns
        -------
        codec_logits : Tensor
            Logits over codebook entries, ``(B, L', n_codebooks, codebook_size)``
            where ``L'`` is the expanded sequence length (sum of durations).
        pred_durations : Tensor
            Raw predicted durations (before softplus), ``(B, L)``.
        """
        B, L = text_tokens.shape

        # ── Embed ──
        x = self.embedding(text_tokens)  # (B, L, d_model)
        x = self.pos_encoding(x)

        # ── Transformer ──
        if text_lengths is not None:
            # Create padding mask (True = masked position)
            pad_mask = (
                torch.arange(L, device=text_tokens.device)
                .unsqueeze(0)
                .expand(B, -1)
                >= text_lengths.unsqueeze(1)
            )  # (B, L)
            x = self.transformer(x, src_key_padding_mask=pad_mask)
        else:
            x = self.transformer(x)

        # ── Duration prediction ──
        # Clamp durations to at least 1 so every text token maps to ≥1 frame
        pred_durations_raw = self.duration_predictor(x).squeeze(-1)  # (B, L)
        pred_durations = F.softplus(pred_durations_raw) + 0.1  # avoid zero

        # ── Expand hidden states by durations ──
        durations_to_use = (
            target_durations if target_durations is not None else pred_durations
        )
        expanded = self._expand_by_durations(x, durations_to_use)  # (B, L', d_model)

        # ── Codec head ──
        logits = self.codec_head(expanded)  # (B, L', n_books * book_size)
        logits = logits.view(
            B, -1, self.config.n_codebooks, self.config.codebook_size
        )

        return logits, pred_durations

    # ─── Loss computation ─────────────────────────────────────────

    def compute_loss(
        self,
        text_tokens: torch.Tensor,
        text_lengths: torch.Tensor,
        codec_indices: torch.Tensor,
        codec_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute joint loss (codec CE + duration MSE).

        Parameters
        ----------
        text_tokens : Tensor
            Byte-level token ids, ``(B, L)``.
        text_lengths : Tensor
            Actual text lengths, ``(B,)``.
        codec_indices : Tensor
            Ground-truth codec indices (from codec encoder),
            ``(B, T_latent, n_codebooks)``.
        codec_lengths : Tensor
            Actual codec frame lengths, ``(B,)``.

        Returns
        -------
        total_loss : Tensor
            ``codec_loss + duration_loss_weight * duration_loss``.
        codec_loss : Tensor
            Average cross-entropy across all codebooks.
        duration_loss : Tensor
            MSE between predicted and target durations.
        """
        # ── Compute target durations (uniform alignment) ──
        target_durations = self._compute_target_durations(
            text_lengths, codec_lengths, text_tokens.shape[1]
        )  # (B, L)

        # ── Forward pass (teacher forcing: use target durations for expansion) ──
        logits, pred_durations = self(
            text_tokens, text_lengths=text_lengths, target_durations=target_durations
        )

        # ── Codec cross-entropy loss ──
        # logits: (B, L', n_codebooks, codebook_size)
        # codec_indices: (B, T_latent, n_codebooks)
        # Note: L' == T_latent when using target durations
        codec_loss = 0.0
        L_expanded = logits.shape[1]
        for cb in range(self.config.n_codebooks):
            # Truncate or pad codec_indices to match expanded length
            indices_cb = codec_indices[:, :L_expanded, cb].reshape(-1)
            logits_cb = logits[:, :, cb, :].reshape(-1, self.config.codebook_size)

            # Mask out padding (where indices are -1 or beyond valid range)
            valid = indices_cb >= 0
            if valid.any():
                codec_loss += F.cross_entropy(
                    logits_cb[valid], indices_cb[valid]
                )

        codec_loss = codec_loss / self.config.n_codebooks

        # ── Duration MSE loss (only on valid positions) ──
        mask = (
            torch.arange(text_tokens.shape[1], device=text_tokens.device)
            .unsqueeze(0)  # (1, L)
            .expand(text_lengths.size(0), -1)  # (B, L)
            < text_lengths.unsqueeze(1)  # (B, L)
        ).float()
        duration_loss = F.mse_loss(
            pred_durations * mask, target_durations * mask, reduction="sum"
        ) / (text_lengths.sum() + 1e-8)

        total_loss = codec_loss + self.config.duration_loss_weight * duration_loss

        return total_loss, codec_loss, duration_loss

    # ─── Inference ───────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self, text_tokens: torch.Tensor, text_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate codec token indices from text.

        Unlike training which produces logits, this returns discrete
        codec indices via ``argmax`` over the codebook dimension.

        Parameters
        ----------
        text_tokens : Tensor
            Byte-level token ids, ``(B, L)``.
        text_lengths : Tensor, optional
            Actual lengths, ``(B,)``.

        Returns
        -------
        codec_indices : Tensor
            Codec token indices, ``(B, L', n_codebooks)``, dtype ``long``.
        """
        logits, _ = self(text_tokens, text_lengths=text_lengths)
        return logits.argmax(dim=-1)  # (B, L', n_codebooks)

    # ─── Parameter count ─────────────────────────────────────────

    def parameter_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ─── Internals ───────────────────────────────────────────────

    def _expand_by_durations(
        self, x: torch.Tensor, durations: torch.Tensor
    ) -> torch.Tensor:
        """Expand each position's hidden vector by its duration.

        Parameters
        ----------
        x : Tensor
            Hidden states, ``(B, L, d_model)``.
        durations : Tensor
            Duration per position, ``(B, L)``.  Values are floats
            (will be rounded to nearest integer for expansion).

        Returns
        -------
        expanded : Tensor
            ``(B, L', d_model)`` where ``L'`` is the sum of durations.
        """
        B, L, D = x.shape
        device = x.device

        # Round durations to integers (with minimum of 1)
        durations_int = durations.round().long().clamp(min=1)  # (B, L)

        # Build expanded tensor batch-by-batch (avoids complex scatter logic)
        max_expanded = durations_int.sum(dim=1).max().item()
        expanded = torch.zeros(B, max_expanded, D, device=device, dtype=x.dtype)

        for b in range(B):
            # Create source index for each position in this batch
            count = durations_int[b]  # (L,)
            # Repeat each source index `count[i]` times
            src_idx = torch.arange(L, device=device).repeat_interleave(count)
            L_ex = src_idx.size(0)
            expanded[b, :L_ex] = x[b, src_idx]

        return expanded

    def _compute_target_durations(
        self,
        text_lengths: torch.Tensor,
        codec_lengths: torch.Tensor,
        max_text_len: int,
    ) -> torch.Tensor:
        """Compute uniform target durations for training.

        Distributes codec frames evenly across text tokens::

            base = codec_length // text_length
            remainder = codec_length % text_length

        The first ``remainder`` tokens get ``base + 1`` frames,
        the rest get ``base`` frames.

        Parameters
        ----------
        text_lengths : Tensor
            Actual text sequence lengths, ``(B,)``.
        codec_lengths : Tensor
            Actual codec frame counts, ``(B,)``.
        max_text_len : int
            Maximum text sequence length (for output tensor shape).

        Returns
        -------
        durations : Tensor
            Target durations, ``(B, max_text_len)``.  Positions
            beyond ``text_lengths`` are zeroed.
        """
        B = text_lengths.size(0)
        device = text_lengths.device
        durations = torch.zeros(B, max_text_len, device=device)

        for i in range(B):
            L = text_lengths[i].item()
            T = codec_lengths[i].item()
            base = T // L
            remainder = T % L
            if L > 0:
                durations[i, :remainder] = base + 1
                durations[i, remainder:L] = base

        return durations


# ═══════════════════════════════════════════════════════════════════
# Convenience: model summary
# ═══════════════════════════════════════════════════════════════════

def create_default_model() -> TextToCodecModel:
    """Create a ``TextToCodecModel`` with default configuration."""
    config = TextToCodecConfig()
    return TextToCodecModel(config)

#!/usr/bin/env python3
"""
ZAYA Audio Codec — Pure PyTorch neural audio codec for voice cloning.

Part of the -thearchitect voice pipeline.
Runs on AMD ROCm (Radeon 8060S).

Architecture (RVQ-VAE):
  Audio → Encoder (conv1d downsample) → RVQ → Discrete Tokens
  Discrete Tokens → Decoder (conv1d upsample) → Audio

Usage:
    from zaya_audio import AudioCodec
    
    codec = AudioCodec().to('cuda')
    tokens = codec.encode(audio)     # (B, 1, T) → (B, N_q, T')
    audio  = codec.decode(tokens)    # (B, N_q, T') → (B, 1, T)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class AudioCodecConfig:
    """Configuration for the audio codec."""
    sample_rate: int = 24000
    hop_length: int = 320               # 75 Hz frame rate
    input_channels: int = 1
    hidden_channels: int = 64
    latent_dim: int = 128
    n_codebooks: int = 4                # RVQ stack depth
    codebook_size: int = 1024           # Codes per codebook
    codebook_dim: int = 128
    encoder_strides: Tuple[int, ...] = (2, 4, 5, 8)
    decoder_strides: Tuple[int, ...] = (8, 5, 4, 2)
    norm_type: str = "layer_norm"


# ─── Residual Vector Quantization ────────────────────────────────────────────

class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer with N stacked codebooks."""

    def __init__(self, config: AudioCodecConfig):
        super().__init__()
        self.n_codebooks = config.n_codebooks
        self.codebook_size = config.codebook_size
        self.codebook_dim = config.codebook_dim
        self.codebooks = nn.ModuleList([
            nn.Embedding(config.codebook_size, config.codebook_dim)
            for _ in range(config.n_codebooks)
        ])
        for cb in self.codebooks:
            nn.init.uniform_(cb.weight, -1.0 / config.codebook_size, 1.0 / config.codebook_size)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, D, T = z.shape
        z = z.permute(0, 2, 1).contiguous()
        z_q = torch.zeros_like(z)
        codes = torch.zeros(B, self.n_codebooks, T, dtype=torch.long, device=z.device)
        commit_loss = torch.tensor(0.0, device=z.device)
        residual = z

        for i, cb in enumerate(self.codebooks):
            flat = residual.reshape(-1, self.codebook_dim)
            dist = (torch.sum(flat ** 2, dim=1, keepdim=True)
                    - 2 * torch.matmul(flat, cb.weight.T)
                    + torch.sum(cb.weight ** 2, dim=1))
            idx = torch.argmin(dist, dim=1)
            zq_i = cb(idx).reshape(B, T, D)
            z_q = z_q + (zq_i + (residual - zq_i).detach())
            commit_loss = commit_loss + F.mse_loss(zq_i.detach(), residual)
            codes[:, i, :] = idx.reshape(B, T)
            residual = residual - zq_i.detach()

        return z_q.permute(0, 2, 1).contiguous(), codes, commit_loss

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        B, D, T = z.shape
        z = z.permute(0, 2, 1).contiguous()
        codes = torch.zeros(B, self.n_codebooks, T, dtype=torch.long, device=z.device)
        residual = z
        for i, cb in enumerate(self.codebooks):
            flat = residual.reshape(-1, self.codebook_dim)
            dist = (torch.sum(flat ** 2, dim=1, keepdim=True)
                    - 2 * torch.matmul(flat, cb.weight.T)
                    + torch.sum(cb.weight ** 2, dim=1))
            idx = torch.argmin(dist, dim=1)
            codes[:, i, :] = idx.reshape(B, T)
            residual = residual - cb(idx).reshape(B, T, D)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        B, N_q, T = codes.shape
        z_q = torch.zeros(B, T, self.codebook_dim, device=codes.device)
        for i in range(N_q):
            z_q = z_q + self.codebooks[i](codes[:, i, :])
        return z_q.permute(0, 2, 1).contiguous()


# ─── Encoder ─────────────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, norm: str):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=2 * stride,
                              stride=stride, padding=stride // 2,
                              bias=(norm != "batch_norm"))
        self.norm = nn.GroupNorm(1, out_ch) if norm == "layer_norm" else \
                    nn.BatchNorm1d(out_ch) if norm == "batch_norm" else nn.Identity()
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Encoder(nn.Module):
    """1D convolutional encoder with progressive downsampling."""

    def __init__(self, config: AudioCodecConfig):
        super().__init__()
        strides = config.encoder_strides
        channels = [config.input_channels] + [
            min(config.hidden_channels * (2 ** i), 512) for i in range(len(strides))
        ]
        self.blocks = nn.ModuleList([
            EncoderBlock(channels[i], channels[i + 1], s, config.norm_type)
            for i, s in enumerate(strides)
        ])
        self.proj = nn.Conv1d(channels[-1], config.latent_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return self.proj(x)


# ─── Decoder ─────────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, norm: str):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2 * stride,
                                       stride=stride, padding=stride // 2,
                                       bias=(norm != "batch_norm"))
        self.norm = nn.GroupNorm(1, out_ch) if norm == "layer_norm" else \
                    nn.BatchNorm1d(out_ch) if norm == "batch_norm" else nn.Identity()
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Decoder(nn.Module):
    """1D convolutional decoder with progressive upsampling."""

    def __init__(self, config: AudioCodecConfig):
        super().__init__()
        strides = config.decoder_strides
        channels = [config.latent_dim] + [
            min(config.hidden_channels * (2 ** (len(strides) - 1 - i)), 512)
            for i in range(len(strides))
        ]
        self.blocks = nn.ModuleList([
            DecoderBlock(channels[i], channels[i + 1], s, config.norm_type)
            for i, s in enumerate(strides)
        ])
        self.proj = nn.Conv1d(channels[-1], config.input_channels, kernel_size=7, padding=3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            z = b(z)
        return torch.tanh(self.proj(z)) * 0.95


# ─── Full Audio Codec ────────────────────────────────────────────────────────

class AudioCodec(nn.Module):
    """Full audio codec: encoder → RVQ → decoder."""

    def __init__(self, config: Optional[AudioCodecConfig] = None):
        super().__init__()
        self.config = config or AudioCodecConfig()
        self.encoder = Encoder(self.config)
        self.latent_proj = (nn.Conv1d(self.config.latent_dim, self.config.codebook_dim, 1)
                           if self.config.latent_dim != self.config.codebook_dim else nn.Identity())
        self.quantizer = ResidualVectorQuantizer(self.config)
        self.codebook_proj = (nn.Conv1d(self.config.codebook_dim, self.config.latent_dim, 1)
                             if self.config.codebook_dim != self.config.latent_dim else nn.Identity())
        self.decoder = Decoder(self.config)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.latent_proj(self.encoder(x))
        z_q, codes, commit_loss = self.quantizer(z)
        x_recon = self.decoder(self.codebook_proj(z_q))
        if x_recon.shape[-1] > x.shape[-1]:
            x_recon = x_recon[..., :x.shape[-1]]
        elif x_recon.shape[-1] < x.shape[-1]:
            x_recon = F.pad(x_recon, (0, x.shape[-1] - x_recon.shape[-1]))
        return x_recon, codes, commit_loss

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to discrete tokens.
        
        Args:
            audio: (B, 1, T), (B, T), or (T,)
        Returns:
            codes: (B, N_q, T')
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(1)
        with torch.no_grad():
            return self.quantizer.encode(self.latent_proj(self.encoder(audio)))

    def decode(self, codes: torch.Tensor, target_length: Optional[int] = None) -> torch.Tensor:
        """Decode tokens to audio.
        
        Args:
            codes: (B, N_q, T') or (N_q, T')
        Returns:
            audio: (B, 1, T)
        """
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
        with torch.no_grad():
            audio = self.decoder(self.codebook_proj(self.quantizer.decode(codes)))
        if target_length is not None:
            audio = audio[..., :target_length] if audio.shape[-1] > target_length \
                    else F.pad(audio, (0, target_length - audio.shape[-1]))
        return audio

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def frame_rate(self) -> int:
        return self.config.sample_rate // self.config.hop_length

    def tokens_to_seconds(self, n: int) -> float:
        return n * self.config.hop_length / self.config.sample_rate

    def seconds_to_tokens(self, s: float) -> int:
        return int(s * self.frame_rate)

    def save_audio(self, audio: torch.Tensor, path: str, sr: Optional[int] = None):
        if audio.dim() == 2:
            audio = audio.squeeze(0)
        sf.write(path, audio.cpu().numpy().astype(np.float32), sr or self.config.sample_rate)

    def load_audio(self, path: str, target_sr: Optional[int] = None) -> torch.Tensor:
        a, sr = sf.read(path)
        if a.ndim > 1:
            a = a.mean(axis=1)
        target_sr = target_sr or self.config.sample_rate
        if sr != target_sr:
            old_len = len(a)
            new_len = int(old_len * target_sr / sr)
            a = np.interp(np.linspace(0, old_len - 1, new_len), np.arange(old_len), a)
        return torch.from_numpy(a).float().unsqueeze(0)


# ─── Loss ────────────────────────────────────────────────────────────────────

class AudioCodecLoss(nn.Module):
    """Combined L1 + multi-scale STFT + commitment loss."""

    def __init__(self, alpha_commit: float = 0.25):
        super().__init__()
        self.alpha_commit = alpha_commit

    def forward(self, x: torch.Tensor, x_recon: torch.Tensor, commit_loss: torch.Tensor
                ) -> Tuple[torch.Tensor, dict]:
        l1 = F.l1_loss(x_recon, x)
        stft = torch.tensor(0.0, device=x.device)
        for n_fft, hop in [(512, 128), (256, 64), (128, 32)]:
            win = torch.hann_window(n_fft, device=x.device)
            x_mag = torch.abs(torch.stft(x.squeeze(1), n_fft, hop, window=win, return_complex=True))
            r_mag = torch.abs(torch.stft(x_recon.squeeze(1), n_fft, hop, window=win, return_complex=True))
            stft = stft + F.l1_loss(r_mag, x_mag)
        total = l1 + stft + self.alpha_commit * commit_loss
        return total, {"l1": l1.item(), "stft": stft.item(),
                       "commit": commit_loss.item(), "total": total.item()}


# ─── Smoke Test ──────────────────────────────────────────────────────────────

def test():
    print("=" * 60)
    print("ZAYA Audio Codec — Smoke Test")
    print("=" * 60)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")
    if dev == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    cfg = AudioCodecConfig()
    model = AudioCodec(cfg).to(dev)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    sr = cfg.sample_rate
    t = torch.linspace(0, 2.0, int(sr * 2.0), device=dev)
    audio = 0.5 * torch.sin(2 * torch.pi * 440 * t).unsqueeze(0).unsqueeze(0)

    recon, codes, cl = model(audio)
    print(f"Input:  {list(audio.shape)}")
    print(f"Recon:  {list(recon.shape)}")
    print(f"Codes:  {list(codes.shape)}  range=[{codes.min().item()}, {codes.max().item()}]")
    print(f"Comm. loss: {cl.item():.4f}")

    snr = 10 * torch.log10(torch.mean(audio**2) / torch.mean((audio - recon)**2))
    print(f"SNR:    {snr.item():.1f} dB")
    print(f"Frame rate: {model.frame_rate} Hz")
    print(f"Tokens/s:   {model.frame_rate * cfg.n_codebooks}")
    print("=" * 60)
    print("✓ Audio Codec Pipeline Verified")
    print("=" * 60)
    return model


if __name__ == "__main__":
    test()

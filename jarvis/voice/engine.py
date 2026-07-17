"""
Voice inference engine — Zaya Co-Host Phase 1.2

Loads a .voice pack and runs text-to-speech inference.
Uses the agnostic RVQ-VAE codec decoder (5.87M params).

Architecture:
    Any LLM → codec tokens → [Codec Decoder] → waveform
                                   ↑
                            Voice Pack (decoder weights + speaker embedding)

Usage:
    from jarvis.voice.engine import VoiceEngine
    engine = VoiceEngine()
    engine.load_pack("./voice/packs/bcloud.voice")
    audio, sr = engine.synthesize("Hello, I'm your AI co-host!")
"""

import json, os, tarfile, tempfile, shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch

VOICE_PACKS_DIR = Path(__file__).parent.parent.parent / "voice" / "packs"

# ── Codec Decoder ──────────────────────────────────────────────────────

class CodecDecoder:
    """Agnostic codec decoder — converts discrete tokens to audio waveform.
    
    Loads the 5.87M parameter RVQ-VAE decoder from a .voice pack.
    Runs on any device (CPU, CUDA, ROCm, NPU via ONNX).
    """

    def __init__(self, model_path: Path, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self._load()

    def _load(self):
        from jarvis.voice.codec import AudioCodec, AudioCodecConfig

        cfg = AudioCodecConfig()
        self.model = AudioCodec(cfg)
        
        # Load trained weights
        state = torch.load(self.model_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"    Codec decoder loaded: {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M params on {self.device}")

    def decode(self, codes: torch.Tensor, target_length: Optional[int] = None) -> np.ndarray:
        """Decode token sequence to audio waveform.
        
        Args:
            codes: (N_q, T') tensor of codec token indices
            target_length: Optional target audio length in samples
            
        Returns:
            Audio waveform as numpy array (float32, [-1, 1])
        """
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
        
        with torch.no_grad():
            audio = self.model.decode(codes.to(self.device), target_length=target_length)
        
        return audio.squeeze().cpu().numpy().astype(np.float32)

    @property
    def sample_rate(self) -> int:
        return self.model.sample_rate if self.model else 24000


# ── Voice Pack ─────────────────────────────────────────────────────────

class VoicePack:
    """Loaded and parsed .voice pack.
    
    Format (tar.gz):
        metadata.json     — Name, sample rate, pitch range, etc.
        decoder.pt        — Codec decoder state dict (5.87M params)
        speaker.pt        — Speaker embedding (512-dim vector)
    """

    def __init__(self, path: Path):
        self.path = path
        self.metadata: dict = {}
        self.speaker_embedding: Optional[np.ndarray] = None
        self.decoder: Optional[CodecDecoder] = None
        self._tmp_dir: Optional[Path] = None

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *args):
        self.cleanup()

    def load(self):
        """Extract and load a .voice pack."""
        self._tmp_dir = Path(tempfile.mkdtemp())
        
        with tarfile.open(self.path, "r:gz") as tar:
            tar.extractall(self._tmp_dir)

        # Load metadata
        meta_path = self._tmp_dir / "metadata.json"
        if meta_path.exists():
            self.metadata = json.loads(meta_path.read_text())

        # Load speaker embedding
        embed_path = self._tmp_dir / "speaker.pt"
        if embed_path.exists():
            self.speaker_embedding = torch.load(
                embed_path, map_location="cpu", weights_only=True
            ).numpy().flatten()
        else:
            self.speaker_embedding = np.random.randn(512).astype(np.float32)

        # Load codec decoder
        decoder_path = self._tmp_dir / "decoder.pt"
        if decoder_path.exists():
            self.decoder = CodecDecoder(decoder_path)

        print(f"  Voice pack loaded: {self.metadata.get('name', 'unknown')}")

    def synthesize(self, text: str, speed: float = 1.0) -> np.ndarray:
        """Synthesize text to audio.
        
        Note: In Phase 1, this uses a simple text→codec token lookup.
        Phase 2 will use ZAYA (or any LLM) for full text→tokens generation.
        
        Args:
            text: Text to synthesize
            speed: Playback speed (1.0 = normal)
            
        Returns:
            Audio waveform as numpy array
        """
        if not self.decoder:
            raise RuntimeError("No codec decoder loaded in voice pack")

        # Phase 1: Simple frame mapping from text characters
        # Phase 2+: This will call ZAYA/LLM for true text→codec token generation
        text_chars = [min(ord(c), 1023) for c in text[:200]]
        n_frames = max(10, len(text_chars) * 3)  # ~3 codec frames per character
        N_q = 4  # Number of codebooks
        
        # Create token sequence (placeholder — LLM will replace this)
        codes = torch.zeros(N_q, n_frames, dtype=torch.long)
        for i in range(min(len(text_chars), n_frames)):
            for q in range(N_q):
                codes[q, i] = text_chars[i] % 1024
        
        audio = self.decoder.decode(codes)
        
        # Apply speed
        if speed != 1.0:
            orig_len = len(audio)
            new_len = int(orig_len / speed)
            audio = np.interp(
                np.linspace(0, orig_len - 1, new_len),
                np.arange(orig_len),
                audio
            )
        
        return audio

    def synthesize_from_tokens(self, codes: torch.Tensor) -> np.ndarray:
        """Synthesize audio from pre-generated codec tokens.
        
        This is the main path once ZAYA generates the tokens.
        
        Args:
            codes: (N_q, T') tensor of codec token indices from LLM
            
        Returns:
            Audio waveform as numpy array
        """
        if not self.decoder:
            raise RuntimeError("No codec decoder loaded")
        return self.decoder.decode(codes)

    def cleanup(self):
        if self._tmp_dir:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None


# ── High-Level Engine ──────────────────────────────────────────────────

class VoiceEngine:
    """Manages multiple voice packs and provides synthesis API.
    
    Agnostic — works with any LLM backend that outputs codec tokens.
    """

    def __init__(self):
        self.voices: dict[str, VoicePack] = {}
        self.active_voice: Optional[str] = None
        self.sample_rate = 24000

    def list_available_packs(self) -> list[dict]:
        """List all .voice packs in the packs directory."""
        packs = []
        if not VOICE_PACKS_DIR.exists():
            return packs
        for f in sorted(VOICE_PACKS_DIR.glob("*.voice")):
            try:
                with tarfile.open(f, "r:gz") as tar:
                    meta = tar.extractfile("metadata.json")
                    if meta:
                        info = json.loads(meta.read())
                        info["path"] = str(f)
                        info["loaded"] = f.stem in self.voices
                        packs.append(info)
            except:
                packs.append({"name": f.stem, "path": str(f), "loaded": False})
        return packs

    def load_pack(self, path: str | Path, name: Optional[str] = None):
        """Load a voice pack."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Voice pack not found: {path}")
        
        pack = VoicePack(path)
        pack.load()
        voice_name = name or pack.metadata.get("name", path.stem)
        self.voices[voice_name] = pack
        self.sample_rate = pack.decoder.sample_rate if pack.decoder else 24000
        if self.active_voice is None:
            self.active_voice = voice_name
        return voice_name

    def unload(self, name: str):
        """Unload a voice pack."""
        if name in self.voices:
            self.voices[name].cleanup()
            del self.voices[name]
            if self.active_voice == name:
                self.active_voice = next(iter(self.voices.keys()), None)

    def activate(self, name: str):
        """Set active voice for synthesis."""
        if name not in self.voices:
            raise KeyError(f"Voice '{name}' not loaded. Available: {list(self.voices.keys())}")
        self.active_voice = name

    def synthesize(self, text: str, voice: Optional[str] = None,
                   speed: float = 1.0) -> tuple[np.ndarray, int]:
        """Synthesize text to audio waveform (text→codec tokens→audio).
        
        Phase 1: Simple text→token mapping (placeholder).
        Phase 2+: Routes through ZAYA LLM for true TTS.
        
        Returns:
            Tuple of (audio_samples, sample_rate)
        """
        v = voice or self.active_voice
        if not v or v not in self.voices:
            raise RuntimeError(f"No voice loaded. Load a voice pack first.")
        audio = self.voices[v].synthesize(text, speed)
        return audio, self.sample_rate

    def synthesize_from_tokens(self, codes: torch.Tensor,
                                voice: Optional[str] = None) -> tuple[np.ndarray, int]:
        """Synthesize audio from LLM-generated codec tokens.
        
        This is the primary path once ZAYA (or any LLM) generates tokens.
        
        Args:
            codes: (N_q, T') codec token tensor from LLM
            voice: Voice pack name to use
            
        Returns:
            Tuple of (audio_samples, sample_rate)
        """
        v = voice or self.active_voice
        if not v or v not in self.voices:
            raise RuntimeError(f"No voice loaded. Load a voice pack first.")
        audio = self.voices[v].synthesize_from_tokens(codes)
        return audio, self.sample_rate

    def cleanup_all(self):
        """Unload all voices."""
        for name in list(self.voices.keys()):
            self.unload(name)


# ── Quick test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=str, required=True,
                        help="Path to .voice pack")
    parser.add_argument("--text", type=str, default="Hello, I'm your AI co-host!",
                        help="Text to synthesize")
    parser.add_argument("--output", type=str, default="/tmp/test_voice.wav",
                        help="Output WAV path")
    args = parser.parse_args()

    import wave
    engine = VoiceEngine()
    name = engine.load_pack(args.pack)
    audio, sr = engine.synthesize(args.text)
    
    with wave.open(args.output, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    
    print(f"\n  Audio saved: {args.output}")
    print(f"  Duration: {len(audio) / sr:.1f}s")
    engine.cleanup_all()

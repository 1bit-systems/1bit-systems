"""JARVIS Voice I/O — STT (whisper) + TTS (piper)."""
import os
import subprocess
import asyncio
import tempfile
from pathlib import Path
from config import JarvisConfig


class VoiceIO:
    """Handles speech-to-text and text-to-speech for JARVIS."""

    def __init__(self, config: JarvisConfig):
        self.config = config
        self.tts_voice = None
        self._init_tts()

    def _init_tts(self):
        """Initialize piper TTS."""
        try:
            from piper import PiperVoice
            # Look for a downloaded voice in multiple locations
            search_paths = [
                Path(self.config.voice_dir) / "en" / "en_US" / "lessac" / "medium",
                Path(self.config.voice_dir) / "en_US" / "lessac" / "medium",
                Path(self.config.voice_dir) / "en_US",
                Path.home() / ".local/share/piper/voices/en/en_US/lessac/medium",
                Path.home() / ".local/share/piper/voices/en_US/lessac/medium",
            ]
            for voice_dir in search_paths:
                if voice_dir.exists():
                    for f in voice_dir.iterdir():
                        if f.suffix == ".onnx":
                            self.tts_voice = PiperVoice.load(str(f))
                            print(f"[JARVIS] TTS loaded: {f.name}")
                            return
            print(f"[JARVIS] No TTS voice found in {search_paths[0]}")
        except ImportError:
            print("[JARVIS] piper-tts not available")

    async def stt(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """Speech-to-text via whisper.cpp server or FLM."""
        # Try FLM whisper endpoint
        try:
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Use whisper-server if running
                if os.path.exists(self.config.whisper_server_bin):
                    return await self._stt_whisper_cpp(audio_data, sample_rate)
                # Fall back to FLM's whisper endpoint
                resp = await client.post(
                    f"http://127.0.0.1:{self.config.flm_port}/v1/audio/transcriptions",
                    files={"file": ("audio.wav", audio_data, "audio/wav")},
                    data={"model": self.config.asr_model, "response_format": "text"},
                    timeout=30.0,
                )
                if resp.status_code == 200:
                    return resp.text.strip()
                return "[STT unavailable]"
        except Exception as e:
            return f"[STT error: {e}]"

    async def _stt_whisper_cpp(self, audio_data: bytes, sample_rate: int) -> str:
        """Use whisper.cpp directly for STT."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_data)
                tmp_path = f.name

            proc = await asyncio.create_subprocess_exec(
                self.config.whisper_server_bin,
                "--model", self.config.whisper_model,
                "--file", tmp_path,
                "--output-txt",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            os.unlink(tmp_path)
            return stdout.decode().strip() or "[no speech detected]"
        except Exception as e:
            return f"[whisper error: {e}]"

    async def tts(self, text: str) -> bytes:
        """Text-to-speech via piper."""
        if self.tts_voice is None:
            return b""  # No TTS available
        try:
            import numpy as np
            import soundfile as sf
            import io

            audio = self.tts_voice.synthesize(text)
            buf = io.BytesIO()
            sf.write(buf, audio, 22050, format="wav")
            return buf.getvalue()
        except Exception as e:
            print(f"[JARVIS] TTS error: {e}")
            return b""

    @staticmethod
    async def download_voice(name: str = "en_US-lessac-medium"):
        """Download a piper voice model."""
        proc = await asyncio.create_subprocess_exec(
            "bash", "-c",
            f"source /home/bcloud/jarvis-env/bin/activate && python3 -m piper download {name}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return stdout.decode(), stderr.decode()

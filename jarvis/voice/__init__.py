"""Zaya Co-Host Voice Module — Phase 1.

Voice cloning, synthesis, and pack management for the Zaya Co-Host platform.

Components:
    record.py   — Record high-quality voice samples (24kHz mono)
    train.py    — Train voice clone: codec + ZAYA adapter → .voice pack
    engine.py   — Load .voice packs, synthesize speech via codec decoder

Architecture (agnostic):
    Audio → [Codec Encoder] → tokens → [Any LLM] → tokens → [Codec Decoder] → Audio
                               ↑  Voice Pack (speaker embedding)  ↑

Quick start:
    python3 jarvis/voice/record.py --name bcloud --session 1
    python3 jarvis/voice/train.py --samples ./voice/samples/bcloud/ --name bcloud
"""

from jarvis.voice.engine import VoiceEngine, VoicePack

__all__ = ["VoiceEngine", "VoicePack"]

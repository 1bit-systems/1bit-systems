---
type: Capability
title: Voice I/O
description: Speech-to-text via Whisper (NPU) + Text-to-speech via Piper (CPU). Real-time bidirectional voice over WebSocket.
tags: [voice, stt, tts, whisper, piper, audio]
timestamp: 2026-07-06T00:00:00Z
---

# Overview

JARVIS supports bidirectional voice I/O: speak to it (STT) and hear its responses (TTS), all locally.

## Speech-to-Text (Whisper)

Two paths:

| Path | Binary | Model | Latency |
|------|--------|-------|---------|
| whisper.cpp | `whisper-server` | `ggml-base.en.bin` | ~1.5s |
| FLM NPU | FLM API | `whisper-v3:turbo` | ~1s |

Source: `server/voice.py` → `VoiceIO.stt()`

## Text-to-Speech (Piper)

- Engine: [Piper TTS](https://github.com/rhasspy/piper)
- Voice: `en_US-lessac-medium`
- Latency: ~50ms
- Voice models at: `~/.local/share/piper/voices/`

Source: `server/voice.py` → `VoiceIO.tts()`

## WebSocket Protocol

```json
// STT: send audio bytes
{"type": "stt", "audio": "<base64-encoded-wav>"}
// Response
{"type": "stt_result", "text": "transcribed text"}

// TTS: send text
{"type": "tts", "text": "Text to speak"}
// Response
{"type": "tts_result", "audio": "<base64-encoded-wav>"}
```

## Citations

[1] [Data Flow](/architecture/data_flow.md)
[2] [Server Stack](/architecture/server_stack.md)

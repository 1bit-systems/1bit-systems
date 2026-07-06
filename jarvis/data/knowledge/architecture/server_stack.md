---
type: Architecture
title: Server Stack
description: Python FastAPI server with WebSocket streaming, tool calling, conversation memory, and Open Knowledge Format integration.
tags: [server, fastapi, python, websocket, architecture]
timestamp: 2026-07-06T00:00:00Z
---

# Overview

The JARVIS server is a Python FastAPI application (`server/server.py`) serving as the orchestration layer between the web UI and inference engines.

## Source Files

| File | Purpose |
|------|---------|
| `server/server.py` | FastAPI app, HTTP/WS endpoints, file upload, knowledge API |
| `server/agent.py` | JarvisAgent — LLM orchestration, tool calling, conversation memory |
| `server/knowledge.py` | OpenKnowledge — CRUD, full-text search, markdown persistence |
| `server/voice.py` | VoiceIO — Whisper STT + Piper TTS |
| `server/config.py` | JarvisConfig — dataclass + env-driven load_config() |

## Key Endpoints

### HTTP

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` or `/chat` | GET | Serve web UI (index.html) |
| `/api/status` | GET | System status — check NPU, GPU, TTS backends |
| `/api/chat` | POST | Send message (form data), stream NDJSON response |
| `/api/upload` | POST | Upload file for RAG or vision |
| `/api/knowledge/*` | GET/POST/DELETE | Knowledge base CRUD + search |

### WebSocket

| Endpoint | Purpose |
|----------|---------|
| `/ws/chat` | Bidirectional streaming: chat, STT, TTS, RAG, clear |

Message types:
- `message` — Chat text (optionally with base64 image)
- `stt` — Audio bytes → transcribed text
- `tts` — Text → synthesized audio bytes
- `rag_query` — Query string → knowledge results
- `clear` — Reset conversation

## Conversation Memory

Per-session message history stored in `conversations` dict (in-memory). Up to `max_history` (50) messages. Trimmed oldest-first (system prompt always preserved).

## Citations

[1] [System Architecture](/architecture/system_architecture.md)
[2] [Data Flow](/architecture/data_flow.md)
[3] [API Reference](/references/api_reference.md)

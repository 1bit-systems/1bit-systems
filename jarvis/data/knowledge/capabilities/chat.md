---
type: Capability
title: Chat / LLM
description: NPU-powered chat at 94 tok/s via Qwen3-0.6B on XDNA2. Streaming response via WebSocket or HTTP SSE. Conversation memory.
tags: [chat, llm, npu, streaming, conversation]
timestamp: 2026-07-06T00:00:00Z
---

# Overview

Primary chat capability powered by the NPU (XDNA2) running Qwen3-0.6B at 94 tok/s via FLM.

## Models

| Model | Backend | Speed | Purpose |
|-------|---------|-------|---------|
| Qwen3-0.6B | FLM NPU (:52625) | 94 tok/s | Primary chat |
| Qwen3-0.6B-NPU2 | Fused Engine (:9090) | 97 tok/s | Fallback (C++ v12) |
| Qwen3-VL-4B | FLM NPU | 11 tok/s | Vision + chat |

## Streaming

Responses are streamed via:
- **HTTP**: `POST /api/chat` returns `application/x-ndjson` (newline-delimited JSON)
- **WebSocket**: `ws://host:8080/ws/chat` with bidirectional streaming

Each chunk: `{"type": "chunk", "text": "..."}` with final `{"type": "done", "session_id": "..."}`.

## Conversation Memory

- In-memory per-session history
- Max 50 messages, system prompt always preserved
- Session ID passed by client or auto-generated

## Backend Selection

1. Try FLM NPU on port 52625 first
2. Fall back to Fused Engine on port 9090
3. Return error message if both unreachable

## Citations

[1] [System Architecture](/architecture/system_architecture.md)
[2] [Voice I/O](/capabilities/voice.md)
[3] [Vision](/capabilities/vision.md)

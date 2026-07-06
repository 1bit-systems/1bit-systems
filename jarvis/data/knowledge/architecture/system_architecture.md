---
type: Architecture
title: System Architecture
description: Full-stack architecture — Web UI (any browser) → Python orchestrator → Fused NPU+GPU+CPU engine. All local, zero cloud.
tags: [architecture, overview, stack]
timestamp: 2026-07-06T00:00:00Z
---

# Overview

JARVIS is a fully-local private AI assistant running on the 1bit.systems fused NPU+GPU+CPU stack. Three layers: web frontend, Python orchestrator, inference engines.

## Layer Diagram

```
┌─────────────────────────────────────────────────┐
│              JARVIS Web UI (any browser)          │
│  Chat · Voice · Vision · RAG · Tools · Status     │
└──────────────────────┬──────────────────────────┘
                       │ HTTP / WebSocket :8080
┌──────────────────────▼──────────────────────────┐
│            JARVIS Orchestrator (Python)           │
│  FastAPI · Agent · Knowledge · TTS/STT · Vision  │
└──────────────────────┬──────────────────────────┘
                       │ HTTP :52625 (FLM) / :9090 (Fused)
┌──────────────────────▼──────────────────────────┐
│            Fused Engine (C++ / Zig)              │
│   NPU (XDNA2) ←→ GPU (Radeon 8060S) ←→ CPU     │
│   Qwen3-0.6B · Whisper · Piper · Qwen3-VL-4B    │
└─────────────────────────────────────────────────┘
```

## Layer 1: Web UI

Single-page HTML/JS application at `web/index.html`. Communicates with the orchestrator via HTTP POST (`/api/chat`) and WebSocket (`/ws/chat`). No build step — open the file directly.

## Layer 2: Python Orchestrator

FastAPI server at `server/server.py` with:

- **Agent** (`server/agent.py`) — LLM orchestration with conversation memory, tool calling, RAG integration
- **Knowledge** (`server/knowledge.py`) — Open Knowledge Format: CRUD, full-text search, markdown persistence
- **Voice** (`server/voice.py`) — Whisper STT + Piper TTS
- **Config** (`server/config.py`) — Environment-driven configuration

## Layer 3: Inference Engines

- **NPU (FLM)**: Primary chat path, 94 tok/s on XDNA2
- **Fused Engine**: Unified daemon on port 9090 (fallback path)
- **GPU (llama.cpp)**: Via Radeon 8060S Vulkan backend
- **CPU (Zen 5)**: Piper TTS, file operations

## Citations

[1] [Server Stack](/architecture/server_stack.md)
[2] [Data Flow](/architecture/data_flow.md)
[3] [Fused Engine](https://github.com/1bit-systems/1bit/tree/main/engine/fusion/) — the underlying NPU+GPU hybrid engine

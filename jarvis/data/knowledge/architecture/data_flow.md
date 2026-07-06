---
type: Architecture
title: Data Flow
description: Request lifecycle for all capabilities — chat, voice, vision, RAG, and tool execution through the NPU/GPU stack.
tags: [data-flow, lifecycle, architecture]
timestamp: 2026-07-06T00:00:00Z
---

# Overview

How data flows through the JARVIS stack for each capability.

## Chat Flow

```
User Message ──▶ WebSocket / HTTP POST
                      │
                      ▼
              JarvisAgent.chat_stream()
                      │
            ┌─────────┴──────────┐
            │                    │
            ▼                    ▼
      FLM NPU (:52625)    Fused Engine (:9090)
      Qwen3-0.6B          (fallback)
            │                    │
            └─────────┬──────────┘
                      │
                      ▼
              Streaming response
                      │
                      ▼
              Web UI (browser)
```

## Voice Flow (STT)

```
Microphone ──▶ Web Audio API ──▶ PCM bytes ──▶ WebSocket "stt"
                                                      │
                                                      ▼
                                              whisper.cpp server
                                              or FLM /v1/audio/transcriptions
                                                      │
                                                      ▼
                                              Transcribed text
                                                      │
                                                      ▼
                                              Chat flow (above)
```

## Voice Flow (TTS)

```
LLM Response ──▶ WebSocket {type:"tts", text:"..."}
                        │
                        ▼
                 PiperVoice.synthesize(text)
                        │
                        ▼
                 WAV bytes ──▶ WebSocket audio chunk
                        │
                        ▼
                 Web Audio API playback
```

## Vision Flow

```
Image + Text ──▶ WebSocket /api/chat
                      │
                      ▼
              User content with image_url
              (base64-JPEG data URI)
                      │
                      ▼
              LLM processes multimodal input
              (Qwen3-VL-4B via FLM)
                      │
                      ▼
              Streaming text response
```

## RAG Flow

```
User Query ──▶ agent.rag_query(query)
                      │
                      ▼
              knowledge.search(query)
              (full-text on content+title+tags)
                      │
                      ▼
              Top-K results formatted as context
                      │
                      ▼
              Prepended to LLM prompt
                      │
                      ▼
              Chat flow (above)
```

## Tool Execution Flow

```
LLM generates tool call ──▶ agent._execute_tool()
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
Calculator  Python    File Ops
              │
    ┌─────────┴─────────┐
    ▼                   ▼
Result             Error message
    │
    ▼
Injected back into LLM context
```

## Citations

[1] [Chat](/capabilities/chat.md)
[2] [Voice I/O](/capabilities/voice.md)
[3] [RAG](/capabilities/rag.md)
[4] [Tool Calling](/capabilities/tool_calling.md)

---
type: Guide
title: Configuration
description: Environment variables, config dataclass, and all tunable parameters for JARVIS.
tags: [config, environment, setup]
timestamp: 2026-07-06T00:00:00Z
---

# Overview

JARVIS is configured via environment variables (loaded by `server/config.py` → `load_config()`).

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JARVIS_API_BASE` | `http://127.0.0.1:9090` | Fused Engine API base URL |
| `JARVIS_LLM_MODEL` | `qwen3:0.6b` | LLM model ID for FLM |
| `JARVIS_PORT` | `8080` | JARVIS server port |
| `JARVIS_DEBUG` | `1` | Enable debug logging |

## Config Dataclass

| Field | Default | Description |
|-------|---------|-------------|
| `api_base` | `http://127.0.0.1:9090` | Unified daemon endpoint |
| `flm_port` | `52625` | FLM NPU server port |
| `llm_model` | `qwen3:0.6b` | Primary chat model |
| `vision_model` | `qwen3vl-it:4b` | Vision model |
| `embed_model` | `embed-gemma:300m` | Embedding model |
| `asr_model` | `whisper-v3:turbo` | Speech recognition model |
| `host` | `0.0.0.0` | Server bind address |
| `port` | `8080` | Server port |
| `max_history` | `50` | Max conversation messages |
| `max_tokens` | `2048` | Max response tokens |
| `temperature` | `0.7` | LLM temperature |
| `tools_enabled` | `[calculator, web_search, python_exec, read_file, list_dir]` | Available tools |

## Citations

[1] [Quick Start](/guides/quick_start.md)
[2] [Server Stack](/architecture/server_stack.md)

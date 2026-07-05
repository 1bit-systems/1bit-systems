# JARVIS — Private AI

**Your private AI assistant. On your hardware. Zero cloud.**

JARVIS is a fully-local AI assistant that runs on the 1bit.systems NPU+GPU+CPU stack. Chat, voice, vision, RAG, and tool-using agents — all on-device. 94 tok/s on an AMD Strix Halo laptop.

```
1bit.systems  ──  Fused NPU+GPU+CPU engine
       │
       └── JARVIS  ──  Your private AI assistant
             │
             ├── Chat (NPU: 94 tok/s)
             ├── Voice (Whisper STT + Piper TTS)
             ├── Vision (Qwen3-VL-4B on NPU)
             ├── RAG (Open Knowledge Format)
             └── Agents (Tool calling, Python exec)
```

## Quick Start

```bash
# 1. Start the NPU backend
sudo flm serve qwen3:0.6b --port 52625 --pmode turbo

# 2. Start JARVIS
source jarvis-env/bin/activate
python3 server.py

# 3. Open http://localhost:8080/chat
```

## Capabilities

| Capability | Engine | Speed |
|-----------|--------|-------|
| Chat/LLM | NPU (FLM) | 94 tok/s |
| Speech-to-Text | NPU (Whisper-v3) | ~1.5s |
| Text-to-Speech | CPU (Piper) | ~50ms |
| Vision | NPU (Qwen3-VL-4B) | 11 tok/s |
| RAG | Open Knowledge Format | Instant |
| Tool Calling | NPU + Python | On-demand |

## Architecture

```
┌─────────────────────────────────────────────────┐
│              JARVIS Web UI (any browser)          │
│  Chat · Voice · Vision · RAG · Tools · Status     │
└──────────────────────┬──────────────────────────┘
                       │ HTTP / WebSocket :8080
┌──────────────────────▼──────────────────────────┐
│            JARVIS Orchestrator (Python)           │
│  Agent · Open Knowledge · TTS/STT · Vision       │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│            Fused Engine (port 9090)              │
│   NPU (FLM) ←→ GPU (llama.cpp) ←→ CPU (Zen 5)  │
└─────────────────────────────────────────────────┘
```

## Open Knowledge Format

All knowledge is stored as human-readable markdown files with YAML frontmatter:

```markdown
---
type: fact
tags: [npu, benchmark]
confidence: 0.95
---
# NPU Speed
Qwen3-0.6B: 94 tok/s on XDNA2
```

- **No lock-in** — plain markdown, edit with any text editor
- **Git-friendly** — version your knowledge
- **Full-text search** — built-in keyword search

## Project Structure

```
jarvis/
├── server/              # Python FastAPI server
│   ├── server.py        # Main entry point
│   ├── agent.py         # LLM agent with tool calling
│   ├── voice.py         # STT (Whisper) + TTS (Piper)
│   └── knowledge.py     # Open Knowledge Format
├── web/                 # Web UI (single HTML file)
│   └── index.html       # JARVIS chat interface
├── mobile/              # Flutter mobile app
├── data/                # Runtime data
│   └── knowledge/       # Open Knowledge files
├── scripts/             # Install and utility scripts
└── docs/                # Documentation
```

## Links

- [1bit.systems](https://1bit.systems) — The fused NPU+GPU+CPU engine
- [GitHub](https://github.com/bong-water-water-bong/1bit-systems)
- [Discord](https://discord.gg/dSyV646eBs)
- Email: admin@1bit.systems

---

*Built with DeepSeek v4 (99.9%) · Shipped with Claude (0.1%) · One human.*
*"Sorry but not Sorry :)"*

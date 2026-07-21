# JARVIS — Local Private AI Agent

A fully local, private AI agent server built on the `1bit-systems` inference
engine. All core inference runs on-device against the local NPU (`npu_xrt`
via the FLM bridge) and/or GPU (ROCm, via Ollama) backends on AMD Radeon
hardware — no cloud dependency, no external API calls for any core function.

## Capabilities

- **Retrieval-augmented generation (RAG)** — full-text search over a local
  markdown knowledge base (`jarvis/rag.py`). Documents can be uploaded via
  API and are automatically injected as context for relevant queries.
- **Local multi-turn memory** — each conversation (`session_id`) is persisted
  server-side to `conversations/<session_id>.md` and recalled on every
  subsequent request to that session, independent of whatever history a
  given client happens to send. This means a phone client, a curl script,
  and a desktop UI hitting the same session all share continuity.
- **Tool invocation** (`jarvis/tools.py`) — the model can call `search_knowledge`,
  `get_time`, `list_models`, or `add_note` via a `TOOL_CALL: {...}` directive
  in its output; the server parses, executes, and feeds the result back for
  a grounded final answer.
- **Multi-step task planning** (`jarvis/planner.py`) — complex requests are
  decomposed into an ordered list of subtasks by a fast local model, each
  subtask is routed to whichever model in the local roster actually fits it
  (vision → `qwen3vl`, heavy reasoning → the larger model, everything else →
  the fast default), and a synthesis pass combines the results — grounded on
  the actual tool outputs, not just each subtask model's own paraphrase of
  them. Because the underlying engine can hold multiple of these models
  resident at once, steps don't pay a cold-load penalty between them.
- **Permission & privacy control** — tools are classed `safe` (read-only,
  always runs) or `sensitive` (mutates local state); sensitive tools only
  execute if the request explicitly passes `allow_write: true`. Every tool
  call — allowed or denied — is appended to a local-only audit log
  (`<knowledge_dir>/tools/audit.log`) that is never transmitted anywhere.
- **Voice** (`jarvis/stt.py`, `jarvis/tts.py`, `jarvis/voice/`) — local
  speech-to-text (faster-whisper) and text-to-speech (Piper, plus a custom
  voice-cloning engine), all on-device.

## Architecture

```
Client (curl / 1bit Mobile app / web UI)
        │  HTTP (OpenAI-compatible /v1/chat/completions, plus /v1/agent/plan)
        ▼
jarvis/server.py  ── session memory (rag.py) ── knowledge base (rag.py)
        │                                              ▲
        ├─ tool-call loop (tools.py) ──────────────────┘  (search_knowledge, add_note)
        ├─ multi-step planner (planner.py) ── routes subtasks across models
        ▼
jarvis/routing.py
        ├─ NPU backend  → npu_xrt engine via FLM bridge  (AMD XDNA 2 NPU)
        └─ GPU backend  → Ollama /api/chat               (AMD Radeon GPU, ROCm)
```

## Setup

Requires Python 3.10+ and a running local backend:

- **NPU backend**: the `1bit-systems` NPU engine / FLM bridge listening on
  `NPU_URL` (default `http://127.0.0.1:52625`).
- **GPU backend**: [Ollama](https://ollama.com) listening on `OLLAMA_URL`
  (default `http://127.0.0.1:11434`) with whichever models from
  `jarvis/routing.py`'s `MODEL_ROUTING` table you want to use pulled locally
  (`ollama pull qwen3.5:9b`, etc.).

### Dependencies

Core server (`server.py`, `rag.py`, `routing.py`, `tools.py`, `planner.py`)
is pure standard library — no install step needed beyond Python itself.

Voice features additionally need:
```
pip install faster-whisper piper-tts torch numpy soundfile
```
(STT uses `faster-whisper`; TTS uses the `piper` CLI plus an optional custom
voice-cloning engine under `jarvis/voice/` built on `torch`/`soundfile`.)

### Run

```bash
cd 1bit-systems
JARVIS_PORT=8080 python3 -m jarvis.server
# → http://localhost:8080/chat
```

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `NPU_URL` | `http://127.0.0.1:52625` | NPU/FLM backend |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | GPU/Ollama backend |
| `JARVIS_PORT` | `8080` | HTTP listen port |
| `JARVIS_KNOWLEDGE_DIR` | `~/jarvis/data/knowledge` | RAG + memory + audit log storage |
| `JARVIS_VENV` | `~/jarvis-env` | venv used for STT/TTS optional deps |
| `PIPER_VOICES_DIR` | `~/piper-voices` | Piper TTS voice models |

## API

- `POST /v1/chat/completions` (OpenAI-compatible) — body accepts the usual
  `model`/`messages`/`stream`/`max_tokens`/`temperature`, plus:
  - `session_id` (str, default `"default"`) — which memory thread to use
  - `rag` (bool, default `true`) — inject knowledge-base context
  - `tools` (bool, default `true` for non-streaming requests) — enable tool calling
  - `allow_write` (bool, default `false`) — permit sensitive tools (`add_note`)
- `POST /v1/agent/plan` — `{"request": "...", "session_id": "...", "allow_write": false}`
  → decomposes and executes a multi-step plan, returns `{"plan": [...], "steps": [...], "answer": "..."}`
- `POST /v1/knowledge/search`, `POST /v1/knowledge/upload`, `GET /v1/knowledge` — RAG
- `POST /v1/audio/transcriptions`, `POST /v1/audio/speech` — STT/TTS
- `GET /v1/models` — list routable models
- `GET /health` — liveness

## Model roster

See `jarvis/routing.py`'s `MODEL_ROUTING` for the full table. The engine
can hold multiple of these resident at once, which is what the multi-step
planner relies on to route different subtasks to different models without
a reload between steps.

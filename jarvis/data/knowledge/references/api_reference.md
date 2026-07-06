---
type: Reference
title: API Reference
description: Complete HTTP and WebSocket API reference for JARVIS — status, chat, upload, knowledge CRUD, voice.
tags: [api, reference, http, websocket, endpoints]
timestamp: 2026-07-06T00:00:00Z
---

# API Reference

## HTTP Endpoints

### GET `/api/status`

System health check. Returns backend status for fused engine, FLM NPU, and TTS.

```bash
curl http://localhost:8080/api/status
```

### POST `/api/chat`

Send a chat message (form data). Returns streaming NDJSON.

```bash
curl -X POST http://localhost:8080/api/chat \
  -F "message=Hello" \
  -F "session_id=abc123"
```

**Parameters**:
- `message` (required): Text message
- `session_id` (optional): Session identifier
- `image` (optional): Image file upload

### POST `/api/upload`

Upload a file.

```bash
curl -X POST http://localhost:8080/api/upload \
  -F "file=@document.pdf" \
  -F "category=rag"
```

**Parameters**:
- `file` (required): File data
- `category` (required): `rag` or `image`

### GET `/api/knowledge/search`

Search the knowledge base.

```bash
curl "http://localhost:8080/api/knowledge/search?q=npu+benchmark&max=5"
```

### GET `/api/knowledge/stats`

Knowledge base statistics.

### GET `/api/knowledge/list`

List all entries, optionally filtered by type.

```bash
curl "http://localhost:8080/api/knowledge/list?type=fact"
```

### GET `/api/knowledge/read`

Read a specific entry by path.

```bash
curl "http://localhost:8080/api/knowledge/read?path=data/knowledge/facts/my_fact.md"
```

### POST `/api/knowledge/add`

Create a new knowledge entry (form data).

### DELETE `/api/knowledge/delete`

Delete a knowledge entry by path.

### POST `/api/tts/download`

Download a TTS voice model.

## WebSocket `/ws/chat`

Bidirectional streaming protocol. Message types:

| Type | Direction | Payload |
|------|-----------|---------|
| `message` | Client→Server | `{text, session_id?, image?}` |
| `stt` | Client→Server | `{audio: "<base64>"}` |
| `tts` | Client→Server | `{text}` |
| `rag_query` | Client→Server | `{query}` |
| `clear` | Client→Server | `{session_id}` |
| `chunk` | Server→Client | `{text, session_id}` |
| `done` | Server→Client | `{session_id}` |
| `stt_result` | Server→Client | `{text}` |
| `tts_result` | Server→Client | `{audio: "<base64>"}` |
| `rag_result` | Server→Client | `{results}` |
| `cleared` | Server→Client | `{session_id}` |

## Citations

[1] [Server Stack](/architecture/server_stack.md)
[2] [Tool Reference](/references/tool_reference.md)

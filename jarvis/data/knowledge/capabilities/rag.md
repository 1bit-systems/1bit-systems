---
type: Capability
title: RAG (Retrieval-Augmented Generation)
description: Search through uploaded documents and knowledge entries using full-text keyword search. Results injected into LLM context.
tags: [rag, search, retrieval, knowledge, context]
timestamp: 2026-07-06T00:00:00Z
---

# Overview

RAG enables JARVIS to answer questions using its stored knowledge — facts, documents, conversation logs, and tool outputs — by retrieving relevant entries and injecting them into the LLM prompt.

## Search Engine

Full-text keyword search on:

| Field | Weight |
|-------|--------|
| Title | ×3 per match |
| Tags | ×2 per match |
| Content body | ×0.5 per occurrence |

Source: `server/knowledge.py` → `OpenKnowledge.search()`

## Knowledge Types

| Type | Directory | Purpose |
|------|-----------|---------|
| `fact` | `knowledge/facts/` | Structured facts learned about the system |
| `document` | `knowledge/documents/` | Uploaded RAG documents |
| `conversation` | `knowledge/conversations/` | Chat history logs |
| `tool_output` | `knowledge/tools/` | Tool execution snapshots |

## RAG Query Flow

1. User sends `rag_query` via WebSocket
2. `agent.rag_query()` calls `knowledge.search(query, top_k=5)`
3. Results formatted as markdown sections with type badges
4. Context prepended to LLM system message
5. LLM generates answer informed by knowledge

## Citations

[1] [Data Flow](/architecture/data_flow.md)
[2] [Open Knowledge](/capabilities/open_knowledge.md)
[3] [Knowledge Format](/references/knowledge_format.md)

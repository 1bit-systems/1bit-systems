---
type: Capability
title: Open Knowledge Format
description: All knowledge stored as human-readable markdown files with YAML frontmatter. Edit with any text editor. No lock-in.
tags: [knowledge, okf, markdown, yaml, persistence]
timestamp: 2026-07-06T00:00:00Z
---

# Overview

JARVIS uses the Open Knowledge Format (OKF) — a transparent, file-based knowledge system. Every piece of knowledge is a human-readable `.md` file with YAML frontmatter.

## Directory Structure

```
data/knowledge/
├── index.json              # Full-text search index (auto-generated)
├── index.md                # OKF bundle index (manual)
├── conversations/          # Chat histories (markdown logs)
├── documents/              # Uploaded RAG documents
├── facts/                  # Structured facts JARVIS learned
├── tools/                  # Tool output snapshots
├── architecture/           # System architecture docs (OKF concepts)
├── capabilities/           # Capability docs (OKF concepts)
├── guides/                 # Setup guides (OKF concepts)
└── references/             # Reference docs (OKF concepts)
```

## File Format

```markdown
---
type: fact | document | conversation | tool_output
created: 2026-07-05T18:30:00Z
tags: [npu, performance, benchmark]
source: user | web | tool | inference
confidence: 0.0-1.0
---
# Title
Content in markdown...
```

## CRUD Operations

All exposed via HTTP API:

- `GET /api/knowledge/search?q=query` — Full-text search
- `GET /api/knowledge/list?type=fact` — List by type
- `GET /api/knowledge/read?path=...` — Read entry
- `POST /api/knowledge/add` — Create entry
- `DELETE /api/knowledge/delete` — Delete entry

## Index

Auto-generated `index.json` with:
- Version, format metadata
- Entry list with type, title, tags, source, confidence
- Size tracking

## Citations

[1] [RAG](/capabilities/rag.md)
[2] [Knowledge Format](/references/knowledge_format.md)
[3] [OKF Specification](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)

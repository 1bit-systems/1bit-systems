---
type: Reference
title: Knowledge Format
description: The markdown + YAML frontmatter format for all JARVIS knowledge entries. Human-readable, git-friendly, no lock-in.
tags: [reference, format, markdown, yaml, okf]
timestamp: 2026-07-06T00:00:00Z
---

# Knowledge Format Reference

## File Path Convention

```
data/knowledge/{type}/{slug}_{YYYYMMDD_HHMMSS}.md
```

Where `{slug}` is the title sanitized (spaces → underscores, non-alphanumeric removed).

## Frontmatter Fields

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `type` | Yes | string | `fact`, `document`, `conversation`, or `tool_output` |
| `created` | Auto | ISO 8601 | Creation timestamp |
| `updated` | Auto | ISO 8601 | Last update timestamp |
| `tags` | No | list | Search keywords |
| `source` | No | string | `user`, `web`, `tool`, or `inference` |
| `confidence` | No | float | 0.0–1.0 confidence score |

## Body Format

Standard markdown. First `# Heading` becomes the title.

```markdown
# Title
Body content in markdown...
```

## Index

`index.json` auto-generated with all entries for fast lookup:

```json
{
  "version": 2,
  "format": "open-knowledge-markdown",
  "updated": "2026-07-05T18:30:00Z",
  "entry_count": 42,
  "entries": {
    "data/knowledge/facts/example_20260705_183000.md": {
      "type": "fact",
      "title": "Example Fact",
      "tags": ["npu", "performance"],
      "source": "inference",
      "confidence": 0.9,
      "created": "2026-07-05T18:30:00Z",
      "size_bytes": 512
    }
  }
}
```

## Citations

[1] [Open Knowledge](/capabilities/open_knowledge.md)
[2] [OKF Specification](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)

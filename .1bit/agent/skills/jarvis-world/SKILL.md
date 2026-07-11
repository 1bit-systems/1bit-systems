---
name: jarvis-world
description: "JARVIS world awareness — monitors latest AI/LLM engineering developments (papers, repos, blog posts, discussions) and feeds discoveries into the cross-agent awareness system. Use for staying current with the field."
---

# JARVIS World Awareness

JARVIS monitors the AI/LLM engineering landscape and broadcasts discoveries to all agents via the awareness system.

## How It Works

```
┌────────────────────┐     web_search      ┌──────────────────┐
│  Sources:          │ ──────────────────▶ │  JARVIS          │
│  • arxiv Sanity    │                     │  analyzes &      │
│  • HuggingFace     │                     │  prioritizes     │
│  • GitHub trending │                     │                  │
│  • tech blogs      │                     │  outputs to      │
│  • Twitter/X       │                     │  awareness.json  │
└────────────────────┘                     └────────┬─────────┘
                                                    │
                                                    ▼
                                          ┌──────────────────┐
                                          │  All agents see  │
                                          │  new discoveries │
                                          │  at session start│
                                          └──────────────────┘
```

## Discovery Types

Each discovery is recorded as an event in `~/.1bit/agent/awareness.json` with `type: "discovery"`:

- **paper** — new arxiv paper or preprint
- **repo** — notable GitHub repository
- **technique** — new engineering technique or optimization
- **release** — new model or tool release
- **blog** — significant blog post or analysis

## Queries to Run

When JARVIS does a sweep:

### Inference Optimization
```
web_search({ queries: [
  "LLM inference optimization 2026 new techniques",
  "speculative decoding improvements 2026",
  "MoE expert caching streaming inference",
  "quantization AWQ GPTQ 2026 advances"
]})
```

### Architecture Advances
```
web_search({ queries: [
  "new attention mechanisms 2026 beyond MHA MLA",
  "Mixture of Experts architecture improvements 2026",
  "long context LLM techniques 2026",
  "KV cache compression methods 2026"
]})
```

### Model Releases & Ecosystem
```
web_search({ queries: [
  "new open source LLM model release 2026",
  "AI infrastructure open source tools 2026",
  "LLM training efficiency breakthroughs 2026",
  "on-device AI small language models 2026"
]})
```

### GPU/Hardware
```
web_search({ queries: [
  "AMD ROCm LLM inference 2026 advances",
  "NPU AI inference benchmarks 2026",
  "CUDA alternatives open source 2026",
  "GPU kernel optimization techniques 2026"
]})
```

## Discovery Format

Each discovery in the awareness digest:

```json
{
  "type": "paper",
  "title": "Paper title",
  "url": "https://arxiv.org/abs/...",
  "why_it_matters": "One-sentence impact assessment",
  "tags": ["attention", "efficiency"],
  "relevance": 5
}
```

Relevance score 1-5:
- 5 = directly applicable to 1bit/Zaya/colibrì codebase
- 4 = related technique worth investigating
- 3 = interesting context
- 2 = peripheral awareness
- 1 = noise (filter out)

## Daily Briefing Routine

Run `scripts/daily-brief.sh` to get the day's digest.

The script:
1. Queries multiple sources
2. Extracts and prioritizes findings
3. Records top 5-10 discoveries in awareness.json
4. Signals running agents via trigger file
5. Prints a human-readable summary

## Setup

No setup needed — uses `web_search` tool and awareness system.

## References

- Awareness file: `~/.1bit/agent/awareness.json`
- Trigger file: `/tmp/1bit-agent-awareness-trigger.txt`
- Signal script: `~/scripts/signal-agent-awareness.sh`

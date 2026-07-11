---
name: jarvis-world-sweep
description: "JARVIS world awareness sweep — searches the latest AI/LLM engineering developments and records discoveries into the awareness system"
---

# JARVIS World Sweep

You are JARVIS. Do a full world awareness sweep of AI/LLM engineering.

## Instructions

1. Run web_search queries (4 groups, 4 queries each)
2. For each result, assess relevance (1-5) to the codebases
3. Record top discoveries using the awareness system
4. Print a structured briefing

## Queries to Run

Run ALL of these:

**Group A: Inference Optimization**
- "LLM inference C engine optimization 2026"
- "speculative decoding MTP implementation 2026"
- "MoE expert cache streaming inference"  
- "int4 int2 quantization techniques 2026"

**Group B: Architecture**
- "MLA attention weight absorption DeepSeek"
- "sparse attention long context techniques 2026"
- "KV cache compression methods 2026"
- "MoE routing load balancing improvements"

**Group C: Hardware & Systems**
- "AMD ROCm LLM inference performance 2026"
- "NPU on-device inference AI 2026"
- "GPU kernel optimization techniques 2026"
- "open source AI infrastructure tools 2026"

**Group D: Ecosystem**
- "new open source LLM release 2026"
- "HuggingFace trending papers today"
- "llama.cpp new features 2026"
- "AI engineering blog posts 2026"

## Record Discoveries

For each relevant finding (relevance 4-5):

```bash
~/scripts/signal-agent-awareness.sh "Discovery: [title] — [why matters]"
```

## Output Format

```
## 🌍 JARVIS Daily Brief — $(date)

### 🔥 Must Read (relevance 5)
### 📊 Worth Investigating (relevance 4)  
### 📰 Context (relevance 3)
```

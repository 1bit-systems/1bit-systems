---
name: 1bit-npu
description: Manage the 1bit NPU inference stack — start, stop, check status, and query models running on the local Strix Halo NPU.
---

# 1bit NPU Stack Management

## Quick Reference

```bash
# Start everything
1bit up

# Check status
1bit status

# Stop everything
1bit down

# Query the NPU endpoint
curl http://127.0.0.1:9090/v1/chat/completions \
  -d '{"model":"qwen3-0.6b-FLM","messages":[{"role":"user","content":"Hello"}],"stream":false}'
```

## What's Running

| Service | Port | URL |
|---------|------|-----|
| 1bit API Bridge | 9090 | `http://127.0.0.1:9090/` |
| Lemond (Chat UI) | 13305 | `http://127.0.0.1:13305/` |
| Lemond WebSocket | 9000 | `ws://127.0.0.1:9000/` |

## Zaya1 74B — iGPU (ROCm) Backend

On port 8081 via llama.cpp Zaya fork. ~18 tok/s on Radeon 8060S.

```bash
# Start Zaya server
1bit zaya

# Query Zaya endpoint
curl http://127.0.0.1:8081/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":false}'
```

### Memory note
Zaya1 74B Q4_K_M is 42.6 GiB and loads entirely in VRAM (63 GiB total on Radeon 8060S), leaving ~15 GiB for KV cache + compute. Context limited to 8192 tokens by default to keep headroom.

## Models available on NPU

- `qwen3-0.6b-FLM` (0.6B) — default, fastest
- `qwen3vl-it-4b-FLM` (4B VL)
- `gemma4-it-e2b-FLM` (4B)
- `llama3.1-8b-FLM` (8B)
- `Bonsai-1.7B-IQ1_S` (1.7B)
- `bonsai-1.7b` (1.7B)

## When to use the NPU vs cloud

- **NPU**: Coding questions, file operations, git, shell commands, light reasoning
- **Cloud**: Heavy reasoning, creative writing, large context tasks

Always prefer local NPU inference unless the user explicitly requests a cloud model or the task requires capabilities beyond the local model.

# FastFlowLM (FLM) Reverse Engineering Analysis

## Overview

This directory contains the reverse engineering analysis of **FastFlowLM (FLM)** v0.9.24 — AMD's commercial C++ LLM inference engine for XDNA2 NPUs. The analysis was performed from the leaked source code available in the `Reverse Engineer Fastflow LM` working directory.

## Key Findings

### Architecture
- **Language**: C++20 with CMake build system
- **NPU Interface**: AMD XDNA2 via DRM ioctls (`/dev/accel/accel*`)
- **Driver**: XRT (Xilinx Runtime Library v2.21.75) + XDNA userspace plugin
- **Quantization**: Custom **Q4NX** 4-bit block format with bf16 scales and int32 zero-points
- **API**: Ollama-compatible REST API (port 52625) + OpenAI-compatible `/v1/chat/completions`
- **Models**: Qwen3, Llama3, Gemma3/4e, Phi4, Whisper, DeepSeek-R1, MoE, VLMs

### NPU Communication Stack
```
Model Code → npu_sequence → npu_app (compiles to ELF) → XRT → amdxdna_accel.h (DRM) → AIE Array
```

### XCLBIN Programs (Pre-compiled AIE Graphs)
Each model variant has dedicated NPU binaries:
| XCLBIN | Size | Purpose |
|---|---|---|
| `mm.xclbin` | 501 KB | Matrix multiply (GEMM) |
| `dequant.xclbin` | 112 KB | Q4NX → bf16 dequantization |
| `attn.xclbin` | 310 KB | Multi-head causal attention |
| `layer.xclbin` | 333 KB | Full transformer layer |
| `lm_head.xclbin` | 316 KB | Final projection (MoE models) |
| `vision_attn.xclbin` | 571 KB | Vision transformer attention |
| `vision_mm.xclbin` | 476 KB | Vision matmul/conv |
| `dequant_mm.xclbin` | 431 KB | Fused dequant+matmul (MoE) |

### Multi-Head Attention on NPU
The `attn.xclbin` implements a 5-stage pipeline across the AIE array:
1. **RoPE** — Rotary Position Embedding (CT00-07)
2. **Q·K^T** — Attention scores (CT00-07)
3. **Scale + Mask** — Causal masking (CT10-17)
4. **Softmax** — Online safe softmax (CT10-17)
5. **Score·V** — Weighted sum (CT20-37)

All 5 stages are pipelined for maximum throughput.

### KV Cache Management
- Per-layer K,V buffers pre-allocated at MAX_L size
- Checkpoint/restore for multi-turn conversation support
- `PromptCache` tracks conversation round checksums for prefix caching
- NPU preemption supports context switching between tasks

### Server Architecture
- Boost.Beast ASIO HTTP server with multiple I/O threads
- Single NPU with global mutex + bounded request queue
- SSE streaming via custom `std::streambuf` with UTF-8 boundary handling
- Client disconnect detection via TCP socket monitoring

## Files

- `q4nx_converter/` — Tools for converting models to/from Q4NX format
- See individual deep-dive documents in subdirectories

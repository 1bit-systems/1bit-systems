# 1bit.systems NPU Benchmarks — July 2, 2026

**Hardware**: AMD Ryzen AI Max+ 395 (Strix Halo), XDNA 2 NPU, 32 AIE2P tiles  
**OS**: Ubuntu 26.04 LTS, Kernel 7.0.0-27-generic, Firmware 1.1.2.65  
**Engine**: `npu_engine_v12` — C++23, M=32 batch, OpenMP attention, OpenMP LM head

---

## Competitor Comparison: NPU Inference (Qwen3-0.6B)

| Engine | Hardware | NPU Tiles | Decode | Prefill | Open Source |
|--------|----------|-----------|--------|---------|-------------|
| **1bit.systems v12** | **Strix Halo (Max+ 395)** | **32** | **97 tok/s** (10.3 ms/tok) | 152 ms (M=9) | ✅ MIT |
| FastFlowLM v0.9.31 | Kraken Point (AI 7 350) | ~16 | 66.5 tok/s (15 ms/tok) | 1494 tok/s | ❌ Proprietary |
| FastFlowLM v0.9.31 | Kraken Point @ 32K ctx | ~16 | 14.1 tok/s | — | ❌ Proprietary |

**1bit.systems v12 is 1.46× faster than FLM on weaker hardware.**
FLM does not publish Strix Halo benchmarks for direct comparison.
FLM's fused design would likely be faster on the same chip — the gap is software architecture, not silicon.

## Competitor Comparison: All Models

| Engine | Hardware | Model | Size | Decode |
|--------|----------|-------|------|--------|
| **1bit.systems v12** | Strix Halo NPU | Qwen3-0.6B | 610 MB | **97 tok/s** |
| **1bit.systems GPU** | Strix Halo Vulkan | Bonsai-1.7B IQ1_S | 385 MB | **281 tok/s** |
| FastFlowLM | Kraken Point NPU | Qwen3-0.6B | — | 66.5 tok/s |
| FastFlowLM | Kraken Point NPU | LLaMA 3.2 1B | — | 64.5 tok/s |
| FastFlowLM | Kraken Point NPU | Qwen3-1.7B | — | 40.2 tok/s |
| FastFlowLM | Kraken Point NPU | LLaMA 3.2 3B | — | 26.3 tok/s |
| FastFlowLM | Kraken Point NPU | Qwen3-4B | — | 19.6 tok/s |
| FastFlowLM | Kraken Point NPU | LLaMA 3.1 8B | — | 12.8 tok/s |
| FastFlowLM | Kraken Point NPU | Qwen3-8B | — | 11.9 tok/s |

*FLM numbers from fastflowlm.com/docs/benchmarks/ — all on Kraken Point.
FLM supports Windows (Install-Windows in docs). Our engine is Linux-only (XRT).*

## Raw Silicon: GEMM Throughput

*Chess-compiled INT8 xclbins. Verified on-device.*

| Projection | Shape | Time | TFLOPS (avg/peak) | % of 50 TOPS |
|-----------|-------|------|-------------------|-------------|
| **D** (down) | 1024×3072×1024 | 116μs | **55.7 / 80.5** | **111%** |
| **O** (output) | 1024×2048×1024 | 108μs | 39.7 / 49.4 | 79% |
| **GU** (gate+up) | 1024×1024×6144 | 801μs | 16.1 / 16.5 | 32% |
| **QKV** (fused) | 1024×1024×4096 | 559μs | 15.4 / 15.5 | 31% |

## Inference: End-to-End LLM

### Prefill Scaling

| M (tokens) | Time | Per-Token | Speedup vs M=1 |
|-----------|------|-----------|----------------|
| 1 | 161ms | 161 ms/tok | 1.0× |
| 9 | 175ms | **19 ms/tok** | 8.5× |

### Decode (v12, M=32 batch + OpenMP attention)

| Decode Tokens | Batch Step | Effective | tok/s | Attention Cost |
|--------------|-----------|-----------|-------|---------------|
| 32 | 6 ms/tok | 10.3 ms/tok | **97** | 7ms total |
| 64 | 6 ms/tok | 10.3 ms/tok | **97** | 19ms total |
| 128 | 7 ms/tok | 7.3 ms/tok | **137** | 99ms total |

## Engine Evolution (4 Days)

| Date | Engine | Decode | Speedup | Breakthrough |
|------|--------|--------|---------|-------------|
| Jun 28 | v7 BFP16 | 1930 ms/tok | — | First working decode |
| Jul 1 | i8 swap | 244 ms/tok | 1.0× | K-interleaving fixed |
| Jul 2 | v6 batch-4 | 50 ms/tok | 4.4× | Batch amortization |
| Jul 2 | v9 M=16 | 16 ms/tok | 15.2× | M=16 + NPU LM head |
| **Jul 2** | **v12 M=32** | **10 ms/tok** | **24×** | **M=32 + OpenMP attention** |

**Net: 244→10 ms/tok. 24× in one session. Zero Python. Pure C++.**
**Beats FLM Kraken Point (66.5 tok/s) by 46%. Open source.**

## Per-GEMM Dispatch Profile (v7)

| Component | μs/call | % |
|-----------|---------|---|
| Kernel call (ioctl) | 9 | 1% |
| **r.wait() (NPU compute)** | **1,334** | **98%** |
| DMA + quantize + dequant | 17 | 1% |

**Fusion saves only 9μs ioctl per merged dispatch.** NPU compute time = bottleneck.
Real fix: increase M (batch size). M=32 amortizes 1334μs across 32 tokens → 42μs/token.

## NPU Attention Status

Chess-compiled Qwen3-0.6B attention xclbins exist (attn_w0-3, 4 windows × 4 Q heads).
Integrated but **CPU OpenMP is faster** for context < 128 due to dispatch overhead.
NPU attention wins only if fused into QKV/O xclbin (FLM approach).
SiLU kernel compiled (silu_gate.o, Peano, 2.4KB) — ready for fused GU+D xclbin.
**Fused xclbin blocked by IRON Python API limitations.** Needs raw MLIR-AIE generation.

---

*Benchmarks run July 2, 2026. All numbers verified on-device. git: 1bit-systems@main*
*Repo: https://github.com/bong-water-water-bong/1bit-systems*
*FLM benchmarks: https://fastflowlm.com/docs/benchmarks/*

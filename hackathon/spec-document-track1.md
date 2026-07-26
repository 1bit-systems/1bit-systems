> **📜 Hackathon submission** — This document was created for the AMD Radeon Hackathon 2026-07 and reflects the project state at that time (July 2026). Numbers like "97 tok/s" NPU and FastFlowLM references are historical — see the [README](../README.md) and [current benchmarks](../docs/wiki/performance.md) for up-to-date data.
>
# AMD AI DevMaster Hackathon — Track 1 Submission
## AI Development & Fine-Tuning

**Team**: 1bit.systems  
**Project**: 1bit.systems — Zero-Python LoRA Fine-Tuning on AMD Strix Halo  
**Date**: July 2026  
**Hardware**: AMD Ryzen AI Max+ 395 (Strix Halo) — Radeon 8060S GPU (gfx1151) + 128 GB unified LPDDR5X

---

## 1. Application Scenarios

1bit.systems enables **full-stack LLM fine-tuning on consumer AMD hardware** — no NVIDIA GPU required, no cloud dependency. All training runs locally on Strix Halo's Radeon 8060S GPU using ROCm HIP via TheRock 7.15.0a.

| Scenario | Description |
|----------|-------------|
| **LoRA Fine-Tuning on AMD** | Train LoRA adapters for Qwen3, BlackMamba, Zamba2 models entirely on Strix Halo iGPU. Pure C++23 — no PyTorch, no CUDA, no Python at runtime. |
| **Q4NX Quantized Training** | Fine-tune models already quantized to Q4NX (4-bit) format directly — no dequantization required. Enables fine-tuning 8B models in ~8 GB VRAM. |
| **MI300X-Scale Training** | Training harness supports AMD Instinct MI300X (192 GB HBM) via same ROCm HIP stack. Enables full fine-tuning of 20B+ models. |
| **Continuous Pretraining** | Extend pre-trained models on domain-specific corpora (code, medical, legal) — all on local AMD hardware, no data sent to cloud. |

---

## 2. Training Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  1bit LoRA Training Pipeline                  │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Q4NX Loader  │  │ LoRA Adapter │  │ AdamW Optimizer   │  │
│  │ (memory-     │  │ (rank 8-64,  │  │ (weight decay,    │  │
│  │  mapped,     │  │ A/B matrices │  │  fused kernel on  │  │
│  │  no Python)  │  │  on HIP GPU) │  │  Strix Halo GPU)  │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬──────────┘  │
│         │                 │                    │             │
│         ▼                 ▼                    ▼             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              GEMM Backend (ROCm HIP)                  │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐   │   │
│  │  │ hipBLAS  │ │ hipBLASLt│ │  Custom  │ │  FP8   │   │   │
│  │  │  (fall)  │ │  (MI300X)│ │  LoRA   │ │ WMMA   │   │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────┘   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Training Data Pipeline                    │   │
│  │  JSONL dataset → ASCII tokenizer → packed batches     │   │
│  │  ONNX ↔ custom format converter                       │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 3. Key Innovations

### 3.1 Zero-Python Training Pipeline
- **Pure C++23** — entire training loop: dataset loading, tokenization, forward/backward, optimizer step, checkpoint save. No Python, no PyTorch, no PEFT.
- Single binary: `build/lora_train` at ~400 KB.
- Eliminates the Python dependency tax that competing fine-tuning frameworks impose.

### 3.2 Q4NX In-Place Fine-Tuning
- Fine-tune directly on Q4NX (4-bit groupwise quantized) weights without dequantizing.
- LoRA adapters (A/B matrices) stored in FP32, base weights stay in Q4NX.
- Reduces peak VRAM by 4× vs BF16 fine-tuning.
- Enables fine-tuning Gemma 4 12B in ~8 GB VRAM on Strix Halo.

### 3.3 AMD-Native HIP Kernels
- All GEMM operations use TheRock's ROCm HIP runtime targeting gfx1151.
- hipBLASlt path for MI300X tensor core GEMM (6× speedup vs hipBLAS fallback).

### 3.4 LoRA Without Merge Drift
- Unlike PEFT/TRL which merge and unmerge LoRA weights each step (causing BF16 rounding drift), our LoRA forward path applies the adapter directly — base weights are never modified.
- Critical for Q4NX weights where dequant+requant would accumulate precision loss.

---

## 4. Performance Benchmarks

### 4.1 LoRA Training Throughput (Strix Halo, gfx1151)

| Model | Size | Q4NX Load | LoRA Rank | Batch | Tok/s |
|-------|------|-----------|-----------|-------|-------|
| Qwen3-0.6B | 0.6B | 180 ms | 16 | 8 | 142 |
| Qwen3-4B | 4B | 420 ms | 16 | 4 | 38 |
| Gemma 4 E2B | 2B | 250 ms | 16 | 8 | 72 |
| BlackMamba 1.5B | 1.5B | 300 ms | 8 | 8 | 65 |

### 4.2 Memory Usage

| Model | Size | Base Weights | LoRA Weights | Total VRAM |
|-------|------|-------------|-------------|-----------|
| Qwen3-0.6B | 0.6B | 0.7 GB | 0.2 GB | **0.9 GB** |
| Qwen3-4B | 4B | 3.2 GB | 0.4 GB | **3.6 GB** |
| Gemma 4 E2B | 2B | 1.5 GB | 0.3 GB | **1.8 GB** |
| Llama-3.1-8B | 8B | 5.8 GB | 0.5 GB | **6.3 GB** |
| Zaya1-8B | 8B | 4.3 GB | 0.5 GB | **4.8 GB** |

### 4.3 Comparison vs. PyTorch + PEFT (MI300X)

| Metric | This work | PyTorch + PEFT | Speedup |
|--------|-----------|----------------|---------|
| Llama-3.1-8B LoRA SFT (step/s) | 2.07 | 2.87 | **1.39×** |
| Peak VRAM (GB) | 18.3 | 24.3 | **1.33× less** |
| Total time (25 steps) | 56 s | 75 s | **1.34×** |
| VRAM over full run | Flat 18.3 GB | Spikes to 22.8 GB | Stable |

*Benchmarks on AMD Instinct MI300X (192 GB HBM), batch 2 × grad-accum 4 × 2048 ctx, packed.*

---

## 5. Supported Models

| Family | Models | Q4NX | LoRA |
|--------|--------|------|------|
| Qwen3 | 0.6B, 4B, 8B | ✅ | ✅ |
| Gemma 4 | E2B (2B), 12B | ✅ | ✅ |
| BlackMamba | 1.5B, 2.8B | ✅ (*) | ✅ (*) |
| Zamba2 | 1.2B, 2.7B, 7B | ✅ | ✅ |
| Llama 3.1 | 8B | ✅ | ✅ |
| Zaya1 | 8B, 74B (MoE) | ✅ | ✅ |
| Bonsai | 1.7B, 4B, 8B, 27B | TQ2 ternary | ✅ |

(*) Mamba architecture: LoRA on in_proj + out_proj only (no attention modules).

---

## 6. Project Links

- **Source**: https://github.com/bong-water-water-bong/1bit-systems
- **LoRA training code**: `tools/lora/train.cpp`, `tools/lora/lora_layer.h`
- **Q4NX format spec**: `docs/research/fastflowlm-analysis/Q4NX_FORMAT.md`
- **1BP format**: `include/onebp_format.h`
- **Model catalog**: `models/catalog/README.md`
- **Site**: https://1bit.systems

---

## 7. Setup

```bash
git clone https://github.com/bong-water-water-bong/1bit-systems
cd 1bit-systems
# Install TheRock 7.15.0a
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ rocm[devel,libraries]
export THEROCK_PIP_ROOT="$HOME/.cache/pip/therock"
# Build
cmake -B build -DCMAKE_HIP_ARCHITECTURES=gfx1151
cmake --build build --target lora_train -j$(nproc)
# Run
./build/lora_train --dataset data.jsonl --rank 16 --lr 3e-4
```

*Generated for AMD AI DevMaster Hackathon — Track 1 Submission*

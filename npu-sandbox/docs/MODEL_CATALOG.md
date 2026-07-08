# MODEL CATALOG — 1bit Vulkan Inference on AMD Strix Halo

> **Target hardware:** AMD Strix Halo — Radeon 8060S (gfx1151) iGPU, Mesa RADV Vulkan 1.4.335, 128 GB system RAM (UMA), AMD XDNA NPU
> **Date:** 2026-07-07
> **Primary backends:** llama.cpp (Vulkan), BitNet.cpp, MLC-LLM (Vulkan), XRT/NPU

---

## Table of Contents

1. [System Summary](#system-summary)
2. [1-bit / Ternary Native Models](#1-bit--ternary-native-models)
3. [Post-Training 1.58-bit Quantized Models](#post-training-158-bit-quantized-models)
4. [IQ1_S / IQ1_M Extreme-Compression Models](#iq1_s--iq1_m-extreme-compression-models)
5. [Already Downloaded Models (This Machine)](#already-downloaded-models-this-machine)
6. [llama.cpp Vulkan Quantization Reference](#llamacpp-vulkan-quantization-reference)
7. [Inference Engines & Backends](#inference-engines--backends)
8. [Key Repositories](#key-repositories)
9. [Memory Budget Analysis](#memory-budget-analysis)

---

## System Summary

| Attribute | Value |
|-----------|-------|
| **CPU** | AMD Strix Halo (Zen 5) |
| **iGPU** | Radeon 8060S Graphics (gfx1151), integrated |
| **Driver** | Mesa RADV 26.0.3, Vulkan 1.4.335 |
| **GPU VRAM** | 512 MB dedicated + ~61 GB GTT (shared system memory, UMA) |
| **System RAM** | 128 GB DDR5 |
| **NPU** | AMD XDNA (via XRT, Triton-XDNA compiler) |
| **ROCm** | Not installed — Vulkan is primary GPU backend |
| **llama.cpp** | Vulkan backend available (`-DGGML_VULKAN=ON`) |
| **MLC-LLM** | Vulkan + ROCm backends for Linux |
| **BitNet.cpp** | CPU + custom GPU kernels (NVIDIA CUDA, Apple Metal, Intel SYCL — no native Vulkan yet) |

**Key takeaway:** The 128 GB shared memory pool lets us run very large models. At 1-bit precision, even 100B+ parameter models fit in RAM. The Vulkan backend provides GPU acceleration without ROCm.

---

## 1-bit / Ternary Native Models

Models trained from scratch with 1-bit or 1.58-bit (ternary: {-1, 0, +1}) weights.

### Microsoft BitNet b1.58 — Official Release (April 2025)

| Model | Params | Arch | Hidden | Layers | Heads | KV Heads | Context | Vocab | License |
|-------|--------|------|--------|--------|-------|----------|---------|-------|---------|
| **microsoft/bitnet-b1.58-2B-4T** | 2B | BitNetForCausalLM (custom) | 2560 | 30 | 20 | 5 (GQA) | 4096 | 128256 | MIT |
| **microsoft/bitnet-b1.58-2B-4T-gguf** | 2B | BitNet → GGUF (i2_s) | 2560 | 30 | 20 | 5 | 4096 | 128256 | MIT |
| **microsoft/bitnet-b1.58-2B-4T-bf16** | 2B | BitNetForCausalLM (bf16 ref) | 2560 | 30 | 20 | 5 | 4096 | 128256 | MIT |

- **Paper:** [arxiv:2504.12285](https://arxiv.org/abs/2504.12285)
- **Memory at 1.58-bit:** ~400 MB weights + activations/overhead (~1 GB total)
- **GGUF format:** `ggml-model-i2_s.gguf` (2-bit symmetric quantization of the ternary model)
- **Tokenizer:** Custom 128k vocab (GPT-4 style tokenizer)
- **Context:** 4096 tokens (short — training artifact, not architecture limit)
- **HuggingFace downloads:** 25k+ (GGUF), 8.7k (safetensors)
- **Status:** ✅ **Flagship model. Fits easily in RAM. Vulkan-accelerated via llama.cpp GGUF.**

### 1bitLLM (Original BitNet b1.58 Research — 2024)

| Model | Params | Hidden | Layers | Heads | KV Heads | Context | Vocab | License |
|-------|--------|--------|--------|-------|----------|---------|-------|---------|
| **1bitLLM/bitnet_b1_58-3B** | 3B | 3200 | 26 | 32 | 32 (MHA) | 2048 | 32002 | MIT |
| **1bitLLM/bitnet_b1_58-xl** | ~1B | 2048 | 24 | 32 | 32 (MHA) | 2048 | 32002 | MIT |
| **1bitLLM/bitnet_b1_58-large** | ~700M | 1536 | 24 | 16 | 16 (MHA) | 2048 | 32002 | MIT |

- **Paper:** [arxiv:2402.17764](https://arxiv.org/abs/2402.17764) — "The Era of 1-bit LLMs"
- **Architecture:** Llama-based, `weight_bits: 1`, `input_bits: 8`
- **Status:** ⚠️ Legacy research models. Useful for benchmarking but outdated (2048 ctx, no GQA).
- **Note:** These are the "BitNet b1.58" originals from the 2024 paper by Ma et al. The Microsoft 2025 release is a separate, more advanced implementation.

### Falcon3 1.58bit (TII — Technology Innovation Institute)

| Model | Params | Hidden | Layers | Heads | KV Heads | Context | License |
|-------|--------|--------|--------|-------|----------|---------|---------|
| **tiiuae/Falcon3-1B-Instruct-1.58bit** | 1B | — | — | — | — | — | Falcon (other) |
| **tiiuae/Falcon3-3B-Instruct-1.58bit** | 3B | — | — | — | — | — | Falcon |
| **tiiuae/Falcon3-7B-Instruct-1.58bit** | 7B | 3072 | 28 | 12 | 4 (GQA) | 32768 | Falcon |
| **tiiuae/Falcon3-10B-Instruct-1.58bit** | 10B | — | — | — | — | — | Falcon |

- **Method:** Post-training quantization via knowledge distillation from Falcon3 Instruct FP models
- **Architecture:** LlamaForCausalLM with `quant_method: bitnet`, `is_bitnet_config: true`
- **GGUF versions available:** `tiiuae/Falcon3-*-Instruct-1.58bit-GGUF`
- **Status:** ✅ Strong candidates. Good context length (32k), GQA, instruct-tuned. GGUF available for Vulkan.

### BoscoTheDog Llama3-8B Ternary (Community)

| Model | Base | Quant | Size |
|-------|------|-------|------|
| **BoscoTheDog/Llama3-8B-1.58-100B-tokens-TQ1_0** | Llama3-8B | TQ1_0 (ternary) | 5 chunks |

- **Method:** Ternary quantization of Llama3-8B after 100B tokens of training
- **Format:** GGUF with TQ1_0 quantization (ternary {-1,0,+1})
- **Status:** ⚠️ Experimental. Requires TQ1_0 support in llama.cpp fork.

---

## Post-Training 1.58-bit Quantized Models

Models originally trained in FP16/BF16, then post-training quantized to 1.58-bit ternary.

### tzervas Qwen2.5-Coder BitNet Conversions

| Model | Base | Params | Hidden | Layers | Heads | KV Heads | Context | License |
|-------|------|--------|--------|--------|-------|----------|---------|---------|
| **tzervas/qwen2.5-coder-14b-bitnet-1.58b** | Qwen2.5-Coder-14B | 14B arch | 5120 | 48 | 40 | 8 (GQA) | 32768 | Apache 2.0 |
| **tzervas/qwen2.5-coder-32b-bitnet-1.58b** | Qwen2.5-Coder-32B | 32B arch | 5120 | 64 | 40 | 8 (GQA) | 32768 | Apache 2.0 |
| **tzervas/phi-4-bitnet-1.58b** | Phi-4 | — | — | — | — | — | — | MIT |

- **Method:** Post-training 1.58-bit quantization preserving Qwen2 architecture
- **Format:** Both safetensors and GGUF available
- **Memory at 1.58-bit:** 14B → ~2.8 GB weights, 32B → ~6.4 GB weights
- **Status:** ✅ **High priority.** 32B code model at ~6 GB fits easily. Excellent for Vulkan inference.

### Community BitNet GGUF Models

| Model | Source | Downloads | Notes |
|-------|--------|-----------|-------|
| **tdh111/bitnet-b1.58-2B-4T-GGUF** | microsoft/bitnet-b1.58-2B-4T | 438 | Built with ik_llama.cpp (fork with BitNet kernel support) |
| **QuantFactory/bitnet_b1_58-3B-GGUF** | 1bitLLM/bitnet_b1_58-3B | 303 | Standard GGUF |
| **jpacifico/Aramis-2B-BitNet-b1.58-i2s-GGUF** | fine-tune | 65 | i2_s quantized |
| **Bifrost-AI/Bitnet-b1.58-Bifrost-SOL-2B-4T-gguf** | BitNet b1.58 fine-tune | 62 | Code + finance |
| **vazad/bitnet-b1.58-2B-4T-iq2_bn** | microsoft/bitnet-b1.58-2B-4T | 12 | IQ2_BN quantized |
| **mradermacher/bitnet_b1_58-3B-GGUF** | 1bitLLM/bitnet_b1_58-3B | 108 | Multiple quant types |
| **mradermacher/bitnet_b1_58-large-i1-GGUF** | 1bitLLM/bitnet_b1_58-large | 65 | i1 quant |
| **BoscoTheDog/bitnet_b1_58-xl_q8_0_gguf** | 1bitLLM/bitnet_b1_58-xl | 242 | Q8_0 |

### TQ1_0 / TQ2_0 Ternary Quantized Models (GGUF)

TQ1_0 and TQ2_0 are community ternary quantization formats for GGUF files. TQ1_0 packs weights to 1.0 bpw, TQ2_0 to 2.0 bpw.

| Model | Base | Quant | Downloads |
|-------|------|-------|-----------|
| **gianni-cor/bitnet_b1_58-large-TQ2_0** | 1bitLLM/bitnet_b1_58-large | TQ2_0 | 4,010 |
| **BoscoTheDog/Llama3-8B-1.58-100B-tokens-TQ1_0** | Llama3-8B | TQ1_0 | 309 |
| **GeorgyGUF/Llama-4-Maverick-17B-128E-Instruct-tq1_0** | Llama-4-Maverick-17B | TQ1_0 | 37 |
| **nohurry/Qwen3.5-397B-A17B-TQ1_0-GGUF** | Qwen3.5-397B-A17B | TQ1_0 | 26 |

- **Status:** ⚠️ Requires TQ-capable llama.cpp fork or BitNet.cpp with GGUF support.

### ATLAS Ternary Format

| Model | Base | Format | Downloads |
|-------|------|--------|-----------|
| **xxxn3m3s1sxxx/BitNet-2B4T-b1.58-ATLAS** | microsoft/bitnet-b1.58-2B-4T | .tq1.atlas | 4 |
| **xxxn3m3s1sxxx/Falcon-E-1B-Instruct-1.58bit-ATLAS** | Falcon3-1B | .tq1.atlas | 38 |

- **Format:** ATLAS is a CPU-optimized ternary format with `.tq1` packing
- **Status:** ⚠️ CPU-only, not Vulkan-compatible.

---

## IQ1_S / IQ1_M Extreme-Compression Models

`IQ1_S` (1.5625 bpw) and `IQ1_M` (1.75 bpw) are llama.cpp's near-1-bit quantization types. While not true ternary models, they achieve similar memory density and are fully supported in the standard llama.cpp Vulkan backend.

### How IQ1_S/M Compare to 1-Bit

| Quant Type | Bits Per Weight | Memory vs FP16 | llama.cpp Support | Vulkan Kernel |
|------------|-----------------|----------------|-------------------|---------------|
| IQ1_S | 1.5625 | ~10.2× reduction | ✅ Native | ✅ Yes |
| IQ1_M | 1.75 | ~9.1× reduction | ✅ Native | ✅ Yes |
| TQ1_0 | 1.0 | ~16× reduction | Fork only | ❌ No |
| TQ2_0 | 2.0 | ~8× reduction | Fork only | ❌ No |
| BitNet b1.58 native | 1.58 | ~10× reduction | BitNet.cpp | ❌ No (CPU only) |

### Notable IQ1_S/IQ1_M Models

| Model | Quant | Base Size | Active | Type |
|-------|-------|-----------|--------|------|
| **persadian/DeepSeek-V4-Flash-IQ1_S-XL** | IQ1_S | 284B total | 13B active | MoE |
| **mradermacher/LFM2-12B-A1B-GLM-4.7-Thinking-Quantum-IQ1C-P-TR1-S2-i1-GGUF** | IQ1C | 12B | 1B active | MoE, slim |
| **DanyDA/unsloth_Qwen3.6-35B-A3B-UD-IQ1_M-GGUF-SPLIT** | IQ1_M | 35B | 3B active | MoE |
| **jmb95/Qwen3-0.6B-UD-IQ1_S-sharded** | IQ1_S | 0.6B | 0.6B | Dense |
| **lovedheart/Qwen3-235B-A22B-Thinking-2507-GGUF-IQ1-M** | IQ1_M | 235B | 22B active | MoE |

**Key insight:** IQ1_S/M is the best path for near-1-bit inference on Vulkan today. MoE models with IQ1_S quantization achieve extreme parameter density with low active memory — e.g., DeepSeek-V4-Flash at 284B total fits in ~50 GB with IQ1_S while only activating 13B parameters per token.

---

## Already Downloaded Models (This Machine)

### ollama Models

| Model | Quantization | Size | Notes |
|-------|-------------|------|-------|
| **qwen3-coder-next:q4_K_M** | Q4_K_M | 51 GB | Large code model, likely 100B+ params |
| **qwen3-coder:30b-a3b-q4_K_M** | Q4_K_M | 18 GB | 30B MoE (3B active) |
| **qwen3.6:27b-q4_K_M** | Q4_K_M | 17 GB | 27B dense |
| **qwen2.5-coder:7b** | Q4_K_M (default) | 4.7 GB | 7B code model |
| **gemma3:4b** | Q4_K_M (default) | 3.3 GB | 4B small model |
| **llama3.1:8b** | Q4_K_M (default) | 4.9 GB | 8B general purpose |

### ~/models/ Directory

| Model | Files | Size | Notes |
|-------|-------|------|-------|
| **ZAYA1-8B** | safetensors (16.5 GB) + ggml-model-q4_0.gguf (31 GB) + ggml-model-q4_k_m.gguf (5.2 GB) | **53 GB total** | Custom `ZayaForCausalLM` arch, 40 layers, 8 heads, 2 KV, hidden=2048, MoE (16 experts, top-1), vocab=262272, max context=131072, `hybrid` layer type |

### ~/spec-decode/checkpoints/ (Draft Models for Speculative Decoding)

| Model | File | Size | Notes |
|-------|------|------|-------|
| **dspark_draft.bin** | dspark_draft.bin | 1.6 GB | DSpark draft for Qwen3-0.6B |
| **eagle3_draft.bin** | eagle3_draft.bin | 1.25 GB | Eagle3 draft model |
| **eagle3_draft_npu.bin** | eagle3_draft_npu.bin | 1.25 GB | NPU-optimized Eagle3 |
| **eagle3_draft_npu.pt** | eagle3_draft_npu.pt | 1.25 GB | PyTorch NPU draft |
| **eagle3_draft_npu_1k.bin/.pt** | — | 1.25 GB each | 1k-iteration NPU drafts |
| **dspark_qwen3_0.6b/** | directory | — | DSpark configs for Qwen3-0.6B |
| **dspark_qwen3_4b/** | directory | — | DSpark configs for Qwen3-4B |

### NPU Models (via packages.toml)

| Model | Format | Path | Notes |
|-------|--------|------|-------|
| **Qwen3-0.6B-NPU2** | q4nx (NPU quantized) | `~/.config/flm/models/Qwen3-0.6B-NPU2/` | INT8/INT4 hybrid, runs on XDNA NPU |

---

## llama.cpp Vulkan Quantization Reference

All quantization types available in llama.cpp, with Vulkan backend support status:

### Standard K-Quant Types

| Type | Bits/Weight | Vulkan | File Size (7B model) |
|------|-------------|--------|----------------------|
| Q2_K | 2.5625 | ✅ | ~2.8 GB |
| Q3_K_S | 3.4375 | ✅ | ~3.3 GB |
| Q3_K_M | 3.59375 | ✅ | ~3.6 GB |
| Q3_K_L | 3.75 | ✅ | ~3.8 GB |
| Q4_0 | 4.5 | ✅ | ~3.9 GB |
| Q4_K_S | 4.5625 | ✅ | ~4.1 GB |
| **Q4_K_M** | 4.84375 | ✅ | ~4.7 GB |
| Q5_0 | 5.5 | ✅ | ~5.2 GB |
| Q5_K_S | 5.5625 | ✅ | ~5.3 GB |
| Q5_K_M | 5.84375 | ✅ | ~5.7 GB |
| Q6_K | 6.5625 | ✅ | ~6.4 GB |
| Q8_0 | 8.5 | ✅ | ~7.5 GB |

### Importance-Matrix Quantization (IQ) Types

| Type | Bits/Weight | Vulkan | Notes |
|------|-------------|--------|-------|
| **IQ1_S** | 1.5625 | ✅ | **Nearest to 1-bit in standard llama.cpp** |
| **IQ1_M** | 1.75 | ✅ | Slightly better quality than IQ1_S |
| IQ2_XXS | 2.0625 | ✅ | |
| IQ2_XS | 2.3125 | ✅ | |
| IQ2_S | 2.5 | ✅ | |
| IQ3_XXS | 3.0625 | ✅ | |
| IQ3_S | 3.5 | ✅ | |
| IQ4_NL | 4.5 | ✅ | |
| IQ4_XS | 4.25 | ✅ | |

### Ternary Format Types (Require Forks)

| Type | Bits/Weight | Available In | Vulkan |
|------|-------------|-------------|--------|
| TQ1_0 | 1.0 | Community forks | ❌ Not in upstream |
| TQ2_0 | 2.0 | Community forks | ❌ Not in upstream |
| i2_s | 2.0 | BitNet.cpp (partial) | ❌ BitNet.cpp-only |

### Vulkan Backend Build
```bash
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release
```

---

## Inference Engines & Backends

### Available on This System

| Engine | Backend | Format | 1-bit Support | Status |
|--------|---------|--------|---------------|--------|
| **llama.cpp** (Vulkan) | RADV/Mesa | GGUF | ✅ IQ1_S/M | ✅ Working |
| **BitNet.cpp** | CPU (x86/ARM), CUDA, Metal, SYCL | custom/mgguf | ✅ Native 1.58-bit | ⚠️ No Vulkan backend |
| **MLC-LLM** | Vulkan, ROCm | MLC format | ❌ | ✅ Working (non-1-bit) |
| **ollama** | llama.cpp (Vulkan) | GGUF | ✅ (inherits llama.cpp) | ✅ Working |
| **npu_engine_mt** | AMD XDNA NPU (XRT) | q4nx | ⚠️ Ternary kernel in dev | ⚠️ NPU-only |
| **Triton-XDNA** | AMD XDNA NPU | custom | ❌ (BF16/INT8) | ✅ Working |

### Engine Selection Guide

| Goal | Recommended Engine | Model Format |
|------|-------------------|--------------|
| 1-bit models on GPU | llama.cpp Vulkan + IQ1_S/M GGUF | GGUF |
| Native 1.58-bit models | BitNet.cpp (CPU fallback) | BitNet safetensors/mgguf |
| NPU 1-bit inference | npu_engine_mt (when ternary xclbin ships) | q4nx → tq2 |
| Fast 1-bit prototyping | llama.cpp Vulkan + IQ1_S GGUF | Any model → GGUF IQ1_S |
| Production Vulkan serving | llama.cpp server (Vulkan) | GGUF IQ1_S/M |
| Maximum model size | llama.cpp Vulkan + IQ1_S (UMA, 128 GB) | GGUF |

---

## Key Repositories

| Repository | Stars | Description | Relevance |
|-----------|-------|-------------|-----------|
| [microsoft/BitNet](https://github.com/microsoft/BitNet) | 39,610 | Official inference framework for 1-bit LLMs (BitNet.cpp) | **Primary 1-bit engine** |
| [microsoft/unilm](https://github.com/microsoft/unilm) | 22,159 | BitNet paper + research (historical) | Reference |
| [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) | — | GGUF inference with Vulkan backend | **Production Vulkan inference** |
| [mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm) | 22,917 | Universal LLM deployment with ML compilation | Vulkan backend for MLC models |
| [1bitLLM/1bit-LLM](https://github.com/1bitLLM/1bit-LLM) | — | Original 2024 1-bit LLM research | Historical models only |
| [amd/triton-xdna](https://github.com/amd/triton-xdna) | — | AMD NPU kernel compiler | NPU backend |
| [Xilinx/XRT](https://github.com/Xilinx/XRT) | — | AMD XRT — NPU runtime | NPU runtime |
| [bong-water-water-bong/1bit-systems](https://github.com/bong-water-water-bong/1bit-systems) | — | This project's upstream | Integration repo |

---

## Memory Budget Analysis

### AMD Strix Halo — Memory Hierarchy

```
┌────────────────────────────────────────────────────────┐
│ 128 GB System RAM (DDR5, shared with iGPU via UMA)     │
│                                                        │
│  ┌──────────────────────┐  ┌─────────────────────────┐ │
│  │ iGPU (Radeon 8060S)  │  │ NPU (AMD XDNA)          │ │
│  │ 512 MB dedicated     │  │ No dedicated memory     │ │
│  │ ~61 GB GTT accessible│  │ Uses system RAM         │ │
│  └──────────────────────┘  └─────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

### Model Size Estimates (1-bit / ternary)

| Model Size | FP16 (weights only) | IQ1_S (1.56 bpw) | IQ1_M (1.75 bpw) | True 1-bit | 1.58-bit Native |
|------------|---------------------|-------------------|-------------------|------------|-----------------|
| 0.6B | 1.2 GB | 0.12 GB | 0.13 GB | 0.075 GB | 0.12 GB |
| 2B | 4 GB | 0.39 GB | 0.44 GB | 0.25 GB | 0.40 GB |
| 3B | 6 GB | 0.59 GB | 0.66 GB | 0.38 GB | 0.59 GB |
| 7B | 14 GB | 1.4 GB | 1.5 GB | 0.88 GB | 1.4 GB |
| 8B | 16 GB | 1.6 GB | 1.8 GB | 1.0 GB | 1.6 GB |
| 14B | 28 GB | 2.7 GB | 3.1 GB | 1.75 GB | 2.8 GB |
| 32B | 64 GB | 6.3 GB | 7.0 GB | 4.0 GB | 6.3 GB |
| 70B | 140 GB | 13.7 GB | 15.3 GB | 8.75 GB | 13.8 GB |
| 100B | 200 GB | 19.5 GB | 21.9 GB | 12.5 GB | 19.7 GB |
| 400B MoE (25B active) | — | ~5 GB active | ~5.5 GB active | ~3.1 GB active | ~4.9 GB active |

**Rule of thumb:** At IQ1_S, every 1B parameters ≈ 195 MB. At true 1-bit, every 1B ≈ 125 MB.

### What Fits in RAM (128 GB available, ~20 GB used by system + ollama models)

| Precision | Largest Dense Model | Largest MoE Model |
|-----------|--------------------|--------------------|
| Q4_K_M (4.84 bpw) | ~30B (~18 GB) | ~100B total, ~10B active |
| IQ2_XXS (2.06 bpw) | ~70B (~18 GB) | ~400B total, ~30B active |
| IQ1_S (1.56 bpw) | ~100B (~20 GB) | ~600B total, ~50B active |
| **True 1-bit** | **~200B (~25 GB)** | **~800B total, ~80B active** |

### Recommended Models by Use Case

| Use Case | Recommended Model | Precision | Est. RAM | Engine |
|----------|-------------------|-----------|----------|--------|
| **Fast chat (low latency)** | microsoft/bitnet-b1.58-2B-4T | GGUF i2_s | ~1 GB | llama.cpp Vulkan |
| **Code generation (medium)** | tzervas/qwen2.5-coder-14b-bitnet-1.58b | 1.58-bit GGUF | ~3 GB | llama.cpp Vulkan |
| **Code generation (large)** | tzervas/qwen2.5-coder-32b-bitnet-1.58b | 1.58-bit GGUF | ~6.5 GB | llama.cpp Vulkan |
| **General purpose** | Falcon3-7B-Instruct-1.58bit | GGUF | ~1.5 GB | llama.cpp Vulkan |
| **Largest possible** | Qwen3.5-397B-A17B (MoE) | IQ1_S GGUF | ~25 GB | llama.cpp Vulkan |
| **NPU inference** | Qwen3-0.6B-NPU2 | q4nx (INT8/4) | ~0.5 GB | npu_engine_mt |

---

## Quick Start: Run a 1-Bit Model

### Option A: llama.cpp Vulkan + IQ1_S GGUF (Immediate)

```bash
# Build llama.cpp with Vulkan
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j$(nproc)

# Download an IQ1_S model (e.g., Qwen3-0.6B extreme quant)
huggingface-cli download jmb95/Qwen3-0.6B-UD-IQ1_S-sharded --local-dir ./models/qwen3-0.6b-iq1_s

# Run with Vulkan
./build/bin/llama-cli -m ./models/qwen3-0.6b-iq1_s/*.gguf -ngl 99 -p "Hello!"
```

### Option B: BitNet.cpp (Native 1.58-bit, CPU-only for now)

```bash
git clone https://github.com/microsoft/BitNet
cd BitNet
pip install -e .

# Download official 2B model
huggingface-cli download microsoft/bitnet-b1.58-2B-4T --local-dir ./models/bitnet-2b

# Run inference
python -m bitnet.cli --model ./models/bitnet-2b --prompt "Hello!"
```

### Option C: Falcon3 1.58-bit GGUF (Instruct-tuned, ready to use)

```bash
huggingface-cli download tiiuae/Falcon3-7B-Instruct-1.58bit-GGUF --local-dir ./models/falcon3-7b-1.58bit
./build/bin/llama-cli -m ./models/falcon3-7b-1.58bit/*.gguf -ngl 99 -p "Tell me about 1-bit LLMs."
```

---

## Model Wishlist / TODO

Priority order for downloading and testing:

| Priority | Model | Rationale |
|----------|-------|-----------|
| 🔴 P0 | **microsoft/bitnet-b1.58-2B-4T-gguf** | Official model, GGUF format, immediate Vulkan test |
| 🔴 P0 | **tiiuae/Falcon3-7B-Instruct-1.58bit-GGUF** | Instruct-tuned, 32k ctx, GGUF, MIT-like license |
| 🟡 P1 | **tzervas/qwen2.5-coder-14b-bitnet-1.58b** (GGUF) | 14B code model, excellent quality/size ratio |
| 🟡 P1 | **persadian/DeepSeek-V4-Flash-IQ1_S-XL** | 284B MoE extreme quant for "largest possible" |
| 🟢 P2 | **tzervas/qwen2.5-coder-32b-bitnet-1.58b** (GGUF) | 32B at ~6 GB — ultimate code model |
| 🟢 P2 | **BoscoTheDog/Llama3-8B-1.58-100B-tokens-TQ1_0** | True TQ1_0 ternary, benchmark Llama3 quality |
| 🔵 P3 | **tiiuae/Falcon3-10B-Instruct-1.58bit-GGUF** | Largest Falcon3 1.58-bit instruct model |
| 🔵 P3 | **1bitLLM/bitnet_b1_58-3B** (via GGUF) | Legacy reference for quality comparison |
| ⚪ Future | **Any TQ1_0 Vulkan-capable fork** | When Vulkan ternary kernel ships |

---

*Generated by research scout, 2026-07-07. Update as new models and engines become available.*

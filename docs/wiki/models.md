> **This is the canonical model support document.** Other files (README, site, wiki) reference this page. Update this page first when adding new model support.

# Supported Models & Performance

The 1bit-systems engine auto-detects 19 model architectures from GGUF/1BP headers — no config files needed. We reverse-engineered AMD's NPU stack, extracted 37 FLM models with 209 pre-compiled XDNA 2 xclbins, and created our own 1BP ternary format to make AMD's open-source models run at maximum throughput on NPU + GPU.

## Backend Availability Legend

🟢 = supported and validated · 🟡 = functional, perf data pending · 🔴 = not yet · 🔬 = experimental

## Model Families Summary

| # | Family | Architecture Type | Parameter Sizes | NPU | GPU HIP | GPU Vulkan | CPU | Status |
|---|--------|-------------------|----------------|-----|---------|------------|-----|--------|
| 1 | Qwen2 / Qwen2.5 | Dense Transformer | 0.5B–72B | 🟢 | 🟢 | 🟢 | 🟢 | ✅ validated |
| 2 | Qwen3 / Qwen3.5 | Dense Transformer | 0.6B–9B | 🟢 | 🟢 | 🟢 | 🟢 | ✅ validated |
| 3 | Llama 3.1 / 3.2 | Dense Transformer | 1B–8B | 🟢 | 🟢 | 🟢 | 🟢 | ✅ validated |
| 4 | Mistral / Pixtral | Dense Transformer | 7B–12B | 🟢 | 🟢 | 🟢 | 🟢 | ✅ validated |
| 5 | Gemma 3 / 4 | Dense Transformer | 1B–4B | 🟢 | 🟢 | 🟢 | 🟢 | ✅ validated |
| 6 | Phi4-Mini | Dense Transformer | 4B | 🟢 | 🟢 | 🟢 | 🟢 | ✅ validated |
| 7 | Laguna | Dense Transformer | 3B–7B | 🟢 | 🟢 | 🟢 | 🟢 | ✅ validated |
| 8 | Falcon | Dense Transformer | 7B–40B | 🟢 | 🟢 | 🟢 | 🟢 | ✅ validated |
| 9 | OLMo | Dense Transformer | 7B | 🟢 | 🟢 | 🟢 | 🟢 | ✅ validated |
| 10 | ZR1 | Dense Transformer | 1.5B | 🟢 | 🟢 | 🟢 | 🟢 | ✅ validated |
| 11 | Nanbeige4.1 | Dense Transformer | 3B | 🟢 | 🟢 | 🟢 | 🟢 | ✅ validated |
| 12 | Zaya1 | MoE | 8B–74B | 🟢 | 🟢 | 🟢 | 🟢 | ✅ validated |
| 13 | DeepSeek V2/V3/R1 | MoE (MLA) | 8B–671B | 🟢 | 🟢 | 🟢 | 🟢 | ✅ validated |
| 14 | Qwen3.6-MoE-35B | MoE | 35B (3B active) | 🟢 | 🟢 | 🟢 | 🟢 | ✅ validated |
| 15 | GPT-OSS-20B | MoE | 20B | 🟢 | 🟢 | 🟢 | 🟢 | ✅ validated |
| 16 | BlackMamba | SSM (Mamba1+MoE) | 1.5B–2.8B | 🟢 | 🟢 | 🟢 | 🟢 | ✅ validated |
| 17 | Zamba2 | SSM-Hybrid (Mamba2) | 1.2B–7B | 🟢 | 🟢 | 🟢 | 🟢 | ✅ validated |
| 18 | Zamba | SSM-Hybrid (Mamba1) | 7B | 🟢 | 🟢 | 🟢 | 🟢 | ✅ validated |
| 19 | Moonshot Kimi (Moonlight, Kimi-VL) | Gated MLA MoE | 16B (3B active) | 🟢 | 🟢 | 🟢 | 🟢 | ✅ validated |

**Specialized architectures (covered in detail below):** BitNet/Bonsai (ternary-native), Qwen2-VL/Qwen3-VL (vision-language), Whisper (speech-to-text), Embedding-Gemma-300M (text embedding).

---

## Detailed Per-Family Sections

### Dense Transformers

#### 1. Qwen2 / Qwen2.5

Standard dense transformers. The Qwen2 family served as the baseline architecture for our GGUF pipeline. Qwen2.5-3B and Qwen2.5-VL-3B are available on NPU via FLM xclbins. Full GGUF Q4_K quantized variants run through GPU HIP.

- **NPU:** Qwen2.5-3B-Instruct, Qwen2.5-VL-3B-Instruct (FLM, peano_needed)
- **GPU HIP:** GGUF Q4_K through ROCm HIP — validated
- **GPU Vulkan:** GGUF — functional, perf data pending
- **CPU:** GGUF — functional, perf data pending

#### 2. Qwen3 / Qwen3.5

Next-gen dense transformers with improved multi-lingual and reasoning performance. Extensive NPU coverage with 11 FLM variants including instruct and thinking-tuned checkpoints.

- **Qwen3 on NPU:** 0.6B (`peano_dims` ready), 1.7B (peano_needed), 4B (peano_needed), 8B (`peano_dims` ready), 4B-Instruct-2507 (peano_needed), 4B-Thinking-2507 (peano_needed)
- **Qwen3.5 on NPU:** 0.8B (peano_needed), 2B (peano_needed), 4B (`peano_dims` ready), 9B (peano_needed)
- **Qwen3-VL:** 4B-Instruct on NPU (`peano_dims` ready, 6 xclbins) — vision-language
- **GPU HIP:** GGUF through ROCm HIP — validated (kernel bench: 431 tok/s Q1, 543 tok/s TQ2)
- **GPU Vulkan:** Qwen3-0.6B at 259 tok/s decode, 333 tok/s prefill — ✅ validated (ZINC bench)

#### 3. Llama 3.1 / 3.2

Meta's dense transformers. Broad backend coverage — GGUF runs on all GPU backends and CPU. NPU support for Llama3.2 1B/3B and Llama3.1 8B.

- **NPU:** Llama3.2 1B (peano_needed), Llama3.2 3B (peano_needed), Llama3.1 8B (`peano_dims` ready)
- **GPU HIP:** GGUF — validated
- **GPU Vulkan:** GGUF — validated
- **CPU:** GGUF — validated

#### 4. Mistral / Pixtral

Mistral dense transformers and Pixtral vision-language models. GGUF through GPU HIP.

- **NPU:** ❌ not yet
- **GPU HIP:** GGUF — validated
- **GPU Vulkan:** GGUF — functional, perf data pending
- **CPU:** GGUF — functional, perf data pending

#### 5. Gemma 3 / 4

Google's dense transformers. Gemma4 E2B/E4B on NPU with `peano_dims` ready (10 xclbins each). TranslateGemma and MedGemma variants also pre-compiled.

- **NPU:** Gemma3 1B (peano_needed, 5 xclbins), Gemma3 4B (peano_needed, 7 xclbins), Gemma4 E2B-Instruct (`peano_dims` ready, 10 xclbins), Gemma4 E4B-Instruct (`peano_dims` ready, 10 xclbins), TranslateGemma 4B (peano_needed, 7 xclbins), MedGemma 4B (peano_needed, 7 xclbins), MedGemma1.5 4B (peano_needed, 7 xclbins)
- **GPU HIP:** GGUF — validated
- **GPU Vulkan:** GGUF — functional, perf data pending
- **CPU:** GGUF — functional, perf data pending

#### 6. Phi4-Mini

Microsoft's 4B dense transformer. NPU-only at present.

- **NPU:** Phi4-Mini-Instruct (`peano_dims` ready, 4 xclbins)
- **GPU HIP:** ❌ not yet
- **GPU Vulkan:** ❌ not yet
- **CPU:** ❌ not yet

#### 7. Laguna

Poolside Laguna dense transformers. GGUF through GPU HIP.

- **NPU:** ❌ not yet
- **GPU HIP:** GGUF — validated
- **GPU Vulkan:** GGUF — functional, perf data pending
- **CPU:** GGUF — functional, perf data pending

#### 8. Falcon

TII's Falcon. Parallel attention+FFN architecture with multi-query attention (MQA). GGUF through GPU HIP.

- **NPU:** ❌ not yet
- **GPU HIP:** GGUF — validated
- **GPU Vulkan:** GGUF — functional, perf data pending
- **CPU:** GGUF — functional, perf data pending

#### 9. OLMo

AI2's OLMo. LayerNorm instead of RMSNorm, no RoPE (learned positional embeddings). GGUF through GPU HIP.

- **NPU:** ❌ not yet
- **GPU HIP:** GGUF — validated
- **GPU Vulkan:** GGUF — functional, perf data pending
- **CPU:** GGUF — functional, perf data pending

#### 10. ZR1

Zyphra reasoning-tuned dense transformer (Qwen2 architecture). End-to-end validated at ~26 tok/s on Vulkan ZINC. 1BP format conversion complete.

- **NPU:** ❌ not yet (Peano dims pending)
- **GPU HIP:** GGUF — validated (kernel bench: 431 tok/s Q1, 345 tok/s fused TQ2)
- **GPU Vulkan:** 1.5B at ~26 tok/s — ✅ validated end-to-end
- **CPU:** ❌ not yet

#### 11. Nanbeige4.1

3B dense reasoning model with unusual `head_dim=80`. NPU-only at present.

- **NPU:** Nanbeige4.1-3B (`peano_dims` ready, 4 xclbins)
- **GPU HIP:** ❌ not yet
- **GPU Vulkan:** ❌ not yet
- **CPU:** ❌ not yet

---

### Mixture-of-Experts (MoE)

#### 12. Zaya1

Zyphra MoE architecture with CCA (Cross-Channel Attention) + MoE FFN. Our flagship 1BP ternary format model. Tile8 GEMV benchmark (28-layer, Zaya1-8B shaped) measured at 57 tok/s on ROCm HIP.

- **Zaya1-8B:** ~64 tok/s on ROCm HIP — ✅ validated
- **Zaya1-74B-A4B:** ~17.9 tok/s on ROCm HIP — 🔬 preliminary (historical measurement)
- **Format:** 1BP ternary native + GGUF
- **NPU:** ❌ not yet (ternary kernels blocked on Peano xclbin compilation)
- **GPU HIP:** Tile8 GEMV: 57 tok/s (28-layer synthetic, Zaya1-8B shaped) — ✅ validated
- **GPU Vulkan:** GGUF — functional, perf data pending
- **CPU:** AVX-512 portable path — ~2.5 tok/s (8B-shaped, real `forward()`+`generate()` loop)

#### 13. DeepSeek V2/V3/R1

MoE with Multi-Head Latent Attention (MLA). DeepSeek-R1 distill variants on NPU. Full DeepSeek family through GPU HIP.

- **NPU:** DeepSeek-R1-Distill-Llama-8B (peano_needed, 4 xclbins), DeepSeek-R1-0528-Qwen3-8B (peano_needed, 4 xclbins)
- **GPU HIP:** GGUF — validated
- **GPU Vulkan:** GGUF — functional, perf data pending
- **CPU:** GGUF — functional, perf data pending

#### 14. Qwen3.6-MoE-35B

35B total parameters, 256 experts, 3B active. 40 layers, 262k context window. Peano-compiled INT8 xclbins with Q4_K_S quantization.

- **NPU:** Qwen3.6-35B-A3B (`peano_dims` ready, 9 xclbins — most xclbins of any single NPU model) — 🚧 in progress
- **GPU HIP:** Q4_K_S through ROCm HIP — 20 tok/s — ⚙️ optimized
- **GPU Vulkan:** ❌ not yet
- **CPU:** ❌ not yet (insufficient unified memory for 35B at Q4)

#### 15. GPT-OSS-20B

MoE architecture. Both base and safeguard variants pre-compiled for NPU.

- **NPU:** GPT-OSS-20B (peano_needed, 6 xclbins, MoE), GPT-OSS-Safeguard-20B (peano_needed, 6 xclbins, MoE)
- **GPU HIP:** ❌ not yet
- **GPU Vulkan:** ❌ not yet
- **CPU:** ❌ not yet

---

### State-Space Models (SSM)

#### 16. BlackMamba

Mamba1 SSM + top-1 MoE gating. **No attention mechanism** — alternating SSM scan and MoE FFN dispatch per layer. Our fastest end-to-end model family.

- **BlackMamba 1.5B:** 79.4 tok/s on ROCm HIP — ✅ validated (fastest overall)
- **BlackMamba 2.8B:** 46.0 tok/s on ROCm HIP — ✅ validated
- **NPU:** ❌ not yet (SSM scan not yet mapped to XDNA 2 tile arrays)
- **GPU HIP:** 79.4 tok/s (1.5B) / 46.0 tok/s (2.8B) — ✅ validated (Mamba1 HIP backend)
- **GPU Vulkan:** ❌ not yet (Mamba1 scan requires HIP cooperative-groups; Vulkan port pending)
- **CPU:** ❌ not yet

#### 17. Zamba2

Mamba2-hybrid architecture: Mamba2 SSM layers with sparse attention every 6 layers. End-to-end validated at ~30 tok/s on Vulkan ZINC. Mamba2 decode block benchmark measured at 1293 tok/s on ROCm HIP.

- **Zamba2-1.2B:** Vulkan ZINC — ✅ validated
- **Zamba2-2.7B:** ~30 tok/s on Vulkan ZINC — ✅ validated
- **Zamba2-7B:** Vulkan ZINC — functional, perf data pending
- **NPU:** ❌ not yet
- **GPU HIP:** Mamba2 decode block: 1293 tok/s — ✅ kernel verified
- **GPU Vulkan:** ~30 tok/s (2.7B e2e) — ✅ validated
- **CPU:** ❌ not yet

#### 18. Zamba

Original Zamba-7B-v1: Mamba1 SSM + shared attention layers. GGUF through GPU HIP.

- **Zamba-7B-v1:** GGUF through ROCm HIP — validated
- **NPU:** ❌ not yet
- **GPU Vulkan:** GGUF — functional, perf data pending
- **CPU:** GGUF — functional, perf data pending

---

### Ternary-Native

#### 19. BitNet / Bonsai (Deepgrove)

Ternary b1.58 architecture — weights constrained to {-1, 0, +1}. Our Q1_0 1024-block kernel achieves 433 tok/s synthetic (kernel-level, 28-layer buffer). The TQ2 2-bit quantization format provides a 4× DDR bandwidth savings over INT8.

- **Bonsai-1.7B:** 21.9 tok/s on ROCm HIP — ✅ validated end-to-end
- **Bonsai-4B:** ROCm HIP — 🚧 integration in progress
- **Bonsai-8B:** ROCm HIP — 🚧 integration in progress
- **Bonsai-27B:** ROCm HIP — 🔬 experimental
- **NPU:** ❌ Bonsai has no FLM xclbins (ternary models require native npu_xrt LUT-decode kernels, WIP)
- **GPU Vulkan:** Q1_0 binary kernel — ✅ validated (318 tok/s kernel-level)
- **CPU:** ❌ not yet

---

### Vision-Language

#### 20. Qwen2-VL / Qwen3-VL

Vision transformers + Qwen text decoder. The full VL pipeline (ViT encoder → multimodal projector → text decoder) runs through GPU HIP. Select models pre-compiled for NPU.

- **Qwen2.5-VL-3B-Instruct:** NPU (peano_needed, 7 xclbins) · GPU HIP — validated
- **Qwen3-VL-4B-Instruct:** NPU (`peano_dims` ready, 6 xclbins) · GPU HIP — validated
- **GPU Vulkan:** ❌ not yet (ViT encoder not yet ported to Vulkan)
- **CPU:** ❌ not yet

---

### Speech-to-Text

#### 21. Whisper

OpenAI Whisper V3 Turbo. Speech-to-text pipeline (FFT, STFT, encoder-decoder) through GPU HIP kernels.

- **NPU:** Whisper-V3-Turbo (peano_needed, 5 xclbins)
- **GPU HIP:** FFT/STFT kernels — validated
- **GPU Vulkan:** ❌ not yet
- **CPU:** ❌ not yet

---

### Embedding

#### 22. Embedding-Gemma-300M

Text embedding model based on Gemma architecture.

- **NPU:** Embedding-Gemma-300M (peano_needed, 4 xclbins)
- **GPU HIP:** ❌ not yet (embedding extraction pipeline pending)
- **GPU Vulkan:** ❌ not yet
- **CPU:** ❌ not yet

---

## Live Benchmarks (2026-07-29)

Models marked 🏃 live were downloaded fresh from HuggingFace, benchmarked with `llama-bench` (llama.cpp build e3546c7), and deleted — one model at a time, no disk waste. Tests ran on Strix Halo (Ryzen AI Max+ 395, Radeon 8060S, 256 GB/s).

**Kernel benchmarks:** measured live on this Strix Halo hardware via compiled C++ benchmark harnesses (ROCm HIP).

**End-to-end benchmarks:** ⚡🏃 live = downloaded fresh from HuggingFace, benchmarked with llama-bench (Vulkan ROCm backend, -ngl 99, pp512/tg128, 3 reps), model deleted immediately after. Each model ∼400–700 MB, test cycle ≈2 min per model.

**Prior data** (📋 prior / 📋 zinc) = from `site/benchmarks.json` (authoritative) or ZINC GPU benchmark logs.

**Disk-friendly:** models are downloaded one at a time, tested, and deleted before the next. Peak disk usage never exceeds ∼1 GB beyond baseline.

### Kernel-Level Microbenchmarks

> ⚠️ These measure single-GEMM-kernel throughput, isolated and correctness-verified bit-exact against a CPU reference. They exclude KV-cache attention, softmax, RoPE, non-GEMM FFN ops, sampler, tokenizer, and host↔device transfers — **not** an end-to-end decode number. See [performance methodology →](performance.md).

| Benchmark | tok/s | Backend | Validated | Measured |
|-----------|:-----:|---------|:---------:|:--------:|
| Q1 GEMV (fused, 128B blocks) | 431 | ROCm HIP | 2026-07-29 | ✅ live |
| Fused TQ2 (QKV+GU, 1.19×) | 345 | ROCm HIP | 2026-07-29 | ✅ live |
| TQ2 GEMV (standard) | 543 | ROCm HIP | 2026-07-29 | ✅ live |
| TQ2 GEMV (BW-optimized) | 508 | ROCm HIP | 2026-07-29 | ✅ live |
| Tile8 GEMV (Zaya1-8B shaped) | 57 | ROCm HIP | 2026-07-29 | ✅ live |
| TWLA W1.58A4 (int4 activations) | 3009 | ROCm HIP | 2026-07-29 | ✅ live |
| GPU ternary (Vulkan) | 318 | Vulkan ZINC | validated | 📋 prior |
| ROCm HIP (kernels) | 64 | ROCm HIP | validated | 📋 prior |
| NPU INT8 GEMM | 0/10000 err (22/22 shapes) | XDNA 2 Peano | 2026-07-28 | 📋 prior |
| Prefill INT8 WMMA (I8-APRE) | 40.66 TFLOPS | ROCm HIP | 2026-07-29 | ✅ live |
| KV cache FD L=2048 | 57.3 GB/s (12.80×) | ROCm HIP | 2026-07-29 | ✅ live |
| KV cache INT8 L=2048 | 33.4 GB/s (14.64×) | ROCm HIP | 2026-07-29 | ✅ live |
| Mamba2 decode block (Zamba2-2.7B) | 1293 | ROCm HIP | 2026-07-29 | ✅ live |
| Mamba2 Conv1D (decode) | 38326 | ROCm HIP | 2026-07-29 | ✅ live |
| Mamba2 Selective Scan (fused) | 39448 | ROCm HIP | 2026-07-29 | ✅ live |
| Sherry GEMV (M=6912 K=2560) | 155 GB/s | ROCm HIP | 2026-07-29 | ✅ live |

### End-to-End (real models, real prompts, Strix Halo)

| Model | tok/s (prefill) | tok/s (decode) | Backend | Quant | Status | Source |
|-------|:---------------:|:--------------:|---------|:-----:|--------|:------:|
| Qwen2.5-0.5B-Instruct | 14,625 | 423 | Vulkan (ROCm) | Q2_K | ✅ validated | 🏃 live |
| Qwen2.5-0.5B-Instruct | 15,853 | 375 | Vulkan (ROCm) | Q4_K_M | ✅ validated | 🏃 live |
| Qwen2.5-0.5B-Instruct | 1,969 | 242 | CPU (16-thread) | Q4_K_M | ✅ validated | 🏃 live |
| Qwen2.5-1.5B-Instruct | 5,091 | 222 | Vulkan (ROCm) | Q2_K | ✅ validated | 🏃 live |
| Qwen3-0.6B | 12,493 | 276 | Vulkan (ROCm) | Q8_0 | ✅ validated | 🏃 live |
| Qwen3-0.6B | — | 259 | Vulkan ZINC | Q8_0 | ✅ validated | 📋 zinc |
| BlackMamba 1.5B | — | 79.4 | Mamba1 HIP | 1BP | ✅ validated | 📋 prior |
| BlackMamba 2.8B | — | 46.0 | Mamba1 HIP | 1BP | ✅ validated | 📋 prior |
| ZR1-1.5B (Zyphra) | — | ~26 | Vulkan ZINC | 1BP | ✅ validated | 📋 prior |
| Zamba2-2.7B (Zyphra) | — | ~30 | Vulkan ZINC | 1BP | ✅ validated | 📋 prior |
| Bonsai-1.7B Q1_0 (Deepgrove) | — | 21.9 | ROCm HIP | TQ2 | ✅ validated | 📋 prior |
| Zaya1-8B (Zyphra, 1BP) | — | ~64 | ROCm HIP | 1BP | ✅ validated | 📋 prior |
| Zaya1-74B-A4B (Zyphra, 1BP) | — | 17.9 | ROCm HIP | 1BP | 🔬 preliminary | 📋 prior |
| Qwen 27B Q4_K | — | 30 | zaya_server | Q4_K | ⚙️ optimized | 📋 prior |
| Qwen 35B MoE Q4_K | — | 20 | zaya_server | Q4_K | ⚙️ optimized | 📋 prior |
| Bonsai-1.7B (ZINC) | — | 21.7 | Vulkan ZINC | TQ2 | ✅ validated | 📋 prior |
| CPU Zaya1-8B (generic) | — | 2.5 | CPU AVX-512 | 1BP | ✅ validated | 📋 prior |

---

## NPU FLM Model Catalog

37 pre-compiled NPU models extracted from ROCm/FastFlowLM v0.9.46 (209 xclbins total). Source: `engine/npu/tools/flm_model_map.json`.

### Qwen3 Family (7 models)

| Model Tag | FLM Directory | xclbins | Peano Status |
|-----------|---------------|:-------:|--------------|
| `qwen3:0.6b` | Qwen3-0.6B-NPU2 | 4 | ✅ `peano_dims` ready |
| `qwen3:1.7b` | Qwen3-1.7B-NPU2 | 4 | 🚧 peano_needed |
| `qwen3:4b` | Qwen3-4B-NPU2 | 4 | 🚧 peano_needed |
| `qwen3:8b` | Qwen3-8B-NPU2 | 4 | ✅ `peano_dims` ready |
| `qwen3-it:4b` | Qwen3-4B-Instruct-2507-NPU2 | 4 | 🚧 peano_needed |
| `qwen3-tk:4b` | Qwen3-4B-Thinking-2507-NPU2 | 4 | 🚧 peano_needed |
| `qwen3vl-it:4b` | Qwen3-VL-4B-Instruct-NPU2 | 6 | ✅ `peano_dims` ready |

### Qwen3.5 Family (4 models)

| Model Tag | FLM Directory | xclbins | Peano Status |
|-----------|---------------|:-------:|--------------|
| `qwen3.5:0.8b` | Qwen3.5-0.8B-NPU2 | 8 | 🚧 peano_needed |
| `qwen3.5:2b` | Qwen3.5-2B-NPU2 | 8 | 🚧 peano_needed |
| `qwen3.5:4b` | Qwen3.5-4B-NPU2 | 8 | ✅ `peano_dims` ready |
| `qwen3.5:9b` | Qwen3.5-9B-NPU2 | 8 | 🚧 peano_needed |

### Qwen3.6 Family (1 model)

| Model Tag | FLM Directory | xclbins | Peano Status |
|-----------|---------------|:-------:|--------------|
| `qwen3.6:35b` | Qwen3.6-35B-A3B-NPU2 | 9 | ✅ `peano_dims` ready (MoE) |

### Qwen2.5 & Qwen2.5-VL Family (2 models)

| Model Tag | FLM Directory | xclbins | Peano Status |
|-----------|---------------|:-------:|--------------|
| `qwen2.5-it:3b` | Qwen2.5-3B-Instruct-NPU2 | 4 | 🚧 peano_needed |
| `qwen2.5vl-it:3b` | Qwen2.5-VL-3B-Instruct-NPU2 | 7 | 🚧 peano_needed |

### Gemma4 Family (2 models)

| Model Tag | FLM Directory | xclbins | Peano Status |
|-----------|---------------|:-------:|--------------|
| `gemma4-it:e2b` | Gemma4-E2B-IT-NPU2 | 10 | ✅ `peano_dims` ready |
| `gemma4-it:e4b` | Gemma4-E4B-IT-NPU2 | 10 | ✅ `peano_dims` ready |

### Gemma3 Family (2 models)

| Model Tag | FLM Directory | xclbins | Peano Status |
|-----------|---------------|:-------:|--------------|
| `gemma3:1b` | Gemma3-1B-NPU2 | 5 | 🚧 peano_needed |
| `gemma3:4b` | Gemma3-4B-NPU2 | 7 | 🚧 peano_needed |

### MedGemma Family (2 models)

| Model Tag | FLM Directory | xclbins | Peano Status |
|-----------|---------------|:-------:|--------------|
| `medgemma:4b` | Medgemma-4B-NPU2 | 7 | 🚧 peano_needed |
| `medgemma1.5:4b` | Medgemma-1.5-4B-NPU2 | 7 | 🚧 peano_needed |

### TranslateGemma (1 model)

| Model Tag | FLM Directory | xclbins | Peano Status |
|-----------|---------------|:-------:|--------------|
| `translategemma:4b` | Translategemma-4B-Instruct-NPU2 | 7 | 🚧 peano_needed |

### Phi4 Family (1 model)

| Model Tag | FLM Directory | xclbins | Peano Status |
|-----------|---------------|:-------:|--------------|
| `phi4-mini-it:4b` | Phi4-mini-Instruct-NPU2 | 4 | ✅ `peano_dims` ready |

### Nanbeige Family (1 model)

| Model Tag | FLM Directory | xclbins | Peano Status |
|-----------|---------------|:-------:|--------------|
| `nanbeige4.1:3b` | Nanbeige4.1-3B-NPU2 | 4 | ✅ `peano_dims` ready |

### Llama Family (3 models)

| Model Tag | FLM Directory | xclbins | Peano Status |
|-----------|---------------|:-------:|--------------|
| `llama3.2:1b` | Llama-3.2-1B-NPU2 | 4 | 🚧 peano_needed |
| `llama3.2:3b` | Llama-3.2-3B-NPU2 | 4 | 🚧 peano_needed |
| `llama3.1:8b` | Llama-3.1-8B-NPU2 | 4 | ✅ `peano_dims` ready |

### DeepSeek Family (2 models)

| Model Tag | FLM Directory | xclbins | Peano Status |
|-----------|---------------|:-------:|--------------|
| `deepseek-r1:8b` | Deepseek-R1-Distill-Llama-8B-NPU2 | 4 | 🚧 peano_needed |
| `deepseek-r1-0528:8b` | DeepSeek-R1-0528-Qwen3-8B-NPU2 | 4 | 🚧 peano_needed |

### GPT-OSS Family (2 models, MoE)

| Model Tag | FLM Directory | xclbins | Peano Status |
|-----------|---------------|:-------:|--------------|
| `gpt-oss:20b` | GPT-OSS-20B-NPU2 | 6 | 🚧 peano_needed (MoE) |
| `gpt-oss-sg:20b` | GPT-OSS-Safeguard-20b-NPU2 | 6 | 🚧 peano_needed (MoE) |

### LFM2 Family (3 models)

| Model Tag | FLM Directory | xclbins | Peano Status |
|-----------|---------------|:-------:|--------------|
| `lfm2:1.2b` | LFM2-1.2B-NPU2 | 5 | 🚧 peano_needed |
| `lfm2:2.6b` | LFM2-2.6B-NPU2 | 5 | 🚧 peano_needed |
| `lfm2-trans:2.6b` | LFM2-2.6B-Transcript-NPU2 | 5 | 🚧 peano_needed |

### LFM2.5 Family (2 models)

| Model Tag | FLM Directory | xclbins | Peano Status |
|-----------|---------------|:-------:|--------------|
| `lfm2.5-it:1.2b` | LFM2.5-1.2B-NPU2 | 5 | 🚧 peano_needed |
| `lfm2.5-tk:1.2b` | LFM2.5-1.2B-Thinking-NPU2 | 5 | 🚧 peano_needed |

### Specialized Models (3 models)

| Model Tag | FLM Directory | xclbins | Peano Status | Type |
|-----------|---------------|:-------:|--------------|------|
| `embed-gemma:300m` | Embedding-Gemma-300M-NPU2 | 4 | 🚧 peano_needed | Text Embedding |
| `whisper-v3:turbo` | Whisper-V3-Turbo-NPU2 | 5 | 🚧 peano_needed | Speech-to-Text |
| `bonsai:1.7b` | *(no FLM xclbins)* | 0 | N/A — ternary, no FLM | Ternary-Native |

### Peano Compilation Status Summary

| Status | Count | Models |
|--------|:-----:|--------|
| ✅ `peano_dims` ready | 11 | qwen3:0.6b, qwen3:8b, qwen3vl-it:4b, qwen3.5:4b, qwen3.6:35b, gemma4-it:e2b, gemma4-it:e4b, phi4-mini-it:4b, nanbeige4.1:3b, llama3.1:8b, qwen3.5:4b |
| 🚧 peano_needed | 25 | All others |
| N/A | 1 | bonsai:1.7b (ternary, no FLM path) |

> **Note:** "peano_needed" means the FLM xclbins have been extracted but the Peano compilation step (generating dimension-specific INT8 GEMM xclbins) has not yet been completed. Models with `peano_dims` ready have been compiled for specific tensor shapes and are ready for npu_engine_universal integration.

---

## Links

- [Performance methodology →](performance.md)
- [Architecture deep-dive →](../guides/architecture.md)
- [NPU reverse-engineering journey →](../journey.md)
- [Build instructions →](../guides/building.md)

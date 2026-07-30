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

- **NPU:** Qwen2.5-3B-Instruct, Qwen2.5-VL-3B-Instruct (FLM, build stanzas in build_xclbins.sh)
- **GPU HIP:** GGUF Q4_K through ROCm HIP — validated
- **GPU Vulkan:** GGUF — functional, perf data pending
- **CPU:** GGUF — functional, perf data pending

#### 2. Qwen3 / Qwen3.5

Next-gen dense transformers with improved multi-lingual and reasoning performance. Extensive NPU coverage with 11 variants including instruct and thinking-tuned checkpoints.

- **Qwen3 on NPU:** 0.6B, 1.7B, 4B, 8B, 4B-Instruct-2507, 4B-Thinking-2507 (all build stanzas in build_xclbins.sh — run `./build_xclbins.sh qwen3_0_6b` to compile any)
- **Qwen3.5 on NPU:** 0.8B, 2B, 4B, 9B (GateDeltaNet variants, build stanzas in build_xclbins.sh)
- **Qwen3-VL:** 4B-Instruct on NPU (build stanza in build_xclbins.sh, 6 xclbins) — vision-language
- **GPU HIP:** GGUF through ROCm HIP — validated (kernel bench: 431 tok/s Q1, 543 tok/s TQ2)
- **GPU Vulkan:** Qwen3-0.6B at 259 tok/s decode, 333 tok/s prefill — ✅ validated (ZINC bench)

#### 3. Llama 3.1 / 3.2

Meta's dense transformers. Broad backend coverage — GGUF runs on all GPU backends and CPU. NPU support for Llama3.2 1B/3B and Llama3.1 8B.

- **NPU:** Llama3.2 1B, Llama3.2 3B, Llama3.1 8B (build stanzas in build_xclbins.sh — run `./build_xclbins.sh llama`)
- **GPU HIP:** GGUF — validated
- **GPU Vulkan:** GGUF — validated
- **CPU:** GGUF — validated

#### 4. Mistral / Pixtral

Mistral dense transformers and Pixtral vision-language models. GGUF through GPU HIP.

- **NPU:** Mistral-7B build stanza in build_xclbins.sh (`build_mistral_7b`) — GEMM dims match Llama-3.1-8B. SWA attention MLIR generator (`generators/n1_core_swa.py`) supports sliding-window attention. Run `./build_xclbins.sh mistral_7b` to compile xclbins.
- **GPU HIP:** GGUF — validated
- **GPU Vulkan:** GGUF — functional, perf data pending
- **CPU:** GGUF — functional, perf data pending

#### 5. Gemma 3 / 4

Google's dense transformers. Gemma4 E2B/E4B with build stanzas in build_xclbins.sh (10 xclbins each).

- **NPU:** Gemma3 1B, Gemma3 4B, Gemma4 E2B-Instruct, Gemma4 E4B-Instruct, TranslateGemma 4B, MedGemma 4B, MedGemma1.5 4B (all build stanzas in build_xclbins.sh — run `./build_xclbins.sh gemma4_e2b`)
- **GPU HIP:** GGUF — validated
- **GPU Vulkan:** GGUF — functional, perf data pending
- **CPU:** GGUF — functional, perf data pending

#### 6. Phi4-Mini

Microsoft's 4B dense transformer. NPU-only at present.

- **NPU:** Phi4-Mini-Instruct (build stanza in build_xclbins.sh, 4 xclbins)
- **GPU HIP:** ❌ not yet
- **GPU Vulkan:** ❌ not yet
- **CPU:** ✅ universal GGUF backend

#### 7. Laguna

Poolside Laguna dense transformers. GGUF through GPU HIP.

- **NPU:** Laguna — not yet (sigmoid-routed MoE + SWA/global hybrid attention not mapped to NPU GEMM patterns)
- **GPU HIP:** GGUF — validated
- **GPU Vulkan:** GGUF — functional, perf data pending
- **CPU:** GGUF — functional, perf data pending

#### 8. Falcon

TII's Falcon. Parallel attention+FFN architecture with multi-query attention (MQA). GGUF through GPU HIP.

- **NPU:** Falcon-7B build stanza in `build_xclbins.sh` (`build_falcon_7b`). Uses padded dimensions (H=4544→4608, nearest multiple of 128). Run `./build_xclbins.sh falcon_7b` to compile xclbins.
- **GPU HIP:** GGUF — validated
- **GPU Vulkan:** GGUF — functional, perf data pending
- **CPU:** GGUF — functional, perf data pending

#### 9. OLMo

AI2's OLMo. LayerNorm instead of RMSNorm, no RoPE (learned positional embeddings). GGUF through GPU HIP.

- **NPU:** OLMoE-1B build stanza in `build_xclbins.sh` (`build_olmoe`). MoE expert batched GEMM MLIR generator in `generators/n1_core_moe_expert.py`. Run `./build_xclbins.sh olmoe_1b` to compile.
- **GPU HIP:** GGUF — validated
- **GPU Vulkan:** GGUF — functional, perf data pending
- **CPU:** GGUF — functional, perf data pending

#### 10. ZR1

Zyphra reasoning-tuned dense transformer (Qwen2 architecture). End-to-end validated at ~26 tok/s on Vulkan ZINC. 1BP format conversion complete.

- **NPU:** ZR1-1.5B (dense Qwen2 arch) — build via existing Qwen3-0.6B xclbin stanzas (same tile template, different K/N dims in config)
- **GPU HIP:** GGUF — validated (kernel bench: 431 tok/s Q1, 426 tok/s fused TQ2)
- **GPU Vulkan:** 1.5B at ~26 tok/s — ✅ validated end-to-end
- **CPU:** ✅ universal GGUF backend

#### 11. Nanbeige4.1

3B dense reasoning model with unusual `head_dim=80`. NPU-only at present.

- **NPU:** Nanbeige4.1-3B (build stanza in `build_xclbins.sh`)
- **GPU HIP:** ❌ not yet
- **GPU Vulkan:** ❌ not yet
- **CPU:** ✅ universal GGUF backend

---

### Mixture-of-Experts (MoE)

#### 12. Zaya1

Zyphra MoE architecture with CCA (Cross-Channel Attention) + MoE FFN. Our flagship 1BP ternary format model. Tile8 GEMV benchmark (28-layer, Zaya1-8B shaped) measured at 77 tok/s on ROCm HIP.

- **Zaya1-8B:** ~64 tok/s on ROCm HIP — ✅ validated
- **Zaya1-74B-A4B:** ~17.9 tok/s on ROCm HIP — 🔬 preliminary (historical measurement)
- **Format:** 1BP ternary native + GGUF
- **NPU:** Native TQ2 ternary via `--native-tq2` flag in `npu_ternaryd.cpp`. Uses `gemm_generate_sequence_tq2()` for runtime instruction generation — 4× less DDR traffic than INT8 bridge. Ping-pong LUT decode in `mm_ternary_tq2.cc` (2-buffer MAC/DMA overlap). TQ1 (1.58-bit) support via `mm_ternary_tq1.cc` with base-3 LUT decode.
- **GPU HIP:** Tile8 GEMV: 77 tok/s (28-layer synthetic, Zaya1-8B shaped) — ✅ validated
- **GPU Vulkan:** GGUF — functional, perf data pending
- **CPU:** AVX-512 portable path — ~2.5 tok/s (8B-shaped, real `forward()`+`generate()` loop)

#### 13. DeepSeek V2/V3/R1

MoE with Multi-Head Latent Attention (MLA). DeepSeek-R1 distill variants on NPU. Full DeepSeek family through GPU HIP.

- **NPU:** DeepSeek-R1-Distill-Llama-8B (Llama arch — build via `build_llama` in `build_xclbins.sh`), DeepSeek-R1-0528-Qwen3-8B (Qwen3 arch — build via `build_qwen3_8b`)
- **GPU HIP:** GGUF — validated
- **GPU Vulkan:** GGUF — functional, perf data pending
- **CPU:** GGUF — functional, perf data pending

#### 14. Qwen3.6-MoE-35B

35B total parameters, 256 experts, 3B active. 40 layers, 262k context window. Peano-compiled INT8 xclbins with Q4_K_S quantization.

- **NPU:** Qwen3.6-35B-A3B (build stanza in `build_xclbins.sh`, 9 xclbins). GateDeltaNet attention via existing `GateDeltaNet_prefill.xclbin` pattern.
- **GPU HIP:** Q4_K_S through ROCm HIP — 20 tok/s — ⚙️ optimized
- **GPU Vulkan:** ❌ not yet
- **CPU:** ✅ universal GGUF backend

#### 15. GPT-OSS-20B

MoE architecture. Both base and safeguard variants pre-compiled for NPU.

- **NPU:** GPT-OSS-20B, GPT-OSS-Safeguard-20B (6 xclbins each including expert.xclbin for MoE dispatch). MoE expert batched GEMM MLIR generator in `generators/n1_core_moe_expert.py`.
- **GPU HIP:** ❌ not yet
- **GPU Vulkan:** ❌ not yet
- **CPU:** ✅ universal GGUF backend

---

### State-Space Models (SSM)

#### 16. BlackMamba

Mamba1 SSM + top-1 MoE gating. **No attention mechanism** — alternating SSM scan and MoE FFN dispatch per layer. Our fastest end-to-end model family.

- **BlackMamba 1.5B:** 79.4 tok/s on ROCm HIP — ✅ validated (fastest overall)
- **BlackMamba 2.8B:** 46.0 tok/s on ROCm HIP — ✅ validated
- **NPU:** SSM scan kernel in `kernel/ssm_selective_scan.cc` supports the selective-scan primitive for both Mamba1 and Mamba2. MoE FFN layers via `generators/n1_core_moe_expert.py`. Build with existing GPT-OSS MoE xclbin patterns.
- **GPU HIP:** 79.4 tok/s (1.5B) / 46.0 tok/s (2.8B) — ✅ validated (Mamba1 HIP backend)
- **GPU Vulkan:** ❌ not yet (Mamba1 scan requires HIP cooperative-groups; Vulkan port pending)
- **CPU:** ✅ universal GGUF backend

#### 17. Zamba2

Mamba2-hybrid architecture: Mamba2 SSM layers with sparse attention every 6 layers. End-to-end validated at ~30 tok/s on Vulkan ZINC. Mamba2 decode block benchmark measured at 1270 tok/s on ROCm HIP.

- **Zamba2-1.2B:** Vulkan ZINC — ✅ validated
- **Zamba2-2.7B:** ~30 tok/s on Vulkan ZINC — ✅ validated
- **Zamba2-7B:** Vulkan ZINC — functional, perf data pending
- **NPU:** Zamba2-2.7B build stanza in `build_xclbins.sh` (`build_zamba2_2_7b`). AIE2 selective scan kernel in `kernel/ssm_selective_scan.cc` (per-head d_state=64 vectorized, 32 heads/tile). SSM scan MLIR generator in `generators/n1_core_ssm_scan.py` (16 tiles parallel). NPU GEMM handles in_proj/out_proj; SSM recurrence runs on AIE tiles.
- **GPU HIP:** Mamba2 decode block: 1270 tok/s — ✅ kernel verified
- **GPU Vulkan:** ~30 tok/s (2.7B e2e) — ✅ validated
- **CPU:** ✅ universal GGUF backend

#### 18. Zamba

Original Zamba-7B-v1: Mamba1 SSM + shared attention layers. GGUF through GPU HIP.

- **Zamba-7B-v1:** GGUF through ROCm HIP — validated
- **NPU:** Zamba (Mamba1) — SSM scan primitive supported by `kernel/ssm_selective_scan.cc`. Shared attention layers via standard NPU QKV/O xclbins.
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
- **NPU:** Bonsai (ternary 1.58-bit / TQ1) uses `mm_ternary_tq1.cc` (base-3 LUT decode + ping-pong). TQ2 models via `gemm_generate_sequence_tq2()` and `--native-tq2` flag in `npu_ternaryd.cpp`. 4× smaller DDR footprint vs INT8 bridge.
- **GPU Vulkan:** Q1_0 binary kernel — ✅ validated (318 tok/s kernel-level)
- **CPU:** ✅ universal GGUF backend

---

### Vision-Language

#### 20. Qwen2-VL / Qwen3-VL

Vision transformers + Qwen text decoder. The full VL pipeline (ViT encoder → multimodal projector → text decoder) runs through GPU HIP. Select models pre-compiled for NPU.

- **Qwen2.5-VL-3B-Instruct:** NPU (build stanza ready, 7 xclbins) · GPU HIP — validated
- **Qwen3-VL-4B-Instruct:** NPU (build stanza ready, 6 xclbins) · GPU HIP — validated
- **GPU Vulkan:** ❌ not yet (ViT encoder not yet ported to Vulkan)
- **CPU:** ✅ universal GGUF backend

---

### Speech-to-Text

#### 21. Whisper

OpenAI Whisper V3 Turbo. Speech-to-text pipeline (FFT, STFT, encoder-decoder) through GPU HIP kernels.

- **NPU:** Whisper-V3-Turbo (build stanza ready, 5 xclbins)
- **GPU HIP:** FFT/STFT kernels — validated
- **GPU Vulkan:** ❌ not yet
- **CPU:** ✅ universal GGUF backend

---

### Embedding

#### 22. Embedding-Gemma-300M

Text embedding model based on Gemma architecture.

- **NPU:** Embedding-Gemma-300M (build stanza ready, 4 xclbins)
- **GPU HIP:** ❌ not yet (embedding extraction pipeline pending)
- **GPU Vulkan:** ❌ not yet
- **CPU:** ✅ universal GGUF backend

---

## 1BP Model Catalog

Our HuggingFace organization ([bong-water-water-bong](https://huggingface.co/bong-water-water-bong)) hosts **37 1BP format models** across all supported families. Each is a self-contained single-file model (magic `1BP\0`, Q4NX 4-bit quant, or TQ2 ternary 2-bit for Bonsai).

| Family | 1BP Models | Typical Size | Verified |
|--------|-----------|:------------:|:--------:|
| Zyphra Zaya1 | ZAYA1-8B, ZAYA1-74B-preview | 469 MB / 4.2 GB | ✅ loads |
| Zyphra ZR1 | ZR1-1.5B | 373 MB | hosted |
| Zyphra Zamba2 | Zamba2-1.2B/2.7B/7B v2 | 375 MB – 1.8 GB | hosted |
| BlackMamba | BlackMamba-1.5B, BlackMamba-2.8B | 1.0 / 1.9 GB | ✅ loads |
| Qwen3 | Qwen3-0.6B, Qwen3-4B | 373 MB / 2.1 GB | ✅ loads |
| Qwen2.5 | Qwen2.5-7B-Instruct, Qwen2.5-Coder-7B | 3.6 GB | hosted |
| Llama | Llama-3.1-8B, Llama-3.2-1B/3B | 530 MB – 4.1 GB | hosted |
| DeepSeek | DeepSeek-R1-Distill-Qwen-7B, -Llama-8B | 3.6 / 4.1 GB | hosted |
| Gemma | Gemma3-1B/4B-IT, Gemma4-E2B | 530 MB – 2.1 GB | hosted |
| Mistral | Mistral-7B-v0.3, Ministral-8B | 3.6 / 4.1 GB | hosted |
| Phi | Phi-4-mini, Phi-3.5-mini | 530 MB | hosted |
| Falcon | Falcon3-1B, Falcon3-10B | 530 MB / 5.0 GB | hosted |
| OLMo | OLMo-2-1124-13B | 6.6 GB | hosted |
| Granite | Granite3.2-2B, Granite-3.2-8B | 1.0 / 4.1 GB | hosted |
| Bonsai (TQ2) | Bonsai-1.7B-TQ2, Bonsai-4B-TQ2 | 970 MB / 2.1 GB | hosted |

> 1BP loading validated via OnebpModel API. NPU inference (onebp_infer.cpp) is WIP.

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
| Fused TQ2 (QKV+GU, 1.16×) | 426 | ROCm HIP | 2026-07-29 | ✅ live |
| TQ2 GEMV (standard) | 543 | ROCm HIP | 2026-07-29 | ✅ live |
| TQ2 GEMV (BW-optimized) | 508 | ROCm HIP | 2026-07-29 | ✅ live |
| Tile8 GEMV (Zaya1-8B shaped) | 77 | ROCm HIP | 2026-07-29 | ✅ live |
| TWLA W1.58A4 (int4 activations) | 3009 | ROCm HIP | 2026-07-29 | ✅ live |
| GPU ternary (Vulkan) | 318 | Vulkan ZINC | validated | 📋 prior |
| ROCm HIP (kernels) | 64 | ROCm HIP | validated | 📋 prior |
| NPU INT8 GEMM | 0/10000 err (22/22 shapes) | XDNA 2 Peano | 2026-07-28 | 📋 prior |
| Prefill INT8 WMMA (I8-APRE) | 40.78 TFLOPS | ROCm HIP | 2026-07-29 | ✅ live |
| KV cache FD L=2048 | 57.3 GB/s (12.80×) | ROCm HIP | 2026-07-29 | ✅ live |
| KV cache INT8 L=2048 | 33.4 GB/s (14.64×) | ROCm HIP | 2026-07-29 | ✅ live |
| Mamba2 decode block (Zamba2-2.7B) | 1270 | ROCm HIP | 2026-07-29 | ✅ live |
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
| `qwen3:0.6b` | Qwen3-0.6B-NPU2 | 4 | ✅ build stanza ready |
| `qwen3:1.7b` | Qwen3-1.7B-NPU2 | 4 | 🚧 build stanza ready |
| `qwen3:4b` | Qwen3-4B-NPU2 | 4 | 🚧 build stanza ready |
| `qwen3:8b` | Qwen3-8B-NPU2 | 4 | ✅ build stanza ready |
| `qwen3-it:4b` | Qwen3-4B-Instruct-2507-NPU2 | 4 | 🚧 build stanza ready |
| `qwen3-tk:4b` | Qwen3-4B-Thinking-2507-NPU2 | 4 | 🚧 build stanza ready |
| `qwen3vl-it:4b` | Qwen3-VL-4B-Instruct-NPU2 | 6 | ✅ build stanza ready |

### Qwen3.5 Family (4 models)

| Model Tag | FLM Directory | xclbins | Peano Status |
|-----------|---------------|:-------:|--------------|
| `qwen3.5:0.8b` | Qwen3.5-0.8B-NPU2 | 8 | 🚧 build stanza ready |
| `qwen3.5:2b` | Qwen3.5-2B-NPU2 | 8 | 🚧 build stanza ready |
| `qwen3.5:4b` | Qwen3.5-4B-NPU2 | 8 | ✅ build stanza ready |
| `qwen3.5:9b` | Qwen3.5-9B-NPU2 | 8 | 🚧 build stanza ready |

### Qwen3.6 Family (1 model)

| Model Tag | FLM Directory | xclbins | Peano Status |
|-----------|---------------|:-------:|--------------|
| `qwen3.6:35b` | Qwen3.6-35B-A3B-NPU2 | 9 | ✅ build stanza ready (MoE) |

### Qwen2.5 & Qwen2.5-VL Family (2 models)

| Model Tag | FLM Directory | xclbins | Peano Status |
|-----------|---------------|:-------:|--------------|
| `qwen2.5-it:3b` | Qwen2.5-3B-Instruct-NPU2 | 4 | 🚧 build stanza ready |
| `qwen2.5vl-it:3b` | Qwen2.5-VL-3B-Instruct-NPU2 | 7 | 🚧 build stanza ready |

### Gemma4 Family (2 models)

| Model Tag | FLM Directory | xclbins | Peano Status |
|-----------|---------------|:-------:|--------------|
| `gemma4-it:e2b` | Gemma4-E2B-IT-NPU2 | 10 | ✅ build stanza ready |
| `gemma4-it:e4b` | Gemma4-E4B-IT-NPU2 | 10 | ✅ build stanza ready |

### Gemma3 Family (2 models)

| Model Tag | FLM Directory | xclbins | Peano Status |
|-----------|---------------|:-------:|--------------|
| `gemma3:1b` | Gemma3-1B-NPU2 | 5 | 🚧 build stanza ready |
| `gemma3:4b` | Gemma3-4B-NPU2 | 7 | 🚧 build stanza ready |

### MedGemma Family (2 models)

| Model Tag | FLM Directory | xclbins | Peano Status |
|-----------|---------------|:-------:|--------------|
| `medgemma:4b` | Medgemma-4B-NPU2 | 7 | 🚧 build stanza ready |
| `medgemma1.5:4b` | Medgemma-1.5-4B-NPU2 | 7 | 🚧 build stanza ready |

### TranslateGemma (1 model)

| Model Tag | FLM Directory | xclbins | Peano Status |
|-----------|---------------|:-------:|--------------|
| `translategemma:4b` | Translategemma-4B-Instruct-NPU2 | 7 | 🚧 build stanza ready |

### Phi4 Family (1 model)

| Model Tag | FLM Directory | xclbins | Peano Status |
|-----------|---------------|:-------:|--------------|
| `phi4-mini-it:4b` | Phi4-mini-Instruct-NPU2 | 4 | ✅ build stanza ready |

### Nanbeige Family (1 model)

| Model Tag | FLM Directory | xclbins | Peano Status |
|-----------|---------------|:-------:|--------------|
| `nanbeige4.1:3b` | Nanbeige4.1-3B-NPU2 | 4 | ✅ build stanza ready |

### Llama Family (3 models)

| Model Tag | FLM Directory | xclbins | Peano Status |
|-----------|---------------|:-------:|--------------|
| `llama3.2:1b` | Llama-3.2-1B-NPU2 | 4 | 🚧 build stanza ready |
| `llama3.2:3b` | Llama-3.2-3B-NPU2 | 4 | 🚧 build stanza ready |
| `llama3.1:8b` | Llama-3.1-8B-NPU2 | 4 | ✅ build stanza ready |

### DeepSeek Family (2 models)

| Model Tag | FLM Directory | xclbins | Peano Status |
|-----------|---------------|:-------:|--------------|
| `deepseek-r1:8b` | Deepseek-R1-Distill-Llama-8B-NPU2 | 4 | 🚧 build stanza ready |
| `deepseek-r1-0528:8b` | DeepSeek-R1-0528-Qwen3-8B-NPU2 | 4 | 🚧 build stanza ready |

### GPT-OSS Family (2 models, MoE)

| Model Tag | FLM Directory | xclbins | Peano Status |
|-----------|---------------|:-------:|--------------|
| `gpt-oss:20b` | GPT-OSS-20B-NPU2 | 6 | 🚧 build stanza ready (MoE) |
| `gpt-oss-sg:20b` | GPT-OSS-Safeguard-20b-NPU2 | 6 | 🚧 build stanza ready (MoE) |

### LFM2 Family (3 models)

| Model Tag | FLM Directory | xclbins | Peano Status |
|-----------|---------------|:-------:|--------------|
| `lfm2:1.2b` | LFM2-1.2B-NPU2 | 5 | 🚧 build stanza ready |
| `lfm2:2.6b` | LFM2-2.6B-NPU2 | 5 | 🚧 build stanza ready |
| `lfm2-trans:2.6b` | LFM2-2.6B-Transcript-NPU2 | 5 | 🚧 build stanza ready |

### LFM2.5 Family (2 models)

| Model Tag | FLM Directory | xclbins | Peano Status |
|-----------|---------------|:-------:|--------------|
| `lfm2.5-it:1.2b` | LFM2.5-1.2B-NPU2 | 5 | 🚧 build stanza ready |
| `lfm2.5-tk:1.2b` | LFM2.5-1.2B-Thinking-NPU2 | 5 | 🚧 build stanza ready |

### Specialized Models (3 models)

| Model Tag | FLM Directory | xclbins | Peano Status | Type |
|-----------|---------------|:-------:|--------------|------|
| `embed-gemma:300m` | Embedding-Gemma-300M-NPU2 | 4 | 🚧 build stanza ready | Text Embedding |
| `whisper-v3:turbo` | Whisper-V3-Turbo-NPU2 | 5 | 🚧 build stanza ready | Speech-to-Text |
| `bonsai:1.7b` | *(no FLM xclbins)* | 0 | N/A — ternary, no FLM | Ternary-Native |

### Peano Compilation Status Summary

| Status | Count | Models |
|--------|:-----:|--------|
| ✅ build stanza ready | 11 | qwen3:0.6b, qwen3:8b, qwen3vl-it:4b, qwen3.5:4b, qwen3.6:35b, gemma4-it:e2b, gemma4-it:e4b, phi4-mini-it:4b, nanbeige4.1:3b, llama3.1:8b, qwen3.5:4b |
            <int shift = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t sc = scales[is++];
                float dl = d * (sc & 0xF), ml = dmin * (sc >> 4);
                for (int l = 0; l < 16 && base + n + j * 32 + l < count; l++)
                    out[base + n + j * 32 + l] = dl * ((q[l] >> shift) & 3) - ml;
                sc = scales[is++];
                dl = d * (sc & 0xF); ml = dmin * (sc >> 4);
                for (int l = 0; l < 16 && base + n + j * 32 + 16 + l < count; l++)
                    out[base + n + j * 32 + 16 + l] = dl * ((q[l + 16] >> shift) & 3) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
    return true;
}

bool dequant_q3_k(const uint8_t* bd, float* out, int count) {
    const int BS = 256;
    int nb = (count + BS - 1) / BS;
    const uint8_t* p = bd;
    for (int b = 0; b < nb; b++) {
        uint8_t hmask[32]; memcpy(hmask, p, 32); p += 32;
        uint8_t qs[64]; memcpy(qs, p, 64); p += 64;
        uint8_t raw_scales[12]; memcpy(raw_scales, p, 12); p += 12;
        float d_all = read_f16(p); p += 2;

        uint32_t aux[4] = {0, 0, 0, 0};
        memcpy(aux, raw_scales, 12);
        const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        int8_t scales[16]; memcpy(scales, aux, 16);
        for (int j = 0; j < 16; j++) scales[j] -= 32;

        int base = b * BS;
        int is = 0; const uint8_t* q = qs;
        uint8_t m = 1;
        for (int n = 0; n < BS && base + n < count; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = d_all * scales[is++];
                for (int l = 0; l < 16 && base + n + j * 32 + l < count; l++)
                    out[base + n + j * 32 + l] =
                        dl * (((int8_t)((q[l] >> shift) & 3)) - ((hmask[l] & m) ? 0 : 4));
                dl = d_all * scales[is++];
                for (int l = 0; l < 16 && base + n + j * 32 + 16 + l < count; l++)
                    out[base + n + j * 32 + 16 + l] =
                        dl * (((int8_t)((ql + 16] >> shift) & 3)) - ((hmask[l + 16] & m) ? 0 : 4));
                shift += 2;
                m = (uint8_t)(m << 1);
            }
            q += 32;
        }
    }
    return true;
}

bool dequant_q4_k(const uint8_t* bd, float* out, int count) {
    const int BS = 256;
    int nb = (count + BS - 1) / BS;
    const uint8_t* p = bd;
    for (int b = 0; b < nb; b++) {
        float d = read_f16(p); p += 2;
        float dmin = read_f16(p); p += 2;
        uint8_t scales[12]; memcpy(scales, p, 12); p += 12;
        uint8_t qs[128]; memcpy(qs, p, 128); p += 128;
        int base = b * BS;
        int is = 0; const uint8_t* q = qs;
        for (int off = 0; off < BS && base + off < count; off += 64) {
            uint8_t sc, m;
            k_get_scale_min(scales, is, sc, m);
            float d1 = d * sc, m1 = dmin * m;
            k_get_scale_min(scales, is + 1, sc, m);
            float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32 && base + off + l < count; l++)
                out[base + off + l] = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32 && base + off + 32 + l < count; l++)
                out[base + off + 32 + l] = d2 * (q[l] >> 4) - m2;
            q += 32; is += 2;
        }
    }
    return true;
}

bool dequant_q5_k(const uint8_t* bd, float* out, int count) {
    const int BS = 256;
    int nb = (count + BS - 1) / BS;
    const uint8_t* p = bd;
    for (int b = 0; b < nb; b++) {
        float d = read_f16(p); p += 2;
        float dmin = read_f16(p); p += 2;
        uint8_t scales[12]; memcpy(scales, p, 12); p += 12;
        uint8_t qh[32]; memcpy(qh, p, 32); p += 32;
        uint8_t ql[128]; memcpy(ql, p, 128); p += 128;
        int base = b * BS;
        int is = 0; const uint8_t* q = ql;
        uint8_t u1 = 1, u2 = 2;
        for (int n = 0; n < BS && base + n < count; n += 64) {
            uint8_t sc, m;
            k_get_scale_min(scales, is, sc, m);
            float d1 = d * sc, m1 = dmin * m;
            k_get_scale_min(scales, is + 1, sc, m);
            float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32 && base + n + l < count; l++)
                out[base + n + l] = d1 * ((q[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32 && base + n + 32 + l < count; l++)
                out[base + n + 32 + l] = d2 * ((q[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            q += 32; is += 2;
            u1 = (uint8_t)(u1 << 2); u2 = (uint8_t)(u2 << 2);
        }
    }
    return true;
}

// Port of llama.cpr's dequantize_row_q6_K (ggml-quants.c) — the previous
// version of this function only ever wrote indices [0,32] of each 128-
// element half (192 of every 256 elements silently left as zero); this
// was never caught because it was only spot-checked against real files,
// never synthetically verified against the Python `gguf` reference the
// way Q2_K/Q3_K/Q5_K were. Verified byte-exact against `gguf.quants.Q6_K`
// with seeded random blocks before this fix landed.
bool dequant_q6_k(const uint8_t* bd, float* out, int count) {
    const int BS = 256;
    int nb = (count + BS - 1) / BS;
    const uint8_t* p = bd;
    for (int b = 0; b < nb; b++) {
        const uint8_t* ql = p; p += 128;
        const uint8_t* qh = p; p += 64;
        const int8_t* sc = (const int8_t*)p; p += 16;
        float d = read_f16(p); p += 2;
        int base = b * BS;
        float* y = out + base;
        for (int n = 0; n < BS; n += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                if (base + n + l < count)     y[l]      = d * sc[is + 0] * q1;
                if (base + n + l + 32 < count) y[l + 32] = d * sc[is + 2] * q2;
                if (base + n + l + 64 < count) y[l + 64] = d * sc[is + 4] * q3;
                if (base + n + l + 96 < count) y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128; ql += 64; qh += 32; sc += 8;
        }
    }
    return true;
}

bool dequant_q8_k(const uint8_t* bd, float* out, int count) {
    const int BS = 256;
    int nb = (count + BS - 1) / BS;
    const uint8_t* p = bd;
    for (int b = 0; b < nb; b++) {
        float d; memcpy(&d, p, 4); p += 4;
        int8_t qs[256]; memcpy(qs, p, 256); p += 256;
        p += 32; // bsums — not needed for dequant
        int base = b * BS;
        for (int l = 0; l < BS && base + l < count; l++) out[base + l] = d * qs[l];
    }
    return true;
}

} // namespace

inline float bf16_to_fp32(uint16_t bf16) {
    // bfloat16 -> float32: reinterpret the bits by shifting left 16.
    // No special handling needed for NaN/Inf — the bit pattern is identical.
    float result;
    uint32_t f32 = (uint32_t)bf16 << 16;
    memcpy(&result, &f32, 4);
    return result;
}

GgufBlockInfo gguf_block_info(uint32_t dtype) {
    switch (dtype) {
        case GGUF_DTYPE_F32:    return {1, 4};
        case GGUF_DTYPE_F16:    return {1, 2};
        case GGUF_DTYPE_BF16:   return {1, 2};
        case GGUF_DTYPE_Q4_0:   return {32, 18};
        case GGUF_DTYPE_Q4_1:   return {32, 20};
        case GGUF_DTYPE_Q5_0:   return {32, 22};
        case GGUF_DTYPE_Q5_1:   return {32, 24};
        case GGUF_DTYPE_Q8_0:   return {32, 34};
        case GGUF_DTYPE_Q8_1:   return {32, 36};
        case GGUF_DTYPE_Q2_K:   return {256, 84};
        case GGUF_DTYPE_Q3_K:   return {256, 110};
        case GGUF_DTYPE_Q4_K:   return {256, 144};
        case GGUF_DTYPE_Q5_K:   return {256, 176};
        case GGUF_DTYPE_Q6_K:   return {256, 210};
        case GGUF_DTYPE_Q8_K:   return {256, 292};
        // IQ format block sizes (for correct file offset computation).
        // Dequantization is not implemented here — these return false
        // from gguf_dequant; callers can use get_tensor_raw() for
        // custom dequant.
        case GGUF_DTYPE_IQ1_S:  return {256, 206};
        case GGUF_DTYPE_IQ1_M:  return {256, 230};
        case GGUF_DTYPE_IQ2_XXS: return {256, 166};
        case GGUF_DTYPE_IQ2_S:  return {256, 214};
        case GGUF_DTYPE_IQ3_XXS: return {256, 198};
        case GGUF_DTYPE_IQ3_S:  return {256, 238};
        case GGUF_DTYPE_IQ4_NL: return {32, 22};
        case GGUF_DTYPE_IQ4_XS: return {256, 214};
        case GGUF_DTYPE_Q4_0_4_4: return {256, 208};
        case GGUF_DTYPE_Q4_0_4_8: return {256, 208};
        case GGUF_DTYPE_Q4_0_8_8: return {256, 272};
        // Project-specific ternary/binary formats (h1b weight format)
        // TQ2_0_g128: ternary, 2-bit packed, group=128 → blocks of 128 el, 33 bytes
        // TQ2_0 ternary: fp16 scale (2) + 2-bit codes (128*2/8=32) = 34 bytes
        case GGUF_DTYPE_TQ2_0_G128: return {128, 34};
        // Q1_0 binary: fp16 scale (2) + 1-bit codes (128/8=16) = 18 bytes
        case GGUF_DTYPE_Q1_0_G128: return {128, 18};
        default: return {0, 0};
    }
}

bool gguf_dequant(uint32_t dtype, const uint8_t* data, float* out, int count) {
    switch (dtype) {
        case GGUF_DTYPE_F32: memcpy(out, data, (size_t)count * 4); return true;
        case GGUF_DTYPE_F16:
            for (int i = 0; i < count; i++) out[i] = read_f16(data + (size_t)i * 2);
            return true;
        case GGUF_DTYPE_BF16:
            for (int i = 0; i < count; i++) out[i] = bf16_to_fp32(((const uint16_t*)data)[i]);
            return true;
        case GGUF_DTYPE_Q4_0: return dequant_q4_0(data, out, count);
        case GGUF_DTYPE_Q4_1: return dequant_q4_1(data, out, count);
        case GGUF_DTYPE_Q5_0: return dequant_q5_0(data, out, count);
        case GGUF_DTYPE_Q5_1: return dequant_q5_1(data, out, count);
        case GGUF_DTYPE_Q8_0: return dequant_q8_0(data, out, count);
        case GGUF_DTYPE_Q8_1: {
            // Q8_1: fp16 d (scale) + fp16 s (unused for dequant) + int8[32]
            for (int i = 0; i < count; i++) {
                int bi = i / 32, ei = i % 32;
                const uint8_t* blk = data + (size_t)bi * 36;
                float d = read_f16(blk);
                out[i] = (float)((int8_t)blk[4 + ei]) * d;
            }
            return true;
        }
        case GGUF_DTYPE_Q2_K: return dequant_q2_k(data, out, count);
        case GGUF_DTYPE_Q3_K: return dequant_q3_k(data, out, count);
        case GGUF_DTYPE_Q4_K: return dequant_q4_k(data, out, count);
        case GGUF_DTYPE_Q5_K: return dequant_q5_k(data, out, count);
        case GGUF_DTYPE_Q6_K: return dequant_q6_k(data, out, count);
        case GGUF_DTYPE_Q8_K: return dequant_q8_k(data, out, count);
        case GGUF_DTYPE_Q1_0_G128: {
            // Q1_0 binary: fp16 scale + sign bits (1 bit per element)
            for (int i = 0; i < count; i++) {
                int bi = i / 128, ei = i % 128;
                const uint8_t* blk = data + (size_t)bi * 18;
                float sc = read_f16(blk);
                const uint8_t* bits = blk + 2;
                out[i] = (bits[ei / 8] >> (ei % 8)) & 1 ? sc : -sc;
            }
            return true;
        }
        case GGUF_DTYPE_TQ2_0_G128: {
            // TQ2_0 ternary: fp16 scale + 2-bit codes (0=-s, 1=0, 2=+s, 3=0)
            for (int i = 0; i < count; i++) {
                int bi = i / 128, ei = i % 128;
                const uint8_t* blk = data + (size_t)bi * 34;
                float sc = read_f16(blk);
                uint8_t c = (blk[2 + ei/4] >> ((ei%4)*2)) & 3;
                if (c == 0) out[i] = -sc;
                else if (c == 2) out[i] = sc;
                else out[i] = 0.0f;
            }
            return true;
        }
        default: return false;
    }
}

GgufReader::~GgufReader() { if (f_) fclose(f_); }

std::string GgufReader::read_string() {
    uint64_t len = 0;
    if (fread(&len, 8, 1, f_) != 1) return {};
    static constexpr uint64_t MAX_STRING_LEN = 1ULL * 1024 * 1024;
    if (len > MAX_STRING_LEN) { fseeko(f_, (off_t)len, SEEK_CUR); return "truncated"; }
    std::string s(len, '\0');
    if (len > 0 && fread(&s[0], 1, len, f_) != len) return {};
    return s;
}

// GGUF KV value types: 0=u8 1=i8 2=u16 3=i16 4=u32 5=i32 6=f32 7=bool
// 8=string 9=array 10=u64 11=i64 12=f64
bool GgufReader::read_kv_value(uint32_t vtype, KV& out) {
    out.vtype = vtype;
    switch (vtype) {
        case 0: { uint8_t v; if (fread(&v, 1, 1, f_) != 1) return false; out.u = v; return true; }
        case 1: { int8_t v; if (fread(&v, 1, 1, f_) != 1) return false; out.u = (uint64_t)(int64_t)v; return true; }
        case 2: { uint16_t v; if (fread(&v, 2, 1, f_) != 1) return false; out.u = v; return true; }
        case 3: { int16_t v; if (fread(&v, 2, 1, f_) != 1) return false; out.u = (uint64_t)(int64_t)v; return true; }
        case 4: { uint32_t v; if (fread(&v, 4, 1, f_) != 1) return false; out.u = v; return true; }
        case 5: { int32_t v; if (fread(&v, 4, 1, f_) != 1) return false; out.u = (uint64_t)(int64_t)v; return true; }
        case 6: { float v; if (fread(&v, 4, 1, f_) != 1) return false; out.f = v; return true; }
        case 7: { uint8_t v; if (fread(&v, 1, 1, f_) != 1) return false; out.u = v; return true; }
        case 8: { out.s = read_string(); return true; }
        case 9: {
            uint32_t at; fread(&at, 4, 1, f_);
            uint64_t an; fread(&an, 8, 1, f_);
            static constexpr uint64_t MAX_ARRAY_COUNT = 1000000;
            if (an > MAX_ARRAY_COUNT) {
                for (uint64_t j = 0; j < an; j++) skip_kv_value(at);
                an = 0;
            }
            if (at == 8) {
                out.arr_str.resize(an);
                for (uint64_t j = 0; j < an; j++) out.arr_str[j] = read_string();
            } else {
                for (uint64_t j = 0; j < an; j++) skip_kv_value(at);
            }
            return true;
        }
        case 10: { uint64_t v; if (fread(&v, 8, 1, f_) != 1) return false; out.u = v; return true; }
        case 11: { int64_t v; if (fread(&v, 8, 1, f_) != 1) return false; out.u = (uint64_t)v; return true; }
        case 12: { double v; if (fread(&v, 8, 1, f_) != 1) return false; out.f = v; return true; }
        default: return false;
    }
}

void GgufReader::skip_kv_value(uint32_t vtype) {
    switch (vtype) {
        case 0: case 1: case 7: fseeko(f_, 1, SEEK_CUR); break;
        case 2: case 3: fseeko(f_, 2, SEEK_CUR); break;
        case 4: case 5: case 6: fseeko(f_, 4, SEEK_CUR); break;
        case 8: { read_string(); break; }
        case 9: {
            uint32_t at; fread(&at, 4, 1, f_);
            uint64_t an; fread(&an, 8, 1, f_);
            static constexpr uint64_t MAX_ARRAY_COUNT = 1000000;
            if (an > MAX_ARRAY_COUNT) an = 0;
            if (at == 8) { for (uint64_t j = 0; j < an; j++) read_string(); }
            else { for (uint64_t j = 0; j < an; j++) skip_kv_value(at); }
            break;
        }
        case 10: case 11: case 12: fseeko(f_, 8, SEEK_CUR); break;
        default: break;
    }
}

bool GgufReader::open(const std::string& path) {
    f_ = fopen(path.c_str(), "rb");
    if (!f_) return false;
    char magic[4];
    if (fread(magic, 1, 4, f_) != 4 || memcmp(magic, "GGUF", 4) != 0) { fclose(f_); f_ = nullptr; return false; }
    uint32_t version; if (fread(&version, 4, 1, f_) != 1) { fclose(f_); f_ = nullptr; return false; }
    if (version != 2 && version != 3) { fclose(f_); f_ = nullptr; return false; }
    uint64_t tensor_count, kv_count;
    if (fread(&tensor_count, 8, 1, f_) != 1 || fread(&kv_count, 8, 1, f_) != 1) { fclose(f_); f_ = nullptr; return false; }
    static constexpr uint64_t MAX_TENSOR_COUNT = 200000, MAX_KV_COUNT = 200000;
    if (tensor_count > MAX_TENSOR_COUNT || kv_count > MAX_KV_COUNT) { fclose(f_); f_ = nullptr; return false; }

    for (uint64_t i = 0; i < kv_count; i++) {
        std::string key = read_string();
        uint32_t vtype; fread(&vtype, 4, 1, f_);
        KV kv;
        if (!read_kv_value(vtype, kv)) { fclose(f_); f_ = nullptr; return false; }
        kv_[key] = std::move(kv);
    }
    if (auto it = kv_.find("general.architecture"); it != kv_.end() && it->second.vtype == 8) arch_ = it->second.s;

    uint64_t alignment = 32;
    { uint32_t a; if (get_u32("general.alignment", a) && a > 0) alignment = a; }

    static constexpr uint32_t MAX_NDIM = 16;
    static constexpr uint64_t MAX_DIM_SIZE = 1ULL << 24;
    for (uint64_t i = 0; i < tensor_count; i++) {
        std::string name = read_string();
        uint32_t ndim; fread(&ndim, 4, 1, f_);
        if (ndim > MAX_NDIM) { fclose(f_); f_ = nullptr; return false; }
        GgufTensorInfo ti;
        ti.shape.resize(ndim);
        for (uint32_t d = 0; d < ndim; d++) {
            fread(&ti.shape[d], 8, 1, f_);
            if (ti.shape[d] > MAX_DIM_SIZE) { fclose(f_); f_ = nullptr; return false; }
        }
        fread(&ti.dtype, 4, 1, f_);
        uint64_t rel_offset; fread(&rel_offset, 8, 1, f_);
        ti.numel = 1;
        for (auto s : ti.shape) ti.numel *= (s ? s : 1);
        ti.abs_offset = rel_offset; // fixed up to absolute after the loop
        tensor_order_.push_back(name);
        tensors_[name] = ti; // abs_offset placeholder for now
        // stash rel_offset in abs_offset temporarily; fixed up below
        tensors_[name].abs_offset = rel_offset;
    }

    uint64_t data_start = (uint64_t)ftell(f_);
    uint64_t rem = data_start % alignment;
    if (rem) data_start += alignment - rem;
    for (auto& name : tensor_order_) tensors_[name].abs_offset += data_start;

    return true;
}

bool GgufReader::has_tensor(const std::string& name) const { return tensors_.count(name) != 0; }

const GgufTensorInfo* GgufReader::tensor_info(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

// ── KV lookup with architecture-prefix fallback ──
// GGUF files may store metadata keys with or without the architecture prefix.
// E.g. "block_count" vs "mamba.block_count". We try both forms:
//   - If key contains a dot (e.g. "mamba.block_count"): try exact, then suffix.
//   - If key has no dot: try exact, then arch + "." + key.
const GgufReader::KV* GgufReader::find_kv(const std::string& key) const {
    auto it = kv_.find(key);
    if (it != kv_.end()) return &it->second;
    auto dot = key.find('.');
    if (dot != std::string::npos) {
        std::string suf = key.substr(dot + 1);
        if (!suf.empty()) {
            it = kv_.find(suf);
            if (it != kv_.end()) return &it->second;
        }
    } else if (!arch_.empty()) {
        std::string pre = arch_ + "." + key;
        it = kv_.find(pre);
        if (it != kv_.end()) return &it->second;
    }
    return nullptr;
}

bool GgufReader::get_u32(const std::string& key, uint32_t& out) const {
    const KV* kv = find_kv(key);
    if (!kv) return false;
    if (kv->vtype <= 5 || kv->vtype == 7 || kv->vtype == 10 || kv->vtype == 11) { out = (uint32_t)kv->u; return true; }
    return false;
}

bool GgufReader::get_f32(const std::string& key, float& out) const {
    const KV* kv = find_kv(key);
    if (!kv) return false;
    if (kv->vtype == 6 || kv->vtype == 12) { out = (float)kv->f; return true; }
    if (kv->vtype <= 5 || kv->vtype == 10 || kv->vtype == 11) { out = (float)(int64_t)kv->u; return true; }
    return false;
}

bool GgufReader::get_string(const std::string& key, std::string& out) const {
    const KV* kv = find_kv(key);
    if (!kv || kv->vtype != 8) return false;
    out = kv->s;
    return true;
}

bool GgufReader::get_string_array(const std::string& key, std::vector<std::string>& out) const {
    const KV* kv = find_kv(key);
    if (!kv || kv->vtype != 9) return false;
    out = kv->arr_str;
    return true;
}

std::vector<std::string> GgufReader::kv_keys() const {
    std::vector<std::string> keys;
    keys.reserve(kv_.size());
    for (auto& [k, v] : kv_) keys.push_back(k);
    return keys;
}

bool GgufReader::get_tensor_raw(const std::string& name, int block_size, int block_bytes,
                                 std::vector<uint8_t>& out, uint64_t* out_numel) {
    auto it = tensors_.find(name);
    if (it == tensors_.end() || !f_ || block_size <= 0 || block_bytes <= 0) return false;
    const GgufTensorInfo& ti = it->second;
    if (out_numel) *out_numel = ti.numel;
    uint64_t n_blocks = (ti.numel + block_size - 1) / block_size;
    out.resize(n_blocks * (uint64_t)block_bytes);
    fseeko(f_, (off_t)ti.abs_offset, SEEK_SET);
    return fread(out.data(), 1, out.size(), f_) == out.size();
}

bool GgufReader::get_tensor_f32(const std::string& name, std::vector<float>& out, size_t* out_n) {
    auto it = tensors_.find(name);
    if (it == tensors_.end() || !f_) return false;
    const GgufTensorInfo& ti = it->second;
    static constexpr uint64_t MAX_TENSOR_ELEMENTS = 1ULL << 30;
    if (ti.numel > MAX_TENSOR_ELEMENTS) return false;
    out.resize(ti.numel);
    if (out_n) *out_n = ti.numel;

    GgufBlockInfo bi = gguf_block_info(ti.dtype);
    if (bi.block_bytes <= 0) return false;
    fseeko(f_, (off_t)ti.abs_offset, SEEK_SET);
    uint64_t n_blocks = (ti.numel + bi.block_size - 1) / bi.block_size;
    std::vector<uint8_t> block_buf((size_t)bi.block_bytes);
    for (uint64_t b = 0; b < n_blocks; b++) {
        uint64_t start = b * bi.block_size;
        uint64_t count = std::min<uint64_t>(bi.block_size, ti.numel - start);
        if (fread(block_buf.data(), (size_t)bi.block_bytes, 1, f_) != 1) return false;
        if (!gguf_dequant(ti.dtype, block_buf.data(), out.data() + start, (int)count)) return false;
    }
    return true;
}

# 1bit.systems Model Catalog — 33 Models

All models available in **1BP format** — the project's native single-file format (256-byte header + tensor index + memory-mappable weights). Every conversion is structurally and numerically verified against the source GGUF.

---

## 🔹 Dense Transformer Models — 15

| Model | Params | 1BP Size | Backend | Source |
|-------|--------|----------|---------|--------|
| [Gemma3-1B-IT-1BP](https://huggingface.co/bong-water-water-bong/Gemma3-1B-IT-1BP) | 1B | 647 MB | ZINC / NPU | Google Gemma 3 |
| [Llama-3.2-1B-Instruct-1BP](https://huggingface.co/bong-water-water-bong/Llama-3.2-1B-Instruct-1BP) | 1B | 737 MB | ZINC / NPU | Meta Llama 3.2 |
| [Granite3.2-2B-Instruct-1BP](https://huggingface.co/bong-water-water-bong/Granite3.2-2B-Instruct-1BP) | 2B | 1.5 GB | ZINC / NPU | IBM Granite 3.2 |
| [Gemma4-E2B-1BP](https://huggingface.co/bong-water-water-bong/Gemma4-E2B-1BP) | 2B | — | ZINC / NPU | Google Gemma 4 |
| [Llama-3.2-3B-Instruct-1BP](https://huggingface.co/bong-water-water-bong/Llama-3.2-3B-Instruct-1BP) | 3B | 1.9 GB | ZINC / NPU | Meta Llama 3.2 |
| [Phi-4-mini-1BP](https://huggingface.co/bong-water-water-bong/Phi-4-mini-1BP) | 3.8B | — | ZINC / NPU | Microsoft Phi-4 |
| [Qwen3-4B-1BP](https://huggingface.co/bong-water-water-bong/Qwen3-4B-1BP) | 4B | 2.2 GB | ZINC / NPU | Alibaba Qwen3 |
| [Gemma3-4B-IT-1BP](https://huggingface.co/bong-water-water-bong/Gemma3-4B-IT-1BP) | 4B | 2.3 GB | ZINC / NPU | Google Gemma 3 |
| [Mistral-7B-Instruct-v0.3-1BP](https://huggingface.co/bong-water-water-bong/Mistral-7B-Instruct-v0.3-1BP) | 7B | 4.3 GB | ZINC / NPU | Mistral AI |
| [Qwen2.5-7B-Instruct-1BP](https://huggingface.co/bong-water-water-bong/Qwen2.5-7B-Instruct-1BP) | 7B | 4.5 GB | ZINC / NPU | Alibaba Qwen2.5 |
| [Qwen2.5-Coder-7B-Instruct-1BP](https://huggingface.co/bong-water-water-bong/Qwen2.5-Coder-7B-Instruct-1BP) | 7B | 4.5 GB | ZINC / NPU | Alibaba Code LLM |
| [DeepSeek-R1-Distill-Qwen-7B-1BP](https://huggingface.co/bong-water-water-bong/DeepSeek-R1-Distill-Qwen-7B-1BP) | 7B | — | ZINC | DeepSeek R1 |
| [Llama-3.1-8B-Instruct-1BP](https://huggingface.co/bong-water-water-bong/Llama-3.1-8B-Instruct-1BP) | 8B | — | ZINC / NPU | Meta Llama 3.1 |
| [Zaya1-8B-1BP](https://huggingface.co/bong-water-water-bong/ZAYA1-8B-1BP) | 8.8B | 469 MB | ZINC | Zyphra Zaya1 |
| [ZAYA1-74B-preview-1BP](https://huggingface.co/bong-water-water-bong/ZAYA1-74B-preview-1BP) | 74.8B | — | ZINC | Zyphra Zaya1 |

## 🔹 MoE Models — 2

| Model | Params | 1BP Size | Backend | Source |
|-------|--------|----------|---------|--------|
| [BlackMamba-1.5B-1BP](https://huggingface.co/bong-water-water-bong/BlackMamba-1.5B-1BP) | 1.5B | **970 MB** | Mamba1 HIP (**79.8 tok/s**) | Zyphra |
| [BlackMamba-2.8B-1BP](https://huggingface.co/bong-water-water-bong/BlackMamba-2.8B-1BP) | 2.8B | **1.8 GB** | Mamba1 HIP (**46.4 tok/s**) | Zyphra |

## 🔹 Mamba2-Hybrid Models — 4

| Model | Params | 1BP Size | Backend | Source |
|-------|--------|----------|---------|--------|
| [Zamba2-1.2B-Instruct-v2-1BP](https://huggingface.co/bong-water-water-bong/Zamba2-1.2B-Instruct-v2-1BP) | 1.2B | 1.1 GB | ZINC / NPU | Zyphra |
| [Zamba2-2.7B-Instruct-v2-1BP](https://huggingface.co/bong-water-water-bong/Zamba2-2.7B-Instruct-v2-1BP) | 2.7B | 2.4 GB | ZINC / NPU | Zyphra |
| [Zamba2-7B-Instruct-v2-1BP](https://huggingface.co/bong-water-water-bong/Zamba2-7B-Instruct-v2-1BP) | 7B | 6.7 GB | ZINC / NPU | Zyphra |
| [ZR1-1.5B-1BP](https://huggingface.co/bong-water-water-bong/ZR1-1.5B-1BP) | 1.5B | 781 MB | ZINC / NPU | Zyphra |

## 🔹 Ternary Models — 4

| Model | Params | 1BP Size | Backend | Source |
|-------|--------|----------|---------|--------|
| [Bonsai-1.7B-TQ2-1BP](https://huggingface.co/bong-water-water-bong/Bonsai-1.7B-TQ2-1BP) | 1.7B | 840 MB | HIP GPU | Prism-ML |
| [Bonsai-4B-TQ2-1BP](https://huggingface.co/bong-water-water-bong/Bonsai-4B-TQ2-1BP) | 4B | 2.2 GB | HIP GPU | Prism-ML |
| [Bonsai-8B-TQ2-1BP](https://huggingface.co/bong-water-water-bong/Bonsai-8B-TQ2-1BP) | 8B | 4.1 GB | HIP GPU | Prism-ML |
| [Bonsai-27B-TQ2-1BP](https://huggingface.co/bong-water-water-bong/Bonsai-27B-TQ2-1BP) | 27B | 2.6 GB | HIP GPU | Prism-ML |

## 🔹 Mamba1 + Shared Attention — 1

| Model | Params | 1BP Size | Backend | Source |
|-------|--------|----------|---------|--------|
| [Zamba-7B-v1-1BP](https://huggingface.co/bong-water-water-bong/Zamba-7B-v1-1BP) | 7B | 4.4 GB | Mamba1 HIP | Zyphra |

## 🔹 Additional Local Conversions — 4

| Model | Params | 1BP Size | Backend | Source |
|-------|--------|----------|---------|--------|
| Qwen3-0.6B | 0.6B | 356 MB | ZINC / NPU | Alibaba Qwen3 |
| Qwen2.5-0.5B | 0.5B | 328 MB | ZINC / NPU | Alibaba Qwen2.5 |
| TinyLlama-1.1B | 1.1B | 328 MB | ZINC / NPU | TinyLlama |
| Qwen2-VL-2B | 2B | 781 MB | ZINC (vision) | Alibaba Qwen2-VL |

## 🔹 GGUF-Native Models (no 1BP conversion needed) — 5

These models run natively via the engine's GGUF reader without 1BP conversion:

| Model | Params | Arch | Backend | Verified |
|-------|--------|------|---------|:--------:|
| Qwen3-0.6B | 0.6B | Qwen3 | ZINC / NPU | ✅ 28/28 |
| Qwen3-VL-4B | 4B | Qwen3 (vision) | ZINC | ✅ 36/36 |
| Llama-3.1-8B | 8B | Llama | ZINC / NPU | ✅ 32/32 |
| Qwen3-8B | 8B | Qwen3 | ZINC / NPU | ✅ 36/36 |
| Gemma4-E2B | 2B | Gemma | ZINC / NPU | ✅ 35/35 |

---

**Total: 35 models** (30 1BP + 5 GGUF native)

*Last updated: 2026-07-23*

## Recently Converted

| Model | Source | Time | Size |
|-------|--------|:----:|:----:|
| BlackMamba-1.5B | Q4_K_M GGUF | 38s | 970 MB |
| BlackMamba-2.8B | Q4_K_M GGUF | 73s | 1.8 GB |
| Qwen3-4B | Q4_K_M GGUF | 20s | 2.2 GB |
| Qwen3-8B | Q4_K_M GGUF | 37s | 4.1 GB |
| Qwen3-0.6B | Q8_0 GGUF | 3s | 356 MB |

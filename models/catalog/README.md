# 1bit.systems Model Catalog

Models in **1BP format** — the project's native single-file format (256-byte header + tensor index + memory-mappable weights). Every conversion is structurally and numerically verified against the source GGUF.

---

## 🔹 Dense Transformer Models

| Model | Params | 1BP Size | Backend | Source |
|-------|--------|----------|---------|--------|
| [Gemma3-1B-IT-1BP](https://huggingface.co/bong-water-water-bong/Gemma3-1B-IT-1BP) | 1B | 647 MB | ZINC / NPU | Google Gemma 3 |
| [Llama-3.2-1B-Instruct-1BP](https://huggingface.co/bong-water-water-bong/Llama-3.2-1B-Instruct-1BP) | 1B | 737 MB | ZINC / NPU | Meta Llama 3.2 |
| [Granite3.2-2B-Instruct-1BP](https://huggingface.co/bong-water-water-bong/Granite3.2-2B-Instruct-1BP) | 2B | 1.5 GB | ZINC / NPU | IBM Granite 3.2 |
| [Gemma4-E2B-1BP](https://huggingface.co/bong-water-water-bong/Gemma4-E2B-1BP) | 2B | — | ZINC / NPU | Google Gemma 4 |
| [Llama-3.2-3B-Instruct-1BP](https://huggingface.co/bong-water-water-bong/Llama-3.2-3B-Instruct-1BP) | 3B | 1.9 GB | ZINC / NPU | Meta Llama 3.2 |
| [Phi-4-mini-1BP](https://huggingface.co/bong-water-water-bong/Phi-4-mini-1BP) | 3.8B | — | ZINC / NPU | Microsoft Phi-4 |
| [Qwen3-4B-1BP](https://huggingface.co/bong-water-water-bong/Qwen3-4B-1BP) | 4B | — | ZINC / NPU | Alibaba Qwen3 |
| [Gemma3-4B-IT-1BP](https://huggingface.co/bong-water-water-bong/Gemma3-4B-IT-1BP) | 4B | 2.3 GB | ZINC / NPU | Google Gemma 3 |
| [Mistral-7B-Instruct-v0.3-1BP](https://huggingface.co/bong-water-water-bong/Mistral-7B-Instruct-v0.3-1BP) | 7B | 4.3 GB | ZINC / NPU | Mistral AI |
| [Qwen2.5-7B-Instruct-1BP](https://huggingface.co/bong-water-water-bong/Qwen2.5-7B-Instruct-1BP) | 7B | 4.5 GB | ZINC / NPU | Alibaba Qwen2.5 |
| [Qwen2.5-Coder-7B-Instruct-1BP](https://huggingface.co/bong-water-water-bong/Qwen2.5-Coder-7B-Instruct-1BP) | 7B | 4.5 GB | ZINC / NPU | Alibaba Code LLM |
| [DeepSeek-R1-Distill-Qwen-7B-1BP](https://huggingface.co/bong-water-water-bong/DeepSeek-R1-Distill-Qwen-7B-1BP) | 7B | — | ZINC | DeepSeek R1 |
| [Llama-3.1-8B-Instruct-1BP](https://huggingface.co/bong-water-water-bong/Llama-3.1-8B-Instruct-1BP) | 8B | — | ZINC / NPU | Meta Llama 3.1 |
| [Zaya1-8B-1BP](https://huggingface.co/bong-water-water-bong/ZAYA1-8B-1BP) | 8.8B | — | ZINC | Zyphra Zaya1 |
| [ZAYA1-74B-preview-1BP](https://huggingface.co/bong-water-water-bong/ZAYA1-74B-preview-1BP) | 74.8B | — | ZINC | Zyphra Zaya1 |

## 🔹 MoE Models

| Model | Params | 1BP Size | Backend | Source |
|-------|--------|----------|---------|--------|
| [BlackMamba-1.5B-1BP](https://huggingface.co/bong-water-water-bong/BlackMamba-1.5B-1BP) | 1.5B | — | Mamba1 HIP GPU (**79.8 tok/s**) | Zyphra |
| [BlackMamba-2.8B-1BP](https://huggingface.co/bong-water-water-bong/BlackMamba-2.8B-1BP) | 2.8B | — | Mamba1 HIP GPU (**46.4 tok/s**) | Zyphra |

## 🔹 Mamba2-Hybrid Models

| Model | Params | 1BP Size | Backend | Source |
|-------|--------|----------|---------|--------|
| [Zamba2-1.2B-Instruct-v2-1BP](https://huggingface.co/bong-water-water-bong/Zamba2-1.2B-Instruct-v2-1BP) | 1.2B | — | ZINC Vulkan | Zyphra |
| [Zamba2-2.7B-Instruct-v2-1BP](https://huggingface.co/bong-water-water-bong/Zamba2-2.7B-Instruct-v2-1BP) | 2.7B | — | ZINC Vulkan | Zyphra |
| [Zamba2-7B-Instruct-v2-1BP](https://huggingface.co/bong-water-water-bong/Zamba2-7B-Instruct-v2-1BP) | 7B | — | ZINC Vulkan | Zyphra |
| [ZR1-1.5B-1BP](https://huggingface.co/bong-water-water-bong/ZR1-1.5B-1BP) | 1.5B | — | ZINC Vulkan | Zyphra |

## 🔹 Ternary Models

| Model | Params | 1BP Size | Backend | Source |
|-------|--------|----------|---------|--------|
| [Bonsai-4B-TQ2-1BP](https://huggingface.co/bong-water-water-bong/Bonsai-4B-TQ2-1BP) | 4B | — | HIP GPU | Prism-ML |

## 🔹 Mamba1 + Shared Attention

| Model | Params | 1BP Size | Backend | Source |
|-------|--------|----------|---------|--------|
| [Zamba-7B-v1-1BP](https://huggingface.co/bong-water-water-bong/Zamba-7B-v1-1BP) | 7B | — | Mamba1 HIP GPU | Zyphra |

---

**Total: 24 models** (8 new additions: Gemma3-1B, Gemma3-4B, Granite3.2-2B, Llama-3.2-1B/3B, Mistral-7B, Qwen2.5-7B, Qwen2.5-Coder-7B)

*Last updated: 2026-07-21*

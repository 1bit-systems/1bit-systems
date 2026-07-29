# New 1BP Model Additions — Phase 1 Plan

## Goal: Expand 46+ → 60+ 1BP models

Targeting the most impactful VLMs, reasoning models, and small efficient models
that this engine's existing backends can already handle (or need minimal additions for).

> **Zyphra ecosystem documented**: Zyphra's **ZUNA1.1** (🧠 EEG, diffusion autoencoder),
> **ZUNA**, and **Zonos-v0.1** / **ZONOS2** (🗣️ TTS) are now documented in the model catalog
> as non-1BP-convertible reference entries. These are not LLMs and cannot run on this engine,
> but are listed for completeness of the Zyphra ecosystem.

---

## 1. Vision-Language Models (VLMs)

| Model | Params | Arch | Existing Support | Effort |
|-------|:------:|------|:----------------:|:------:|
| **SmolVLM-256M-Instruct** | 256M | smolvlm (SigLIP ViT + LLM) | ViT ✅ (SigLIP), LLM → new | **Low** |
| **SmolVLM-2.2B-Instruct** | 2.2B | smolvlm | ViT ✅ (SigLIP), LLM → new | **Low** |
| **LLaVA-NeXT-LLaMA3-8B** | 8B | llava (CLIP ViT-L + llama3) | ViT ✅, llama ✅ | **Low** |
| **LLaVA-1.6-Mistral-7B** | 7B | llava (CLIP ViT-L + mistral) | ViT ✅, mistral ✅ | **Low** |
| **Molmo-7B-D** | 7B | molmo (CLIP ViT-L + OLMoE) | ViT ✅, new MoE variant | **Medium** |
| **Ovis-1.6-Gemma2-9B** | 9B | ovis (SigLIP + gemma2) | ViT ✅, gemma2 ✅ | **Low** |
| **PaliGemma-3B** | 3B | paligemma (SigLIP + gemma) | ViT ✅, gemma ✅ | **Low** |
| **Florence-2-base** | 0.23B | florence | New arch | **Medium** |
| **Qwen2.5-VL-3B** | 3B | qwen2vl | ✅ Already supported | **Trivial** |
| **Qwen2.5-VL-7B** | 7B | qwen2vl | ✅ Already supported | **Trivial** |

### Implementation for VLMs (all Low effort):
- Extend `gguf_to_onebp.cpp`: add arch string → `OnebpArch::ONEBP_VISION` mapping
- Extend `vision_encoder.h`: add `SmolVLM` arch config (SigLIP-ViT + projector)
- Most VLMs use standard CLIP/SigLIP ViT encoders + standard LLM decoder
- The `vision_server` already handles `/v1/chat/completions` with image_url

## 2. State-of-the-Art Reasoning Models

| Model | Params | Arch | Support | Effort |
|-------|:------:|------|:-------:|:------:|
| **DeepSeek-R1-Distill-Qwen-7B** | 7B | qwen2 | ✅ | **Trivial** |
| **DeepSeek-R1-Distill-Llama-8B** | 8B | llama | ✅ | **Trivial** |
| **DeepSeek-R1-Distill-Qwen-14B** | 14B | qwen2 | ✅ | **Trivial** |
| **DeepSeek-R1-Distill-Qwen-32B** | 32B | qwen2 | ✅ | **Trivial** |
| **DeepSeek-V3-R1-671B (FP8)** | 671B | deepseek2 | New MLA MoE | **High** |
| **Qwen3-14B-A4B** | 14B | qwen3 | ✅ | **Trivial** |
| **Qwen3-32B-A4B** | 32B | qwen3 | ✅ | **Trivial** |
| **Phi-4-mini-instruct** | 3.8B | phi3 | ✅ | **Trivial** |
| **Phi-4-moe-instruct** | 14B | phi3 (MoE) | New MoE variant | **Medium** |
| **Mistral-Small-3.1-24B-Instruct** | 24B | mistral | ✅ | **Low** |
| **Gemma-3-12B-it** | 12B | gemma3 | ⚠️ Need arch check | **Low** |

## 3. Small & Efficient Models (NPU-friendly)

| Model | Params | Arch | Support | Effort |
|-------|:------:|------|:-------:|:------:|
| **SmolLM2-135M** | 135M | llama | ✅ | **Trivial** |
| **SmolLM2-360M** | 360M | llama | ✅ | **Trivial** |
| **SmolLM2-1.7B** | 1.7B | llama | ✅ | **Trivial** |
| **TinyLlama-1.1B** | 1.1B | qwen2 | ✅ Already in catalog | **Trivial** |
| **Qwen3-0.6B** | 0.6B | qwen3 | ✅ Already in catalog | **Trivial** |
| **Granite-3.2-8B-Instruct** | 8B | granite | ✅ | **Trivial** |
| **Falcon3-10B-Instruct** | 10B | llama | ✅ (llama compat) | **Trivial** |

## 4. LoRA-Ready Adaptations

| Adapter | For Model | Rank | Source |
|---------|-----------|:----:|--------|
| Qwen3-instruct-lora | Qwen3-8B | 64 | Fine-tune via Unsloth → GGUF |
| DeepSeek-R1-style-lora | Qwen3-8B | 128 | Reasoning style transfer |
| Zaya1-chat-lora | Zaya1-8B | 32 | Chat fine-tune |
| BlackMamba-instruct-lora | BlackMamba-1.5B | 16 | Mamba SSM LoRA |
| Bonsai-code-lora | Bonsai-1.7B | 32 | Code fine-tune |

## Conversion Commands

```bash
# Trivial: just download GGUF and convert
./build/gguf_to_onebp smolvlm-256m.gguf models/SmolVLM-256M.1bp
./build/gguf_to_onebp llava-next-llama3-8b.gguf models/LLaVA-NeXT-8B.1bp
./build/gguf_to_onebp deepseek-r1-7b.gguf models/DeepSeek-R1-Distill-7B.1bp

# Medium: may need minor arch string additions
# See src/gguf_loader.cpp -> rcpp_arch_from_string()
# And include/onebp_format.h -> OnebpArch enum
```

## Architecture String Mapping

Add these to `src/gguf_loader.cpp` in `rcpp_arch_from_string()`:

```cpp
// New VLMs
if (arch == "smolvlm")     return ONEBP_ARCH_VISION;
if (arch == "llava")        return ONEBP_ARCH_VISION;
if (arch == "molmo")        return ONEBP_ARCH_DENSE;  // uses OLMoE arch
if (arch == "ovis")         return ONEBP_ARCH_VISION;
if (arch == "paligemma")    return ONEBP_ARCH_VISION;
if (arch == "florence")     return ONEBP_ARCH_VISION;

// New MoE reasoning
if (arch == "deepseek2")    return ONEBP_ARCH_MOE;
if (arch == "phi_moe")      return ONEBP_ARCH_MOE;
```

## Batch Conversion Script

Use the existing `tools/batch_convert.sh`:

```bash
# Convert all newly downloaded GGUF models
bash tools/batch_convert.sh --all

# Or convert specific models
for model in smolvlm-256m llava-next-8b deepseek-r1-7b; do
    ./build/gguf_to_onebp models/${model}.gguf models/${model}.1bp
done
```

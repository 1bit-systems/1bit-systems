# `huggingface.co/amd` &mdash; what loads on this stack

AMD publishes 300 models on HuggingFace. **Zero of them are GGUFs.** The headline LLMs are AWQ / Quark / OGA-hybrid checkpoints intended for AMD's closed Ryzen AI EP &mdash; which on Linux ships only as Ubuntu 24.04 `.deb` packages bound to AMD's own `amdxdna` kernel module. On a CachyOS / Arch host this stack is unreachable today, even via Distrobox (the closed EP requires the AMD-shipped `.ko`, not the in-kernel `amdxdna` your distro provides). See [`docs/reddit-npu-gate.md`](reddit-npu-gate.md) Correction #2.5 for the live test result.

The rest of the catalog is vision / ASR / embedding / non-LLM artifacts, and a long tail of FP8 / MXFP4 quantizations targeted at datacenter Instinct chips, not Strix Halo.

For the **`flm:npu`** lane on Strix Halo, FastFlowLM ships its own model collection (qwen3, gemma3, phi4-mini, lfm2, llama3.2, deepseek-r1, gpt-oss); pull those via `lemonade pull <name>-FLM`, not from `huggingface.co/amd`.

For the **`llamacpp:rocm`** / **`llamacpp:vulkan`** lanes, GGUFs come from Bonsai, BitNet, Manojb, unsloth, gianni-cor, lilyanatia &mdash; not AMD.

## Tier A &mdash; loadable on `llamacpp:*` today (0 / 300)

None. AMD ships no GGUF on HuggingFace.

## Tier B &mdash; loadable on `flm:npu` today (0 / 300)

None. FLM uses its own (overlapping in family, distinct in binary) model collection.

## Tier C &mdash; AWQ / Quark / OGA / RyzenAI &mdash; **NOT loadable on Arch** (79 / 300)

These need AMD's closed Ryzen AI EP under Ubuntu 24.04 with AMD's amdxdna `.ko`. Top 30 by download:

| Downloads | Model |
|---:|---|
| 18,083 | `amd/Llama-3.1-8B-Instruct-FP8-KV-Quark-test` |
| 10,037 | `amd/Qwen2.5-1.5B-Instruct-ptpc-Quark-ts` |
| 77 | `amd/Llama-2-7b-hf-awq-g128-int4-asym-bf16-onnx-ryzen-strix` |
| 64 | `amd/DeepSeek-R1-Distill-Llama-70B-dml-int4-awq-block-128` |
| 53 | `amd/LFM2-1.2B-ONNX_rai_1.7.1` |
| 49 | `amd/Llama-3.2-1B-Instruct-awq-g128-int4-asym-fp16-onnx-hybrid` |
| 24 | `amd/Qwen3-8B-awq-quant-onnx-hybrid` |
| 21 | `amd/DeepSeek-R1-Distill-Qwen-7B-awq-asym-uint4-g128-lmhead-onnx-hybrid` |
| 19 | `amd/Qwen2.5-7B-Instruct-awq-g128-int4-asym-bf16-onnx-ryzen-strix` |
| 18 | `amd/DeepSeek-R1-Distill-Llama-8B-awq-asym-uint4-g128-lmhead` |
| 15 | `amd/Phi-3-mini-4k-instruct-awq-g128-int4-asym-bf16-onnx-ryzen-strix` |
| 15 | `amd/Llama-3.2-3B-Instruct-awq-g128-int4-asym-fp16-onnx-hybrid` |
| 13 | `amd/DeepSeek-R1-Distill-Llama-8B-awq-asym-uint4-g128-lmhead-onnx-hybrid` |
| 13 | `amd/Qwen3-1.7B-awq-quant-onnx-hybrid` |
| 12 | `amd/Qwen1.5-7B-Chat-awq-g128-int4-asym-fp16-onnx-hybrid` |
| 12 | `amd/Llama-3.2-3B-Instruct-awq-g128-int4-asym-bf16-onnx-ryzen-strix` |
| 11 | `amd/Qwen2.5-7B-Instruct-awq-uint4-asym-g128-lmhead-g32-fp16-onnx-hybrid` |
| 10 | `amd/Llama2-7b-chat-awq-g128-int4-asym-bf16-onnx-ryzen-strix` |
| 9 | `amd/Phi-3.5-mini-instruct-awq-g128-int4-asym-bf16-onnx-ryzen-strix` |
| 9 | `amd/Llama-3.2-1B-Instruct-awq-g128-int4-asym-bf16-onnx-ryzen-strix` |
| 8 | `amd/Qwen1.5-7B-Chat-awq-g128-int4-asym-bf16-onnx-ryzen-strix` |
| 8 | `amd/Phi-3.5-mini-instruct-awq-g128-int4-asym-fp16-onnx-hybrid` |
| 8 | `amd/DeepSeek-R1-Distill-Qwen-1.5B-awq-asym-uint4-g128-lmhead-onnx-hybrid` |
| 8 | `amd/Llama-3.1-8B-Instruct-awq-asym-uint4-g128-lmhead-onnx-hybrid` |
| 8 | `amd/Qwen2.5-1.5B-Instruct-awq-g128-int4-asym-bf16-onnx-ryzen-strix` |
| 7 | `amd/Phi-3-mini-4k-instruct-awq-g128-int4-asym-fp16-onnx-hybrid` |
| 7 | `amd/Mistral-7B-Instruct-v0.3-awq-g128-int4-asym-fp16-onnx-hybrid` |
| 7 | `amd/Mistral-7B-v0.3-awq-asym-uint4-g128-lmhead-onnx-hybrid` |
| 7 | `amd/DeepSeek-R1-Distill-Llama-8B-awq-g128-int4-asym-bf16-onnx-ryzen-strix` |
| 7 | `amd/AMD-OLMo-1B-SFT-DPO-awq-g128-int4-asym-bf16-onnx-ryzen-strix` |

## Tier D &mdash; Vision / CNN (10 / 300)

ResNet50, FDViT (b/s/ti), YOLO (v8m / v5s / v3 / x-s). ONNX format, separate inference path; not LLMs. Use directly with `onnxruntime` if needed.

## Tier E &mdash; Whisper / ASR (4 / 300)

`amd/NPU-Whisper-Base-Small`, `amd/whisper-small-onnx-npu`, `amd/whisper-medium-onnx-npu`, `amd/whisper-large-turbo-onnx-npu`. NPU-specific, not loadable via `whispercpp:vulkan`. Our STT lane uses `Whisper-Tiny` from openai/whisper.cpp anyway.

## Tier F &mdash; Embedding (1 / 300)

`amd/NPU-Nomic-embed-text-v1.5-ryzen-strix-cpp`. NPU-specific. We use the `nomic-embed-text-v2-moe-GGUF` GGUF for our embedding lane instead.

## Tier G &mdash; Other (206 / 300)

Datacenter quantizations (FP8-KV, MXFP4 for Instinct), research checkpoints, intermediate artifacts. Top by downloads:

| Downloads | Model |
|---:|---|
| 118,070 | `amd/DeepSeek-R1-MXFP4` |
| 90,005 | `amd/Kimi-K2.5-MXFP4` |
| 39,418 | `amd/Llama-3.1-8B-Instruct-FP8-KV` |
| 35,690 | `amd/DeepSeek-R1-0528-MXFP4` |
| 13,180 | `amd/PARD-Llama-3.2-1B` |
| 11,186 | `amd/Llama-3.3-70B-Instruct-FP8-KV` |

MXFP4 / FP8-KV are AMD MI300-class server quantizations, not for Strix Halo NPU.

## Summary

| Tier | Count | Loadable on this Strix Halo / Arch box today? |
|---|---:|---|
| A &mdash; GGUF (`llamacpp:*`) | 0 | n/a — none shipped |
| B &mdash; FLM (`flm:npu`) | 0 | n/a — FLM uses own collection |
| C &mdash; AWQ / Quark / OGA (Ryzen AI EP) | 79 | ❌ Ubuntu 24.04 + AMD `.ko` only |
| D &mdash; Vision | 10 | partial — needs raw `onnxruntime` |
| E &mdash; Whisper | 4 | partial — NPU-specific, not via whispercpp |
| F &mdash; Embedding | 1 | partial — NPU-specific |
| G &mdash; Other (FP8/MXFP4/research) | 206 | mostly server-targeted |

Captured **2026-04-28** via `https://huggingface.co/api/models?author=amd&limit=300`. Re-run that query before relying on these counts; AMD adds models frequently.

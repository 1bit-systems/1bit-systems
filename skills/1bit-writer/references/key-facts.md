# 1bit-systems: Key Facts Quick Reference

## Project

- **Name**: 1bit.systems
- **Repo**: https://github.com/bong-water-water-bong/1bit-systems
- **Site**: https://1bit.systems
- **License**: MIT
- **Language**: Pure C++23
- **Binary size**: ~400 KB (server + CLI + daemon)
- **Runtime deps**: Zero Python. Zero Node.js. Zero Rust.
- **Author**: bong-water-water-bong

## Hardware

- **Primary target**: AMD Strix Halo (Ryzen AI Max+ 395)
- **GPU**: Radeon 8060S iGPU (ROCm HIP / Vulkan)
- **NPU**: XDNA 2 (50 TOPS INT8)
- **CPU**: AVX-512 capable

## Backends (9 total)

| Backend | API | Status |
|---------|-----|--------|
| NPU XDNA 2 | XRT xclbin | Production |
| Mamba1 HIP | ROCm HIP | Production |
| ROCm HIP | HIP kernels | Production |
| Vulkan ZINC | GLSL compute | Production |
| Vulkan raw | GLSL compute | Production |
| CPU AVX-512 | Scalar | Production |
| CPU scalar | Scalar | Production |
| GGUF CPU | llama.cpp | Production |
| CUDA (via llama.cpp) | CUDA | Delegated |
| Metal (via llama.cpp) | Metal | Delegated |

## Models Supported — 35 Total (30 1BP + 5 GGUF native)

### By Family

| Family | Count | Models | Primary Backend | Highlight Perf |
|--------|------:|--------|-----------------|---------------|
| **Dense Transformer** | 15 | Qwen3, Qwen2.5, Llama, Gemma, Phi, Mistral, Granite, DeepSeek, Zaya1 | ZINC / NPU | Zaya1-8B tiny 469MB 1BP |
| **Mamba2-Hybrid** | 4 | Zamba2 1.2B/2.7B/7B, ZR1-1.5B | ZINC / NPU | SSM + attention hybrid |
| **Ternary (TQ2)** | 4 | Bonsai 1.7B/4B/8B/27B | HIP GPU | Fused TQ2 415 tok/s |
| **MoE (Mamba1)** | 2 | BlackMamba 1.5B/2.8B | Mamba1 HIP | 79.8 tok/s e2e |
| **Mamba1+SharedAttn** | 1 | Zamba-7B-v1 | Mamba1 HIP | Legacy hybrid |
| **GGUF Native** | 5 | Qwen3-0.6B, Qwen3-8B, Llama-3.1-8B, Qwen3-VL-4B, Gemma4-E2B | All backends | Run without 1BP conversion |
| **Additional Local** | 4 | TinyLlama-1.1B, Qwen2.5-0.5B, Qwen2-VL-2B, Qwen3-0.6B (local) | ZINC / NPU | Small / vision models |

All 1BP models hosted at: https://huggingface.co/bong-water-water-bong
Full catalog: `models/catalog/README.md`

## Key Engineering Milestones

- NPU stack reverse-engineered in 4 days (22 proprietary `.so` → 17.5 MB open source)
- 209 xclbin bitstreams traced back to AIE generators
- 1800+ hour engineering effort across 28 layers of GEMM kernels
- 3 correctness bugs found and fixed in Mamba1 kernel (conv state overflow, A_log exponentiation, HIP device stubs)
- FastFlowLM fully replaced: 87.8 MB closed binary → 17.5 MB open one

## Packaging

deb, snap, AppImage, Homebrew, AUR, Docker, Ollama, curl|bash

## Key Directories

| Path | Content |
|------|---------|
| `src/` | Main source code |
| `engine/npu/` | NPU engine |
| `include/` | Headers (1BP format, GGUF loader, tokenizer, etc.) |
| `site/` | Website (landing page, blog, docs) |
| `site/blog/` | Blog posts (HTML + markdown) |
| `site/benchmarks.json` | SSOT for all benchmark numbers |
| `docs/` | Engineering docs, journey, social media drafts |
| `docs/journey.md` | 1800+ line engineering journal |
| `docs/wiki/` | Wiki content |
| `prompts/` | Agent personality prompts |
| `models/` | Model catalog |
| `tests/` | Test suite |
| `tools/` | Diagnostic and utility tools |
| `scripts/` | Build and benchmark scripts |

## Benchmark SSOT (`site/benchmarks.json`)

Always reference this file for current numbers. Status tags:
- `validated` — Confirmed on real hardware
- `optimized` — Known good but not fully validated end-to-end  
- `end_to_end` — Real model, real prompt
- `corrected` — Previously wrong, now fixed
- `broken` — Does not produce correct output
- `unresolved` — Known issue, not yet fixed
- `raw` — Unvalidated measurement

## Blog Topics Already Covered

1. Reverse-engineered AMD NPU in 4 days
2. Three bugs that broke 97 tok/s
3. One binary to rule them all (model auto-detection)
4. Token router (model-agnostic GGUF routing)
5. NPU optimization sprint (72× speedup)
6. NPU benchmarks (55 TFLOPS)
7. NPU 132× more efficient than GPU
8. Fused layer engine (one xclbin per layer)
9. What reviews didn't tell you about Ryzen AI Halo
10. DSpark speculative decoding (572 tok/s target)
11. Reddit post: Zaya + DFlash draft

## Social Media Accounts

- GitHub: https://github.com/bong-water-water-bong
- Site: https://1bit.systems

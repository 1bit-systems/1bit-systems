# Unified Stack Bench — Strix Halo, both lanes through one Lemonade endpoint

_Captured 2026-04-28 via direct OpenAI-compat curls against `lemond` on `:13305`. Same prompt and `max_tokens` per row. Numbers are single-shot (no warmup, no run averaging) — meant to validate end-to-end stack health, not to publish steady-state peaks._

**Box:** AMD Ryzen AI MAX+ 395 / Radeon 8060S `gfx1151` / XDNA 2 NPU, 124 GB LPDDR5x. Kernel `7.0.2-1-cachyos`. Lemonade Server 10.2.0 with the version-pin patch applied (`flm.npu` bumped from `v0.9.38` → `v0.9.39` to match the AUR `fastflowlm` package). `flm validate` all green; `Lemonade backends list` shows `llamacpp:rocm` (`b1231`), `llamacpp:vulkan` (`b8668`), and `flm:npu` (`v0.9.39`) all `installed`.

## What we are claiming

**Both lanes serve, through the same OpenAI-compat endpoint, on Linux, today.** No proxy gymnastics — just Lemonade's native recipe routing.

```
GET  /v1/models               → both llamacpp and flm models listed
POST /v1/chat/completions     → routes to ROCm (iGPU) or NPU per requested model
```

## Numbers

| Model | Backend | Lane | Prompt tok | Decode tok | Prefill tok/s | Decode tok/s | TTFT |
|---|---|---|---:|---:|---:|---:|---:|
| `LFM2-1.2B-GGUF` (short) | `llamacpp:rocm` | iGPU | 17 | 30 | 891.6 | **222.2** | ~0 |
| `LFM2-1.2B-GGUF` (med) | `llamacpp:rocm` | iGPU | 26 | 180 | 1600.2 | **217.1** | ~0 |
| `LFM2-1.2B-GGUF` (long) | `llamacpp:rocm` | iGPU | 35 | 318 | 2026.5 | **216.6** | ~0 |
| `BitNet-b1.58-3B` (short) | `llamacpp:rocm` | iGPU (1-bit) | 35 | 30 | 859.3 | **76.1** | ~0 |
| `BitNet-b1.58-3B` (med) | `llamacpp:rocm` | iGPU (1-bit) | 38 | 200 | 978.9 | **73.3** | ~0 |
| `BitNet-b1.58-3B` (long) | `llamacpp:rocm` | iGPU (1-bit) | 48 | 350 | 1211.1 | **70.8** | ~0 |
| `qwen3-0.6b-FLM` (short) | `flm:npu` | NPU | 24 | 8 | 52.5 | **87.3** | 457 ms |
| `qwen3-0.6b-FLM` (med) | `flm:npu` | NPU | 35 | 163 | 76.0 | **94.8** | 460 ms |
| `qwen3-0.6b-FLM` (long) | `flm:npu` | NPU | 45 | 175 | 97.6 | **94.5** | 461 ms |
| `qwen3-1.7b-FLM` (short) | `flm:npu` | NPU | 20 | 30 | 35.2 | **43.0** | 568 ms |
| `qwen3-1.7b-FLM` (med) | `flm:npu` | NPU | 31 | 136 | 54.4 | **42.1** | 570 ms |
| `qwen3-1.7b-FLM` (long) | `flm:npu` | NPU | 41 | 200 | 71.3 | **41.8** | 575 ms |
| `deepseek-r1-8b-FLM` (med) | `flm:npu` | NPU | 21 | 300 | 14.7 | **11.3** | 1430 ms |

`BitNet-b1.58-3B` is the `bitnet_b1_58-3B-TQ2_0` ternary checkpoint, served through `llamacpp:rocm` on the iGPU — that row is genuine sub-2-bit ternary inference behind the unified endpoint.

## Headlines

- **iGPU (`llamacpp:rocm`) on Strix Halo serves a 1.2B Q4_K_M at ~217 tok/s** decode steady-state across short/med/long prompts. Prefill scales from ~0.9k tps (short) to ~2.0k tps (long) as the model finds its rhythm.
- **iGPU on a 3B 1-bit/ternary (TQ2_0) checkpoint holds ~71-76 tok/s** decode. Sub-2-bit lane is real — same endpoint, different model name.
- **NPU (`flm:npu`) holds ~95 tok/s decode on qwen3-0.6b** and ~42 tok/s on qwen3-1.7b. NPU is not the throughput champion against the iGPU on small models — its value is *offload* (free up the iGPU for bigger ROCm models) and *low power*. It's also a separate compute lane the user can route to per-request.
- **8B on NPU (`deepseek-r1-8b-FLM`) at ~11 tok/s** with ~1.4 s TTFT. Usable for background agents; not the headline number.
- **Same OpenAI endpoint on `:13305` routes to either lane.** No proxy. Lemonade's native `flm:npu` recipe + `llamacpp:rocm` recipe under one `lemond`.

## What's still red (so we're honest about what "green" means)

- `user.bitnet-b1.58-2B-4T-GGUF-bitnet1582b4t-iq2_bn.gguf` — load fails (`llama-server failed to start`). Stale tensor format (`TYPE_IQ4_NL_4_4` was removed in current `llama-cpp` `b1231`). Needs re-quantizing. Not a stack bug.
- AWQ/OGA model-loading on Linux — still no `xrt`-based equivalent of `onnxruntime_providers_ryzenai.dll`. AMD's 200+ AWQ HF checkpoints don't run on Linux NPU yet. FLM's own model collection (qwen3, gemma3, phi4-mini, lfm2, llama3.2, deepseek-r1, gpt-oss) is the working alternative.
- Brevitas (sub-INT8 / N-bit) integration — still "Coming soon" in optimum-amd docs.
- Diffusion + CLIP on NPU on Linux — still mostly nothing.

## Reproduce

`lemond` running on `:13305` (Lemonade Server with `flm:npu` patched per `install.sh`). For each model, an OpenAI-compat curl against `/v1/chat/completions` with the prompt and `max_tokens` shown. Lemonade returns `timings.*` (llama.cpp lane) or `usage.decoding_speed_tps` / `usage.prefill_speed_tps` / `usage.prefill_duration_ttft` (FLM lane); both are normalized into the table above.

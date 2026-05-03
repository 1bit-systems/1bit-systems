# Fork-everything — 1bit ownership strategy

2026-04-23 lock-in.

## Why own the whole stack

Four concrete reasons, each already bit us:

1. **Wan 2.2 TI2V-5B** failed to load on upstream sd.cpp today (ggml 4D tensor cap vs Wan's 5D patch embedding). Upstream maintainers hadn't patched despite shipping Wan 2.2 docs. Wait or fix — we fixed.
2. **mlx-engine** CI broke on a GitHub 500 during `NripeshN/mlx` clone; we reached into Geramy's upstream and landed the cache mitigation (lemonade PR #19). Being able to PR upstream matters.
3. **BitNet-1.58 support** for the TTS, image, and video lanes exists **nowhere** upstream. If we want first-public ternary video, we have to author the kernels + loaders ourselves.
4. **ggml's 4D tensor ceiling** and non-HIP ternary kernels are a structural limit that upstream won't lift on our timeline.

So the policy is: **everything we depend on at runtime lives in a `bong-water-water-bong/` fork we control.** Upstreams stay pinned as `upstream` remotes for rebasing good changes back in.

## Active forks (2026-04-23)

| Upstream | Our fork | Purpose |
|---|---|---|
| [khimaros/qwen3-tts.cpp](https://github.com/khimaros/qwen3-tts.cpp) | [bong/1bit-tts.cpp](https://github.com/bong-water-water-bong/1bit-tts.cpp) | TTS engine — Qwen3-TTS C++ server |
| [leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) | [bong/stable-diffusion.cpp](https://github.com/bong-water-water-bong/stable-diffusion.cpp) | image + video engine; planned rename to `1bit-image.cpp` |
| [ggerganov/whisper.cpp](https://github.com/ggerganov/whisper.cpp) | [bong/1bit-whisper.cpp](https://github.com/bong-water-water-bong/1bit-whisper.cpp) | STT engine — freshly forked 2026-04-23 |
| [ggml-org/ggml](https://github.com/ggml-org/ggml) | [bong/1bit-ggml](https://github.com/bong-water-water-bong/1bit-ggml) | tensor runtime — base for 5D support + HIP ternary ops |
| [lemonade-sdk/lemonade](https://github.com/lemonade-sdk/lemonade) | [bong/lemonade](https://github.com/bong-water-water-bong/lemonade) | OpenAI-compat frontend (caller-side gateway) |
| `lemonade-sdk/lemon-mlx-engine` | `bong/lemon-mlx-engine` | MLX backend — ours for Apple Silicon + ROCm |
| [lemonade-sdk/llamacpp-rocm](https://github.com/lemonade-sdk/llamacpp-rocm) | [bong/llamacpp-rocm](https://github.com/bong-water-water-bong/llamacpp-rocm) | ROCm-specific llama.cpp builds |
| [cmetz/ec-su_axb35-linux](https://github.com/cmetz/ec-su_axb35-linux) | (tracked via 1bit-watchdog; no fork yet) | EC driver for Bosgame / AXB35-02 |
| [amd/IRON](https://github.com/amd/IRON) | (tracked read-only; compile-time tool, no serve-path fork) | NPU kernel authoring |

## Native originals (always ours)

| Repo | Purpose |
|---|---|
| [`bong/1bit-systems`](https://github.com/bong-water-water-bong/1bit-systems) | this monorepo — Rust orchestration + kernel source of truth |
| [`bong/rocm-cpp`](https://github.com/bong-water-water-bong/rocm-cpp) | HIP kernels (ternary GEMV, Flash-Decoding, rotor-quant KV, sd.cpp-style image ops). Mirror of the folded `rocm-cpp/` subtree inside this repo. |

## Update flow

All tracked forks go through 1bit-watchdog with a 24-hour dwell before auto-merge + rebuild. See `packages.toml [watch.*]` section. Human-gated merges are still fine for risky releases; the watchdog is the default lane.

## Rename queue

- `bong/stable-diffusion.cpp` → `1bit-image.cpp` — wait until Stage 1 of the 1bit-image plan (5D loader + HIP ternary port) lands. Rename is noisy on contributor history; one clean rename rather than two.
- `bong/lemon-mlx-engine` — keep name for now (Geramy publishes releases under this name); revisit if Geramy rebrands upstream.
- `bong/lemonade` — keep name; caller-side gateway only, not a hot-path surface.

## What we still depend on upstream for

- Model weights on Hugging Face (Qwen3-TTS, Wan 2.2, SDXL, whisper, BitNet base). We don't re-host. HF versions are tracked by 1bit-watchdog with their own 24-hour dwell.
- AMD ROCm system runtime (libhip, hipBLAS-via-Tensile path, amdgpu DRM). Host prereq.
- AMD XRT + amdxdna kernel driver for the NPU lane.
- Linux kernel + glibc + standard sysadmin stack.

Everything else is forked.

## Rule alignment

- Rule A (no Python at runtime) — forks don't change this. Python build-time remains allowed for IRON + hf-CLI + model conversion.
- Rule B (C++20 for kernels) — every fork is C++ except IRON (Python DSL → xclbin, compile-time only).
- Rule C (no hipBLAS at runtime) — enforced across every fork that touches the serving path.
- Rule E (NPU stack) — FastFlowLM is the live serving lane. IRON is author-time only for custom kernels; we do not fork it.

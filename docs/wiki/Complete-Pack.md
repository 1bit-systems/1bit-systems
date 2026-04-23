# Complete pack — the six lanes

The full media + inference + NPU surface of 1bit systems as of **2026-04-23**. Every lane is a separate [`packages.toml`](../../packages.toml) component; all six compose under `1bit install all`. Status column is honest — green when the service answers on a live port today, yellow when the pipeline is staged but not yet dispatching end-to-end, blue when the toolchain is proven but the shipping kernel isn't written.

| lane | component (`packages.toml`) | binary / service | model source (HF) | license | status (2026-04-23) | one-command install |
|---|---|---|---|---|---|---|
| **LLM** | `core` | `1bit-server` / `strix-server.service` :8180 | `bong-water-water-bong/bitnet-2p4b-sparse` (Run 5 ship target, `model.bitnet-2p4b`) · current in-service weights are the pre-retrain BitNet-1.58 2B base | MIT | **live** — 66.8 tok/s decode @ 64-tok verified on gfx1151 | `1bit install core` |
| **TTS** | `tts-engine` | `1bit-tts-server` / `1bit-tts.service` :8095 | `khimaros/Qwen3-TTS-12Hz-0.6B-Base-GGUF` (fallback) · `khimaros/Qwen3-TTS-Tokenizer-12Hz-GGUF` (vocoder) · `bong-water-water-bong/qwen3-tts-0p6b-ternary` (Run 6 ternary, pending) | Qwen permissive | **live** — `/v1/audio/voices` responds on :8095 | `1bit install tts-engine` |
| **STT** | `stt-engine` | whisper.cpp / `1bit-whisper.service` :8190 | `ggerganov/whisper.cpp` (`ggml-large-v3-q5_0.bin`) | MIT | **live** — Vulkan on sliger B580 audio node | `1bit install stt-engine` |
| **image** | `image-engine` | `1bit-sd-server` / `1bit-sd.service` :1234 | `bong-water-water-bong/sdxl-gguf` (`sd_xl_base_1.0.Q8_0.gguf`) | OpenRAIL-M (SDXL) | **live** — native HIP, zero hipBLAS | `1bit install image-engine` |
| **video** | `video-engine` | shares `1bit-sd-server` with image lane | `QuantStack/Wan2.2-TI2V-5B-GGUF` (`Wan2.2-TI2V-5B-Q4_K_M.gguf`) | Apache-2.0 | **staged** — weights pulled, TI2V dispatch wiring in flight | `1bit install video-engine` |
| **NPU (authoring lane)** | `npu-engine` | IRON + MLIR-AIE + Peano + libxrt · no persistent service yet | `amd/IRON` toolchain (no BitNet model yet — port pending) | Apache-2.0 | **toolchain proven** — IRON axpy 160/160 on npu5, ternary kernel TBD. see [NPU unlock 2026-04-23](./NPU-Unlock-20260423.md) | `1bit install npu-engine` |

Install the whole stack in one shot:

```sh
1bit install all
```

which resolves to `core + tts-engine + stt-engine + image-engine + video-engine + npu-engine + power + burnin`. First-boot OOBE on a fresh CachyOS box should land in ~5 minutes plus model-download time, without touching a single `.service` file by hand.

## What "live" means on the status column

- **green / live** — the systemd unit is up, the healthcheck URL listed in `packages.toml` returns 2xx, and we ran real inference against it on the strixhalo box today or yesterday.
- **yellow / staged** — weights are downloaded and sha-verified, the engine builds, but the router / dispatcher isn't yet routing that modality end-to-end. Video-gen sits here: Wan 2.2 TI2V-5B is GGUF'd on disk under the shared sd.cpp engine; the TI2V codepath in `sd-server` still needs its HTTP endpoint wired.
- **blue / toolchain proven** — authoring chain compiles and runs test kernels on the target silicon, but the production kernel for our specific model is not yet written. NPU sits here: axpy 160/160 passed, `bitnet_gemm` MLIR-AIE operator is the ~1–2 wk of remaining work before BitNet-1.58 decodes on the NPU.

## Why these six

Five media/inference lanes plus one authoring lane is the whole local-AI surface we care about: text in, text out (LLM); voice in (STT); voice out (TTS); image out (image-gen); video out (video-gen); and the NPU authoring lane that lets us write new accelerator kernels without waiting on AMD's userspace timeline. No telemetry, no cloud leg, no Python at runtime on any of them (Rule A holds per-lane). Anything not in this table — avatar / music-gen / embeddings — isn't in the complete pack; if we add it later it ships as a new component, not a retrofit to an existing one.

## See also

- [Training-Runs.md](./Training-Runs.md) — Run 5 (LLM Sparse-BitNet 2:4) + Run 6 (ternary Qwen3-TTS) queue.
- [NPU-Unlock-20260423.md](./NPU-Unlock-20260423.md) — today's ship-gate crack, naming reconciliation, axpy matrix.
- [Benchmarks.md](./Benchmarks.md) — measured throughput for the live lanes.
- [Installation.md](./Installation.md) — full fresh-box walkthrough.

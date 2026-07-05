# Video Diffusion Engine — 1bit.systems

**Zero Python. Pure C++23. NPU + CPU.**

C++ video diffusion engine powered by [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (ggml) with 1bit.systems NPU acceleration.

## What It Is

A thin C++ wrapper that uses stable-diffusion.cpp for proven Wan2.2/T5/VAE inference and adds NPU (XDNA 2 INT8) acceleration via the 1bit NPU engine.

## Architecture

```
┌─────────────────────────────────────────────────┐
│               video_main (CLI)                   │
│  --prompt, --frames, --steps, --model, --output  │
└──────────────────────┬──────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐ ┌──────────┐ ┌──────────┐
│ stable-      │ │   T5     │ │  Wan     │
│ diffusion    │ │ Encoder  │ │  VAE     │
│ .cpp (ggml)  │ │ (C++)    │ │  Decoder │
│ DiT forward  │ └──────────┘ └──────────┘
└──────┬───────┘
       │
┌──────▼───────┐
│  1bit NPU    │
│  I8Ctx INT8  │
│  (XDNA 2)    │
└──────────────┘
```

## Quick Start

```bash
# Clone with submodule
git clone --recursive git@github.com:bong-water-water-bong/1bit-systems.git
cd 1bit-systems/engine/video

# Build (CPU only)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Generate video (Wan2.2 T2V)
./video_engine --model Wan-AI/Wan2.2-T2V-A14B-GGUF \
               --prompt "a cat walking, cinematic lighting" \
               --frames 81 --steps 50

# Generate video (Wan2.2 I2V)
./video_engine --model Wan-AI/Wan2.2-I2V-A14B-GGUF \
               --prompt "a cat walking" \
               --input-image ./cat.png \
               --frames 81 --steps 50
```

## Features

| Feature | Status | Source |
|---------|--------|--------|
| **Wan2.2 T2V** | ✅ | stable-diffusion.cpp ggml |
| **Wan2.2 I2V** | ✅ | stable-diffusion.cpp ggml |
| **Wan2.2 TI2V** | ✅ | stable-diffusion.cpp ggml |
| **T5 text encoder** | ✅ C++ | stable-diffusion.cpp |
| **Wan VAE decoder** | ✅ C++ | stable-diffusion.cpp |
| **NPU acceleration** | 🔧 | I8Ctx bridge (in progress) |
| **CPU SIMD** | ✅ | ggml AVX2/AVX512 |
| **CUDA backend** | ✅ | ggml CUDA |
| **Vulkan backend** | ✅ | ggml Vulkan |
| **Metal backend** | ✅ | ggml Metal |
| **Negative prompt** | ✅ | CLI flag |
| **LoRA** | ✅ | stable-diffusion.cpp |
| **ControlNet** | ⏳ | stable-diffusion.cpp SD1.5 |
| **CacheDiT** | ✅ | DBCache + TaylorSeer + EasyCache + Spectrum |
| **Flash attention** | ✅ | ggml built-in |
| **FLUX.2** | ✅ | stable-diffusion.cpp |

## References

- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) — C++ diffusion inference (ggml)
- [Wan2.2](https://github.com/Wan-Video/Wan2.2) — Official Wan video model
- [ggml](https://github.com/ggml-org/ggml) — Tensor library

# Build Guide

## Prerequisites

```bash
# System deps
sudo apt install build-essential cmake git

# For CUDA backend
sudo apt install nvidia-cuda-toolkit

# For Vulkan backend
sudo apt install libvulkan-dev glslc
```

## Build

```bash
cd engine/video

# 1. Init submodule (first time only)
git submodule update --init --recursive

# 2. Build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Enable CUDA
cmake .. -DCMAKE_BUILD_TYPE=Release -DSD_CUBLAS=ON
make -j$(nproc)

# Enable Vulkan
cmake .. -DCMAKE_BUILD_TYPE=Release -DSD_VULKAN=ON
make -j$(nproc)
```

## Download Models

```bash
# Wan2.2 T2V 1.3B (2.6 GB, INT4 quantized)
wget https://huggingface.co/Wan-AI/Wan2.2-T2V-1.3B-GGUF/resolve/main/wan2.2-t2v-1.3b-Q4_0.gguf

# Wan2.2 I2V 14B (8.4 GB, INT4 quantized)
wget https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2-i2v-a14b-Q4_0.gguf
```

## Run

```bash
# Text-to-video (CPU, 16 frames, 50 steps)
./video_engine \
    --model wan2.2-t2v-1.3b-Q4_0.gguf \
    --prompt "a cat walking, cinematic lighting" \
    --frames 16 --steps 50

# Image-to-video (CUDA, 81 frames)
./video_engine \
    --model wan2.2-i2v-a14b-Q4_0.gguf \
    --prompt "a cat walking" \
    --input-image ./cat.png \
    --frames 81 --steps 50 \
    --backend cuda

# Benchmark mode
./video_engine \
    --model wan2.2-t2v-1.3b-Q4_0.gguf \
    --prompt "test" \
    --frames 8 --steps 10 \
    --benchmark --flash-attn
```

## Backend Comparison

| Backend | Hardware | Speed | Memory |
|---------|----------|-------|--------|
| `cpu` | AVX2/AVX512 | 1× baseline | ~4 GB |
| `cuda` | NVIDIA GPU | 5-10× | ~2 GB VRAM |
| `vulkan` | AMD/NVIDIA/Intel GPU | 5-8× | ~2 GB VRAM |
| `metal` | Apple Silicon | 3-5× | ~4 GB unified |
| `npu` | XDNA 2 (future) | target 10-20× | ~1 GB |

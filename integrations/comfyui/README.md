# 1bit-systems ComfyUI Integration

Custom nodes that expose the 1bit inference engine to ComfyUI workflows,
giving you NPU-accelerated LLM inference, image generation, vision-language
understanding, and audio synthesis — all from ComfyUI's visual graph editor.

## Installation

```bash
cd /path/to/ComfyUI/custom_nodes/
git clone https://github.com/bong-water-water-bong/1bit-systems comfyui_1bit_systems
pip install httpx Pillow numpy

# Also install submodule dependencies
cd comfyui_1bit_systems
git submodule update --init --recursive
```

## Prerequisite: Start the 1bit Servers

Before using any node, start the backend servers (in separate terminals or
as systemd services):

```bash
# Start the LLM + VLM inference server
cd /path/to/1bit-systems
cmake -B build -G Ninja && cmake --build build --target unified_server -j$(nproc)
./build/unified_server -w /path/to/models/ -p 8088

# Start the image generation server (requires stable-diffusion.cpp)
# cmake -B build -DUSE_DIFFUSION=ON
# ./build/image_server -p 8089

# Start the audio server
# cmake -B build -DUSE_AUDIO_CPP=ON
# ./build/jarvis_server --port 8090
```

## Available Nodes

### 1BP LLM Generate
Text generation using any 1BP model via the unified_server.
- Input: prompt text, system prompt, model name
- Parameters: max_tokens, temperature
- Output: generated text string

### 1BP VLM Analyze Image
Vision-language understanding. Send an image + question, get a description.
- Input: image from any ComfyUI image node, text prompt
- Parameters: model name (e.g. `zaya1-vl-8b`, `qwen2-vl-2b`)
- Output: text description

### 1BP Image Generate
Text-to-image generation via stable-diffusion.cpp backend.
- Input: text prompt, negative prompt
- Parameters: width, height, steps, CFG scale, seed
- Supports: SD, SDXL, FLUX, Qwen-Image, Z-Image, and more
- Optional: LoRA path + strength
- Output: image tensor (compatible with ComfyUI image pipeline)

### 1BP Text-to-Speech
Synthesize speech from text.
- Input: text, voice name
- Output: audio waveform

### 1BP LoRA Loader
Hot-load a LoRA adapter into the inference engine.
- Input: LoRA GGUF path, target model name
- Output: status string

## Architecture

```
ComfyUI (Python nodes)
    │
    ├── HTTP POST /v1/chat/completions  ──►  unified_server (C++, port 8088)
    ├── HTTP POST /v1/images/generations ──►  image_server   (C++, port 8089)
    └── HTTP POST /v1/audio/speech      ──►  jarvis_server  (C++, port 8090)
```

All heavy computation happens in the C++ backends (HIP CUDA Vulkan NPU CPU).
ComfyUI is just a thin UI layer — no Python inference overhead.

## NPU Acceleration

For NPU users (AMD Strix Halo):
- LLM inference automatically routes to NPU when available
- VLM image encoding uses NPU for image preprocessing
- Image generation uses GPU (diffusion is GPU-heavy)

## LoRA Workflow

```
[1BP LoRA Loader] ─► [1BP LLM Generate]
    │                          │
    │ loads adapter            │ uses adapted weights
    │ into server              │ for generation
    └──────────────────────────┘
```

Load a LoRA at the start of your workflow, then any LLM generation nodes
will use the adapted model. LoRA can be hot-swapped between generations
without restarting the server.

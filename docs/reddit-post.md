# Reddit r/LocalLLaMA Post Draft

## Title:
**74KB C++23 binary. 22 models. No Python. I unlocked AMD's NPU in 4 days.**

## Body:

**tl;dr**: 74KB binary runs 22+ model architectures (LLMs, video gen, image gen,
audio gen) on an AMD Strix Halo NPU + GPU + CPU. Zero Python. MIT.

```
curl -sL https://1bit.systems/npu-install.sh | bash
1bit pull qwen3-0.6b
1bit chat
```

### Why this exists

I bought a Strix Halo laptop because the NPU promised 50 TOPS. What I got was
a vendor-locked runtime that only works with their proprietary model format,
behind a closed-source binary, and INT8 physically works on the silicon but
the software stack won't let you use it without paying for FastFlowLM.

So I reverse-engineered it. Took 4 days.

### What's in the 74KB

| Feature | Detail |
|---------|--------|
| **NPU inference** | 94 tok/s via FLM proxy (production), 97 tok/s C++ engine |
| **GPU inference** | 22 tok/s on Radeon 8060S via Vulkan compute (Zig) |
| **CPU scheduler** | Unified KV cache, RadixAttention, H2O eviction |
| **Video generation** | Wan2.2, CogVideoX, HunyuanVideo, LTX-Video, Sana, Mochi, ... (14 models) |
| **Image generation** | Flux, Flux Schnell, SDXL, SD3.5, Flux.2 |
| **Audio generation** | Stable Audio Open (44.1kHz stereo), AudioLDM2 |
| **LoRA support** | All models, unified loader |
| **Dependencies** | 0. No Python. No pip. No Docker. No MLIR. |

### The model-agnostic bit

I built a single `AgnosticPipeline` that accepts any HuggingFace model ID and
auto-detects the correct pipeline, defaults, and modality. `--model flux` works
the same as `--model stabilityai/stable-audio-open-1.0` or
`--model Wan-AI/Wan2.1-T2V-1.3B-Diffusers`.

```bash
# LLM
1bit chat

# Video
video-lora generate --model wan --prompt "cat walking, cinematic"

# Photography
video-lora generate --model flux --prompt "portrait, soft lighting"

# Audio
video-lora generate --model stable-audio --prompt "rain on window" --audio-end-s 30
```

### The catch

The C++ NPU engine (97 tok/s) has a coherence bug — output is garbled.
Tracking in docs/journey.md. The FLM proxy (94 tok/s) is production-ready.
If you're an XRT/XDNA low-level person, I could use the help.

### Links

GitHub: https://github.com/1bit-systems/1bit  
Benchmarks: docs/wiki/performance.md  
Audit trail: docs/journey.md (1,200+ lines, every crash and fix)

MIT. Open source. Your hardware. Not AMD's.

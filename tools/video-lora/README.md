# Video LoRA

Video generation with LoRA support on Strix Halo. Supports **Wan2.2**, **LTX-Video**, **AnimateDiff**, and **CogVideoX**.

> Vendored into [1bit.systems](../../README.md) at `tools/video-lora/` from
> [bong-water-water-bong/video-lora](https://github.com/bong-water-water-bong/video-lora).
> CI runs from the repo root via `.github/workflows/video-lora-ci.yml`.

## Models & LoRAs

| Model | LoRA Support | Size | Notes |
|-------|-------------|------|-------|
| **Wan2.2-Fun** | Reward LoRAs, Camera Control LoRAs | 1.3B / 14B | Best open-source T2V |
| **LTX-Video** | IC LoRA detailer (in-context) | 13B | Video-to-video control |
| **AnimateDiff** | Motion + Style LoRAs | 1.5B | Largest community LoRA ecosystem |
| **CogVideoX** | Transformer LoRA | 2B / 5B | Good for coherent motion |
| **Stable Video Diff.** | UNet LoRA | 2.5B | Image-to-video |

## Quick Start

### Python (CPU — works everywhere)

```bash
pip install -e ".[dev]"

# AnimateDiff (SD1.5 + motion module) on CPU
python -m video_lora generate --model animatediff --prompt "cat walking, cinematic" --frames 8

# List available models and LoRAs
python -m video_lora list-models
```

### Zig + Vulkan (GPU — Strix Halo Radeon 8060S)

```bash
cd vulkan
zig build run -- --prompt "cinematic dolly zoom through cherry blossoms" --frames 16
zig build run -- --prompt "cat walking" --lora ./motion-lora.safetensors
```

Requires Zig 0.15.2+ and `glslc` (for shader compilation).

## Project Structure

```
tools/video-lora/
├── src/video_lora/
│   ├── __init__.py
│   ├── cli.py              # CLI entry point
│   ├── models/
│   │   ├── __init__.py
│   │   ├── wan.py          # Wan2.2 pipeline + LoRA
│   │   ├── ltx.py          # LTX-Video pipeline + LoRA
│   │   ├── animatediff.py  # AnimateDiff pipeline + LoRA
│   │   └── cogvideo.py     # CogVideoX pipeline + LoRA
│   ├── core/
│   │   ├── __init__.py
│   │   ├── lora_loader.py  # Unified LoRA loading
│   │   ├── pipeline.py     # Base pipeline abstraction
│   │   └── scheduler.py    # Scheduler configs
│   └── utils/
│       ├── __init__.py
│       └── export.py       # Export to GIF/MP4
├── tests/
│   ├── __init__.py
│   └── test_models.py
├── pyproject.toml
└── README.md
```

(CI workflow lives at repo root: `.github/workflows/video-lora-ci.yml`.)

# Video LoRA — Model-Agnostic Video Engine

**14+ video generation models, one unified API.** Auto-detects the correct pipeline
from any HuggingFace model ID or short alias. LoRA support across all backends.

> Vendored into [1bit.systems](../../README.md) at `tools/video-lora/` from
> [bong-water-water-bong/video-lora](https://github.com/bong-water-water-bong/video-lora).
> CI runs from the repo root via `.github/workflows/video-lora-ci.yml`.

## Quick Start

```bash
pip install -e ".[dev]"

# By alias
video-lora generate --model wan --prompt "cat walking, cinematic"
video-lora generate --model hunyuan --prompt "cinematic dolly zoom through cherry blossoms"
video-lora generate --model sana --prompt "sunset over ocean" --frames 81

# By full HuggingFace model ID (auto-detected)
video-lora generate --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers --prompt "cat"

# With LoRA
video-lora generate --model cogvideo --prompt "cat" --lora THUDM/CogVideoX-Fun-Video-LoRA

# With image input (ConsisID identity-preserving I2V)
video-lora generate --model consisid --prompt "smiling" --image face.jpg

# Override any default
video-lora generate --model ltx2 --prompt "cat" --frames 121 --width 1280 --height 720

# List known models
video-lora list-models
```

## Python API

```python
from video_lora import AgnosticPipeline

# By alias
pipe = AgnosticPipeline("wan")
pipe.generate("a cat walking")

# By HF model ID (auto-detected)
pipe = AgnosticPipeline("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
pipe.generate("a cat walking")

# With overrides
pipe.generate("a cat walking", num_frames=81, width=1280, height=720, seed=42)

# With LoRA
pipe.generate("a cat walking", lora_path="alibaba-pai/Wan2.2-Fun-Reward-LoRAs")
```

## Supported Models (auto-detected)

| Model | Aliases | Size | Description |
|-------|---------|------|-------------|
| **Wan2.2** | `wan`, `wan2`, `wan2.2` | 1.3B / 14B | Reward + Camera LoRAs |
| **LTX-Video** | `ltx`, `ltx1` | 13B | IC LoRA detailer |
| **LTX2** | `ltx2` | 2B | Next-gen LTX, IC LoRA, HDR |
| **CogVideoX** | `cogvideo`, `cog`, `cogvideox` | 2B / 5B | Transformer, coherent motion |
| **HunyuanVideo** | `hunyuan`, `hunyuanvideo` | 13B | Tencent flagship |
| **Sana Video** | `sana`, `sanavideo` | 2B | Efficient linear attention, fast |
| **Mochi** | `mochi` | 10B | Genmo, strong prompt adherence |
| **EasyAnimate** | `easyanimate`, `easy` | 3B / 7B / 12B | Alibaba, control + inpaint |
| **Cosmos** | `cosmos` | 7B | NVIDIA world model, physics-first |
| **Allegro** | `allegro` | — | Rhymes AI, custom float32 VAE |
| **Motif Video** | `motif`, `motifvideo` | 2B | Coherent long-form |
| **ConsisID** | `consisid`, `consis` | preview | Identity-consistent I2V (--image) |
| **SkyReels V2** | `skyreels`, `skyreelsv2` | 14B | Diffusion forcing, 540P / 720P |
| **AnimateDiff** | `animatediff`, `animate` | 1.5B | 1000+ community LoRAs |

Any other HuggingFace video model is also auto-detected at runtime.

## Architecture

```
tools/video-lora/
├── src/video_lora/
│   ├── __init__.py
│   ├── cli.py                  # Model-agnostic CLI (no hardcoded choices)
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── registry.py         # 14+ model entries: aliases, defaults, special setup
│   │   ├── resolver.py         # Auto-detect pipeline from any model ID
│   │   └── agnostic.py         # AgnosticPipeline class — detect + generate
│   ├── core/
│   │   ├── __init__.py
│   │   ├── pipeline.py         # Abstract base (keep for backward compat)
│   │   ├── lora_loader.py      # Unified LoRA loading
│   │   └── scheduler.py        # Scheduler configs
│   └── models/                 # Optional per-model overrides (kept for custom tuning)
│       └── ...
├── tests/
│   └── test_models.py
├── pyproject.toml
└── README.md
```

(CI workflow lives at repo root: `.github/workflows/video-lora-ci.yml`.)

## How it works

1. **`--model` accepts anything**: short alias (`wan`), alias variant (`hunyuanvideo`),
   or full HuggingFace ID (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`).

2. **Registry lookup**: checks 14+ known models by alias, HF org prefix, or exact ID.

3. **Auto-detect fallback**: for unknown models, calls `DiffusionPipeline.from_pretrained()`
   which auto-detects the correct pipeline class from the model's `config.json`.

4. **Smart defaults**: per-model resolution, frame count, guidance scale, and step count.

5. **Parameter remapping**: Sana's `frames` → unified `num_frames`, etc.

6. **Special setup**: custom float32 VAEs (Allegro, SkyReels), motion adapters (AnimateDiff),
   identity preservation (ConsisID) — all handled transparently.

## Zig + Vulkan (GPU — Strix Halo Radeon 8060S)

```bash
cd vulkan
zig build run -- --prompt "cinematic dolly zoom through cherry blossoms" --frames 16
zig build run -- --prompt "cat walking" --lora ./motion-lora.safetensors
```

Requires Zig 0.15.2+ and `glslc` (for shader compilation).

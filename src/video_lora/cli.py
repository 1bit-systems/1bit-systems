"""CLI entry point for video-lora."""

import argparse
from pathlib import Path
from typing import Optional


def register_models():
    """Lazy-import model backends so CLI stays fast."""
    models = {}

    try:
        from .models.animatediff import AnimateDiffVideo
        models["animatediff"] = lambda: AnimateDiffVideo()
    except ImportError:
        pass

    try:
        from .models.wan import WanVideo
        models["wan"] = lambda: WanVideo()
    except ImportError:
        pass

    try:
        from .models.ltx import LTXVideo
        models["ltx"] = lambda: LTXVideo()
    except ImportError:
        pass

    return models


def main():
    parser = argparse.ArgumentParser(description="Video LoRA Generator")
    sub = parser.add_subparsers(dest="command", required=True)

    # `generate` subcommand
    gen = sub.add_parser("generate", help="Generate a video")
    gen.add_argument("--model", "-m", required=True,
                     choices=["wan", "ltx", "animatediff", "cogvideo"],
                     help="Model backend")
    gen.add_argument("--prompt", "-p", required=True, help="Text prompt")
    gen.add_argument("--lora", help="LoRA path (HF repo ID or local file)")
    gen.add_argument("--lora-weight", type=float, default=0.7,
                     help="LoRA merge weight")
    gen.add_argument("--frames", type=int, default=16, help="Number of frames")
    gen.add_argument("--width", type=int, default=640)
    gen.add_argument("--height", type=int, default=480)
    gen.add_argument("--seed", type=int, help="Random seed")
    gen.add_argument("--output", "-o", type=Path, help="Output path")

    # `list-models` subcommand
    list_cmd = sub.add_parser("list-models", help="List available models")

    args = parser.parse_args()

    if args.command == "list-models":
        print("Available models and their top LoRAs:")
        print()
        print("  wan         - Wan2.2 (1.3B / 14B)")
        print("                LoRAs: alibaba-pai/Wan2.2-Fun-Reward-LoRAs")
        print("                LoRAs: alibaba-pai/Wan2.2-Fun-A14B-Control-Camera")
        print("  ltx         - LTX-Video (13B)")
        print("                LoRAs: Lightricks/LTX-Video-ICLoRA-detailer-13b-0.9.8")
        print("  animatediff - AnimateDiff (1.5B base)")
        print("                LoRAs: 1000+ community LoRAs on CivitAI")
        print("  cogvideo    - CogVideoX (2B / 5B)")
        return

    models = register_models()
    if args.model not in models:
        print(f"Model '{args.model}' not available. Install extras:")
        print(f"  pip install -e '.[{args.model}]'")
        return

    pipe = models[args.model]()
    output = pipe.generate(
        prompt=args.prompt,
        lora_path=args.lora,
        lora_weight=args.lora_weight,
        num_frames=args.frames,
        width=args.width,
        height=args.height,
        seed=args.seed,
        output=args.output,
    )
    print(f"Output: {output}")


if __name__ == "__main__":
    main()

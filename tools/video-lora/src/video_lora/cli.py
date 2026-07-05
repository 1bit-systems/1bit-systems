"""CLI entry point for video-lora — model-agnostic engine.

Usage::

    # By alias
    video-lora generate --model wan --prompt "cat walking"
    video-lora generate --model hunyuan --prompt "cinematic dolly zoom"

    # By full HuggingFace model ID (auto-detected)
    video-lora generate --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers --prompt "cat walking"

    # With LoRA
    video-lora generate --model cogvideo --prompt "cat" --lora THUDM/CogVideoX-Fun-Video-LoRA

    # With image input (ConsisID, I2V models)
    video-lora generate --model consisid --prompt "smiling" --image face.jpg

    # Override defaults
    video-lora generate --model wan --prompt "cat" --frames 81 --width 1280 --height 720

    # Audio generation
    video-lora generate --model stable-audio --prompt "rain on window" --audio-end-s 30
    video-lora generate --model audioldm2 --prompt "dog barking" --audio-length-s 5

    # List known models
    video-lora list-models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .engine.agnostic import AgnosticPipeline
from .engine.registry import all_known


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Video LoRA Generator — model-agnostic video & audio generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  video-lora generate --model wan --prompt 'cat walking'\n"
            "  video-lora generate --model hunyuan --prompt 'cinematic zoom'\n"
            "  video-lora generate --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers --prompt 'cat'\n"
            "  video-lora generate --model consisid --prompt 'smiling' --image face.jpg\n"
            "  video-lora generate --model cogvideo --prompt 'cat' --lora THUDM/CogVideoX-Fun-Video-LoRA\n"
            "  video-lora generate --model stable-audio --prompt 'rain' --audio-end-s 30\n"
            "  video-lora list-models"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # `generate` subcommand
    gen = sub.add_parser("generate", help="Generate a video")
    gen.add_argument(
        "--model", "-m", required=True,
        help=(
            "Model backend — short alias (wan, hunyuan, cogvideo, sana, ltx, ...) "
            "or full HuggingFace model ID (Wan-AI/Wan2.1-T2V-1.3B-Diffusers)"
        ),
    )
    gen.add_argument("--prompt", "-p", required=True, help="Text prompt")
    gen.add_argument("--lora", help="LoRA path (HF repo ID or local .safetensors file)")
    gen.add_argument("--lora-weight", type=float, default=0.7, help="LoRA merge weight")
    gen.add_argument("--frames", type=int, help="Number of frames (model default if not set)")
    gen.add_argument("--width", type=int, help="Output width (model default if not set)")
    gen.add_argument("--height", type=int, help="Output height (model default if not set)")
    gen.add_argument("--seed", type=int, help="Random seed")
    gen.add_argument("--image", type=Path, help="Input image path (required for I2V models like consisid)")
    gen.add_argument("--output", "-o", type=Path, help="Output file path")
    gen.add_argument("--guidance-scale", type=float, help="CFG guidance scale")
    gen.add_argument("--steps", type=int, help="Number of inference steps")

    # `list-models` subcommand
    sub.add_parser("list-models", help="List known models and their defaults")

    args = parser.parse_args()

    if args.command == "list-models":
        known = all_known()
        audio_models = {k: v for k, v in known.items() if v.modality == "audio"}
        video_models = {k: v for k, v in known.items() if v.modality == "video"}

        print("Video models:")
        print()
        for name, info in sorted(video_models.items()):
            d = info.defaults
            aliases = ", ".join(info.aliases)
            print(f"  {name}")
            print(f"    Aliases:     {aliases}")
            print(f"    Description: {info.description}")
            print(f"    Defaults:    {d['num_frames']} frames, "
                  f"{d['width']}×{d['height']}, "
                  f"guidance {d['guidance_scale']}, "
                  f"{d.get('num_inference_steps', 50)} steps")
            if info.requires_image:
                print(f"    Note:        Requires --image input")
            print(f"    Examples:    {', '.join(info.example_ids)}")
            print()

        print("Audio models:")
        print()
        for name, info in sorted(audio_models.items()):
            d = info.defaults
            aliases = ", ".join(info.aliases)
            print(f"  {name}")
            print(f"    Aliases:     {aliases}")
            print(f"    Description: {info.description}")
            print(f"    Defaults:    {d.get('audio_end_in_s', d.get('audio_length_in_s', '?'))}s duration, "
                  f"guidance {d['guidance_scale']}, "
                  f"{d.get('num_inference_steps', 50)} steps")
            print(f"    Examples:    {', '.join(info.example_ids)}")
            print()
        return

    # Build pipeline kwargs
    pipe_kwargs: dict = {}
    generate_kwargs: dict = {}

    if args.frames is not None:
        generate_kwargs["num_frames"] = args.frames
    if args.width is not None:
        generate_kwargs["width"] = args.width
    if args.height is not None:
        generate_kwargs["height"] = args.height
    if args.seed is not None:
        generate_kwargs["seed"] = args.seed
    if args.image is not None:
        generate_kwargs["image_path"] = args.image
    if args.output is not None:
        generate_kwargs["output"] = args.output
    if args.guidance_scale is not None:
        generate_kwargs["guidance_scale"] = args.guidance_scale
    if args.steps is not None:
        generate_kwargs["num_inference_steps"] = args.steps

    try:
        pipe = AgnosticPipeline(args.model, **pipe_kwargs)
        output = pipe.generate(
            prompt=args.prompt,
            lora_path=args.lora,
            lora_weight=args.lora_weight,
            **generate_kwargs,
        )
        print(f"Output: {output}")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model '{args.model}': {e}", file=sys.stderr)
        print("Try using a full HuggingFace model ID or check the model name.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

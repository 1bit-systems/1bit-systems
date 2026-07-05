"""Agnostic pipeline — auto-detect and generate with any video model.

Usage::

    from video_lora.engine.agnostic import AgnosticPipeline

    # By alias
    pipe = AgnosticPipeline("wan")
    pipe.generate("a cat walking")

    # By HF model ID (auto-detected)
    pipe = AgnosticPipeline("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    pipe.generate("a cat walking")

    # With LoRA
    pipe.generate("a cat walking", lora_path="alibaba-pai/Wan2.2-Fun-Reward-LoRAs")

    # Override any default
    pipe.generate("a cat walking", width=1280, height=720, num_frames=81)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video  # type: ignore[attr-defined]
from PIL import Image

from ..core.lora_loader import load_lora_into_pipe
from .registry import ModelInfo, lookup
from .resolver import resolve_pipeline_class

logger = logging.getLogger(__name__)


class AgnosticPipeline:
    """Model-agnostic video generation pipeline.

    Accepts any model ID — short alias (``wan``), exact alias (``hunyuanvideo``),
    HF repo ID (``Wan-AI/Wan2.1-T2V-1.3B-Diffusers``), or unknown model
    (auto-detected from config).

    Automatically applies per-model sensible defaults and handles special
    setup (custom VAEs, parameter name differences, image inputs).
    """

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        **from_pretrained_kwargs,
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_info: Optional[ModelInfo] = None

        # Resolve metadata
        pipeline_cls = resolve_pipeline_class(model_id)
        self.model_info = lookup(model_id)

        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        # Load pipeline
        if pipeline_cls:
            self.pipe = pipeline_cls.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map="auto",
                **from_pretrained_kwargs,
            )
        else:
            # Unknown model — let diffusers auto-detect
            logger.info("Unknown model %s — auto-detecting pipeline", model_id)
            self.pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map="auto",
                **from_pretrained_kwargs,
            )

        # Apply special setup (custom VAE, motion adapter, etc.)
        if self.model_info and self.model_info.setup_fn:
            self.pipe = self.model_info.setup_fn(self.pipe, self.device, model_id)

    def _get_defaults(self) -> dict[str, Any]:
        """Get sensible defaults for this model."""
        if self.model_info:
            return dict(self.model_info.defaults)
        return {
            "num_frames": 16,
            "width": 640,
            "height": 480,
            "guidance_scale": 6.0,
            "num_inference_steps": 50,
        }

    def _remap_params(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Remap parameter names to match the pipeline's ``__call__`` signature.

        E.g. Sana Video uses ``frames`` instead of ``num_frames``.
        """
        if not self.model_info:
            return kwargs
        aliases = self.model_info.param_aliases
        if not aliases:
            return kwargs
        for our_name, their_name in aliases.items():
            if our_name in kwargs:
                kwargs[their_name] = kwargs.pop(our_name)
        return kwargs

    def generate(
        self,
        prompt: str,
        lora_path: Optional[str] = None,
        lora_weight: float = 0.7,
        num_frames: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
        output: Optional[Path] = None,
        image_path: Optional[Path] = None,
        **extra_kwargs,
    ) -> Path:
        """Generate a video from a text prompt.

        Args:
            prompt: Text prompt.
            lora_path: Optional LoRA path (HF repo ID or ``.safetensors`` file).
            lora_weight: LoRA merge weight.
            num_frames: Number of frames. Uses model default if not set.
            width: Output width. Uses model default if not set.
            height: Output height. Uses model default if not set.
            seed: Random seed for reproducibility.
            output: Output file path. Auto-named if not set.
            image_path: Input image path (required for I2V models like ConsisID).
            **extra_kwargs: Passed through to the pipeline's ``__call__``.

        Returns:
            Path to the generated video file.
        """
        if output is None:
            model_slug = self.model_id.replace("/", "-").replace(".", "-")
            output = Path(f"{model_slug}_output_{abs(hash(prompt))}.mp4")

        # Apply defaults
        defaults = self._get_defaults()
        pipe_kwargs: dict[str, Any] = dict(defaults)

        # Override with explicit params
        if num_frames is not None:
            pipe_kwargs["num_frames"] = num_frames
        if width is not None:
            pipe_kwargs["width"] = width
        if height is not None:
            pipe_kwargs["height"] = height

        # Apply overrides from extra_kwargs
        pipe_kwargs.update(extra_kwargs)

        # Handle image input (ConsisID, I2V models)
        requires_image = self.model_info and self.model_info.requires_image
        if image_path is not None:
            pipe_kwargs["image"] = Image.open(image_path).convert("RGB")
        elif requires_image:
            raise ValueError(
                f"{self.model_info.name} requires an image input. "
                "Pass image_path=<Path> to generate()."
            )

        # Load LoRA if specified
        if lora_path:
            load_lora_into_pipe(self.pipe, lora_path, lora_weight)

        # Seed
        generator: Optional[torch.Generator] = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)

        # Remap parameter names (e.g., num_frames → frames for Sana)
        pipe_kwargs = self._remap_params(pipe_kwargs)

        # Generate
        pipe_kwargs["prompt"] = prompt
        pipe_kwargs["generator"] = generator

        result = self.pipe(**pipe_kwargs)
        video = result.frames[0]

        export_to_video(video, str(output))
        return output

    def load_lora(self, lora_path: str, weight: float = 0.7) -> None:
        """Load a LoRA into the pipeline."""
        load_lora_into_pipe(self.pipe, lora_path, weight)

    def unload_lora(self) -> None:
        """Remove the current LoRA from the pipeline."""
        self.pipe.unload_lora_weights()

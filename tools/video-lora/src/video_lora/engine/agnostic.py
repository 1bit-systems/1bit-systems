"""Agnostic pipeline — auto-detect and generate with any video or audio model.

Usage::

    from video_lora.engine.agnostic import AgnosticPipeline

    # Video by alias
    pipe = AgnosticPipeline("wan")
    pipe.generate("a cat walking")

    # Audio by alias
    pipe = AgnosticPipeline("stable-audio")
    pipe.generate("128 BPM tech house drum loop", audio_end_in_s=30.0)

    # By HF model ID (auto-detected)
    pipe = AgnosticPipeline("stabilityai/stable-audio-open-1.0")
    pipe.generate("rain on window", audio_end_in_s=10.0)
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
from ..utils.export import export_to_wav
from .registry import ModelInfo, lookup
from .resolver import resolve_pipeline_class

logger = logging.getLogger(__name__)


class AgnosticPipeline:
    """Model-agnostic video and audio generation pipeline.

    Accepts any model ID — short alias (``wan``, ``stable-audio``),
    HF repo ID (``Wan-AI/Wan2.1-T2V-1.3B-Diffusers``,
    ``stabilityai/stable-audio-open-1.0``), or unknown model
    (auto-detected from config).

    Automatically applies per-model sensible defaults and handles special
    setup (custom VAEs, parameter name differences, image inputs, audio params).
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

    def _modality(self) -> str:
        """Return the modality (``\"video\"``, ``\"audio\"``, or ``\"image\"``)."""
        if self.model_info:
            return self.model_info.modality
        return "video"

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
        # Audio-specific params
        audio_end_in_s: Optional[float] = None,
        audio_start_in_s: Optional[float] = 0.0,
        audio_length_in_s: Optional[float] = None,
        **extra_kwargs,
    ) -> Path:
        """Generate video, audio, or image from a text prompt.

        Args:
            prompt: Text prompt.
            lora_path: Optional LoRA path (HF repo ID or ``.safetensors`` file).
            lora_weight: LoRA merge weight.
            num_frames: Number of frames (video only).
            width: Output width.
            height: Output height.
            seed: Random seed for reproducibility.
            output: Output file path. Auto-named if not set.
            image_path: Input image path (for I2V / img2img models).
            audio_end_in_s: End time in seconds (audio only).
            audio_start_in_s: Start time in seconds (audio only, default 0).
            audio_length_in_s: Duration in seconds (audio only, AudioLDM2).
            **extra_kwargs: Passed through to the pipeline's ``__call__``.

        Returns:
            Path to the generated file (``.mp4`` for video, ``.wav`` for audio,
            ``.png`` for image).
        """
        modality = self._modality()

        if modality == "audio":
            return self._generate_audio(
                prompt=prompt,
                lora_path=lora_path,
                lora_weight=lora_weight,
                seed=seed,
                output=output,
                audio_end_in_s=audio_end_in_s,
                audio_start_in_s=audio_start_in_s,
                audio_length_in_s=audio_length_in_s,
                **extra_kwargs,
            )

        if modality == "image":
            return self._generate_image(
                prompt=prompt,
                lora_path=lora_path,
                lora_weight=lora_weight,
                width=width,
                height=height,
                seed=seed,
                output=output,
                **extra_kwargs,
            )

        return self._generate_video(
            prompt=prompt,
            lora_path=lora_path,
            lora_weight=lora_weight,
            num_frames=num_frames,
            width=width,
            height=height,
            seed=seed,
            output=output,
            image_path=image_path,
            **extra_kwargs,
        )

    # ------------------------------------------------------------------
    # Video generation
    # ------------------------------------------------------------------

    def _generate_video(
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
        """Generate a video from a text prompt."""
        if output is None:
            model_slug = self.model_id.replace("/", "-").replace(".", "-")
            output = Path(f"{model_slug}_output_{abs(hash(prompt))}.mp4")

        defaults = self._get_defaults()
        pipe_kwargs: dict[str, Any] = dict(defaults)

        if num_frames is not None:
            pipe_kwargs["num_frames"] = num_frames
        if width is not None:
            pipe_kwargs["width"] = width
        if height is not None:
            pipe_kwargs["height"] = height

        pipe_kwargs.update(extra_kwargs)

        requires_image = self.model_info and self.model_info.requires_image
        if image_path is not None:
            pipe_kwargs["image"] = Image.open(image_path).convert("RGB")
        elif requires_image:
            raise ValueError(
                f"{self.model_info.name} requires an image input. "
                "Pass image_path=<Path> to generate()."
            )

        if lora_path:
            load_lora_into_pipe(self.pipe, lora_path, lora_weight)

        generator: Optional[torch.Generator] = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)

        pipe_kwargs = self._remap_params(pipe_kwargs)
        pipe_kwargs["prompt"] = prompt
        pipe_kwargs["generator"] = generator

        result = self.pipe(**pipe_kwargs)
        video = result.frames[0]

        export_to_video(video, str(output))
        return output

    # ------------------------------------------------------------------
    # Audio generation
    # ------------------------------------------------------------------

    def _generate_audio(
        self,
        prompt: str,
        lora_path: Optional[str] = None,
        lora_weight: float = 0.7,
        seed: Optional[int] = None,
        output: Optional[Path] = None,
        audio_end_in_s: Optional[float] = None,
        audio_start_in_s: Optional[float] = 0.0,
        audio_length_in_s: Optional[float] = None,
        **extra_kwargs,
    ) -> Path:
        """Generate audio from a text prompt."""
        if output is None:
            model_slug = self.model_id.replace("/", "-").replace(".", "-")
            output = Path(f"{model_slug}_output_{abs(hash(prompt))}.wav")

        defaults = self._get_defaults()
        pipe_kwargs: dict[str, Any] = dict(defaults)

        # Audio-specific params
        if audio_end_in_s is not None:
            pipe_kwargs["audio_end_in_s"] = audio_end_in_s
        if audio_start_in_s is not None:
            pipe_kwargs["audio_start_in_s"] = audio_start_in_s
        if audio_length_in_s is not None:
            pipe_kwargs["audio_length_in_s"] = audio_length_in_s

        # Strip video-only defaults
        pipe_kwargs.pop("num_frames", None)
        pipe_kwargs.pop("width", None)
        pipe_kwargs.pop("height", None)

        pipe_kwargs.update(extra_kwargs)

        if lora_path:
            load_lora_into_pipe(self.pipe, lora_path, lora_weight)

        generator: Optional[torch.Generator] = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)

        pipe_kwargs["prompt"] = prompt
        pipe_kwargs["generator"] = generator

        result = self.pipe(**pipe_kwargs)
        audio = result.audios

        # result.audios shape: (batch, channels, samples) — take first
        if isinstance(audio, (list, tuple)):
            audio = audio[0]
        elif hasattr(audio, "ndim") and audio.ndim == 3:
            audio = audio[0]

        sample_rate = getattr(self.pipe.vae.config, "sampling_rate", 44100)
        export_to_wav(audio, str(output), sample_rate=sample_rate)
        return output

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------

    def _generate_image(
        self,
        prompt: str,
        lora_path: Optional[str] = None,
        lora_weight: float = 0.7,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
        output: Optional[Path] = None,
        **extra_kwargs,
    ) -> Path:
        """Generate an image from a text prompt."""
        if output is None:
            model_slug = self.model_id.replace("/", "-").replace(".", "-")
            output = Path(f"{model_slug}_output_{abs(hash(prompt))}.png")

        defaults = self._get_defaults()
        pipe_kwargs: dict[str, Any] = dict(defaults)

        if width is not None:
            pipe_kwargs["width"] = width
        if height is not None:
            pipe_kwargs["height"] = height

        # Strip non-image defaults
        pipe_kwargs.pop("num_frames", None)

        pipe_kwargs.update(extra_kwargs)

        if lora_path:
            load_lora_into_pipe(self.pipe, lora_path, lora_weight)

        generator: Optional[torch.Generator] = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)

        pipe_kwargs["prompt"] = prompt
        pipe_kwargs["generator"] = generator

        result = self.pipe(**pipe_kwargs)
        image = result.images[0]

        image.save(str(output))
        return output

    # ------------------------------------------------------------------
    # LoRA helpers
    # ------------------------------------------------------------------

    def load_lora(self, lora_path: str, weight: float = 0.7) -> None:
        """Load a LoRA into the pipeline."""
        load_lora_into_pipe(self.pipe, lora_path, weight)

    def unload_lora(self) -> None:
        """Remove the current LoRA from the pipeline."""
        self.pipe.unload_lora_weights()

"""EasyAnimate pipeline — Alibaba's lightweight video model."""

from pathlib import Path
from typing import Optional

import torch
from diffusers import EasyAnimatePipeline
from diffusers.utils import export_to_video  # type: ignore[attr-defined]

from ..core.pipeline import VideoPipeline


class EasyAnimateVideo(VideoPipeline):
    """EasyAnimate text-to-video — Alibaba's lightweight model (3B / 7B / 12B)."""

    def __init__(
        self,
        model_id: str = "alibaba-pai/EasyAnimateV5.1-7b-zh-diffusers",
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pipe = EasyAnimatePipeline.from_pretrained(  # type: ignore[no-untyped-call]
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.device = device

    def generate(
        self,
        prompt: str,
        lora_path: Optional[str] = None,
        lora_weight: float = 0.7,
        num_frames: int = 49,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
        output: Optional[Path] = None,
    ) -> Path:
        if output is None:
            output = Path(f"easyanimate_output_{abs(hash(prompt))}.mp4")

        if lora_path:
            self.load_lora(lora_path, lora_weight)

        generator = torch.Generator().manual_seed(seed) if seed else None
        video = self.pipe(
            prompt=prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            guidance_scale=5,
            num_inference_steps=50,
            generator=generator,
        ).frames[0]

        export_to_video(video, str(output))
        return output

    def load_lora(self, lora_path: str, weight: float = 0.7) -> None:
        from ..core.lora_loader import load_lora_into_pipe
        load_lora_into_pipe(self.pipe, lora_path, weight)

    def unload_lora(self) -> None:
        self.pipe.unload_lora_weights()

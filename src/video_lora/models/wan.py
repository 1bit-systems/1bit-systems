"""Wan2.2 pipeline — reward LoRAs + camera control."""

from pathlib import Path
from typing import Optional

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

from ..core.pipeline import VideoPipeline


class WanVideo(VideoPipeline):
    """Wan2.2 text-to-video with reward / camera LoRA support."""

    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pipe = DiffusionPipeline.from_pretrained(
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
        num_frames: int = 16,
        width: int = 640,
        height: int = 480,
        seed: Optional[int] = None,
        output: Optional[Path] = None,
    ) -> Path:
        if output is None:
            output = Path(f"wan_output_{abs(hash(prompt))}.mp4")

        if lora_path:
            self.load_lora(lora_path, lora_weight)

        generator = torch.Generator().manual_seed(seed) if seed else None
        video = self.pipe(
            prompt=prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            generator=generator,
        ).frames[0]

        export_to_video(video, str(output))
        return output

    def load_lora(self, lora_path: str, weight: float = 0.7) -> None:
        from ..core.lora_loader import load_lora_into_pipe
        load_lora_into_pipe(self.pipe, lora_path, weight)

    def unload_lora(self) -> None:
        self.pipe.unload_lora_weights()

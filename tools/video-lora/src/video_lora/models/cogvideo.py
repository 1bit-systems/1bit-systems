"""CogVideoX pipeline — transformer-based, best coherent motion."""

from pathlib import Path
from typing import Optional

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video  # type: ignore[attr-defined]

from ..core.pipeline import VideoPipeline


class CogVideoXVideo(VideoPipeline):
    """CogVideoX text-to-video with transformer LoRA support."""

    def __init__(
        self,
        model_id: str = "THUDM/CogVideoX-2B",
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pipe = CogVideoXPipeline.from_pretrained(  # type: ignore[no-untyped-call]
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
        num_frames: int = 48,
        width: int = 720,
        height: int = 480,
        seed: Optional[int] = None,
        output: Optional[Path] = None,
    ) -> Path:
        if output is None:
            output = Path(f"cogvideo_output_{abs(hash(prompt))}.mp4")

        if lora_path:
            self.load_lora(lora_path, lora_weight)

        generator = torch.Generator().manual_seed(seed) if seed else None
        video = self.pipe(
            prompt=prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            guidance_scale=6,
            generator=generator,
        ).frames[0]

        export_to_video(video, str(output))
        return output

    def load_lora(self, lora_path: str, weight: float = 0.7) -> None:
        from ..core.lora_loader import load_lora_into_pipe
        load_lora_into_pipe(self.pipe, lora_path, weight)

    def unload_lora(self) -> None:
        self.pipe.unload_lora_weights()

"""ConsisID pipeline — identity-consistent video generation (image-to-video)."""

from pathlib import Path
from typing import Optional

import torch
from diffusers import ConsisIDPipeline
from diffusers.utils import export_to_video  # type: ignore[attr-defined]
from PIL import Image

from ..core.pipeline import VideoPipeline


class ConsisIDVideo(VideoPipeline):
    """ConsisID image-to-video — identity-consistent video from a face image."""

    def __init__(
        self,
        model_id: str = "BestWishYsh/ConsisID-preview",
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pipe = ConsisIDPipeline.from_pretrained(  # type: ignore[no-untyped-call]
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
        width: int = 720,
        height: int = 480,
        seed: Optional[int] = None,
        output: Optional[Path] = None,
        image_path: Optional[Path] = None,
    ) -> Path:
        """Generate video. ConsisID requires an ``image_path`` (face reference)."""
        if output is None:
            output = Path(f"consisid_output_{abs(hash(prompt))}.mp4")

        if image_path is None:
            raise ValueError(
                "ConsisID requires an image_path argument "
                "(path to a face image for identity preservation). "
                "Usage: generate(prompt=..., image_path=Path('face.jpg'))"
            )

        if lora_path:
            self.load_lora(lora_path, lora_weight)

        generator = torch.Generator().manual_seed(seed) if seed else None
        image = Image.open(image_path).convert("RGB")
        video = self.pipe(
            image=image,
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

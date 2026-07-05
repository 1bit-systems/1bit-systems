"""Abstract base for video generation pipelines."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class VideoPipeline(ABC):
    """Base class for all video generation pipelines."""

    @abstractmethod
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
        """Generate a video from a text prompt, optionally with a LoRA."""
        ...

    @abstractmethod
    def load_lora(self, lora_path: str, weight: float = 0.7) -> None:
        """Load a LoRA into the pipeline."""
        ...

    @abstractmethod
    def unload_lora(self) -> None:
        """Remove the current LoRA from the pipeline."""
        ...

"""Video LoRA — model-agnostic video generation with LoRA support."""

from .engine.agnostic import AgnosticPipeline
from .engine.registry import all_known, lookup

__version__ = "0.3.0"
__all__ = ["AgnosticPipeline", "all_known", "lookup"]

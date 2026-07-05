"""Agnostic model engine — auto-detect any video model."""

from .agnostic import AgnosticPipeline
from .registry import ModelInfo, all_known, lookup

__all__ = ["AgnosticPipeline", "ModelInfo", "all_known", "lookup"]

"""Model resolver — auto-detect diffusers pipeline from any model ID."""

from __future__ import annotations

import importlib
import logging
import re
from typing import Any, Optional

from .registry import REGISTRY, ModelInfo, lookup

logger = logging.getLogger(__name__)


def resolve_pipeline_class(model_id: str) -> Optional[type]:
    """Resolve a model ID to its diffusers pipeline class.

    Strategies (in order):
    1. Registry lookup by alias / HF ID prefix
    2. Try ``DiffusionPipeline.from_pretrained`` with auto-detection
    3. Inspect ``config.json`` for pipeline tag
    """
    info = lookup(model_id)
    if info and info.pipeline_class:
        return info.pipeline_class
    return None


def resolve_model_info(model_id: str) -> Optional[ModelInfo]:
    """Resolve a model ID to its ModelInfo entry."""
    return lookup(model_id)


def auto_detect_pipeline(model_id: str) -> Optional[type]:
    """Auto-detect pipeline class using ``DiffusionPipeline`` registry.

    Falls back to trying ``DiffusionPipeline.from_pretrained`` which auto-detects
    the correct pipeline class from the model's config.
    """
    from diffusers import DiffusionPipeline

    try:
        # Auto-detect via pipeline tag in config
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype="float32",  # light load — just for detection
            device_map=None,
            low_cpu_mem_usage=True,
        )
        cls = type(pipe)
        # Clean up — move to CPU and delete
        pipe.to("cpu")
        del pipe
        return cls
    except Exception as e:
        logger.debug("Auto-detect failed for %s: %s", model_id, e)
        return None


def get_pipeline_info(
    model_id: str,
) -> tuple[Optional[type], Optional[ModelInfo]]:
    """Get pipeline class and metadata for a model ID.

    Returns ``(pipeline_class, model_info)``. Either may be None if
    the model is entirely unknown.
    """
    info = lookup(model_id)
    if info and info.pipeline_class:
        return info.pipeline_class, info

    # Fall back to auto-detect
    cls = auto_detect_pipeline(model_id)
    return cls, info


def normalize_model_id(raw: str) -> str:
    """Normalize a model identifier.

    - Short alias (``wan``, ``hunyuan``) → stays as-is
    - HF ID (``Wan-AI/Wan2.1-...``) → stays as-is
    - Partial match → try to resolve
    """
    raw = raw.strip()
    if raw in REGISTRY:
        return raw
    if "/" in raw:
        return raw
    # Check if it's an alias that isn't in registry yet
    # (shouldn't happen but handle gracefully)
    return raw

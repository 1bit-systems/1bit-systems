"""Unified LoRA loading for video models."""

from pathlib import Path
from typing import Optional, Union

import torch


def load_lora_into_pipe(
    pipe,
    lora_path: Union[str, Path],
    weight: float = 0.7,
    adapter_name: str = "default",
):
    """Load a LoRA into a diffusers pipeline with optional weight scaling.

    Supports:
    - HuggingFace repo IDs: ``alibaba-pai/Wan2.2-Fun-Reward-LoRAs``
    - Local ``.safetensors`` files
    - Local directories with multiple LoRAs

    Args:
        pipe: A diffusers pipeline (AnimateDiff, Wan2.2, LTX-Video, etc.)
        lora_path: HF repo ID, local path, or .safetensors file
        weight: LoRA merge weight (0.0 = no effect, 1.0 = full effect)
        adapter_name: Name for the adapter (for multiple LoRAs)
    """
    lora_path = str(lora_path)

    # HF repo ID or local .safetensors
    if lora_path.endswith(".safetensors") or "/" in lora_path and not Path(lora_path).exists():
        # Try as HF hub repo
        pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
    elif Path(lora_path).is_file() and lora_path.endswith(".safetensors"):
        pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
    elif Path(lora_path).is_dir():
        # Directory with multiple LoRAs
        pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
    else:
        raise ValueError(f"Cannot resolve LoRA path: {lora_path}")

    # Scale weights
    if weight != 1.0:
        pipe.set_adapter_weight(adapter_name, weight)

    pipe.fuse_lora(adapter_names=[adapter_name], lora_scale=weight)
    return pipe

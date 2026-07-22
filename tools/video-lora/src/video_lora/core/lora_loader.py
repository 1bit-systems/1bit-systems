"""Unified LoRA loading for video models."""

from pathlib import Path
from typing import Any, Union


def load_lora_into_pipe(
    pipe: Any,
    lora_path: Union[str, Path],
    weight: float = 0.7,
    adapter_name: str = "default",
) -> Any:
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
    lora_path_obj = Path(lora_path)

    # Check local paths FIRST, then fall back to HF hub repo IDs
    if lora_path_obj.exists():
        if lora_path_obj.is_file() and lora_path.endswith(".safetensors"):
            pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
        elif lora_path_obj.is_dir():
            # Directory with multiple LoRAs
            pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
        else:
            raise ValueError(
                f"Local path exists but is not a .safetensors file or directory: {lora_path}"
            )
    elif "/" in lora_path:
        # Not a local path — try as HF hub repo ID
        pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
    else:
        raise ValueError(f"Cannot resolve LoRA path: {lora_path}")

    # Scale weights
    if weight != 1.0:
        pipe.set_adapter_weight(adapter_name, weight)

    pipe.fuse_lora(adapter_names=[adapter_name], lora_scale=weight)
    return pipe

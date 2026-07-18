# SPDX-License-Identifier: Apache-2.0
"""Cache runtime helpers for 1bit-mlx.

Adapted from Rapid-MLX's runtime.cache module. Provides scheduler
resolution for the cache protocol layer.
"""

from typing import Any


def _resolve_scheduler(engine: Any) -> Any:
    """Resolve the scheduler from an engine instance.

    Original Rapid-MLX had complex scheduler resolution through
    async engines. This stub provides a simple hook that subclasses
    can override.
    """
    scheduler = getattr(engine, "scheduler", None)
    if scheduler is not None:
        return scheduler
    # Fallback: check for a _scheduler attribute
    scheduler = getattr(engine, "_scheduler", None)
    if scheduler is not None:
        return scheduler
    return None

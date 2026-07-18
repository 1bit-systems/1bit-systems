# SPDX-License-Identifier: Apache-2.0
"""JSON nesting-depth helpers (D-TOOL-RECUR / D-DEEP-JSON).

Adapted for 1bit.systems from Rapid-MLX.
"""

from __future__ import annotations

import os
from typing import Any

MAX_TOOL_SCHEMA_DEPTH_ENV = "MAX_TOOL_SCHEMA_DEPTH"
MAX_BODY_DEPTH_ENV = "MAX_BODY_DEPTH"
DEFAULT_MAX_TOOL_SCHEMA_DEPTH = 64
DEFAULT_MAX_BODY_DEPTH = 64


def _resolve_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def resolve_max_body_depth() -> int:
    """Return the effective max JSON body nesting depth."""
    return _resolve_env_int(MAX_BODY_DEPTH_ENV, DEFAULT_MAX_BODY_DEPTH)


def resolve_max_tool_schema_depth() -> int:
    """Return the effective max tool schema nesting depth."""
    return _resolve_env_int(MAX_TOOL_SCHEMA_DEPTH_ENV, DEFAULT_MAX_TOOL_SCHEMA_DEPTH)


def json_nesting_depth_exceeds(value: Any, limit: int) -> bool:
    """Check if JSON value nesting depth exceeds limit.

    Walks iteratively with an explicit work stack so the depth
    measurement itself cannot crash the worker via recursion.
    """
    if limit < 1:
        return False
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > limit:
            return True
        if isinstance(node, dict):
            for v in node.values():
                if isinstance(v, (dict, list)):
                    stack.append((v, depth + 1))
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, (dict, list)):
                    stack.append((item, depth + 1))
    return False

# SPDX-License-Identifier: Apache-2.0
"""
Server configuration — single source of truth for server-wide state.

Adapted for 1bit.systems from Rapid-MLX. All server state lives here
in a single ServerConfig instance, accessible via ``get_config()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ServerConfig:
    """All server-wide mutable state in one place."""

    # --- Engine ---
    engine: Any = None  # BaseEngine-compatible instance
    model_name: str | None = None
    model_alias: str | None = None
    model_path: str | None = None
    ready: bool = False
    draining: bool = False
    bind_host: str | None = None
    bind_port: int | None = None
    bind_listen_fd: int | None = None
    api_key: str | None = None
    sse_keepalive_seconds: float = 20.0
    max_request_bytes: int = 8 * 1024 * 1024
    body_receive_timeout_seconds: float = 15.0
    default_timeout: float = 1800.0
    rate_limit: int = 0  # 0 = disabled


_config: ServerConfig | None = None


def get_config() -> ServerConfig:
    """Return the global ServerConfig singleton."""
    global _config
    if _config is None:
        _config = ServerConfig()
    return _config


def reset_config() -> None:
    """Reset the global config (for testing)."""
    global _config
    _config = None

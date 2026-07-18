# SPDX-License-Identifier: Apache-2.0
"""
Hardware-compatibility shims.

Adapted for 1bit.systems — MLX-specific shims are replaced with
no-ops. The install() function is a no-op for non-MLX backends.
"""


def install() -> None:
    """Install compatibility shims (no-op for 1bit backend).

    The original Rapid-MLX shim handled MLX M5 single-stream GPU
    compatibility. For 1bit.systems, this is a no-op.
    """
    pass

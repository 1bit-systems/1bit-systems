#!/bin/bash
# 1bit environment setup — source this before running the engine.
# Usage: source env.sh
#    or: source env.sh /path/to/1bit  # override install dir
#
# Pure C++ stack — no Rust, no Python, no Node.js required.
# Built binaries: zaya_server, onebitd, onebit, unified_router, bitnet_tui
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  set -euo pipefail
fi

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LINK_DIR="${1:-$DIR}"

# HSA_OVERRIDE_GFX_VERSION: only needed for ROCm <7.x where the kernel driver
# doesn't report the correct GPU target for Strix Halo (gfx1151).
if command -v hipconfig &>/dev/null; then
    ROCM_VER=$(hipconfig --version 2>/dev/null | cut -d. -f1)
    if [ -n "$ROCM_VER" ] && [ "$ROCM_VER" -lt 7 ] 2>/dev/null; then
        export HSA_OVERRIDE_GFX_VERSION=11.5.1
    fi
elif [ -d /opt/rocm-6.2 ] || [ -d /opt/rocm-6.1 ] || [ -d /opt/rocm-6.0 ]; then
    export HSA_OVERRIDE_GFX_VERSION=11.5.1
fi
export HSA_ENABLE_SDMA=0
export LD_LIBRARY_PATH="$LINK_DIR/build:${LD_LIBRARY_PATH:-}"
export PATH="$LINK_DIR/build:$PATH"

echo "[1bit] Environment ready (pure C++ stack):"
echo "  HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-not set}"
echo "  HSA_ENABLE_SDMA=$HSA_ENABLE_SDMA"
echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "  Binaries available:"
echo "    zaya_server     — HIP inference engine + HTTP server"
echo "    onebitd         — daemon (spawns backend, proxies HTTP)"
echo "    onebit / 1bit   — CLI agent (chat, up, down, status, build, config)"
echo "    unified_router  — NPU+GPU routing proxy"
echo "    bitnet_tui      — FTXUI terminal chat UI"
echo ""
echo "  Quick start:"
echo "    onebit chat     — interactive agent session"
echo "    onebit up       — start NPU stack"
echo "    onebit status   — check stack health"

#!/bin/bash
# 1bit environment setup — source this before running the engine.
# Usage: source env.sh
#    or: source env.sh /path/to/1bit  # override install dir
#
# When executed directly (not sourced), fail on first error.
# When sourced, let the caller's shell handle error handling.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  set -euo pipefail
fi

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LINK_DIR="${1:-$DIR}"

# HSA_OVERRIDE_GFX_VERSION: only needed for ROCm <7.x where the kernel driver
# doesn't report the correct GPU target for Strix Halo (gfx1151).
# For ROCm 7.2+, the driver reports gfx1151 correctly and overriding
# to gfx1100 causes wrong code paths (see issue #251).
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

echo "[1bit] Environment ready:"
echo "  HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-not set}"
echo "  HSA_ENABLE_SDMA=$HSA_ENABLE_SDMA"
echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "  Run: zaya_server"
echo "  Or: zaya_gpu_decode model.q4nx --tokens 64"

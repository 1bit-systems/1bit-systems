#!/bin/bash
set -euo pipefail
#!/usr/bin/env bash
# 1bit environment setup — source this before running the engine.
# Usage: source env.sh
#    or: source env.sh /path/to/1bit  # override install dir
#
# NOTE: this file is meant to be `source`d, so `set -e` is applied only when
# it is executed directly (not sourced) — otherwise an error here would kill
# the caller's interactive shell. See issue #171.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  set -euo pipefail
fi

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LINK_DIR="${1:-$DIR}"

export HSA_OVERRIDE_GFX_VERSION=11.5.1
export HSA_ENABLE_SDMA=0
export LD_LIBRARY_PATH="$LINK_DIR/build:${LD_LIBRARY_PATH:-}"
export PATH="$LINK_DIR/build:$PATH"

echo "[1bit] Environment ready:"
echo "  HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
echo "  HSA_ENABLE_SDMA=$HSA_ENABLE_SDMA"
echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "  Run: zaya_server"
echo "  Or: zaya_gpu_decode model.q4nx --tokens 64"

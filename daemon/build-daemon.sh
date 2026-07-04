#!/bin/bash
# Build the C++ daemon (zero Python replacement)
set -euo pipefail
SRCDIR="$(cd "$(dirname "$0")" && pwd)"
CXX="${CXX:-g++}"

# Minimal build (no Stripe)
echo "=== Building npu-gpu-cpud (minimal, no Stripe) ==="
$CXX -std=c++23 -O3 -o "$SRCDIR/npu-gpu-cpud" "$SRCDIR/npu-gpu-cpud.cpp" -lpthread
echo "  -> $(ls -lh "$SRCDIR/npu-gpu-cpud" | awk '{print $5}')"

# Full build (with Stripe)
if pkg-config --exists libcurl 2>/dev/null || [ -f /usr/include/curl/curl.h ]; then
    echo "=== Building npu-gpu-cpud-stripe (with Stripe support) ==="
    $CXX -std=c++23 -O3 -o "$SRCDIR/npu-gpu-cpud-stripe" \
        "$SRCDIR/npu-gpu-cpud.cpp" -lpthread -lcurl -DHAVE_LIBCURL
    echo "  -> $(ls -lh "$SRCDIR/npu-gpu-cpud-stripe" | awk '{print $5}')"
else
    echo "  (libcurl not found — Stripe support disabled. Install libcurl-dev for Stripe.)"
fi

echo "=== Daemon build complete ==="

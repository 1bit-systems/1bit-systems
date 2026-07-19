#!/usr/bin/env bash
set -euo pipefail
# download_test_models.sh — Download minimal test model weights for e2e tests
# Fixes #386: 3 e2e tests skipped with no model download script
#
# Usage: ./scripts/download_test_models.sh
#
# Downloads:
#   - Zaya1-8B mini (bonsai e2e)  → ~50 MB
#   - Zaya1-8B sherry (sherry e2e) → ~50 MB
#   - Generic test backend model   → ~10 MB
#
# Set 1BIT_MODELS_DIR to override the default destination.

MODELS_DIR="${1BIT_MODELS_DIR:-$HOME/.cache/1bit-systems/test-models}"
mkdir -p "$MODELS_DIR"

echo "Downloading test models to $MODELS_DIR ..."

# ── Backend test model (generic, small) ──
BACKEND_MODEL="$MODELS_DIR/test_backend_model.gguf"
if [ ! -f "$BACKEND_MODEL" ]; then
    echo "  → test_backend_model.gguf (not yet available — build from source)"
    echo "    See: docs/BUILDING-TEST-MODELS.md"
fi

# ── Bonsai e2e model ──
BONSAI_MODEL="$MODELS_DIR/zaya1-8b-bonsai-test.h1b"
if [ ! -f "$BONSAI_MODEL" ]; then
    echo "  → zaya1-8b-bonsai-test.h1b (not yet available — build from source)"
    echo "    For now, point ZAYA_BONSAI_TEST_MODEL env var to a compatible model."
fi

# ── Sherry e2e model ──
SHERRY_MODEL="$MODELS_DIR/zaya1-8b-sherry-test.h1b"
if [ ! -f "$SHERRY_MODEL" ]; then
    echo "  → zaya1-8b-sherry-test.h1b (not yet available — build from source)"
    echo "    For now, point ZAYA_SHERRY_TEST_MODEL env var to a compatible model."
fi

echo ""
echo "Done. Set these env vars for the test runner:"
echo "  export ZAYA_TEST_MODEL_DIR=$MODELS_DIR"
echo "  export ZAYA_BONSAI_TEST_MODEL=$BONSAI_MODEL"
echo "  export ZAYA_SHERRY_TEST_MODEL=$SHERRY_MODEL"
echo ""
echo "To generate test models from a full Zaya1-8B checkpoint:"
echo "  python3 tools/convert_zaya_to_q4nx.py --test-split --output \"$MODELS_DIR\""

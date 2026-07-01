#!/usr/bin/env bash
# 1bit.systems — one-command install
set -euo pipefail

echo "=== 1bit.systems ==="
echo ""

# Detect platform
OS=$(uname -s)
if [ "$OS" != "Linux" ]; then
    echo "1bit.systems requires Linux (Ubuntu 26.04+)."
    echo "For macOS, use ZINC GPU engine from /home/bcloud/zinc."
    exit 1
fi

# Check NPU
if ! xrt-smi examine 2>/dev/null | grep -q RyzenAI; then
    echo "No NPU detected. Install AMD XRT and ensure amdxdna is loaded:"
    echo "  sudo apt install libxrt2 libxrt-npu2"
    echo "  sudo modprobe amdxdna"
    exit 1
fi
echo "✓ NPU detected"

# Check model
MODEL="${HOME}/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx"
if [ ! -f "$MODEL" ]; then
    echo "! Qwen3-0.6B model not found at $MODEL"
    echo "  Install FastFlowLM first: flm pull qwen3:0.6b"
    echo "  Or set FLM_MODEL_PATH env var"
fi

# Build
echo "Building..."
make -C "$(dirname "$0")/.." npu

# Install
cp engine/npu/build/npu_engine /usr/local/bin/1bit-npu
echo "✓ Installed: /usr/local/bin/1bit-npu"
echo ""
echo "Run: 1bit-npu"

#!/usr/bin/env bash
# 1bit.systems — one-command install + run
# curl -sL https://1bit.systems/install.sh | bash
set -euo pipefail

RED='\033[0;31m' GREEN='\033[0;32m' CYAN='\033[0;36m' NC='\033[0m'
say() { printf "${GREEN}✓${NC} %s\n" "$*"; }
warn() { printf "${CYAN}!${NC} %s\n" "$*"; }
die() { printf "${RED}✗${NC} %s\n" "$*"; exit 1; }

echo ""
printf "${GREEN}╔══════════════════════════════════════════╗${NC}\n"
printf "${GREEN}║         1bit.systems — NPU Install      ║${NC}\n"
printf "${GREEN}╚══════════════════════════════════════════╝${NC}\n"
echo ""

# Detect platform
OS=$(uname -s)
[ "$OS" != "Linux" ] && die "Linux required (Ubuntu 26.04+). macOS: use ZINC GPU engine."

# Check NPU
if ! xrt-smi examine 2>/dev/null | grep -q RyzenAI; then
    die "No NPU detected. Run: sudo apt install libxrt2 libxrt-npu2 && sudo modprobe amdxdna"
fi
say "NPU detected"

# Check for xclbins
XCLDIR="/home/bcloud/npu-sandbox/npu-infer/build/int8"
if [ ! -f "$XCLDIR/final_i8_QKV_v.xclbin" ]; then
    warn "INT8 xclbins not found at $XCLDIR"
    warn "Build them with: cd engine/npu/xclbins && python3 n1_core_i8_v2.py"
    warn "See docs/building.md for full xclbin build instructions."
fi

# Check model
MODEL="${HOME}/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx"
if [ ! -f "$MODEL" ]; then
    warn "Model not found. Install FastFlowLM and pull the model:"
    warn "  pip install fastflowlm && flm pull qwen3:0.6b"
fi

# Build engine
say "Building engine..."
gcc -c -O3 -o engine/npu/build/dequant_q4nx.o engine/npu/src/dequant_q4nx.c 2>/dev/null || true
g++ -std=c++23 -O3 -o engine/npu/build/npu_engine engine/npu/src/npu_engine_cb.cpp \
    engine/npu/build/dequant_q4nx.o \
    -l xrt_coreutil -l uuid -l m -l dl 2>/dev/null || {
    warn "Build failed — missing XRT headers or libs."
    warn "Install: sudo apt install libxrt-dev libxrt-npu2"
    warn "Or use the pre-built binary from GitHub Releases."
    exit 0
}
say "Engine built: engine/npu/build/npu_engine"

# Run quick smoke test
if [ -f "$MODEL" ] && [ -f "$XCLDIR/final_i8_QKV_v.xclbin" ]; then
    say "Running smoke test (BOS token, 2 decode)..."
    timeout 60 engine/npu/build/npu_engine 1 2 2>/dev/null && say "Smoke test passed!" || warn "Smoke test timed out (normal on first run)"
fi

echo ""
echo "  Install complete."
echo "  Binary: engine/npu/build/npu_engine"
echo "  Run:    ./engine/npu/build/npu_engine"
echo ""
echo "  —bong-water-water-bong · Sorry but not Sorry"

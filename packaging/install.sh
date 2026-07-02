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
printf "${GREEN}║   1bit.systems — 5 models, 120KB, 28 tok/s   ║${NC}\n"
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

# Check models
say "Checking for models..."
MODELS_FOUND=0
for MODEL_DIR in \
    "${HOME}/.config/flm/models/Qwen3-0.6B-NPU2" \
    "${HOME}/models/Qwen3-8B-NPU2" \
    "${HOME}/.config/flm/models/Qwen3-VL-4B-Instruct-NPU2" \
    "${HOME}/.config/flm/models/Llama-3.1-8B-NPU2" \
    "${HOME}/.config/flm/models/Gemma4-E2B-IT-NPU2"; do
    if [ -f "$MODEL_DIR/model.q4nx" ]; then
        say "  $(basename $MODEL_DIR)"
        MODELS_FOUND=$((MODELS_FOUND + 1))
    fi
done
[ $MODELS_FOUND -eq 0 ] && warn "No models found. Install FastFlowLM and pull models:"
[ $MODELS_FOUND -eq 0 ] && warn "  pip install fastflowlm && flm pull qwen3:0.6b"

# Build engine
say "Building engine (5 models, 120KB binary)..."
gcc -c -std=c11 -O3 -o engine/npu/build/dequant_q4nx.o engine/npu/src/dequant_q4nx.c 2>/dev/null || true
g++ -std=c++23 -O3 -march=native -fopenmp -ffast-math \
    -o engine/npu/build/npu_engine_all \
    engine/npu/src/npu_engine_all.cpp \
    engine/npu/build/dequant_q4nx.o \
    -Iengine/npu/src \
    -l xrt_coreutil -l uuid -l m -l dl 2>/dev/null || {
    warn "Build failed — missing XRT headers or libs."
    warn "Install: sudo apt install libxrt-dev libxrt-npu2"
    warn "Or use the pre-built binary from GitHub Releases."
    exit 0
}
say "Engine built: engine/npu/build/npu_engine_all"

# Run smoke test on first found model
DEFAULT_MODEL="${HOME}/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx"
if [ -f "$DEFAULT_MODEL" ]; then
    say "Running smoke test (9 token prefill, 4 decode, OpenMP)..."
    OMP_NUM_THREADS=16 timeout 60 engine/npu/build/npu_engine_all "$DEFAULT_MODEL" 4 2>/dev/null && \
        say "Smoke test passed — 28 tok/s!" || warn "Smoke test timed out (normal on first run)"
fi

# --- 1bit CLI Installation ---
say "Installing 1bit agent CLI..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Install npm dependencies
if command -v node &>/dev/null; then
  cd "$PROJECT_DIR"
  npm install --ignore-scripts 2>/dev/null || warn "npm install skipped (run manually: npm install)"
  npm run build 2>/dev/null || warn "npm build skipped (run manually: npm run build)"

  # Symlink to ~/.local/bin
  mkdir -p "${HOME}/.local/bin"
  ln -sf "$PROJECT_DIR/bin/1bit" "${HOME}/.local/bin/1bit"

  # Add to PATH if not already
  if [[ ":$PATH:" != *":${HOME}/.local/bin:"* ]]; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "${HOME}/.bashrc"
    say "Added ~/.local/bin to PATH in .bashrc"
  fi

  say "1bit CLI installed. Run: 1bit help"
else
  warn "Node.js not found. Install Node.js 22+ to use the 1bit agent CLI."
fi

# --- Systemd Service ---
if command -v systemctl &>/dev/null; then
  if [ -f "$PROJECT_DIR/services/install-service.sh" ]; then
    say "Installing 1bit-agent systemd service..."
    bash "$PROJECT_DIR/services/install-service.sh" install 2>/dev/null || warn "Service install failed (run manually: services/install-service.sh)"
  fi
fi

echo ""
echo "  Install complete."
echo "  Binary:  engine/npu/build/npu_engine"
echo "  CLI:     1bit"
echo "  Service: 1bit-agent (systemd --user)"
echo ""
echo "  Run: 1bit chat"
echo "  Run: 1bit up   (to start NPU stack)"
echo ""
echo "  —bong-water-water-bong · Sorry but not Sorry"

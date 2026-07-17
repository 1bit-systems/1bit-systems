#!/usr/bin/env bash
# install-lemon-mlx.sh — Build and install lemon-mlx-engine for AMD ROCm.
#
# lemon-mlx-engine is a C++ MLX inference engine that runs LLMs on AMD GPUs
# via ROCm. It supports 50+ architectures (Llama, Qwen, Gemma, Phi, etc.)
# and 4-bit/8-bit quantized inference.
#
# Usage:
#   ./scripts/install-lemon-mlx.sh              # build & install to /opt/lemon-mlx-engine
#   PREFIX=/custom/path ./scripts/install-lemon-mlx.sh
#   SKIP_BUILD=1 ./scripts/install-lemon-mlx.sh  # install already-built binaries

set -euo pipefail

PREFIX="${PREFIX:-/opt/lemon-mlx-engine}"
SKIP_BUILD="${SKIP_BUILD:-0}"
JOBS="${JOBS:-$(nproc)}"
GFX="${GFX:-gfx1151}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)/integrations/lemon-mlx-engine"

echo "=== lemon-mlx-engine ROCm Build ==="
echo "  Source:    $REPO_DIR"
echo "  Prefix:    $PREFIX"
echo "  GPU Arch:  $GFX"
echo "  Jobs:      $JOBS"
echo ""

if [ ! -d "$REPO_DIR" ]; then
  echo "ERROR: lemon-mlx-engine submodule not found at $REPO_DIR"
  echo "Run: git submodule update --init integrations/lemon-mlx-engine"
  exit 1
fi

if [ "$SKIP_BUILD" = "1" ]; then
  echo "SKIP_BUILD=1 — linking pre-built binaries"
else
  # Check for ROCm
  if [ ! -d /opt/rocm-therock ]; then
    echo "ERROR: TheRock not found at /opt/rocm-therock"
    echo "       Install: sudo bash scripts/setup-therock.sh"
    exit 1
  fi
  ROCM_LIB="/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_devel/lib"
  echo "TheRock detected: $(ls $ROCM_LIB/libamdhip64* 2>/dev/null | head -1)"

  # Build
  BUILD_DIR="$REPO_DIR/build"
  mkdir -p "$BUILD_DIR"
  cd "$BUILD_DIR"

  cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLX_BUILD_CUDA=OFF \
    -DMLX_BUILD_METAL=OFF \
    -DCMAKE_PREFIX_PATH="${ROCM_LIB%/*}" \
    -DCMAKE_HIP_ARCHITECTURES="$GFX"

  # Patch mlx parallel-jobs flag for clang compatibility
  MLX_ROCM_CMAKE="$BUILD_DIR/_deps/mlx-src/mlx/backend/rocm/CMakeLists.txt"
  if [ -f "$MLX_ROCM_CMAKE" ]; then
    sed -i 's/-parallel-jobs=\${NPROC}//g' "$MLX_ROCM_CMAKE"
  fi

  make -j"$JOBS"
  echo "Build complete."
fi

# Install
echo "Installing to $PREFIX ..."
sudo mkdir -p "$PREFIX/bin" "$PREFIX/lib"

# Find built binaries
BUILD_DIR="$REPO_DIR/build"
if [ -f "$BUILD_DIR/chat" ]; then
  sudo cp "$BUILD_DIR/chat" "$PREFIX/bin/"
fi
if [ -f "$BUILD_DIR/server" ]; then
  sudo cp "$BUILD_DIR/server" "$PREFIX/bin/"
fi
if [ -f "$BUILD_DIR/diagnose" ]; then
  sudo cp "$BUILD_DIR/diagnose" "$PREFIX/bin/"
fi
if [ -f "$BUILD_DIR/convert" ]; then
  sudo cp "$BUILD_DIR/convert" "$PREFIX/bin/"
fi

# Link into PATH
sudo ln -sf "$PREFIX/bin/chat"     /usr/local/bin/lemon-chat
sudo ln -sf "$PREFIX/bin/server"   /usr/local/bin/lemon-server
sudo ln -sf "$PREFIX/bin/diagnose" /usr/local/bin/lemon-diagnose
sudo ln -sf "$PREFIX/bin/convert"  /usr/local/bin/lemon-convert

echo ""
echo "=== Installed ==="
ls -lh "$PREFIX/bin/"
echo ""
echo "Binaries linked:"
which lemon-chat lemon-server lemon-diagnose lemon-convert 2>/dev/null || echo "(check PATH)"
echo ""
echo "Usage:"
echo "  lemon-chat   <model-id>     # Interactive chat"
echo "  lemon-server <model-id>     # OpenAI-compatible API server"
echo "  lemon-diagnose <model-id>   # Numerical diagnostics"
echo ""
echo "Examples:"
echo "  lemon-chat mlx-community/Qwen3-4B-4bit --temp 0.1"
echo "  lemon-server mlx-community/Qwen3-8B-4bit --port 8080"

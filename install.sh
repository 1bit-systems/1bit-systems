#!/usr/bin/env bash
set -euo pipefail
GREEN='\033[0;32m'; NC='\033[0m'; YELLOW='\033[1;33m'
log() { echo -e "${GREEN}[1bit]${NC} $*"; }
warn() { echo -e "${YELLOW}[1bit]${NC} $*"; }

REPO_URL="https://github.com/bong-water-water-bong/1bit-systems.git"
INSTALL_DIR="${INSTALL_DIR:-$HOME/1bit}"
SKIP_ROCM=false; [ "${1:-}" = "--skip-rocm" ] && SKIP_ROCM=true
MODELS_DIR="${MODELS_DIR:-$HOME/models}"

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    echo "Usage: curl -fsSL https://raw.githubusercontent.com/bong-water-water-bong/1bit-systems/main/install.sh | bash"
    echo "       bash install.sh [--skip-rocm]"
    echo ""
    echo "  --skip-rocm  Skip kernel build (use pre-build librocm_cpp.so)"
    echo ""
    echo "Installs 1bit inference engine for AMD Strix Halo (gfx1151)."
    echo "Builds the pure C++ zaya_server (207 KB) — no Rust, no Python."
    exit 0
fi

# ── Detect if running standalone (curl-piped) vs from repo root ────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" 2>/dev/null && pwd || echo ".")"
if [ -f "$SCRIPT_DIR/CMakeLists.txt" ]; then
    DIR="$SCRIPT_DIR"
    log "Running from repo root: $DIR"
else
    log "Cloning 1bit repository..."
    if [ -d "$INSTALL_DIR/.git" ]; then
        log "Repo already exists at $INSTALL_DIR — pulling latest"
        git -C "$INSTALL_DIR" pull --ff-only || warn "pull failed; continuing with existing copy"
    else
        git clone --depth 1 "$REPO_URL" "$INSTALL_DIR"
    fi
    DIR="$INSTALL_DIR"
fi

# ── Install deps ──────────────────────────────────────────────────────────────
install_deps() {
    if command -v apt-get &>/dev/null; then
        log "Installing build deps (apt)..."
        sudo apt-get update -qq && sudo apt-get install -y -qq build-essential cmake ninja-build git curl rocm-hip-sdk || true
        [ ! -f /opt/rocm/bin/hipcc ] && [ -f /usr/bin/hipcc ] && sudo ln -sf /usr/bin/hipcc /opt/rocm/bin/hipcc 2>/dev/null || true
        [ ! -f /opt/rocm/bin/hipconfig ] && [ -f /usr/bin/hipconfig ] && sudo ln -sf /usr/bin/hipconfig /opt/rocm/bin/hipconfig 2>/dev/null || true
    elif command -v pacman &>/dev/null; then
        log "Installing build deps (pacman)..."
        sudo pacman -Sy --noconfirm base-devel cmake ninja git curl rocm-hip-sdk
    elif command -v dnf &>/dev/null; then
        log "Installing build deps (dnf)..."
        sudo dnf install -y gcc-c++ cmake ninja-build git curl rocm-hip-devel
    else
        warn "Unknown package manager. Please install: cmake ninja git curl build-essential rocm-hip-sdk"
    fi
}

install_deps
mkdir -p "$MODELS_DIR"

# ── Build kernels + server (pure C++, no Rust) ───────────────────────────────
if [ "$SKIP_ROCM" = false ]; then
    log "Building kernels (rocm-cpp) + server (zaya_server)..."
    cd "$DIR"
    cmake -B build -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1151
    cmake --build build --target zaya_server -j"$(nproc)"
    log "Build complete: $DIR/build/zaya_server ($(stat -c%s "$DIR/build/zaya_server") bytes)"
else
    warn "Skipping kernel build. Set LD_LIBRARY_PATH to find librocm_cpp.so."
    log "Building server only (requires pre-built librocm_cpp.so)..."
    cd "$DIR"
    cmake -B build -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1151
    cmake --build build --target zaya_server -j"$(nproc)"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
log ""
log "Done. Run:"
log "  export HSA_OVERRIDE_GFX_VERSION=11.5.1"
log "  export HSA_ENABLE_SDMA=0"
log "  export LD_LIBRARY_PATH=$DIR/build:\$LD_LIBRARY_PATH"
log "  $DIR/build/zaya_server"
log ""
log "Then send requests:"
log '  curl -X POST http://localhost:8088/completion \'
log '    -H "Content-Type: application/json" \'
log '    -d '\''{"prompt":"Hello","n_predict":16}'\'
log ""
log "Or use any OpenAI-compatible client:"
log '  from openai import OpenAI'
log '  client = OpenAI(base_url="http://localhost:8088/v1", api_key="any")'

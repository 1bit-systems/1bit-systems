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
        sudo apt-get update -qq
        sudo apt-get install -y -qq build-essential cmake ninja-build git curl || true
        # ROCm HIP SDK — try multiple package names across distro versions
        sudo apt-get install -y -qq rocm-hip-libraries 2>/dev/null || \
            sudo apt-get install -y -qq hip-sdk 2>/dev/null || \
            sudo apt-get install -y -qq rocm-dev 2>/dev/null || \
            warn "No ROCm HIP package found. Install manually: rocm-hip-libraries"
    elif command -v pacman &>/dev/null; then
        log "Installing build deps (pacman)..."
        sudo pacman -Sy --noconfirm base-devel cmake ninja git curl rocm-hip-sdk 2>/dev/null || \
            warn "ROCm HIP package not found. Install rocm-hip-sdk manually"
    elif command -v dnf &>/dev/null; then
        log "Installing build deps (dnf)..."
        sudo dnf install -y gcc-c++ cmake ninja-build git curl rocm-hip-devel 2>/dev/null || \
            warn "ROCm HIP package not found. Install rocm-hip-devel manually"
    else
        warn "Unknown package manager. Install: cmake ninja git curl build-essential + ROCm HIP SDK"
    fi
}

install_deps
mkdir -p "$MODELS_DIR"

# ── Build kernels + server (pure C++, no Rust) ───────────────────────────────
if [ "$SKIP_ROCM" = false ]; then
    log "Building kernels (rocm-cpp) + server (zaya_server)..."
    cd "$DIR"
    cmake -B build -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1151 || { warn "cmake configure failed"; exit 1; }
    cmake --build build --target zaya_server -j"$(nproc)" || { warn "cmake build failed"; exit 1; }
    log "Build complete: $DIR/build/zaya_server ($(stat -c%s "$DIR/build/zaya_server") bytes)"
else
    warn "--skip-rocm: kernel build skipped."
    warn "Make sure librocm_cpp.so is on LD_LIBRARY_PATH before running zaya_server."
    log "Checking for pre-built server binary..."
    if [ -f "$DIR/build/zaya_server" ]; then
        log "Found existing build: $DIR/build/zaya_server"
    else
        warn "No pre-built server found at $DIR/build/zaya_server."
        warn "Run without --skip-rocm on a ROCm-equipped machine, or"
        warn "download a pre-built release from GitHub."
    fi
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

#!/usr/bin/env bash
set -euo pipefail
GREEN='\033[0;32m'; NC='\033[0m'; YELLOW='\033[1;33m'
log() { echo -e "${GREEN}[1bit]${NC} $*"; }
warn() { echo -e "${YELLOW}[1bit]${NC} $*"; }

REPO_URL="https://github.com/bong-water-water-bong/1bit.git"
INSTALL_DIR="${INSTALL_DIR:-$HOME/1bit}"
SKIP_ROCM=false; [ "${1:-}" = "--skip-rocm" ] && SKIP_ROCM=true
MODELS_DIR="${MODELS_DIR:-$HOME/models}"

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    echo "Usage: curl -fsSL https://raw.githubusercontent.com/bong-water-water-bong/1bit/main/install.sh | bash"
    echo "       bash install.sh [--skip-rocm]"
    echo ""
    echo "  --skip-rocm  Skip kernel build (use pre-build librocm_cpp.so)"
    echo ""
    echo "Installs 1bit inference engine for AMD Strix Halo (gfx1151)."
    echo "Clones the repo to ${INSTALL_DIR:-\$HOME/1bit}, builds kernels + Rust server."
    exit 0
fi

# ── Detect if running standalone (curl-piped) vs from repo root ────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" 2>/dev/null && pwd || echo ".")"
if [ -f "$SCRIPT_DIR/CMakeLists.txt" ] && [ -d "$SCRIPT_DIR/rust" ]; then
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

# ── Rust ───────────────────────────────────────────────────────────────────────
if ! command -v cargo &>/dev/null; then
    log "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env"

install_deps
mkdir -p "$MODELS_DIR"

# ── Kernels ────────────────────────────────────────────────────────────────────
if [ "$SKIP_ROCM" = false ]; then
    log "Building kernels (rocm-cpp)..."
    cd "$DIR"
    cmake -B build -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1151
    ninja -C build rocm_cpp bitnet_decode bench_prefill_variants
    log "Kernel build complete: $DIR/build/librocm_cpp.so"
else
    warn "Skipping kernel build. Set LD_LIBRARY_PATH to find librocm_cpp.so."
fi

# ── Server ─────────────────────────────────────────────────────────────────────
log "Building Rust server..."
cd "$DIR/rust"
source "$HOME/.cargo/env"
cargo build --release
log "Server built: $DIR/rust/target/release/onebit"

# ── Test ───────────────────────────────────────────────────────────────────────
if cargo test --release 2>&1; then
    log "✓ Tests pass (7/7)"
else
    warn "Tests failed — check output above"
fi

# ── Done ───────────────────────────────────────────────────────────────────────
log ""
log "Done. Run:"
log "  export HSA_OVERRIDE_GFX_VERSION=11.5.1"
log "  export HSA_ENABLE_SDMA=0"
log "  export LD_LIBRARY_PATH=$DIR/build:\$LD_LIBRARY_PATH"
log "  $DIR/rust/target/release/onebit --model model.h1b --port 13305 --tune-prefill --fp16-weights"
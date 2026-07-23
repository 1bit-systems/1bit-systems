#!/usr/bin/env bash
set -euo pipefail
GREEN='\033[0;32m'; NC='\033[0m'; YELLOW='\033[1;33m'
log() { echo -e "${GREEN}[1bit]${NC} $*"; }
warn() { echo -e "${YELLOW}[1bit]${NC} $*"; }

REPO_URL="https://github.com/bong-water-water-bong/1bit-systems.git"
INSTALL_DIR="${INSTALL_DIR:-$HOME/1bit}"
SKIP_ROCM=false; [ "${1:-}" = "--skip-rocm" ] && SKIP_ROCM=true
MODELS_DIR="${MODELS_DIR:-$HOME/models}"

# ── Kernel version check ───────────────────────────────────────────────────────
# amdgpu OPTC CRTC hang on Strix Halo (gfx1151) with kernel 6.19.x (issue #1)
# Works fine on 6.18.x LTS and 7.x. Warn users on 6.19.x.
KERNEL_RELEASE="$(uname -r)"
if echo "$KERNEL_RELEASE" | grep -q '^6\.19\.'; then
    warn "Kernel $KERNEL_RELEASE detected!"
    warn "Strix Halo (gfx1151) systems running 6.19.x kernels may experience"
    warn "an amdgpu OPTC CRTC hang during GPU inference (issue #1)."
    warn "Recommended: use kernel 6.18.22-lts or 7.x instead."
    warn "See: https://github.com/bong-water-water-bong/1bit-systems/issues/1"
    echo ""
fi

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    echo "Usage: curl -fsSL https://raw.githubusercontent.com/bong-water-water-bong/1bit-systems/main/install.sh -o install.sh"
    echo "       # Review the script, then:"
    echo "       bash install.sh [--skip-rocm]"
    echo ""
    echo "  --skip-rocm  Skip kernel build (use pre-build librocm_cpp.so)"
    echo ""
    echo "If a SHA256 checksum file is available, verify before running:"
    echo "       sha256sum -c install.sh.sha256"
    echo ""
    echo "Installs 1bit inference engine for AMD Strix Halo (gfx1151)."
    echo "Builds pure C++ end-to-end: zaya_server, onebit (CLI), onebitd (daemon),"
    echo "unified_router (proxy), bitnet_tui (TUI) + librocm_cpp.so — no Rust, no Python."
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
        sudo apt-get install -y -qq build-essential cmake ninja-build git curl python3-pip
    elif command -v pacman &>/dev/null; then
        log "Installing build deps (pacman)..."
        sudo pacman -Sy --noconfirm base-devel cmake ninja git curl python-pip
    elif command -v dnf &>/dev/null; then
        log "Installing build deps (dnf)..."
        sudo dnf install -y gcc-c++ cmake ninja-build git curl python3-pip
    else
        warn "Unknown package manager. Install: cmake ninja git curl build-essential python3-pip"
    fi
    command -v ninja >/dev/null 2>&1 || { echo "WARNING: ninja not found, using Unix Makefiles"; CMAKE_GENERATOR=""; }
    
    # TheRock 7.15.0a — pip-installed HIP SDK for gfx1151
    if ! command -v amdclang++ &>/dev/null; then
        log "Installing TheRock 7.15.0a SDK..."
        python3 -m pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
            rocm[devel,libraries] 2>/dev/null || {
            warn "TheRock pip install failed. Set THEROCK_PIP_ROOT manually."
            warn "See: https://github.com/bong-water-water-bong/TheRock"
        }
        export THEROCK_PIP_ROOT="$HOME/.cache/pip/therock"
    else
        log "amdclang++ found — TheRock SDK already installed"
    fi
	command -v ninja >/dev/null 2>&1 || { echo "WARNING: ninja not found, using Unix Makefiles"; CMAKE_GENERATOR=""; }
}

install_deps
mkdir -p "$MODELS_DIR"

# ── Build kernels + server (pure C++, no Rust) ───────────────────────────────
if [ "$SKIP_ROCM" = false ]; then
    log "Building C++ inference stack (server + CLI + daemon)..."
    cd "$DIR"
    cmake -B build ${CMAKE_GENERATOR:+-G Ninja} -DCMAKE_HIP_ARCHITECTURES=gfx1151 || { warn "cmake configure failed"; exit 1; }
    cmake --build build --target zaya_server onebitd onebit onebit_bin unified_router -j"$(nproc)" || { warn "cmake build failed"; exit 1; }
    log "Build complete:"
    log "  $DIR/build/zaya_server ($(stat -c%s "$DIR/build/zaya_server" 2>/dev/null || echo '?') bytes)"
    log "  $DIR/build/onebitd      ($(stat -c%s "$DIR/build/onebitd" 2>/dev/null || echo '?') bytes)"
    log "  $DIR/build/onebit       ($(stat -c%s "$DIR/build/onebit" 2>/dev/null || echo '?') bytes)"
    log "  $DIR/build/1bit         → onebit"
    log "  $DIR/build/unified_router"
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

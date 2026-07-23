#!/usr/bin/env bash
# 1bit.systems — NPU Engine Install
# Pure NPU inference engine. No Python. No Node. No Docker. No dependencies.
#
# Usage:
#   curl -sL https://1bit.systems/npu-install.sh | bash
#   # or from a release tarball:
#   tar xzf 1bit-systems-*.tar.gz && cd 1bit-systems-* && bash npu-install.sh
#
# Requires: Linux on AMD Strix Halo (Ryzen AI Max+ 395), XRT 2.21+
# Optional: g++ for building from source
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
say()  { printf "${GREEN}✓${NC} %s\n" "$*"; }
warn() { printf "${CYAN}!${NC} %s\n" "$*"; }
die()  { printf "${RED}✗${NC} %s\n" "$*"; exit 1; }

echo ""
printf "${GREEN}╔══════════════════════════════════════════════╗${NC}\n"
printf "${GREEN}║  1bit.systems — NPU Inference Engine        ║${NC}\n"
printf "${GREEN}║  ~400 KB · 35 models · 9 backends · Zero deps  ║${NC}\n"
printf "${GREEN}╚══════════════════════════════════════════════╝${NC}\n"
echo ""

# ── Detect OS ──
[ "$(uname -s)" != "Linux" ] && die "Only Linux (AMD Strix Halo) is supported."

# ── Detect NPU hardware ──
if lspci 2>/dev/null | grep -qi "XDNA\|NPU\|AIE\|ACCEL"; then
  say "NPU hardware detected"
else
  warn "No NPU detected. This machine needs an AMD Strix Halo (Ryzen AI Max+ 395) for NPU inference."
  warn "You can still use the GPU engine: see engine/gpu/ for Zig Vulkan backend."
  echo ""
fi

# ── Check XRT ──
if ldconfig -p 2>/dev/null | grep -q libxrt_coreutil; then
  say "XRT runtime found"
else
  die "XRT 2.21+ not found. Install: sudo apt install libxrt2 libxrt-npu2 libxrt-dev"
fi

# ── Destination ──
INSTALL_DIR="${HOME}/.local/1bit-npu"
BIN_DIR="${HOME}/.local/bin"
mkdir -p "${INSTALL_DIR}/bin" "${BIN_DIR}"

# ── Determine source ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "${SCRIPT_DIR}/bin/1bit-npu" ]; then
  # Installing from a local tarball or repo checkout
  say "Installing from local source: ${SCRIPT_DIR}"
  cp "${SCRIPT_DIR}/bin/1bit-npu"     "${INSTALL_DIR}/bin/" 2>/dev/null || true
  cp "${SCRIPT_DIR}/bin/1bit-server"  "${INSTALL_DIR}/bin/" 2>/dev/null || true
  cp "${SCRIPT_DIR}/lib/"*.o          "${INSTALL_DIR}/lib/" 2>/dev/null || true
else
  # Download latest release from GitHub
  REPO="bong-water-water-bong/1bit-systems"
  say "Downloading latest release from ${REPO}..."
  DOWNLOAD_URL=$(curl -sfL "https://api.github.com/repos/${REPO}/releases/latest" \
    | grep "browser_download_url.*runtime.tar.gz" | head -1 | cut -d'"' -f4)
  [ -z "$DOWNLOAD_URL" ] && die "Could not find latest release asset."
  TMPDIR=$(mktemp -d)
  curl -sfL "$DOWNLOAD_URL" -o "${TMPDIR}/runtime.tar.gz"
  tar xzf "${TMPDIR}/runtime.tar.gz" -C "${TMPDIR}"
  cp -r "${TMPDIR}"/* "${INSTALL_DIR}/"
  rm -rf "${TMPDIR}"
fi

# ── Symlink binaries ──
for BIN in 1bit-npu 1bit-server; do
  if [ -f "${INSTALL_DIR}/bin/${BIN}" ]; then
    ln -sf "${INSTALL_DIR}/bin/${BIN}" "${BIN_DIR}/${BIN}"
    chmod +x "${BIN_DIR}/${BIN}"
    say "Linked ${BIN}"
  fi
done

# ── Add to PATH ──
if [[ ":$PATH:" != *":${HOME}/.local/bin:"* ]]; then
  SHELL_CONFIG="${HOME}/.bashrc"
  [ -n "$ZSH_VERSION" ] && SHELL_CONFIG="${HOME}/.zshrc"
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "${SHELL_CONFIG}"
  say "Added ~/.local/bin to PATH in ${SHELL_CONFIG}"
fi

# ── Post-install instructions ──
echo ""
say "Installation complete!"
echo ""
echo "  ┌─ Quick start ─────────────────────────────────────────────────┐"
echo "  │                                                               │"
echo "  │  # 1. Get a model (Q4NX format):                              │"
echo "  │    ~/.local/bin/1bit-npu --download qwen3:0.6b                │"
echo "  │    # or place your .q4nx file in ~/.local/share/1bit/models/  │"
echo "  │                                                               │"
echo "  │  # 2. Run inference:                                          │"
echo "  │    ~/.local/bin/1bit-npu model.q4nx 16                        │"
echo "  │                                                               │"
echo "  │  # 3. Start HTTP server (OpenAI-compatible API):               │"
echo "  │    ~/.local/bin/1bit-server 8081                               │"
echo "  │    # Then: curl http://localhost:8081/v1/chat/completions ...  │"
echo "  │                                                               │"
echo "  │  # 4. Build from source (optional):                           │"
echo "  │    g++ -std=c++23 -O3 -march=native -fopenmp -ffast-math      │"
echo "  │        -o npu_engine engine/npu/src/npu_engine_all.cpp         │"
echo "  │        engine/npu/build/dequant_q4nx.o -lxrt_coreutil          │"
echo "  │                                                               │"
echo "  └───────────────────────────────────────────────────────────────┘"
echo ""
echo "  Docs:  https://1bit.systems"
echo "  Repo:  https://github.com/bong-water-water-bong/1bit-systems"
echo "  Help:  1bit-npu --help"
echo ""

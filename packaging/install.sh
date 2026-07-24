#!/usr/bin/env bash
# 1bit.systems — instant install from tarball or curl
# curl -sL https://1bit.systems/install.sh | bash
# Or: tar xzf 1bit-systems-*.tar.gz && cd 1bit-systems-* && bash install.sh
set -euo pipefail

RED='\033[0;31m' GREEN='\033[0;32m' CYAN='\033[0;36m' NC='\033[0m'
say() { printf "${GREEN}✓${NC} %s\n" "$*"; }
warn() { printf "${CYAN}!${NC} %s\n" "$*"; }
die() { printf "${RED}✗${NC} %s\n" "$*"; exit 1; }

echo ""
printf "${GREEN}╔══════════════════════════════════════════╗${NC}\n"
printf "${GREEN}║   1bit.systems — 94 tok/s NPU · 100% local ║${NC}\n"
printf "${GREEN}╚══════════════════════════════════════════╝${NC}\n"
echo ""

# ── Where we're installing ──
INSTALL_DIR="${HOME}/.local/1bit-systems"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Node.js check ──
if ! command -v node &>/dev/null; then
  die "Node.js 22+ required. Install: curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - && sudo apt install -y nodejs"
fi

NODE_VER=$(node -v | cut -d. -f1 | tr -d v)
[ "$NODE_VER" -lt 22 ] && warn "Node $(node -v) detected. Recommend Node 22+."

# ── Install files ──
say "Installing to ${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}"

# Copy from tarball source
cp -r "${SCRIPT_DIR}"/* "${INSTALL_DIR}/" 2>/dev/null || true

# ── npm install (optional but recommended) ──
cd "${INSTALL_DIR}"
if [ -f package.json ]; then
  say "Installing npm dependencies..."
  npm install --ignore-scripts --no-audit --no-fund 2>/dev/null && say "npm deps installed" || warn "npm install skipped — CLI works from dist/ regardless"
fi

# ── Symlink CLI ──
mkdir -p "${HOME}/.local/bin"
ln -sf "${INSTALL_DIR}/bin/1bit" "${HOME}/.local/bin/1bit"
say "1bit CLI linked to ~/.local/bin/1bit"

if [[ ":$PATH:" != *":${HOME}/.local/bin:"* ]]; then
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "${HOME}/.bashrc"
  say "Added ~/.local/bin to PATH in .bashrc"
fi

# ── NPU check ──
echo ""
if lspci 2>/dev/null | grep -qi "XDNA\|NPU\|AIE"; then
  say "NPU hardware detected"
  echo ""
  echo "  To run inference:"
  echo "    1. Start the NPU stack (daemon + server): 1bit up"
  echo "    2. Chat: 1bit chat"
else
  warn "No NPU detected. You can still use the CLI, but need a Strix Halo for inference."
  echo "  'unified_server' and '1bit up' fall back to GPU/CPU backends without an NPU."
fi

echo ""
echo "  Quick start:"
echo "    1bit chat          # interactive session"
echo "    1bit help          # show commands"
echo "    1bit up            # start NPU stack"
echo "    1bit status        # check daemon health"
echo ""
echo "  Docs:  https://1bit.systems"
echo "  Repo:  https://github.com/bong-water-water-bong/1bit-systems"
echo ""
echo "  —bong-water-water-bong · Sorry but not Sorry"

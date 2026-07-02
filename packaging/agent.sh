#!/usr/bin/env sh
# 1bit.systems — /1bit coding agent install
# curl -fsSL https://1bit.systems/agent.sh | sh
set -e

echo "╔══════════════════════════════════════════════╗"
echo "║       /1bit coding agent — install          ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Check Node.js
if ! command -v node >/dev/null 2>&1; then
    echo "Node.js required. Install: https://nodejs.org (>= 22)"
    exit 1
fi

NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 22 ]; then
    echo "Node.js >= 22 required. Current: $(node -v)"
    exit 1
fi
echo "✅ Node.js $(node -v)"

# Install via npm
echo ""
echo "Installing /1bit coding agent..."
npm install -g @1bit/1bit-coding-agent 2>/dev/null || {
    echo "npm publish pending — using source install"
    echo "git clone https://github.com/bong-water-water-bong/1bit-agent.git"
    echo "cd 1bit-agent && npm install && npm run build"
    echo "npm link packages/coding-agent"
    exit 0
}

echo ""
echo "✅ /1bit installed!"
echo ""
echo "Usage:"
echo "  1bit \"write a hello world\""
echo "  1bit -p \"refactor this file\""
echo "  1bit --help"
echo ""
echo "Docs: https://1bit.systems"
echo "Repo: https://github.com/bong-water-water-bong/1bit-agent"

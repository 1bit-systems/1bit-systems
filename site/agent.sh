#!/usr/bin/env bash
# 1bit.systems — /1bit coding agent install + NPU model registration
# curl -fsSL https://1bit.systems/agent.sh | sh
set -euo pipefail

echo "╔══════════════════════════════════════════════════╗"
echo "║       /1bit coding agent — install             ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Check Node.js
if ! command -v node >/dev/null 2>&1; then
    echo "Node.js required. Install: https://nodejs.org (>= 22)"
    exit 1
fi
echo "✅ Node.js $(node -v)"

# Install /1bit agent
echo ""
echo "Installing /1bit coding agent..."
npm install -g @1bit/1bit-coding-agent 2>/dev/null && echo "✅ /1bit installed via npm" || true

if ! command -v 1bit >/dev/null 2>&1; then
    echo "npm publish pending — using source install"
    if [ ! -d /tmp/1bit-agent ]; then
        git clone https://github.com/bong-water-water-bong/1bit-agent.git /tmp/1bit-agent 2>/dev/null
    fi
    cd /tmp/1bit-agent
    npm install --ignore-scripts 2>/dev/null
    npm run build 2>/dev/null
    npm link 2>/dev/null
    echo "✅ /1bit installed from source"
fi

# Register NPU models if server detected
echo ""
echo "Checking for NPU server..."
if curl -s http://localhost:8081/health >/dev/null 2>&1; then
    echo "✅ NPU server detected on port 8081"
    
    # Create config dir if needed
    mkdir -p ~/.1bit/agent

    # Write models.json
    cat > ~/.1bit/agent/models.json << 'MODELS'
{
  "providers": {
    "1bit-npu": {
      "baseUrl": "http://localhost:8081/v1",
      "api": "openai-completions",
      "apiKey": "1bit",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false
      },
      "models": [
        { "id": "qwen3-0.6b", "name": "Qwen3 0.6B (NPU · 28 tok/s)", "contextWindow": 4096 },
        { "id": "gemma4-e2b", "name": "Gemma4 E2B (NPU · 16 tok/s)", "contextWindow": 4096 },
        { "id": "qwen3-vl-4b", "name": "Qwen3 VL 4B (NPU · 11 tok/s)", "contextWindow": 4096 },
        { "id": "llama-3.1-8b", "name": "Llama 3.1 8B (NPU · 10 tok/s)", "contextWindow": 4096 },
        { "id": "qwen3-8b", "name": "Qwen3 8B (NPU · 8 tok/s)", "contextWindow": 4096 }
      ]
    }
  }
}
MODELS
    echo "✅ 5 NPU models registered"

    # Set 1bit-npu as default provider if settings file exists
    if [ -f ~/.1bit/agent/settings.json ]; then
        python3 -c "
import json
with open('$HOME/.1bit/agent/settings.json') as f: s=json.load(f)
s.setdefault('apiKeys',{})['1bit-npu']='1bit'
s['provider']='1bit-npu'; s['model']='1bit-npu/qwen3-0.6b'
with open('$HOME/.1bit/agent/settings.json','w') as f: json.dump(s,f,indent=2)
" 2>/dev/null && echo "✅ Default provider set to 1bit-npu" || echo "⚠️  Settings update skipped"
    fi
else
    echo "⚠️  NPU server not detected on port 8081"
    echo "   Start server: OMP_NUM_THREADS=16 ./npu_engine_all model.q4nx 16"
    echo "   Models auto-register on next /1bit launch when server is running"
fi

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║         /1bit ready — 5 NPU models             ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "Usage:"
echo "  1bit                 # interactive mode"
echo "  1bit \"write rust\"    # one-shot prompt"  
echo "  1bit -c              # continue last session"
echo "  /model               # switch models in TUI"
echo ""
echo "Models: qwen3-0.6b · gemma4-e2b · qwen3-vl-4b · llama-3.1-8b · qwen3-8b"
echo "Docs:   https://1bit.systems"
echo "Repo:   https://github.com/bong-water-water-bong/1bit-agent"

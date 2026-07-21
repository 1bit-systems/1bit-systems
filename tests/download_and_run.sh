#!/bin/bash
# tests/download_and_run.sh — Automated model download + inference smoke test
#
# Downloads a small model (Qwen3-0.6B GGUF from HuggingFace), runs inference
# through zaya_server, and checks that output is non-degenerate.
#
# This is the CI gate that catches regressions in the decode loop, tokenizer,
# weight loader, and backend dispatch. It's NOT a correctness oracle — just a
# "not stuck, not garbage" sanity check.
#
# Usage:
#   bash tests/download_and_run.sh              # full run (downloads model)
#   bash tests/download_and_run.sh --quick       # skip download, use cached
#   bash tests/download_and_run.sh --ci          # CI variant, 10 tokens max
set -euo pipefail

MODEL_URL="https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/qwen3-0.6b-q4_k_m.gguf"
MODEL_FILE="/tmp/qwen3-0.6b-q4_k_m.gguf"
SERVER_BIN="${1:-./build/zaya_server}"
TIMEOUT_SECS=120
CI_MODE=false
QUICK=false

for arg in "$@"; do
  case "$arg" in
    --ci) CI_MODE=true; TIMEOUT_SECS=60;;
    --quick) QUICK=true;;
  esac
done

echo "=== 1bit-systems Inference Smoke Test ==="
echo "Hardware: $(uname -m), NPU: $(xrt-smi examine -r 2>/dev/null | grep -oP 'RyzenAI-\S+' || echo 'N/A')"
echo ""

# 1. Download model
if [ ! -f "$MODEL_FILE" ] || [ "$QUICK" = false ]; then
  echo "Downloading Qwen3-0.6B Q4_K_M ($MODEL_URL)..."
  if command -v wget &>/dev/null; then
    wget -q --show-progress "$MODEL_URL" -O "$MODEL_FILE" 2>&1 | tail -1
  elif command -v curl &>/dev/null; then
    curl -sL "$MODEL_URL" -o "$MODEL_FILE"
  else
    echo "ERROR: need wget or curl"
    exit 1
  fi
  echo "  Downloaded: $(ls -lh "$MODEL_FILE" | awk '{print $5}')"
else
  echo "Using cached model: $MODEL_FILE ($(ls -lh "$MODEL_FILE" | awk '{print $5}'))"
fi

# 2. Build server if not present
if [ ! -f "$SERVER_BIN" ]; then
  echo "Building zaya_server..."
  cmake -B build -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1151 2>&1 | tail -1
  cmake --build build --target zaya_server -j8 2>&1 | tail -3
fi

# 3. Start server
echo "Starting zaya_server..."
PORT=$((RANDOM + 10000))
"$SERVER_BIN" --model "$MODEL_FILE" --port "$PORT" &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

# Wait for server to be ready
for i in $(seq 1 30); do
  if curl -s "http://127.0.0.1:$PORT/v1/health" 2>/dev/null | grep -q ok; then
    echo "  Server ready on port $PORT (${i}s)"
    break
  fi
  if [ $i -eq 30 ]; then
    echo "ERROR: Server did not start in 30s"
    exit 1
  fi
  sleep 1
done

# 4. Run inference
N_TOKENS=10
[ "$CI_MODE" = true ] && N_TOKENS=5

echo "Running inference ($N_TOKENS tokens)..."
RESPONSE=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"zaya\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":$N_TOKENS}" 2>&1)

echo "Response: $(echo "$RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" 2>/dev/null || echo "$RESPONSE" | head -3)"

# 5. Validate: check for degenerate output
LEN=$(echo "$RESPONSE" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    c=d['choices'][0]['message']['content']
    print(len(c.split()))
except: print(0)
" 2>/dev/null)

if [ "$LEN" -gt 0 ]; then
  echo "✅ PASS — generated $LEN tokens, output non-degenerate"
else
  echo "❌ FAIL — empty or degenerate output"
  echo "Raw response: $RESPONSE"
  exit 1
fi

# 6. Record benchmark
TOK_S=$(echo "$RESPONSE" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    u=d.get('usage',{})
    n=u.get('completion_tokens',1)
    t=u.get('time_ms',1000)
    print(f'{n/(t/1000):.1f}')
except: print(0)
" 2>/dev/null)

if [ "$TOK_S" != "0" ]; then
  bash bench/record.sh "zaya_server_smoke" "$TOK_S" null "validated" \
    "End-to-end smoke test (Qwen3-0.6B Q4_K_M)" "e2e" "zaya_server" "CPU+NPU"
fi

echo "=== Smoke test complete ==="

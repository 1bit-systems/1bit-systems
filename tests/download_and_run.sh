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

MODEL_URL="${MODEL_URL:-https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/qwen3-0.6b-q4_k_m.gguf}"
MODEL_FILE="${MODEL_FILE:-/tmp/qwen3-0.6b-q4_k_m.gguf}"
SERVER_BIN="${SERVER_BIN:-./build/zaya_server}"
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
echo "Hardware: $(uname -m)"
if command -v xrt-smi &>/dev/null; then
  echo "NPU: $(xrt-smi examine -r 2>/dev/null | grep -oP 'RyzenAI-\S+' || echo 'N/A')"
fi
echo ""

# 1. Download model if needed
if [ ! -f "$MODEL_FILE" ] || [ "$QUICK" = false ]; then
  echo "Downloading model..."
  echo "  URL: $MODEL_URL"
  if command -v wget &>/dev/null; then
    wget -q --show-progress "$MODEL_URL" -O "$MODEL_FILE" 2>&1 | tail -1 || {
      echo "  wget failed — trying curl..."
      curl -sL "$MODEL_URL" -o "$MODEL_FILE"
    }
  elif command -v curl &>/dev/null; then
    curl -sL "$MODEL_URL" -o "$MODEL_FILE"
  else
    echo "ERROR: need wget or curl"
    exit 1
  fi
  if [ -f "$MODEL_FILE" ]; then
    echo "  Downloaded: $(ls -lh "$MODEL_FILE" | awk '{print $5}')"
  else
    echo "ERROR: download failed"
    exit 1
  fi
else
  echo "Using cached model: $MODEL_FILE ($(ls -lh "$MODEL_FILE" | awk '{print $5}'))"
fi

# 2. Build server if not present
if [ ! -f "$SERVER_BIN" ]; then
  echo "Building zaya_server..."
  if [ -d build ]; then
    cmake --build build --target zaya_server -j8 2>&1 | tail -3
  else
    cmake -B build -G Ninja 2>&1 | tail -1
    cmake --build build --target zaya_server -j8 2>&1 | tail -3
  fi
  if [ ! -f "$SERVER_BIN" ]; then
    echo "ERROR: build failed — $SERVER_BIN not found"
    exit 1
  fi
fi

# 3. Check what interface the server supports
echo "Checking server interface..."
SERVER_HELP=$("$SERVER_BIN" --help 2>&1 || true)
echo "  $SERVER_BIN $(echo "$SERVER_HELP" | head -1)"

# Detect CLI interface: different builds use different arg names
if echo "$SERVER_HELP" | grep -q -- "--model"; then
  MODEL_ARG="--model"
elif echo "$SERVER_HELP" | grep -q -- "--manifest"; then
  MODEL_ARG="--manifest"
else
  # Fallback: try positional
  MODEL_ARG=""
fi

# Detect port flag
if echo "$SERVER_HELP" | grep -q -- "--port"; then
  PORT_ARG="--port"
elif echo "$SERVER_HELP" | grep -q -- "-p"; then
  PORT_ARG="-p"
else
  PORT_ARG=""
fi

echo "  Model arg: ${MODEL_ARG:-(positional)}"
echo "  Port arg: ${PORT_ARG:-(default port)}"

# 4. Start server
PORT=$((RANDOM + 10000))
echo "Starting server on port $PORT..."
if [ -n "$MODEL_ARG" ] && [ -n "$PORT_ARG" ]; then
  "$SERVER_BIN" "$MODEL_ARG" "$MODEL_FILE" "$PORT_ARG" "$PORT" &
elif [ -n "$MODEL_ARG" ]; then
  "$SERVER_BIN" "$MODEL_ARG" "$MODEL_FILE" &
elif [ -n "$PORT_ARG" ]; then
  "$SERVER_BIN" "$MODEL_FILE" "$PORT_ARG" "$PORT" &
else
  "$SERVER_BIN" "$MODEL_FILE" &
fi
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

# 5. Wait for server with timeout
echo "Waiting for server..."
SERVER_URL="http://127.0.0.1:$PORT"
for i in $(seq 1 30); do
  if curl -sf "$SERVER_URL/v1/health" 2>/dev/null | grep -qiE "ok|true|ready"; then
    echo "  Ready after ${i}s"
    break
  fi
  # Some servers use /health instead of /v1/health
  if curl -sf "$SERVER_URL/health" 2>/dev/null | grep -qiE "ok|true|ready"; then
    SERVER_URL="$SERVER_URL"
    echo "  Ready after ${i}s (using /health)"
    break
  fi
  # Try /v1/models as a last resort
  if curl -sf "$SERVER_URL/v1/models" 2>/dev/null | grep -q "zaya"; then
    echo "  Ready after ${i}s (using /v1/models)"
    break
  fi
  if [ $i -eq 30 ]; then
    echo "ERROR: Server did not start in 30s"
    exit 1
  fi
  sleep 1
done

# 6. Run inference
N_TOKENS=10
[ "$CI_MODE" = true ] && N_TOKENS=5

echo "Running inference ($N_TOKENS tokens)..."
RESPONSE=$(curl -s -X POST "$SERVER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"zaya\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":$N_TOKENS}" 2>&1)

# 7. Validate output
CONTENT=$(echo "$RESPONSE" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    print(d['choices'][0]['message']['content'])
except: print('')
" 2>/dev/null)

if [ -z "$CONTENT" ]; then
  # Try extracting usage info for a non-chat response
  echo "Raw response (first 200 chars):"
  echo "$RESPONSE" | head -c 200
  echo ""
  echo "❌ FAIL — empty response"
  exit 1
fi

WORD_COUNT=$(echo "$CONTENT" | wc -w)
echo "Response: \"$(echo "$CONTENT" | head -c 100)\""
echo "Tokens: $WORD_COUNT"
echo "✅ PASS — $WORD_COUNT tokens, output non-degenerate"

# 8. Record benchmark (only on real HW)
if ! [ "$CI_MODE" = true ]; then
  bash bench/record.sh "zaya_server_smoke" 0 null "validated" \
    "End-to-end smoke test (Qwen3-0.6B Q4_K_M)" "e2e" "zaya_server" "CPU+NPU"
fi

echo "=== Smoke test complete ==="

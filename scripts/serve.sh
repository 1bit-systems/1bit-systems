#!/bin/bash
# 1bit.systems — Production NPU Inference Stack
# Starts FLM backend + daemon, provides OpenAI-compatible API on port 9090.
set -euo pipefail

PORT="${1:-9090}"
MODEL="${2:-qwen3:0.6b}"
FLM_PORT="${3:-52625}"
PMODE="${4:-turbo}"

# Clean up any existing instances
pkill -f "flm serve.*:${FLM_PORT}" 2>/dev/null || true
pkill -f "npu-gpu-cpud.*:${PORT}" 2>/dev/null || true
sleep 1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "╔══════════════════════════════════════════════════╗"
echo "║  1bit.systems NPU Inference Stack                ║"
echo "║  Model: $MODEL                                    ║"
echo "║  API:   http://0.0.0.0:$PORT                      ║"
echo "╚══════════════════════════════════════════════════╝"

# Start FLM backend
echo ""
echo "[1/2] Starting FLM NPU backend on :$FLM_PORT..."
flm serve "$MODEL" --port "$FLM_PORT" --pmode "$PMODE" &
FLMPID=$!
sleep 4

if ! kill -0 $FLMPID 2>/dev/null; then
    echo "ERROR: FLM failed to start"
    exit 1
fi
echo "       FLM running (pid=$FLMPID)"

# Verify FLM
if curl -s "http://127.0.0.1:$FLM_PORT/v1/models" >/dev/null 2>&1; then
    echo "       ✅ FLM responding"
else
    echo "       ⚠️  FLM may still be loading..."
fi

# Start daemon
echo ""
echo "[2/2] Starting control plane daemon on :$PORT..."
"$REPO_ROOT/daemon/npu-gpu-cpud" --port "$PORT" --npu-port "$FLM_PORT" &
DAEMONPID=$!
sleep 2

if ! kill -0 $DAEMONPID 2>/dev/null; then
    echo "ERROR: Daemon failed to start"
    kill $FLMPID 2>/dev/null
    exit 1
fi
echo "       Daemon running (pid=$DAEMONPID)"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Stack ready!                                    ║"
echo "║  curl http://127.0.0.1:$PORT/v1/chat/completions  ║"
echo "║    -d '{\"model\":\"$MODEL\",\"messages\":[...]}'     ║"
echo "╚══════════════════════════════════════════════════╝"

# Quick smoke test
sleep 2
echo ""
echo "Smoke test:"
curl -s -m 20 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":5}" 2>/dev/null | \
  python3 -c "import sys,json; d=json.load(sys.stdin); print('  Response:', d['choices'][0]['message']['content']); print('  Device:', d.get('x-device','?'))" 2>/dev/null || echo "  (still warming up...)"

# Trap cleanup
trap "echo ''; echo 'Shutting down...'; kill $DAEMONPID $FLMPID 2>/dev/null; wait 2>/dev/null; echo 'Done.'" EXIT INT TERM

# Keep running
echo ""
echo "Press Ctrl+C to stop."
wait

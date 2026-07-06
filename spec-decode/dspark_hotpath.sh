#!/usr/bin/env bash
# DSpark Hot-Path Daemon — Starts FLM (NPU target) + spec_proxy (DSpark draft)
set -ueo pipefail

FLM_PORT=52625
PROXY_PORT=9090

echo "═══ DSpark Hot Path ═══"
echo "Starting FLM NPU target on port $FLM_PORT..."

# Kill any stale FLM
pkill -f "flm serve qwen3" 2>/dev/null || true
sleep 1

# Start FLM (NPU backend)
/usr/bin/flm serve qwen3:0.6b --port $FLM_PORT --host 127.0.0.1 &
FLM_PID=$!

# Wait for FLM to be ready
for i in $(seq 1 30); do
  if ss -tlnp | grep -q :$FLM_PORT; then
    echo "  FLM ready (PID $FLM_PID)"
    break
  fi
  sleep 1
done

if ! ss -tlnp | grep -q :$FLM_PORT; then
  echo "  ERROR: FLM failed to start"
  exit 1
fi

echo "Starting DSpark spec_proxy on port $PROXY_PORT..."
exec /home/bcloud/spec-decode/build/spec_proxy --port $PROXY_PORT --flm-port $FLM_PORT

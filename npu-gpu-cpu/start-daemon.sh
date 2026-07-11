#!/bin/bash
# Start the NPU+GPU+CPU Control Plane Daemon
# Usage: ./start-daemon.sh [port]

PORT=${1:-8080}
DIR="$(cd "$(dirname "$0")" && pwd)"

# Reload NPU driver
sudo -n modprobe -r amdxdna 2>/dev/null
sudo -n modprobe amdxdna
sleep 1

# Start daemon (needs disown to survive shell exit)
sudo -n python3 "$DIR/daemon/npu-gpu-cpud.py" --port "$PORT" &
disown

sleep 3
echo ""
echo "Daemon started on http://localhost:$PORT"
echo "  GET  http://localhost:$PORT/v1/health"
echo "  POST http://localhost:$PORT/v1/chat/completions"
echo ""
echo "Example:"
echo "  curl -X POST http://localhost:$PORT/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"qwen3:0.6b\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":16}'"

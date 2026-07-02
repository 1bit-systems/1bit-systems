#!/bin/bash
# Launch the NPU control plane daemon with sudo privileges
# Usage: ./start.sh [port]

PORT=${1:-8080}
DIR="$(cd "$(dirname "$0")" && pwd)"

# Reload NPU driver to clear stuck BOs
modprobe -r amdxdna 2>/dev/null
modprobe amdxdna
sleep 1

# Start daemon
exec python3 "$DIR/npu-gpu-cpud.py" --port "$PORT"

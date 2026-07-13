#!/bin/bash
# run_flm_gdb.sh — Launch FLM under GDB for BO protocol capture
set -e

CAPTURE_DIR="/tmp/gdb_flm_capture"
mkdir -p "$CAPTURE_DIR"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GDB_SCRIPT="$SCRIPT_DIR/gdb_capture_bo.py"

# Kill existing FLM
pkill -9 flm 2>/dev/null || true
sleep 1

echo "Starting FLM under GDB..."
echo "Output: $CAPTURE_DIR/gdb_capture.log"

# Launch FLM via GDB in background
gdb -batch \
    -x "$GDB_SCRIPT" \
    --args /opt/fastflowlm/bin/flm serve --host 127.0.0.1 --port 8080 \
    2>&1 | tee "$CAPTURE_DIR/gdb_output.txt" &

GDB_PID=$!
echo "GDB PID: $GDB_PID"

# Wait for FLM to be ready
for i in {1..30}; do
    if ss -tlnp 2>/dev/null | grep -q 8080; then
        echo "FLM ready after ${i}s"
        break
    fi
    sleep 1
done

# Send test request
echo "Sending inference request..."
python3 -c "
import requests, json
r = requests.post('http://127.0.0.1:8080/v1/chat/completions',
    json={
        'model': 'qwen3:0.6b', 
        'messages': [{'role': 'user', 'content': 'What is the capital of France?'}], 
        'max_tokens': 5,
        'stream': False
    },
    headers={'Content-Type': 'application/json'},
    timeout=60)
resp = r.json()
if 'choices' in resp:
    print('Content:', resp['choices'][0]['message']['content'])
elif 'error' in resp:
    print('Error:', resp['error']['message'][:200])
"

# Allow GDB to process
sleep 3

# Kill FLM and GDB
kill $GDB_PID 2>/dev/null || true
sleep 1
pkill -9 flm 2>/dev/null || true

echo ""
echo "=== Capture complete ==="
echo "Log: $CAPTURE_DIR/gdb_capture.log"
echo "Output: $CAPTURE_DIR/gdb_output.txt"
ls -la "$CAPTURE_DIR/"

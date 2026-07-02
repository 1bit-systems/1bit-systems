#!/bin/bash
# Post-reboot test: compare FLM vs our engine quality
# Run as: bash post-reboot-test.sh

set -e

echo "============================================"
echo " NPU Quality Test — FLM vs Custom Engine"
echo "============================================"
echo ""

# 1. Reload driver
echo "1. Reloading NPU driver..."
sudo modprobe -r amdxdna 2>/dev/null || true
sudo modprobe amdxdna
sleep 2
echo "   OK"

# 2. Start FLM (reference quality)
echo ""
echo "2. FLM baseline (reference quality)..."
flm serve qwen3:0.6b --port 52627 > /tmp/flm_ref.log 2>&1 &
FLM_PID=$!
sleep 8

echo "   FLM health:"
curl -s http://localhost:52627/v1/health | python3 -c "import sys,json; d=json.load(sys.stdin); print('   status:', d.get('status','unknown'))" 2>/dev/null || echo "   FLM failed to start"

echo ""
echo "   FLM chat (Hello):"
curl -s -X POST http://localhost:52627/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3:0.6b","messages":[{"role":"user","content":"Hello, how are you?"}],"max_tokens":32}' | \
  python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    txt = d['choices'][0]['message']['content']
    print(f'   FLM: {txt[:200]}')
except Exception as e:
    print(f'   FLM error: {e}')
    print(sys.stdin.read()[:200])
" 2>/dev/null

kill $FLM_PID 2>/dev/null

# 3. Reload driver for our engine
echo ""
echo "3. Reloading driver for custom engine..."
sudo modprobe -r amdxdna 2>/dev/null || true
sudo modprobe amdxdna
sleep 2

# 4. Our daemon
echo ""
echo "4. Custom engine daemon..."
sudo python3 /home/bcloud/npu-gpu-cpu/daemon/npu-gpu-cpud.py --port 9090 > /tmp/our_daemon.log 2>&1 &
DAEMON_PID=$!
sleep 8

echo "   Health:"
curl -s http://localhost:9090/v1/health | python3 -c "import sys,json; d=json.load(sys.stdin); print('   NPU available:', d['devices']['npu']['available'])" 2>/dev/null

echo ""
echo "   Chat (Hello):"
curl -s -X POST http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3:0.6b","messages":[{"role":"user","content":"Hello, how are you?"}],"max_tokens":32}' | \
  python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    txt = d['choices'][0]['message']['content']
    print(f'   US: {txt[:200]}')
    print(f'   ms/tok: {d.get(\"x-ms-per-tok\",\"?\")}')
except Exception as e:
    print(f'   Error: {e}')
" 2>/dev/null

echo ""
echo "============================================"
echo " Comparison complete."
echo " If FLM produces coherent text and we don't,"
echo " the issue is in npu_engine_stdio.cpp"
echo "============================================"

kill $DAEMON_PID 2>/dev/null

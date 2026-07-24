#!/bin/bash
# run_zr1_zaya_demo.sh — Build + launch ZR1→Zaya speculative decode demo
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

echo "╔══════════════════════════════════════════════╗"
echo "║  ZR1→Zaya Speculative Decode Demo           ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# 1. Build zaya_server if needed
SERVER_BIN="./build/zaya_server"
if [ ! -f "$SERVER_BIN" ]; then
    echo "Building zaya_server..."
    cmake -B build -G Ninja 2>/dev/null || cmake -B build
    cmake --build build --target zaya_server -j8
    echo ""
fi

# 2. Verify model files exist
ZR1_MODEL="models/ZR1-1.5B.1bp"
ZAYA_MODEL="models/ZAYA1-8B-Q4_K_M.gguf"

if [ ! -f "$ZR1_MODEL" ]; then
    echo "❌ ZR1 model not found at $ZR1_MODEL"
    echo "   Convert first: ./build/gguf_to_onebp models/Zyphra_ZR1-1.5B-Q4_K_M.gguf $ZR1_MODEL"
    exit 1
fi

if [ ! -f "$ZAYA_MODEL" ]; then
    echo "❌ Zaya model not found at $ZAYA_MODEL"
    exit 1
fi

# 3. Kill any existing servers on our ports
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $PID_ZR1 2>/dev/null || true
    kill $PID_ZAYA 2>/dev/null || true
    wait 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

for port in 8081 8082; do
    lsof -ti :$port 2>/dev/null | xargs kill 2>/dev/null || true
done
sleep 1

# 4. Start ZR1 draft server
echo "Starting ZR1-1.5B draft server (port 8081)..."
"$SERVER_BIN" --model "$ZR1_MODEL" --port 8081 &
PID_ZR1=$!
echo "  PID: $PID_ZR1"

# 5. Start Zaya target server
echo "Starting Zaya1-8B target server (port 8082)..."
"$SERVER_BIN" --model "$ZAYA_MODEL" --port 8082 &
PID_ZAYA=$!
echo "  PID: $PID_ZAYA"

# 6. Wait for both servers to be ready
echo ""
echo "Waiting for servers..."
for i in $(seq 1 30); do
    zr1_ok=$(curl -sf http://127.0.0.1:8081/v1/health 2>/dev/null && echo 1 || echo 0)
    zaya_ok=$(curl -sf http://127.0.0.1:8082/v1/health 2>/dev/null && echo 1 || echo 0)
    if [ "$zr1_ok" = "1" ] && [ "$zaya_ok" = "1" ]; then
        echo "  Both ready after ${i}s"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "  Timed out waiting for servers"
        exit 1
    fi
    sleep 1
done

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Running speculative decode...               ║"
echo "╚══════════════════════════════════════════════╝"

# 7. Run the coordinator
python3 "$SCRIPT_DIR/zr1_zaya_spec_demo.py" \
    --draft-port 8081 \
    --verify-port 8082 \
    --n-draft 5 \
    --n-rounds 5 \
    --prompt "Write a short poem about artificial intelligence."

echo ""
echo "Demo complete."

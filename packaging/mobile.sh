#!/usr/bin/env bash
# 1bit.systems — mobile setup: ngrok + QR + 1bit-mobile
# curl -fsSL https://1bit.systems/mobile.sh | sh
set -euo pipefail

echo "╔══════════════════════════════════════════════════╗"
echo "║     1bit Mobile — QR code pairing              ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

PORT="${1:-8081}"
NGROK_AUTH="${BIT_NGROK_AUTH:-}"

# Step 1: Check NPU server
echo "📡 Checking NPU server on port $PORT..."
if curl -s http://localhost:$PORT/health >/dev/null 2>&1; then
    echo "✅ NPU server running on port $PORT"
else
    echo "⚠️  NPU server not running. Start it first:"
    echo "   1bit-server $PORT &"
    echo "   Or: OMP_NUM_THREADS=16 ./npu_engine_all model.q4nx"
    echo ""
fi

# Step 2: Setup ngrok
if command -v ngrok >/dev/null 2>&1; then
    echo "✅ ngrok found"
    if [ -n "$NGROK_AUTH" ]; then
        ngrok config add-authtoken "$NGROK_AUTH" 2>/dev/null || true
    fi

    # Kill existing ngrok
    pkill ngrok 2>/dev/null || true
    sleep 1

    # Start ngrok in background
    nohup ngrok http $PORT --log=stdout > /tmp/ngrok-1bit.log 2>&1 &
    NGROK_PID=$!
    echo "   ngrok PID: $NGROK_PID"

    # Wait for ngrok to start and get URL
    echo "   Waiting for ngrok tunnel..."
    for _i in $(seq 1 10); do
        NGROK_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | grep -o '"public_url":"[^"]*"' | head -1 | cut -d'"' -f4)
        if [ -n "$NGROK_URL" ]; then
            break
        fi
        sleep 1
    done

    if [ -z "$NGROK_URL" ]; then
        echo "   ⚠️  Could not get ngrok URL. Check /tmp/ngrok-1bit.log"
        echo "   Tunnel may still be starting. Check: http://localhost:4040"
    else
        echo ""
        echo "╔══════════════════════════════════════════════════╗"
        echo "║  🌐 Public URL: $NGROK_URL"
        echo "╚══════════════════════════════════════════════════╝"
    fi
else
    echo "⚠️  ngrok not installed."
    echo "   Install: snap install ngrok"
    echo "   Or: curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc"
    echo "   Then: ngrok config add-authtoken <your-token>"
fi

# Step 3: Generate QR code
if command -v qrencode >/dev/null 2>&1; then
    if [ -n "$NGROK_URL" ]; then
        echo ""
        echo "═══════════════════════════════════════════════════"
        qrencode -t ANSIUTF8 "$NGROK_URL/v1/chat/completions" 2>/dev/null || \
        qrencode -t UTF8 "$NGROK_URL/v1/chat/completions" 2>/dev/null || true
        echo "═══════════════════════════════════════════════════"
        echo ""
        echo "📱 Scan this QR code with 1bit Mobile"
        echo "   iOS: https://apps.apple.com/app/1bit-mobile/id6757372210"
        echo "   Android: https://play.google.com/store/apps/details?id=com.1bit.mobile"
    fi
else
    echo ""
    echo "⚠️  qrencode not installed (needed for QR code)"
    echo "   Install: sudo apt install qrencode"
    if [ -n "$NGROK_URL" ]; then
        echo ""
        echo "📱 Point 1bit Mobile to: $NGROK_URL/v1/chat/completions"
    fi
fi

# Step 4: Instructions
echo ""
echo "═══════════════════════════════════════════════════"
echo "  1bit Mobile Setup Complete"
echo "═══════════════════════════════════════════════════"
echo ""
echo "  Server: http://localhost:$PORT"
if [ -n "$NGROK_URL" ]; then
    echo "  Public: $NGROK_URL"
fi
echo ""
echo "  To reconnect later:"
echo "    curl -fsSL https://1bit.systems/mobile.sh | sh"
echo ""
echo "  To kill tunnel:"
echo "    pkill ngrok"
echo ""

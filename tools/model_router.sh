#!/usr/bin/env bash
# Model Router — dispatches inference requests to the right backend.
# Usage: ./model_router.sh MODEL PROMPT [N_PREDICT]
# Models: zaya, albert, qwen
set -euo pipefail

MODEL="${1:-zaya}"
PROMPT="${2:-Hello}"
N="${3:-16}"
cd ${HOME}

case "$MODEL" in
  zaya|zaya1|ZAYA)
    echo "→ ZAYA1-8B (GPU HIP)"
    PORT=8088
    # Start server if not running
    if ! curl -sf http://127.0.0.1:$PORT/ >/dev/null 2>&1; then
        echo "  Starting server..."
        build/zaya_server $PORT &>/tmp/zaya_server.log &
        for i in $(seq 1 30); do
            sleep 2
            if curl -sf http://127.0.0.1:$PORT/ >/dev/null 2>&1; then
                echo "  Server ready after $((i*2))s"
                break
            fi
        done
    fi
    # Send request
    curl -s -X POST http://127.0.0.1:$PORT/completion \
      -H "Content-Type: application/json" \
      -d "{\"prompt\":\"$PROMPT\",\"n_predict\":$N}"
    ;;

  albert|albert-moe|ALBERT)
    echo "→ Albert-MoE-13 (Rust CPU)"
    BIN="/tmp/ternary-intelligence-stack/albert-moe-13/target/release/moe-test"
    if [ ! -f "$BIN" ]; then
        echo "  Building Albert..."
        cd /tmp/ternary-intelligence-stack/albert-moe-13
        cargo build --release --bin moe-test 2>&1 | tail -3
    fi
    # Run benchmark with prompt (PROMPT passed via env var to prevent shell injection)
    echo "  Running benchmark... (use screen for interactive TUI)"
    export PROMPT_VAL="$PROMPT" BIN_VAL="$BIN"
    screen -dmS albert bash -c 'cd /tmp/ternary-intelligence-stack/albert-moe-13 && echo "$PROMPT_VAL" | timeout 30 "$BIN_VAL" 2>&1 | tee /tmp/albert_out.log'
    sleep 2
    echo "  Started in screen session 'albert'"
    echo "  View: screen -r albert"
    ;;

  qwen|qwen3|fused)
    echo "→ Qwen3-0.6B (NPU+GPU Fused)"
    BIN="${HOME}/engine/fusion/zig-out/bin/fused-engine"
    if [ ! -f "$BIN" ]; then
        echo "  Building fused engine..."
        cd ${HOME}/engine/fusion && zig build 2>&1 | tail -3
    fi
    $BIN --model ${HOME}/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx \
          --prompt "$PROMPT" --max-tokens $N
    ;;

  bitnet|BITNET)
    echo "→ BitNet b1.58 (C++ CPU) — build pending"
    echo "  Model downloaded, binary not yet available"
    echo "  Try: cd ${HOME}/bitnet.cpp && cmake --build build"
    ;;

  list|help|*)
    echo "Available models:"
    echo "  zaya    - ZAYA1-8B          GPU    6.5 tok/s  ✅"
    echo "  albert  - Albert-MoE-13     CPU    40 tok/s   ✅"
    echo "  qwen    - Qwen3-0.6B        NPU+GPU 18 tok/s  ✅"
    echo "  bitnet  - BitNet b1.58      CPU    pending    🔄"
    echo ""
    echo "Usage: $0 <model> [prompt] [n_predict]"
    ;;
esac

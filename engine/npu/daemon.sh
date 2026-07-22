#!/usr/bin/env bash
set -euo pipefail
# 1bit.systems live daemon — runs NPU engine and serves output for live dashboard
# Usage: ./daemon.sh [port]  (default: 8001)

PORT="${1:-8001}"
ENGINE="./build/npu_engine_cb"
STATEDIR="/tmp/1bit-live"
mkdir -p "$STATEDIR"

# Run NPU engine, capture output to state file
run_engine() {
    local OUTFILE="$STATEDIR/run_$1.txt"
    local JSONFILE="$STATEDIR/latest.json"
    echo "{" > "$JSONFILE"
    echo "  \"status\": \"running\"," >> "$JSONFILE"
    echo "  \"timestamp\": \"$(date -Iseconds)\"," >> "$JSONFILE"
    echo "  \"prompt_tokens\": $1," >> "$JSONFILE"
    echo "  \"decode_tokens\": $2" >> "$JSONFILE"
    echo "}" >> "$JSONFILE"

    # Run engine
    "$ENGINE" "$1" "$2" 2>&1 | tee "$OUTFILE"
    local EXIT=$?

    # Parse output for JSON
    local SPEED
    local TOKENS
    local PREFILL
    SPEED=$(grep "ms/tok" "$OUTFILE" | tail -1 | awk '{print $2}')
    TOKENS=$(grep -c "^  \\[" "$OUTFILE")
    PREFILL=$(grep "Prefill:" "$OUTFILE" | tail -1 | awk '{print $2}' | tr -d 'ms')

    cat > "$JSONFILE" << JSONEOF
{
  "status": "complete",
  "exit": $EXIT,
  "speed_ms_tok": "$SPEED",
  "tokens_generated": $TOKENS,
  "prefill_ms": "$PREFILL",
  "timestamp": "$(date -Iseconds)",
  "output": $(python3 -c "import sys,json; print(json.dumps(open('$OUTFILE').read()))")
}
JSONEOF
    echo "{\"status\":\"done\",\"speed_ms_tok\":$SPEED,\"tokens\":$TOKENS}" >> "$STATEDIR/runlog.jsonl"
}

# Serve state file via netcat HTTP
serve() {
    while true; do
        nc -l -p "$PORT" -q 1 2>/dev/null <<-HTTP
HTTP/1.1 200 OK
Content-Type: application/json
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, OPTIONS
Access-Control-Allow-Headers: Content-Type

$(cat $STATEDIR/latest.json 2>/dev/null || echo '{"status":"idle"}')
HTTP
    done
}

case "${1:-serve}" in
    run)
        run_engine "${2:-9}" "${3:-8}"
        ;;
    serve)
        echo "1bit.systems live daemon on :$PORT"
        echo "State: $STATEDIR"
        serve
        ;;
    *)
        echo "Usage: $0 [serve|run PROMPT DECODE]"
        ;;
esac

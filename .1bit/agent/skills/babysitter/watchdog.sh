#!/usr/bin/env bash
# watchdog.sh — Agent Babysitter
# Monitors interactive_shell sessions for stalls, hangs, and failures.
# Auto-recovers with degraded retry strategies.
#
# Usage:
#   watchdog.sh --session-id <id> --max-stall 180 --max-runtime 600
#   watchdog.sh --swarm-id <id> --concurrency 5
#   watchdog.sh --list                         # show all watched sessions
#   watchdog.sh --journal                      # show babysitter journal

set -euo pipefail

JOURNAL="${HOME}/.pi/agent/cache/babysitter-journal.jsonl"
CACHE_DIR="${HOME}/.pi/agent/cache"
mkdir -p "$CACHE_DIR"

# ─── Args ──────────────────────────────────────────────────

SESSION_ID=""
SWARM_ID=""
MAX_STALL=180
MAX_RUNTIME=600
MODE="session"
ACTION="watch"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --session-id) SESSION_ID="$2"; shift 2 ;;
        --swarm-id) SWARM_ID="$2"; MODE="swarm"; shift 2 ;;
        --max-stall) MAX_STALL="$2"; shift 2 ;;
        --max-runtime) MAX_RUNTIME="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --on-fail) ON_FAIL="$2"; shift 2 ;;
        --list) ACTION="list"; shift ;;
        --journal) ACTION="journal"; shift ;;
        --kill) ACTION="kill"; SESSION_ID="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

ON_FAIL="${ON_FAIL:-retry-simpler}"
CONCURRENCY="${CONCURRENCY:-5}"

# ─── Colors ────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;90m'
NC='\033[0m'
BOLD='\033[1m'

# ─── Journal ───────────────────────────────────────────────

log_journal() {
    local session_id="$1" status="$2" runtime_s="${3:-0}" stalls="${4:-0}" retries="${5:-0}" output_lines="${6:-0}" recovery="${7:-}"
    local entry
    entry=$(jq -nc \
        --arg sid "$session_id" \
        --arg status "$status" \
        --arg runtime "$runtime_s" \
        --arg stalls "$stalls" \
        --arg retries "$retries" \
        --arg lines "$output_lines" \
        --arg recovery "$recovery" \
        --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        '{session_id: $sid, status: $status, runtime_s: ($runtime|tonumber), stalls: ($stalls|tonumber), retries: ($retries|tonumber), output_lines: ($lines|tonumber), recovery: $recovery, timestamp: $ts}')
    echo "$entry" >> "$JOURNAL"
}

# ─── Stall Detection ───────────────────────────────────────

check_stall() {
    local session_id="$1"
    local output_file="$CACHE_DIR/watchdog-${session_id}.out"
    local stall_file="$CACHE_DIR/watchdog-${session_id}.stalls"

    # Get current output
    local current_output
    current_output=$(pi -p "show session $session_id" 2>/dev/null || echo "")

    if [[ -z "$current_output" ]]; then
        # Try interactive_shell approach
        current_output="[no output available — session may be headless]"
    fi

    # Check if output is same as last poll
    if [[ -f "$output_file" ]]; then
        local prev_output
        prev_output=$(cat "$output_file")
        if [[ "$current_output" == "$prev_output" ]]; then
            local stall_count
            stall_count=$(cat "$stall_file" 2>/dev/null || echo 0)
            stall_count=$((stall_count + 1))
            echo "$stall_count" > "$stall_file"

            local stall_seconds=$((stall_count * 90))  # assume 90s poll interval
            if [[ $stall_seconds -ge $MAX_STALL ]]; then
                echo "STALLED:$stall_seconds:$stall_count"
                return 1
            else
                echo "SAME:$stall_seconds:$stall_count"
                return 0
            fi
        else
            echo "0" > "$stall_file"
            echo "CHANGED"
            return 0
        fi
    else
        echo "0" > "$stall_file"
        echo "FIRST_POLL"
        return 0
    fi

    echo "$current_output" > "$output_file"
}

# ─── Error Loop Detection ──────────────────────────────────

detect_error_loop() {
    local output_file="$CACHE_DIR/watchdog-${1}.out"
    if [[ -f "$output_file" ]]; then
        local apologize_count
        apologize_count=$(grep -ci "I apologize\|Let me try\|Unfortunately\|I couldn't" "$output_file" 2>/dev/null || echo 0)
        if [[ $apologize_count -ge 3 ]]; then
            echo "ERROR_LOOP:$apologize_count"
            return 1
        fi
    fi
    echo "NO_LOOP"
    return 0
}

# ─── Recovery ──────────────────────────────────────────────

recover_session() {
    local session_id="$1" strategy="$2"
    echo -e "  ${YELLOW}[watchdog]${NC} Recovery: ${CYAN}${strategy}${NC} for session ${GRAY}${session_id}${NC}"

    case "$strategy" in
        retry-same)
            echo "  → Killing and respawning with same task"
            pi -p "kill session $session_id" 2>/dev/null || true
            log_journal "$session_id" "retrying" 0 0 1 0 "retry-same"
            ;;
        retry-simpler)
            echo "  → Killing. Task will be split into smaller sub-tasks."
            pi -p "kill session $session_id" 2>/dev/null || true
            log_journal "$session_id" "retrying" 0 0 1 0 "retry-simpler"
            ;;
        retry-different-agent)
            echo "  → Killing. Will retry with different agent type."
            pi -p "kill session $session_id" 2>/dev/null || true
            log_journal "$session_id" "retrying" 0 0 1 0 "retry-different-agent"
            ;;
        escalate)
            echo -e "  ${RED}→ ESCALATING to supervisor. Cannot auto-recover.${NC}"
            log_journal "$session_id" "escalated" 0 0 1 0 "escalate"
            ;;
        skip)
            echo "  → Skipping. Non-critical task in batch."
            log_journal "$session_id" "skipped" 0 0 1 0 "skip"
            ;;
        *)
            echo "  → Unknown strategy: $strategy. Escalating."
            log_journal "$session_id" "escalated" 0 0 1 0 "unknown-strategy"
            ;;
    esac
}

# ─── Single Session Watch Cycle ────────────────────────────

watch_session() {
    local session_id="$1" start_time="$2"
    local stall_count=0 retry_count=0

    while true; do
        local now
        now=$(date +%s)
        local elapsed=$((now - start_time))

        # Check max runtime
        if [[ $elapsed -ge $MAX_RUNTIME ]]; then
            echo -e "  ${RED}[watchdog]${NC} Session ${GRAY}${session_id}${NC} exceeded max runtime (${elapsed}s > ${MAX_RUNTIME}s)"
            recover_session "$session_id" "escalate"
            return 1
        fi

        # Check for error loops
        local loop_result
        loop_result=$(detect_error_loop "$session_id")
        if [[ "$loop_result" == ERROR_LOOP:* ]]; then
            echo -e "  ${RED}[watchdog]${NC} Session ${GRAY}${session_id}${NC} in error loop: ${loop_result}"
            if [[ $retry_count -lt 2 ]]; then
                retry_count=$((retry_count + 1))
                recover_session "$session_id" "retry-simpler"
                return 2  # signal to caller: retry needed
            else
                recover_session "$session_id" "escalate"
                return 1
            fi
        fi

        # Check for stalls
        local stall_result
        stall_result=$(check_stall "$session_id")
        if [[ "$stall_result" == STALLED:* ]]; then
            local stall_secs="${stall_result#STALLED:}"; stall_secs="${stall_secs%%:*}"
            echo -e "  ${YELLOW}[watchdog]${NC} Session ${GRAY}${session_id}${NC} stalled for ${stall_secs}s"
            if [[ $retry_count -lt 2 ]]; then
                retry_count=$((retry_count + 1))
                recover_session "$session_id" "$ON_FAIL"
                return 2
            else
                recover_session "$session_id" "escalate"
                return 1
            fi
        fi

        # Session still running normally
        local runtime_min=$((elapsed / 60))
        echo -e "  ${GREEN}[watchdog]${NC} ${GRAY}${session_id}${NC} running (${runtime_min}m, stalls=${stall_count}, retries=${retry_count})"

        sleep 90
    done
}

# ─── List Watched Sessions ─────────────────────────────────

cmd_list() {
    echo ""
    echo -e "  ${BOLD}═══ Babysitter: Watched Sessions ═══${NC}"
    echo ""

    if [[ -f "$JOURNAL" ]]; then
        # Show active watches (last entry per session)
        local sessions
        sessions=$(tail -100 "$JOURNAL" | jq -r '.session_id' 2>/dev/null | sort -u || echo "")

        if [[ -z "$sessions" ]]; then
            echo "  ${GRAY}No watched sessions in journal${NC}"
            return
        fi

        for sid in $sessions; do
            local last_status
            last_status=$(grep "\"session_id\":\"$sid\"" "$JOURNAL" | tail -1 | jq -r '.status' 2>/dev/null || echo "unknown")
            local last_ts
            last_ts=$(grep "\"session_id\":\"$sid\"" "$JOURNAL" | tail -1 | jq -r '.timestamp' 2>/dev/null || echo "")
            local runtime_s
            runtime_s=$(grep "\"session_id\":\"$sid\"" "$JOURNAL" | tail -1 | jq -r '.runtime_s' 2>/dev/null || echo "0")

            case "$last_status" in
                running)     echo -e "  ${GREEN}🟢${NC} ${sid}  running  ${GRAY}${runtime_s}s${NC}  ${GRAY}${last_ts}${NC}" ;;
                completed)   echo -e "  ${GREEN}✅${NC} ${sid}  completed  ${GRAY}${runtime_s}s${NC}" ;;
                stalled)     echo -e "  ${YELLOW}⚠️${NC} ${sid}  stalled  ${GRAY}${last_ts}${NC}" ;;
                retrying)    echo -e "  ${CYAN}🔄${NC} ${sid}  retrying  ${GRAY}${last_ts}${NC}" ;;
                escalated)   echo -e "  ${RED}🚨${NC} ${sid}  escalated  ${GRAY}${last_ts}${NC}" ;;
                failed)      echo -e "  ${RED}❌${NC} ${sid}  failed  ${GRAY}${last_ts}${NC}" ;;
                *)           echo -e "  ${GRAY}  ${sid}  ${last_status}${NC}" ;;
            esac
        done
    else
        echo "  ${GRAY}No journal found${NC}"
    fi
    echo ""
}

# ─── Journal ────────────────────────────────────────────────

cmd_journal() {
    echo ""
    echo -e "  ${BOLD}═══ Babysitter Journal (last 20 entries) ═══${NC}"
    echo ""
    if [[ -f "$JOURNAL" ]]; then
        tail -20 "$JOURNAL" | while IFS= read -r line; do
            local sid status runtime stalls retries recovery
            sid=$(echo "$line" | jq -r '.session_id' 2>/dev/null || echo "?")
            status=$(echo "$line" | jq -r '.status' 2>/dev/null || echo "?")
            runtime=$(echo "$line" | jq -r '.runtime_s' 2>/dev/null || echo "0")
            stalls=$(echo "$line" | jq -r '.stalls' 2>/dev/null || echo "0")
            retries=$(echo "$line" | jq -r '.retries' 2>/dev/null || echo "0")
            rec=$(echo "$line" | jq -r '.recovery' 2>/dev/null || echo "")

            local icon=""
            case "$status" in
                completed) icon="${GREEN}✅${NC}" ;;
                running)   icon="${GREEN}🟢${NC}" ;;
                stalled)   icon="${YELLOW}⚠️${NC}" ;;
                retrying)  icon="${CYAN}🔄${NC}" ;;
                escalated) icon="${RED}🚨${NC}" ;;
                failed)    icon="${RED}❌${NC}" ;;
                skipped)   icon="${GRAY}⏭${NC}" ;;
                *)         icon="${GRAY}  ${NC}" ;;
            esac

            printf "  %b %-14s %-10s %4ss  stalls=%-2s retries=%-2s" "$icon" "$sid" "$status" "$runtime" "$stalls" "$retries"
            if [[ -n "$rec" ]]; then
                printf "  recovery=%-20s" "$rec"
            fi
            echo ""
        done
    else
        echo "  ${GRAY}No journal found${NC}"
    fi
    echo ""
}

# ─── Kill Session ───────────────────────────────────────────

cmd_kill() {
    local sid="$1"
    echo -e "  ${YELLOW}[watchdog]${NC} Killing session ${GRAY}${sid}${NC}"
    pi -p "kill session $sid" 2>/dev/null || true
    log_journal "$sid" "killed" 0 0 0 0 "manual-kill"
    rm -f "$CACHE_DIR/watchdog-${sid}.out" "$CACHE_DIR/watchdog-${sid}.stalls"
}

# ─── Main ──────────────────────────────────────────────────

case "$ACTION" in
    list)
        cmd_list
        ;;
    journal)
        cmd_journal
        ;;
    kill)
        cmd_kill "$SESSION_ID"
        ;;
    watch)
        if [[ "$MODE" == "session" ]]; then
            if [[ -z "$SESSION_ID" ]]; then
                echo "Usage: watchdog.sh --session-id <id> [--max-stall 180] [--max-runtime 600]"
                echo "       watchdog.sh --list"
                echo "       watchdog.sh --journal"
                exit 1
            fi
            echo ""
            echo -e "  ${BOLD}═══ Watchdog: ${CYAN}${SESSION_ID}${NC} ═══${NC}"
            echo -e "  Max stall: ${MAX_STALL}s  |  Max runtime: ${MAX_RUNTIME}s  |  On fail: ${ON_FAIL}"
            echo ""

            log_journal "$SESSION_ID" "running" 0 0 0 0 ""
            watch_session "$SESSION_ID" "$(date +%s)"
            exit_code=$?

            if [[ $exit_code -eq 0 ]]; then
                log_journal "$SESSION_ID" "completed" 0 0 0 0 ""
                echo -e "  ${GREEN}[watchdog]${NC} Session ${GRAY}${SESSION_ID}${NC} completed successfully"
            fi
        elif [[ "$MODE" == "swarm" ]]; then
            if [[ -z "$SWARM_ID" ]]; then
                echo "Usage: watchdog.sh --swarm-id <id> [--concurrency 5]"
                exit 1
            fi
            echo ""
            echo -e "  ${BOLD}═══ Swarm Watchdog: ${CYAN}${SWARM_ID}${NC} ═══${NC}"
            echo -e "  Concurrency: ${CONCURRENCY}  |  Max stall: ${MAX_STALL}s  |  Max runtime: ${MAX_RUNTIME}s"
            echo ""
            echo "  ${GRAY}Swarm watchdog active — monitor with:${NC}"
            echo "  ${GRAY}  watchdog.sh --list${NC}"
            echo "  ${GRAY}  watchdog.sh --journal${NC}"
        fi
        ;;
esac

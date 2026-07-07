#!/usr/bin/env bash
# tracker.sh — Token Budget Tracker
# Logs and summarizes token consumption across agents.
#
# Usage:
#   tracker.sh summary                      # daily summary
#   tracker.sh log <session-id> <provider> <tokens-in> <tokens-out> <cost> <task>
#   tracker.sh check                        # check against budget
#   tracker.sh reset                         # start new day

set -euo pipefail

BUDGET_JOURNAL="${HOME}/.pi/agent/cache/token-budget.jsonl"
mkdir -p "$(dirname "$BUDGET_JOURNAL")"

# ─── Colors ────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;90m'
NC='\033[0m'
BOLD='\033[1m'

# ─── Cost estimation per provider ──────────────────────────

estimate_cost() {
    local provider="$1" tokens_in="$2" tokens_out="$3"
    case "$provider" in
        deepseek)
            echo "scale=6; ($tokens_in / 1000000 * 0.50) + ($tokens_out / 1000000 * 2.00)" | bc -l 2>/dev/null || echo "0"
            ;;
        anthropic|claude)
            echo "scale=6; ($tokens_in / 1000000 * 15.00) + ($tokens_out / 1000000 * 75.00)" | bc -l 2>/dev/null || echo "0"
            ;;
        openai|codex)
            echo "scale=6; ($tokens_in / 1000000 * 3.00) + ($tokens_out / 1000000 * 15.00)" | bc -l 2>/dev/null || echo "0"
            ;;
        npu-local|npu|local)
            echo "0"
            ;;
        *)
            echo "0"
            ;;
    esac
}

# ─── Log Entry ─────────────────────────────────────────────

cmd_log() {
    local session_id="$1" provider="$2" tokens_in="$3" tokens_out="$4" cost_est="$5" task="$6"

    if [[ "$cost_est" == "auto" ]]; then
        cost_est=$(estimate_cost "$provider" "$tokens_in" "$tokens_out")
    fi

    local entry
    entry=$(jq -nc \
        --arg date "$(date +%Y-%m-%d)" \
        --arg provider "$provider" \
        --arg session "$session_id" \
        --arg in "$tokens_in" \
        --arg out "$tokens_out" \
        --arg cost "$cost_est" \
        --arg task "$task" \
        --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        '{
            date: $date,
            provider: $provider,
            session_id: $session,
            tokens_in: ($in|tonumber),
            tokens_out: ($out|tonumber),
            cost_est: ($cost|tonumber),
            task: $task,
            timestamp: $ts
        }')
    echo "$entry" >> "$BUDGET_JOURNAL"
    echo -e "  ${GREEN}Logged:${NC} ${session_id} | ${provider} | ${tokens_in}/${tokens_out} tokens | \$${cost_est} | ${task}"
}

# ─── Summary ────────────────────────────────────────────────

cmd_summary() {
    local today
    today=$(date +%Y-%m-%d)

    echo ""
    echo -e "  ${BOLD}═══ Token Budget: ${today} ═══${NC}"
    echo ""

    if [[ ! -f "$BUDGET_JOURNAL" ]]; then
        echo -e "  ${GRAY}No budget journal yet${NC}"
        echo ""
        return
    fi

    # Today only
    local today_entries
    today_entries=$(grep "\"date\":\"$today\"" "$BUDGET_JOURNAL" 2>/dev/null || echo "")

    if [[ -z "$today_entries" ]]; then
        echo -e "  ${GRAY}No entries for today${NC}"
        echo ""
        return
    fi

    # Per-provider totals
    local total=0
    local deepseek_total=0 anthropic_total=0 openai_total=0 npu_calls=0
    local deepseek_calls=0 anthropic_calls=0 openai_calls=0

    while IFS= read -r line; do
        local provider cost
        provider=$(echo "$line" | jq -r '.provider' 2>/dev/null)
        cost=$(echo "$line" | jq -r '.cost_est' 2>/dev/null)
        total=$(echo "$total + $cost" | bc -l)

        case "$provider" in
            deepseek) deepseek_total=$(echo "$deepseek_total + $cost" | bc -l); deepseek_calls=$((deepseek_calls + 1)) ;;
            anthropic|claude) anthropic_total=$(echo "$anthropic_total + $cost" | bc -l); anthropic_calls=$((anthropic_calls + 1)) ;;
            openai|codex) openai_total=$(echo "$openai_total + $cost" | bc -l); openai_calls=$((openai_calls + 1)) ;;
            npu*) npu_calls=$((npu_calls + 1)) ;;
        esac
    done <<< "$today_entries"

    local total_sessions=$((deepseek_calls + anthropic_calls + openai_calls + npu_calls))

    printf "  Daily limit: \$25.00 | Spent: \$%.2f (%.1f%%)\n" "$total" "$(echo "$total * 100 / 25" | bc -l)"
    echo ""

    # Per provider
    printf "  DeepSeek:    \$%.2f / \$15.00  (%d calls)\n" "$deepseek_total" "$deepseek_calls"
    printf "  Claude:      \$%.2f / \$8.00   (%d calls)\n" "$anthropic_total" "$anthropic_calls"
    printf "  Codex/GPT:   \$%.2f / \$2.00   (%d calls)\n" "$openai_total" "$openai_calls"
    printf "  NPU-local:   \$0.00 (free, %d calls)\n" "$npu_calls"
    echo ""

    if [[ $total_sessions -gt 0 ]]; then
        local avg_cost
        avg_cost=$(echo "scale=4; $total / $total_sessions" | bc -l)
        printf "  Sessions: %d | Avg cost: \$%.4f/session\n" "$total_sessions" "$avg_cost"
    fi

    # Warning thresholds
    local pct
    pct=$(echo "scale=1; $total * 100 / 25" | bc -l)
    if (( $(echo "$pct >= 100" | bc -l) )); then
        echo -e "  ${RED}🚫 BUDGET EXCEEDED — no more paid calls${NC}"
    elif (( $(echo "$pct >= 75" | bc -l) )); then
        echo -e "  ${YELLOW}⚠️  Approaching budget limit (${pct}%)${NC}"
    fi

    echo ""
}

# ─── Check Budget ──────────────────────────────────────────

cmd_check() {
    local today
    today=$(date +%Y-%m-%d)

    if [[ ! -f "$BUDGET_JOURNAL" ]]; then
        echo "OK:0:0"
        return
    fi

    local total=0
    while IFS= read -r line; do
        local cost
        cost=$(echo "$line" | jq -r '.cost_est' 2>/dev/null)
        total=$(echo "$total + $cost" | bc -l)
    done < <(grep "\"date\":\"$today\"" "$BUDGET_JOURNAL" 2>/dev/null || echo "")

    local pct
    pct=$(echo "scale=0; $total * 100 / 25" | bc -l)

    if (( $(echo "$pct >= 100" | bc -l) )); then
        echo "BLOCKED:${total}:${pct}"
    elif (( $(echo "$pct >= 75" | bc -l) )); then
        echo "WARN:${total}:${pct}"
    else
        echo "OK:${total}:${pct}"
    fi
}

# ─── Reset ──────────────────────────────────────────────────

cmd_reset() {
    local today
    today=$(date +%Y-%m-%d)
    echo "  ${YELLOW}Archiving today's journal...${NC}"
    if [[ -f "$BUDGET_JOURNAL" ]]; then
        cp "$BUDGET_JOURNAL" "${BUDGET_JOURNAL}.$(date +%Y%m%d).bak"
        # Keep only entries not from today
        grep -v "\"date\":\"$today\"" "$BUDGET_JOURNAL" > "${BUDGET_JOURNAL}.tmp" 2>/dev/null || true
        mv "${BUDGET_JOURNAL}.tmp" "$BUDGET_JOURNAL"
        echo -e "  ${GREEN}Reset complete${NC}"
    fi
}

# ─── Main ──────────────────────────────────────────────────

case "${1:-summary}" in
    summary|sum) cmd_summary ;;
    log)
        if [[ $# -lt 6 ]]; then
            echo "Usage: tracker.sh log <session-id> <provider> <tokens-in> <tokens-out> <cost|auto> <task>"
            exit 1
        fi
        cmd_log "$2" "$3" "$4" "$5" "$6" "$7"
        ;;
    check) cmd_check ;;
    reset) cmd_reset ;;
    estimate)
        estimate_cost "$2" "$3" "$4"
        ;;
    *)
        echo "Usage: tracker.sh {summary|log|check|reset|estimate}"
        echo ""
        echo "  summary              Show daily budget summary"
        echo "  log <sid> <prov> <in> <out> <cost> <task>  Log an agent call"
        echo "  check                Check budget status (OK/WARN/BLOCKED)"
        echo "  reset                Archive and reset for new day"
        echo "  estimate <prov> <in> <out>  Estimate cost for a call"
        exit 1
        ;;
esac

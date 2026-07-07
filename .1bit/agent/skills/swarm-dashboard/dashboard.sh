#!/usr/bin/env bash
# dashboard.sh — Swarm Dashboard
# Shows health of all running agents at a glance.
#
# Usage:
#   dashboard.sh                  # full dashboard
#   dashboard.sh --compact         # at-a-glance only
#   dashboard.sh --running         # only running sessions
#   dashboard.sh --json            # export as JSON
#   dashboard.sh --watch           # refresh every 15s
#   dashboard.sh --graph           # dependency graph

set -euo pipefail

CACHE_DIR="${HOME}/.pi/agent/cache"
STATE_FILE="${CACHE_DIR}/swarm-state.json"
JOURNAL="${CACHE_DIR}/babysitter-journal.jsonl"
BUDGET_JOURNAL="${CACHE_DIR}/token-budget.jsonl"
CHECKPOINT_DIR="${CACHE_DIR}"

MODE="full"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --compact) MODE="compact"; shift ;;
        --running) MODE="running"; shift ;;
        --json) MODE="json"; shift ;;
        --watch) MODE="watch"; shift ;;
        --graph) MODE="graph"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$CACHE_DIR"

# ─── Colors ────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
PINK='\033[0;35m'
GRAY='\033[0;90m'
NC='\033[0m'
BOLD='\033[1m'

# ─── Helpers ───────────────────────────────────────────────

token_bar() {
    local used="$1" max="$2"
    local pct=$((used * 100 / max))
    local bars=$((pct / 10))
    local bar=""
    for ((i=0; i<10; i++)); do
        if [[ $i -lt $bars ]]; then bar+="█"; else bar+="░"; fi
    done
    echo -n "${bar} ${pct}%"
}

format_runtime() {
    local seconds="$1"
    local m=$((seconds / 60))
    local s=$((seconds % 60))
    printf "%dm%02ds" "$m" "$s"
}

# ─── Resource Monitor ──────────────────────────────────────

get_resources() {
    # CPU
    local cpu_pct
    cpu_pct=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1 2>/dev/null || echo "?")
    cpu_pct=$(printf "%.0f" "$cpu_pct" 2>/dev/null || echo "?")

    # RAM
    local ram_used ram_total
    ram_used=$(free -g | awk '/Mem:/{print $3}' 2>/dev/null || echo "?")
    ram_total=$(free -g | awk '/Mem:/{print $2}' 2>/dev/null || echo "?")

    # Disk
    local disk_used disk_total
    disk_used=$(df -BG /home | awk 'NR==2{print $3}' 2>/dev/null | sed 's/G//' || echo "?")
    disk_total=$(df -BG /home | awk 'NR==2{print $2}' 2>/dev/null | sed 's/G//' || echo "?")

    # NPU usage
    local npu_pct="?"
    if command -v xrt-smi &>/dev/null; then
        npu_pct=$(xrt-smi examine 2>/dev/null | grep -oP '\d+(?=%)' | head -1 || echo "?")
    fi

    # GPU usage
    local gpu_pct="?"
    if command -v rocm-smi &>/dev/null; then
        gpu_pct=$(rocm-smi --showuse 2>/dev/null | grep -oP '\d+(?=%)' | head -1 || echo "?")
    fi

    echo "${cpu_pct}|${ram_used}|${ram_total}|${disk_used}|${disk_total}|${npu_pct}|${gpu_pct}"
}

# ─── Session Status ────────────────────────────────────────

get_sessions() {
    # Parse journal for latest status per session
    if [[ -f "$JOURNAL" ]]; then
        local sessions
        sessions=$(tail -200 "$JOURNAL" 2>/dev/null | jq -r '.session_id' 2>/dev/null | sort -u || echo "")

        for sid in $sessions; do
            local last
            last=$(grep "\"session_id\":\"$sid\"" "$JOURNAL" | tail -1)
            local status runtime_s stalls retries
            status=$(echo "$last" | jq -r '.status' 2>/dev/null || echo "unknown")
            runtime_s=$(echo "$last" | jq -r '.runtime_s' 2>/dev/null || echo "0")
            stalls=$(echo "$last" | jq -r '.stalls' 2>/dev/null || echo "0")
            retries=$(echo "$last" | jq -r '.retries' 2>/dev/null || echo "0")
            echo "${sid}|${status}|${runtime_s}|${stalls}|${retries}"
        done
    fi
}

# ─── Budget ─────────────────────────────────────────────────

get_budget() {
    local spent=0
    if [[ -f "$BUDGET_JOURNAL" ]]; then
        spent=$(tail -100 "$BUDGET_JOURNAL" 2>/dev/null | jq -s 'map(.cost_est | tonumber) | add' 2>/dev/null || echo "0")
    fi
    printf "%.2f" "$spent"
}

# ─── Display: Full Dashboard ───────────────────────────────

show_full() {
    echo ""
    echo -e "  ${BOLD}═══ Swarm Dashboard ═══${NC} $(date '+%Y-%m-%d %H:%M:%S') ${BOLD}═══${NC}"
    echo ""

    # ── Running ──
    echo -e "  ${BOLD}RUNNING${NC}"
    local has_running=false
    while IFS='|' read -r sid status runtime_s stalls retries; do
        [[ "$status" != "running" && "$status" != "retrying" ]] && continue
        has_running=true
        local icon="🟢"
        [[ "$status" == "retrying" ]] && icon="🔄"
        local rt=$(format_runtime "$runtime_s")
        printf "  %b %-14s  %-10s ${GRAY}%8s${NC}  stalls=%-2s retries=%-2s\n" "$icon" "$sid" "$status" "$rt" "$stalls" "$retries"
    done < <(get_sessions)
    [[ $has_running == false ]] && echo -e "  ${GRAY}  (none)${NC}"

    # ── Completed ──
    echo ""
    echo -e "  ${BOLD}COMPLETED${NC}"
    local has_completed=false
    while IFS='|' read -r sid status runtime_s stalls retries; do
        [[ "$status" != "completed" ]] && continue
        has_completed=true
        local rt=$(format_runtime "$runtime_s")
        printf "  ${GREEN}✅${NC} %-14s  completed  ${GRAY}%8s${NC}\n" "$sid" "$rt"
    done < <(get_sessions)
    [[ $has_completed == false ]] && echo -e "  ${GRAY}  (none)${NC}"

    # ── Failed/Stalled ──
    echo ""
    echo -e "  ${BOLD}FAILED / STALLED${NC}"
    local has_failed=false
    while IFS='|' read -r sid status runtime_s stalls retries; do
        case "$status" in
            failed|stalled|escalated) has_failed=true ;;
            *) continue ;;
        esac
        local icon="❌"
        [[ "$status" == "stalled" ]] && icon="⚠️"
        [[ "$status" == "escalated" ]] && icon="🚨"
        printf "  ${RED}%b${NC} %-14s  %-10s  stalls=%-2s retries=%-2s\n" "$icon" "$sid" "$status" "$stalls" "$retries"
    done < <(get_sessions)
    [[ $has_failed == false ]] && echo -e "  ${GRAY}  (none)${NC}"

    # ── Resources ──
    echo ""
    echo -e "  ${BOLD}RESOURCES${NC}"
    local resources
    resources=$(get_resources)
    IFS='|' read -r cpu_pct ram_used ram_total disk_used disk_total npu_pct gpu_pct <<< "$resources"
    printf "  CPU:  %s\n" "$(token_bar "$cpu_pct" 100)"
    printf "  RAM:  %s  (%sG / %sG)\n" "$(token_bar "$ram_used" "$ram_total")" "$ram_used" "$ram_total"
    printf "  Disk: %s  (%sG / %sG)\n" "$(token_bar "$disk_used" "$disk_total")" "$disk_used" "$disk_total"
    printf "  NPU:  %s  (max 50 TOPS)\n" "$(token_bar "${npu_pct:-0}" 100)"
    printf "  GPU:  %s\n" "$(token_bar "${gpu_pct:-0}" 100)"

    # ── Budget ──
    echo ""
    echo -e "  ${BOLD}BUDGET${NC}"
    local spent
    spent=$(get_budget)
    printf "  Spent: \$%s / \$25.00 (%.1f%%)\n" "$spent" "$(echo "$spent * 100 / 25" | bc -l 2>/dev/null || echo "?")"
    echo ""
}

# ─── Display: Compact ──────────────────────────────────────

show_compact() {
    echo ""
    echo -e "  ${BOLD}Swarm${NC} $(date '+%H:%M:%S')"

    local running=0 completed=0 failed=0
    while IFS='|' read -r sid status runtime_s stalls retries; do
        case "$status" in
            running|retrying) running=$((running + 1)) ;;
            completed) completed=$((completed + 1)) ;;
            failed|stalled|escalated) failed=$((failed + 1)) ;;
        esac
    done < <(get_sessions)

    echo -ne "  ${GREEN}🟢 ${running} running${NC}  "
    echo -ne "  ✅ ${completed} done  "
    echo -ne "  ${RED}❌ ${failed} failed${NC}"
    local spent
    spent=$(get_budget)
    echo -ne "  |  \$${spent}"
    echo ""
}

# ─── Display: Running Only ─────────────────────────────────

show_running() {
    echo ""
    echo -e "  ${BOLD}Running Sessions${NC}"
    local has=false
    while IFS='|' read -r sid status runtime_s stalls retries; do
        [[ "$status" != "running" && "$status" != "retrying" ]] && continue
        has=true
        local rt=$(format_runtime "$runtime_s")
        printf "  %-14s  %-10s  %8s  stalls=%s  retries=%s\n" "$sid" "$status" "$rt" "$stalls" "$retries"
    done < <(get_sessions)
    [[ $has == false ]] && echo -e "  ${GRAY}(none)${NC}"
    echo ""
}

# ─── Display: JSON ─────────────────────────────────────────

show_json() {
    local resources
    resources=$(get_resources)
    IFS='|' read -r cpu_pct ram_used ram_total disk_used disk_total npu_pct gpu_pct <<< "$resources"
    local spent
    spent=$(get_budget)

    local sessions_json="["
    local first=true
    while IFS='|' read -r sid status runtime_s stalls retries; do
        [[ $first == false ]] && sessions_json+=","
        first=false
        sessions_json+="{\"session_id\":\"$sid\",\"status\":\"$status\",\"runtime_s\":$runtime_s,\"stalls\":$stalls,\"retries\":$retries}"
    done < <(get_sessions)
    sessions_json+="]"

    jq -n \
        --argjson sessions "$sessions_json" \
        --arg cpu "$cpu_pct" \
        --arg ram_used "$ram_used" \
        --arg ram_total "$ram_total" \
        --arg disk_used "$disk_used" \
        --arg disk_total "$disk_total" \
        --arg npu "$npu_pct" \
        --arg gpu "$gpu_pct" \
        --arg spent "$spent" \
        --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        '{
            timestamp: $ts,
            sessions: $sessions,
            resources: { cpu_pct: $cpu, ram_used_gb: $ram_used, ram_total_gb: $ram_total, disk_used_gb: $disk_used, disk_total_gb: $disk_total, npu_pct: $npu, gpu_pct: $gpu },
            budget: { spent: ($spent|tonumber), limit: 25.00 }
        }'
}

# ─── Display: Dependency Graph ─────────────────────────────

show_graph() {
    echo ""
    echo -e "  ${BOLD}═══ Dependency Graph ═══${NC}"
    echo ""

    # Build a simple graph from journal entries
    local running=() completed=() failed=() pending=()
    while IFS='|' read -r sid status runtime_s stalls retries; do
        case "$status" in
            running|retrying) running+=("$sid") ;;
            completed) completed+=("$sid") ;;
            failed|stalled|escalated) failed+=("$sid") ;;
            *) pending+=("$sid") ;;
        esac
    done < <(get_sessions)

    for s in "${completed[@]}"; do
        echo -e "  ${GREEN}[${s}]${NC} ── ✅ done"
    done
    for s in "${running[@]}"; do
        echo -e "  ${CYAN}[${s}]${NC} ── 🟢 running"
    done
    for s in "${failed[@]}"; do
        echo -e "  ${RED}[${s}]${NC} ── ❌ failed"
    done
    for s in "${pending[@]}"; do
        echo -e "  ${GRAY}[${s}]${NC} ── ⏳ pending"
    done

    if [[ ${#completed[@]} -eq 0 && ${#running[@]} -eq 0 && ${#failed[@]} -eq 0 ]]; then
        echo -e "  ${GRAY}No sessions in journal${NC}"
    fi
    echo ""
}

# ─── Main ──────────────────────────────────────────────────

case "$MODE" in
    full)    show_full ;;
    compact) show_compact ;;
    running) show_running ;;
    json)    show_json ;;
    graph)   show_graph ;;
    watch)
        # Auto-show when agents are active, auto-hide when done
        local was_visible=false
        while true; do
            local has_active=false
            while IFS='|' read -r sid status runtime_s stalls retries; do
                case "$status" in
                    running|retrying) has_active=true; break ;;
                esac
            done < <(get_sessions)

            if [[ "$has_active" == true ]]; then
                clear 2>/dev/null || true
                show_compact
                was_visible=true
            elif [[ "$was_visible" == true ]]; then
                # Agents all done — clear dashboard
                clear 2>/dev/null || true
                echo ""
                echo -e "  ${GREEN}✅ All agents completed${NC}"
                echo ""
                was_visible=false
            fi
            sleep 5
        done
        ;;
esac

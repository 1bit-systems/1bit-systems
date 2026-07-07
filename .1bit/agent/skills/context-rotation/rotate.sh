#!/usr/bin/env bash
# rotate.sh — Context Rotation Helper
# Checkpoints agent context and prepares rotation.
#
# Usage:
#   rotate.sh check <session-id> <provider>          # check if rotation needed
#   rotate.sh checkpoint <session-id> <task> <done> <remaining> <files>
#   rotate.sh status                                  # show all session token depths

set -euo pipefail

CACHE_DIR="${HOME}/.pi/agent/cache"
CHECKPOINT_DIR="${CACHE_DIR}/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

# ─── Thresholds per provider ───────────────────────────────

get_threshold() {
    local provider="$1"
    case "$provider" in
        deepseek) echo "60000" ;;
        anthropic|claude) echo "90000" ;;
        openai|codex|gpt) echo "80000" ;;
        npu-local|npu|qwen) echo "16000" ;;
        *) echo "60000" ;;
    esac
}

get_max() {
    local provider="$1"
    case "$provider" in
        deepseek) echo "128000" ;;
        anthropic|claude) echo "200000" ;;
        openai|codex|gpt) echo "200000" ;;
        npu-local|npu|qwen) echo "32000" ;;
        *) echo "128000" ;;
    esac
}

# ─── Colors ────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;90m'
NC='\033[0m'
BOLD='\033[1m'

# ─── Check if Rotation Needed ──────────────────────────────

cmd_check() {
    local session_id="$1" provider="${2:-deepseek}" tokens="${3:-0}"
    local threshold max
    threshold=$(get_threshold "$provider")
    max=$(get_max "$provider")

    local pct=$((tokens * 100 / max))

    echo ""
    echo -e "  ${BOLD}Context Health: ${CYAN}${session_id}${NC} (${provider})${NC}"
    echo -e "  Tokens: ${tokens} / ${max} (${pct}%)"

    if [[ $tokens -ge $max ]]; then
        echo -e "  ${RED}🚫 AT LIMIT — must rotate immediately${NC}"
        echo "ROTATE_NOW:${tokens}:${max}:${pct}"
    elif [[ $tokens -ge $threshold ]]; then
        echo -e "  ${YELLOW}⚠️  Above rotation threshold (${threshold}) — rotate soon${NC}"
        echo "ROTATE_SOON:${tokens}:${max}:${pct}"
    elif [[ $tokens -ge $((threshold * 75 / 100)) ]]; then
        echo -e "  ${GREEN}✅ Approaching threshold — prepare checkpoint${NC}"
        echo "PREPARE:${tokens}:${max}:${pct}"
    else
        echo -e "  ${GREEN}✅ Healthy — no rotation needed${NC}"
        echo "HEALTHY:${tokens}:${max}:${pct}"
    fi
    echo ""
}

# ─── Create Checkpoint ──────────────────────────────────────

cmd_checkpoint() {
    local session_id="$1" task="$2" completed="$3" remaining="$4" in_progress="$5"
    local checkpoint_file="${CHECKPOINT_DIR}/checkpoint-${session_id}.md"
    local rotation_num=1

    # Check if this is a re-rotation
    if [[ -f "$checkpoint_file" ]]; then
        rotation_num=$(grep -c "^Rotation:" "$checkpoint_file" 2>/dev/null || echo 0)
        rotation_num=$((rotation_num + 2))
    fi

    cat > "$checkpoint_file" << EOF
# Checkpoint: ${task}
Rotation: ${rotation_num} | Time: $(date -u +%Y-%m-%dT%H:%M:%SZ) | Session: ${session_id}

## Completed
${completed}

## In Progress
${in_progress:-"(none)"}

## Remaining
${remaining}

## Files Modified
${FILES:-$(git -C /home/bcloud diff --name-only HEAD 2>/dev/null | head -20 | sed 's/^/- /' || echo "(none)")}

## Key Context
- Task: ${task}
- Provider: ${PROVIDER:-unknown}
- Checkpoint created by context-rotation skill

## Blockers
${BLOCKERS:-None}
EOF

    echo -e "  ${GREEN}✅ Checkpoint written:${NC} ${checkpoint_file}"
    echo -e "  ${GRAY}$(wc -l < "$checkpoint_file") lines${NC}"
    echo ""

    # Return the checkpoint content for LLM to consume
    echo "CHECKPOINT_FILE:${checkpoint_file}"
}

# ─── Status — All Sessions ─────────────────────────────────

cmd_status() {
    echo ""
    echo -e "  ${BOLD}═══ Context Health ═══${NC}"
    echo ""

    local has_checkpoints=false
    for cp in "$CHECKPOINT_DIR"/checkpoint-*.md; do
        [[ -f "$cp" ]] || continue
        has_checkpoints=true
        local sid rotation task
        sid=$(basename "$cp" .md | sed 's/^checkpoint-//')
        rotation=$(grep "^Rotation:" "$cp" | head -1 | awk '{print $2}' || echo "?")
        task=$(grep "^# Checkpoint:" "$cp" | sed 's/^# Checkpoint: //' || echo "?")
        printf "  %-14s  rotation=%-2s  %s\n" "$sid" "$rotation" "$task"
    done

    [[ $has_checkpoints == false ]] && echo -e "  ${GRAY}No checkpoints found${NC}"
    echo ""
}

# ─── Compaction Summary ────────────────────────────────────

cmd_compact() {
    local session_id="$1" task="$2" completed="$3" in_progress="$4" next="$5" files="$6"

    cat << EOF

<context_summary>
Task: ${task}
Completed:
${completed}
In progress: ${in_progress}
Files modified:
${files}
Next: ${next}
</context_summary>
EOF
}

# ─── Main ──────────────────────────────────────────────────

case "${1:-help}" in
    check)
        cmd_check "${2:-unknown}" "${3:-deepseek}" "${4:-0}"
        ;;
    checkpoint)
        if [[ $# -lt 5 ]]; then
            echo "Usage: rotate.sh checkpoint <session-id> <task> <completed> <remaining> [in-progress]"
            echo "  completed/remaining/in-progress: markdown bullet lists (use \$'...' for multiline)"
            exit 1
        fi
        cmd_checkpoint "$2" "$3" "$4" "$5" "${6:-(none)}"
        ;;
    status)
        cmd_status
        ;;
    compact)
        cmd_compact "${2:-}" "${3:-}" "${4:-}" "${5:-}" "${6:-}" "${7:-}"
        ;;
    *)
        echo "Usage: rotate.sh {check|checkpoint|status|compact}"
        echo ""
        echo "  check <sid> <provider> [tokens]     Check if rotation needed"
        echo "  checkpoint <sid> <task> <done> <remaining> [in-progress]"
        echo "  status                               Show all session checkpoints"
        echo "  compact <sid> <task> <done> <in-progress> <next> <files>"
        exit 1
        ;;
esac

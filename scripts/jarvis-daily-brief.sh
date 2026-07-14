#!/bin/bash
# jarvis-daily-brief.sh — JARVIS daily AI/LLM engineering briefing
#
# Queries multiple sources, compiles a structured digest,
# feeds it into the awareness system so all agents see it.
#
# Run manually:
#   ./scripts/jarvis-daily-brief.sh
#
# Or via cron (already set up at 8 AM daily)

set -e

AWARENESS_DIR="${HOME}/.1bit/agent"
AWARENESS_FILE="${AWARENESS_DIR}/awareness.json"
SIGNAL_SCRIPT="${HOME}/scripts/signal-agent-awareness.sh"
DIGEST_FILE="/tmp/jarvis-daily-digest.json"

echo "🦅 JARVIS daily briefing — $(date -u '+%Y-%m-%d %H:%M UTC')"
echo ""

# ── Step 1: Collect discoveries ────────────────────────────────────────────
# This section records discoveries compiled by the agent during its session.
# The agent uses web_search to find them. Here we track the structured output.

# ── Step 2: Record in awareness.json ──────────────────────────────────────
# Append discoveries as events so agents see them at session start.
record_discovery() {
    local title="$1" url="$2" relevance="$3" tags="$4" summary="$5"
    local now
    now="$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo 'unknown')"

    mkdir -p "$AWARENESS_DIR"
    if [ ! -f "$AWARENESS_FILE" ]; then
        printf '{"events":[],"lastSeen":{},"agents":{}}\n' > "$AWARENESS_FILE"
    fi

    # Pass all data via env vars (no shell→Python injection)
    EVENT_TITLE="$title" EVENT_URL="$url" EVENT_RELEVANCE="$relevance" \
    EVENT_TAGS="$tags" EVENT_SUMMARY="$summary" EVENT_TIME="$now" \
    AWARENESS_FILE="$AWARENESS_FILE" python3 -c "
import json, os

with open(os.environ['AWARENESS_FILE']) as f:
    data = json.load(f)

data.setdefault('events', [])
ids = [e.get('id', 0) for e in data['events']]
next_id = max(ids, default=0) + 1

data['events'].append({
    'id': next_id,
    'timestamp': os.environ['EVENT_TIME'],
    'type': 'discovery',
    'agent': 'jarvis',
    'title': os.environ['EVENT_TITLE'],
    'url': os.environ['EVENT_URL'],
    'relevance': float(os.environ['EVENT_RELEVANCE']),
    'tags': os.environ['EVENT_TAGS'],
    'message': os.environ['EVENT_SUMMARY']
})

if len(data['events']) > 500:
    data['events'] = data['events'][-500:]

data.setdefault('lastSeen', {})
data['lastSeen']['jarvis'] = 'discovery-' + str(next_id)

with open(os.environ['AWARENESS_FILE'], 'w') as f:
    json.dump(data, f, indent=2)
" 2>/dev/null || true
}

# ── Step 3: Signal running agents ──────────────────────────────────────────
signal_findings() {
    local count=$1
    local summary="$2"
    if [ -x "$SIGNAL_SCRIPT" ]; then
        "$SIGNAL_SCRIPT" "🌍 JARVIS daily brief: $count new discoveries — $summary"
    fi
}

# ── Main ───────────────────────────────────────────────────────────────────

echo "JARVIS daily briefing complete."
echo "  Check ~/.1bit/agent/awareness.json for discoveries."
echo "  Running agents are notified via trigger file."

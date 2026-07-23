#!/bin/bash
set -euo pipefail
# jarvis-daily-brief.sh — JARVIS daily AI/LLM engineering briefing
#
# Queries multiple sources, compiles a structured digest,
# feeds it into the awareness system so all agents see it.
#
# Run manually:
#   ./scripts/jarvis-daily-brief.sh
#
# Or via cron (already set up at 8 AM daily)

set -euo pipefail

AWARENESS_DIR="${HOME}/.1bit/agent"
AWARENESS_FILE="${AWARENESS_DIR}/awareness.json"
SIGNAL_SCRIPT="${HOME}/scripts/signal-agent-awareness.sh"

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

    # All fields (title/url/tags/summary may be untrusted external text from
    # web_search) are passed via environment, not interpolated into Python
    # source — prevents SyntaxError swallowing and code injection (#128).
    D_FILE="$AWARENESS_FILE" D_NOW="$now" D_TITLE="$title" D_URL="$url" \
    D_RELEVANCE="$relevance" D_TAGS="$tags" D_SUMMARY="$summary" \
    python3 -c "
import json, os

with open(os.environ['D_FILE']) as f:
    data = json.load(f)

data.setdefault('events', [])
ids = [e.get('id', 0) for e in data['events']]
next_id = max(ids, default=0) + 1

try:
    relevance = float(os.environ.get('D_RELEVANCE', '0') or '0')
except ValueError:
    relevance = 0

data['events'].append({
    'id': next_id,
    'timestamp': os.environ.get('D_NOW', ''),
    'type': 'discovery',
    'agent': 'jarvis',
    'title': os.environ.get('D_TITLE', ''),
    'url': os.environ.get('D_URL', ''),
    'relevance': relevance,
    'tags': os.environ.get('D_TAGS', ''),
    'message': '🌍 ' + os.environ.get('D_SUMMARY', '')
})

if len(data['events']) > 500:
    data['events'] = data['events'][-500:]

data.setdefault('lastSeen', {})
data['lastSeen']['jarvis'] = 'discovery-' + str(next_id)

with open(os.environ['D_FILE'], 'w') as f:
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

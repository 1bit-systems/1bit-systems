#!/bin/bash
# daily-brief.sh — JARVIS daily briefing for AI/LLM engineering
#
# Lightweight sweep that records a structured digest into the awareness system.
# Can be run via cron or manually.
#
# Usage:
#   ./daily-brief.sh                    # Record today's brief
#   ./daily-brief.sh --verbose          # Also print to stdout

set -e

AWARENESS_FILE="${HOME}/.1bit/agent/awareness.json"
SIGNAL_SCRIPT="${HOME}/scripts/signal-agent-awareness.sh"
_TRIGGER_FILE="/tmp/1bit-agent-awareness-trigger.txt"
VERBOSE=0
[ "$1" = "--verbose" ] && VERBOSE=1

DATE_STR="$(date -u '+%Y-%m-%d')"
TIMESTAMP="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

mkdir -p "$(dirname "$AWARENESS_FILE")"

if [ ! -f "$AWARENESS_FILE" ]; then
    printf '{"events":[],"lastSeen":{},"agents":{}}\n' > "$AWARENESS_FILE"
fi

if [ $VERBOSE -eq 1 ]; then
    echo "🦅 JARVIS Daily Brief — $DATE_STR"
    echo ""
    echo "This brief runs when JARVIS is activated in a 1bit session."
    echo "The agent uses web_search + fetch_content to gather discoveries."
    echo ""
    echo "Topics covered:"
    echo "  • LLM inference optimization (speculative decoding, quantization)"
    echo "  • MoE architecture advances (routing, expert caching)"
    echo "  • New model releases and tools"
    echo "  • GPU/NPU inference ecosystem"
    echo "  • Attention mechanism innovations"
    echo ""
fi

# Write a daily stub into awareness so agents know a brief exists
python3 -c "
import json, datetime

with open('$AWARENESS_FILE') as f:
    data = json.load(f)

data.setdefault('events', [])
ids = [e.get('id', 0) for e in data['events']]
next_id = max(ids, default=0) + 1

data['events'].append({
    'id': next_id,
    'timestamp': '$TIMESTAMP',
    'type': 'discovery',
    'agent': 'jarvis',
    'title': 'Daily Brief: $DATE_STR',
    'url': '',
    'relevance': 3,
    'tags': 'daily-brief',
    'message': '🌍 JARVIS daily engineering brief for $DATE_STR — run a sweep to get the latest discoveries'
})

if len(data['events']) > 500:
    data['events'] = data['events'][-500:]

with open('$AWARENESS_FILE', 'w') as f:
    json.dump(data, f, indent=2)
" 2>/dev/null || true

# Signal running agents
if [ -x "$SIGNAL_SCRIPT" ]; then
    "$SIGNAL_SCRIPT" "🌍 JARVIS daily brief for $DATE_STR ready — activates on next sweep"
fi

if [ $VERBOSE -eq 1 ]; then
    echo "✅ Brief recorded in awareness system."
    echo "   All agents will see it at session start."
fi

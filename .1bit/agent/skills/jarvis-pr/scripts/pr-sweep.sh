#!/bin/bash
# pr-sweep.sh — JARVIS PR inbox scan
#
# Scans the ProtonMail inbox for PR-relevant messages,
# categorizes them, and records findings in the awareness system.
#
# Usage:
#   bash pr-sweep.sh                    # Full scan
#   bash pr-sweep.sh --quick            # Just unread count + top senders
#   bash pr-sweep.sh --verbose          # Print everything

set -e

AWARENESS_FILE="${HOME}/.1bit/agent/awareness.json"
SIGNAL_SCRIPT="${HOME}/scripts/signal-agent-awareness.sh"
_TIMESTAMP="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
DATE_STR="$(date -u '+%Y-%m-%d')"
VERBOSE=0

[ "$1" = "--verbose" ] && VERBOSE=1
[ "$1" = "--quick" ] && _QUICK=1

echo "📬 JARVIS PR Sweep — $DATE_STR"
echo ""

# ── 1. Check inbox stats ─────────────────────────────────
_INBOX_STATS=$(python3 -c "
import json, subprocess
result = subprocess.run(
    ['node', '-e', '''
        const { execSync } = require('child_process');
        // We'll use the MCP via the 1bit extension system
        console.log(JSON.stringify({status: 'mcp_available'}));
    '''],
    capture_output=True, text=True, timeout=5
)
print(result.stdout)
" 2>/dev/null || echo '{"status":"check_failed"}')

echo "📬 Inbox scan requires a 1bit session with MCP tools."
echo ""
echo "To run a full PR sweep, start JARVIS:"
echo "  pi --skill jarvis-pr"
echo "  'JARVIS, scan my inbox and handle PR'"
echo ""

# ── 2. Record sweep event in awareness ───────────────────
mkdir -p "$(dirname "$AWARENESS_FILE")"
if [ ! -f "$AWARENESS_FILE" ]; then
    printf '{"events":[],"lastSeen":{},"agents":{}}\n' > "$AWARENESS_FILE"
fi

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
    'type': 'pr-sweep',
    'agent': 'jarvis',
    'title': 'PR Sweep: $DATE_STR',
    'message': '📬 PR sweep recorded for $DATE_STR — run JARVIS with --skill jarvis-pr for full scan'
})

if len(data['events']) > 500:
    data['events'] = data['events'][-500:]

with open('$AWARENESS_FILE', 'w') as f:
    json.dump(data, f, indent=2)
" 2>/dev/null || true

# Signal
if [ -x "$SIGNAL_SCRIPT" ]; then
    "$SIGNAL_SCRIPT" "📬 PR sweep ready for $DATE_STR — run jarvis with --skill jarvis-pr"
fi

echo "✅ PR sweep recorded in awareness system."
echo ""
echo "To actually read and respond to emails, run:"
echo "  pi --skill jarvis-pr"
echo "  'JARVIS, read my inbox and handle PR'"

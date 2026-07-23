#!/usr/bin/env bash
set -euo pipefail
# jarvis-daily-routine.sh — JARVIS complete daily routine
#
# Runs all JARVIS sweeps in sequence:
#   1. Analytics (GitHub + Cloudflare) — project metrics
#   2. PR brief — inbox scan (lightweight stub)
#   3. World brief — daily engineering digest
#
# All results feed into the awareness system.
#
# Usage:
#   ./scripts/jarvis-daily-routine.sh           # Full routine
#   ./scripts/jarvis-daily-routine.sh --quiet   # No stdout

set -euo pipefail

DATE_STR="$(date -u '+%Y-%m-%d')"
TIMESTAMP="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
AWARENESS_FILE="${HOME}/.1bit/agent/awareness.json"
SIGNAL_SCRIPT="${HOME}/scripts/signal-agent-awareness.sh"

echo "🦅 JARVIS Daily Routine — $DATE_STR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Step 1: Analytics ──────────────────────────────────
echo "📊 [1/3] Project Analytics..."
if [ -x "$HOME/scripts/jarvis-analytics-sweep.sh" ]; then
    bash "$HOME/scripts/jarvis-analytics-sweep.sh" 2>/dev/null && echo "  ✓ Analytics recorded" || echo "  ⚠ Analytics skipped"
fi
echo ""

# ── Step 2: World brief ─────────────────────────────────
echo "🌍 [2/3] World Engineering Brief..."
if [ -x "$HOME/.1bit/agent/skills/jarvis-world/scripts/daily-brief.sh" ]; then
    bash "$HOME/.1bit/agent/skills/jarvis-world/scripts/daily-brief.sh" 2>/dev/null && echo "  ✓ World brief recorded" || echo "  ⚠ World brief skipped"
fi
echo ""

# ── Step 3: PR brief ────────────────────────────────────
echo "📬 [3/3] PR Brief..."
if [ -x "$HOME/.1bit/agent/skills/jarvis-pr/scripts/pr-sweep.sh" ]; then
    bash "$HOME/.1bit/agent/skills/jarvis-pr/scripts/pr-sweep.sh" 2>/dev/null && echo "  ✓ PR sweep recorded" || echo "  ⚠ PR sweep skipped"
fi
echo ""

# ── Final: Summarize ────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ JARVIS daily routine complete."
echo ""
echo "To run a full interactive sweep with all tools:"
echo "  pi --skill jarvis-world --skill jarvis-pr"
echo "  'JARVIS, full daily routine'"
echo ""

# Record completion in awareness
mkdir -p "$(dirname "$AWARENESS_FILE")"
if [ ! -f "$AWARENESS_FILE" ]; then
    printf '{"events":[],"lastSeen":{},"agents":{}}\n' > "$AWARENESS_FILE"
fi

AWARENESS_FILE="$AWARENESS_FILE" TIMESTAMP="$TIMESTAMP" DATE_STR="$DATE_STR" \
python3 << 'PYEOF' 2>/dev/null || true
import json, os
with open(os.environ['AWARENESS_FILE']) as f:
    data = json.load(f)
data.setdefault('events', [])
ids = [e.get('id', 0) for e in data['events']]
next_id = max(ids, default=0) + 1
data['events'].append({
    'id': next_id,
    'timestamp': os.environ['TIMESTAMP'],
    'type': 'daily-routine',
    'agent': 'jarvis',
    'title': f"Daily Routine: {os.environ['DATE_STR']}",
    'message': f"🦅 JARVIS daily routine complete for {os.environ['DATE_STR']}"
})
if len(data['events']) > 500:
    data['events'] = data['events'][-500:]
with open(os.environ['AWARENESS_FILE'], 'w') as f:
    json.dump(data, f, indent=2)
PYEOF

if [ -x "$SIGNAL_SCRIPT" ]; then
    "$SIGNAL_SCRIPT" "🦅 JARVIS daily routine complete for $DATE_STR"
fi

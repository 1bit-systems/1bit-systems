#!/usr/bin/env bash
set -euo pipefail
# signal-agent-awareness.sh — Signal all running agents about a codebase change
#
# Use this in CI pipelines, post-push hooks, or any external process
# that wants running agents to know about changes.
#
# Usage:
#   ./signal-agent-awareness.sh "message describing what changed"
#   ./signal-agent-awareness.sh "Agent A pushed CCA kernel fix to main"
#
# This writes to:
#   1. The shared awareness.json (for session-start awareness)
#   2. The trigger file (for real-time awareness in running sessions)

set -euo pipefail

MESSAGE="${1:-Codebase was updated}"
RECORD_SCRIPT="${HOME}/scripts/record-agent-change.sh"
TRIGGER_FILE="/tmp/1bit-agent-awareness-trigger.txt"

# 1. Record the change if we're in a git repo
if [ -d ".git" ] || git rev-parse --git-dir >/dev/null 2>&1; then
    BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
    if [ -x "$RECORD_SCRIPT" ]; then
        "$RECORD_SCRIPT" commit "$BRANCH"
    fi
fi

# 2. Signal running agents via trigger file
#    The git-awareness extension watches this file via fs.watch
echo "$MESSAGE" > "$TRIGGER_FILE"

echo "✅ Awareness signal sent: $MESSAGE"

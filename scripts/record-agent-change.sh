#!/bin/bash
# record-agent-change.sh — Log git changes to the cross-agent awareness file.
#
# Called from git hooks (post-commit, post-merge, post-checkout, pre-push).
# Writes structured events to .1bit/agent/awareness.json so any agent
# starting a session can see what changed since it last ran.
#
# Usage:
#   ./record-agent-change.sh commit [branch]
#   ./record-agent-change.sh merge  [branch]
#   ./record-agent-change.sh checkout <old-ref> <new-ref>
#
# Environment (optional):
#   GIT_AUTHOR_NAME / GIT_COMMITTER_NAME — identifies the agent/user

set -e

AWARENESS_DIR="${HOME}/.1bit/agent"
AWARENESS_FILE="${AWARENESS_DIR}/awareness.json"
AWARENESS_PAYLOAD="${AWARENESS_DIR}/.awareness-payload.json"
MAX_EVENTS=200

ensure_file() {
    mkdir -p "$AWARENESS_DIR"
    if [ ! -f "$AWARENESS_FILE" ]; then
        printf '{"events":[],"lastSeen":{},"agents":{}}\n' > "$AWARENESS_FILE"
    fi
}

get_agent_name() {
    local name
    name="${GIT_AUTHOR_NAME:-${GIT_COMMITTER_NAME:-}}"
    if [ -z "$name" ] || [ "$name" = "bcloud" ]; then
        name="$(git config user.name 2>/dev/null || echo "unknown")"
    fi
    echo "$name"
}

# Write payload as JSON to temp file, then have python3 merge it into awareness
# This avoids all shell escaping issues with commit messages, filenames, etc.
write_payload() {
    local type="$1"
    local agent_name
    agent_name="$(get_agent_name)"
    shift
    # Pass ALL untrusted values (type, agent name, and the key=value args) as
    # process arguments / environment, never interpolated into Python source.
    # A commit author name (or filename/message) containing a quote or
    # backslash previously broke the literal or allowed code injection (#128).
    PAYLOAD_TYPE="$type" \
    PAYLOAD_AGENT="$agent_name" \
    PAYLOAD_OUT="$AWARENESS_PAYLOAD" \
    python3 -c "
import json, os, sys
from datetime import datetime, timezone

payload = {
    'type': os.environ['PAYLOAD_TYPE'],
    'agent': os.environ['PAYLOAD_AGENT'],
    'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
}
for arg in sys.argv[1:]:
    k, _, v = arg.partition('=')
    payload[k] = v

with open(os.environ['PAYLOAD_OUT'], 'w') as f:
    json.dump(payload, f)
" "$@" 2>/dev/null || true
}

# Main merge step: read awareness.json and payload.json, merge, write back
merge_awareness() {
    if [ ! -f "$AWARENESS_PAYLOAD" ]; then
        return 0
    fi

    ensure_file

    python3 -c "
import json, sys

# Read current awareness
with open('$AWARENESS_FILE') as f:
    data = json.load(f)

# Read payload
with open('$AWARENESS_PAYLOAD') as f:
    payload = json.load(f)

data.setdefault('events', [])
data.setdefault('lastSeen', {})
data.setdefault('agents', {})

payload_type = payload.get('type', 'commit')

# Dedup for commits
if payload_type == 'commit':
    commit_hash = payload.get('commit', '')
    for e in data['events']:
        if e.get('commit') == commit_hash and e.get('type') == 'commit':
            sys.exit(0)  # duplicate, skip

# Generate monotonic ID
ids = [e.get('id', 0) for e in data['events']]
next_id = max(ids, default=0) + 1

event = {
    'id': next_id,
    'timestamp': payload.get('timestamp', ''),
    'type': payload_type,
    'agent': payload.get('agent', 'unknown'),
    'branch': payload.get('branch', 'unknown'),
    'commit': payload.get('commit', ''),
    'message': payload.get('message', ''),
}

# Handle files (stored as comma-separated string in payload, convert to list)
files_str = payload.get('files', '')
if files_str:
    event['files'] = [f for f in files_str.split(',') if f]

# Handle checkout-specific fields
if payload_type == 'checkout':
    event['oldBranch'] = payload.get('oldBranch', '')

data['events'].append(event)

# Trim to MAX_EVENTS
data['events'] = data['events'][-${MAX_EVENTS}:]

# Update lastSeen and agents for commit events
if payload_type == 'commit':
    agent = payload.get('agent', 'unknown')
    commit_hash = payload.get('commit', '')
    branch = payload.get('branch', 'unknown')
    ts = payload.get('timestamp', '')
    data['lastSeen'][agent] = commit_hash
    data['agents'][agent] = {
        'lastCommit': commit_hash,
        'lastSeen': ts,
        'branch': branch
    }

with open('$AWARENESS_FILE', 'w') as f:
    json.dump(data, f)

# Clean up payload
import os
os.remove('$AWARENESS_PAYLOAD')
" 2>/dev/null || true
}

record_commit() {
    local branch="${1:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')}"
    local commit_hash commit_msg files agent_name

    commit_hash="$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
    [ "$commit_hash" = "unknown" ] && return 0

    commit_msg="$(git log -1 --format='%s' 2>/dev/null || echo 'unknown')"
    files="$(git diff-tree --no-commit-id --name-only -r HEAD 2>/dev/null | head -20 | paste -sd ',' - || echo '')"

    write_payload "commit" \
        "branch=$branch" \
        "commit=$commit_hash" \
        "message=$commit_msg" \
        "files=$files"

    merge_awareness
}

record_merge() {
    local branch="${1:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')}"
    local msg
    msg="$(git log -1 --format='%s' 2>/dev/null || echo 'merged')"

    if echo "$msg" | grep -qi "merge.*main\|main.*merge"; then
        record_commit "$branch"
    fi
}

record_checkout() {
    local old_ref="$1"
    local new_ref="$2"
    local flag="${3:-0}"
    [ "$flag" != "1" ] && return 0

    local new_branch
    new_branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
    [ "$new_branch" = "$old_ref" ] && return 0

    local commit_hash
    commit_hash="$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"

    write_payload "checkout" \
        "branch=$new_branch" \
        "commit=$commit_hash" \
        "oldBranch=$old_ref" \
        "message=Switched to $new_branch from $old_ref"

    merge_awareness
}

case "${1:-}" in
    commit)  record_commit "${2:-}" ;;
    merge)   record_merge "${2:-}" ;;
    checkout) record_checkout "${2:-}" "${3:-}" "${4:-}" ;;
    *)
        echo "Usage: $0 {commit|merge|checkout} [args...]" >&2
        exit 1
        ;;
esac

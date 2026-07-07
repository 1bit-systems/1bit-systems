#!/usr/bin/env bash
set -euo pipefail

# Auto-Commits: displays recent git commit history with full context
# Usage: bash commits.sh [--full|--count N|--since DATE|--watch]

COUNT=10
FULL=false
SINCE=""
WATCH=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full) FULL=true; COUNT=30 ;;
    --count) COUNT="$2"; shift ;;
    --since) SINCE="$2"; shift ;;
    --watch) WATCH=true ;;
    *) echo "Usage: $0 [--full|--count N|--since DATE|--watch]"; exit 1 ;;
  esac
  shift
done

# Detect git repo
if ! ROOT=$(git rev-parse --show-toplevel 2>/dev/null); then
  echo "Not in a git repository."
  exit 1
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "detached")
SINCE_ARG=""
if [[ -n "$SINCE" ]]; then
  SINCE_ARG="--since=\"$SINCE\""
fi

echo ""
echo "═══ Auto-Commits: $ROOT ═══"
echo "Branch: $BRANCH" $(if [[ -n "$SINCE" ]]; then echo "| Since: $SINCE"; fi)
echo ""

# Uncommitted changes
if [[ -n "$(git status --short 2>/dev/null)" ]]; then
  echo "⚠ Uncommitted changes:"
  git status --short
  echo ""
fi

# Recent contributors
echo "Recent Contributors (last $COUNT commits):"
git shortlog -sne -"$COUNT" 2>/dev/null | while read -r commits author; do
  bar=$(printf '%*s' "$commits" | tr ' ' '█')
  echo "  $author  $bar  $commits commits"
done
echo ""

# Most active files
echo "Most Active Files (last $COUNT commits):"
git log --oneline --name-only -"$COUNT" 2>/dev/null \
  | grep -v '^[0-9a-f]\{7\}' \
  | grep -v '^$' \
  | sort \
  | uniq -c \
  | sort -rn \
  | head -10 \
  | while read -r count file; do
    bar=$(printf '%*s' "$((count * 2))" | tr ' ' '▌')
    echo "  $file  $bar  $count changes"
  done
echo ""

# Commit log
echo "Last $COUNT Commits:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if $FULL; then
  # Detailed view with stats
  git log -"$COUNT" --format="%H" 2>/dev/null | while read -r sha; do
    if [[ -z "$sha" ]]; then continue; fi
    short=$(echo "$sha" | cut -c1-7)
    date=$(git log -1 --format="%ai" "$sha" 2>/dev/null | cut -d' ' -f1)
    author=$(git log -1 --format="%an" "$sha" 2>/dev/null)
    subject=$(git log -1 --format="%s" "$sha" 2>/dev/null)
    body=$(git log -1 --format="%b" "$sha" 2>/dev/null | head -10)
    stats=$(git diff --stat "$sha^..$sha" 2>/dev/null | tail -1 || echo "0 files changed")

    echo ""
    echo "$short  $date  $author"
    echo "  $subject"
    if [[ -n "$body" ]]; then
      echo "$body" | sed 's/^/    /'
    fi
    echo "  ── $stats"
  done
else
  # Compact view
  git log -"$COUNT" --format="%h  %as  %an
  %s" --stat 2>/dev/null | awk '
    /^[0-9a-f]{7}/ {
      if (sha) print ""
      sha = $1; date = $2; rest = substr($0, index($0,$3))
      print sha, date, rest
      next
    }
    / file.*changed/ { print "  ── " $0; next }
    /^$/ { next }
    /^ / { next }
    { print "  " $0 }
  '
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Watch mode
if $WATCH; then
  echo "Watching for new commits... (Ctrl+C to stop)"
  while true; do
    if command -v inotifywait &>/dev/null; then
      inotifywait -q -e modify "$ROOT/.git/refs/heads/$BRANCH" 2>/dev/null || sleep 30
    else
      sleep 30
    fi
    clear
    "$0" --count "$COUNT" 2>/dev/null || true
  done
fi

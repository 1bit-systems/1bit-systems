---
name: auto-commits
description: Automatically loads recent git commit history with full details — message, author, date, files changed, diffs. Use before making changes to understand the recent trajectory of the codebase, or when the user hasn't provided context about what's changed recently. For commit-level impact analysis, pair with the gitnexus skills.
---

# Auto-Commits — Automatic Commit Context

When this skill activates, it automatically fetches and presents the
recent git commit history for the current repository. This gives you
full context about what's been changing, who changed it, and why —
before you touch any code.

## How It Works

1. Detects the current git repository (`git rev-parse --show-toplevel`)
2. Fetches the last 30 commits with full details
3. Summarizes: recent files touched, active contributors, commit cadence
4. Shows the diffstat for each commit (files changed, insertions, deletions)
5. Presents everything in a structured format

## Usage

```bash
# Quick summary (default: last 10 commits, compact)
bash ~/.pi/agent/skills/auto-commits/commits.sh

# Full detail: 30 commits with diffs
bash ~/.pi/agent/skills/auto-commits/commits.sh --full

# Custom number of commits
bash ~/.pi/agent/skills/auto-commits/commits.sh --count 50

# Since a specific date
bash ~/.pi/agent/skills/auto-commits/commits.sh --since "2026-07-01"

# Watch mode — check for new commits every 30s
bash ~/.pi/agent/skills/auto-commits/commits.sh --watch
```

## Auto-Load Behavior

On activation (when working in a git repo), this skill auto-runs:

```
# In the current repo, this runs automatically:
#   - Checks if git repo
#   - Fetches last 30 commits
#   - Reports recent activity summary
```

## Output Format

```
═══ Auto-Commits: /home/bcloud/1bit ═══
Branch: main | Since: 2026-07-01

Recent Contributors:
  admin       ████████████████  15 commits
  dependabot  ██                 2 commits

Most Active Files:
  engine/npu/src/npu_engine_cb.cpp    8 changes
  docs/wiki/performance.md            5 changes
  engine/fusion/dispatcher.zig        3 changes

Last 10 Commits:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
abc1234  2026-07-06  admin
  feat(npu): M=32 batched dispatch amortization
  ── 3 files changed, +84 -12

abc1233  2026-07-06  admin
  fix(fusion): KV cache sync race condition
  ── 2 files changed, +22 -8
...
```

## Pairing with GitNexus

For deeper analysis of specific commits:

| Need | Tool |
|------|------|
| "What flows does this commit affect?" | `gitnexus_detect_changes(...)` |
| "Who depends on this changed symbol?" | `gitnexus_impact(...)` |
| "Trace this commit's logic" | `gitnexus_query({query: "..."})` |

## Notes

- Works in any git repo (no GitHub dependency)
- --watch mode uses `inotifywait` if available, otherwise polls every 30s
- Respects `.gitignore` — only tracks tracked files
- Shows your uncommitted changes if any (`git status --short`)

# Reddit Scraper & PR Agent Context Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI tool (`tools/reddit_scraper.py`) that fetches Reddit posts + comments from a given URL, stores the raw JSON locally, and can re-fetch to check for new replies. Also expose a GitHub Actions workflow that runs it on schedule and commits new replies to the repo, so PR agents can reference community feedback in their reviews.

**Architecture:** Pure Python tool using PRAW (Reddit API) to fetch post content and comments. Stores a session JSON (`.reddit_sessions/reddit_post_comments.json`) with post metadata + flattened comments. On re-fetch, merges new comments (detected by ID). Optionally outputs a markdown summary that gets committed so PR agents can reference community context.

**Tech Stack:** Python 3.14, PRAW 8, GitHub Actions cron, bash/git for commit + push

---
## Global Constraints

- Every task produces independently testable output
- The tool must work with or without a Reddit API client ID/secret (fallback to scraping HTML)
- PRAW credentials configured via env vars: `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`
- Store session data in `.reddit_sessions/` directory at repo root
- Output formats: JSON (full data) and Markdown (summary)
- All steps are 100% complete code — no placeholders, no TBDs

---

### Task 1: PRAW Configuration Module

**Files:**
- Create: `tools/reddit_scraper.py` (start with the config/auth base)

**Interfaces:**
- Produces: `get_reddit_instance()` — returns `praw.Reddit` or `None` (if no credentials)
- Produces: `REDDIT_SESSION_DIR = ".reddit_sessions"` constant

- [ ] **Step 1: Create the initial scraper module with auth logic**

```python
#!/usr/bin/env python3
"""
Reddit scraper — fetch posts + comments for PR agent context.

Usage:
  python3 tools/reddit_scraper.py fetch <url> [--output ./reddit_session.json]
  python3 tools/reddit_scraper.py recheck <session_path> [--output ./reddit_session.json]
  python3 tools/reddit_scraper.py summary <session_path> [--format markdown|json]

Environment:
  REDDIT_CLIENT_ID     — Reddit app client ID (optional)
  REDDIT_CLIENT_SECRET — Reddit app client secret (optional)
  REDDIT_USER_AGENT    — User agent string for PRAW (optional)
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REDDIT_SESSION_DIR = Path(".reddit_sessions")


def get_reddit_instance():
    """Create a PRAW Reddit instance from env vars, or return None."""
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = os.environ.get(
        "REDDIT_USER_AGENT",
        "linux:npu-gpu-cpu-pr-agent:v1.0 (by /u/bong-water-water-bong)",
    )
    if not client_id or not client_secret:
        return None
    try:
        import praw
        return praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )
    except ImportError:
        return None


def parse_reddit_url(url: str):
    """Extract subreddit, post_id from a Reddit URL."""
    m = re.search(r'/r/(\w+)/comments/(\w+)', url)
    if not m:
        raise ValueError(f"Could not parse Reddit URL: {url}")
    return m.group(1), m.group(2)
```

- [ ] **Step 2: Write argparse CLI entry point**

```python
def main():
    parser = argparse.ArgumentParser(description="Reddit scraper for PR agent context")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # fetch
    fetch_parser = subparsers.add_parser("fetch", help="Fetch a Reddit post + comments")
    fetch_parser.add_argument("url", help="Reddit post URL")
    fetch_parser.add_argument("--output", "-o", type=Path, help="Output path (default: .reddit_sessions/<post_id>.json)")

    # recheck
    recheck_parser = subparsers.add_parser("recheck", help="Re-fetch a previously saved session for new comments")
    recheck_parser.add_argument("session", type=Path, help="Path to existing session JSON")
    recheck_parser.add_argument("--output", "-o", type=Path, help="Output path (default: overwrite session)")

    # summary
    summary_parser = subparsers.add_parser("summary", help="Generate a summary from a session")
    summary_parser.add_argument("session", type=Path, help="Path to session JSON")
    summary_parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    summary_parser.add_argument("--output", "-o", type=Path, help="Output path (default: stdout)")

    args = parser.parse_args()

    if args.command == "fetch":
        do_fetch(args.url, args.output)
    elif args.command == "recheck":
        do_recheck(args.session, args.output)
    elif args.command == "summary":
        do_summary(args.session, args.format, args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
cd /home/bcloud/npu-gpu-cpu
git add tools/reddit_scraper.py
git commit -m "feat(reddit-scraper): add PRAW auth + CLI skeleton"
```

---

### Task 2: Fetch & Comment Extraction

**Files:**
- Modify: `tools/reddit_scraper.py`

**Interfaces:**
- Consumes: `get_reddit_instance()`, `parse_reddit_url()` from Task 1
- Produces: `do_fetch(url, output_path)` — fetches post + all comments, saves JSON
- Produces: `_extract_comments(submission)` — recursive comment flattening
- Produces: Session JSON schema (see below)

**Session JSON Schema:**
```json
{
  "meta": {
    "url": "https://...",
    "title": "Post title",
    "subreddit": "StrixHalo",
    "post_id": "1uitnkr",
    "author": "username",
    "score": 28,
    "upvote_ratio": 0.95,
    "created_utc": 1719680000,
    "num_comments": 5,
    "fetched_at": "2026-06-29T17:30:00Z"
  },
  "comments": [
    {
      "id": "t1_abc123",
      "parent_id": null,
      "author": "user1",
      "score": 10,
      "body": "Comment text",
      "created_utc": 1719680100,
      "depth": 0,
      "replies": [
        {"id": "t1_def456", "parent_id": "t1_abc123", ...}
      ]
    }
  ],
  "comments_flat": [
    {"id": "t1_abc123", "author": "user1", "score": 10, "body": "...", "depth": 0},
    {"id": "t1_def456", ...}
  ]
}
```

- [ ] **Step 1: Implement do_fetch() — fetch post via PRAW or HTML scraping**

```python
import requests
from html.parser import HTMLParser


def do_fetch(url: str, output_path: Path = None):
    subreddit, post_id = parse_reddit_url(url)
    reddit = get_reddit_instance()

    if reddit:
        print(f"[reddit-scraper] Using PRAW API to fetch {url}")
        submission = reddit.submission(id=post_id)
        # Ensure comments are loaded
        submission.comments.replace_more(limit=None)
        session = _build_session_from_praw(submission, url)
    else:
        print(f"[reddit-scraper] No PRAW credentials. Falling back to HTML scrape of {url}")
        session = _build_session_from_html(url, post_id, subreddit)

    if output_path is None:
        REDDIT_SESSION_DIR.mkdir(parents=True, exist_ok=True)
        output_path = REDDIT_SESSION_DIR / f"{post_id}.json"

    output_path.write_text(json.dumps(session, indent=2, default=str))
    print(f"[reddit-scraper] Saved session to {output_path}")
    print(f"[reddit-scraper]   Title: {session['meta']['title']}")
    print(f"[reddit-scraper]   Comments: {session['meta']['num_comments']}")
    return session


def _build_session_from_praw(submission, url: str) -> dict:
    meta = {
        "url": url,
        "title": submission.title,
        "subreddit": str(submission.subreddit),
        "post_id": submission.id,
        "author": str(submission.author) if submission.author else "[deleted]",
        "score": submission.score,
        "upvote_ratio": getattr(submission, "upvote_ratio", None),
        "created_utc": submission.created_utc,
        "num_comments": submission.num_comments,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "selftext": getattr(submission, "selftext", "")[:2000],
    }
    comments, comments_flat = _extract_comments_praw(submission.comments)
    return {
        "meta": meta,
        "comments": comments,
        "comments_flat": comments_flat,
        "comment_ids": {c["id"] for c in comments_flat},
    }


def _extract_comments_praw(comments_list, depth=0):
    result = []
    flat = []
    for comment in comments_list:
        if isinstance(comment, praw.models.MoreComments):
            continue
        entry = {
            "id": comment.id,
            "parent_id": str(comment.parent_id) if comment.parent_id else None,
            "author": str(comment.author) if comment.author else "[deleted]",
            "score": comment.score,
            "body": comment.body,
            "created_utc": comment.created_utc,
            "depth": depth,
            "is_submitter": comment.is_submitter if hasattr(comment, "is_submitter") else False,
        }
        replies, reply_flat = _extract_comments_praw(comment.replies, depth + 1)
        if replies:
            entry["replies"] = replies
        result.append(entry)
        flat.append(entry)
        flat.extend(reply_flat)
    return result, flat
```

- [ ] **Step 2: Write the HTML fallback parser**

```python
class _CommentHTMLParser(HTMLParser):
    """Minimal parser for old.reddit.com comment pages."""
    def __init__(self):
        super().__init__()
        self.in_md = False
        self.in_author = False
        self.in_score = False
        self.current = {}
        self.comments = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        classes = attrs_dict.get("class", "")

        if "usertext-body" in classes:
            self.in_md = True
            self.current = {}
        if "author" in classes:
            self.in_author = True
        if "score" in classes:
            self.in_score = True

    def handle_endtag(self, tag):
        if tag == "div" and self.in_md:
            self.in_md = False
            if self.current.get("body"):
                self.comments.append(self.current)
            self.current = {}

    def handle_data(self, data):
        if self.in_author:
            self.current["author"] = data.strip()
            self.in_author = False
        if self.in_score:
            self.current["score"] = data.strip()
            self.in_score = False
        if self.in_md:
            self.current["body"] = self.current.get("body", "") + data.strip() + " "


def _build_session_from_html(url: str, post_id: str, subreddit: str) -> dict:
    """Fallback: scrape old.reddit.com HTML for comments."""
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"}
    html_url = f"https://old.reddit.com/r/{subreddit}/comments/{post_id}/"
    resp = requests.get(html_url, headers=headers, timeout=30)
    resp.raise_for_status()

    # Try extracting title
    title_match = re.search(r'<title>(.*?)</title>', resp.text, re.DOTALL)
    title = title_match.group(1).replace(" - Reddit", "").strip() if title_match else "(unknown)"

    # Extract comment body text via regex
    comment_texts = re.findall(
        r'<div class="md"><p>(.*?)</p></div>',
        resp.text,
        re.DOTALL,
    )

    # Extract usernames
    authors = re.findall(
        r'<a[^>]*class="author[^"]*"[^>]*>(\w+)</a>',
        resp.text,
    )

    # Build flat comments
    comments_flat = []
    seen_ids = set()
    for i, text in enumerate(comment_texts):
        body = re.sub(r'<[^>]+>', ' ', text)
        body = re.sub(r'\s+', ' ', body).strip()
        if not body or len(body) < 20 or "TL;DR" in body or "A place for" in body:
            continue
        comment_id = f"html_{i}"
        if comment_id in seen_ids:
            continue
        seen_ids.add(comment_id)
        author = authors[i] if i < len(authors) else "[unknown]"
        comments_flat.append({
            "id": comment_id,
            "author": author,
            "body": body,
            "score": None,
            "depth": 0,
            "created_utc": None,
        })

    meta = {
        "url": url,
        "title": title,
        "subreddit": subreddit,
        "post_id": post_id,
        "author": "(scraped)",
        "score": None,
        "upvote_ratio": None,
        "created_utc": None,
        "num_comments": len(comments_flat),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "selftext": "(scraped)",
        "scraped": True,
    }

    return {
        "meta": meta,
        "comments": comments_flat,
        "comments_flat": comments_flat,
        "comment_ids": {c["id"] for c in comments_flat},
    }
```

- [ ] **Step 3: Implement do_recheck() — merge new comments**

```python
def do_recheck(session_path: Path, output_path: Path = None):
    """Re-fetch a post and merge any new comments."""
    if not session_path.exists():
        print(f"[reddit-scraper] Session not found: {session_path}", file=sys.stderr)
        sys.exit(1)

    old_session = json.loads(session_path.read_text())
    url = old_session["meta"]["url"]

    print(f"[reddit-scraper] Re-checking {url}")
    new_session = do_fetch(url)

    # Compare comment IDs
    old_ids = set(old_session.get("comment_ids", []))
    new_ids = set(new_session.get("comment_ids", []))

    added = new_ids - old_ids
    removed = old_ids - new_ids

    if not added and not removed:
        print(f"[reddit-scraper] No new comments (still {len(old_ids)} total)")
    else:
        print(f"[reddit-scraper] Changes detected:")
        print(f"  +{len(added)} new comments")
        print(f"  -{len(removed)} removed/deleted comments")

    # Merge: keep old comments, add new ones
    old_comments = {c["id"] for c in old_session.get("comments_flat", [])}
    new_comments = new_session.get("comments_flat", [])
    merged_flat = list(old_session.get("comments_flat", []))
    seen = old_comments.copy()
    for c in new_comments:
        if c["id"] not in seen:
            merged_flat.append(c)
            seen.add(c["id"])

    merged = {
        "meta": new_session["meta"],
        "comments": merged_flat,
        "comments_flat": merged_flat,
        "comment_ids": list(seen),
        "recheck_history": {
            "previous_fetch": old_session["meta"]["fetched_at"],
            "current_fetch": new_session["meta"]["fetched_at"],
            "comments_added": len(added),
            "comments_removed": len(removed),
        },
    }

    if output_path is None:
        output_path = session_path
    output_path.write_text(json.dumps(merged, indent=2, default=str))
    print(f"[reddit-scraper] Updated session saved to {output_path}")
    return merged
```

- [ ] **Step 4: Implement do_summary() — markdown output**

```python
def do_summary(session_path: Path, fmt: str = "markdown", output_path: Path = None):
    session = json.loads(session_path.read_text())
    meta = session["meta"]
    comments = session.get("comments_flat", [])

    if fmt == "json":
        output = json.dumps(session, indent=2, default=str)
    else:
        lines = []
        lines.append(f"# Reddit Post: {meta.get('title', '(no title)')}")
        lines.append(f"")
        lines.append(f"- **Subreddit:** r/{meta.get('subreddit', '?')}")
        lines.append(f"- **Author:** u/{meta.get('author', '?')}")
        lines.append(f"- **Score:** {meta.get('score', '?')}")
        if meta.get("upvote_ratio"):
            lines.append(f"- **Upvote ratio:** {meta['upvote_ratio']}")
        lines.append(f"- **Comments:** {meta.get('num_comments', len(comments))}")
        lines.append(f"- **Fetched:** {meta.get('fetched_at', '?')}")
        lines.append(f"- **URL:** {meta.get('url', '?')}")
        lines.append(f"")

        if meta.get("selftext"):
            lines.append(f"## Post Body")
            lines.append(f"")
            lines.append(f"{meta['selftext'][:500]}")
            lines.append(f"")

        lines.append(f"## Comments ({len(comments)})")
        lines.append(f"")

        for c in sorted(comments, key=lambda x: x.get("score") or 0, reverse=True):
            indent = "  " * c.get("depth", 0)
            author = c.get("author", "[deleted]")
            score = c.get("score")
            score_str = f"({score} pts)" if score is not None else ""
            body = c.get("body", "")[:500]
            submitter_tag = " [OP]" if c.get("is_submitter") else ""
            lines.append(f"{indent}- **u/{author}**{submitter_tag} {score_str}: {body}")
            lines.append(f"")

        output = "\n".join(lines)

    if output_path:
        output_path.write_text(output)
        print(f"[reddit-scraper] Summary saved to {output_path}")
    else:
        print(output)

    return output
```

- [ ] **Step 5: Make sure imports are correct at the top of reddit_scraper.py**

Ensure the final file has all imports at the top:

```python
import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    import praw
    HAS_PRAW = True
except ImportError:
    HAS_PRAW = False
```

- [ ] **Step 6: Test the tool end-to-end**

```bash
cd /home/bcloud/npu-gpu-cpu
python3 tools/reddit_scraper.py fetch "https://www.reddit.com/r/StrixHalo/comments/1uitnkr/we_ran_qwen306b_on_the_strix_halo_npu_at_48_toks/"
```

Expected output:
```
[reddit-scraper] Using PRAW API to fetch https://www.reddit.com/r/StrixHalo/comments/1uitnkr...
[reddit-scraper] Saved session to .reddit_sessions/1uitnkr.json
[reddit-scraper]   Title: We ran Qwen3-0.6B on the Strix Halo NPU at 4.8 tok/s...
[reddit-scraper]   Comments: 5
```

- [ ] **Step 7: Generate markdown summary**

```bash
cd /home/bcloud/npu-gpu-cpu
python3 tools/reddit_scraper.py summary .reddit_sessions/1uitnkr.json --format markdown --output docs/REDDIT_COMMENTS.md
cat docs/REDDIT_COMMENTS.md
```

Expected: A markdown file with the post metadata + all comments.

- [ ] **Step 8: Recheck workflow**

```bash
cd /home/bcloud/npu-gpu-cpu
python3 tools/reddit_scraper.py recheck .reddit_sessions/1uitnkr.json
```

Expected: "No new comments" if nothing changed, or merge new ones.

- [ ] **Step 9: Commit**

```bash
cd /home/bcloud/npu-gpu-cpu
git add tools/reddit_scraper.py .reddit_sessions/ docs/REDDIT_COMMENTS.md
git commit -m "feat(reddit-scraper): implement fetch/recheck/summary with PRAW + HTML fallback"
```

---

### Task 3: GitHub Actions Workflow — Scheduled Reddit Check

**Files:**
- Create: `.github/workflows/reddit-check.yml`

**Interfaces:**
- Consumes: `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` secrets
- Consumes: `GITHUB_TOKEN` (built-in) for committing results
- Produces: Check data every 6 hours, commit new comments to repo

- [ ] **Step 1: Create the workflow**

```yaml
name: Reddit Check

on:
  schedule:
    # Every 6 hours
    - cron: '0 */6 * * *'
  workflow_dispatch:
    inputs:
      force:
        description: 'Force re-fetch all posts'
        required: false
        default: 'false'

jobs:
  check-reddit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.14'

      - name: Install dependencies
        run: |
          pip install praw requests

      - name: Fetch/Recheck Reddit posts
        env:
          REDDIT_CLIENT_ID: ${{ secrets.REDDIT_CLIENT_ID }}
          REDDIT_CLIENT_SECRET: ${{ secrets.REDDIT_CLIENT_SECRET }}
          REDDIT_USER_AGENT: ${{ secrets.REDDIT_USER_AGENT }}
        run: |
          python3 tools/reddit_scraper.py fetch "https://www.reddit.com/r/StrixHalo/comments/1uitnkr/we_ran_qwen306b_on_the_strix_halo_npu_at_48_toks/"
          python3 tools/reddit_scraper.py summary .reddit_sessions/1uitnkr.json --format markdown --output docs/REDDIT_COMMENTS.md

      - name: Commit changes
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add .reddit_sessions/ docs/REDDIT_COMMENTS.md
          if git diff --cached --quiet; then
            echo "No new comments — skipping commit"
          else
            git commit -m "chore(reddit): update Reddit comments feed [skip ci]"
            git push
          fi
```

- [ ] **Step 2: Commit**

```bash
cd /home/bcloud/npu-gpu-cpu
git add .github/workflows/reddit-check.yml
git commit -m "ci(reddit): add scheduled workflow to fetch Reddit comments every 6 hours"
```

---

### Task 4: Wire Reddit Context Into PR Agent Config

**Files:**
- Modify: `.pr_agent.toml` (add reference to `docs/REDDIT_COMMENTS.md` in extra_instructions)

**Interfaces:**
- Consumes: `docs/REDDIT_COMMENTS.md` (from Task 2)
- Modifies: `extra_instructions` in `[pr_reviewer]` section

- [ ] **Step 1: Add a note in extra_instructions telling the PR agent to check the Reddit context file**

Edit `.pr_agent.toml` and add near the top of `[pr_reviewer]` extra_instructions:

```
## Community Context
This project has been discussed on Reddit at r/StrixHalo. The file
docs/REDDIT_COMMENTS.md contains the latest comments from that thread,
including technical feedback. Before reviewing a PR, check that file
for any community-reported issues or suggestions relevant to the code
being changed. In particular, look for:
- INT8 DMA stride alignment advice (AIE2 DMA addresses in 32-bit words)
- BF16 Chess compiler workarounds (Peano/LLVM-AIE backend suggestion)
- Any other community-leveraged bug reports or fixes
```

- [ ] **Step 2: Commit**

```bash
cd /home/bcloud/npu-gpu-cpu
git add .pr_agent.toml
git commit -m "feat(pr-agent): wire Reddit community context into PR review instructions"
```

---

### Task 5: Push Everything

- [ ] **Step 1: Push to origin**

```bash
cd /home/bcloud/npu-gpu-cpu
git push origin main
```

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
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    import praw
    HAS_PRAW = True
except ImportError:
    HAS_PRAW = False

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


def do_fetch(url: str, output_path: Path = None):
    """Fetch a Reddit post and its comments."""
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
    """Build session dict from a PRAW submission object."""
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
        "comment_ids": sorted({c["id"] for c in comments_flat}),
    }


def _extract_comments_praw(comments_list, depth=0):
    """Recursively flatten PRAW comment tree."""
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


def _build_session_from_html(url: str, post_id: str, subreddit: str) -> dict:
    """Fallback: scrape old.reddit.com HTML for comments.

    Parses the old.reddit.com page by iterating over each data-fullname anchor
    and extracting author, score, and body text from a fixed-size window
    following the anchor. t3_ fullnames are the post; t1_ are comments.
    """
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"}
    html_url = f"https://old.reddit.com/r/{subreddit}/comments/{post_id}/"
    resp = requests.get(html_url, headers=headers, timeout=30)
    resp.raise_for_status()
    html = resp.text.replace('&#32;', ' ')  # normalize HTML-encoded spaces

    # Extract title
    title_match = re.search(r'<title>(.*?)</title>', html, re.DOTALL)
    title = title_match.group(1).replace(" - Reddit", "").strip() if title_match else "(unknown)"

    comments_flat = []
    post_author = "(scraped)"
    post_body = ""
    post_score = None

    # Iterate over each fullname anchor and extract data from a window after it
    for m in re.finditer(r'data-fullname="(t\d+_[a-z0-9]+)"', html):
        fullname = m.group(1)
        # Take a 5000-char window starting from the fullname
        chunk = html[m.start():m.start() + 12000]

        # Extract author from data-author attribute (available on all thing divs)
        author_m = re.search(r'data-author="([^"]+)"', chunk)
        author = author_m.group(1) if author_m else "[unknown]"

        # Extract score from data-score attribute
        score_m = re.search(r'data-score="(\d+)"', chunk)
        score = int(score_m.group(1)) if score_m else None

        # Extract body text from the first <div class="md"> inside this chunk
        md_m = re.search(r'<div class="md"[^>]*>(.*?)</div>', chunk, re.DOTALL)
        body = ""
        if md_m:
            text = re.sub(r'<[^>]+>', ' ', md_m.group(1))
            text = re.sub(r'\s+', ' ', text).strip()
            body = text

        # Skip sidebar/sustainability noise
        skip_patterns = [
            "A place for news, tips, comments",
            "Posts are automatically archived",
            "Check out our Discord",
            "Community settings",
        ]
        if any(p in body for p in skip_patterns):
            continue
        if len(body) < 10:
            continue

        if fullname.startswith("t3_"):  # post
            post_author = author
            post_body = body
            post_score = score
        elif fullname.startswith("t1_"):  # comment
            comments_flat.append({
                "id": fullname,
                "author": author,
                "body": body,
                "score": score,
                "depth": 0,
                "created_utc": None,
            })

    meta = {
        "url": url,
        "title": title,
        "subreddit": subreddit,
        "post_id": post_id,
        "author": post_author,
        "score": post_score,
        "upvote_ratio": None,
        "created_utc": None,
        "num_comments": len(comments_flat),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "selftext": post_body[:2000] if post_body else "(scraped)",
        "scraped": True,
    }

    return {
        "meta": meta,
        "comments": comments_flat,
        "comments_flat": comments_flat,
        "comment_ids": sorted({c["id"] for c in comments_flat}),
    }


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
        "comment_ids": sorted(seen),
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


def do_summary(session_path: Path, fmt: str = "markdown", output_path: Path = None):
    """Generate a summary from a session."""
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

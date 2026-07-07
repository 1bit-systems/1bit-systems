---
name: reddit-scraper
description: Scrape Reddit posts, comments, and subreddit listings without API keys using old.reddit.com HTML parsing. Use when you need Reddit content not available via API (the .json endpoint was shut down May 2026). Includes Python functions and bash one-liner.
---

# Reddit Scraper Skill

Scrape Reddit posts, comments, and subreddit listings without API keys.
Uses `old.reddit.com` HTML parsing (server-rendered, stable DOM).
The `.json` API was shut down May 2026 (returns 403).

## Usage

### Fetch a single post with comments

```python
import requests
from bs4 import BeautifulSoup

def fetch_post(url):
    """Fetch a Reddit post from old.reddit.com with all comments."""
    # Ensure old.reddit.com URL
    url = url.replace("www.reddit.com", "old.reddit.com")
    url = url.replace("reddit.com", "old.reddit.com")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    
    soup = BeautifulSoup(resp.text, "lxml")
    
    # Post content
    post = {}
    title_el = soup.select_one("a.title")
    if title_el:
        post["title"] = title_el.get_text(strip=True)
    
    user_el = soup.select_one("p.tagline a.author")
    if user_el:
        post["author"] = user_el.get_text(strip=True)
    
    score_el = soup.select_one("div.score.unvoted")
    if score_el:
        post["score"] = score_el.get("title", score_el.get_text(strip=True))
    
    body_el = soup.select_one("div.usertext-body div.md")
    if body_el:
        post["body"] = body_el.get_text(strip=True)
    
    # Comments
    post["comments"] = []
    for comment in soup.select("div.comment"):
        comment_data = {}
        c_user = comment.select_one("a.author")
        if c_user:
            comment_data["author"] = c_user.get_text(strip=True)
        c_body = comment.select_one("div.usertext-body div.md")
        if c_body:
            comment_data["body"] = c_body.get_text(strip=True)
        c_score = comment.select_one("span.score-unvoted")
        if c_score:
            comment_data["score"] = c_score.get_text(strip=True)
        if comment_data:
            post["comments"].append(comment_data)
    
    return post
```

### Fetch subreddit listings

```python
def fetch_subreddit(subreddit, sort="hot", limit=25):
    """Fetch subreddit posts. sort: hot, new, top, controversial."""
    url = f"https://old.reddit.com/r/{subreddit}/{sort}/"
    headers = {"User-Agent": "Mozilla/5.0 ..."}
    
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    
    soup = BeautifulSoup(resp.text, "lxml")
    posts = []
    for thing in soup.select("div.thing"):
        post = {}
        title_el = thing.select_one("a.title")
        if title_el:
            post["title"] = title_el.get_text(strip=True)
            post["permalink"] = title_el.get("href", "")
        post["score"] = thing.get("data-score", "")
        post["author"] = thing.get("data-author", "")
        post["comments"] = thing.get("data-comments-count", "")
        post["url"] = thing.get("data-url", "")
        posts.append(post)
    return posts
```

## Bash One-Liner (quick check)

```bash
# Fetch a post's raw HTML (use w3m/lynx for text extraction)
curl -s -A "Mozilla/5.0" "https://old.reddit.com/r/StrixHalo/comments/1uitnkr/we_ran_qwen306b_on_the_strix_halo_npu_at_48_toks/" | w3m -dump -T text/html -O utf8 | head -100
```

## Notes

- Rate limit: ~1 request per 2 seconds
- Always use a real browser User-Agent
- `old.reddit.com` returns fully server-rendered HTML — no JS needed
- The `div.thing` selector grabs posts; `div.comment` selector grabs comments
- For RSS: `https://www.reddit.com/r/subreddit/.rss` still works (XML)
- For structured data without scraping: use PRAW (requires OAuth app registration at reddit.com/prefs/apps)
- The `.json` endpoint has been blocked since May 2026 — do not rely on it

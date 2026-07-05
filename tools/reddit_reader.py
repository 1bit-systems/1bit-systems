#!/usr/bin/env python3
"""
Reddit reader — scrapes old.reddit.com with real browser headers.
Bypasses Cloudflare/blocking by using curl with proper User-Agent + cookies.

Usage:
    ./reddit_reader.py search "query" [--limit 5]
    ./reddit_reader.py subreddit "LocalLLaMA" [--sort hot|new|top]
"""

import subprocess, json, sys, re, html
from urllib.parse import quote

USER_AGENTS = [
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]

COOKIE = "reddit_session=; _ga=GA1.2.0; _gid=GA1.2.0;"

def curl_get(url, timeout=15):
    """Fetch URL using curl with real browser headers."""
    cmd = [
        "curl", "-s", "-L",
        "-A", USER_AGENTS[0],
        "-H", "Accept: text/html,application/json,*/*",
        "-H", "Accept-Language: en-US,en;q=0.9",
        "-H", "Cache-Control: no-cache",
        "-H", "Pragma: no-cache",
        "-H", f"Cookie: {COOKIE}",
        "--connect-timeout", "10",
        "--max-time", str(timeout),
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout+5)
    if result.returncode != 0 or not result.stdout.strip():
        # Retry with different UA
        cmd[3] = USER_AGENTS[1]
        cmd[5] = "text/html,application/xhtml+xml"
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout+5)
    return result.stdout

def parse_json_from_html(html_text):
    """Extract JSON from old.reddit HTML page."""
    # Try to find JSON embedded in script tags
    matches = re.findall(r'window\.__r\s*=\s*({.*?});', html_text, re.DOTALL)
    if matches:
        return json.loads(matches[0])
    return None

def parse_listing(html_text):
    """Parse old.reddit listing page for posts."""
    posts = []
    # Find post entries
    entries = re.findall(
        r'<div class="[^"]*entry[^"]*"[^>]*>(.*?)</div>\s*</div>\s*</div>',
        html_text, re.DOTALL
    )
    
    if not entries:
        # Try alternate parsing — find thing elements
        things = re.findall(
            r'<div[^>]*class="[^"]*thing[^"]*"[^>]*id="thing_(t3_\w+)"[^>]*>(.*?)</div>\s*<!--\s*/\s*thing',
            html_text, re.DOTALL
        )
        for tid, content in things:
            post = parse_thing(tid, content)
            if post:
                posts.append(post)
    
    return posts

def parse_thing(thing_id, content):
    """Parse a single thing element."""
    # Extract title
    title_match = re.search(r'<a[^>]*class="[^"]*title[^"]*"[^>]*>(.*?)</a>', content, re.DOTALL)
    if not title_match:
        return None
    
    title = html.unescape(re.sub(r'<[^>]+>', '', title_match.group(1)).strip())
    
    # Extract score
    score_match = re.search(r'<div[^>]*class="[^"]*score[^"]*"[^>]*unvoted[^"]*"[^>]*>(.*?)</div>', content)
    score = score_match.group(1).strip() if score_match else "?"
    
    # Extract comments
    comments_match = re.search(r'<a[^>]*>(\d+)\s*comments?</a>', content)
    comments = comments_match.group(1) if comments_match else "0"
    
    # Extract subreddit
    subreddit_match = re.search(r'href="/r/(\w+)"', content)
    subreddit = subreddit_match.group(1) if subreddit_match else "?"
    
    # Extract URL
    url_match = re.search(r'<a[^>]*class="[^"]*title[^"]*"[^>]*href="([^"]*)"', content)
    url = url_match.group(1) if url_match else ""
    if url and not url.startswith('http'):
        url = 'https://old.reddit.com' + url
    
    # Extract time
    time_match = re.search(r'time[^>]*title="([^"]*)"', content)
    time = time_match.group(1) if time_match else "?"
    
    return {
        'id': thing_id,
        'title': title,
        'score': score,
        'comments': comments,
        'subreddit': subreddit,
        'url': url,
        'time': time,
    }

def search(query, limit=5):
    """Search Reddit (HTML path — JSON API is blocked)."""
    url = f"https://old.reddit.com/search?q={quote(query)}&limit={limit}&sort=new&t=all"
    html_data = curl_get(url)
    posts = parse_listing(html_data)
    if not posts:
        # Try alternate page structure
        posts = parse_alt_listing(html_data)
    return posts[:limit]

def parse_alt_listing(html_text):
    """Alternative parser for old.reddit.com structure."""
    posts = []
    for m in re.finditer(r'<a[^>]*class="[^"]*search-title[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html_text, re.DOTALL):
        href, title_raw = m.group(1), m.group(2)
        title = html.unescape(re.sub(r'<[^>]+>', '', title_raw).strip())
        url = href if href.startswith('http') else 'https://old.reddit.com' + href
        # Try to extract subreddit
        sub = '?'
        sm = re.search(r'/r/(\w+)', href)
        if sm:
            sub = sm.group(1)
        posts.append({
            'id': '', 'title': title, 'score': '?',
            'comments': '?', 'subreddit': sub, 'url': url, 'time': '',
        })
    return posts

def get_subreddit(subreddit, sort='hot', limit=5):
    """Get posts from a subreddit (HTML path)."""
    url = f"https://old.reddit.com/r/{subreddit}/{sort}/"
    html_data = curl_get(url)
    posts = parse_listing(html_data)
    if not posts:
        posts = parse_alt_listing(html_data)
    return posts[:limit]

def get_post(post_id):
    """Get a single post by ID."""
    url = f"https://old.reddit.com/comments/{post_id}.json"
    data = curl_get(url)
    try:
        j = json.loads(data)
        if j and len(j) > 0:
            post_data = j[0]['data']['children'][0]['data']
            comments = []
            if len(j) > 1:
                for c in j[1]['data']['children']:
                    if c['kind'] == 't1':
                        cd = c['data']
                        comments.append({
                            'author': cd.get('author', '[deleted]'),
                            'body': cd.get('body', '')[:500],
                            'score': cd.get('ups', 0),
                            'time': cd.get('created_utc', 0),
                        })
            return {
                'title': post_data.get('title', ''),
                'score': post_data.get('ups', 0),
                'comments_count': post_data.get('num_comments', 0),
                'selftext': post_data.get('selftext', '')[:1000],
                'url': post_data.get('url', ''),
                'comments': comments,
            }
    except:
        pass
    return None

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  reddit_reader.py search <query> [--limit N]")
        print("  reddit_reader.py subreddit <name> [--sort hot|new|top] [--limit N]")
        print("  reddit_reader.py post <post_id>")
        sys.exit(1)

    cmd = sys.argv[1]
    args = sys.argv[2:]

    limit = 5
    sort = 'hot'
    i = 0
    while i < len(args):
        if args[i] == '--limit' and i+1 < len(args):
            limit = int(args[i+1])
            i += 2
        elif args[i] == '--sort' and i+1 < len(args):
            sort = args[i+1]
            i += 2
        else:
            i += 1

    if cmd == 'search':
        query = ' '.join([a for a in args if not a.startswith('--')])
        posts = search(query, limit)
        for p in posts:
            print(f"r/{p['subreddit']} | ▲{p['score']} | {p['comments']}cmts")
            print(f"  {p['title']}")
            print(f"  {p['url']}")
            print()

    elif cmd == 'subreddit':
        sub = args[0] if args else 'LocalLLaMA'
        posts = get_subreddit(sub, sort, limit)
        for p in posts:
            print(f"▲{p['score']} | {p['comments']}cmts")
            print(f"  {p['title']}")
            print(f"  {p['url']}")
            print()

    elif cmd == 'post':
        pid = args[0] if args else ''
        post = get_post(pid)
        if post:
            print(f"▲{post['score']} | {post['comments_count']}cmts")
            print(f"  {post['title']}")
            print(f"  {post['selftext'][:500]}")
            print()
            for c in post['comments'][:5]:
                print(f"  [{c['score']}] {c['author']}: {c['body'][:200]}")
                print()

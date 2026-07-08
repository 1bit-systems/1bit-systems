---
name: youtube-transcripts
description: Fetch YouTube video transcripts (captions) without API keys using yt-dlp or the innertube API. Use when you need video transcripts, timestamps, or want to extract text content from YouTube videos.
---

# YouTube Transcripts Skill

Fetch YouTube video transcripts (captions/subtitles) without API keys.
Two methods: `yt-dlp` (preferred, supports auto-generated captions) and raw innertube API (Python, no deps beyond `requests`).

## Usage

### Method 1: yt-dlp (recommended)

```bash
# Fetch transcript as plain text
yt-dlp --skip-download --write-auto-subs --convert-subs srt --output "%(id)s" "https://www.youtube.com/watch?v=VIDEO_ID"
# Then extract text from the .srt file

# One-liner: fetch + strip timestamps
yt-dlp --skip-download --write-auto-subs --convert-subs vtt --output - "https://www.youtube.com/watch?v=VIDEO_ID" 2>/dev/null | \
  sed 's/<[^>]*>//g' | grep -v '^$' | grep -v 'WEBVTT' | grep -v '^[0-9:]*\.[0-9]* -->' | \
  awk '!seen[$0]++'

# Or output to file
yt-dlp --skip-download --write-auto-subs --convert-subs srt --output "%(title)s.%(ext)s" "URL"
```

### Method 2: Python (innertube API, no yt-dlp needed)

```python
import requests
import re
import json

def get_transcript(video_id, lang="en"):
    """
    Fetch transcript for a YouTube video using the innertube API.
    video_id: YouTube video ID (the v= parameter)
    lang: language code (en, es, fr, etc.)
    Returns: list of {text, start, duration} dicts, or None
    """
    # Step 1: Get the video page to extract captions data
    url = f"https://www.youtube.com/watch?v={video_id}"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    }
    
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    
    # Extract ytInitialPlayerResponse JSON
    match = re.search(r'ytInitialPlayerResponse\s*=\s*({.+?});', resp.text)
    if not match:
        raise ValueError("Could not find player response")
    
    player_response = json.loads(match.group(1))
    
    # Navigate to captions
    captions = (
        player_response.get("captions", {})
        .get("playerCaptionsTracklistRenderer", {})
        .get("captionTracks", [])
    )
    
    if not captions:
        return None
    
    # Find the right language track
    track = None
    for t in captions:
        if t.get("languageCode") == lang:
            track = t
            break
    if not track and captions:
        track = captions[0]  # fallback to first available
    
    # Some tracks have a URL, others use signature
    base_url = track.get("baseUrl", "")
    if not base_url:
        raise ValueError("No transcript URL found")
    
    # Fetch the actual transcript XML
    transcript_resp = requests.get(base_url, headers=headers, timeout=15)
    transcript_resp.raise_for_status()
    
    # Parse the XML
    from xml.etree import ElementTree
    root = ElementTree.fromstring(transcript_resp.text)
    
    segments = []
    for text_el in root.findall(".//text"):
        segments.append({
            "text": text_el.text or "",
            "start": float(text_el.get("start", 0)),
            "duration": float(text_el.get("dur", 0)),
        })
    
    return segments


def transcript_to_text(segments):
    """Convert transcript segments to plain text."""
    return " ".join(s["text"] for s in segments)


def transcript_to_srt(segments):
    """Convert transcript segments to SRT format."""
    lines = []
    for i, s in enumerate(segments, 1):
        start_ts = format_timestamp(s["start"])
        end_ts = format_timestamp(s["start"] + s["duration"])
        lines.append(f"{i}\n{start_ts} --> {end_ts}\n{s['text']}\n")
    return "\n".join(lines)


def format_timestamp(seconds):
    """Convert seconds to SRT timestamp HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")
```

### Bash one-liner (video info + available captions)

```bash
# List available caption tracks
yt-dlp --list-subs "https://www.youtube.com/watch?v=VIDEO_ID"

# Download auto-generated English captions as plain text
yt-dlp --skip-download --write-auto-subs --sub-lang en --convert-subs srt -o - "URL" 2>/dev/null | sed '/^$/d'
```

## Notes

- yt-dlp supports auto-generated captions from YouTube's speech recognition
- The innertube API method works without yt-dlp but only returns manually uploaded captions (not auto-generated)
- Rate limit: ~1 request per 2-3 seconds for innertube API
- Always use a real browser User-Agent
- For videos without any captions, yt-dlp's `--write-auto-subs` will fail — no workaround exists
- Timestamps are in seconds from video start

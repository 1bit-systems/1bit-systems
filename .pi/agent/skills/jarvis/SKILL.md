---
name: jarvis
description: JARVIS / 1bit.systems theme and tools — applies matching TUI themes (1bit green, JARVIS pink), provides server endpoints (JARVIS, NPU, Fused engine), and CLI shortcuts. Use when working with JARVIS, the NPU engine, or the 1bit.systems project.
---

# JARVIS / 1bit.systems Theme & Tools

When JARVIS or 1bit.systems work is detected, this skill activates the matching TUI theme.

## Theme Commands

- `theme 1bit` — Apply the 1bit.systems dark deck theme (green primary, teal-black bg)
- `theme jarvis` — Apply the JARVIS theme (pink primary, matrix accents)
- `theme dark` — Back to default dark

## Available Tools & Context

- **JARVIS server**: `http://localhost:8080` — Web UI at `/chat`, API at `/api/status`
- **NPU engine**: `http://127.0.0.1:9090` — fused layer daemon, 291 tok/s (production)
- **NPU fallback**: `http://127.0.0.1:52625` — FLM daemon, 94 tok/s (fallback v2)
- **CLI**: `/home/bcloud/jarvis/jarvis` — Start/stop/chat/demo/mobile

## Reddit Watch (r/StrixHalo Monitor)

JARVIS continuously monitors r/StrixHalo for mentions of the 1bit.systems
project. New finds are pushed into JARVIS's knowledge base automatically.

```bash
# Check what Reddit Watch has found
curl -s http://localhost:8080/api/reddit-watch | python3 -m json.tool

# One-shot manual check (pushes to JARVIS KB)
/home/bcloud/jarvis/venv/bin/python /home/bcloud/jarvis/scripts/reddit-watch.py --to-jarvis

# Watch logs
tail -f /home/bcloud/.local/share/reddit-watch/watch.log

# Service status
systemctl --user status reddit-watch.service
```

The watcher polls every 5 minutes and stores state at
`~/.local/share/reddit-watch/state.json`. It follows posts already seen,
so you only get notified about new ones.

## Theme Colors

### 1bit theme (default)
- Background: `#021621` (deep teal-black)
- Primary: `#00ff00` (matrix green)
- Accent: `#f00fd2` (hot pink)
- Secondary: `#12a0ed` (cyan-blue)
- Text: `#e7f6fd` (bright cyan-white)

### JARVIS theme (alternative)
- Same background, pink accent as primary
- Green as border accent
- Blue for secondary elements

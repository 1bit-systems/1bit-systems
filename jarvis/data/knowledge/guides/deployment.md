---
type: Guide
title: Deployment
description: Deploy JARVIS to Cloudflare Pages (web UI) or as a self-hosted service on the 1bit.systems stack.
tags: [deployment, cloudflare, hosting]
timestamp: 2026-07-06T00:00:00Z
---

# Deployment

## Self-Hosted (Default)

Just run the server:

```bash
python3 server/server.py
```

JARVIS listens on `0.0.0.0:8080` by default. Access from any device on your LAN.

## Cloudflare Pages (Web UI Only)

The static web UI can be deployed to Cloudflare Pages:

```bash
cd ~/jarvis
./deploy.sh
```

The `wrangler.toml` configures the deployment. Note: the API server must still run locally or on a reachable host.

## Subdomain Setup

```bash
cd ~/jarvis
./scripts/setup-subdomain.sh
```

## Production Considerations

- Run behind a reverse proxy (nginx/Caddy) for TLS
- Set `JARVIS_DEBUG=0` in production
- Consider systemd service for auto-restart
- TTS voices downloaded via API: `POST /api/tts/download`

## Citations

[1] [Quick Start](/guides/quick_start.md)
[2] [Configuration](/capabilities/configuration.md)

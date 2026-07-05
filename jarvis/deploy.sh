#!/usr/bin/env bash
# Deploy jarvis.1bit.systems to Cloudflare Pages
set -e

echo "=== Deploying JARVIS site to Cloudflare Pages ==="

# Build step: copy the landing page
mkdir -p /tmp/jarvis-deploy
cp -r /home/bcloud/1bit-site/jarvis/* /tmp/jarvis-deploy/
cp /home/bcloud/1bit-site/assets/favicon.svg /tmp/jarvis-deploy/ 2>/dev/null || true

# Deploy
cd /tmp/jarvis-deploy
npx wrangler pages deploy . --project-name=jarvis --branch=main

echo "=== Done ==="
echo "Set custom domain: jarvis.1bit.systems in Cloudflare Dashboard"

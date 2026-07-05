#!/usr/bin/env bash
# Set up jarvis.1bit.systems subdomain
# Run this after getting a Cloudflare API token with DNS:Edit permissions

set -e

echo "=== Setting up jarvis.1bit.systems ==="
echo ""

# Your Cloudflare API token with DNS:Edit and Pages:Write permissions
read -sp "Cloudflare API token (with DNS + Pages perms): " CF_TOKEN
echo ""

# Zone ID for 1bit.systems
ZONE_ID="fb8da40a864cd0c59bf39b689024624e"
ACCOUNT_ID="687e774105ecfa11892e7faba5b362e6"

# 1. Add CNAME record
echo "1. Adding CNAME record..."
curl -s -X POST "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records" \
  -H "Authorization: Bearer $CF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "CNAME",
    "name": "jarvis",
    "content": "jarvis-1mp.pages.dev",
    "ttl": 1,
    "proxied": true
  }' | python3 -c 'import sys,json;d=json.load(sys.stdin);print("✅" if d["success"] else "❌", d.get("errors",[{}])[0].get("message",""))'

# 2. Verify domain in Pages project  
echo "2. Verifying Pages domain..."
curl -s "https://api.cloudflare.com/client/v4/accounts/$ACCOUNT_ID/pages/projects/jarvis/domains" \
  -H "Authorization: Bearer $CF_TOKEN" | python3 -c '
import sys,json
d=json.load(sys.stdin)
for dom in d.get("result",[]):
    print(f"  {dom[\"name\"]}: {dom[\"status\"]}")
'

echo ""
echo "=== Done ==="
echo "Access JARVIS at: https://jarvis.1bit.systems"
echo "May take up to 5 minutes for DNS + SSL to provision."

#!/bin/bash
set -euo pipefail
# jarvis-analytics-sweep.sh — JARVIS analytics pull
#
# Calls GitHub + Cloudflare analytics, extracts key metrics,
# records them in the awareness system.
#
# Usage: ./scripts/jarvis-analytics-sweep.sh

set -euo pipefail

ANALYTICS_DIR="${HOME}/.1bit/agent/skills/analytics"
AWARENESS_FILE="${HOME}/.1bit/agent/awareness.json"
SIGNAL_SCRIPT="${HOME}/scripts/signal-agent-awareness.sh"
TIMESTAMP="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
DATE_STR="$(date -u '+%Y-%m-%d')"

# Export so python3 heredoc can read them
export TIMESTAMP DATE_STR AWARENESS_FILE GH_TMP CF_TMP

mkdir -p "$(dirname "$AWARENESS_FILE")" "$ANALYTICS_DIR"
echo "📊 JARVIS Analytics Sweep — $DATE_STR"

# Run analytics to temp files (strip ANSI codes)
GH_TMP=$(mktemp /tmp/jarvis-gh-output.XXXXXX) || exit 1
CF_TMP=$(mktemp /tmp/jarvis-cf-output.XXXXXX) || exit 1
trap 'rm -f "$GH_TMP" "$CF_TMP"' EXIT
bash "$ANALYTICS_DIR/github-analytics.sh" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' > "$GH_TMP" || echo "" > "$GH_TMP"
bash "$ANALYTICS_DIR/cf-analytics.sh" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' > "$CF_TMP" || echo "" > "$CF_TMP"

# Extract and record via python3 (no shell escaping issues)
python3 << 'PYEOF'
import json, re, os

AWARENESS_FILE = os.environ.get('AWARENESS_FILE', os.path.expanduser('~/.1bit/agent/awareness.json'))
TIMESTAMP = os.environ.get('TIMESTAMP', '')
DATE_STR = os.environ.get('DATE_STR', '')

def extract_from(path, patterns):
    try:
        with open(path) as f:
            text = f.read()
    except:
        return {k: '0' for k in patterns}
    result = {}
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        result[key] = m.group(1).replace(',','') if m else '0'
    return result

# GitHub
gh_tmp = os.environ.get('GH_TMP', '/tmp/jarvis-gh-output.txt')
gh = extract_from(gh_tmp, {
    'stars': r'Stars:\s*([0-9,]+)',
    'forks': r'Forks:\s*([0-9,]+)',
    'clones_total': r'Clones:\s*([0-9,]+)',
})
# unique counts
gh_text = open(gh_tmp).read()
vhits = list(re.finditer(r'unique\s*([0-9,]+)', gh_text))
gh['clones_unique'] = vhits[0].group(1).replace(',','') if len(vhits) >= 1 else '0'
gh['views_unique'] = vhits[1].group(1).replace(',','') if len(vhits) >= 2 else '0'
# views total
vm = re.search(r'Page Views:\s*([0-9,]+)', gh_text)
gh['views_total'] = vm.group(1).replace(',','') if vm else '0'

# Cloudflare
cf_tmp = os.environ.get('CF_TMP', '/tmp/jarvis-cf-output.txt')
cf = extract_from(cf_tmp, {
    'page_views': r'Page Views:\s*([0-9,]+)',
    'visitors': r'Unique Visitors:\s*([0-9,]+)',
    'requests': r'Total Requests:\s*([0-9,]+)',
    'bandwidth': r'Bandwidth:\s*([0-9.]+ ?[A-Z]*[Bb]?)',
    'cached': r'Cached:\s*([0-9.]+%)',
})

entry = {
    'id': 0,
    'timestamp': TIMESTAMP,
    'type': 'analytics',
    'agent': 'jarvis',
    'title': f'Analytics: {DATE_STR}',
    'github': {
        'stars': int(gh.get('stars', 0)),
        'forks': int(gh.get('forks', 0)),
        'clones_total': int(gh.get('clones_total', 0)),
        'clones_unique': int(gh.get('clones_unique', 0)),
        'views_total': int(gh.get('views_total', 0)),
        'views_unique': int(gh.get('views_unique', 0)),
    },
    'cloudflare': {
        'page_views': int(cf.get('page_views', 0)),
        'unique_visitors': int(cf.get('visitors', 0)),
        'requests': int(cf.get('requests', 0)),
        'bandwidth': cf.get('bandwidth', '0'),
        'cached': cf.get('cached', '0%'),
    },
    'message': f'📊 Analytics {DATE_STR} — ⭐{gh.get("stars","?")} ★ 👁{gh.get("views_total","?")} views ★ 🌐{cf.get("page_views","?")} page views'
}

# Read existing awareness
try:
    with open(AWARENESS_FILE) as f:
        data = json.load(f)
except:
    data = {'events': [], 'lastSeen': {}, 'agents': {}}

data.setdefault('events', [])
ids = [e.get('id', 0) for e in data['events']]
entry['id'] = max(ids, default=0) + 1
data['events'].append(entry)

if len(data['events']) > 500:
    data['events'] = data['events'][-500:]
data.setdefault('lastSeen', {})
data['lastSeen']['jarvis-analytics'] = f'analytics-{entry["id"]}'

with open(AWARENESS_FILE, 'w') as f:
    json.dump(data, f, indent=2)

# Print summary
g = entry['github']
c = entry['cloudflare']
print(f'  ✅ Stars: {g["stars"]}  Views: {g["views_total"]}  Clones: {g["clones_total"]}')
print(f'     CF: {c["page_views"]} page views, {c["unique_visitors"]} visitors, {c["bandwidth"]} bw, {c["cached"]} cached')
print(f'     Message: {entry["message"]}')
PYEOF

# Signal
SUMMARY="$(python3 -c "
import json, os
with open(os.environ.get('AWARENESS_FILE', '')) as f:
    d = json.load(f)
e = [x for x in d['events'] if x.get('type')=='analytics'][-1]
print(f'⭐{e[\"github\"][\"stars\"]} ★ 👁{e[\"github\"][\"views_total\"]} views ★ 🌐{e[\"cloudflare\"][\"page_views\"]} page views')
" 2>/dev/null || echo 'analytics recorded')"

if [ -x "$SIGNAL_SCRIPT" ]; then
    "$SIGNAL_SCRIPT" "📊 Analytics: $SUMMARY"
fi

# Cleanup
rm -f "$GH_TMP" "$CF_TMP"

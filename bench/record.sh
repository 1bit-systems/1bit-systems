#!/bin/bash
# bench/record.sh — Record a benchmark result to site/benchmarks.json
#
# Usage: bench/record.sh <engine_key> <tok_s> [tflops] [status] [label] [table] [display_name] [backend]
#
# Appends/replaces the entry for <engine_key> in site/benchmarks.json and
# updates the "updated" and "commit" fields. Creates the file if absent.
set -euo pipefail

ENGINE_KEY="${1:?usage: bench/record.sh engine_key tok_s [tflops] [status] [label] [table] [display_name] [backend]}"
TOK_S="${2:-null}"
TFLOPS="${3:-null}"
STATUS="${4:-validated}"
LABEL="${5:-}"
TABLE="${6:-kernel}"
DISPLAY_NAME="${7:-$ENGINE_KEY}"
BACKEND="${8:-}"

COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
FILE="site/benchmarks.json"

# Create or update
if [ ! -f "$FILE" ]; then
  cat > "$FILE" <<-JSON
{
 "updated": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
 "commit": "$COMMIT",
 "hardware": { "cpu": "AMD Ryzen AI Max+ 395", "npu": "XDNA 2", "gpu": "Radeon 8060S" },
 "engines": {}
}
JSON
fi

python3 -c "
import json, sys
with open('$FILE') as f:
    d = json.load(f)
d['updated'] = '$(date -u +%Y-%m-%dT%H:%M:%SZ)'
d['commit'] = '$COMMIT'
d.setdefault('hardware', {})
d['hardware'].setdefault('cpu', 'AMD Ryzen AI Max+ 395')
d['hardware'].setdefault('npu', 'XDNA 2')
d['hardware'].setdefault('gpu', 'Radeon 8060S')
entry = {'tok_s': $TOK_S, 'tflops': $TFLOPS, 'status': '$STATUS', 'label': '$LABEL', 'table': '$TABLE', 'display_name': '$DISPLAY_NAME', 'backend': '$BACKEND'}
# Remove null values
entry = {k: v for k, v in entry.items() if v is not None and v != '' and v != 'null'}
d['engines']['$ENGINE_KEY'] = entry
with open('$FILE', 'w') as f:
    json.dump(d, f, indent=1)
    f.write('\n')
print(f'Recorded {ENGINE_KEY}: {TOK_S} tok/s')
" 2>&1

# Also update badge JSONs for shields.io
for badge_file in site/badge_gpu.json site/badge_npu.json; do
  key="$(basename "$badge_file" .json | sed 's/badge_//')"
  # Map engine_key to badge file
  case "$ENGINE_KEY" in
    q1_gemv|fused_tq2|ternary|tq2_gemv|rocm_hip|gpu_zinc|prefill_i8)
      TARGET="site/badge_gpu.json";;
    npu_v12|npu_flm)
      TARGET="site/badge_npu.json";;
    *)
      TARGET="";;
  esac
  if [ -n "$TARGET" ] && [ "$badge_file" = "$TARGET" ]; then
    MSG="${TOK_S} tok/s"
    python3 -c "
import json
with open('$badge_file') as f:
    b = json.load(f)
b['message'] = '$MSG'
b['updated'] = '$(date -u +%Y-%m-%dT%H:%M:%SZ)'
with open('$badge_file', 'w') as f:
    json.dump(b, f, indent=1)
    f.write('\n')
" 2>/dev/null || true
  fi
done

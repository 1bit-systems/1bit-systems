#!/bin/bash
set -euo pipefail
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

# Validate numeric inputs to prevent code injection
validate_numeric() {
    local val="$1" name="$2"
    if [ "$val" = "null" ] || [ "$val" = "" ]; then
        return 0
    fi
    if ! echo "$val" | grep -qE '^[0-9]+\.?[0-9]*$'; then
        echo "ERROR: $name must be a numeric value, got: '$val'" >&2
        exit 1
    fi
}
validate_numeric "$TOK_S" "TOK_S"
validate_numeric "$TFLOPS" "TFLOPS"

export TOK_S TFLOPS STATUS LABEL TABLE DISPLAY_NAME BACKEND ENGINE_KEY COMMIT FILE
python3 << 'PYEOF'
import json, os

tok_s_val = os.environ.get('TOK_S', 'null')
tflops_val = os.environ.get('TFLOPS', 'null')
status_val = os.environ.get('STATUS', 'validated')
label_val = os.environ.get('LABEL', '')
table_val = os.environ.get('TABLE', 'kernel')
display_val = os.environ.get('DISPLAY_NAME', os.environ.get('ENGINE_KEY', ''))
backend_val = os.environ.get('BACKEND', '')
engine_key = os.environ.get('ENGINE_KEY', '')
commit_val = os.environ.get('COMMIT', 'unknown')
file_path = os.environ.get('FILE', 'site/benchmarks.json')

with open(file_path) as f:
    d = json.load(f)
d['updated'] = os.popen('date -u +%Y-%m-%dT%H:%M:%SZ').read().strip()
d['commit'] = commit_val
d.setdefault('hardware', {})
d['hardware'].setdefault('cpu', 'AMD Ryzen AI Max+ 395')
d['hardware'].setdefault('npu', 'XDNA 2')
d['hardware'].setdefault('gpu', 'Radeon 8060S')
entry = {
    'tok_s': float(tok_s_val) if tok_s_val not in ('null', '') else None,
    'tflops': float(tflops_val) if tflops_val not in ('null', '') else None,
    'status': status_val,
    'label': label_val,
    'table': table_val,
    'display_name': display_val,
    'backend': backend_val,
}
# Remove None/empty values
entry = {k: v for k, v in entry.items() if v is not None and v != '' and v != 'null'}
d['engines'][engine_key] = entry
with open(file_path, 'w') as f:
    json.dump(d, f, indent=1)
    f.write('\n')
print(f"Recorded {engine_key}: {tok_s_val} tok/s")
PYEOF

# Also update badge JSONs for shields.io
for badge_file in site/badge_gpu.json site/badge_npu.json; do
  # shellcheck disable=SC2034
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
    export MSG badge_file
    python3 << 'PYEOF' 2>/dev/null || true
import json, os
msg_val = os.environ.get('MSG', '')
bfile = os.environ.get('badge_file', '')
with open(bfile) as f:
    b = json.load(f)
b['message'] = msg_val
b['updated'] = os.popen('date -u +%Y-%m-%dT%H:%M:%SZ').read().strip()
with open(bfile, 'w') as f:
    json.dump(b, f, indent=1)
    f.write('\n')
PYEOF
  fi
done

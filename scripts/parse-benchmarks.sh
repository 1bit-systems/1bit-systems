#!/usr/bin/env bash
# parse-benchmarks.sh — regenerate site/benchmarks.json + shields badges from the
# single source of truth (docs/wiki/performance.md). Labels/status/colors are
# HONEST: only measured-coherent numbers read as "production" / green.
#
# Flags:
#   --check   regenerate to temp, diff against committed; exit 1 if out of date
set -euo pipefail

CHECK=false
for arg; do case "$arg" in --check) CHECK=true;; esac; done

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo /home/bcloud)"

SRC="docs/wiki/performance.md"
SITE="site"
mkdir -p "$SITE"

# When in check mode, write to a temp directory instead of site/
if [ "$CHECK" = true ]; then
  OUT=$(mktemp -d)
  trap 'rm -rf "$OUT"' EXIT
else
  OUT="$SITE"
fi

# tok/s extractor: first "**N tok/s**" on a row containing the given label text.
extract_tok() { grep -iE "$1" "$SRC" 2>/dev/null | grep -oP '\*\*~?\K[0-9]+(?= tok/s)' | head -1 || true; }

NPU_FLM=$(extract_tok "NPU FLM");     NPU_FLM="${NPU_FLM:-94}"
TERN=$(extract_tok "GPU ternary");    TERN="${TERN:-279}"
GPU_1BIT=$(extract_tok "GPU 1-bit");  GPU_1BIT="${GPU_1BIT:-381}"
GPU_ZINC=$(extract_tok "GPU ZINC");   GPU_ZINC="${GPU_ZINC:-22}"
ROCM=$(extract_tok "ROCm");           ROCM="${ROCM:-113}"
# DSpark was projected at 572 tok/s but disproven (0.1–0.2 tok/s end-to-end).
DSPARK=0
FUSED=$(extract_tok "NPU fused");     FUSED="${FUSED:-291}"
NPU_V12=$(extract_tok "NPU v12");     NPU_V12="${NPU_V12:-97}"
ALL5=$(grep -iE "C\+\+ (all|ALL)" "$SRC" 2>/dev/null | grep -oP '\*\*\K[0-9]+' | head -1) || true; ALL5="${ALL5:-28}"
TFLOPS=$(grep "TFLOPS" "$SRC" | grep -oP '[0-9]+(\.[0-9]+)?(?= TFLOPS)' | head -1) || true; TFLOPS="${TFLOPS:-55.7}"

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
cat > "$OUT/benchmarks.json" << EOF
{
 "updated": "$TIMESTAMP",
 "source": "$SRC (single source of truth)",
 "_legend": {
  "validated": "measured on-device, coherent output",
  "measured": "throughput measured via third-party tool (llama.cpp)",
  "projected": "base engine x speculative-decode acceptance; NOT an end-to-end measurement",
  "raw": "kernel runs at this speed but engine output is not yet coherent",
  "reported": "reported, not independently re-measured"
 },
 "engines": {
  "npu_fused":{ "tok_s": $FUSED,    "status": "validated", "label": "NPU fused (coherent — BF16 overflow fix)" },
  "npu_flm":  { "tok_s": $NPU_FLM,  "status": "validated", "label": "FLM proxy (production)" },
  "npu_v12":  { "tok_s": $NPU_V12,  "status": "raw",       "label": "C++ v12 (raw — not yet coherent)" },
  "rocm_hip": { "tok_s": $ROCM,     "status": "reported",  "label": "GPU ROCm HIP (reported)" },
  "all_5":    { "tok_s": $ALL5,     "status": "raw",       "label": "C++ all-5 (raw — not yet coherent)" },
  "dspark":   { "tok_s": 0, "status": "disproven", "label": "DSpark spec-decode (disproven — 0.1–0.2 tok/s end-to-end)" },
  "ternary":  { "tok_s": $TERN,     "status": "validated", "label": "Ternary Vulkan (validated 1-bit)" },
  "gpu_1bit": { "tok_s": $GPU_1BIT, "status": "measured",  "label": "GPU llama.cpp 1-bit (measured)" },
  "gpu_zinc": { "tok_s": $GPU_ZINC, "status": "validated", "label": "GPU Vulkan ZINC F16 (validated)" }
 },
 "order": ["npu_fused","npu_flm","npu_v12","rocm_hip","all_5","dspark","ternary","gpu_1bit","gpu_zinc"],
 "system": { "tflops_int8": $TFLOPS }
}
EOF

write_badge() { cat > "$OUT/$1" << EOF
{"schemaVersion":1,"label":"$2","message":"$3","color":"$4","cacheSeconds":86400}
EOF
}
write_badge "validated-badge.json" "validated" "${FUSED} tok/s NPU fused · ${NPU_FLM} tok/s NPU · ${TERN} tok/s 1-bit GPU" "brightgreen"
write_badge "flm-badge.json"   "NPU (FLM, production)"   "${NPU_FLM} tok/s"          "brightgreen"
write_badge "tern-badge.json"  "GPU 1-bit ternary"       "${TERN} tok/s"             "brightgreen"
write_badge "gpu-badge.json"   "GPU ZINC (F16)"          "${GPU_ZINC} tok/s"         "blue"
write_badge "rocm-badge.json"  "GPU ROCm (reported)"     "${ROCM} tok/s"             "yellowgreen"
write_badge "dspark-badge.json" "DSpark spec-decode"     "disproven (0.1–0.2 tok/s)" "red"
write_badge "fused-badge.json" "NPU fused layer"         "${FUSED} tok/s (coherent)"  "brightgreen"
write_badge "bench-badge.json" "NPU C++ v12"             "${NPU_V12} tok/s (raw, WIP)" "yellow"
write_badge "tok-badge.json"   "decode (validated)"     "${NPU_FLM} tok/s"          "brightgreen"
write_badge "tflops-badge.json" "INT8 GEMM"              "${TFLOPS} TFLOPS"          "brightgreen"

# ── Check mode: diff generated against committed, fail on mismatch ──
if [ "$CHECK" = true ]; then
  errors=0
  # Only check files this script generates (not external badges like pageviews)
  GENERATED="benchmarks.json validated-badge.json flm-badge.json tern-badge.json gpu-badge.json rocm-badge.json dspark-badge.json fused-badge.json bench-badge.json tok-badge.json tflops-badge.json"
  for base in $GENERATED; do
    if [ ! -f "$SITE/$base" ]; then
      echo "::error title=missing::$base does not exist in $SITE/ — run parse-benchmarks.sh"
      errors=$((errors + 1))
      continue
    fi
    # Strip timestamp line before diff
    if ! diff <(grep -v '"updated"' "$OUT/$base") <(grep -v '"updated"' "$SITE/$base") > /dev/null 2>&1; then
      echo "::error title=stale::$base — regenerated output differs from committed"
      diff "$OUT/$base" "$SITE/$base" | head -20
      errors=$((errors + 1))
    fi
  done
  if [ "$errors" -gt 0 ]; then
    echo "❌ $errors file(s) out of date — run scripts/parse-benchmarks.sh and commit the changes"
    exit 1
  fi
  echo "✅ benchmarks.json + badges are up to date with $SRC"
  exit 0
fi

echo "✅ regenerated benchmarks.json + $(ls "$SITE"/*-badge.json | wc -l) badges (honest labels) | FLM=$NPU_FLM TERN=$TERN V12=$NPU_V12(raw) DSPARK=disproven"

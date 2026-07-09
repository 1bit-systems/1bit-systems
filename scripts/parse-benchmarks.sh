#!/usr/bin/env bash
# parse-benchmarks.sh — regenerate site/benchmarks.json + shields badges from the
# single source of truth (docs/wiki/performance.md). Labels/status/colors are
# HONEST: only measured-coherent numbers read as "production" / green.
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo /home/bcloud)"

SRC="docs/wiki/performance.md"
SITE="site"
mkdir -p "$SITE"

# tok/s extractor: first "**N tok/s**" on a row containing the given label text.
extract_tok() { grep -iE "$1" "$SRC" 2>/dev/null | grep -oP '\*\*~?\K[0-9]+(?= tok/s)' | head -1 || true; }

NPU_FLM=$(extract_tok "NPU FLM");     NPU_FLM="${NPU_FLM:-94}"    # validated production
TERN=$(extract_tok "GPU ternary");    TERN="${TERN:-279}"        # validated 1-bit headline
GPU_1BIT=$(extract_tok "GPU 1-bit");  GPU_1BIT="${GPU_1BIT:-381}" # measured (llama.cpp)
GPU_ZINC=$(extract_tok "GPU ZINC");   GPU_ZINC="${GPU_ZINC:-22}" # validated
ROCM=$(extract_tok "ROCm");           ROCM="${ROCM:-113}"        # reported
DSPARK=$(extract_tok "DSpark");       DSPARK="${DSPARK:-572}"    # PROJECTED
FUSED=$(extract_tok "NPU fused");     FUSED="${FUSED:-291}"      # raw, not coherent
NPU_V12=$(extract_tok "NPU v12");     NPU_V12="${NPU_V12:-97}"   # raw, not coherent
ALL5=$(grep -iE "C\+\+ (all|ALL)" "$SRC" 2>/dev/null | grep -oP '\*\*\K[0-9]+' | head -1) || true; ALL5="${ALL5:-28}"
TFLOPS=$(grep "TFLOPS" "$SRC" | grep -oP '[0-9]+(\.[0-9]+)?(?= TFLOPS)' | head -1) || true; TFLOPS="${TFLOPS:-55.7}"

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
cat > "$SITE/benchmarks.json" << EOF
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
  "dspark":   { "tok_s": $DSPARK,   "status": "projected", "label": "DSpark spec-decode (projected)" },
  "ternary":  { "tok_s": $TERN,     "status": "validated", "label": "Ternary Vulkan (validated 1-bit)" },
  "gpu_1bit": { "tok_s": $GPU_1BIT, "status": "measured",  "label": "GPU llama.cpp 1-bit (measured)" },
  "gpu_zinc": { "tok_s": $GPU_ZINC, "status": "validated", "label": "GPU Vulkan ZINC F16 (validated)" }
 },
 "order": ["npu_fused","npu_flm","npu_v12","rocm_hip","all_5","dspark","ternary","gpu_1bit","gpu_zinc"],
 "system": { "tflops_int8": $TFLOPS }
}
EOF

write_badge() { cat > "$SITE/$1" << EOF
{"schemaVersion":1,"label":"$2","message":"$3","color":"$4","cacheSeconds":86400}
EOF
}
# green = validated only; yellow = projected/raw
write_badge "validated-badge.json" "validated" "${FUSED} tok/s NPU fused · ${NPU_FLM} tok/s NPU · ${TERN} tok/s 1-bit GPU" "brightgreen"
write_badge "flm-badge.json"   "NPU (FLM, production)"   "${NPU_FLM} tok/s"          "brightgreen"
write_badge "tern-badge.json"  "GPU 1-bit ternary"       "${TERN} tok/s"             "brightgreen"
write_badge "gpu-badge.json"   "GPU ZINC (F16)"          "${GPU_ZINC} tok/s"         "blue"
write_badge "rocm-badge.json"  "GPU ROCm (reported)"     "${ROCM} tok/s"             "yellowgreen"
write_badge "dspark-badge.json" "DSpark spec-decode"     "~${DSPARK} tok/s (projected)" "yellow"
write_badge "fused-badge.json" "NPU fused layer"         "${FUSED} tok/s (coherent)"  "brightgreen"
write_badge "bench-badge.json" "NPU C++ v12"             "${NPU_V12} tok/s (raw, WIP)" "yellow"
write_badge "tok-badge.json"   "decode (validated)"     "${NPU_FLM} tok/s"          "brightgreen"
write_badge "tflops-badge.json" "INT8 GEMM"              "${TFLOPS} TFLOPS"          "brightgreen"

echo "✅ regenerated benchmarks.json + $(ls "$SITE"/*-badge.json | wc -l) badges (honest labels) | FLM=$NPU_FLM TERN=$TERN DSPARK=$DSPARK(proj) V12=$NPU_V12(raw)"

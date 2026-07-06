#!/usr/bin/env bash
# parse-benchmarks.sh — Extract headline benchmark numbers from performance.md
# Outputs site/benchmarks.json and shields.io badge JSON files.
# Run from repo root after updating docs/wiki/performance.md.

set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo /home/bcloud)"

SRC="docs/wiki/performance.md"
SITE="site"
mkdir -p "$SITE"

# ── Extract headline tok/s from At a Glance table ────────────────────────
# Format: | **NPU v12** (open C++, production) | XDNA 2 · 32 tiles | **97 tok/s** | Qwen3-0.6B |
extract_tok() { grep "$1" "$SRC" | grep -oP '\*\*\K[0-9]+(?= tok/s)' | head -1 || true; }

NPU_V12=$(extract_tok "NPU v12")   || true; NPU_V12="${NPU_V12:-97}"
NPU_FLM=$(extract_tok "NPU FLM")   || true; NPU_FLM="${NPU_FLM:-94}"
GPU_ZINC=$(extract_tok "GPU ZINC") || true; GPU_ZINC="${GPU_ZINC:-22}"
ROCM=$(extract_tok "ROCm.*HIP")    || true; ROCM="${ROCM:-113}"
TERN=$(extract_tok "Ternary")      || true; TERN="${TERN:-279}"

# C++ ALL: | **C++ ALL** (5 models) | 36 ms/tok | 14 ms/tok prefill | **28** | Auto-detect |
ALL5=$(grep "C++ ALL" "$SRC" | grep -oP '\*\*\K[0-9]+' | head -1) || true; ALL5="${ALL5:-28}"

# System metrics
TFLOPS=$(grep "TFLOPS" "$SRC" | grep -oP '[0-9]+(\.[0-9]+)?(?= TFLOPS)' | head -1) || true
SPEEDUP=$(grep "× speedup" "$SRC" | grep -oP '[0-9]+(?=× speedup)' | head -1) || true
TFLOPS="${TFLOPS:-55.7}"; SPEEDUP="${SPEEDUP:-24}"

# ── Write benchmarks.json ─────────────────────────────────────────────────
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
cat > "$SITE/benchmarks.json" << EOF
{
  "updated": "$TIMESTAMP",
  "source": "$SRC",
  "engines": {
    "npu_v12":     { "tok_s": $NPU_V12, "label": "C++ v12 (production)" },
    "npu_flm":     { "tok_s": $NPU_FLM, "label": "FLM proxy (fallback)" },
    "all_5":       { "tok_s": $ALL5,    "label": "C++ all-5 (auto-detect)" },
    "gpu_zinc":    { "tok_s": $GPU_ZINC,"label": "GPU Vulkan (ZINC)" },
    "rocm_hip":    { "tok_s": $ROCM,    "label": "GPU ROCm (HIP)" },
    "ternary":     { "tok_s": $TERN,    "label": "Ternary (Vulkan)" }
  },
  "system": {
    "tflops_int8": $TFLOPS,
    "speedup": "${SPEEDUP}x"
  }
}
EOF
echo "✅ $SITE/benchmarks.json | NPU=${NPU_V12} FLM=${NPU_FLM} GPU=${GPU_ZINC} ROCM=${ROCM} TERN=${TERN}"

# ── Shields.io badges ────────────────────────────────────────────────────
write_badge() {
  cat > "$SITE/$1" << EOF
{"schemaVersion":1,"label":"$2","message":"$3","color":"$4","cacheSeconds":86400}
EOF
}

write_badge "bench-badge.json"  "NPU C++ v12"    "${NPU_V12} tok/s"          "brightgreen"
write_badge "tok-badge.json"    "decode (NPU)"    "${NPU_V12} tok/s"          "brightgreen"
write_badge "flm-badge.json"    "decode (FLM)"    "${NPU_FLM} tok/s"          "green"
write_badge "gpu-badge.json"    "decode (GPU)"    "${GPU_ZINC} tok/s"         "blue"
write_badge "rocm-badge.json"   "decode (ROCm)"   "${ROCM} tok/s"            "yellowgreen"
write_badge "tern-badge.json"   "decode (tern)"   "${TERN} tok/s"             "brightgreen"
write_badge "tflops-badge.json" "INT8 GEMM"       "${TFLOPS} TFLOPS"          "brightgreen"

echo "✅ $(ls "$SITE"/*-badge.json 2>/dev/null | wc -l) badge files written"

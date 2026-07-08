#!/usr/bin/env bash
# ── 1bit.systems numbers — single source of truth ──
# Auto-updates ALL site files with current binary sizes.
# Run: npm run build  (auto)  or  npm run measure  (standalone)
set -euo pipefail
cd "$(dirname "$0")/.."

# Measure binaries (KB, rounded nearest)
measure() { [ -f "$1" ] && echo $(( ($(stat -c%s "$1") + 512) / 1024 )) || echo "0"; }

FUSED=$(measure engine/npu/build/npu_engine_fused)
SPEC=$(measure spec-decode/build/npu_spec_decode)
MAIN=$(measure engine/npu/build/npu_engine)
UNIVERSAL=$(measure engine/npu/build/npu_engine_universal)
SERVER=$(measure engine/npu/build/npu_engine_server)

echo "  fused: ${FUSED} KB   spec-decode: ${SPEC} KB   main: ${MAIN} KB"

# ── Write numbers.json (source of truth) ──
cat > site/numbers.json <<EOF
{"binary":{"fused_kb":${FUSED},"spec_decode_kb":${SPEC},"main_engine_kb":${MAIN},"universal_kb":${UNIVERSAL},"server_kb":${SERVER}},"benchmarks":{"npu_validated_tok_s":94,"npu_fused_raw_tok_s":291,"gpu_ternary_tok_s":279,"gpu_1bit_llama_tok_s":381,"gpu_f16_baseline_tok_s":22,"tflops":55.7},"models":73,"backends":6}
EOF

# ── sed patterns to replace stale KB numbers ──
# $SPEC = spec-decode binary (the "single binary" marketing number)
# $FUSED = fused layer engine (the "fused layer" technical detail)

update_file() {
  local file="$1"
  [ ! -f "$file" ] && return
  # Replace all "38 KB" / "38KB" with actual fused layer size
  sed -i \
    -e "s/38KB/${FUSED}KB/g" \
    -e "s/38 KB /${FUSED} KB /g" \
    -e "s/38 KB</${FUSED} KB</g" \
    -e "s/38 KB\*\*/${FUSED} KB**/g" \
    -e "s/| 38 KB |/| ${FUSED} KB |/g" \
    -e "s/38 KB\./${FUSED} KB./g" \
    -e "s/38 KB,/${FUSED} KB,/g" \
    -e "s/38 KB)/${FUSED} KB)/g" \
    "$file"
  # Replace spec-decode binary size
  sed -i \
    -e "s/[0-9][0-9]* KB Single Binary/${SPEC} KB Single Binary/g" \
    -e "s/[0-9][0-9]* KB single binary/${SPEC} KB single binary/g" \
    -e "s/\"[0-9][0-9]* KB single binary\"/\"${SPEC} KB single binary\"/g" \
    -e "s/>[0-9][0-9]* KB binary</>${SPEC} KB binary</g" \
    -e "s/>[0-9][0-9]* KB unified</>${SPEC} KB unified</g" \
    -e "s/>[0-9][0-9]* KB · MIT</>${SPEC} KB · MIT</g" \
    -e "s/fused layer · one xclbin/spec-decode · NPU+GPU+CPU/g" \
    "$file"
}

# ── Update ALL site files ──
update_file site/index.html
update_file site/demo/index.html
update_file site/demo/gpu.html
update_file site/demo/text.html
update_file site/404.html
update_file site/store/index.html
for f in site/blog/*.html site/blog/*.md; do update_file "$f"; done

# ── Update repo docs ──
update_file CLAUDE.md
update_file README.md
for f in docs/*.md docs/wiki/*.md; do update_file "$f"; done

# ── Fix specific known stale numbers ──
# CLAUDE.md: "342 tok/s" was never validated → 279
sed -i 's/342 tok\/s/279 tok\/s/g' CLAUDE.md 2>/dev/null || true
# site/index.html: 342 tok/s GPU ternary → 279 (validated)
sed -i 's/342 tok\/s/279 tok\/s/g' site/index.html 2>/dev/null || true
# README badge: 307 tok/s → 279 (validated number)
sed -i 's/307%20tok%2Fs-ternary/279%20tok%2Fs-ternary/g' README.md 2>/dev/null || true
sed -i 's/307 tok\/s ternary/279 tok\/s ternary/g' README.md 2>/dev/null || true
# README: "4 days. 307 tok/s" → 279
sed -i 's/\(4 days\. \)307/\1279/g' README.md 2>/dev/null || true
# wiki: 437 → 436 (current spec-decode size)
sed -i 's/437 KB npu_spec_decode/436 KB npu_spec_decode/g' docs/wiki/npu-architecture.md 2>/dev/null || true
sed -i 's/437 KB (4-xclbin)/436 KB (4-xclbin)/g' docs/wiki/performance.md 2>/dev/null || true
# src/commands/chat.ts: 94 tok/s is still correct (validated FLM number)
# "50 TOPS" in chat.ts is correct (AMD spec)

echo "  → site/* updated"
echo "  → CLAUDE.md updated"
echo "  → README.md updated"
echo "  → docs/ updated"

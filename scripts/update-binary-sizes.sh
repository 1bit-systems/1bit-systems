#!/usr/bin/env bash
# Auto-update binary size numbers in site/index.html from actual build artifacts
set -euo pipefail
cd "$(dirname "$0")/.."

# Measure actual sizes (KB, rounded nearest)
fused=$(stat -c%s engine/npu/build/npu_engine_fused 2>/dev/null || echo 0)
fused_kb=$(( (fused + 512) / 1024 ))

spec=$(stat -c%s spec-decode/build/npu_spec_decode 2>/dev/null || echo 0)
spec_kb=$(( (spec + 512) / 1024 ))

main=$(stat -c%s engine/npu/build/npu_engine 2>/dev/null || echo 0)
main_kb=$(( (main + 512) / 1024 ))

echo "fused layer: ${fused_kb} KB (${fused} bytes)"
echo "spec-decode: ${spec_kb} KB (${spec} bytes)"
echo "main engine: ${main_kb} KB (${main} bytes)"

# Update site/index.html — replace all "38 KB" with "${spec_kb} KB" for the unified binary
# but keep the fused layer reference distinct
sed -i \
  -e "s/38 KB Fused Layer/${spec_kb} KB Single Binary/g" \
  -e "s/38 KB fused layer binary/${spec_kb} KB single binary/g" \
  -e "s/38 KB fused layer/${fused_kb} KB fused layer/g" \
  -e "s/38 KB binary/${spec_kb} KB single binary/g" \
  -e "s/38 KB Fused/${spec_kb} KB Single/g" \
  -e "s/38 KB layer/${fused_kb} KB fused layer/g" \
  -e "s/38 KB\./${spec_kb} KB./g" \
  site/index.html

# Fix the "38 KB" in the inline text about fused layer + spec-decode sizes
sed -i \
  "s/38 KB fused layer \`/${fused_kb} KB fused layer per dispatch; /" \
  site/index.html

echo ""
echo "Updated site/index.html"
grep -n "KB" site/index.html | grep -i "single\|fused\|binary" | head -10

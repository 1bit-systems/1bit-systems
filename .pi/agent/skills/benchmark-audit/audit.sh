#!/usr/bin/env bash
# audit.sh — Benchmark Stale Number Audit
# Cross-references all benchmark claims against docs/wiki/performance.md
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
REPO="${HOME}"
ST_FILE="${REPO}/docs/wiki/performance.md"
QUICK=false; REPORT=false
TMPDIR="${TMPDIR:-/tmp}/benchmark-audit"; mkdir -p "$TMPDIR"

for arg in "$@"; do
  case "$arg" in --quick) QUICK=true ;; --report) REPORT=true ;; esac
done

if [ ! -f "$ST_FILE" ]; then
  echo "Error: source of truth not found at $ST_FILE"
  exit 1
fi

echo -e "${BOLD}${CYAN}══════════════════════════════════════════════${NC}"
echo -e "${BOLD}${CYAN}  BENCHMARK AUDIT — Stale Number Detection${NC}"
echo -e "${BOLD}${CYAN}  Source of truth: docs/wiki/performance.md${NC}"
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════${NC}"$'\n'

# ═══════════════════════════════════════════════
# DEFINITIONS: Current (authoritative) vs Obsolete numbers
# ═══════════════════════════════════════════════

# Current authoritative claims (what SHOULD be in files)
# Format: "exact string"|context
CURRENT_CLAIMS=$(cat << 'EOC'
291 tok/s|NPU fused layer production
97 tok/s|NPU C++ v12 fallback
94 tok/s|NPU FLM fallback
113 tok/s|ROCm Bonsai TQ2
28 tok/s|C++ all-5 models
22 tok/s|GPU ZINC
279 tok/s|ternary Vulkan
38 KB|binary size fused engine
55.7 TFLOPS|INT8 GEMM peak D projection
50 TOPS|NPU INT8 spec
73+ models|total model count
22 multi-modal|multi-modal models
3.4 ms/tok|fused layer decode
10 ms/tok|C++ v12 decode
36 ms/tok|C++ all Qwen3-0.6B
381 tok/s|GPU Qwen2-0.5B IQ1_S
312 tok/s|GPU Qwen3.5-0.8B Q1_0
267 tok/s|GPU Hy-MT2-1.8B STQ1_0
158 tok/s|GPU gemma-2-2b IQ1_S
122 tok/s|GPU gemma3-4B IQ1_S
79 tok/s|GPU Nemo-8B IQ1_S
70 tok/s|GPU Qwen3.5-9B Q1_0
EOC
)

# Obsolete numbers that were replaced (should NOT appear in any file)
# Format: "exact obsolete string"|replaced by|context
OBSOLETE_CLAIMS=$(cat << 'EOC'
244 tok/s|291 tok/s|old NPU decode before fused layer engine
74 KB|38 KB|old binary size before stripped build (check context: C++ v12 standalone is 74 KB, fused is 38 KB)
120 KB|38 KB|older binary size claim (check context: standalone C++ binary is ~120 KB unstripped)
EOC
)

# Files where "74 KB" references are contextually correct (historical/comparative/technical)
ALLOWLIST_FILES=(
  "docs/journey.md"
  "1bit-systems/docs/journey.md"
  "docs/launch.md"
  "1bit-systems/docs/launch.md"
  "site/blog/"
  "1bit-systems/site/blog/"
  ".kb/"
  "1bit-systems/.kb/"
  "engine/npu/BENCHMARKS.md"
  "1bit-systems/engine/npu/BENCHMARKS.md"
)

# ═══════════════════════════════════════════════
# FILES TO SCAN
# ═══════════════════════════════════════════════
echo -e "${BOLD}Selecting files...${NC}"

if $QUICK; then
  SCAN_FILES=(
    "${REPO}/CLAUDE.md"
    "${REPO}/README.md"
    "${REPO}/site/index.html"
    "${REPO}/site/tok-badge.json"
    "${REPO}/site/bench-badge.json"
    "${REPO}/site/gpu-badge.json"
    "${REPO}/site/rocm-badge.json"
    "${REPO}/site/tern-badge.json"
    "${REPO}/site/tflops-badge.json"
    "${REPO}/site/flm-badge.json"
    "${REPO}/site/live.html"
  )
elif $REPORT; then
  if [ -f "${TMPDIR}/last-scan-files.txt" ]; then
    mapfile -t SCAN_FILES < "${TMPDIR}/last-scan-files.txt"
  else
    echo -e "${RED}No cached scan. Run without --report first.${NC}"; exit 1
  fi
else
  mapfile -t SCAN_FILES < <(find "$REPO" -maxdepth 4 -type f \( -name "*.md" -o -name "*.html" -o -name "*.json" \) \
    -not -path "*/.git/*" -not -path "*/.claude/*" -not -path "*/node_modules/*" -not -path "*/snap/*" \
    -not -path "*/.local/*" -not -path "*/.cache/*" -not -path "*/go/pkg/*" \
    -not -path "*/1bit-agent/*" -not -path "*/.1bit/*" -not -path "*/.npm/*" -not -path "*/.ollama/*" \
    -not -path "*/build-rocm/*" -not -path "*/Xilinx/*" -not -path "*/zig-*/*" \
    -not -path "*/whisper.cpp/*" -not -path "*/mlir-aie/*" -not -path "*/zaya-llama.cpp/*" \
    -not -path "*/engine/zaya/*" -not -path "*/engine/video/*" -not -path "*/xilinx-vitis-install/*" \
    -not -path "*/torch2aie/*" -not -path "*/npu-sandbox/*" -not -path "*/npu-gpu-cpu/*" \
    -not -path "*/1bit/*" -not -path "*/1bit-systems/1bit/*" -not -path "*/1bit-systems/snap/*" -not -path "*/lemonade/*" \
    -not -path "*/lemon-mlx-engine/*" -not -path "*/strixhalo-npu-setup/*" \
    -not -path "*/jarvis-mobile/*" -not -path "*/lemonade-mobile-local/*" \
    -not -path "*/store-designs/*" -not -path "*/.pi/*" -not -path "*/Desktop/*" \
    -not -path "*/DeepSpec/*" -not -path "*/site/store/*" -not -path "*/site/jarvis/*" \
    -not -path "*/engine/gpu/src/server/chat.html" \
    -not -path "*/Xilinx/*" -not -path "*/npu_benchmark/*" \
    | sort)
  printf "%s\n" "${SCAN_FILES[@]}" > "${TMPDIR}/last-scan-files.txt"
fi

echo -e "  ${YELLOW}${#SCAN_FILES[@]}${NC} files to check"
echo ""

# ═══════════════════════════════════════════════
# SCAN: Check each file for stale/current claims
# ═══════════════════════════════════════════════
echo -e "${BOLD}Scanning files...${NC}"

RESULTS_FILE="${TMPDIR}/results.txt"
MISSING_FILE="${TMPDIR}/missing-claims.txt"
echo -n > "$RESULTS_FILE"
echo -n > "$MISSING_FILE"

scan_file() {
  local file="$1" f relpath
  # Strip common repo prefix for relative paths
  if [[ "$file" == ${REPO}/1bit-systems/* ]]; then
    relpath="1bit-systems/${file#${REPO}/1bit-systems/}"
  else
    relpath="${file#$REPO/}"
  fi
  [ "$file" = "$ST_FILE" ] && return
  [ ! -f "$file" ] && return

  local content; content=$(cat "$file" 2>/dev/null) || return
  local stale_found=false

  # Check if this file is in the allowlist (historical/comparative context, not stale)
  local is_allowlisted=false
  for pattern in "${ALLOWLIST_FILES[@]}"; do
    if [[ "$relpath" == *"$pattern"* ]]; then
      is_allowlisted=true
      break
    fi
  done

  if $is_allowlisted; then
    echo "  ${relpath}: ${YELLOW}⚠ ALLOWLISTED${NC} (historical/comparative context)" >> "$RESULTS_FILE"
    return
  fi

  # Check for OBSOLETE numbers in this file
  while IFS='|' read -r obsolete_str replaced_by context; do
    if echo "$content" | grep -qF "$obsolete_str" 2>/dev/null; then
      echo "  ${relpath}: ${RED}✗ STALE${NC} '${obsolete_str}' (should be '${replaced_by}' — ${context})" >> "$RESULTS_FILE"
      stale_found=true
    fi
  done <<< "$OBSOLETE_CLAIMS"

  if ! $stale_found; then
    echo "  ${relpath}: ${GREEN}✓${NC}" >> "$RESULTS_FILE"
  fi
}

# Run in parallel
MAX_JOBS=8; job_count=0
for file in "${SCAN_FILES[@]}"; do
  scan_file "$file" &
  job_count=$((job_count + 1))
  if [ $job_count -ge $MAX_JOBS ]; then
    wait -n 2>/dev/null || true; job_count=$((job_count - 1))
  fi
done
wait

# ═══════════════════════════════════════════════
# VERIFY: Current claims are present in key files
# ═══════════════════════════════════════════════
echo -e "${BOLD}Verifying key claims in critical files...${NC}"

KEY_FILES=(
  "CLAUDE.md"
  "README.md"
  "site/index.html"
  "1bit-systems/CLAUDE.md"
  "1bit-systems/README.md"
  "1bit-systems/site/index.html"
)

while IFS='|' read -r claim_str label; do
  claim_str_trim=$(echo "$claim_str" | xargs)
  for keyfile in "${KEY_FILES[@]}"; do
    fpath="${REPO}/${keyfile}"
    if [ -f "$fpath" ]; then
      if grep -qF "$claim_str_trim" "$fpath" 2>/dev/null; then
        : # found
      else
        echo "${keyfile}: ${YELLOW}⚠ MISSING${NC} '${claim_str_trim}' (${label})" >> "$MISSING_FILE"
      fi
    fi
  done
done <<< "$CURRENT_CLAIMS"

# ═══════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════
echo ""
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════${NC}"
echo -e "${BOLD}${CYAN}  AUDIT RESULTS${NC}"
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════${NC}"$'\n'

# Stale numbers
stale_lines=$(grep 'STALE' "$RESULTS_FILE" 2>/dev/null || true)
stale_count=$(echo "$stale_lines" | grep -c . 2>/dev/null || true)

if [ -n "$stale_lines" ] && [ "$stale_count" -gt 0 ] 2>/dev/null; then
  echo -e "${BOLD}${RED}✗ STALE NUMBERS FOUND (${stale_count}):${NC}"$'\n'
  echo "$stale_lines"
else
  echo -e "${GREEN}✓ No stale numbers found!${NC}"
fi
echo ""

# Missing claims
missing_lines=$(cat "$MISSING_FILE" 2>/dev/null || true)
missing_count=$(echo "$missing_lines" | grep -c . 2>/dev/null || echo 0)

if [ "$missing_count" -gt 0 ]; then
  echo -e "${BOLD}${YELLOW}⚠ MISSING CLAIMS (${missing_count}):${NC}"$'\n'
  echo "$missing_lines"
  echo ""
  echo -e "  These claims exist in docs/wiki/performance.md but are missing from"
  echo -e "  CLAUDE.md, README.md, or site/index.html. Not necessarily a bug —"
  echo -e "  some claims are only relevant in the deep benchmark doc."
else
  echo -e "${GREEN}✓ All key claims present in critical files.${NC}"
fi
echo ""

# Clean files
clean_count=$(grep -c "${GREEN}✓" "$RESULTS_FILE" 2>/dev/null || echo 0)
echo -e "${GREEN}✓ ${clean_count} files clean${NC}"
echo ""

# ═══════════════════════════════════════════════
# CLAUDE.md HEADER CHECK
# ═══════════════════════════════════════════════
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════${NC}"
echo -e "${BOLD}  CLAUDE.md Header Cross-Check${NC}"
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════${NC}"$'\n'

header=$(sed -n '3p' "${REPO}/1bit-systems/CLAUDE.md" 2>/dev/null)
if [ -z "$header" ]; then header=$(sed -n '3p' "${REPO}/CLAUDE.md" 2>/dev/null); fi
echo -e "  Current header: ${YELLOW}${header:0:120}...${NC}"$'\n'

HEADER_CLAIMS=("291 tok/s" "97 tok/s" "94 tok/s" "113 tok/s" "28 tok/s" "22 tok/s" "279 tok/s" "38 KB")
all_present=true
for claim in "${HEADER_CLAIMS[@]}"; do
  if echo "$header" | grep -qF "$claim" 2>/dev/null; then
    echo -e "    ${GREEN}✓${NC} $claim"
  else
    echo -e "    ${RED}✗${NC} $claim ${RED}MISSING${NC}"
    all_present=false
  fi
done

if $all_present; then
  echo -e "\n  ${GREEN}CLAUDE.md header is complete.${NC}"
else
  echo -e "\n  ${RED}CLAUDE.md header is missing some key claims.${NC}"
fi

echo ""
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════${NC}"
echo -e "${BOLD}${CYAN}  AUDIT COMPLETE${NC}"
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════${NC}"$'\n'
echo -e "  Source of truth: ${ST_FILE}"
echo -e "  Cache: ${TMPDIR}/"
echo ""
echo -e "${YELLOW}  Next steps if stale numbers found:${NC}"
echo -e "    1. Edit each stale file to match docs/wiki/performance.md"
echo -e "    2. Run again: ${BOLD}bash ~/.pi/agent/skills/benchmark-audit/audit.sh${NC}"
echo -e "    3. Verify clean before committing"$'\n'

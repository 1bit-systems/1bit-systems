#!/bin/bash
set -euo pipefail
# ───────────────────────────────────────────────────────────────
# bench_all_archs.sh — Run inference benchmark on all available
# Q4NX model directories, mapping each to its architecture tag.
#
# For each model: runs fused-engine with --max-tokens 10,
# captures tokens/s, total inference time, and status.
# Outputs a Markdown table.
# ───────────────────────────────────────────────────────────────
set -euo pipefail

ENGINE="${1:-./zig-out/bin/fused-engine}"
MODEL_DIR="${HOME}/.config/flm/models"
TIMEOUT_PER_MODEL="${2:-120}"   # seconds per model
MAX_TOKENS=10
PROMPT="Hello"
RESULTS_FILE="/tmp/bench_results_$$.txt"
exec > >(tee -a "${RESULTS_FILE}") 2>&1

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

PASS=0
FAIL=0
SKIP=0

# ── Architecture mapping (same as test_all_archs.sh) ──
declare -A MODEL_ARCH_MAP
MODEL_ARCH_MAP["Qwen3-0.6B-NPU2"]="qwen3_0_6b"
MODEL_ARCH_MAP["Qwen3-1.7B-NPU2"]="qwen3_1_5b"
MODEL_ARCH_MAP["Qwen3-4B-NPU2"]="qwen3_7b"
MODEL_ARCH_MAP["Qwen3.5-0.8B-NPU2"]="qwen3_0_6b"
MODEL_ARCH_MAP["Qwen3.5-9B-NPU2"]="qwen3_7b"
MODEL_ARCH_MAP["Llama-3.1-8B-NPU2"]="llama3_1_8b"
MODEL_ARCH_MAP["Llama-3.2-1B-NPU2"]="llama3_2_1b"
MODEL_ARCH_MAP["Llama-3.2-3B-NPU2"]="llama3_2_3b"
MODEL_ARCH_MAP["Gemma3-4B-NPU2"]="gemma2_2b"
MODEL_ARCH_MAP["Gemma4-E4B-IT-NPU2"]="gemma2_2b"
MODEL_ARCH_MAP["DeepSeek-R1-0528-Qwen3-8B-NPU2"]="deepseek_v2_lite"
MODEL_ARCH_MAP["Phi4-mini-Instruct-NPU2"]="qwen3_0_6b"
MODEL_ARCH_MAP["Medgemma-4B-NPU2"]="gemma2_2b"
# Additional mappings for models discovered in the model directory (best-effort guesses)
MODEL_ARCH_MAP["Bonsai-1.7B-Q4NX"]="qwen3_0_6b"         # small model, closest match
MODEL_ARCH_MAP["Deepseek-R1-Distill-Llama-8B-NPU2"]="llama3_1_8b"  # Llama-based distill
MODEL_ARCH_MAP["Gemma3-1B-NPU2"]="gemma2_2b"
MODEL_ARCH_MAP["Gemma4-E2B-IT-NPU2"]="gemma2_2b"
MODEL_ARCH_MAP["Medgemma-1.5-4B-NPU2"]="gemma2_2b"
MODEL_ARCH_MAP["Nanbeige4.1-3B-NPU2"]="qwen3_1_5b"       # ~3B, closest small arch
MODEL_ARCH_MAP["Qwen2.5-3B-Instruct-NPU2"]="qwen2_5_7b"   # Qwen2.5 family
MODEL_ARCH_MAP["Qwen3-4B-Instruct-2507-NPU2"]="qwen3_7b"
MODEL_ARCH_MAP["Qwen3-4B-Thinking-2507-NPU2"]="qwen3_7b"
MODEL_ARCH_MAP["Qwen3-8B-NPU2"]="qwen3_7b"               # 8B → qwen3_7b closest
MODEL_ARCH_MAP["Qwen3.5-2B-NPU2"]="qwen3_1_5b"
MODEL_ARCH_MAP["Qwen3.5-4B-NPU2"]="qwen3_7b"
MODEL_ARCH_MAP["Qwen3-VL-4B-Instruct-NPU2"]="qwen3_7b"
MODEL_ARCH_MAP["Qwen3.6-35B-A3B-NPU2"]="qwen3_7b"        # MoE, but closest arch
MODEL_ARCH_MAP["GPT-OSS-20B-NPU2"]="llama3_1_8b"          # 20B dense, Llama-like
MODEL_ARCH_MAP["GPT-OSS-Safeguard-20b-NPU2"]="llama3_1_8b"
MODEL_ARCH_MAP["Translategemma-4B-Instruct-NPU2"]="gemma2_2b"
MODEL_ARCH_MAP["LFM2-1.2B-NPU2"]="qwen3_0_6b"
MODEL_ARCH_MAP["LFM2-2.6B-NPU2"]="qwen3_1_5b"
MODEL_ARCH_MAP["LFM2-2.6B-Transcript-NPU2"]="qwen3_1_5b"
MODEL_ARCH_MAP["LFM2.5-1.2B-NPU2"]="qwen3_0_6b"
MODEL_ARCH_MAP["LFM2.5-1.2B-Thinking-NPU2"]="qwen3_0_6b"

# Models to skip (non-LLM or known incompatible)
declare -A SKIP_MODELS
SKIP_MODELS["Embedding-Gemma-300M-NPU2"]="embedding model, not generative"
SKIP_MODELS["Whisper-V3-Turbo-NPU2"]="audio model, not text generative"

echo -e "${BOLD}═══ Fused Engine — All-Architecture Benchmark ═══${NC}"
echo "Engine:    ${ENGINE}"
echo "Models:    ${MODEL_DIR}"
echo "Max tokens: ${MAX_TOKENS} per model"
echo "Timeout:   ${TIMEOUT_PER_MODEL}s per model"
echo "Prompt:    \"${PROMPT}\""
echo ""

# ── Discover model directories ──
declare -a MODELS
while IFS= read -r -d '' dir; do
  name=$(basename "$dir")
  [[ -f "${dir}/model.q4nx" ]] || continue
  # Skip test_ directories
  [[ "$name" == test_* ]] && continue
  # Check skip list
  if [[ -n "${SKIP_MODELS[$name]:-}" ]]; then
    echo -e "  ${YELLOW}⊘${NC} Skipping ${name}: ${SKIP_MODELS[$name]}"
    SKIP=$((SKIP + 1))
    continue
  fi
  MODELS+=("$name")
done < <(find "${MODEL_DIR}" -mindepth 1 -maxdepth 1 -type d -print0)

MODEL_COUNT=${#MODELS[@]}
echo -e "${CYAN}Found ${MODEL_COUNT} model(s) to benchmark${NC}"
echo ""

# ── Run benchmarks ──
declare -A RESULTS_TOK_S
declare -A RESULTS_TIME
declare -A RESULTS_STATUS
declare -A RESULTS_ARCH
declare -A RESULTS_MODEL_SIZE

# shellcheck disable=SC2034
for name in "${MODELS[@]}"; do
  model_file="${MODEL_DIR}/${name}/model.q4nx"
  arch="${MODEL_ARCH_MAP[$name]:-}"
  results_arch="${arch:-auto}"

  echo -ne "${BOLD}[$((PASS+FAIL+1))/${MODEL_COUNT}]${NC} ${name} "
  if [[ -n "$arch" ]]; then
    echo -ne "(${CYAN}${arch}${NC}) "
  else
    echo -ne "(${YELLOW}no mapping, trying auto-detect${NC}) "
  fi
  echo "..."

  # Build command
  cmd=("timeout" "${TIMEOUT_PER_MODEL}" "${ENGINE}" "--model" "${model_file}" "--max-tokens" "${MAX_TOKENS}" "--prompt" "${PROMPT}")
  if [[ -n "$arch" ]]; then
    cmd+=("--model-arch" "${arch}")
  fi

  # Capture output and timing
  SECONDS=0
  output=$("${cmd[@]}" 2>&1 || true)
  elapsed=$SECONDS

  # Parse results from output
  tok_s_line=$(echo "$output" | grep -oP '\d+ tokens in \d+ms \(\d+ tok/s\)' | tail -1)

  # Also try to extract model config for size info
  h_line=$(echo "$output" | grep -oP 'H=\d+')
  nv_line=$(echo "$output" | grep -oP 'NV=\d+')

  if [[ -n "$tok_s_line" ]]; then
    # Extract numbers from e.g. "10 tokens in 1234ms (8 tok/s)"
    tok_count=$(echo "$tok_s_line" | grep -oP '^\d+')
    time_ms=$(echo "$tok_s_line" | grep -oP 'in \K\d+')
    tok_s=$(echo "$tok_s_line" | grep -oP '\(\K\d+')
    RESULTS_TOK_S["$name"]=$tok_s
    RESULTS_TIME["$name"]=$time_ms
    RESULTS_STATUS["$name"]="✅"
    RESULTS_ARCH["$name"]="${arch:-auto}"
    RESULTS_MODEL_SIZE["$name"]="${h_line:-unknown}"
    echo -e "    ${GREEN}✓${NC} ${tok_s} tok/s, ${time_ms}ms (${elapsed}s wall)"
    PASS=$((PASS + 1))
  elif echo "$output" | grep -qiE "error|panic|segmentation|not found|no such"; then
    err=$(echo "$output" | grep -iE "error|panic|segmentation|not found" | head -1)
    RESULTS_STATUS["$name"]="❌"
    RESULTS_ARCH["$name"]="${arch:-auto}"
    RESULTS_MODEL_SIZE["$name"]="${h_line:-unknown}"
    echo -e "    ${RED}✗${NC} ${err:0:100}"
    FAIL=$((FAIL + 1))
  else
    RESULTS_STATUS["$name"]="⚠️"
    RESULTS_ARCH["$name"]="${arch:-auto}"
    RESULTS_MODEL_SIZE["$name"]="${h_line:-unknown}"
    echo -e "    ${YELLOW}⚠${NC} No result line (possibly timed out or no GPU)"
    SKIP=$((SKIP + 1))
  fi
done

# ── Output Markdown table ──
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}Benchmark Results${NC}"
echo ""

cat <<'TABLE_HEADER'
| # | Model | Arch | Tok/s | Time (ms) | Status |
|---|-------|------|-------|-----------|--------|
TABLE_HEADER

idx=0
for name in "${MODELS[@]}"; do
  idx=$((idx + 1))
  ts="${RESULTS_TOK_S[$name]:---}"
  tm="${RESULTS_TIME[$name]:---}"
  st="${RESULTS_STATUS[$name]:--}"
  ar="${RESULTS_ARCH[$name]:--}"
  printf "| %d | %s | %s | %s | %s | %s |\n" "$idx" "$name" "$ar" "$ts" "$tm" "$st"
done

echo ""
echo "---"
echo ""
echo -e "**${GREEN}${PASS} passed${NC}**, **${RED}${FAIL} failed${NC}**, **${YELLOW}${SKIP} skipped${NC}** | ${MODEL_COUNT} total models"
echo ""

# ── Summary: Top 5 by tokens/s ──
echo ""
echo -e "${BOLD}🏆 Top 5 by tokens/second${NC}"
echo "| # | Model | Arch | Tok/s | Status |"
echo "|---|-------|------|-------|--------|"
# Sort by tok/s descending
for name in "${MODELS[@]}"; do
  ts="${RESULTS_TOK_S[$name]:-0}"
  ar="${RESULTS_ARCH[$name]:--}"
  st="${RESULTS_STATUS[$name]:--}"
  echo "${ts}|${name}|${ar}|${st}"
done | sort -t'|' -k1 -rn | head -5 | awk -F'|' '{printf "| %d | %s | %s | %s | %s |\n", NR, $2, $3, $1, $4}'

# ── Failure summary ──
if [[ $FAIL -gt 0 ]]; then
  echo ""
  echo -e "${BOLD}${RED}Failed models${NC}"
  for name in "${MODELS[@]}"; do
    if [[ "${RESULTS_STATUS[$name]:-}" == "❌" ]]; then
      echo "  • ${name} (arch: ${RESULTS_ARCH[$name]:--})"
    fi
  done
fi

# Cleanup
rm -f "${RESULTS_FILE}"
echo ""
echo -e "${BOLD}Done.${NC}"

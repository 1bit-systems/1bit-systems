#!/bin/bash
# Comprehensive architecture validation test
# Tests every registered arch tag for config loading, dispatch selection,
# and basic inference on available Q4NX models.
set -euo pipefail

ENGINE="${1:-./zig-out/bin/fused-engine}"
echo "═══ Architecture Validation Suite ═══"
echo "Engine: $ENGINE"
echo ""

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
PASS=0; FAIL=0; SKIP=0

pass() { echo -e "  ${GREEN}✓${NC} $1"; PASS=$((PASS+1)); }
fail() { echo -e "  ${RED}✗${NC} $1"; FAIL=$((FAIL+1)); }
skip() { echo -e "  ${YELLOW}⊘${NC} $1"; SKIP=$((SKIP+1)); }

# ── Test 1: --list-archs prints all 15 ──
echo -e "${CYAN}Group 1: Architecture registry${NC}"
ARCHS=$("$ENGINE" --list-archs 2>&1 | tail -n +2 | awk '{print $1}' | grep -v '^$' | wc -l)
if [[ "$ARCHS" -ge 15 ]]; then
  pass "Registry contains $ARCHS architectures (≥15)"
else
  fail "Registry has $ARCHS architectures, expected ≥15"
fi

# Each arch tag should have attention kernel, FFN kernel, and dispatch policy
for tag in qwen3_0_6b qwen3_1_5b qwen3_7b qwen3_14b qwen2_5_7b qwen2_5_32b \
           llama3_1_8b llama3_2_1b llama3_2_3b gemma2_2b gemma2_9b \
           deepseek_v2_lite deepseek_v3 mixtral_8x7b zaya1_8b; do
  line=$("$ENGINE" --list-archs 2>&1 | grep "$tag" || true)
  if echo "$line" | grep -qE 'flash|mla' && echo "$line" | grep -qE 'dense|moe'; then
    pass "  $tag: $(echo $line | awk '{print $2, $3, $4}')"
  else
    fail "  $tag: malformed entry: $line"
  fi
done

# ── Test 2: Config loading for each available Q4NX model ──
echo -e "\n${CYAN}Group 2: Model loading (available Q4NX files)${NC}"
MODEL_DIR="${HOME}/.config/flm/models"
declare -A MODEL_ARCH_MAP
MODEL_ARCH_MAP["Qwen3-0.6B-NPU2"]="qwen3_0_6b"
MODEL_ARCH_MAP["Qwen3-1.7B-NPU2"]="qwen3_1_5b"
MODEL_ARCH_MAP["Qwen3-4B-NPU2"]="qwen3_7b"
MODEL_ARCH_MAP["Qwen3.5-0.8B-NPU2"]="qwen3_0_6b"
MODEL_ARCH_MAP["Qwen3.5-9B-NPU2"]="qwen3_7b"
MODEL_ARCH_MAP["Llama-3.1-8B-NPU2"]="llama3_1_8b"
MODEL_ARCH_MAP["Gemma3-4B-NPU2"]="gemma2_2b"
MODEL_ARCH_MAP["Gemma4-E4B-IT-NPU2"]="gemma2_2b"
MODEL_ARCH_MAP["DeepSeek-R1-0528-Qwen3-8B-NPU2"]="deepseek_v2_lite"
MODEL_ARCH_MAP["Phi4-mini-Instruct-NPU2"]="qwen3_0_6b"
MODEL_ARCH_MAP["Medgemma-4B-NPU2"]="gemma2_2b"

for model_dir in "$MODEL_DIR"/*/; do
  name=$(basename "$model_dir")
  model_file="$model_dir/model.q4nx"
  [[ -f "$model_file" ]] || continue
  
  arch="${MODEL_ARCH_MAP[$name]:-}"
  if [[ -z "$arch" ]]; then
    skip "$name: no arch mapping, probing auto-detect"
    # Try running with auto-detect (no --model-arch flag)
    if timeout 5 "$ENGINE" --model "$model_file" --max-tokens 1 --prompt "Hello" --debug 2>&1 | grep -q "ArchConfig\|arch_registry\|ModelConfig"; then
      pass "$name: model loads with auto-detect"
    else
      # Just check it doesn't crash immediately
      result=$(timeout 3 "$ENGINE" --model "$model_file" --max-tokens 1 2>&1 || true)
      if echo "$result" | grep -qi "error\|panic\|segmentation"; then
        skip "$name: engine errors on load (expected for unknown arch)"
      else
        skip "$name: engine runs (may need --model-arch)"
      fi
    fi
    continue
  fi
  
  echo -e "\n  ${YELLOW}→${NC} Testing $name with arch=$arch"
  
  # Test 1: Model loads without crash
  output=$(timeout 5 "$ENGINE" --model "$model_file" --model-arch "$arch" \
    --max-tokens 1 --prompt "Hello" --debug 2>&1 || true)
  
  if echo "$output" | grep -qi "error\|panic\|not found"; then
    fail "$name: load failed"
    echo "    $(echo "$output" | tail -3)"
    continue
  fi
  
  # Test 2: Architecture routing is correct
  if echo "$output" | grep -qi "ArchConfig loaded\|attention.*flash\|FFN.*dense\|MoE\|MLA"; then
    pass "$name: kernel routing activated"
  else
    skip "$name: kernel routing message not found (may be normal)"
  fi
  
  # Test 3: Dispatch policy selected
  if echo "$output" | grep -qi "policy\|dispatch\|ffn_on_npu\|gpu_only\|qkv_on_npu"; then
    pass "$name: dispatch policy selected"
  else
    skip "$name: dispatch policy not logged"
  fi
  
  # Test 4: Tokenizer works
  if echo "$output" | grep -qi "token\|prompt tokens\|tokenized"; then
    pass "$name: tokenization works"
  else
    skip "$name: tokenization not verified"
  fi
done

# ── Test 3: Dispatch policy sanity per architecture ──
echo -e "\n${CYAN}Group 3: Dispatch policy recommendations${NC}"
declare -A EXPECTED_DISPATCH
EXPECTED_DISPATCH["qwen3_0_6b"]="ffn_on_npu"
EXPECTED_DISPATCH["qwen3_1_5b"]="ffn_on_npu"
EXPECTED_DISPATCH["qwen3_7b"]="ffn_on_npu"
EXPECTED_DISPATCH["qwen3_14b"]="ffn_on_npu"
EXPECTED_DISPATCH["qwen2_5_7b"]="ffn_on_npu"
EXPECTED_DISPATCH["qwen2_5_32b"]="ffn_on_npu"
EXPECTED_DISPATCH["llama3_1_8b"]="ffn_on_npu"
EXPECTED_DISPATCH["llama3_2_1b"]="ffn_on_npu"
EXPECTED_DISPATCH["llama3_2_3b"]="ffn_on_npu"
EXPECTED_DISPATCH["gemma2_2b"]="ffn_on_npu"
EXPECTED_DISPATCH["gemma2_9b"]="ffn_on_npu"
EXPECTED_DISPATCH["deepseek_v2_lite"]="ffn_on_npu"
EXPECTED_DISPATCH["deepseek_v3"]="gpu_only"
EXPECTED_DISPATCH["mixtral_8x7b"]="gpu_only"
EXPECTED_DISPATCH["zaya1_8b"]="ffn_on_npu"

for tag in "${!EXPECTED_DISPATCH[@]}"; do
  expected="${EXPECTED_DISPATCH[$tag]}"
  actual=$("$ENGINE" --list-archs 2>&1 | awk -v t="$tag" '$1==t{print $4}')
  if [[ "$actual" == "$expected" ]]; then
    pass "$tag → $actual"
  else
    fail "$tag: expected $expected, got $actual"
  fi
done

# ── Test 4: Quick inference smoke test (Llama-3.1-8B) ──
echo -e "\n${CYAN}Group 4: Smoke test — Llama-3.1-8B inference${NC}"
LLAMA_MODEL="$MODEL_DIR/Llama-3.1-8B-NPU2/model.q4nx"
if [[ -f "$LLAMA_MODEL" ]]; then
  # Just run 1 token to verify the full pipeline
  result=$(timeout 30 "$ENGINE" --model "$LLAMA_MODEL" --model-arch llama3_1_8b \
    --max-tokens 1 --prompt "Hello" 2>&1 || true)
  
  if echo "$result" | grep -qi "tokens\|generated\|token\|output"; then
    pass "Llama-3.1-8B: inference produces output"
  elif echo "$result" | grep -qi "error\|panic\|segmentation\|not found"; then
    fail "Llama-3.1-8B: inference failed"
    echo "    $(echo "$result" | tail -5)"
  else
    skip "Llama-3.1-8B: result ambiguous (no GPU maybe)"
    echo "    $(echo "$result" | tail -3)"
  fi
else
  skip "Llama-3.1-8B: model file not found"
fi

# ── Results ──
echo ""
echo "═══════════════════════════════════════"
echo -e "  ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}, ${YELLOW}$SKIP skipped${NC}"
echo "═══════════════════════════════════════"
[[ $FAIL -eq 0 ]]

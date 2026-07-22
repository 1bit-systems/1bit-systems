#!/bin/bash
# run_trace_validate.sh — NPU vs HuggingFace FP32 end-to-end validation
#
# 1. Generate HF FP32 reference traces (ground truth)
# 2. Run NPU engine with --trace to dump all intermediates
# 3. Compare: NPU output vs HF reference
#
# Usage:
#   sudo bash tools/run_trace_validate.sh
#
# Environment:
#   MODEL_PATH  Path to HF model (default: ~/.cache/.../Qwen3-0.6B)
#   PROMPT      Input text (default: "Hi")
#   NPU_MODEL   Path to Q4NX model (default: ~/.config/flm/.../model.q4nx)
set -euo pipefail

HF_MODEL="${HF_MODEL:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"
NPU_MODEL="${NPU_MODEL:-$HOME/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx}"
PROMPT="${PROMPT:-Hi}"
REF_DIR="/tmp/trace_hf"
NPU_DIR="/tmp/trace_npu"
# shellcheck disable=SC2034
SUMMARY="/tmp/trace_compare.csv"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║   NPU vs HuggingFace FP32 Validation Pipeline          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "HF model:  $HF_MODEL"
echo "NPU model: $NPU_MODEL"
echo "Prompt:    $PROMPT"
echo ""

# ── Step 1: Generate HF FP32 reference ──
echo ""
echo "=== Step 1: Generating HuggingFace FP32 reference ==="
rm -rf "$REF_DIR"
cd "$HOME/1bit-systems"
"$HOME/venv-hf/bin/python3" tools/hf_ref_forward.py \
    --model "$HF_MODEL" \
    --prompt "$PROMPT" \
    --output "$REF_DIR" 2>&1
echo ""
echo "HF reference saved to: $REF_DIR"

# Read token IDs for NPU
TOKENS=$(cat "$REF_DIR/token_ids.txt" | tr '\n' ',')
echo "Tokens for NPU: $TOKENS"

# ── Step 2: Build NPU engine with trace ──
echo ""
echo "=== Step 2: Building NPU engine (trace mode) ==="
cd "$HOME/npu-sandbox/npu-infer/build"
g++ -std=gnu++17 -O0 -g -I../include \
    -o npu_engine_trace ../src/npu_engine_universal.cpp dequant_q4nx.o \
    -lxrt_coreutil -lm -luuid \
    2>&1
echo "  Built: npu_engine_trace"

# ── Step 3: Run NPU engine with --trace ──
echo ""
echo "=== Step 3: Running NPU engine with --trace ==="
rm -rf "$NPU_DIR"
echo "$TOKENS" | tr ',' '\n' > /tmp/trace_tokens.txt

sudo ./npu_engine_trace "$NPU_MODEL" 1 /tmp/trace_tokens.txt --trace "$NPU_DIR" \
    2>&1 | tee /tmp/npu_trace_output.log
echo "NPU trace saved to: $NPU_DIR"

# ── Step 4: Compare hidden states ──
echo ""
echo "=== Step 4: Comparing hidden states ==="
cd "$HOME/1bit-systems"

# Compare per-layer hidden states
echo ""
echo "Per-layer hidden state comparison (NPU h_out vs HF hidden_state):"
printf "  %-8s %-12s %-12s %-12s %s\n" "Layer" "|NPU|" "|HF|" "Cos Sim" "Max Diff"
echo "  " "$(printf '=%.0s' {1..65})"

for l in $(seq 0 27); do
    NPU_FILE="$NPU_DIR/layer_$(printf '%02d' $l)/h_out.f32"
    HF_FILE="$REF_DIR/layer_$(printf '%02d' $l)_hidden.f32"
    
    if [ ! -f "$NPU_FILE" ]; then
        printf "  L%-5s  NPU trace missing\n" "$l"
        continue
    fi
    if [ ! -f "$HF_FILE" ]; then
        printf "  L%-5s  HF ref missing\n" "$l"
        continue
    fi
    
    # Read both binaries
    NPU_DATA=$(python3 -c "
import numpy as np
n = np.fromfile('$NPU_FILE', dtype=np.float32)
h = np.fromfile('$HF_FILE', dtype=np.float32)
n_norm = float(np.linalg.norm(n))
h_norm = float(np.linalg.norm(h))
cos = float(np.dot(n, h) / (n_norm * h_norm + 1e-30))
max_diff = float(np.max(np.abs(n - h)))
print(f'{n_norm:.4f} {h_norm:.4f} {cos:.6f} {max_diff:.6e}')
" 2>&1)
    
    read n_norm h_norm cos max_diff <<< "$NPU_DATA"
    
    STATUS=""
    if (( $(echo "$cos < 0.99" | bc -l 2>/dev/null || echo 1) )); then
        STATUS="⚠️ DIVERGE"
    elif (( $(echo "$cos < 0.999" | bc -l 2>/dev/null || echo 0) )); then
        STATUS="~close"
    else
        STATUS="OK"
    fi
    
    printf "  L%-5s  %-12.4f %-12.4f %-12.6f %-12.2e %s\n" \
        "$l" "$n_norm" "$h_norm" "$cos" "$max_diff" "$STATUS"
done

# Compare final norm
echo ""
echo "Final norm comparison:"
NPU_FINAL="$NPU_DIR/final_norm_output.f32"
HF_FINAL="$REF_DIR/final_norm_output.f32"
if [ -f "$NPU_FINAL" ] && [ -f "$HF_FINAL" ]; then
    python3 -c "
import numpy as np
n = np.fromfile('$NPU_FINAL', dtype=np.float32)
h = np.fromfile('$HF_FINAL', dtype=np.float32)
n_norm = np.linalg.norm(n)
h_norm = np.linalg.norm(h)
cos = np.dot(n, h) / (n_norm * h_norm + 1e-30)
max_diff = np.max(np.abs(n - h))
print(f'  NPU norm={n_norm:.4f}  HF norm={h_norm:.4f}  cos_sim={cos:.6f}  max_diff={max_diff:.6e}')
if cos > 0.99:
    print('  ✅ Final norm MATCHES')
else:
    print('  ⚠️  Final norm DIVERGES')
"
fi

# Compare logits
echo ""
echo "Logit comparison:"
NPU_LOGITS="$NPU_DIR/logits.f32"
HF_LOGITS="$REF_DIR/logits.f32"
if [ -f "$NPU_LOGITS" ] && [ -f "$HF_LOGITS" ]; then
    python3 -c "
import numpy as np
n = np.fromfile('$NPU_LOGITS', dtype=np.float32)
h = np.fromfile('$HF_LOGITS', dtype=np.float32)
n_norm = np.linalg.norm(n)
h_norm = np.linalg.norm(h)
cos = np.dot(n, h) / (n_norm * h_norm + 1e-30)
max_diff = np.max(np.abs(n - h))
n_top5 = np.argsort(n)[-5:][::-1]
h_top5 = np.argsort(h)[-5:][::-1]
print(f'  NPU norm={n_norm:.4f}  HF norm={h_norm:.4f}  cos_sim={cos:.6f}  max_diff={max_diff:.6e}')
print(f'  NPU top-5 tokens: {n_top5.tolist()}')
print(f'  HF  top-5 tokens: {h_top5.tolist()}')
match = len(set(n_top5.tolist()) & set(h_top5.tolist()))
print(f'  Top-5 overlap: {match}/5')
if cos > 0.99:
    print('  ✅ Logits MATCH')
elif cos > 0.9:
    print('  ⚠️  Logits close but not exact')
else:
    print('  ❌ Logits DIVERGE')
"
fi

echo ""
echo "=== Done ==="

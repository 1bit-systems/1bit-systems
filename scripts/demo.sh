#!/usr/bin/env bash
# 1bit.systems — One Binary Demo
# Run: bash scripts/demo.sh
# Records terminal session showing: build, model load, inference, watchdog
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="/tmp/1bit-demo-${TIMESTAMP}.log"
MODEL="${DEMO_MODEL:-$HOME/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx}"
NPU_ENGINE="${DEMO_NPU:-$HOME/npu-infer/build/npu_engine_universal}"
TOKENIZER="${DEMO_TOK:-$HOME/.config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json}"

echo "═══════════════════════════════════════════════" | tee -a "$LOGFILE"
echo "  1bit.systems — One Binary Demo" | tee -a "$LOGFILE"
echo "  $(date)" | tee -a "$LOGFILE"
echo "═══════════════════════════════════════════════" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# ── Step 1: System info ──
echo "=== 1. Hardware ===" | tee -a "$LOGFILE"
# TheRock 7.15.0a C++ SDK
THEROCK_BIN="/opt/rocm-therock/bin"
if [ ! -f "$THEROCK_BIN/rocminfo" ]; then
    THEROCK_BIN="/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/bin"
fi
"$THEROCK_BIN/rocminfo" 2>/dev/null | grep "Marketing Name" | head -2 | tee -a "$LOGFILE"
echo "GPU temp: $(cat /sys/class/drm/card1/device/hwmon/hwmon*/temp1_input 2>/dev/null | head -1 | awk '{print $1/1000 "°C"}')" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# ── Step 2: Build ──
echo "=== 2. Build ===" | tee -a "$LOGFILE"
echo "  cmake --build build --target zaya_server -j\$(nproc)" | tee -a "$LOGFILE"
cmake --build build --target zaya_server -j"$(nproc)" 2>&1 | tail -3 | tee -a "$LOGFILE"
echo "  Binary: $(ls -lh build/zaya_server | awk '{print $5}')" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# ── Step 3: Tests ──
echo "=== 3. Tests ===" | tee -a "$LOGFILE"
PASS=0; FAIL=0; for t in build/test_*; do
  name=$(basename $t)
  result=$(timeout 20 $t 2>&1 | grep -E "PASS|FAIL|Verdict" | tail -1)
  if echo "$result" | grep -q "PASS"; then
    echo "  ✅ $name" | tee -a "$LOGFILE"; PASS=$((PASS + 1))
  else
    echo "  ➖ $name" | tee -a "$LOGFILE"; FAIL=$((FAIL + 1))
  fi
done
echo "  $PASS/$((PASS+FAIL)) tests passed" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# ── Step 4: Kernel benchmarks ──
echo "=== 4. Kernel Benchmarks ===" | tee -a "$LOGFILE"
for bench in bench_bonsai_q1_1024 bench_fused_tq2_1024; do
  if [ -f "build/$bench" ]; then
    result=$(timeout 15 build/$bench 2>&1 | grep -E "Throughput|tok/s" | head -2)
    echo "  $bench: $result" | tee -a "$LOGFILE"
  fi
done
echo "" | tee -a "$LOGFILE"

# ── Step 5: Fused Engine Inference ──
echo "=== 5. Fused Engine ===" | tee -a "$LOGFILE"
if [ -f "$MODEL" ] && [ -x "$NPU_ENGINE" ] && [ -f "$TOKENIZER" ]; then
  echo "  Model: $(basename $MODEL)" | tee -a "$LOGFILE"
  echo "  Generating 10 tokens..." | tee -a "$LOGFILE"
  command time -f "  Time: %e seconds" \
    ./engine/fusion/zig-out/bin/fused-engine \
    -m "$MODEL" \
    --npu-engine "$NPU_ENGINE" \
    --tokenizer "$TOKENIZER" \
    -n 10 -p "The capital of France is" 2>&1 | \
    grep -E "GPU.*ready|Generating|tokens in|tok/s|error" | head -10 | tee -a "$LOGFILE"
else
  echo "  ⚠️ Model/NPU engine not found — running help instead" | tee -a "$LOGFILE"
  ./engine/fusion/zig-out/bin/fused-engine --help 2>&1 | head -5 | tee -a "$LOGFILE"
fi
echo "" | tee -a "$LOGFILE"

# ── Step 6: GGUF Loader ──
echo "=== 6. GGUF Model Loader ===" | tee -a "$LOGFILE"
GGUF_MODEL="${GGUF_MODEL:-$HOME/models/tinylama-1.1b-q4.gguf}"
cat > /tmp/test_gguf_demo.cpp << EOF
#include "rocm_cpp/bitnet_model.h"
#include <cstdio>
int main() {
    rcpp_bitnet_model_t model = {};
    rcpp_status_t st = rcpp_bitnet_load_gguf("${GGUF_MODEL}", &model);
    printf("st=%d H=%d L=%d NH=%d NKV=%d V=%d emb=%p\n", (int)st,
           model.hidden_size, model.num_layers, model.num_heads,
           model.num_kv_heads, model.vocab_size, model.embedding_dev);
    if (st == RCPP_OK) rcpp_bitnet_free(&model);
    return st == RCPP_OK ? 0 : 1;
}
EOF
g++ -std=c++17 -I include -O2 /tmp/test_gguf_demo.cpp -o /tmp/test_gguf_demo \
  -L build -lrocm_cpp -L/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_devel/lib -lamdhip64 -Wl,-rpath,/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_devel/lib 2>&1 | tail -1
LD_LIBRARY_PATH=build:/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_devel/lib timeout 15 /tmp/test_gguf_demo 2>&1 | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# ── Step 7: Server health check ──
echo "=== 7. HTTP API Server ===" | tee -a "$LOGFILE"
echo "  Binary: build/zaya_server ($(ls -lh build/zaya_server | awk '{print $5}'))" | tee -a "$LOGFILE"
echo "  Start: ./build/zaya_server --port 8088" | tee -a "$LOGFILE"
echo "  Send: curl http://localhost:8088/v1/models" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# ── Summary ──
echo "═══════════════════════════════════════════════" | tee -a "$LOGFILE"
echo "  DEMO COMPLETE" | tee -a "$LOGFILE"
echo "  Log: $LOGFILE" | tee -a "$LOGFILE"
echo "═══════════════════════════════════════════════" | tee -a "$LOGFILE"

# Copy log to site for the demo page
mkdir -p site/demo
cp "$LOGFILE" site/demo/latest.log
echo "Demo log copied to site/demo/latest.log"

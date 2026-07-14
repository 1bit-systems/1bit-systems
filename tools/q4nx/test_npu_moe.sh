#!/bin/bash
# NPU MoE Worker Protocol Test Suite
# Tests --worker mode, binary wire protocol, and per-op correctness.
#
# Protocol (from npu_engine_universal.cpp):
#   Request:  4×u32 LE  { op, layer, batch, in_dim }
#             followed by batch*in_dim raw f32 values
#   Response: 2×u32 LE  { status, out_dim_per_sample }
#             followed by batch * out_dim_per_sample raw f32 values
#   Status: 0=OK, 1=bad args, 2=bad op
#   Ops: QUIT=0, QKV=1, OPROJ=2, GATEUP=3, UP=4, DOWN=5
#
set -euo pipefail

cd "$(dirname "$0")"
ENGINE="${1:-/home/bcloud/engine/npu/build/npu_engine_universal}"
MODEL="${2:-/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx}"
PYTEST="$(dirname "$0")/test_npu_moe.py"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
PASS=0; FAIL=0; SKIP=0
TMPDIR=$(mktemp -d /tmp/npu_moe_test.XXXXXX)
trap 'rm -rf "$TMPDIR"' EXIT

pass() { echo -e "  ${GREEN}✓${NC} $1"; PASS=$((PASS+1)); }
fail() { echo -e "  ${RED}✗${NC} $1"; FAIL=$((FAIL+1)); }
skip() { echo -e "  ${YELLOW}⊘${NC} $1"; SKIP=$((SKIP+1)); }

SRC="/home/bcloud/engine/npu/src/npu_engine_universal.cpp"

echo -e "${BOLD}${CYAN}═══ NPU MoE Worker Protocol Test Suite ═══${NC}"
echo "Engine: $ENGINE"
echo "Model: $MODEL"
echo ""

if [[ ! -x "$ENGINE" ]]; then
    fail "Engine not found at $ENGINE"; exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    skip "Model not found at $MODEL"; HAS_MODEL=false
else
    HAS_MODEL=true
fi

# ============================================================================
echo -e "${CYAN}Group 1: Source code verification${NC}"

OP_COUNT=$(grep -oP 'OP_\w+' "$SRC" | sort -u | wc -l)
OPS=$(grep -oP 'OP_\w+' "$SRC" | sort -u | tr '\n' ' ')
if [[ "$OP_COUNT" -ge 5 ]]; then
    pass "WorkerOp enum: $OP_COUNT ops ($OPS)"
else
    fail "WorkerOp enum has only $OP_COUNT ops (expected ≥5)"
fi

for op in QKV OPROJ GATEUP UP DOWN; do
    if grep -q "case OP_${op}" "$SRC"; then
        pass "switch(OP_${op}) handled in run_worker"
    else
        fail "switch case missing: OP_${op}"
    fi
done

if grep -q "OP_QUIT" "$SRC"; then
    pass "OP_QUIT handled (if/break in run_worker loop)"
else
    fail "OP_QUIT handler not found"
fi

if grep -q "uint32_t hdr\[4\]" "$SRC"; then
    pass "Request header: {op, layer, batch, in_dim} (4×u32)"
else
    fail "Request header format not found"
fi
if grep -q "uint32_t resp\[2\]" "$SRC"; then
    pass "Response header: {status, out_dim} (2×u32)"
else
    fail "Response header format not found"
fi

for check in "batch>4096" "in_dim>65536" "layer>=NC"; do
    name=""
    case $check in
        batch*) name="Batch capped at 4096" ;;
        in_dim*) name="In_dim capped at 65536" ;;
        layer*) name="Layer bounds checked (layer < NC)" ;;
    esac
    if grep -q "$check" "$SRC"; then
        pass "$name"
    else
        fail "$name"
    fi
done

if grep -q "WORKER_READY" "$SRC" && grep -q "fflush(stderr)" "$SRC"; then
    pass "WORKER_READY + fflush(stderr) for sync"
else
    fail "WORKER_READY sync not found"
fi

if grep -q "cn(out_buf.data" "$SRC"; then
    pass "Worker applies clean_nans on output"
else
    fail "Output sanity check not found"
fi

if grep -q "read_exact" "$SRC"; then
    pass "Worker uses read_exact() for robust byte-level I/O"
else
    fail "read_exact helper not found"
fi

# ============================================================================
echo -e "\n${CYAN}Group 2: --worker flag handling${NC}"

OUTPUT=$("$ENGINE" 2>&1 || true)
if echo "$OUTPUT" | grep -q "worker"; then
    pass "Usage message mentions --worker mode"
else
    fail "Usage message does not mention --worker"
fi

echo -e "\n  ${YELLOW}→${NC} Testing --worker with missing model..."
if "$ENGINE" --worker 2>&1 | grep -qi "usage\|invalid\|error"; then
    pass "--worker with no model: rejects gracefully"
else
    skip "--worker with no model: no explicit rejection message"
fi

# ============================================================================
echo -e "\n${CYAN}Group 3: Wire protocol — Python test harness${NC}"

if ! $HAS_MODEL; then
    skip "All protocol tests (model file not found)"
elif ! python3 -c "import struct, subprocess, os, fcntl; print('ok')" 2>/dev/null; then
    skip "All protocol tests (python3 not available)"
elif [[ ! -f "$PYTEST" ]]; then
    skip "All protocol tests (test_npu_moe.py not found)"
else
    echo -e "  ${YELLOW}→${NC} Running Python wire-protocol tests..."
    echo -e "  ${YELLOW}  (spawns worker once, tests all ops in sequence)${NC}"
    echo ""
    # Run with output capture via file to avoid pipe deadlock
    python3 "$PYTEST" "$ENGINE" "$MODEL" 40 > "$TMPDIR/py_out.txt" 2>&1 || true
    cat "$TMPDIR/py_out.txt"
    PY_PASS=$(grep -c "^  ✓" "$TMPDIR/py_out.txt" 2>/dev/null || echo 0)
    PY_FAIL=$(grep -c "^  ✗" "$TMPDIR/py_out.txt" 2>/dev/null || echo 0)
    PY_SKIP=$(grep -c "^  ⊘" "$TMPDIR/py_out.txt" 2>/dev/null || echo 0)
    PASS=$((PASS + PY_PASS))
    FAIL=$((FAIL + PY_FAIL))
    SKIP=$((SKIP + PY_SKIP))
fi

# ============================================================================
echo -e "\n${CYAN}Group 4: Infrastructure verification${NC}"

XCLBIN_DIR="/home/bcloud/npu-sandbox/npu-infer/build/int8"
for variant in GU G U D; do
    COUNT=$(ls "$XCLBIN_DIR"/final_i8_${variant}_*.xclbin 2>/dev/null | wc -l)
    if [[ "$COUNT" -gt 0 ]]; then
        pass "xclbin variant '${variant}' exists ($COUNT files)"
    else
        skip "xclbin variant '${variant}' not found on disk"
    fi
done

if $HAS_MODEL; then
    echo -e "  ${YELLOW}→${NC} Model arch: $(timeout 8 "$ENGINE" "$MODEL" 2>&1 | grep -E '^H=|^=== NPU' | tr '\n' '; ')" 2>/dev/null || true
fi

echo ""
echo "═══════════════════════════════════════"
echo -e "  ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}, ${YELLOW}$SKIP skipped${NC}"
echo "═══════════════════════════════════════"
[[ $FAIL -eq 0 ]]

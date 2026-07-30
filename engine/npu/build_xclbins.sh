#!/bin/bash
# Build all model xclbins using the proven make flow (n1_core_placed.py)
#
# Usage:
#   ./build_xclbins.sh [--verify] [model_tag]
#
# Examples:
#   ./build_xclbins.sh                      # Build all models
#   ./build_xclbins.sh qwen3_0_6b           # Build one model
#   ./build_xclbins.sh --verify qwen3_0_6b  # Build + run verification
#
# Models: qwen3_0_6b qwen3_8b qwen3_vl_4b llama gemma4_e2b
#
# Environment:
#   AIE_TOOLS_DIR   — MLIR-AIE toolchain root (default: ~/mlir-aie/install_tmp)
#   TORCH2AIE_DIR   — torch2aie repo root (default: ~/torch2aie)
#   INT8_DIR        — Output directory for xclbins (default: ~/npu-sandbox/npu-infer/build/int8)
#
# Known-good toolchain setup (as of 2026-07-29):
#   export AIE_TOOLS_DIR=~/mlir-aie/install_tmp
#   export PATH=$AIE_TOOLS_DIR/bin:$PATH
#   export PYTHONPATH=$AIE_TOOLS_DIR/python:$PYTHONPATH

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GENERATORS_DIR="${SCRIPT_DIR}/generators"
TORCH2AIE_DIR="${TORCH2AIE_DIR:-${HOME}/torch2aie}"
AIE_TOOLS_DIR="${AIE_TOOLS_DIR:-${TORCH2AIE_DIR}/toolchain}"

# Source the fix_toolchain.sh helper if available (defines functions only)
if [ -f "${GENERATORS_DIR}/fix_toolchain.sh" ]; then
    # shellcheck source=generators/fix_toolchain.sh
    # We only source the file to get function definitions; don't trigger auto-run
    source "${GENERATORS_DIR}/fix_toolchain.sh" 2>/dev/null || true
fi
CFG1="${TORCH2AIE_DIR}/examples/gemm_asymmetric_tile_buffering/config1"
INT8_DIR="${INT8_DIR:-${HOME}/npu-sandbox/npu-infer/build/int8}"
mk_dir="$CFG1/build"

# ── Environment Checks ───────────────────────────────────────────────

check_env() {
    local errors=0

    # Check torch2aie repo
    if [ ! -d "$TORCH2AIE_DIR" ]; then
        echo "ERROR: torch2aie directory not found at: $TORCH2AIE_DIR"
        echo "  Set TORCH2AIE_DIR or clone the repo:"
        echo "    git clone https://github.com/bong-water-water-bong/torch2aie ~/torch2aie"
        errors=$((errors + 1))
    fi

    # Check MLIR-AIE toolchain binaries
    if [ ! -d "$AIE_TOOLS_DIR/bin" ]; then
        echo "ERROR: MLIR-AIE toolchain not found at: $AIE_TOOLS_DIR"
        echo "  Set AIE_TOOLS_DIR or download the toolchain:"
        echo "    cd ~/torch2aie && ./toolchain/download.sh"
        errors=$((errors + 1))
    else
        # Verify key tools exist
        local aiecc_path="$AIE_TOOLS_DIR/bin/aiecc"
        if [ ! -x "$aiecc_path" ]; then
            echo "ERROR: aiecc not found or not executable at $aiecc_path"
            echo "  Make sure PATH includes \$AIE_TOOLS_DIR/bin"
            errors=$((errors + 1))
        else
            # Check aiecc version for opaque pointer support
            local aiecc_ver
            aiecc_ver=$("$aiecc_path" --version 2>/dev/null | head -1 || echo "unknown")
            echo "  aiecc version: $aiecc_ver"
            # Check if aiecc supports -opaque-pointers
            if "$aiecc_path" --help 2>/dev/null | grep -q 'opaque'; then
                echo "  aiecc: opaque-pointer support detected ✓"
            else
                echo "  aiecc: legacy pointer mode (no -opaque-pointers flag)"
                echo "         This is fine — fix_toolchain.sh handles the conversion."
            fi
        fi
        if ! command -v "$AIE_TOOLS_DIR/bin/make" &>/dev/null && ! command -v make &>/dev/null; then
            echo "ERROR: 'make' not found. Install build-essential."
            errors=$((errors + 1))
        fi
    fi

    # Check for v24 generator
    if [ -f "${GENERATORS_DIR}/n1_core_i8_v24.py" ]; then
        echo "  v24 MLIR generator found at ${GENERATORS_DIR}/n1_core_i8_v24.py ✓"
    fi

    # Check for fix_toolchain.sh
    if [ -f "${GENERATORS_DIR}/fix_toolchain.sh" ]; then
        echo "  fix_toolchain.sh found ✓"
    fi

    # Check config1 directory
    if [ -n "$TORCH2AIE_DIR" ] && [ ! -d "$CFG1" ]; then
        echo "WARNING: GEMM config directory not found at: $CFG1"
        echo "  The Makefile-based build flow may not work."
        echo "  Expected path: examples/gemm_asymmetric_tile_buffering/config1/"
        ls "${TORCH2AIE_DIR}/examples/" 2>/dev/null | head -10 || true
    fi

    # Check xclbin output dir
    mkdir -p "$INT8_DIR" || {
        echo "ERROR: Cannot create output directory: $INT8_DIR"
        echo "  Set INT8_DIR to a writable location."
        errors=$((errors + 1))
    }

    if [ $errors -gt 0 ]; then
        echo ""
        echo "Aborting: $errors error(s) found. Fix them and re-run."
        exit 1
    fi

    echo "Environment checks passed."
    echo "  TORCH2AIE_DIR: $TORCH2AIE_DIR"
    echo "  AIE_TOOLS_DIR: $AIE_TOOLS_DIR"
    echo "  INT8_DIR:      $INT8_DIR"
    echo ""
}

# ── Verification ─────────────────────────────────────────────────────

run_verify() {
    local tag="$1"
    local xclbin_dir="$INT8_DIR"

    echo "  Verifying xclbins for $tag..."

    # Check xclbin files exist
    local count
    count=$(ls "${xclbin_dir}"/final_i8_*_${tag}.xclbin 2>/dev/null | wc -l)
    if [ "$count" -eq 0 ]; then
        echo "  ✗ FAIL: No xclbins found for $tag"
        return 1
    fi
    echo "  ✓ $count xclbins present"

    # Check instruction files exist for each xclbin
    local missing_insts=0
    for xc in "${xclbin_dir}"/final_i8_*_${tag}.xclbin; do
        local label
        label=$(basename "$xc" | sed "s/final_i8_//; s/_${tag}.xclbin//")
        local inst="${xclbin_dir}/insts_i8_${label}_${tag}.txt"
        if [ ! -f "$inst" ]; then
            echo "  ✗ FAIL: Missing instruction file for $label: $inst"
            missing_insts=$((missing_insts + 1))
        fi
    done
    if [ "$missing_insts" -gt 0 ]; then
        return 1
    fi
    echo "  ✓ All instruction files present"

    # Verify xclbin sizes are non-zero and reasonable
    for xc in "${xclbin_dir}"/final_i8_*_${tag}.xclbin; do
        local size
        size=$(stat -c%s "$xc" 2>/dev/null || echo 0)
        if [ "$size" -lt 1000 ]; then
            echo "  ✗ FAIL: $(basename "$xc") is too small: $size bytes"
            return 1
        fi
    done
    echo "  ✓ All xclbin sizes are valid"

    # If bench_gemm binary exists, run it
    local bench_bin
    bench_bin=$(find "${SCRIPT_DIR}/build_cmake" -name "bench_gemm" -type f 2>/dev/null | head -1)
    if [ -n "$bench_bin" ] && [ -x "$bench_bin" ]; then
        echo "  Running bench_gemm verification..."
        NPU_XCLBIN_DIR="$xclbin_dir" "$bench_bin" 2>&1 || {
            echo "  ✗ FAIL: bench_gemm exited with error"
            return 1
        }
        echo "  ✓ bench_gemm passed"
    else
        echo "  ⚡ bench_gemm not found — skipping runtime verification"
        echo "    To enable, build with: cd engine/npu && mkdir -p build_cmake && cd build_cmake && cmake .. && make bench_gemm"
    fi

    echo "  ✓ Verification passed for $tag"
    return 0
}

# ── Helper: parse args ──────────────────────────────────────────────

DO_VERIFY=false
DO_V24=false
DO_GEN_MLIR=false
REQUESTED_TAG=""

while [ $# -gt 0 ]; do
    case "$1" in
        --verify|-v)
            DO_VERIFY=true
            shift
            ;;
        --v24)
            DO_V24=true
            echo "  Using v24 MLIR generator (BD descriptor pipelining)"
            shift
            ;;
        --generate-mlir-only)
            DO_GEN_MLIR=true
            shift
            ;;
        --help|-h)
            sed -n '2,22p' "$0" | sed 's/^#//'
            echo ""
            echo "Flags:"
            echo "  --verify, -v         Build + run verification"
            echo "  --v24                Use v24 MLIR generator (BD pipelining)"
            echo "  --generate-mlir-only Generate MLIR only (no aiecc compilation)"
            echo "  --help, -h           Show this help"
            exit 0
            ;;
        *)
            if [ -z "$REQUESTED_TAG" ]; then
                REQUESTED_TAG="$1"
            else
                echo "ERROR: Unexpected argument: $1"
                echo "Usage: ./build_xclbins.sh [--verify] [--v24] [model_tag]"
                exit 1
            fi
            shift
            ;;
    esac
done

# If --v24 mode, set flag for build function
if [ "$DO_V24" = true ]; then
    export NPU_USE_V24=1
fi

# ── Main Build ───────────────────────────────────────────────────────

# Handle --generate-mlir-only (no environment needed)
if [ "$DO_GEN_MLIR" = true ]; then
    echo "=== MLIR Generation Only Mode ==="
    echo ""
    if [ -f "${GENERATORS_DIR}/n1_core_i8_v24.py" ]; then
        echo "Generating v24 MLIR (BD pipelined)..."
        echo "Usage: python3 ${GENERATORS_DIR}/n1_core_i8_v24.py -M 128 -K 1024 -N 4096 > output.mlir"
        echo ""
        echo "Example for D projection:"
        echo "  python3 ${GENERATORS_DIR}/n1_core_i8_v24.py -M 128 -K 3072 -N 1024 > d_v24.mlir"
        exit 0
    else
        echo "ERROR: v24 generator not found at ${GENERATORS_DIR}/n1_core_i8_v24.py"
        exit 1
    fi
fi

# Run environment checks first
echo "=== build_xclbins.sh — Environment Check ==="
check_env

# Set up toolchain paths
export PATH="${AIE_TOOLS_DIR}/bin:$PATH"
export PYTHONPATH="${AIE_TOOLS_DIR}/python:$PYTHONPATH"
mkdir -p "$INT8_DIR" "$mk_dir"

build() {
    local tag=$1 M=$2 K=$3 N=$4 label=$5
    local target="${mk_dir}/final_${M}x${K}x${N}_128x64x128.xclbin"
    local insts_target="${mk_dir}/insts_${M}x${K}x${N}_128x64x128.txt"
    local out_xclbin="${INT8_DIR}/final_i8_${label}_${tag}.xclbin"
    local out_insts="${INT8_DIR}/insts_i8_${label}_${tag}.txt"

    # If output already exists and matches expected size, skip
    if [ -f "$out_xclbin" ] && [ -f "$out_insts" ]; then
        echo "  ✓ $label (cached, $(stat -c%s "$out_xclbin") bytes)"
        return 0
    fi

    if [ ! -f "$target" ]; then
        echo "  Building ${M}x${K}x${N} for $tag/$label (torch2aie Makefile flow)..."

        # ── Step 1: Generate MLIR (try v24 generator first, then torch2aie) ──
        local gen_mlir="${mk_dir}/${M}x${K}x${N}.mlir"
        if [ -f "${GENERATORS_DIR}/n1_core_i8_v24.py" ]; then
            echo "    Generating MLIR via v24 generator (BD pipelined)..."
            if ! python3 "${GENERATORS_DIR}/n1_core_i8_v24.py" \
                -M "$M" -K "$K" -N "$N" -m 128 -k 64 -n 128 \
                > "$gen_mlir" 2>"${mk_dir}/${M}x${K}x${N}_gen.log"; then
                echo "    ⚡ v24 MLIR generation failed — see ${mk_dir}/${M}x${K}x${N}_gen.log"
                echo "    Falling back to torch2aie Makefile flow..."
            else
                echo "    ✓ MLIR generated: $(wc -l < "$gen_mlir") lines"

                # ── Step 2: Fix opaque pointer issues in generated IR ──
                if [ -f "${GENERATORS_DIR}/fix_toolchain.sh" ] && [ "${NPU_FIX_TOOLCHAIN:-0}" = "1" ]; then
                    echo "    Fixing opaque-pointer IR..."
                    "${GENERATORS_DIR}/fix_toolchain.sh" --fix "$gen_mlir" || true
                fi

                # ── Step 3: Compile MLIR → xclbin via aiecc ──
                # Note: This requires the torch2aie Makefile to be adapted for
                # external MLIR input. If aiecc is available directly, we use it.
                # Otherwise, we fall through to the Makefile path.
                local aiecc_path="$AIE_TOOLS_DIR/bin/aiecc"
                if [ -x "$aiecc_path" ] && [ -f "${CFG1}/Makefile" ]; then
                    # Use torch2aie Makefile with the generated MLIR
                    echo "    Compiling via aiecc (torch2aie Makefile)..."

                    # Copy MLIR into the build tree for the Makefile
                    local build_mlir="${mk_dir}/n1_core_${M}x${K}x${N}.mlir"
                    cp "$gen_mlir" "$build_mlir"
                fi
            fi
        fi

        # ── Step 4: torch2aie Makefile flow (fallback) ──
        echo "    Running torch2aie make (M=${M} K=${K} N=${N})..."
        local make_log="${mk_dir}/make_${M}x${K}x${N}.log"
        if ! make -C "$CFG1" "M=$M" "K=$K" "N=$N" m=128 k=64 n=128 use_placed=1 targetname=n1_core \
            aie_py_src=n1_core_placed.py \
            "build/final_${M}x${K}x${N}_128x64x128.xclbin" > "$make_log" 2>&1; then
            local make_exit=$?
            echo "  ✗ FAIL: make failed for ${M}x${K}x${N} ($label)"
            echo "    Exit code: $make_exit"
            echo "    Last 20 lines of build log:"
            tail -20 "$make_log" | sed 's/^/      /'
            echo ""
            echo "    Full build log: $make_log"
            echo "    Common issues:"
            echo "      1. Opaque pointer LLVM IR mismatch — run fix_toolchain.sh"
            echo "         ${GENERATORS_DIR}/fix_toolchain.sh --fix ${mk_dir}"
            echo "      2. Missing Peano compiler — set PEANO_DIR"
            echo "      3. Outdated toolchain — re-run torch2aie/toolchain/download.sh"
            return 1
        fi
        local size
        size=$(stat -c%s "$target" 2>/dev/null || echo 0)
        echo "    Done: $size bytes"

        # ── Step 5: Post-build opaque pointer fix (retry if needed) ──
        if [ -f "${GENERATORS_DIR}/fix_toolchain.sh" ]; then
            "${GENERATORS_DIR}/fix_toolchain.sh" --fix "${mk_dir}" 2>/dev/null || true
        fi
    else
        echo "  Using cached ${M}x${K}x${N}"
    fi

    # ── Step 6: Copy outputs to INT8_DIR ──
    if [ ! -f "$target" ]; then
        echo "  ✗ FAIL: Target xclbin not found after build: $target"
        return 1
    fi

    cp "$target" "$out_xclbin"
    if [ -f "$insts_target" ]; then
        cp "$insts_target" "$out_insts"
        echo "  ✓ $label: $(stat -c%s "$out_xclbin") bytes, $(wc -l < "$out_insts") insts"
    else
        echo "  ⚡ $label: xclbin copied but instruction file missing: $insts_target"
        touch "$out_insts"
    fi
}

# ── Model Definitions ────────────────────────────────────────────────

# Each model defines GEMM dimensions based on its architecture.
# GEMM labels:
#   QKV: M×H×(NH·HD + 2·NKV·HD)  — query/key/value fused projection
#   O:   M×(NH·HD)×H              — attention output projection
#   GU:  M×H×(2·IM)               — fused gate+up (split into G U if >14336)
#   D:   M×IM×H                   — down projection

build_qwen3_0_6b() {
    echo "=== Qwen3-0.6B (H=1024, NH=16, NKV=8, HD=128, IM=3072) ==="
    local TAG="qwen3_0_6b"
    build "$TAG" 128 1024 4096   QKV
    build "$TAG" 128 2048 1024   O
    build "$TAG" 128 1024 6144   GU
    build "$TAG" 128 3072 1024   D
}

build_qwen3_8b() {
    echo "=== Qwen3-8B (H=4096, NH=32, NKV=8, HD=128, IM=12288) ==="
    local TAG="qwen3_8b"
    build "$TAG" 128 4096 6144   QKV
    build "$TAG" 128 4096 4096   O
    build "$TAG" 128 4096 12288  G
    build "$TAG" 128 4096 12288  U
    build "$TAG" 128 12288 4096  D
}

build_qwen3_vl_4b() {
    echo "=== Qwen3-VL-4B (H=2560, NH=32, NKV=8, HD=128, IM=9728) ==="
    local TAG="qwen3_vl_4b"
    build "$TAG" 128 2560 6144   QKV
    build "$TAG" 128 4096 2560   O
    build "$TAG" 128 2560 9728   G
    build "$TAG" 128 2560 9728   U
    build "$TAG" 128 9728 2560   D
}

build_llama() {
    echo "=== Llama-3.1-8B (H=4096, NH=32, NKV=8, HD=128, IM=14336) ==="
    local TAG="llama"
    build "$TAG" 128 4096 6144   QKV
    build "$TAG" 128 4096 4096   O
    build "$TAG" 128 4096 14336  G
    build "$TAG" 128 4096 14336  U
    build "$TAG" 128 14336 4096  D
}

build_gemma4_e2b() {
    echo "=== Gemma4-E2B (H=1536, NH=8, NKV=1, HD=256, IM=6144) ==="
    local TAG="gemma4_e2b"
    # Note: actual Q4NX file has H=1536 (not 3584 as in model name)
    build "$TAG" 128 1536 2560   QKV
    build "$TAG" 128 2048 1536   O
    build "$TAG" 128 1536 12288  GU
    build "$TAG" 128 6144 1536   D
}

# ── NEW ARCHITECTURES ────────────────────────────────────────────────

build_mistral_7b() {
    echo "=== Mistral-7B (H=4096, NH=32, NKV=8, HD=128, IM=14336, SWA) ==="
    echo "  Note: GEMM dims match Llama but attn uses sliding window (needs SWA xclbin)"
    local TAG="mistral_7b"
    build "$TAG" 128 4096 6144   QKV
    build "$TAG" 128 4096 4096   O
    build "$TAG" 128 4096 14336  G
    build "$TAG" 128 4096 14336  U
    build "$TAG" 128 14336 4096  D
}

build_falcon_7b() {
    echo "=== Falcon-7B (H=4544, NH=71, NKV=1, HD=64, IM=18176) ==="
    echo "  Note: H=4544 is NOT multiple of 128 — uses padded dims (4608)"
    local TAG="falcon_7b"
    # Padded to nearest multiple of 128: H=4608, NH=72, NKV=1, HD=64
    # QKV = 4608× (72*64 + 2*1*64) = 4608×4736
    build "$TAG" 128 4608 4736   QKV
    # O = 4608×4608
    build "$TAG" 128 4608 4608   O
    # GU = 4608×(2*18176) = 4608×36352 — split to avoid >14336 tile limit
    build "$TAG" 128 4608 18176  G
    build "$TAG" 128 4608 18176  U
    # D = 18176×4608 — padded to 18304×4608
    build "$TAG" 128 18304 4608  D
}

build_olmoe() {
    echo "=== OLMoE-1B-7B (H=2048, NH=16, NKV=16, HD=128, IM=2048, MoE) ==="
    echo "  Note: Reuses GPT-OSS expert pattern: 4 GEMM + expert.xclbin"
    local TAG="olmoe_1b"
    build "$TAG" 128 2048 4096   QKV   # QKV: H→NH*HD+2*NKV*HD = 2048→4096
    build "$TAG" 128 2048 2048   O     # O: NH*HD→H = 2048→2048
    build "$TAG" 128 2048 4096   GU    # GU: H→2*IM = 2048→4096
    build "$TAG" 128 2048 2048   D     # D: IM→H = 2048→2048
}

build_zamba2_2_7b() {
    echo "=== Zamba2-2.7B (H=2560, NH=32, NKV=32, HD=80, IM=5120, SSM hybrid) ==="
    echo "  Note: SSM layers run on CPU; NPU handles GEMM for attn layers + in/out proj"
    local TAG="zamba2_2_7b"
    # Attn layers: QKV = 2560×7680, O = 2560×2560
    build "$TAG" 128 2560 7680   QKV
    build "$TAG" 128 2560 2560   O
    # FFN: GU = 2560×10240, D = 5120×2560
    build "$TAG" 128 2560 10240  GU
    build "$TAG" 128 5120 2560   D
}

# ── Execute ──────────────────────────────────────────────────────────

BUILD_ALL_MODELS="qwen3_0_6b qwen3_8b qwen3_vl_4b llama gemma4_e2b mistral_7b falcon_7b olmoe_1b zamba2_2_7b"

if [ -n "$REQUESTED_TAG" ]; then
    # Build only the requested model
    case "$REQUESTED_TAG" in
        qwen3_0_6b)  build_qwen3_0_6b ;;
        qwen3_8b)    build_qwen3_8b ;;
        qwen3_vl_4b) build_qwen3_vl_4b ;;
        llama)       build_llama ;;
        gemma4_e2b)  build_gemma4_e2b ;;
        mistral_7b)  build_mistral_7b ;;
        falcon_7b)   build_falcon_7b ;;
        olmoe_1b)    build_olmoe ;;
        zamba2_2_7b) build_zamba2_2_7b ;;
        *)
            echo "ERROR: Unknown model tag: $REQUESTED_TAG"
            echo "  Valid tags: $BUILD_ALL_MODELS"
            exit 1
            ;;
    esac
    BUILT_TAGS="$REQUESTED_TAG"
else
    # Build all models
    build_qwen3_0_6b
    build_qwen3_8b
    build_qwen3_vl_4b
    build_llama
    build_gemma4_e2b
    BUILT_TAGS="$BUILD_ALL_MODELS"
fi

# ── Summary ──────────────────────────────────────────────────────────

echo ""
echo "=== Build Summary ==="
for tag in $BUILT_TAGS; do
    count=$(ls "${INT8_DIR}"/final_i8_*_${tag}.xclbin 2>/dev/null | wc -l)
    # shellcheck disable=SC2034
    size_total=$(du -sh "${INT8_DIR}"/final_i8_*_${tag}.xclbin 2>/dev/null | tail -1 | awk '{print $1}')
    echo "  $tag: $count xclbins"
done
echo "  Output directory: $INT8_DIR"

# ── Verification ─────────────────────────────────────────────────────

if [ "$DO_VERIFY" = true ]; then
    echo ""
    echo "=== Verification ==="
    FAILURES=0
    for tag in $BUILT_TAGS; do
        if ! run_verify "$tag"; then
            FAILURES=$((FAILURES + 1))
        fi
    done
    echo ""
    if [ "$FAILURES" -eq 0 ]; then
        echo "✓ All verifications passed"
    else
        echo "✗ $FAILURES model(s) failed verification"
        exit 1
    fi
fi

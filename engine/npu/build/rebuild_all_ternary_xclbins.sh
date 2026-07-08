#!/bin/bash
# rebuild_all_ternary_xclbins.sh — Build all native ternary xclbins for npu_ternaryd
#
# Builds 3 xclbin variants (32-core native ternary, 4×8 grid):
#   Q.xclbin  — M=2048 K=1024  (for Q, Up, Gate projections)
#   KV.xclbin — M=1024 K=1024  (for K, V projections)
#   O.xclbin  — M=1024 K=2048  (for O, Down projections)
#
# Prerequisites:
#   source engine/npu/build/env.sh
#
# Usage:
#   bash engine/npu/build/rebuild_all_ternary_xclbins.sh [output_dir]
#
# Output: xclbins + insts in output_dir/ (default: engine/npu/build/ternary_native_xclbins/)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENGINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="${1:-$SCRIPT_DIR/ternary_native_xclbins}"

if [ -z "${TOOLCHAIN:-}" ]; then
    source "$SCRIPT_DIR/env.sh"
fi

CC="${TOOLCHAIN}/bin/xchesscc_wrapper"
AIECC="${TOOLCHAIN}/mlir_aie/bin/aiecc.py"
AIECC_PYTHON="${TOOLCHAIN}/../.venv/bin/python3"
AIECC_PYTHONPATH="${MLIR_AIE_DIR}/python"
GEN_PYTHON="${AIECC_PYTHON}"
MLIR_GEN="$ENGINE_DIR/kernel/n1_core_native_ternary_32core.py"
# Kernel source lives in 1bit-systems subdirectory (cross-module)
KERNEL_SRC="$SCRIPT_DIR/../../../1bit-systems/engine/npu/kernel/mm_ternary_32x64x128.cpp"
if [ ! -f "$KERNEL_SRC" ]; then
    KERNEL_SRC="$ENGINE_DIR/kernel/mm_ternary_32x64x128.cpp"
fi

mkdir -p "$OUT_DIR"

# ── xclbin configurations ────────────────────────────
# Format: "name|M|K"
# M = output rows (must be multiple of 32)
# K = total ternary values (= packed_K * 4)
CONFIGS=(
    "Q|2048|1024"
    "KV|1024|1024"
    "O|1024|2048"
)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Native Ternary 32-Core xclbin Builder — All Variants"
echo "  Output dir: $OUT_DIR"
echo "  Configs: ${#CONFIGS[@]} variants"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r NAME TOTAL_M TOTAL_K <<< "$cfg"
    DIM_K=$(( TOTAL_K / 4 ))       # packed K bytes
    PER_CORE_M=$(( TOTAL_M / 32 )) # rows per core
    PER_COL_M=$(( TOTAL_M / 8 ))   # rows per column buffer
    BUILD_DIR="$OUT_DIR/build_${NAME}"
    KERNEL_ENTRY="mm_ternary_32x64x128"
    OBJ_FILE="$BUILD_DIR/${KERNEL_ENTRY}.o"

    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "  Building: $NAME.xclbin  (M=$TOTAL_M, K=$TOTAL_K, K_packed=$DIM_K)"
    echo "    per_core=$PER_CORE_M  per_col=$PER_COL_M"
    echo "────────────────────────────────────────────────────────────────"
    mkdir -p "$BUILD_DIR"

    # ── Step 1: Compile Chess kernel ──────────────────
    echo "  [1/3] Compiling kernel (DIM_M=$PER_COL_M DIM_K_PACKED=$DIM_K)..."
    $CC aie2p \
        -I"$AIETOOLS_DIR/include" \
        -I"$MLIR_AIE_DIR/include" \
        -I"$MLIR_AIE_DIR/include/aie_kernels" \
        -I"$MLIR_AIE_DIR/include/aie_kernels/aie2p" \
        -DDIM_M="$PER_COL_M" \
        -DDIM_K_PACKED="$DIM_K" \
        -DDIM_N=128 \
        -c "$KERNEL_SRC" \
        -o "$OBJ_FILE" 2>&1 | tail -2
    echo "  ✅ Kernel: $(du -h "$OBJ_FILE" | cut -f1)"

    # ── Step 2: Generate MLIR ─────────────────────────
    echo "  [2/3] Generating 32-core MLIR..."
    MLIR_FILE="$BUILD_DIR/design.mlir"
    PYTHONPATH="${AIECC_PYTHONPATH}:${PYTHONPATH:-}" \
        $GEN_PYTHON "$MLIR_GEN" -M "$TOTAL_M" -K "$DIM_K" \
        > "$MLIR_FILE"
    echo "  ✅ MLIR: $(wc -l < "$MLIR_FILE") lines"

    # ── Step 3: Build xclbin ─────────────────────────
    echo "  [3/3] aiecc → xclbin (this takes minutes)..."
    mkdir -p "$BUILD_DIR/design.mlir.prj"
    cp "$OBJ_FILE" "$BUILD_DIR/design.mlir.prj/"

    cd "$BUILD_DIR"
    PYTHONPATH="${AIECC_PYTHONPATH}:${PYTHONPATH:-}" \
        $AIECC_PYTHON \
        $AIECC \
        -v \
        -j4 \
        --aietools="$AIETOOLS_DIR" \
        --no-compile-host \
        --alloc-scheme=basic-sequential \
        --aie-generate-xclbin \
        --xclbin-name="${NAME}.xclbin" \
        --xclbin-kernel-name=MLIR_AIE \
        --aie-generate-npu-insts \
        --npu-insts-name="insts_${NAME}.txt" \
        "$MLIR_FILE" 2>&1 | tail -5

    # ── Copy outputs to flat directory ───────────────
    cp "$BUILD_DIR/${NAME}.xclbin" "$OUT_DIR/${NAME}.xclbin"
    cp "$BUILD_DIR/insts_${NAME}.txt" "$OUT_DIR/insts_${NAME}.txt"

    echo "  ✅ Done: $NAME.xclbin ($(stat -c%s "$OUT_DIR/${NAME}.xclbin") bytes)"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ All xclbins built!"
echo "  Directory: $OUT_DIR"
ls -lh "$OUT_DIR"/*.xclbin "$OUT_DIR"/insts_*.txt 2>/dev/null
echo ""
echo "  To use with npu_ternaryd:"
echo "    ./engine/npu/build/npu_ternaryd model.ternary/ $OUT_DIR/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

#!/bin/bash
# run_build.sh — Build all 25 new xclbins
set -euo pipefail

PYTHON=/home/bcloud/mlir-aie/.venv/bin/python3
AIECC=/home/bcloud/mlir-aie/build_tmp/bin/aiecc
PEANO=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/llvm-aie
AIETOOLS=/home/bcloud/mlir-aie/build_tmp
KERNEL_O=/home/bcloud/1bit-systems/engine/npu/generators/mm_32x64x128.o

export PYTHONPATH=/home/bcloud/mlir-aie/install_tmp/python:/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/install_tmp/python/aie/_mlir_libs

GENERATOR_DIR="$(cd "$(dirname "$0")" && pwd)"
XCLBIN_DIR="$GENERATOR_DIR/../xclbins"
mkdir -p "$XCLBIN_DIR"

SHAPES=(
    "qwen3.6-moe_35b:QKV:2048:5120:8"
    "qwen3.6-moe_35b:O:4096:2048:8"
    "qwen3.6-moe_35b:G:2048:512:4"
    "qwen3.6-moe_35b:U:2048:512:4"
    "qwen3.6-moe_35b:D:512:2048:4"
    "qwen3.5_4b:QKV:2560:6144:8"
    "qwen3.5_4b:O:4096:2560:4"
    "qwen3.5_4b:G:2560:9216:8"
    "qwen3.5_4b:U:2560:9216:8"
    "qwen3.5_4b:D:9216:2560:4"
    "gemma4_e4b:QKV:2560:6144:8"
    "gemma4_e4b:O:4096:2560:4"
    "gemma4_e4b:G:2560:12288:8"
    "gemma4_e4b:U:2560:12288:8"
    "gemma4_e4b:D:12288:2560:4"
    "phi4-mini_4b:QKV:3072:5120:8"
    "phi4-mini_4b:O:3072:3072:4"
    "phi4-mini_4b:G:3072:8192:8"
    "phi4-mini_4b:U:3072:8192:8"
    "phi4-mini_4b:D:8192:3072:4"
    "nanbeige4.1_3b:QKV:2560:3840:8"
    "nanbeige4.1_3b:O:2560:2560:4"
    "nanbeige4.1_3b:G:2560:8192:8"
    "nanbeige4.1_3b:U:2560:8192:8"
    "nanbeige4.1_3b:D:8192:2560:4"
)

build_one() {
    local tag="$1" proj="$2" K="$3" N="$4" cols="$5"
    local design="/tmp/design_${proj}_${tag}.mlir"
    local xclbin="$XCLBIN_DIR/final_i8_${proj}_${tag}.xclbin"
    local insts_dir="$GENERATOR_DIR/insts"
    mkdir -p "$insts_dir"
    
    echo ""
    echo "══════ Building ${tag} ${proj} K=${K} N=${N} cols=${cols} ══════"
    
    # Generate clean MLIR (stderr to /dev/null, stdout to file)
    $PYTHON "$GENERATOR_DIR/n1_core_i8_v23.py" \
        -M 128 -K "$K" -N "$N" -m 32 -k 64 -n 128 -c "$cols" \
        2>/dev/null > "$design"
    
    # aiecc needs kernel .o in CWD and runs from the design directory
    local workdir; workdir=$(dirname "$design")
    cp "$KERNEL_O" "$workdir/mm_32x64x128.o" 2>/dev/null || true
    
    cd "$workdir"
    $AIECC --peano="$PEANO" --aietools="$AIETOOLS" \
        --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
        --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
        --aie-generate-npu-insts \
        --xclbin-name="$xclbin" \
        --npu-insts-name="$insts_dir/insts_i8_${proj}_${tag}.txt" \
        "$design" 2>&1 | tail -1
    cd "$GENERATOR_DIR"
    
    if [ -f "$xclbin" ]; then
        local size; size=$(stat -c%s "$xclbin" 2>/dev/null)
        echo "  ✅ $(basename "$xclbin") ($(numfmt --to=iec "$size"))"
        return 0
    else
        echo "  ❌ FAILED"
        return 1
    fi
}

ok=0
fail=0
for entry in "${SHAPES[@]}"; do
    IFS=':' read -r tag proj K N cols <<< "$entry"
    if build_one "$tag" "$proj" "$K" "$N" "$cols"; then
        ((ok++))
    else
        ((fail++))
    fi
done

echo ""
echo "══════ RESULTS: ${ok} OK, ${fail} FAILED ══════"
echo "Xclbins in: $XCLBIN_DIR"
find "$XCLBIN_DIR" -name "final_i8_*.xclbin" -not -path "*backup*" 2>/dev/null | wc -l
echo "total xclbins"

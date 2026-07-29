#!/bin/bash
# Sequential build of all new xclbins with isolated workdirs
set -e

export PYTHON=/home/bcloud/mlir-aie/.venv/bin/python3
export AIECC=/home/bcloud/mlir-aie/build_tmp/bin/aiecc
export PEANO=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/llvm-aie
export AIETOOLS=/home/bcloud/mlir-aie/build_tmp
export KERNEL=/home/bcloud/1bit-systems/engine/npu/generators/mm_32x64x128.o
export XDIR=/home/bcloud/1bit-systems/engine/npu/xclbins
export GEN=/home/bcloud/1bit-systems/engine/npu/generators
export PYTHONPATH=/home/bcloud/mlir-aie/install_tmp/python:/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/install_tmp/python/aie/_mlir_libs

build_one() {
    local tag="$1" proj="$2" K="$3" N="$4" cols="$5"
    local workdir="/tmp/b_${proj}_${tag}"
    rm -rf "$workdir"
    mkdir -p "$workdir"
    
    $PYTHON "$GEN/n1_core_i8_v23.py" -M 128 -K "$K" -N "$N" -m 32 -k 64 -n 128 -c "$cols" \
        2>/dev/null > "$workdir/design.mlir"
    cp "$KERNEL" "$workdir/"
    
    cd "$workdir"
    $AIECC --peano="$PEANO" --aietools="$AIETOOLS" \
        --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
        --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
        --aie-generate-npu-insts \
        --xclbin-name="$XDIR/final_i8_${proj}_${tag}.xclbin" \
        --npu-insts-name="$workdir/insts.txt" \
        design.mlir > /dev/null 2>&1
    
    if [ -f "$XDIR/final_i8_${proj}_${tag}.xclbin" ]; then
        local sz; sz=$(stat -c%s "$XDIR/final_i8_${proj}_${tag}.xclbin" 2>/dev/null)
        echo "  ✅ $proj $tag ($(numfmt --to=iec "$sz"))"
    else
        echo "  ❌ $proj $tag"
    fi
    
    cd "$GEN"
    rm -rf "$workdir"
    sleep 2
}

ok=0
fail=0

for entry in \
    "qwen3.6-moe_35b:O:4096:2048:8" \
    "qwen3.6-moe_35b:U:2048:512:4" \
    "qwen3.6-moe_35b:D:512:2048:4" \
    "qwen3.5_4b:QKV:2560:6144:8" \
    "qwen3.5_4b:O:4096:2560:4" \
    "qwen3.5_4b:G:2560:9216:8" \
    "qwen3.5_4b:U:2560:9216:8" \
    "qwen3.5_4b:D:9216:2560:4" \
    "gemma4_e4b:QKV:2560:6144:8" \
    "gemma4_e4b:O:4096:2560:4" \
    "gemma4_e4b:G:2560:12288:8" \
    "gemma4_e4b:U:2560:12288:8" \
    "gemma4_e4b:D:12288:2560:4" \
    "phi4-mini_4b:QKV:3072:5120:8" \
    "phi4-mini_4b:O:3072:3072:4" \
    "phi4-mini_4b:G:3072:8192:8" \
    "phi4-mini_4b:U:3072:8192:8" \
    "phi4-mini_4b:D:8192:3072:4" \
    "nanbeige4.1_3b:QKV:2560:3840:8" \
    "nanbeige4.1_3b:O:2560:2560:4" \
    "nanbeige4.1_3b:G:2560:8192:8" \
    "nanbeige4.1_3b:U:2560:8192:8" \
    "nanbeige4.1_3b:D:8192:2560:4"; do
    
    IFS=':' read -r tag proj K N cols <<< "$entry"
    if build_one "$tag" "$proj" "$K" "$N" "$cols"; then
        ((ok++))
    else
        ((fail++))
    fi
done

echo ""
echo "=== Done: $ok OK, $fail FAILED ==="
find "$XDIR" -name "final_i8_*.xclbin" -not -path "*backup*" 2>/dev/null | wc -l
echo "total xclbins"

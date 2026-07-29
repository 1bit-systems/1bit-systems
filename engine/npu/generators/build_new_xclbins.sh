#!/bin/bash
# Build new xclbin shapes for models extracted from FLM v0.9.45/46
# Each projection = one xclbin with (K,N) tile dims + cols parameter
set -euo pipefail

export PEANO_INSTALL_DIR=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/llvm-aie
export AIETOOLS_DIR=/home/bcloud/mlir-aie/npu2_40_toolchain
export MLIR_AIE_DIR=/home/bcloud/mlir-aie
export VITIS_INC=/opt/amd/vitis/install/2026.1/Vitis/aietools/include
export PATH=$AIETOOLS_DIR/bin:$PEANO_INSTALL_DIR/bin:$PATH

GENERATOR_DIR="$(cd "$(dirname "$0")" && pwd)"
XCLBIN_DIR="$GENERATOR_DIR/../xclbins"
KERNEL_OBJ="$GENERATOR_DIR/mm_32x64x128.o"

# Verify kernel obj exists
if [ ! -f "$KERNEL_OBJ" ]; then
    echo "ERROR: Kernel object not found. Run build_kernel.sh first."
    exit 1
fi

# Model shapes: each entry is "model_tag:QKV_K,QKV_N:O_K,O_N:GU_K,GU_N or G_K,G_N,U_K,U_N:D_K,D_N:cols"
# For models with separate G and U, use "G_U:g_k,g_n:u_k,u_n"
# For models with fused GU, use "GU:gu_k,gu_n"
declare -a MODELS=(
    # qwen3.6-moe_35b-a3b (MoE, 256 experts, 40 layers, Q4_K_S)
    # QKV: h=2048, n_q=16*256=4096, n_kv=2*256=512 → N=4096+2*512=5120
    # O: n_q*hd=4096 × h=2048
    # G: h=2048 × ie=512 (per expert)
    # U: h=2048 × ie=512 (per expert)
    # D: ie=512 × h=2048
    "qwen3.6-moe_35b:QKV:2048:5120:8"
    "qwen3.6-moe_35b:O:4096:2048:8"
    "qwen3.6-moe_35b:G:2048:512:4"
    "qwen3.6-moe_35b:U:2048:512:4"
    "qwen3.6-moe_35b:D:512:2048:8"
    
    # qwen3.5_4b (VLM, 32 layers)
    # QKV: h=2560, n_q=16*256=4096, n_kv=4*256=1024 → N=4096+2*1024=6144
    # O: 4096 × 2560
    # G: 2560 × 9216
    # U: 2560 × 9216
    # D: 9216 × 2560
    "qwen3.5_4b:QKV:2560:6144:8"
    "qwen3.5_4b:O:4096:2560:4"
    "qwen3.5_4b:G:2560:9216:8"
    "qwen3.5_4b:U:2560:9216:8"
    "qwen3.5_4b:D:9216:2560:4"
    
    # gemma4_e4b (8B, 26 layers, larger gemma4)
    # QKV: h=2560, n_q=16*256=4096, n_kv=4*256=1024 → N=6144
    # O: 4096 × 2560
    # G: 2560 × 12288
    # U: 2560 × 12288
    # D: 12288 × 2560
    "gemma4_e4b:QKV:2560:6144:8"
    "gemma4_e4b:O:4096:2560:4"
    "gemma4_e4b:G:2560:12288:8"
    "gemma4_e4b:U:2560:12288:8"
    "gemma4_e4b:D:12288:2560:4"
    
    # phi4-mini-it_4b (32 layers, GQA=3)
    # VERIFIED from Q4NX: H=3072, NH=24, NKV=8, HD=128, IM=8192
    # QKV: h=3072, n_q=3072, n_kv=1024 → N=3072+2048=5120
    # O: 3072 × 3072
    # G: 3072 × 8192
    # U: 3072 × 8192
    # D: 8192 × 3072
    "phi4-mini_4b:QKV:3072:5120:8"
    "phi4-mini_4b:O:3072:3072:4"
    "phi4-mini_4b:G:3072:8192:8"
    "phi4-mini_4b:U:3072:8192:8"
    "phi4-mini_4b:D:8192:3072:4"
    
    # nanbeige4.1_3b (32 layers, reasoning)
    # QKV: h=2560, n_q=32*80=2560, n_kv=8*80=640 → N=2560+2*640=3840
    # O: 2560 × 2560
    # G: 2560 × 8192
    # U: 2560 × 8192
    # D: 8192 × 2560
    "nanbeige4.1_3b:QKV:2560:3840:8"
    "nanbeige4.1_3b:O:2560:2560:4"
    "nanbeige4.1_3b:G:2560:8192:8"
    "nanbeige4.1_3b:U:2560:8192:8"
    "nanbeige4.1_3b:D:8192:2560:4"
)

build_xclbin() {
    local model_tag="$1"
    local proj="$2"
    local K="$3"
    local N="$4"
    local cols="$5"
    
    local xclbin_name="final_i8_${proj}_${model_tag}.xclbin"
    local insts_name="insts_i8_${proj}_${model_tag}.txt"
    
    echo ""
    echo "═══════════════════════════════════════════════"
    echo "  Building ${model_tag} ${proj} (K=${K}, N=${N}, cols=${cols})"
    echo "═══════════════════════════════════════════════"
    
    # Generate MLIR
    python3 "$GENERATOR_DIR/n1_core_i8_v23.py" \
        -M 128 -K "$K" -N "$N" \
        -m 32 -k 64 -n 128 \
        -c "$cols" > "$GENERATOR_DIR/design_${proj}_${model_tag}.mlir" 2>&1
    
    # Compile to xclbin
    aiecc --aietools="$AIETOOLS_DIR" --peano="$PEANO_INSTALL_DIR" \
        --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
        --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
        --aie-generate-npu-insts \
        --xclbin-name="$XCLBIN_DIR/$xclbin_name" \
        --npu-insts-name="$GENERATOR_DIR/$insts_name" \
        "$GENERATOR_DIR/design_${proj}_${model_tag}.mlir" 2>&1
    
    local status=$?
    if [ $status -eq 0 ]; then
        local size
        size=$(stat -c%s "$XCLBIN_DIR/$xclbin_name" 2>/dev/null || echo "0")
        echo "  ✅ ${xclbin_name} ($(numfmt --to=iec "$size"))"
    else
        echo "  ❌ FAILED (exit=$status)"
    fi
    
    # Clean up MLIR design file
    rm -f "$GENERATOR_DIR/design_${proj}_${model_tag}.mlir"
    
    return $status
}

echo "================================================"
echo "  Building $(( ${#MODELS[@]} )) new xclbin shapes"
echo "================================================"

total_ok=0
total_fail=0

for entry in "${MODELS[@]}"; do
    IFS=':' read -r model_tag proj K N cols <<< "$entry"
    
    if build_xclbin "$model_tag" "$proj" "$K" "$N" "$cols"; then
        total_ok=$((total_ok + 1))
    else
        total_fail=$((total_fail + 1))
    fi
done

echo ""
echo "================================================"
echo "  Build complete: ${total_ok} OK, ${total_fail} failed"
echo "================================================"
echo ""
echo "New xclbins in: $XCLBIN_DIR"
find "$XCLBIN_DIR" -name "final_i8_*.xclbin" -not -path "*backup*" -exec ls -la {} \; 2>/dev/null | tail -30
echo ""
echo "Total xclbins: $(find "$XCLBIN_DIR" -name "final_i8_*.xclbin" -not -path "*backup*" 2>/dev/null | wc -l)"

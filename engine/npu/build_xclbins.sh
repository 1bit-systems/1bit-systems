#!/bin/bash
# Build all model xclbins using the proven make flow (n1_core_placed.py)
# This uses BF16 MLIR generation which has correct AIE memory allocation.
# The engine will use INT8 data which the kernel handles internally.

set -e
CFG1="${HOME}/torch2aie/examples/gemm_asymmetric_tile_buffering/config1"
INT8_DIR="${HOME}/npu-sandbox/npu-infer/build/int8"
mk_dir="$CFG1/build"
mkdir -p "$INT8_DIR"

export PATH="${HOME}/torch2aie/toolchain/bin:$PATH"
export PYTHONPATH="${HOME}/torch2aie/toolchain/mlir_aie/python"

build() {
    local tag=$1 M=$2 K=$3 N=$4 label=$5
    local target="${mk_dir}/final_${M}x${K}x${N}_128x64x128.xclbin"
    local insts_target="${mk_dir}/insts_${M}x${K}x${N}_128x64x128.txt"

    if [ ! -f "$target" ]; then
        echo "  Building ${M}x${K}x${N} for $tag/$label..."
        make -C "$CFG1" "M=$M" "K=$K" "N=$N" m=128 k=64 n=128 use_placed=1 targetname=n1_core \
            aie_py_src=n1_core_placed.py \
            "build/final_${M}x${K}x${N}_128x64x128.xclbin" 2>&1 | tail -3
        echo "  Done: $(stat -c%s "$target") bytes"
    else
        echo "  Using cached ${M}x${K}x${N}"
    fi

    cp "$target" "${INT8_DIR}/final_i8_${label}_${tag}.xclbin"
    if [ -f "$insts_target" ]; then
        cp "$insts_target" "${INT8_DIR}/insts_i8_${label}_${tag}.txt"
    fi
}

# === Qwen3-0.6B (H=1024, NH=16, NKV=8, HD=128, IM=3072) ===
echo "=== Qwen3-0.6B ==="
# QKV: K=H=1024, N=HD*(NH+2*NKV)=128*32=4096
# O: K=NH*HD=2048, N=H=1024
# GU: K=H=1024, N=2*IM=6144 (fused)
# D: K=IM=3072, N=H=1024
TAG="qwen3_0_6b"
build "$TAG" 128 1024 4096   QKV
build "$TAG" 128 2048 1024   O
build "$TAG" 128 1024 6144   GU
build "$TAG" 128 3072 1024   D

# === Qwen3-8B (H=4096, NH=32, NKV=8, HD=128, IM=12288) ===
echo "=== Qwen3-8B ==="
# QKV: K=4096, N=HD*(NH+2*NKV)=4096+1024+1024=6144
# O: K=4096, N=4096
# G: K=4096, N=12288 (split)
# U: K=4096, N=12288 (split)
# D: K=12288, N=4096
TAG="qwen3_8b"
build "$TAG" 128 4096 6144   QKV
build "$TAG" 128 4096 4096   O
build "$TAG" 128 4096 12288  G
build "$TAG" 128 4096 12288  U
build "$TAG" 128 12288 4096  D

# === Qwen3-VL-4B (H=2560, NH=32, NKV=8, HD=128, IM=9728) ===
echo "=== Qwen3-VL-4B ==="
# QKV: K=2560, N=HD*(NH+2*NKV)=128*48=6144
# O: K=4096, N=2560
# G: K=2560, N=9728 (split)
# U: K=2560, N=9728 (split)
# D: K=9728, N=2560
TAG="qwen3_vl_4b"
build "$TAG" 128 2560 6144   QKV
build "$TAG" 128 4096 2560   O
build "$TAG" 128 2560 9728   G
build "$TAG" 128 2560 9728   U
build "$TAG" 128 9728 2560   D

# === Llama-3.1-8B (H=4096, NH=32, NKV=8, HD=128, IM=14336) ===
echo "=== Llama-3.1-8B ==="
# QKV: K=4096, N=6144 (same as qwen3_8b - same H,NH,NKV,HD)
# O: K=4096, N=4096
# G: K=4096, N=14336
# U: K=4096, N=14336
# D: K=14336, N=4096
TAG="llama"
build "$TAG" 128 4096 6144   QKV
build "$TAG" 128 4096 4096   O
build "$TAG" 128 4096 14336  G
build "$TAG" 128 4096 14336  U
build "$TAG" 128 14336 4096  D

# === Gemma4-E2B (H=1536, NH=8, NKV=1, HD=256, IM=6144) ===
# Note: actual Q4NX file has H=1536 (not 3584 as in model name)
# QKV: K=H=1536, N=HD*(NH+2*NKV)=256*(8+2)=2560
# O: K=NH*HD=2048, N=H=1536
# GU: K=H=1536, N=2*IM=12288 (fused, 12288<=14336)
# D: K=IM=6144, N=H=1536
TAG="gemma4_e2b"
build "$TAG" 128 1536 2560   QKV
build "$TAG" 128 2048 1536   O
build "$TAG" 128 1536 12288  GU
build "$TAG" 128 6144 1536   D

echo ""
echo "=== All xclbins built ==="
for tag in qwen3_0_6b qwen3_8b qwen3_vl_4b llama gemma4_e2b; do
    count=$(ls "${INT8_DIR}"/final_i8_*_${tag}.xclbin 2>/dev/null | wc -l)
    echo "  $tag: $count xclbins"
done

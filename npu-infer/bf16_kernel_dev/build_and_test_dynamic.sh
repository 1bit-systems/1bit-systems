#!/usr/bin/env bash
# Full build: xclbin + C++ TXN builder for dynamic INT8 GEMM
# Usage: ./build_and_test_dynamic.sh [M] [K] [N]

set -eo pipefail

M=${1:-128}
K=${2:-1024}
N=${3:-4096}
DIR="${4:-/tmp/build_dynamic_int8}"
NAME="int8_dynamic"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DIR"

# Paths
export LD_LIBRARY_PATH="${HOME}/airenv/lib/python3.14/site-packages/llvm-aie/lib:${HOME}/torch2aie/toolchain/aietools/lib/lnx64.o"
export PATH="${HOME}/mlir-aie/install_tmp/bin:${HOME}/mlir-aie/build_tmp/bin:${HOME}/airenv/lib/python3.14/site-packages/llvm-aie/bin:${PATH}"

PEANO_CLANG=${HOME}/airenv/lib/python3.14/site-packages/llvm-aie/bin/clang
KERNEL_SRC=/tmp/mlir-aie/aie_kernels/aie2p/mm.cc

echo "=== 1. Compile Peano kernel ==="
$PEANO_CLANG --target=aie2p-none-unknown-elf -O2 \
  -std=c++20 -DNDEBUG -D__AIE_API_AIE_ADF_HPP__ \
  -DDIM_M=32 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY \
  -I${HOME}/mlir-aie/include \
  -I${HOME}/mlir-aie/third_party/aie_api/include \
  -I${HOME}/mlir-aie/aie_kernels/aie2p \
  -c "$KERNEL_SRC" -o "$DIR/mm_32x64x128.o" 2>&1 | tail -3

echo "=== 2. Generate static MLIR (for xclbin) ==="
PYTHONPATH=${HOME}/torch2aie/toolchain/mlir_aie/python \
  ${HOME}/torch2aie/.venv/bin/python \
  "$SCRIPT_DIR/gen_mlir_dynamic.py" -M $M -K $K -N $N > "$DIR/design_static.mlir"

echo "=== 3. Build xclbin ==="
${HOME}/mlir-aie/install_tmp/bin/aiecc \
  --aietools=${HOME}/torch2aie/toolchain/aietools \
  --peano=${HOME}/airenv/lib/python3.14/site-packages/llvm-aie \
  --alloc-scheme=basic-sequential \
  --aie-generate-xclbin \
  --no-compile-host \
  --no-xchesscc \
  --no-xbridge \
  --xclbin-name="$DIR/${NAME}.xclbin" \
  --unified \
  --dynamic-objFifos \
  -I"$DIR" \
  "$DIR/design_static.mlir" 2>&1 | grep -E "edge|error" | tail -3

echo "=== 4. Generate dynamic MLIR (for TXN builder) ==="
PYTHONPATH=${HOME}/torch2aie/toolchain/mlir_aie/python \
  ${HOME}/torch2aie/.venv/bin/python \
  "$SCRIPT_DIR/gen_mlir_dynamic.py" --dynamic -M $M -K $K -N $N > "$DIR/design_dynamic.mlir"

echo "=== 5. Lower with BD pool pass ==="
aie-opt "$DIR/design_dynamic.mlir" \
  --aie-materialize-bd-chains \
  --aie-substitute-shim-dma-allocations \
  --aie-unroll-runtime-sequence-loops \
  --canonicalize \
  --aie-lower-dynamic-bd-pool \
  --canonicalize \
  --aie-assign-runtime-sequence-bd-ids \
  --aie-dma-tasks-to-npu \
  --aie-dma-to-npu \
  --aie-lower-set-lock \
  -o "$DIR/lowered.mlir"

echo "=== 6. Generate C++ TXN builder ==="
aie-translate --aie-npu-to-cpp "$DIR/lowered.mlir" > "$DIR/txn_builder.h"

echo ""
echo "=== Results ==="
echo "  xclbin:       $(ls -lh "$DIR/${NAME}.xclbin" | awk '{print $5}')"
echo "  TXN builder:  $(wc -l < "$DIR/txn_builder.h") lines"
echo ""
echo "=== To test on NPU hardware ==="
echo "  g++ -std=c++23 \\"
echo "    -I\${HOME}/mlir-aie/install_tmp/include \\"
echo "    -DGEN_HDR='\"$DIR/txn_builder.h\"' \\"
echo "    -DXCLBIN='\"$DIR/${NAME}.xclbin\"' \\"
echo "    $SCRIPT_DIR/test_dynamic_int8.cpp \\"
echo "    -o $DIR/test -lxrt_core -lxrt_coreutil -lpthread"
echo "  $DIR/test $M $K $N"

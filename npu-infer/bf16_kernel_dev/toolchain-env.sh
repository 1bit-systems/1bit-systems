#!/usr/bin/env bash
# ── Operational MLIR-AIE / xclbin build environment for 1bit-systems INT8 GEMMs ──
# AMD Xilinx IP toolchain (Chess compiler).  License required.
#
#   source /home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev/toolchain-env.sh
#
export PATH="/home/bcloud/torch2aie/toolchain/bin:/home/bcloud/torch2aie/toolchain/aietools/bin:${PATH}"
export PYTHONPATH="/home/bcloud/torch2aie/toolchain/mlir_aie/python:${PYTHONPATH}"
export AIETOOLS_DIR="/home/bcloud/torch2aie/toolchain/aietools"
export AIETOOLS="/home/bcloud/torch2aie/toolchain/aietools"
export MLIR_AIE_DIR="/home/bcloud/torch2aie/toolchain/mlir_aie"
export LM_LICENSE_FILE="/home/bcloud/torch2aie/licenses/Xilinx.lic"
export LD_LIBRARY_PATH="/home/bcloud/torch2aie/toolchain/mlir_aie.libs:/home/bcloud/torch2aie/toolchain/xrt/lib64:${LD_LIBRARY_PATH}"

# Fixed tool paths:
export AIECC_BIN="/home/bcloud/torch2aie/toolchain/bin/aiecc"   # AMD Xilinx IP (Chess-enabled)
export GEN_PYTHON="/home/bcloud/torch2aie/.venv/bin/python" # py3.12 venv (has aie/aiex/aie.ir)
export INT8_DIR="/home/bcloud/npu-sandbox/npu-infer/build/int8"  # prebuilt kernel .o + working xclbins

# Canonical proven build of one INT8 GEMM xclbin from a design .py:
#   $GEN_PYTHON <design.py> -M 128 -K 3072 -N 1024 -m 32 -k 64 -n 128 > D.mlir
#   $AIECC_BIN --aietools=$AIETOOLS_DIR --alloc-scheme=basic-sequential \
#     --aie-generate-xclbin --no-compile-host --xclbin-name=D.xclbin --unified --dynamic-objFifos \
#     --aie-generate-npu-insts --npu-insts-name=D.txt -I$INT8_DIR D.mlir
# ── New MLIR-AIE build with dynamic BD pool support (PR #3358+) ──
# Built from Xilinx/mlir-aie main (2026-07-21).
# Supports --aie-lower-dynamic-bd-pool and HasAncestor on DMA task ops.
# Use `source toolchain-env.sh && aie-opt --help | grep bd-pool` to verify.
# To use the new binaries, set PATH to pick up the install_tmp/bin first:
#   export PATH="/home/bcloud/mlir-aie/install_tmp/bin:${PATH}"
#
# The new aie-opt accepts keyword-format dma_bd syntax:
#   aie.dma_bd(%buf : memref<...> offset = 0 len = 1024 sizes = [1,4,8,32] strides = [4096,512,32,1])

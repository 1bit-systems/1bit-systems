#!/bin/bash
# Source the Chess/MLIR toolchain for NPU xclbin compilation
# Run before any build_*.sh script
export TOOLCHAIN=/home/bcloud/torch2aie/toolchain
export PATH=$TOOLCHAIN/bin:$TOOLCHAIN/aietools/bin:$PATH
export LD_LIBRARY_PATH=$TOOLCHAIN/mlir_aie/lib:$TOOLCHAIN/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$TOOLCHAIN/mlir_aie/python:$PYTHONPATH
export XCHESS=$TOOLCHAIN/bin/xchesscc

echo "Toolchain ready:"
echo "  chess:  $(which xchesscc 2>/dev/null)"
echo "  aiecc:  $(which aiecc.py 2>/dev/null)"
echo "  xclbin: $(which xclbinutil 2>/dev/null)"
echo "  mlir:   $(python3 -c 'import aie; print(aie.__file__)' 2>/dev/null || echo 'not found')"

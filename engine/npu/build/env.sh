#!/bin/bash
# Source the Chess/MLIR toolchain for NPU xclbin compilation
# Run before any build_*.sh script
SCRIPT_DIR="$(cd "$(dirname "$BASH_SOURCE")" && pwd)"
ENGINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export TOOLCHAIN=/home/bcloud/torch2aie/toolchain
export AIETOOLS_DIR=$TOOLCHAIN/aietools
export MLIR_AIE_DIR=$TOOLCHAIN/mlir_aie
export PATH=$TOOLCHAIN/bin:$AIETOOLS_DIR/bin:$TOOLCHAIN/mlir_aie/bin${PATH:+:$PATH}
export LD_LIBRARY_PATH=$TOOLCHAIN/mlir_aie/lib:$TOOLCHAIN/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$TOOLCHAIN/mlir_aie/python${PYTHONPATH:+:$PYTHONPATH}
export XCHESS=$TOOLCHAIN/bin/xchesscc

echo "Toolchain ready:"
echo "  chess:      $(which xchesscc 2>/dev/null || echo 'NOT FOUND')"
echo "  aiecc.py:   $(which aiecc.py 2>/dev/null || echo 'NOT FOUND')"
echo "  aiecc:      $(ls /home/bcloud/mlir-aie/build/bin/aiecc 2>/dev/null || echo 'NOT FOUND')"
echo "  xclbinutil: $(which xclbinutil 2>/dev/null || echo 'NOT FOUND')"
echo "  peano:      $(ls /home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/llvm-aie/bin/clang++ 2>/dev/null || echo 'NOT FOUND')"
echo "  gen-python: $(ls /home/bcloud/torch2aie/.venv/bin/python 2>/dev/null || echo 'NOT FOUND')"

# Verify aie module can be imported
if PYTHONPATH=$TOOLCHAIN/mlir_aie/python /home/bcloud/torch2aie/.venv/bin/python -c 'from aie.extras.context import mlir_mod_ctx' 2>/dev/null; then
    echo "  aie-mlir:   OK"
else
    echo "  aie-mlir:   WARNING - import failed"
fi

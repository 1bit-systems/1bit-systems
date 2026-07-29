#!/bin/bash
# Fix the Peano toolchain to build v24 batched-BD xclbins.
# The LLVM 13 opt in target_aie2p can't handle typed-pointer IR from mlir-aie.
# This wrapper redirects opt/llc to the LLVM 21 versions in the LLVM-AIE venv,
# which support opaque pointers and AIE2P targets.
#
# Known limitation: LLVM-AIE's lld can't link Chess-compiled kernel objects
# (has .tctmemtab/.chesstypeannotationtab sections). To build v24 xclbins,
# compile mm.cc with the LLVM-AIE clang instead.
PEANO_DIR="$(dirname "$(which opt)" 2>/dev/null)"
if [ -z "$PEANO_DIR" ]; then
  echo "Error: opt not found on PATH. Run with Peano path configured."
  exit 1
fi
LLVM_AIE_DIR="/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/llvm-aie/bin"
for tool in opt llc; do
  if [ ! -f "$PEANO_DIR/$tool.real" ]; then
    cp "$PEANO_DIR/$tool" "$PEANO_DIR/$tool.real"
  fi
  cat > "$PEANO_DIR/$tool" << EOF
#!/bin/bash
exec "$LLVM_AIE_DIR/$tool" "\$@"
EOF
  chmod +x "$PEANO_DIR/$tool"
done
echo "Toolchain fixed. opt: $($PEANO_DIR/opt --version 2>&1 | head -1)"

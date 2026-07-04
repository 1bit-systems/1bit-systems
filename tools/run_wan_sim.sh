#!/bin/bash
# WAN NPU Kernel Debug — aiesimulator setup
# Run this from the 1bit-systems repo root
set -e

export XILINX_VITIS=/home/bcloud/Xilinx/2026.1/Vitis
export AIETOOLS=$XILINX_VITIS/aietools
export RDI_DATADIR=$AIETOOLS
export LD_LIBRARY_PATH=$AIETOOLS/lib/lnx64.o:$AIETOOLS/lib:$XILINX_VITIS/lib/lnx64.o
export PATH=$AIETOOLS/bin:$HOME/torch2aie/toolchain/bin:$PATH

WAN_DIR=/home/bcloud/npu-sandbox/npu-infer/build/wan

echo "=== 1. Build xclbin with aiesim support ==="
cd "$WAN_DIR"

# Ensure device JSON exists (symlink fix applied)
ls -la $AIETOOLS/data/aie2p/devices/aie2p_8x4_device.json 2>/dev/null || \
  sudo ln -sf /home/bcloud/Xilinx/2026.1/data/aie2ps/devices/XC2VE3804.json \
    $AIETOOLS/data/aie2p/devices/aie2p_8x4_device.json

# Build with --aiesim (may take 5+ minutes)
aiecc --aietools="$AIETOOLS_DIR" \
  --unified --dynamic-objFifos \
  --aie-generate-xclbin --no-compile-host \
  --xclbin-name="wan_sim.xclbin" \
  --npu-insts-name="insts_wan_sim.txt" \
  --aiesim \
  wan_sim_v2.mlir 2>&1 | tee /tmp/aiecc_build.log

echo "=== 2. Run aiesimulator ==="
if [ -d "wan_sim_v2.mlir.prj" ]; then
  cd wan_sim_v2.mlir.prj
  mkdir -p sim
  echo "aie2p" > sim/.target
  
  aiesimulator \
    --pkg-dir=sim \
    --dump-vcd=/tmp/wan_sim.vcd \
    --simulation-cycle-timeout=100000 2>&1 | tee /tmp/aiesim_run.log
fi

echo "=== Done. VCD at /tmp/wan_sim.vcd ==="

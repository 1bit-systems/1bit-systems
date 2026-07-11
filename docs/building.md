# Building the INT8 Inference Engine

## Prerequisites

- AMD Strix Halo (Ryzen AI Max+ 395) with NPU enabled
- Ubuntu 26.04 LTS (kernel 7.0.0+)
- AMD XRT 2.21+ (`sudo apt install libxrt2 libxrt-npu2 libxrt-dev`)
- GCC 15+ (for C++23 support)
- Chess compiler license (free from [AMD Ryzen AI EA](https://account.amd.com/en/member/ryzenai-sw-ea.html))
- Qwen3-0.6B Q4NX model (from FastFlowLM or extracted from FLM installation)

## Step 1: Verify NPU

```bash
sudo xrt-smi examine
# Expected: [0000:c6:00.1] |RyzenAI-npu5|
lsmod | grep amdxdna
# Expected: amdxdna module loaded
```

## Step 2: Set up torch2aie toolchain

The torch2aie toolchain provides `aiecc` (MLIR-AIE compiler) and `xchesscc_wrapper` (Chess C++ compiler). These are needed to build xclbins.

```bash
git clone https://github.com/taowen/torch2aie.git ~/torch2aie
cd ~/torch2aie
./scripts/setup_python.sh
source scripts/env.sh

# Place Chess license
mkdir -p licenses/
cp /path/to/Xilinx.lic licenses/
```

## Step 3: Build INT8 xclbins (one-time)

The engine uses 4 INT8 xclbins for QKV, O, GU, and D projections.
Each is built with the K-interleaved-fixed generator.

```bash
cd engine/xclbins
source ~/torch2aie/scripts/env.sh

# Build kernel once
xchesscc_wrapper aie2p -c \
  -I $AIETOOLS_DIR/include -I $MLIR_AIE_DIR/include \
  -DDIM_M=32 -DDIM_K=64 -DDIM_N=64 -Di8_i16_ONLY \
  $MLIR_AIE_DIR/include/aie_kernels/aie2p/mm.cc \
  -o ../build/mm_i8.o

# Generate xclbins for each projection
for shape in "128 1024 4096 QKV" "128 2048 1024 O" "128 1024 6144 GU" "128 3072 1024 D"; do
    read M K N name <<< "$shape"
    python3 n1_core_i8_v2.py -M $M -K $K -N $N > $name.mlir
    aiecc --aietools=$AIETOOLS_DIR --alloc-scheme=basic-sequential \
          --aie-generate-xclbin --no-compile-host \
          --xclbin-name=../build/final_i8_${name}_v.xclbin \
          --aie-generate-npu-insts --npu-insts-name=../build/insts_i8_${name}_v.txt \
          $name.mlir
done
```

## Step 4: Build the dequantizer

```bash
gcc -c -O3 -o engine/build/dequant_q4nx.o engine/src/dequant_q4nx.c
```

## Step 5: Build the engine

```bash
g++ -std=c++23 -O3 -o engine/build/npu_engine \
    engine/src/npu_engine_i8.cpp \
    engine/build/dequant_q4nx.o \
    -I$XRT/include \
    -L$XRT/lib64 \
    -lxrt_coreutil -luuid -lm -ldl
```

## Step 6: Run

```bash
./engine/build/npu_engine
```

Expected output:

```
=== NPU Engine i8 + Attention ===
Init 8 contexts...
Dequant+pack: 4.5s
=== Prefill 9 ===
Done
=== Generate ===
  [0] 107325 (245ms)
  [1] 40469 (241ms)
  ...
=== 243 ms/tok ===
```

## Model location

The engine expects the Qwen3-0.6B Q4NX model at:
```
/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx
```

This is the default FastFlowLM model cache location. To use a different path,
edit the `mp` variable in `engine/src/npu_engine_i8.cpp`.

## Performance tuning

- **4-live contexts**: All 4 GEMM contexts stay alive. No swapping overhead.
- **Pre-loaded BOs**: Each layer's INT8 weights are uploaded once at startup.
- **Activation scale**: Fixed at 5.0f/127.0f (verified on Strix Halo).
- **NPU attention**: 4 attention xclbins registered. Falls back to CPU for <32 tokens.

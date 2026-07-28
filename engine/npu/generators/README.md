# NPU INT8 GEMM MLIR generator

`n1_core_i8_v23.py` generates the MLIR for every INT8 GEMM xclbin under
`engine/npu/xclbins/final_i8_{QKV,O,GU,G,U,D}_<model_tag>.xclbin`. Written
2026-07-28 after the prior generator source (used for the original D/O xclbins)
was lost — only committed as compiled binaries, never as source. Don't repeat
that: if this file changes, commit the change along with any regenerated
xclbins.

`mm_kernel_reference.cc` is the exact `mm.cc` kernel source (from mlir-aie's
`aie_kernels/aie2p/`) this generator was built and verified against — kept
here as a pinned reference copy in case the upstream mlir-aie tree changes.

## Build (per shape)

```bash
# 1. Kernel object (Peano, matches ABI below)
$PEANO_INSTALL_DIR/bin/clang --target=aie2p-none-unknown-elf -O2 -std=c++20 \
  -I $AIETOOLS_DIR/include -I $MLIR_AIE_DIR/include \
  -I $MLIR_AIE_DIR/include/aie_kernels -DDIM_M=32 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY \
  -c $MLIR_AIE_DIR/include/aie_kernels/aie2p/mm.cc -o mm_32x64x128.o

# 2. MLIR + xclbin (M is always 128; K,N,cols depend on model/projection —
#    see npu_dims.h for per-model dimensions; cols must evenly divide N/128)
python3 n1_core_i8_v23.py -M 128 -K <K> -N <N> -m 32 -k 64 -n 128 -c <cols> > design.mlir
aiecc --aietools="$AIETOOLS_DIR" --peano="$PEANO_INSTALL_DIR" \
  --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
  --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
  --aie-generate-npu-insts \
  --xclbin-name=final_i8_<PROJ>_<model_tag>.xclbin \
  --npu-insts-name=insts_i8_<PROJ>_<model_tag>.txt \
  design.mlir
```

## Why Peano + scalar kernel, not Chess + vectorized

Chess hangs on real NPU2 hardware for this kernel/loop pattern (root cause
still open — see project notes). Peano doesn't. The vectorized
`matmul_i8_i32` kernel needs data pre-arranged in AIE microtile (8x8) order,
which this generator doesn't produce — use the `matmul_scalar_i8_i32` /
`zero_scalar_i32` entry points (already what's linked into the checked-in
xclbins). Correct, not yet at vectorized-kernel throughput.

## Per-model shapes verified 2026-07-28 (0 errors on real hardware, all 22)

| model | QKV (K,N) | O (K,N) | G/U or GU (K,N) | D (K,N) | cols |
|---|---|---|---|---|---|
| qwen3_0_6b | 1024,4096 | 2048,1024 | GU: 1024,6144 | 3072,1024 | 8 |
| qwen3_8b | 4096,6144 | 4096,4096 | G/U: 4096,12288 | 12288,4096 | 8 |
| qwen3_vl_4b | 2560,6144 | 4096,2560 | G/U: 2560,9728 | 9728,2560 | 4 |
| llama | 4096,6144 | 4096,4096 | G/U: 4096,14336 | 14336,4096 | 8 |
| gemma4_e2b | 1536,2560 | 2048,1536 | GU: 1536,12288 | 6144,1536 | 4 |

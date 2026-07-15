# Engine Rewrite Plan — Custom torch2aie Kernel

## The Problem
FLM's black-box xclbins (mm.xclbin, attn.xclbin, layer.xclbin, dequant.xclbin) all produce
IDENTITY output regardless of opcode, instructions, or BO configuration. They require
internal tile SRAM state (weight DMA + runlist context) that we can't reproduce.

Custom torch2aie xclbins from `gemm_asymmetric_tile_buffering/config1` produce REAL GEMM
at 12.12 TFLOPS with correct output (verified against CPU reference).

## The Plan

### Files to Modify

1. **`include/engine.h`** — New `CustomNpuEngine` class replacing `NpuInferenceEngine`
2. **`src/engine.cpp`** — Complete rewrite with custom-kernel dataflow
3. **`include/model.h`** — Add weight packing for BFP16 format
4. **`src/model.c`** — Add `pack_weight_bfp_block()` that shuffles + BFP16 encodes
5. **`CMakeLists.txt`** — Add torch2aie include paths to npu_infer
6. **`src/main.cpp`** — Minimal changes for new engine API

### Dataflow (per GEMM)

For each GEMM: `A[M,K] @ B[K,N] = C[M,N]`, the xclbin is designed for:
- XM = 4096, XK = 4096, XN = 2048 (config1)
- tile_m = 128, tile_k = 64, tile_n = 128
- 4 AIE rows × 8 AIE cols = 32 cores

**A (activation, BF16 path, BO group 3):**
1. Pad input [M, K] to [XM, XK] with zeros
2. Shuffle: `layout_A_L1_2x1_8x8block(A_float, XM, XK, 128, 64)`
3. Convert to BF16 uint16 pairs
4. Upload to BO

**B (weights, BFP16 path, BO group 4):**
1. Pre-packed at model load time:
   a. B matrix = transpose of weight matrix W [out_features, in_features]
   b. Pad to [XK, XN] with zeros
   c. Shuffle: `layout_transpose_L1_1x2_8x8block(B_float, XK, XN, 64, 128)`
   d. Encode: `floatToBfp16(8, XK*XN, shuffled_data, 0)`
2. Upload once per layer (reused for all decode steps)

**C (output, BF16 path, BO group 5):**
1. Clear output BO (zero)
2. Kernel: `k(3, instr_bo, ninstr, bo_A, bo_B, bo_C)`
3. Read back, unshuffle: `layout_inverse_C_L1_2x2_8x8block(C_float, XM, XN, 512, 128)`
4. Extract valid [M, N] region

### Key Dimensions

| Projection | M | K | N | Instr count |
|-----------|-----|------|-------|-----------|
| Q_proj (single tok) | 1 | 1024 | 1024 | same as config1 |
| K_proj | 1 | 1024 | 1024 | same |
| V_proj | 1 | 1024 | 1024 | same |
| O_proj | 1 | 1024 | 1024 | same |
| Gate_proj | 1 | 1024 | 3072 | same (N ≤ 2048, 2 calls for 3072) |
| Up_proj | 1 | 1024 | 3072 | same, 2 calls |
| Down_proj | 1 | 3072 | 1024 | same (K ≤ 4096, fine) |
| LM_head | 1 | 1024 | ? | need to check N=151936 vs XN=2048 → 75 calls |

For N > XN=2048, split into N/XN chunks (ceil division):
- gate/up proj: 3072 → ceil(3072/2048)=2 chunks
- LM head: 151936 → ceil(151936/2048)=75 chunks

### Weight Packing (one-time cost)

For each weight tensor W[out_features, in_features]:
1. Transpose to B[K=in_features, N=out_features] row-major
2. For each N-chunk [0..N, 0..min(N+XN, N)]:
   a. Copy K×XN slice into [XK, XN] zero-padded float buffer
   b. Shuffle: `layout_transpose_L1_1x2_8x8block`
   c. Encode: `floatToBfp16(8, XK*XN, shuffled, 0)`
   d. Store packed chunk

### Full Layer Pipeline

```
input [1, 1024] BF16
  ↓
RMS Norm (CPU BF16)
  ↓
Q_proj NPU (M=1, K=1024, N=1024)
K_proj NPU (M=1, K=1024, N=1024)
V_proj NPU (M=1, K=1024, N=1024)
  ↓                   ┌─────────────┐
Q*K^T attn weights    │ Attention   │
softmax(QK^T)         │ (CPU only)  │
softmax * V           └─────────────┘
  ↓
O_proj NPU (M=1, K=1024, N=1024)
  ↓ residual +=
  ↓
RMS Norm (CPU BF16)
  ↓
Gate_proj NPU (M=1, K=1024, N=3072, 2 chunks)
Up_proj NPU   (M=1, K=1024, N=3072, 2 chunks)
  ↓
SiLU(gate) * up → down input
  ↓
Down_proj NPU (M=1, K=3072, N=1024)
  ↓ residual +=
  ↓
→ next layer
```

### Changes to CMakeLists.txt

Add torch2aie include path to npu_infer:
```
target_include_directories(npu_infer PRIVATE
    /home/bcloud/torch2aie/examples
    /home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering
)
```

And link paths for XRT:
```
target_link_directories(npu_infer PRIVATE
    /home/bcloud/torch2aie/toolchain/xrt/lib64
)
```

### Instruction File for Config1

The pre-built instruction file at:
`/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/insts_4096x4096x2048_128x64x128.txt`
(44KB binary, ~11K instructions)

This works for ALL projection shapes because the padded dimensions [XM,XK,XN] = [4096,4096,2048]
are fixed. The activations and weights are padded to these dimensions.

### Weight Strategy

At model load time:
1. For each tensor, compute number of N-chunks
2. For each chunk, pre-pack (transpose → pad → shuffle → BFP16 encode)
3. Store in a `PackedWeight` struct with:
   - `std::vector<uint8_t> data` — BFP16 bytes
   - `int orig_rows, orig_cols` — original dimensions
   - `int n_chunks` — number of 2048-wide N-chunks

For GEMM execution at runtime:
1. Pad activation to [XM, XK], shuffle, BF16
2. Upload activation BO
3. Upload weight chunk BO (pre-packed, or cache in BO from init)
4. Clear output BO
5. Kernel call
6. Read back, unshuffle, extract [M, N] region
7. Repeat for each N-chunk

### Optimization: Cache weights in BOs

At init time, upload all packed weight chunks to per-layer BOs (group 4).
Then at runtime, just sync the activation BO and call the kernel.
No weight upload per layer — only activation sync + kernel submit + output readback.

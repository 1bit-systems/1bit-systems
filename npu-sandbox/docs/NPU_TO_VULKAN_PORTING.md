# NPU → Vulkan Compute Porting Guide

**Date:** 2026-07-07  
**Target Hardware:** AMD XDNA2 NPU (Strix Halo) → Vulkan 1.3 Compute (any GPU with `VK_KHR_cooperative_matrix` or subgroup ops)  
**Source codebase:** `/home/bcloud/npu-sandbox/npu-infer/` + `/home/bcloud/spec-decode/`

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Component 1: Ternary NPU Kernel](#component-1-ternary-npu-kernel)
3. [Component 2: INT8 GEMM Kernels](#component-2-int8-gemm-kernels)
4. [Component 3: Chess Kernel Objects](#component-3-chess-kernel-objects)
5. [Component 4: BFP16 Kernel (blocked)](#component-4-bfp16-kernel-blocked)
6. [Component 5: NPU Engine Architecture](#component-5-npu-engine-architecture)
7. [Component 6: BitNet Server](#component-6-bitnet-server)
8. [Component 7: Speculative Decode Engine](#component-7-speculative-decode-engine)
9. [Vulkan Memory Model Mapping](#vulkan-memory-model-mapping)
10. [Expected Performance Delta](#expected-performance-delta)
11. [Implementation Roadmap](#implementation-roadmap)

---

## Architecture Overview

The NPU inference stack operates on AMD XDNA2 (Strix Halo) with these constraints:

| Property | NPU (XDNA2) | Notes |
|----------|------------|-------|
| Compute tiles | 8 AIE2 cores (1 row × 8 columns) | Each has 64 KB local SRAM |
| Tile micro-GEMM | 32×64×128 (M×K×N) per core | INT8 8×8 MACs → i32 accumulator |
| DMA pipeline | ObjectFifo: L3(DRAM)→L2(mem tile)→L1(core SRAM) | Depth-2 hardware fifos |
| Native format | INT8, BFP16 (8bf16×8ebs) | Raw BF16 DMA unsupported (hardware bug) |
| Peak throughput | 31 TFLOPS (8 cols, BFP16) | Limited by 8-column firmware cap |
| XCLBIN dispatch | 4 separate xclbins: QKV, O, GU, D | Model-specific K/N dimensions |
| Weight format | Q4NX: I8 quantized, packed in per-layer BOs | Per-tensor scale, 1 MB BO blocks |
| KV cache | INT4 quantized on CPU | group=32, bf16 scale+zp |

### Vulkan GPU Mapping (High-Level)

| NPU Concept | Vulkan Equivalent |
|-------------|-------------------|
| AIE2 core tile (32×64×128 matmul) | Compute shader workgroup with `cooperative_matrix` or subgroup intrinsics |
| ObjectFifo DMA pipeline (L3→L2→L1) | Explicit buffer copies + pipeline barriers; shared memory for L1 |
| 4 separate xclbins (QKV/O/GU/D) | 4 compute pipelines (or 1 dispatch with push constants selecting operation) |
| Per-layer weight BOs (pre-packed on device) | Storage buffers bound per descriptor set |
| Instruction stream (MLIR-generated) | Precompiled SPIR-V (no runtime interpretation) |
| CPU attention fallback | Subgroup-level attention shader |
| LM head (CPU BF16 dot product) | Compute shader reduction |

---

## Component 1: Ternary NPU Kernel

**Source:** `bf16_kernel_dev/n1_core_ternary.py`

### What It Does

This is a **MLIR generator**, not a computed kernel. It generates the same MLIR IR as the INT8 kernel but targets ternary weights. The actual ternary compute uses the IDENTICAL `mm_32x64x128.o` Chess kernel — weights are pre-converted from 2-bit packed (−1,0,+1) to INT8 before hitting the NPU.

```
GGUF 2-bit weights → tools/q2_0_to_q4nx.py → INT8 {-1,0,1} with scale baked in → NPU (thinks it's INT8)
```

### Key Architecture Details

```python
# Tile dimensions (same as INT8 kernel)
m, k, n = 32, 64, 128          # Micro-tile per AIE core
M, K, N = 128, 1024, 4096      # Typical full dimensions
mtk = 512                       # L2 K-tile (streamed as 8×64 chunks)
n_aie_cols = 8                  # 8 core tiles in parallel
n_aie_rows = 1                  # Single row of cores

# Data types
dtype_in = np.int8
dtype_out = np.int32
```

### Tiling Strategy

```
A [M×K] → L2 tiles: 32×512 (row-major in K)  → L1 tiles: 32×64 (per K-iteration)
B [K×N] → L2 tiles: 64×128                    → L1 tiles: 64×128 (same, full tiles)
C [M×N] → L1 tiles: 32×128                    → L2 tiles: 32×128 (per row × n_aie_cols)

Each core: for K//k iterations: C += matmul(A[m×k], B[k×n])
```

The runtime sequence feeds all 8 columns from the same B tile but different C stripes. A is replicated to all columns (broadcast pattern during tiling, not per-column copies — the same L2 fifo feeds all 8 cores).

### Vulkan Porting Assessment

| Aspect | NPU | Vulkan Mapping |
|--------|-----|----------------|
| **Micro-tile GEMM** | Chess obj: `matmul_i8_i32(A[32×64], B[64×128], C[32×128])` with 8×8 MACs | **Cooperative matrix**: `coopmat<int8_t, 32, 128, 64>` or **subgroup tensor**: 8×8 int8 MAD via subgroup intrinsics. Decompose 32×128 by 64 into 4×16 subgroup tiles of 8×8 each. |
| **K-loop accumulation** | K//k iters inside core, C stays SRAM (i32 accum) | Accumulate in shared memory (`groupshared int`); use `subgroupAdd` for parallel reduction. |
| **Ternary dequant** | On CPU, before NPU: 2-bit → INT8 with scale | **Same**: pre-convert 2-bit packed → INT8 on host, upload to storage buffer. OR: dequant on-the-fly in shader from 2-bit packed input (saves 4× bandwidth, costs unpack ALU). |
| **Per-block scale** | Baked into INT8 values | **Pass as uniform**: `uniform float block_scale` in push constant; multiply post-GEMM. |
| **ObjectFifo streaming** | Hardware DMA with back-pressure | **Pipeline barriers**: `VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT` → host→device transfer → compute dispatch. Double-buffer for overlap. |
| **Zero-init C** | `zero_i32(C[32×128])` kernel call before K-loop | Shared memory zero-init: `for (uint i = tid; i < 32*128; i += WORKGROUP_SIZE) smem_C[i] = 0; barrier();` |

### Native Ternary Kernel (Future)

The future "native ternary" xclbin replaces `mm_32x64x128.o` with a Chess kernel that uses 2-bit packed tiles natively (AIE SIMD ternary ops). In Vulkan this maps to:

- **Storage buffer with 2-bit packed weights**: 4 values per byte
- **Compute shader with bit-extract**: `int bitpair = (packed_byte >> (2 * bit_idx)) & 0x3; int weight = bitpair - 1; // maps {00,01,10}→{-1,0,1}`
- **No dequant pass needed**: saves host→device bandwidth
- **Per-block scale multiplies output** not input

**Vulkan native ternary GLSL pseudocode:**
```glsl
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require

layout(local_size_x=32, local_size_y=4) in; // 128 threads = 32 M × 4 N

shared int smem_C[32][128];
shared int8_t smem_A_block[32][64];
shared int8_t smem_B_block[64][128];

void main() {
    uint m = gl_LocalInvocationID.x;  // 0..31
    uint n = gl_LocalInvocationID.y * 32 + gl_LocalInvocationID.z * ...; // tile

    // Zero accumulator
    smem_C[m][n] = 0;
    barrier();

    for (uint k_block = 0; k_block < K / 64; k_block++) {
        // Load A tile (int8 activations)
        smem_A_block[m][gl_LocalInvocationID.x * ...] = A[...];
        // Load B from 2-bit packed: bit-extract + expand to int8
        uint byte_idx = (k_block * 64 + ...) * (N / 4) + n / 4;
        uint shift = (n % 4) * 2;
        int w = int((B_packed[byte_idx] >> shift) & 0x3) - 1;  // {-1,0,1}
        smem_B_block[gl_LocalInvocationID.x][n] = int8_t(w * 127); // scale baked in
        barrier();

        // MAC accumulate (32×64×128 = unroll 8 × 8 × 16 iterations)
        for (uint k = 0; k < 64; k++) {
            smem_C[m][n] += int(smem_A_block[m][k]) * int(smem_B_block[k][n]);
        }
        barrier();
    }

    // Write output (with block scale)
    C[m * N + n] = float(smem_C[m][n]) * block_scale;
}
```

**Performance expectation**: 2-bit native should be ~3-4× faster than INT8 dequant path (4× less weight data movement, same compute).

---

## Component 2: INT8 GEMM Kernels

**Sources:** `n1_core_i8_v2.py`, `n1_core_i8_v20.py`, `n1_core_i8_minimal.py`  
**Tests:** `test_int8_v2.cpp`, `test_i8_v2_fixed.cpp`, `test_int8_xclbin.cpp`

### What They Do

These are MLIR generators that produce xclbins for INT8 GEMM on the NPU. Each generator targets a different K/N dimension combo:

| XCLBIN | M | K | N | Purpose |
|--------|---|---|---|--------|
| `final_i8_QKV_v` | 128 | H=1024 | 4096 | Q+K+V concatenated projection |
| `final_i8_O_v` | 128 | NH×HD=2048 | H=1024 | Attention output projection |
| `final_i8_GU_v` | 128 | H=1024 | 6144 | Gate+Up concatenated |
| `final_i8_D_v` | 128 | IM=3072 | H=1024 | FFN down projection |

Tile size is always `m×k×n = 32×64×128` per core, mapped across 8 AIE columns.

### Key Pipeline: v2 (Broadcast A)

```
A flow: shim → mem (depth 2, [32,512]) → 8 cores (depth 2, [32,64])
B flow: shim → mem (depth 2, [64,128]) → 1 core  (depth 2, [64,128])
C flow: 8 cores → mem → shim (depth 1 then 2)
```

A is **broadcast** to all 8 cores via a single ObjectFifo with `dimensionsFromStream` sub-views: each core reads `32×64` at offset `j*64` from the `32×512` L2 buffer (K-interleaved). This means all cores compute on the same A data but different K-slices.

> **Note:** v20 redesigns this to per-shim **unlinked** fifos (each shim→mem→core is independent, depth=16, no K-interleaving). This avoids linked-pool depth limits but doubles A data movement.

### Minimal (n1_core_i8_minimal.py)

Single-core, single-tile test: `32×64×128` matmul. Used for debugging — verifies the Chess kernel works in isolation.

### Vulkan Porting Assessment

| Aspect | NPU | Vulkan Mapping |
|--------|-----|----------------|
| **4 separate xclbins** | Each has different K/N dimensions compiled into the MLIR | **Single compute pipeline + push constants**: `layout(push_constant) struct { uint K, N, op_type; }`. Build SPIR-V once, select operation by push constant. Specialize K with `VK_EXT_shader_module_identifier` or just branching. |
| **Tile dimensions 32×64×128** | Hardware-fixed (AIE register file size) | **Parameterize in shader**: `BM=32, BK=64, BN=128`. Workgroup size: 32×4=128 threads (1 per M, 4 per N via vector). Or `cooperative_matrix` with `gl_CooperativeMatrixSize=32×128, K=64`. |
| **A broadcast (all cores same A)** | ObjectFifo links to all 8 cores | **Shared memory broadcast**: one workgroup per column-tile. All workgroups read same A rows from global. Use `subgroupBroadcast` for intra-workgroup sharing. |
| **K-interleaving (v2)** | `dimensionsFromStream` splits 32×512 into 8×32×64 sub-views | **Explicit offset**: `A[(m_tile + subgroup_id)*K + k_tile*64 + k]`. Tile assignment per workgroup. |
| **ObjectFifo depth=2** | Hardware pipe with back-pressure | **Double-buffered shared memory**: load next block to `smem_A[1]` while computing on `smem_A[0]`. Swap buffers with barrier. |
| **INT8 → i32 accumulator** | 8×8 MACs in Chess kernel | **`i8vec4` + `i16vec4` intermediate**: `subgroupInclusiveAdd` for row reduces. Or use `cooperative_matrix<int8_t>` which accumulates in i32 natively. |
| **Scale application** | Post-GEMM: `float(Cm[i]*scale_a*scale_b)` | **Post-GEMM in shader**: `float val = float(accum) * scales.a * scales.b;` |
| **Test pattern (row=index)** | `a[i] = i/K; b[0..N] = 1` → expected `C[m][0] = m*K` | Same test in Vulkan: upload test BOs, dispatch, read back, verify. Use Vulkan debug printf or write-then-read pattern. |

### Vulkan INT8 GEMM Shader Sketch

```glsl
#extension GL_KHR_cooperative_matrix : require

layout(local_size_x_id=0, local_size_y_id=1) local_size_x = 32;
layout(push_constant) uniform PC { uint M, K, N; float ascale, bscale; } pc;

coopmat<int8_t, gl_ScopeSubgroup, 32, 128, 64, gl_MatrixOperandA> matA;
coopmat<int8_t, gl_ScopeSubgroup, 32, 128, 64, gl_MatrixOperandB> matB;
coopmat<int32_t, gl_ScopeSubgroup, 32, 128, 64, gl_MatrixOperandAccumulator> matC;

void main() {
    // Load A (32×64 tile from global), B (64×128 tile from global)
    coopMatLoad(matA, A_buf, K, 0, gl_CooperativeMatrixLayoutRowMajor);
    coopMatLoad(matB, B_buf, N, 0, gl_CooperativeMatrixLayoutRowMajor);

    // Accumulate over K dimension
    for (uint kb = 64; kb < pc.K; kb += 64) {
        coopmat<int8_t> matA_next, matB_next;
        coopMatLoad(matA_next, A_buf + kb, pc.K, 0, gl_CooperativeMatrixLayoutRowMajor);
        coopMatLoad(matB_next, B_buf + kb * pc.N, pc.N, 0, gl_CooperativeMatrixLayoutRowMajor);
        matC = coopMatMulAdd(matA_next, matB_next, matC);
    }
    matC = coopMatMulAdd(matA, matB, matC); // final accumulate

    // Store with scale
    coopMatStore(matC, C_buf, pc.N, 0, gl_CooperativeMatrixLayoutRowMajor);
}
```

**If `cooperative_matrix` not available**, fall back to subgroup tiled matmul:
- 32 threads, each computes one M-row of 128 N-columns
- Load 64 K-values to registers, unrolled 8×8 dot product per N
- Accumulate in `int` register, store with scale

---

## Component 3: Chess Kernel Objects

**Sources:** `mm_bf16_v3.cc`, `attn_scalar.cc`, `silu_gate.cc`

These are the C++ kernels compiled by the **Chess compiler** (AMD AIE toolchain) and linked as `.o` files into the xclbin. They define what happens inside a single AIE core tile.

### Kernel: `mm_32x64x128.o`

Referenced by all INT8 MLIR generators. Provides:

```cpp
// Equivalent Chess API (not the actual .o source, but behavior inferred from tests)
extern "C" {
    void zero_i32(int32_t *C);      // Zero 32×128 i32 array
    void matmul_i8_i32(int8_t *A, int8_t *B, int32_t *C);  // C += A×B
}
```

The actual kernel uses AIE vector intrinsics:
- `aie::mmul<8,8,8,int8,int8>` — 8×8×8 INT8 MAC producing accfloat
- Loops: `m/8 × n/8 × k/8 = 4 × 16 × 8 = 512` MMUL instructions per call
- A is loaded in 8×8 chunks, B in 8×8 chunks, accumulator stays in accfloat registers
- `aie::mac(acc, a_vec, b_vec)` — fused multiply-accumulate

### Kernel: BF16 Matmul (mm_bf16_v3.cc) — **NON-WORKING**

This tries raw BF16 matmul but hangs due to DMA limitation. The code structure is:

```cpp
void matmul_vectorized_bf16(bfloat16 *pA, bfloat16 *pB, bfloat16 *pC) {
    // 4×16 tiles of 8×8 = 512 mac operations
    // Uses global counter to shift output offset (tile rotation)
    aie::accum<accfloat, 8*8> acc(aie::load_v<8*8>(&pCs[...]));
    acc = aie::mac(acc, av, bv);  // av=8×8, bv=8×8
    aie::store_v(&pCs[...], acc.to_vector<bfloat16>());
}
```

**Why it hangs**: The XDNA2 DMA engine cannot stream raw BF16 data — it only supports BFP16 (packed 9 bytes/8 values) and INT8. All BF16 xclbin variants (emulation, identity, native) hang on the first kernel call because DMA tries to fetch BF16 from DRAM to tile SRAM and stalls.

### Kernel: Attention Scalar (attn_scalar.cc)

Scalar attention for NPU on-tile computation:

```cpp
void attn_scalar_bf16(int32_t *q, int32_t *kv, int32_t *out, int32_t C) {
    // q: NH*HD bf16 pairs packed as i32 (2048 bf16 = 1024 i32)
    // kv: [K part][V part], each NKV*C*HD bf16
    // For each head h (0..15), kvh = h/2:
    //   - Score: Q_h · K_kvh^T / sqrt(HD) → softmax
    //   - Output: weighted sum of V_kvh
}
```

This is the per-window xclbin (attn_w0..w3.xclbin) that handles attention for 4 KV-head windows in parallel. However, the engine code actually uses **CPU attention** (SIMD with AVX-512) because the BF16 packing overhead makes NPU attention slower for <32 tokens.

### Kernel: SiLU Gate (silu_gate.cc)

```cpp
void silu_gate_bf16(int32_t *gu_out, int32_t *d_in, int32_t n) {
    // gu_out: [gate[0..n-1], up[0..n-1]] — 6144 bf16 for IM=3072
    // Computes: d_in[i] = sigmoid(gu_out[i]) * gu_out[i + n]
    // Fast sigmoid via polynomial approximation of exp(-|x|)
}
```

### Vulkan Porting Assessment

| Chess Kernel | Vulkan Equivalent |
|-------------|-------------------|
| **zero_i32** | `for (uint i=tid; i<32*128; i+=WGSIZE) smem_C[i]=0; barrier();` |
| **matmul_i8_i32** | Cooperative matrix `coopMatMulAdd` as shown above, or subgroup tiled dot product |
| **matmul_bf16** | **Irrelevant** — raw BF16 DMA won't work on NPU, use BFP16 or fp32 on Vulkan |
| **attn_scalar** | Subgroup attention with shared memory QK^T scores + softmax + weighted V sum |
| **silu_gate** | Trivial element-wise: `sigmoid(gate[i]) * up[i]` — single compute shader, 1 workgroup |
| **8×8×8 MAC intrinsic** | `cooperative_matrix<int8_t, 8, 8, 8>` or manual `subgroupAdd` with `i32` accumulator |
| **Global counter (tile rotation)** | Workgroup ID: `gl_WorkGroupID.x, .y` for M/N tile indices |

---

## Component 4: BFP16 Kernel (Blocked)

**Source discussion:** `CONCLUSION.md`, `FIRMWARE-NOTES.md`, `README.md`

### Status

BFP16 (Block Floating Point 16) is the **only working BF16-like format** on XDNA2. Raw BF16 is a hardware DMA limitation, not a compiler bug — confirmed by 4 variants all hanging while BFP16 works at 31 TFLOPS.

**BFP16 format:** 8 BF16 values → 9 bytes: `[exp_u8][mant0_s8]...[mant7_s8]`

The working BFP16 xclbin uses the `v8bfp16ebs8` Vitis type and `mac_8x8_8x8T` intrinsic designed for BFP16. However the engine uses INT8 quantized weights because BFP16:
- Requires weight conversion to 8-wide block format
- Has instruction count issues in aiecc for M=256 (multi-token)
- Achieves RMSE 0.0003 with scale=1.0 (practically lossless)

### Vulkan Relevance

**NONE.** Vulkan GPUs natively support fp16 and fp32. There's no need for BFP16. Simply use fp16 precision for all activations and weights if the GPU supports it (most modern GPUs do via `VK_KHR_shader_float16_int8`). For peak throughput, use fp16 MACs with fp32 accumulation.

---

## Component 5: NPU Engine Architecture

**Sources:** `src/npu_engine_mt.cpp` (production, 97 tok/s), `src/npu_engine_i8.cpp` (earlier), `include/engine.h` (OOP wrapper), `include/model_config.h` (Q4NX parser)

### Engine Flow

```
1. Init:
   - mmap Q4NX model file (I8 weights + BF16 norms/embeddings)
   - Open NPU device (xrt::device(0))
   - Load 4-5 xclbins (QKV, O, GU/G+U, D)
   - Dequant + repack all 28-layer weights into per-layer INT8 BOs on device
   - Pre-compute RoPE tables (4096 × HD=128 per position)

2. Prefill (batch M input tokens):
   embed[M*H] = lookup(embed_tokens, input_tokens)
   for layer 0..27:
     rms_norm(h, in_norm[l])
     qkv = GEMM_QKV(h[M×1024])          // → [M×4096] = Q + K + V concatenated
     split: Q[M×2048], K[M×1024], V[M×1024]
     rms_norm(Q), rms_norm(K)
     RoPE(Q, K) at absolute positions 0..M-1
     append K, V to KV cache (INT4 quantized, group=32)
     attn_out = SIMD_attention(Q, all cached K/V, M tokens)
     o = GEMM_O(attn_out[M×2048])       // → [M×1024]
     residual_add(h, o)
     rms_norm(h, pa_norm[l])
     gate_up = GEMM_GU(h[M×1024])       // → [M×6144]
     silu_gate × up → [M×3072]
     down = GEMM_D(silu[M×3072])        // → [M×1024]
     residual_add(h, down)

3. Decode (autoregressive, M=1):
   Same as layer loop but M=1
   After all layers:
     rms_norm(h, final_norm)
     lm_head = h · embed_tokens^T (CPU BF16 dot product, 151936×1024)
     argmax → next token

4. Output:
   Prefill: output top-1 token IDs (one per input token)
   Decode: output generated token IDs (one per step)
```

### Key Design Choices

- **4 xclbins, not 1**: Different K/N dimensions per operation require different MLIR tensor shapes. A "unified xclbin" would need dynamic shapes which aiecc doesn't support.
- **CPU attention**: AVX-512 SIMD attention (`_mm512_fmadd_ps`) is faster than NPU attention for <32 tokens due to BF16 packing overhead.
- **CPU LM head**: 151,936 × 1,024 = 155M BF16 dot products. AVX-512 makes this ~1ms on CPU. Moving to NPU would require another GEMM xclbin.
- **INT4 KV cache**: 32-element groups, BF16 scale+zero. Compresses KV from fp32 to ~1.125 bytes/element (vs 4 bytes fp32).
- **Dynamic activation scale**: Computes `amax(abs(activations)) / 127` per GEMM call to avoid clipping.

### Vulkan Porting Assessment

| Engine Component | NPU Implementation | Vulkan Mapping |
|-----------------|-------------------|----------------|
| **Weight loading** | mmap Q4NX → dequant I4→f32 → quant f32→I8 → upload to NPU BOs | **Vulkan host→device transfer**: dequant on CPU, upload to `VkBuffer` with `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`. Pipeline: `vkMapMemory` → copy → `vkUnmapMemory` → `vkQueueSubmit` with transfer barrier. |
| **4 xclbins** | Separate kernel objects loaded per type | **4 compute pipelines**: `VkPipeline` per operation type (QKV/O/GU/D). Same SPIR-V with specialization constants for K/N dimensions. |
| **Layer loop (28 iterations)** | Sequential: mm→attn→mm→mlp | **Sequential dispatches**: each layer is 4-5 vkCmdDispatch calls with barriers between. No pipelining across layers (KV cache dependency). |
| **KV cache** | INT4 quantized on CPU, dequant per-attention call | **GPU-side KV cache**: store fp16 in device-local buffer. No quantization needed (GPU VRAM is abundant). Index by `[layer][head][position][dim]`. |
| **RMS Norm** | CPU scalar loop | **Compute shader**: reduction (sum of squares) via subgroup ops, then element-wise multiply. `<100μs per call`. |
| **RoPE** | Pre-computed cos/sin tables, element-wise apply | **Pre-computed texel buffer**: `samplerBuffer` with `[4096][64]` float cos/sin pairs. Or compute on-the-fly in shader. |
| **Attention** | CPU AVX-512 SIMD | **FlashAttention-style compute shader**: tile Q·K^T into shared memory, softmax with online rescaling, accumulate V. Subgroups of 32 handle one head each. 2-pass: QK scores + softmax → V reduction. |
| **SiLU gate** | CPU scalar `g/(1+exp(-g)) * u` | **Single compute dispatch**: element-wise, trivially parallel. Merge with GEMM_GU as post-pass. |
| **LM head** | CPU BF16 dot product | **GEMM shader**: [1, 1024] × [1024, 151936] = [1, 151936]. Single workgroup reduction with atomic max for argmax. Or use GEMM_D xclbin-like shader. |
| **Token sampling** | CPU argmax or temperature sampling | **GPU reduction**: `subgroupMax`/`subgroupAdd` for top-k. For temperature sampling, `subgroupInclusiveAdd` prefix sum + binary search. |
| **Multi-token batch (M>1)** | single-threaded for loop M times | **Batch dispatch**: M rows in parallel. Each workgroup handles one token row through all layers. Requires M× memory but gains M× parallelism. |

---

## Component 6: BitNet Server

**Source:** `1bit/npu_server.py`

### What It Does

OpenAI-compatible HTTP API server:

```
GET  /health              → {"status": "ok"}
GET  /v1/models           → model list
POST /v1/chat/completions → inference request
     - messages: [{"role":...}, ...]  → Qwen3 chat template
     - stream: bool                    → SSE streaming
     - max_tokens: int (max 64)
```

### Inference Loop

```
1. build_prompt(messages) → Qwen3 chat template string
2. encode(prompt)         → C tokenizer subprocess → token IDs
3. Truncate to 256 tokens
4. run_engine(tokens, gen_count):
     sudo sh -c "NPU_GEN={gen} npu_engine_mt {model} {token_ids...}"
     → parses stdout: first n_input tokens = prefill predictions
     → remaining tokens = generated (autoregressive)
5. For streaming: decode each token as it arrives
   For non-streaming: decode all, return as content
```

### Performance

The server adds minimal overhead — it's a thin wrapper around `subprocess.call`. The bottleneck is entirely in `npu_engine_mt` (97 tok/s baseline).

### Vulkan Porting Assessment

| Aspect | NPU Server | Vulkan Server |
|--------|-----------|---------------|
| **Architecture** | Python HTTP → subprocess C++ engine | **In-process C++ server** with Vulkan dispatch. No subprocess overhead. Use `libhttpserver` or `cpp-httplib`. |
| **Chat template** | Qwen3: `<\|im_start\|>role\nmsg<\|im_end\|>` | **Same tokenizer**: port C tokenizer or use HuggingFace `tokenizers` C API. |
| **Streaming** | SSE chunks per token | **Same SSE format**: write to socket after each vkQueueSubmit completes. |
| **Model loading** | One-time at server start | **One-time at init**: create Vulkan instance, device, pipelines, upload weights. |
| **sudo requirement** | Yes (NPU needs IOMMU permissions) | **No** (Vulkan runs in userspace with GPU driver). |
| **Concurrency** | Single-request (blocks on engine subprocess) | **Multi-queue**: multiple VkQueues or timestamp-interleaved dispatches for concurrent requests. |
| **Environment config** | `NPU_GEN`, `NPU_TEMP`, `PORT` env vars | **Config file** or CLI flags. |
| **KV cache isolation** | Per-request (engine process owns KV) | **Session-based**: assign KV cache buffer per session ID. Cache persists across requests. |

---

## Component 7: Speculative Decode Engine

**Sources:** `engine/spec_decode.h`, `engine/npu_spec_integration.cpp`, `engine/npu_target_model.h`, `draft/dspark_draft.h`

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   SpeculativeDecoder                        │
│                                                             │
│  ┌──────────┐    hidden     ┌─────────────┐                │
│  │  TARGET  │──────────────▶│   DRAFT     │                │
│  │  (NPU)   │               │ (DSpark/MT) │                │
│  │  94 tok/s│               │  CPU model  │                │
│  └────┬─────┘               └──────┬──────┘                │
│       │                            │                        │
│       │ logits              draft_tokens[0..N-1]            │
│       ▼                            │                        │
│  ┌──────────┐                      │                        │
│  │ VERIFY   │◄─────────────────────┘                        │
│  │ one pass │  target.forward_with_kv(all_tokens, past)     │
│  └────┬─────┘                                               │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────┐                                               │
│  │ REJECT   │  per-token comparison: draft[i] == target[i]   │
│  │ SAMPLE   │  accept match, break on first mismatch         │
│  └──────────┘                                               │
└─────────────────────────────────────────────────────────────┘
```

### Key Parameters

```cpp
block_size = 7;              // N speculative tokens per draft round
num_target_layers = 5;       // hidden states extracted from layers {1,6,12,18,24}
num_draft_layers = 5;        // DSpark backbone depth (Eagle3=1)
target_layer_ids = {1,6,12,18,24};  // which target layers feed draft model
vocab_size = 151936;
hidden_size = 1024;
```

### Current Status

- **DSpark** (5-layer draft): 0.1-0.2 tok/s (broken). Draft trained on HF FP hidden states, rejected 100% against INT8 NPU features.
- **Target model** (NPUQwen3Target): Same I8Ctx dispatch as `npu_engine_mt`, but packaged as `TargetModelInterface`.
- **MTP/Eagle3** (mtp_draft.h): 1-layer draft, untrained. Fallback draft model.
- **Needs**: Draft retrained on NPU INT8 hidden states for >0% acceptance.

### Vulkan Porting Assessment

| Aspect | NPU Implementation | Vulkan Mapping |
|--------|-------------------|----------------|
| **Target model** | NPUQwen3Target: 4 xclbin I8Ctx dispatch on NPU | **GPU GEMM shaders**: same as Component 5 Vulkan mapping. |
| **Draft model** | DSpark 5-layer CPU model; Eagle3 1-layer CPU model | **GPU draft model**: run draft on GPU as lightweight compute shaders. DSpark is 5× small transformer layers. Eagle3 is 1 layer. Both fit easily in shared memory for M=7 block. |
| **Draft autoregression** | CPU scalar for-loop, 7 iterations | **GPU sequential with shared memory state**: 7 iterations in single shader with `barrier()` between. KV cache for draft is 7× small (7-token context). |
| **Verification** | `forward_with_kv(tokens, verify_len=1+N, past_len)` | **Single dispatch**: run target forward on [1+N] tokens with existing KV cache. Same as prefill but with past positions. |
| **Rejection sampling** | CPU scalar comparison | **GPU reduction**: `subgroupAllEqual` per draft position. Early-exit at first mismatch. |
| **KV cache rollback** | `commit_accepted(start_pos, n_accepted)` → just sets `kv.n` | **Track KV length**: store KV count in uniform. If rejected, subsequent dispatches only read `n_accepted` entries. No data movement needed. |
| **Target hidden extraction** | CPU copy from `layer_hidden_snapshot_[]` | **Pipeline barrier + read**: after target forward completes, read hidden states from output buffer (specific layers specified by push constant). |
| **End-to-end orchestration** | `spec_decode.h` template class on CPU | **CPU orchestration**: same pattern. GPU dispatches are async; use `vkFence` or `vkQueueWaitIdle` to synchronize draft ↔ verify steps. |

---

## Vulkan Memory Model Mapping

### NPU Memory Hierarchy → Vulkan

| NPU Memory | Size | Access Pattern | Vulkan Equivalent |
|-----------|------|---------------|-------------------|
| L0: AIE register file | ~2 KB per core | 8×8 tiles, 1 cycle | Subgroup invocation private registers |
| L1: Core SRAM | 64 KB per core | ObjectFifo, 1-3 cycle load | **Shared memory** (`groupshared`, 32-96 KB per workgroup) |
| L2: Mem tile SRAM | 512 KB per column | DMA buffer pool | Shared memory (for larger tiles) or device-local cache |
| L3: NPU DRAM | 32 GB shared | DMA, 64B bursts | **Device-local VkBuffer** (`VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`) |
| Host DRAM | MMAP'd model file | XRT `sync_to_device()` | **Host-visible staging buffer** → `vkCmdCopyBuffer` → device-local |

### Weight Buffer Layout

```
NPU: 28 layers × 7 weight types × N_blocks × 1 MB BOs
     Each BO: [block_rows=256 × block_cols=1024 × 1 byte (int8)] = 256 KB
     Total: ~28 × 7 × 4 (avg blocks) × 256 KB ≈ 200 MB

Vulkan: 
  - Single large storage buffer per operation type
  - QKV weights: [28 layers, H=1024, 4096] int8 = 112 MB
  - O weights:   [28 layers, 2048, 1024] int8 = 56 MB
  - GU weights:  [28 layers, 1024, 6144] int8 = 168 MB
  - D weights:   [28 layers, 3072, 1024] int8 = 84 MB
  Total: ~420 MB (2× NPU due to less efficient packing, but Vulkan GPUs have 8-24 GB VRAM)
```

### Descriptor Set Layout

```cpp
// Set 0: Per-operation weights (bound once, reused all layers)
layout(set=0, binding=0) buffer Weights { int8_t data[]; } weights;
layout(set=0, binding=1) uniform Params { uint K, N, layer_stride; float scale[]; } params;

// Set 1: Per-layer dynamic state
layout(set=1, binding=0) buffer Activations { float data[]; } activations;
layout(set=1, binding=1) buffer KV_Cache { float data[]; } kv_cache;
layout(set=1, binding=2) buffer Output { float data[]; } output;
layout(set=1, binding=3) buffer RMSNorm_Weights { float data[]; } rms_weights;
```

---

## Expected Performance Delta

| Metric | NPU (Strix Halo) | Vulkan GPU (Radeon 8060S) | Ratio |
|--------|-----------------|---------------------------|-------|
| **Decode speed** | 97 tok/s (v12 baseline) | **381 tok/s** (measured, GPU infer) | **3.9× faster** |
| **Peak GEMM TFLOPS** | 31 TFLOPS (BFP16, 8 cols) | ~40 TFLOPS (fp16, 8060S) | ~1.3× |
| **Memory bandwidth** | ~100 GB/s (NPU DRAM) | 256-512 GB/s (GPU VRAM) | 2.5-5× |
| **KV cache capacity** | 4096 pos × INT4 (20 MB) | 4096 pos × fp16 (160 MB) | 8× more VRAM used, but VRAM is abundant |
| **Weight loading time** | ~200 ms (dequant + upload) | ~10-50 ms (PCIe DMA) | 4-20× faster |
| **Batch inference (M>1)** | Linear scaling (CPU for-loop) | Linear scaling (parallel workgroups) | 1-8× depending on M |
| **Attention (1 token)** | ~2 ms CPU AVX-512 | ~0.1 ms GPU (shared memory) | 20× faster |
| **LM head** | ~1 ms CPU BF16 dot | ~0.05 ms GPU reduction | 20× faster |
| **Speculative decode** | 0.1-0.2 tok/s (broken) | **1.5-2.5× over baseline** (projected) | 10-20× current, 1.5-2.5× vs baseline |

### Where Vulkan Wins

1. **Attention**: CPU attention is the bottleneck in the NPU pipeline (~2 ms per layer, 28 layers = 56 ms for attention alone). GPU attention is 20× faster via shared memory tiling.
2. **LM head**: 151,936 × 1,024 dot product is trivially parallelized on GPU.
3. **Multi-token batches**: M=8 gives 8× speedup on GPU (parallel workgroups) vs linear scaling on CPU.
4. **Memory bandwidth**: 2.5-5× more bandwidth → faster weight streaming during K-loop.
5. **Wider GEMM**: GPU can use larger tiles (64×64 or 128×128) vs NPU's fixed 32×64×128.

### Where NPU Wins

1. **Power efficiency**: NPU ~15W vs GPU ~45W (3× less power).
2. **Sustained throughput**: NPU's streaming architecture avoids GPU launch overhead for small batches.
3. **Speculative decode (fixed)**: Higher acceptance rates on same-model features (no precision mismatch between target INT8 and draft FP hidden states).

### Realistic Vulkan GPU Target

| Model | NPU tok/s | GPU tok/s (projected) | GPU tok/s (measured) |
|-------|-----------|----------------------|---------------------|
| Qwen3-0.6B INT8 | 97 | 350-400 | 381 (confirmed) |
| Qwen3-0.6B FP16 | N/A | 250-300 | ? |
| Qwen3-0.6B + spec decode | 0.1 (broken) | 500-600 (1.5x baseline) | ? |
| Qwen3-4B INT8 | ? | 80-120 | ? |

---

## Implementation Roadmap

### Phase 1: Single-Operation Vulkan GEMM (Week 1)

1. **Set up Vulkan boilerplate**: instance, device, queue, command pool
2. **Implement cooperative matrix GEMM** or subgroup-tiled GEMM for INT8
3. **Verify against NPU reference output**: test with same 128×1024×4096 dimensions
4. **Benchmark**: compare single GEMM latency vs NPU (target: 2-5× faster)

### Phase 2: Full Layer Pipeline (Week 2)

1. **Port RMS Norm**: shared memory reduction, element-wise apply
2. **Port RoPE**: pre-computed cos/sin in uniform buffer or texel buffer
3. **Port attention**: FlashAttention-style with shared memory tiling
4. **Port SiLU gate**: element-wise compute
5. **Chain into layer pipeline**: GEMM→norm→attention→GEMM→norm→GEMM→silu→GEMM
6. **Verify per-layer output** against NPU for same input tokens

### Phase 3: Full Model + Server (Week 3)

1. **Load Q4NX weights**: mmap → dequant → upload to Vulkan buffers
2. **Implement KV cache**: device-local buffer with position tracking
3. **LM head + sampling**: final GEMM, argmax, temperature sampling
4. **Server integration**: port `npu_server.py` to C++ Vulkan server
5. **End-to-end benchmark**: compare tok/s against NPU baseline

### Phase 4: Speculative Decode (Week 4)

1. **Port draft model**: DSpark 5-layer or Eagle3 1-layer as GPU compute shaders
2. **Implement verification pass**: `forward_with_kv()` target model dispatch
3. **Rejection sampling**: GPU-side comparison + early-exit
4. **Benchmark**: measure acceptance rate and total speedup

### Phase 5: Optimization (Ongoing)

1. **Double-buffering**: overlap compute with data transfer
2. **Fused kernels**: merge RMS norm into GEMM pre-pass
3. **INT4 KV cache**: optional compression for larger models
4. **Pipeline barriers audit**: minimize synchronization overhead
5. **Multi-queue**: separate compute queues for concurrent inference

---

## Key Design Decisions

### 1. cooperative_matrix vs Subgroup Tiling

| Approach | Pros | Cons |
|----------|------|------|
| **cooperative_matrix** | High-level, driver-optimized, matches NPU 8×8 MAC | Requires `VK_KHR_cooperative_matrix` (available on RDNA3+, Adreno, NV Turing+) |
| **Subgroup tiling** | Universal support, full control | More code, requires manual tuning for each GPU |

**Recommendation**: Use cooperative_matrix where available, fall back to subgroup tiling.

### 2. Weight Precision

- **Keep INT8 weights**: Proven on NPU with minimal accuracy loss
- **Optionally use fp16**: For GPU with abundant bandwidth, fp16 eliminates quantization noise
- **2-bit ternary path**: 4× bandwidth reduction, only for ternary models

### 3. Attention Implementation

Don't port NPU's CPU SIMD attention. Use GPU-native FlashAttention approach:
- Tile Q·K^T into shared memory blocks (32×32)
- Softmax rescaling (online, numerically stable)
- Accumulate V in registers

### 4. KV Cache Format

NPU uses INT4 + BF16 scales to fit 4096 tokens in CPU memory. On GPU:
- **Use fp16 directly**: 4096 pos × 8 heads × 128 dim × 2 bytes × 28 layers = 235 MB — fits easily in 8-24 GB VRAM
- Optional INT8 compression for 9B+ models

### 5. Pipeline Layout

NPU uses 4 separate xclbins with different K/N dimensions. Vulkan can use:
- **Single compute pipeline** with push constants for operation type
- **Or 4 specialized pipelines** (simpler SPIR-V, no runtime branching)
- **Recommendation**: 4 pipelines (compiler optimizes better for fixed K/N)

---

## File Inventory (Quick Reference)

| File | Lines | Purpose |
|------|-------|---------|
| `bf16_kernel_dev/n1_core_ternary.py` | 119 | Ternary MLIR generator (same as INT8, different weights) |
| `bf16_kernel_dev/n1_core_i8_v2.py` | 121 | INT8 MLIR generator v2 (broadcast A, K-interleaved) |
| `bf16_kernel_dev/n1_core_i8_v20.py` | 120 | INT8 MLIR generator v20 (per-shim fifos, unlinked) |
| `bf16_kernel_dev/n1_core_i8_minimal.py` | 82 | Minimal 32×64×128 single-core test |
| `bf16_kernel_dev/mm_bf16_v3.cc` | 37 | Chess BF16 kernel (hangs — DMA limitation) |
| `bf16_kernel_dev/attn_scalar.cc` | 70 | Chess attention kernel (scalar, per-window xclbin) |
| `bf16_kernel_dev/silu_gate.cc` | 46 | Chess SiLU gate+up kernel |
| `bf16_kernel_dev/test_int8_v2.cpp` | 49 | Host-side test: row-pattern validation |
| `bf16_kernel_dev/test_bfp16_random.cpp` | 88 | Host-side test: BFP16 random validation |
| `src/npu_engine_mt.cpp` | ~350 | Production engine: 97 tok/s, multi-token decode |
| `src/npu_engine_i8.cpp` | ~200 | Earlier engine with per-window attention xclbins |
| `src/engine.cpp` | ~250 | OOP engine wrapper (NpuInferenceEngine class) |
| `include/engine.h` | 120 | Engine class API |
| `include/model_config.h` | ~180 | Q4NX JSON parser → ModelConfig derivation |
| `include/model.h` | ~100 | ModelWeights/TensorDesc C structs |
| `include/common.h` | ~80 | Constants, xclbin paths, opcodes |
| `src/kv_quant.h` | ~30 | INT4 KV cache quantization (group=32, BF16 scale) |
| `1bit/npu_server.py` | 120 | OpenAI-compatible HTTP server |
| `spec-decode/engine/spec_decode.h` | ~320 | Speculative decode orchestrator |
| `spec-decode/engine/npu_target_model.h` | ~400 | NPUQwen3Target: I8Ctx dispatch for spec decode |
| `spec-decode/engine/npu_spec_integration.cpp` | ~100 | Main: wires target + draft + decoder |
| `spec-decode/draft/dspark_draft.h` | 746 | DSpark 5-layer draft model |
| `spec-decode/NPU_VS_GPU.md` | 54 | Performance comparison summary |
| `bf16_kernel_dev/CONCLUSION.md` | 46 | BFP16 vs BF16 DMA root cause |
| `bf16_kernel_dev/README.md` | 45 | BF16 development status |
| `bf16_kernel_dev/FIRMWARE-NOTES.md` | 42 | Firmware column limit investigation |

# Vulkan Backend Research: Pure 1-Bit Inference Engine

**Date:** 2026-07-07
**Target:** Radeon 8060S (RADV STRIX_HALO, gfx1151 / RDNA3.5, Vulkan 1.4.335)

---

## 1. Executive Summary

Building a "pure Vulkan 1-bit inference engine" is viable. The RDNA3.5 GPU (41 GB device memory, 64-wide subgroups, INT8 cooperative matrix, integer dot product) provides ample compute for ternary-weight inference. The best architecture is **subgroup-shuffle GEMM with in-shader ternary dequantization**, optionally leveraging INT8 cooperative matrix (16×16×16) for the accumulation step.

**Key Findings:**
- VK_KHR_cooperative_matrix exposes 14 configs on RADV, all 16×16×16 — INT8 inputs → I32/U32 output, plus FP16
- No existing engine does pure 1-bit Vulkan inference — closest is ggml-vulkan IQ1_S (~1.56 bpw, not pure ternary)
- Subgroup shuffle is the recommended matmul path for custom ternary formats
- We can write raw SPIR-V compute shaders (glslangValidator/glslc installed)
- IQ1_S is NOT pure ternary — it's a 4-level scalar quantization with LUT

---

## 2. llama.cpp / GGML Vulkan Backend

### 2.1 Architecture Overview

**Location:** `/home/bcloud/zaya-llama.cpp/ggml/src/ggml-vulkan/`

GGML-Vulkan uses **compute shaders compiled from GLSL** (embedded as SPIR-V byte arrays in `ggml-vulkan-shaders.hpp`). Each GGML operation has a corresponding `.comp` shader, compiled via `vulkan-shaders-gen`.

**Pipeline architecture:**
- 3 pipeline tiers per quantization: `l` (large), `m` (medium), `s` (small) — selected based on matrix dimensions
- Aligned (`a_l`, `a_m`, `a_s`) variants for aligned memory access
- Separate pipelines for FP16 accumulation vs FP32 accumulation (`f16acc` / `f32acc` struct)
- Cooperative matrix used **only for flash attention** (FA_COOPMAT1/FA_COOPMAT2 paths), NOT for matrix multiply

### 2.2 How Matrix Multiply Works (subgroup-based GEMM)

ggml-vulkan uses **subgroup-shuffle-based tiled GEMM**, NOT cooperative matrix for matmul:

```
matmul pipeline structure (vk_matmul_pipeline2):
├── f32acc (FP32 accumulation)
│   ├── l (large tile: ~64x64)
│   ├── m (medium tile: ~32x32)
│   └── s (small tile: ~16x16)
└── f16acc (FP16 accumulation)
    ├── l, m, s
```

Tile dimensions are determined by subgroup size:
- RDNA3.5: 64-wide subgroups → `coopmat_m=16, coopmat_n=16` used as tile dimensions
- On non-coopmat GPUs: tile sizes drop to 4×4 or 2×2

**Key detection code (line 3353-3356):**
```cpp
const uint32_t tm_l = device->coopmat_support ? device->coopmat_m : 4;
const uint32_t tm_m = device->coopmat_support ? device->coopmat_m : 4;
const uint32_t tm_s = device->coopmat_support ? device->coopmat_m : 2;
const uint32_t tn_l = device->coopmat_support ? device->coopmat_n : 4;
```

### 2.3 Dequantization Strategy

**In-shader dequantization** — weights are read from packed buffer and dequantized on-the-fly within the compute shader during matmul. No pre-computed weight arrays.

The `dequant_funcs.glsl` header provides `dequantize()` and `dequantize4()` functions for each quantization type. These are `#ifdef`-selected at shader compile time via `DATA_A_IQ1_S`, `DATA_A_Q4_0`, etc.

### 2.4 1-Bit Quantization Types in GGML Vulkan

| Type | Enum Value | bpw | Block Size | Block Structure | Notes |
|------|-----------|-----|-----------|----------------|-------|
| **Q1_0** | 41 | 1.0625 | 128el (QK1_0) | `d`(fp16) + `qs[16B]` = 18 bytes | True 1-bit: each bit → ±d |
| **IQ1_S** | 19 | 1.5625 | 256el (QK_K) | `d`(fp16) + `qs[32B]` + `qh[16B]` = 50 bytes | 2-bit index → 1024-entry LUT |
| **IQ1_M** | 29 | 1.75 | 256el (QK_K) | `d`(fp16) + `qs[32B]` + `qh[16B]` + `scales[8B]` = 58 bytes | Higher precision variant |
| **TQ1_0** | 34 | ? | ? | ? | Ternary quant (new in ggml) |

**IQ1_S is NOT pure 1-bit/ternary.** It's a 4-level quantization:
```
weight = block_scale * (LUT_2bit_value + delta)
delta = ±0.125 (per-subblock sign flip)
LUT_2bit_value ∈ {0, 1, 2, 3}
```
Effective values range: -0.875, -0.125, 0.875, 1.875, 2.875 (before scale)

**Q1_0 IS true 1-bit** — each weight is either `+d` or `-d`:
```glsl
const uint bit = (qs[idx/8] >> (idx % 8)) & 1u;
return bit != 0u ? d : -d; // from flash_attn_cm2.comp
```

### 2.5 Vulkan Backend Selection in Ollama

Ollama ships GGML CPU-only backends (`libggml-cpu-*.so`). **There is no `libggml-vulkan.so` in the Ollama installation** at `/usr/local/lib/ollama/`. The Vulkan backend must be compiled separately from source with `-DGGML_VULKAN=ON`.

Current Ollama uses CPU inference only:
```
ollama ps → no running models
ollama list → qwen3-coder-next:q4_K_M, qwen3-coder:30b-a3b-q4_K_M, etc.
```

### 2.6 Full List of Vulkan Shaders (vulkan-shaders/)

**Dequantization shaders:**
`dequant_f32.comp`, `dequant_iq1_m.comp`, `dequant_iq1_s.comp`, `dequant_iq2_s.comp`, `dequant_iq2_xs.comp`, `dequant_iq2_xxs.comp`, `dequant_iq3_s.comp`, `dequant_iq3_xxs.comp`, `dequant_iq4_nl.comp`, `dequant_iq4_xs.comp`, `dequant_q1_0.comp`, `dequant_q2_k.comp` through `dequant_q6_k.comp`

**Matrix multiply shaders:**
`mul_mat_vec.comp` (matrix-vector, generic), `mul_mat_vec_iq1_s.comp`, `mul_mat_vec_iq1_m.comp`, `mul_mat_vec_iq2_s.comp` through `mul_mat_vec_q6_k.comp`, `mul_mat_vecq.comp` (quantized vec), `mul_mat_split_k_reduce.comp`

**Flash attention shaders:**
`flash_attn.comp` (scalar), `flash_attn_cm1.comp` (cooperative matrix v1), `flash_attn_cm2.comp` (cooperative matrix v2/NV), `flash_attn_mask_opt.comp`, `flash_attn_split_k_reduce.comp`

---

## 3. MLC-LLM / TVM Vulkan Backend

**Not installed.** No `mlc`, `tvm`, or related pip packages found.

MLC-LLM uses Apache TVM with a Vulkan backend via SPIR-V code generation. It compiles model IR → TVM Relay → Vulkan compute shaders. Supports FP16 and INT4/INT8 quantization but has no specific 1-bit support. The TVM Vulkan backend is mature but heavyweight for a custom 1-bit engine.

---

## 4. Other Vulkan Inference Engines

### 4.1 vLLM Vulkan

No Vulkan backend. vLLM is CUDA-centric (with ROCm for AMD). No Vulkan compute support exists.

### 4.2 Rust Vulkan Compute Crates

| Crate | Version | Description | Relevance |
|-------|---------|-------------|-----------|
| **rlx-vulkan** | 0.2.12 | Native Vulkan compute for RLX (MIT-RLX), raw ash + embedded SPIR-V kernels | ★★★ Best starting point — GPL-3.0, ML-focused |
| **kronos-compute** | 0.2.3-rc3 | Compute-only Vulkan, MIT/Apache-2.0 | ★★★ Lean, no ML though |
| **vulkane** | 0.8.3 | Complete safe RAII Vulkan wrapper | ★★ For hand-rolled approach |
| oxicuda-vulkan | 0.4.1 | CUDA→Vulkan transpiler | ★ Novel but immature |

### 4.3 Candle / Burn

- **Candle** (HuggingFace): CUDA, Metal, MKL backends. **No Vulkan backend.** The `candle-ug` (user-generated) crate could theoretically host Vulkan kernels.
- **Burn 0.21.0**: Uses `burn-cubecl`, `burn-ndarray`, `burn-candle`. **No Vulkan backend.** Burn-candle is deprecated.

### 4.4 WebGPU / WebLLM

- WebGPU compute shaders (WGSL) are the browser equivalent of Vulkan compute
- WebLLM (MLC) uses WebGPU for browser inference
- Not directly applicable to native Vulkan, but algorithm patterns transfer

### 4.5 ONNX Runtime Vulkan

ONNX Runtime has a Vulkan execution provider but it's not widely used for LLMs. Primarily targets image models.

---

## 5. VK_KHR_cooperative_matrix Analysis

### 5.1 Hardware-Specific Properties (Radeon 8060S / RADV)

**All 14 cooperative matrix configurations are 16×16×16:**

```
Config  A_type   B_type   C_type   Result   Saturating
[0]     UINT8    UINT8    UINT32   UINT32   No
[1]     UINT8    UINT8    INT32    INT32    No
[2]     UINT8    UINT8    INT32    INT32    Yes
[3]     UINT8    INT8     UINT32   UINT32   No
[4]     UINT8    INT8     INT32    INT32    No
[5]     UINT8    INT8     INT32    INT32    Yes
[6]     INT8     UINT8    UINT32   UINT32   No
[7]     INT8     UINT8    INT32    INT32    No
[8]     INT8     UINT8    INT32    INT32    Yes
[9]     INT8     INT8     UINT32   UINT32   No
[10]    INT8     INT8     INT32    INT32    No
[11]    INT8     INT8     INT32    INT32    Yes
[12]    FLOAT16  FLOAT16  FLOAT16  FLOAT16  No
[13]    FLOAT16  FLOAT16  FLOAT32  FLOAT32  No
```

### 5.2 What This Means for 1-Bit Inference

**INT8 cooperative matrix CAN be used for ternary weights:**
- Map ternary values {−1, 0, +1} → INT8 {−1, 0, +1}
- 16×16×16 tile performs 256 multiply-accumulates per cooperative matrix op
- INT32 accumulation (16× result precision) — safe for large K dimensions
- Saturating accumulation available (prevents overflow)

**FP16 cooperative matrix for activations:**
- Activations are FP16, accumulated to FP16 or FP32
- 16×16×16 FP16 cooperative matrix provides 2× throughput vs scalar

**Limitations:**
- No pure INT4 or INT2 cooperative matrix — must use INT8 for 1-bit
- No BF16 cooperative matrix (though `shaderBFloat16CooperativeMatrix` feature exists)
- RDNA3.5 cooperative matrix is limited to 16×16×16 — no larger tiles
- Cooperative matrix + subgroup shuffle hybrid approach is optimal

### 5.3 Cooperative Matrix in GLSL

SPIR-V requires `GL_KHR_cooperative_matrix` extension:
```glsl
#extension GL_KHR_cooperative_matrix : enable

coopmat<float16_t, gl_ScopeSubgroup, 16, 16> matA;
coopmat<float16_t, gl_ScopeSubgroup, 16, 16> matB;
coopmat<float32_t, gl_ScopeSubgroup, 16, 16> matC;

coopMatMulAdd(matA, matB, matC, matC);  // C += A × B
```

Required for cooperative matrix to work correctly:
- GLSL compilation with `--target-env vulkan1.2` or higher
- `VK_KHR_cooperative_matrix` device extension enabled
- `cooperativeMatrix` feature enabled in device creation
- Workgroup size must be a multiple of subgroup size (64 on RDNA3.5)

---

## 6. Integer Dot Product (VK_KHR_shader_integer_dot_product)

### 6.1 RDNA3.5 Support

| Operation | Accelerated |
|-----------|-------------|
| 8-bit unsigned dot product | ✅ Yes |
| 8-bit signed dot product | ✅ Yes |
| 8-bit mixed signedness dot product | ✅ Yes |
| 4×8-bit packed unsigned | ✅ Yes |
| 4×8-bit packed signed | ✅ Yes |
| 4×8-bit packed mixed | ✅ Yes |
| 8-bit saturating accumulate (all) | ✅ Yes |
| **16-bit** dot product (all variants) | ❌ **No** |
| **32-bit** dot product (all variants) | ❌ No |
| **64-bit** dot product (all variants) | ❌ No |

### 6.2 Relevance for 1-Bit Inference

Integer dot product via `dot4add_i8packed` in GLSL provides hardware-accelerated 4×8-bit dot product per instruction. For ternary weights packed as INT8, this gives a 4× throughput multiplier over scalar math. Example:

```glsl
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

// Packed int8 dot product
uint32_t result = 0u;
result = dot4add_i8packed(packed_weights, packed_activations, result);
```

---

## 7. Subgroup Operations on RDNA3.5

### 7.1 Subgroup Properties

| Property | Value |
|----------|-------|
| Subgroup size | **64** |
| Min subgroup size | 32 (via VK_EXT_subgroup_size_control) |
| Max compute workgroup subgroups | 4,294,967,295 |
| Max compute workgroup invocations | 1024 |
| Max compute shared memory | 65,536 bytes |
| Required subgroup size stages | Compute, Fragment, Mesh, Task |

### 7.2 Supported Subgroup Operations

All operations supported: `BASIC`, `VOTE`, `ARITHMETIC`, `BALLOT`, `SHUFFLE`, `SHUFFLE_RELATIVE`, `CLUSTERED`, `QUAD`, `ROTATE`, `ROTATE_CLUSTERED`

Extended types (`VK_KHR_shader_subgroup_extended_types`): int8, int16, float16 supported for subgroup operations.

### 7.3 Subgroup Shuffle Throughput Estimate

On RDNA3.5 (gfx1151), a 64-wide subgroup shuffle has ~4-cycle latency per shuffle. With `subgroupShuffleXor`, the classic Kogge-Stone reduction takes log2(64) = 6 steps → ~24 cycles for a full subgroup sum reduction. This is extremely fast for matmul inner product accumulation.

### 7.4 Subgroup Shuffle GEMM Pattern

The standard approach (used by ggml-vulkan's `mul_mat_vec`):
```
1. Each thread loads a tile of A and B into registers
2. Multiply individual elements (FP16 or INT8)
3. Butterfly reduction via subgroupShuffleXor across 64 threads
4. First thread writes result to shared memory / output buffer
5. Performs well when K-dimension tile aligns with subgroup size
```

---

## 8. SPIR-V Compute Shader Approach

### 8.1 Toolchain Available

| Tool | Version | Path |
|------|---------|------|
| glslangValidator | 16.2.0 | `/usr/bin/glslangValidator` |
| glslc (shaderc) | 2026.1 | `/usr/bin/glslc` |
| spirv-tools | 2026.1-1 | `/usr/bin/spirv-*` |
| spirv-headers | 1.6.1+1.4.341.0 | system |

### 8.2 GLSL → SPIR-V Compilation

```bash
# Compile GLSL compute shader to SPIR-V
glslangValidator -V --target-env vulkan1.2 \
    -o kernel.spv kernel.comp

# Or using glslc (preferred for Vulkan)
glslc --target-env=vulkan1.2 \
    -fshader-stage=compute \
    -o kernel.spv kernel.comp
```

### 8.3 Recommended Shader Architecture for Ternary GEMM

```glsl
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_shuffle : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in; // subgroup-sized

// Packed ternary weights: 2 bits per weight, 32 weights per uint64_t
layout(binding = 0) readonly buffer Weights { uint64_t packed_weights[]; };
layout(binding = 1) readonly buffer Scales  { float16_t scales[]; };
layout(binding = 2) readonly buffer Input   { float16_t input[]; };
layout(binding = 3) writeonly buffer Output { float16_t output[]; };

// Ternary decode: 0b00→-1, 0b01→0, 0b10→+1, 0b11→unused
float16_t decode_ternary(uint64_t packed, uint idx) {
    uint bits = uint((packed >> (2 * idx)) & 3);
    // Map 0→-1, 1→0, 2→+1 using subgroup ops or lookup
    return float16_t(int(bits) - 1);  // 0→-1, 1→0, 2→+1
}

void main() {
    uint row = gl_WorkGroupID.x;
    uint lane = gl_LocalInvocationID.x;  // 0..63

    // Cooperative load: each lane loads 1 activation
    float16_t activation = input[lane];
    float16_t scale = scales[row];

    // Each lane decodes its weight from packed buffer
    float16_t weight = decode_ternary(packed_weights[row * PACK_RATIO + lane/32], lane % 32);

    // Multiply
    float32_t product = float32_t(activation) * float32_t(weight) * float32_t(scale);

    // Subgroup reduction (butterfly)
    for (uint offset = 32; offset > 0; offset >>= 1) {
        product += subgroupShuffleXor(product, offset);
    }

    // First lane writes result
    if (lane == 0) {
        output[row] = float16_t(product);
    }
}
```

---

## 9. Key Design Decisions

### 9.1 Cooperative Matrix vs Hand-Rolled Subgroup GEMM

| Factor | Cooperative Matrix (INT8, 16×16×16) | Subgroup Shuffle GEMM |
|--------|--------------------------------------|----------------------|
| **Throughput** | 256 MACs/op (hardware tensor cores) | ~64 MACs/reduction (SIMD) |
| **Flexibility** | Fixed 16×16×16 tile, limited types | Any tile shape, any type |
| **Dequant overhead** | Must dequant BEFORE coop mat (extra pass) | Dequant in inner loop (fused) |
| **Ternary support** | Map to INT8, waste 6 bits/value | Native 2-bit packing possible |
| **Mixed precision** | ACC types limited to I32/U32/F32/F16 | Any combination |
| **Portability** | VK_KHR_cooperative_matrix required | Works on ANY Vulkan 1.1+ GPU |

**Recommendation: Hybrid approach**
- Use **subgroup shuffle** for dequantization + element-wise multiply (fused, no wasted bandwidth)
- Use **INT8 cooperative matrix** for the accumulation step if the tile dimensions match (16×16)
- For small matrices (prompt processing), subgroup shuffle alone is sufficient
- For large matmuls (decode phase), cooperative matrix wins

### 9.2 FP16 vs FP32 Accumulation

| Factor | FP16 | FP32 |
|--------|------|------|
| Register pressure | 1 register/value | 2 registers/value |
| Precision | ~3.3 decimal digits | ~7.2 decimal digits |
| Coop mat support | YES (config [12]) | YES (config [13]) |
| Subgroup shuffle | Faster (1 cycle) | Slower (2 cycles) |
| Risk for large K | **Overflow risk above K~2048** | Safe to K~1M+ |

**Recommendation: FP32 accumulation for safety, FP16 for speed-critical inner loops**

For ternary GEMM specifically:
- The product range is small (±activation × scale) — FP16 accumulation is safe for K ≤ 4096
- For larger K, use FP32 or split-K reduction
- ggml-vulkan uses both: `f16acc` for speed, `f32acc` for precision, selected at pipeline creation

### 9.3 2-Bit Weight Packing in Vulkan Buffers

**Format options for ternary weights:**

```
Option A: 2 bits per weight (packed u32)
  PROS: Memory efficient (4 weights/byte), 16 weights per uint32_t
  CONS: Requires bitfieldExtract in shader, no native Vulkan format
  Storage: uint32_t packed[] where each 32-bit word holds 16 ternary values

Option B: 8-bit per weight (unpacked INT8)
  PROS: Native cooperative matrix support, no decode overhead
  CONS: 4× memory waste (8 bits vs 2 bits)
  Storage: int8_t weights[] where each byte is -1, 0, or 1

Option C: 2 bits + scale (block format like IQ1_S)
  PROS: Same memory as Option A + per-block scaling
  CONS: Decode overhead similar to IQ1_S
  Storage: block { fp16 scale; uint32_t packed[BLOCK_SIZE*2/32]; }
```

**Recommendation: Option A (2-bit packed) for weight storage, decode to FP16** during matmul.
- Memory: 0.25 bytes/weight (pure 2-bit packing) + ~0.03 bytes/weight (fp16 per-block scale)
- Decode: 1 `bitfieldExtract` per weight, cheap on RDNA3

### 9.4 Dequantization: In-Shader vs Pre-Computed

| Approach | Memory | Compute | Best For |
|----------|--------|---------|----------|
| In-shader dequant | Low (packed weights only) | Slight decode cost per matmul | Large models, memory-bound workloads |
| Pre-computed FP16 | High (2 bytes/weight) | Zero decode cost | Small models, compute-bound workloads |
| Hybrid (cache block) | Medium (shared memory cache) | Amortized decode | Repeated weight reads |

**Recommendation: In-shader dequantization** — the decode cost of 2-bit → FP16 is minimal (~2 instructions) and the memory savings (8× vs FP16) dominate for LLM inference where memory bandwidth is the bottleneck.

---

## 10. Reference Architecture: Pure Vulkan 1-Bit Engine

### 10.1 Component Stack

```
┌──────────────────────────────────────────┐
│  Model Loader (GGUF / safetensors)       │
│  - Parse model weights                   │
│  - Pack ternary weights (2-bit format)   │
├──────────────────────────────────────────┤
│  Vulkan Runtime                          │
│  - Device enumeration & selection         │
│  - Memory allocator (VMA-style suballoc)  │
│  - Pipeline cache (pre-compiled SPIR-V)   │
│  - Command buffer management              │
├──────────────────────────────────────────┤
│  Compute Pipeline                       │
│  ├── gemv_ternary_fp16 (single token)   │
│  ├── gemm_ternary_fp16 (batch/prompt)   │
│  ├── attention_flash (optional)          │
│  ├── rms_norm / layer_norm              │
│  ├── rope / positional encoding          │
│  ├── silu / gelu activation              │
│  └── residual_add                       │
├──────────────────────────────────────────┤
│  SPIR-V Kernel Library                  │
│  - Compiled at build time                │
│  - Embedded as byte arrays               │
│  - Specialized per quantization type     │
└──────────────────────────────────────────┘
```

### 10.2 Gemv (Matrix-Vector) for Autoregressive Decode

The dominant operation during generation (1 token at a time):

```
Workgroup: 64 threads (subgroup-sized)
Algorithm:
  1. Each thread loads 1 activation (FP16)
  2. Cooperative load of packed ternary weights (CS_WEIGHTS / 64 per thread)
  3. For each weight block:
     a. Decode 16 ternary weights from packed uint32_t
     b. Multiply by corresponding activation, accumulate in FP32
  4. Subgroup shuffle reduction → FP16 output
  5. Apply per-channel scale
```

Expected performance (RDNA3.5, 41 GB/s bandwidth for 20GB model):
- GEMV is strictly memory-bound: 0.25 bytes/weight × model_dim weights per token
- Throughput = bandwidth / bytes_per_token

### 10.3 Gemm (Matrix-Matrix) for Prompt Processing

For batched prefill:

```
Workgroup: 256 threads (4 subgroups × 64)
Algorithm:
  1. Cooperative matrix tile: 16×16×16 INT8
  2. Each subgroup loads 16×16 weight tile + 16×K activation tile
  3. Decode ternary weights → INT8
  4. coopMatMulAdd (16×16×16 cooperative matrix)
  5. Accumulate results in shared memory / FP32 registers
  6. Write output tile
```

### 10.4 Memory Requirements (20GB Model Example)

| Component | FP16 | Ternary (2-bit) | Savings |
|-----------|------|-----------------|---------|
| Weights (forward) | 20 GB | 2.5 GB | 8× |
| KV Cache (4096 ctx) | 2 GB | 2 GB | same |
| Activations | 0.5 GB | 0.5 GB | same |
| **Total** | **22.5 GB** | **5 GB** | **4.5×** |

With 41 GB device memory (Radeon 8060S), a 20GB-equivalent ternary model easily fits entirely in VRAM.

---

## 11. Risks and Gaps

| Risk | Impact | Mitigation |
|------|--------|------------|
| RADV cooperative matrix bugs | Medium | Fall back to subgroup GEMM path |
| No BF16 cooperative matrix | Low | Use FP16 instead |
| VK_KHR_cooperative_matrix adoption | Low | All modern AMD/NVIDIA support it |
| 2-bit decode overhead in inner loop | Low | Pre-compute into shared memory for repeated reads |
| Subgroup size variability (32 vs 64) | Medium | Specialize shaders for both subgroup sizes |
| GGML's `TQ1_0` (ternary) format not Vulkan-ported yet | High | Must implement custom ternary format |
| VK_KHR_cooperative_matrix revisions (spec v2) | Low | RADV already supports rev 2 |

---

## 12. Build Environment

| Component | Status |
|-----------|--------|
| Vulkan SDK | System headers (vulkan_core.h, vulkan.hpp) |
| SPIR-V compiler | glslang-tools 16.2.0, glslc 2026.1 |
| Vulkan loader | libvulkan.so.1.4.341 (RADV) |
| GPU | Radeon 8060S (gfx1151, RDNA3.5) |
| API Version | Vulkan 1.4.335 |
| Driver | RADV 26.0.3 |
| Cooperative Matrix | Rev 2, 14 configs |
| Subgroup Size | 64 (min 32) |
| Device Memory | 41.23 GiB |

---

## 13. Key References

- **zaya-llama.cpp Vulkan backend:** `/home/bcloud/zaya-llama.cpp/ggml/src/ggml-vulkan/`
- **Vulkan shaders:** `/home/bcloud/zaya-llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/`
- **IQ1_S format:** `/home/bcloud/zaya-llama.cpp/ggml/src/ggml-common.h` line 415-419
- **Cooperative matrix configs:** Self-measured (Section 5.1)
- **rlx-vulkan crate:** `https://github.com/MIT-RLX/rlx` (crates.io: rlx-vulkan 0.2.12)
- **kronos-compute crate:** `https://github.com/LynnColeArt/kronos-compute` (crates.io 0.2.3-rc3)

---

*Research compiled from: Vulkan capability queries, source code analysis of zaya-llama.cpp ggml-vulkan, crate registries, and hardware introspection on Radeon 8060S.*

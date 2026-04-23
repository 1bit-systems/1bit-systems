# TQ2_0 HIP port plan — halo ternary GEMV → ggml-hip

**Status:** staged, not merged.
**Task:** #9 (port halo `ternary_gemv_halo.hip` → ggml-hip TQ2_0 dispatch).
**Targets:** both `1bit-tts.cpp/ggml/` and `stable-diffusion.cpp/ggml/` submodules.

## Why this isn't a one-line swap

Halo and ggml disagree on everything except the ternary alphabet.

| Concern           | Halo kernel                                       | ggml TQ2_0                                                |
|-------------------|---------------------------------------------------|-----------------------------------------------------------|
| Alphabet          | `0→-1, 1→0, 2→+1, 3=unused`                       | `q-1` → `{-1,0,+1}` (so `q∈{0,1,2}` same mapping)         |
| Packing layout    | K-contig: byte holds 4 consecutive k values       | Stride-32: byte `[j+m]` holds k = `{m, m+32, m+64, m+96}` |
| Block granularity | Per-row: one fp32 scale covers entire K           | Per-256-element block: fp16 scale per block               |
| Activation path   | Pre-quantised `x_i8` + per-tensor fp32 scale      | `q8_1` blocks (int8 + fp16 scale + fp16 sum)              |
| Dot primitive     | `__builtin_amdgcn_sudot4` on 16-k chunks          | same primitive available but different k-stride           |

**Conclusion:** dispatching TQ2_0 to the existing halo kernel is wrong — it would treat stride-32 bytes as contiguous k and produce garbage. Must write a native TQ2_0 kernel.

## Design — `ternary_gemv_tq2_0_q8_1_halo`

### Signature (extern "C")
```cpp
extern "C" void ternary_gemv_tq2_0_q8_1_launch(
    const void* vx,      // block_tq2_0* — [M rows, K/QK_K blocks per row]
    const void* vy,      // block_q8_1*  — [K/QK8_1 blocks]
    void*       dst,     // float* — [M]
    int32_t     ncols,   // K
    int32_t     nrows,   // M
    void*       stream);
```
Matches the shape of existing `mul_mat_vec_q_cuda` launch in `ggml/src/ggml-cuda/mmvq.cuh`.

### Per-thread work plan

1. **Grid:** `dim3(M / ROWS_PER_BLOCK)`, `block(128)`.
2. **Per-block:** 4 waves × 32 lanes. Each wave handles 2 rows (mirrors halo's `ROWS_PER_BLOCK=8`, `ROWS_PER_WARP=2`).
3. **Per-row iteration:** loop over K/256 TQ2_0 blocks.
4. **Per-block inner loop:**
   - Load 64-byte `qs` + fp16 `d_w` for this weight block.
   - Load corresponding `q8_1` activation blocks (8 × 32-element blocks cover 256 elements).
   - For each 32-byte sub-block `j ∈ {0, 32}`:
     - Lane `l` (0..31) owns byte `qs[j+l]`, which decodes to 4 signed values at k = `{l, l+32, l+64, l+96}` relative to the sub-block start.
     - Fetch matching `x_i8[k]` from the q8_1 blocks (strided).
     - Accumulate `sum_{n=0..3} sign(bits>>(2n) & 3) * x_i8[k_n]` using `sudot4` *after gathering the 4 x values into a uint32*.
   - Scale per-block accumulator by `d_w * d_x_block` (fp16→fp32 on both) and add to row accumulator.
5. **Wave reduction:** `__shfl_xor` butterfly; lane 0 writes `dst[row]`.

### Why not dot4-friendly

Halo gets 4 contiguous k-values per byte ⇒ `xw` fits in one uint32 ⇒ 1 × `sudot4` = 4 MACs. TQ2_0's stride-32 packing means the 4 values from a byte come from k-positions separated by 32 — the x-side fetch is *not* contiguous. Two ways to cope:

- **(A) Gather + dot4:** 4 scalar `x_i8[k_n]` loads, manually pack into uint32, then `sudot4`. Same MAC count, more load pressure.
- **(B) Shuffle the weight side:** unpack 4 lanes of stride-32 bytes into LDS-rearranged contiguous layout per 128-element chunk, then resume halo's dot4. Costs one extra LDS round-trip per block but keeps contiguous x loads.

Pick (A) for first landing — simpler, correctness-first. Benchmark (A) vs halo baseline; if >20% regression, add (B).

### Memory traffic sanity check

256 elements / 64 bytes qs = 2 bpw × 256 = 512 bits = 64 bytes. Plus 2 bytes fp16 scale = 66 B per 256-weight block. Halo: row scale amortised to zero ⇒ 64 B / 256 weights. TQ2_0 overhead = 66/64 ≈ 3% more bandwidth. Acceptable.

At 92% of LPDDR5X peak (225 GB/s × 0.92 ≈ 207 GB/s), 3% overhead = 89.3% of peak. Still well ahead of anything else on gfx1151.

## Files to add / change

### rocm-cpp side (lives in `rocm-cpp/kernels/`, Rule B)

- **NEW** `kernels/ternary_gemv_tq2_0.hip` — the kernel + `extern "C"` launch.
- **EDIT** `kernels/CMakeLists.txt` — add new `.hip` to the library.

### ggml-hip side (must land in BOTH forks' submodules)

- **NEW** `ggml/src/ggml-cuda/ternary-tq2_0.cu` — dispatch glue. Wraps the `extern "C"` launch and exposes:
  ```cpp
  void ggml_cuda_op_mul_mat_vec_tq2_0_q8_1(ggml_backend_cuda_context & ctx,
                                           const ggml_tensor * src0,
                                           const ggml_tensor * src1,
                                           ggml_tensor * dst);
  ```
- **EDIT** `ggml/src/ggml-cuda/mmvq.cuh` — declare the new op.
- **EDIT** `ggml/src/ggml-cuda/mmvq.cu` — in `ggml_cuda_mul_mat_vec_q`, add:
  ```cpp
  case GGML_TYPE_TQ2_0: return ggml_cuda_op_mul_mat_vec_tq2_0_q8_1(...);
  ```
- **EDIT** `ggml/src/ggml-cuda/ggml-cuda.cu` — in `ggml_backend_cuda_device_supports_op`, advertise TQ2_0 × Q8_1 mul_mat.
- **EDIT** `ggml/src/ggml-cuda/CMakeLists.txt` — link against `rocm-cpp` library; add new `.cu`.

### Workspace plumbing

- **EDIT** `rocm-cpp/CMakeLists.txt` — export the kernel as a static lib (`ternary_hip_kernels`) so ggml-cuda can link it without rebuilding.
- **EDIT** `crates/1bit-hip/build.rs` — *no change*; 1bit-hip continues to call the halo-format kernel for our native path. TQ2_0 kernel is consumed only by ggml.

## Test matrix

1. **Unit**: CPU reference dequant vs GPU kernel output for random 256×4096 TQ2_0 tensors. Max-abs diff < 1e-4 (fp32) / 1e-2 (fp16 cast).
2. **End-to-end**: load an existing TriLM-3.9B GGUF after requantising a linear to TQ2_0; run perplexity on wikitext-103; compare vs CPU-only TQ2_0 baseline. Target ≤ +0.01 PPL delta.
3. **Perf**: `llama-bench -p 0 -n 128 -m trilm-3.9b-tq2_0.gguf -ngl 99`. Baseline fp16-on-cpu ~4 tok/s. Target: ≥ 50 tok/s decode on gfx1151.

## Landing order

1. Land rocm-cpp kernel + its self-tests FIRST (no ggml changes yet).
2. Land ggml-hip glue in `1bit-tts.cpp` submodule, validate with a TQ2_0 speech model.
3. Cherry-pick glue commit into `stable-diffusion.cpp` submodule.
4. Once both green, promote into the shared `1bit-ggml` submodule if/when we consolidate.

## Out of scope for this port

- MMQ (batched matmul) path — only MMVQ (batch=1 decode) lands.
- TQ1_0 support. TQ1_0 uses the base-3 pow-trit packing and needs a separate kernel.
- Prefill / compute-bound path — this is the LPDDR5X-bandwidth-bound decode lane.

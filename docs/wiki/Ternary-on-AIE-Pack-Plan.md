# Ternary-on-AIE: packing & MAC plan

Design doc for how `BitNet-b1.58-2B-4T` weights (ternary, values `{-1, 0, +1}`) land on the XDNA 2 AIE2P tile array. It extends the private NPU kernel design notes covering the 8x4 tile grid and INT8 GEMM gotchas.

## Packing on the host side (Rust, offline)

1. Each ternary weight takes 2 bits. We pack 4 weights per byte, row-major. This runs once in the requantizer (`tools/requantize/ternary.rs`, same shape as the existing `.h1b` codec).
2. Encoding: `-1 → 0b10`, `0 → 0b00`, `+1 → 0b01`, `0b11 reserved` (BitNet v2 uses it as a saturation sentinel).
3. Per-layer scale stays bf16, stored alongside the packed block (one scale per row of 4096 for decoder weights; Microsoft's layout).
4. Pre-tile for the AIE 4D DMA reorder pattern (`design.py:368-375` in IRON). A naïve row-major blob costs cycles at shim-DMA time and we can just bake the reorder into the requantizer. One 4D transpose-in-advance, zero run-time cost.

Output of step 1 → a per-layer `weights.bin` consumed by step 2.

## Unpack on the tile (Peano C++)

On AIE2P the int8 MAC pipeline wants int8 inputs for both A and B. We unpack 2-bit ternary to int8 **inside the tile core**, amortising over the MAC latency.

```cpp
// inside aie_kernels/onebit_ternary_mm.cpp
// Lane = 32-wide vector; we run 4 lanes in parallel.
aie::vector<int8, 32> unpack_ternary(uint64_t packed) {
    aie::vector<int8, 32> out;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        uint8_t b = (packed >> (2 * i)) & 0b11;
        out[i] = (b == 0b01) ? 1 : (b == 0b10) ? -1 : 0;  // 0b11 → 0 (sentinel)
    }
    return out;
}
```

Cost: roughly one shift + one predicated-set per lane per pair, pipelined into the MAC issue slot. Theoretical hit: ~0 cycles if unpack and MAC issue on alternating slots (AIE2P VLIW has two VEC slots); worst case: 2-3 cycles added per 32-MAC block. Budget allows.

## MAC core — adapted from `mm.cc:83-208`

Mirror `matmul_vectorized_2x2_mmul` but replace the A-side i8 load with our unpack:

- `a_tile[2][2]` gets `unpack_ternary(load_as_i64(A_raw, ...))` instead of `aie::load_v<>(A_raw, ...)`.
- B-side (activations) stays int8 — prefill activations quantise to int8 upstream (that's the `a8` path; `a4` is next-gen).
- C accumulator stays int32, drained through shim DMA to L3.
- Unit shape r=s=t=8 holds: per-core step does 16×16 int32 out from 16×16 int8 A + 16×16 int8 B. Same tile shape as the upstream INT8 recipe.

Top 3 implementation gotchas (cribbed from the IRON analysis):

1. **4D A-reorder BD** (`design.py:368-375`): bake this into the packing step. The tile expects a specific sub-tile pre-order.
2. **Transpose-on-load** for `c_row_maj=false` (`mm.cc:135-145`): easy to skip; produces bit-exact-wrong C. We want row-major C for compatibility with the router's hidden-state shape.
3. **Alternating shim placement** on 8-col NPU2 (`design.py:385`): `Tile(2*i, 1)`, not `Tile(i, 1)`. Linear indexing double-assigns shim DMAs.

## Bandwidth-vs-compute crossover

At ternary-packed int2, each weight byte yields 4 MAC cycles on-tile. For BitNet-2B hidden=2560 × hidden=2560 × 30-layer prefill:

- Shim-DMA budget: 8 shim DMAs × 2 GB/s each ≈ 16 GB/s weight fetch. 2560 × 2560 / 4 = ~1.6 MB per layer weight block. Per-token fetch: 30 layers × 1.6 MB = 48 MB. At 16 GB/s: 3 ms / token best case.
- Compute budget: 50 TOPS theoretical × 42% demonstrated = 21 effective TOPS = 21 × 10⁹ MACs/s. Per token: 30 × 2 × 2560 × 2560 ≈ 393 M MACs → 19 ms / token compute.

Crossover: compute-bound at all prefill lengths we'll see. Bandwidth is not the ceiling for NPU prefill — compute is. This is why NPU is the correct prefill surface: iGPU is bandwidth-limited on the same problem, NPU is compute-limited, and we care about compute throughput at large M.

## Activation path

Activations flow in as int8 from a stage upstream (CPU-side quantiser or the previous layer's output). Today's 1bit-server has bf16 activations end-to-end; for NPU prefill we add:

- `activation_quantise_int8(bf16 in, i8 out, bf16 scale)` — one pass before tile dispatch.
- `activation_dequantise_bf16(i32 in, bf16 out, bf16 scale)` — one pass after tile drain.

Both run on the iGPU (HIP), not the NPU. Overhead: ~0.3 ms per layer at 2B. Absorbed by overlapping with NPU MAC time.

## Checklist (reimplementer)

1. Requantizer output: packed ternary bytes + per-row bf16 scales + pre-baked 4D reorder.
2. Peano C++ kernel source: unpack fn + MAC loop cribbed from `mm.cc:83-208`, A-path swapped for unpacked int8.
3. Per-tile memory sized to <48 KiB of L1 (see NPU-Kernel-Design.md).
4. Shim DMA bindings: 4 A-lanes (broadcast across rows), 8 B-lanes (broadcast across cols), 8 C-drain.
5. Alternating shim placement on NPU2 (8 cols).
6. Host-side: quantise bf16 activations → int8 + scale on iGPU, drain int32 → dequantise to bf16.
7. `xclbin` produced by Peano, loaded through `libxrt` from the native runtime. The live serving lane remains FastFlowLM; this authoring lane is for custom AIE kernels.
8. Test: bit-exact match against the iGPU HIP reference kernel on a fixed prompt.

## Sources

- Parent NPU kernel design notes are in the private memory archive until restored to this repo.
- IRON `aie_kernels/aie2p/mm.cc:83-208` — INT8 MAC template we adapt.
- IRON `iron/operators/gemm/design.py:368-415` — DMA descriptor patterns.
- Microsoft BitNet b1.58 paper: 2-bit ternary encoding.
- Our rocm-cpp ternary_gemv_halo.hip — HIP reference kernel for parity testing.

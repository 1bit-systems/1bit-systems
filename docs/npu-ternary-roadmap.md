# NPU Ternary Roadmap — XDNA 2 AIE2 Architecture Analysis

## Summary

The bridge is **not a workaround** — it's the correct approach for compute-bound
scenarios. A true AIE2 ternary microkernel is feasible but only wins for
memory-bandwidth-bound models (batch=1 decode). Both paths have their place.

---

## AIE2 Architecture Constraints

### Vector ISA
- Minimum `load_v<N>`: N=16 for `uint8_t` (128 bits)
- `mac_8x8_8x8T`: 64 INT8×BF16 MACs/cycle, expects `bfp16ebs8` format
- No native 2-bit or sub-byte arithmetic
- Has `aie::compare` → mask, `aie::select` (blend), `aie::shuffle` (byte/word permute)
- Has `aie::shuffle_down_bits` for bit-level extraction within elements
- 512-bit vector unit per AIE core, 32 cores on Strix Halo

### Memory Hierarchy
```
DDR (64GB) → L3 (global SRAM) → L2 (tile-group SRAM) → L1 (tile-local SRAM, ~64KB)
  ↓              ↓                       ↓                       ↓
  ~200 GB/s     ~1 TB/s                 ~5 TB/s                ~50 TB/s
```

The INT8 weights sit in DDR and stream through the hierarchy. The bottleneck for
batch=1 decode is **DDR bandwidth** (~200 GB/s shared across 32 cores). TQ2 ternary
cuts weight traffic by 4× (2-bit vs 8-bit), directly relieving the bottleneck.

---

## Phase 1: Bridge ✅ (For compute-bound workloads)

```
DDR (2-bit TQ2) → Host CPU ternary→INT8 → L3 (INT8) → AIE INT8 GEMV
```

- **When to use**: Model fits in AIE tile SRAM with INT8 weights, OR model is
  compute-bound rather than bandwidth-bound
- **Cost**: 8× DDR storage vs true 2-bit (but INT8 is already what AIE needs)
- **Throughput**: Full vector MAC — no overhead
- **Complexity**: Zero AIE changes, pure host-side C++

## Phase 2: True AIE Ternary Microkernel 🔲 (For memory-bandwidth-bound decode)

```
DDR (2-bit TQ2) → L3 → L2 → L1 (2-bit) → Scalar unpack → INT8 buffer → Vector MAC
                    ↓ 4× less traffic            ↑ ping-pong
```

### Design

```cpp
// Two buffers in L1 SRAM for ping-pong:
alignas(32) uint8_t unpack_buf_ping[64 * 128];  // 8KB — 128 cols × 64 int8
alignas(32) uint8_t unpack_buf_pong[64 * 128];  // 8KB

// Each K=64 reduction step:
// 1. DMA-load 16 bytes (128 ternary codes) into L1
// 2. Scalar-unpack (LUT-based):
//    For each byte, extract 4 × 2-bit codes:
//      precomputed LUT[256] = { -1,0,+1,  -1,0,+1, ... }  // 256 entries × 4 values
//      Actually use uint32_t LUT[256] = packed 4 int8 values per byte
//    LUT lookup: val[3:0] = LUT[byte] >> (code_idx * 8) & 0xFF
// 3. Vector MAC: standard mac_8x8_8x8T on unpacked int8 buffer
// 4. Ping-pong: unpack next 128 codes while current batch runs through MAC
```

### LUT-based unpack (the key trick)

The ternary unpack maps each byte (4 × 2-bit codes) to 4 int8 values:
```
byte  = [code3(2b) | code2(2b) | code1(2b) | code0(2b)]
LUT[256] = uint32_t where each byte = int8 value for that code position
         = code==0 → -1, code==1 → 0, code==2 → +1, code==3 → 0
val = (int8_t*)(&LUT[byte])[code_idx]
```

This is **one LUT load + one byte extract** per 4 ternary codes — fast enough
for the scalar unit to keep ahead of the vector MAC.

### Expected throughput

| Scenario | Bridge (INT8) | True AIE Ternary | Winner |
|----------|:------------:|:----------------:|:------:|
| Compute-bound (large batch) | Full MAC throughput | ~15% MAC stall from unpack | Bridge |
| DDR-bound (batch=1, 8B params) | DDR bottleneck at 200 GB/s | 4× less DDR traffic → 4× faster | **Ternary** |
| DDR-bound (batch=1, 27B params) | Doesn't fit (OOM) | Fits (4× less DDR) | **Ternary** only option |

### What ships in which xclbin

| xclbin prefix | Weight source | AIE vector operation | DDR savings |
|---------------|--------------|---------------------|:-----------:|
| `final_i8_*` | INT8 from host | `mac_8x8_8x8T` | 1× |
| `ternary_*` | TQ2 → INT8 via host bridge | `mac_8x8_8x8T` | 4× |
| `native_tq2_*` | TQ2 raw (2-bit on AIE) | scalar LUT unpack + `mac_8x8_8x8T` | 4× |

### Build

Already have the Chess microkernel (`mm_ternary_tq2.cc`) compiling to 6948 bytes.
The LUT-based scalar unpack needs to be added to the `ternary_tq2_gemv` function.
The `n1_core_tq2_placed.py` MLIR design is correct and generates valid MLIR.

Next steps:
1. Add LUT-based scalar unpack to `mm_ternary_tq2.cc` (replace the current INT8-only body)
2. Add ping-pong buffer management
3. Build and test on hardware

---

## Recommendation

Use the **bridge** for all current models (Qwen3-0.6B, Qwen3-8B, etc.) — they are
compute-bound at batch=1 on 32 AIE tiles. The bridge gives full MAC throughput.

Design the **true ternary microkernel** for future large models (Bonsai-27B,
Zaya1-74B, etc.) that are DDR-bandwidth-bound at batch=1. For those, the 4× DDR
savings directly translates to 4× faster inference.

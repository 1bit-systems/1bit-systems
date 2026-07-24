# Zamba2 on NPU — Integration Design

## Architecture: Zamba2-2.7B (mid-size reference)

| Property | Value |
|----------|-------|
| d_model (H) | 2560 |
| d_inner | 5120 |
| d_state | 64 |
| d_conv | 4 |
| n_head (SSM) | 80 |
| n_group | 1 |
| head_dim | 64 |
| Total layers | 54 |
| SSM layers | 45 (indices: 0-5, 7-11, 13-17, 19-23, 25-29, 31-35, 37-41, 43-46, 48-53) |
| Hybrid attn layers | 9 (indices: 6, 12, 18, 24, 30, 36, 42, 47, 51) |
| Attention config | NH=32, NKV=32, attn_HD=80, attn_H=5120 |

## NPU xclbin reuse analysis

### Attention hybrid layers (9/54 = 17%)
These use standard QKV → RoPE → Attn → O → residual → FFN (GU+D) pattern.

- **QKV projection**: H=2560 → qkv_n = NH*HD + 2*NKV*HD = 32*80 + 2*32*80 = 7680
  - Need `final_i8_QKV_qwen3_8b.xclbin` tile configuration? No — Qwen3-8B has H=4096, different tile setup
  - Need a new Zamba2-specific QKV xclbin (or reuse the "v" universal variant)
  
- **O projection**: NH*HD=2560 → H=2560
  - Closest xclbin: `final_i8_O_v.xclbin` (generic dimensions?)
  
- **GU (gate/up)**: H=2560 → IM=8960 (Zamba2 FFN intermediate)
  - Need dimension-specific instructions
  
- **D (down)**: IM=8960 → H=2560
  - Similar dimension mismatch

**Verdict**: The xclbins are tile-size-specific. Zamba2's H=2560 doesn't match Qwen3 (H=1024, 4096), Llama (H=4096), Gemma4 (H=2048), or Phi4 (H=3840). Need new xclbins compiled at Zamba2 tile sizes, OR use the "v" (variable) universal variants which may auto-size.

### SSM layers (45/54 = 83%)
These require: in_proj → conv1d → selective_scan → group_norm → gate → out_proj

- **in_proj**: H=2560 → d_inner + conv_dim + n_head = 5120 + (5120+2*1*64) + 80 = 10384
  - GEMM. Need NPU xclbin at these dimensions
  
- **out_proj**: d_inner=5120 → H=2560
  - GEMM. Need NPU xclbin

- **conv1d**: kernel=4, conv_dim=5248
  - Trivial CPU operation (~10 µs per token)
  
- **selective_scan**: 80 heads × d_state=64
  - ~10K scalar ops per token — CPU cost is O(10 µs)
  - NOT a GEMM — purely sequential recurrence, no GEMM acceleration possible
  - AIE tiles COULD accelerate this (vectorized per-head scan) but need custom xclbin

- **group_norm + gate**: O(d_inner) element-wise ops
  - Trivial CPU

**Bottleneck analysis** (estimated per 2.7B layer on NPU):

| Operation | FLOPS | NPU speed (est.) | CPU speed (est.) |
|-----------|-------|------------------|-------------------|
| in_proj (MNK=1×10384×2560) | 53M MACs | ~3 µs | ~200 µs |
| out_proj (MNK=1×2560×5120) | 26M MACs | ~2 µs | ~100 µs |
| conv1d (4×5248) | 21K ops | N/A | ~5 µs |
| selective_scan (80×64) | 10K ops | N/A | ~10 µs |
| group_norm + gate | 15K ops | N/A | ~5 µs |
| **Total per SSM layer** | | **~5 µs** (NPU) | **~320 µs** (CPU) |

## Implementation strategy

### Phase 1: CPU-only SSM + NPU attention (this PR)
Run the full model on CPU+NPU hybrid:
- SSM layers → CPU via `mamba2_cpu_forward()`
- Attention layers → NPU for QKV+O+GU+D, CPU for RoPE+attn
- Target: ~5-10 tok/s (CPU-bound on SSM layers)

### Phase 2: NPU GEMM for in_proj/out_proj
Add new xclbins for Zamba2 dimensions (H=2560):
- `final_i8_D_zamba2_2_7b.xclbin` — in_proj GEMM
- `final_i8_D_zamba2_2_7b.xclbin` — out_proj (same tile pattern, different dimensions)
- Also need QKV/O/GU/D for attention layers
- Target: ~20-40 tok/s

### Phase 3: Custom AIE kernel for selective_scan
Design an AIE graph for the selective scan recurrence:
- Per-head independent → map 80 heads across 80 AIE tiles
- Each tile: state update (64 MACs) + output (64 MACs) = O(100 cycles/token)
- All 80 heads in parallel: O(100 cycles total)
- Target: ~80+ tok/s (bottleneck shifts to NPU GEMM again)

## Next steps for Phase 1

1. Write `tools/zamba2_npu.cpp` — standalone tool that:
   - Loads Zamba2 1BP model
   - Runs SSM layers via CPU `mamba2_cpu_forward()`
   - Runs attention layers via existing NPU xclbins (or CPU fallback)
   - Reports per-layer latency and overall tok/s

2. Add Zamba2 model config constants for NPU tile sizing

3. Build and benchmark on Strix Halo

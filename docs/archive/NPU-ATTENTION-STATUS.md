# NPU Attention Kernel — Status July 2, 2026

## What was built

| Artifact | Status | Size | Details |
|----------|--------|------|---------|
| `attn_scalar.cc` | ✅ AMD Xilinx IP (Chess) | 4KB | Scalar bf16 Q×K^T, fast_exp softmax, V-weight |
| `attn_c8.mlir` | ✅ IRON-generated | 53 lines | Single core, shim+mem, 2 input ObjectFifos (Q + KV) |
| `attn_c8.xclbin` | ✅ AMD Xilinx IP (Chess) | 14KB | C=8 context, Chess-compiled |
| `insts_attn_c8.txt` | ✅ Generated | 368B | 46 NPU instructions |

## Why it's not integrated

NPU attention as a **separate dispatch** adds 28 × 1334μs = **+37ms per token**.
CPU OpenMP attention at context 40 costs only **7ms total per token**.

```
Separate NPU attn:  28 dispatches × 1334μs = 37ms  ← worse
CPU OpenMP attn:    16 heads parallel on 16 cores = 7ms  ← better
```

NPU attention mathematically only wins if **fused** into the QKV/O xclbin
(one dispatch for QKV+attention, not two). FLM does exactly this with their
`design.py` fused full-layer design.

## GU Combined Program Memory Overflow

Attempted: `128×4096×28672` combined G+U xclbin. Instruction memory per
AIE core is 16KB (4096 words). GU_v (128×1024×6144) uses 901 words (22%).
GU combined at N=28672 would need ~3600 words (88%):

```
Estimated: 901 × (28672 / 6144) × (4096 / 1024) ≈ 3600 words
```

aiecc rejects this during allocation, not because of actual overflow but
because the instruction buffer can't be statically guaranteed to fit.

**Solution already implemented**: Llama xclbins split G and U.
```
G_llama: 377 words  (9%) — separate gate projection
U_llama: 377 words  (9%) — separate up projection
GU_v:    901 words  (22%) — combined gate+up for Qwen3
```

For larger models, use split G+U xclbins (2 dispatches) instead of fused GU.

## Next: Fused Full-Layer XCLBIN

FLM's design shows the path:
- `design.py` generates MLIR for QKV→postprocess→attention→O→norm→GU→D
- All 5 kernels (main16, postprocess_qkv, edge_attention, full_vector_station, swiglu) compiled
- One dispatch per layer instead of 4 (QKV+O+GU+D)
- Plus attention on-chip = zero extra cost

This requires porting FLM's MLIR generator to Qwen3-0.6B dimensions.
Estimated 1-2 weeks of MLIR-AIE engineering.

## What wins today

v12 OpenMP: 97 tok/s (10ms/tok) on Strix Halo — beats FLM Kraken Point.
The NPU attention kernel exists as proof the toolchain works for custom
kernels. Integration blocked by dispatch overhead math, not by feasibility.

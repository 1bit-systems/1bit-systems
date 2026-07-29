# Zyphra Family NPU XCLBIN Generation

## Dimension Specs

### Zaya1-8B (1BP, flagship model)
- Architecture: MoE with CCA attention
- H=2048, L=40, NH=8, NKV=2, HD=128
- IM=2048, N_EXP=16, N_EXP_T=17, RTR_H=256
- Vocab: 262272
- Quant: TQ2 (ternary 2-bit, 1BP format)

### ZR1-1.5B
- Architecture: Dense transformer (Qwen2-based)
- H=1152, L=28, V=50304
- Quant: Q4NX

### Zamba2-2.7B
- Architecture: Mamba2-hybrid (SSM + sparse attention)
- d_model=2560, d_inner=5120, d_state=64
- Quant: Q4NX

## Peano Compilation

To generate xclbins for the NPU, run the Peano pipeline with the dimension
specs above. Each model needs 4 xclbins (QKV, O, GU, D) plus optional attn
xclbin for NPU-resident attention.

```bash
# Example: ZR1-1.5B (dense, same process as Qwen3-0.6B)
# Uses the same Peano pipeline as existing models

# For Zaya1-8B (MoE + ternary), the TQ2 kernel needs to be compiled first:
# 1. Generate ternary NPU instructions via gemm_npu_instructions.cpp
# 2. Compile TQ2 GEMM xclbin via Peano
# 3. Compile CCA attention xclbin via Peano
```

## Status
- ZR1-1.5B: dimension specs match existing xclbin patterns (dense Qwen2 arch)
  → can reuse Qwen3-0.6B xclbins with retargeted dims
- Zaya1-8B: needs TQ2 ternary xclbin compilation (new kernel type)
- Zamba2-2.7B: needs Mamba2-specific xclbins (SSM scan not yet mapped)

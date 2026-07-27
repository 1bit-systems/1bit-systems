# NPU Contract Authoring Guide

## Overview

Each model that runs on the XDNA 2 NPU needs a **contract** — a set of per-model constants
that define tile layout, phase dimensions, packet sizes, and weight-streaming parameters.
The contract feeds into MLIR-AIE compilation through `aiecc` to produce FPGA bitstreams
(`.xclbin`) and instruction sequences (`.bin`).

## Affected Models & Their Dimensions

| Model | Tag | H | NH | NKV | HD | IM | Layers | GU_FUSED | Cloned Kernels |
|-------|-----|---|----|-----|----|----|--------|----------|----------------|
| Qwen3-0.6B | `qwen3_0_6b` | 1024 | 16 | 8 | 128 | 3072 | 28 | 1 | ✅ **Known good** |
| Gemma4-E2B | `gemma4_e2b` | 1536 | 8 | 1 | 256 | 6144 | 35 | 1 | GU, D |
| Qwen3-8B | `qwen3_8b` | 4096 | 32 | 8 | 128 | 12288 | 36 | 0 | G, U |
| Qwen3-VL-4B | `qwen3_vl_4b` | 2560 | 32 | 8 | 128 | 9728 | 36 | 0 | G, U |
| Llama-3.1-8B | `llama` | 4096 | 32 | 8 | 128 | 14336 | 32 | 0 | G, U |

## Contract Constants Derivation

### Base dimensions (from `npu_dims.h`)

```
H  = hidden_dim
NH = num_q_heads
NKV = num_kv_heads
HD = head_dim
IM = intermediate_dim (ffn)
NC = num_layers
```

### Phase definitions (7 phases)

```
Q:     [H, NH*HD]           — input=H, output=NH*HD
K:     [H, NKV*HD]           — input=H, output=NKV*HD
V:     [H, NKV*HD]           — input=H, output=NKV*HD
O:     [NH*HD, H]            — input=NH*HD, output=H
UP:    [H, IM]               — input=H, output=IM
GATE:  [H, IM]               — input=H, output=IM
DOWN:  [IM, H]               — input=IM, output=H
```

When `GU_FUSED=1`: 2*IM <= 14336 (AIE tile limit), G and U share one kernel "GU".
When `GU_FUSED=0`: 2*IM > 14336, G and U need separate kernels.

### Derived quantities

```
PHASE_BLOCKS  = output_dim ÷ 512   (OUTPUT_BLOCK_ROWS)
PHASE_CHUNKS  = input_dim ÷ 256    (K_CHUNK)
TOTAL_PATCHES = sum(blocks × 4 columns × 2 patches_per_column)
```

### Packet sizes

```
C1R2_PACKET_DWORDS     = 1 + H//2
ATTENTION_PACKET_DWORDS = (NH*HD)//2
DOWN_PACKET_DWORDS     = IM//2
COMPACT_PACKET_DWORDS  = 1 + 4×4×16 = 257  (fixed: 1 + cols×rows×(RECORD_DWORDS-1))
O_CHUNKS               = (NH*HD) ÷ 256
SWIGLU_SLICES          = IM ÷ 512
AUX_DWORDS             = (2*H + (NH+NKV)*HD) // 2
HIDDEN_DWORDS          = H//2
OUTPUT_DWORDS          = H//2
```

## Contract Values Per Model

### Qwen3-0.6B (reference — known working)

```python
HIDDEN_DIM = 1024
INTERMEDIATE_DIM = 3072
HEAD_DIM = 128
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
GU_FUSED = 1  # 2*IM = 6144 <= 14336

PHASE_BLOCKS = (4, 2, 2, 2, 6, 6, 2)  # Q=4, K=2, V=2, O=2, UP=6, GATE=6, DOWN=2
PHASE_CHUNKS = (4, 4, 4, 8, 4, 4, 12) # Q=4, K=4, V=4, O=8, UP=4, GATE=4, DOWN=12
TOTAL_PATCHES = 192                     # sum of blocks × 4 × 2
C1R2_PACKET_DWORDS = 513                # 1 + 1024/2
ATTENTION_PACKET_DWORDS = 1024          # (16*128)/2
DOWN_PACKET_DWORDS = 1536               # 3072/2
O_CHUNKS = 8                            # 2048/256
SWIGLU_SLICES = 6                       # 3072/512
AUX_DWORDS = 2560                       # (2*1024 + (16+8)*128)/2
```

### Gemma4-E2B

```python
HIDDEN_DIM = 1536
INTERMEDIATE_DIM = 6144
HEAD_DIM = 256
NUM_Q_HEADS = 8
NUM_KV_HEADS = 1
GU_FUSED = 1  # 2*IM = 12288 <= 14336

# Phase I/O:
# Q: [1536, 8*256=2048]
# K: [1536, 1*256=256]
# V: [1536, 1*256=256]
# O: [2048, 1536]
# UP: [1536, 6144]
# GATE: [1536, 6144]
# DOWN: [6144, 1536]

PHASE_BLOCKS = (2048//512, 256//512, 256//512, 1536//512, 6144//512, 6144//512, 1536//512)
             = (4, 1, 1, 3, 12, 12, 3)

PHASE_CHUNKS = (1536//256, 1536//256, 1536//256, 2048//256, 1536//256, 1536//256, 6144//256)
             = (6, 6, 6, 8, 6, 6, 24)

TOTAL_PATCHES = (4+1+1+3+12+12+3) * 4 * 2 = 288

C1R2_PACKET_DWORDS = 1 + 1536//2 = 769
ATTENTION_PACKET_DWORDS = 2048//2 = 1024
DOWN_PACKET_DWORDS = 6144//2 = 3072
O_CHUNKS = 2048//256 = 8
SWIGLU_SLICES = 6144//512 = 12
AUX_DWORDS = (2*1536 + (8+1)*256)//2 = (3072 + 2304)//2 = 2688  # Note: HD=256 changes QK RoPE calc
```

**Key difference**: Gemma4 has `HD=256` (not 128), `NKV=1`, and `NH=8`. The QK RoPE
side-load calculation needs attention — with `HD=256`, the norm/RoPE constants are:
`(NUM_Q_HEADS + NUM_KV_HEADS) * HD = 9 * 256 = 2304` bf16 = 1152 i32.

### Qwen3-8B

```python
HIDDEN_DIM = 4096
INTERMEDIATE_DIM = 12288
HEAD_DIM = 128
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
GU_FUSED = 0  # 2*IM = 24576 > 14336 — separate G and U kernels

# Phase I/O:
# Q: [4096, 32*128=4096]
# K: [4096, 8*128=1024]
# V: [4096, 8*128=1024]
# O: [4096, 4096]
# UP: [4096, 12288]
# GATE: [4096, 12288]
# DOWN: [12288, 4096]

PHASE_BLOCKS = (4096//512, 1024//512, 1024//512, 4096//512, 12288//512, 12288//512, 4096//512)
             = (8, 2, 2, 8, 24, 24, 8)

PHASE_CHUNKS = (4096//256, 4096//256, 4096//256, 4096//256, 4096//256, 4096//256, 12288//256)
             = (16, 16, 16, 16, 16, 16, 48)

TOTAL_PATCHES = (8+2+2+8+24+24+8) * 4 * 2 = 608

C1R2_PACKET_DWORDS = 1 + 4096//2 = 2049
ATTENTION_PACKET_DWORDS = 4096//2 = 2048
DOWN_PACKET_DWORDS = 12288//2 = 6144
O_CHUNKS = 4096//256 = 16
SWIGLU_SLICES = 12288//512 = 24
AUX_DWORDS = (2*4096 + (32+8)*128)//2 = (8192 + 5120)//2 = 6656
```

**Key difference**: `GU_FUSED=0` means G and U need separate xclbins. The G kernel
is `(M, H, IM)` = `(M, 4096, 12288)` and the U kernel has the same shape but
different weights.

### Qwen3-VL-4B

```python
HIDDEN_DIM = 2560
INTERMEDIATE_DIM = 9728
HEAD_DIM = 128
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
GU_FUSED = 0  # 2*IM = 19456 > 14336

# Phase I/O:
# Q: [2560, 32*128=4096]
# K: [2560, 8*128=1024]
# V: [2560, 8*128=1024]
# O: [4096, 2560]
# UP: [2560, 9728]
# GATE: [2560, 9728]
# DOWN: [9728, 2560]

PHASE_BLOCKS = (4096//512, 1024//512, 1024//512, 2560//512, 9728//512, 9728//512, 2560//512)
             = (8, 2, 2, 5, 19, 19, 5)  # 2560/512 = 5 (not integer! floors to 5)

PHASE_CHUNKS = (2560//256, 2560//256, 2560//256, 4096//256, 2560//256, 2560//256, 9728//256)
             = (10, 10, 10, 16, 10, 10, 38)

TOTAL_PATCHES = (8+2+2+5+19+19+5) * 4 * 2 = 480

C1R2_PACKET_DWORDS = 1 + 2560//2 = 1281
ATTENTION_PACKET_DWORDS = 4096//2 = 2048
DOWN_PACKET_DWORDS = 9728//2 = 4864
O_CHUNKS = 4096//256 = 16
SWIGLU_SLICES = 9728//512 = 19
AUX_DWORDS = (2*2560 + (32+8)*128)//2 = (5120 + 5120)//2 = 5120
```

**Key difference**: Qwen3-VL-4B has `H=2560` which is not evenly divisible by 512
for the O phase output (2560/512 = 5 exactly, so it does divide evenly, but
other derived values need checking).

### Llama-3.1-8B

```python
HIDDEN_DIM = 4096
INTERMEDIATE_DIM = 14336
HEAD_DIM = 128
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
GU_FUSED = 0  # 2*IM = 28672 > 14336

# Phase I/O:
# Q: [4096, 32*128=4096]
# K: [4096, 8*128=1024]
# V: [4096, 8*128=1024]
# O: [4096, 4096]
# UP: [4096, 14336]
# GATE: [4096, 14336]
# DOWN: [14336, 4096]

PHASE_BLOCKS = (4096//512, 1024//512, 1024//512, 4096//512, 14336//512, 14336//512, 4096//512)
             = (8, 2, 2, 8, 28, 28, 8)

PHASE_CHUNKS = (4096//256, 4096//256, 4096//256, 4096//256, 4096//256, 4096//256, 14336//256)
             = (16, 16, 16, 16, 16, 16, 56)

TOTAL_PATCHES = (8+2+2+8+28+28+8) * 4 * 2 = 672

C1R2_PACKET_DWORDS = 1 + 4096//2 = 2049
ATTENTION_PACKET_DWORDS = 4096//2 = 2048
DOWN_PACKET_DWORDS = 14336//2 = 7168
O_CHUNKS = 4096//256 = 16
SWIGLU_SLICES = 14336//512 = 28
AUX_DWORDS = (2*4096 + (32+8)*128)//2 = (8192 + 5120)//2 = 6656
ROPE_THETA = 500000.0  # Different from Qwen3's 1000000.0
```

## Build Pipeline

Each xclbin goes through:

1. **Chess kernel compilation**: `xchesscc_wrapper aie2p` compiles `.cc` → `.o`
2. **MLIR generation**: Python script generates `.mlir` with per-model constants
3. **aiecc compilation**: `aiecc` compiles `.mlir` + `.o` → `.xclbin` + `.bin`

For the full-layer design (which produces the G/U/GU/D Q4NX GEMM kernels), the
pipeline follows the pattern in `~/torch2aie/examples/qwen3-decode-layer/`:

```
build_attn_06b.sh  # Example: builds edge attention for 0.6B
  → compile kernel .cc → .o
  → generate MLIR with model constants
  → aiecc: .mlir + .o → .xclbin + .bin
  → output: engine/npu/xclbins/final_i8_{KERNEL}_{TAG}.xclbin
```

## Validation

Each new contract must be validated:

1. **Static validation**: `contract.validate_contract()` checks all derived values
2. **Chess kernel test**: each `.o` passes its oracle gate
3. **Full-layer NPU test**: `npu_engine_universal` runs the model and output
   is compared against CPU reference
4. **End-to-end**: `unified_server` serves the model via OpenAI API

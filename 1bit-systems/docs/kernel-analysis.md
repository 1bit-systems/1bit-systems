# Kernel Analysis — FLM xclbin Architecture & Instruction Format

## 1. FLM's xclbin Kernels

FLM ships 4 xclbins per model under `/opt/fastflowlm/share/flm/xclbins/<model>/`:

| xclbin | Purpose | Opcode (in ELF) | Size (Qwen3-0.6B) |
|--------|---------|-----------------|-------------------|
| `QKV.mtx.xclbin` | Q/K/V projection | 0 | 183 KB |
| `O.mtx.xclbin` | Output projection | 1 | 160 KB |
| `GU.mtx.xclbin` | Gate+Up projection | 2 | 187 KB |
| `D.mtx.xclbin` | Down projection | 3 | 148 KB |
| `attn.xclbin` | Fused attention | 6 | 312 KB |

Each xclbin contains a single `MLIR_AIE` kernel compiled for AIE2 (XDNA 2).

## 2. Kernel ABI

All 4 GEMM xclbins use ABI v2 (opcode-based dispatch):

| Arg | Name | Description |
|-----|------|-------------|
| 0 | opcode | 0-3 maps to QKV/O/GU/D respectively. 3 = generic GEMM |
| 1 | A | Activation BO (tokens × hidden) in fp32 |
| 2 | B | Weight BO (hidden × hidden) in packed int4 |
| 3 | C | Output BO (tokens × hidden) in fp32 |
| 4 | M | Number of rows (tokens) |
| 5 | N | Number of columns (hidden) |
| 6 | K | Inner dimension (hidden / expert dim) |

The kernel reads A and B, computes C = A @ B, and blocks until complete.

## 3. Instruction Format

FLM's instruction files (`insts_i8_*.txt`) describe the NPU control code sequences
that set up DMA buffers, configure compute tiles, and synchronize execution.

Format key-value pairs:
```
M = 128
N = 128
K = 128
...
```

See `include/npu_instr.hpp` for the serialization format.

## 4. Dynamic Instruction Generation (FLM-style)

FLM generates attention instructions at runtime based on sequence length.
The format uses `xrt::ext::kernel` with ELF generation via aiebu assembler.

The open-source approach (this repo):
1. Parse base instructions from a reference `.insts` file
2. Patch BO addresses via DDR_PATCH commands
3. Issue DMA tokens and wait for completion
4. Compile to instruction stream → ELF → xrt::ext::kernel

See `engine/npu/README-dynamic.md` for the detailed implementation guide.

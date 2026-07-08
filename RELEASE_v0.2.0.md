# v0.2.0 — Native Ternary NPU: Hardware Verified

**Strix Halo NPU (RyzenAI-npu5) — 32 cores, bit-exact BF16**

## 🎉 Breakthrough

For the first time, a **native ternary (2-bit packed) kernel runs on AMD NPU hardware** with bit-exact BF16 precision. The 32-core xclbin processes 128 rows × 256 ternary values in 118.9 µs on a single NPU call.

## ✅ Verified

| Test | Result |
|------|--------|
| Single-core all-ones | -256.0000 exactly (32/32) |
| **32-core all-ones** | **-256.0000 exactly (128/128)** |
| BF16 precision | Bit-exact vs CPU reference |

## 📦 What's Included

- **mm_ternary_32x64x128** Chess C++ kernel with row_start/num_rows tiling
- **32-core MLIR generator** — 4×8 grid, per-column DMA, object_fifo dataflow
- **Single-core MLIR generator** — for debugging and verification
- **Q2_0 bit-exact decoder** — cos=1.000000 vs F16 reference
- **Q2_0→Q4NX converter** — INT8 passthrough for existing engine
- **npu_ternary_target** — spec-decode TargetModelInterface integration
- **NPU unlock tool** — UEFI NVRAM patching for Strix Halo

## 🏎️ Models Benchmarked

| Model | Size | Format | Tensors | Hidden | Layers |
|-------|------|--------|---------|--------|--------|
| **Bonsai-1.7B-Q1_0** 🔵 | 250 MB | Q1_0 (1-bit) | 310 | 2048 | 28 |
| ZAYA1-8B-zaya | 5.6 GB | Q4_K/F32 | 1041 | 2048 | 40 |
| ZAYA1-8B-Q4_K_M | 5.6 GB | Q4_K | 921 | 2048 | 40 |

## 🔧 NPU Performance

| Configuration | Latency | Throughput | XCLBin |
|--------------|---------|------------|--------|
| 1 core | 68.3 µs | 14,636/s | 16 KB |
| **32 cores** | **118.9 µs** | **8,410/s** | **314 KB** |

## 🧠 GPU Kernels

13 HIP kernel sources across 5 packing formats (Q1_0, Q2_0, TQ1, Sherry, ZAYA MoE) + Vulkan ternary GEMM shader — all targeting RDNA 3.5 (gfx1151).

## 📐 Ternary Packing Formats

| Format | bpw | Use |
|--------|-----|-----|
| Q1_0 | 1.0 | Binary ±1 (Bonsai) |
| Q2_0 | 2.0 | {-1,0,+1} (PrismML) |
| NPU packed | 2.0 | 4×2bit/byte, on-the-fly decode |
| TQ1 halo | 1.6 | Base-3, 5 values/byte |
| Sherry | 1.25 | 3:4 sparsity (training) |

---

**Built on Strix Halo XDNA2. AMD Ryzen AI Max+ 395.**

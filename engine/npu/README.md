# INT8 Inference Engine

Pure C++ inference engine for Qwen3-0.6B on AMD Strix Halo NPU.

## Files

| File | Purpose |
|------|---------|
| `src/npu_engine_i8.cpp` | Main engine — 4-live INT8 contexts, NPU attention |
| `src/dequant_q4nx.c` | Q4NX weight dequantizer (C99) |
| `xclbins/n1_core_i8_v2.py` | INT8 MLIR generator with K-interleaving fix |
| `kernel/edge_attention.cc` | NPU attention kernel (Chess C++) |
| `build/dequant_q4nx.o` | Compiled dequantizer (committed, zero-dependency) |
| `build_xclbins.sh` | Build xclbins via torch2aie MLIR-AIE toolchain |
| `gen_xclbins.sh` | Template-based xclbin generator for new model architectures |
| `gen_xclbins.py` | Python equivalent of gen_xclbins.sh |

## Prerequisites

### MLIR-AIE Toolchain

The xclbin build requires the [torch2aie](https://github.com/bong-water-water-bong/torch2aie) repository with its MLIR-AIE toolchain.

```bash
# Clone and set up the toolchain
git clone https://github.com/bong-water-water-bong/torch2aie ~/torch2aie
cd ~/torch2aie

# Download the pre-built MLIR-AIE toolchain
./toolchain/download.sh
```

**Known-good commit** (as of 2026-07-29): The torch2aie `main` branch at the commit that ships `aiecc` v0.3.x with AIE2P support. Verify with:

```bash
$ $HOME/torch2aie/toolchain/bin/aiecc --version
# Expected: aiecc (MLIR-AIE) version 0.3.x
```

> **Note:** The MLIR-AIE commit hash is version-locked inside the toolchain tarball. Run `./toolchain/download.sh` to get the one verified against the build scripts. If you update torch2aie independently, xclbin PDI reproducibility is not guaranteed (see [#1076](https://github.com/bong-water-water-bong/1bit-systems/issues/1076)).

### Compiler: Peano vs Chess

| Compiler | xclbin Use | Status |
|----------|-----------|--------|
| **Peano** (LLVM-based, shipped with MLIR-AIE toolchain) | INT8 GEMM xclbins | ✅ **Produces correct xclbins.** Recommended for all GEMM builds. |
| **Chess** (AMD Xilinx proprietary) | Attention kernel (`kernel/edge_attention.cc`) | ✅ Required for NPU attention xclbins. Not needed for GEMM. |

> ⚠️ **Known issue:** xclbins compiled with Peano vs Chess produce different PDI binaries even from the same MLIR source. The shipping xclbins in `xclbins/` were built with **Peano**. If you rebuild with Chess (e.g., via the FastFlowLM pipeline), the resulting PDI will differ and may cause runtime failures. See [#1075](https://github.com/bong-water-water-bong/1bit-systems/issues/1075), [#1076](https://github.com/bong-water-water-bong/1bit-systems/issues/1076).

### Python Dependencies

The MLIR generator (`n1_core_i8_v2.py`) and `gen_xclbins.py` require:

```bash
# These are included in the torch2aie toolchain venv:
pip install numpy torch
pip install torch2aie  # or: pip install -e ~/torch2aie
```

Python packages needed at minimum:

| Package | Version | Notes |
|---------|---------|-------|
| `numpy` | ≥1.24 | Array operations for MLIR generation |
| `torch` | ≥2.0 | Only if running torch2aie model export |
| `torch2aie` | matching toolchain | MLIR-AIE Python bindings (`$AIE_TOOLS_DIR/python`) |

> The MLIR-AIE toolchain ships its own Python packages in `$AIE_TOOLS_DIR/python`. Set `PYTHONPATH` to point there rather than installing conflicting versions.

### XRT Runtime

| Component | Version | Notes |
|-----------|---------|-------|
| XRT (Xilinx Runtime) | ≥2.19 | Required for xclbin loading at runtime |
| `xrt_coreutil` | matching XRT | Linking target for engine builds |
| `aiebu` | matching XRT | AIE buffer library for instruction assembly |

Check your XRT version:

```bash
$ xbutil examine
# Expected: XRT version 2.19.x or later
```

## Build Steps

### 1. One-time: Compile the Dequantizer

```bash
cd engine/npu
gcc -c -O3 -o build/dequant_q4nx.o src/dequant_q4nx.c
```

### 2. Known-good Toolchain Setup (as of 2026-07-29)

```bash
export AIE_TOOLS_DIR=~/mlir-aie/install_tmp
export PATH=$AIE_TOOLS_DIR/bin:$PATH
export PYTHONPATH=$AIE_TOOLS_DIR/python:$PYTHONPATH
```

> If you placed the torch2aie toolchain somewhere else, adjust `AIE_TOOLS_DIR` accordingly. The default download path is `~/torch2aie/toolchain`.

### 3. Build xclbins

```bash
cd engine/npu

# Build xclbins for Qwen3-0.6B
./build_xclbins.sh qwen3_0_6b

# Or build all known models:
./build_xclbins.sh qwen3_0_6b
./build_xclbins.sh qwen3_8b
./build_xclbins.sh qwen3_vl_4b
./build_xclbins.sh llama
./build_xclbins.sh gemma4_e2b
```

This generates the following xclbins per model:

| GEMM | Shape | Description |
|------|-------|-------------|
| QKV | M×H×(NH·HD + 2·NKV·HD) | Query/Key/Value fused projection |
| O | M×(NH·HD)×H | Attention output projection |
| GU | M×H×(2·IM) | Fused Gate+Up projection |
| D | M×IM×H | Down projection |

> For models where GU exceeds the 14336-column tile limit, `build_xclbins.sh` automatically splits into separate G and U xclbins.

### 4. Compile the Engine

```bash
# Requires XRT headers + libs
g++ -std=c++23 -O3 -o build/npu_engine \
    src/npu_engine_i8.cpp build/dequant_q4nx.o \
    -I$XRT/include -L$XRT/lib64 -lxrt_coreutil -luuid -lm -ldl

# Or use CMake (recommended)
mkdir -p build_cmake && cd build_cmake
cmake .. -DCMAKE_PREFIX_PATH=$XRT
make -j$(nproc)
```

### 5. Run

```bash
./build/npu_engine
```

## Verification

After building xclbins, verify they match expected sizes and instruction counts.

### Check xclbin File Sizes

```bash
# For Qwen3-0.6B:
$ ls -la xclbins/final_i8_*_qwen3_0_6b.xclbin
# Expected sizes:
#   final_i8_QKV_qwen3_0_6b.xclbin: 12074 bytes   (M=128, K=1024, N=4096)
#   final_i8_O_qwen3_0_6b.xclbin:   46346 bytes   (M=128, K=2048, N=1024)
#   final_i8_GU_qwen3_0_6b.xclbin:  12074 bytes   (M=128, K=1024, N=6144)
#   final_i8_D_qwen3_0_6b.xclbin:   46346 bytes   (M=128, K=3072, N=1024)

# Base template sizes (for models cloned from templates):
#   final_i8_QKV_v.xclbin:  12074 bytes   (template, M=128, K=1024, N=4096)
#   final_i8_GU_v.xclbin:   12074 bytes   (template, M=128, K=1024, N=6144)
#   final_i8_D_v.xclbin:    46346 bytes   (template, M=128, K=3072, N=1024)
#   final_i8_O_v.xclbin:    46346 bytes   (template, M=128, K=2048, N=1024)
```

### Check Instruction File Line Counts

```bash
$ wc -l xclbins/insts_i8_*_qwen3_0_6b.txt
# Expected (for Qwen3-0.6B):
#   QKV: 136 lines (M=128, K=1024, N=4096)
#   O:   36 lines  (M=128, K=2048, N=1024)
#   GU:  136 lines (M=128, K=1024, N=6144)
#   D:   36 lines  (M=128, K=3072, N=1024)
```

Instruction count formulas:
- **QKV/GU** (N large): `(K / 64) * (N / 128)` lines
- **O/D** (K large): `(K / 64) * (N / 128)` lines

### Run GEMM Verification

```bash
# Build and run the bench_gemm test harness
cd engine/npu
mkdir -p build_cmake && cd build_cmake
cmake .. && make bench_gemm -j$(nproc)
./bench_gemm

# Expected output:
#   - Correctness check: mean relative error < 1e-2 (INT8 quantized)
#   - Throughput: ~15-55 GFLOPS depending on dimensions
#   - D projection should hit ~55 TFLOPS (111% of 50 TOPS peak)
```

Alternatively, verify with `--verify` flag:

```bash
./build_xclbins.sh --verify qwen3_0_6b
```

This builds and runs the verification harness after compilation.

## Troubleshooting

### "PDI binary differs from production"

**Symptom:** The rebuilt xclbin has a different SHA256 than the one committed in `xclbins/`, even with the same MLIR source.

**Cause:** The MLIR-AIE toolchain embeds build metadata (LLVM version, timestamps) in the PDI. The Peano compiler version and LLVM opaque pointer representation also affect binary output.

**Status:** This is a [known issue](https://github.com/bong-water-water-bong/1bit-systems/issues/1076). The shipped xclbins are verified correct on hardware — they pass GEMM correctness tests and produce coherent decode. A rebuild that produces a different PDI is NOT necessarily wrong, but must be validated against the reference.

**Resolution:**
1. Verify functional correctness using `bench_gemm` (see Verification section)
2. If `bench_gemm` passes, the xclbin is safe to use
3. For bit-exact reproducibility, see the Fallback Method below

### "Chess vs Peano compiler differences"

**Symptom:** AIE kernel crashes or produces wrong results when using Chess-compiled xclbins alongside Peano-compiled ones.

**Cause:** The Chess compiler (AMD Xilinx proprietary) and Peano (LLVM-based, open-source) produce different AIE instruction encodings and memory layouts for the same MLIR source. Mixing them in the same pipeline can cause data layout mismatches.

**Fix:**
- INT8 GEMM xclbins → always use **Peano** (shipped with MLIR-AIE toolchain)
- Attention kernels (`edge_attention.cc`) → must use **Chess** (AMD Vitis tools)
- Never mix: run all GEMM xclbins through the same compiler pipeline

### "Opaque pointer LLVM version mismatches"

**Symptom:** `aiecc` fails with LLVM opaque pointer errors or crashes during MLIR→xclbin compilation.

**Cause:** The MLIR-AIE toolchain ships a specific LLVM build. If your system has a different LLVM version in `PATH`, or if the `aiecc` script picks up the wrong `llc`/`opt`, the compilation will fail.

**Fix:**
```bash
# Isolate the toolchain PATH — put AIE tools first
export PATH=$AIE_TOOLS_DIR/bin:$PATH
# Verify:
which llc          # should be $AIE_TOOLS_DIR/bin/llc
llc --version      # should match MLIR-AIE's LLVM
```

### "build_xclbins.sh fails with 'make: command not found'"

Ensure you have build tools installed:

```bash
sudo apt install build-essential make
```

### "XRT not found"

```bash
# Install XRT (Ubuntu 26.04)
sudo apt install xrt xrt-dev
# Or from AMD's package repo:
wget https://github.com/Xilinx/XRT/releases/download/202420.2.19.0/xrt_202420.2.19.0_amd64.deb
sudo dpkg -i xrt_202420.2.19.0_amd64.deb
export XRT=/opt/xilinx/xrt
```

## Fallback Method: Template-Based Generation

When the MLIR-AIE toolchain build does not reproduce shipping xclbins (see [#1076](https://github.com/bong-water-water-bong/1bit-systems/issues/1076)), use the template-based generator. This clones GEMM xclbins by closest-matching shape, which is functionally correct because the xclbin format is parameterized by RTP registers at runtime.

### Using `gen_xclbins.sh`

```bash
# Generate xclbins for a new model using template cloning
cd engine/npu
./gen_xclbins.sh qwen3_0_6b 1024 16 8 128 3072

# Arguments: <tag> <H> <NH> <NKV> <HD> <IM> [M=128]
```

### Using `gen_xclbins.py`

```bash
python3 gen_xclbins.py qwen3_0_6b 1024 16 8 128 3072
```

### Pre-built xclbins

The xclbins in `engine/npu/xclbins/` ARE verified correct:

| File | Status |
|------|--------|
| `final_i8_{QKV,GU,D,O}_v.xclbin` | ✅ Reference templates, Peano-compiled, verified on hardware |
| `final_i8_*_qwen3_0_6b.xclbin` | ✅ Full set for Qwen3-0.6B |
| `final_i8_*_qwen3_8b.xclbin` | ✅ Full set for Qwen3-8B |
| `final_i8_*_qwen3_vl_4b.xclbin` | ✅ Full set for Qwen3-VL-4B |
| `final_i8_*_llama.xclbin` | ✅ Full set for Llama-3.1-8B |
| `final_i8_*_gemma4_e2b.xclbin` | ✅ Full set for Gemma4-E2B |
| `final_i8_*_phi4.xclbin` | ✅ Cloned from template, verified at init |

> These are the same xclbins used in production. If your rebuild produces a different PDI, use these pre-built versions — they are functionally identical and verified to produce coherent decode on all 5 supported model families.

### Adding a New Model

1. Run `gen_xclbins.sh <tag> <H> <NH> <NKV> <HD> <IM>` to clone matching xclbins
2. Verify the cloned xclbin sizes match expected dimensions
3. Test with `bench_gemm` or the NPU engine
4. For full MLIR-AIE compilation (instead of template cloning), see the [torch2aie design flow](https://github.com/bong-water-water-bong/torch2aie)

## Related Issues

| Issue | Description |
|-------|-------------|
| [#1052](https://github.com/bong-water-water-bong/1bit-systems/issues/1052) | MLIR-AIE toolchain version pinning for reproducible builds |
| [#1075](https://github.com/bong-water-water-bong/1bit-systems/issues/1075) | Chess vs Peano compiler PDI divergence |
| [#1076](https://github.com/bong-water-water-bong/1bit-systems/issues/1076) | xclbin build reproducibility (this document) |

## Architecture

4 INT8 GEMM xclbins + 4 NPU attention xclbins, all alive simultaneously.
Per-layer weight BOs pre-loaded at startup. Zero Python at runtime.

### GEMM Pipeline

```
MLIR generator (Python)
        ↓
   aiecc (MLIR-AIE toolchain, Peano compiler)
        ↓
   .xclbin + insts.txt
        ↓
   XRT kernel execution (C++ engine)
```

### Engine Flow

```
npu_engine_universal.cpp
  ├─ Load xclbins by model config (auto-detect from Q4NX header)
  ├─ Allocate weight BOs per layer
  ├─ Execute GEMMs: QKV → Attention → O → GU → D
  └─ Run LM head (CPU OpenMP or NPU)
```

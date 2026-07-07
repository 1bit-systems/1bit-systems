# Building 1bit.systems NPU Engine on Windows

This guide documents how to build the NPU inference engine for Windows (AMD Strix Halo XDNA 2 NPU).

> **Historical**: the `engine/npu/` C++ source this guide builds was retired
> from this repo (commit `cd232a091`) — superseded by the `spec-decode/`
> stack, which does not yet have an equivalent Windows port. Kept as a
> reference for the XRT/toolchain setup steps, which still apply to any future
> Windows port.

## Prerequisites

### 1. Hardware
- AMD Strix Halo APU (Ryzen AI Max 300 series) with XDNA 2 NPU
- Windows 11 24H2 or later

### 2. AMD NPU Driver

Download and install the Windows NPU driver from the [AMD Ryzen AI software page](https://www.amd.com/en/products/software/ryzen-ai.html).

The driver package (`NPU_RAI_376_WHQL.zip`) contains:
- **`ipustack.sys`** — NPU kernel driver
- **`xrt_core.dll`** / **`xrt_coreutil.dll`** — XRT user-space DLLs (same API as Linux)
- **`xrt-smi.exe`** — XRT System Management Interface
- **`RadeonML.dll`** / **`RadeonML_IPU.dll`** — High-level NPU inference API
- **`xclbinutil.exe`** — XCLBIN utility
- C headers under `rml/include/rml/`

After installation, verify with:
```powershell
xrt-smi.exe examine
```

Expected output shows NPU device detected with XDNA 2.

### 3. Visual Studio 2022

Install Visual Studio 2022 with:
- "Desktop development with C++" workload
- MSVC v143 build tools
- Windows 11 SDK (10.0.26100.0 or later)

### 4. CMake

Install CMake 3.20+:
```powershell
winget install Kitware.CMake
```

### 5. XRT Development Headers

The XRT C++ headers (`xrt/xrt_device.h`, `xrt/xrt_bo.h`, `xrt/xrt_kernel.h`) ship with the AMD Ryzen AI LT (Light Toolkit) package.

Install location: typically `C:\Program Files\AMD\XRT\include\`
Import library: `C:\Program Files\AMD\XRT\lib\xrt_coreutil.lib`

### 6. NPU xclbins

Pre-compiled INT8 xclbins for the NPU GEMM kernels. These are the same `.xclbin` files used on Linux:
- `final_i8_QKV_v.xclbin` — Fused QKV projection
- `final_i8_O_v.xclbin` — Attention output projection
- `final_i8_GU_v.xclbin` — Gate + Up projection
- `final_i8_D_v.xclbin` — Down projection

Place these in a known path (e.g., `C:\npu\xclbins\`).

### 7. Model File

Q4NX format model file (e.g., `model.q4nx` for Qwen3-0.6B). Same format as Linux.

## Building

### Using CMake (Recommended)

```powershell
# Configure
cmake -B build -S engine\npu ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DXRT_DIR="C:\Program Files\AMD\XRT"

# Build all model variants
cmake --build build --config Release

# Build a specific variant
cmake --build build --config Release --target npu_engine_qwen3_0_6b
```

### Manual MSVC Build

```powershell
cl /std:c++23 /O2 /arch:AVX2 /openmp ^
    /I engine\npu\src ^
    /I "C:\Program Files\AMD\XRT\include" ^
    /DMODEL_qwen3_0_6b ^
    engine\npu\src\npu_engine_universal.cpp ^
    engine\npu\src\dequant_q4nx.c ^
    /link ^
    /LIBPATH:"C:\Program Files\AMD\XRT\lib" ^
    xrt_coreutil.lib
```

## Architecture

### Portability Layer

The file `engine/npu/src/platform.h` provides a unified API across Linux and Windows:

| POSIX API | Linux | Windows |
|-----------|-------|---------|
| `open(path, O_RDONLY)` | `::open()` | `CreateFileW()` |
| `close(fd)` | `::close()` | `CloseHandle()` |
| `fstat(fd, &st)` | `::fstat()` | `GetFileSizeEx()` |
| `mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0)` | `::mmap()` | `CreateFileMapping()` + `MapViewOfFileEx()` |
| `munmap(addr, len)` | `::munmap()` | `UnmapViewOfFile()` |
| `memmem(h, hl, n, nl)` | `::memmem()` | Custom scan implementation |

### XRT API

The same XRT C++ API is available on both platforms:
- `xrt::device dev(0)` — open NPU device
- `xrt::xclbin` — load compiled kernel
- `xrt::bo` — buffer objects for NPU I/O
- `xrt::kernel` — NPU kernel dispatch

On Windows, link against `xrt_coreutil.lib` (from `xrt_coreutil.dll`) instead of `libxrt_coreutil.so`.

### OpenMP

NPU engine uses `#pragma omp parallel for` in:
- Attention score computation
- LM head (final projection)

MSVC supports `/openmp` (VS 2022+). For environments without OpenMP, the code falls back to single-threaded execution (slower but correct).

## Running

```powershell
# Set paths to xclbins and model
set XCLBIN_DIR=C:\npu\xclbins
set MODEL_PATH=C:\npu\models\qwen3-0.6b.q4nx

# Run inference
npu_engine_qwen3_0_6b.exe %MODEL_PATH% 9
```

## Status

| Component | Windows | Notes |
|-----------|---------|-------|
| dequant_q4nx.c | ✅ Verified | Pure C, portable |
| platform.h | ✅ Verified | Compiles on MSVC |
| CMakeLists.txt | ✅ Ready | Cross-platform build |
| XRT xrt_coreutil.dll | ✅ Available | From AMD NPU driver |
| NPU driver (ipustack.sys) | ✅ Available | WHQL-signed |
| Engine source | ✅ Patched | All _WIN32 conditionals in place |
| xclbins | ✅ No change | Same .xclbin files as Linux |
| Model loading | ✅ Portable | Q4NX parser is platform-independent |

**Not yet tested**: The Windows XRT DLLs need to be verified against actual Strix Halo hardware running Windows. The NPU driver (`ipustack.sys`) is WHQL-signed and the RadeonML DirectML backend is confirmed working, but our custom XRT path (`xrt_coreutil.dll` → `xrt::bo` → `xrt::kernel`) needs validation on real Windows hardware.

## See Also

- [Build Guide (Linux)](building.md)
- [Benchmarks](wiki/performance.md)
- [NPU Driver Download](https://www.amd.com/en/products/software/ryzen-ai.html)
- `NPU_RAI_376_WHQL.zip` — Windows NPU driver pack (26 MB)
- `ryzen-ai-lt-1.8.0-beta.exe` — Ryzen AI LT toolkit (3.3 GB)

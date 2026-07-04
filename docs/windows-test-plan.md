# Windows Port — First Build Test Plan

**Target**: Build and run `npu_engine_universal` (qwen3-0.6b variant) on Windows 11 with AMD Strix Halo NPU.
**Estimated time**: 2-4 hours on a Windows Strix Halo machine.

---

## Phase 0 — Environment Setup (30 min)

### 0.1 Install NPU Driver
- Run `npu_sw_installer.exe` from `NPU_RAI_376_WHQL.zip`
- Or install `ryzen-ai-lt-1.8.0-beta.exe` (3.3 GB, includes headers + libs)
- Verify: `xrt-smi.exe examine` shows NPU device

**Expected**: NPU detected, firmware version ≥ 1.1.2.65

**Failure modes**:
- Driver not WHQL-signed for this Windows build → enable test signing
- NPU not visible in Device Manager → check BIOS: NPU must be enabled

### 0.2 Install Build Tools
```powershell
winget install Microsoft.VisualStudio.2022.Community --silent --custom "--add Microsoft.VisualStudio.Workload.NativeDesktop --includeRecommended"
winget install Kitware.CMake
```

Verify:
```powershell
cl /?        # MSVC compiler
cmake --version  # ≥ 3.20
```

### 0.3 Locate XRT SDK
Find XRT headers and .lib files:
```powershell
# Typical install paths:
dir "C:\Program Files\AMD\XRT\include\xrt\"   # xrt_device.h, xrt_bo.h, xrt_kernel.h
dir "C:\Program Files\AMD\XRT\lib\"            # xrt_coreutil.lib
```

**Expected**: Headers present matching the Linux XRT C++ API.

---

## Phase 1 — Build System Validation (30 min)

### 1.1 CMake Configure
```powershell
cd engine\npu
cmake -B build -S . ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DXRT_DIR="C:\Program Files\AMD\XRT"
```

**Expected**: Configures without errors. Outputs:
```
Platform: Windows
XRT found: C:\Program Files\AMD\XRT\include
OpenMP found: /openmp
NPU engine targets:
  + npu_engine_qwen3_0_6b
  + npu_engine
```

**Failure modes**:
- `xrt/xrt_device.h` not found → set correct `-DXRT_DIR`
- OpenMP not found → OK, builds without (slower but correct)

### 1.2 CMake Build
```powershell
cmake --build build --config Release
```

**Expected**: Compiles cleanly. Key warnings to watch:
- C4267 (size_t truncation) — expected from MSVC, benign
- D9025 (override `/O2` with `/openmp`) — expected, MSVC quirk

**Critical failures**:
- `xrt::device` / `xrt::bo` / `xrt::kernel` not found in `xrt_coreutil.lib` ⬅ **HIGHEST RISK**
- `#pragma omp` ignored with warning — OK, falls back to serial
- `std::c++23` features not supported → need VS 2022 17.12+

### 1.3 Manual MSVC Build (Fallback)
If CMake fails, try direct command line:
```powershell
cl /std:c++23 /O2 /arch:AVX2 ^
    /I engine\npu\src ^
    /I "C:\Program Files\AMD\XRT\include" ^
    /DMODEL_qwen3_0_6b ^
    engine\npu\src\npu_engine_universal.cpp ^
    engine\npu\src\dequant_q4nx.c ^
    /link /LIBPATH:"C:\Program Files\AMD\XRT\lib" xrt_coreutil.lib
```

---

## Phase 2 — Platform Shim Validation (15 min)

### 2.1 Build a Shim Test Harness
Create a small test that exercises every `platform_*` function independently:
```cpp
#include "platform.h"
#include <cstdio>

int main() {
    // platform_open_read + platform_fstat + platform_mmap + platform_close
    auto fd = platform_open_read("test.q4nx");
    platform_stat st;
    platform_fstat(fd, &st);
    printf("File size: %lld\n", (long long)st.st_size);
    auto* data = (uint8_t*)platform_mmap(st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    printf("First byte: %02x\n", data[0]);
    platform_munmap(data, st.st_size);
    platform_close(fd);

    // platform_memmem
    const char* h = "hello world";
    auto* r = (const char*)platform_memmem(h, 11, "world", 5);
    printf("memmem: %s\n", r ? "PASS" : "FAIL");

    printf("All shim tests passed\n");
    return 0;
}
```

**Expected**: File maps correctly, memory contents match source file, memmem finds the needle.

---

## Phase 3 — XRT API Smoke Test (30 min)

### 3.1 Device Enumeration
```cpp
#include <xrt/xrt_device.h>
#include <cstdio>

int main() {
    try {
        auto dev = xrt::device(0);
        printf("Device name: %s\n", dev.get_info<xrt::info::device::name>().c_str());
        printf("Device bdf: %s\n", dev.get_info<xrt::info::device::bdf>().c_str());
    } catch (const std::exception& e) {
        printf("FAIL: %s\n", e.what());
        return 1;
    }
    return 0;
}
```

**Expected**: Device opens, name mentions "NPU" or "IPU" or "Strix".

**Failure modes**: ⬅ **HIGHEST RISK**
- `xrt::device(0)` throws `std::system_error` with "No such device" — XRT DLL might use a different device path on Windows (`\\.\NPU0` vs `/dev/accel/accel0`)
- `get_info<xrt::info::device::name>()` returns empty — XRT info API might differ on Windows
- DLL load failure (`xrt_coreutil.dll` not found) — PATH issue

### 3.2 xclbin Load
```cpp
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <cstdio>

int main() {
    try {
        auto dev = xrt::device(0);
        auto xc = xrt::xclbin("final_i8_QKV_v.xclbin");
        dev.register_xclbin(xc);
        auto hc = xrt::hw_context(dev, xc.get_uuid());
        auto k = xrt::kernel(hc, "MLIR_AIE");
        printf("xclbin loaded, kernel created\n");
    } catch (const std::exception& e) {
        printf("FAIL: %s\n", e.what());
        return 1;
    }
    return 0;
}
```

**Expected**: xclbin loads, kernel created from `"MLIR_AIE"` name.

**Failure modes**: ⬅ **HIGH RISK**
- xclbin version mismatch — Windows driver may require different xclbin format
- `register_xclbin` fails — Windows NPU firmware may reject Linux-compiled xclbins
- Kernel name `"MLIR_AIE"` not found — Windows xclbins may use a different symbol

### 3.3 Buffer Allocation + DMA
```cpp
// Buffer BO + sync
auto bo = xrt::bo(dev, 4096, XRT_BO_FLAGS_HOST_ONLY, k.group_id(1));
auto* ptr = bo.map();
memset(ptr, 0x42, 4096);
bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
printf("First byte: %02x\n", ((uint8_t*)ptr)[0]);  // should be 0x42
```

**Expected**: Buffer allocated, host-visible, DMA sync completes without crash.

---

## Phase 4 — End-to-End Engine Test (45 min)

### 4.1 Model Loading from Q4NX
With a `model.q4nx` on the Windows filesystem:
```powershell
npu_engine_qwen3_0_6b.exe C:\npu\models\qwen3-0.6b.q4nx 1
```

**Expected**: Engine loads model, prints model info, doesn't crash during `mmap`/parse.

**Failure modes**:
- `platform_mmap` returns `MAP_FAILED` → check file path encoding (wide char conversion)
- JSON parsing fails → `platform_memmem` has bug in the fallback implementation
- Model dimensions mismatch → verify model file is correct Q4NX format

### 4.2 GEMM Kernel Dispatch (Dry Run)
Same command but with a short prompt:
```powershell
npu_engine_qwen3_0_6b.exe C:\npu\models\qwen3-0.6b.q4nx 1
```

**Expected**: Engine initializes 4 GEMM contexts (QKV, O, GU, D), dispatches kernels, prints timing.

**Failure modes**: ⬅ **HIGH RISK**
- `xrt::bo` allocation fails → `XRT_BO_FLAGS_HOST_ONLY` may not be supported on Windows
- `xrt::kernel::operator()` throws → NPU might not support the xclbin at kernel group IDs used
- `r.wait()` hangs indefinitely → Windows XRT event handling may differ

### 4.3 Inference — Tokens Produced
```powershell
npu_engine_qwen3_0_6b.exe C:\npu\models\qwen3-0.6b.q4nx 9
```

**Expected**: Engine produces output tokens (may be garbage — same as Linux v12 correctness state, this is expected).

**Measurement**:
- Latency per token (ms/tok)
- Tokens per second
- Compare against Linux baseline: ~36 ms/tok (28 tok/s) for "all models" variant, ~10-16 ms/tok for v12 variant

---

## Phase 5 — Regression Matrix

| Test | Linux Baseline | Windows Target | Pass Criteria |
|------|---------------|----------------|--------------|
| Compile (dequant only) | ✅ `gcc -O3` | MSVC `/O2` | No errors, single binary |
| Compile (full engine) | ✅ `g++ -std=c++23 -O3` | MSVC `/std:c++23 /O2` | No errors, one .exe |
| `platform_open_read` | Returns fd | Returns HANDLE | File size matches |
| `platform_mmap` | `PROT_READ\|MAP_PRIVATE` | `FILE_MAP_READ` | Data matches file |
| `platform_memmem` | GNU libc | Custom scan | Same result as Linux |
| `xrt::device(0)` | Opens NPU | Opens NPU | Non-null device handle |
| `xrt::xclbin` | Loads .xclbin | Loads .xclbin | No exception |
| `xrt::bo(map+sync)` | DMA works | DMA works | Buffer roundtrip correct |
| `xrt::kernel()` | Dispatch works | Dispatch works | No hang, no crash |
| Engine: model load | Parses Q4NX | Parses Q4NX | Model config printed |
| Engine: prefill | 14 ms/tok | < 50 ms/tok | Tokens produced |
| Engine: decode | 36 ms/tok | < 100 ms/tok | Measurable throughput |
| Engine: munmap | Clean exit | Clean exit | No crash, exit code 0 |

---

## Phase 6 — Debugging Toolkit

### Build a "no-XRT" stub for shim testing
If XRT DLLs fail, verify the platform shim independently:
```cpp
// test_shim_only.cpp — no XRT dependency
#include "platform.h"
// ... run all tests from Phase 2
```

### Enable debug logging
Add to `platform.h` temporarily:
```cpp
#if PLATFORM_WIN32
#define PLATFORM_LOG(fmt, ...) printf("[platform] " fmt "\n", __VA_ARGS__)
// Call in each platform_ function
#endif
```

### Check Windows XRT DLL exports
```powershell
dumpbin /exports "C:\Program Files\AMD\XRT\bin\xrt_coreutil.dll"
```
Look for: `xrtDeviceOpen`, `xrtBOAlloc`, `xrtXCLBinLoad`, `xrtKernelRun` — these are the C API functions the C++ headers call internally.

### If xrt_coreutil.lib link fails
The Windows XRT DLLs might export a C-only API without C++ headers. Check if the Linux C++ headers compile against the Windows DLL by examining `dumpbin /exports` output and comparing with the symbols the C++ headers reference.

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-----------|--------|-----------|
| XRT C++ symbols differ in Windows DLL | 🔴 High | Blocker | Use C XRT API directly, or write a thin Win32 NPU backend |
| xclbin format mismatch | 🟡 Medium | Blocker | Compile xclbins on Windows via `xclbinutil.exe` or use AMD's Windows xclbins |
| NPU device path different | 🟡 Medium | Blocker | Check `xrt-smi.exe examine` for device enumeration, patch `xrt::device(0)` if needed |
| mmap shim bug for large files | 🟢 Low | Annoying | Test with small (1KB) and large (1GB) files |
| OpenMP performance worse | 🟢 Low | Performance | Acceptable — MSVC OpenMP is slower than libgomp |
| `std::isfinite` missing on MSVC | 🟢 Low | Compile error | Already handled — MSVC has `_finite()` |

---

## First Build Checklist

- [ ] Phase 0.1: NPU driver installed, `xrt-smi.exe examine` passes
- [ ] Phase 0.2: MSVC + CMake installed
- [ ] Phase 0.3: XRT headers + .lib located
- [ ] Phase 1.1: CMake configure succeeds
- [ ] Phase 1.2: CMake build succeeds (no link errors)
- [ ] Phase 2.0: Shim test harness runs (mmap, memmem work)
- [ ] Phase 3.1: `xrt::device(0)` opens NPU
- [ ] Phase 3.2: xclbin loads, kernel created
- [ ] Phase 3.3: Buffer alloc + DMA sync works
- [ ] Phase 4.1: Model file loads and parses
- [ ] Phase 4.2: GEMM dispatch completes
- [ ] Phase 4.3: Tokens produced, timing recorded

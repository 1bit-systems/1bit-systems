# Windows Support

The AMD XDNA 2 NPU exists on Windows too (Ryzen AI 300 series, Strix Halo).
Here's the path to get 1bit.systems running there.

## Current Status

| Feature | Status | Notes |
|---------|--------|-------|
| NPU engine (C++23) | ❌ Untested | Requires XRT for Windows + MSVC/clang-cl |
| GPU engine (Zinc) | ❌ Vulkan untested | Zinc has Vulkan backend but hasn't been tested on Windows |
| HTTP server | ✅ Should work | Pure C++ sockets, no platform-specific code |
| install.sh | ❌ | Bash script, needs PowerShell equivalent |

## What's Needed

### 1. XRT for Windows

AMD distributes XRT for Windows through the NPU driver package.
Check if `xrt_coreutil.dll` is available after installing the
official AMD NPU driver:

```powershell
# Check for XRT
where.exe xrt_coreutil.dll 2>$null
# Or check Program Files
Get-ChildItem "$env:ProgramFiles\AMD\XRT" -Recurse -Filter "*.dll"
```

If XRT isn't available, the engine can't talk to the NPU directly.
The GPU (Vulkan) backend is the more practical path.

### 2. Build Toolchain

```powershell
# Option A: MSVC (cl.exe)
cl /std:c++23 /O2 /EHsc npu_engine_all.cpp dequant_q4nx.c /I engine\npu\src /link xrt_coreutil.lib

# Option B: clang-cl (via Visual Studio)
clang-cl -std=c++23 -O3 npu_engine_all.cpp dequant_q4nx.c -Iengine/npu/src -lxrt_coreutil
```

### 3. PowerShell Install Script

A `npu-install.ps1` would replace the bash script:

```powershell
# npu-install.ps1 (stub — not yet implemented)
$installDir = "$env:LOCALAPPDATA\1bit-npu"
$binDir = "$env:USERPROFILE\.local\bin"

# Download latest release from GitHub
$release = Invoke-RestMethod "https://api.github.com/repos/bong-water-water-bong/1bit-systems/releases/latest"
# ... extract and install
```

### 4. Driver Requirements

| Component | Notes |
|-----------|-------|
| AMD NPU Driver | From AMD.com or Windows Update |
| XDNA 2 Runtime | Included with driver on Strix Halo |
| Vulkan Runtime | Included with Windows (or from LunarG) |

## Community Help Wanted

If you have a Strix Halo machine running Windows and want to help:

1. Test if `xrt_coreutil.dll` exists after installing AMD drivers
2. Try building with MSVC (report any C++23 compatibility issues)
3. Try running the HTTP server (it's platform-independent)
4. Open an issue with your findings

## Quick Test (Windows)

```powershell
# 1. Open "Developer PowerShell" (as admin)
# 2. Clone the repo
git clone https://github.com/bong-water-water-bong/1bit-systems
cd 1bit-systems

# 3. Try building the HTTP server (most portable component)
cl /std:c++23 /O2 /EHsc packaging\binary\server.cpp /Fe:1bit-server.exe

# 4. Run it
.\1bit-server.exe 8081

# 5. Test the API
curl -X POST http://localhost:8081/v1/chat/completions -d "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}"
```

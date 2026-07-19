# Building zaya_server (C++ / ROCm)

This document covers building **zaya_server** — a pure C++ inference server with optional
GPU decoding support. No Rust, no Python at runtime. The host CPU is **AMD Strix Halo**
(Ryzen AI Max+ 395) and GPU acceleration uses **ROCm 7.2.4** targeting `gfx1151`.

---

## Prerequisites

| Package            | Version / Notes                                     |
|--------------------|-----------------------------------------------------|
| Ubuntu             | 24.04 LTS or later (CachyOS / Arch also works)      |
| Kernel             | 6.18.22-lts or 7.x — **not** 6.19.x (issue #1 hang) |
| ROCm               | 7.2.4                                               |
| CMake              | ≥ 3.28                                              |
| Ninja              | ≥ 1.12                                              |
| GCC                | ≥ 13 (C++20) or ≥ 14 (C++23)                        |
| Git                | —                                                   |

Install system dependencies:

```bash
sudo apt update
sudo apt install -y cmake ninja-build build-essential git
```

---

## ROCm 7.2.4

Install ROCm via the official AMD repository or a local package install.

**Example — repository install**

```bash
# Add AMD ROCm repository (adjust for your distro)
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
echo "deb [signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2.4 noble main" \
  | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install -y rocm-dev hipcc

# Verify
/opt/rocm/bin/rocminfo
/opt/rocm/bin/hipconfig --full
```

**Set `CMAKE_HIP_ARCHITECTURES`** so that HIP kernels are compiled for the Strix Halo GPU:

```bash
export CMAKE_HIP_ARCHITECTURES=gfx1151
```

It is convenient to add this to your shell profile:

```bash
echo 'export CMAKE_HIP_ARCHITECTURES=gfx1151' >> ~/.bashrc
```

---

## Build: zaya_server (required)

Clone the repository and build the main server binary:

```bash
# Clone (adjust URL to match your remote)
cd ~
git clone <your-repo-url> zaya
cd zaya

# Configure
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_PREFIX_PATH=/opt/rocm

# Build the server binary
cmake --build build --target zaya_server
```

The resulting binary is `build/zaya_server`.

---

## Build: zaya_gpu_decode (optional)

If your model uses the **Q4NX** quantisation format, you can build `zaya_gpu_decode`
to offload the dequantisation and matmul steps to the GPU:

```bash
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DZAYA_ENABLE_GPU_DECODE=ON

cmake --build build --target zaya_gpu_decode
```

The resulting shared library (or object) is `build/libzaya_gpu_decode.so`.

> **Note:** `zaya_server` will auto-detect the presence of this library at startup
> and use it when loading Q4NX models. Building without `ZAYA_ENABLE_GPU_DECODE`
> disables GPU decode; the server still runs, but inference stays entirely on CPU.

---

## Build: llama.cpp with ROCm backend (optional)

If the server depends on **llama.cpp** and you want its inference to use the same
ROCm device:

```bash
# Either bundled in the zaya repo or standalone
cd path/to/llama.cpp

cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DGGML_HIP=ON \
  -DCMAKE_PREFIX_PATH=/opt/rocm

cmake --build build --target llama
```

Then ensure `zaya_server`'s CMake configuration points to this build (e.g. via
`-DLLAMA_DIR=/path/to/llama.cpp/build` during the zaya configure step).

---

## CMake option summary

| Option                     | Default | Description                                |
|----------------------------|---------|--------------------------------------------|
| `ZAYA_ENABLE_GPU_DECODE`   | OFF     | Build `zaya_gpu_decode` for Q4NX GPU offload |
| `ZAYA_USE_LLAMACPP_ROCM`   | OFF     | Link llama.cpp compiled with `GGML_HIP=ON` |
| `CMAKE_HIP_ARCHITECTURES`  | —       | **Must** be set to `gfx1151`               |

---

## Running

```bash
./build/zaya_server --model /path/to/model
```

If `libzaya_gpu_decode.so` was built and is findable, the server will print a
message at startup confirming GPU decode is active.

---

## Troubleshooting

### `hipErrorNoBinaryForGPU`

The `CMAKE_HIP_ARCHITECTURES` variable was not set, or was set to the wrong target.
Ensure it is `gfx1151` and that ROCm 7.2.4 is installed (older ROCm releases may
not include code-objects for gfx1151).

### `cannot find -lamdhip64`

ROCm is not on the linker path. Pass `-DCMAKE_PREFIX_PATH=/opt/rocm` during
CMake configuration.

### No GPU decode even though `zaya_gpu_decode` was built

Check that the shared library is in the library search path:

```bash
export LD_LIBRARY_PATH=/path/to/zaya/build:$LD_LIBRARY_PATH
```

Also verify the model file is actually Q4NX (check the file header or extension).

### Kernel hang on first inference on Strix Halo (issue #1)

Strix Halo (gfx1151) systems may hit an amdgpu OPTC hang on the first GPU kernel launch
after cold boot. The hang is intermittent (~1 in 5 boots).

**Symptoms:** first inference call hangs; `dmesg` shows OPTC lockup messages; reboot required.

**Mitigation:**
```bash
export HSA_ENABLE_SDMA=0   # avoids the triggering OPTC code path
```
See [issue #1](https://github.com/bong-water-water-bong/1bit-systems/issues/1) for details.

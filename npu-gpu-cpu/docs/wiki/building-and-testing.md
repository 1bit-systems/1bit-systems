# Building and Testing

## Prerequisites

- AMD Strix Halo system (Ryzen AI MAX+ 395 or similar)
- Ubuntu 26.04+ with kernel 7.0.0+
- ROCm 7.1+ (`sudo apt install rocm-dev hipcc`)
- XRT runtime (`libxrt_coreutil`) for NPU inference
- libdrm-dev (`sudo apt install libdrm-dev libdrm-amdgpu-dev`)

## Quick Start

```bash
# Clone the repo
git clone https://github.com/bong-water-water-bong/npu-gpu-cpu.git
cd npu-gpu-cpu

# Build everything
make

# Run hardware tests
sudo make test

# Run benchmarks
sudo make bench

# Build NPU parallelism tools
make npu-tools

# Start the control plane daemon
python3 daemon/npu-gpu-cpud.py --port 8080
```

## Individual Build Steps

### 1. NPU Device Test
```bash
make
./build/test_npu_dev
# Expected: NPU buffer alloc + dma-buf export OK
```

### 2. DMA-Buf Zero-Copy Test
```bash
sudo ./build/test_gtt_dmabuf
# Expected: PASS — GPU and NPU access same physical pages
```

### 3. DMA-Buf Bandwidth Benchmark
```bash
sudo ./build/bench_gtt_dmabuf
# Expected: 27 GB/s read, 56 GB/s write
```

### 4. HIP NPU Shim
```bash
cmake -S rocm-npu -B rocm-npu/build -DHIP_INCLUDE_DIR=/usr/include
cmake --build rocm-npu/build
# Produces: rocm-npu/build/libhip_npu.so

# Test HIP device enumeration
LD_PRELOAD=./rocm-npu/build/libhip_npu.so ./tests/hip_list_devices
# Expected: 2 devices — GPU 0 + NPU 1
```

### 5. Control Plane Daemon
```bash
python3 daemon/npu-gpu-cpud.py --port 8080

# Health check
curl http://localhost:8080/v1/health

# Chat completion (via NPU engine)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"npu://151643,872,198,11852,151644,198","messages":[]}'
```

### 6. NPU Parallelism Tools
```bash
make npu-tools

# Multi-context verification
sudo ./build/npu_multi_context_engine 4

# Scaling test (1..8 contexts)
sudo ./build/npu_server --scaling --ctx 8

# Concurrent benchmark
sudo ./build/npu_server --bench --ctx 8 --requests 32
```

## Kernel Module Build

```bash
# Requires full kernel source tree
cd /usr/src/linux-source-7.0.0  # or your kernel tree

# Apply NPU IP block patches
cp patches/amdgpu_npu.c     drivers/gpu/drm/amd/amdgpu/
cp patches/amdgpu_npu.h     drivers/gpu/drm/amd/amdgpu/
cp patches/amdgpu_npu_mgr.c drivers/gpu/drm/amd/amdgpu/
cp patches/amdgpu_npu_sched.c drivers/gpu/drm/amd/amdgpu/

# Apply integration patch
git am patches/0002-add-npu-ip-block.patch

# Build just the amdgpu module
make -C /lib/modules/$(uname -r)/build \
  M=$PWD/drivers/gpu/drm/amd/amdgpu modules -j$(nproc)
```

## Testing on Different Hardware

| Test | Strix Halo | Phoenix | Phoenix2 |
|------|-----------|---------|----------|
| dma-buf zero-copy | ✅ Tested | Untested | Untested |
| NPU inference (INT8) | ✅ 28 tok/s | Untested | Untested |
| SMU init fix | ✅ Tested | Should work | Should work |
| HIP shim | ✅ 2 devices | Untested | Untested |
| Multi-context pool | ✅ 7.9× scaling | Untested | Untested |

## CI

Tests require a self-hosted Strix Halo runner with:
- `/dev/accel/accel0` — NPU device
- ROCm 7.1+ — GPU compute

See `.github/workflows/` for CI configuration.

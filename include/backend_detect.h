// backend_detect.h — Hardware detection probes for all supported backends
// Portable: no linking required — uses dlopen/popen/sysfs for probing.

#pragma once
#include "backend.h"

/// Check if a HIP-capable GPU is available (looks for renderD* devices + HIP runtime)
bool has_hip_gpu();

/// Check if Vulkan 1.2+ is available with at least one physical device
bool has_vulkan();

/// Check if AMD XDNA NPU is accessible via XRT, sysfs, or amdxdna driver
bool has_npu();

/// Check if a CUDA-capable NVIDIA GPU is available
bool has_cuda();

/// Check if an Apple Metal-capable GPU is available (macOS only)
bool has_metal();

/// Check if the CPU supports AVX-512
bool has_avx512();

/// Probe all backends, return the best available type
BackendType detect_backends();

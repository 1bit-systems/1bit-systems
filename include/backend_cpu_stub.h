// backend_cpu_stub.h — Always-available CPU backend creation
// The CPU backend is pure C++ (no HIP, no GPU) so it ships in the backend_manager library.
// This header is included by backend_factory.cpp and backend_manager.cpp.

#pragma once
#include "backend.h"

/// Create a CPU backend (scalar or AVX-512 depending on runtime detection).
/// Always returns a valid backend — no GPU/NPU dependencies.
Backend* create_cpu_backend();

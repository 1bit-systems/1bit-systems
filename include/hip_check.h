// hip_check.h — Shared HIP error-checking macro for the 1bit-systems codebase.
//
// Replaces the 12+ independent per-file HIP_CHECK definitions with one canonical
// version.  Does NOT call abort() — errors are logged and propagated so a
// multi-backend server can fail over to CPU/Vulkan instead of crashing.
//
// Every kernel file should include this header instead of defining its own
// HIP_CHECK / HIP_OK / HIP_OK_V macros.

#pragma once

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>

// ── Non-fatal error macros (log + throw, server survives) ────────────────────

#ifndef HIP_CHECK
#define HIP_CHECK(e) do {                                                       \
    hipError_t _s_ = (e);                                                       \
    if (_s_ != hipSuccess) {                                                    \
        fprintf(stderr, "HIP Error %s:%d: %s (code %d)\n",                      \
                __FILE__, __LINE__, hipGetErrorString(_s_), (int)_s_);          \
        throw std::runtime_error(std::string("HIP error at ") + __FILE__ +      \
                                 ":" + std::to_string(__LINE__) + ": " +        \
                                 hipGetErrorString(_s_));                       \
    }                                                                           \
} while(0)
#endif

#ifndef HIP_OK
#define HIP_OK(e) do {                                                          \
    hipError_t _s_ = (e);                                                       \
    if (_s_ != hipSuccess) {                                                    \
        fprintf(stderr, "HIP Error %s:%d: %s (code %d)\n",                      \
                __FILE__, __LINE__, hipGetErrorString(_s_), (int)_s_);          \
        throw std::runtime_error(std::string("HIP error at ") + __FILE__ +      \
                                 ":" + std::to_string(__LINE__) + ": " +        \
                                 hipGetErrorString(_s_));                       \
    }                                                                           \
} while(0)
#endif

#ifndef HIP_OK_V
#define HIP_OK_V(e) do {                                                        \
    hipError_t _s_ = (e);                                                       \
    if (_s_ != hipSuccess) {                                                    \
        fprintf(stderr, "HIP Error %s:%d: %s (code %d)\n",                      \
                __FILE__, __LINE__, hipGetErrorString(_s_), (int)_s_);          \
        return;                                                                 \
    }                                                                           \
} while(0)
#endif

#ifndef HIP_OK_R
#define HIP_OK_R(e, retval) do {                                                \
    hipError_t _s_ = (e);                                                       \
    if (_s_ != hipSuccess) {                                                    \
        fprintf(stderr, "HIP Error %s:%d: %s (code %d)\n",                      \
                __FILE__, __LINE__, hipGetErrorString(_s_), (int)_s_);          \
        return (retval);                                                        \
    }                                                                           \
} while(0)
#endif

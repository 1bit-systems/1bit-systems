/** platform.h — Platform abstraction for NPU engine (Linux → Win32 port)
 *
 *  Provides portable equivalents for POSIX APIs used by the engine:
 *    - open / close / read / fstat  → Win32 wrappers
 *    - mmap / munmap                → Win32 Memory Mapped Files
 *    - memmem (GNU extension)       → portable fallback
 *    - XRT includes                 → conditionally select path
 *
 *  Include this as the first non-system header in every engine .cpp file.
 *  On Linux it's a transparent passthrough; on Windows it provides the shim.
 */
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>
#include <memory>
#include <cassert>

// ============================================================
// Platform detection
// ============================================================
#if defined(_WIN32) || defined(_WIN64)
#  define PLATFORM_WIN32 1
#else
#  define PLATFORM_LINUX 1
#endif

// ============================================================
// Include XRT headers (path differs on Windows vs Linux)
// ============================================================
#if PLATFORM_LINUX
#  include <xrt/xrt_device.h>
#  include <xrt/xrt_bo.h>
#  include <xrt/xrt_kernel.h>
#elif PLATFORM_WIN32
   // Windows XRT DLLs export the same C++ API via xrt_core.dll / xrt_coreutil.dll
   // Headers ship with the AMD Ryzen AI LT package or NPU driver SDK.
   // Set XRT_INCLUDE_DIR /XRT_LIB_DIR in CMake.
#  include <xrt/xrt_device.h>
#  include <xrt/xrt_bo.h>
#  include <xrt/xrt_kernel.h>
#endif

// ============================================================
// POSIX file I/O → portable wrappers
// ============================================================
#if PLATFORM_LINUX
#  include <fcntl.h>
#  include <unistd.h>
#  include <sys/stat.h>
#  include <sys/mman.h>

   // Already have open/close/fstat/mmap — no shim needed.
   using PlatformFileHandle = int;
   using platform_stat = struct stat;

   inline PlatformFileHandle platform_open_read(const char* path) {
       return ::open(path, O_RDONLY);
   }
   inline int platform_close(PlatformFileHandle fd) {
       return ::close(fd);
   }
   inline int platform_fstat(PlatformFileHandle fd, struct stat* st) {
       return ::fstat(fd, st);
   }
   inline void* platform_mmap(size_t len, int prot, int flags, PlatformFileHandle fd, off_t off) {
       return ::mmap(NULL, len, prot, flags, fd, off);
   }
   inline int platform_munmap(void* addr, size_t len) {
       return ::munmap(addr, len);
   }

   // memmem is a GNU extension available on Linux
   inline void* platform_memmem(const void* haystack, size_t hl, const void* needle, size_t nl) {
       return ::memmem(haystack, hl, needle, nl);
   }

#elif PLATFORM_WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <io.h>
#  include <fcntl.h>

   using PlatformFileHandle = HANDLE;

   inline PlatformFileHandle platform_open_read(const char* path) {
       // Convert UTF-8 path to wide for proper Unicode support
       int wlen = MultiByteToWideChar(CP_UTF8, 0, path, -1, NULL, 0);
       if (wlen <= 0) return INVALID_HANDLE_VALUE;
       wchar_t* wpath = (wchar_t*)_malloca(wlen * sizeof(wchar_t));
       MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, wlen);
       HANDLE h = CreateFileW(wpath, GENERIC_READ, FILE_SHARE_READ, NULL,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
       _freea(wpath);
       return h;
   }

   inline int platform_close(PlatformFileHandle fd) {
       return CloseHandle(fd) ? 0 : -1;
   }

   // Minimal stat replacement — extract file size from handle
   struct platform_stat {
       int64_t st_size;
   };

   inline int platform_fstat(PlatformFileHandle fd, platform_stat* st) {
       LARGE_INTEGER size;
       if (!GetFileSizeEx(fd, &size)) return -1;
       st->st_size = size.QuadPart;
       return 0;
   }

   // mmap equivalent via CreateFileMapping + MapViewOfFile
   inline void* platform_mmap(size_t len, int prot, int flags,
                              PlatformFileHandle fd, int64_t off) {
       DWORD flProtect = PAGE_READONLY;
       DWORD dwDesiredAccess = FILE_MAP_READ;
       (void)prot; (void)flags; // both implied by read-only mapping

       HANDLE hMap = CreateFileMappingW(fd, NULL, flProtect,
                                        (DWORD)(len >> 32), (DWORD)len, NULL);
       if (!hMap) return MAP_FAILED;

       void* addr = MapViewOfFileEx(hMap, dwDesiredAccess,
                                    (DWORD)(off >> 32), (DWORD)off, len, NULL);
       CloseHandle(hMap); // view retains reference to the mapping
       return addr ? addr : MAP_FAILED;
   }

   inline int platform_munmap(void* addr, size_t len) {
       (void)len;
       return UnmapViewOfFile(addr) ? 0 : -1;
   }

   // memmem is not available on Windows — implement via Boyer-Moore-like scan
   inline void* platform_memmem(const void* haystack, size_t hl,
                                 const void* needle, size_t nl) {
       if (!nl) return const_cast<void*>(haystack);
       if (nl > hl || !haystack || !needle) return NULL;
       const unsigned char* h = (const unsigned char*)haystack;
       const unsigned char* n = (const unsigned char*)needle;
       for (size_t i = 0; i <= hl - nl; i++) {
           if (h[i] == n[0] && memcmp(h + i, n, nl) == 0)
               return const_cast<unsigned char*>(h + i);
       }
       return NULL;
   }

   // Define MAP_FAILED if not present (Win32 doesn't have <sys/mman.h>)
#  ifndef MAP_FAILED
#    define MAP_FAILED ((void*)(intptr_t)-1)
#  endif
   // Dummy PROT_READ / MAP_PRIVATE — not used in the Win32 path but kept for compat
#  ifndef PROT_READ
#    define PROT_READ 1
#  endif
#  ifndef MAP_PRIVATE
#    define MAP_PRIVATE 2
#  endif

#endif // PLATFORM_WIN32

// ============================================================
// OpenMP abstraction
// ============================================================
// The engine uses #pragma omp parallel for in attention and LM head.
// On MSVC: /openmp flag enables it (or use C++17 parallel algorithms).
// No source changes needed — handled by compiler flag.

// ============================================================
// bf16 helpers — pure portable C++
// ============================================================
static inline float bf16_to_float(uint16_t v) {
    uint32_t b = (uint32_t)v << 16;
    float f;
    memcpy(&f, &b, 4);
    return f;
}

static inline float bf16_to_float_safe(uint16_t v) {
    // Return 0 for NaN-like bf16 values (exponent = 0xFF)
    return (v & 0x7F80) == 0x7F80 ? 0.0f : bf16_to_float(v);
}

// Short aliases matching original engine naming convention
static inline float bf16f(uint16_t v) { return bf16_to_float(v); }
static inline float bf16g(uint16_t v) { return bf16_to_float_safe(v); }

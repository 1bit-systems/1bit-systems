/** platform_mmap.cpp — Win32 mmap implementation for NPU engine.
 *
 *  On Linux the mmap/munmap wrappers are inlined in platform.h.
 *  On Windows the CreateFileMapping + MapViewOfFile logic is also inlined there.
 *  This file exists as a compilation unit for the CMake build system.
 *
 *  Currently empty — all definitions are header-only via platform.h.
 */
// Platform includes are handled via the header

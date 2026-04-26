// onebit::aie::DyLoader — runtime dlopen wrapper for libxrt_coreutil.so.
//
// We do not link against libxrt at compile time. Instead we dlopen
// the library at first use and resolve the symbols we need (about a
// dozen for kernel dispatch). This means:
//
//   * The binary builds clean on a sandbox / CI host without libxrt.
//   * Runtime probe is centralised — `DyLoader::available()` is the
//     single source of truth.
//   * If libxrt is not on the runtime LD_LIBRARY_PATH, every
//     Backend method returns LibraryUnavailable rather than failing
//     to start up.
//
// XRT's public C++ surface is template-heavy; we don't bind it here.
// The intent is to dlopen one of the AMD-provided C shims (e.g. the
// `xrt_coreutil` or `xrt_xclbin` C entry points) once kernel-author
// finishes the C ABI surface that mirrors the Rust crate's
// `native/xrt_c_shim.cpp`. Until that lands, `available()` returns
// true on hosts that have libxrt_coreutil.so but no resolvable
// dispatch symbols — `Backend::load_xclbin` then fails with
// NotYetWired.

#pragma once

#include <expected>
#include <string>
#include <string_view>

#include "onebit/aie/error.hpp"

namespace onebit::aie {

// Names of the libxrt SOs we probe in order. The first one that
// dlopen's wins. We don't hard-code a path — let ld.so respect
// LD_LIBRARY_PATH + /etc/ld.so.cache.
inline constexpr std::string_view kXrtSonames[] = {
    "libxrt_coreutil.so.2",
    "libxrt_coreutil.so",
    "libxrt_core.so.2",
    "libxrt_core.so",
};

class DyLoader {
public:
    DyLoader() = default;

    DyLoader(const DyLoader&)            = delete;
    DyLoader& operator=(const DyLoader&) = delete;
    DyLoader(DyLoader&&) noexcept;
    DyLoader& operator=(DyLoader&&) noexcept;

    ~DyLoader();

    // Try to dlopen libxrt. Idempotent — repeated calls are cheap.
    // On success returns the SONAME that loaded; on failure returns
    // LibraryUnavailable carrying dlerror() output.
    [[nodiscard]] std::expected<std::string, Error> open();

    // True when `open()` returned success at least once. Cheap probe.
    [[nodiscard]] bool is_open() const noexcept { return handle_ != nullptr; }

    // Resolve a named symbol from the loaded library. Returns nullptr
    // if not loaded or symbol not found. Does NOT call dlerror() —
    // callers should branch on nullptr and report NotYetWired.
    [[nodiscard]] void* resolve(const char* name) const noexcept;

    // Force-close the handle. Mainly for tests; production callers
    // can rely on the destructor.
    void close() noexcept;

    // Label of the library that opened, or empty if not open.
    [[nodiscard]] std::string_view soname() const noexcept { return soname_; }

private:
    void*        handle_{nullptr};
    std::string  soname_{};
};

} // namespace onebit::aie

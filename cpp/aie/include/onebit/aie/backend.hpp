// onebit::aie::Backend — public NPU dispatch interface.
//
// Two implementations:
//   * StubBackend — every method returns NotYetWired. Always available.
//   * XrtBackend  — dlopens libxrt at construction; returns
//                    LibraryUnavailable when dlopen fails.
//
// Mirrors the Rust crate's AieBackend trait + StubAieBackend impl.

#pragma once

#include <expected>
#include <filesystem>
#include <memory>

#include "onebit/aie/error.hpp"
#include "onebit/aie/types.hpp"

namespace onebit::aie {

class Backend {
public:
    virtual ~Backend() = default;

    Backend()                          = default;
    Backend(const Backend&)            = delete;
    Backend& operator=(const Backend&) = delete;
    Backend(Backend&&)                 = delete;
    Backend& operator=(Backend&&)      = delete;

    // Load a compiled .xclbin and prepare it for dispatch.
    [[nodiscard]] virtual std::expected<KernelHandle, Error>
    load_xclbin(const std::filesystem::path& path) = 0;

    // Dispatch a BitNet-1.58 ternary GEMV. Shape and dtype invariants
    // documented on AieBackend::bitnet_gemv in the Rust crate.
    [[nodiscard]] virtual std::expected<void, Error>
    bitnet_gemv(KernelHandle  k,
                Buffer        weights,
                Buffer        x,
                Buffer&       out,
                Buffer        scales) = 0;

    // Cheap, call-at-startup device introspection. Never fails in the
    // stub; the real path returns NotYetWired or a populated struct.
    [[nodiscard]] virtual DeviceInfo device_info() const = 0;
};

// Stub backend. Always compiled. Every method returns NotYetWired.
class StubBackend final : public Backend {
public:
    StubBackend() = default;

    [[nodiscard]] std::expected<KernelHandle, Error>
    load_xclbin(const std::filesystem::path& path) override;

    [[nodiscard]] std::expected<void, Error>
    bitnet_gemv(KernelHandle k,
                Buffer       weights,
                Buffer       x,
                Buffer&      out,
                Buffer       scales) override;

    [[nodiscard]] DeviceInfo device_info() const override;
};

// XRT-backed backend. Constructs by dlopen-ing libxrt at runtime.
// Methods return LibraryUnavailable until the dispatch shims are
// resolved; once they are, they return NotYetWired until the C ABI
// surface is wired (kernel-author task).
class XrtBackend final : public Backend {
public:
    XrtBackend();
    ~XrtBackend() override;

    XrtBackend(const XrtBackend&)            = delete;
    XrtBackend& operator=(const XrtBackend&) = delete;

    // Returns the loaded SONAME (e.g. "libxrt_coreutil.so.2") or
    // LibraryUnavailable if dlopen failed.
    [[nodiscard]] std::expected<std::string, Error> probe_runtime();

    [[nodiscard]] std::expected<KernelHandle, Error>
    load_xclbin(const std::filesystem::path& path) override;

    [[nodiscard]] std::expected<void, Error>
    bitnet_gemv(KernelHandle k,
                Buffer       weights,
                Buffer       x,
                Buffer&      out,
                Buffer       scales) override;

    [[nodiscard]] DeviceInfo device_info() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Construct the best backend available given the runtime environment.
// Tries XrtBackend first; if libxrt isn't loadable, returns a
// StubBackend. Owners hold a `std::unique_ptr<Backend>`.
[[nodiscard]] std::unique_ptr<Backend> make_default_backend();

} // namespace onebit::aie

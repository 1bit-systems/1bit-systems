// backend.cpp — StubBackend + XrtBackend implementations.
//
// XrtBackend dlopens libxrt at probe_runtime() time. Until kernel-author
// finishes the C ABI surface (analogous to the Rust crate's
// xrt_c_shim.cpp), every dispatch method returns NotYetWired even when
// libxrt loaded successfully.

#include "onebit/aie/backend.hpp"

#include "onebit/aie/dyloader.hpp"

#include <utility>

namespace onebit::aie {

// ---------------------------------------------------------------------------
// StubBackend
// ---------------------------------------------------------------------------

std::expected<KernelHandle, Error>
StubBackend::load_xclbin(const std::filesystem::path& path) {
    (void)path;
    return std::unexpected(Error{ErrorKind::NotYetWired,
                                 "StubBackend::load_xclbin"});
}

std::expected<void, Error>
StubBackend::bitnet_gemv(KernelHandle k,
                         Buffer       weights,
                         Buffer       x,
                         Buffer&      out,
                         Buffer       scales) {
    (void)k; (void)weights; (void)x; (void)out; (void)scales;
    return std::unexpected(Error{ErrorKind::NotYetWired,
                                 "StubBackend::bitnet_gemv"});
}

DeviceInfo StubBackend::device_info() const {
    DeviceInfo info;
    info.device_name      = "stub";
    info.firmware_version = "0.0.0-stub";
    info.columns          = 0;
    info.tile_class       = "AIE2P";
    return info;
}

// ---------------------------------------------------------------------------
// XrtBackend — dlopen at probe; everything else NotYetWired until the
// C ABI shims are filled in by kernel-author.
// ---------------------------------------------------------------------------

struct XrtBackend::Impl {
    DyLoader loader;
    bool     probed{false};
    bool     have_runtime{false};
    std::string runtime_soname;
};

XrtBackend::XrtBackend() : impl_{std::make_unique<Impl>()} {}
XrtBackend::~XrtBackend() = default;

std::expected<std::string, Error> XrtBackend::probe_runtime() {
    if (!impl_->probed) {
        auto r = impl_->loader.open();
        impl_->probed = true;
        if (r) {
            impl_->have_runtime   = true;
            impl_->runtime_soname = *r;
        } else {
            return std::unexpected(r.error());
        }
    }
    if (!impl_->have_runtime) {
        return std::unexpected(Error{ErrorKind::LibraryUnavailable,
                                     "libxrt not loadable"});
    }
    return impl_->runtime_soname;
}

std::expected<KernelHandle, Error>
XrtBackend::load_xclbin(const std::filesystem::path& path) {
    auto rt = probe_runtime();
    if (!rt) return std::unexpected(rt.error());

    namespace fs = std::filesystem;
    std::error_code ec;
    if (!fs::exists(path, ec) || !fs::is_regular_file(path, ec)) {
        return std::unexpected(Error{ErrorKind::XclbinNotFound, path.string()});
    }
    // Probe the canonical entry symbol from the Rust crate's planned
    // shim. When kernel-author lands xrt_c_shim.cpp this will dispatch;
    // until then we report NotYetWired so callers can route around.
    if (impl_->loader.resolve("xrt_shim_xclbin_load") == nullptr) {
        return std::unexpected(Error{
            ErrorKind::NotYetWired,
            "xrt_shim_xclbin_load symbol not present in loaded libxrt"});
    }
    return std::unexpected(Error{ErrorKind::NotYetWired,
                                 "XrtBackend::load_xclbin dispatch path"});
}

std::expected<void, Error>
XrtBackend::bitnet_gemv(KernelHandle k,
                        Buffer       weights,
                        Buffer       x,
                        Buffer&      out,
                        Buffer       scales) {
    (void)k; (void)weights; (void)x; (void)out; (void)scales;
    auto rt = probe_runtime();
    if (!rt) return std::unexpected(rt.error());
    return std::unexpected(Error{ErrorKind::NotYetWired,
                                 "XrtBackend::bitnet_gemv"});
}

DeviceInfo XrtBackend::device_info() const {
    DeviceInfo info;
    info.device_name      = impl_->have_runtime ? "RyzenAI-npu5" : "stub-xrt";
    info.firmware_version = impl_->have_runtime ? "unknown" : "0.0.0-stub";
    info.columns          = 0;
    info.tile_class       = "AIE2P";
    return info;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<Backend> make_default_backend() {
    auto x = std::make_unique<XrtBackend>();
    auto rt = x->probe_runtime();
    if (rt.has_value()) {
        return x;
    }
    // Fall back to stub. The router will see NotYetWired on every
    // dispatch, which it already handles.
    return std::make_unique<StubBackend>();
}

} // namespace onebit::aie

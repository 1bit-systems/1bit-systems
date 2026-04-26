// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 halo-ai-core / bong-water-water-bong.
//
// onebit::aie::BitnetGemmAIE2P implementation.
//
// Direct-link libxrt path. We compile this TU only when ONEBIT_AIE_HAVE_XRT_RT
// is defined; otherwise every entry point returns LibraryUnavailable so
// the rest of the tower can build clean on hosts without xrt installed.

#include "onebit/aie/bitnet_gemm_aie2p.hpp"

#include <fstream>
#include <ios>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#if defined(ONEBIT_AIE_HAVE_XRT_RT) && ONEBIT_AIE_HAVE_XRT_RT
#  include <xrt/xrt_bo.h>
#  include <xrt/xrt_device.h>
#  include <xrt/xrt_hw_context.h>
#  include <xrt/xrt_kernel.h>
#  include <xrt/experimental/xrt_xclbin.h>
#endif

namespace onebit::aie {

#if defined(ONEBIT_AIE_HAVE_XRT_RT) && ONEBIT_AIE_HAVE_XRT_RT

namespace {

// Slurp a file into bytes. Returns Error{XclbinNotFound} on any I/O error.
// Anonymous-namespace local; only used in the real-XRT build.
//
// We size the buffer up-front from filesystem::file_size and read in one shot
// so we don't pay the istreambuf_iterator cost. Also dodges the C++23 trap
// where vector<byte>(streambuf_it<char>, ...) fails because byte has no
// implicit char ctor (libstdc++ 15 enforces construct_at constraint).
[[nodiscard]] std::expected<std::vector<std::byte>, Error>
read_file(const std::filesystem::path& path)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    if (!fs::exists(path, ec) || !fs::is_regular_file(path, ec)) {
        return std::unexpected(Error{ErrorKind::XclbinNotFound, path.string()});
    }
    const auto sz = fs::file_size(path, ec);
    if (ec) {
        return std::unexpected(Error{ErrorKind::XclbinNotFound,
                                     "stat failed: " + path.string()});
    }
    std::ifstream f{path, std::ios::binary};
    if (!f) {
        return std::unexpected(Error{ErrorKind::XclbinNotFound,
                                     "open failed: " + path.string()});
    }
    std::vector<std::byte> buf(static_cast<std::size_t>(sz));
    if (sz > 0) {
        f.read(reinterpret_cast<char*>(buf.data()),
               static_cast<std::streamsize>(sz));
        if (!f) {
            return std::unexpected(Error{ErrorKind::XclbinNotFound,
                                         "read failed: " + path.string()});
        }
    }
    return buf;
}

} // namespace

// ----------------------------------------------------------------------------
// Real-XRT path. Mirrors run_pyxrt_bitnet.py:
//   opcode = 3  (kernel arg 0)
//   bo_i        (kernel arg 1, group_id(1), cacheable)
//   n_insts     (kernel arg 2, scalar uint32)
//   bo_a        (kernel arg 3, group_id(3), host_only)
//   bo_w        (kernel arg 4, group_id(4), host_only)
//   bo_c        (kernel arg 5, group_id(5), host_only)
// ----------------------------------------------------------------------------

struct BitnetGemmAIE2P::Impl {
    xrt::device      device;
    xrt::xclbin      xclbin;
    xrt::hw_context  ctx;
    xrt::kernel      kernel;
    xrt::bo          bo_insts;
    xrt::bo          bo_a;
    xrt::bo          bo_w;
    xrt::bo          bo_c;
    std::uint32_t    n_insts{0};
    std::string      kernel_name;
    bool             ready{false};
};

BitnetGemmAIE2P::BitnetGemmAIE2P() : p_{std::make_unique<Impl>()} {}
BitnetGemmAIE2P::~BitnetGemmAIE2P() = default;
BitnetGemmAIE2P::BitnetGemmAIE2P(BitnetGemmAIE2P&&) noexcept            = default;
BitnetGemmAIE2P& BitnetGemmAIE2P::operator=(BitnetGemmAIE2P&&) noexcept = default;

std::expected<BitnetGemmAIE2P, Error>
BitnetGemmAIE2P::load(const std::filesystem::path& xclbin_path,
                      const std::filesystem::path& insts_path)
{
    auto insts_bytes = read_file(insts_path);
    if (!insts_bytes) return std::unexpected(insts_bytes.error());
    if (insts_bytes->size() % sizeof(std::uint32_t) != 0) {
        return std::unexpected(Error{
            ErrorKind::ShapeMismatch,
            "insts file size not a multiple of 4 bytes: " + insts_path.string()});
    }
    const auto n_insts = static_cast<std::uint32_t>(
        insts_bytes->size() / sizeof(std::uint32_t));
    if (n_insts == 0) {
        return std::unexpected(Error{ErrorKind::ShapeMismatch,
                                     "empty insts file"});
    }

    BitnetGemmAIE2P out;

    // All XRT calls below can throw xrt::system_error / std::runtime_error.
    // Per Rule F we report std::expected, so we wrap in a try/catch and
    // map exceptions onto Error{Xrt}. This is the *only* try/catch in the
    // path; once we're past load() the engine is in a known-good state.
    try {
        out.p_->device = xrt::device{0};

        // xrt::xclbin from filesystem path. The xclbin object owns the
        // parsed axlf; register_xclbin associates it with the device.
        out.p_->xclbin = xrt::xclbin{xclbin_path.string()};
        const auto uuid = out.p_->device.register_xclbin(out.p_->xclbin);

        out.p_->ctx = xrt::hw_context{out.p_->device, uuid};

        // Pull the first kernel name out of the xclbin metadata, matching
        // the python reference's `xclbin.get_kernels()[0].get_name()`.
        const auto kernels = out.p_->xclbin.get_kernels();
        if (kernels.empty()) {
            return std::unexpected(Error{ErrorKind::Xrt,
                                         "xclbin contains no kernels"});
        }
        out.p_->kernel_name = kernels.front().get_name();
        out.p_->kernel = xrt::kernel{out.p_->ctx, out.p_->kernel_name};

        // BO sizing in bytes for the compiled tile.
        constexpr std::size_t kAbytes = a_elems()     * sizeof(std::uint16_t);
        constexpr std::size_t kWbytes = w_elems_u32() * sizeof(std::uint32_t);
        constexpr std::size_t kCbytes = c_elems()     * sizeof(std::uint16_t);

        const std::size_t insts_bytes_sz = insts_bytes->size();

        // xrt::kernel::group_id returns int but xrt::bo wants xrt::memory_group
        // (unsigned int). Cast explicitly to silence -Wnarrowing without
        // changing observed behaviour.
        const auto gid1 = static_cast<xrt::memory_group>(out.p_->kernel.group_id(1));
        const auto gid3 = static_cast<xrt::memory_group>(out.p_->kernel.group_id(3));
        const auto gid4 = static_cast<xrt::memory_group>(out.p_->kernel.group_id(4));
        const auto gid5 = static_cast<xrt::memory_group>(out.p_->kernel.group_id(5));

        out.p_->bo_insts = xrt::bo{out.p_->device, insts_bytes_sz,
                                   xrt::bo::flags::cacheable, gid1};
        out.p_->bo_a = xrt::bo{out.p_->device, kAbytes,
                               xrt::bo::flags::host_only, gid3};
        out.p_->bo_w = xrt::bo{out.p_->device, kWbytes,
                               xrt::bo::flags::host_only, gid4};
        out.p_->bo_c = xrt::bo{out.p_->device, kCbytes,
                               xrt::bo::flags::host_only, gid5};

        // Stage the instruction stream once at load time — kernel keeps
        // re-using the same BO across gemm() calls. We reinterpret the
        // raw byte buffer as uint32 words via memcpy into the BO so we
        // don't violate strict aliasing.
        out.p_->bo_insts.write(insts_bytes->data(), insts_bytes_sz, 0);
        out.p_->bo_insts.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        out.p_->n_insts = n_insts;
        out.p_->ready   = true;
    } catch (const std::exception& e) {
        return std::unexpected(Error{ErrorKind::Xrt, e.what()});
    }

    return out;
}

std::expected<void, Error>
BitnetGemmAIE2P::gemm(std::span<const std::uint16_t> a_bf16,
                      std::span<const std::uint32_t> w_packed,
                      std::span<std::uint16_t>       c_bf16)
{
    if (!p_ || !p_->ready) {
        return std::unexpected(Error{ErrorKind::NotYetWired,
                                     "BitnetGemmAIE2P::gemm on un-ready engine"});
    }

    if (a_bf16.size() != a_elems()) {
        return std::unexpected(Error{
            ErrorKind::ShapeMismatch,
            "a_bf16 length != M*K (" + std::to_string(a_elems()) + ")"});
    }
    if (w_packed.size() != w_elems_u32()) {
        return std::unexpected(Error{
            ErrorKind::ShapeMismatch,
            "w_packed length != K*N/16 (" + std::to_string(w_elems_u32()) + ")"});
    }
    if (c_bf16.size() != c_elems()) {
        return std::unexpected(Error{
            ErrorKind::ShapeMismatch,
            "c_bf16 length != M*N (" + std::to_string(c_elems()) + ")"});
    }

    constexpr std::size_t kAbytes = a_elems()     * sizeof(std::uint16_t);
    constexpr std::size_t kWbytes = w_elems_u32() * sizeof(std::uint32_t);
    constexpr std::size_t kCbytes = c_elems()     * sizeof(std::uint16_t);

    try {
        // H2D: stage activations + weights.
        p_->bo_a.write(a_bf16.data(),   kAbytes, 0);
        p_->bo_w.write(w_packed.data(), kWbytes, 0);
        p_->bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        p_->bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        // bo_insts already synced at load() time.

        // Dispatch. Argument order *must* match the python reference:
        //   (opcode=3, bo_insts, n_insts, bo_a, bo_w, bo_c)
        constexpr std::uint64_t kOpcode = 3ULL;
        auto run = p_->kernel(kOpcode,
                              p_->bo_insts,
                              p_->n_insts,
                              p_->bo_a,
                              p_->bo_w,
                              p_->bo_c);
        run.wait();

        // D2H: pull the result.
        p_->bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        p_->bo_c.read(c_bf16.data(), kCbytes, 0);
    } catch (const std::exception& e) {
        return std::unexpected(Error{ErrorKind::Xrt, e.what()});
    }

    return {};
}

std::string_view BitnetGemmAIE2P::kernel_name() const noexcept {
    return (p_ && p_->ready) ? std::string_view{p_->kernel_name}
                             : std::string_view{};
}

bool BitnetGemmAIE2P::is_ready() const noexcept {
    return p_ && p_->ready;
}

#else  // ONEBIT_AIE_HAVE_XRT_RT == 0

// ----------------------------------------------------------------------------
// No-XRT compile-clean path. Identical ABI; every entry point reports
// LibraryUnavailable. This keeps the static lib linkable on CI hosts.
// ----------------------------------------------------------------------------

struct BitnetGemmAIE2P::Impl {
    bool ready{false};
};

BitnetGemmAIE2P::BitnetGemmAIE2P() : p_{std::make_unique<Impl>()} {}
BitnetGemmAIE2P::~BitnetGemmAIE2P() = default;
BitnetGemmAIE2P::BitnetGemmAIE2P(BitnetGemmAIE2P&&) noexcept            = default;
BitnetGemmAIE2P& BitnetGemmAIE2P::operator=(BitnetGemmAIE2P&&) noexcept = default;

std::expected<BitnetGemmAIE2P, Error>
BitnetGemmAIE2P::load(const std::filesystem::path& xclbin_path,
                      const std::filesystem::path& insts_path)
{
    (void)xclbin_path;
    (void)insts_path;
    return std::unexpected(Error{ErrorKind::LibraryUnavailable,
                                 "onebit::aie built without XRT linkage"});
}

std::expected<void, Error>
BitnetGemmAIE2P::gemm(std::span<const std::uint16_t> a_bf16,
                      std::span<const std::uint32_t> w_packed,
                      std::span<std::uint16_t>       c_bf16)
{
    (void)a_bf16; (void)w_packed; (void)c_bf16;
    return std::unexpected(Error{ErrorKind::LibraryUnavailable,
                                 "onebit::aie built without XRT linkage"});
}

std::string_view BitnetGemmAIE2P::kernel_name() const noexcept { return {}; }
bool             BitnetGemmAIE2P::is_ready()    const noexcept { return false; }

#endif // ONEBIT_AIE_HAVE_XRT_RT

} // namespace onebit::aie

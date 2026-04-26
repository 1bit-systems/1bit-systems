// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 halo-ai-core / bong-water-water-bong.
//
// onebit::aie::BitnetGemmAIE2P implementation.
//
// Direct-link libxrt path. We compile this TU only when ONEBIT_AIE_HAVE_XRT_RT
// is defined; otherwise every entry point returns LibraryUnavailable so
// the rest of the tower can build clean on hosts without xrt installed.

#include "onebit/aie/bitnet_gemm_aie2p.hpp"

#include <cmath>
#include <cstdint>
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

// Solve the square tile size T from the W (HALO_V2 packed) arg byte
// count. HALO_V2 packs 16 ternary codes per uint32 (4 bytes), so each
// W byte holds 4 codes. For a TxT weight matrix:
//   total_codes = T*T
//   W_bytes     = total_codes / 4 = T*T / 4
// Inverse:
//   T = sqrt(W_bytes * 4)
//
// Sanity check on the live xclbins (16 codes/u32 invariant):
//   T=64  -> W_bytes = 64*64/4 = 1024
//   T=512 -> W_bytes = 512*512/4 = 65536
//
// Returns 0 when w_bytes is zero, the implied T isn't a positive integer
// (round-trip square fails), or T isn't a multiple of 16 (HALO_V2 word
// boundary requirement — partial words cross uint32 lines).
[[nodiscard]] std::uint32_t solve_tile_from_w_bytes(std::uint64_t w_bytes) noexcept
{
    if (w_bytes == 0) return 0;
    const std::uint64_t codes = w_bytes * 4ULL;
    const auto t_d = std::sqrt(static_cast<double>(codes));
    const auto t   = static_cast<std::uint32_t>(t_d + 0.5);
    if (t == 0) return 0;
    if (static_cast<std::uint64_t>(t) * t != codes) return 0;
    if ((t % kBitnetGemmAIE2P_CodesPerU32) != 0) return 0;
    return t;
}

// Filename-string fallback solver for hosts where xrt::xclbin::arg::get_size()
// returns 0 for the BO args (some XRT-AIE backends report only the scalar
// args). Looks for a "<NNN>x<NNN>x<NNN>" triple anywhere in the path stem
// and pulls the first dim. Returns 0 when no triple matches.
[[nodiscard]] std::uint32_t solve_tile_from_filename(std::string_view path) noexcept
{
    // Scan left-to-right for runs of "<digits>x<digits>x<digits>".
    auto is_digit = [](char c) noexcept { return c >= '0' && c <= '9'; };
    for (std::size_t i = 0; i + 4 < path.size(); ++i) {
        if (!is_digit(path[i])) continue;
        std::size_t j = i;
        std::uint64_t a = 0;
        while (j < path.size() && is_digit(path[j])) {
            a = a * 10 + static_cast<std::uint64_t>(path[j] - '0');
            ++j;
        }
        if (j >= path.size() || path[j] != 'x') continue;
        ++j;
        std::size_t k0 = j;
        std::uint64_t b = 0;
        while (j < path.size() && is_digit(path[j])) {
            b = b * 10 + static_cast<std::uint64_t>(path[j] - '0');
            ++j;
        }
        if (j == k0) continue;
        if (j >= path.size() || path[j] != 'x') continue;
        ++j;
        std::size_t l0 = j;
        std::uint64_t c = 0;
        while (j < path.size() && is_digit(path[j])) {
            c = c * 10 + static_cast<std::uint64_t>(path[j] - '0');
            ++j;
        }
        if (j == l0) continue;
        // Require all three dims equal AND a HALO_V2-legal multiple of 16.
        if (a == b && b == c && a > 0 && (a % kBitnetGemmAIE2P_CodesPerU32) == 0) {
            return static_cast<std::uint32_t>(a);
        }
        // Not a match: continue scanning past this run.
        i = j; // outer loop's ++i still fires; harmless.
    }
    return 0;
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
    // Tile dims read out of the xclbin at load(); see probe_tile_dim()
    // below for the heuristic. Zero on un-ready handle.
    std::uint32_t    tile_m{0};
    std::uint32_t    tile_k{0};
    std::uint32_t    tile_n{0};
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

        // ----- Tile-dim probe ---------------------------------------------
        // Strategy:
        //   1) Ask the xclbin for the W-arg byte size (kernel arg index 4).
        //      If it reports a sane value, T = sqrt(W_bytes * 4) with the
        //      square round-trip checked. AIE/IPU xclbins on some XRT
        //      versions report a scalar host-type size for global BO args,
        //      in which case (1) returns 0 and we fall through to (2).
        //   2) Filename heuristic: scan path stem for "<NNN>x<NNN>x<NNN>"
        //      with all three dims equal and a multiple of 16. This is what
        //      the build emits today (final_<T>x<T>x<T>_64x64x64.xclbin and
        //      final_<T>_<T>x<T>x<T>_64x64x64.xclbin).
        //   3) Both failed -> fail load() loudly. We do NOT silently fall
        //      back to the compile-time default — a wrong tile would
        //      mis-size the BOs and produce garbage at gemm() time.
        std::uint32_t tile_dim = 0;
        const auto kargs = kernels.front().get_args();
        if (kargs.size() > 4) {
            const auto w_bytes = static_cast<std::uint64_t>(kargs[4].get_size());
            tile_dim = solve_tile_from_w_bytes(w_bytes);
        }
        if (tile_dim == 0) {
            tile_dim = solve_tile_from_filename(xclbin_path.string());
        }
        if (tile_dim == 0) {
            return std::unexpected(Error{ErrorKind::ShapeMismatch,
                "could not infer tile dim from xclbin (W-arg get_size and "
                "filename heuristic both failed): " + xclbin_path.string()});
        }
        out.p_->tile_m = tile_dim;
        out.p_->tile_k = tile_dim;
        out.p_->tile_n = tile_dim;

        // BO sizing in bytes for the loaded tile.
        const std::size_t kAbytes =
            static_cast<std::size_t>(tile_dim) * tile_dim * sizeof(std::uint16_t);
        const std::size_t kWbytes =
            static_cast<std::size_t>(tile_dim) * tile_dim / kBitnetGemmAIE2P_CodesPerU32
            * sizeof(std::uint32_t);
        const std::size_t kCbytes =
            static_cast<std::size_t>(tile_dim) * tile_dim * sizeof(std::uint16_t);

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

    // Per-instance expected sizes from the loaded xclbin's tile dims.
    const std::size_t a_expected = static_cast<std::size_t>(p_->tile_m) * p_->tile_k;
    const std::size_t w_expected =
        (static_cast<std::size_t>(p_->tile_k) * p_->tile_n) /
        kBitnetGemmAIE2P_CodesPerU32;
    const std::size_t c_expected = static_cast<std::size_t>(p_->tile_m) * p_->tile_n;

    if (a_bf16.size() != a_expected) {
        return std::unexpected(Error{
            ErrorKind::ShapeMismatch,
            "a_bf16 length != M*K (" + std::to_string(a_expected) + ")"});
    }
    if (w_packed.size() != w_expected) {
        return std::unexpected(Error{
            ErrorKind::ShapeMismatch,
            "w_packed length != K*N/16 (" + std::to_string(w_expected) + ")"});
    }
    if (c_bf16.size() != c_expected) {
        return std::unexpected(Error{
            ErrorKind::ShapeMismatch,
            "c_bf16 length != M*N (" + std::to_string(c_expected) + ")"});
    }

    const std::size_t kAbytes = a_expected * sizeof(std::uint16_t);
    const std::size_t kWbytes = w_expected * sizeof(std::uint32_t);
    const std::size_t kCbytes = c_expected * sizeof(std::uint16_t);

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

std::uint32_t BitnetGemmAIE2P::loaded_tile_m() const noexcept {
    return (p_ && p_->ready) ? p_->tile_m : 0u;
}
std::uint32_t BitnetGemmAIE2P::loaded_tile_k() const noexcept {
    return (p_ && p_->ready) ? p_->tile_k : 0u;
}
std::uint32_t BitnetGemmAIE2P::loaded_tile_n() const noexcept {
    return (p_ && p_->ready) ? p_->tile_n : 0u;
}
std::size_t BitnetGemmAIE2P::loaded_a_elems() const noexcept {
    return (p_ && p_->ready)
        ? static_cast<std::size_t>(p_->tile_m) * p_->tile_k
        : 0u;
}
std::size_t BitnetGemmAIE2P::loaded_w_elems_u32() const noexcept {
    return (p_ && p_->ready)
        ? (static_cast<std::size_t>(p_->tile_k) * p_->tile_n) /
          kBitnetGemmAIE2P_CodesPerU32
        : 0u;
}
std::size_t BitnetGemmAIE2P::loaded_c_elems() const noexcept {
    return (p_ && p_->ready)
        ? static_cast<std::size_t>(p_->tile_m) * p_->tile_n
        : 0u;
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

std::uint32_t BitnetGemmAIE2P::loaded_tile_m() const noexcept { return 0u; }
std::uint32_t BitnetGemmAIE2P::loaded_tile_k() const noexcept { return 0u; }
std::uint32_t BitnetGemmAIE2P::loaded_tile_n() const noexcept { return 0u; }
std::size_t   BitnetGemmAIE2P::loaded_a_elems()     const noexcept { return 0u; }
std::size_t   BitnetGemmAIE2P::loaded_w_elems_u32() const noexcept { return 0u; }
std::size_t   BitnetGemmAIE2P::loaded_c_elems()     const noexcept { return 0u; }

#endif // ONEBIT_AIE_HAVE_XRT_RT

} // namespace onebit::aie

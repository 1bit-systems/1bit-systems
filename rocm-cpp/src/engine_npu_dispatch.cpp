// engine_npu_dispatch.cpp — C++23 island bridging Engine::ternary_gemv to
// onebit::aie::tiled_gemv (which itself sits on top of BitnetGemmAIE2P).
//
// Compiled with cxx_std_23 + linked against onebit::aie_bitnet_gemm and
// onebit::aie_tiled_gemv.  The only HIP call here is hipMemcpy through
// the runtime API.  engine.cpp (C++17) consumes this TU through
// engine_npu_dispatch.h.
//
// Two correctness-critical transforms happen on the host:
//   1) ENCODING swap.  rocm-cpp's HALO_V2 packed weights use
//        bits=0 → -1, bits=1 → 0, bits=2 → +1
//      (see kernels/ternary_gemv_phase5_halo.hip:35 unpack_ternary_halo_4
//      and the device-side `(bits==2) - (bits==0)` derivation).  The
//      AIE2P xclbin and its pack_halo_v2 reference use a DIFFERENT
//      encoding:
//        bits=00 → 0, bits=01 → +1, bits=10 → -1
//      so the two-bit code is NOT a re-interpretation; we LUT every
//      code on the host before submission.  Skipping this step yields
//      garbage outputs (the kernel still runs without error).
//   2) Per-row W scale fold.  The AIE2P kernel emits unscaled bf16
//      partials.  We multiply by the engine's `row_scales[N]` post-mmul
//      on the host before downcast to fp16.  The activation scale
//      (`x_scale_dev`) is NOT used — the AIE path consumes fp16 acts
//      directly off RMSNorm, mirroring the SHERRY_FP16 lane.
//
// See engine_npu_dispatch.h for the full Phase-2 contract notes.

#include "engine_npu_dispatch.h"

#include "onebit/aie/bitnet_gemm_aie2p.hpp"
#include "onebit/aie/error.hpp"
#include "onebit/aie/tiled_gemv.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace {

// fp16 -> fp32 -> bf16 (round-to-nearest-even).  fp16 and bf16 do NOT
// share an exponent layout, so we round-trip through fp32.  Same biasing
// trick as cpp/aie/tests/bitnet_gemm_aie2p_test.cpp's fp32_to_bf16().
//
// Precision floor: bf16 has 7 mantissa bits vs fp16's 10, so this cast
// drops ~3 bits of precision; expect rel_err ~1e-3 on downstream math
// (the AIE2P tile-test threshold is 5e-3).
[[nodiscard]] inline std::uint16_t fp16_to_bf16(_Float16 h) noexcept {
    const float f = static_cast<float>(h);
    const std::uint32_t u   = std::bit_cast<std::uint32_t>(f);
    const std::uint32_t lsb = (u >> 16) & 1u;
    const std::uint32_t bias = 0x7FFFu + lsb;
    const std::uint32_t rnd  = u + bias;
    return static_cast<std::uint16_t>(rnd >> 16);
}

[[nodiscard]] inline _Float16 bf16_to_fp16(std::uint16_t b) noexcept {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
    const float f = std::bit_cast<float>(u);
    return static_cast<_Float16>(f);
}

// HIP-halo -> AIE-halo_v2 code remap.  Per-2-bit lookup:
//   HIP 0b00 (-1) -> AIE 0b10
//   HIP 0b01 ( 0) -> AIE 0b00
//   HIP 0b10 (+1) -> AIE 0b01
//   HIP 0b11 (defensive 0) -> AIE 0b00
constexpr std::array<std::uint8_t, 4> kCodeRemap = {0b10, 0b00, 0b01, 0b00};

// Re-encode 16 codes per u32 from HIP-halo to AIE-halo_v2.
[[nodiscard]] inline std::uint32_t remap_word(std::uint32_t hip_word) noexcept {
    std::uint32_t out = 0u;
    for (std::uint32_t slot = 0; slot < 16u; ++slot) {
        const std::uint32_t code = (hip_word >> (2u * slot)) & 0b11u;
        out |= static_cast<std::uint32_t>(kCodeRemap[code]) << (2u * slot);
    }
    return out;
}

inline void copy_error(char* buf, std::size_t cap, std::string_view sv) noexcept {
    if (!buf || cap == 0) return;
    const std::size_t n = std::min(sv.size(), cap - 1);
    std::memcpy(buf, sv.data(), n);
    buf[n] = '\0';
}

} // namespace

// ---------------------------------------------------------------------------
// Handle.  Owns the AIE2P engine plus host scratch reused across dispatches.
// Per-call buffers are resized lazily on first use (and grow only) — the
// engine reuses the same shapes across all decode tokens after warmup.
// ---------------------------------------------------------------------------

struct rcpp_npu_dispatch_t {
    onebit::aie::BitnetGemmAIE2P eng;

    // Host mirrors of device buffers.  Sized to (current_N, current_K).
    std::vector<_Float16>      normed_host;     // fp16 [K]
    std::vector<std::uint16_t> a_bf16_host;     // bf16 [K]
    std::vector<std::uint32_t> w_packed_host;   // u32  [N*K/16] (re-encoded)
    std::vector<float>         row_scales_host; // fp32 [N]
    std::vector<std::uint16_t> c_bf16_host;     // bf16 [N]
    std::vector<_Float16>      out_fp16_host;   // fp16 [N]

    std::string last_error;

    explicit rcpp_npu_dispatch_t(onebit::aie::BitnetGemmAIE2P&& e) noexcept
        : eng{std::move(e)} {}
};

extern "C" rcpp_npu_status_t
rcpp_npu_dispatch_create(const char*  xclbin_path,
                         const char*  insts_path,
                         char*        error_buf,
                         std::size_t  error_buf_size,
                         rcpp_npu_dispatch_t** out)
{
    if (!out) return RCPP_NPU_LOAD_FAILED;
    *out = nullptr;

    if (!xclbin_path || !insts_path) {
        copy_error(error_buf, error_buf_size, "null xclbin/insts path");
        return RCPP_NPU_LOAD_FAILED;
    }

    auto eng_or = onebit::aie::BitnetGemmAIE2P::load(xclbin_path, insts_path);
    if (!eng_or) {
        const auto& err = eng_or.error();
        const std::string msg =
            std::string{"BitnetGemmAIE2P::load failed: kind="} +
            std::string{onebit::aie::label(err.kind())} +
            " detail=" + std::string{err.detail()};
        copy_error(error_buf, error_buf_size, msg);
        return RCPP_NPU_LOAD_FAILED;
    }

    *out = new rcpp_npu_dispatch_t{std::move(*eng_or)};
    return RCPP_NPU_OK;
}

extern "C" void
rcpp_npu_dispatch_free(rcpp_npu_dispatch_t* h) { delete h; }

extern "C" const char*
rcpp_npu_dispatch_last_error(rcpp_npu_dispatch_t* h) {
    return h ? h->last_error.c_str() : "";
}

extern "C" rcpp_npu_status_t
rcpp_npu_dispatch_gemv(rcpp_npu_dispatch_t* h,
                       const void* packed_dev,
                       const float* row_scales_dev,
                       const void* normed_fp16_dev,
                       void*       out_fp16_dev,
                       int N, int K)
{
    if (!h) return RCPP_NPU_NOT_INITIALIZED;
    h->last_error.clear();

    const std::uint32_t tile = onebit::aie::BitnetGemmAIE2P::tile_n(); // 64 today
    if (N <= 0 || K <= 0 ||
        (static_cast<std::uint32_t>(N) % tile) != 0 ||
        (static_cast<std::uint32_t>(K) % tile) != 0)
    {
        h->last_error = "N or K not a multiple of tile (" +
                        std::to_string(tile) + ")";
        return RCPP_NPU_SHAPE_MISMATCH;
    }

    const std::size_t N_sz = static_cast<std::size_t>(N);
    const std::size_t K_sz = static_cast<std::size_t>(K);
    const std::size_t W_words = (N_sz * K_sz) / 16u;

    // Lazy resize host scratch.
    if (h->normed_host.size()     < K_sz)        h->normed_host.assign(K_sz, _Float16{0.0f});
    if (h->a_bf16_host.size()     < K_sz)        h->a_bf16_host.assign(K_sz, 0u);
    if (h->w_packed_host.size()   < W_words)     h->w_packed_host.assign(W_words, 0u);
    if (h->row_scales_host.size() < N_sz)        h->row_scales_host.assign(N_sz, 0.0f);
    if (h->c_bf16_host.size()     < N_sz)        h->c_bf16_host.assign(N_sz, 0u);
    if (h->out_fp16_host.size()   < N_sz)        h->out_fp16_host.assign(N_sz, _Float16{0.0f});

    // --- D->H copies ---
    if (auto e = hipMemcpy(h->normed_host.data(), normed_fp16_dev,
                           K_sz * sizeof(_Float16), hipMemcpyDeviceToHost);
        e != hipSuccess) {
        h->last_error = std::string{"hipMemcpy(normed) failed: "} +
                        hipGetErrorString(e);
        return RCPP_NPU_HIP_ERROR;
    }
    if (auto e = hipMemcpy(h->w_packed_host.data(), packed_dev,
                           W_words * sizeof(std::uint32_t),
                           hipMemcpyDeviceToHost);
        e != hipSuccess) {
        h->last_error = std::string{"hipMemcpy(packed) failed: "} +
                        hipGetErrorString(e);
        return RCPP_NPU_HIP_ERROR;
    }
    if (auto e = hipMemcpy(h->row_scales_host.data(), row_scales_dev,
                           N_sz * sizeof(float), hipMemcpyDeviceToHost);
        e != hipSuccess) {
        h->last_error = std::string{"hipMemcpy(row_scales) failed: "} +
                        hipGetErrorString(e);
        return RCPP_NPU_HIP_ERROR;
    }

    // --- fp16 -> bf16 cast on the activation row.  No SIMD here; K is at
    // most ~7K elements per call and this is dwarfed by the H<->D copies.
    for (std::size_t k = 0; k < K_sz; ++k) {
        h->a_bf16_host[k] = fp16_to_bf16(h->normed_host[k]);
    }

    // --- Re-encode the weight buffer in place from HIP-halo to AIE-halo_v2.
    // We DON'T mutate the device buffer — only this host mirror.  Cost is
    // 16 LUT-lookups per u32; with W_words ~= 11M for halo-1bit-2b's lm_head
    // this is ~5ms on a single core, but lm_head gates OUT (V%64 != 0) so
    // no production layer hits this size.  Layers we do hit have W_words
    // ~= 25K-700K; sub-millisecond.
    for (std::size_t i = 0; i < W_words; ++i) {
        h->w_packed_host[i] = remap_word(h->w_packed_host[i]);
    }

    // --- Tiled GEMV through the AIE library helper.  Same call the
    // sibling tiled_gemv_test.cpp uses; we just plumb our host buffers
    // into spans.
    onebit::aie::TiledGemvCfg cfg{
        .n_total = N,
        .k_total = K,
        .tile    = static_cast<int>(tile),
    };

    auto rc = onebit::aie::tiled_gemv(
        h->eng,
        std::span<const std::uint16_t>{h->a_bf16_host.data(), K_sz},
        std::span<const std::uint32_t>{h->w_packed_host.data(), W_words},
        std::span<std::uint16_t>{h->c_bf16_host.data(), N_sz},
        cfg);
    if (!rc) {
        const auto& err = rc.error();
        h->last_error = std::string{"tiled_gemv failed: kind="} +
                        std::string{onebit::aie::label(err.kind())} +
                        " detail=" + std::string{err.detail()};
        return RCPP_NPU_DISPATCH_FAILED;
    }

    // --- Apply per-row W scale post-mmul, downcast to fp16.
    for (std::size_t n = 0; n < N_sz; ++n) {
        const std::uint32_t bf  = static_cast<std::uint32_t>(h->c_bf16_host[n]) << 16;
        const float         f   = std::bit_cast<float>(bf);
        const float         scaled = f * h->row_scales_host[n];
        h->out_fp16_host[n] = static_cast<_Float16>(scaled);
    }

    // --- H->D writeback.
    if (auto e = hipMemcpy(out_fp16_dev, h->out_fp16_host.data(),
                           N_sz * sizeof(_Float16), hipMemcpyHostToDevice);
        e != hipSuccess) {
        h->last_error = std::string{"hipMemcpy(out) failed: "} +
                        hipGetErrorString(e);
        return RCPP_NPU_HIP_ERROR;
    }

    return RCPP_NPU_OK;
}

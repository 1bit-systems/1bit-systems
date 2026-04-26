// SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
// Sherry — see LICENSE-SHERRY.md and SHERRY-FILES.txt at the repo root.
// Commercial use requires a separate license.
//
// sherry_per_row_probe.cpp — diagnostic CPU oracle for Sherry vs HALO_V2
// per-row dot products. One-shot tool: NOT linked into the runtime path.
//
// Purpose: localize residual PPL gap between v2 baseline (PPL 12.13) and
// Sherry v3 (PPL 8131 as of 2026-04-26). Computes for layer 0 q_proj:
//
//   ref[r]    = sum_k a[k] * w_v2[r,k] * scales_v2[r]
//   sherry[r] = sum_k a[k] * w_sh[r,k] * scales_sh[r]
//   rel_err[r] = |sherry[r] - ref[r]| / max(|ref[r]|, eps)
//
// Output: per-row JSON to stdout. We compare:
//   (a) mean / median / p99 rel_err — uniform vs row-varying tells us
//       whether per-row scale is wrong (uniform) or sign distribution
//       is wrong (high variance).
//   (b) ratio scales_sh[r] / scales_v2[r] — whether the L1 rescale is
//       consistent or jittery per row.
//   (c) magnitude of |ref| and |sherry| — saturation check.
//
// Activation vector: deterministic standard-normal via xorshift64 (seed
// fixed) so probe is reproducible across runs. Length = hs (model hidden
// size), read from the v2 file header.
//
// Build via tools/CMakeLists addition (target sherry_per_row_probe).

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

static constexpr int8_t kHaloCodeToTernary[4] = {-1, 0, +1, 0};

bool read_all(const std::string& path, std::vector<uint8_t>& buf) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const std::streamsize n = f.tellg();
    if (n < 0) return false;
    buf.resize(static_cast<size_t>(n));
    f.seekg(0, std::ios::beg);
    f.read(reinterpret_cast<char*>(buf.data()), n);
    return static_cast<bool>(f);
}

template<typename T>
T read_scalar_le(const uint8_t* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

void halo_unpack_row(const uint8_t* row, int cols, int8_t* out) {
    const int row_bytes = (cols + 3) / 4;
    for (int b = 0; b < row_bytes; ++b) {
        const uint8_t byte = row[b];
        for (int slot = 0; slot < 4; ++slot) {
            const int k = b * 4 + slot;
            if (k >= cols) break;
            out[k] = kHaloCodeToTernary[(byte >> (slot * 2)) & 0x3];
        }
    }
}

// Sherry decode (matches kernel LUT exactly).
void sherry_decode_row(const uint8_t* packed, int cols, int8_t* out) {
    const int macrogroups = cols / 32;
    for (int mg = 0; mg < macrogroups; ++mg) {
        uint64_t bits40 = 0;
        for (int b = 0; b < 5; ++b) {
            bits40 |= static_cast<uint64_t>(packed[mg * 5 + b]) << (8 * b);
        }
        for (int sg = 0; sg < 8; ++sg) {
            const uint32_t code = static_cast<uint32_t>((bits40 >> (5 * sg)) & 0x1F);
            const uint32_t zp    = (code >> 3) & 0x3u;
            const uint32_t signs =  code       & 0x7u;
            uint32_t si = 0;
            for (int p = 0; p < 4; ++p) {
                int8_t q;
                if (static_cast<uint32_t>(p) == zp) {
                    q = 0;
                } else {
                    q = ((signs >> si) & 1u) ? +1 : -1;
                    ++si;
                }
                out[mg * 32 + sg * 4 + p] = q;
            }
        }
    }
}

uint64_t xorshift64(uint64_t& s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

float gauss_from_uniform(float u1, float u2) {
    // Box-Muller. u1,u2 in (0,1).
    if (u1 < 1e-7f) u1 = 1e-7f;
    const float r = std::sqrt(-2.0f * std::log(u1));
    const float t = 6.28318530718f * u2;
    return r * std::cos(t);
}

void make_activations(int n, std::vector<float>& out) {
    out.assign(n, 0.0f);
    uint64_t s = 0xC0FFEEDEADBEEFULL;
    for (int i = 0; i < n; i += 2) {
        const uint64_t a = xorshift64(s);
        const uint64_t b = xorshift64(s);
        const float u1 = static_cast<float>((a >> 11) & 0x1FFFFFu) / 2097151.0f;
        const float u2 = static_cast<float>((b >> 11) & 0x1FFFFFu) / 2097151.0f;
        const float g0 = gauss_from_uniform(u1 + 1e-9f, u2);
        const float g1 = gauss_from_uniform(u2 + 1e-9f, u1);
        out[i]                            = g0;
        if (i + 1 < n) out[i + 1]         = g1;
    }
}

struct LayerOffsets {
    size_t v2_layer_norm_block;     // offset to start of layer i in v2
    size_t v3_layer_norm_block;     // offset to start of layer i in v3
};

// Walk to layer 0's q_proj weight + scales pair in both files. Mirrors
// the reader logic in h1b_repack_sherry::do_repack.
struct ModelView {
    int hs, is_, L, nh, nkv, V, hd;
    int rows_q;            // = nh * hd
    int cols_q;            // = hs
    size_t v2_q_weights;   // offset
    size_t v2_q_scales;
    size_t v3_q_weights;
    size_t v3_q_scales;
};

bool open_view(const std::vector<uint8_t>& v2,
               const std::vector<uint8_t>& v3,
               ModelView& out)
{
    if (v2.size() < 8 + 36 || v3.size() < 8 + 36) return false;
    if (std::memcmp(v2.data(), "H1B\x00", 4) != 0) return false;
    if (std::memcmp(v3.data(), "H1B\x00", 4) != 0) return false;
    const int32_t v2_ver = read_scalar_le<int32_t>(v2.data() + 4);
    const int32_t v3_ver = read_scalar_le<int32_t>(v3.data() + 4);
    if (v2_ver != 2) { std::fprintf(stderr, "v2 must be HALO_V2 (got v%d)\n", v2_ver); return false; }
    if (v3_ver != 3) { std::fprintf(stderr, "v3 must be Sherry v3 (got v%d)\n", v3_ver); return false; }

    int32_t cfg[9];
    std::memcpy(cfg, v2.data() + 8, sizeof(cfg));
    out.hs  = cfg[0]; out.is_ = cfg[1]; out.L   = cfg[2];
    out.nh  = cfg[3]; out.nkv = cfg[4]; out.V   = cfg[5];
    out.hd  = out.hs / out.nh;
    out.rows_q = out.nh * out.hd;
    out.cols_q = out.hs;

    // v2 layout: H1B(4) + ver(4) + cfg(36) + rope+rms(8) + emb + final_norm + per-layer
    const size_t hdr_v2 = 4 + 4 + 36 + 8;
    const size_t hdr_v3 = hdr_v2;  // same layout
    const size_t emb     = static_cast<size_t>(out.V) * static_cast<size_t>(out.hs) * 4;
    const size_t final_norm = static_cast<size_t>(out.hs) * 4;

    // Layer 0 norm block bytes (matches encoder).
    const size_t norm_block =
        static_cast<size_t>(out.hs) * 4 * (1 + 1 + 4 + 2)
        + static_cast<size_t>(out.is_) * 4;

    // q is the FIRST tensor in the per-layer block. Weight bytes:
    const int row_bytes_v2 = (out.cols_q + 3) / 4;
    const size_t q_w_v2 = static_cast<size_t>(out.rows_q) * static_cast<size_t>(row_bytes_v2);
    const size_t q_s_v2 = static_cast<size_t>(out.rows_q) * 4;

    const size_t row_bytes_v3 = static_cast<size_t>(out.cols_q) * 5 / 32;
    const size_t q_w_v3 = static_cast<size_t>(out.rows_q) * row_bytes_v3;
    const size_t q_s_v3 = static_cast<size_t>(out.rows_q) * 4;

    out.v2_q_weights = hdr_v2 + emb + final_norm + norm_block;
    out.v2_q_scales  = out.v2_q_weights + q_w_v2;
    out.v3_q_weights = hdr_v3 + emb + final_norm + norm_block;
    out.v3_q_scales  = out.v3_q_weights + q_w_v3;

    if (out.v2_q_scales + q_s_v2 > v2.size()) return false;
    if (out.v3_q_scales + q_s_v3 > v3.size()) return false;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string v2_path, v3_path;
    int probe_n_rows = 32;
    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        if      (a == "--v2"   && i + 1 < argc) v2_path = argv[++i];
        else if (a == "--v3"   && i + 1 < argc) v3_path = argv[++i];
        else if (a == "--rows" && i + 1 < argc) probe_n_rows = std::atoi(argv[++i]);
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (v2_path.empty() || v3_path.empty()) {
        std::fprintf(stderr, "usage: sherry_per_row_probe --v2 halo-1bit-2b.h1b --v3 halo-1bit-2b-sherry-cpp.h1b [--rows N]\n");
        return 2;
    }

    std::vector<uint8_t> v2, v3;
    if (!read_all(v2_path, v2)) { std::fprintf(stderr, "cannot open %s\n", v2_path.c_str()); return 1; }
    if (!read_all(v3_path, v3)) { std::fprintf(stderr, "cannot open %s\n", v3_path.c_str()); return 1; }

    ModelView mv{};
    if (!open_view(v2, v3, mv)) { std::fprintf(stderr, "header parse failed\n"); return 1; }
    std::fprintf(stderr, "[probe] hs=%d is=%d L=%d nh=%d nkv=%d hd=%d rows_q=%d cols_q=%d\n",
                 mv.hs, mv.is_, mv.L, mv.nh, mv.nkv, mv.hd, mv.rows_q, mv.cols_q);

    std::vector<float> act;
    make_activations(mv.cols_q, act);

    const int rows_to_probe = probe_n_rows < mv.rows_q ? probe_n_rows : mv.rows_q;
    const int row_bytes_v2  = (mv.cols_q + 3) / 4;
    const size_t row_bytes_v3 = static_cast<size_t>(mv.cols_q) * 5 / 32;

    const float* scales_v2 =
        reinterpret_cast<const float*>(v2.data() + mv.v2_q_scales);
    const float* scales_v3 =
        reinterpret_cast<const float*>(v3.data() + mv.v3_q_scales);

    std::vector<int8_t> w_v2(mv.cols_q), w_sh(mv.cols_q);

    // Aggregate stats.
    double sum_relerr = 0.0, sum_relerr2 = 0.0;
    double max_relerr = 0.0;
    int n_rows_used = 0;

    std::printf("{\n  \"rows\": [\n");
    for (int r = 0; r < rows_to_probe; ++r) {
        halo_unpack_row(v2.data() + mv.v2_q_weights + static_cast<size_t>(r) * row_bytes_v2,
                        mv.cols_q, w_v2.data());
        sherry_decode_row(v3.data() + mv.v3_q_weights + static_cast<size_t>(r) * row_bytes_v3,
                          mv.cols_q, w_sh.data());

        // Reference: pure v2 dot, scaled.
        double ref_unscaled = 0.0;
        int l1_v2 = 0;
        for (int k = 0; k < mv.cols_q; ++k) {
            ref_unscaled += static_cast<double>(act[k]) * static_cast<double>(w_v2[k]);
            if (w_v2[k] != 0) ++l1_v2;
        }
        const double ref = ref_unscaled * static_cast<double>(scales_v2[r]);

        // Sherry: 3:4 sparse dot, scaled.
        double sh_unscaled = 0.0;
        int l1_sh = 0;          // always 3 * groups
        int sign_disagree = 0;  // count of k where sign(w_v2)!=sign(w_sh)
        int phantom_pos = 0;    // count of k where w_v2==0 && w_sh!=0
        int dropped = 0;        // count of k where w_v2!=0 && w_sh==0
        for (int k = 0; k < mv.cols_q; ++k) {
            sh_unscaled += static_cast<double>(act[k]) * static_cast<double>(w_sh[k]);
            if (w_sh[k] != 0) ++l1_sh;
            if (w_v2[k] != 0 && w_sh[k] == 0) ++dropped;
            if (w_v2[k] == 0 && w_sh[k] != 0) ++phantom_pos;
            if (w_v2[k] != 0 && w_sh[k] != 0 && w_v2[k] != w_sh[k]) ++sign_disagree;
        }
        const double sh = sh_unscaled * static_cast<double>(scales_v3[r]);

        const double abs_err = std::abs(sh - ref);
        const double denom   = std::max(std::abs(ref), 1e-9);
        const double relerr  = abs_err / denom;
        sum_relerr  += relerr;
        sum_relerr2 += relerr * relerr;
        if (relerr > max_relerr) max_relerr = relerr;
        ++n_rows_used;

        const double scale_ratio = static_cast<double>(scales_v3[r]) / static_cast<double>(scales_v2[r]);
        const double l1_ratio_expected = static_cast<double>(l1_v2) / static_cast<double>(l1_sh);

        std::printf("    {\"r\":%d,\"ref\":%.6e,\"sh\":%.6e,\"abs_err\":%.6e,\"rel_err\":%.6e,"
                    "\"scales_v2\":%.6e,\"scales_v3\":%.6e,\"scale_ratio\":%.6e,"
                    "\"l1_v2\":%d,\"l1_sh\":%d,\"l1_ratio_expected\":%.6f,"
                    "\"dropped\":%d,\"phantom_pos\":%d,\"sign_disagree\":%d}%s\n",
                    r, ref, sh, abs_err, relerr,
                    static_cast<double>(scales_v2[r]),
                    static_cast<double>(scales_v3[r]),
                    scale_ratio,
                    l1_v2, l1_sh, l1_ratio_expected,
                    dropped, phantom_pos, sign_disagree,
                    (r + 1 == rows_to_probe) ? "" : ",");
    }

    const double mean_rel = sum_relerr / static_cast<double>(n_rows_used);
    const double var_rel  = sum_relerr2 / static_cast<double>(n_rows_used) - mean_rel * mean_rel;
    std::printf("  ],\n");
    std::printf("  \"summary\": {\"n_rows\":%d,\"mean_rel_err\":%.6f,\"std_rel_err\":%.6f,\"max_rel_err\":%.6f}\n",
                n_rows_used, mean_rel, std::sqrt(var_rel > 0.0 ? var_rel : 0.0), max_relerr);
    std::printf("}\n");
    return 0;
}

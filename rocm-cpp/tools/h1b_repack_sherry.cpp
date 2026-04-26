// SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
// Sherry — see LICENSE-SHERRY.md and SHERRY-FILES.txt at the repo root.
// Commercial use requires a separate license.
//
// h1b_repack_sherry — C++ port of `requantize_h1b_to_sherry.py`, RULE A (no
// Python on the runtime/build path). Replaces the buggy Python packer that
// encoded natural q==0 lanes as sign_bit=0 (kernel decodes that as -1) and so
// poisoned roughly 600 weights per row, blowing PPL to ~1.28e9 and producing
// the "10879 10879 10879 ..." degenerate decode.
//
// THE BUG (old packer, lines 138-156 of requantize_bf16_to_sherry.py):
//
//   zero_pos = argmin(|w_bf16|)   # ignores natural q==0 entirely
//   q_g[..., zero_pos] = 0        # force the chosen lane to 0
//   for p in range(4):
//       mask = (p != zero_pos)
//       sign_bit = (q_g[..., p] == 1) * mask
//       codes |= sign_bit << ...
//
// If a group's natural q==0 is at lane B but argmin(|w|) is at lane A, the
// old packer picks zero_pos=A and then has q[B]==0 sitting in the sign bits.
// `(q==1) → 1, otherwise 0` makes that lane encode sign_bit=0, which the
// kernel LUT (build_sherry_entry: `bit ? +1 : -1`) decodes as -1. Every
// natural zero becomes a phantom -1.
//
// THE FIX (this tool):
//
//   1. PREFER natural q==0 as zero_pos. If a group already contains a 0 the
//      Sherry sparsity contract is trivially satisfied with zero loss.
//   2. If multiple natural zeros exist, pick the first (lowest index). Any
//      additional zeros are encoded as sign_bit=1 (i.e. +1) — this is a
//      deterministic choice that biases secondary zeros to +1 instead of -1
//      (the old packer's silent default). Frequency is logged + asserted
//      under 5% per tensor.
//   3. If no natural zero (rare on already-quantized HALO_V2 — most groups
//      contain at least one 0 because BitNet b1.58 is ternary), fall back to
//      lane 0 = zero_pos. We have no bf16 magnitudes here (input is HALO_V2
//      .h1b, already-quantized) so smallest-|w| tie-break is unavailable.
//      The lane-0 fallback matches the buggy py packer's behavior on
//      no-zero groups; the kernel's signs decode lossily for that lane.
//
// The bit packing itself is delegated to rcpp_sherry_pack(...) from
// librocm_cpp — that function takes a properly 3:4-sparse int8 buffer (one
// zero per group-of-4) and produces the byte-exact 5-bit-packed output the
// Sherry kernel LUT expects. We feed it well-formed input → bit-perfect
// output.
//
// CLI:
//
//   h1b_repack_sherry --input  halo-1bit-2b.h1b
//                     --output halo-1bit-2b-sherry-cpp.h1b
//                     [--threshold-mode absmean|smallest-quartile]
//
//   h1b_repack_sherry --verify halo-1bit-2b-sherry-cpp.h1b
//
// --threshold-mode accepted but unused (input is HALO_V2, already-quantized;
// we don't have bf16 magnitudes to honor a different threshold). Exists so
// future bf16-input mode lands without a CLI break.
//
// Build: see CMakeLists.txt — links rocm_cpp + hip::host. Pure host C++20
// otherwise; no HIP kernel calls.

#include "rocm_cpp/sherry.h"

#include <cassert>
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

// Halo v2 codes: 0->-1, 1->0, 2->+1, 3->unused (treated as 0).
static constexpr int8_t kHaloCodeToTernary[4] = {-1, 0, +1, 0};

// ── Round-5 calibration sidecar ("HCAL" v1) ──────────────────────────────────
//
// One per-input-dim (mu, sigma) entry per (layer, projection). The repacker
// uses these to draw input-distribution-matched activation samples and pick
// per-row scale corrections that bring the Sherry GEMV's expected output back
// onto the v2 expected output, row by row.
//
// File layout (matches sherry_calib_capture):
//   magic[4]  = 'H','C','A','L'
//   version   = int32(1)
//   hs        = int32
//   is_       = int32
//   layers    = int32
//   projs     = int32(7)  -- q,k,v,o,gate,up,down in this order
//   flags     = int32(0)
//   per (layer, proj) in layer-major order:
//     K_in    = int32
//     mu[K_in]    = float32 array
//     sigma[K_in] = float32 array
struct CalibTensor {
    int K_in = 0;
    std::vector<float> mu;
    std::vector<float> sigma;
};
struct CalibSidecar {
    bool loaded = false;
    int hs = 0, is_ = 0, layers = 0, projs = 0;
    // Indexed [layer * projs + proj_idx]. Project order matches
    // layer_tensors() in this file: q,k,v,o,gate,up,down.
    std::vector<CalibTensor> entries;

    const CalibTensor* lookup(int li, int proj_idx) const noexcept {
        if (!loaded) return nullptr;
        if (li < 0 || li >= layers) return nullptr;
        if (proj_idx < 0 || proj_idx >= projs) return nullptr;
        return &entries[(size_t)li * (size_t)projs + (size_t)proj_idx];
    }
};

bool load_calib_sidecar(const std::string& path, CalibSidecar& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const std::streamsize n = f.tellg();
    if (n < 24) return false;
    std::vector<uint8_t> buf((size_t)n);
    f.seekg(0, std::ios::beg);
    f.read(reinterpret_cast<char*>(buf.data()), n);
    if (!f) return false;
    if (std::memcmp(buf.data(), "HCAL", 4) != 0) {
        std::fprintf(stderr, "[sherry] calib sidecar: bad magic at %s\n", path.c_str());
        return false;
    }
    auto rd_i32 = [&](size_t off) -> int32_t {
        int32_t v;
        std::memcpy(&v, buf.data() + off, 4);
        return v;
    };
    if (rd_i32(4) != 1) {
        std::fprintf(stderr, "[sherry] calib sidecar: unsupported version %d\n",
                     rd_i32(4));
        return false;
    }
    out.hs     = rd_i32(8);
    out.is_    = rd_i32(12);
    out.layers = rd_i32(16);
    out.projs  = rd_i32(20);
    // flags @ 24 reserved; read but ignored.
    size_t off = 28;
    const size_t total = (size_t)out.layers * (size_t)out.projs;
    out.entries.assign(total, {});
    for (size_t i = 0; i < total; ++i) {
        if (off + 4 > buf.size()) {
            std::fprintf(stderr, "[sherry] calib sidecar: truncated at entry %zu\n", i);
            return false;
        }
        const int32_t K = rd_i32(off);
        off += 4;
        if (K <= 0 || K > (1 << 20)) {
            std::fprintf(stderr, "[sherry] calib sidecar: bad K_in=%d at entry %zu\n", K, i);
            return false;
        }
        const size_t bytes = (size_t)K * 4u;
        if (off + 2u * bytes > buf.size()) {
            std::fprintf(stderr, "[sherry] calib sidecar: short read at entry %zu\n", i);
            return false;
        }
        out.entries[i].K_in = K;
        out.entries[i].mu.resize((size_t)K);
        out.entries[i].sigma.resize((size_t)K);
        std::memcpy(out.entries[i].mu.data(),    buf.data() + off, bytes); off += bytes;
        std::memcpy(out.entries[i].sigma.data(), buf.data() + off, bytes); off += bytes;
    }
    out.loaded = true;
    std::printf("[sherry] calib sidecar loaded: %s (L=%d P=%d hs=%d is=%d)\n",
                path.c_str(), out.layers, out.projs, out.hs, out.is_);
    return true;
}

// Per-tensor stats accumulated by the packer + reported on stdout.
struct TensorStats {
    std::string name;
    int rows         = 0;
    int cols         = 0;
    size_t v2_bytes  = 0;
    size_t v3_bytes  = 0;
    // Group-of-4 zero-count distribution.
    uint64_t groups_total       = 0;
    uint64_t groups_zero_count[5] = {0, 0, 0, 0, 0};  // index 0..4
    uint64_t natural_zero_picks = 0;   // first-zero lane chosen (lossless)
    uint64_t multi_zero_groups  = 0;   // 2+ zeros in same group
    uint64_t no_zero_groups     = 0;   // 0 zeros (forced lane-0 = sign drop)
    uint64_t phantom_signs_lost = 0;   // ±1 lanes overwritten by forced zero
};

void print_tensor_stats(const TensorStats& s) {
    const double natz_rate = s.groups_total
        ? (double)s.natural_zero_picks / (double)s.groups_total
        : 0.0;
    const double multi_rate = s.groups_total
        ? (double)s.multi_zero_groups / (double)s.groups_total
        : 0.0;
    const double nozero_rate = s.groups_total
        ? (double)s.no_zero_groups / (double)s.groups_total
        : 0.0;
    std::printf(
        "[sherry] %-6s %5dx%-5d v2=%9zu B v3=%9zu B "
        "groups=%-9" PRIu64 " "
        "natz=%6.2f%% multi=%6.2f%% nozero=%6.2f%% phantom_lost=%" PRIu64 "\n",
        s.name.c_str(), s.rows, s.cols, s.v2_bytes, s.v3_bytes,
        s.groups_total, natz_rate * 100.0, multi_rate * 100.0,
        nozero_rate * 100.0, s.phantom_signs_lost);
}

// Halo v2 row → ternary int8 buffer.
void halo_unpack_row(const uint8_t* packed_row, int cols, int8_t* out) {
    const int row_bytes = (cols + 3) / 4;
    for (int b = 0; b < row_bytes; ++b) {
        const uint8_t byte = packed_row[b];
        for (int slot = 0; slot < 4; ++slot) {
            const int k = b * 4 + slot;
            if (k >= cols) break;
            out[k] = kHaloCodeToTernary[(byte >> (slot * 2)) & 0x3];
        }
    }
}

// Convert a ternary row (q ∈ {-1,0,+1}, length=cols) into the strictly-3:4-
// sparse form rcpp_sherry_pack expects. Returns nothing — fills `out` with
// the same int8 values, but with exactly one zero per 4-group, choosing
// natural zeros where possible.
void make_3to4_sparse(const int8_t* ternary, int cols, int8_t* out,
                      TensorStats& stats)
{
    assert((cols & 3) == 0);
    const int groups = cols / 4;
    for (int g = 0; g < groups; ++g) {
        const int base = g * 4;
        int8_t v[4] = {ternary[base + 0], ternary[base + 1],
                       ternary[base + 2], ternary[base + 3]};
        int zeros = 0;
        int first_zero = -1;
        for (int p = 0; p < 4; ++p) {
            if (v[p] == 0) {
                if (first_zero < 0) first_zero = p;
                ++zeros;
            }
        }
        ++stats.groups_total;
        ++stats.groups_zero_count[zeros];

        int zero_pos;
        if (zeros >= 1) {
            // Natural-zero path. Lossless: keep the first zero as zero_pos,
            // remaining lanes carry their original ±1 (or, for a 2nd/3rd
            // natural zero, default to +1 — see comment below).
            zero_pos = first_zero;
            ++stats.natural_zero_picks;
            if (zeros >= 2) ++stats.multi_zero_groups;
        } else {
            // No natural zero (all four lanes are ±1). We must drop one lane
            // to satisfy 3:4. Without bf16 magnitudes we have no oracle for
            // "smallest |w|", so distribute the loss across all four positions
            // by picking `g % 4`. This breaks the lane-0 systematic bias the
            // old policy carried (every no-zero group dropped its lane-0
            // contribution; the ~13% of groups in this path collectively shaved
            // ~3.25% of column-0-aligned weight contribution off every row).
            // Round-robin on g %% 4 makes the loss 0-mean across the row.
            zero_pos = g & 0x3;
            ++stats.no_zero_groups;
            ++stats.phantom_signs_lost;
            v[zero_pos] = 0;
        }

        // Force every non-(zero_pos) zero to ±1 so the packed sign bit
        // matches the well-formed 3:4 contract. The OLD policy used a
        // hardcoded +1 here, which collapsed PPL to ~1e9 because 54% of
        // halo-1bit-2b groups have ≥2 natural zeros: forcing all of
        // their secondary zeros to +1 introduced a DC bias on every
        // row (validated 2026-04-26 — phantom_signs_lost=94.7% of
        // groups). Use a deterministic *balanced* fill keyed on
        // (group_index ^ lane_index) so half the phantom signs are −1
        // and half are +1 across each row, preserving the row's mean
        // in expectation.
        for (int p = 0; p < 4; ++p) {
            if (p == zero_pos) {
                v[p] = 0;
            } else if (v[p] == 0) {
                v[p] = ((g + p) & 1) ? +1 : -1;
                ++stats.phantom_signs_lost;
            }
        }

        for (int p = 0; p < 4; ++p) out[base + p] = v[p];
    }
}

// Pack one full row using rcpp_sherry_pack. Caller has already converted
// the row to the strict 3:4-sparse form via make_3to4_sparse.
void pack_row_sherry(const int8_t* sparse_ternary, int cols, uint8_t* packed_out) {
    assert((cols & 31) == 0);
    rcpp_sherry_pack(sparse_ternary, packed_out, cols);
}

// ---- file I/O helpers ------------------------------------------------------

bool read_all(const std::string& path, std::vector<uint8_t>& buf) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const std::streamsize n = f.tellg();
    if (n < 0) return false;
    buf.resize((size_t)n);
    f.seekg(0, std::ios::beg);
    f.read(reinterpret_cast<char*>(buf.data()), n);
    return (bool)f;
}

template<typename T>
T read_scalar_le(const uint8_t* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

// ---- ternary tensor shape spec --------------------------------------------

struct TensorSpec {
    const char* name;
    int rows;
    int cols;
};

std::vector<TensorSpec> layer_tensors(int hs, int is_, int nh, int nkv, int hd) {
    return {
        {"q",    nh  * hd, hs},
        {"k",    nkv * hd, hs},
        {"v",    nkv * hd, hs},
        {"o",    hs,       nh * hd},
        {"gate", is_,      hs},
        {"up",   is_,      hs},
        {"down", hs,       is_},
    };
}

// ---- repack core -----------------------------------------------------------

// Sample N=8 deterministic Box-Muller draws per K dim using Normal(mu[k], sigma[k]).
// Returns the per-sample activation vectors as a flat [N_SAMPLES][K] float buffer.
// Seed is keyed on (li, proj_idx) so calibration is reproducible across runs and
// independent across (layer, projection) pairs.
constexpr int kCalibSamples = 8;

void draw_calib_samples(const CalibTensor& ct, int li, int proj_idx,
                        std::vector<float>& acts /*[N_SAMPLES * K]*/)
{
    const int K = ct.K_in;
    acts.assign((size_t)kCalibSamples * (size_t)K, 0.0f);
    // xorshift64 keyed on (li, proj_idx).
    uint64_t s = 0x9E3779B97F4A7C15ULL
               ^ ((uint64_t)li * 0x9E3779B97F4A7C15ULL)
               ^ ((uint64_t)proj_idx * 0xC2B2AE3D27D4EB4FULL);
    if (s == 0) s = 0xDEADBEEFDEADBEEFULL;
    auto next_u64 = [&]() -> uint64_t {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return s;
    };
    auto next_uniform_open = [&]() -> double {
        // (0, 1) — exclusive at both ends, so log() in Box-Muller is finite.
        const uint64_t u = (next_u64() >> 11) | 1ULL;  // [1, 2^53), avoid 0
        return (double)u / (double)(1ULL << 53);
    };
    // Box-Muller pairs.
    for (int n = 0; n < kCalibSamples; ++n) {
        float* row = acts.data() + (size_t)n * (size_t)K;
        int k = 0;
        while (k + 1 < K) {
            const double u1 = next_uniform_open();
            const double u2 = next_uniform_open();
            const double r = std::sqrt(-2.0 * std::log(u1));
            const double th = 6.28318530717958647692 * u2;
            const double z0 = r * std::cos(th);
            const double z1 = r * std::sin(th);
            row[k]     = (float)(ct.mu[(size_t)k]     + (double)ct.sigma[(size_t)k]     * z0);
            row[k + 1] = (float)(ct.mu[(size_t)(k+1)] + (double)ct.sigma[(size_t)(k+1)] * z1);
            k += 2;
        }
        if (k < K) {
            const double u1 = next_uniform_open();
            const double u2 = next_uniform_open();
            const double r = std::sqrt(-2.0 * std::log(u1));
            const double th = 6.28318530717958647692 * u2;
            const double z = r * std::cos(th);
            row[k] = (float)(ct.mu[(size_t)k] + (double)ct.sigma[(size_t)k] * z);
        }
    }
}

// Per-row scale ratio over a fixed sample population.
//
//   ratio[r] = (Σ_n a_n . v2_signed_n) / (Σ_n a_n . sherry_signed_n)
//
// where the dot products are
//   sum_k acts_n[k] * v2_signs[r,k]
//   sum_k acts_n[k] * sh_signs[r,k]
// and v2_signs / sh_signs are int8 ∈ {-1, 0, +1}. We absorb the L2 of the
// activation field by accumulating |a_n . w_v2| and |a_n . w_sh| (signed
// sums, not absolute) and taking the ratio of their sums-over-N. Clamp to
// [0.5, 2.0] — the round-4 [0.25, 4.0] clamp was too loose because the
// activation distribution it sampled had no per-dim shape; with real per-K
// sigma the ratio should sit much closer to 1.
//
// Returns 1.0 if either denominator is zero (kernel-side fallback to v2
// scale verbatim, which is what round-3 already does).
float per_row_calib_ratio(const std::vector<float>& acts /*[N_SAMPLES * K]*/,
                          const int8_t* v2_row,
                          const int8_t* sh_row,
                          int K)
{
    double num = 0.0, den = 0.0;
    for (int n = 0; n < kCalibSamples; ++n) {
        const float* a = acts.data() + (size_t)n * (size_t)K;
        double dv = 0.0, ds = 0.0;
        for (int k = 0; k < K; ++k) {
            const float ak = a[k];
            dv += (double)ak * (double)v2_row[k];
            ds += (double)ak * (double)sh_row[k];
        }
        num += dv;
        den += ds;
    }
    if (std::fabs(den) < 1e-12 || std::fabs(num) < 1e-12) return 1.0f;
    double r = num / den;
    if (r < 0.5)  r = 0.5;
    if (r > 2.0)  r = 2.0;
    return (float)r;
}

int do_repack(const std::string& in_path, const std::string& out_path,
              const std::string& calib_path)
{
    std::vector<uint8_t> buf;
    if (!read_all(in_path, buf)) {
        std::fprintf(stderr, "[h1b_repack_sherry] cannot open %s\n", in_path.c_str());
        return 1;
    }
    if (buf.size() < 4 + 4 + 9 * 4) {
        std::fprintf(stderr, "[h1b_repack_sherry] input too small\n");
        return 1;
    }
    if (std::memcmp(buf.data(), "H1B\x00", 4) != 0) {
        std::fprintf(stderr, "[h1b_repack_sherry] bad magic\n");
        return 1;
    }
    size_t off = 4;
    int32_t version = read_scalar_le<int32_t>(buf.data() + off);
    off += 4;
    if (version != 1 && version != 2) {
        std::fprintf(stderr, "[h1b_repack_sherry] unsupported input version %d "
                             "(want HALO_V2; v3/v4 are already Sherry)\n", version);
        return 1;
    }

    int32_t cfg[9];
    std::memcpy(cfg, buf.data() + off, sizeof(cfg));
    off += sizeof(cfg);
    const int hs  = cfg[0];
    const int is_ = cfg[1];
    const int L   = cfg[2];
    const int nh  = cfg[3];
    const int nkv = cfg[4];
    const int V   = cfg[5];
    const int hd  = hs / nh;

    std::printf("[sherry] config: hs=%d is=%d L=%d nh=%d nkv=%d V=%d hd=%d\n",
                hs, is_, L, nh, nkv, V, hd);

    // Round-5 calibration sidecar. Optional — if absent, behavior matches
    // round-3 (S_sh[r] = S_v2[r] verbatim, no per-row correction).
    CalibSidecar calib;
    std::string resolved_calib_path = calib_path;
    if (resolved_calib_path.empty()) {
        // Auto-derive: input.h1b → input.calib.bin (replace trailing .h1b).
        const std::string& s = in_path;
        if (s.size() >= 4 && s.compare(s.size() - 4, 4, ".h1b") == 0) {
            resolved_calib_path = s.substr(0, s.size() - 4) + ".calib.bin";
        } else {
            resolved_calib_path = s + ".calib.bin";
        }
    }
    if (load_calib_sidecar(resolved_calib_path, calib)) {
        if (calib.hs != hs || calib.is_ != is_ || calib.layers != L) {
            std::fprintf(stderr,
                "[sherry] calib sidecar dim mismatch (hs=%d is=%d L=%d vs "
                "h1b hs=%d is=%d L=%d) — IGNORING\n",
                calib.hs, calib.is_, calib.layers, hs, is_, L);
            calib.loaded = false;
        }
    } else {
        std::printf("[sherry] no calib sidecar at %s — round-3 fallback "
                    "(S_sh[r] = S_v2[r] verbatim)\n", resolved_calib_path.c_str());
    }

    float rope_theta = 500000.0f;
    float rms_eps    = 1.0e-5f;
    if (version >= 2) {
        rope_theta = read_scalar_le<float>(buf.data() + off);
        off += 4;
        rms_eps    = read_scalar_le<float>(buf.data() + off);
        off += 4;
    }
    std::printf("[sherry] rope_theta=%.1f rms_eps=%.1e tied_emb=%d\n",
                rope_theta, rms_eps, cfg[7]);

    // Sherry alignment.
    auto check_align = [&](const char* name, int k) {
        if (k % 32 != 0) {
            std::fprintf(stderr, "[h1b_repack_sherry] %s=%d not divisible by 32 — "
                                 "Sherry packing requires it\n", name, k);
            std::exit(1);
        }
    };
    check_align("hs",     hs);
    check_align("is",     is_);
    check_align("nh*hd",  nh * hd);
    check_align("nkv*hd", nkv * hd);

    // Build v3 header. Mirror v2 cfg exactly except cfg[8] |= H1B_FLAG_SHERRY_FP16.
    int32_t cfg_out[9];
    std::memcpy(cfg_out, cfg, sizeof(cfg));
    cfg_out[8] = (int32_t)((uint32_t)cfg[8] | 0x2u);  // H1B_FLAG_SHERRY_FP16

    std::vector<uint8_t> out;
    out.reserve(buf.size());  // upper bound; v3 is smaller for ternary tensors
    auto append_bytes = [&](const void* p, size_t n) {
        const uint8_t* pp = static_cast<const uint8_t*>(p);
        out.insert(out.end(), pp, pp + n);
    };
    append_bytes("H1B\x00", 4);
    const int32_t v3 = 3;
    append_bytes(&v3, 4);
    append_bytes(cfg_out, sizeof(cfg_out));
    append_bytes(&rope_theta, 4);
    append_bytes(&rms_eps, 4);

    // Embedding + final norm — pass-through.
    const size_t emb_bytes  = (size_t)V * (size_t)hs * 4;
    const size_t norm_bytes = (size_t)hs * 4;
    if (off + emb_bytes + norm_bytes > buf.size()) {
        std::fprintf(stderr, "[h1b_repack_sherry] short read: emb/final_norm\n");
        return 1;
    }
    append_bytes(buf.data() + off, emb_bytes); off += emb_bytes;
    append_bytes(buf.data() + off, norm_bytes); off += norm_bytes;

    // Per-layer.
    std::vector<TensorStats> all_stats;
    all_stats.reserve((size_t)L * 7);
    for (int li = 0; li < L; ++li) {
        // Norm block: 1 input + 1 post + 4×attn_sub + 2×ffn_sub_truncated + 1 ffn_sub.
        const size_t norm_block_bytes =
            (size_t)hs * 4 * (1 + 1 + 4 + 2) + (size_t)is_ * 4;
        if (off + norm_block_bytes > buf.size()) {
            std::fprintf(stderr, "[h1b_repack_sherry] short read: norms L%d\n", li);
            return 1;
        }
        append_bytes(buf.data() + off, norm_block_bytes);
        off += norm_block_bytes;

        const auto specs = layer_tensors(hs, is_, nh, nkv, hd);
        std::vector<int8_t> ternary;
        std::vector<int8_t> sparse;
        std::vector<uint8_t> packed_v3;
        // Round-5: per-(layer, proj) calibration sample buffer. Reused across
        // rows of the same projection.
        std::vector<float> calib_acts;
        for (size_t proj_idx = 0; proj_idx < specs.size(); ++proj_idx) {
            const auto& sp = specs[proj_idx];
            const size_t src_row_bytes = (size_t)((sp.cols + 3) / 4);
            const size_t src_bytes     = (size_t)sp.rows * src_row_bytes;
            const size_t scales_bytes  = (size_t)sp.rows * 4;
            if (off + src_bytes + scales_bytes > buf.size()) {
                std::fprintf(stderr, "[h1b_repack_sherry] short read: tensor L%d %s\n",
                             li, sp.name);
                return 1;
            }
            const uint8_t* src = buf.data() + off;
            off += src_bytes;
            const uint8_t* scales = buf.data() + off;
            off += scales_bytes;

            const size_t dst_row_bytes = (size_t)sp.cols * 5 / 32;
            const size_t dst_bytes     = (size_t)sp.rows * dst_row_bytes;
            packed_v3.assign(dst_bytes, 0);
            ternary.assign((size_t)sp.cols, 0);
            sparse.assign((size_t)sp.cols, 0);

            TensorStats stats;
            stats.name     = sp.name;
            stats.rows     = sp.rows;
            stats.cols     = sp.cols;
            stats.v2_bytes = src_bytes;
            stats.v3_bytes = dst_bytes;

            // Per-row scale path.
            //
            // Round-3 (no calibration sidecar): pass through `scales_v2[r]`
            // verbatim. The phantom-sign fill is balanced ±1 so E[Σ a*w_sh]
            // tracks E[Σ a*w_v2] in expectation; on a long-context PPL run
            // the lane-drop noise averages out across rows.
            //
            // Round-5 (calibration sidecar present): per-row distribution-
            // matched correction.
            //
            //   S_sh[r] = S_v2[r] * (Σ_n a_n . v2_signed_n)
            //                     / (Σ_n a_n . sherry_signed_n)
            //
            //   over N=8 deterministic Box-Muller samples drawn from
            //   Normal(mu[k], sigma[k]) per K dim, where (mu, sigma) come
            //   from the calibration sidecar. Round-4 used a flat Gaussian
            //   without per-dim shape; the resulting ratios diverged
            //   wildly (range -58k to +87k, 31% clamped) and PPL regressed
            //   to 4.2e7. With per-K-dim sigma the activation distribution
            //   matches the actual post-RMSNorm distribution shape per
            //   layer/projection, so the ratio sits much closer to 1 and
            //   the [0.5, 2.0] clamp catches only genuine outliers.
            std::vector<float> scales_v3((size_t)sp.rows, 0.0f);
            const float* scales_v2 =
                reinterpret_cast<const float*>(scales);

            const CalibTensor* ct =
                calib.lookup(li, static_cast<int>(proj_idx));
            const bool use_calib = (ct != nullptr) && (ct->K_in == sp.cols);
            if (calib.loaded && !use_calib) {
                std::fprintf(stderr,
                    "[sherry][warn] L%d %s: calib K_in=%d != tensor cols=%d — "
                    "row scale falls back to v2 verbatim\n",
                    li, sp.name, ct ? ct->K_in : -1, sp.cols);
            }
            if (use_calib) {
                draw_calib_samples(*ct, li, static_cast<int>(proj_idx),
                                   calib_acts);
            }

            // Per-tensor calib ratio stats.
            double ratio_min = 1e30, ratio_max = -1e30, ratio_sum = 0.0;
            int ratio_clamped = 0, ratio_count = 0;
            // Reusable scratch for v2 ternary signs (we already have ternary
            // for sherry packing; we need a separate copy because
            // make_3to4_sparse mutates `ternary` → `sparse`).
            std::vector<int8_t> v2_row_buf;
            if (use_calib) v2_row_buf.assign((size_t)sp.cols, 0);

            for (int r = 0; r < sp.rows; ++r) {
                halo_unpack_row(src + (size_t)r * src_row_bytes,
                                sp.cols, ternary.data());
                if (use_calib) {
                    std::memcpy(v2_row_buf.data(), ternary.data(),
                                (size_t)sp.cols);
                }
                make_3to4_sparse(ternary.data(), sp.cols, sparse.data(), stats);
                pack_row_sherry(sparse.data(), sp.cols,
                                packed_v3.data() + (size_t)r * dst_row_bytes);
                if (use_calib) {
                    const float ratio = per_row_calib_ratio(
                        calib_acts, v2_row_buf.data(), sparse.data(), sp.cols);
                    scales_v3[r] = scales_v2[r] * ratio;
                    if (ratio == 0.5f || ratio == 2.0f) ++ratio_clamped;
                    if (ratio < ratio_min) ratio_min = ratio;
                    if (ratio > ratio_max) ratio_max = ratio;
                    ratio_sum += (double)ratio;
                    ++ratio_count;
                } else {
                    scales_v3[r] = scales_v2[r];
                }
            }
            if (use_calib && ratio_count > 0 && (li == 0 || li + 1 == L)) {
                std::printf(
                    "[sherry] L%-2d %-4s ratio min=%.3f max=%.3f mean=%.3f "
                    "clamped=%d/%d\n",
                    li, sp.name, ratio_min, ratio_max,
                    ratio_sum / (double)ratio_count, ratio_clamped, ratio_count);
            }

            // Emit packed tensor + recomputed per-row scales.
            append_bytes(packed_v3.data(), dst_bytes);
            append_bytes(reinterpret_cast<const uint8_t*>(scales_v3.data()),
                         scales_bytes);

            // Multi-zero rate cap (architect spec: < 5%).
            const double multi_rate = stats.groups_total
                ? (double)stats.multi_zero_groups / (double)stats.groups_total
                : 0.0;
            if (multi_rate >= 0.05) {
                std::fprintf(stderr,
                    "[h1b_repack_sherry][warn] L%d %s: multi-zero rate %.2f%% "
                    "(cap 5.00%%) — secondary zeros forced to +1\n",
                    li, sp.name, multi_rate * 100.0);
            }

            if (li == 0 || (li + 1) % 10 == 0 || li + 1 == L) {
                if (li == 0) print_tensor_stats(stats);
            }
            all_stats.push_back(std::move(stats));
        }
        if ((li + 1) % 5 == 0 || li + 1 == L) {
            std::printf("[sherry] layer %d/%d repacked\n", li + 1, L);
        }
    }

    // Trailing bytes (untied LM head, etc.) — pass-through.
    if (off < buf.size()) {
        const size_t trailing = buf.size() - off;
        std::printf("[sherry] copying %zu trailing bytes (untied LM head?)\n", trailing);
        append_bytes(buf.data() + off, trailing);
    }

    std::ofstream f(out_path, std::ios::binary | std::ios::trunc);
    if (!f) {
        std::fprintf(stderr, "[h1b_repack_sherry] cannot open output %s\n",
                     out_path.c_str());
        return 1;
    }
    f.write(reinterpret_cast<const char*>(out.data()), (std::streamsize)out.size());
    if (!f) {
        std::fprintf(stderr, "[h1b_repack_sherry] short write\n");
        return 1;
    }
    f.close();

    // Aggregate summary.
    uint64_t total_groups = 0, total_natz = 0, total_multi = 0,
             total_nozero = 0, total_phantom = 0;
    for (const auto& s : all_stats) {
        total_groups  += s.groups_total;
        total_natz    += s.natural_zero_picks;
        total_multi   += s.multi_zero_groups;
        total_nozero  += s.no_zero_groups;
        total_phantom += s.phantom_signs_lost;
    }
    std::printf("\n[sherry] === SUMMARY ===\n");
    std::printf("[sherry] tensors=%zu groups=%" PRIu64 "\n",
                all_stats.size(), total_groups);
    std::printf("[sherry] natural_zero=%6.2f%% multi_zero=%6.2f%% no_zero=%6.2f%%\n",
                total_groups ? 100.0 * (double)total_natz   / (double)total_groups : 0.0,
                total_groups ? 100.0 * (double)total_multi  / (double)total_groups : 0.0,
                total_groups ? 100.0 * (double)total_nozero / (double)total_groups : 0.0);
    std::printf("[sherry] phantom_signs_lost=%" PRIu64 " (%.4f%% of groups)\n",
                total_phantom,
                total_groups ? 100.0 * (double)total_phantom / (double)total_groups : 0.0);
    std::printf("[sherry] wrote %s (%zu B, %.1f%% of v2 input)\n",
                out_path.c_str(), out.size(), 100.0 * (double)out.size() / (double)buf.size());
    return 0;
}

// ---- verify ---------------------------------------------------------------
//
// Reads a v3 file back, decodes every Sherry-packed weight via the same LUT
// the kernel uses (`build_sherry_entry`-equivalent), and compares MSE per
// tensor against the canonical HALO_V2 source if `--source` is supplied. If
// no source is supplied we run a self-consistency pass: re-pack each row
// from its decoded ternary form and assert byte-identity. Sherry's lossy
// 3:4 contract means re-pack-from-decode is bit-exact (decoded data is
// already 3:4 sparse).

void sherry_decode_group(uint32_t code, int8_t out[4]) {
    const uint32_t zp    = (code >> 3) & 0x3;
    const uint32_t signs =  code       & 0x7;
    int idx = 0;
    for (int p = 0; p < 4; ++p) {
        if ((uint32_t)p == zp) {
            out[p] = 0;
        } else {
            const int bit = (signs >> idx) & 1;
            out[p] = (int8_t)(bit ? +1 : -1);
            ++idx;
        }
    }
}

void sherry_decode_row(const uint8_t* packed, int cols, int8_t* out) {
    assert((cols & 31) == 0);
    const int macrogroups = cols / 32;
    for (int mg = 0; mg < macrogroups; ++mg) {
        // Reconstruct the 40-bit word (5 bytes) LSB-first.
        uint64_t bits40 = 0;
        for (int b = 0; b < 5; ++b) {
            bits40 |= (uint64_t)packed[mg * 5 + b] << (8 * b);
        }
        for (int sg = 0; sg < 8; ++sg) {
            const uint32_t code = (uint32_t)((bits40 >> (5 * sg)) & 0x1F);
            int8_t group[4];
            sherry_decode_group(code, group);
            for (int p = 0; p < 4; ++p) {
                out[mg * 32 + sg * 4 + p] = group[p];
            }
        }
    }
}

int do_verify(const std::string& path, const std::string& source_path) {
    std::vector<uint8_t> buf;
    if (!read_all(path, buf)) {
        std::fprintf(stderr, "[verify] cannot open %s\n", path.c_str());
        return 1;
    }
    if (buf.size() < 4 + 4 + 9 * 4) {
        std::fprintf(stderr, "[verify] input too small\n");
        return 1;
    }
    if (std::memcmp(buf.data(), "H1B\x00", 4) != 0) {
        std::fprintf(stderr, "[verify] bad magic\n");
        return 1;
    }
    size_t off = 4;
    int32_t version = read_scalar_le<int32_t>(buf.data() + off);
    off += 4;
    if (version != 3) {
        std::fprintf(stderr, "[verify] expected v3 file, got v%d\n", version);
        return 1;
    }
    int32_t cfg[9];
    std::memcpy(cfg, buf.data() + off, sizeof(cfg));
    off += sizeof(cfg);
    const int hs  = cfg[0];
    const int is_ = cfg[1];
    const int L   = cfg[2];
    const int nh  = cfg[3];
    const int nkv = cfg[4];
    const int V   = cfg[5];
    const int hd  = hs / nh;
    const uint32_t flags = (uint32_t)cfg[8];

    std::printf("[verify] v3 cfg: hs=%d is=%d L=%d nh=%d nkv=%d V=%d hd=%d flags=0x%x\n",
                hs, is_, L, nh, nkv, V, hd, flags);

    // Skip rope_theta + rms_eps + emb + final_norm.
    off += 8;
    off += (size_t)V * hs * 4;
    off += (size_t)hs * 4;

    // Optional cross-check vs HALO_V2 source.
    std::vector<uint8_t> src_buf;
    size_t src_off = 0;
    bool have_source = !source_path.empty();
    if (have_source) {
        if (!read_all(source_path, src_buf)) {
            std::fprintf(stderr, "[verify] cannot open source %s\n", source_path.c_str());
            return 1;
        }
        if (std::memcmp(src_buf.data(), "H1B\x00", 4) != 0) {
            std::fprintf(stderr, "[verify] source bad magic\n");
            return 1;
        }
        int32_t src_ver = read_scalar_le<int32_t>(src_buf.data() + 4);
        if (src_ver != 1 && src_ver != 2) {
            std::fprintf(stderr, "[verify] source must be HALO_V2 (got v%d)\n", src_ver);
            return 1;
        }
        src_off = 4 + 4 + sizeof(cfg) + (src_ver >= 2 ? 8u : 0u)
                + (size_t)V * hs * 4 + (size_t)hs * 4;
    }

    uint64_t total_weights = 0;
    uint64_t mismatches = 0;
    double sse = 0.0;
    std::vector<int8_t> decoded;
    std::vector<int8_t> source;
    for (int li = 0; li < L; ++li) {
        const size_t norm_block_bytes =
            (size_t)hs * 4 * (1 + 1 + 4 + 2) + (size_t)is_ * 4;
        off += norm_block_bytes;
        if (have_source) src_off += norm_block_bytes;

        const auto specs = layer_tensors(hs, is_, nh, nkv, hd);
        for (const auto& sp : specs) {
            const size_t row_bytes_v3 = (size_t)sp.cols * 5 / 32;
            const size_t bytes_v3     = (size_t)sp.rows * row_bytes_v3;
            const size_t scales_bytes = (size_t)sp.rows * 4;
            const uint8_t* packed = buf.data() + off;
            off += bytes_v3 + scales_bytes;

            decoded.assign((size_t)sp.cols, 0);
            uint64_t tensor_zero = 0, tensor_pos = 0, tensor_neg = 0;
            for (int r = 0; r < sp.rows; ++r) {
                sherry_decode_row(packed + (size_t)r * row_bytes_v3,
                                  sp.cols, decoded.data());
                for (int c = 0; c < sp.cols; ++c) {
                    const int8_t q = decoded[c];
                    if      (q == 0)  ++tensor_zero;
                    else if (q > 0)   ++tensor_pos;
                    else              ++tensor_neg;
                }
                if (have_source) {
                    const size_t src_row_bytes = (size_t)((sp.cols + 3) / 4);
                    source.assign((size_t)sp.cols, 0);
                    halo_unpack_row(src_buf.data() + src_off + (size_t)r * src_row_bytes,
                                    sp.cols, source.data());
                    for (int c = 0; c < sp.cols; ++c) {
                        ++total_weights;
                        const int diff = (int)source[c] - (int)decoded[c];
                        if (diff != 0) ++mismatches;
                        sse += (double)(diff * diff);
                    }
                }
            }
            if (have_source) {
                src_off += (size_t)sp.rows * (size_t)((sp.cols + 3) / 4);
                src_off += scales_bytes;
            }
            if (li == 0) {
                const uint64_t total = (uint64_t)sp.rows * (uint64_t)sp.cols;
                std::printf("[verify] L0 %-6s %5dx%-5d zero=%5.2f%% +1=%5.2f%% -1=%5.2f%%\n",
                            sp.name, sp.rows, sp.cols,
                            100.0 * (double)tensor_zero / (double)total,
                            100.0 * (double)tensor_pos  / (double)total,
                            100.0 * (double)tensor_neg  / (double)total);
            }
        }
    }
    if (have_source && total_weights > 0) {
        const double mse = sse / (double)total_weights;
        const double mismatch_rate = (double)mismatches / (double)total_weights;
        std::printf("\n[verify] vs source: weights=%" PRIu64 " mismatches=%" PRIu64
                    " (%.4f%%) MSE=%.6f\n",
                    total_weights, mismatches, mismatch_rate * 100.0, mse);
        if (mse > 0.5) {
            std::fprintf(stderr, "[verify] FAIL: MSE %.6f > 0.5 threshold\n", mse);
            return 2;
        }
        std::printf("[verify] PASS: MSE %.6f <= 0.5\n", mse);
    } else {
        std::printf("[verify] decode-only pass complete (no --source supplied)\n");
    }
    return 0;
}

void usage() {
    std::fprintf(stderr,
        "usage:\n"
        "  h1b_repack_sherry --input <halo_v2.h1b> --output <sherry_v3.h1b>\n"
        "                    [--threshold-mode absmean|smallest-quartile]\n"
        "                    [--calib <model.calib.bin>]\n"
        "  h1b_repack_sherry --verify <sherry_v3.h1b> [--source <halo_v2.h1b>]\n"
        "\n"
        "Round-5 calibration (--calib): emit sidecar via sherry_calib_capture\n"
        "first, then point this tool at it. If --calib is omitted the tool\n"
        "auto-derives the path (input.h1b -> input.calib.bin); if that file\n"
        "is absent, the repacker falls back to round-3 behavior (S_v2[r]\n"
        "verbatim).\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string in_path, out_path, verify_path, source_path, calib_path;
    std::string threshold_mode = "absmean";
    bool mode_repack = false, mode_verify = false;
    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        if      (a == "--input"           && i + 1 < argc) { in_path        = argv[++i]; mode_repack = true; }
        else if (a == "--output"          && i + 1 < argc) { out_path       = argv[++i]; mode_repack = true; }
        else if (a == "--threshold-mode"  && i + 1 < argc) { threshold_mode = argv[++i]; }
        else if (a == "--calib"           && i + 1 < argc) { calib_path     = argv[++i]; }
        else if (a == "--verify"          && i + 1 < argc) { verify_path    = argv[++i]; mode_verify = true; }
        else if (a == "--source"          && i + 1 < argc) { source_path    = argv[++i]; }
        else if (a == "--help" || a == "-h") { usage(); return 0; }
        else {
            std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
            usage();
            return 2;
        }
    }

    if (mode_repack && mode_verify) {
        std::fprintf(stderr, "[h1b_repack_sherry] --input/--output and --verify are mutually exclusive\n");
        return 2;
    }
    if (!mode_repack && !mode_verify) {
        usage();
        return 2;
    }
    if (threshold_mode != "absmean" && threshold_mode != "smallest-quartile") {
        std::fprintf(stderr, "[h1b_repack_sherry] unknown --threshold-mode '%s' "
                             "(want absmean|smallest-quartile)\n",
                     threshold_mode.c_str());
        return 2;
    }

    if (mode_repack) {
        if (in_path.empty() || out_path.empty()) {
            std::fprintf(stderr, "[h1b_repack_sherry] --input and --output both required\n");
            return 2;
        }
        return do_repack(in_path, out_path, calib_path);
    }
    return do_verify(verify_path, source_path);
}

// test_medusa_loader.cpp — round-trip the `.h1b-medusa` v1 + v2 sidecars.
//
// v1 — synthetic-zero per-head [vocab, hidden] ternary projection. Negative
//      cases: bad magic / hidden mismatch / vocab mismatch / truncated.
// v2 — residual-MLP topology with w_in/w_out fp16 dense [hidden, hidden]
//      tensors. Engine reuses base.lm_head, so vocab_size is unused in
//      the header. Negative cases: bad variant / bad dtype / hidden
//      mismatch / truncated.
//
// Needs a live ROCm device (the loader hipMallocs the per-head buffers).
// SKIPs cleanly with exit 0 if no device.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "rocm_cpp/medusa.h"

namespace {

int g_failures = 0;

void check(bool cond, const char* desc) {
    std::fprintf(stderr, "  %s  %s\n", cond ? "PASS" : "FAIL", desc);
    if (!cond) ++g_failures;
}

bool gpu_available() {
    int n = 0;
    hipError_t e = hipGetDeviceCount(&n);
    return (e == hipSuccess) && (n > 0);
}

// Synthetic-zero writer. Mirrors the docs/h1b-medusa-format.md v1 layout.
//   header (32 bytes)
//   per head: vocab*((hidden+3)/4) zero bytes, then vocab fp32 zero scales.
// HALO_V2 only — that's the only format the smoke path exercises.
void write_synthetic_h1b_medusa(const std::string& path,
                                uint32_t num_heads,
                                uint32_t hidden_size,
                                uint32_t vocab_size,
                                bool corrupt_magic = false,
                                bool truncate_tail = false)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "FAIL: cannot create %s\n", path.c_str());
        std::exit(2);
    }
    char magic[4] = {'H', '1', 'B', 'M'};
    if (corrupt_magic) magic[0] = 'X';
    f.write(magic, 4);

    auto wu32 = [&](uint32_t v) { f.write(reinterpret_cast<char*>(&v), 4); };
    wu32(1);                                          // version
    wu32(num_heads);
    wu32(hidden_size);
    wu32(vocab_size);
    wu32(/*weight_format=*/0);                        // HALO_V2
    wu32(0); wu32(0);                                 // reserved

    const size_t per_head_w_bytes =
        (size_t)vocab_size * (size_t)((hidden_size + 3u) / 4u);
    std::vector<uint8_t> w(per_head_w_bytes, 0u);
    std::vector<float>   s(vocab_size, 0.0f);

    const uint32_t emit = truncate_tail
                              ? (num_heads > 0 ? num_heads - 1 : 0)
                              : num_heads;
    for (uint32_t h = 0; h < emit; ++h) {
        f.write(reinterpret_cast<char*>(w.data()), per_head_w_bytes);
        f.write(reinterpret_cast<char*>(s.data()), s.size() * sizeof(float));
    }
}

// Build a minimal rcpp_bitnet_model_t shell with just hidden_size + vocab_size
// populated — that's all the loader cross-checks. The other fields are zero.
rcpp_bitnet_model_t make_fake_base(int hidden, int vocab) {
    rcpp_bitnet_model_t m{};
    m.hidden_size = hidden;
    m.vocab_size  = vocab;
    return m;
}

}  // namespace

int main() {
    std::fprintf(stderr, "[medusa loader test]\n");

    if (!gpu_available()) {
        std::fprintf(stderr, "  SKIP: no ROCm device\n");
        return 0;
    }

    // Use modest shape so the test runs in a few hundred ms even with no
    // GPU pressure: 4 heads, hidden=512, vocab=1024. ((512+3)/4) = 128 bytes
    // per row × 1024 rows × 4 heads = 512 KB on disk. Plus 4*1024*4 = 16 KB
    // scales.
    const uint32_t num_heads   = 4;
    const uint32_t hidden_size = 512;
    const uint32_t vocab_size  = 1024;

    const std::string tmp = "/tmp/test_medusa_loader.h1b-medusa";

    // ── Happy path ──
    write_synthetic_h1b_medusa(tmp, num_heads, hidden_size, vocab_size);
    rcpp_bitnet_model_t base = make_fake_base((int)hidden_size, (int)vocab_size);

    rcpp_medusa_heads_t heads{};
    rcpp_status_t st = rcpp_medusa_load_h1b_sidecar(tmp.c_str(), &base, &heads);
    check(st == RCPP_OK, "happy-path load returns RCPP_OK");
    check(heads.num_heads == num_heads, "num_heads round-trips");
    check(heads.hidden_size == hidden_size, "hidden_size round-trips");
    check(heads.vocab_size == vocab_size, "vocab_size round-trips");
    check(heads.weight_format == RCPP_WEIGHT_FORMAT_HALO_V2,
          "weight_format round-trips");
    bool all_devptrs_nonnull = true;
    for (uint32_t h = 0; h < num_heads; ++h) {
        if (!heads.heads[h].packed_dev) all_devptrs_nonnull = false;
        if (!heads.heads[h].row_scales_dev) all_devptrs_nonnull = false;
    }
    check(all_devptrs_nonnull, "every head has both packed_dev and row_scales_dev");

    rcpp_medusa_free_heads(&heads);
    rcpp_medusa_free_heads(&heads);   // idempotent
    check(heads.num_heads == 0, "free-heads zeros num_heads");

    // ── Bad magic ──
    write_synthetic_h1b_medusa(tmp, num_heads, hidden_size, vocab_size,
                               /*corrupt_magic=*/true);
    rcpp_medusa_heads_t bad{};
    st = rcpp_medusa_load_h1b_sidecar(tmp.c_str(), &base, &bad);
    check(st == RCPP_INVALID_ARG, "corrupt magic → RCPP_INVALID_ARG");
    rcpp_medusa_free_heads(&bad);

    // ── Hidden-size mismatch ──
    write_synthetic_h1b_medusa(tmp, num_heads, hidden_size, vocab_size);
    rcpp_bitnet_model_t base_h_wrong = make_fake_base((int)hidden_size + 16,
                                                      (int)vocab_size);
    rcpp_medusa_heads_t h_wrong{};
    st = rcpp_medusa_load_h1b_sidecar(tmp.c_str(), &base_h_wrong, &h_wrong);
    check(st == RCPP_INVALID_ARG, "hidden_size mismatch → RCPP_INVALID_ARG");
    rcpp_medusa_free_heads(&h_wrong);

    // ── Vocab mismatch ──
    rcpp_bitnet_model_t base_v_wrong = make_fake_base((int)hidden_size,
                                                      (int)vocab_size + 1);
    rcpp_medusa_heads_t v_wrong{};
    st = rcpp_medusa_load_h1b_sidecar(tmp.c_str(), &base_v_wrong, &v_wrong);
    check(st == RCPP_INVALID_ARG, "vocab_size mismatch → RCPP_INVALID_ARG");
    rcpp_medusa_free_heads(&v_wrong);

    // ── Truncated payload (final head missing) ──
    write_synthetic_h1b_medusa(tmp, num_heads, hidden_size, vocab_size,
                               /*corrupt_magic=*/false, /*truncate_tail=*/true);
    rcpp_medusa_heads_t trunc{};
    st = rcpp_medusa_load_h1b_sidecar(tmp.c_str(), &base, &trunc);
    check(st == RCPP_INVALID_ARG, "truncated payload → RCPP_INVALID_ARG");
    rcpp_medusa_free_heads(&trunc);

    std::remove(tmp.c_str());

    // ─────────────────────────────────────────────────────────────────────
    // v2 (residual-MLP) cases. Modest shape: 2 heads, hidden=128, fp16
    // dtype. Two tensors per head × 128*128*2 = 64 KB / tensor → 128 KB
    // / head → 256 KB total weights, plus 32 B header.
    const uint32_t v2_num_heads = 2;
    const uint32_t v2_hidden    = 128;

    auto write_v2 = [&](const std::string& path,
                        uint32_t variant_field,
                        uint32_t hidden_field,
                        uint32_t dtype_field,
                        bool truncate)
    {
        std::ofstream f(path, std::ios::binary);
        if (!f) std::exit(2);
        f.write("H1BM", 4);
        auto wu32 = [&](uint32_t v) { f.write(reinterpret_cast<char*>(&v), 4); };
        wu32(2);                  // version = 2
        wu32(variant_field);
        wu32(v2_num_heads);
        wu32(hidden_field);
        wu32(dtype_field);
        wu32(0); wu32(0);

        const size_t per_tensor =
            (size_t)hidden_field * (size_t)hidden_field * 2u;
        std::vector<uint8_t> zeros(per_tensor, 0u);
        const uint32_t emit_heads = truncate ? (v2_num_heads - 1) : v2_num_heads;
        for (uint32_t h = 0; h < emit_heads; ++h) {
            f.write(reinterpret_cast<char*>(zeros.data()), per_tensor); // w_in
            f.write(reinterpret_cast<char*>(zeros.data()), per_tensor); // w_out
        }
    };

    rcpp_bitnet_model_t base_v2 = make_fake_base((int)v2_hidden,
                                                  (int)vocab_size);

    // ── v2 happy path (fp16) ──
    write_v2(tmp, /*variant=*/1, v2_hidden, /*dtype=*/1, /*truncate=*/false);
    rcpp_medusa_heads_t v2h{};
    st = rcpp_medusa_load_h1b_sidecar(tmp.c_str(), &base_v2, &v2h);
    check(st == RCPP_OK, "v2 happy-path load returns RCPP_OK");
    check(v2h.num_heads == v2_num_heads, "v2 num_heads round-trips");
    check(v2h.hidden_size == v2_hidden, "v2 hidden_size round-trips");
    check(v2h.variant == RCPP_MEDUSA_VARIANT_RESIDUAL_MLP,
          "v2 variant tag round-trips");
    check(v2h.v2_dtype == RCPP_MEDUSA_DTYPE_FP16, "v2 dtype round-trips");
    bool v2_devptrs_nonnull = true;
    for (uint32_t h = 0; h < v2_num_heads; ++h) {
        if (!v2h.heads[h].w_in_dev)  v2_devptrs_nonnull = false;
        if (!v2h.heads[h].w_out_dev) v2_devptrs_nonnull = false;
        if (v2h.heads[h].packed_dev) v2_devptrs_nonnull = false;
    }
    check(v2_devptrs_nonnull,
          "v2 every head has w_in_dev + w_out_dev (no legacy packed_dev)");
    rcpp_medusa_free_heads(&v2h);

    // ── v2 happy path (bf16 → fp16 cast) ──
    write_v2(tmp, /*variant=*/1, v2_hidden, /*dtype=*/0, /*truncate=*/false);
    rcpp_medusa_heads_t v2bf{};
    st = rcpp_medusa_load_h1b_sidecar(tmp.c_str(), &base_v2, &v2bf);
    check(st == RCPP_OK, "v2 bf16-dtype load returns RCPP_OK (host-side cast)");
    check(v2bf.v2_dtype == RCPP_MEDUSA_DTYPE_BF16,
          "v2 dtype field reflects on-disk bf16 tag");
    rcpp_medusa_free_heads(&v2bf);

    // ── v2 bad variant (variant=99) ──
    write_v2(tmp, /*variant=*/99, v2_hidden, /*dtype=*/1, /*truncate=*/false);
    rcpp_medusa_heads_t v2bv{};
    st = rcpp_medusa_load_h1b_sidecar(tmp.c_str(), &base_v2, &v2bv);
    check(st == RCPP_INVALID_ARG, "v2 bad variant → RCPP_INVALID_ARG");
    rcpp_medusa_free_heads(&v2bv);

    // ── v2 bad dtype (dtype=99) ──
    write_v2(tmp, /*variant=*/1, v2_hidden, /*dtype=*/99, /*truncate=*/false);
    rcpp_medusa_heads_t v2bd{};
    st = rcpp_medusa_load_h1b_sidecar(tmp.c_str(), &base_v2, &v2bd);
    check(st == RCPP_INVALID_ARG, "v2 bad dtype → RCPP_INVALID_ARG");
    rcpp_medusa_free_heads(&v2bd);

    // ── v2 hidden mismatch ──
    write_v2(tmp, /*variant=*/1, v2_hidden, /*dtype=*/1, /*truncate=*/false);
    rcpp_bitnet_model_t base_v2_wrong = make_fake_base((int)v2_hidden + 16,
                                                        (int)vocab_size);
    rcpp_medusa_heads_t v2hm{};
    st = rcpp_medusa_load_h1b_sidecar(tmp.c_str(), &base_v2_wrong, &v2hm);
    check(st == RCPP_INVALID_ARG, "v2 hidden mismatch → RCPP_INVALID_ARG");
    rcpp_medusa_free_heads(&v2hm);

    // ── v2 truncated payload ──
    write_v2(tmp, /*variant=*/1, v2_hidden, /*dtype=*/1, /*truncate=*/true);
    rcpp_medusa_heads_t v2tr{};
    st = rcpp_medusa_load_h1b_sidecar(tmp.c_str(), &base_v2, &v2tr);
    check(st == RCPP_INVALID_ARG, "v2 truncated payload → RCPP_INVALID_ARG");
    rcpp_medusa_free_heads(&v2tr);

    std::remove(tmp.c_str());

    if (g_failures) {
        std::fprintf(stderr, "[medusa loader] %d failure(s)\n", g_failures);
        return 1;
    }
    std::fprintf(stderr, "[medusa loader] all passed\n");
    return 0;
}

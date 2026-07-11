// bench_kv_i8 — microbench for INT8 KV cache attention vs FP16 baseline.
//
// For each seq_len in {64, 256, 512, 1024, 2048}:
//   1. Generate random FP16 Q/K/V tensors in BitNet-2B-4T shape
//      (num_q_heads=20, num_kv_heads=5, head_dim=128).
//   2. Quantize K/V to INT8 + FP16 per-(pos, kv_head) scales using the new
//      rcpp_quantize_fp16_to_i8_rowscale helper.
//   3. Run both rcpp_kv_cache_attn_decode (fp16) and
//      rcpp_kv_cache_attn_decode_i8 on the same Q.
//   4. Measure wall µs/call for each (N_ITERS warm-up + N_TIMED timed).
//   5. Measure max-abs-diff between the two outputs (should be < 0.5 in fp16).
//
// Usage: bench_kv_i8 [out_path]
//   out_path defaults to "/home/bcloud/claude output/kv_i8_bench-<UTC>.txt"

#include "rocm_cpp/ck_gemm.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define HIP_CHECK(x) do { auto _s = (x); if (_s != hipSuccess) { \
    fprintf(stderr, "HIP err %d %s:%d\n", _s, __FILE__, __LINE__); std::exit(1); } } while (0)
#define RCPP_CHECK(x) do { auto _s = (x); if (_s != RCPP_OK) { \
    fprintf(stderr, "rcpp err %d %s:%d\n", (int)_s, __FILE__, __LINE__); std::exit(1); } } while (0)

struct HipBuf {
    void* p = nullptr;
    size_t n = 0;
    HipBuf() = default;
    explicit HipBuf(size_t bytes) : n(bytes) { HIP_CHECK(hipMalloc(&p, bytes)); }
    ~HipBuf() { if (p) hipFree(p); }
    HipBuf(HipBuf&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; o.n = 0; }
    HipBuf& operator=(HipBuf&& o) noexcept { if (this != &o) { if (p) hipFree(p); p = o.p; n = o.n; o.p = nullptr; o.n = 0; } return *this; }
    HipBuf(const HipBuf&) = delete;
    HipBuf& operator=(const HipBuf&) = delete;
};

static std::string utc_stamp() {
    std::time_t t = std::time(nullptr);
    std::tm* g = std::gmtime(&t);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04d%02d%02dT%02d%02d%02dZ",
                  g->tm_year + 1900, g->tm_mon + 1, g->tm_mday,
                  g->tm_hour, g->tm_min, g->tm_sec);
    return std::string(buf);
}

struct BenchRow {
    int seq_len;
    double us_fp16, us_i8;
    double gbs_fp16, gbs_i8;
    double max_abs_diff;
    double mean_abs_diff;
    double ref_max_abs;
};

int main(int argc, char** argv) {
    const int NH  = 20;   // BitNet-2B-4T
    const int NKV = 5;
    const int HD  = 128;
    const std::vector<int> seqs = {64, 256, 512, 1024, 2048};
    const int N_WARM  = 5;
    const int N_TIMED = 20;

    const std::string out_path = (argc > 1)
        ? std::string(argv[1])
        : ("/home/bcloud/claude output/kv_i8_bench-" + utc_stamp() + ".txt");

    std::mt19937 rng(0xC0DEF00D);
    // BitNet's post-RoPE Q/K magnitudes land roughly in [-4, 4]; V (post-proj)
    // is tighter. Use a unit-ish distribution scaled similarly so the scale
    // clamp (1e-8) never trips.
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<BenchRow> rows;

    for (int L : seqs) {
        const size_t Q_bytes    = (size_t)NH  * HD * sizeof(__half);
        const size_t KV_bytes   = (size_t)L * NKV * HD * sizeof(__half);
        const size_t KV_i8_bytes= (size_t)L * NKV * HD * sizeof(int8_t);
        const size_t SC_bytes   = (size_t)L * NKV * sizeof(__half);
        const size_t OUT_bytes  = (size_t)NH  * HD * sizeof(__half);

        // Host data
        std::vector<__half> h_Q(NH  * HD);
        std::vector<__half> h_K(L * NKV * HD);
        std::vector<__half> h_V(L * NKV * HD);
        for (auto& x : h_Q) x = __float2half(nd(rng));
        for (auto& x : h_K) x = __float2half(nd(rng));
        for (auto& x : h_V) x = __float2half(nd(rng));

        HipBuf d_Q(Q_bytes);
        HipBuf d_K(KV_bytes), d_V(KV_bytes);
        HipBuf d_K_i8(KV_i8_bytes), d_V_i8(KV_i8_bytes);
        HipBuf d_K_sc(SC_bytes), d_V_sc(SC_bytes);
        HipBuf d_out_fp16(OUT_bytes), d_out_i8(OUT_bytes);

        HIP_CHECK(hipMemcpy(d_Q.p, h_Q.data(), Q_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_K.p, h_K.data(), KV_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_V.p, h_V.data(), KV_bytes, hipMemcpyHostToDevice));

        // Quantize K/V: for each pos in 0..L-1, one call of num_rows=NKV, row_len=HD.
        // (The rowscale helper treats the full tensor as L*NKV rows of HD; the
        // scale array comes out [L*NKV] which matches our [L, NKV] layout.)
        RCPP_CHECK(rcpp_quantize_fp16_to_i8_rowscale(
            d_K.p, d_K_i8.p, d_K_sc.p, L * NKV, HD, nullptr));
        RCPP_CHECK(rcpp_quantize_fp16_to_i8_rowscale(
            d_V.p, d_V_i8.p, d_V_sc.p, L * NKV, HD, nullptr));

        const float scale = 1.0f / std::sqrt((float)HD);

        // ---- Correctness ----
        RCPP_CHECK(rcpp_kv_cache_attn_decode(
            d_Q.p, d_K.p, d_V.p, d_out_fp16.p,
            NH, NKV, HD, L, scale, nullptr));
        RCPP_CHECK(rcpp_kv_cache_attn_decode_i8(
            d_Q.p, d_K_i8.p, d_V_i8.p, d_K_sc.p, d_V_sc.p, d_out_i8.p,
            NH, NKV, HD, L, scale, nullptr));
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<__half> h_out_fp16(NH * HD), h_out_i8(NH * HD);
        HIP_CHECK(hipMemcpy(h_out_fp16.data(), d_out_fp16.p, OUT_bytes, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_out_i8.data(),   d_out_i8.p,   OUT_bytes, hipMemcpyDeviceToHost));
        double max_abs = 0.0, sum_abs = 0.0, ref_max = 0.0;
        for (size_t i = 0; i < h_out_fp16.size(); ++i) {
            double a = (double)(float)h_out_fp16[i];
            double b = (double)(float)h_out_i8[i];
            double d = std::fabs(a - b);
            if (d > max_abs) max_abs = d;
            sum_abs += d;
            double aa = std::fabs(a);
            if (aa > ref_max) ref_max = aa;
        }
        double mean_abs = sum_abs / h_out_fp16.size();

        // ---- Timing FP16 ----
        for (int i = 0; i < N_WARM; ++i) {
            RCPP_CHECK(rcpp_kv_cache_attn_decode(
                d_Q.p, d_K.p, d_V.p, d_out_fp16.p,
                NH, NKV, HD, L, scale, nullptr));
        }
        HIP_CHECK(hipDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_TIMED; ++i) {
            RCPP_CHECK(rcpp_kv_cache_attn_decode(
                d_Q.p, d_K.p, d_V.p, d_out_fp16.p,
                NH, NKV, HD, L, scale, nullptr));
        }
        HIP_CHECK(hipDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        double us_fp16 = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_TIMED;

        // ---- Timing INT8 ----
        for (int i = 0; i < N_WARM; ++i) {
            RCPP_CHECK(rcpp_kv_cache_attn_decode_i8(
                d_Q.p, d_K_i8.p, d_V_i8.p, d_K_sc.p, d_V_sc.p, d_out_i8.p,
                NH, NKV, HD, L, scale, nullptr));
        }
        HIP_CHECK(hipDeviceSynchronize());
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_TIMED; ++i) {
            RCPP_CHECK(rcpp_kv_cache_attn_decode_i8(
                d_Q.p, d_K_i8.p, d_V_i8.p, d_K_sc.p, d_V_sc.p, d_out_i8.p,
                NH, NKV, HD, L, scale, nullptr));
        }
        HIP_CHECK(hipDeviceSynchronize());
        t1 = std::chrono::high_resolution_clock::now();
        double us_i8 = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_TIMED;

        // KV DRAM traffic (read-once per decode): 2 * L * NKV * HD * size.
        // Q and output are constant-size, negligible at L >= 64.
        double fp16_bytes = 2.0 * L * NKV * HD * 2.0;
        double i8_bytes   = 2.0 * L * NKV * HD * 1.0 + 2.0 * L * NKV * 2.0;
        double gbs_fp16 = fp16_bytes / (us_fp16 * 1e-6) / 1e9;
        double gbs_i8   = i8_bytes   / (us_i8   * 1e-6) / 1e9;

        rows.push_back({L, us_fp16, us_i8, gbs_fp16, gbs_i8, max_abs, mean_abs, ref_max});

        fprintf(stderr,
                "[L=%5d] fp16=%7.2f us (%5.1f GB/s)  i8=%7.2f us (%5.1f GB/s)  "
                "speedup=%.2fx  max|diff|=%.4f (ref_max=%.2f, mean=%.4f)\n",
                L, us_fp16, gbs_fp16, us_i8, gbs_i8, us_fp16 / us_i8,
                max_abs, ref_max, mean_abs);
    }

    // Write report
    std::ostringstream os;
    os << "# KV int8 cache microbench\n";
    os << "# model: BitNet-2B-4T shape (NH=" << NH << " NKV=" << NKV
       << " HD=" << HD << ")  arch: gfx1151\n";
    os << "# kernels: rcpp_kv_cache_attn_decode (fp16) vs rcpp_kv_cache_attn_decode_i8\n";
    os << "# warm=" << N_WARM << " timed=" << N_TIMED
       << "  UTC=" << utc_stamp() << "\n";
    os << "#\n";
    os << "# seq_len  us_fp16  us_i8  gbs_fp16  gbs_i8  speedup  max_abs_diff  mean_abs_diff  ref_max_abs\n";
    for (auto& r : rows) {
        os << r.seq_len
           << "  " << r.us_fp16
           << "  " << r.us_i8
           << "  " << r.gbs_fp16
           << "  " << r.gbs_i8
           << "  " << (r.us_fp16 / r.us_i8)
           << "  " << r.max_abs_diff
           << "  " << r.mean_abs_diff
           << "  " << r.ref_max_abs
           << "\n";
    }
    std::ofstream f(out_path);
    if (!f) {
        fprintf(stderr, "cannot write %s — dumping to stdout\n", out_path.c_str());
        std::fputs(os.str().c_str(), stdout);
    } else {
        f << os.str();
        fprintf(stderr, "wrote %s\n", out_path.c_str());
    }
    return 0;
}

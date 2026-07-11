// bench_kv_fd — microbench for split-KV Flash-Decoding KV attention vs fp16.
//
// For each seq_len in {64, 256, 512, 1024, 2048}:
//   1. Generate random FP16 Q/K/V tensors in BitNet-2B-4T shape
//      (num_q_heads=20, num_kv_heads=5, head_dim=128).
//   2. Run rcpp_kv_cache_attn_decode       (baseline, one block per head)
//      and  rcpp_kv_cache_attn_decode_fd   (split-KV, pass1+reduce) on same Q.
//   3. Measure wall µs/call for each (N_WARM + N_TIMED).
//   4. Measure max-abs-diff fp16-vs-fd (should be < 0.05 fp16 units).
//
// Usage: bench_kv_fd [out_path]
//   out_path defaults to "/home/bcloud/claude output/kv_fd_bench-<UTC>.txt"

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
    double us_fp16, us_fd;
    double gbs_fp16, gbs_fd;
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
        : ("/home/bcloud/claude output/kv_fd_bench-" + utc_stamp() + ".txt");

    std::mt19937 rng(0xC0DEF00D);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<BenchRow> rows;

    for (int L : seqs) {
        const size_t Q_bytes   = (size_t)NH  * HD * sizeof(__half);
        const size_t KV_bytes  = (size_t)L * NKV * HD * sizeof(__half);
        const size_t OUT_bytes = (size_t)NH  * HD * sizeof(__half);

        std::vector<__half> h_Q(NH  * HD);
        std::vector<__half> h_K(L * NKV * HD);
        std::vector<__half> h_V(L * NKV * HD);
        for (auto& x : h_Q) x = __float2half(nd(rng));
        for (auto& x : h_K) x = __float2half(nd(rng));
        for (auto& x : h_V) x = __float2half(nd(rng));

        HipBuf d_Q(Q_bytes);
        HipBuf d_K(KV_bytes), d_V(KV_bytes);
        HipBuf d_out_fp16(OUT_bytes), d_out_fd(OUT_bytes);

        HIP_CHECK(hipMemcpy(d_Q.p, h_Q.data(), Q_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_K.p, h_K.data(), KV_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_V.p, h_V.data(), KV_bytes, hipMemcpyHostToDevice));

        const float scale = 1.0f / std::sqrt((float)HD);

        // ---- Correctness ----
        RCPP_CHECK(rcpp_kv_cache_attn_decode(
            d_Q.p, d_K.p, d_V.p, d_out_fp16.p,
            NH, NKV, HD, L, scale, nullptr));
        RCPP_CHECK(rcpp_kv_cache_attn_decode_fd(
            d_Q.p, d_K.p, d_V.p, d_out_fd.p,
            NH, NKV, HD, L, scale, nullptr));
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<__half> h_out_fp16(NH * HD), h_out_fd(NH * HD);
        HIP_CHECK(hipMemcpy(h_out_fp16.data(), d_out_fp16.p, OUT_bytes, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_out_fd.data(),   d_out_fd.p,   OUT_bytes, hipMemcpyDeviceToHost));
        double max_abs = 0.0, sum_abs = 0.0, ref_max = 0.0;
        for (size_t i = 0; i < h_out_fp16.size(); ++i) {
            double a = (double)(float)h_out_fp16[i];
            double b = (double)(float)h_out_fd[i];
            double d = std::fabs(a - b);
            if (d > max_abs) max_abs = d;
            sum_abs += d;
            double aa = std::fabs(a);
            if (aa > ref_max) ref_max = aa;
        }
        double mean_abs = sum_abs / h_out_fp16.size();

        // Correctness gate — STOP on diverged numerics per task rules.
        if (max_abs > 0.05) {
            fprintf(stderr,
                    "FATAL: fd kernel diverges from fp16 baseline at L=%d: "
                    "max|diff|=%.6f > 0.05 (ref_max=%.3f, mean=%.6f)\n",
                    L, max_abs, ref_max, mean_abs);
            return 2;
        }

        // ---- Timing FP16 baseline ----
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

        // ---- Timing split-KV FD ----
        for (int i = 0; i < N_WARM; ++i) {
            RCPP_CHECK(rcpp_kv_cache_attn_decode_fd(
                d_Q.p, d_K.p, d_V.p, d_out_fd.p,
                NH, NKV, HD, L, scale, nullptr));
        }
        HIP_CHECK(hipDeviceSynchronize());
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_TIMED; ++i) {
            RCPP_CHECK(rcpp_kv_cache_attn_decode_fd(
                d_Q.p, d_K.p, d_V.p, d_out_fd.p,
                NH, NKV, HD, L, scale, nullptr));
        }
        HIP_CHECK(hipDeviceSynchronize());
        t1 = std::chrono::high_resolution_clock::now();
        double us_fd = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_TIMED;

        // KV DRAM traffic (read-once per decode): 2 * L * NKV * HD * 2 bytes.
        double bytes = 2.0 * L * NKV * HD * 2.0;
        double gbs_fp16 = bytes / (us_fp16 * 1e-6) / 1e9;
        double gbs_fd   = bytes / (us_fd   * 1e-6) / 1e9;

        rows.push_back({L, us_fp16, us_fd, gbs_fp16, gbs_fd, max_abs, mean_abs, ref_max});

        fprintf(stderr,
                "[L=%5d] fp16=%7.2f us (%5.1f GB/s)  fd=%7.2f us (%5.1f GB/s)  "
                "speedup=%.2fx  max|diff|=%.5f (ref_max=%.2f, mean=%.5f)\n",
                L, us_fp16, gbs_fp16, us_fd, gbs_fd, us_fp16 / us_fd,
                max_abs, ref_max, mean_abs);
    }

    // Write report
    std::ostringstream os;
    os << "# KV split-KV Flash-Decoding microbench\n";
    os << "# model: BitNet-2B-4T shape (NH=" << NH << " NKV=" << NKV
       << " HD=" << HD << ")  arch: gfx1151  TILE=128\n";
    os << "# kernels: rcpp_kv_cache_attn_decode (fp16 baseline) vs "
          "rcpp_kv_cache_attn_decode_fd (split-KV FD)\n";
    os << "# warm=" << N_WARM << " timed=" << N_TIMED
       << "  UTC=" << utc_stamp() << "\n";
    os << "#\n";
    os << "# seq_len  us_fp16  us_fd  gbs_fp16  gbs_fd  speedup  "
          "max_abs_diff  mean_abs_diff  ref_max_abs\n";
    for (auto& r : rows) {
        os << r.seq_len
           << "  " << r.us_fp16
           << "  " << r.us_fd
           << "  " << r.gbs_fp16
           << "  " << r.gbs_fd
           << "  " << (r.us_fp16 / r.us_fd)
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

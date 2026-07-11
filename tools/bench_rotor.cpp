// bench_rotor — PlanarQuant-3 rotorquant KV-cache decode microbench.
//
// For each seq_len in {512, 1024, 2048, 4096}:
//   1. Build random FP16 Q/K/V at BitNet-2B-4T shape (NH=20, NKV=5, HD=128).
//   2. Run the baseline fp16 split-KV Flash-Decoding attention.
//   3. Requantize K/V -> 3-bit packed via rcpp_kv_requantize_pq3(_v).
//   4. Run rcpp_kv_cache_attn_decode_fd_pq3 on the packed caches.
//   5. Report per-kernel us/call, implied tok/s, per-token ms, and peak KV
//      bytes held on device.
//
// PPL comparison is NOT in scope here — wire it via
// `bitnet_decode --ppl --kv-rotor` once the kernel builds clean.
//
// Usage: bench_rotor [out_path]
//   out_path defaults to "/home/bcloud/claude output/rotor_bench-<UTC>.txt"

#include "rocm_cpp/ck_gemm.h"
#include "rocm_cpp/kv_rotorquant.h"

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
    HipBuf& operator=(HipBuf&& o) noexcept {
        if (this != &o) { if (p) hipFree(p); p = o.p; n = o.n; o.p = nullptr; o.n = 0; }
        return *this;
    }
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
    double us_fp16_attn;
    double us_pq3_attn;
    double us_pq3_requant;
    double us_pq3_total;      // requant(K) + requant(V) + attn
    double tok_per_s_fp16;
    double tok_per_s_pq3;
    double ms_per_tok_fp16;
    double ms_per_tok_pq3;
    size_t kv_bytes_fp16;
    size_t kv_bytes_pq3;
    double max_abs_diff;
    double mean_abs_diff;
    double ref_max_abs;
};

int main(int argc, char** argv) {
    const int NH  = 20;   // BitNet-2B-4T
    const int NKV = 5;
    const int HD  = 128;
    const std::vector<int> seqs = {512, 1024, 2048, 4096};
    const int N_WARM  = 5;
    const int N_TIMED = 20;
    const int LAYER_IDX = 7;   // arbitrary non-zero layer for seed variety

    const std::string out_path = (argc > 1)
        ? std::string(argv[1])
        : ("/home/bcloud/claude output/rotor_bench-" + utc_stamp() + ".txt");

    std::mt19937 rng(0xC0DEF00D);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<BenchRow> rows;

    for (int L : seqs) {
        const size_t Q_bytes    = (size_t)NH  * HD * sizeof(__half);
        const size_t KV_bytes   = (size_t)L * NKV * HD * sizeof(__half);
        const size_t OUT_bytes  = (size_t)NH  * HD * sizeof(__half);
        const size_t PQ3_row    = (size_t)(HD * 3 / 8);
        const size_t KV_pq3_bytes = (size_t)L * NKV * PQ3_row;

        std::vector<__half> h_Q(NH * HD);
        std::vector<__half> h_K(L * NKV * HD);
        std::vector<__half> h_V(L * NKV * HD);
        for (auto& x : h_Q) x = __float2half(nd(rng));
        for (auto& x : h_K) x = __float2half(nd(rng));
        for (auto& x : h_V) x = __float2half(nd(rng));

        HipBuf d_Q(Q_bytes);
        HipBuf d_K(KV_bytes),  d_V(KV_bytes);
        HipBuf d_K_pq3(KV_pq3_bytes), d_V_pq3(KV_pq3_bytes);
        HipBuf d_out_fp16(OUT_bytes), d_out_pq3(OUT_bytes);

        HIP_CHECK(hipMemcpy(d_Q.p, h_Q.data(), Q_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_K.p, h_K.data(), KV_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_V.p, h_V.data(), KV_bytes, hipMemcpyHostToDevice));

        const float scale = 1.0f / std::sqrt((float)HD);

        // ---- Correctness: fp16 baseline vs pq3 round-trip --------------
        RCPP_CHECK(rcpp_kv_cache_attn_decode_fd(
            d_Q.p, d_K.p, d_V.p, d_out_fp16.p,
            NH, NKV, HD, L, scale, nullptr));

        rcpp_kv_requantize_pq3  (d_K.p, d_K_pq3.p, L, NKV, HD, LAYER_IDX, nullptr);
        rcpp_kv_requantize_pq3_v(d_V.p, d_V_pq3.p, L, NKV, HD, LAYER_IDX, nullptr);
        int rc = rcpp_kv_cache_attn_decode_fd_pq3(
            d_Q.p, d_K_pq3.p, d_V_pq3.p, d_out_pq3.p,
            NH, NKV, HD, L, LAYER_IDX, scale, nullptr);
        if (rc != 0) {
            fprintf(stderr, "rcpp_kv_cache_attn_decode_fd_pq3 rc=%d at L=%d\n", rc, L);
            return 1;
        }
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<__half> h_out_fp16(NH * HD), h_out_pq3(NH * HD);
        HIP_CHECK(hipMemcpy(h_out_fp16.data(), d_out_fp16.p, OUT_bytes, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_out_pq3.data(),  d_out_pq3.p,  OUT_bytes, hipMemcpyDeviceToHost));
        double max_abs = 0.0, sum_abs = 0.0, ref_max = 0.0;
        for (size_t i = 0; i < h_out_fp16.size(); ++i) {
            double a = (double)(float)h_out_fp16[i];
            double b = (double)(float)h_out_pq3[i];
            double d = std::fabs(a - b);
            if (d > max_abs) max_abs = d;
            sum_abs += d;
            double aa = std::fabs(a);
            if (aa > ref_max) ref_max = aa;
        }
        double mean_abs = sum_abs / h_out_fp16.size();

        // ---- Timing fp16 FD attention ---------------------------------
        for (int i = 0; i < N_WARM; ++i) {
            RCPP_CHECK(rcpp_kv_cache_attn_decode_fd(
                d_Q.p, d_K.p, d_V.p, d_out_fp16.p,
                NH, NKV, HD, L, scale, nullptr));
        }
        HIP_CHECK(hipDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_TIMED; ++i) {
            RCPP_CHECK(rcpp_kv_cache_attn_decode_fd(
                d_Q.p, d_K.p, d_V.p, d_out_fp16.p,
                NH, NKV, HD, L, scale, nullptr));
        }
        HIP_CHECK(hipDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        double us_fp16 = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_TIMED;

        // ---- Timing pq3 FD attention (attn only, caches already packed) ----
        for (int i = 0; i < N_WARM; ++i) {
            rcpp_kv_cache_attn_decode_fd_pq3(
                d_Q.p, d_K_pq3.p, d_V_pq3.p, d_out_pq3.p,
                NH, NKV, HD, L, LAYER_IDX, scale, nullptr);
        }
        HIP_CHECK(hipDeviceSynchronize());
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_TIMED; ++i) {
            rcpp_kv_cache_attn_decode_fd_pq3(
                d_Q.p, d_K_pq3.p, d_V_pq3.p, d_out_pq3.p,
                NH, NKV, HD, L, LAYER_IDX, scale, nullptr);
        }
        HIP_CHECK(hipDeviceSynchronize());
        t1 = std::chrono::high_resolution_clock::now();
        double us_pq3 = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_TIMED;

        // ---- Timing the requantize side (one (pos, kv_head) row, NOT the
        // full seq) — this is the per-token write cost. Requantize ONE
        // new token's K+V at the tail of the cache each iteration.
        for (int i = 0; i < N_WARM; ++i) {
            rcpp_kv_requantize_pq3  (d_K.p, d_K_pq3.p, /*seq_len=*/1, NKV, HD, LAYER_IDX, nullptr);
            rcpp_kv_requantize_pq3_v(d_V.p, d_V_pq3.p, /*seq_len=*/1, NKV, HD, LAYER_IDX, nullptr);
        }
        HIP_CHECK(hipDeviceSynchronize());
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_TIMED; ++i) {
            rcpp_kv_requantize_pq3  (d_K.p, d_K_pq3.p, 1, NKV, HD, LAYER_IDX, nullptr);
            rcpp_kv_requantize_pq3_v(d_V.p, d_V_pq3.p, 1, NKV, HD, LAYER_IDX, nullptr);
        }
        HIP_CHECK(hipDeviceSynchronize());
        t1 = std::chrono::high_resolution_clock::now();
        double us_requant = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_TIMED;

        double us_total_pq3 = us_pq3 + us_requant;

        // tok/s approximation — attention is run per token in decode. A full
        // tok latency = attn (seq_len=pos+1) + requant(1). For the fp16 path
        // the KV write is a tiny device-to-device memcpy — count it as 0 us
        // for the comparison (it's within noise).
        double tok_per_s_fp16 = 1e6 / us_fp16;
        double tok_per_s_pq3  = 1e6 / us_total_pq3;
        double ms_per_tok_fp16 = us_fp16       / 1000.0;
        double ms_per_tok_pq3  = us_total_pq3  / 1000.0;

        // KV bytes (both K and V).
        size_t kv_bytes_fp16 = 2 * KV_bytes;
        size_t kv_bytes_pq3  = 2 * KV_pq3_bytes;

        rows.push_back({
            L, us_fp16, us_pq3, us_requant, us_total_pq3,
            tok_per_s_fp16, tok_per_s_pq3,
            ms_per_tok_fp16, ms_per_tok_pq3,
            kv_bytes_fp16, kv_bytes_pq3,
            max_abs, mean_abs, ref_max
        });

        fprintf(stderr,
                "[L=%5d] fp16_attn=%7.2f us  pq3_attn=%7.2f us  pq3_requant=%7.2f us  "
                "tok/s fp16=%.1f pq3=%.1f  KV bytes fp16=%zu pq3=%zu (%.2fx)  "
                "max|diff|=%.4f (ref=%.2f, mean=%.4f)\n",
                L, us_fp16, us_pq3, us_requant, tok_per_s_fp16, tok_per_s_pq3,
                kv_bytes_fp16, kv_bytes_pq3,
                (double)kv_bytes_fp16 / (double)kv_bytes_pq3,
                max_abs, ref_max, mean_abs);
    }

    std::ostringstream os;
    os << "# rotorquant PlanarQuant-3 KV-cache microbench\n";
    os << "# model: BitNet-2B-4T shape (NH=" << NH << " NKV=" << NKV
       << " HD=" << HD << ")  arch: gfx1151  TILE=128  layer_idx=" << LAYER_IDX << "\n";
    os << "# kernels: rcpp_kv_cache_attn_decode_fd (fp16) vs "
          "rcpp_kv_cache_attn_decode_fd_pq3 (+ rcpp_kv_requantize_pq3*)\n";
    os << "# warm=" << N_WARM << " timed=" << N_TIMED
       << "  UTC=" << utc_stamp() << "\n#\n";
    os << "# seq_len  us_fp16_attn  us_pq3_attn  us_pq3_requant  us_pq3_total  "
          "tok/s_fp16  tok/s_pq3  ms/tok_fp16  ms/tok_pq3  "
          "kv_bytes_fp16  kv_bytes_pq3  max_abs_diff  mean_abs_diff  ref_max_abs\n";
    for (auto& r : rows) {
        os << r.seq_len
           << "  " << r.us_fp16_attn
           << "  " << r.us_pq3_attn
           << "  " << r.us_pq3_requant
           << "  " << r.us_pq3_total
           << "  " << r.tok_per_s_fp16
           << "  " << r.tok_per_s_pq3
           << "  " << r.ms_per_tok_fp16
           << "  " << r.ms_per_tok_pq3
           << "  " << r.kv_bytes_fp16
           << "  " << r.kv_bytes_pq3
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

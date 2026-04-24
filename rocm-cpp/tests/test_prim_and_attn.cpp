// Correctness tests for the MLX-free decode kernels:
//   quantize_fp16_to_i8, rmsnorm, rope, silu_glu, embedding_lookup,
//   kv_cache_attn_decode.
//
// Each tested against a scalar CPU reference.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "rocm_cpp/ck_gemm.h"

#define HIP_OK(e) do { auto _s=(e); if(_s!=hipSuccess){fprintf(stderr,"HIP %d %s:%d\n",_s,__FILE__,__LINE__); std::abort();}} while(0)

struct Result { const char* name; bool pass; float max_abs; float max_rel; };
static std::vector<Result> results;

static void check(const char* name, float max_abs, float threshold) {
    bool pass = max_abs < threshold;
    results.push_back({name, pass, max_abs, 0.0f});
    printf("  %-28s : max_abs=%.6f  threshold=%.4f  %s\n",
           name, max_abs, threshold, pass ? "PASS" : "FAIL");
}

static float diff_max(const std::vector<_Float16>& a, const std::vector<_Float16>& b) {
    float m = 0;
    for (size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::fabs((float)a[i] - (float)b[i]));
    return m;
}
static float diff_max_f(const std::vector<float>& a, const std::vector<_Float16>& b) {
    float m = 0;
    for (size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::fabs(a[i] - (float)b[i]));
    return m;
}

static void test_quant() {
    const int K = 2560;
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> rd(-0.5f, 0.5f);
    std::vector<_Float16> x(K);
    for (auto& v : x) v = (_Float16)rd(rng);

    _Float16* dX; int8_t* dI8; float* dScale;
    HIP_OK(hipMalloc(&dX, K * 2)); HIP_OK(hipMalloc(&dI8, K));
    HIP_OK(hipMalloc(&dScale, 4));
    HIP_OK(hipMemcpy(dX, x.data(), K*2, hipMemcpyHostToDevice));
    rcpp_quantize_fp16_to_i8(dX, dI8, dScale, K, nullptr);
    HIP_OK(hipDeviceSynchronize());

    std::vector<int8_t> i8(K);
    float scale;
    HIP_OK(hipMemcpy(i8.data(), dI8, K, hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(&scale, dScale, 4, hipMemcpyDeviceToHost));

    // CPU ref
    float xmax = 0.0f;
    for (auto v : x) xmax = std::max(xmax, std::fabs((float)v));
    float ref_scale = xmax / 127.0f;

    float max_scale_err = std::fabs(scale - ref_scale);
    // Reconstruct and check max error
    float max_recon = 0.0f;
    for (int k = 0; k < K; ++k) {
        float recon = (float)i8[k] * scale;
        max_recon = std::max(max_recon, std::fabs(recon - (float)x[k]));
    }
    printf("  %-28s : scale_err=%.2e  recon_max=%.4f  (scale=%.4f  |x|max=%.4f)\n",
           "quant_fp16_to_i8", max_scale_err, max_recon, scale, xmax);
    results.push_back({"quant_fp16_to_i8", max_scale_err < 1e-4f && max_recon < scale,
                       max_recon, 0});

    HIP_OK(hipFree(dX)); HIP_OK(hipFree(dI8)); HIP_OK(hipFree(dScale));
}

static void test_rmsnorm() {
    const int K = 2560;
    const float eps = 1e-6f;
    std::mt19937 rng(2);
    std::uniform_real_distribution<float> rd(-1.0f, 1.0f);
    std::vector<_Float16> x(K), w(K);
    for (auto& v : x) v = (_Float16)rd(rng);
    for (auto& v : w) v = (_Float16)(rd(rng) * 0.5f + 1.0f);

    // CPU reference
    double sumsq = 0;
    for (auto v : x) sumsq += (double)v * (double)v;
    double mean = sumsq / K;
    double inv = 1.0 / std::sqrt(mean + eps);
    std::vector<float> ref(K);
    for (int k = 0; k < K; ++k) ref[k] = (float)x[k] * (float)inv * (float)w[k];

    _Float16 *dX, *dW, *dY;
    HIP_OK(hipMalloc(&dX, K*2)); HIP_OK(hipMalloc(&dW, K*2)); HIP_OK(hipMalloc(&dY, K*2));
    HIP_OK(hipMemcpy(dX, x.data(), K*2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dW, w.data(), K*2, hipMemcpyHostToDevice));
    rcpp_rmsnorm_fp16(dX, dW, dY, eps, K, nullptr);
    HIP_OK(hipDeviceSynchronize());

    std::vector<_Float16> y(K);
    HIP_OK(hipMemcpy(y.data(), dY, K*2, hipMemcpyDeviceToHost));
    check("rmsnorm_fp16", diff_max_f(ref, y), 0.02f);

    HIP_OK(hipFree(dX)); HIP_OK(hipFree(dW)); HIP_OK(hipFree(dY));
}

static void test_rope() {
    const int num_heads = 20, head_dim = 128;
    const int pos = 17;
    const float theta = 500000.0f;
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> rd(-1.0f, 1.0f);
    std::vector<_Float16> x(num_heads * head_dim);
    for (auto& v : x) v = (_Float16)rd(rng);

    // CPU reference
    std::vector<float> ref(x.size());
    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < head_dim / 2; ++i) {
            float freq = 1.0f / std::pow(theta, 2.0f * (float)i / (float)head_dim);
            float angle = (float)pos * freq;
            float c = std::cos(angle), s = std::sin(angle);
            float x0 = (float)x[h*head_dim + 2*i + 0];
            float x1 = (float)x[h*head_dim + 2*i + 1];
            ref[h*head_dim + 2*i + 0] = c*x0 - s*x1;
            ref[h*head_dim + 2*i + 1] = s*x0 + c*x1;
        }
    }

    _Float16* dX;
    HIP_OK(hipMalloc(&dX, x.size()*2));
    HIP_OK(hipMemcpy(dX, x.data(), x.size()*2, hipMemcpyHostToDevice));
    rcpp_rope_fp16(dX, pos, theta, num_heads, head_dim, nullptr);
    HIP_OK(hipDeviceSynchronize());
    std::vector<_Float16> y(x.size());
    HIP_OK(hipMemcpy(y.data(), dX, y.size()*2, hipMemcpyDeviceToHost));
    check("rope_fp16", diff_max_f(ref, y), 0.01f);
    HIP_OK(hipFree(dX));
}

static void test_silu_glu() {
    const int N = 6912;
    std::mt19937 rng(4);
    std::uniform_real_distribution<float> rd(-2.0f, 2.0f);
    std::vector<_Float16> u(N), g(N);
    for (auto& v : u) v = (_Float16)rd(rng);
    for (auto& v : g) v = (_Float16)rd(rng);

    std::vector<float> ref(N);
    for (int i = 0; i < N; ++i) {
        float uv = (float)u[i];
        float silu = uv / (1.0f + std::exp(-uv));
        ref[i] = silu * (float)g[i];
    }

    _Float16 *dU, *dG, *dY;
    HIP_OK(hipMalloc(&dU, N*2)); HIP_OK(hipMalloc(&dG, N*2)); HIP_OK(hipMalloc(&dY, N*2));
    HIP_OK(hipMemcpy(dU, u.data(), N*2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dG, g.data(), N*2, hipMemcpyHostToDevice));
    rcpp_silu_glu_fp16(dU, dG, dY, N, nullptr);
    HIP_OK(hipDeviceSynchronize());

    std::vector<_Float16> y(N);
    HIP_OK(hipMemcpy(y.data(), dY, N*2, hipMemcpyDeviceToHost));
    check("silu_glu_fp16", diff_max_f(ref, y), 0.02f);

    HIP_OK(hipFree(dU)); HIP_OK(hipFree(dG)); HIP_OK(hipFree(dY));
}

static void test_embedding() {
    const int vocab = 128256, hidden = 2560;
    const int tok = 12345;
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> rd(-0.2f, 0.2f);
    std::vector<_Float16> embed((size_t)vocab * hidden);
    // Only init the row we look up — too expensive otherwise.
    for (int k = 0; k < hidden; ++k) {
        embed[(size_t)tok * hidden + k] = (_Float16)rd(rng);
    }

    _Float16 *dE, *dY;
    HIP_OK(hipMalloc(&dE, embed.size()*2));
    HIP_OK(hipMalloc(&dY, hidden*2));
    HIP_OK(hipMemcpy(dE, embed.data(), embed.size()*2, hipMemcpyHostToDevice));
    rcpp_embedding_lookup_fp16(dE, tok, dY, hidden, nullptr);
    HIP_OK(hipDeviceSynchronize());

    std::vector<_Float16> y(hidden);
    HIP_OK(hipMemcpy(y.data(), dY, hidden*2, hipMemcpyDeviceToHost));
    float m = 0;
    for (int k = 0; k < hidden; ++k) {
        m = std::max(m, std::fabs((float)y[k] - (float)embed[(size_t)tok*hidden + k]));
    }
    check("embedding_lookup_fp16", m, 1e-6f);
    HIP_OK(hipFree(dE)); HIP_OK(hipFree(dY));
}

static void test_kv_attn() {
    // BitNet-2B-4T shape
    const int nh = 20, nkv = 5, hd = 128, seq_len = 64;
    const int gqa = nh / nkv;
    const float scale = 1.0f / std::sqrt((float)hd);

    std::mt19937 rng(6);
    std::uniform_real_distribution<float> rd(-0.3f, 0.3f);
    std::vector<_Float16> Q((size_t)nh * hd);
    std::vector<_Float16> Kc((size_t)seq_len * nkv * hd);
    std::vector<_Float16> Vc((size_t)seq_len * nkv * hd);
    for (auto& v : Q)  v = (_Float16)rd(rng);
    for (auto& v : Kc) v = (_Float16)rd(rng);
    for (auto& v : Vc) v = (_Float16)rd(rng);

    // CPU reference
    std::vector<float> ref((size_t)nh * hd, 0.0f);
    for (int h = 0; h < nh; ++h) {
        int kv_h = h / gqa;
        std::vector<float> S(seq_len);
        float S_max = -INFINITY;
        for (int t = 0; t < seq_len; ++t) {
            float s = 0;
            for (int d = 0; d < hd; ++d) {
                s += (float)Q[h*hd + d] *
                     (float)Kc[(t*nkv + kv_h)*hd + d];
            }
            s *= scale;
            S[t] = s;
            S_max = std::max(S_max, s);
        }
        float Z = 0;
        for (int t = 0; t < seq_len; ++t) {
            S[t] = std::exp(S[t] - S_max);
            Z += S[t];
        }
        for (int t = 0; t < seq_len; ++t) S[t] /= Z;
        for (int d = 0; d < hd; ++d) {
            float o = 0;
            for (int t = 0; t < seq_len; ++t) {
                o += S[t] * (float)Vc[(t*nkv + kv_h)*hd + d];
            }
            ref[h*hd + d] = o;
        }
    }

    _Float16 *dQ, *dK, *dV, *dO;
    HIP_OK(hipMalloc(&dQ, Q.size()*2));  HIP_OK(hipMemcpy(dQ, Q.data(), Q.size()*2, hipMemcpyHostToDevice));
    HIP_OK(hipMalloc(&dK, Kc.size()*2)); HIP_OK(hipMemcpy(dK, Kc.data(), Kc.size()*2, hipMemcpyHostToDevice));
    HIP_OK(hipMalloc(&dV, Vc.size()*2)); HIP_OK(hipMemcpy(dV, Vc.data(), Vc.size()*2, hipMemcpyHostToDevice));
    HIP_OK(hipMalloc(&dO, (size_t)nh*hd*2));

    rcpp_kv_cache_attn_decode(dQ, dK, dV, dO, nh, nkv, hd, seq_len, scale, nullptr);
    HIP_OK(hipDeviceSynchronize());

    std::vector<_Float16> y((size_t)nh*hd);
    HIP_OK(hipMemcpy(y.data(), dO, y.size()*2, hipMemcpyDeviceToHost));
    check("kv_cache_attn_decode", diff_max_f(ref, y), 0.01f);

    HIP_OK(hipFree(dQ)); HIP_OK(hipFree(dK)); HIP_OK(hipFree(dV)); HIP_OK(hipFree(dO));
}

// ---------------------------------------------------------------------------
// Argmax over fp16 logits (paired with forward_token_greedy's D→H-skip
// fast path). Seeds a ~vocab-sized fp16 vector with uniformly random
// values and asserts the kernel's returned index matches
// std::max_element's baseline. Exact-integer match — argmax on a
// uniquely-maxed random fp16 vector is deterministic under
// fast-math reductions (no summations, just pairwise `>`).
// ---------------------------------------------------------------------------
static void test_argmax_fp16_top1() {
    const int V = 128256;  // Llama-3 vocab, the production fast-path size.
    std::mt19937 rng(0xA17EA5u);
    std::uniform_real_distribution<float> rd(-4.0f, 4.0f);
    std::vector<_Float16> logits(V);
    for (auto& v : logits) v = (_Float16)rd(rng);

    // Plant a clearly-separated max so the reduction has a unique
    // winner under fp16 rounding.
    const int planted_idx = 77337;
    logits[planted_idx] = (_Float16)8.0f;

    _Float16* dLogits; int* dIdx;
    HIP_OK(hipMalloc(&dLogits, V * sizeof(_Float16)));
    HIP_OK(hipMalloc(&dIdx, sizeof(int)));
    HIP_OK(hipMemcpy(dLogits, logits.data(), V * sizeof(_Float16),
                     hipMemcpyHostToDevice));

    auto st = rcpp_argmax_fp16_top1(dLogits, dIdx, V, nullptr);
    if (st != RCPP_OK) {
        fprintf(stderr, "rcpp_argmax_fp16_top1 returned %d\n", (int)st);
        std::abort();
    }
    HIP_OK(hipDeviceSynchronize());

    int device_idx = -1;
    HIP_OK(hipMemcpy(&device_idx, dIdx, sizeof(int), hipMemcpyDeviceToHost));

    // Reference: host-side linear scan.
    int ref_idx = 0;
    float ref_val = (float)logits[0];
    for (int i = 1; i < V; ++i) {
        float v = (float)logits[i];
        if (v > ref_val) { ref_val = v; ref_idx = i; }
    }

    bool pass = (device_idx == ref_idx) && (device_idx == planted_idx);
    printf("  %-28s : device=%d ref=%d planted=%d  %s\n",
           "argmax_fp16_top1", device_idx, ref_idx, planted_idx,
           pass ? "PASS" : "FAIL");
    results.push_back({"argmax_fp16_top1", pass, 0.0f, 0.0f});

    HIP_OK(hipFree(dLogits));
    HIP_OK(hipFree(dIdx));
}

int main() {
    printf("=== rocm-cpp prim + attention kernel tests ===\n");
    test_quant();
    test_rmsnorm();
    test_rope();
    test_silu_glu();
    test_embedding();
    test_kv_attn();
    test_argmax_fp16_top1();
    int fails = 0;
    for (auto& r : results) if (!r.pass) ++fails;
    printf("\n%zu tests: %d pass / %d fail\n", results.size(), (int)results.size() - fails, fails);
    return fails ? 1 : 0;
}

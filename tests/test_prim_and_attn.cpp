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
    printf("  %-28s : max_abs=%.6f  threshold=%.4f  %s\n", name, max_abs, threshold, pass ? "PASS" : "FAIL");
}

static float diff_max(const std::vector<_Float16>& a, const std::vector<_Float16>& b) {
    float m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs((float)a[i] - (float)b[i]));
    return m;
}
static float diff_max_f(const std::vector<float>& a, const std::vector<_Float16>& b) {
    float m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - (float)b[i]));
    return m;
}

static void test_quant() { /* unchanged */ }
static void test_rmsnorm() { /* unchanged */ }

static void test_rope() {
    const int num_heads = 20, head_dim = 128;
    const int pos = 17;
    const float theta = 500000.0f;
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> rd(-1.0f, 1.0f);
    std::vector<_Float16> x(num_heads * head_dim);
    for (auto& v : x) v = (_Float16)rd(rng);

    // CPU reference -- split-half RoPE (matches kernel convention):
    //   row[i] <- cos*row[i] - sin*row[i+hd/2]
    //   row[i+hd/2] <- sin*row[i]_orig + cos*row[i+hd/2]
    std::vector<float> ref(x.size());
    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < head_dim / 2; ++i) {
            float freq = 1.0f / std::pow(theta, 2.0f * (float)i / (float)head_dim);
            float angle = (float)pos * freq;
            float c = std::cos(angle), s = std::sin(angle);
            float x0 = (float)x[h*head_dim + i];
            float x1 = (float)x[h*head_dim + i + head_dim/2];
            ref[h*head_dim + i] = c*x0 - s*x1;
            ref[h*head_dim + i + head_dim/2] = s*x0 + c*x1;
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

static void test_silu_glu() { /* unchanged */ }
static void test_embedding() { /* unchanged */ }
static void test_kv_attn() { /* unchanged */ }

int main() {
    int dev_count = 0;
    if (hipGetDeviceCount(&dev_count) != hipSuccess || dev_count == 0) {
        fprintf(stderr, "no HIP device available, skipping\n");
        return 77;
    }
    printf("=== rocm-cpp prim + attention kernel tests ===\n");
    test_quant(); test_rmsnorm(); test_rope();
    test_silu_glu(); test_embedding(); test_kv_attn();
    int fails = 0;
    for (auto& r : results) if (!r.pass) ++fails;
    printf("\n%zu tests: %d pass / %d fail\n", results.size(), (int)results.size() - fails, fails);
    return fails ? 1 : 0;
}
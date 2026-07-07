// test_gemm_quant.cpp — Verify activation quantization/dequantization correctness
// Build: g++ -std=c++23 -O2 -o test_gemm_quant test_gemm_quant.cpp -lm && ./test_gemm_quant
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static constexpr float FIXED_ASCALE = 8.0f / 127.0f;

// ─── Replicas of the engine quant/dequant logic ─────────────────────────

// Universal/split engine: single global weight scale
static float packB_global(const float* w, int K, int N, std::vector<int8_t>& Bm) {
    float amax = 0;
    for (int i = 0; i < K * N; i++) {
        float a = fabsf(w[i]);
        if (std::isfinite(a) && a > amax) amax = a;
    }
    if (amax < 1e-12f) amax = 1.0f;
    float Bscale = amax / 127.0f;
    float is = 127.0f / amax;
    Bm.resize(K * N);
    for (int i = 0; i < K * N; i++) {
        float v = w[i];
        int x = (int)roundf(v * is);
        if (x > 127) x = 127; else if (x < -127) x = -127;
        Bm[i] = (int8_t)x;
    }
    return Bscale;
}

// Server engine: per-column weight scale
static float packB_percol(const float* w, int K, int N, std::vector<int8_t>& Bm, std::vector<float>& col_scale) {
    col_scale.resize(N);
    float amax = 0;
    for (int n = 0; n < N; n++) {
        float camax = 0;
        for (int k = 0; k < K; k++) {
            float a = fabsf(w[k * N + n]);
            if (std::isfinite(a) && a > camax) camax = a;
        }
        if (camax > amax) amax = camax;
        col_scale[n] = camax > 1e-12f ? camax / 127.0f : 1.0f;
    }
    float Bscale = amax / 127.0f;
    Bm.resize(K * N);
    for (int n = 0; n < N; n++) {
        float is = 1.0f / col_scale[n];
        for (int k = 0; k < K; k++) {
            float v = w[k * N + n];
            if (!std::isfinite(v)) v = 0;
            int x = (int)roundf(v * is);
            if (x > 127) x = 127; else if (x < -127) x = -127;
            Bm[k * N + n] = (int8_t)x;
        }
    }
    return Bscale;
}

// Activation quantization (same as engine go() path)
static void quantA(const float* A, int M, int K, std::vector<int8_t>& Am, int KD) {
    // ascale = amax/127, so ais = 1/ascale = 127/amax
    float ais = 1.0f / FIXED_ASCALE;
    Am.resize(M * KD);
    memset(Am.data(), 0, M * KD);
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++) {
            float v = A[m * K + k];
            if (!std::isfinite(v)) v = 0;
            int q = (int)roundf(v * ais);
            if (q > 127) q = 127; else if (q < -127) q = -127;
            Am[m * KD + k] = (int8_t)q;
        }
}

// INT8 GEMM (simulates NPU)
static void i8gemm(const int8_t* A, const int8_t* B, int M, int K, int N, int KD, std::vector<int32_t>& C) {
    C.assign(M * N, 0);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++)
                sum += (int32_t)A[m * KD + k] * (int32_t)B[k * N + n];
            C[m * N + n] = sum;
        }
}

// Dequant output (universal/split style)
static void dequant_global(const int32_t* C, int M, int N, float Bscale, float* out) {
    float cs = FIXED_ASCALE * Bscale;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float val = (float)C[m * N + n] * cs;
            if (!std::isfinite(val)) val = 0;
            out[m * N + n] = val;
        }
}

// Dequant output (server style: per-column)
static void dequant_percol(const int32_t* C, int M, int N, const std::vector<float>& col_scale, float* out) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float val = (float)C[m * N + n] * FIXED_ASCALE * col_scale[n];
            if (!std::isfinite(val)) val = 0;
            out[m * N + n] = val;
        }
}

// ─── CPU reference GEMM ─────────────────────────────────────────────────

static void cpu_gemm(const float* A, const float* B, int M, int K, int N, float* C) {
    memset(C, 0, (size_t)M * N * sizeof(float));
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++)
                C[m * N + n] += A[m * K + k] * B[k * N + n];
}

// ─── Tests ───────────────────────────────────────────────────────────────

static bool check_close(const float* a, const float* b, int n, float rtol, const char* name) {
    float max_err = 0, max_val = 0;
    int worst = 0;
    for (int i = 0; i < n; i++) {
        float err = fabsf(a[i] - b[i]);
        if (err > max_err) { max_err = err; worst = i; }
        if (fabsf(a[i]) > max_val) max_val = fabsf(a[i]);
    }
    float rel = max_val > 0 ? max_err / max_val : max_err;
    bool pass = rel <= rtol;
    printf("  %s: max_err=%.6f max_val=%.4f rel=%.2e %s (worst @%d: %.4f vs %.4f)\n",
        name, max_err, max_val, rel, pass ? "OK" : "FAIL", worst, a[worst], b[worst]);
    return pass;
}

int main() {
    int errors = 0;
    srand(42);

    for (int t = 0; t < 20; t++) {
        int M = 1 + (rand() % 16);
        int K = 32 + (rand() % 512);
        int N = 32 + (rand() % 2048);
        int KD = ((K + 31) / 32) * 32; // round up to 32

        // Generate random weights and activations
        std::vector<float> A(M * K), B(K * N);
        for (int i = 0; i < M * K; i++) A[i] = (float)rand() / RAND_MAX * 8.0f - 4.0f;  // [-4,4]
        for (int i = 0; i < K * N; i++) B[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;   // [-1,1]

        // CPU reference
        std::vector<float> C_ref(M * N);
        cpu_gemm(A.data(), B.data(), M, K, N, C_ref.data());

        // ── Global scale (universal/split engine) ──
        std::vector<int8_t> Bq_global;
        float Bscale = packB_global(B.data(), K, N, Bq_global);
        std::vector<int8_t> Aq;
        quantA(A.data(), M, K, Aq, KD);
        std::vector<int32_t> Cq;
        i8gemm(Aq.data(), Bq_global.data(), M, K, N, KD, Cq);
        std::vector<float> C_out(M * N);
        dequant_global(Cq.data(), M, N, Bscale, C_out.data());

        if (!check_close(C_out.data(), C_ref.data(), M * N, 0.05f, "global")) {
            errors++;
            printf("    M=%d K=%d N=%d KD=%d Bscale=%.6f\n", M, K, N, KD, Bscale);
        }

        // ── Per-column scale (server engine) ──
        std::vector<int8_t> Bq_percol;
        std::vector<float> col_scale;
        packB_percol(B.data(), K, N, Bq_percol, col_scale);
        std::vector<int32_t> Cq2;
        i8gemm(Aq.data(), Bq_percol.data(), M, K, N, KD, Cq2);
        std::vector<float> C_out2(M * N);
        dequant_percol(Cq2.data(), M, N, col_scale, C_out2.data());

        if (!check_close(C_out2.data(), C_ref.data(), M * N, 0.05f, "percol")) {
            errors++;
            printf("    M=%d K=%d N=%d KD=%d\n", M, K, N, KD);
        }
    }

    // ── Edge case: activations at scale boundary ──
    printf("\n── Edge cases ──\n");
    {
        int M = 1, K = 256, N = 256, KD = 256;
        std::vector<float> A(M * K), B(K * N);
        // Push activations to the scale boundary [-8, 8]
        for (int i = 0; i < K; i++) A[i] = (i % 2 == 0 ? 7.9f : -7.9f);
        for (int i = 0; i < K * N; i++) B[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;

        std::vector<float> C_ref(M * N);
        cpu_gemm(A.data(), B.data(), M, K, N, C_ref.data());

        std::vector<int8_t> Bq;
        float Bscale = packB_global(B.data(), K, N, Bq);
        std::vector<int8_t> Aq;
        quantA(A.data(), M, K, Aq, KD);
        std::vector<int32_t> Cq;
        i8gemm(Aq.data(), Bq.data(), M, K, N, KD, Cq);
        std::vector<float> C_out(M * N);
        dequant_global(Cq.data(), M, N, Bscale, C_out.data());

        if (!check_close(C_out.data(), C_ref.data(), M * N, 0.05f, "boundary")) errors++;
    }

    // ── Edge case: very small activations ──
    {
        int M = 1, K = 128, N = 128, KD = 128;
        std::vector<float> A(M * K), B(K * N);
        for (int i = 0; i < K; i++) A[i] = 1e-4f * ((float)rand() / RAND_MAX * 2.0f - 1.0f);
        for (int i = 0; i < K * N; i++) B[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;

        std::vector<float> C_ref(M * N);
        cpu_gemm(A.data(), B.data(), M, K, N, C_ref.data());

        std::vector<int8_t> Bq;
        float Bscale = packB_global(B.data(), K, N, Bq);
        std::vector<int8_t> Aq;
        quantA(A.data(), M, K, Aq, KD);
        std::vector<int32_t> Cq;
        i8gemm(Aq.data(), Bq.data(), M, K, N, KD, Cq);
        std::vector<float> C_out(M * N);
        dequant_global(Cq.data(), M, N, Bscale, C_out.data());

        if (!check_close(C_out.data(), C_ref.data(), M * N, 0.10f, "tiny_act")) errors++;
    }

    printf("\n%d errors total\n", errors);
    return errors > 0 ? 1 : 0;
}

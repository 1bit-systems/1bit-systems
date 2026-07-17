// tools/oscar_calib.cpp — OSCAR offline calibration for KV cache rotation matrices
//
// Runs a few forward passes with random calibration data, collects Q/K/V
// activations, computes covariance matrices, derives attention-aware rotation
// matrices per the OSCAR paper (arXiv:2605.17757), exports to binary.
//
// Build: g++ -O3 -march=native -std=c++17 -Iengine/fusion -Iinclude \
//        -o tools/oscar_calib tools/oscar_calib.cpp engine/fusion/cpu_layer.cpp -lm
// Run:   ./tools/oscar_calib model.q4nx [calib_tokens=64] [output=oscar_rots.bin]

#include "cpu_layer.h"
#include "cpu_q4nx_loader.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <fstream>
#include <algorithm>

// ── Simple symmetric eigendecomposition (Jacobi method) for small matrices ──
// Only for head_dim <= 128. Returns eigenvalues in lambda and eigenvectors as
// columns of Q (column j = eigenvector for lambda[j]).
static bool jacobi_eigh(float* A, float* lambda, float* Q, int n, int max_iter = 50) {
    // Copy A to current matrix
    std::vector<float> B(n * n);
    std::vector<float> Qv(n * n);
    for (int i = 0; i < n * n; i++) { B[i] = A[i]; Qv[i] = 0; }
    for (int i = 0; i < n; i++) Qv[i * n + i] = 1.0f; // Identity

    for (int iter = 0; iter < max_iter; iter++) {
        // Find largest off-diagonal element
        float max_off = 0;
        int p = 0, q = 0;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) {
                float v = fabsf(B[i * n + j]);
                if (v > max_off) { max_off = v; p = i; q = j; }
            }
        if (max_off < 1e-10f) break; // Converged

        // Compute Jacobi rotation
        float beta = (B[q * n + q] - B[p * n + p]) / (2.0f * B[p * n + q]);
        float t = (beta >= 0 ? 1.0f : -1.0f) / (fabsf(beta) + sqrtf(beta * beta + 1.0f));
        float c = 1.0f / sqrtf(1.0f + t * t);
        float s = t * c;

        // Apply rotation to B
        for (int i = 0; i < n; i++) {
            if (i == p || i == q) continue;
            float Bip = B[i * n + p];
            float Biq = B[i * n + q];
            B[i * n + p] = Bip * c + Biq * s;
            B[i * n + q] = -Bip * s + Biq * c;
            // Symmetric
            B[p * n + i] = B[i * n + p];
            B[q * n + i] = B[i * n + q];
        }
        float App = B[p * n + p];
        float Aqq = B[q * n + q];
        float Apq = B[p * n + q];
        B[p * n + p] = App * c * c + 2.0f * Apq * c * s + Aqq * s * s;
        B[q * n + q] = App * s * s - 2.0f * Apq * c * s + Aqq * c * c;
        B[p * n + q] = 0.0f;
        B[q * n + p] = 0.0f;

        // Update eigenvectors
        for (int i = 0; i < n; i++) {
            float Qip = Qv[i * n + p];
            float Qiq = Qv[i * n + q];
            Qv[i * n + p] = Qip * c + Qiq * s;
            Qv[i * n + q] = -Qip * s + Qiq * c;
        }
    }

    // Extract eigenvalues and eigenvectors, sorted descending
    struct E { int idx; float val; };
    std::vector<E> evs(n);
    for (int i = 0; i < n; i++) { evs[i] = {i, B[i * n + i]}; }
    std::sort(evs.begin(), evs.end(), [](auto& a, auto& b) { return a.val > b.val; });

    for (int i = 0; i < n; i++) {
        lambda[i] = evs[i].val;
        for (int j = 0; j < n; j++)
            Q[j * n + i] = Qv[j * n + evs[i].idx];
    }
    return true;
}

// ── Hadamard matrix (size must be power of 2) ──
static void hadamard(float* H, int n) {
    for (int i = 0; i < n; i++) H[i] = 1.0f;
    for (int sz = 2; sz <= n; sz *= 2)
        for (int i = 0; i < sz/2; i++)
            for (int j = 0; j < n; j += sz)
                H[j * n + sz/2 + i] = H[j * n + i];
    // Normalize: H/sqrt(n)
    float inv_sqrt = 1.0f / sqrtf((float)n);
    for (int i = 0; i < n * n; i++) H[i] *= inv_sqrt;
}

// ── Bit-reversal permutation matrix ──
// P_br[i, reverse_bits(i, log2n)] = 1
static void bit_reversal_perm(float* P, int n) {
    int log2n = 0; while ((1 << log2n) < n) log2n++;
    for (int i = 0; i < n * n; i++) P[i] = 0;
    for (int i = 0; i < n; i++) {
        int rev = 0;
        for (int b = 0; b < log2n; b++)
            if (i & (1 << b)) rev |= (1 << (log2n - 1 - b));
        P[i * n + rev] = 1.0f;
    }
}

// ── Matrix multiply: C = A * B  (all n×n row-major) ──
static void matmul(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0;
            for (int k = 0; k < n; k++)
                s += (double)A[i * n + k] * (double)B[k * n + j];
            C[i * n + j] = (float)s;
        }
}

// ── Transpose: B = A^T ──
static void transpose(const float* A, float* B, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            B[j * n + i] = A[i * n + j];
}

// ── Binary output format ──
struct OscarHeader {
    char magic[8] = {'O','S','C','A','R','R','O','T'};
    int32_t head_dim;           // 128 for Qwen3
    int32_t n_layers;
    int32_t n_heads_q;
    int32_t n_heads_kv;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s model.q4nx [calib_tokens=64] [output=oscar_rots.bin]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    int calib_tokens = argc > 2 ? atoi(argv[2]) : 64;
    const char* out_path = argc > 3 ? argv[3] : "oscar_rots.bin";

    printf("=== OSCAR Calibration ===\n");
    printf("  Model: %s\n", model_path);
    printf("  Tokens: %d\n", calib_tokens);
    printf("  Output: %s\n", out_path);

    auto t0 = std::chrono::high_resolution_clock::now();

    // ── Load model ──
    Q4nxModel qm;
    if (!qm.load(model_path)) return 1;
    int H = qm.hidden_dim, IM = qm.inter_size, NH = qm.n_heads;
    int NKV = qm.n_kv_heads, HD = qm.head_dim, V = qm.vocab_size, L = qm.n_layers;

    printf("  H=%d IM=%d NH=%d NKV=%d HD=%d V=%d L=%d\n", H, IM, NH, NKV, HD, V, L);

    auto gt = [&](auto n) {
        auto i = qm.tensors.find(n);
        return i == qm.tensors.end() ? std::vector<float>() : i->second.fp32;
    };

    // ── Precompute rotations per layer ──
    // For each layer, we compute R_K and R_V (both HD×HD matrices)
    // using random calibration tokens (uniform random hidden states).
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> R_K_all(L * HD * HD);
    std::vector<float> R_V_all(L * HD * HD);

    // For each layer, compute the covariance from random inputs through Q/K/V projections
    for (int l = 0; l < L; l++) {
        if (l % 4 == 0) printf("  Layer %d/%d...\n", l, L);

        char k[256];
        auto gl = [&](auto f) { snprintf(k,256,f,l); return gt(k); };
        auto wq = gl("model.layers.%d.self_attn.q_proj.weight");
        auto wk = gl("model.layers.%d.self_attn.k_proj.weight");
        auto wv = gl("model.layers.%d.self_attn.v_proj.weight");

        if (wq.empty() || wk.empty() || wv.empty()) continue;

        // Accumulate covariance matrices
        std::vector<double> C_Q(HD * HD, 0.0);  // query covariance
        std::vector<double> C_S(HD * HD, 0.0);  // score-aware value cov

        for (int t = 0; t < calib_tokens; t++) {
            // Random hidden state (one head's worth)
            std::vector<float> x(H);
            for (int i = 0; i < H; i++) x[i] = dist(rng);

            // Compute Q, K, V for one head
            std::vector<float> q(HD), k(HD), v(HD);
            for (int i = 0; i < HD; i++) {
                double qs = 0, ks = 0, vs = 0;
                for (int j = 0; j < H; j++) {
                    qs += (double)x[j] * wq[(size_t)i * H + j];
                    ks += (double)x[j] * wk[(size_t)i * H + j];
                    vs += (double)x[j] * wv[(size_t)i * H + j];
                }
                q[i] = (float)qs; k[i] = (float)ks; v[i] = (float)vs;
            }

            // C_Q += q^T * q  (outer product)
            for (int i = 0; i < HD; i++)
                for (int j = 0; j < HD; j++)
                    C_Q[i * HD + j] += (double)q[i] * q[j];

            // For C_S, we need attention scores, approximated as uniform attention
            // over all tokens: S ~= 1/seq_len. So C_S ~= V^T * V / seq_len^2
            // Simplified: C_S += v^T * v (uniform attention approximation)
            for (int i = 0; i < HD; i++)
                for (int j = 0; j < HD; j++)
                    C_S[i * HD + j] += (double)v[i] * v[j];
        }

        // Normalize
        float inv_n = 1.0f / calib_tokens;
        std::vector<float> CQ_f(HD * HD), CS_f(HD * HD);
        for (int i = 0; i < HD * HD; i++) {
            CQ_f[i] = (float)(C_Q[i] * inv_n);
            CS_f[i] = (float)(C_S[i] * inv_n);
        }

        // Eigendecomposition: C_Q = U_Q * Lambda_Q * U_Q^T
        std::vector<float> lambda_Q(HD), U_Q(HD * HD);
        std::vector<float> lambda_S(HD), U_S(HD * HD);
        jacobi_eigh(CQ_f.data(), lambda_Q.data(), U_Q.data(), HD);
        jacobi_eigh(CS_f.data(), lambda_S.data(), U_S.data(), HD);

        // Build R_K = U_Q * H_Had * P_br
        std::vector<float> H_had(HD * HD), P_br(HD * HD);
        hadamard(H_had.data(), HD);
        bit_reversal_perm(P_br.data(), HD);

        std::vector<float> temp(HD * HD);
        // R_K = U_Q * (H_Had * P_br)
        matmul(H_had.data(), P_br.data(), temp.data(), HD);
        matmul(U_Q.data(), temp.data(), &R_K_all[(size_t)l * HD * HD], HD);

        // R_V = U_S * H_Had * P_br
        matmul(U_S.data(), temp.data(), &R_V_all[(size_t)l * HD * HD], HD);
    }

    // ── Export to binary ──
    OscarHeader hdr;
    hdr.head_dim = HD;
    hdr.n_layers = L;
    hdr.n_heads_q = NH;
    hdr.n_heads_kv = NKV;

    std::ofstream f(out_path, std::ios::binary);
    if (!f) { fprintf(stderr, "Cannot write: %s\n", out_path); return 1; }
    f.write((char*)&hdr, sizeof(hdr));
    f.write((char*)R_K_all.data(), R_K_all.size() * sizeof(float));
    f.write((char*)R_V_all.data(), R_V_all.size() * sizeof(float));
    f.close();

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();

    printf("\n  Done: %d layers × %dx%d rotation matrices exported\n", L, HD, HD);
    printf("  File: %s (%.0f KB)\n", out_path, (R_K_all.size() + R_V_all.size()) * 4.0 / 1024);
    printf("  Time: %.0f ms\n", ms);
    return 0;
}

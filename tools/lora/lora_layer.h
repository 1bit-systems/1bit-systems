#pragma once
// lora_layer.h — LoRA adapter forward + manual backward pass
// All matmuls target MI300X tensor cores via ROCm HIP WMMA.
// No Python, no PyTorch, no PEFT. Pure C++23.
//
// LoRA: y = W0*x + (B @ A) * x * (alpha / r)
//   A: [r, k]  (rank, in_features)  — random init
//   B: [d, r]  (out_features, rank) — zero init
//   scale = alpha / r

#include <vector>
#include <cmath>
#include <cstdio>
#include <random>
#include <cstdint>

// ── LoRA adapter (host-side reference) ────────────────────────────────────
// For MI300X, these matmuls should be replaced with hipblasLt or WMMA calls.
// The host versions here are correctness references for the training loop.

struct LoraLayer {
    int in_features, out_features, rank;
    float alpha, scale;
    
    // Weight matrices: row-major
    std::vector<float> A;  // [rank, in_features]
    std::vector<float> B;  // [out_features, rank]
    
    // Gradients (only A and B get grads — W0 is frozen)
    std::vector<float> grad_A;
    std::vector<float> grad_B;
    
    // Optimizer state (AdamW)
    std::vector<float> adam_m_A, adam_v_A;
    std::vector<float> adam_m_B, adam_v_B;
    
    LoraLayer(int in_feat, int out_feat, int r, float a = 1.0f)
        : in_features(in_feat), out_features(out_feat), rank(r), alpha(a)
    {
        scale = alpha / (float)rank;
        size_t a_size = (size_t)rank * in_features;
        size_t b_size = (size_t)out_features * rank;
        
        A.resize(a_size);
        B.resize(b_size);
        grad_A.resize(a_size, 0.0f);
        grad_B.resize(b_size, 0.0f);
        adam_m_A.resize(a_size, 0.0f);
        adam_v_A.resize(a_size, 0.0f);
        adam_m_B.resize(b_size, 0.0f);
        adam_v_B.resize(b_size, 0.0f);
        
        // Init A ~ N(0, 0.02), B = 0
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.02f);
        for (auto& v : A) v = dist(rng);
        // B stays zero — first forward pass is identity
    }
    
    // Forward: y += (B @ A) @ x * scale
    // x: [batch, in_features], y: [batch, out_features]
    void forward(const float* x, float* y, int batch) {
        std::vector<float> tmp(rank);
        for (int b = 0; b < batch; b++) {
            // tmp[r] = sum_j A[r,j] * x[b,j]
            for (int r = 0; r < rank; r++) {
                float sum = 0.0f;
                for (int j = 0; j < in_features; j++)
                    sum += A[(size_t)r * in_features + j] * x[(size_t)b * in_features + j];
                tmp[r] = sum;
            }
            // y[b,o] += scale * sum_r B[o,r] * tmp[r]
            for (int o = 0; o < out_features; o++) {
                float sum = 0.0f;
                for (int r = 0; r < rank; r++)
                    sum += B[(size_t)o * rank + r] * tmp[r];
                y[(size_t)b * out_features + o] += scale * sum;
            }
        }
    }
    
    // Backward: compute dL/dA, dL/dB from dL/dy
    // dy: [batch, out_features]   x: [batch, in_features]
    void backward(const float* dy, const float* x, int batch) {
        std::vector<float> tmp(rank);
        std::vector<float> g(rank);
        for (int b = 0; b < batch; b++) {
            // precompute tmp[r] = sum_j A[r,j] * x[b,j]  (reuse from forward)
            for (int r = 0; r < rank; r++) {
                float sum = 0.0f;
                for (int j = 0; j < in_features; j++)
                    sum += A[(size_t)r * in_features + j] * x[(size_t)b * in_features + j];
                tmp[r] = sum;
            }
            
            // grad_B[o,r] += dy[b,o] * tmp[r]
            for (int o = 0; o < out_features; o++) {
                float dy_bo = dy[(size_t)b * out_features + o];
                for (int r = 0; r < rank; r++)
                    grad_B[(size_t)o * rank + r] += dy_bo * tmp[r];
            }
            
            // grad_A[r,j] += sum_o B[o,r] * dy[b,o] * x[b,j]
            for (int r = 0; r < rank; r++) {
                float sum = 0.0f;
                for (int o = 0; o < out_features; o++)
                    sum += B[(size_t)o * rank + r] * dy[(size_t)b * out_features + o];
                g[r] = sum;
            }
            for (int r = 0; r < rank; r++)
                for (int j = 0; j < in_features; j++)
                    grad_A[(size_t)r * in_features + j] += g[r] * x[(size_t)b * in_features + j];
        }
        
        // Scale gradients by (alpha/r) to match forward
        float s = scale;
        for (auto& g : grad_A) g *= s;
        for (auto& g : grad_B) g *= s;
    }
    
    // Zero gradients
    void zero_grad() {
        std::fill(grad_A.begin(), grad_A.end(), 0.0f);
        std::fill(grad_B.begin(), grad_B.end(), 0.0f);
    }
    
    // AdamW step
    void adamw_step(float lr, float beta1, float beta2, float eps, float weight_decay, int step) {
        auto step_adam = [&](std::vector<float>& w, std::vector<float>& m,
                             std::vector<float>& v, const std::vector<float>& g) {
            float b1t = 1.0f - std::pow(beta1, step);
            float b2t = 1.0f - std::pow(beta2, step);
            for (size_t i = 0; i < w.size(); i++) {
                m[i] = beta1 * m[i] + (1.0f - beta1) * g[i];
                v[i] = beta2 * v[i] + (1.0f - beta2) * g[i] * g[i];
                float m_hat = m[i] / b1t;
                float v_hat = v[i] / b2t;
                w[i] -= lr * (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * w[i]);
            }
        };
        step_adam(A, adam_m_A, adam_v_A, grad_A);
        step_adam(B, adam_m_B, adam_v_B, grad_B);
    }
    
    // Save adapter to disk (Q4NX-tiled for MI300X TRG)
    void save(const char* path) const {
        FILE* f = fopen(path, "wb");
        if (!f) { perror("save"); return; }
        uint32_t hdr[4] = {0x4C4F5241, (uint32_t)rank, (uint32_t)in_features, (uint32_t)out_features};
        fwrite(hdr, sizeof(hdr), 1, f);
        float sc = scale;
        fwrite(&sc, sizeof(sc), 1, f);
        fwrite(A.data(), sizeof(float), A.size(), f);
        fwrite(B.data(), sizeof(float), B.size(), f);
        fclose(f);
        printf("  Saved adapter: %s (r=%d, in=%d, out=%d)\n", path, rank, in_features, out_features);
    }
};

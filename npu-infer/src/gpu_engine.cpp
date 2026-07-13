/**
 * GPU Inference Engine for Qwen3-0.6B using ROCm/hipBLAS.
 *
 * Implements FP32 inference with hipBLAS GEMM operations.
 * Targets Radeon 8060S (gfx1151) with ROCm 7.1.
 *
 * Build:
 *   cd build && hipcc -std=c++17 -O2 -I../include \
 *       -I/home/bcloud/torch2aie/examples \
 *       -I/opt/rocm/include -L/opt/rocm/lib \
 *       ../src/gpu_engine.cpp dequant_q4nx.o \
 *       -lhipblas -lm -o gpu_engine
 */

#include "gpu_engine.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

extern "C" float* dequant_i8_to_float(const uint8_t*, int, int*, int*);

// HIP check macro
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "[HIP ERROR] %s:%d: %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(err)); \
        } \
    } while (0)

// hipBLAS check macro
#define HIPBLAS_CHECK(call) \
    do { \
        hipblasStatus_t err = call; \
        if (err != HIPBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "[HIPBLAS ERROR] %s:%d: %d\n", __FILE__, __LINE__, err); \
        } \
    } while (0)

// RoPE tables (global for performance)
static std::vector<float> g_rope_cos;
static std::vector<float> g_rope_sin;

// Initialize RoPE tables for given head_dim and base
static void init_rope(int hd, float base, int max_pos) {
    g_rope_cos.resize(max_pos * hd);
    g_rope_sin.resize(max_pos * hd);
    for (int p = 0; p < max_pos; p++) {
        for (int d = 0; d < hd; d += 2) {
            float freq = 1.0f / powf(base, (float)d / hd);
            float angle = p * freq;
            g_rope_cos[p * hd + d] = cosf(angle);
            g_rope_sin[p * hd + d] = sinf(angle);
            g_rope_cos[p * hd + d + 1] = cosf(angle);
            g_rope_sin[p * hd + d + 1] = sinf(angle);
        }
    }
}

// Apply rotary embeddings (in-place on q and k)
static void apply_rope_impl(float* q, float* k, int hd, int num_heads, int position) {
    for (int h = 0; h < num_heads; h++) {
        float* qh = q + h * hd;
        float* kh = k + h * hd;
        for (int d = 0; d < hd; d += 2) {
            float q0 = qh[d], q1 = qh[d + 1];
            float c = g_rope_cos[position * hd + d];
            float s = g_rope_sin[position * hd + d];
            qh[d] = q0 * c - q1 * s;
            qh[d + 1] = q0 * s + q1 * c;
            if (kh) {
                float k0 = kh[d], k1 = kh[d + 1];
                kh[d] = k0 * c - k1 * s;
                kh[d + 1] = k0 * s + k1 * c;
            }
        }
    }
}

//======= GpuInferenceEngine Implementation =======

GpuInferenceEngine::GpuInferenceEngine() : initialized_(false) {
    // Initialize working buffers
    h_q_.resize(NH * HD);
    h_k_.resize(NKV * HD);
    h_v_.resize(NKV * HD);
    h_q_out_.resize(NH * HD + NKV * HD + NKV * HD);  // QKV output: 4096
    h_attn_out_.resize(NH * HD);
    h_s1_.resize(H);
    h_s2_.resize(H);
    h_gate_out_.resize(IM);
    h_up_out_.resize(IM);
    h_swiglu_out_.resize(IM);
    h_logits_.resize(NV);
    h_scor_.resize(MAX_SEQ_LEN);

    // Initialize KV cache structures
    kv_cache_k_.resize(NC);
    kv_cache_v_.resize(NC);
    kv_cache_len_.resize(NC, 0);
    for (int i = 0; i < NC; i++) {
        kv_cache_k_[i].resize(MAX_SEQ_LEN * NKV * HD);
        kv_cache_v_[i].resize(MAX_SEQ_LEN * NKV * HD);
    }
}

GpuInferenceEngine::~GpuInferenceEngine() {
    if (initialized_) {
        free_gpu_buffers();
    }
    if (model_data_) {
        munmap(model_data_, model_size_);
        model_data_ = nullptr;
    }
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
    if (hipblas_handle_) {
        hipblasDestroy(hipblas_handle_);
        hipblas_handle_ = nullptr;
    }
}

bool GpuInferenceEngine::init(const char* model_path) {
    printf("[GPU Engine] Initializing...\n");

    // Initialize hipBLAS
    HIPBLAS_CHECK(hipblasCreate(&hipblas_handle_));
    printf("[GPU Engine] hipBLAS handle created\n");

    // Memory-map the model file
    fd_ = open(model_path, O_RDONLY);
    if (fd_ < 0) {
        fprintf(stderr, "[ERROR] Cannot open model file: %s\n", model_path);
        return false;
    }

    struct stat st;
    if (fstat(fd_, &st) < 0) {
        fprintf(stderr, "[ERROR] Cannot stat model file\n");
        return false;
    }
    model_size_ = st.st_size;

    model_data_ = (uint8_t*)mmap(NULL, model_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (model_data_ == MAP_FAILED) {
        fprintf(stderr, "[ERROR] Cannot mmap model file\n");
        return false;
    }
    close(fd_);  // fd can be closed after mmap
    fd_ = -1;

    // Parse header
    uint64_t header_size;
    memcpy(&header_size, model_data_, 8);
    json_size_ = header_size;
    const char* json_start = (const char*)(model_data_ + 8);
    size_t json_len = header_size;

    printf("[GPU Engine] JSON metadata size: %zu bytes\n", json_len);

    // Parse layer offsets
    struct LayerOffsets {
        uint64_t qp, kp, vp, op, gp, up, dp;
        uint64_t in_off, pa_off, qn_off, kn_off;
    };
    std::vector<LayerOffsets> lo(NC);

    auto i8p = [&](uint64_t offset) -> const uint8_t* {
        return model_data_ + 8 + json_size_ + offset;
    };

    char buf[256];
    for (int l = 0; l < NC; l++) {
        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.q_proj.weight", l);
        lo[l].qp = parse_json_offset(json_start, json_len, buf);
        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.k_proj.weight", l);
        lo[l].kp = parse_json_offset(json_start, json_len, buf);
        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.v_proj.weight", l);
        lo[l].vp = parse_json_offset(json_start, json_len, buf);
        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.o_proj.weight", l);
        lo[l].op = parse_json_offset(json_start, json_len, buf);
        snprintf(buf, sizeof(buf), "model.layers.%d.mlp.gate_proj.weight", l);
        lo[l].gp = parse_json_offset(json_start, json_len, buf);
        snprintf(buf, sizeof(buf), "model.layers.%d.mlp.up_proj.weight", l);
        lo[l].up = parse_json_offset(json_start, json_len, buf);
        snprintf(buf, sizeof(buf), "model.layers.%d.mlp.down_proj.weight", l);
        lo[l].dp = parse_json_offset(json_start, json_len, buf);
        snprintf(buf, sizeof(buf), "model.layers.%d.input_layernorm.weight", l);
        lo[l].in_off = parse_json_offset(json_start, json_len, buf);
        snprintf(buf, sizeof(buf), "model.layers.%d.post_attention_layernorm.weight", l);
        lo[l].pa_off = parse_json_offset(json_start, json_len, buf);
        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.q_norm.weight", l);
        lo[l].qn_off = parse_json_offset(json_start, json_len, buf);
        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.k_norm.weight", l);
        lo[l].kn_off = parse_json_offset(json_start, json_len, buf);
    }

    uint64_t norm_off = parse_json_offset(json_start, json_len, "model.norm.weight");
    uint64_t lm_off = parse_json_offset(json_start, json_len, "lm_head.weight");
    uint64_t embed_off = parse_json_offset(json_start, json_len, "model.embed_tokens.weight");

    printf("[GPU Engine] Loading weights...\n");

    // Embedding table
    embed_tokens_ = (uint16_t*)(model_data_ + 8 + json_size_ + embed_off);

    // Final norm weight
    final_norm_weight_ = (uint16_t*)(model_data_ + 8 + json_size_ + norm_off);

    // LM head - dequantize and convert to float
    int lr, lc;
    float* lf = dequant_i8_to_float(i8p(lm_off), 18992, &lr, &lc);
    // lm_head_dequant_ will hold float weights in (H, NV) layout for efficient GEMM
    lm_head_dequant_.reset(new float[NV * H]);
    for (int h = 0; h < H; h++) {
        for (int v = 0; v < NV; v++) {
            lm_head_dequant_[h * NV + v] = lf[v * H + h];  // Transpose: (H, NV)
        }
    }
    free(lf);

    // Dequantize and load layer weights
    layer_weights_.resize(NC);
    input_layernorm_weights_.resize(NC * H);
    post_attention_layernorm_weights_.resize(NC * H);
    q_norm_weights_.resize(NC * HD);
    k_norm_weights_.resize(NC * HD);

    auto t0 = std::chrono::steady_clock::now();

    for (int l = 0; l < NC; l++) {
        int qr, kr, vr, unused;

        // QKV weights (Q: 1024×2048, K: 1024×1024, V: 1024×1024)
        float* qw = dequant_i8_to_float(i8p(lo[l].qp), 256, &qr, &unused);
        float* kw = dequant_i8_to_float(i8p(lo[l].kp), 128, &kr, &unused);
        float* vw = dequant_i8_to_float(i8p(lo[l].vp), 128, &vr, &unused);

        int tqkv = qr + kr + vr;  // 2048 + 1024 + 1024 = 4096
        layer_weights_[l].qkv_w.resize((size_t)H * tqkv);
        for (int k = 0; k < H; k++) {
            memcpy(&layer_weights_[l].qkv_w[k * tqkv], &qw[k * qr], qr * sizeof(float));
            memcpy(&layer_weights_[l].qkv_w[k * tqkv + qr], &kw[k * kr], kr * sizeof(float));
            memcpy(&layer_weights_[l].qkv_w[k * tqkv + qr + kr], &vw[k * vr], vr * sizeof(float));
        }
        free(qw); free(kw); free(vw);

        // O projection: NH*HD × H = 2048 × 1024
        float* ow = dequant_i8_to_float(i8p(lo[l].op), 256, &unused, &unused);
        layer_weights_[l].o_w.resize((size_t)NH * HD * H);
        memcpy(layer_weights_[l].o_w.data(), ow, layer_weights_[l].o_w.size() * sizeof(float));
        free(ow);

        // Gate projection: H × IM = 1024 × 3072
        float* gw = dequant_i8_to_float(i8p(lo[l].gp), 384, &unused, &unused);
        layer_weights_[l].gate_w.resize((size_t)H * IM);
        memcpy(layer_weights_[l].gate_w.data(), gw, layer_weights_[l].gate_w.size() * sizeof(float));
        free(gw);

        // Up projection: H × IM = 1024 × 3072
        float* uw = dequant_i8_to_float(i8p(lo[l].up), 384, &unused, &unused);
        layer_weights_[l].up_w.resize((size_t)H * IM);
        memcpy(layer_weights_[l].up_w.data(), uw, layer_weights_[l].up_w.size() * sizeof(float));
        free(uw);

        // Down projection: IM × H = 3072 × 1024
        float* dw = dequant_i8_to_float(i8p(lo[l].dp), 384, &unused, &unused);
        layer_weights_[l].down_w.resize((size_t)IM * H);
        memcpy(layer_weights_[l].down_w.data(), dw, layer_weights_[l].down_w.size() * sizeof(float));
        free(dw);

        // RMSNorm weights (stored as BF16)
        const uint16_t* in_w = (const uint16_t*)(model_data_ + 8 + json_size_ + lo[l].in_off);
        memcpy(&input_layernorm_weights_[l * H], in_w, H * sizeof(uint16_t));

        const uint16_t* pa_w = (const uint16_t*)(model_data_ + 8 + json_size_ + lo[l].pa_off);
        memcpy(&post_attention_layernorm_weights_[l * H], pa_w, H * sizeof(uint16_t));

        const uint16_t* qn_w = (const uint16_t*)(model_data_ + 8 + json_size_ + lo[l].qn_off);
        memcpy(&q_norm_weights_[l * HD], qn_w, HD * sizeof(uint16_t));

        const uint16_t* kn_w = (const uint16_t*)(model_data_ + 8 + json_size_ + lo[l].kn_off);
        memcpy(&k_norm_weights_[l * HD], kn_w, HD * sizeof(uint16_t));

        if (l % 7 == 0) {
            printf("  Loaded layer %d/%d\n", l + 1, NC);
        }
    }

    printf("[GPU Engine] Weights loaded in %.1fs\n",
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    // Initialize RoPE tables
    printf("[GPU Engine] Initializing RoPE tables...\n");
    init_rope(HD, 1000000.0f, MAX_SEQ_LEN);

    // Allocate GPU buffers
    printf("[GPU Engine] Allocating GPU buffers...\n");
    if (!allocate_gpu_buffers()) {
        fprintf(stderr, "[ERROR] Failed to allocate GPU buffers\n");
        return false;
    }

    // Copy weights to GPU
    printf("[GPU Engine] Copying weights to GPU...\n");
    if (!copy_weights_to_gpu()) {
        fprintf(stderr, "[ERROR] Failed to copy weights to GPU\n");
        return false;
    }

    initialized_ = true;
    printf("[GPU Engine] Initialization complete!\n\n");
    return true;
}

bool GpuInferenceEngine::allocate_gpu_buffers() {
    // Allocate GPU buffers for activations
    HIP_CHECK(hipMalloc(&d_q_, NH * HD * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_k_, NKV * HD * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_v_, NKV * HD * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_qk_, MAX_SEQ_LEN * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_attn_, NH * HD * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_h1_, H * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_h2_, H * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_gate_out_, IM * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_up_out_, IM * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_swiglu_, IM * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_down_out_, H * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_logits_, NV * sizeof(float)));

    // KV cache on GPU
    HIP_CHECK(hipMalloc(&d_k_cache_, NC * MAX_SEQ_LEN * NKV * HD * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_v_cache_, NC * MAX_SEQ_LEN * NKV * HD * sizeof(float)));

    return true;
}

void GpuInferenceEngine::free_gpu_buffers() {
    auto hip_free = [](void*& p) {
        if (p) {
            hipFree(p);
            p = nullptr;
        }
    };

    hip_free((void*&)d_q_);
    hip_free((void*&)d_k_);
    hip_free((void*&)d_v_);
    hip_free((void*&)d_qk_);
    hip_free((void*&)d_attn_);
    hip_free((void*&)d_h1_);
    hip_free((void*&)d_h2_);
    hip_free((void*&)d_gate_out_);
    hip_free((void*&)d_up_out_);
    hip_free((void*&)d_swiglu_);
    hip_free((void*&)d_down_out_);
    hip_free((void*&)d_logits_);
    hip_free((void*&)d_k_cache_);
    hip_free((void*&)d_v_cache_);
}

bool GpuInferenceEngine::copy_weights_to_gpu() {
    // Initialize KV cache to zeros
    HIP_CHECK(hipMemset(d_k_cache_, 0, NC * MAX_SEQ_LEN * NKV * HD * sizeof(float)));
    HIP_CHECK(hipMemset(d_v_cache_, 0, NC * MAX_SEQ_LEN * NKV * HD * sizeof(float)));

    return true;
}

int GpuInferenceEngine::generate(const int* input_ids, int input_len,
                                  int* output_ids, int max_output) {
    if (!initialized_) {
        fprintf(stderr, "[ERROR] Engine not initialized\n");
        return 0;
    }

    // Reset KV cache
    for (int l = 0; l < NC; l++) {
        kv_cache_len_[l] = 0;
        memset(kv_cache_k_[l].data(), 0, MAX_SEQ_LEN * NKV * HD * sizeof(float));
        memset(kv_cache_v_[l].data(), 0, MAX_SEQ_LEN * NKV * HD * sizeof(float));
    }

    std::vector<float> hidden(H);
    int seq_pos = 0;

    // Prefill phase: process all input tokens
    printf("[GPU Engine] Prefill: %d tokens\n", input_len);
    auto t_start = std::chrono::steady_clock::now();

    for (int pi = 0; pi < input_len; pi++) {
        // Embed token
        int tok = input_ids[pi];
        for (int i = 0; i < H; i++) {
            hidden[i] = bf16_to_float(embed_tokens_[tok * H + i]);
        }

        // Process all layers
        for (int l = 0; l < NC; l++) {
            compute_layer(l, hidden.data(), seq_pos);
        }

        seq_pos++;
    }

    printf("[GPU Engine] Prefill done in %.1fms\n",
           std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count());

    // Generate tokens
    printf("[GPU Engine] Generate: up to %d tokens\n", max_output);
    int gen_count = 0;

    for (int step = 0; step < max_output; step++) {
        auto t_step = std::chrono::steady_clock::now();

        // Compute logits
        compute_logits(hidden.data(), h_logits_.data());

        // Sample token
        int tok = sample_token(h_logits_.data());
        output_ids[gen_count++] = tok;

        // Embed new token and continue
        for (int i = 0; i < H; i++) {
            hidden[i] = bf16_to_float(embed_tokens_[tok * H + i]);
        }

        // Process all layers
        for (int l = 0; l < NC; l++) {
            compute_layer(l, hidden.data(), seq_pos);
        }

        seq_pos++;

        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_step).count();

        // Stop on EOS token
        if (tok == 151643 || tok == 151644) {
            printf("  [%d] %d (%.0fms) - EOS\n", step, tok, ms);
            break;
        }

        if (step < 8 || step % 10 == 0) {
            printf("  [%d] %d (%.0fms)\n", step, tok, ms);
        }
    }

    double total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_start).count();
    printf("\n[GPU Engine] %.0f ms/tok average\n", total_ms / (input_len + gen_count));

    return gen_count;
}

uint64_t GpuInferenceEngine::parse_json_offset(const char* js, size_t jl, const char* name) {
    size_t nl = strlen(name);
    const char* p = js;
    const char* e = js + jl;

    while (p < e) {
        auto q = (const char*)memmem(p, e - p, name, nl);
        if (!q) return 0;
        if (q > js && *(q - 1) == '"' && *(q + nl) == '"') {
            auto o = strstr(q, "\"data_offsets\"");
            if (o) {
                auto a = strchr(o, '[');
                if (a) return strtoull(a + 1, NULL, 10);
            }
        }
        p = q + 1;
    }
    return 0;
}

void GpuInferenceEngine::rms_norm(float* out, const float* in,
                                   const uint16_t* weight, int n) {
    // Compute sum of squares
    double ss = 0;
    for (int i = 0; i < n; i++) {
        float val = std::isfinite(in[i]) ? in[i] : 0.0f;
        ss += (double)val * val;
    }

    // Compute inverse rms
    float irms = 1.0f / sqrtf((float)(ss / n) + EPS);

    // Normalize and scale
    for (int i = 0; i < n; i++) {
        float norm = std::isfinite(in[i]) ? in[i] * irms : 0.0f;
        float w = bf16_to_float(weight[i]);
        out[i] = norm * w;
    }
}

void GpuInferenceEngine::softmax(float* scores, int n) {
    // NaN guard
    for (int i = 0; i < n; i++) {
        if (!std::isfinite(scores[i])) scores[i] = 0.0f;
    }

    // Find max
    float mx = scores[0];
    for (int i = 1; i < n; i++) {
        if (scores[i] > mx) mx = scores[i];
    }

    // Exponentiate and sum
    double sum = 0;
    for (int i = 0; i < n; i++) {
        float d = scores[i] - mx;
        d = d > 80 ? 80 : (d < -80 ? -80 : d);
        scores[i] = expf(d);
        sum += scores[i];
    }

    // Normalize
    if (sum <= 0) {
        float iv = 1.0f / n;
        for (int i = 0; i < n; i++) scores[i] = iv;
        return;
    }

    float is = 1.0f / (float)sum;
    for (int i = 0; i < n; i++) {
        scores[i] *= is;
    }
}

void GpuInferenceEngine::compute_layer(int layer_idx, float* h, int seq_pos) {
    const LayerWeights& lw = layer_weights_[layer_idx];
    const uint16_t* in_w = &input_layernorm_weights_[layer_idx * H];
    const uint16_t* pa_w = &post_attention_layernorm_weights_[layer_idx * H];
    const uint16_t* qn_w = &q_norm_weights_[layer_idx * HD];
    const uint16_t* kn_w = &k_norm_weights_[layer_idx * HD];

    // 1. Save residual
    memcpy(h_s1_.data(), h, H * sizeof(float));

    // 2. RMSNorm with input_layernorm
    rms_norm(h_s2_.data(), h, in_w, H);

    // 3. QKV GEMM: h (1×H) × W_qkv (H×4096) → qkv (1×4096)
    // hipBLAS is column-major: C = A * B
    // We store weights in row-major (H×4096), so we use transpose: C = h * W^T
    float alpha = 1.0f, beta = 0.0f;

    // QKV projection: hidden(1×1024) × QKV_weight(1024×4096)^T → (1×4096)
    // In row-major storage: weight is (H, tqkv), we want (1, H) × (H, tqkv) = (1, tqkv)
    // hipBLAS column-major interpretation of row-major data:
    // weight (H×tqkv) as col-maj: lda = tqkv, with columns being the H-sized chunks
    int tqkv = NH * HD + NKV * HD + NKV * HD;  // 2048 + 1024 + 1024 = 4096

    HIPBLAS_CHECK(hipblasSgemm(
        hipblas_handle_,
        HIPBLAS_OP_N,    // B not transposed
        HIPBLAS_OP_T,    // A transposed = hidden * weight^T
        tqkv,             // N = 4096 (output dim)
        1,                // M = 1
        H,                // K = 1024
        &alpha,
        lw.qkv_w.data(),  // B: (tqkv, H) interpreted from (H, tqkv) row-major
        tqkv,             // ldb = tqkv
        h_s2_.data(),     // A: (H, 1) = (K, M) row-major hidden
        H,                // lda = H
        &beta,
        h_q_out_.data(),  // C: (N, M) = (tqkv, 1)
        tqkv              // ldc
    ));

    HIP_CHECK(hipDeviceSynchronize());

    // Split Q, K, V from concatenated output
    // h_q_out_: [Q(2048), K(1024), V(1024)] = 4096 total
    memcpy(h_q_.data(), h_q_out_.data(), NH * HD * sizeof(float));                  // Q: 2048
    memcpy(h_k_.data(), &h_q_out_[NH * HD], NKV * HD * sizeof(float));             // K: 1024
    memcpy(h_v_.data(), &h_q_out_[NH * HD + NKV * HD], NKV * HD * sizeof(float));   // V: 1024

    // 4. Q RMSNorm per head
    for (int hh = 0; hh < NH; hh++) {
        double sq = 0;
        for (int d = 0; d < HD; d++) {
            sq += (double)h_q_[hh * HD + d] * h_q_[hh * HD + d];
        }
        float iq = 1.0f / sqrtf((float)(sq / HD) + EPS);
        for (int d = 0; d < HD; d++) {
            h_q_[hh * HD + d] *= iq * bf16_to_float(qn_w[d]);
        }
    }

    // K RMSNorm per KV head
    for (int kh = 0; kh < NKV; kh++) {
        double sk = 0;
        for (int d = 0; d < HD; d++) {
            sk += (double)h_k_[kh * HD + d] * h_k_[kh * HD + d];
        }
        float ik = 1.0f / sqrtf((float)(sk / HD) + EPS);
        for (int d = 0; d < HD; d++) {
            h_k_[kh * HD + d] *= ik * bf16_to_float(kn_w[d]);
        }
    }

    // Apply RoPE
    apply_rope_impl(h_q_.data(), h_k_.data(), HD, NH, seq_pos);

    // 5. Store K,V in KV cache
    int cache_len = kv_cache_len_[layer_idx];
    memcpy(&kv_cache_k_[layer_idx][cache_len * NKV * HD],
           h_k_.data(), NKV * HD * sizeof(float));
    memcpy(&kv_cache_v_[layer_idx][cache_len * NKV * HD],
           h_v_.data(), NKV * HD * sizeof(float));
    kv_cache_len_[layer_idx]++;

    // 6. Attention: Q × K^T → softmax → × V
    int cur_len = kv_cache_len_[layer_idx];

    for (int hh = 0; hh < NH; hh++) {
        int kvh = hh / GQA;  // Which KV head this Q head attends to

        // Compute Q[hh] · K^T[p] for all positions p
        for (int p = 0; p < cur_len; p++) {
            double s = 0;
            for (int d = 0; d < HD; d++) {
                s += (double)h_q_[hh * HD + d] *
                     kv_cache_k_[layer_idx][p * NKV * HD + kvh * HD + d];
            }
            h_scor_[p] = (float)(s / sqrtf(HD));
        }

        // Softmax
        softmax(h_scor_.data(), cur_len);

        // Compute attention output
        for (int d = 0; d < HD; d++) {
            double s = 0;
            for (int p = 0; p < cur_len; p++) {
                s += (double)h_scor_[p] *
                     kv_cache_v_[layer_idx][p * NKV * HD + kvh * HD + d];
            }
            h_attn_out_[hh * HD + d] = (float)s;
        }
    }

    // 7. O projection GEMM: attn (1×2048) × W_o (2048×1024) → out (1×1024)
    // O weight is (NH*HD, H) = (2048, 1024), stored row-major
    // Result: (1, NH*HD) × (NH*HD, H)^T = (1, H)
    alpha = 1.0f; beta = 0.0f;
    HIPBLAS_CHECK(hipblasSgemm(
        hipblas_handle_,
        HIPBLAS_OP_N,
        HIPBLAS_OP_T,
        H,                 // N = 1024
        1,                 // M = 1
        NH * HD,           // K = 2048
        &alpha,
        h_attn_out_.data(), // A: (1, 2048)
        NH * HD,           // lda
        lw.o_w.data(),     // B: (2048, 1024) row-major → transposed for col-maj
        NH * HD,           // ldb = K for transposed
        &beta,
        h_s2_.data(),      // C: (1, 1024)
        H                  // ldc
    ));

    HIP_CHECK(hipDeviceSynchronize());

    // 8. Residual add: h = saved + o_out
    for (int i = 0; i < H; i++) {
        h[i] = h_s1_[i] + h_s2_[i];
    }

    // 9. Save residual
    memcpy(h_s1_.data(), h, H * sizeof(float));

    // 10. RMSNorm with post_attention_layernorm
    rms_norm(h_s2_.data(), h, pa_w, H);

    // 11. Gate + Up GEMMs: h (1×1024) × W_g (1024×3072) → gate (1×3072)
    alpha = 1.0f; beta = 0.0f;

    // Gate projection: (1, H) × (H, IM) = (1, IM)
    // Gate weight stored row-major (H, IM), transposed for col-maj GEMM
    HIPBLAS_CHECK(hipblasSgemm(
        hipblas_handle_,
        HIPBLAS_OP_N,
        HIPBLAS_OP_T,
        IM,                // N = 3072
        1,                 // M = 1
        H,                 // K = 1024
        &alpha,
        h_s2_.data(),      // A: (1, 1024)
        H,                 // lda
        lw.gate_w.data(),  // B: (1024, 3072) → transposed
        H,                 // ldb
        &beta,
        h_gate_out_.data(), // C: (1, 3072)
        IM                 // ldc
    ));

    // Up projection: (1, H) × (H, IM) = (1, IM)
    HIPBLAS_CHECK(hipblasSgemm(
        hipblas_handle_,
        HIPBLAS_OP_N,
        HIPBLAS_OP_T,
        IM,                // N = 3072
        1,                 // M = 1
        H,                 // K = 1024
        &alpha,
        h_s2_.data(),      // A: (1, 1024)
        H,                 // lda
        lw.up_w.data(),    // B: (1024, 3072) → transposed
        H,                 // ldb
        &beta,
        h_up_out_.data(),  // C: (1, 3072)
        IM                 // ldc
    ));

    HIP_CHECK(hipDeviceSynchronize());

    // 12. SwiGLU: silu(gate) * up
    for (int i = 0; i < IM; i++) {
        float g = h_gate_out_[i];
        if (!std::isfinite(g)) g = 0.0f;
        // silu(x) = x / (1 + exp(-x))
        float sigmoid = 1.0f / (1.0f + expf(-g));
        float silu = g * sigmoid;
        h_swiglu_out_[i] = silu * h_up_out_[i];
    }

    // 13. Down GEMM: swiglu (1×3072) × W_d (3072×1024) → out (1×1024)
    alpha = 1.0f; beta = 0.0f;
    HIPBLAS_CHECK(hipblasSgemm(
        hipblas_handle_,
        HIPBLAS_OP_N,
        HIPBLAS_OP_T,
        H,                 // N = 1024
        1,                 // M = 1
        IM,                // K = 3072
        &alpha,
        h_swiglu_out_.data(), // A: (1, 3072)
        IM,                // lda
        lw.down_w.data(),  // B: (3072, 1024) → transposed
        IM,                // ldb
        &beta,
        h_s2_.data(),      // C: (1, 1024)
        H                  // ldc
    ));

    // 14. Residual add: h = saved + down_out
    for (int i = 0; i < H; i++) {
        h[i] = h_s1_[i] + h_s2_[i];
    }
}

void GpuInferenceEngine::compute_logits(const float* hidden, float* logits) {
    // 1. Final RMSNorm
    std::vector<float> normed(H);
    rms_norm(normed.data(), hidden, final_norm_weight_, H);

    // 2. LM head GEMM: normed (1×1024) × lm_head (1024×151936) → logits (1×151936)
    // lm_head_dequant_ is stored as (H, NV) in row-major for easy GEMM
    float alpha = 1.0f, beta = 0.0f;

    // CPU fallback for now - the LM head is very wide (NV = 151936)
    // This is O(H*NV) = ~155M multiplications per token
    for (int v = 0; v < NV; v++) {
        double s = 0;
        for (int h = 0; h < H; h++) {
            s += (double)normed[h] * lm_head_dequant_[h * NV + v];
        }
        logits[v] = (float)s;
    }
}

int GpuInferenceEngine::sample_token(const float* logits) {
    // Find max
    float mx = logits[0];
    for (int i = 1; i < NV; i++) {
        if (logits[i] > mx) mx = logits[i];
    }

    // Softmax with temperature = 1.0
    std::vector<float> probs(NV);
    double sum = 0;
    for (int i = 0; i < NV; i++) {
        float d = logits[i] - mx;
        d = d > 80 ? 80 : (d < -80 ? -80 : d);
        probs[i] = expf(d);
        sum += probs[i];
    }

    // Sample
    float r = (float)rand() / (float)RAND_MAX * (float)sum;
    float acc = 0;
    for (int i = 0; i < NV; i++) {
        acc += probs[i];
        if (acc >= r) return i;
    }

    return NV - 1;  // Fallback
}

//======= Standalone main() for testing =======

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("=== GPU Inference Engine for Qwen3-0.6B ===\n");
    printf("Using ROCm with hipBLAS\n\n");

    const char* model_path = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
    if (argc > 1) {
        model_path = argv[1];
    }

    GpuInferenceEngine engine;
    if (!engine.init(model_path)) {
        fprintf(stderr, "[ERROR] Failed to initialize engine\n");
        return 1;
    }

    // Test prompt
    int prompt[] = {151643, 872, 198, 11852, 151644, 198, 151643, 77091, 198};
    int prompt_len = sizeof(prompt) / sizeof(prompt[0]);

    printf("\n=== Running inference ===\n");
    printf("Prompt tokens: %d\n", prompt_len);

    int output[256];
    int num_generated = engine.generate(prompt, prompt_len, output, 32);

    printf("\n=== Generated %d tokens ===\n", num_generated);
    printf("Output: ");
    for (int i = 0; i < num_generated; i++) {
        printf("%d ", output[i]);
    }
    printf("\n");

    return 0;
}

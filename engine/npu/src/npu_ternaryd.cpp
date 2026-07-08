/**
 * npu_ternaryd — Native Ternary NPU Inference Daemon
 *
 * Reads packed ternary weights (from q2_0_to_packed.py), dispatches
 * native ternary xclbins via XRT, and provides stdin/stdout JSON API.
 *
 * Protocol (one JSON object per line, stdin→stdout):
 *   Input:  {"tokens": [int...], "max_new_tokens": int}
 *   Output: {"tokens": [int...], "finished": bool, "error": str?}
 *
 * Build:
 *   g++ -std=c++23 -O3 -o npu_ternaryd npu_ternaryd.cpp \
 *       -I/usr/include/xrt -L/usr/lib -lxrt_coreutil -lxrt_core -luuid -lm
 *
 * Usage:
 *   ./npu_ternaryd <model.ternary/> <xclbin_dir/>
 *
 * Model dir (from q2_0_to_packed.py):
 *   model.ternary/
 *   ├── manifest.json   — tensor metadata
 *   └── weights.bin     — packed ternary weights + bf16 scales + f16 norms
 *
 * Xclbin: 32-core single-shot ternary_32c_oneshot.xclbin (M=128 K=256)
 *   Built via: bash engine/npu/build/build_32c_oneshot.sh
 *   Also: single-core ternary_m52_k64.xclbin for partial tiles (M=52 K=256)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <fstream>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

// ═══════════════════════════════════════════════════════════════
//  BF16 helpers
// ═══════════════════════════════════════════════════════════════

static inline uint16_t f2bf(float v) {
    uint32_t b; memcpy(&b, &v, 4);
    return (uint16_t)((b + 0x8000) >> 16);
}
static inline float bf16f(uint16_t v) {
    uint32_t b = (uint32_t)v << 16; float f;
    memcpy(&f, &b, 4); return f;
}
static inline float bf16g(uint16_t v) {
    return (v & 0x7F80) == 0x7F80 ? 0.0f : bf16f(v);
}
static inline float f16f(uint16_t v) {
    // Convert fp16 to float32
    uint32_t s = (v >> 15) & 1;
    uint32_t e = (v >> 10) & 0x1f;
    uint32_t m = v & 0x3ff;
    if (e == 0) { if (m == 0) return s ? -0.0f : 0.0f; float r = m / 1024.0f; r *= 1.0f / 16384.0f; return s ? -r : r; }
    if (e == 31) { return m ? NAN : (s ? -INFINITY : INFINITY); }
    float r = 1.0f + m / 1024.0f;
    r *= ldexpf(1.0f, (int)e - 15);
    return s ? -r : r;
}

// ═══════════════════════════════════════════════════════════════
//  RMSNorm on float32
// ═══════════════════════════════════════════════════════════════

static inline void rms_norm(float* x, const float* w, int n) {
    double ss = 0;
    for (int i = 0; i < n; i++) if (std::isfinite(x[i])) ss += (double)x[i] * x[i];
    float ir = 1.0f / sqrtf((float)(ss / n) + 1e-6f);
    for (int i = 0; i < n; i++) x[i] = std::isfinite(x[i]) ? x[i] * ir * w[i] : 0.0f;
}

// ═══════════════════════════════════════════════════════════════
//  Softmax
// ═══════════════════════════════════════════════════════════════

static inline void softmax(float* sc, int n) {
    if (n <= 0) return;
    float mx = sc[0];
    for (int i = 1; i < n; i++) if (sc[i] > mx) mx = sc[i];
    double s = 0;
    for (int i = 0; i < n; i++) {
        float d = sc[i] - mx;
        if (d > 80) d = 80; else if (d < -80) d = -80;
        sc[i] = expf(d);
        s += sc[i];
    }
    float is = (s > 0) ? 1.0f / (float)s : 1.0f / n;
    for (int i = 0; i < n; i++) sc[i] *= is;
}

// ═══════════════════════════════════════════════════════════════
//  RoPE (simplified — single position)
// ═══════════════════════════════════════════════════════════════

struct RoPECache {
    std::vector<float> cos, sin;
    int head_dim = 0;

    void init(int hd, float theta) {
        head_dim = hd;
        int hd2 = hd / 2;
        cos.resize(hd2);
        sin.resize(hd2);
        for (int d = 0; d < hd2; d++) {
            float f = 1.0f / powf(theta, (float)d / hd2);
            cos[d] = f; sin[d] = f; // placeholder — computed at apply time
        }
    }

    void apply(float* x, int pos) {
        int hd2 = head_dim / 2;
        for (int d = 0; d < hd2; d++) {
            float theta_pos = (float)pos * cos[d];
            float c = cosf(theta_pos);
            float s = sinf(theta_pos);
            float a = x[d], b = x[d + hd2];
            x[d] = a * c - b * s;
            x[d + hd2] = b * c + a * s;
        }
    }
};

// ═══════════════════════════════════════════════════════════════
//  Native Ternary xclbin context (one per projection)
// ═══════════════════════════════════════════════════════════════

struct TernaryCtx {
    static constexpr int K_TERNARY = 256;   // ternary values per kernel call
    static constexpr int K_PACKED  = 64;    // packed bytes for K=256
    static constexpr int M_SINGLE  = 52;    // rows per single-core dispatch (max L1 fit)
    static constexpr int M_MULTI   = 128;   // rows per 32-core dispatch
    static constexpr int N_COLS    = 8;     // 32-core grid columns
    static constexpr int N_ROWS    = 4;     // 32-core grid rows

    int M_total, K_total;  // dimensions
    int K_packed_total;    // K_total / 4 (actual packed bytes per row)
    xrt::device* device = nullptr;

    // Single-core xclbin (M=52, remainder tiles)
    xrt::uuid sc_uuid;
    std::vector<uint32_t> sc_instr;

    // 32-core xclbin (M=128, bulk tiles)
    xrt::uuid mc_uuid;
    std::vector<uint32_t> mc_instr;

    // Pre-loaded weight buffer: [M_total * K_packed uint8] [M_total * 2 bf16]
    std::unique_ptr<xrt::bo> bo_weights;
    size_t weights_bytes = 0, scales_bytes = 0;

    bool init(xrt::device& dev,
              const xrt::uuid& single_uuid, const std::vector<uint32_t>& single_instr,
              const xrt::uuid& multi_uuid,  const std::vector<uint32_t>& multi_instr,
              const uint8_t* packed_weights,
              const uint16_t* weight_scales,
              int M, int K) {
        M_total = M; K_total = K;
        K_packed_total = K_total / 4;
        device = &dev;
        sc_uuid = single_uuid;  sc_instr = single_instr;
        mc_uuid = multi_uuid;   mc_instr = multi_instr;

        weights_bytes = (size_t)M_total * K_packed_total;
        scales_bytes = (size_t)M_total * 2;
        bo_weights = std::make_unique<xrt::bo>(dev, weights_bytes + scales_bytes,
                                                xrt::bo::flags::host_only, 0);
        memcpy(bo_weights->map(), packed_weights, weights_bytes);
        memcpy((char*)bo_weights->map() + weights_bytes, weight_scales, scales_bytes);
        bo_weights->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        fprintf(stderr, "  [TernaryCtx] M=%d K=%d weights=%zuB (SC=%d MC=%d) OK\n",
                M, K, weights_bytes + scales_bytes, M_SINGLE, M_MULTI);
        return true;
    }

    // ── dispatch helpers ─────────────────────────────────────

    // Dispatch single-core kernel (flat buffer layout)
    void dispatch_sc(const uint16_t* act_slice, int k_act_count, int m_chunk, int m_start,
                     int k_start, uint16_t* output_bf16) {
        auto hc = std::make_unique<xrt::hw_context>(*device, sc_uuid);
        auto k  = std::make_unique<xrt::kernel>(*hc, "MLIR_AIE");
        int wg = k->group_id(3), og = k->group_id(5), ig = k->group_id(1);

        int tile_in  = M_SINGLE * K_PACKED + M_SINGLE * 2 + K_TERNARY * 2;
        int tile_out = M_SINGLE * 2;
        auto bo_in  = std::make_unique<xrt::bo>(*device, tile_in, xrt::bo::flags::host_only, wg);
        auto bo_out = std::make_unique<xrt::bo>(*device, tile_out, xrt::bo::flags::host_only, og);

        const uint8_t*  wgt = (const uint8_t*)bo_weights->map();
        const uint16_t* sc  = (const uint16_t*)((const char*)bo_weights->map() + weights_bytes);
        uint8_t* in_map = (uint8_t*)bo_in->map();
        int k_packed_chunk = k_act_count / 4;

        // Weight slice (from k_start offset within each row)
        for (int r = 0; r < m_chunk; r++) {
            memcpy(in_map + r * K_PACKED,
                   wgt + (size_t)(m_start + r) * K_packed_total + k_start / 4,
                   (size_t)k_packed_chunk);
        }
        if (k_packed_chunk < K_PACKED)
            for (int r = 0; r < m_chunk; r++)
                memset(in_map + r * K_PACKED + k_packed_chunk, 0, K_PACKED - k_packed_chunk);

        // Scale slice
        memcpy(in_map + M_SINGLE * K_PACKED, sc + m_start, (size_t)m_chunk * 2);

        // Activation slice
        uint16_t* act_dst = (uint16_t*)(in_map + M_SINGLE * K_PACKED + M_SINGLE * 2);
        memcpy(act_dst, act_slice, (size_t)k_act_count * 2);
        if (k_act_count < K_TERNARY)
            memset(act_dst + k_act_count, 0, (K_TERNARY - k_act_count) * 2);

        bo_in->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto bo_instr = xrt::bo(*device, sc_instr.size() * 4, XCL_BO_FLAGS_CACHEABLE, ig);
        memcpy(bo_instr.map(), sc_instr.data(), sc_instr.size() * 4);
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto run = (*k)((unsigned)3, bo_instr, (unsigned)sc_instr.size(), *bo_in, *bo_out);
        run.wait();

        bo_out->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        uint16_t* out_map = (uint16_t*)bo_out->map();
        for (int r = 0; r < m_chunk; r++) {
            float partial = bf16f(out_map[r]);
            float cur = bf16f(output_bf16[m_start + r]);
            output_bf16[m_start + r] = f2bf(cur + partial);
        }
    }

    // Dispatch 32-core kernel (per-column data layout, f32 output)
    void dispatch_mc(const uint16_t* act_slice, int k_act_count, int m_start,
                     int k_start, uint16_t* output_bf16) {
        auto hc = std::make_unique<xrt::hw_context>(*device, mc_uuid);
        auto k  = std::make_unique<xrt::kernel>(*hc, "MLIR_AIE");
        int wg = k->group_id(3), og = k->group_id(5), ig = k->group_id(1);

        int m_per_core = M_MULTI / (N_COLS * N_ROWS);  // 4
        int m_per_col  = m_per_core * N_ROWS;           // 16
        size_t col_in  = (size_t)m_per_col * K_PACKED + (size_t)m_per_col * 2 + (size_t)K_TERNARY * 2;
        size_t col_in_dw = (col_in + 3) / 4;             // 392
        size_t col_out_elems = (size_t)m_per_core * N_ROWS; // 16 f32

        auto bo_in  = std::make_unique<xrt::bo>(*device, col_in_dw * N_COLS * 4, xrt::bo::flags::host_only, wg);
        auto bo_out = std::make_unique<xrt::bo>(*device, col_out_elems * N_COLS * 4, xrt::bo::flags::host_only, og);

        const uint8_t*  wgt = (const uint8_t*)bo_weights->map();
        const uint16_t* sc  = (const uint16_t*)((const char*)bo_weights->map() + weights_bytes);
        uint8_t* in_map = (uint8_t*)bo_in->map();

        // Per-column data layout
        for (int col = 0; col < N_COLS; col++) {
            uint8_t* col_ptr = in_map + col * col_in_dw * 4;
            for (int row = 0; row < N_ROWS; row++) {
                int rs = m_start + col * m_per_col + row * m_per_core;
                if (rs < M_total) {
                    memcpy(col_ptr, wgt + (size_t)rs * K_packed_total + k_start / 4, (size_t)m_per_core * K_PACKED);
                    col_ptr += (size_t)m_per_core * K_PACKED;
                    memcpy(col_ptr, sc + rs, (size_t)m_per_core * 2);
                    col_ptr += (size_t)m_per_core * 2;
                } else {
                    memset(col_ptr, 0, (size_t)m_per_core * (K_PACKED + 2));
                    col_ptr += (size_t)m_per_core * (K_PACKED + 2);
                }
            }
            memcpy(col_ptr, act_slice, (size_t)k_act_count * 2);
            if (k_act_count < K_TERNARY)
                memset(col_ptr + k_act_count * 2, 0, (K_TERNARY - k_act_count) * 2);
        }
        bo_in->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto bo_instr = xrt::bo(*device, mc_instr.size() * 4, XCL_BO_FLAGS_CACHEABLE, ig);
        memcpy(bo_instr.map(), mc_instr.data(), mc_instr.size() * 4);
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto run = (*k)((unsigned)3, bo_instr, (unsigned)mc_instr.size(), *bo_in, *bo_out);
        run.wait();

        // Read f32 output, reorder from per-column layout, accumulate as bf16
        bo_out->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        const float* out_f32 = (const float*)bo_out->map();
        for (int r = 0; r < M_MULTI && (m_start + r) < M_total; r++) {
            int col  = r / m_per_col;
            int ri   = r % m_per_col;
            int core = ri / m_per_core;
            int loc  = ri % m_per_core;
            size_t idx = (size_t)col * col_out_elems + (size_t)core * m_per_core + loc;
            float partial = out_f32[idx];
            float cur = bf16f(output_bf16[m_start + r]);
            output_bf16[m_start + r] = f2bf(cur + partial);
        }
    }

    /**
     * Tiled GEMV: uses 32-core xclbin for 128-row chunks, single-core for remainder.
     * Both xclbins are single-shot — fresh hw_context per dispatch.
     */
    void gemv(const uint16_t* activation_bf16, uint16_t* output_bf16) {
        memset(output_bf16, 0, (size_t)M_total * 2);

        for (int m_start = 0; m_start < M_total; ) {
            int remaining = M_total - m_start;
            int m_chunk = (remaining >= M_MULTI) ? M_MULTI : M_SINGLE;
            if (m_chunk > remaining) m_chunk = remaining;
            bool use_multi = (m_chunk >= M_MULTI);

            for (int k_start = 0; k_start < K_total; k_start += K_TERNARY) {
                int k_act_count = std::min(K_TERNARY, K_total - k_start);

                if (use_multi)
                    dispatch_mc(activation_bf16 + k_start, k_act_count, m_start, k_start, output_bf16);
                else
                    dispatch_sc(activation_bf16 + k_start, k_act_count, m_chunk, m_start, k_start, output_bf16);
            }

            m_start += m_chunk;
        }
    }
};

// ═══════════════════════════════════════════════════════════════
//  Model loader (packed ternary format)
// ═══════════════════════════════════════════════════════════════

struct PackedModel {
    int hidden_size = 1024;
    int num_layers = 28;
    int vocab_size = 151936;
    int num_heads = 16;
    int num_kv_heads = 8;
    int head_dim = 128;
    int intermediate_size = 2048;
    float rope_theta = 1000000.0f;

    // Mapped model file
    uint8_t* model_base = nullptr;
    size_t model_size = 0;

    // Norm weights (f32, converted from f16 on load)
    std::vector<float> final_norm_w;
    std::vector<std::vector<float>> layer_in_norm, layer_pa_norm;
    std::vector<std::vector<float>> layer_q_norm, layer_k_norm;

    // Embeddings and lm_head (f16, read on-the-fly)
    int embed_elems = 0;  // vocab_size * hidden_size
    size_t embed_offset = 0;  // byte offset in model_base

    RoPECache rope;

    struct LayerWeights {
        uint8_t* q_weights = nullptr;
        uint16_t* q_scales = nullptr;
        int q_M = 0, q_K = 0;
        uint8_t* k_weights = nullptr;
        uint16_t* k_scales = nullptr;
        int k_M = 0, k_K = 0;
        uint8_t* v_weights = nullptr;
        uint16_t* v_scales = nullptr;
        int v_M = 0, v_K = 0;
        uint8_t* o_weights = nullptr;
        uint16_t* o_scales = nullptr;
        int o_M = 0, o_K = 0;
        uint8_t* up_weights = nullptr;
        uint16_t* up_scales = nullptr;
        int up_M = 0, up_K = 0;
        uint8_t* gate_weights = nullptr;
        uint16_t* gate_scales = nullptr;
        int gate_M = 0, gate_K = 0;
        uint8_t* down_weights = nullptr;
        uint16_t* down_scales = nullptr;
        int down_M = 0, down_K = 0;
    };

    std::vector<LayerWeights> layers;

    bool load(const std::string& model_dir) {
        std::string manifest_path = model_dir + "/manifest.json";
        std::string weights_path = model_dir + "/weights.bin";

        // Read manifest
        std::ifstream mf(manifest_path);
        if (!mf) { fprintf(stderr, "manifest.json not found: %s\n", manifest_path.c_str()); return false; }
        std::string mjson((std::istreambuf_iterator<char>(mf)),
                          std::istreambuf_iterator<char>());

        // mmap weights.bin
        int fd = open(weights_path.c_str(), O_RDONLY);
        if (fd < 0) { fprintf(stderr, "weights.bin not found: %s\n", weights_path.c_str()); return false; }
        struct stat st; fstat(fd, &st);
        model_size = st.st_size;
        model_base = (uint8_t*)mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);

        // Parse manifest (simple key-value extraction — no full JSON parser needed)
        layers.resize(num_layers);
        layer_in_norm.resize(num_layers);
        layer_pa_norm.resize(num_layers);
        layer_q_norm.resize(num_layers);
        layer_k_norm.resize(num_layers);

        // Extract tensor entries from manifest
        // Format: {"tensor.name": {"M":..., "K":..., "offset_weights":..., ...}, ...}

        for (int l = 0; l < num_layers; l++) {
            auto& lw = layers[l];
            char key[256];

            snprintf(key, 256, "model.layers.%d.self_attn.q_proj.weight", l);
            load_packed_weights(key, &lw.q_weights, &lw.q_scales, &lw.q_M, &lw.q_K);
            snprintf(key, 256, "model.layers.%d.self_attn.k_proj.weight", l);
            load_packed_weights(key, &lw.k_weights, &lw.k_scales, &lw.k_M, &lw.k_K);
            snprintf(key, 256, "model.layers.%d.self_attn.v_proj.weight", l);
            load_packed_weights(key, &lw.v_weights, &lw.v_scales, &lw.v_M, &lw.v_K);
            snprintf(key, 256, "model.layers.%d.self_attn.o_proj.weight", l);
            load_packed_weights(key, &lw.o_weights, &lw.o_scales, &lw.o_M, &lw.o_K);
            snprintf(key, 256, "model.layers.%d.mlp.up_proj.weight", l);
            load_packed_weights(key, &lw.up_weights, &lw.up_scales, &lw.up_M, &lw.up_K);
            snprintf(key, 256, "model.layers.%d.mlp.gate_proj.weight", l);
            load_packed_weights(key, &lw.gate_weights, &lw.gate_scales, &lw.gate_M, &lw.gate_K);
            snprintf(key, 256, "model.layers.%d.mlp.down_proj.weight", l);
            load_packed_weights(key, &lw.down_weights, &lw.down_scales, &lw.down_M, &lw.down_K);

            // Norm weights (f16 from GGUF)
            snprintf(key, 256, "model.layers.%d.input_layernorm.weight", l);
            load_norm_weights(key, layer_in_norm[l], hidden_size);
            snprintf(key, 256, "model.layers.%d.post_attention_layernorm.weight", l);
            load_norm_weights(key, layer_pa_norm[l], hidden_size);
            snprintf(key, 256, "model.layers.%d.self_attn.q_norm.weight", l);
            load_norm_weights(key, layer_q_norm[l], head_dim);
            snprintf(key, 256, "model.layers.%d.self_attn.k_norm.weight", l);
            load_norm_weights(key, layer_k_norm[l], head_dim);
        }

        // Final norm
        load_norm_weights("model.norm.weight", final_norm_w, hidden_size);

        // Embeddings / lm_head
        load_embed_table("model.embed_tokens.weight");

        rope.init(head_dim, rope_theta);

        // Auto-detect architecture from first layer weights
        if (!layers.empty() && layers[0].q_K > 0) {
            hidden_size = layers[0].q_K * 4;  // K_packed * 4 = K (ternary values)
            num_heads = layers[0].q_M / head_dim;
            num_kv_heads = layers[0].k_M / head_dim;
            if (layers[0].up_M > 0) intermediate_size = layers[0].up_M;
            fprintf(stderr, "[PackedModel] Auto-detected: H=%d IM=%d NH=%d NKV=%d\n",
                    hidden_size, intermediate_size, num_heads, num_kv_heads);
        }

        fprintf(stderr, "[PackedModel] Loaded %d layers, H=%d NH=%d NKV=%d HD=%d\n",
                num_layers, hidden_size, num_heads, num_kv_heads, head_dim);
        return true;
    }

private:
    // Simple JSON value extractor (no dependencies)
    int64_t extract_int(const std::string& json, const std::string& key) {
        auto kq = "\"" + key + "\"";
        auto p = json.find(kq);
        if (p == std::string::npos) return -1;
        p += kq.size();
        while (p < json.size() && (json[p] == ':' || json[p] == ' ')) p++;
        return strtoll(json.c_str() + p, nullptr, 10);
    }

    void load_packed_weights(const char* name, uint8_t** out_weights, uint16_t** out_scales,
                             int* out_M, int* out_K) {
        // Linear search in manifest JSON for name
        std::string needle = "\"" + std::string(name) + "\"";
        auto p = model_json.find(needle);
        if (p == std::string::npos) {
            *out_weights = nullptr;
            *out_scales = nullptr;
            return;
        }

        size_t off_w = (size_t)extract_int(model_json.substr(p), "offset_weights");
        size_t off_s = (size_t)extract_int(model_json.substr(p), "offset_scales");
        *out_weights = model_base + off_w;
        *out_scales = (uint16_t*)(model_base + off_s);
        *out_M = (int)extract_int(model_json.substr(p), "M");
        *out_K = (int)extract_int(model_json.substr(p), "K_packed");
        if (*out_K <= 0) *out_K = (int)extract_int(model_json.substr(p), "K") / 4;
    }

    void load_norm_weights(const char* name, std::vector<float>& out, int n) {
        out.resize(n);
        std::string needle = "\"" + std::string(name) + "\"";
        auto p = model_json.find(needle);
        if (p == std::string::npos) {
            memset(out.data(), 0, n * 4);
            return;
        }
        size_t off = (size_t)extract_int(model_json.substr(p), "offset_bytes");
        // Norm weights are f16 in GGUF
        auto* src = (const uint16_t*)(model_base + off);
        for (int i = 0; i < n; i++) out[i] = f16f(src[i]);
    }

    void load_embed_table(const char* name) {
        std::string needle = "\"" + std::string(name) + "\"";
        auto p = model_json.find(needle);
        if (p == std::string::npos) return;
        embed_offset = (size_t)extract_int(model_json.substr(p), "offset_bytes");
        embed_elems = vocab_size * hidden_size;
    }

    // Cached manifest for lookups (set by load)
    std::string model_json;

public:
    void set_manifest(const std::string& json) { model_json = json; }

    void embed_token(int token, float* out) {
        auto* src = (const uint16_t*)(model_base + embed_offset + (size_t)token * hidden_size * 2);
        for (int i = 0; i < hidden_size; i++) out[i] = f16f(src[i]);
    }

    void lm_head(const float* hidden, float* logits) {
        auto* emb = (const uint16_t*)(model_base + embed_offset);
        for (int v = 0; v < vocab_size; v++) {
            double s = 0;
            for (int k = 0; k < hidden_size; k++)
                s += (double)hidden[k] * f16f(emb[(size_t)v * hidden_size + k]);
            logits[v] = (float)s;
        }
    }
};

// ═══════════════════════════════════════════════════════════════
//  NPU Ternary Daemon
// ═══════════════════════════════════════════════════════════════

struct TernaryDaemon {
    PackedModel model;
    xrt::device device{0};
    std::unique_ptr<xrt::xclbin> sc_xclbin;   // single-core (M=52)
    std::unique_ptr<xrt::xclbin> mc_xclbin;   // 32-core (M=128)

    // One TernaryCtx per (projection, layer) — xclbin shared, weights differ
    struct LayerCtx {
        std::unique_ptr<TernaryCtx> q, k, v, o, up, gate, down;
    };
    std::vector<LayerCtx> layer_ctxs;
    int max_pos = 32;  // max KV cache positions

    bool init(const std::string& model_dir, const std::string& xclbin_dir) {
        // Load manifest JSON
        std::string manifest_path = model_dir + "/manifest.json";
        std::ifstream mf(manifest_path);
        if (!mf) { fprintf(stderr, "manifest.json not found\n"); return false; }
        std::string mjson((std::istreambuf_iterator<char>(mf)),
                          std::istreambuf_iterator<char>());

        model.set_manifest(mjson);
        if (!model.load(model_dir)) { fprintf(stderr, "Model load failed\n"); return false; }

        // Load xclbins (one per projection type, shared across layers)
        std::string xb = xclbin_dir;
        if (xb.back() != '/') xb += '/';

        layer_ctxs.resize(model.num_layers);

        int H = model.hidden_size;
        int HD = model.head_dim;
        int NH = model.num_heads;
        int NKV = model.num_kv_heads;
        int IM = model.intermediate_size;

        int q_dim = NH * HD;   // 2048
        int kv_dim = NKV * HD; // 1024

        fprintf(stderr, "[Daemon] Loading %d layers of xclbins...\n", model.num_layers);
        fflush(stderr);
        fprintf(stderr, "  Q: M=%d K=%d  K: M=%d K=%d  V: M=%d K=%d\n", q_dim, H, kv_dim, H, kv_dim, H);
        fflush(stderr);
        fprintf(stderr, "  O: M=%d K=%d  Up: M=%d K=%d  Gate: M=%d K=%d  Down: M=%d K=%d\n",
                H, q_dim, IM, H, IM, H, H, IM);

        // For now: use one xclbin per dimension set
        // QKV dims: M=q_dim/kv_dim, K=H
        // O dims: M=H, K=q_dim
        // Up/Gate dims: M=IM, K=H (same xclbin)
        // Down dims: M=H, K=IM

        // Two xclbins: 32-core for bulk (M=128), single-core for remainder (M=52)
        // Xclbin dir must contain:
        //   ternary_32c_oneshot.xclbin + insts_ternary_32c_oneshot.txt
        //   ternary_m52_k64.xclbin     + insts_ternary_m52_k64.txt
        std::string mc_xclbin_file = xb + "ternary_32c_oneshot.xclbin";
        std::string mc_instr_file  = xb + "insts_ternary_32c_oneshot.txt";
        std::string sc_xclbin_file = xb + "ternary_m52_k64.xclbin";
        std::string sc_instr_file  = xb + "insts_ternary_m52_k64.txt";

        auto load_xclbin = [&](const std::string& path, xrt::uuid& uuid, std::vector<uint32_t>& instr) -> bool {
            std::ifstream xf(path, std::ios::binary);
            if (!xf) { fprintf(stderr, "xclbin not found: %s\n", path.c_str()); return false; }
            xf.seekg(0, std::ios::end); auto xs = (size_t)xf.tellg(); xf.seekg(0);
            std::vector<char> xd(xs); xf.read(xd.data(), xs);
            auto xc = std::make_unique<xrt::xclbin>(xd);
            device.register_xclbin(*xc);
            uuid = xc->get_uuid();
            // Store in appropriate member based on path
            if (path.find("32c") != std::string::npos)
                mc_xclbin = std::move(xc);
            else
                sc_xclbin = std::move(xc);
            return true;
        };
        auto load_instr = [&](const std::string& path, std::vector<uint32_t>& instr) -> bool {
            std::ifstream nf(path, std::ios::binary);
            if (!nf) { fprintf(stderr, "instr not found: %s\n", path.c_str()); return false; }
            nf.seekg(0, std::ios::end); auto ns = (size_t)nf.tellg(); nf.seekg(0);
            instr.resize(ns / 4 + 1);
            nf.read((char*)instr.data(), ns);
            return true;
        };

        xrt::uuid sc_uuid, mc_uuid;
        std::vector<uint32_t> sc_instr, mc_instr;
        if (!load_xclbin(sc_xclbin_file, sc_uuid, sc_instr)) return false;
        if (!load_instr(sc_instr_file, sc_instr)) return false;
        if (!load_xclbin(mc_xclbin_file, mc_uuid, mc_instr)) return false;
        if (!load_instr(mc_instr_file, mc_instr)) return false;

        for (int l = 0; l < model.num_layers; l++) {
            auto& lw = model.layers[l];
            auto& lc = layer_ctxs[l];

            lc.q = std::make_unique<TernaryCtx>();
            if (!lc.q->init(device, sc_uuid, sc_instr, mc_uuid, mc_instr,
                            lw.q_weights, lw.q_scales, lw.q_M, lw.q_K * 4)) return false;

            lc.k = std::make_unique<TernaryCtx>();
            if (!lc.k->init(device, sc_uuid, sc_instr, mc_uuid, mc_instr,
                            lw.k_weights, lw.k_scales, lw.k_M, lw.k_K * 4)) return false;

            lc.v = std::make_unique<TernaryCtx>();
            if (!lc.v->init(device, sc_uuid, sc_instr, mc_uuid, mc_instr,
                            lw.v_weights, lw.v_scales, lw.v_M, lw.v_K * 4)) return false;

            lc.o = std::make_unique<TernaryCtx>();
            if (!lc.o->init(device, sc_uuid, sc_instr, mc_uuid, mc_instr,
                            lw.o_weights, lw.o_scales, lw.o_M, lw.o_K * 4)) return false;

            lc.up = std::make_unique<TernaryCtx>();
            if (!lc.up->init(device, sc_uuid, sc_instr, mc_uuid, mc_instr,
                             lw.up_weights, lw.up_scales, lw.up_M, lw.up_K * 4)) return false;

            lc.gate = std::make_unique<TernaryCtx>();
            if (!lc.gate->init(device, sc_uuid, sc_instr, mc_uuid, mc_instr,
                               lw.gate_weights, lw.gate_scales, lw.gate_M, lw.gate_K * 4)) return false;

            lc.down = std::make_unique<TernaryCtx>();
            if (!lc.down->init(device, sc_uuid, sc_instr, mc_uuid, mc_instr,
                               lw.down_weights, lw.down_scales, lw.down_M, lw.down_K * 4)) return false;
        }

        fprintf(stderr, "[Daemon] All %d layers loaded. Ready.\n", model.num_layers);
        return true;
    }

    void layer_forward(int l, float* hidden, std::vector<uint16_t>& buf_bf16,
                       int pos, RoPECache& rope) {
        fprintf(stderr, "  [L%d] start\n", l);
        int H = model.hidden_size;
        int HD = model.head_dim;
        int NH = model.num_heads;
        int NKV = model.num_kv_heads;
        int IM = model.intermediate_size;
        int q_dim = NH * HD;
        int kv_dim = NKV * HD;

        auto& lc = layer_ctxs[l];

        // Input RMSNorm
        float residual[H];
        memcpy(residual, hidden, H * 4);
        rms_norm(hidden, model.layer_in_norm[l].data(), H);

        // ── Q projection ──────────────────────────────────
        // Convert hidden (f32) → bf16
        for (int i = 0; i < H; i++) buf_bf16[i] = f2bf(hidden[i]);
        std::vector<uint16_t> q_out(q_dim);
        lc.q->gemv(buf_bf16.data(), q_out.data());
        fprintf(stderr, "  [L%d] Q done\n", l);

        // ── K projection ──────────────────────────────────
        std::vector<uint16_t> k_out(kv_dim);
        lc.k->gemv(buf_bf16.data(), k_out.data());
        fprintf(stderr, "  [L%d] K done\n", l);

        // ── V projection ──────────────────────────────────
        std::vector<uint16_t> v_out(kv_dim);
        lc.v->gemv(buf_bf16.data(), v_out.data());
        fprintf(stderr, "  [L%d] V done\n", l);

        // Convert to f32 for attention
        std::vector<float> q_f32(q_dim), k_f32(kv_dim), v_f32(kv_dim);
        fprintf(stderr, "  [L%d] q_out[0]=%f k_out[0]=%f v_out[0]=%f\n", l,
                bf16f(q_out[0]), bf16f(k_out[0]), bf16f(v_out[0]));
        for (int i = 0; i < q_dim; i++) q_f32[i] = bf16f(q_out[i]);
        for (int i = 0; i < kv_dim; i++) k_f32[i] = bf16f(k_out[i]);
        for (int i = 0; i < kv_dim; i++) v_f32[i] = bf16f(v_out[i]);
        fprintf(stderr, "  [L%d] f32 convert done\n", l);

        // Q/K norms + RoPE
        rms_norm(q_f32.data(), model.layer_q_norm[l].data(), HD);
        rms_norm(k_f32.data(), model.layer_k_norm[l].data(), HD);
        // Apply RoPE per head
        for (int h = 0; h < NH; h++)
            rope.apply(q_f32.data() + h * HD, pos);
        for (int h = 0; h < NKV; h++)
            rope.apply(k_f32.data() + h * HD, pos);

        // ── Attention (self-attend, no KV cache yet) ──────
        // For single-token: Q[seq=1] @ K[seq=1]^T → single score → V[seq=1]
        // Simplified: identity (since single token attends to itself)
        // Full multi-head attention for longer sequences requires KV cache
        std::vector<float> attn_out(q_dim);  // q_dim = NH * HD
        // Self-attention: Q @ K^T * V (for single token, this passes through)
        for (int h = 0; h < NH; h++) {
            int kvh = h / (NH / NKV);  // GQA: map head to KV head
            float* qh = q_f32.data() + h * HD;
            float* kh = k_f32.data() + kvh * HD;
            float* vh = v_f32.data() + kvh * HD;

            // Dot product attention score
            double score = 0;
            for (int d = 0; d < HD; d++) score += (double)qh[d] * kh[d];
            score *= 1.0 / sqrtf((float)HD);

            // Single position → softmax is trivially 1.0
            // attn = V * 1.0
            float* aoh = attn_out.data() + h * HD;
            for (int d = 0; d < HD; d++) aoh[d] = vh[d];
        }

        // ── O projection ──────────────────────────────────
        // Convert attn_out → bf16
        for (int i = 0; i < q_dim; i++) buf_bf16[i] = f2bf(attn_out[i]);
        std::vector<uint16_t> o_out(H);
        lc.o->gemv(buf_bf16.data(), o_out.data());
        fprintf(stderr, "  [L%d] O done\n", l);

        // Residual + post-attention norm
        for (int i = 0; i < H; i++) {
            hidden[i] = residual[i] + bf16f(o_out[i]);
        }
        memcpy(residual, hidden, H * 4);
        rms_norm(hidden, model.layer_pa_norm[l].data(), H);

        // ── FFN: Up, Gate, SwiGLU, Down ──────────────────
        for (int i = 0; i < H; i++) buf_bf16[i] = f2bf(hidden[i]);

        std::vector<uint16_t> up_out(IM), gate_out(IM);
        lc.up->gemv(buf_bf16.data(), up_out.data());
        fprintf(stderr, "  [L%d] Up done\n", l);
        lc.gate->gemv(buf_bf16.data(), gate_out.data());
        fprintf(stderr, "  [L%d] Gate done\n", l);

        // SwiGLU (CPU)
        for (int i = 0; i < IM; i++) {
            float u = bf16f(up_out[i]);
            float g = bf16f(gate_out[i]);
            // SiLU(g) * u = g * sigmoid(g) * u
            float silu = g / (1.0f + expf(-g));
            up_out[i] = f2bf(u * silu);
        }

        std::vector<uint16_t> down_out(H);
        lc.down->gemv(up_out.data(), down_out.data());
        fprintf(stderr, "  [L%d] Down done\n", l);

        // Final residual
        for (int i = 0; i < H; i++)
            hidden[i] = residual[i] + bf16f(down_out[i]);
    }

    void generate(const std::vector<int>& tokens, int max_new,
                  std::vector<int>& out_tokens) {
        int H = model.hidden_size;
        std::vector<float> hidden(H);
        std::vector<uint16_t> buf_bf16(H);
        std::vector<float> logits(model.vocab_size);

        // Embed the last token
        model.embed_token(tokens.back(), hidden.data());

        RoPECache rope;
        rope.init(model.head_dim, model.rope_theta);

        // Prefill: run all layers once
        for (int l = 0; l < model.num_layers; l++) {
            layer_forward(l, hidden.data(), buf_bf16, 0, rope);
        }

        // Decode loop
        for (int step = 0; step < max_new; step++) {
            // Final RMSNorm
            float norm_hidden[H];
            memcpy(norm_hidden, hidden.data(), H * 4);
            rms_norm(norm_hidden, model.final_norm_w.data(), H);

            // LM head → logits → argmax
            model.lm_head(norm_hidden, logits.data());

            float best = -1e30f;
            int best_tok = 0;
            for (int v = 0; v < model.vocab_size; v++) {
                if (logits[v] > best) { best = logits[v]; best_tok = v; }
            }
            out_tokens.push_back(best_tok);

            // Stop if EOS (approximate: last few tokens)
            if (best_tok >= model.vocab_size - 5) break;

            // Embed next token and run layers
            model.embed_token(best_tok, hidden.data());
            for (int l = 0; l < model.num_layers; l++) {
                layer_forward(l, hidden.data(), buf_bf16, step + 1, rope);
            }
        }
    }
};

// ═══════════════════════════════════════════════════════════════
//  JSON helpers (minimal, no dependencies)
// ═══════════════════════════════════════════════════════════════

static int json_get_int(const char* js, size_t jl, const char* key) {
    std::string kq = "\"" + std::string(key) + "\"";
    auto p = strstr(js, kq.c_str());
    if (!p) return -1;
    p += kq.size();
    while (*p == ':' || *p == ' ') p++;
    return atoi(p);
}

static std::vector<int> json_get_int_array(const char* js, size_t jl, const char* key) {
    std::vector<int> result;
    std::string kq = "\"" + std::string(key) + "\"";
    auto p = strstr(js, kq.c_str());
    if (!p) return result;
    p += kq.size();
    while (*p == ':' || *p == ' ' || *p == '[') p++;
    while (*p && *p != ']') {
        char* end;
        long val = strtol(p, &end, 10);
        if (end == p) break;
        result.push_back((int)val);
        p = end;
        while (*p == ',' || *p == ' ') p++;
    }
    return result;
}

// ═══════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.ternary/> <xclbin_dir/>\n", argv[0]);
        fprintf(stderr, "  Reads JSON from stdin, writes JSON to stdout.\n");
        fprintf(stderr, "  Input:  {\"tokens\": [...], \"max_new_tokens\": N}\n");
        fprintf(stderr, "  Output: {\"tokens\": [...], \"finished\": bool}\n");
        return 1;
    }

    std::string model_dir = argv[1];
    std::string xclbin_dir = argv[2];

    setvbuf(stdout, nullptr, _IONBF, 0);
    fprintf(stderr, "[npu_ternaryd] Loading model from %s\n", model_dir.c_str());
    fprintf(stderr, "[npu_ternaryd] Xclbins from %s\n", xclbin_dir.c_str());

    TernaryDaemon daemon;
    if (!daemon.init(model_dir, xclbin_dir)) {
        fprintf(stderr, "[npu_ternaryd] Init failed\n");
        return 1;
    }

    fprintf(stderr, "[npu_ternaryd] Ready — waiting for requests...\n");

    // Request loop
    char line[65536];
    while (fgets(line, sizeof(line), stdin)) {
        size_t ll = strlen(line);
        while (ll > 0 && (line[ll-1] == '\n' || line[ll-1] == '\r')) line[--ll] = 0;
        if (ll == 0) continue;

        auto tokens = json_get_int_array(line, ll, "tokens");
        int max_new = json_get_int(line, ll, "max_new_tokens");
        if (max_new <= 0) max_new = 64;

        if (tokens.empty()) {
            printf("{\"error\":\"no tokens in request\"}\n");
            continue;
        }

        std::vector<int> out_tokens;
        daemon.generate(tokens, max_new, out_tokens);

        printf("{\"tokens\":[");
        for (size_t i = 0; i < out_tokens.size(); i++) {
            if (i) printf(",");
            printf("%d", out_tokens[i]);
        }
        printf("],\"finished\":true}\n");
    }

    fprintf(stderr, "[npu_ternaryd] stdin closed, exiting.\n");
    return 0;
}

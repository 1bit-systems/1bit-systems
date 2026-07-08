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
 * Xclbin: single-shot single-core ternary_oneshot.xclbin
 *   Built via: bash engine/npu/build/build_oneshot_xclbin.sh
 *   One xclbin for all projections — host tiles M and K dimensions.
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
    static constexpr int M_PER_CORE = 32;   // output rows per kernel call

    int M_total, K_total;  // dimensions
    xrt::device* device = nullptr;

    std::unique_ptr<xrt::xclbin> xc;
    xrt::uuid xclbin_uuid;
    std::vector<uint32_t> instr;

    // Pre-loaded weight buffer: [M_total * K_packed uint8] [M_total * 2 bf16]
    std::unique_ptr<xrt::bo> bo_weights;
    size_t weights_bytes = 0, scales_bytes = 0;

    // IO buffer sizes per tile (fits single-tile data: M_PER_CORE rows × K_TERNARY values)
    static constexpr int TILE_IN_BYTES  = M_PER_CORE * K_PACKED + M_PER_CORE * 2 + K_TERNARY * 2;
    static constexpr int TILE_OUT_BYTES = M_PER_CORE * 2;  // bf16

    bool init(xrt::device& dev,
              const std::string& xclbin_path,
              const std::string& instr_path,
              const uint8_t* packed_weights,  // [M_total * K_packed]
              const uint16_t* weight_scales,  // [M_total] bf16
              int M, int K) {
        M_total = M; K_total = K;
        device = &dev;

        // Load xclbin once, cache uuid
        std::ifstream xf(xclbin_path, std::ios::binary);
        if (!xf) { fprintf(stderr, "  [TernaryCtx] xclbin not found: %s\n", xclbin_path.c_str()); return false; }
        xf.seekg(0, std::ios::end); auto xs = (size_t)xf.tellg(); xf.seekg(0);
        std::vector<char> xd(xs); xf.read(xd.data(), xs);
        xc = std::make_unique<xrt::xclbin>(xd);
        dev.register_xclbin(*xc);
        xclbin_uuid = xc->get_uuid();

        // Load instructions
        std::ifstream nf(instr_path, std::ios::binary);
        if (!nf) { fprintf(stderr, "  [TernaryCtx] instr not found: %s\n", instr_path.c_str()); return false; }
        nf.seekg(0, std::ios::end); auto ns = (size_t)nf.tellg(); nf.seekg(0);
        instr.resize(ns / 4 + 1);
        nf.read((char*)instr.data(), ns);

        // Upload packed weights (flat, sliced per-tile in gemv)
        weights_bytes = (size_t)M_total * K_PACKED;
        scales_bytes = (size_t)M_total * 2;
        bo_weights = std::make_unique<xrt::bo>(dev, weights_bytes + scales_bytes,
                                                xrt::bo::flags::host_only, 0);
        memcpy(bo_weights->map(), packed_weights, weights_bytes);
        memcpy((char*)bo_weights->map() + weights_bytes, weight_scales, scales_bytes);
        bo_weights->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        fprintf(stderr, "  [TernaryCtx] M=%d K=%d weights=%zuB tile_in=%d OK\n",
                M, K, weights_bytes + scales_bytes, TILE_IN_BYTES);
        return true;
    }

    /**
     * Tiled GEMV: out[m] = sum_k(dot(weight[m][k*K_TERNARY:(k+1)*K_TERNARY], act[k_chunk]) * scale[m])
     *
     * Uses single-shot single-core xclbin (kernel exits after one input).
     * Tiles M into M_PER_CORE chunks, K into K_TERNARY chunks.
     * Fresh hw_context per dispatch avoids multi-dispatch crash.
     */
    void gemv(const uint16_t* activation_bf16, uint16_t* output_bf16) {
        memset(output_bf16, 0, (size_t)M_total * 2);

        const uint8_t*  wgt_map = (const uint8_t*)bo_weights->map();
        const uint16_t* sc_map  = (const uint16_t*)((const char*)bo_weights->map() + weights_bytes);

        for (int m_start = 0; m_start < M_total; m_start += M_PER_CORE) {
            int m_chunk = std::min(M_PER_CORE, M_total - m_start);

            for (int k_start = 0; k_start < K_total; k_start += K_TERNARY) {
                int k_act_count = std::min(K_TERNARY, K_total - k_start);
                int k_packed_chunk = k_act_count / 4;

                // Create per-dispatch hw_context + kernel (single-shot, exits cleanly)
                auto hc = std::make_unique<xrt::hw_context>(*device, xclbin_uuid);
                auto k  = std::make_unique<xrt::kernel>(*hc, "MLIR_AIE");

                int wg = k->group_id(3);
                int og = k->group_id(5);
                int ig = k->group_id(1);

                auto bo_in  = std::make_unique<xrt::bo>(*device, TILE_IN_BYTES, xrt::bo::flags::host_only, wg);
                auto bo_out = std::make_unique<xrt::bo>(*device, TILE_OUT_BYTES, xrt::bo::flags::host_only, og);

                // Build tile input: [m_chunk×K_packed weights | m_chunk×2 scales | k_act×2 activations]
                uint8_t* in_map = (uint8_t*)bo_in->map();

                // Copy weight slice
                uint8_t* wt_src = (uint8_t*)wgt_map + (size_t)m_start * K_PACKED;
                for (int r = 0; r < m_chunk; r++) {
                    memcpy(in_map + r * K_PACKED,
                           wt_src + r * ((size_t)K_total / 4) + k_start / 4,
                           (size_t)k_packed_chunk);
                }
                // Zero-pad if partial K chunk
                if (k_packed_chunk < K_PACKED) {
                    for (int r = 0; r < m_chunk; r++)
                        memset(in_map + r * K_PACKED + k_packed_chunk, 0,
                               (size_t)(K_PACKED - k_packed_chunk));
                }

                // Copy scale slice
                memcpy(in_map + M_PER_CORE * K_PACKED, sc_map + m_start, (size_t)m_chunk * 2);

                // Copy activation slice
                uint16_t* act_dst = (uint16_t*)(in_map + M_PER_CORE * K_PACKED + M_PER_CORE * 2);
                memcpy(act_dst, activation_bf16 + k_start, (size_t)k_act_count * 2);
                if (k_act_count < K_TERNARY)
                    memset(act_dst + k_act_count, 0, (K_TERNARY - k_act_count) * 2);

                bo_in->sync(XCL_BO_SYNC_BO_TO_DEVICE);

                // Instruction buffer
                auto bo_instr = xrt::bo(*device, instr.size() * 4, XCL_BO_FLAGS_CACHEABLE, ig);
                memcpy(bo_instr.map(), instr.data(), instr.size() * 4);
                bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

                // Dispatch single-shot kernel
                auto run = (*k)((unsigned)3, bo_instr, (unsigned)instr.size(),
                                *bo_in, *bo_out);
                run.wait();

                // Read back and accumulate
                bo_out->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                uint16_t* out_map = (uint16_t*)bo_out->map();
                for (int r = 0; r < m_chunk; r++) {
                    float partial = bf16f(out_map[r]);
                    float cur = bf16f(output_bf16[m_start + r]);
                    output_bf16[m_start + r] = f2bf(cur + partial);
                }
            }
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
        uint8_t* k_weights = nullptr;
        uint16_t* k_scales = nullptr;
        uint8_t* v_weights = nullptr;
        uint16_t* v_scales = nullptr;
        uint8_t* o_weights = nullptr;
        uint16_t* o_scales = nullptr;
        uint8_t* up_weights = nullptr;
        uint16_t* up_scales = nullptr;
        uint8_t* gate_weights = nullptr;
        uint16_t* gate_scales = nullptr;
        uint8_t* down_weights = nullptr;
        uint16_t* down_scales = nullptr;
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
            load_packed_weights(key, &lw.q_weights, &lw.q_scales);
            snprintf(key, 256, "model.layers.%d.self_attn.k_proj.weight", l);
            load_packed_weights(key, &lw.k_weights, &lw.k_scales);
            snprintf(key, 256, "model.layers.%d.self_attn.v_proj.weight", l);
            load_packed_weights(key, &lw.v_weights, &lw.v_scales);
            snprintf(key, 256, "model.layers.%d.self_attn.o_proj.weight", l);
            load_packed_weights(key, &lw.o_weights, &lw.o_scales);
            snprintf(key, 256, "model.layers.%d.mlp.up_proj.weight", l);
            load_packed_weights(key, &lw.up_weights, &lw.up_scales);
            snprintf(key, 256, "model.layers.%d.mlp.gate_proj.weight", l);
            load_packed_weights(key, &lw.gate_weights, &lw.gate_scales);
            snprintf(key, 256, "model.layers.%d.mlp.down_proj.weight", l);
            load_packed_weights(key, &lw.down_weights, &lw.down_scales);

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

    void load_packed_weights(const char* name, uint8_t** out_weights, uint16_t** out_scales) {
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
        fprintf(stderr, "  Q: M=%d K=%d  K: M=%d K=%d  V: M=%d K=%d\n", q_dim, H, kv_dim, H, kv_dim, H);
        fprintf(stderr, "  O: M=%d K=%d  Up: M=%d K=%d  Gate: M=%d K=%d  Down: M=%d K=%d\n",
                H, q_dim, IM, H, IM, H, H, IM);

        // For now: use one xclbin per dimension set
        // QKV dims: M=q_dim/kv_dim, K=H
        // O dims: M=H, K=q_dim
        // Up/Gate dims: M=IM, K=H (same xclbin)
        // Down dims: M=H, K=IM

        // Single oneshot xclbin for all projections — host tiles M/K.
        // Xclbin dir must contain: ternary_oneshot.xclbin + insts_ternary_oneshot.txt
        std::string xclbin_file = xb + "ternary_oneshot.xclbin";
        std::string instr_file  = xb + "insts_ternary_oneshot.txt";

        for (int l = 0; l < model.num_layers; l++) {
            auto& lw = model.layers[l];
            auto& lc = layer_ctxs[l];

            lc.q = std::make_unique<TernaryCtx>();
            if (!lc.q->init(device, xclbin_file, instr_file,
                            lw.q_weights, lw.q_scales, q_dim, H)) return false;

            lc.k = std::make_unique<TernaryCtx>();
            if (!lc.k->init(device, xclbin_file, instr_file,
                            lw.k_weights, lw.k_scales, kv_dim, H)) return false;

            lc.v = std::make_unique<TernaryCtx>();
            if (!lc.v->init(device, xclbin_file, instr_file,
                            lw.v_weights, lw.v_scales, kv_dim, H)) return false;

            lc.o = std::make_unique<TernaryCtx>();
            if (!lc.o->init(device, xclbin_file, instr_file,
                            lw.o_weights, lw.o_scales, H, q_dim)) return false;

            lc.up = std::make_unique<TernaryCtx>();
            if (!lc.up->init(device, xclbin_file, instr_file,
                             lw.up_weights, lw.up_scales, IM, H)) return false;

            lc.gate = std::make_unique<TernaryCtx>();
            if (!lc.gate->init(device, xclbin_file, instr_file,
                               lw.gate_weights, lw.gate_scales, IM, H)) return false;

            lc.down = std::make_unique<TernaryCtx>();
            if (!lc.down->init(device, xclbin_file, instr_file,
                               lw.down_weights, lw.down_scales, H, IM)) return false;
        }

        fprintf(stderr, "[Daemon] All %d layers loaded. Ready.\n", model.num_layers);
        return true;
    }

    void layer_forward(int l, float* hidden, std::vector<uint16_t>& buf_bf16,
                       int pos, RoPECache& rope) {
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

        // ── K projection ──────────────────────────────────
        std::vector<uint16_t> k_out(kv_dim);
        lc.k->gemv(buf_bf16.data(), k_out.data());

        // ── V projection ──────────────────────────────────
        std::vector<uint16_t> v_out(kv_dim);
        lc.v->gemv(buf_bf16.data(), v_out.data());

        // Convert to f32 for attention
        std::vector<float> q_f32(q_dim), k_f32(kv_dim), v_f32(kv_dim);
        for (int i = 0; i < q_dim; i++) q_f32[i] = bf16f(q_out[i]);
        for (int i = 0; i < kv_dim; i++) k_f32[i] = bf16f(k_out[i]);
        for (int i = 0; i < kv_dim; i++) v_f32[i] = bf16f(v_out[i]);

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
        lc.gate->gemv(buf_bf16.data(), gate_out.data());

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

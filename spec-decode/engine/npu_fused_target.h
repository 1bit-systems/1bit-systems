#pragma once
/** NPU Fused Target — speculative-decode target model using the fused-layer xclbin.
 *
 * Uses the numerics-enabled full-layer xclbin (qwen3-decode-layer-capacity-token127,
 * one call per layer: QKV→attn→O→GU→SiLU→D all on NPU) with 28 reference-format
 * wref_l*.bin weight files (aux@pos0 + packed Q4NX), per-position design-token127-to-token*.bin
 * instruction streams, a per-position rope_table.bin (cos/sin patched per token), and
 * per-layer 256 KB persistent KV caches. NOTE: the earlier weight-stream xclbin
 * (md5 515b2af1) was numerics-DISABLED (compute replaced by consume/drop) — fast but
 * incoherent; this driver targets the compute-real build (md5 2f0f0858).
 *
 * Conforms to TargetModelInterface for use with spec_decode.h's SpeculativeDecoder.
 *
 * Build: link with -lxrt_coreutil -luuid -lm -ldl
 */

#include "spec_decode.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <omp.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

extern "C" float* dequant_i8_to_float(const uint8_t*, int, int*, int*);
extern "C" float* dequant_i8_to_float_ex(const uint8_t*, int, int, int*, int*);

namespace npu_fused_detail {

// BF16 helpers
static inline uint16_t f2bf(float v) {
    uint32_t b; memcpy(&b, &v, 4);
    return (uint16_t)((b + 0x8000) >> 16);
}
static inline float bf16f(uint16_t v) {
    uint32_t b = (uint32_t)v << 16; float f;
    memcpy(&f, &b, 4); return f;
}

// RMSNorm on float32 data
static inline void rms_norm(float* x, const float* w, int n) {
    double ss = 0;
    for (int i = 0; i < n; i++) if (std::isfinite(x[i])) ss += (double)x[i] * x[i];
    float ir = 1.0f / sqrtf((float)(ss / n) + 1e-6f);
    for (int i = 0; i < n; i++) x[i] = std::isfinite(x[i]) ? x[i] * ir * w[i] : 0.0f;
}

// Clamp infinities to zero
static inline void clamp_finite(float* x, int n) {
    for (int i = 0; i < n; i++) if (!std::isfinite(x[i])) x[i] = 0.0f;
}

// Simple argmax
static inline int argmax_f32(const float* logits, int n) {
    return (int)std::distance(logits, std::max_element(logits, logits + n));
}

// Load a binary file into a vector
static std::vector<char> load_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    size_t sz = f.tellg(); f.seekg(0);
    std::vector<char> data(sz); f.read(data.data(), sz);
    return data;
}

// Parse Q4NX model header metadata (JSON offset, embed table offset, layer weight offsets)
struct Q4NXModel {
    uint8_t* base = nullptr;
    size_t size = 0;
    uint64_t data_off = 0;
    const uint16_t* embed_table = nullptr;  // [vocab_size, hidden] bf16
    struct LayerMeta {
        uint64_t qp, kp, vp, op, gp, up, dp;
        uint64_t in_off, pa_off, qn_off, kn_off;
    };
    std::vector<LayerMeta> layers;
    const float* lm_head_f32 = nullptr;
    int lm_head_rows = 0, lm_head_cols = 0;
    std::vector<float> in_norm_w, pa_norm_w, q_norm_w, k_norm_w, final_norm_w;
    int vocab_size = 151936;
    int hidden = 1024;
    int num_layers = 28;
    int num_heads = 16;
    int num_kv_heads = 8;
    int head_dim = 128;

    bool load(const char* path) {
        int fd = ::open(path, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "Cannot open %s\n", path); return false; }
        struct stat st; fstat(fd, &st);
        size = st.st_size;
        base = (uint8_t*)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        
        uint64_t hsz; memcpy(&hsz, base, 8);
        const char* js = (const char*)(base + 8);
        size_t jl = hsz;
        data_off = 8 + hsz;
        embed_table = (const uint16_t*)(base + data_off);
        
        auto json_off = [&](const char* name) -> uint64_t {
            size_t nl = strlen(name);
            const char* p = js, *e = js + jl;
            while (p < e) {
                auto q = (const char*)memmem(p, e - p, name, nl);
                if (!q) return 0;
                if (q > js && *(q-1) == '"' && *(q + nl) == '"') {
                    auto o = strstr(q, "\"data_offsets\"");
                    if (o) { auto a = strchr(o, '['); if (a) return strtoull(a + 1, NULL, 10); }
                }
                p = q + 1;
            }
            return 0;
        };
        
        layers.resize(num_layers);
        char buf[128];
        for (int l = 0; l < num_layers; l++) {
            snprintf(buf, 128, "model.layers.%d.self_attn.q_proj.weight", l);
            layers[l].qp = json_off(buf);
            snprintf(buf, 128, "model.layers.%d.self_attn.k_proj.weight", l);
            layers[l].kp = json_off(buf);
            snprintf(buf, 128, "model.layers.%d.self_attn.v_proj.weight", l);
            layers[l].vp = json_off(buf);
            snprintf(buf, 128, "model.layers.%d.self_attn.o_proj.weight", l);
            layers[l].op = json_off(buf);
            snprintf(buf, 128, "model.layers.%d.mlp.gate_proj.weight", l);
            layers[l].gp = json_off(buf);
            snprintf(buf, 128, "model.layers.%d.mlp.up_proj.weight", l);
            layers[l].up = json_off(buf);
            snprintf(buf, 128, "model.layers.%d.mlp.down_proj.weight", l);
            layers[l].dp = json_off(buf);
            snprintf(buf, 128, "model.layers.%d.input_layernorm.weight", l);
            layers[l].in_off = json_off(buf);
            snprintf(buf, 128, "model.layers.%d.post_attention_layernorm.weight", l);
            layers[l].pa_off = json_off(buf);
            snprintf(buf, 128, "model.layers.%d.self_attn.q_norm.weight", l);
            layers[l].qn_off = json_off(buf);
            snprintf(buf, 128, "model.layers.%d.self_attn.k_norm.weight", l);
            layers[l].kn_off = json_off(buf);
        }
        
        // Load norm weights
        auto load_bf16_vec = [&](uint64_t off, int n) -> std::vector<float> {
            std::vector<float> v(n);
            auto* src = (const uint16_t*)(base + data_off + off);
            for (int i = 0; i < n; i++) v[i] = bf16f(src[i]);
            return v;
        };
        
        in_norm_w.resize(hidden);
        pa_norm_w.resize(hidden);
        q_norm_w.resize(head_dim);
        k_norm_w.resize(head_dim);
        
        // These are per-layer, we need the layer 0 versions for the non-fused parts
        // (the fused xclbin handles norms internally for its own layers)
        
        uint64_t no = json_off("model.norm.weight");
        if (no) final_norm_w = load_bf16_vec(no, hidden);
        
        // Load lm_head (dequantized I8 weight) — same as original engine
        uint64_t lo = json_off("lm_head.weight");
        {
            int lr = 0, lc = 0;
            float* raw = dequant_i8_to_float(base + data_off + lo, 18992, &lr, &lc);
            if (raw && lr > 0 && lc > 0) {
                lm_head_f32 = new float[(size_t)lr * lc];
                memcpy(const_cast<float*>(lm_head_f32), raw, (size_t)lr * lc * sizeof(float));
                lm_head_rows = lr;
                lm_head_cols = lc;
                vocab_size = lr;
                free(raw);
                printf("  lm_head: %dx%d (loaded via dequant, rows=%d cols=%d)\n", lr, lc, lm_head_rows, lm_head_cols);
                printf("  lm_head[0][0..3]: %.4f %.4f %.4f %.4f\n",
                       lm_head_f32[0], lm_head_f32[1], lm_head_f32[2], lm_head_f32[3]);
            }
        }
        
        return true;
    }
    
    float* embed_lookup(int token_id) const {
        // embed_table is [vocab_size, hidden] bf16
        // Returns a float32 buffer (caller manages lifetime)
        auto* result = new float[hidden];
        const uint16_t* row = embed_table + (size_t)token_id * hidden;
        for (int i = 0; i < hidden; i++) result[i] = bf16f(row[i]);
        return result;
    }
    
    void embed_lookup_bf16(int token_id, uint16_t* out_bf16) const {
        const uint16_t* row = embed_table + (size_t)token_id * hidden;
        memcpy(out_bf16, row, hidden * 2);
    }
};

} // namespace npu_fused_detail

class NpuFusedTarget : public TargetModelInterface {
public:
    static constexpr int H = 1024;
    static constexpr int NC = 28;
    static constexpr int NV = 151936;
    static constexpr int MAX_SEQ = 4096;
    static constexpr int MAX_INSTR_POS = 128;
    static constexpr int B = 2048;  // bytes per bf16 hidden block: 1024 * 2
    // Numerics-enabled capacity-token127 build KV cache: kv_blocks(8) * CACHE_BLOCK_DWORDS(8192)
    // = 65536 dwords per layer, per K and per V. Persistent across tokens, zeroed at reset.
    static constexpr size_t KV_DWORDS = 65536;
    static constexpr size_t KV_BYTES  = KV_DWORDS * 4;   // 262144
    // Position-dependent cos/sin slot inside each layer's aux-prefixed weight buffer.
    // aux dword layout: input_norm(512) post_norm(512) q_norm(64) k_norm(64) cos(32) sin(32).
    static constexpr size_t ROPE_COSSIN_DWORD_OFFSET = 1152;
    static constexpr size_t ROPE_COSSIN_DWORDS = 64;

    NpuFusedTarget(const char* model_path,
                   const char* xclbin_dir,
                   const char* weights_dir,
                   const int32_t* target_layer_ids,
                   int32_t num_target_layers)
        : target_layer_ids_(target_layer_ids, target_layer_ids + num_target_layers)
    {
        using namespace npu_fused_detail;
        
        printf("[NpuFusedTarget] Loading model: %s\n", model_path);
        if (!model_.load(model_path)) {
            fprintf(stderr, "Failed to load model\n");
            return;
        }
        
        printf("[NpuFusedTarget] Loading fused xclbin...\n");
        std::string xclbin_path = std::string(xclbin_dir) + "/design.xclbin";
        auto xclbin_data = load_binary(xclbin_path);
        if (xclbin_data.empty()) {
            fprintf(stderr, "Failed to load xclbin: %s\n", xclbin_path.c_str());
            return;
        }
        
        dev_ = std::make_unique<xrt::device>(0);
        xrt::xclbin xclbin(xclbin_data);
        dev_->register_xclbin(xclbin);
        // Explicit hw_context (matches the proven v12 GEMM path and the gate test) — the
        // implicit device-context kernel stalls on the 2nd consecutive real-compute run.
        ctx_ = std::make_unique<xrt::hw_context>(*dev_, xclbin.get_uuid());
        kernel_ = std::make_unique<xrt::kernel>(*ctx_, "MLIR_AIE");
        
        // Per-arg memory banks (kernel args after the opcode/instr/count prefix):
        //   arg3=k_cache arg4=v_cache arg5=weights arg6=output arg7=hidden ; instr=arg1.
        int ig = kernel_->group_id(1);
        int kg = kernel_->group_id(3);
        int vg = kernel_->group_id(4);
        int wg = kernel_->group_id(5);
        int og = kernel_->group_id(6);
        int hg = kernel_->group_id(7);
        if (getenv("FUSED_DBG"))
            fprintf(stderr, "[dbg] group_id 1=%d 3=%d 4=%d 5=%d 6=%d 7=%d\n",
                    ig, kg, vg, wg, og, hg);

        // Load instruction files (per-position). Numerics-enabled capacity build ships
        // design-token127-to-token{pos}.bin for pos 0..126, all 1723 words.
        printf("[NpuFusedTarget] Loading instruction files...\n");
        auto generic_data = load_binary(std::string(xclbin_dir) + "/design.bin");
        generic_bo_ = std::make_unique<xrt::bo>(*dev_, generic_data.size(), xrt::bo::flags::cacheable, ig);
        memcpy(generic_bo_->map(), generic_data.data(), generic_data.size());
        generic_bo_->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        instr_bos_.resize(MAX_INSTR_POS);
        int instr_loaded = 0;
        for (int pos = 0; pos < MAX_INSTR_POS; pos++) {
            char fname[256];
            snprintf(fname, 256, "%s/design-token127-to-token%d.bin", xclbin_dir, pos);
            auto data = load_binary(fname);
            if (!data.empty()) {
                auto bo = std::make_unique<xrt::bo>(*dev_, data.size(), xrt::bo::flags::cacheable, ig);
                memcpy(bo->map(), data.data(), data.size());
                bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
                instr_bos_[pos] = std::move(bo);
                instr_loaded++;
            }
        }
        printf("[NpuFusedTarget] Loaded %d per-position instruction streams\n", instr_loaded);

        // Load reference-format per-layer weights (aux@pos0 + packed Q4NX), 9.835 MB each.
        printf("[NpuFusedTarget] Loading reference-format weight files...\n");
        weight_bos_.resize(NC);
        for (int l = 0; l < NC; l++) {
            char fname[256];
            snprintf(fname, 256, "%s/wref_l%d.bin", weights_dir, l);
            auto data = load_binary(fname);
            if (data.empty()) {
                fprintf(stderr, "Missing weights for layer %d (%s)\n", l, fname);
                return;
            }
            auto bo = std::make_unique<xrt::bo>(*dev_, data.size(), xrt::bo::flags::host_only, wg);
            memcpy(bo->map(), data.data(), data.size());
            bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            weight_bos_[l] = std::move(bo);
        }

        // Load the per-position cos/sin RoPE table (int32[MAX_INSTR_POS][ROPE_COSSIN_DWORDS]).
        {
            auto rt = load_binary(std::string(weights_dir) + "/rope_table.bin");
            if (rt.empty()) { fprintf(stderr, "Missing rope_table.bin\n"); return; }
            rope_table_.resize(rt.size() / 4);
            memcpy(rope_table_.data(), rt.data(), rt.size());
            printf("[NpuFusedTarget] rope_table: %zu positions\n", rope_table_.size() / ROPE_COSSIN_DWORDS);
        }

        // Per-layer persistent KV caches (256 KB each, one K + one V per layer).
        printf("[NpuFusedTarget] Creating per-layer KV caches (%zu KB each) + data BOs...\n", KV_BYTES / 1024);
        kCache_.resize(NC);
        vCache_.resize(NC);
        for (int l = 0; l < NC; l++) {
            kCache_[l] = std::make_unique<xrt::bo>(*dev_, KV_BYTES, xrt::bo::flags::host_only, kg);
            vCache_[l] = std::make_unique<xrt::bo>(*dev_, KV_BYTES, xrt::bo::flags::host_only, vg);
        }
        bHidden_ = std::make_unique<xrt::bo>(*dev_, B, xrt::bo::flags::host_only, hg);
        bOutput_ = std::make_unique<xrt::bo>(*dev_, B, xrt::bo::flags::host_only, og);

        clear_kv_cache();

        layer_hidden_snapshots_.resize(NC, std::vector<float>(H));
        
        printf("[NpuFusedTarget] Ready. %d layers, %d target layers for draft features\n",
               NC, num_target_layers);
    }

    ~NpuFusedTarget() override = default;

    // --- TargetModelInterface implementation ---

    void forward(const int32_t* input_ids, int32_t seq_len,
                 float* logits, float* hidden_states) override {
        // Prefill: run all positions from scratch, reset KV cache
        batch_forward(input_ids, seq_len, /*start_pos=*/0, /*kv_reset=*/true,
                      logits, hidden_states, /*logits_all=*/false);
    }

    void forward_with_kv(const int32_t* input_ids, int32_t n_tokens, int32_t past_len,
                          float* logits, float* hidden_states) override {
        // Verify: run new tokens using existing KV cache
        batch_forward(input_ids, n_tokens, past_len, /*kv_reset=*/false,
                      logits, hidden_states, /*logits_all=*/true);
    }

    void get_layer_hidden(const float* /*all_hidden*/, int32_t /*num_layers*/,
                           const int32_t* target_ids, int32_t num_target,
                           float* out) override {
        for (int i = 0; i < num_target; i++) {
            int layer = target_ids[i];
            memcpy(out + (size_t)i * H, layer_hidden_snapshots_[layer].data(), H * 4);
        }
    }

    void commit_accepted(int32_t start_pos, int32_t n_accept) override {
        // KV cache is already in the correct state — the fused engine writes
        // positions sequentially and we control past_len externally.
        // This is a no-op for fused target since we always write to position
        // past_len + i and never past that.
        (void)start_pos; (void)n_accept;
    }

private:
    // Get instruction BO for a position (falls back to generic)
    xrt::bo& get_instr_bo(int pos) {
        if (pos >= 0 && pos < MAX_INSTR_POS && instr_bos_[pos]) return *instr_bos_[pos];
        return *generic_bo_;
    }
    
    void clear_kv_cache() {
        for (int l = 0; l < NC; l++) {
            memset(kCache_[l]->map(), 0, KV_BYTES);
            memset(vCache_[l]->map(), 0, KV_BYTES);
            kCache_[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            vCache_[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
    }

    // Overwrite the position-dependent cos/sin slot (layer-independent, 256B) in every
    // layer's weight buffer, syncing only that sub-range to device.
    void patch_rope(int pos) {
        size_t npos = rope_table_.size() / ROPE_COSSIN_DWORDS;
        if (npos == 0) return;
        if (pos < 0) pos = 0;
        if ((size_t)pos >= npos) pos = (int)npos - 1;   // clamp to table (context cap)
        const int32_t* cs = rope_table_.data() + (size_t)pos * ROPE_COSSIN_DWORDS;
        for (int l = 0; l < NC; l++) {
            int32_t* wmap = (int32_t*)weight_bos_[l]->map();
            memcpy(wmap + ROPE_COSSIN_DWORD_OFFSET, cs, ROPE_COSSIN_DWORDS * 4);
            weight_bos_[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE,
                                 ROPE_COSSIN_DWORDS * 4, ROPE_COSSIN_DWORD_OFFSET * 4);
        }
    }
    
    // Core forward logic shared by prefill and verify
    void batch_forward(const int32_t* tokens, int n, int32_t start_pos,
                        bool reset_kv,
                        float* out_logits, float* out_hidden,
                        bool logits_all_positions) {
        using namespace npu_fused_detail;
        
        if (reset_kv) clear_kv_cache();
        
        // Per-position final hidden state buffers (for logits_all_positions)
        std::vector<std::vector<float>> per_pos_hidden;
        if (logits_all_positions && out_logits) {
            per_pos_hidden.resize(n, std::vector<float>(H));
        }
        
        // For each position, run all 28 layers.
        for (int pi = 0; pi < n; pi++) {
            int pos = start_pos + pi;

            // Patch position-dependent RoPE cos/sin (layer-independent) into every
            // layer's aux-prefixed weight buffer, sub-buffer sync only the 256B slot.
            patch_rope(pos);

            // Embed lookup: copy bf16 embedding into the hidden BO.
            uint16_t* hdata = (uint16_t*)bHidden_->map();
            model_.embed_lookup_bf16(tokens[pi], hdata);
            bHidden_->sync(XCL_BO_SYNC_BO_TO_DEVICE);

            for (int l = 0; l < NC; l++) {
                auto& ibo = get_instr_bo(pos);
                unsigned opcode = getenv("FUSED_OPCODE") ? atoi(getenv("FUSED_OPCODE")) : 3;
                auto run = (*kernel_)(opcode, ibo, (uint32_t)1723,
                                     *kCache_[l], *vCache_[l], *weight_bos_[l],
                                     *bOutput_, *bHidden_);
                run.wait();
                bOutput_->sync(XCL_BO_SYNC_BO_FROM_DEVICE);

                if (getenv("FUSED_DBG") && pi == 0) {
                    uint16_t* od = (uint16_t*)bOutput_->map();
                    fprintf(stderr, "[dbg] pos0 l=%d done out[%.4f %.4f %.4f]\n",
                            l, bf16f(od[0]), bf16f(od[1]), bf16f(od[2]));
                }

                // Snapshot hidden at target layers for draft features.
                for (int ti = 0; ti < (int)target_layer_ids_.size(); ti++) {
                    if (target_layer_ids_[ti] == l) {
                        uint16_t* odata = (uint16_t*)bOutput_->map();
                        float* snap = layer_hidden_snapshots_[l].data();
                        for (int i = 0; i < H; i++) snap[i] = bf16f(odata[i]);
                        break;
                    }
                }

                // Layer output becomes next layer's input.
                memcpy(bHidden_->map(), bOutput_->map(), B);
                bHidden_->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            }

            // Capture this position's final hidden state (for per-position logits).
            if (logits_all_positions && out_logits) {
                uint16_t* od = (uint16_t*)bOutput_->map();
                for (int i = 0; i < H; i++) per_pos_hidden[pi][i] = bf16f(od[i]);
            }
        }
        
        // Copy last position's final hidden state (for out_hidden)
        if (out_hidden) {
            // If we captured per-position, use last entry; otherwise read from BO
            if (!per_pos_hidden.empty()) {
                memcpy(out_hidden, per_pos_hidden.back().data(), H * 4);
            } else {
                uint16_t* final_hdata = (uint16_t*)bOutput_->map();
                for (int i = 0; i < H; i++) out_hidden[i] = bf16f(final_hdata[i]);
            }
        }
        
        // Compute logits: final RMSNorm + lm_head matmul on CPU
        if (out_logits) {
            int num_logit_positions = logits_all_positions ? n : 1;
            int last_pi = n - 1;
            
            for (int pi = 0; pi < num_logit_positions; pi++) {
                // Use the captured position if available, otherwise the last position
                int src_idx = logits_all_positions ? pi : last_pi;
                float* hsrc = per_pos_hidden.empty()
                    ? (src_idx == last_pi ? nullptr : nullptr)  // shouldn't happen
                    : per_pos_hidden[src_idx].data();
                
                // Fallback: read from BO if no per-position cache (prefill path)
                std::vector<float> fallback_hidden;
                if (!hsrc) {
                    fallback_hidden.resize(H);
                    uint16_t* fh = (uint16_t*)bOutput_->map();
                    for (int i = 0; i < H; i++) fallback_hidden[i] = bf16f(fh[i]);
                    hsrc = fallback_hidden.data();
                }
                
                // Compute logits for this position
                rms_norm(hsrc, model_.final_norm_w.data(), H);
                clamp_finite(hsrc, H);
                
                float* lg = out_logits + (size_t)pi * NV;
                #pragma omp parallel for
                for (int v = 0; v < NV; v++) {
                    double s = 0;
                    const float* wrow = model_.lm_head_f32 + (size_t)v * H;
                    for (int k = 0; k < H; k++) s += (double)hsrc[k] * wrow[k];
                    lg[v] = (float)s;
                }
            }
        }
    }

    npu_fused_detail::Q4NXModel model_;
    std::unique_ptr<xrt::device> dev_;
    std::unique_ptr<xrt::hw_context> ctx_;
    std::unique_ptr<xrt::kernel> kernel_;
    std::unique_ptr<xrt::bo> generic_bo_;
    std::vector<std::unique_ptr<xrt::bo>> instr_bos_;
    std::vector<std::unique_ptr<xrt::bo>> weight_bos_;
    // Per-layer KV caches (one pair per layer, persistent across positions).
    std::vector<std::unique_ptr<xrt::bo>> kCache_;
    std::vector<std::unique_ptr<xrt::bo>> vCache_;
    std::unique_ptr<xrt::bo> bHidden_, bOutput_;
    // Per-position cos/sin RoPE table (loaded once, shared across layers).
    std::vector<int32_t> rope_table_;
    std::vector<std::vector<float>> layer_hidden_snapshots_;
    std::vector<int32_t> target_layer_ids_;
};

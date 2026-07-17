/**
 * safetensors_to_raw.cpp
 *
 * Reads Qwen3-0.6B model.safetensors (BF16) and exports as raw float32 files
 * in the exact layout the ROCm GPU engine expects:
 *
 *   B[k * N + col]  where B has shape (K, N) = (in_features, out_features)
 *
 * This transposes from HF's (out_features, in_features) row-major storage.
 *
 * Usage: ./safetensors_to_raw <model.safetensors> <output_dir>
 *
 * Output directory will contain:
 *   embed.F32      : [NV * H] float32, identity (same as lm_head, tied)
 *   lm_head.F32    : [NV * H] float32, row-major (NV × H), NO TRANSPOSE — see note below
 *   final_norm.F32 : [H] float32
 *   layer_0.qkv.F32  : [H * 4096] float32, B[k * 4096 + col]
 *   layer_0.gu.F32   : [H * 6144] float32, B[k * 6144 + col]
 *   layer_0.o.F32    : [2048 * 1024] float32, B[k * 1024 + col]
 *   layer_0.down.F32 : [3072 * 1024] float32, B[k * 1024 + col]
 *   layer_0.in_norm.F32  : [H] float32
 *   layer_0.pa_norm.F32  : [H] float32
 *   layer_0.q_norm.F32   : [128] float32
 *   layer_0.k_norm.F32   : [128] float32
 *   ... (same for layers 1..27)
 *
 * Note on lm_head: The GPU engine stores lm_head as [H × NV] = B[k * NV + col].
 * gemv_kernel reads B[k][col] for k=0..H-1, col=0..NV-1.
 * This means B[k][col] = lm_head[col][k] (HF's row-major lm_head is [NV][H]).
 * So we transpose lm_head from [NV, H] to [H, NV].
 *
 * But the safetensors stores lm_head before embed (same data, tied weights).
 * We still output lm_head.F32 as [NV * H] row-major (same as embed) and let
 * the loader do the transpose, OR we do the transpose here.
 *
 * Actually: The GPU engine expects lm_head_ as [H * NV] where
 * lm_head_[k * NV + n] = weight connecting hidden[k] to logits[n].
 * HF stores lm_head as [NV, H] row-major: lm_hf[n * H + k].
 * So lm_head_[k * NV + n] = lm_hf[n * H + k] → this is a full transpose.
 *
 * For simplicity: output lm_head as [H * NV] already transposed.
 * And embed (same data) also as [H * NV] — but the engine's embed_table_ is
 * loaded as [NV * H] row-major for direct indexing embed_table_[token * H + dim].
 * So embed must stay as [NV * H], NOT transposed.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

// nlohmann/json
#include <nlohmann/json.hpp>
using json = nlohmann::json;

// Constants matching Qwen3-0.6B config
static const int H = 1024;       // hidden_size
static const int NH = 16;        // num_attention_heads
static const int NKV = 8;        // num_key_value_heads
static const int HD = 128;       // head_dim
static const int IM = 3072;      // intermediate_size
static const int NV = 151936;    // vocab_size
static const int NC = 28;        // num_hidden_layers
static const int GQA = NH / NKV; // 2

// BF16 → float32 conversion
inline float bf16_to_f32(uint16_t v) {
    union { uint32_t u; float f; } conv;
    conv.u = (uint32_t)v << 16;
    return conv.f;
}

// Read entire file into memory
static std::vector<uint8_t> read_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return {}; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(sz);
    if ((long)fread(buf.data(), 1, sz, f) != sz) {
        fprintf(stderr, "Short read from %s\n", path); fclose(f); return {};
    }
    fclose(f);
    return buf;
}

// Write raw float32 array to file
static bool write_f32(const char* path, const float* data, size_t n) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return false; }
    size_t written = fwrite(data, 4, n, f);
    fclose(f);
    if (written != n) {
        fprintf(stderr, "Short write to %s (%zu/%zu)\n", path, written, n);
        return false;
    }
    printf("  Wrote %s (%zu elements, %.2f MB)\n", path, n, n * 4.0 / 1e6);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.safetensors> <output_dir>\n", argv[0]);
        return 1;
    }

    const char* safetensors_path = argv[1];
    const char* out_dir = argv[2];

    // Create output directory
    mkdir(out_dir, 0755);

    // Read safetensors file
    auto buf = read_file(safetensors_path);
    if (buf.empty()) return 1;

    // Parse header (8 bytes length prefix + JSON)
    uint64_t hdr_len = 0;
    if (buf.size() < 8) { fprintf(stderr, "File too small\n"); return 1; }
    memcpy(&hdr_len, buf.data(), 8);
    if (8 + hdr_len > buf.size()) {
        fprintf(stderr, "Header length %llu exceeds file size %zu\n",
                (unsigned long long)hdr_len, buf.size());
        return 1;
    }

    std::string hdr_str((const char*)buf.data() + 8, hdr_len);
    json hdr = json::parse(hdr_str);

    // Data starts after header
    const uint8_t* tensors = buf.data() + 8 + hdr_len;

    // Helper: get pointer to BF16 data for a tensor name
    auto get_tensor = [&](const std::string& name) -> const uint16_t* {
        if (!hdr.contains(name)) {
            fprintf(stderr, "Tensor '%s' not found in safetensors header\n", name.c_str());
            return nullptr;
        }
        auto& info = hdr[name];
        auto& off = info["data_offsets"];
        size_t start = off[0].get<size_t>();
        return (const uint16_t*)(tensors + start);
    };

    // Helper: get tensor shape
    auto get_shape = [&](const std::string& name) -> std::vector<int> {
        std::vector<int> shape;
        if (!hdr.contains(name)) return shape;
        for (auto& s : hdr[name]["shape"])
            shape.push_back(s.get<int>());
        return shape;
    };

    // ============================================================
    // 1. Embedding table (tied with lm_head)
    //    HF shape: [NV, H] row-major in BF16
    //    Engine expects: embed_table_[token * H + dim], so [NV * H] as-is
    // ============================================================
    printf("Exporting embed_table...\n");
    // Check if embed_tokens exists (usually first in file)
    std::vector<float> embed(NV * H, 0.0f);
    const uint16_t* embed_src = get_tensor("model.embed_tokens.weight");
    if (!embed_src) {
        // Fall back to lm_head since weights are tied
        embed_src = get_tensor("lm_head.weight");
    }
    if (embed_src) {
        for (int n = 0; n < NV; n++)
            for (int d = 0; d < H; d++)
                embed[(size_t)n * H + d] = bf16_to_f32(embed_src[(size_t)n * H + d]);
        write_f32((std::string(out_dir) + "/embed.F32").c_str(), embed.data(), (size_t)NV * H);
    } else {
        fprintf(stderr, "ERROR: Could not find embed_tokens or lm_head\n");
    }

    // ============================================================
    // 2. LM head (tied weights, same data as embed)
    //    HF shape: [NV, H] row-major (lm_head[token][dim])
    //    Engine expects: lm_head_[k * NV + n] = weight[k][n] with (K, N) = (H, NV)
    //    So: lm_head_[k * NV + n] = lm_hf[n * H + k] → transpose
    // ============================================================
    printf("Exporting lm_head...\n");
    {
        std::vector<float> lm_head((size_t)H * NV, 0.0f);
        const uint16_t* lm_src = get_tensor("lm_head.weight");
        if (lm_src) {
            for (int k = 0; k < H; k++)
                for (int n = 0; n < NV; n++)
                    lm_head[(size_t)k * NV + n] = bf16_to_f32(lm_src[(size_t)n * H + k]);
            write_f32((std::string(out_dir) + "/lm_head.F32").c_str(), lm_head.data(), (size_t)H * NV);
        }
    }

    // ============================================================
    // 3. Final norm
    // ============================================================
    printf("Exporting final_norm...\n");
    {
        const uint16_t* fn_src = get_tensor("model.norm.weight");
        if (fn_src) {
            std::vector<float> fn(H);
            for (int i = 0; i < H; i++) fn[i] = bf16_to_f32(fn_src[i]);
            write_f32((std::string(out_dir) + "/final_norm.F32").c_str(), fn.data(), H);
        }
    }

    // ============================================================
    // 4. Per-layer weights
    // ============================================================
    for (int l = 0; l < NC; l++) {
        printf("Exporting layer %d...\n", l);
        char prefix[64];
        snprintf(prefix, 64, "%s/layer_%d", out_dir, l);
        mkdir(prefix, 0755);

        char fpath[256];

        // ---- Fused QKV ----
        // HF: q_proj [NH*HD, H] = [2048, 1024], k_proj [NKV*HD, H] = [1024, 1024], v_proj [1024, 1024]
        // Engine: qkv[k * 4096 + i] where:
        //   i=0..2047: q_proj[i][k]    (transpose from HF's [2048][1024])
        //   i=2048..3071: k_proj[i-2048][k]
        //   i=3072..4095: v_proj[i-3072][k]
        {
            char tensor_name[128];
            snprintf(tensor_name, 128, "model.layers.%d.self_attn.q_proj.weight", l);
            const uint16_t* qp = get_tensor(tensor_name);
            snprintf(tensor_name, 128, "model.layers.%d.self_attn.k_proj.weight", l);
            const uint16_t* kp = get_tensor(tensor_name);
            snprintf(tensor_name, 128, "model.layers.%d.self_attn.v_proj.weight", l);
            const uint16_t* vp = get_tensor(tensor_name);

            if (qp && kp && vp) {
                const int tqkv = 4096; // NH*HD + 2*NKV*HD = 2048 + 1024 + 1024
                std::vector<float> qkv((size_t)H * tqkv);

                for (int k = 0; k < H; k++) {
                    // Q: i=0..2047, q_proj[i][k] = qp[i * H + k]
                    for (int i = 0; i < NH*HD; i++)
                        qkv[(size_t)k * tqkv + i] = bf16_to_f32(qp[(size_t)i * H + k]);
                    // K: i=2048..3071, k_proj[i-2048][k] = kp[(i-2048) * H + k]
                    for (int i = 0; i < NKV*HD; i++)
                        qkv[(size_t)k * tqkv + NH*HD + i] = bf16_to_f32(kp[(size_t)i * H + k]);
                    // V: i=3072..4095, v_proj[i-3072][k] = vp[(i-3072) * H + k]
                    for (int i = 0; i < NKV*HD; i++)
                        qkv[(size_t)k * tqkv + NH*HD + NKV*HD + i] = bf16_to_f32(vp[(size_t)i * H + k]);
                }

                snprintf(fpath, 256, "%s/qkv.F32", prefix);
                write_f32(fpath, qkv.data(), (size_t)H * tqkv);
            }
        }

        // ---- Fused GU (gate + up) ----
        // HF: gate_proj [IM, H] = [3072, 1024], up_proj [IM, H] = [3072, 1024]
        // Engine: gu[k * 6144 + i] where:
        //   i=0..3071: gate_proj[i][k]
        //   i=3072..6143: up_proj[i-3072][k]
        {
            char tensor_name[128];
            snprintf(tensor_name, 128, "model.layers.%d.mlp.gate_proj.weight", l);
            const uint16_t* gp = get_tensor(tensor_name);
            snprintf(tensor_name, 128, "model.layers.%d.mlp.up_proj.weight", l);
            const uint16_t* up = get_tensor(tensor_name);

            if (gp && up) {
                const int tgu = 2 * IM; // 6144
                std::vector<float> gu((size_t)H * tgu);

                for (int k = 0; k < H; k++) {
                    for (int i = 0; i < IM; i++)
                        gu[(size_t)k * tgu + i] = bf16_to_f32(gp[(size_t)i * H + k]);
                    for (int i = 0; i < IM; i++)
                        gu[(size_t)k * tgu + IM + i] = bf16_to_f32(up[(size_t)i * H + k]);
                }

                snprintf(fpath, 256, "%s/gu.F32", prefix);
                write_f32(fpath, gu.data(), (size_t)H * tgu);
            }
        }

        // ---- O projection ----
        // HF: o_proj [H, NH*HD] = [1024, 2048]
        // Engine: o_proj has shape (K=NH*HD=2048, N=H=1024)
        // o_proj[k * 1024 + col] with k=0..2047, col=0..1023
        // o_proj[k][col] = hf[col][k] (transpose)
        {
            char tensor_name[128];
            snprintf(tensor_name, 128, "model.layers.%d.self_attn.o_proj.weight", l);
            const uint16_t* op = get_tensor(tensor_name);
            if (op) {
                std::vector<float> o((size_t)(NH*HD) * H);
                for (int k = 0; k < NH*HD; k++)
                    for (int col = 0; col < H; col++)
                        o[(size_t)k * H + col] = bf16_to_f32(op[(size_t)col * (NH*HD) + k]);
                snprintf(fpath, 256, "%s/o.F32", prefix);
                write_f32(fpath, o.data(), (size_t)(NH*HD) * H);
            }
        }

        // ---- Down projection ----
        // HF: down_proj [H, IM] = [1024, 3072]
        // Engine: down has shape (K=IM=3072, N=H=1024)
        // down[k * 1024 + col] with k=0..3071, col=0..1023
        // down[k][col] = hf[col][k] (transpose)
        {
            char tensor_name[128];
            snprintf(tensor_name, 128, "model.layers.%d.mlp.down_proj.weight", l);
            const uint16_t* dp = get_tensor(tensor_name);
            if (dp) {
                std::vector<float> down((size_t)IM * H);
                for (int k = 0; k < IM; k++)
                    for (int col = 0; col < H; col++)
                        down[(size_t)k * H + col] = bf16_to_f32(dp[(size_t)col * IM + k]);
                snprintf(fpath, 256, "%s/down.F32", prefix);
                write_f32(fpath, down.data(), (size_t)IM * H);
            }
        }

        // ---- Norms (1D BF16 vectors) ----
        {
            char tensor_name[128];
            auto copy_norm = [&](const std::string& name, float* dst, int n, const char* suffix) {
                snprintf(tensor_name, 128, "%s", name.c_str());
                const uint16_t* src = get_tensor(tensor_name);
                if (src) {
                    for (int i = 0; i < n; i++) dst[i] = bf16_to_f32(src[i]);
                    snprintf(fpath, 256, "%s/%s", prefix, suffix);
                    write_f32(fpath, dst, n);
                }
            };

            std::vector<float> in_norm(H), pa_norm(H), q_norm(HD), k_norm(HD);

            snprintf(tensor_name, 128, "model.layers.%d.input_layernorm.weight", l);
            copy_norm(tensor_name, in_norm.data(), H, "in_norm.F32");

            snprintf(tensor_name, 128, "model.layers.%d.post_attention_layernorm.weight", l);
            copy_norm(tensor_name, pa_norm.data(), H, "pa_norm.F32");

            snprintf(tensor_name, 128, "model.layers.%d.self_attn.q_norm.weight", l);
            copy_norm(tensor_name, q_norm.data(), HD, "q_norm.F32");

            snprintf(tensor_name, 128, "model.layers.%d.self_attn.k_norm.weight", l);
            copy_norm(tensor_name, k_norm.data(), HD, "k_norm.F32");
        }
    }

    printf("\nDone. Output in %s\n", out_dir);
    return 0;
}

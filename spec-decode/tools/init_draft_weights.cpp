// Initialize draft weights from Qwen3-0.6B target model's first layer
// Creates a "warm-start" draft checkpoint without training
// g++ -O3 -o init_draft_weights init_draft_weights.cpp -lm && ./init_draft_weights
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cstdint>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static const char* kModelPath = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
static const char* kOutputPath = "/home/bcloud/spec-decode/checkpoints/eagle3_draft.bin";

extern "C" float* dequant_i8_to_float(const uint8_t*, int, int*, int*);

static float bf16g(uint16_t v) {
    if ((v & 0x7F80) == 0x7F80) return 0.0f;
    uint32_t b = (uint32_t)v << 16;
    float f; memcpy(&f, &b, 4); return f;
}

// Shapes matching MTPDraftConfig:
constexpr int H = 1024;        // hidden_size
constexpr int NH = 16;         // num_heads
constexpr int NKV = 8;         // num_kv_heads
constexpr int HD = 128;        // head_dim
constexpr int VOCAB = 151936;  // vocab_size
constexpr int IM = 3072;       // inter_dim
constexpr int NTL = 5;         // num_target_layers (layer features to fuse)
constexpr int FC_IN = H * NTL; // 5120

// Write a flat float32 array
static void write_tensor(FILE* out, const float* data, size_t n) {
    fwrite(data, sizeof(float), n, out);
}

int main() {
    printf("Initializing draft weights from %s\n", kModelPath);
    int fd = open(kModelPath, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st; fstat(fd, &st);
    uint8_t* map = (uint8_t*)mmap(0, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    
    uint64_t hsz; memcpy(&hsz, map, 8);
    const char* js = (const char*)(map + 8);
    const uint16_t* data_start = (const uint16_t*)(map + 8 + hsz);
    auto i8p = [&](uint64_t off) { return (const uint8_t*)(map + 8 + hsz + off); };
    
    // Helper: find weight offset in JSON
    auto find_offset = [&](const char* name) -> uint64_t {
        size_t nl = strlen(name);
        const char* p = js, *e = js + hsz;
        while (p < e) {
            auto q = (const char*)memmem(p, e-p, name, nl);
            if (!q) return 0;
            if (q > js && *(q-1) == '"' && *(q+nl) == '"') {
                auto o = strstr(q, "\"data_offsets\"");
                if (o) { auto a = strchr(o, '['); if (a) return strtoull(a+1, NULL, 10); }
            }
            p = q + 1;
        }
        return 0;
    };
    
    FILE* out = fopen(kOutputPath, "wb");
    if (!out) { perror("fopen out"); return 1; }
    
    // Layer 0 weight names for Qwen3-0.6B
    // embed_tokens: [VOCAB, H] from model.embed_tokens.weight
    printf("Extracting embed_tokens [%dx%d]...\n", VOCAB, H);
    std::vector<float> embed_tokens((size_t)VOCAB * H, 0);
    for (int i = 0; i < VOCAB && i < H; i++)  // copy first H rows
        for (int j = 0; j < H; j++)
            embed_tokens[(size_t)i * H + j] = bf16g(data_start[i * H + j]);
    write_tensor(out, embed_tokens.data(), embed_tokens.size());
    printf("  embed_tokens: %.1f MB\n", embed_tokens.size() * 4 / 1e6);
    
    // fc projection: [H, FC_IN] - initialize as identity on first H, zero rest
    printf("Initializing fc [%dx%d]...\n", H, FC_IN);
    std::vector<float> fc((size_t)H * FC_IN, 0);
    for (int i = 0; i < H; i++) fc[(size_t)i * FC_IN + i] = 1.0f; // identity for 1st layer
    write_tensor(out, fc.data(), fc.size());
    printf("  fc: %.1f MB\n", fc.size() * 4 / 1e6);
    
    // Layer 0 weights from target model's first layer
    auto layer0 = [&](const char* name, auto& vec, int rows, int cols) {
        uint64_t off = find_offset(name);
        if (!off) { printf("  WARNING: %s not found, using zeros\n", name); return; }
        int ur, uc;
        float* w = dequant_i8_to_float(i8p(off), cols, &ur, &uc);
        if (w) {
            int r = std::min(ur, rows);
            int c = std::min(uc, cols);
            for (int i = 0; i < r * c; i++) vec[i] = w[i];
            free(w);
        }
    };
    
    auto init_vec = [&](const char* name, auto& vec, int n) {
        uint64_t off = find_offset(name);
        if (!off) return;
        const uint16_t* wp = (const uint16_t*)(map + 8 + hsz + off);
        for (int i = 0; i < std::min(n, 1024); i++) vec[i] = bf16g(wp[i]);
    };
    
    // hidden_norm [H], input_layernorm [H]
    printf("Extracting norms...\n");
    std::vector<float> hidden_norm(H, 1.0f);
    std::vector<float> input_layernorm(H, 1.0f);
    init_vec("model.layers.0.input_layernorm.weight", input_layernorm, H);
    write_tensor(out, hidden_norm.data(), H);
    write_tensor(out, input_layernorm.data(), H);
    
    // QKV projections
    printf("Extracting attention weights...\n");
    std::vector<float> q_proj((size_t)NH*HD * 2*H, 0);
    std::vector<float> k_proj((size_t)NKV*HD * 2*H, 0);
    std::vector<float> v_proj((size_t)NKV*HD * 2*H, 0);
    layer0("model.layers.0.self_attn.q_proj.weight", q_proj, NH*HD, 2*H);
    layer0("model.layers.0.self_attn.k_proj.weight", k_proj, NKV*HD, 2*H);
    layer0("model.layers.0.self_attn.v_proj.weight", v_proj, NKV*HD, 2*H);
    write_tensor(out, q_proj.data(), q_proj.size());
    write_tensor(out, k_proj.data(), k_proj.size());
    write_tensor(out, v_proj.data(), v_proj.size());
    
    // O projection
    std::vector<float> o_proj((size_t)H * NH*HD, 0);
    layer0("model.layers.0.self_attn.o_proj.weight", o_proj, H, NH*HD);
    write_tensor(out, o_proj.data(), o_proj.size());
    
    // Q/K norms
    printf("Extracting QK norms...\n");
    std::vector<float> q_norm(HD, 1.0f);
    std::vector<float> k_norm(HD, 1.0f);
    write_tensor(out, q_norm.data(), HD);
    write_tensor(out, k_norm.data(), HD);
    
    // Post-attention layernorm
    std::vector<float> post_attention_layernorm(H, 1.0f);
    init_vec("model.layers.0.post_attention_layernorm.weight", post_attention_layernorm, H);
    write_tensor(out, post_attention_layernorm.data(), H);
    
    // FFN gate/up/down projections
    printf("Extracting FFN weights...\n");
    std::vector<float> gate_proj((size_t)IM * H, 0);
    std::vector<float> up_proj((size_t)IM * H, 0);
    std::vector<float> down_proj((size_t)H * IM, 0);
    layer0("model.layers.0.mlp.gate_proj.weight", gate_proj, IM, H);
    layer0("model.layers.0.mlp.up_proj.weight", up_proj, IM, H);
    layer0("model.layers.0.mlp.down_proj.weight", down_proj, H, IM);
    write_tensor(out, gate_proj.data(), gate_proj.size());
    write_tensor(out, up_proj.data(), up_proj.size());
    write_tensor(out, down_proj.data(), down_proj.size());
    
    // Final norm + lm_head
    printf("Extracting final norm + LM head...\n");
    std::vector<float> norm(H, 1.0f);
    write_tensor(out, norm.data(), H);
    
    std::vector<float> lm_head((size_t)VOCAB * H, 0);
    // Try lm_head.weight, fall back to embed_tokens (tied)
    uint64_t lm_off = find_offset("lm_head.weight");
    if (lm_off) {
        int lr, lc;
        float* lw = dequant_i8_to_float(i8p(lm_off), H, &lr, &lc);
        if (lw) {
            int r = std::min(lr, VOCAB);
            int c = std::min(lc, H);
            for (int i = 0; i < r * c; i++) lm_head[i] = lw[i];
            free(lw);
        }
    } else {
        printf("  lm_head not found, copying from embed_tokens (tied weights)\n");
        memcpy(lm_head.data(), embed_tokens.data(), embed_tokens.size() * 4);
    }
    write_tensor(out, lm_head.data(), lm_head.size());
    
    fclose(out);
    munmap(map, st.st_size);
    
    // Check file size
    struct stat ost;
    stat(kOutputPath, &ost);
    printf("\n✅ Wrote %s (%.1f MB)\n", kOutputPath, ost.st_size / 1e6);
    printf("Ready for spec decode\n");
    return 0;
}

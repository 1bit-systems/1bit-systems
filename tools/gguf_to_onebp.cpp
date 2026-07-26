/** gguf_to_onebp.cpp — Convert GGUF models to 1BP format.
 *  Build target: `gguf_to_onebp` (see CMakeLists.txt) — pure C++, no Python.
 *  Run:   ./build/gguf_to_onebp model.gguf output.1bp          (Q4NX 4-bit)
 *         ./build/gguf_to_onebp model.gguf output.1bp --tq2    (symmetric ternary)
 *         ./build/gguf_to_onebp model.gguf output.1bp --tq1    (1.58-bit base-3)
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <cstring>
#include "onebp_format.h"
#include <signal.h>
#include "gguf_reader.h"

static void sigfpe_handler(int sig) {
#ifdef _WIN32
    fprintf(stderr, "SIGFPE\n");
#else
    fprintf(stderr, "SIGFPE at %p\n", __builtin_return_address(0));
#endif
    fflush(stderr);
    _exit(1);
}

static inline uint16_t f32b(float v) {
    uint32_t b; memcpy(&b, &v, 4); return (uint16_t)(b >> 16);
}

int main(int argc, char** argv) {
    //signal(SIGFPE, sigfpe_handler);
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input.gguf output.1bp [--tq2 | --tq1]\n", argv[0]);
        return 1;
    }
    // --- Parse quant selection (Q4NX default, --tq2 for symmetric ternary, --tq1 for 1.58-bit) ---
    OnebpQuant quant = ONEBP_Q4NX;
    for (int ai = 3; ai < argc; ai++) {
        if (strcmp(argv[ai], "--tq2") == 0)       quant = ONEBP_TQ2;
        else if (strcmp(argv[ai], "--tq1") == 0)  quant = ONEBP_TQ1;
        else if (strcmp(argv[ai], "--q4nx") == 0) quant = ONEBP_Q4NX;
        else { fprintf(stderr, "Unknown option: %s\n", argv[ai]); return 1; }
    }
    const char* quant_name = (quant == ONEBP_TQ2) ? "TQ2 (ternary 2-bit)" :
                             (quant == ONEBP_TQ1) ? "TQ1 (ternary 1.58-bit)" :
                             "Q4NX (4-bit)";
    GgufReader reader;
    if (!reader.open(argv[1])) {
        fprintf(stderr, "Failed to open GGUF: %s\n", argv[1]);
        return 1;
    }
    printf("GGUF opened: arch=%s tensors=%zu\n",
           reader.architecture().c_str(), reader.tensor_names().size());
    fflush(stdout);

    OnebpHeader hdr;
    hdr.init();
    hdr.quant = quant;
    auto gu = [&](const char* k, int& v) {
        uint32_t x; if (reader.get_u32(k, x)) { v = (int)x; return true; }
        // Try architecture-specific prefix
        std::string arch = reader.architecture();
        if (!arch.empty()) {
            std::string ak = arch + "." + k;
            // Map generic names to arch-specific keys
            if (strcmp(k, "hidden_size") == 0)
                ak = arch + ".embedding_length";
            else if (strcmp(k, "num_hidden_layers") == 0)
                ak = arch + ".block_count";
            else if (strcmp(k, "num_attention_heads") == 0)
                ak = arch + ".attention.head_count";
            else if (strcmp(k, "num_key_value_heads") == 0)
                ak = arch + ".attention.head_count_kv";
            else if (strcmp(k, "head_dim") == 0)
                ak = arch + ".attention.key_length";
            else if (strcmp(k, "intermediate_size") == 0)
                ak = arch + ".feed_forward_length";
            else if (strcmp(k, "vocab_size") == 0)
                ak = arch + ".vocab_size";
            if (reader.get_u32(ak, x)) { v = (int)x; return true; }
        }
        return false;
    };
    gu("hidden_size", hdr.hidden_size) || gu("embedding_length", hdr.hidden_size);
    gu("num_hidden_layers", hdr.num_layers) || gu("block_count", hdr.num_layers);
    // Attention heads are optional — Mamba/MoE architectures have none.
    // Key-value heads default to attention heads; head_dim derived if absent.
    gu("num_attention_heads", hdr.num_attention_heads) || gu("attention.head_count", hdr.num_attention_heads);
    gu("num_key_value_heads", hdr.num_kv_heads) || gu("attention.head_count_kv", hdr.num_kv_heads);
    if (hdr.num_attention_heads > 0 && !hdr.num_kv_heads)
        hdr.num_kv_heads = hdr.num_attention_heads;
    gu("head_dim", hdr.head_dim) || gu("attention.key_length", hdr.head_dim);
    if (!hdr.head_dim && hdr.num_attention_heads > 0)
        hdr.head_dim = hdr.hidden_size / hdr.num_attention_heads;
    gu("intermediate_size", hdr.intermediate_size) || gu("feed_forward_length", hdr.intermediate_size);
    // Try explicit vocab_size; fall back to token_embd.weight rows or tokens array.
    gu("vocab_size", hdr.vocab_size);
    if (!hdr.vocab_size) {
        // Some GGUF files omit vocab_size — infer from token_embd.weight shape.
        auto* emb = reader.tensor_info("token_embd.weight");
        if (emb && emb->shape.size() >= 1) hdr.vocab_size = (int)emb->shape[0];
    }
    if (!hdr.vocab_size) {
        // Fallback: try tokenizer.ggml.tokens array count
        std::vector<std::string> tokens;
        if (reader.get_string_array("tokenizer.ggml.tokens", tokens))
            hdr.vocab_size = (int)tokens.size();
    }
    if (!hdr.valid()) {
        fprintf(stderr, "Bad config: H=%d L=%d NH=%d V=%d\n",
                hdr.hidden_size, hdr.num_layers, hdr.num_attention_heads, hdr.vocab_size);
        return 1;
    }
    printf("Model: H=%d L=%d NH=%d NKV=%d HD=%d IM=%d V=%d\n",
           hdr.hidden_size, hdr.num_layers, hdr.num_attention_heads,
           hdr.num_kv_heads, hdr.head_dim, hdr.intermediate_size, hdr.vocab_size);

    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror("fopen"); return 1; }

    struct TInfo { std::string name; int rows, cols; uint64_t offset, tiled; };
    std::vector<TInfo> tensors;
    uint64_t data_off = 0;
    int tr = 32, tc = 256, gs = 32;

    for (auto& tn : reader.tensor_names()) {
        auto* inf = reader.tensor_info(tn);
        if (!inf || inf->shape.size() != 2) continue;
        int c = (int)inf->shape[0], r = (int)inf->shape[1];
        if (r <= 0 || c <= 0) continue;
        if ((uint64_t)r * (uint64_t)c > 200000000) continue;
        uint64_t tiled = onebp_tiled_size(r, c, tr, tc, gs, quant);
        tensors.push_back({tn, r, c, data_off, tiled});
        data_off += tiled;
        if (tensors.size() <= 3) printf("  tensor %s: %dx%d tiled=%lu\n", tn.c_str(), r, c, tiled);
    }
    printf("  Total tensors: %zu, data size: %.1f MB\n", tensors.size(), data_off / (1024.0*1024.0));
    fflush(stdout);
    hdr.tensor_count = (uint32_t)tensors.size();
    // Write header NOW with correct tensor_count
    fwrite(&hdr, sizeof(hdr), 1, fout);

    // Compute index size and fix up tensor offsets to account for header + index
    uint64_t index_size = 0;
    for (auto& t : tensors) {
        uint32_t nl = std::min((uint32_t)t.name.size(), (uint32_t)63);
        index_size += 4 + nl + 1 + 4 + 8 + 8 + 8;  // name_len + name + \0 + ndim + dims[2] + offset + bytes
    }
    uint64_t data_base = sizeof(OnebpHeader) + index_size;
    for (auto& t : tensors) {
        t.offset += data_base;
    }

    // Write tensor index
    for (auto& t : tensors) {
        uint32_t nl = std::min((uint32_t)t.name.size(), (uint32_t)63);
        fwrite(&nl, 4, 1, fout);
        fwrite(t.name.data(), 1, nl, fout);
        fwrite("\0", 1, 1, fout);
        uint32_t nd = 2;
        fwrite(&nd, 4, 1, fout);
        uint32_t d[2] = {(uint32_t)t.rows, (uint32_t)t.cols};
        fwrite(d, 8, 1, fout);
        fwrite(&t.offset, 8, 1, fout);
        fwrite(&t.tiled, 8, 1, fout);
    }

    printf("Quantizing %zu tensors as %s...\n", tensors.size(), quant_name);
    fflush(stdout);
    auto t0 = std::chrono::steady_clock::now();

    int count = 0;
    for (auto& ti : tensors) {
        count++;
        printf("  [%d/%zu] %s... ", count, tensors.size(), ti.name.c_str()); fflush(stdout);
        // All tensors processed
        std::vector<float> fw;
        auto* inf = reader.tensor_info(ti.name);
        if (inf) printf("%lu elements at offset %lu\n", inf->numel, inf->abs_offset);
        fflush(stdout);
        if (!reader.get_tensor_f32(ti.name, fw)) {
            printf("SKIP (get_tensor_f32 failed)\n"); continue;
        }
        printf("got %zu floats, tiling...\n", fw.size()); fflush(stdout);
        int R = ti.rows, C = ti.cols;
        int ntr = (R + tr - 1) / tr, ntc = (C + tc - 1) / tc;
        printf("  tiles: %dx%d fout=%p\n", ntr, ntc, (void*)fout); fflush(stdout);
        for (int r = 0; r < ntr; r++) {
            for (int c = 0; c < ntc; c++) {
                int r0 = r * tr, c0 = c * tc;
                int grps = tc / gs;
                if (grps <= 0) grps = 1;
                if (quant == ONEBP_TQ2) {
                    // ── TQ2: symmetric ternary (-scale, 0, +scale), no zero-point ──
                    // Per tile: [scales: tr*grps*bf16][codes: tr*tc packed 4/byte].
                    // Scale = max|v| per 32-group => lossless when the source is
                    // already ternary within the group, round-to-nearest otherwise.
                    // code: 0=-scale, 1=0, 2=+scale (LSB-first, 4 codes per byte).
                    size_t sb = (size_t)tr * grps * 2, cb = (size_t)tr * tc / 4;
                    std::vector<uint8_t> tdata(sb + cb, 0);
                    uint16_t* sc = (uint16_t*)tdata.data();
                    uint8_t*  qd = tdata.data() + sb;
                    for (int rr = 0; rr < tr; rr++) {
                        for (int g = 0; g < grps; g++) {
                            int ar = r0 + rr, acs = c0 + g * gs;
                            float maxabs = 0.0f;
                            for (int i = 0; i < gs; i++) {
                                int ac = acs + i;
                                if (ar < R && ac < C) {
                                    float v = fw[(size_t)ar * C + ac];
                                    if (std::isfinite(v)) { float a = fabsf(v); if (a > maxabs) maxabs = a; }
                                }
                            }
                            float s = maxabs;
                            if (s < 1e-20f) s = 1.0f;  // all-zero / padding group
                            float inv_s = 1.0f / s;
                            sc[rr * grps + g] = f32b(s);
                            for (int i = 0; i < gs; i++) {
                                int ac = acs + i;
                                uint8_t code = 1;  // default 0 == +0
                                if (ar < R && ac < C) {
                                    float v = fw[(size_t)ar * C + ac];
                                    if (std::isfinite(v)) {
                                        int t = (int)roundf(v * inv_s);
                                        if (t < -1) t = -1; else if (t > 1) t = 1;
                                        code = (uint8_t)(t + 1);  // -1->0, 0->1, +1->2
                                    }
                                }
                                int local_c = (acs - c0) + i;
                                size_t pos = (size_t)rr * tc + local_c;
                                qd[pos / 4] |= (uint8_t)(code << ((pos & 3) * 2));
                            }
                        }
                    }
                    fwrite(tdata.data(), 1, tdata.size(), fout);
                    continue;
                }
                if (quant == ONEBP_TQ1) {
                    // ── TQ1: 1.58-bit base-3 ternary (5 codes/byte) ──
                    // Groups of 5 elements: bf16 scale + 1 byte with 5 base-3 codes.
                    // code: 0=-scale, 1=0, 2=+scale
                    // packed = code0 + code1*3 + code2*9 + code3*27 + code4*81
                    static const int tq1_pow3[5] = {1, 3, 9, 27, 81};
                    int tq1_grps = (tc + 4) / 5;  // ceil(tc/5)
                    size_t sb = (size_t)tr * tq1_grps * 2;
                    size_t cb = (size_t)tr * tq1_grps;
                    std::vector<uint8_t> tdata(sb + cb, 0);
                    uint16_t* sc = (uint16_t*)tdata.data();
                    uint8_t*  qd = tdata.data() + sb;
                    for (int rr = 0; rr < tr; rr++) {
                        for (int g = 0; g < tq1_grps; g++) {
                            int ar = r0 + rr, acs = c0 + g * 5;
                            float maxabs = 0.0f;
                            for (int i = 0; i < 5; i++) {
                                int ac = acs + i;
                                if (ar < R && ac < C) {
                                    float v = fw[(size_t)ar * C + ac];
                                    if (std::isfinite(v)) { float a = fabsf(v); if (a > maxabs) maxabs = a; }
                                }
                            }
                            float s = maxabs > 1e-20f ? maxabs : 1.0f;
                            float inv_s = 1.0f / s;
                            sc[rr * tq1_grps + g] = f32b(s);
                            uint8_t packed = 0;
                            for (int i = 0; i < 5; i++) {
                                int ac = acs + i;
                                uint8_t code = 1;  // default: 0
                                if (ar < R && ac < C) {
                                    float v = fw[(size_t)ar * C + ac];
                                    if (std::isfinite(v)) {
                                        float q = v * inv_s;
                                        if (q > 0.5f) code = 2;       // +1
                                        else if (q < -0.5f) code = 0;  // -1
                                        else code = 1;                 // 0
                                    }
                                }
                                packed += (uint8_t)(code * tq1_pow3[i]);
                            }
                            qd[rr * tq1_grps + g] = packed;
                        }
                    }
                    fwrite(tdata.data(), 1, tdata.size(), fout);
                    continue;
                }
                // ── Q4NX: asymmetric 4-bit (min + scale per group) ──
                size_t sb = (size_t)tr * grps * 2, zb = sb, db = (size_t)tr * tc / 2;
                std::vector<uint8_t> tdata(sb + zb + db, 0);
                uint16_t* sc = (uint16_t*)tdata.data();
                uint16_t* zp = (uint16_t*)(tdata.data() + sb);
                uint8_t*  qd = tdata.data() + sb + zb;
                for (int rr = 0; rr < tr; rr++) {
                    for (int g = 0; g < grps; g++) {
                        int ar = r0 + rr, acs = c0 + g * gs;
                        float mx = -1e10f, mn = 1e10f;
                        int valid_cnt = 0;
                        for (int i = 0; i < gs; i++) {
                            int ac = acs + i;
                            if (ar < R && ac < C) {
                                float v = fw[(size_t)ar * C + ac];
                                if (std::isfinite(v)) { if (v > mx) mx = v; if (v < mn) mn = v; valid_cnt++; }
                            }
                        }
                        // Handle degenerate groups (all zeros or padding)
                        float s;
                        if (valid_cnt < 2 || mx == mn) { s = 1.0f; mn = 0.0f; }
                        else { s = (mx - mn) / 15.0f; }
                        if (s < 1e-10f) { s = 1.0f; mn = 0.0f; }
                        sc[rr * grps + g] = f32b(s);
                        zp[rr * grps + g] = f32b(mn);
                        for (int i = 0; i < gs; i += 2) {
                            int ac0 = acs + i, ac1 = acs + i + 1;
                            uint8_t v0 = 0, v1 = 0;
                            float inv_s = 1.0f / s;
                            if (ar < R && ac0 < C) {
                                float v = fw[(size_t)ar * C + ac0];
                                v0 = (uint8_t)std::max(0, std::min(15, (int)roundf((v - mn) * inv_s)));
                            }
                            if (ar < R && ac1 < C) {
                                float v = fw[(size_t)ar * C + ac1];
                                v1 = (uint8_t)std::max(0, std::min(15, (int)roundf((v - mn) * inv_s)));
                            }
                            int local_c = (acs - c0) + i; // column within tile
                            qd[((size_t)rr * tc + local_c) / 2] = (v1 << 4) | v0;
                        }
                    }
                }
                fwrite(tdata.data(), 1, tdata.size(), fout);
            }
        }
        printf("  %-50s %4dx%-4d -> %zu KB\n", ti.name.c_str(), R, C, ti.tiled / 1024);
    }

    fseek(fout, 0, SEEK_SET);
    fwrite(&hdr, sizeof(hdr), 1, fout);
    fclose(fout);
    FILE* fc = fopen(argv[2], "rb"); fseek(fc, 0, SEEK_END);
    long fsz = ftell(fc); fclose(fc);
    auto t1 = std::chrono::steady_clock::now();
    printf("\n=== DONE: %s (%.1f MB) in %.0f seconds ===\n",
           argv[2], fsz / (1024.0*1024.0),
           std::chrono::duration<double>(t1 - t0).count());
    return 0;
}

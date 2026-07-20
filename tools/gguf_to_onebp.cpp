/** gguf_to_onebp.cpp — Convert GGUF models to 1BP format.
 *  Build: g++ -std=c++17 -O3 -mavx2 -I include -I /usr/include \
 *         tools/gguf_to_onebp.cpp src/gguf_reader.cpp -o build/gguf_to_onebp -lpthread
 *  Run:   ./build/gguf_to_onebp model.gguf output.1bp
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
    fprintf(stderr, "SIGFPE at %p\n", __builtin_return_address(0));
    fflush(stderr);
    _exit(1);
}

static inline uint16_t f32b(float v) {
    uint32_t b; memcpy(&b, &v, 4); return (uint16_t)(b >> 16);
}

int main(int argc, char** argv) {
    //signal(SIGFPE, sigfpe_handler);
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input.gguf output.1bp\n", argv[0]);
        return 1;
    }
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
    hdr.quant = ONEBP_Q4NX;
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
    gu("num_attention_heads", hdr.num_attention_heads) || gu("attention.head_count", hdr.num_attention_heads);
    gu("num_key_value_heads", hdr.num_kv_heads) || gu("attention.head_count_kv", hdr.num_kv_heads);
    if (!hdr.num_kv_heads) hdr.num_kv_heads = hdr.num_attention_heads;
    gu("head_dim", hdr.head_dim) || gu("attention.key_length", hdr.head_dim);
    if (!hdr.head_dim) hdr.head_dim = hdr.hidden_size / hdr.num_attention_heads;
    gu("intermediate_size", hdr.intermediate_size) || gu("feed_forward_length", hdr.intermediate_size);
    gu("vocab_size", hdr.vocab_size);
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
    hdr.tensor_count = 0;
    fwrite(&hdr, sizeof(hdr), 1, fout);

    struct TInfo { std::string name; int rows, cols; uint64_t offset, tiled; };
    std::vector<TInfo> tensors;
    uint64_t data_off = sizeof(OnebpHeader);
    int tr = 32, tc = 256, gs = 32;

    for (auto& tn : reader.tensor_names()) {
        auto* inf = reader.tensor_info(tn);
        if (!inf || inf->shape.size() != 2) continue;
        int c = (int)inf->shape[0], r = (int)inf->shape[1];
        if (r <= 0 || c <= 0) continue;
        if ((uint64_t)r * c > 200000000) continue;
        uint64_t tiled = onebp_tiled_size(r, c, tr, tc, gs, ONEBP_Q4NX);
        tensors.push_back({tn, r, c, data_off, tiled});
        data_off += tiled;
        if (tensors.size() <= 3) printf("  tensor %s: %dx%d tiled=%lu\n", tn.c_str(), r, c, tiled);
    }
    printf("  Total tensors: %zu, data size: %.1f MB\n", tensors.size(), data_off / (1024.0*1024.0));
    fflush(stdout);
    hdr.tensor_count = (uint32_t)tensors.size();

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

    printf("Quantizing %zu tensors...\n", tensors.size());
    fflush(stdout);
    auto t0 = std::chrono::steady_clock::now();

    int count = 0;
    for (auto& ti : tensors) {
        count++;
        printf("  [%d/%zu] %s... ", count, tensors.size(), ti.name.c_str()); fflush(stdout);
        if (count > 5) { printf("stopping after 5\n"); break; }
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
                int rh = std::min(tr, R - r0), cw = std::min(tc, C - c0);
                int grps = tc / gs;
                if (grps <= 0) grps = 1;
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
                            qd[((size_t)rr * tc + acs + i) / 2] = (v1 << 4) | v0;
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

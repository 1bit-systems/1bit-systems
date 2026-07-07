/** NPU Fused Server — single NPU call per layer (QKV→Attn→O→GU→D).
 *  Uses design_full_layer.xclbin with per-position instructions + capref weights.
 *  Protocol (stdin/stdout binary):
 *    LAYER <layer> <pos> <batch>
 *      → writes batch*H bf16 embedding to hidden BO, calls kernel, outputs final H floats
 *    LM_HEAD <batch>
 *      → reads batch*H floats, outputs batch int32 token IDs (CPU lm_head)
 *    EXIT
 *
 *  Target: ~250+ tok/s (single fused NPU call eliminates PCIe/CPU between ops)
 *  Startup: loads xclbin + 128 per-position instruction files + 28 weight files
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <unistd.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

#include "platform.h"
#include "model_config.h"

static constexpr int H = 1024;
static constexpr int NC = 28;
static constexpr int NV = 151936;
static constexpr int MAX_POS = 4096;
static constexpr int MAX_INSTR_POS = 128;
static constexpr int INSTR_WORDS = 1723;
static constexpr size_t KV_BYTES = 256 * 1024;  // 256KB per K or V cache per layer

// BF16 helpers (bf16f is in platform.h, define only helper that's missing)
static inline uint16_t f32bf(float v) {
    uint32_t b; memcpy(&b, &v, 4);
    return (uint16_t)((b + 0x8000) >> 16);
}

// Load binary file
static std::vector<char> load_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    size_t sz = f.tellg(); f.seekg(0);
    std::vector<char> data(sz); f.read(data.data(), sz);
    return data;
}

// Embedding table (bf16)
static std::vector<uint16_t> g_embed;
static std::vector<float> g_lm_head_f32;

extern "C" float* dequant_i8_to_float_ex(const uint8_t*, int, int, int*, int*);

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stdin, NULL, _IONBF, 0);

    const char* model_path = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
    const char* xclbin_dir = "/home/bcloud/torch2aie/examples/qwen3-decode-layer/build/qwen3-decode-layer-capacity-token127";
    const char* weights_dir = "/home/bcloud/npu-sandbox/npu-infer/build/int8/capref";

    if (auto* env = getenv("FUSED_MODEL")) model_path = env;
    if (auto* env = getenv("FUSED_XCLBIN_DIR")) xclbin_dir = env;
    if (auto* env = getenv("FUSED_WEIGHTS_DIR")) weights_dir = env;

    printf("FUSED: Loading model %s\n", model_path);
    printf("FUSED: xclbin=%s weights=%s\n", xclbin_dir, weights_dir);
    fflush(stdout);

    // Load model for embeddings + lm_head
    printf("FUSED: opening model...\n"); fflush(stdout);
    auto fd = platform_open_read(model_path);
    printf("FUSED: fd=%d\n", fd); fflush(stdout);
    platform_stat st; platform_fstat(fd, &st);
    printf("FUSED: size=%lu\n", (unsigned long)st.st_size); fflush(stdout);
    auto* md = (const uint8_t*)platform_mmap((size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    printf("FUSED: mmap=%p\n", (void*)md); fflush(stdout);
    platform_close(fd);
    uint64_t hsz; memcpy(&hsz, md, 8);
    uint64_t df = 8 + hsz;
    const char* js = (const char*)(md + 8);
    size_t jl = hsz;

    // BF16 embeddings
    auto emb = (const uint16_t*)(md + df);
    g_embed.assign(emb, emb + (size_t)NV * H);

    // Load lm_head from model
    auto json_off = [&](const char* name) -> uint64_t {
        size_t nl = strlen(name);
        const char* p = js, *e = js + jl;
        while (p < e) {
            auto q = (const char*)platform_memmem(p, e - p, name, nl);
            if (!q) return 0;
            if (q > js && *(q-1) == '"' && *(q+nl) == '"') {
                auto o = strstr(q, "\"data_offsets\"");
                if (o) { auto a = strchr(o, '['); if (a) return strtoull(a + 1, NULL, 10); }
            }
            p = q + 1;
        }
        return 0;
    };

    auto i8p = [&](uint64_t o) { return md + df + o; };
    // Find tensor INT8 rows
    auto gi8 = [&](const char* k) -> int {
        int r = 0; find_tensor_info(js, jl, k, &r); return r;
    };
    int lm_i8 = gi8("lm_head.weight");
    printf("FUSED: lm_head i8_rows=%d\n", lm_i8);
    uint64_t lo = json_off("lm_head.weight");
    if (lo && lm_i8 > 0) {
        int lr = 0, lc = 0;
        float* lm_raw = dequant_i8_to_float_ex(i8p(lo), lm_i8, H, &lr, &lc);
        if (lr > 0 && lc > 0) {
            g_lm_head_f32.assign(lm_raw, lm_raw + (size_t)lr * lc);
            free(lm_raw);
            printf("FUSED: LM head %dx%d\n", lr, lc);
        }
    }
    const float* lm_emb = g_lm_head_f32.empty() ? (const float*)emb : g_lm_head_f32.data();
    int lm_rows = g_lm_head_f32.empty() ? NV : (int)(g_lm_head_f32.size() / H);
    if (lm_rows == 0) lm_rows = NV;

    // Final norm weights from model
    std::vector<float> fin_w(H, 1.0f);
    uint64_t fn_off = json_off("model.norm.weight");
    if (fn_off) {
        auto* fwp = (const uint16_t*)(md + df + fn_off);
        for (int i = 0; i < H; i++) fin_w[i] = bf16f(fwp[i]);
    }

    // Load fused xclbin
    printf("FUSED: Loading xclbin...\n");
    auto xclbin_data = load_bin(std::string(xclbin_dir) + "/design.xclbin");
    if (xclbin_data.empty()) { fprintf(stderr, "No xclbin at %s\n", xclbin_dir); return 1; }
    xrt::device dev(0);
    xrt::xclbin xclbin(xclbin_data);
    dev.register_xclbin(xclbin);
    xrt::hw_context hctx(dev, xclbin.get_uuid());
    xrt::kernel kernel(hctx, "MLIR_AIE");

    // Kernel argument groups
    int ig = kernel.group_id(1);  // instructions
    int kg = kernel.group_id(3);  // K cache
    int vg = kernel.group_id(4);  // V cache
    int wg = kernel.group_id(5);  // weights
    int og = kernel.group_id(6);  // output
    int hg = kernel.group_id(7);  // hidden

    // Load instruction files
    printf("FUSED: Loading instruction files...\n");
    std::vector<xrt::bo> instr_bos(MAX_INSTR_POS);
    auto generic_data = load_bin(std::string(xclbin_dir) + "/design.bin");
    xrt::bo generic_bo(dev, std::max(generic_data.size(), (size_t)65536),
                       xrt::bo::flags::cacheable, ig);
    memset(generic_bo.map(), 0, 65536);
    if (!generic_data.empty()) memcpy(generic_bo.map(), generic_data.data(), generic_data.size());
    generic_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    int instr_loaded = 0;
    for (int pos = 0; pos < MAX_INSTR_POS; pos++) {
        char fname[256];
        snprintf(fname, 256, "%s/design-token127-to-token%d.bin", xclbin_dir, pos);
        auto data = load_bin(fname);
        if (!data.empty()) {
            auto bo = xrt::bo(dev, std::max(data.size(), (size_t)65536),
                              xrt::bo::flags::cacheable, ig);
            memset(bo.map(), 0, 65536);
            memcpy(bo.map(), data.data(), data.size());
            bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            instr_bos[pos] = std::move(bo);
            instr_loaded++;
        }
    }
    printf("FUSED: %d per-position instruction files loaded\n", instr_loaded);

    // Load reference weights
    printf("FUSED: Loading per-layer weight files...\n");
    std::vector<xrt::bo> weight_bos(NC);
    for (int l = 0; l < NC; l++) {
        char fname[256];
        snprintf(fname, 256, "%s/wref_l%d.bin", weights_dir, l);
        auto data = load_bin(fname);
        if (data.empty()) { fprintf(stderr, "Missing %s\n", fname); return 1; }
        auto bo = xrt::bo(dev, data.size(), xrt::bo::flags::host_only, wg);
        memcpy(bo.map(), data.data(), data.size());
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        weight_bos[l] = std::move(bo);
    }

    // Load RoPE table
    std::vector<int32_t> rope_table;
    {
        auto rt = load_bin(std::string(weights_dir) + "/rope_table.bin");
        if (rt.empty()) { fprintf(stderr, "Missing rope_table.bin\n"); return 1; }
        rope_table.resize(rt.size() / 4);
        memcpy(rope_table.data(), rt.data(), rt.size());
    }
    static constexpr int ROPE_COSSIN_DWORDS = 64;
    static constexpr int ROPE_COSSIN_DWORD_OFFSET = 1152;

    // Create BOs: KV caches (per-layer), hidden, output
    printf("FUSED: Creating BOs...\n");
    std::vector<xrt::bo> kCache, vCache;
    for (int l = 0; l < NC; l++) {
        kCache.emplace_back(dev, KV_BYTES, xrt::bo::flags::host_only, kg);
        vCache.emplace_back(dev, KV_BYTES, xrt::bo::flags::host_only, vg);
    }
    xrt::bo bHidden(dev, H * 2, xrt::bo::flags::host_only, hg);  // bf16 hidden state
    xrt::bo bOutput(dev, H * 2, xrt::bo::flags::host_only, og);  // bf16 output

    printf("FUSED: READY\n");
    fflush(stdout);

    // Pre-load xclbin bytes for AIE2 array reset between layers
    auto xclbin_bytes = load_bin(std::string(xclbin_dir) + "/design.xclbin");

    char cmd[512];
    int last_rope_pos = -1;
    bool running = true;
    while (running && fgets(cmd, sizeof(cmd), stdin)) {
        size_t clen = strlen(cmd);
        while (clen > 0 && (cmd[clen-1] == '\n' || cmd[clen-1] == '\r')) cmd[--clen] = '\0';

        char op[32] = {0};
        int layer = 0, pos = 0, batch = 0;
        int parsed = sscanf(cmd, "%31s %d %d %d", op, &layer, &pos, &batch);
        if (parsed < 1) continue;

        // ── EXIT ──
        if (strcmp(op, "EXIT") == 0) break;

        // ── LAYER <layer> <pos> <batch> ──
        // Reads batch*H BF16 embedding values from stdin, runs fused kernel,
        // outputs batch*H BF16 hidden state to stdout
        else if (strcmp(op, "LAYER") == 0 && parsed == 4) {
            if (layer < 0 || layer >= NC || batch <= 0 || batch > 1) {
                fprintf(stderr, "FUSED: invalid LAYER args (B must be 1 for fused kernel)\n");
                continue;
            }

            // Read input hidden state (BF16, batch=1 for now)
            uint16_t hidden_in[H];
            size_t got = fread(hidden_in, 2, H, stdin);
            if ((int)got != H) { fprintf(stderr, "FUSED: short read\n"); break; }

            // Copy embedding to hidden BO
            memcpy(bHidden.map(), hidden_in, H * 2);
            bHidden.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            // Patch RoPE once per position (not per layer)
            if (pos != last_rope_pos) {
                size_t npos = rope_table.size() / ROPE_COSSIN_DWORDS;
                int cp = pos;
                if (cp < 0) cp = 0;
                if ((size_t)cp >= npos) cp = (int)npos - 1;
                const int32_t* cs = rope_table.data() + (size_t)cp * ROPE_COSSIN_DWORDS;
                for (int pl = 0; pl < NC; pl++) {
                    int32_t* wmap = (int32_t*)weight_bos[pl].map();
                    memcpy(wmap + ROPE_COSSIN_DWORD_OFFSET, cs, ROPE_COSSIN_DWORDS * 4);
                    weight_bos[pl].sync(XCL_BO_SYNC_BO_TO_DEVICE,
                                       ROPE_COSSIN_DWORDS * 4, ROPE_COSSIN_DWORD_OFFSET * 4);
                }
                last_rope_pos = pos;
            }

            // Reload xclbin to reset AIE2 array (cores halt after each kernel run)
            {
                xrt::xclbin fresh_xc(xclbin_bytes);
                dev.register_xclbin(fresh_xc);
                xrt::hw_context fresh_hctx(dev, fresh_xc.get_uuid());
                kernel = xrt::kernel(fresh_hctx, "MLIR_AIE");
            }
            // Run fused kernel for this layer
            auto& ibo = pos < MAX_INSTR_POS && instr_bos[pos] ? instr_bos[pos] : generic_bo;
            unsigned opcode = 3;
            if (getenv("FUSED_OPCODE")) opcode = (unsigned)atoi(getenv("FUSED_OPCODE"));
            auto run = kernel(opcode, ibo, (uint32_t)INSTR_WORDS,
                             kCache[layer], vCache[layer], weight_bos[layer],
                             bOutput, bHidden);
            run.wait();

            // Read output
            bOutput.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            uint16_t* out_data = (uint16_t*)bOutput.map();

            // Output BF16 hidden state
            fwrite(out_data, 2, H, stdout);
            fflush(stdout);
        }

        // ── LM_HEAD <batch> ──
        // Reads batch*H floats, outputs batch int32 token IDs
        else if (strcmp(op, "LM_HEAD") == 0 && parsed >= 2) {
            int bsize = batch > 0 ? batch : 1;
            if (bsize > 1) bsize = 1;
            std::vector<float> hidden_in(bsize * H);
            size_t got = fread(hidden_in.data(), 4, bsize * H, stdin);
            if ((int)got != bsize * H) break;

            for (int b = 0; b < bsize; b++) {
                float* h = &hidden_in[b * H];
                // RMSNorm
                double ss = 0;
                for (int i = 0; i < H; i++) ss += (double)h[i] * h[i];
                float ir = 1.0f / sqrtf((float)(ss / H) + 1e-6f);
                for (int i = 0; i < H; i++) h[i] *= ir * fin_w[i];

                // Argmax
                int best = 0;
                double best_s = -1e30;
                for (int n = 0; n < lm_rows; n++) {
                    double s = 0;
                    const float* row = lm_emb + (size_t)n * H;
                    for (int k = 0; k < H; k++) s += (double)h[k] * row[k];
                    if (s > best_s) { best_s = s; best = n; }
                }
                int32_t tid = (int32_t)best;
                fwrite(&tid, 4, 1, stdout);
            }
            fflush(stdout);
        }

        // ── LAYER_BATCH <layer> <pos> <batch> ──
        // Multi-token batch (CPU-attention fallback, uses component QKV+FFN)
        // For B>1, we need to do the CPU-side work that the fused kernel handles internally
        // This is a future optimization for B>1 with the fused xclbin
        else {
            fprintf(stderr, "FUSED: unknown cmd: %s\n", cmd);
        }
    }

    platform_munmap((void*)md, (size_t)st.st_size);
    fprintf(stderr, "FUSED: DONE\n");
    return 0;
}

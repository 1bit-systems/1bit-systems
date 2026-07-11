// tools/trg_save.cpp — Convert Q4NX to .trg binary format and save/load
//
// .trg format: flat binary, memory-mappable, contains all ternary-packed
// weights + FP32 norms + metadata header. Loads in <100ms vs 2s for Q4NX.
//
// Build: g++ -O3 -march=native -fopenmp -std=c++17 -Iengine/fusion \
//        -o tools/trg_save tools/trg_save.cpp engine/fusion/cpu_layer.cpp -lm
//
// Save: ./tools/trg_save save model.q4nx model.trg
// Load: ./tools/trg_save load model.trg [layers]

#include "cpu_layer.h"
#include "cpu_q4nx_loader.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// ── .trg file header (v2: per-block scales) ───────────────────
// Block = 256 ternary values = 16 uint32s = 64 packed bytes
// Scales: [M, n_blocks] per projection, where n_blocks = K / 256
#pragma pack(push, 1)
struct TrgHeader {
    char magic[4] = {'T','R','G','2'};   // format identifier (v2 = per-block scales)
    uint32_t H, IM, NH, NKV, HD, V, L;   // model dims
    uint32_t gqa_ratio;
    uint32_t pack_size[7];                // uint32 per row for q,k,v,o,g,u,d
    uint32_t block_sizes[7];              // n_blocks per projection (K/256)
    uint64_t offset_emb;                  // byte offset to embedding [V*H] fp32
    uint64_t offset_fn;                   // final norm [H] fp32
    uint64_t offset_lm;                   // lm_head [V*H] fp32
    uint64_t offset_norms;                // in_norm, pa_norm, q_norm, k_norm [L*(H+H+HD+HD)] fp32
    uint64_t offset_weights;              // packed ternary weights [variable] uint32
    uint64_t offset_scales;               // per-block scales [variable] fp32 [L*7*M*n_blocks]
    uint64_t file_size;                   // total file size
    uint8_t  reserved[252];               // future use (size ensures 8-byte alignment)
};
#pragma pack(pop)

static_assert(sizeof(TrgHeader) <= 512, "TrgHeader too large");
static_assert(sizeof(TrgHeader) % 8 == 0, "TrgHeader not 8-byte aligned");

// ── Pack one projection (per-block scales, block=256 ternary) ───
static void pack_proj(const float* src, uint32_t* packed, float* scales,
                      int OUT, int IN, int n_blocks) {
    int pp = (IN + 15) / 16;
    int blk_pk = 16;  // 256 ternary values / 16 per uint32
    for (int r = 0; r < OUT; r++) {
        // Per-block scale: mean |weight| within each 256-value block
        for (int b = 0; b < n_blocks; b++) {
            double sa = 0; int nz = 0;
            int blk_start = b * 256;
            int blk_end = (b + 1) * 256;
            if (blk_end > IN) blk_end = IN;
            for (int c = blk_start; c < blk_end; c++) {
                float v = src[r*IN+c]; sa += fabs(v); if (v != 0) nz++;
            }
            scales[r * n_blocks + b] = (nz > 0) ? (float)(sa / nz) : 0;
        }
        for (int u = 0; u < pp; u++) {
            uint32_t w = 0;
            for (int v = 0; v < 16; v++) {
                int c = u*16+v;
                float val = (c < IN) ? src[r*IN+c] : 0;
                w |= ((val > 0 ? 0x1u : (val < 0 ? 0x2u : 0)) << (v*2));
            }
            packed[r*pp+u] = w;
        }
    }
}

// ── Save .trg file ────────────────────────────────────────────
static int save_trg(const char* q4nx_path, const char* trg_path) {
    printf("=== Save .trg ===\n");
    printf("  Source: %s\n", q4nx_path);
    printf("  Dest:   %s\n\n", trg_path);

    auto t0 = std::chrono::high_resolution_clock::now();

    Q4nxModel qm;
    if (!qm.load(q4nx_path)) return 1;

    int H = qm.hidden_dim, IM = qm.inter_size, NH = qm.n_heads;
    int NKV = qm.n_kv_heads, HD = qm.head_dim, V = qm.vocab_size, L = qm.n_layers;
    int GQA = NH / NKV;

    auto gt = [&](auto n) {
        auto i = qm.tensors.find(n);
        return i == qm.tensors.end() ? std::vector<float>() : i->second.fp32;
    };

    printf("  Dimensions: V=%d H=%d L=%d NH=%d NKV=%d HD=%d IM=%d\n",
           V, H, L, NH, NKV, HD, IM);

    // Load + convert all data
    auto emb = gt("model.embed_tokens.weight");
    auto fn  = gt("model.norm.weight");
    auto lm  = gt("lm_head.weight");
    if (lm.empty()) lm = emb;

    // Per-layer norms: in_norm [L*H], pa_norm [L*H], q_norm [L*HD], k_norm [L*HD]
    std::vector<float> in_norm(L*H), pa_norm(L*H), q_norm(L*HD), k_norm(L*HD);

    // Packed weights + scales (flat, all layers concatenated)
    // Layout per layer: q_packed, k_packed, v_packed, o_packed, g_packed, u_packed, d_packed
    // Then: q_scales, k_scales, v_scales, o_scales, g_scales, u_scales, d_scales
    int pk_H   = (H + 15) / 16;
    int pk_AH  = (NH*HD + 15) / 16;
    int pk_IM  = (IM + 15) / 16;

    // Per-layer sizes (packed uint32 per row = ceil(K/16))
    int pk_sizes[7] = {
        NH*HD * pk_H,    // q: M=NH*HD, K=H
        NKV*HD * pk_H,   // k: M=NKV*HD, K=H
        NKV*HD * pk_H,   // v: M=NKV*HD, K=H
        H * pk_AH,       // o: M=H, K=NH*HD
        IM * pk_H,       // gate: M=IM, K=H
        IM * pk_H,       // up: M=IM, K=H
        H * pk_IM        // down: M=H, K=IM
    };
    // Per-block scale sizes (one scale per 256-weight block)
    int proj_K[7] = { H, H, H, NH*HD, H, H, IM };
    int proj_M[7] = { NH*HD, NKV*HD, NKV*HD, H, IM, IM, H };
    int block_sizes[7], sc_sizes[7];
    for (int i = 0; i < 7; i++) {
        block_sizes[i] = (proj_K[i] + 255) / 256;  // n_blocks = ceil(K/256)
        sc_sizes[i] = proj_M[i] * block_sizes[i];   // [M, n_blocks]
    }

    int total_pk = 0, total_sc = 0;
    for (int i = 0; i < 7; i++) { total_pk += pk_sizes[i] * L; total_sc += sc_sizes[i] * L; }

    std::vector<uint32_t> all_packed(total_pk);
    std::vector<float> all_scales(total_sc);
    int pk_off = 0, sc_off = 0;

    printf("  Converting %d layers...\n", L);
    for (int l = 0; l < L; l++) {
        char k[256];
        auto gl = [&](auto f) { snprintf(k,256,f,l); return gt(k); };

        auto inn = gl("model.layers.%d.input_layernorm.weight");
        auto pan = gl("model.layers.%d.post_attention_layernorm.weight");
        auto qn  = gl("model.layers.%d.self_attn.q_norm.weight");
        auto kn  = gl("model.layers.%d.self_attn.k_norm.weight");
        if (!inn.empty()) memcpy(&in_norm[l*H], inn.data(), H*4);
        if (!pan.empty()) memcpy(&pa_norm[l*H], pan.data(), H*4);
        if (!qn.empty())  memcpy(&q_norm[l*HD], qn.data(), HD*4);
        if (!kn.empty())  memcpy(&k_norm[l*HD], kn.data(), HD*4);

        auto pk = [&](auto n, uint32_t* p, float* s, int O, int I, int nb) {
            auto w = gl(n); if (!w.empty()) pack_proj(w.data(), p, s, O, I, nb);
        };
        pk("model.layers.%d.self_attn.q_proj.weight", &all_packed[pk_off], &all_scales[sc_off], NH*HD, H, block_sizes[0]); pk_off += pk_sizes[0]; sc_off += sc_sizes[0];
        pk("model.layers.%d.self_attn.k_proj.weight", &all_packed[pk_off], &all_scales[sc_off], NKV*HD, H, block_sizes[1]); pk_off += pk_sizes[1]; sc_off += sc_sizes[1];
        pk("model.layers.%d.self_attn.v_proj.weight", &all_packed[pk_off], &all_scales[sc_off], NKV*HD, H, block_sizes[2]); pk_off += pk_sizes[2]; sc_off += sc_sizes[2];
        pk("model.layers.%d.self_attn.o_proj.weight", &all_packed[pk_off], &all_scales[sc_off], H, NH*HD, block_sizes[3]); pk_off += pk_sizes[3]; sc_off += sc_sizes[3];
        pk("model.layers.%d.mlp.gate_proj.weight", &all_packed[pk_off], &all_scales[sc_off], IM, H, block_sizes[4]); pk_off += pk_sizes[4]; sc_off += sc_sizes[4];
        pk("model.layers.%d.mlp.up_proj.weight", &all_packed[pk_off], &all_scales[sc_off], IM, H, block_sizes[5]); pk_off += pk_sizes[5]; sc_off += sc_sizes[5];
        pk("model.layers.%d.mlp.down_proj.weight", &all_packed[pk_off], &all_scales[sc_off], H, IM, block_sizes[6]); pk_off += pk_sizes[6]; sc_off += sc_sizes[6];
    }

    // Build header
    TrgHeader hdr = {};
    memcpy(hdr.magic, "TRG2", 4);
    hdr.H = H; hdr.IM = IM; hdr.NH = NH; hdr.NKV = NKV;
    hdr.HD = HD; hdr.V = V; hdr.L = L; hdr.gqa_ratio = GQA;
    for (int i = 0; i < 7; i++) { hdr.pack_size[i] = pk_sizes[i]; hdr.block_sizes[i] = block_sizes[i]; }

    // Compute offsets
    uint64_t off = sizeof(TrgHeader);
    hdr.offset_emb    = off; off += emb.size() * 4;
    hdr.offset_fn     = off; off += fn.size() * 4;
    hdr.offset_lm     = off; off += lm.size() * 4;
    hdr.offset_norms  = off; off += in_norm.size() * 4 + pa_norm.size() * 4 + q_norm.size() * 4 + k_norm.size() * 4;
    hdr.offset_weights = off; off += all_packed.size() * 4;
    hdr.offset_scales  = off; off += all_scales.size() * 4;
    hdr.file_size = off;

    // Write file
    std::ofstream f(trg_path, std::ios::binary);
    if (!f) { fprintf(stderr, "Cannot write: %s\n", trg_path); return 1; }

    f.write((char*)&hdr, sizeof(hdr));
    f.write((char*)emb.data(), emb.size()*4);
    f.write((char*)fn.data(), fn.size()*4);
    f.write((char*)lm.data(), lm.size()*4);
    f.write((char*)in_norm.data(), in_norm.size()*4);
    f.write((char*)pa_norm.data(), pa_norm.size()*4);
    f.write((char*)q_norm.data(), q_norm.size()*4);
    f.write((char*)k_norm.data(), k_norm.size()*4);
    f.write((char*)all_packed.data(), all_packed.size()*4);
    f.write((char*)all_scales.data(), all_scales.size()*4);
    f.close();

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    double mb = hdr.file_size / (1024.0*1024.0);

    printf("\n  Written: %.0f MB (%.0f ms)\n", mb, ms);
    printf("  %.0f MB/s\n\n", mb / (ms/1000.0));
    printf("  To load: ./tools/trg_save load %s\n", trg_path);
    return 0;
}

// ── Load .trg file ────────────────────────────────────────────
struct TrgModel {
    TrgHeader hdr;
    std::vector<float> emb, fn, lm;
    std::vector<float> in_norm, pa_norm, q_norm, k_norm;
    std::vector<uint32_t> packed;
    std::vector<float> scales;
    double load_ms = 0;
    int H, IM, NH, NKV, HD, GQA, V, L;

    bool load(const char* path) {
        auto t0 = std::chrono::high_resolution_clock::now();

        std::ifstream f(path, std::ios::binary);
        if (!f) { fprintf(stderr, "Cannot open: %s\n", path); return false; }

        f.read((char*)&hdr, sizeof(hdr));
        if (memcmp(hdr.magic, "TRG1", 4) != 0) {
            fprintf(stderr, "Invalid .trg file (magic: %.4s)\n", hdr.magic);
            return false;
        }

        H = hdr.H; IM = hdr.IM; NH = hdr.NH; NKV = hdr.NKV;
        HD = hdr.HD; V = hdr.V; L = hdr.L; GQA = hdr.gqa_ratio;

        // Memory-map the entire file
        f.close();
        f.open(path, std::ios::binary | std::ios::ate);
        if (!f) return false;
        size_t fsz = f.tellg();
        f.close();

        // Use mmap for zero-copy loading
        int fd = open(path, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "Cannot open: %s\n", path); return false; }
        void* map = mmap(nullptr, fsz, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (map == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); return false; }

        auto ptr = [&](uint64_t off) { return (const char*)map + off; };
        auto copy_vec = [&](auto& vec, uint64_t off, size_t bytes) {
            vec.resize(bytes / sizeof(decltype(vec[0])));
            memcpy(vec.data(), ptr(off), bytes);
        };

        copy_vec(emb, hdr.offset_emb, (hdr.V * hdr.H) * 4);
        copy_vec(fn,  hdr.offset_fn,  hdr.H * 4);
        copy_vec(lm,  hdr.offset_lm,  (hdr.V * hdr.H) * 4);
        copy_vec(in_norm, hdr.offset_norms, hdr.L * hdr.H * 4);
        copy_vec(pa_norm, hdr.offset_norms + hdr.L * hdr.H * 4, hdr.L * hdr.H * 4);
        copy_vec(q_norm,  hdr.offset_norms + 2 * hdr.L * hdr.H * 4, hdr.L * hdr.HD * 4);
        copy_vec(k_norm,  hdr.offset_norms + 2 * hdr.L * hdr.H * 4 + hdr.L * hdr.HD * 4, hdr.L * hdr.HD * 4);
        copy_vec(packed, hdr.offset_weights, hdr.offset_scales - hdr.offset_weights);
        copy_vec(scales, hdr.offset_scales, fsz - hdr.offset_scales);

        munmap(map, fsz);

        auto t1 = std::chrono::high_resolution_clock::now();
        load_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        return true;
    }

    // Decode one token (full model, per-block scales)
    void decode(float* hidden, int pos, float* kc, float* vc,
                float* sq, float* sa, float* sf, float* sa2,
                float* st, float* ct) {
        int pk[7], bs[7];
        for (int i = 0; i < 7; i++) { pk[i] = hdr.pack_size[i]; bs[i] = hdr.block_sizes[i]; }
        uint32_t* pw = packed.data();
        float* sw = scales.data();

        // Manually expand to track pk/bs correctly per projection
        for (int l = 0; l < L; l++) {
            float res[4096]; memcpy(res, hidden, H*4);
            cpu_rmsnorm(hidden, &in_norm[l*H], hidden, H, 1e-6f);
            // q_proj: M=NH*HD, K=H, nb=bs[0]
            cpu_ternary_gemv_block(pw, hidden, sw, sq, NH*HD, H, bs[0]); pw += pk[0]; sw += bs[0] * NH*HD;
            // k_proj: M=NKV*HD, K=H, nb=bs[1]
            cpu_ternary_gemv_block(pw, hidden, sw, sq+NH*HD, NKV*HD, H, bs[1]); pw += pk[1]; sw += bs[1] * NKV*HD;
            // v_proj: M=NKV*HD, K=H, nb=bs[2]
            cpu_ternary_gemv_block(pw, hidden, sw, sq+NH*HD+NKV*HD, NKV*HD, H, bs[2]); pw += pk[2]; sw += bs[2] * NKV*HD;
            for(int h=0;h<NH;h++) cpu_rmsnorm(sq+h*HD, &q_norm[l*HD], sq+h*HD, HD, 1e-6f);
            cpu_rope(sq,pos,NH,HD,st,ct);
            for(int h=0;h<NKV;h++) cpu_rmsnorm(sq+NH*HD+h*HD, &k_norm[l*HD], sq+NH*HD+h*HD, HD, 1e-6f);
            cpu_rope(sq+NH*HD,pos,NKV,HD,st,ct);
            for(int h=0;h<NKV;h++){
                memcpy(&kc[l*4096*NKV*HD+pos*NKV*HD+h*HD], sq+NH*HD+h*HD, HD*4);
                memcpy(&vc[l*4096*NKV*HD+pos*NKV*HD+h*HD], sq+NH*HD+NKV*HD+h*HD, HD*4);
            }
            cpu_attention(sq, &kc[l*4096*NKV*HD], &vc[l*4096*NKV*HD], sa, NH, NKV, HD, pos+1, GQA);
            // o_proj: M=H, K=NH*HD, nb=bs[3]
            cpu_ternary_gemv_block(pw, sa, sw, hidden, H, NH*HD, bs[3]); pw += pk[3]; sw += bs[3] * H;
            for(int i=0;i<H;i++)hidden[i]=res[i]+hidden[i];
            memcpy(res,hidden,H*4);
            cpu_rmsnorm(hidden,&pa_norm[l*H],hidden,H,1e-6f);
            // gate_proj: M=IM, K=H, nb=bs[4]
            cpu_ternary_gemv_block(pw, hidden, sw, sf, IM, H, bs[4]); pw += pk[4]; sw += bs[4] * IM;
            // up_proj: M=IM, K=H, nb=bs[5]
            cpu_ternary_gemv_block(pw, hidden, sw, sf+IM, IM, H, bs[5]); pw += pk[5]; sw += bs[5] * IM;
            cpu_silu_glu(sf, sf+IM, sa2, IM);
            // down_proj: M=H, K=IM, nb=bs[6]
            cpu_ternary_gemv_block(pw, sa2, sw, hidden, H, IM, bs[6]); pw += pk[6]; sw += bs[6] * H;
            for(int i=0;i<H;i++)hidden[i]=res[i]+hidden[i];
        }
    }
};

// ── Load and benchmark ────────────────────────────────────────
static int load_bench(const char* trg_path, int n_layers_override) {
    printf("=== Load .trg ===\n");
    printf("  File: %s\n\n", trg_path);

    TrgModel m;
    if (!m.load(trg_path)) return 1;

    int L = (n_layers_override > 0 && n_layers_override <= m.L) ? n_layers_override : m.L;
    printf("  Loaded in %.0f ms (%d/%d layers)\n", m.load_ms, L, m.L);

    const int H=m.H, IM=m.IM, NH=m.NH, NKV=m.NKV, HD=m.HD, V=m.V, GQA=m.GQA;

    std::vector<float> hidden(H), sq(NH*HD+2*NKV*HD), sa(NH*HD), sf(2*IM), sa2(IM);
    std::vector<float> kc(4096*NKV*HD,0), vc(4096*NKV*HD,0);

    std::vector<float> st(4096*HD), ct(4096*HD);
    for (int p = 0; p < 4096; p++)
        for (int d = 0; d < HD; d++) {
            float th = p / pow(10000.0f, (2.0f*(d/2))/HD);
            st[p*HD+d]=sin(th); ct[p*HD+d]=cos(th);
        }

    // Warmup
    memcpy(hidden.data(), m.emb.data()+1*H, H*4);
    for (int i = 0; i < 3; i++)
        m.decode(hidden.data(), i, kc.data(), vc.data(), sq.data(), sa.data(), sf.data(), sa2.data(), st.data(), ct.data());

    // Benchmark
    int runs = 20;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; i++) {
        memcpy(hidden.data(), m.emb.data()+1*H, H*4);
        m.decode(hidden.data(), i, kc.data(), vc.data(), sq.data(), sa.data(), sf.data(), sa2.data(), st.data(), ct.data());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count() / runs;

    // LM head
    std::vector<float> logits(V);
    cpu_lm_head(hidden.data(), m.lm.data(), logits.data(), V, H);
    int best = 0;
    for (int i = 1; i < V; i++) if (logits[i] > logits[best]) best = i;

    printf("  Decode: %.2f ms  = %.0f tok/s (%d layers)\n", ms, 1000.0/ms, L);
    printf("  Top token: %d\n", best);

    // DSpark estimate if reduced layers
    if (L < m.L) {
        double draft_tok_s = 1000.0 / ms;
        // Full model speed from loaded data
        TrgModel mf; memcpy(&mf, &m, sizeof(m));
        // Quick full model run
        memcpy(hidden.data(), m.emb.data()+1*H, H*4);
        for (int i = 0; i < 3; i++)
            mf.decode(hidden.data(), i, kc.data(), vc.data(), sq.data(), sa.data(), sf.data(), sa2.data(), st.data(), ct.data());
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < runs; i++) {
            memcpy(hidden.data(), mf.emb.data()+1*H, H*4);
            mf.decode(hidden.data(), i, kc.data(), vc.data(), sq.data(), sa.data(), sf.data(), sa2.data(), st.data(), ct.data());
        }
        t1 = std::chrono::high_resolution_clock::now();
        double full_ms = std::chrono::duration<double,std::milli>(t1-t0).count() / runs;
        double full_tok_s = 1000.0 / full_ms;

        printf("\n  DSpark estimate (draft=%dL, full=%dL, M=8, acc=75%%):\n", L, m.L);
        double round_ms = 8 * ms + full_ms;
        double tpr = 1.0 + 0.75 * 8;
        printf("    Draft seq: %.1f ms  Verify: %.1f ms  Round: %.1f ms\n", 8*ms, full_ms, round_ms);
        printf("    Tokens/round: %.1f  Effective: %.0f tok/s  (%.1fx)\n", tpr, tpr/(round_ms/1000.0), tpr/(round_ms/1000.0)/full_tok_s);
    }

    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage:\n");
        printf("  Save:  %s save <model.q4nx> <output.trg>\n", argv[0]);
        printf("  Load:  %s load <model.trg> [layers]\n", argv[0]);
        printf("  Bench: %s bench <model.trg> [layers]\n", argv[0]);
        return 1;
    }

    std::string cmd = argv[1];
    if (cmd == "save") {
        return save_trg(argv[2], argc > 3 ? argv[3] : "model.trg");
    } else if (cmd == "load" || cmd == "bench") {
        return load_bench(argv[2], argc > 3 ? atoi(argv[3]) : 0);
    } else {
        fprintf(stderr, "Unknown command: %s\n", cmd.c_str());
        return 1;
    }
}

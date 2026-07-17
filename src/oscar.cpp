// src/oscar.cpp — Host-side wrappers for OSCAR INT2 KV cache kernels

#include "rocm_cpp/oscar.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

// -- HIP launcher prototypes (defined in kernels/oscar_quant.hip) ------------
extern "C" void rcpp_oscar_quantize_launch(
    const void* in_fp16, void* out_idx, void* out_scale,
    int seq_len, int num_kv_heads, int HD, int layer_idx,
    const float* R_host, void* stream);

extern "C" int rcpp_oscar_attn_decode_launch(
    const void* Q_dev, const void* K_idx, const void* K_scale,
    const void* V_idx, const void* V_scale,
    void* out_dev,
    int num_q_heads, int num_kv_heads, int HD, int seq_len,
    int layer_idx, float scale,
    const float* R_K_host, const float* R_V_host,
    void* stream);

// Binary format header (must match tools/oscar_calib.cpp)
#pragma pack(push, 1)
typedef struct {
    char magic[8];
    int32_t head_dim;
    int32_t n_layers;
    int32_t n_heads_q;
    int32_t n_heads_kv;
} OscarFileHeader;
#pragma pack(pop)

// ── Load rotation matrices from binary file ────────────────────────────────
extern "C" int oscar_load_rotations(const char* path, OscarRots* rots) {
    if (!path || !rots) return -1;

    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "oscar: cannot open %s\n", path); return -1; }

    // Read header
    OscarFileHeader hdr;
    f.read((char*)&hdr, sizeof(hdr));
    if (memcmp(hdr.magic, "OSCARROT", 8) != 0) {
        fprintf(stderr, "oscar: bad magic in %s\n", path);
        return -1;
    }

    rots->head_dim = hdr.head_dim;
    rots->n_layers = hdr.n_layers;
    rots->n_heads_q = hdr.n_heads_q;
    rots->n_heads_kv = hdr.n_heads_kv;

    size_t n = (size_t)rots->n_layers * rots->head_dim * rots->head_dim * 2;
    rots->data = (float*)malloc(n * sizeof(float));
    if (!rots->data) {
        fprintf(stderr, "oscar: malloc %zu floats failed\n", n);
        return -1;
    }

    f.read((char*)rots->data, n * sizeof(float));
    f.close();
    return 0;
}

extern "C" void oscar_free_rotations(OscarRots* rots) {
    if (rots && rots->data) {
        free(rots->data);
        rots->data = nullptr;
    }
}

// ── Public C API: quantize K ──────────────────────────────────────────────
extern "C" void rcpp_kv_quantize_oscar_k(
    const void* K_fp16_dev, void* K_idx_dev, void* K_scale_dev,
    int seq_len, int num_kv_heads, int HD, int layer_idx,
    const OscarRots* rots, oscar_stream_t stream)
{
    if (!K_fp16_dev || !K_idx_dev || !K_scale_dev || !rots) return;
    if (seq_len <= 0 || num_kv_heads <= 0 || HD <= 0) return;
    if (layer_idx >= rots->n_layers) return;

    const float* Rk = oscar_Rk(rots, layer_idx);
    rcpp_oscar_quantize_launch(K_fp16_dev, K_idx_dev, K_scale_dev,
                               seq_len, num_kv_heads, HD, layer_idx,
                               Rk, stream);
}

extern "C" void rcpp_kv_quantize_oscar_v(
    const void* V_fp16_dev, void* V_idx_dev, void* V_scale_dev,
    int seq_len, int num_kv_heads, int HD, int layer_idx,
    const OscarRots* rots, oscar_stream_t stream)
{
    // Same code path. Separate symbol for future asymmetric handling.
    rcpp_kv_quantize_oscar_k(V_fp16_dev, V_idx_dev, V_scale_dev,
                             seq_len, num_kv_heads, HD, layer_idx,
                             rots, stream);
}

// ── Public C API: attention decode ─────────────────────────────────────────
extern "C" int rcpp_kv_cache_attn_decode_oscar(
    const void* Q_dev, const void* K_idx_dev, const void* K_scale_dev,
    const void* V_idx_dev, const void* V_scale_dev,
    void* out_dev,
    int num_q_heads, int num_kv_heads, int HD,
    int seq_len, int layer_idx,
    const OscarRots* rots, float scale, oscar_stream_t stream)
{
    if (!Q_dev || !K_idx_dev || !V_idx_dev || !out_dev || !rots) return -1;
    if (layer_idx >= rots->n_layers) return -1;

    const float* Rk = oscar_Rk(rots, layer_idx);
    const float* Rv = oscar_Rv(rots, layer_idx);

    return rcpp_oscar_attn_decode_launch(
        Q_dev, K_idx_dev, K_scale_dev,
        V_idx_dev, V_scale_dev, out_dev,
        num_q_heads, num_kv_heads, HD, seq_len,
        layer_idx, scale, Rk, Rv, stream);
}

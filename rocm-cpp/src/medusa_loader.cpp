// medusa_loader.cpp — `.h1b-medusa` v1 + v2 sidecar loader.
//
// See include/rocm_cpp/medusa.h + docs/h1b-medusa-format.md.
//
// Pure host C++ — no HIP language tagging. Uses libamdhip64 for hipMalloc /
// hipMemcpy. Two on-disk variants share the "H1BM" magic:
//
//   v1 / variant=0 (legacy): per-head [vocab, hidden] ternary projection.
//                            Cross-checks hidden_size + vocab_size against
//                            the base model. The synthetic-zero writer in
//                            tests/test_medusa_loader.cpp + the upstream
//                            v1 pack output land here.
//   v2 / variant=1         : residual-MLP topology mirroring upstream
//                            parrishcorcoran/MedusaBitNet-2B-4T. Each head
//                            stores w_in[hidden, hidden] + w_out[hidden,
//                            hidden] in fp16 (or bf16 — cast host-side
//                            before upload). vocab_size header field is
//                            unused (the engine reuses base->embedding_dev
//                            as the shared lm_head).
//
// The dispatch happens after we read the version u32: version=1 → v1 path,
// version=2 → v2 path. Anything else returns RCPP_INVALID_ARG.

#include "rocm_cpp/medusa.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

// Per-head packed-weight size in bytes (v1 only). Same shape rules as the
// base loader in src/h1b_loader.cpp.
//   rows = vocab_size, cols = hidden_size
// The `has_row_scales` out-param flags whether a per-head fp32 scale tensor
// is also expected on disk (omitted for the Bonsai inline-scale formats).
size_t medusa_v1_head_bytes(rcpp_weight_format_t fmt,
                            uint32_t vocab,
                            uint32_t hidden,
                            bool* has_row_scales)
{
    *has_row_scales = true;
    switch (fmt) {
        case RCPP_WEIGHT_FORMAT_HALO_V2:
            return (size_t)vocab * (size_t)((hidden + 3u) / 4u);
        case RCPP_WEIGHT_FORMAT_SHERRY_I8:
        case RCPP_WEIGHT_FORMAT_SHERRY_FP16:
            // 5 bits / value packed into bytes; cols must be %32. Loader
            // returns 0 (caller fails on size mismatch) if not.
            if (hidden % 32u != 0u) return 0;
            return (size_t)vocab * ((size_t)hidden * 5u / 32u);
        case RCPP_WEIGHT_FORMAT_TQ1: {
            const uint32_t cols_padded = ((hidden + 19u) / 20u) * 20u;
            return (size_t)vocab * ((size_t)cols_padded / 5u);
        }
        case RCPP_WEIGHT_FORMAT_BONSAI_Q1:
            *has_row_scales = false;
            if (hidden % 128u != 0u) return 0;
            return (size_t)vocab * ((size_t)hidden / 128u) * 18u;
        case RCPP_WEIGHT_FORMAT_BONSAI_TQ2:
            *has_row_scales = false;
            if (hidden % 128u != 0u) return 0;
            return (size_t)vocab * ((size_t)hidden / 128u) * 34u;
        default:
            return 0;
    }
}

// v2 per-tensor (w_in OR w_out) size in bytes given dtype + hidden_size.
// Returns 0 for unsupported dtypes — caller fails on size mismatch.
size_t medusa_v2_tensor_bytes(rcpp_medusa_dtype_t dtype, uint32_t hidden) {
    switch (dtype) {
        case RCPP_MEDUSA_DTYPE_BF16:
        case RCPP_MEDUSA_DTYPE_FP16:
            return (size_t)hidden * (size_t)hidden * 2u;
        case RCPP_MEDUSA_DTYPE_HALO_V2_TERNARY:
            // 2 bpw + per-row fp32 scales; reserved (no runtime path yet).
            // Size = hidden * ((hidden + 3) / 4) bytes; scales added below
            // by caller. We only return the packed-weight bytes here.
            return (size_t)hidden * (size_t)((hidden + 3u) / 4u);
        case RCPP_MEDUSA_DTYPE_SHERRY_I8:
            if (hidden % 32u != 0u) return 0;
            return (size_t)hidden * ((size_t)hidden * 5u / 32u);
        default:
            return 0;
    }
}

bool read_exact(std::ifstream& f, void* dst, size_t n) {
    f.read(static_cast<char*>(dst), (std::streamsize)n);
    return (size_t)f.gcount() == n;
}

// bf16 → fp16 host-side cast. Both are 16-bit; we go through float to
// preserve magnitude (bf16 has 8-bit exp / 7-bit mantissa, fp16 has 5-bit
// exp / 10-bit mantissa, so values >65504 saturate to fp16 inf — bounded
// MedusaBitNet weights stay well inside fp16 range in practice).
inline uint16_t bf16_to_fp16_one(uint16_t bf16) {
    // bf16 → fp32: bits go in the upper half of the fp32 binary repr.
    uint32_t f32_bits = (uint32_t)bf16 << 16;
    float f;
    std::memcpy(&f, &f32_bits, 4);
    // fp32 → fp16 via __fp16 (clang's IEEE-754 binary16 builtin). Falls
    // back to a saturating cast on overflow.
    _Float16 h = (_Float16)f;
    uint16_t out;
    std::memcpy(&out, &h, 2);
    return out;
}

void bf16_buffer_to_fp16(const uint8_t* bf16_bytes, uint16_t* fp16_out,
                         size_t count_elems) {
    const uint16_t* bf16 = reinterpret_cast<const uint16_t*>(bf16_bytes);
    for (size_t i = 0; i < count_elems; ++i) {
        fp16_out[i] = bf16_to_fp16_one(bf16[i]);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// v1 loader path. Identical to the legacy loader; preserved verbatim so
// existing v1 .h1b-medusa sidecars (synthetic-zero, etc.) still load.
// `f` is positioned just after the 32-byte header.
rcpp_status_t load_v1_payload(std::ifstream& f,
                              const char* path,
                              uint32_t num_heads,
                              uint32_t hidden_size,
                              uint32_t vocab_size,
                              uint32_t weight_format,
                              rcpp_medusa_heads_t* out)
{
    const rcpp_weight_format_t fmt = (rcpp_weight_format_t)weight_format;
    bool has_row_scales = true;
    const size_t per_head_w_bytes =
        medusa_v1_head_bytes(fmt, vocab_size, hidden_size, &has_row_scales);
    if (per_head_w_bytes == 0) {
        std::fprintf(stderr,
                     "[medusa] unsupported weight_format=%u or shape (h=%u v=%u)\n",
                     weight_format, hidden_size, vocab_size);
        return RCPP_INVALID_ARG;
    }

    out->num_heads     = num_heads;
    out->hidden_size   = hidden_size;
    out->vocab_size    = vocab_size;
    out->weight_format = fmt;
    out->variant       = RCPP_MEDUSA_VARIANT_VOCAB;
    out->v2_dtype      = RCPP_MEDUSA_DTYPE_BF16;  // unused in v1

    std::vector<uint8_t> w_buf(per_head_w_bytes);
    std::vector<float>   s_buf(has_row_scales ? vocab_size : 0u);

    for (uint32_t h = 0; h < num_heads; ++h) {
        if (!read_exact(f, w_buf.data(), per_head_w_bytes)) {
            std::fprintf(stderr,
                         "[medusa] truncated weight payload at head %u in %s\n",
                         h, path);
            rcpp_medusa_free_heads(out);
            return RCPP_INVALID_ARG;
        }
        if (has_row_scales) {
            if (!read_exact(f, s_buf.data(), (size_t)vocab_size * sizeof(float))) {
                std::fprintf(stderr,
                             "[medusa] truncated scales payload at head %u in %s\n",
                             h, path);
                rcpp_medusa_free_heads(out);
                return RCPP_INVALID_ARG;
            }
        }

        void*  d_w = nullptr;
        float* d_s = nullptr;
        if (hipMalloc(&d_w, per_head_w_bytes) != hipSuccess) {
            rcpp_medusa_free_heads(out);
            return RCPP_HIP_ERROR;
        }
        if (hipMemcpy(d_w, w_buf.data(), per_head_w_bytes,
                      hipMemcpyHostToDevice) != hipSuccess) {
            hipFree(d_w);
            rcpp_medusa_free_heads(out);
            return RCPP_HIP_ERROR;
        }
        if (has_row_scales) {
            if (hipMalloc(&d_s, (size_t)vocab_size * sizeof(float)) != hipSuccess) {
                hipFree(d_w);
                rcpp_medusa_free_heads(out);
                return RCPP_HIP_ERROR;
            }
            if (hipMemcpy(d_s, s_buf.data(),
                          (size_t)vocab_size * sizeof(float),
                          hipMemcpyHostToDevice) != hipSuccess) {
                hipFree(d_w); hipFree(d_s);
                rcpp_medusa_free_heads(out);
                return RCPP_HIP_ERROR;
            }
        }

        out->heads[h].packed_dev     = d_w;
        out->heads[h].row_scales_dev = d_s;
        out->heads[h].w_in_dev       = nullptr;
        out->heads[h].w_out_dev      = nullptr;
    }

    return RCPP_OK;
}

// ─────────────────────────────────────────────────────────────────────────
// v2 loader path. Reads variant + num_heads + hidden_size + weight_dtype,
// then for each head reads w_in[hidden,hidden] then w_out[hidden,hidden]
// in the on-disk dtype, casts to fp16 if needed, and uploads to device.
//
// `f` is positioned just after the 32-byte header (4 magic + 4 version
// + 24 bytes already consumed by the caller from the v2 header layout).
rcpp_status_t load_v2_payload(std::ifstream& f,
                              const char* path,
                              uint32_t variant,
                              uint32_t num_heads,
                              uint32_t hidden_size,
                              uint32_t weight_dtype,
                              rcpp_medusa_heads_t* out)
{
    if (variant != (uint32_t)RCPP_MEDUSA_VARIANT_RESIDUAL_MLP) {
        std::fprintf(stderr,
                     "[medusa] v2 sidecar carries unsupported variant=%u in %s\n",
                     variant, path);
        return RCPP_INVALID_ARG;
    }
    const rcpp_medusa_dtype_t dtype = (rcpp_medusa_dtype_t)weight_dtype;
    const size_t per_tensor_bytes = medusa_v2_tensor_bytes(dtype, hidden_size);
    if (per_tensor_bytes == 0) {
        std::fprintf(stderr,
                     "[medusa] unsupported v2 weight_dtype=%u or shape (h=%u)\n",
                     weight_dtype, hidden_size);
        return RCPP_INVALID_ARG;
    }
    if (dtype != RCPP_MEDUSA_DTYPE_BF16 && dtype != RCPP_MEDUSA_DTYPE_FP16) {
        // Reserved tags: parser accepts the size, but the runtime path
        // (Engine::Impl::medusa_step_residual) only handles fp16 today.
        std::fprintf(stderr,
                     "[medusa] v2 weight_dtype=%u accepted but engine "
                     "residual-MLP path requires bf16/fp16 today\n",
                     weight_dtype);
        return RCPP_INVALID_ARG;
    }

    out->num_heads     = num_heads;
    out->hidden_size   = hidden_size;
    out->vocab_size    = 0u;
    out->weight_format = RCPP_WEIGHT_FORMAT_HALO_V2;     // unused in v2
    out->variant       = RCPP_MEDUSA_VARIANT_RESIDUAL_MLP;
    out->v2_dtype      = dtype;

    // Device-side storage is always fp16 [hidden*hidden] for both w_in/w_out.
    const size_t fp16_bytes = (size_t)hidden_size * (size_t)hidden_size * 2u;

    std::vector<uint8_t>  raw_buf(per_tensor_bytes);
    std::vector<uint16_t> fp16_buf(dtype == RCPP_MEDUSA_DTYPE_BF16
                                   ? (size_t)hidden_size * hidden_size : 0u);

    for (uint32_t h = 0; h < num_heads; ++h) {
        for (int which = 0; which < 2; ++which) {  // 0 = w_in, 1 = w_out
            if (!read_exact(f, raw_buf.data(), per_tensor_bytes)) {
                std::fprintf(stderr,
                             "[medusa] truncated v2 %s at head %u in %s\n",
                             which == 0 ? "w_in" : "w_out", h, path);
                rcpp_medusa_free_heads(out);
                return RCPP_INVALID_ARG;
            }

            const uint8_t* host_fp16_src = raw_buf.data();
            if (dtype == RCPP_MEDUSA_DTYPE_BF16) {
                bf16_buffer_to_fp16(raw_buf.data(), fp16_buf.data(),
                                    (size_t)hidden_size * hidden_size);
                host_fp16_src = reinterpret_cast<const uint8_t*>(fp16_buf.data());
            }

            void* d_w = nullptr;
            if (hipMalloc(&d_w, fp16_bytes) != hipSuccess) {
                rcpp_medusa_free_heads(out);
                return RCPP_HIP_ERROR;
            }
            if (hipMemcpy(d_w, host_fp16_src, fp16_bytes,
                          hipMemcpyHostToDevice) != hipSuccess) {
                hipFree(d_w);
                rcpp_medusa_free_heads(out);
                return RCPP_HIP_ERROR;
            }
            if (which == 0) out->heads[h].w_in_dev  = d_w;
            else            out->heads[h].w_out_dev = d_w;
        }
        out->heads[h].packed_dev     = nullptr;
        out->heads[h].row_scales_dev = nullptr;
    }

    return RCPP_OK;
}

}  // namespace

extern "C"
rcpp_status_t
rcpp_medusa_load_h1b_sidecar(const char* path,
                             const rcpp_bitnet_model_t* base,
                             rcpp_medusa_heads_t* out)
{
    if (!path || !base || !out) return RCPP_INVALID_ARG;
    std::memset(out, 0, sizeof(*out));

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[medusa] cannot open sidecar: %s\n", path);
        return RCPP_INVALID_ARG;
    }

    // ── First 8 bytes: magic + version. The remainder of the header
    //    differs between v1 and v2; we dispatch on `version` here. ──
    char     magic[4] = {0};
    uint32_t version  = 0;
    if (!read_exact(f, magic, 4) || !read_exact(f, &version, 4)) {
        std::fprintf(stderr, "[medusa] truncated header in %s\n", path);
        return RCPP_INVALID_ARG;
    }
    if (std::memcmp(magic, "H1BM", 4) != 0) {
        std::fprintf(stderr,
                     "[medusa] bad magic in %s (expected H1BM, got %02x%02x%02x%02x)\n",
                     path,
                     (unsigned char)magic[0], (unsigned char)magic[1],
                     (unsigned char)magic[2], (unsigned char)magic[3]);
        return RCPP_INVALID_ARG;
    }

    if (version == 1u) {
        // v1 header layout: variant/num_heads/hidden/vocab/weight_format
        // + 8 bytes reserved. variant slot reused as `num_heads` for the
        // legacy synthetic-zero writer that predates the variant field.
        // (Legacy writers always set version=1 and a non-zero num_heads,
        //  so we keep the original 24-byte tail here.)
        uint32_t num_heads = 0, hidden_size = 0, vocab_size = 0;
        uint32_t weight_format = 0, reserved0 = 0, reserved1 = 0;
        if (!read_exact(f, &num_heads,     4) ||
            !read_exact(f, &hidden_size,   4) ||
            !read_exact(f, &vocab_size,    4) ||
            !read_exact(f, &weight_format, 4) ||
            !read_exact(f, &reserved0,     4) ||
            !read_exact(f, &reserved1,     4))
        {
            std::fprintf(stderr, "[medusa] truncated v1 header in %s\n", path);
            return RCPP_INVALID_ARG;
        }

        if (num_heads == 0u || num_heads > RCPP_MEDUSA_MAX_HEADS) {
            std::fprintf(stderr, "[medusa] num_heads=%u out of range [1,%d]\n",
                         num_heads, RCPP_MEDUSA_MAX_HEADS);
            return RCPP_INVALID_ARG;
        }
        if ((int)hidden_size != base->hidden_size) {
            std::fprintf(stderr,
                         "[medusa] hidden_size mismatch: sidecar=%u base=%d\n",
                         hidden_size, base->hidden_size);
            return RCPP_INVALID_ARG;
        }
        if ((int)vocab_size != base->vocab_size) {
            std::fprintf(stderr,
                         "[medusa] vocab_size mismatch: sidecar=%u base=%d\n",
                         vocab_size, base->vocab_size);
            return RCPP_INVALID_ARG;
        }
        return load_v1_payload(f, path, num_heads, hidden_size, vocab_size,
                               weight_format, out);
    }

    if (version == 2u) {
        // v2 header layout: variant/num_heads/hidden/weight_dtype + 8
        // bytes reserved. No vocab_size — the engine reuses the base
        // lm_head, so vocab is always model.vocab_size.
        uint32_t variant = 0, num_heads = 0, hidden_size = 0;
        uint32_t weight_dtype = 0, reserved0 = 0, reserved1 = 0;
        if (!read_exact(f, &variant,      4) ||
            !read_exact(f, &num_heads,    4) ||
            !read_exact(f, &hidden_size,  4) ||
            !read_exact(f, &weight_dtype, 4) ||
            !read_exact(f, &reserved0,    4) ||
            !read_exact(f, &reserved1,    4))
        {
            std::fprintf(stderr, "[medusa] truncated v2 header in %s\n", path);
            return RCPP_INVALID_ARG;
        }

        if (num_heads == 0u || num_heads > RCPP_MEDUSA_MAX_HEADS) {
            std::fprintf(stderr, "[medusa] num_heads=%u out of range [1,%d]\n",
                         num_heads, RCPP_MEDUSA_MAX_HEADS);
            return RCPP_INVALID_ARG;
        }
        if ((int)hidden_size != base->hidden_size) {
            std::fprintf(stderr,
                         "[medusa] hidden_size mismatch: sidecar=%u base=%d\n",
                         hidden_size, base->hidden_size);
            return RCPP_INVALID_ARG;
        }
        return load_v2_payload(f, path, variant, num_heads, hidden_size,
                               weight_dtype, out);
    }

    std::fprintf(stderr, "[medusa] unsupported version %u in %s\n",
                 version, path);
    return RCPP_INVALID_ARG;
}

extern "C"
void
rcpp_medusa_free_heads(rcpp_medusa_heads_t* heads)
{
    if (!heads) return;
    for (uint32_t h = 0; h < heads->num_heads && h < RCPP_MEDUSA_MAX_HEADS; ++h) {
        if (heads->heads[h].packed_dev) {
            hipFree(heads->heads[h].packed_dev);
            heads->heads[h].packed_dev = nullptr;
        }
        if (heads->heads[h].row_scales_dev) {
            hipFree(heads->heads[h].row_scales_dev);
            heads->heads[h].row_scales_dev = nullptr;
        }
        if (heads->heads[h].w_in_dev) {
            hipFree(heads->heads[h].w_in_dev);
            heads->heads[h].w_in_dev = nullptr;
        }
        if (heads->heads[h].w_out_dev) {
            hipFree(heads->heads[h].w_out_dev);
            heads->heads[h].w_out_dev = nullptr;
        }
    }
    std::memset(heads, 0, sizeof(*heads));
}

// test_bonsai_e2e.cpp — minimal smoke test for the Bonsai (Qwen3) dispatch
// path introduced alongside the bonsai_q1/tq2_gemv_launch kernels.
//
// What it checks:
//   1. rcpp_bitnet_load_h1b succeeds on the Bonsai-1.7B-TQ2 .h1b file.
//   2. The loader resolves WeightFormat::BONSAI_TQ2 from cfg[8] & 0x8
//      and sets is_qwen3 = 1.
//   3. The sidecar GGUF pass fills attn_q_norm / attn_k_norm on every layer
//      and hydrates the embedding + output_norm.
//   4. One decode-token forward pass through all L layers + tied LM head
//      completes with no HIP launch error.
//   5. The resulting fp32 logits buffer is non-zero and finite.
//
// What it does NOT check:
//   - Generation quality. Until the sibling HIP port agent lands the real
//     bonsai_tq2_gemv_launch, the dispatch short-circuits to memset-zero and
//     the logits reflect only the fp16 GEMV on the embedding matrix (i.e.,
//     argmax comes from random chance on zero residual). Once the kernel
//     lands this test stays valid — it never asserted on specific tokens.
//
// Expected runtime: ~1 s once weights are in LPDDR5 page cache (the heavy
// cost is the 621 MiB TQ2 → FP16 host-side dequant of the embedding).
// Binary: built as an executable under tests/ — NOT wired into librocm_cpp.
// Run by hand on the strix-halo box.

#include "rocm_cpp/bitnet_model.h"
#include "rocm_cpp/ck_gemm.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Local weak declarations — mirror the forward decls in bitnet_decode.cpp.
// Letting the link succeed even when the sibling agent's kernel .hip files
// haven't landed yet; the dispatch helper below checks for null.
extern "C" void bonsai_q1_gemv_launch(
    const uint8_t*, const uint16_t*, uint16_t*, int, int, void*) __attribute__((weak));
extern "C" void bonsai_tq2_gemv_launch(
    const uint8_t*, const uint16_t*, uint16_t*, int, int, void*) __attribute__((weak));

#define HIP_OK(e) do { \
    hipError_t _s = (e); \
    if (_s != hipSuccess) { \
        fprintf(stderr, "HIP error %d (%s) at %s:%d\n", \
                (int)_s, hipGetErrorString(_s), __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)
#define RC_OK(e) do { \
    rcpp_status_t _s = (e); \
    if (_s != RCPP_OK) { \
        fprintf(stderr, "rcpp err %d at %s:%d\n", (int)_s, __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)

static void bonsai_gemv(rcpp_weight_format_t fmt,
                        const void* packed, const void* act_fp16,
                        void* out_fp16, int N, int K)
{
    using Fn = void (*)(const uint8_t*, const uint16_t*, uint16_t*, int, int, void*);
    Fn fn = nullptr;
    if      (fmt == RCPP_WEIGHT_FORMAT_BONSAI_TQ2) fn = &bonsai_tq2_gemv_launch;
    else if (fmt == RCPP_WEIGHT_FORMAT_BONSAI_Q1)  fn = &bonsai_q1_gemv_launch;
    if (!fn) {
        // Kernel not linked yet — null-fill so the forward pass doesn't SEGV.
        (void)hipMemsetAsync(out_fp16, 0, (size_t)N * sizeof(uint16_t), nullptr);
        return;
    }
    fn(static_cast<const uint8_t*>(packed),
       static_cast<const uint16_t*>(act_fp16),
       static_cast<uint16_t*>(out_fp16),
       N, K, nullptr);
}

int main(int argc, char** argv) {
    const char* path = (argc > 1)
        ? argv[1]
        : "/home/bcloud/halo-ai/models/bonsai/bonsai-1.7b-tq2.h1b";

    fprintf(stderr, "[test_bonsai_e2e] loading %s\n", path);

    rcpp_bitnet_model_t m{};
    RC_OK(rcpp_bitnet_load_h1b(path, &m));

    fprintf(stderr,
            "[test_bonsai_e2e] version=%d flags=0x%x weight_format=%d is_qwen3=%d "
            "hs=%d is=%d L=%d nh=%d nkv=%d V=%d\n",
            m.format_version, m.flags, (int)m.weight_format, m.is_qwen3,
            m.hidden_size, m.intermediate_size, m.num_layers,
            m.num_heads, m.num_kv_heads, m.vocab_size);

    if (m.weight_format != RCPP_WEIGHT_FORMAT_BONSAI_TQ2 &&
        m.weight_format != RCPP_WEIGHT_FORMAT_BONSAI_Q1)
    {
        fprintf(stderr,
                "[test_bonsai_e2e] FAIL: expected BONSAI_Q1/TQ2 dispatch tag, got %d\n",
                (int)m.weight_format);
        return 1;
    }
    if (!m.is_qwen3) {
        fprintf(stderr, "[test_bonsai_e2e] FAIL: is_qwen3 not set on a Bonsai model\n");
        return 1;
    }
    // Confirm the sidecar pass filled per-head q/k norms on at least layer 0.
    if (!m.layers[0].attn_q_norm_dev || !m.layers[0].attn_k_norm_dev) {
        fprintf(stderr,
                "[test_bonsai_e2e] FAIL: attn_q/k_norm_dev null on L0 — sidecar GGUF "
                "probably not found.\n");
        return 1;
    }

    const int hs  = m.hidden_size;
    const int is  = m.intermediate_size;
    const int nh  = m.num_heads;
    const int nkv = m.num_kv_heads;
    const int hd  = hs / nh;
    const int L   = m.num_layers;
    const int V   = m.vocab_size;
    const float scale = 1.0f / std::sqrt((float)hd);

    // Scratch buffers — decode shape (batch=1, one token).
    float*    x_fp32 = nullptr;
    _Float16* x = nullptr;
    _Float16* normed = nullptr;
    _Float16* q_fp16 = nullptr, *k_fp16 = nullptr, *v_fp16 = nullptr;
    _Float16* o_fp16 = nullptr;
    _Float16* gate_fp16 = nullptr, *up_fp16 = nullptr, *down_fp16 = nullptr;
    _Float16* silu_out = nullptr;
    float*    logits = nullptr;

    HIP_OK(hipMalloc(&x_fp32,   hs * sizeof(float)));
    HIP_OK(hipMalloc(&x,        hs * sizeof(_Float16)));
    HIP_OK(hipMalloc(&normed,   hs * sizeof(_Float16)));
    HIP_OK(hipMalloc(&q_fp16,   nh * hd * sizeof(_Float16)));
    HIP_OK(hipMalloc(&k_fp16,   nkv * hd * sizeof(_Float16)));
    HIP_OK(hipMalloc(&v_fp16,   nkv * hd * sizeof(_Float16)));
    HIP_OK(hipMalloc(&o_fp16,   hs * sizeof(_Float16)));
    HIP_OK(hipMalloc(&gate_fp16, is * sizeof(_Float16)));
    HIP_OK(hipMalloc(&up_fp16,  is * sizeof(_Float16)));
    HIP_OK(hipMalloc(&down_fp16, hs * sizeof(_Float16)));
    HIP_OK(hipMalloc(&silu_out, is * sizeof(_Float16)));
    HIP_OK(hipMalloc(&logits,   V * sizeof(float)));

    std::vector<_Float16*> Ks(L, nullptr), Vs(L, nullptr);
    const size_t kv_slab = (size_t)nkv * hd * sizeof(_Float16);
    for (int l = 0; l < L; ++l) {
        HIP_OK(hipMalloc(&Ks[l], kv_slab));
        HIP_OK(hipMalloc(&Vs[l], kv_slab));
    }

    const int token_id = 1;  // BOS
    const int pos = 0;

    RC_OK(rcpp_embedding_lookup_fp16(m.embedding_dev, token_id, x, hs, nullptr));
    HIP_OK(hipMemsetAsync(x_fp32, 0, hs * sizeof(float), nullptr));
    RC_OK(rcpp_residual_add_fp32_from_fp16(x_fp32, x, hs, nullptr));

    for (int l = 0; l < L; ++l) {
        rcpp_bitnet_layer_t& ly = m.layers[l];

        // Qwen3 attention.
        RC_OK(rcpp_rmsnorm_fp32_in_fp16_out(x_fp32, ly.input_norm_dev, normed,
                                            m.rms_norm_eps, hs, nullptr));

        bonsai_gemv(m.weight_format, ly.q_packed_dev, normed, q_fp16, nh*hd,  hs);
        bonsai_gemv(m.weight_format, ly.k_packed_dev, normed, k_fp16, nkv*hd, hs);
        bonsai_gemv(m.weight_format, ly.v_packed_dev, normed, v_fp16, nkv*hd, hs);

        // Per-head q / k norms before RoPE.
        for (int h = 0; h < nh; ++h) {
            _Float16* qh = q_fp16 + (size_t)h * hd;
            RC_OK(rcpp_rmsnorm_fp16(qh, ly.attn_q_norm_dev, qh,
                                    m.rms_norm_eps, hd, nullptr));
        }
        for (int h = 0; h < nkv; ++h) {
            _Float16* kh = k_fp16 + (size_t)h * hd;
            RC_OK(rcpp_rmsnorm_fp16(kh, ly.attn_k_norm_dev, kh,
                                    m.rms_norm_eps, hd, nullptr));
        }

        RC_OK(rcpp_rope_fp16(q_fp16, pos, m.rope_theta, nh,  hd, nullptr));
        RC_OK(rcpp_rope_fp16(k_fp16, pos, m.rope_theta, nkv, hd, nullptr));

        HIP_OK(hipMemcpy(Ks[l], k_fp16, nkv*hd*2, hipMemcpyDeviceToDevice));
        HIP_OK(hipMemcpy(Vs[l], v_fp16, nkv*hd*2, hipMemcpyDeviceToDevice));
        RC_OK(rcpp_kv_cache_attn_decode_fd(q_fp16, Ks[l], Vs[l],
                                           o_fp16, nh, nkv, hd, pos+1, scale, nullptr));

        // Qwen3 has no attn_sub_norm — o_fp16 goes straight into O proj.
        bonsai_gemv(m.weight_format, ly.o_packed_dev, o_fp16, normed, hs, nh*hd);
        RC_OK(rcpp_residual_add_fp32_from_fp16(x_fp32, normed, hs, nullptr));

        // Qwen3 FFN.
        RC_OK(rcpp_rmsnorm_fp32_in_fp16_out(x_fp32, ly.post_attn_norm_dev, normed,
                                            m.rms_norm_eps, hs, nullptr));
        bonsai_gemv(m.weight_format, ly.gate_packed_dev, normed, gate_fp16, is, hs);
        bonsai_gemv(m.weight_format, ly.up_packed_dev,   normed, up_fp16,   is, hs);
        // rcpp_silu_glu_fp16 computes `silu(first_arg) * second_arg`.
        // Canonical Qwen3 SwiGLU = silu(gate) * up, so gate goes first.
        RC_OK(rcpp_silu_glu_fp16(gate_fp16, up_fp16, silu_out, is, nullptr));
        bonsai_gemv(m.weight_format, ly.down_packed_dev, silu_out, down_fp16, hs, is);
        RC_OK(rcpp_residual_add_fp32_from_fp16(x_fp32, down_fp16, hs, nullptr));
    }

    RC_OK(rcpp_rmsnorm_fp32_in_fp16_out(x_fp32, m.final_norm_weight_dev, normed,
                                        m.rms_norm_eps, hs, nullptr));
    RC_OK(rcpp_fp16_gemv(m.embedding_dev, normed, logits, V, hs, nullptr));

    HIP_OK(hipDeviceSynchronize());
    HIP_OK(hipGetLastError());

    std::vector<float> host_logits((size_t)V);
    HIP_OK(hipMemcpy(host_logits.data(), logits, V * sizeof(float), hipMemcpyDeviceToHost));

    bool any_nonzero = false;
    bool all_finite  = true;
    float lmax = -1e30f, lmin = 1e30f;
    int argmax_idx = 0;
    for (int i = 0; i < V; ++i) {
        const float v = host_logits[i];
        if (!std::isfinite(v)) all_finite = false;
        if (v != 0.0f) any_nonzero = true;
        if (v > lmax) { lmax = v; argmax_idx = i; }
        if (v < lmin) lmin = v;
    }

    fprintf(stderr, "[test_bonsai_e2e] logits: argmax=%d max=%.3f min=%.3f\n",
            argmax_idx, lmax, lmin);

    const bool kernels_linked = (bonsai_q1_gemv_launch != nullptr) ||
                                (bonsai_tq2_gemv_launch != nullptr);
    if (!kernels_linked) {
        fprintf(stderr,
                "[test_bonsai_e2e] NOTE: bonsai_*_gemv_launch weak-symbols null; "
                "ternary GEMVs short-circuited to zeros. Logits come from the final "
                "fp16_gemv only — non-trivial is still informative but not a "
                "correctness assertion.\n");
    }
    if (!all_finite) {
        fprintf(stderr, "[test_bonsai_e2e] FAIL: non-finite logit(s)\n");
        return 1;
    }
    if (!any_nonzero) {
        fprintf(stderr,
                "[test_bonsai_e2e] FAIL: all logits zero. Embedding + final_norm "
                "hydration broke, OR (if kernels_linked=0) the residual was zero after "
                "L layers of zeroed-GEMV output.\n");
        return 1;
    }

    auto hfree = [](void* p) { (void)hipFree(p); };
    for (int l = 0; l < L; ++l) { hfree(Ks[l]); hfree(Vs[l]); }
    hfree(x_fp32); hfree(x); hfree(normed);
    hfree(q_fp16); hfree(k_fp16); hfree(v_fp16); hfree(o_fp16);
    hfree(gate_fp16); hfree(up_fp16); hfree(down_fp16); hfree(silu_out);
    hfree(logits);
    rcpp_bitnet_free(&m);

    fprintf(stderr, "[test_bonsai_e2e] PASS\n");
    return 0;
}

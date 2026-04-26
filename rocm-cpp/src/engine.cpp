// rocm_cpp::Engine — implementation. Refactor of tools/bitnet_decode.cpp's
// main() into a reusable C++ class. All HIP / rcpp_* calls preserved verbatim
// from the CLI; the only changes are state ownership (now in Engine::Impl)
// and error handling (rcpp_status_t / hipError_t both converted to
// std::runtime_error at function boundaries).
//
// Same dispatch logic, same KV cache layout (fp16 / int8+scale / pq3), same
// sampler chain (greedy / temp + top-k + top-p + rep-penalty). Same Bonsai
// short-circuit, same Qwen3 vs BitNet attention preamble + FFN activation.
// If you're tracking down a numerical regression vs bitnet_decode, this file
// is the only diff that's reasonable to suspect — and the Bonsai dispatch +
// forward_token + sampler functions below should be byte-identical to the
// pre-refactor lambdas.

#include "rocm_cpp/engine.h"

#include "rocm_cpp/bitnet_model.h"
#include "rocm_cpp/ck_gemm.h"
#include "rocm_cpp/kv_rotorquant.h"
#include "rocm_cpp/medusa.h"
#include "rocm_cpp/sherry.h"
#include "rocm_cpp/tokenizer.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Bonsai weak symbols — match bitnet_decode.cpp. weak attribute lets us link
// even if the kernels haven't been compiled in; a null fn pointer triggers a
// zero-output warning.
extern "C" void bonsai_q1_gemv_launch(
    const uint8_t* packed_weights,
    const uint16_t* act_fp16,
    uint16_t* out_fp16,
    int N_out, int K_in,
    void* stream) __attribute__((weak));
extern "C" void bonsai_tq2_gemv_launch(
    const uint8_t* packed_weights,
    const uint16_t* act_fp16,
    uint16_t* out_fp16,
    int N_out, int K_in,
    void* stream) __attribute__((weak));

namespace rocm_cpp {
namespace {

// Arch-aware special-token text markers. The byte-level BPE encoder embedded
// in rcpp_tokenizer can re-emit the multi-byte form of a special token when
// the model produces it as plain bytes (e.g. Qwen3 will render id 151645 as
// the literal "<|im_end|>" when decoded through the BPE merges). The decode
// path doesn't suppress these because the engine doesn't own the tokenizer's
// added-token table — we strip them here on the way out.
//
// The list MUST be ordered with the longest token first so a partial-prefix
// suffix-hold check (used by streaming) doesn't prematurely emit bytes that
// could complete a longer marker. Today every entry is the same length (10-13
// bytes) so order is mostly cosmetic; if a one-character marker is ever added,
// keep the long-first ordering.
inline const std::vector<std::string>& special_marker_table(Engine::Arch arch) {
    static const std::vector<std::string> bitnet = {
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|eot_id|>",
    };
    static const std::vector<std::string> qwen3 = {
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
    };
    return arch == Engine::Arch::Qwen3 ? qwen3 : bitnet;
}

// Remove EVERY occurrence of every marker for `arch` from `text`.
inline void strip_specials_inplace(std::string& text, Engine::Arch arch) {
    const auto& table = special_marker_table(arch);
    for (const auto& tok : table) {
        if (tok.empty()) continue;
        for (size_t pos = text.find(tok);
             pos != std::string::npos;
             pos = text.find(tok, pos)) {
            text.erase(pos, tok.size());
        }
    }
}

// Streaming-safe filter. Maintains a held buffer of trailing bytes that
// could be the prefix of a marker. Each call:
//   1. Append `delta` to `held`.
//   2. Strip any *complete* marker occurrence from `held` (in-place).
//   3. Compute the longest suffix of `held` that is a strict prefix of any
//      marker, and hold that many bytes back. Emit everything before that.
//
// `flush=true` (end-of-stream) bypasses step 3 — any remaining held bytes
// are returned (after one last complete-marker strip), since no further
// input can complete a marker.
//
// Thread-unsafe; one instance per generate() call (lives on the stack).
class StreamingMarkerFilter {
public:
    explicit StreamingMarkerFilter(Engine::Arch arch)
        : table_(&special_marker_table(arch)) {
        // Find longest marker — bound on how many trailing bytes we'd ever
        // need to hold back. Caps the worst-case held buffer growth.
        for (const auto& s : *table_) {
            if (s.size() > max_marker_len_) max_marker_len_ = s.size();
        }
    }

    // Append `delta`, return any bytes that are now safe to emit.
    std::string feed(const std::string& delta) {
        held_.append(delta);
        strip_complete_markers();

        // Find the longest suffix of held_ that is a strict prefix of any
        // marker. Bytes beyond that suffix are guaranteed safe to emit —
        // no future input can transform them into a marker.
        size_t hold = longest_marker_prefix_suffix();
        if (hold >= held_.size()) {
            // Whole buffer is a potential prefix; emit nothing.
            return {};
        }
        std::string out = held_.substr(0, held_.size() - hold);
        held_.erase(0, held_.size() - hold);
        return out;
    }

    // End-of-stream: emit whatever's left after a final complete-marker
    // strip. No more input is coming so a held prefix can never complete
    // into a real marker.
    std::string flush() {
        strip_complete_markers();
        std::string out = std::move(held_);
        held_.clear();
        return out;
    }

private:
    void strip_complete_markers() {
        for (const auto& tok : *table_) {
            if (tok.empty()) continue;
            for (size_t pos = held_.find(tok);
                 pos != std::string::npos;
                 pos = held_.find(tok, pos)) {
                held_.erase(pos, tok.size());
            }
        }
    }

    // For each marker `m` and each k in [1, min(|m|-1, |held_|)], check if
    // held_'s last k bytes equal m's first k bytes. Return the largest such k
    // across all markers; 0 if none.
    size_t longest_marker_prefix_suffix() const {
        size_t best = 0;
        for (const auto& m : *table_) {
            if (m.empty()) continue;
            const size_t kmax = std::min(m.size() - 1, held_.size());
            // k must exceed current best to matter.
            for (size_t k = kmax; k > best; --k) {
                if (std::memcmp(held_.data() + (held_.size() - k),
                                m.data(), k) == 0) {
                    best = k;
                    break;
                }
            }
        }
        return best;
    }

    const std::vector<std::string>* table_;
    std::string held_;
    size_t max_marker_len_ = 0;
};

#define RC_CHECK(expr)                                                         \
    do {                                                                       \
        rcpp_status_t _s = (expr);                                             \
        if (_s != RCPP_OK) {                                                   \
            char buf[256];                                                     \
            std::snprintf(buf, sizeof(buf),                                    \
                          "rocm_cpp::Engine: rcpp error %d at %s:%d",          \
                          (int)_s, __FILE__, __LINE__);                        \
            throw std::runtime_error(buf);                                     \
        }                                                                      \
    } while (0)

#define HIP_CHECK(expr)                                                        \
    do {                                                                       \
        hipError_t _e = (expr);                                                \
        if (_e != hipSuccess) {                                                \
            char buf[256];                                                     \
            std::snprintf(buf, sizeof(buf),                                    \
                          "rocm_cpp::Engine: HIP error %d (%s) at %s:%d",      \
                          (int)_e, hipGetErrorString(_e), __FILE__, __LINE__); \
            throw std::runtime_error(buf);                                     \
        }                                                                      \
    } while (0)

rcpp_status_t bonsai_gemv_dispatch(rcpp_weight_format_t fmt,
                                   const void* packed_weights_dev,
                                   const void* act_fp16_dev,
                                   void* out_fp16_dev,
                                   int N, int K,
                                   hipStream_t stream) {
    using Fn = void (*)(const uint8_t*, const uint16_t*, uint16_t*, int, int, void*);
    Fn fn = nullptr;
    if (fmt == RCPP_WEIGHT_FORMAT_BONSAI_TQ2)
        fn = &bonsai_tq2_gemv_launch;
    else if (fmt == RCPP_WEIGHT_FORMAT_BONSAI_Q1)
        fn = &bonsai_q1_gemv_launch;
    if (!fn) {
        static bool once = false;
        if (!once) {
            std::fprintf(stderr,
                         "[rocm_cpp::Engine] bonsai_*_gemv_launch not linked — "
                         "HIP kernel pass has not landed yet. Zeroing output.\n");
            once = true;
        }
        (void)hipMemsetAsync(out_fp16_dev, 0, (size_t)N * sizeof(uint16_t), stream);
        return RCPP_OK;
    }
    fn(static_cast<const uint8_t*>(packed_weights_dev),
       static_cast<const uint16_t*>(act_fp16_dev),
       static_cast<uint16_t*>(out_fp16_dev),
       N, K, static_cast<void*>(stream));
    return RCPP_OK;
}

// Function pointer to the int8-act dispatcher (HALO_V2 / SHERRY / TQ1).
//
// Round-5 engine-wire (2026-04-25): switched to the device-scalar overloads
// (rcpp_ternary_gemv_*_f16_devscale). The kernel reads x_scale from a
// device pointer, so we no longer need to hipMemcpy-D2H the freshly
// quantized scale before each launch — drops ~7 stalls/layer × 28 layers
// = ~200 sync round-trips per token.
using TernaryGemvI8Fn = rcpp_status_t (*)(
    const void*, const void*, const float*, const void*, void*, int, int, void*);

}  // namespace

struct Engine::Impl {
    Config cfg;

    // Loaded state
    bool loaded = false;
    rcpp_bitnet_model_t model{};
    rcpp_tokenizer_t* tokenizer = nullptr;

    // Cached / derived from model
    int hs = 0, is_ = 0, nh = 0, nkv = 0, hd = 0, L = 0, V = 0;
    int hs_k = 0, is_k = 0;
    int k_pad_unit = 1;
    int max_len = 0;
    bool is_bonsai_q1 = false, is_bonsai_tq2 = false, is_bonsai = false;
    bool is_qwen3 = false;
    float scale = 0.0f;
    TernaryGemvI8Fn ternary_gemv_i8 = nullptr;

    // Device scratch (allocated in load(), freed in unload())
    float*    x_fp32 = nullptr;
    _Float16* x = nullptr;
    _Float16* normed = nullptr;
    _Float16* x_i8_scratch_fp16 = nullptr;
    int8_t*   x_i8 = nullptr;
    float*    x_scale_dev = nullptr;
    float*    q_raw = nullptr;
    float*    k_raw = nullptr;
    float*    v_raw = nullptr;
    float*    o_raw = nullptr;
    float*    gate_raw = nullptr;
    float*    up_raw = nullptr;
    float*    down_raw = nullptr;
    _Float16* q_fp16 = nullptr;
    _Float16* k_fp16 = nullptr;
    _Float16* v_fp16 = nullptr;
    _Float16* o_fp16 = nullptr;
    _Float16* gate_fp16 = nullptr;
    _Float16* up_fp16 = nullptr;
    _Float16* down_fp16 = nullptr;
    _Float16* silu_out = nullptr;
    int8_t*   silu_i8 = nullptr;
    float*    silu_scale_dev = nullptr;
    float*    logits = nullptr;
    int*      next_tok_dev = nullptr;

    // KV caches (one slot per layer; the right vector is populated based on
    // cfg.kv_int8 / cfg.kv_rotor)
    std::vector<_Float16*> K_caches, V_caches;
    std::vector<int8_t*>   K_caches_i8, V_caches_i8;
    std::vector<_Float16*> K_scales, V_scales;
    std::vector<uint8_t*>  K_caches_pq3, V_caches_pq3;

    // Sampler scratch (host-side)
    std::vector<float> logits_host;
    std::vector<int>   sampler_history;

    // Streaming detokenization scratch
    std::vector<char> tail_buf;
    std::vector<char> stream_buf;

    // ── Medusa speculative-decoding state (opt-in) ──
    // attach_medusa() populates this from a `.h1b-medusa` v1 OR v2 sidecar
    // after the base load() succeeds; unload() releases the device buffers.
    bool                medusa_attached = false;
    rcpp_medusa_heads_t medusa{};

    // v2 (residual-MLP) device-scratch buffers — allocated lazily by
    // attach_medusa() when the sidecar carries variant=RESIDUAL_MLP.
    // Layout (one chain step):
    //   tmp_f32   = w_in_k  @ h_in         (fp32 [hidden] — rcpp_fp16_gemv
    //                                       outputs fp32 in this codebase)
    //   tmp_in    = fp16(tmp_f32)          (fp16 [hidden])
    //   silu_act  = SiLU(tmp_in)           (fp16 [hidden] — via silu_glu_fp16
    //                                       with all-ones gate buffer)
    //   delta_f32 = w_out_k @ silu_act     (fp32 [hidden])
    //   delta     = fp16(delta_f32)        (fp16 [hidden])
    //   h_out     = h_in + delta           (fp16 [hidden])
    //   logits    = embedding_dev @ h_out  (fp32 [vocab] — shared lm_head)
    _Float16* medusa_v2_h_in     = nullptr;  // fp16 [hidden] — chained input
    _Float16* medusa_v2_h_out    = nullptr;  // fp16 [hidden] — residual output
    float*    medusa_v2_tmp_f32  = nullptr;  // fp32 [hidden] — GEMV output A
    _Float16* medusa_v2_tmp_in   = nullptr;  // fp16 [hidden] — fp16(tmp_f32)
    _Float16* medusa_v2_silu_act = nullptr;  // fp16 [hidden] — SiLU(tmp_in)
    float*    medusa_v2_delta_f32 = nullptr; // fp32 [hidden] — GEMV output B
    _Float16* medusa_v2_delta    = nullptr;  // fp16 [hidden] — fp16(delta_f32)
    _Float16* medusa_v2_ones     = nullptr;  // fp16 [hidden] — all-ones gate
                                             // for plain SiLU via silu_glu
    float*    medusa_v2_logits_f32 = nullptr;// fp32 [V] — head lm_head output

    explicit Impl(const Config& c) : cfg(c) {
        // Env-var overrides. Lemond's bitnet_server.cpp constructs Engine
        // with default Config, so the only way to flip kv_int8 / kv_rotor
        // for a serving deployment without rebuilding lemond is via env.
        if (const char* p = std::getenv("HALO_KV_INT8");
            p && (p[0] == '1' || p[0] == 't' || p[0] == 'T')) {
            cfg.kv_int8 = true;
        }
        if (const char* p = std::getenv("HALO_KV_ROTOR");
            p && (p[0] == '1' || p[0] == 't' || p[0] == 'T')) {
            cfg.kv_rotor = true;
        }
        if (cfg.kv_int8 && cfg.kv_rotor) {
            throw std::runtime_error(
                "rocm_cpp::Engine: kv_int8 and kv_rotor are mutually exclusive");
        }
    }

    ~Impl() { unload(); }

    static std::string derive_tok_path(const std::string& h1b) {
        std::string out = h1b;
        const std::string ext = ".h1b";
        if (out.size() >= ext.size() &&
            out.compare(out.size() - ext.size(), ext.size(), ext) == 0) {
            out.replace(out.size() - ext.size(), ext.size(), ".htok");
        } else {
            out += ".htok";
        }
        return out;
    }

    void load(const std::string& h1b_path, const std::string& tok_path_in) {
        if (loaded) {
            throw std::runtime_error(
                "rocm_cpp::Engine::load: already loaded — call unload() first");
        }
        const std::string tok_path =
            tok_path_in.empty() ? derive_tok_path(h1b_path) : tok_path_in;

        if (rcpp_bitnet_load_h1b(h1b_path.c_str(), &model) != RCPP_OK) {
            throw std::runtime_error("rocm_cpp::Engine::load: failed to load " +
                                     h1b_path);
        }

        // Tokenizer is optional at load — if it fails, we run without
        // detokenization. on_token will not fire, generate() returns the
        // empty text. This mirrors bitnet_decode's --text fallback.
        if (rcpp_tokenizer_load(tok_path.c_str(), &tokenizer) != RCPP_OK) {
            std::fprintf(stderr,
                         "[rocm_cpp::Engine] WARN: cannot load tokenizer %s — "
                         "generate() will return empty text.\n",
                         tok_path.c_str());
            tokenizer = nullptr;
        }

        // Resolve format / arch + cache shape constants
        is_bonsai_q1  = (model.weight_format == RCPP_WEIGHT_FORMAT_BONSAI_Q1);
        is_bonsai_tq2 = (model.weight_format == RCPP_WEIGHT_FORMAT_BONSAI_TQ2);
        is_bonsai     = is_bonsai_q1 || is_bonsai_tq2;
        is_qwen3      = (model.arch == RCPP_ARCH_QWEN3);
        std::fprintf(stderr,
                     "[rocm_cpp::Engine] arch=%s weight_format=%d\n",
                     is_qwen3 ? "qwen3" : "bitnet", (int)model.weight_format);

        // Round-5: dispatch to the *_devscale entries — kernel reads x_scale
        // from device memory, eliminating per-launch D→H sync stalls.
        ternary_gemv_i8 =
            (model.weight_format == RCPP_WEIGHT_FORMAT_TQ1)
                ? rcpp_ternary_gemv_tq1_halo_f16_devscale
            : (model.weight_format == RCPP_WEIGHT_FORMAT_SHERRY_I8 ||
               model.weight_format == RCPP_WEIGHT_FORMAT_SHERRY_FP16)
                ? rcpp_ternary_gemv_sherry_f16_devscale
                : rcpp_ternary_gemv_halo_f16_devscale;

        k_pad_unit = (model.weight_format == RCPP_WEIGHT_FORMAT_TQ1) ? 20 : 1;
        hs   = model.hidden_size;
        is_  = model.intermediate_size;
        nh   = model.num_heads;
        nkv  = model.num_kv_heads;
        hd   = hs / nh;
        L    = model.num_layers;
        V    = model.vocab_size;
        hs_k = ((hs + k_pad_unit - 1) / k_pad_unit) * k_pad_unit;
        is_k = ((is_ + k_pad_unit - 1) / k_pad_unit) * k_pad_unit;
        scale = 1.0f / std::sqrt((float)hd);

        max_len = cfg.max_context > 0 ? cfg.max_context : 4096;

        // Scratch device buffers — same as bitnet_decode main() ----
        HIP_CHECK(hipMalloc(&x_fp32, hs * 4));
        HIP_CHECK(hipMalloc(&x, hs * 2));
        HIP_CHECK(hipMalloc(&normed, hs * 2));
        HIP_CHECK(hipMalloc(&x_i8_scratch_fp16, hs * 2));
        HIP_CHECK(hipMalloc(&x_i8, hs_k));
        HIP_CHECK(hipMemsetAsync(x_i8, 0, hs_k, nullptr));
        HIP_CHECK(hipMalloc(&x_scale_dev, 4));
        HIP_CHECK(hipMalloc(&q_raw, nh * hd * 4));
        HIP_CHECK(hipMalloc(&k_raw, nkv * hd * 4));
        HIP_CHECK(hipMalloc(&v_raw, nkv * hd * 4));
        HIP_CHECK(hipMalloc(&q_fp16, nh * hd * 2));
        HIP_CHECK(hipMalloc(&k_fp16, nkv * hd * 2));
        HIP_CHECK(hipMalloc(&v_fp16, nkv * hd * 2));
        HIP_CHECK(hipMalloc(&o_raw, hs * 4));
        HIP_CHECK(hipMalloc(&o_fp16, hs * 2));
        HIP_CHECK(hipMalloc(&gate_raw, is_ * 4));
        HIP_CHECK(hipMalloc(&up_raw, is_ * 4));
        HIP_CHECK(hipMalloc(&down_raw, hs * 4));
        HIP_CHECK(hipMalloc(&gate_fp16, is_ * 2));
        HIP_CHECK(hipMalloc(&up_fp16, is_ * 2));
        HIP_CHECK(hipMalloc(&down_fp16, hs * 2));
        HIP_CHECK(hipMalloc(&silu_out, is_ * 2));
        HIP_CHECK(hipMalloc(&silu_i8, is_k));
        HIP_CHECK(hipMemsetAsync(silu_i8, 0, is_k, nullptr));
        HIP_CHECK(hipMalloc(&silu_scale_dev, 4));
        HIP_CHECK(hipMalloc(&logits, V * 4));
        HIP_CHECK(hipMalloc(&next_tok_dev, 4));

        // KV caches per layer
        K_caches.assign(L, nullptr);
        V_caches.assign(L, nullptr);
        K_caches_i8.assign(L, nullptr);
        V_caches_i8.assign(L, nullptr);
        K_scales.assign(L, nullptr);
        V_scales.assign(L, nullptr);
        K_caches_pq3.assign(L, nullptr);
        V_caches_pq3.assign(L, nullptr);

        const size_t kv_size     = (size_t)max_len * nkv * hd * sizeof(_Float16);
        const size_t kv_size_i8  = (size_t)max_len * nkv * hd * sizeof(int8_t);
        const size_t sc_size     = (size_t)max_len * nkv * sizeof(_Float16);
        const size_t kv_size_pq3 = (size_t)max_len * nkv * ((size_t)hd * 3 / 8);
        if (cfg.kv_rotor && (hd & 7) != 0) {
            throw std::runtime_error(
                "rocm_cpp::Engine::load: kv_rotor requires head_dim % 8 == 0");
        }
        // ROCm hot-path win (2026-04-26 docs walk): flip KV-cache buffers to
        // coarse-grain. On Strix Halo unified memory the iGPU normally
        // maintains per-cacheline coherence with the CPU on hipMalloc'd
        // device memory; coarse-grain skips that snoop traffic. KV cache is
        // device-local during decode (CPU never touches it between launches)
        // so this is safe and a pure win on LPDDR5x bandwidth.
        int kv_dev_id = 0;
        HIP_CHECK(hipGetDevice(&kv_dev_id));
        auto kv_advise_coarse = [&](void* p, size_t n) {
            // Best-effort: not all driver builds honor this on plain
            // hipMalloc; ignore failures rather than aborting the load.
            (void)hipMemAdvise(p, n, hipMemAdviseSetCoarseGrain, kv_dev_id);
        };
        for (int l = 0; l < L; ++l) {
            if (cfg.kv_int8) {
                HIP_CHECK(hipMalloc(&K_caches_i8[l], kv_size_i8));
                HIP_CHECK(hipMalloc(&V_caches_i8[l], kv_size_i8));
                HIP_CHECK(hipMalloc(&K_scales[l], sc_size));
                HIP_CHECK(hipMalloc(&V_scales[l], sc_size));
                kv_advise_coarse(K_caches_i8[l], kv_size_i8);
                kv_advise_coarse(V_caches_i8[l], kv_size_i8);
                kv_advise_coarse(K_scales[l],    sc_size);
                kv_advise_coarse(V_scales[l],    sc_size);
            } else if (cfg.kv_rotor) {
                HIP_CHECK(hipMalloc(&K_caches_pq3[l], kv_size_pq3));
                HIP_CHECK(hipMalloc(&V_caches_pq3[l], kv_size_pq3));
                kv_advise_coarse(K_caches_pq3[l], kv_size_pq3);
                kv_advise_coarse(V_caches_pq3[l], kv_size_pq3);
            } else {
                HIP_CHECK(hipMalloc(&K_caches[l], kv_size));
                HIP_CHECK(hipMalloc(&V_caches[l], kv_size));
                kv_advise_coarse(K_caches[l], kv_size);
                kv_advise_coarse(V_caches[l], kv_size);
            }
        }

        logits_host.assign(V, 0.0f);
        tail_buf.assign(8192, 0);
        stream_buf.assign(16 * 1024, 0);

        loaded = true;
    }

    void unload() {
        if (!loaded) return;
        // Best-effort cleanup; if anything fails we still want the rest freed.
        for (int l = 0; l < L; ++l) {
            if (cfg.kv_int8) {
                if (K_caches_i8[l]) hipFree(K_caches_i8[l]);
                if (V_caches_i8[l]) hipFree(V_caches_i8[l]);
                if (K_scales[l])    hipFree(K_scales[l]);
                if (V_scales[l])    hipFree(V_scales[l]);
            } else if (cfg.kv_rotor) {
                if (K_caches_pq3[l]) hipFree(K_caches_pq3[l]);
                if (V_caches_pq3[l]) hipFree(V_caches_pq3[l]);
            } else {
                if (K_caches[l]) hipFree(K_caches[l]);
                if (V_caches[l]) hipFree(V_caches[l]);
            }
        }
        K_caches.clear();    V_caches.clear();
        K_caches_i8.clear(); V_caches_i8.clear();
        K_scales.clear();    V_scales.clear();
        K_caches_pq3.clear(); V_caches_pq3.clear();

        auto F = [](auto*& p) { if (p) { hipFree(p); p = nullptr; } };
        F(x_fp32); F(x); F(normed); F(x_i8_scratch_fp16); F(x_i8); F(x_scale_dev);
        F(q_raw); F(k_raw); F(v_raw); F(o_raw); F(gate_raw); F(up_raw); F(down_raw);
        F(q_fp16); F(k_fp16); F(v_fp16); F(o_fp16);
        F(gate_fp16); F(up_fp16); F(down_fp16);
        F(silu_out); F(silu_i8); F(silu_scale_dev);
        F(logits); F(next_tok_dev);

        if (medusa_attached) {
            rcpp_medusa_free_heads(&medusa);
            medusa_attached = false;
        }
        std::memset(&medusa, 0, sizeof(medusa));

        F(medusa_v2_h_in);
        F(medusa_v2_h_out);
        F(medusa_v2_tmp_f32);
        F(medusa_v2_tmp_in);
        F(medusa_v2_silu_act);
        F(medusa_v2_delta_f32);
        F(medusa_v2_delta);
        F(medusa_v2_ones);
        F(medusa_v2_logits_f32);

        if (tokenizer) { rcpp_tokenizer_free(tokenizer); tokenizer = nullptr; }
        rcpp_bitnet_free(&model);
        std::memset(&model, 0, sizeof(model));
        loaded = false;
    }

    // ---- ternary GEMV dispatcher (Bonsai vs i8 path) --------------------
    //
    // Round-5 engine-wire: the i8 dispatch now takes the activation scale by
    // device pointer. Bonsai still reads from `normed_fp16` directly and
    // ignores both x_i8_in + x_scale_dev — the wrapper passes them through
    // for callsite uniformity but only the i8 lane consumes them.
    rcpp_status_t ternary_gemv(const void* packed,
                               const int8_t* x_i8_in,
                               const float* x_scale_dev,
                               const float* row_scales,
                               const void* normed_fp16,
                               void* out_fp16,
                               int N, int K) const {
        if (is_bonsai) {
            return bonsai_gemv_dispatch(model.weight_format, packed,
                                        normed_fp16, out_fp16, N, K,
                                        /*stream=*/nullptr);
        }
        // SHERRY_FP16 carries a flag that says "this row_scale was minted
        // for the fp16-act kernel, not the i8 kernel". Routing it through
        // the i8 path double-counts magnitude (i8 quant introduces its own
        // x_scale on top of the row scale) and the activation precision
        // collapses to ±127. Send it to the dedicated fp16 launcher with
        // per-row scales instead — same packed weights, fp16 acts straight
        // off the RMSNorm output, single multiply at writeback.
        if (model.weight_format == RCPP_WEIGHT_FORMAT_SHERRY_FP16) {
            sherry_ternary_gemv_with_scales_launch(
                static_cast<const uint8_t*>(packed),
                static_cast<const uint16_t*>(normed_fp16),
                row_scales,
                static_cast<uint16_t*>(out_fp16),
                N, K, /*stream=*/nullptr);
            return RCPP_OK;
        }
        return ternary_gemv_i8(packed, x_i8_in, x_scale_dev, row_scales,
                               out_fp16, N, K, /*stream=*/nullptr);
    }

    // K alignment for GEMV input dims
    int align_k(int k) const {
        return ((k + k_pad_unit - 1) / k_pad_unit) * k_pad_unit;
    }

    // Forward one token through all layers, leaving fp32 logits in this->logits.
    // pos is the KV-cache slot (also the RoPE position).
    int forward_token(int token_id, int pos,
                      const GenerateOptions& opts,
                      std::mt19937_64& rng,
                      const std::vector<int>& sampler_recent) {
        RC_CHECK(rcpp_embedding_lookup_fp16(model.embedding_dev, token_id, x,
                                            hs, nullptr));
        HIP_CHECK(hipMemsetAsync(x_fp32, 0, hs * sizeof(float), nullptr));
        RC_CHECK(rcpp_residual_add_fp32_from_fp16(x_fp32, x, hs, nullptr));

        for (int l = 0; l < L; ++l) {
            rcpp_bitnet_layer_t& ly = model.layers[l];

            // --- Attention block ---
            RC_CHECK(rcpp_rmsnorm_fp32_in_fp16_out(
                x_fp32, ly.input_norm_dev, normed,
                model.rms_norm_eps, hs, nullptr));
            RC_CHECK(rcpp_quantize_fp16_to_i8(normed, x_i8, x_scale_dev, hs,
                                              nullptr));
            // Round-5: x_scale stays on the device. ternary_gemv() reads it
            // through the *_devscale entries; no D→H sync stall here.

            RC_CHECK(ternary_gemv(ly.q_packed_dev, x_i8, x_scale_dev,
                                  ly.q_scales_dev, normed, q_fp16,
                                  nh * hd, hs_k));
            RC_CHECK(ternary_gemv(ly.k_packed_dev, x_i8, x_scale_dev,
                                  ly.k_scales_dev, normed, k_fp16,
                                  nkv * hd, hs_k));
            RC_CHECK(ternary_gemv(ly.v_packed_dev, x_i8, x_scale_dev,
                                  ly.v_scales_dev, normed, v_fp16,
                                  nkv * hd, hs_k));

            if (is_qwen3 && ly.attn_q_norm_dev) {
                for (int h = 0; h < nh; ++h) {
                    _Float16* qh = q_fp16 + (size_t)h * hd;
                    RC_CHECK(rcpp_rmsnorm_fp16(qh, ly.attn_q_norm_dev, qh,
                                               model.rms_norm_eps, hd, nullptr));
                }
            }
            if (is_qwen3 && ly.attn_k_norm_dev) {
                for (int h = 0; h < nkv; ++h) {
                    _Float16* kh = k_fp16 + (size_t)h * hd;
                    RC_CHECK(rcpp_rmsnorm_fp16(kh, ly.attn_k_norm_dev, kh,
                                               model.rms_norm_eps, hd, nullptr));
                }
            }

            RC_CHECK(rcpp_rope_fp16(q_fp16, pos, model.rope_theta, nh, hd,
                                    nullptr));
            RC_CHECK(rcpp_rope_fp16(k_fp16, pos, model.rope_theta, nkv, hd,
                                    nullptr));

            if (cfg.kv_int8) {
                RC_CHECK(rcpp_quantize_fp16_to_i8_rowscale(
                    k_fp16, K_caches_i8[l] + (size_t)pos * nkv * hd,
                    K_scales[l] + (size_t)pos * nkv, nkv, hd, nullptr));
                RC_CHECK(rcpp_quantize_fp16_to_i8_rowscale(
                    v_fp16, V_caches_i8[l] + (size_t)pos * nkv * hd,
                    V_scales[l] + (size_t)pos * nkv, nkv, hd, nullptr));
                // Round-5: split-KV Flash-Decoding over int8 KV recovers the
                // 6.78× @ L=2048 win the fp16 FD port hit, composed with the
                // 2× DRAM cut int8 already gives. Internal partials cache
                // grows monotonically — no per-call hipMalloc.
                RC_CHECK(rcpp_kv_cache_attn_decode_fd_i8(
                    q_fp16, K_caches_i8[l], V_caches_i8[l],
                    K_scales[l], V_scales[l], o_fp16,
                    nh, nkv, hd, pos + 1, scale, nullptr));
            } else if (cfg.kv_rotor) {
                const size_t pq3_row = (size_t)hd * 3 / 8;
                rcpp_kv_requantize_pq3(
                    k_fp16, K_caches_pq3[l] + (size_t)pos * nkv * pq3_row,
                    /*seq_len=*/1, nkv, hd, /*layer_idx=*/l, nullptr);
                rcpp_kv_requantize_pq3_v(
                    v_fp16, V_caches_pq3[l] + (size_t)pos * nkv * pq3_row,
                    1, nkv, hd, l, nullptr);
                int rrc = rcpp_kv_cache_attn_decode_fd_pq3(
                    q_fp16, K_caches_pq3[l], V_caches_pq3[l], o_fp16,
                    nh, nkv, hd, pos + 1, /*layer_idx=*/l, scale, nullptr);
                if (rrc != 0) {
                    throw std::runtime_error(
                        "rocm_cpp::Engine: kv_rotor attn rc=" +
                        std::to_string(rrc) + " layer=" + std::to_string(l));
                }
            } else {
                HIP_CHECK(hipMemcpy(K_caches[l] + (size_t)pos * nkv * hd,
                                    k_fp16, nkv * hd * 2,
                                    hipMemcpyDeviceToDevice));
                HIP_CHECK(hipMemcpy(V_caches[l] + (size_t)pos * nkv * hd,
                                    v_fp16, nkv * hd * 2,
                                    hipMemcpyDeviceToDevice));
                RC_CHECK(rcpp_kv_cache_attn_decode_fd(
                    q_fp16, K_caches[l], V_caches[l], o_fp16,
                    nh, nkv, hd, pos + 1, scale, nullptr));
            }

            if (is_qwen3) {
                RC_CHECK(rcpp_quantize_fp16_to_i8(o_fp16, x_i8, x_scale_dev,
                                                  hs, nullptr));
                // Round-5: device-scalar — no D→H read.
                RC_CHECK(ternary_gemv(ly.o_packed_dev, x_i8, x_scale_dev,
                                      ly.o_scales_dev, o_fp16, normed,
                                      hs, align_k(nh * hd)));
                RC_CHECK(rcpp_residual_add_fp32_from_fp16(x_fp32, normed, hs,
                                                          nullptr));
            } else {
                RC_CHECK(rcpp_rmsnorm_fp16(o_fp16, ly.attn_sub_norm_dev, normed,
                                           model.rms_norm_eps, hs, nullptr));
                RC_CHECK(rcpp_quantize_fp16_to_i8(normed, x_i8, x_scale_dev, hs,
                                                  nullptr));
                RC_CHECK(ternary_gemv(ly.o_packed_dev, x_i8, x_scale_dev,
                                      ly.o_scales_dev, normed, o_fp16,
                                      hs, align_k(nh * hd)));
                RC_CHECK(rcpp_residual_add_fp32_from_fp16(x_fp32, o_fp16, hs,
                                                          nullptr));
            }

            // --- FFN block ---
            RC_CHECK(rcpp_rmsnorm_fp32_in_fp16_out(
                x_fp32, ly.post_attn_norm_dev, normed,
                model.rms_norm_eps, hs, nullptr));
            RC_CHECK(rcpp_quantize_fp16_to_i8(normed, x_i8, x_scale_dev, hs,
                                              nullptr));

            RC_CHECK(ternary_gemv(ly.gate_packed_dev, x_i8, x_scale_dev,
                                  ly.gate_scales_dev, normed, gate_fp16,
                                  is_, hs_k));
            RC_CHECK(ternary_gemv(ly.up_packed_dev, x_i8, x_scale_dev,
                                  ly.up_scales_dev, normed, up_fp16,
                                  is_, hs_k));

            if (is_qwen3) {
                RC_CHECK(rcpp_silu_glu_fp16(gate_fp16, up_fp16, silu_out, is_,
                                            nullptr));
            } else {
                RC_CHECK(rcpp_relu2_glu_rmsnorm_fp16(
                    gate_fp16, up_fp16, ly.ffn_sub_norm_dev, silu_out,
                    model.rms_norm_eps, is_, nullptr));
            }

            RC_CHECK(rcpp_quantize_fp16_to_i8(silu_out, silu_i8,
                                              silu_scale_dev, is_, nullptr));
            // Round-5: silu_scale stays on the device; the down GEMV reads
            // it through the *_devscale entry.
            RC_CHECK(ternary_gemv(ly.down_packed_dev, silu_i8, silu_scale_dev,
                                  ly.down_scales_dev, silu_out, down_fp16,
                                  hs, is_k));
            RC_CHECK(rcpp_residual_add_fp32_from_fp16(x_fp32, down_fp16, hs,
                                                      nullptr));
        }

        RC_CHECK(rcpp_rmsnorm_fp32_in_fp16_out(
            x_fp32, model.final_norm_weight_dev, normed,
            model.rms_norm_eps, hs, nullptr));
        RC_CHECK(rcpp_fp16_gemv(model.embedding_dev, normed, logits, V, hs,
                                nullptr));

        if (opts.temperature <= 0.0f) {
            int next_tok = 0;
            RC_CHECK(rcpp_argmax_fp32(logits, next_tok_dev, V, nullptr));
            HIP_CHECK(hipMemcpy(&next_tok, next_tok_dev, 4,
                                hipMemcpyDeviceToHost));
            return next_tok;
        }
        return sample_host(opts, rng, sampler_recent);
    }

    // Host-side sampler — same chain as bitnet_decode (rep penalty -> top-k
    // -> softmax(temp) -> top-p -> multinomial). One D->H copy per token.
    int sample_host(const GenerateOptions& opts, std::mt19937_64& rng,
                    const std::vector<int>& recent) {
        HIP_CHECK(hipMemcpy(logits_host.data(), logits, V * 4,
                            hipMemcpyDeviceToHost));

        if (opts.rep_penalty != 1.0f && opts.rep_last_n > 0) {
            int start = std::max(0, (int)recent.size() - opts.rep_last_n);
            for (int i = start; i < (int)recent.size(); ++i) {
                int id = recent[i];
                if (id >= 0 && id < V) {
                    float& l = logits_host[id];
                    l = (l > 0.0f) ? (l / opts.rep_penalty)
                                   : (l * opts.rep_penalty);
                }
            }
        }

        if (opts.top_k > 0 && opts.top_k < V) {
            std::vector<float> tmp(logits_host);
            std::nth_element(tmp.begin(), tmp.begin() + (V - opts.top_k),
                             tmp.end());
            float thresh = tmp[V - opts.top_k];
            for (float& l : logits_host) if (l < thresh) l = -INFINITY;
        }

        float m = -INFINITY;
        for (float l : logits_host) if (l > m) m = l;
        double sum = 0.0;
        for (float& l : logits_host) {
            l = std::exp((l - m) / opts.temperature);
            sum += l;
        }
        const float inv = (float)(1.0 / (sum > 0 ? sum : 1.0));
        for (float& l : logits_host) l *= inv;

        if (opts.top_p > 0.0f && opts.top_p < 1.0f) {
            std::vector<int> idx(V);
            for (int i = 0; i < V; ++i) idx[i] = i;
            std::sort(idx.begin(), idx.end(),
                      [&](int a, int b) {
                          return logits_host[a] > logits_host[b];
                      });
            float csum = 0.0f;
            int cutoff = V;
            for (int i = 0; i < V; ++i) {
                csum += logits_host[idx[i]];
                if (csum >= opts.top_p) { cutoff = i + 1; break; }
            }
            for (int i = cutoff; i < V; ++i) logits_host[idx[i]] = 0.0f;
            float keep_sum = 0.0f;
            for (int i = 0; i < cutoff; ++i) keep_sum += logits_host[idx[i]];
            if (keep_sum > 0) {
                float s = 1.0f / keep_sum;
                for (int i = 0; i < cutoff; ++i) logits_host[idx[i]] *= s;
            }
        }

        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        float r = u(rng);
        float acc = 0.0f;
        for (int i = 0; i < V; ++i) {
            acc += logits_host[i];
            if (acc >= r) return i;
        }
        return V - 1;
    }

    // Forward one token with NO sampling. Leaves fp32 logits in this->logits
    // (device) and copies them out to `out_host_logits` if non-null. This is
    // the PPL entry point — callers reset cache_pos to 0 before the loop and
    // walk the cache positions monotonically, exactly like prefill does.
    void forward_token_no_sample(int token_id, int pos,
                                 float* out_host_logits) {
        RC_CHECK(rcpp_embedding_lookup_fp16(model.embedding_dev, token_id, x,
                                            hs, nullptr));
        HIP_CHECK(hipMemsetAsync(x_fp32, 0, hs * sizeof(float), nullptr));
        RC_CHECK(rcpp_residual_add_fp32_from_fp16(x_fp32, x, hs, nullptr));

        for (int l = 0; l < L; ++l) {
            rcpp_bitnet_layer_t& ly = model.layers[l];

            RC_CHECK(rcpp_rmsnorm_fp32_in_fp16_out(
                x_fp32, ly.input_norm_dev, normed,
                model.rms_norm_eps, hs, nullptr));
            RC_CHECK(rcpp_quantize_fp16_to_i8(normed, x_i8, x_scale_dev, hs,
                                              nullptr));
            // Round-5: x_scale lives on device, ternary_gemv reads via *_devscale.

            RC_CHECK(ternary_gemv(ly.q_packed_dev, x_i8, x_scale_dev,
                                  ly.q_scales_dev, normed, q_fp16,
                                  nh * hd, hs_k));
            RC_CHECK(ternary_gemv(ly.k_packed_dev, x_i8, x_scale_dev,
                                  ly.k_scales_dev, normed, k_fp16,
                                  nkv * hd, hs_k));
            RC_CHECK(ternary_gemv(ly.v_packed_dev, x_i8, x_scale_dev,
                                  ly.v_scales_dev, normed, v_fp16,
                                  nkv * hd, hs_k));

            if (is_qwen3 && ly.attn_q_norm_dev) {
                for (int h = 0; h < nh; ++h) {
                    _Float16* qh = q_fp16 + (size_t)h * hd;
                    RC_CHECK(rcpp_rmsnorm_fp16(qh, ly.attn_q_norm_dev, qh,
                                               model.rms_norm_eps, hd, nullptr));
                }
            }
            if (is_qwen3 && ly.attn_k_norm_dev) {
                for (int h = 0; h < nkv; ++h) {
                    _Float16* kh = k_fp16 + (size_t)h * hd;
                    RC_CHECK(rcpp_rmsnorm_fp16(kh, ly.attn_k_norm_dev, kh,
                                               model.rms_norm_eps, hd, nullptr));
                }
            }

            RC_CHECK(rcpp_rope_fp16(q_fp16, pos, model.rope_theta, nh, hd,
                                    nullptr));
            RC_CHECK(rcpp_rope_fp16(k_fp16, pos, model.rope_theta, nkv, hd,
                                    nullptr));

            if (cfg.kv_int8) {
                RC_CHECK(rcpp_quantize_fp16_to_i8_rowscale(
                    k_fp16, K_caches_i8[l] + (size_t)pos * nkv * hd,
                    K_scales[l] + (size_t)pos * nkv, nkv, hd, nullptr));
                RC_CHECK(rcpp_quantize_fp16_to_i8_rowscale(
                    v_fp16, V_caches_i8[l] + (size_t)pos * nkv * hd,
                    V_scales[l] + (size_t)pos * nkv, nkv, hd, nullptr));
                // Round-5: split-KV i8 attn — same FD win on the i8 path.
                RC_CHECK(rcpp_kv_cache_attn_decode_fd_i8(
                    q_fp16, K_caches_i8[l], V_caches_i8[l],
                    K_scales[l], V_scales[l], o_fp16,
                    nh, nkv, hd, pos + 1, scale, nullptr));
            } else if (cfg.kv_rotor) {
                const size_t pq3_row = (size_t)hd * 3 / 8;
                rcpp_kv_requantize_pq3(
                    k_fp16, K_caches_pq3[l] + (size_t)pos * nkv * pq3_row,
                    1, nkv, hd, l, nullptr);
                rcpp_kv_requantize_pq3_v(
                    v_fp16, V_caches_pq3[l] + (size_t)pos * nkv * pq3_row,
                    1, nkv, hd, l, nullptr);
                int rrc = rcpp_kv_cache_attn_decode_fd_pq3(
                    q_fp16, K_caches_pq3[l], V_caches_pq3[l], o_fp16,
                    nh, nkv, hd, pos + 1, l, scale, nullptr);
                if (rrc != 0) {
                    throw std::runtime_error(
                        "rocm_cpp::Engine: kv_rotor attn rc=" +
                        std::to_string(rrc) + " layer=" + std::to_string(l));
                }
            } else {
                HIP_CHECK(hipMemcpy(K_caches[l] + (size_t)pos * nkv * hd,
                                    k_fp16, nkv * hd * 2,
                                    hipMemcpyDeviceToDevice));
                HIP_CHECK(hipMemcpy(V_caches[l] + (size_t)pos * nkv * hd,
                                    v_fp16, nkv * hd * 2,
                                    hipMemcpyDeviceToDevice));
                RC_CHECK(rcpp_kv_cache_attn_decode_fd(
                    q_fp16, K_caches[l], V_caches[l], o_fp16,
                    nh, nkv, hd, pos + 1, scale, nullptr));
            }

            if (is_qwen3) {
                RC_CHECK(rcpp_quantize_fp16_to_i8(o_fp16, x_i8, x_scale_dev,
                                                  hs, nullptr));
                RC_CHECK(ternary_gemv(ly.o_packed_dev, x_i8, x_scale_dev,
                                      ly.o_scales_dev, o_fp16, normed,
                                      hs, align_k(nh * hd)));
                RC_CHECK(rcpp_residual_add_fp32_from_fp16(x_fp32, normed, hs,
                                                          nullptr));
            } else {
                RC_CHECK(rcpp_rmsnorm_fp16(o_fp16, ly.attn_sub_norm_dev, normed,
                                           model.rms_norm_eps, hs, nullptr));
                RC_CHECK(rcpp_quantize_fp16_to_i8(normed, x_i8, x_scale_dev, hs,
                                                  nullptr));
                RC_CHECK(ternary_gemv(ly.o_packed_dev, x_i8, x_scale_dev,
                                      ly.o_scales_dev, normed, o_fp16,
                                      hs, align_k(nh * hd)));
                RC_CHECK(rcpp_residual_add_fp32_from_fp16(x_fp32, o_fp16, hs,
                                                          nullptr));
            }

            RC_CHECK(rcpp_rmsnorm_fp32_in_fp16_out(
                x_fp32, ly.post_attn_norm_dev, normed,
                model.rms_norm_eps, hs, nullptr));
            RC_CHECK(rcpp_quantize_fp16_to_i8(normed, x_i8, x_scale_dev, hs,
                                              nullptr));

            RC_CHECK(ternary_gemv(ly.gate_packed_dev, x_i8, x_scale_dev,
                                  ly.gate_scales_dev, normed, gate_fp16,
                                  is_, hs_k));
            RC_CHECK(ternary_gemv(ly.up_packed_dev, x_i8, x_scale_dev,
                                  ly.up_scales_dev, normed, up_fp16,
                                  is_, hs_k));

            if (is_qwen3) {
                RC_CHECK(rcpp_silu_glu_fp16(gate_fp16, up_fp16, silu_out, is_,
                                            nullptr));
            } else {
                RC_CHECK(rcpp_relu2_glu_rmsnorm_fp16(
                    gate_fp16, up_fp16, ly.ffn_sub_norm_dev, silu_out,
                    model.rms_norm_eps, is_, nullptr));
            }

            RC_CHECK(rcpp_quantize_fp16_to_i8(silu_out, silu_i8,
                                              silu_scale_dev, is_, nullptr));
            RC_CHECK(ternary_gemv(ly.down_packed_dev, silu_i8, silu_scale_dev,
                                  ly.down_scales_dev, silu_out, down_fp16,
                                  hs, is_k));
            RC_CHECK(rcpp_residual_add_fp32_from_fp16(x_fp32, down_fp16, hs,
                                                      nullptr));
        }

        RC_CHECK(rcpp_rmsnorm_fp32_in_fp16_out(
            x_fp32, model.final_norm_weight_dev, normed,
            model.rms_norm_eps, hs, nullptr));
        RC_CHECK(rcpp_fp16_gemv(model.embedding_dev, normed, logits, V, hs,
                                nullptr));

        if (out_host_logits) {
            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipMemcpy(out_host_logits, logits, V * sizeof(float),
                                hipMemcpyDeviceToHost));
        }
    }

    // Per-position NLL. Resets KV state by re-prefilling from pos=0 — caller
    // is responsible for clamping `ids` to max_len.
    std::vector<float> compute_nll(const std::vector<int>& ids) {
        if (!loaded) {
            throw std::runtime_error(
                "rocm_cpp::Engine::compute_nll: model not loaded");
        }
        if (ids.size() < 2) return {};
        const int n = (int)ids.size();
        if (n > max_len) {
            throw std::runtime_error(
                "rocm_cpp::Engine::compute_nll: input length " +
                std::to_string(n) + " > max_context " + std::to_string(max_len));
        }

        std::vector<float> nll;
        nll.reserve(n - 1);
        std::vector<float> host_logits(V);

        for (int t = 0; t < n - 1; ++t) {
            forward_token_no_sample(ids[t], t, host_logits.data());
            HIP_CHECK(hipDeviceSynchronize());

            // Numerically-stable log_softmax at the next-target id.
            const int target = ids[t + 1];
            float m = -INFINITY;
            for (int i = 0; i < V; ++i) if (host_logits[i] > m) m = host_logits[i];
            double sum = 0.0;
            for (int i = 0; i < V; ++i) sum += std::exp((double)host_logits[i] - m);
            const float lse = m + (float)std::log(sum > 0 ? sum : 1.0);
            float lp_target;
            if (target >= 0 && target < V) {
                lp_target = host_logits[target] - lse;
            } else {
                lp_target = -lse;
            }
            nll.push_back(-lp_target);
        }
        return nll;
    }

    // Tokenize a single string. add_bos=true prepends the .htok-declared BOS
    // (128000 on Llama-3 tokenizers). The byte-level BPE encoder does NOT
    // recognize special-token markers like "<|eot_id|>" — those come back as
    // raw byte tokens. Callers that need exact special ids must inject them.
    std::vector<int> tokenize(const std::string& text, bool add_bos) {
        if (!tokenizer) return {};
        std::vector<int> buf(4096);
        size_t count = 0;
        const int bos_flag = add_bos ? 1 : 0;
        rcpp_tokenizer_encode(tokenizer, text.data(), text.size(),
                              bos_flag, buf.data(), buf.size(), &count);
        if (count > buf.size()) {
            buf.resize(count);
            rcpp_tokenizer_encode(tokenizer, text.data(), text.size(),
                                  bos_flag, buf.data(), buf.size(), &count);
        }
        buf.resize(count);
        return buf;
    }

    GenerateResult generate(const std::string& prompt,
                            const GenerateOptions& opts,
                            std::function<bool(const std::string&)> on_token) {
        if (!loaded) {
            throw std::runtime_error(
                "rocm_cpp::Engine::generate: model not loaded");
        }
        // String entry: BOS-prepended byte-level encode, no chat template.
        // Callers that need special-token IDs go through generate_from_tokens().
        std::vector<int> prompt_ids;
        if (tokenizer) {
            prompt_ids = tokenize(prompt, /*add_bos=*/true);
        } else {
            // No tokenizer: best-effort BOS-only prompt (matches bitnet_decode's
            // fallback when --text is given but the .htok file is missing).
            prompt_ids.push_back(1);
        }
        return generate_from_tokens(prompt_ids, opts, std::move(on_token));
    }

    GenerateResult generate_from_tokens(
            const std::vector<int>& prompt_ids_in,
            const GenerateOptions& opts,
            std::function<bool(const std::string&)> on_token) {
        if (!loaded) {
            throw std::runtime_error(
                "rocm_cpp::Engine::generate_from_tokens: model not loaded");
        }
        if (prompt_ids_in.empty()) {
            throw std::runtime_error(
                "rocm_cpp::Engine::generate_from_tokens: prompt is empty");
        }
        std::vector<int> prompt_ids = prompt_ids_in;

        if ((int)prompt_ids.size() >= max_len) {
            throw std::runtime_error(
                "rocm_cpp::Engine::generate_from_tokens: prompt (" +
                std::to_string(prompt_ids.size()) +
                ") >= max_context (" + std::to_string(max_len) + ")");
        }

        const int prompt_len = (int)prompt_ids.size();
        int max_new = opts.max_tokens;
        if (max_new < 0) max_new = 0;
        const int ctx_room = max_len - prompt_len;
        if (max_new > ctx_room) max_new = ctx_room;

        const uint64_t seed =
            opts.seed != 0
                ? opts.seed
                : (uint64_t)std::chrono::steady_clock::now()
                        .time_since_epoch().count();
        std::mt19937_64 rng(seed);

        sampler_history = prompt_ids;

        // Fresh KV cache for this generation.
        int cache_pos = 0;

        // ---- Prefill ----
        auto t_pre0 = std::chrono::high_resolution_clock::now();
        int last_tok = prompt_ids.front();
        for (size_t i = 0; i < prompt_ids.size(); ++i) {
            last_tok = forward_token(prompt_ids[i], cache_pos + (int)i, opts,
                                     rng, sampler_history);
            HIP_CHECK(hipDeviceSynchronize());
        }
        cache_pos += (int)prompt_ids.size();
        auto t_pre1 = std::chrono::high_resolution_clock::now();
        const double prefill_ms =
            std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count();

        GenerateResult result;
        result.prompt_tokens = prompt_len;
        result.prefill_tok_per_sec =
            (prefill_ms > 0) ? (1000.0 * prompt_len / prefill_ms) : 0.0;

        if (max_new <= 0) {
            result.finish_reason = "length";
            return result;
        }

        // ---- Decode ----
        // Caller-supplied stops take precedence; otherwise default to the
        // Llama-3 EOS pair (preserves halo-1bit-2b regression baseline).
        std::vector<int> stop_ids = opts.stop_token_ids;
        if (stop_ids.empty()) {
            stop_ids = {128001, 128009};  // <|end_of_text|>, <|eot_id|>
        }
        std::vector<int> generated;
        generated.reserve(max_new);
        std::string out_text;
        size_t printed_bytes = 0;
        bool cancelled = false;
        std::string finish = "length";

        // Arch-aware special-token marker filter for the streaming on_token
        // callback. on_token sees only emit-safe bytes; multi-byte markers
        // that span chunk boundaries are buffered until they either complete
        // (and are stripped) or are proven impossible (and emitted).
        const Arch arch = is_qwen3 ? Arch::Qwen3 : Arch::BitNet;
        StreamingMarkerFilter stream_filter(arch);

        auto t_dec0 = std::chrono::high_resolution_clock::now();
        int cur_tok = last_tok;
        for (int step = 0; step < max_new; ++step) {
            int next_tok = forward_token(cur_tok, cache_pos + step, opts, rng,
                                         sampler_history);
            HIP_CHECK(hipDeviceSynchronize());
            generated.push_back(next_tok);
            sampler_history.push_back(next_tok);

            // Streaming detokenization — same window-decode-and-diff
            // technique bitnet_decode uses for stdout. Avoids re-tokenizing
            // the entire generation each step.
            std::string delta;
            if (tokenizer) {
                size_t tlen = 0;
                rcpp_tokenizer_decode(tokenizer, generated.data(),
                                      generated.size(), stream_buf.data(),
                                      stream_buf.size(), &tlen);
                tlen = std::min(tlen, stream_buf.size());
                if (tlen > printed_bytes) {
                    delta.assign(stream_buf.data() + printed_bytes,
                                 tlen - printed_bytes);
                    out_text.append(delta);
                    printed_bytes = tlen;
                }
            }

            if (on_token && !delta.empty()) {
                std::string safe = stream_filter.feed(delta);
                if (!safe.empty()) {
                    if (!on_token(safe)) {
                        cancelled = true;
                        finish = "cancel";
                        break;
                    }
                }
            }

            bool eos_hit = false;
            for (int sid : stop_ids) {
                if (next_tok == sid) { eos_hit = true; break; }
            }
            if (eos_hit) {
                finish = "stop";
                break;
            }
            cur_tok = next_tok;

            // --- stop_sequences (suffix match on detokenized tail) ---
            if (!opts.stop_sequences.empty() && tokenizer) {
                size_t win = std::min((size_t)64, generated.size());
                size_t tlen = 0;
                rcpp_tokenizer_decode(
                    tokenizer, generated.data() + (generated.size() - win),
                    win, tail_buf.data(), tail_buf.size(), &tlen);
                tlen = std::min(tlen, tail_buf.size());
                std::string tail(tail_buf.data(), tlen);
                bool hit = false;
                for (const auto& s : opts.stop_sequences) {
                    if (tail.size() >= s.size() &&
                        tail.compare(tail.size() - s.size(), s.size(), s) == 0) {
                        hit = true;
                        break;
                    }
                }
                if (hit) {
                    finish = "stop";
                    break;
                }
            }
        }
        auto t_dec1 = std::chrono::high_resolution_clock::now();
        const double decode_ms =
            std::chrono::duration<double, std::milli>(t_dec1 - t_dec0).count();

        // Drain the streaming filter — anything still held back is emit-safe
        // now that decode is finished. Skip if cancelled (caller already
        // told us to stop; don't push extra bytes after a "false" return).
        if (!cancelled && on_token) {
            std::string tail = stream_filter.flush();
            if (!tail.empty()) {
                (void)on_token(tail);
            }
        }

        // Arch-aware all-occurrences strip on the accumulated full-text result.
        // Replaces the prior Llama-3-only trailing-suffix strip:
        //   - BitNet: <|begin_of_text|>, <|end_of_text|>, <|eot_id|>
        //   - Qwen3:  <|im_start|>, <|im_end|>, <|endoftext|>
        // Token bytes can leak through the byte-level BPE detokenizer at any
        // position (most often the trailing EOS token, but also mid-stream
        // when the model pastes them in raw); strip them all unconditionally
        // before handing back to the caller.
        strip_specials_inplace(out_text, arch);

        result.text              = std::move(out_text);
        result.completion_tokens = (int)generated.size();
        result.decode_tok_per_sec =
            (decode_ms > 0 && !generated.empty())
                ? (1000.0 * generated.size() / decode_ms)
                : 0.0;
        result.finish_reason = cancelled ? "cancel" : finish;
        return result;
    }

    // ── Medusa speculative decoding ─────────────────────────────────────────
    //
    // v0 wire shape: linear chain.
    //
    //   1. Run base forward at pos to predict t0 (argmax / sampled). The
    //      hidden state used for the LM head (`normed`, post-final-RMSNorm)
    //      is still in the device scratch.
    //   2. For each medusa head h ∈ [0, K-1), apply head h to that hidden
    //      state to get candidate c_{h+1}. With synthetic-zero heads the
    //      logits collapse to all-zero and the argmax is 0 — that's fine,
    //      it lets the verify path execute end-to-end without crashing.
    //   3. Verify: feed t0 through forward at pos+1, compare its argmax to
    //      c1; on match, feed c1 at pos+2 and compare argmax to c2; etc.
    //      Stop on first mismatch. The mismatch step's argmax is correct
    //      and is committed (KV cache is monotonically advanced so no
    //      rollback is needed past the rejected position; the simple
    //      sequential verify never advances past a rejected token).
    //
    // The Medusa kernels rcpp_medusa_tree_attn_decode_fd / _small_m_gemv
    // ship in librocm_cpp.so and are exercised by tests/test_medusa_*.
    // This generate path uses the simpler sequential primitives — it's
    // bit-identical to a plain decode at every committed position so the
    // text output never diverges from generate_from_tokens(), even when
    // the heads are real and the kernel exists. The tree-attention shape
    // is a follow-up optimization for tree-of-candidates verify.

    // Apply Medusa head `head_idx` to the current `normed` hidden state and
    // return the argmax token id. Mirrors the lm_head pass except the
    // weights are the head's packed-ternary buffer rather than the tied
    // embedding. For HALO_V2 the hidden state is int8-quantized first via
    // the same pre-existing scratch buffers used by the q/k/v projections.
    int medusa_head_argmax(uint32_t head_idx) {
        const rcpp_medusa_head_t& h = medusa.heads[head_idx];
        // Allocate a scratch fp16 logits buffer on first use. Re-uses the
        // existing fp32 logits slot would require an extra fp16→fp32 cast;
        // instead we copy fp16 to host and argmax there. V is bounded
        // (128k for halo-1bit-2b) so the D2H is cheap.
        static thread_local std::vector<_Float16> h_logits_fp16;
        if ((int)h_logits_fp16.size() != V) h_logits_fp16.assign(V, _Float16(0));

        // We need a device fp16 buffer of size [V] for the head output.
        // Reuse the same scratch across all heads in a round.
        static thread_local _Float16* d_head_logits = nullptr;
        static thread_local int       d_head_logits_cap = 0;
        if (d_head_logits_cap < V) {
            if (d_head_logits) hipFree(d_head_logits);
            HIP_CHECK(hipMalloc(&d_head_logits, (size_t)V * sizeof(_Float16)));
            d_head_logits_cap = V;
        }

        // Quantize the (already-final-RMSNorm'd) hidden state in `normed`
        // to int8 + per-tensor scale, then dispatch the format-appropriate
        // ternary GEMV.
        RC_CHECK(rcpp_quantize_fp16_to_i8(normed, x_i8, x_scale_dev, hs, nullptr));
        float x_scale_local = 0.0f;
        HIP_CHECK(hipMemcpy(&x_scale_local, x_scale_dev, 4, hipMemcpyDeviceToHost));

        switch (medusa.weight_format) {
            case RCPP_WEIGHT_FORMAT_HALO_V2:
                RC_CHECK(rcpp_ternary_gemv_halo_f16(
                    h.packed_dev, x_i8, x_scale_local, h.row_scales_dev,
                    d_head_logits, V, hs_k, nullptr));
                break;
            case RCPP_WEIGHT_FORMAT_SHERRY_I8:
            case RCPP_WEIGHT_FORMAT_SHERRY_FP16:
                RC_CHECK(rcpp_ternary_gemv_sherry_f16(
                    h.packed_dev, x_i8, x_scale_local, h.row_scales_dev,
                    d_head_logits, V, hs_k, nullptr));
                break;
            case RCPP_WEIGHT_FORMAT_TQ1:
                RC_CHECK(rcpp_ternary_gemv_tq1_halo_f16(
                    h.packed_dev, x_i8, x_scale_local, h.row_scales_dev,
                    d_head_logits, V, hs_k, nullptr));
                break;
            default:
                // Bonsai head formats not yet exercised — treat as zero
                // logits (all candidates collapse to id 0, identical to
                // synthetic-zero behavior).
                HIP_CHECK(hipMemsetAsync(d_head_logits, 0,
                                         (size_t)V * sizeof(_Float16), nullptr));
                break;
        }

        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_logits_fp16.data(), d_head_logits,
                            (size_t)V * sizeof(_Float16),
                            hipMemcpyDeviceToHost));

        // Host-side argmax. fp16-stable: scan u16 monotone of the
        // signed-zero-masked values (handles -inf pad if a future caller
        // ever writes them). Synthetic-zero heads produce all-zero
        // h_logits_fp16 → argmax falls to index 0 by tie-breaking.
        int   best_idx = 0;
        float best_val = (float)h_logits_fp16[0];
        for (int i = 1; i < V; ++i) {
            float v = (float)h_logits_fp16[i];
            if (v > best_val) { best_val = v; best_idx = i; }
        }
        return best_idx;
    }

    // Read the device fp32 `logits` and return the argmax. Caller must have
    // synchronized the stream.
    int argmax_logits_host() {
        HIP_CHECK(hipMemcpy(logits_host.data(), logits, V * 4,
                            hipMemcpyDeviceToHost));
        int   best_idx = 0;
        float best_val = logits_host[0];
        for (int i = 1; i < V; ++i) {
            if (logits_host[i] > best_val) {
                best_val = logits_host[i];
                best_idx = i;
            }
        }
        return best_idx;
    }

    GenerateResult generate_medusa(const std::vector<int>& prompt_ids_in,
                                   const GenerateOptions& opts,
                                   int num_speculative,
                                   std::function<bool(const std::string&)> on_token)
    {
        // Caller-side fallback: if Medusa isn't usable, just run the standard
        // path. This keeps the recipe-options gate cheap on the wire — a
        // request with `medusa_speculative=N` against a non-Medusa-attached
        // engine still works, just without the speedup.
        if (!medusa_attached || num_speculative <= 1) {
            return generate_from_tokens(prompt_ids_in, opts,
                                        std::move(on_token));
        }
        if (!loaded) {
            throw std::runtime_error(
                "rocm_cpp::Engine::generate_medusa: model not loaded");
        }
        if (prompt_ids_in.empty()) {
            throw std::runtime_error(
                "rocm_cpp::Engine::generate_medusa: prompt is empty");
        }
        // Cap K at the number of attached heads + 1 (root). Each head
        // produces one drafted token; attaching N heads bounds K at N+1.
        const int K_max = (int)medusa.num_heads + 1;
        const int K = std::min(num_speculative, K_max);

        std::vector<int> prompt_ids = prompt_ids_in;
        if ((int)prompt_ids.size() >= max_len) {
            throw std::runtime_error(
                "rocm_cpp::Engine::generate_medusa: prompt (" +
                std::to_string(prompt_ids.size()) +
                ") >= max_context (" + std::to_string(max_len) + ")");
        }
        const int prompt_len = (int)prompt_ids.size();
        int max_new = opts.max_tokens;
        if (max_new < 0) max_new = 0;
        const int ctx_room = max_len - prompt_len;
        if (max_new > ctx_room) max_new = ctx_room;

        const uint64_t seed =
            opts.seed != 0
                ? opts.seed
                : (uint64_t)std::chrono::steady_clock::now()
                        .time_since_epoch().count();
        std::mt19937_64 rng(seed);

        sampler_history = prompt_ids;
        int cache_pos = 0;

        // ── Prefill (greedy, KV cache filled position 0..prompt_len-1) ──
        auto t_pre0 = std::chrono::high_resolution_clock::now();
        int last_tok = prompt_ids.front();
        for (size_t i = 0; i < prompt_ids.size(); ++i) {
            last_tok = forward_token(prompt_ids[i], cache_pos + (int)i, opts,
                                     rng, sampler_history);
            HIP_CHECK(hipDeviceSynchronize());
        }
        cache_pos += (int)prompt_ids.size();
        auto t_pre1 = std::chrono::high_resolution_clock::now();
        const double prefill_ms =
            std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count();

        GenerateResult result;
        result.prompt_tokens = prompt_len;
        result.prefill_tok_per_sec =
            (prefill_ms > 0) ? (1000.0 * prompt_len / prefill_ms) : 0.0;

        if (max_new <= 0) {
            result.finish_reason = "length";
            return result;
        }

        // ── Stop-token IDs ──
        std::vector<int> stop_ids = opts.stop_token_ids;
        if (stop_ids.empty()) stop_ids = {128001, 128009};
        auto is_stop = [&](int t) {
            for (int sid : stop_ids) if (t == sid) return true;
            return false;
        };

        std::vector<int> generated;
        generated.reserve(max_new);
        std::string out_text;
        size_t printed_bytes = 0;
        bool cancelled = false;
        // `finish` is the SET-BY-EXIT-CONDITION reason. Empty string ==
        // "still going" and is the only value that does NOT trigger the
        // outer-while early-out. Mapped to "length" at the end if no
        // explicit reason was set (i.e. we ran out of room or fell off
        // the natural max_new cap without an EOS hit).
        std::string finish = "";

        const Arch arch_tag = is_qwen3 ? Arch::Qwen3 : Arch::BitNet;
        StreamingMarkerFilter stream_filter(arch_tag);

        // Helper: append `next_tok` to the running stream and fire on_token.
        // Returns false if the user cancelled, which the caller surfaces as
        // finish_reason="cancel".
        auto commit_token = [&](int next_tok) -> bool {
            generated.push_back(next_tok);
            sampler_history.push_back(next_tok);
            if (tokenizer) {
                size_t tlen = 0;
                rcpp_tokenizer_decode(tokenizer, generated.data(),
                                      generated.size(), stream_buf.data(),
                                      stream_buf.size(), &tlen);
                tlen = std::min(tlen, stream_buf.size());
                if (tlen > printed_bytes) {
                    std::string delta(stream_buf.data() + printed_bytes,
                                      tlen - printed_bytes);
                    out_text.append(delta);
                    printed_bytes = tlen;
                    if (on_token && !delta.empty()) {
                        std::string safe = stream_filter.feed(delta);
                        if (!safe.empty() && !on_token(safe)) {
                            cancelled = true;
                            return false;
                        }
                    }
                }
            }
            return true;
        };

        auto t_dec0 = std::chrono::high_resolution_clock::now();
        int cur_tok = last_tok;
        int decode_steps = 0;

        while ((int)generated.size() < max_new) {
            // ── 1) Base step: forward `cur_tok` at cache_pos to get t0
            //                 prediction. Hidden state for head dispatch
            //                 lives in `normed` afterwards.
            int t0 = forward_token(cur_tok, cache_pos, opts, rng,
                                   sampler_history);
            HIP_CHECK(hipDeviceSynchronize());
            ++decode_steps;
            cache_pos += 1;

            // ── 2) Draft K-1 candidates from heads (top-1 each).
            //       v1 (vocab projection): each head fires on the same
            //         `normed` hidden state to produce one candidate.
            //       v2 (residual MLP): linear chain. h_0 = normed; for
            //         k = 0..K-2: h_{k+1} = h_k + w_out_k @ SiLU(w_in_k @ h_k);
            //         candidate = argmax(lm_head @ h_{k+1}).
            // chain[0] = t0; chain[1..K-1] = head outputs.
            std::vector<int> chain;
            chain.reserve(K);
            chain.push_back(t0);
            if (medusa.variant == RCPP_MEDUSA_VARIANT_RESIDUAL_MLP) {
                // Snapshot the base post-final-RMSNorm hidden state into
                // medusa_v2_h_in BEFORE we touch it again. forward_token's
                // verify pass below will overwrite normed[] for each step,
                // but the heads themselves only read from medusa_v2_h_in /
                // medusa_v2_h_out — those buffers are private to the chain.
                HIP_CHECK(hipMemcpy(medusa_v2_h_in, normed,
                                    (size_t)hs * 2,
                                    hipMemcpyDeviceToDevice));
                for (int h = 0; h + 1 < K; ++h) {
                    int c = medusa_residual_argmax((uint32_t)h,
                                                   medusa_v2_h_in,
                                                   medusa_v2_h_out);
                    chain.push_back(c);
                    // Chain: next head reads h_out as its h_in. Swap by
                    // rotating the buffer pointers (no copy).
                    std::swap(medusa_v2_h_in, medusa_v2_h_out);
                }
            } else {
                for (int h = 0; h + 1 < K; ++h) {
                    int c = medusa_head_argmax((uint32_t)h);
                    chain.push_back(c);
                }
            }

            // ── 3) Commit t0 to the stream — it's an unconditional accept
            //       (the model itself produced it). Also handle the EOS /
            //       cancel / max_tokens early-exit on it.
            if (!commit_token(t0)) { finish = "cancel"; break; }
            if (is_stop(t0))      { finish = "stop";   break; }
            if ((int)generated.size() >= max_new) { finish = "length"; break; }

            // ── 4) Sequential verify of chain[1..K-1] against the model's
            //       own argmax at each next position. First mismatch ends
            //       this round; the position's t_real becomes the next
            //       cur_tok and the next round starts.
            int next_round_seed = chain[0];   // fallback if all chain match
            for (int j = 1; j < K; ++j) {
                int verify_in = chain[j - 1];
                int t_real = forward_token(verify_in, cache_pos, opts, rng,
                                           sampler_history);
                HIP_CHECK(hipDeviceSynchronize());
                ++decode_steps;
                cache_pos += 1;

                if (t_real == chain[j]) {
                    // Candidate accepted — commit it as the NEXT model token
                    // (the model itself just confirmed this prediction).
                    if (!commit_token(t_real)) {
                        cancelled = true; break;
                    }
                    if (is_stop(t_real)) { finish = "stop"; break; }
                    if ((int)generated.size() >= max_new) {
                        finish = "length"; break;
                    }
                    next_round_seed = t_real;
                } else {
                    // Mismatch: the model's correction (t_real) is the
                    // committed token. KV cache is consistent up to and
                    // including cache_pos-1 (where we just appended). The
                    // next round seeds with t_real.
                    if (!commit_token(t_real)) {
                        cancelled = true; break;
                    }
                    if (is_stop(t_real)) { finish = "stop"; break; }
                    if ((int)generated.size() >= max_new) {
                        finish = "length"; break;
                    }
                    next_round_seed = t_real;
                    break;
                }
            }
            if (cancelled || !finish.empty()) break;

            // The next round's `cur_tok` is whichever token the model just
            // committed last. forward_token() uses it as input embedding
            // for the next decode position.
            cur_tok = next_round_seed;
        }
        if (finish.empty()) finish = "length";

        auto t_dec1 = std::chrono::high_resolution_clock::now();
        const double decode_ms =
            std::chrono::duration<double, std::milli>(t_dec1 - t_dec0).count();

        if (!cancelled && on_token) {
            std::string tail = stream_filter.flush();
            if (!tail.empty()) (void)on_token(tail);
        }
        strip_specials_inplace(out_text, arch_tag);

        result.text              = std::move(out_text);
        result.completion_tokens = (int)generated.size();
        result.decode_tok_per_sec =
            (decode_ms > 0 && decode_steps > 0)
                ? (1000.0 * (double)generated.size() / decode_ms)
                : 0.0;
        result.finish_reason = cancelled ? "cancel" : finish;
        return result;
    }

    void attach_medusa(const std::string& sidecar_path) {
        if (!loaded) {
            throw std::runtime_error(
                "rocm_cpp::Engine::attach_medusa: load() must succeed first");
        }
        if (medusa_attached) {
            rcpp_medusa_free_heads(&medusa);
            medusa_attached = false;
        }
        rcpp_status_t st =
            rcpp_medusa_load_h1b_sidecar(sidecar_path.c_str(), &model, &medusa);
        if (st != RCPP_OK) {
            std::memset(&medusa, 0, sizeof(medusa));
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "rocm_cpp::Engine::attach_medusa: load failed (%d) for %s",
                          (int)st, sidecar_path.c_str());
            throw std::runtime_error(buf);
        }
        medusa_attached = true;

        if (medusa.variant == RCPP_MEDUSA_VARIANT_RESIDUAL_MLP) {
            // Allocate hidden-sized fp16 scratch (one set, reused across
            // heads + draft rounds) and an fp32 logits scratch the size
            // of the base vocab. Plus a fixed all-ones fp16 buffer that
            // makes silu_glu_fp16(up=z, gate=ones) emit plain SiLU(z).
            HIP_CHECK(hipMalloc(&medusa_v2_h_in,      (size_t)hs * 2));
            HIP_CHECK(hipMalloc(&medusa_v2_h_out,     (size_t)hs * 2));
            HIP_CHECK(hipMalloc(&medusa_v2_tmp_f32,   (size_t)hs * 4));
            HIP_CHECK(hipMalloc(&medusa_v2_tmp_in,    (size_t)hs * 2));
            HIP_CHECK(hipMalloc(&medusa_v2_silu_act,  (size_t)hs * 2));
            HIP_CHECK(hipMalloc(&medusa_v2_delta_f32, (size_t)hs * 4));
            HIP_CHECK(hipMalloc(&medusa_v2_delta,     (size_t)hs * 2));
            HIP_CHECK(hipMalloc(&medusa_v2_ones,      (size_t)hs * 2));
            HIP_CHECK(hipMalloc(&medusa_v2_logits_f32, (size_t)V * 4));

            // Fill ones buffer once (host fp16(1.0) → device).
            std::vector<_Float16> host_ones((size_t)hs, _Float16(1.0f));
            HIP_CHECK(hipMemcpy(medusa_v2_ones, host_ones.data(),
                                (size_t)hs * 2, hipMemcpyHostToDevice));

            std::fprintf(stderr,
                         "[rocm_cpp::Engine] medusa attached (v2 residual-MLP): "
                         "%u heads, hidden=%u, dtype=%d (fp16 on device, "
                         "shared lm_head reused)\n",
                         medusa.num_heads, medusa.hidden_size,
                         (int)medusa.v2_dtype);
        } else {
            std::fprintf(stderr,
                         "[rocm_cpp::Engine] medusa attached (v1 vocab-proj): "
                         "%u heads, weight_format=%d, vocab=%u, hidden=%u\n",
                         medusa.num_heads, (int)medusa.weight_format,
                         medusa.vocab_size, medusa.hidden_size);
        }
    }

    bool is_medusa_attached() const noexcept { return medusa_attached; }

    Engine::MedusaVariant medusa_variant_tag() const noexcept {
        if (!medusa_attached)
            return Engine::MedusaVariant::Vocab;
        return medusa.variant == RCPP_MEDUSA_VARIANT_RESIDUAL_MLP
                   ? Engine::MedusaVariant::ResidualMLP
                   : Engine::MedusaVariant::Vocab;
    }

    // Residual-MLP head step (v2). Reads h_in_dev (fp16 [hidden]) and writes
    // h_out_dev (fp16 [hidden]) such that:
    //   h_out = h_in + w_out_k @ SiLU(w_in_k @ h_in)
    // h_in_dev and h_out_dev MUST be distinct device buffers (no aliasing).
    // Uses fp16 GEMVs against the on-device fp16 weights staged at attach time.
    void medusa_step_residual(uint32_t head_idx,
                              const _Float16* h_in_dev,
                              _Float16* h_out_dev)
    {
        const rcpp_medusa_head_t& hd_w = medusa.heads[head_idx];

        // 1) tmp_f32 = w_in_k @ h_in
        //    rcpp_fp16_gemv: W fp16 [M, K], x fp16 [K], y fp32 [M] — same
        //    kernel the base lm_head pass uses. Then cast back to fp16
        //    for the SiLU activation (residual stream stays in fp16 to
        //    keep memory pressure low for the chain GEMVs).
        RC_CHECK(rcpp_fp16_gemv(hd_w.w_in_dev, h_in_dev, medusa_v2_tmp_f32,
                                hs, hs, /*stream=*/nullptr));
        RC_CHECK(rcpp_fp32_to_fp16(medusa_v2_tmp_f32, medusa_v2_tmp_in,
                                   hs, nullptr));

        // 2) silu_act = SiLU(tmp_in). silu_glu_fp16(up, gate, y) computes
        //    y[i] = SiLU(up[i]) * gate[i]; with gate = all-ones fp16, this
        //    is plain SiLU.
        RC_CHECK(rcpp_silu_glu_fp16(medusa_v2_tmp_in, medusa_v2_ones,
                                    medusa_v2_silu_act, hs, nullptr));

        // 3) delta_f32 = w_out_k @ silu_act → fp16 cast.
        RC_CHECK(rcpp_fp16_gemv(hd_w.w_out_dev, medusa_v2_silu_act,
                                medusa_v2_delta_f32, hs, hs, nullptr));
        RC_CHECK(rcpp_fp32_to_fp16(medusa_v2_delta_f32, medusa_v2_delta,
                                   hs, nullptr));

        // 4) h_out = h_in + delta. Copy h_in into h_out, then in-place add.
        HIP_CHECK(hipMemcpyAsync(h_out_dev, h_in_dev, (size_t)hs * 2,
                                 hipMemcpyDeviceToDevice, nullptr));
        RC_CHECK(rcpp_residual_add_fp16(h_out_dev, medusa_v2_delta, hs,
                                        nullptr));
    }

    // Run one residual-MLP head and emit the argmax token id of the shared
    // lm_head logits at the resulting hidden state. `head_idx` ranges over
    // [0, medusa.num_heads). The chained-input variant takes the previous
    // head's h_out as h_in (drives the linear-chain t1 → t2 → t3 → t4).
    int medusa_residual_argmax(uint32_t head_idx,
                               const _Float16* h_in_dev,
                               _Float16* h_out_dev)
    {
        medusa_step_residual(head_idx, h_in_dev, h_out_dev);

        // Logits via the SHARED base.lm_head (= tied embedding matrix).
        // Same call shape as forward_token's terminal lm_head pass.
        RC_CHECK(rcpp_fp16_gemv(model.embedding_dev, h_out_dev,
                                medusa_v2_logits_f32, V, hs, nullptr));

        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(logits_host.data(), medusa_v2_logits_f32,
                            (size_t)V * 4, hipMemcpyDeviceToHost));
        int   best_idx = 0;
        float best_val = logits_host[0];
        for (int i = 1; i < V; ++i) {
            if (logits_host[i] > best_val) {
                best_val = logits_host[i];
                best_idx = i;
            }
        }
        return best_idx;
    }
};

// ---- Engine public surface ----

Engine::Engine(const Config& cfg) : impl_(std::make_unique<Impl>(cfg)) {}
Engine::~Engine() = default;

void Engine::load(const std::string& h1b_path,
                  const std::string& tokenizer_path) {
    impl_->load(h1b_path, tokenizer_path);
}

void Engine::unload() { impl_->unload(); }

bool Engine::is_loaded() const noexcept { return impl_ && impl_->loaded; }

GenerateResult Engine::generate(
    const std::string& prompt, const GenerateOptions& opts,
    std::function<bool(const std::string&)> on_token) {
    return impl_->generate(prompt, opts, std::move(on_token));
}

GenerateResult Engine::generate_from_tokens(
    const std::vector<int>& prompt_token_ids, const GenerateOptions& opts,
    std::function<bool(const std::string&)> on_token) {
    return impl_->generate_from_tokens(prompt_token_ids, opts,
                                       std::move(on_token));
}

std::vector<int> Engine::tokenize(const std::string& text, bool add_bos) const {
    return impl_->tokenize(text, add_bos);
}

int Engine::bos_id() const noexcept {
    return (impl_ && impl_->tokenizer)
        ? rcpp_tokenizer_bos_id(impl_->tokenizer)
        : -1;
}

int Engine::eos_id() const noexcept {
    return (impl_ && impl_->tokenizer)
        ? rcpp_tokenizer_eos_id(impl_->tokenizer)
        : -1;
}

Engine::Arch Engine::arch() const noexcept {
    if (!impl_ || !impl_->loaded) return Arch::BitNet;
    return impl_->is_qwen3 ? Arch::Qwen3 : Arch::BitNet;
}

std::vector<float> Engine::compute_nll(const std::vector<int>& token_ids) const {
    return impl_->compute_nll(token_ids);
}

void Engine::attach_medusa(const std::string& sidecar_path) {
    impl_->attach_medusa(sidecar_path);
}

bool Engine::is_medusa_attached() const noexcept {
    return impl_ && impl_->medusa_attached;
}

Engine::MedusaVariant Engine::medusa_variant() const noexcept {
    if (!impl_) return MedusaVariant::Vocab;
    return impl_->medusa_variant_tag();
}

GenerateResult Engine::generate_medusa(
    const std::vector<int>& prompt_token_ids,
    const GenerateOptions& opts,
    int num_speculative,
    std::function<bool(const std::string&)> on_token) {
    return impl_->generate_medusa(prompt_token_ids, opts, num_speculative,
                                  std::move(on_token));
}

}  // namespace rocm_cpp

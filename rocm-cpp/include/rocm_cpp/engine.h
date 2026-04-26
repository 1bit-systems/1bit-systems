// rocm_cpp::Engine — in-process C++ wrapper around librocm_cpp's HIP decode
// pipeline. Owns the loaded BitNet model, all device scratch buffers, and the
// per-layer KV cache. One Engine instance services one logical conversation
// or one-shot generation; concurrent generate() calls on the same instance
// are NOT safe — the caller must serialize. (Lemond's Router does this via
// a per-WrappedServer mutex, so the typical multi-tenant deployment is one
// Engine per loaded model with a shared lock around generate().)
//
// All errors throw std::runtime_error with a human-readable message —
// internal HIP / rcpp_status_t failures are converted at the API boundary so
// the caller doesn't need to thread a status enum through every call site.
// Engine never throws across the HIP runtime — exceptions only originate
// inside C++ helper code, never inside an extern "C" callback.

#ifndef ROCM_CPP_ENGINE_H
#define ROCM_CPP_ENGINE_H

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace rocm_cpp {

// Static knobs set once at construction. Mirrors the CLI flags on
// bitnet_decode that affect device-side memory layout — once load() has run,
// these can't change without an unload() / re-load() cycle.
struct Config {
    // KV cache compression mode. At most one may be true; both = throw.
    //   kv_int8  : INT8 K/V tensor + per-(pos, kv_head) FP16 scale. ~2x DRAM.
    //   kv_rotor : PQ3 packed-3bit (8 idx / 3 bytes). ~5.33x DRAM.
    bool kv_int8  = false;
    bool kv_rotor = false;

    // Pre-allocated KV slab depth in tokens. Caps the longest single
    // generate() prompt+completion. Defaults to 4096 (BitNet-2B-4T's
    // max_position_embeddings — RoPE is trained to here).
    int  max_context = 4096;
};

// Per-call sampler + stop config. Defaults match bitnet_decode's CLI defaults
// (greedy, no stops, no rep-penalty).
struct GenerateOptions {
    int      max_tokens   = 256;
    float    temperature  = 0.0f;   // 0 = greedy argmax (skips host sampler)
    int      top_k        = 0;      // 0 = disabled
    float    top_p        = 1.0f;   // 1.0 = disabled
    float    rep_penalty  = 1.0f;   // 1.0 = disabled
    int      rep_last_n   = 64;
    uint64_t seed         = 0;      // 0 = derive from steady_clock at call time
    std::vector<std::string> stop_sequences;
    // Optional caller-supplied stop token IDs. If non-empty, the decode loop
    // exits when the next sampled token matches any entry. If empty, the
    // engine falls back to its built-in Llama-3 stops (128001, 128009) —
    // preserves halo-1bit-2b regression baseline.
    std::vector<int> stop_token_ids;
};

struct GenerateResult {
    std::string text;
    int         prompt_tokens     = 0;
    int         completion_tokens = 0;
    double      prefill_tok_per_sec = 0.0;
    double      decode_tok_per_sec  = 0.0;
    // One of: "stop" (EOS or stop_sequences hit), "length" (max_tokens hit),
    // "cancel" (on_token returned false), "error" (forward pass failure).
    std::string finish_reason     = "stop";
};

class Engine {
public:
    explicit Engine(const Config& cfg = {});
    ~Engine();

    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&)                 = delete;
    Engine& operator=(Engine&&)      = delete;

    // Load weights + tokenizer. If tokenizer_path is empty, derives it from
    // h1b_path by replacing the trailing ".h1b" with ".htok" (or appending
    // ".htok" if the input doesn't end in .h1b).
    //
    // Throws std::runtime_error if either file fails to load, or if the
    // engine is already loaded (call unload() first to swap models).
    void load(const std::string& h1b_path,
              const std::string& tokenizer_path = "");

    // Idempotent — safe to call before load() or after a previous unload().
    void unload();

    bool is_loaded() const noexcept;

    // Run a generation. Resets the KV cache to position 0, prefills the
    // prompt, then decodes up to opts.max_tokens.
    //
    // on_token (optional) is invoked once per decoded token with the
    // incremental UTF-8 text delta (already detokenized + buffered against
    // multi-byte boundaries — same logic bitnet_decode uses for streaming
    // stdout). Return false to cancel; the partial result is still returned
    // with finish_reason = "cancel". An empty std::function (default) just
    // accumulates into result.text.
    //
    // NOT thread-safe across calls on the same Engine — caller serializes.
    //
    // The string overload BOS-prepends + byte-level-BPE-encodes the prompt.
    // Special tokens like <|eot_id|> are NOT recognized — they are emitted
    // as raw bytes. Callers that need bit-exact special-token IDs (e.g.
    // chat templates) must build a token-id stream and call
    // generate_from_tokens() instead.
    GenerateResult generate(
        const std::string& prompt,
        const GenerateOptions& opts = {},
        std::function<bool(const std::string& token_text)> on_token = {});

    // Token-id workhorse. The prompt_token_ids vector is fed verbatim into
    // prefill — caller is responsible for any BOS / chat-template / special
    // token framing. Same KV / sampler / streaming semantics as generate().
    GenerateResult generate_from_tokens(
        const std::vector<int>& prompt_token_ids,
        const GenerateOptions& opts = {},
        std::function<bool(const std::string& token_text)> on_token = {});

    // Byte-level BPE encode of `text`. NO special-token recognition —
    // "<|eot_id|>" comes back as ~10 BPE byte tokens, not id 128009.
    // If add_bos=true, the tokenizer's BOS id is prepended.
    // Empty vector if no tokenizer is loaded.
    [[nodiscard]] std::vector<int>
    tokenize(const std::string& text, bool add_bos = false) const;

    // Special-token ids from the loaded .htok header. Both return -1 if no
    // tokenizer is loaded. Llama-3-derived tokenizers (BitNet-2B-4T's
    // included) report bos=128000, eos=128001; the other specials
    // (<|eot_id|>=128009, <|start_header_id|>=128006, <|end_header_id|>=128007)
    // are not in the header so callers hard-code those ids.
    [[nodiscard]] int bos_id() const noexcept;
    [[nodiscard]] int eos_id() const noexcept;

    // Resolved model architecture, mirroring `model.arch` from the loaded
    // .h1b. Drives chat-template selection in callers (BitNet/Llama-3 vs
    // Qwen3/ChatML). Returns BitNet if no model is loaded — callers should
    // gate on is_loaded() first.
    enum class Arch { BitNet = 0, Qwen3 = 1 };
    [[nodiscard]] Arch arch() const noexcept;

    // Per-position negative log-likelihood. Resets the KV cache, prefills
    // greedily through `token_ids` one-token-at-a-time, and at each step
    // reads the host-side fp32 logits buffer to compute
    //   nll[t] = -log_softmax(logits_at_pos_t)[token_ids[t+1]].
    //
    // Result length = token_ids.size() - 1. Empty vector if fewer than 2
    // tokens are passed in. Throws std::runtime_error on forward failure
    // or if the model is not loaded.
    //
    // NOT thread-safe across calls on the same Engine — caller serializes
    // (same contract as generate()). Sliding-window evaluation is the
    // caller's responsibility; this entry point processes the input
    // contiguously up to max_context.
    [[nodiscard]] std::vector<float>
    compute_nll(const std::vector<int>& token_ids) const;

    // -------------------------------------------------------------------------
    // Medusa speculative decoding (opt-in).
    //
    // Loads a `.h1b-medusa` v1 OR v2 sidecar (see docs/h1b-medusa-format.md)
    // from disk and uploads heads to device memory. Must be called AFTER
    // load(); throws std::runtime_error on header / shape / IO failure. The
    // base model's hidden_size must match the sidecar header. (For v1 the
    // base vocab_size must also match; v2 reuses base.lm_head and ignores
    // the sidecar vocab field.)
    void attach_medusa(const std::string& sidecar_path);

    // True iff a sidecar has been attached since the last load() / unload().
    [[nodiscard]] bool is_medusa_attached() const noexcept;

    // On-disk wire format of the currently attached sidecar. Vocab = legacy
    // v1 per-head [vocab, hidden] ternary projection. ResidualMLP = v2
    // residual-MLP topology with shared base.lm_head. Returns Vocab when no
    // sidecar is attached — callers should gate on is_medusa_attached().
    enum class MedusaVariant { Vocab = 0, ResidualMLP = 1 };
    [[nodiscard]] MedusaVariant medusa_variant() const noexcept;

    // Run a generation with Medusa speculative decoding.
    //
    //   num_speculative — K, the number of speculative tokens drafted per
    //                     base step. Typical values 2..4. K <= 1 OR
    //                     !is_medusa_attached() falls back to the plain
    //                     generate_from_tokens() path.
    //
    // Wire shape (v0): linear chain. The base model's argmax becomes t0, and
    // each Medusa head's top-1 over the same hidden state contributes one
    // candidate at offset h+1 — total K = num_speculative tokens drafted per
    // round. A batched verify pass commits the longest greedy-matching
    // prefix; the remaining tail re-decodes one token at a time.
    GenerateResult generate_medusa(
        const std::vector<int>& prompt_token_ids,
        const GenerateOptions& opts,
        int num_speculative,
        std::function<bool(const std::string& token_text)> on_token = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rocm_cpp

#endif  // ROCM_CPP_ENGINE_H

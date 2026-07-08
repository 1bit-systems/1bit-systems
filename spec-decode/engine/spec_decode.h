#pragma once
// Speculative Decoding Engine — orchestrates draft model + target NPU model
//
// Flow:
//   1. Target model (NPU) generates first token + hidden states
//   2. Draft model predicts N=block_size candidate tokens
//   3. Target model verifies all candidates in ONE forward pass
//   4. Rejection sampling: accept/reject each draft token
//   5. Update KV caches, repeat from step 2
//
// Expected speedup: 1.5x-2.5x on NPU (94 tok/s -> 141-235 tok/s)
//
// Template parameter DraftModelT must provide:
//   - using State = ... (KV cache state type)
//   - using Config = ... (config type)
//   - forward(trunk_hidden, input_id, pos, state, draft_logits, draft_hidden)
//   - weights_loaded()
//   - load_weights(path)

#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstring>

#include "../draft/dspark_draft.h"  // DSpark is the DEFAULT draft model
#include "../draft/mtp_draft.h"     // MTP (Eagle3) for compatibility trait

// Default alias — DSpark is the production draft
using DefaultDraftConfig = DSparkDraftConfig;
using DefaultDraftModel = DSparkDraftModel;
using DefaultDraftState = DSparkDraftState;

// Target model interface — implemented by the NPU engine
struct TargetModelInterface {
    // Forward pass through target model
    // input_ids: [seq_len] token IDs
    // Returns: logits [vocab_size], hidden_states [num_layers, hidden_size]
    virtual void forward(
        const int32_t* input_ids,
        int32_t seq_len,
        float* logits,
        float* hidden_states
    ) = 0;

    // Forward pass with KV cache (for verification step)
    // input_ids: [n_tokens] — prompt + draft tokens to verify
    // past_len: number of tokens already in KV cache
    virtual void forward_with_kv(
        const int32_t* input_ids,
        int32_t n_tokens,
        int32_t past_len,
        float* logits,
        float* hidden_states
    ) = 0;

    // Get hidden states from specific layers (for draft model input)
    virtual void get_layer_hidden(
        const float* all_hidden,
        int32_t num_layers,
        const int32_t* target_layer_ids,
        int32_t num_target_layers,
        float* out
    ) = 0;

    // Commit the accepted prefix of the most recent forward_with_kv() call to the KV cache,
    // discarding any rejected trailing positions. start_pos must match the past_len passed to
    // that forward_with_kv() call; n_accept is how many of its input positions to keep.
    // Default no-op for implementations that don't need explicit rollback (e.g. simulated
    // targets that recompute KV from scratch every call).
    virtual void commit_accepted(int32_t /*start_pos*/, int32_t /*n_accept*/) {}

    virtual ~TargetModelInterface() = default;
};

struct SpecDecodeConfig {
    int32_t block_size = 7;           // N speculative tokens per forward
    float temperature = 0.0f;         // 0 = greedy, >0 = sampling
    int32_t max_new_tokens = 2048;
    int32_t vocab_size = 151936;
    int32_t hidden_size = 1024;
    int32_t num_kv_heads = 8;
    int32_t head_dim = 128;
    int32_t max_seq = 4096;
    int32_t num_target_layers = 5;
    int32_t num_draft_layers = 5;   // draft backbone layers (DSpark=5, Eagle3=1)
    int32_t target_layer_ids[5] = {1, 6, 12, 18, 24};
    int32_t eos_token_id = 151645;
    uint64_t seed = 42;
};

struct SpecDecodeStats {
    int64_t total_tokens = 0;
    int64_t accepted_draft_tokens = 0;
    int64_t total_draft_proposed = 0;
    int64_t verify_calls = 0;

    float acceptance_rate() const {
        return total_draft_proposed > 0
            ? (float)accepted_draft_tokens / total_draft_proposed
            : 0.0f;
    }
    float speedup_factor() const {
        return verify_calls > 0
            ? (float)total_tokens / verify_calls
            : 1.0f;
    }
};

// Resize trait — dispatches to the right resize() signature per draft state type
template <typename T> struct StateResizeTraits;

template <>
struct StateResizeTraits<DSparkDraftState> {
    static void resize(DSparkDraftState& s, int num_layers, int num_kv_heads, int head_dim, int block_size) {
        s.resize(num_layers, num_kv_heads, head_dim, block_size);
    }
};

template <>
struct StateResizeTraits<MTPDraftState> {
    static void resize(MTPDraftState& s, int /*num_layers*/, int num_kv_heads, int head_dim, int block_size) {
        s.resize(num_kv_heads, head_dim, block_size);
    }
};

// ─── Scratch buffer arena (pre-allocated, reused across generate calls) ─────
struct SpecDecodeArena {
    std::vector<float> logits;             // [max_vocab]
    std::vector<float> all_hidden;         // [max_model_layers * max_hidden]
    std::vector<float> target_hidden;      // [max_target_layers * max_hidden]
    std::vector<float> draft_logits;       // [max_block * max_vocab]
    std::vector<float> draft_hidden_step;  // [max_hidden]
    std::vector<int32_t> draft_tokens;     // [max_block]
    std::vector<int32_t> verify_input;     // [1 + max_block]
    std::vector<float> verify_hidden;      // [max_model_layers * max_hidden]
    std::vector<float> verify_logits;      // [(1 + max_block) * max_vocab]

    void ensure(int32_t block_size, int32_t hidden_size, int32_t vocab_size,
                int32_t num_target_layers, int32_t num_model_layers) {
        auto grow = [](auto& v, size_t n) { if (v.size() < n) v.resize(n); };
        grow(logits,           (size_t)vocab_size);
        grow(all_hidden,       (size_t)num_model_layers * hidden_size);
        grow(target_hidden,    (size_t)num_target_layers * hidden_size);
        grow(draft_logits,     (size_t)block_size * vocab_size);
        grow(draft_hidden_step, (size_t)hidden_size);
        grow(draft_tokens,     (size_t)block_size);
        grow(verify_input,     (size_t)(1 + block_size));
        grow(verify_hidden,    (size_t)num_model_layers * hidden_size);
        grow(verify_logits,    (size_t)(1 + block_size) * vocab_size);
    }
};

// SpeculativeDecoder — templated on draft model type for flexibility
template <typename DraftModelT = DefaultDraftModel,
          typename DraftStateT = DefaultDraftState,
          typename DraftConfigT = DefaultDraftConfig>
class SpeculativeDecoderT {
public:
    SpeculativeDecoderT(
        TargetModelInterface& target,
        DraftModelT& draft,
        const SpecDecodeConfig& cfg
    ) : target_(target), draft_(draft), cfg_(cfg),
        rng_(cfg.seed) {
        // Pre-allocate all buffers at construction time
        ensure_arena();
    }

    // Generate tokens with speculative decoding
    // prompt_ids: [prompt_len] input token IDs
    // output_ids: [max_output_len] output buffer (will be filled)
    // Returns: number of tokens generated
    int32_t generate(
        const int32_t* prompt_ids,
        int32_t prompt_len,
        int32_t* output_ids,
        int32_t max_output_len
    ) {
        stats_ = SpecDecodeStats{};
        std::copy(prompt_ids, prompt_ids + prompt_len, output_ids);
        int32_t generated = prompt_len;
        int32_t max_len = prompt_len + max_output_len;

        // Ensure arena is sized for current config (grows only if sizes changed)
        ensure_arena();

        const int32_t num_model_layers = 28; // Qwen3-0.6B full layers
        float* logits       = arena_.logits.data();
        float* all_hidden   = arena_.all_hidden.data();
        float* target_hidden = arena_.target_hidden.data();

        // Prefill: run target on full prompt
        target_.forward(
            prompt_ids, prompt_len,
            logits, all_hidden
        );

        // Extract target layer features for draft
        target_.get_layer_hidden(
            all_hidden, cfg_.num_target_layers,
            cfg_.target_layer_ids, cfg_.num_target_layers,
            target_hidden
        );

        // Sample first token greedily (avoid vector allocation — reuse logits buffer)
        int32_t next_token = argmax_inline(logits, cfg_.vocab_size);
        if (next_token == cfg_.eos_token_id) {
            output_ids[prompt_len] = next_token;
            stats_.total_tokens = 1;
            return prompt_len + 1;
        }
        output_ids[prompt_len] = next_token;
        generated = prompt_len + 1;
        stats_.total_tokens++;

        // Speculative decoding loop
        while (generated < max_len) {
            // 1. Draft: autoregress block_size candidate tokens
            float* draft_logits      = arena_.draft_logits.data();
            float* draft_hidden_step = arena_.draft_hidden_step.data();
            int32_t* draft_tokens    = arena_.draft_tokens.data();

            DraftStateT draft_state;
            StateResizeTraits<DraftStateT>::resize(draft_state, cfg_.num_draft_layers, cfg_.num_kv_heads, cfg_.head_dim, cfg_.block_size);
            int32_t draft_input_id = output_ids[generated - 1];
            for (int i = 0; i < cfg_.block_size; i++) {
                const float* draft_input_hidden = (i == 0) ? target_hidden : draft_hidden_step;
                draft_.forward(
                    draft_input_hidden,
                    draft_input_id,
                    /*pos=*/i,
                    draft_state,
                    draft_logits + (size_t)i * cfg_.vocab_size,
                    draft_hidden_step
                );
                draft_input_id = argmax_inline(
                    draft_logits + (size_t)i * cfg_.vocab_size, cfg_.vocab_size
                );
            }

            // Sample draft tokens greedily
            for (int i = 0; i < cfg_.block_size; i++) {
                float* logits_i = draft_logits + (size_t)i * cfg_.vocab_size;
                draft_tokens[i] = argmax_inline(logits_i, cfg_.vocab_size);
            }
            stats_.total_draft_proposed += cfg_.block_size;

            // 2. Verify: run target on [last_token | draft_tokens...] against existing KV cache
            int32_t verify_len = 1 + cfg_.block_size;
            int32_t past_len = generated - 1;
            int32_t* verify_input = arena_.verify_input.data();
            verify_input[0] = output_ids[generated - 1];
            std::copy(draft_tokens, draft_tokens + cfg_.block_size,
                      verify_input + 1);

            float* verify_hidden = arena_.verify_hidden.data();
            float* verify_logits = arena_.verify_logits.data();

            target_.forward_with_kv(
                verify_input, verify_len, past_len,
                verify_logits, verify_hidden
            );

            // 3. Rejection sampling — inline argmax for the verify step
            int n_accepted = 0;
            bool rejected_early = false;
            bool hit_eos = false;
            for (int i = 0; i < cfg_.block_size && generated < max_len; i++) {
                int32_t target_token = argmax_inline(
                    verify_logits + (size_t)i * cfg_.vocab_size,
                    cfg_.vocab_size
                );

                if (draft_tokens[i] == target_token) {
                    output_ids[generated] = draft_tokens[i];
                    generated++;
                    n_accepted++;
                    stats_.total_tokens++;
                    if (draft_tokens[i] == cfg_.eos_token_id) { hit_eos = true; break; }
                } else {
                    output_ids[generated] = target_token;
                    generated++;
                    stats_.total_tokens++;
                    rejected_early = true;
                    break;
                }
            }

            // Bonus token
            if (!rejected_early && !hit_eos && generated < max_len) {
                int32_t bonus_token = argmax_inline(
                    verify_logits + (size_t)cfg_.block_size * cfg_.vocab_size,
                    cfg_.vocab_size
                );
                output_ids[generated] = bonus_token;
                generated++;
                stats_.total_tokens++;
                if (bonus_token == cfg_.eos_token_id) hit_eos = true;
            }

            stats_.accepted_draft_tokens += n_accepted;
            stats_.verify_calls++;

            target_.commit_accepted(past_len, n_accepted + 1);

            if (hit_eos) return generated;

            // 4. Update target hidden states for next draft
            target_.get_layer_hidden(
                verify_hidden, cfg_.num_target_layers,
                cfg_.target_layer_ids, cfg_.num_target_layers,
                target_hidden
            );

            if (output_ids[generated - 1] == cfg_.eos_token_id)
                break;
        }

        return generated;
    }

    const SpecDecodeStats& stats() const { return stats_; }

private:
    TargetModelInterface& target_;
    DraftModelT& draft_;
    SpecDecodeConfig cfg_;
    std::mt19937 rng_;
    SpecDecodeStats stats_;
    SpecDecodeArena arena_;

    // Ensure arena is sized for the current config (grows only on config change)
    void ensure_arena() {
        arena_.ensure(cfg_.block_size, cfg_.hidden_size, cfg_.vocab_size,
                      cfg_.num_target_layers, 28);
    }

    // Inline argmax — no function call overhead, no std::distance
    static inline int32_t argmax_inline(const float* logits, int32_t n) {
        if (n <= 0) return 0;
        int32_t best_i = 0;
        float best_v = logits[0];
        for (int32_t i = 1; i < n; i++) {
            float v = logits[i];
            if (v > best_v) { best_v = v; best_i = i; }
        }
        return best_i;
    }
};

// Default instantiation using DSpark
using SpeculativeDecoder = SpeculativeDecoderT<DefaultDraftModel, DefaultDraftState, DefaultDraftConfig>;

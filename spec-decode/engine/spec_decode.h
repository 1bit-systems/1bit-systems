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

#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstring>

#include "../draft/mtp_draft.h"

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
        // 1 + accepted/per_verify * accept_rate
        return verify_calls > 0
            ? (float)total_tokens / verify_calls
            : 1.0f;
    }
};

class SpeculativeDecoder {
public:
    SpeculativeDecoder(
        TargetModelInterface& target,
        MTPDraftModel& draft,
        const SpecDecodeConfig& cfg
    ) : target_(target), draft_(draft), cfg_(cfg),
        rng_(cfg.seed) {
        state_.resize(cfg_.num_kv_heads, cfg_.head_dim, cfg_.max_seq);
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

        // Buffer for hidden states from target
        const int32_t num_model_layers = 28; // Qwen3-0.6B full layers
        std::vector<float> target_hidden(
            cfg_.num_target_layers * cfg_.hidden_size
        );

        // Prefill: run target on full prompt
        std::vector<float> logits(cfg_.vocab_size);
        std::vector<float> all_hidden(
            num_model_layers * cfg_.hidden_size
        );
        target_.forward(
            prompt_ids, prompt_len,
            logits.data(), all_hidden.data()
        );

        // Extract target layer features for draft
        target_.get_layer_hidden(
            all_hidden.data(), cfg_.num_target_layers,
            cfg_.target_layer_ids, cfg_.num_target_layers,
            target_hidden.data()
        );

        // Sample first token greedily
        int32_t next_token = argmax(logits.data(), cfg_.vocab_size);
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
            // 1. Draft: autoregress block_size candidate tokens. The draft has its own
            // tiny attention window that resets each round (positions 0..block_size-1) —
            // it only needs to model the immediate continuation, not full history; that's
            // the target model's job. Each step embeds the previous step's token internally
            // (real token id, not a stub) and reuses the same target_hidden trunk features
            // for the whole round (matches training: one fc-projected feature set per round).
            std::vector<int32_t> draft_tokens(cfg_.block_size);
            std::vector<float> draft_logits(
                (size_t)cfg_.block_size * cfg_.vocab_size
            );
            std::vector<float> draft_hidden_step(cfg_.hidden_size);

            MTPDraftState draft_state;
            draft_state.resize(cfg_.num_kv_heads, cfg_.head_dim, cfg_.block_size);
            int32_t draft_input_id = output_ids[generated - 1];
            for (int i = 0; i < cfg_.block_size; i++) {
                draft_.forward(
                    target_hidden.data(),
                    draft_input_id,
                    /*pos=*/i,
                    draft_state,
                    draft_logits.data() + (size_t)i * cfg_.vocab_size,
                    draft_hidden_step.data()
                );
                draft_input_id = argmax(
                    draft_logits.data() + (size_t)i * cfg_.vocab_size, cfg_.vocab_size
                );
            }

            // Sample draft tokens greedily
            for (int i = 0; i < cfg_.block_size; i++) {
                float* logits_i = draft_logits.data() + (size_t)i * cfg_.vocab_size;
                draft_tokens[i] = argmax(logits_i, cfg_.vocab_size);
            }
            stats_.total_draft_proposed += cfg_.block_size;

            // 2. Verify: run target on [last_token | draft_tokens...] against the existing
            // KV cache (past_len = generated - 1, since output_ids[generated-1] hasn't had
            // its own KV entry written yet — see commit_accepted below).
            int32_t verify_len = 1 + cfg_.block_size; // last + N draft
            int32_t past_len = generated - 1;
            std::vector<int32_t> verify_input(verify_len);
            verify_input[0] = output_ids[generated - 1];
            std::copy(draft_tokens.begin(), draft_tokens.end(),
                      verify_input.begin() + 1);

            const int32_t num_model_layers = 28;
            std::vector<float> verify_hidden(
                num_model_layers * cfg_.hidden_size
            );
            std::vector<float> verify_logits(
                (size_t)verify_len * cfg_.vocab_size
            );

            target_.forward_with_kv(
                verify_input.data(), verify_len, past_len,
                verify_logits.data(), verify_hidden.data()
            );

            // 3. Rejection sampling
            int n_accepted = 0;
            bool rejected_early = false;
            bool hit_eos = false;
            for (int i = 0; i < cfg_.block_size && generated < max_len; i++) {
                // Greedy acceptance: if draft == argmax(target), accept
                int32_t target_token = argmax(
                    verify_logits.data() + (size_t)i * cfg_.vocab_size,
                    cfg_.vocab_size
                );

                if (draft_tokens[i] == target_token) {
                    // Accept draft token
                    output_ids[generated] = draft_tokens[i];
                    generated++;
                    n_accepted++;
                    stats_.total_tokens++;
                    if (draft_tokens[i] == cfg_.eos_token_id) { hit_eos = true; break; }
                } else {
                    // Reject: use target's token instead. Do NOT also emit a bonus token —
                    // verify_logits[block_size] predicts what follows ALL draft tokens, which
                    // is no longer a valid continuation once one of them was rejected.
                    output_ids[generated] = target_token;
                    generated++;
                    stats_.total_tokens++;
                    rejected_early = true;
                    break;
                }
            }

            // Bonus token: target's prediction after all draft tokens were accepted.
            // Only valid when nothing was rejected — always at verify_logits[block_size].
            if (!rejected_early && !hit_eos && generated < max_len) {
                int32_t bonus_token = argmax(
                    verify_logits.data() + (size_t)cfg_.block_size * cfg_.vocab_size,
                    cfg_.vocab_size
                );
                output_ids[generated] = bonus_token;
                generated++;
                stats_.total_tokens++;
                if (bonus_token == cfg_.eos_token_id) hit_eos = true;
            }

            stats_.accepted_draft_tokens += n_accepted;
            stats_.verify_calls++;

            // Commit this round's KV cache: keep entries for the previously-uncached last
            // token plus every accepted/corrective token (n_accepted+1 positions starting at
            // past_len); the bonus token (if any) gets its own KV entry on the next round.
            target_.commit_accepted(past_len, n_accepted + 1);

            if (hit_eos) return generated;

            // 4. Update target hidden states for next draft
            target_.get_layer_hidden(
                verify_hidden.data(), cfg_.num_target_layers,
                cfg_.target_layer_ids, cfg_.num_target_layers,
                target_hidden.data()
            );

            if (output_ids[generated - 1] == cfg_.eos_token_id)
                break;
        }

        return generated;
    }

    const SpecDecodeStats& stats() const { return stats_; }

private:
    TargetModelInterface& target_;
    MTPDraftModel& draft_;
    MTPDraftState state_;
    SpecDecodeConfig cfg_;
    std::mt19937 rng_;
    SpecDecodeStats stats_;

    int32_t argmax(const float* logits, int32_t n) {
        return std::distance(logits, std::max_element(logits, logits + n));
    }
};

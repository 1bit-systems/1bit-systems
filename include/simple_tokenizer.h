#pragma once
// simple_tokenizer.h — shared tokenizer wrapper, extracted from
// tools/unified_server.cpp so backends that need to bridge token IDs to text
// (e.g. FlmBackend, which talks to an external text-only subprocess) can use
// the exact same vocabulary as the rest of the server.
//
// Three tokenizer sources, in priority order:
//   1. ZINC's GGUF-native BPE tokenizer (load_from_gguf) — reads the real
//      vocab embedded in the active model's own GGUF file, so arbitrary
//      downloaded models get correct tokenization with no extra files
//      needed. Requires the ZINC integration (see CMakeLists.txt
//      ZINC_ENABLED) — falls through when unavailable.
//   2. RCPP BPE tokenizer (.htok) — the project's own Zaya-specific format.
//   3. ASCII/UTF-8 byte passthrough — last-resort fallback, degrades badly
//      for real vocabularies (kept only so the server never hard-fails).

#include "rocm_cpp/tokenizer.h"
#include <string>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cstddef>

#ifndef ZINC_DISABLED
extern "C" {
    void* zinc_tokenizer_init(const char* gguf_path);
    void zinc_tokenizer_destroy(void* state);
    uint32_t zinc_tokenizer_bos_id(void* state);
    uint32_t zinc_tokenizer_eos_id(void* state);
    long long zinc_tokenizer_encode(void* state, const char* text, uint32_t* out_ids, size_t out_cap);
    long long zinc_tokenizer_decode(void* state, const uint32_t* ids, size_t n_ids, char* out_buf, size_t out_cap);
}
#endif

struct SimpleTokenizer {
    int bos_id = 2;
    int eos_id = 1;
    bool use_bpe = false;
    rcpp_tokenizer_t* bpe_tok = nullptr;
    bool use_gguf_native = false;
#ifndef ZINC_DISABLED
    void* gguf_tok_state = nullptr;
#endif

    // Try loading the real tokenizer embedded in a GGUF file (ZINC's native
    // BPE tokenizer). Returns false (leaving any existing tokenizer state
    // untouched) if ZINC isn't built in, the path isn't a GGUF file, or it
    // has no embedded vocab — caller should keep using whatever was loaded
    // before, not treat this as fatal.
    bool load_from_gguf(const std::string& gguf_path) {
#ifndef ZINC_DISABLED
        if (gguf_path.size() < 5 || gguf_path.substr(gguf_path.size() - 5) != ".gguf")
            return false;
        void* st = zinc_tokenizer_init(gguf_path.c_str());
        if (!st) {
            fprintf(stderr, "SimpleTokenizer: no usable embedded vocab in %s\n", gguf_path.c_str());
            return false;
        }
        unload_gguf_native();
        gguf_tok_state = st;
        use_gguf_native = true;
        bos_id = (int)zinc_tokenizer_bos_id(st);
        eos_id = (int)zinc_tokenizer_eos_id(st);
        fprintf(stderr, "Loaded GGUF-native tokenizer from %s (BOS=%d, EOS=%d)\n",
                gguf_path.c_str(), bos_id, eos_id);
        return true;
#else
        (void)gguf_path;
        return false;
#endif
    }

    bool load(const std::string& vocab_path) {
        if (vocab_path.size() >= 5 && vocab_path.substr(vocab_path.size() - 5) == ".htok") {
            rcpp_tokenizer_t* tok = nullptr;
            rcpp_status_t st = rcpp_tokenizer_load(vocab_path.c_str(), &tok);
            if (st == RCPP_OK && tok) {
                bpe_tok = tok;
                use_bpe = true;
                bos_id = rcpp_tokenizer_bos_id(bpe_tok);
                eos_id = rcpp_tokenizer_eos_id(bpe_tok);
                fprintf(stderr, "Loaded BPE tokenizer from %s (BOS=%d, EOS=%d)\n",
                        vocab_path.c_str(), bos_id, eos_id);
                return true;
            }
            fprintf(stderr, "Failed to load BPE tokenizer from %s\n", vocab_path.c_str());
        }
        fprintf(stderr, "No .htok at %s — using fallback ASCII tokenizer\n", vocab_path.c_str());
        return false;
    }

    void unload_gguf_native() {
#ifndef ZINC_DISABLED
        if (gguf_tok_state) { zinc_tokenizer_destroy(gguf_tok_state); gguf_tok_state = nullptr; }
#endif
        use_gguf_native = false;
    }

    ~SimpleTokenizer() {
        if (bpe_tok) rcpp_tokenizer_free(bpe_tok);
        unload_gguf_native();
    }

    std::vector<int> encode(const std::string& text) {
#ifndef ZINC_DISABLED
        if (use_gguf_native && gguf_tok_state) {
            std::vector<uint32_t> r(4096);
            long long n = zinc_tokenizer_encode(gguf_tok_state, text.c_str(), r.data(), r.size());
            if (n > 0) return std::vector<int>(r.begin(), r.begin() + n);
            return {bos_id};
        }
#endif
        if (use_bpe && bpe_tok) {
            std::vector<int> r(4096);
            size_t out_n = 0;
            rcpp_status_t st = rcpp_tokenizer_encode(bpe_tok, text.c_str(), text.size(),
                                                      1, r.data(), r.size(), &out_n);
            if (st == RCPP_OK && out_n > 0) {
                r.resize(out_n);
                return r;
            }
            return {bos_id};
        }
        // Fallback: ASCII + UTF-8 byte passthrough
        std::vector<int> r = {bos_id};
        for (unsigned char c : text) {
            if (c >= 32 && c <= 126)
                r.push_back((int)c + 100);
            else if (c != 0)
                r.push_back((int)c + 200);
        }
        return r;
    }

    /// Encode with per-token log-probabilities (for cascade strategy).
    /// Returns token IDs and fills logprobs_out with corresponding logprobs.
    std::vector<int> encode_with_logprobs(const std::string& text,
                                           std::vector<double>& logprobs_out) {
        logprobs_out.clear();
        if (use_bpe && bpe_tok) {
            std::vector<int> r(4096);
            std::vector<double> lp(4096);
            size_t out_n = 0;
            rcpp_status_t st = rcpp_tokenizer_encode_with_logprobs(
                bpe_tok, text.c_str(), text.size(),
                1, r.data(), lp.data(), r.size(), &out_n);
            if (st == RCPP_OK && out_n > 0) {
                r.resize(out_n);
                logprobs_out.assign(lp.begin(), lp.begin() + out_n);
                return r;
            }
            return {bos_id};
        }
        // Fallback (also used for GGUF-native, which has no logprob API):
        // encode without logprobs
        auto tokens = encode(text);
        logprobs_out.resize(tokens.size(), -1.0);
        return tokens;
    }

    std::string decode(const std::vector<int>& tokens) {
#ifndef ZINC_DISABLED
        if (use_gguf_native && gguf_tok_state) {
            std::vector<uint32_t> ids(tokens.begin(), tokens.end());
            std::string r(8192, '\0');
            long long n = zinc_tokenizer_decode(gguf_tok_state, ids.data(), ids.size(), r.data(), r.size());
            if (n >= 0) { r.resize(n); return r; }
            return "";
        }
#endif
        if (use_bpe && bpe_tok) {
            std::string r(4096, '\0');
            size_t out_len = 0;
            rcpp_status_t st = rcpp_tokenizer_decode(bpe_tok, tokens.data(), tokens.size(),
                                                      r.data(), r.size(), &out_len);
            if (st == RCPP_OK && out_len > 0) {
                r.resize(out_len);
                return r;
            }
            return "";
        }
        // Fallback: ASCII + UTF-8 byte passthrough
        std::string r;
        for (int v : tokens) {
            if (v == bos_id || v == eos_id) continue;
            if (v > 100 && v < 200)
                r += (char)(v - 100);
            else if (v > 200 && v < 456)
                r += (char)(v - 200);
            else {
                r += '['; r += std::to_string(v); r += ']';
            }
        }
        return r;
    }
};

// Shared global instance — defined once in src/simple_tokenizer.cpp, used
// by unified_server.cpp and by any backend (e.g. FlmBackend) that needs to
// bridge token IDs to text using the exact same vocabulary the rest of the
// server uses.
extern SimpleTokenizer g_tokenizer;

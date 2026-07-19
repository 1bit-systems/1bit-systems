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
    bool load_from_gguf(const std::string& gguf_path);

    bool load(const std::string& vocab_path);

    void unload_gguf_native();

    ~SimpleTokenizer();

    std::vector<int> encode(const std::string& text);

    /// Encode with per-token log-probabilities (for cascade strategy).
    /// Returns token IDs and fills logprobs_out with corresponding logprobs.
    std::vector<int> encode_with_logprobs(const std::string& text,
                                           std::vector<double>& logprobs_out);

    std::string decode(const std::vector<int>& tokens);
};

// Shared global instance — defined once in src/simple_tokenizer.cpp, used
// by unified_server.cpp and by any backend (e.g. FlmBackend) that needs to
// bridge token IDs to text using the exact same vocabulary the rest of the
// server uses.
extern SimpleTokenizer g_tokenizer;

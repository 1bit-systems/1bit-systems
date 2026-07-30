#pragma once
// BitNet tokenizer — pure C ABI. No runtime Python. No JSON parser.
//
// Consumes a side-file (.tok) produced by the halo-1bit exporter at
// build time. Binary layout (little-endian):
//
//   magic     : char[4] = "HTOK"
//   version   : u32     = 1
//   vocab_size: u32
//   merges_n  : u32
//   bos_id    : i32
//   eos_id    : i32
//   --- vocab table, vocab_size entries, in id order ---
//     token_len : u16
//     token_bytes: u8[token_len]
//   --- merges table, merges_n entries ---
//     a_id    : i32
//     b_id    : i32
//     new_id  : i32   (the merged token id; rank = insertion order, lower = priority)
//
// The encoder is minimal:
//   * Byte-level BPE identical to tiktoken / LLaMA-3 at the merge level.
//   * LLaMA-3 / cl100k_base regex pre-tokenizer (contractions + Unicode
//     categories + number grouping + whitespace handling).
//   * whitespace is handled the way LLaMA-3 bytes it out (space = 0x20)
//
// For English prompt input to BitNet-b1.58-2B-4T this produces the
// same token IDs as the reference tokenizer on clean text (no
// unicode apostrophes / emoji / mixed-script content).

#ifndef ROCM_CPP_TOKENIZER_H
#define ROCM_CPP_TOKENIZER_H

#include "rocm_cpp/ck_gemm.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rcpp_tokenizer rcpp_tokenizer_t;

// Load a .tok file from disk. Caller owns the returned handle.
rcpp_status_t
rcpp_tokenizer_load(const char* tok_path, rcpp_tokenizer_t** out);

void
rcpp_tokenizer_free(rcpp_tokenizer_t* tok);

// Encode text (UTF-8) to token IDs. Writes up to max_out ids; returns
// the actual count in *out_count. If the buffer is too small, fills
// it up to max_out and still reports the true count. Prepends BOS if
// add_bos != 0.
rcpp_status_t
rcpp_tokenizer_encode(const rcpp_tokenizer_t* tok,
                      const char* text, size_t text_len,
                      int add_bos,
                      int* ids_out, size_t max_out, size_t* out_count);

// Decode a span of IDs back to UTF-8. Writes up to max_bytes; *out_len
// gets the true byte count (may exceed max_bytes; buffer stays capped).
rcpp_status_t
rcpp_tokenizer_decode(const rcpp_tokenizer_t* tok,
                      const int* ids, size_t n_ids,
                      char* out, size_t max_bytes, size_t* out_len);

// BOS / EOS IDs as stored in the .tok header.
int rcpp_tokenizer_bos_id(const rcpp_tokenizer_t* tok);
int rcpp_tokenizer_eos_id(const rcpp_tokenizer_t* tok);

// ── Logprob API — merge-rank-based token frequency scoring (fixes #81) ──

/// Return the pseudo-logprob for a single token ID.
double rcpp_tokenizer_logprob(const rcpp_tokenizer_t* tok, int token_id);

/// Encode text and return per-token pseudo-logprobs.
rcpp_status_t
rcpp_tokenizer_encode_with_logprobs(const rcpp_tokenizer_t* tok,
                                     const char* text, size_t text_len,
                                     int add_bos,
                                     int* ids_out, double* logprobs_out,
                                     size_t max_out, size_t* out_count);

#ifdef __cplusplus
}
#endif
#endif  // ROCM_CPP_TOKENIZER_H

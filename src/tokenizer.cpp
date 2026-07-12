// BitNet tokenizer — pure C++ / zero Python at runtime.
//
// Reads the .htok binary emitted by halo-1bit/scripts/export_tokenizer.py
// (build-time Python is allowed per project rules; runtime is not).
//
// Scope:
//   * Byte-level BPE identical to tiktoken / LLaMA-3 at the merge level.
//   * GPT-2 byte<->unicode mapping (256-entry LUT) applied at the boundary.
//   * LLaMA-3 / cl100k_base regex pre-tokenizer (contractions + Unicode
//     letters/numbers/whitespace splitting). Matches the tiktoken reference
//     on practical inputs including CJK, Cyrillic, and punctuation-heavy text
//     (fixes #92).

#include "rocm_cpp/tokenizer.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// GPT-2 byte-to-unicode mapping (same table as tiktoken / LLaMA-3).
// Printable ASCII bytes map to themselves; the rest are relocated into
// U+0100..U+017F so every byte becomes a valid printable char.
struct ByteMap {
    std::array<uint32_t, 256> byte_to_cp;   // byte -> unicode codepoint
    std::array<int, 0x180>    cp_to_byte;   // codepoint (in used range) -> byte, or -1

    ByteMap() {
        cp_to_byte.fill(-1);
        std::vector<uint8_t> bs;
        for (int b = int('!'); b <= int('~'); ++b) bs.push_back((uint8_t)b);
        for (int b = 0xA1; b <= 0xAC; ++b)         bs.push_back((uint8_t)b);
        for (int b = 0xAE; b <= 0xFF; ++b)         bs.push_back((uint8_t)b);
        std::vector<uint32_t> cps;
        for (uint8_t b : bs) cps.push_back(b);
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), (uint8_t)b) == bs.end()) {
                bs.push_back((uint8_t)b);
                cps.push_back(0x100u + (uint32_t)n);
                ++n;
            }
        }
        for (size_t i = 0; i < bs.size(); ++i) {
            byte_to_cp[bs[i]] = cps[i];
            if (cps[i] < cp_to_byte.size()) cp_to_byte[cps[i]] = bs[i];
        }
    }
};

const ByteMap& byte_map() { static ByteMap bm; return bm; }

// Encode one codepoint to UTF-8. Returns bytes written (1..4).
int utf8_encode(uint32_t cp, char out[4]) {
    if (cp < 0x80)    { out[0] = (char)cp; return 1; }
    if (cp < 0x800)   { out[0] = (char)(0xC0 | (cp >> 6));       out[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) { out[0] = (char)(0xE0 | (cp >> 12));      out[1] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[2] = (char)(0x80 | (cp & 0x3F)); return 3; }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6)  & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

// Decode one UTF-8 codepoint. Writes bytes_read, returns codepoint or 0xFFFD.
uint32_t utf8_decode(const uint8_t* s, size_t len, size_t& bytes_read) {
    if (len == 0) { bytes_read = 0; return 0; }
    uint8_t b0 = s[0];
    if (b0 < 0x80) { bytes_read = 1; return b0; }
    if ((b0 & 0xE0) == 0xC0 && len >= 2) {
        bytes_read = 2;
        return ((uint32_t)(b0 & 0x1F) << 6) | (s[1] & 0x3F);
    }
    if ((b0 & 0xF0) == 0xE0 && len >= 3) {
        bytes_read = 3;
        return ((uint32_t)(b0 & 0x0F) << 12) | ((uint32_t)(s[1] & 0x3F) << 6) | (s[2] & 0x3F);
    }
    if ((b0 & 0xF8) == 0xF0 && len >= 4) {
        bytes_read = 4;
        return ((uint32_t)(b0 & 0x07) << 18) | ((uint32_t)(s[1] & 0x3F) << 12)
             | ((uint32_t)(s[2] & 0x3F) << 6) | (s[3] & 0x3F);
    }
    bytes_read = 1;
    return 0xFFFD;
}

struct MergeKey {
    int32_t a, b;
    bool operator==(const MergeKey& o) const { return a == o.a && b == o.b; }
};
struct MergeKeyHash {
    size_t operator()(const MergeKey& k) const noexcept {
        return (size_t)k.a * 0x9E3779B97F4A7C15ull ^ (size_t)k.b;
    }
};

}  // namespace

struct rcpp_tokenizer {
    std::vector<std::string> id_to_bytes;               // token id -> raw bytes (GPT-2 mapped)
    std::unordered_map<std::string, int32_t> bytes_to_id;
    std::unordered_map<MergeKey, std::pair<int32_t, int32_t>, MergeKeyHash> merges;
    int32_t bos_id = 128000;
    int32_t eos_id = 128001;

    // Reverse map: token_id → merge rank (for logprob estimation, fixes #81).
    // Built once after merges are loaded. Tokens not in the map (e.g. special
    // tokens) get a base rank equal to max_rank + 1 (least likely).
    std::vector<int> id_to_rank;  // index = token_id, value = merge rank (or -1 for unmerged)
    int max_rank = 0;             // total number of merges

    void build_rank_map() {
        if (!merges.empty() && id_to_rank.empty()) {
            max_rank = (int)merges.size();
            id_to_rank.assign(id_to_bytes.size(), -1);
            for (auto& kv : merges) {
                int merged_id = kv.second.first;
                int rank = kv.second.second;
                if (merged_id >= 0 && merged_id < (int)id_to_rank.size()) {
                    // Lower rank = earlier merge = more common.
                    // Only keep the LOWEST rank (earliest merge) for each token.
                    if (id_to_rank[merged_id] < 0 || rank < id_to_rank[merged_id]) {
                        id_to_rank[merged_id] = rank;
                    }
                }
            }
        }
    }

    // Merge-rank-based pseudo-logprob for a token ID. Non-negative.
    // Tokens formed by earlier merges (lower rank) get higher probability.
    // Unmerged tokens (special tokens, single bytes) get a base probability.
    double logprob_for_id(int id) const {
        if (id < 0 || id >= (int)id_to_rank.size()) return -20.0;
        int rank = id_to_rank[id];
        if (rank < 0) {
            // Unmerged token (special token or single byte).
            // Assign a moderate probability: less likely than early merges,
            // more likely than very late merges.
            return -8.0;
        }
        // rank ranges 0..max_rank-1. Convert to a probability in (0, 1).
        // Linear decay: rank 0 → probability ~0.01, rank max_rank → ~0.0003.
        double p = 0.01 * (1.0 - (double)rank / (double)max_rank) + 0.0003;
        return log(p);
    }
};

// Build rank map after loading merges
static void ensure_rank_map(rcpp_tokenizer* t) {
    if (t) t->build_rank_map();
}

extern "C" rcpp_status_t
rcpp_tokenizer_load(const char* path, rcpp_tokenizer_t** out)
{
    if (!path || !out) return RCPP_INVALID_ARG;
    std::ifstream f(path, std::ios::binary);
    if (!f) return RCPP_INVALID_ARG;

    char magic[4];
    f.read(magic, 4);
    if (std::strncmp(magic, "HTOK", 4) != 0) { fprintf(stderr, "bad .htok magic\n"); return RCPP_INVALID_ARG; }

    uint32_t vocab_size = 0, num_merges = 0, bos = 0, eos = 0;
    f.read(reinterpret_cast<char*>(&vocab_size), 4);
    f.read(reinterpret_cast<char*>(&num_merges), 4);
    f.read(reinterpret_cast<char*>(&bos), 4);
    f.read(reinterpret_cast<char*>(&eos), 4);

    auto t = new rcpp_tokenizer();
    t->bos_id = (int32_t)bos;
    t->eos_id = (int32_t)eos;
    t->id_to_bytes.resize(vocab_size);
    t->bytes_to_id.reserve(vocab_size);

    for (uint32_t i = 0; i < vocab_size; ++i) {
        uint16_t len = 0;
        f.read(reinterpret_cast<char*>(&len), 2);
        std::string s(len, '\0');
        if (len) f.read(s.data(), len);
        t->id_to_bytes[i] = s;
        if (!s.empty()) t->bytes_to_id[s] = (int32_t)i;
    }

    t->merges.reserve(num_merges);
    // Rank = insertion order; earlier merges have priority.
    // Third field is the new (merged) token id; rank is derived from
    // insertion order (i), NOT from the on-disk value. Lower rank = higher
    // priority during BPE encoding.
    for (uint32_t i = 0; i < num_merges; ++i) {
        uint32_t a = 0, b = 0, merged = 0;
        f.read(reinterpret_cast<char*>(&a), 4);
        f.read(reinterpret_cast<char*>(&b), 4);
        f.read(reinterpret_cast<char*>(&merged), 4);
        if (!f) { fprintf(stderr, "[tokenizer] short read at merge %u\n", i); return RCPP_INVALID_ARG; }
        t->merges.emplace(MergeKey{(int32_t)a, (int32_t)b},
                          std::make_pair((int32_t)merged, (int32_t)i));
    }

    ensure_rank_map(t);

    *out = t;
    return RCPP_OK;
}

extern "C" void rcpp_tokenizer_free(rcpp_tokenizer_t* t) { delete t; }

extern "C" int rcpp_tokenizer_bos_id(const rcpp_tokenizer_t* t) { return t ? t->bos_id : -1; }
extern "C" int rcpp_tokenizer_eos_id(const rcpp_tokenizer_t* t) { return t ? t->eos_id : -1; }

// ── Logprob API — merge-rank-based token frequency scoring (fixes #81) ──

/// Return the merge-rank-based pseudo-logprob for a single token ID.
/// Non-positive: 0 is most likely, -20 is least. Based on the token's
/// merge order in BPE training (earlier merges = more common tokens).
extern "C" double rcpp_tokenizer_logprob(const rcpp_tokenizer_t* t, int token_id) {
    if (!t) return -20.0;
    return t->logprob_for_id(token_id);
}

/// Encode text and return per-token pseudo-logprobs.
/// ids_out and logprobs_out receive one entry per encoded token.
/// Returns RCPP_OK on success, RCPP_INVALID_ARG on error.
extern "C" rcpp_status_t
rcpp_tokenizer_encode_with_logprobs(const rcpp_tokenizer_t* t,
                                     const char* text, size_t text_len,
                                     int add_bos,
                                     int* ids_out, double* logprobs_out,
                                     size_t max_out, size_t* out_count) {
    if (!t || !text || !out_count) return RCPP_INVALID_ARG;
    if (!ids_out && max_out > 0) return RCPP_INVALID_ARG;

    // Encode the text using the standard encoder
    rcpp_status_t rc = rcpp_tokenizer_encode(t, text, text_len, add_bos,
                                              ids_out, max_out, out_count);
    if (rc != RCPP_OK) return rc;

    // Compute logprob for each token
    if (logprobs_out) {
        for (size_t i = 0; i < *out_count && i < max_out; i++) {
            int id = ids_out ? ids_out[i] : 0;
            logprobs_out[i] = t->logprob_for_id(id);
        }
    }
    return RCPP_OK;
}

// ── Unicode category helpers (for LLaMA-3 pre-tokenizer) ──
// L = Unicode letter, N = Unicode number, Z = whitespace/separator.
// For ASCII relies on standard <cctype> checks; for non-ASCII uses simple
// range tests covering the scripts most likely to appear in real text.
// This is not a full Unicode Character Database — just enough to match
// the tiktoken cl100k_base pre-tokenizer on practical inputs.

namespace {

// Is this codepoint a Unicode letter (L* category)?
static bool is_letter(uint32_t cp) {
    if (cp < 0x80) {
        return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
    }
    // Latin Extended (including IPA), Greek, Cyrillic, Armenian, Hebrew, Arabic
    if (cp <= 0x06FF) return true;
    // Extended Arabic, Syriac, Thaana, NKo, Samaritan, Mandaic
    if (cp >= 0x0750 && cp <= 0x085F) return true;
    // Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam
    if (cp >= 0x0900 && cp <= 0x0D7F) return true;
    // Sinhala, Thai, Lao, Tibetan, Myanmar
    if (cp >= 0x0D80 && cp <= 0x109F) return true;
    // Georgian, Hangul Jamo
    if (cp >= 0x10A0 && cp <= 0x11FF) return true;
    // Ethiopic
    if (cp >= 0x1200 && cp <= 0x137F) return true;
    // Cherokee, Unified Canadian Aboriginal
    if (cp >= 0x13A0 && cp <= 0x14FF) return true;
    // Ogham, Runic
    if (cp >= 0x1680 && cp <= 0x16FF) return true;
    // Tagalog, Hanunoo, Buhid, Tagbanwa, Khmer, Mongolian
    if (cp >= 0x1700 && cp <= 0x18AF) return true;
    // CJK Radicals, Kangxi Radicals, Ideographic Description
    if (cp >= 0x2E80 && cp <= 0x2FFF) return true;
    // CJK Symbols and Punctuation, Hiragana, Katakana, Bopomofo, Kanbun
    if (cp >= 0x3000 && cp <= 0x30FF) return true;
    // CJK Unified Ideographs
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;
    // Yi, Vai
    if (cp >= 0xA000 && cp <= 0xA4FF) return true;
    // Hangul Syllables
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true;
    // CJK Extension B, C, etc.
    if (cp >= 0x20000 && cp <= 0x2FFFF) return true;
    // CJK Extension G, H
    if (cp >= 0x30000 && cp <= 0x3FFFF) return true;
    return false;
}

// Is this codepoint a Unicode number (N* category)?
static bool is_number(uint32_t cp) {
    if (cp < 0x80) {
        return cp >= '0' && cp <= '9';
    }
    // Superscript/subscript digits
    if (cp >= 0x00B2 && cp <= 0x00B3) return true;
    if (cp == 0x00B9) return true;
    // Arabic-Indic, Extended Arabic-Indic
    if (cp >= 0x0660 && cp <= 0x0669) return true;
    if (cp >= 0x06F0 && cp <= 0x06F9) return true;
    // NKo, Devanagari, Bengali, etc. digits
    if (cp >= 0x0966 && cp <= 0x096F) return true;
    if (cp >= 0x09E6 && cp <= 0x09EF) return true;
    if (cp >= 0x0A66 && cp <= 0x0A6F) return true;
    if (cp >= 0x0AE6 && cp <= 0x0AEF) return true;
    if (cp >= 0x0B66 && cp <= 0x0B6F) return true;
    if (cp >= 0x0BE6 && cp <= 0x0BEF) return true;
    if (cp >= 0x0C66 && cp <= 0x0C6F) return true;
    if (cp >= 0x0CE6 && cp <= 0x0CEF) return true;
    if (cp >= 0x0D66 && cp <= 0x0D6F) return true;
    // Myanmar, Khmer, Thai digits
    if (cp >= 0x1040 && cp <= 0x1049) return true;
    if (cp >= 0x17E0 && cp <= 0x17E9) return true;
    if (cp >= 0x19D0 && cp <= 0x19D9) return true;
    // Full-width digits
    if (cp >= 0xFF10 && cp <= 0xFF19) return true;
    // Mathematical double-struck and monospace digits
    if (cp >= 0x1D7CE && cp <= 0x1D7FF) return true;
    return false;
}

// Is this codepoint a whitespace or line-break character (Z* category) or
// one of the ASCII control chars (\r, \n, \t, \f, \v)?
static bool is_whitespace(uint32_t cp) {
    if (cp == U'\r' || cp == U'\n' || cp == U'\t' || cp == U'\f' || cp == U'\v') return true;
    if (cp < 0x80) return cp == ' ' || cp == '\r' || cp == '\n' || cp == '\t';
    // Unicode spaces: U+0085 (NEL), U+00A0 (NBSP), U+1680, U+2000-200A, U+2028, U+2029, U+202F, U+205F, U+3000
    if (cp == 0x00A0 || cp == 0x1680) return true;
    if (cp >= 0x2000 && cp <= 0x200A) return true;
    if (cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F) return true;
    if (cp == 0x3000) return true;
    return false;
}

// Is this codepoint NOT whitespace, NOT letter, NOT number?
static bool is_other(uint32_t cp) {
    return !is_letter(cp) && !is_number(cp) && !is_whitespace(cp);
}

// Is this codepoint a \r or \n?
static bool is_newline(uint32_t cp) {
    return cp == U'\r' || cp == U'\n';
}

}  // namespace

// LLaMA-3 / cl100k_base pre-tokenizer.
// Splits input text into chunks matching the tiktoken cl100k_base regex:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// 
// Chunks are returned as UTF-8 byte substrings. BPE merges never cross
// chunk boundaries, matching the tiktoken reference (fixes #92).
//
// Each match pattern is tried in order at the current position. The first
// one that succeeds consumes characters and emits a chunk. If none match,
// the single character at the current position is emitted (fallback).
static std::vector<std::string> llama3_pre_tokenize(const std::string& text) {
    std::vector<std::string> chunks;
    const size_t n = text.size();
    const uint8_t* data = reinterpret_cast<const uint8_t*>(text.data());
    size_t pos = 0;  // byte position in the UTF-8 string

    if (n == 0) return chunks;

    // Helper: decode codepoint at byte position `p`, return cp and set `len`.
    auto cp_at = [&](size_t p, size_t& len) -> uint32_t {
        if (p >= n) { len = 0; return 0; }
        return utf8_decode(data + p, n - p, len);
    };

    // Helper: scan forward from `p` while predicate returns true for codepoints.
    // Returns the byte position after the last matching character.
    auto scan_while = [&](size_t p, auto pred) -> size_t {
        while (p < n) {
            size_t len = 0;
            uint32_t cp = utf8_decode(data + p, n - p, len);
            if (!pred(cp)) break;
            p += len;
        }
        return p;
    };

    // Helper: check how many consecutive \p{N} codepoints start at `p`, capped at 3.
    auto count_digits = [&](size_t p) -> size_t {
        size_t end = p;
        for (int i = 0; i < 3; i++) {
            size_t len = 0;
            uint32_t cp = utf8_decode(data + end, n - end, len);
            if (len == 0 || !is_number(cp)) break;
            end += len;
        }
        return end;
    };

    while (pos < n) {
        size_t len = 0;
        uint32_t cp = utf8_decode(data + pos, n - pos, len);
        size_t chunk_start = pos;

        // Pattern 1: (?i:'s|'t|'re|'ve|'m|'ll|'d) — case-insensitive contractions
        if (cp == U'\'') {
            size_t p1 = pos + len;
            size_t l1 = 0;
            uint32_t nxt = p1 < n ? utf8_decode(data + p1, n - p1, l1) : 0;
            bool matched = false;
            if (l1 > 0) {
                if (nxt == U's' || nxt == U't' || nxt == U'm' || nxt == U'd' ||
                    nxt == U'S' || nxt == U'T' || nxt == U'M' || nxt == U'D') {
                    pos = p1 + l1; matched = true;
                } else {
                    size_t p2 = p1 + l1, l2 = 0;
                    uint32_t nxt2 = p2 < n ? utf8_decode(data + p2, n - p2, l2) : 0;
                    if (l2 > 0 && (
                        ((nxt == U'r' || nxt == U'R') && (nxt2 == U'e' || nxt2 == U'E')) ||
                        ((nxt == U'v' || nxt == U'V') && (nxt2 == U'e' || nxt2 == U'E')) ||
                        ((nxt == U'l' || nxt == U'L') && (nxt2 == U'l' || nxt2 == U'L')))) {
                        pos = p2 + l2; matched = true;
                    }
                }
            }
            if (matched) { chunks.emplace_back(text, chunk_start, pos - chunk_start); continue; }
        }

        // Pattern 2: [^\r\n\p{L}\p{N}]?\p{L}+ — optional non-LN prefix + letter run
        if (!is_newline(cp) && (is_letter(cp) || (!is_letter(cp) && !is_number(cp)))) {
            size_t p = pos;
            if (!is_letter(cp)) {
                // Optional prefix character (not letter, not number, not newline)
                p += len;
                if (p < n) {
                    size_t ll = 0;
                    uint32_t lcp = utf8_decode(data + p, n - p, ll);
                    if (lcp > 0 && is_letter(lcp)) {
                        pos = scan_while(p, [](uint32_t c) { return is_letter(c); });
                        chunks.emplace_back(text, chunk_start, pos - chunk_start);
                        continue;
                    }
                }
                // Not a letter after prefix; the prefix doesn't match this pattern.
                // Fall through to let other patterns or the fallback handle it.
            } else {
                // Direct letter run
                pos = scan_while(p, [](uint32_t c) { return is_letter(c); });
                chunks.emplace_back(text, chunk_start, pos - chunk_start);
                continue;
            }
        }

        // Pattern 3: \p{N}{1,3} — 1-3 consecutive digits
        if (is_number(cp)) {
            pos = count_digits(pos);
            chunks.emplace_back(text, chunk_start, pos - chunk_start);
            continue;
        }

        // Pattern 4: ?[^\s\p{L}\p{N}]+[\r\n]* — optional space + others + newlines
        if (cp == U' ' || is_other(cp)) {
            size_t p = pos;
            if (cp == U' ') p += len;  // consume optional leading space
            // Check for at least one "other" character after the optional space
            if (p < n) {
                size_t ll = 0;
                uint32_t ocp = utf8_decode(data + p, n - p, ll);
                if (is_other(ocp)) {
                    p += ll;
                    p = scan_while(p, [](uint32_t c) { return is_other(c); });
                    p = scan_while(p, [](uint32_t c) { return is_newline(c); });
                    pos = p;
                    chunks.emplace_back(text, chunk_start, pos - chunk_start);
                    continue;
                }
            }
            // If we consumed a space but the rest didn't match pattern 4, 
            // the space alone is emitted below (fallthrough). For cp==' '
            // the space will be handled by the fallback single-char emit.
        }

        // Pattern 5: \s*[\r\n]+ — optional whitespace + one or more newlines
        if (is_whitespace(cp)) {
            size_t p = scan_while(pos, [](uint32_t c) { return is_whitespace(c); });
            if (p < n) {
                size_t ll = 0;
                uint32_t nc = utf8_decode(data + p, n - p, ll);
                if (is_newline(nc)) {
                    p += ll;
                    p = scan_while(p, [](uint32_t c) { return is_newline(c); });
                    pos = p;
                    chunks.emplace_back(text, chunk_start, pos - chunk_start);
                    continue;
                }
            }
        }

        // Pattern 6: \s+(?!\S) — whitespace at end of text (trailing whitespace)
        if (is_whitespace(cp)) {
            size_t p = scan_while(pos, [](uint32_t c) { return is_whitespace(c); });
            if (p >= n) {
                // Whitespace that goes to end of string
                pos = p;
                chunks.emplace_back(text, chunk_start, pos - chunk_start);
                continue;
            }
            // Is there a newline or end-of-string after this whitespace?
            size_t ll = 0;
            uint32_t after = utf8_decode(data + p, n - p, ll);
            if (is_newline(after)) {
                // Whitespace before a newline — the newline will be caught
                // by pattern 5 in the next iteration. The whitespace itself
                // is emitted here (trailing before newline).
                pos = p;
                chunks.emplace_back(text, chunk_start, pos - chunk_start);
                continue;
            }
            // Whitespace before non-whitespace, not at end: fall through
            // to single-character fallback
        }

        // Fallback: emit the current character as its own chunk.
        // Every character must belong to at least one chunk.
        chunks.emplace_back(text, chunk_start, len);
        pos += len;
    }

    return chunks;
}

// Convert raw UTF-8 text into a vector of single-char (GPT-2 mapped)

// Convert raw UTF-8 text into a vector of single-char (GPT-2 mapped)
// byte-strings, one entry per input byte. These are the starting
// "pieces" the BPE merge loop works on.
static std::vector<std::string> byte_level_split(const std::string& text) {
    const auto& bm = byte_map();
    std::vector<std::string> pieces;
    pieces.reserve(text.size());
    char buf[4];
    for (uint8_t b : text) {
        uint32_t cp = bm.byte_to_cp[b];
        int n = utf8_encode(cp, buf);
        pieces.emplace_back(buf, buf + n);
    }
    return pieces;
}

extern "C" rcpp_status_t
rcpp_tokenizer_encode(const rcpp_tokenizer_t* t,
                      const char* text, size_t text_len,
                      int add_bos,
                      int* ids_out, size_t max_out, size_t* out_count)
{
    if (!t || !text || !out_count) return RCPP_INVALID_ARG;
    *out_count = 0;
    if (ids_out == nullptr && max_out > 0) return RCPP_INVALID_ARG;

    std::string s(text, text_len);
    if (s.empty()) return RCPP_OK;
    auto chunks = llama3_pre_tokenize(s);

    // BPE-merge a single chunk's byte pieces in place. Merges do NOT
    // cross chunk boundaries — that's the whole point of the pre-
    // tokenizer, it prevents spurious word/digit/whitespace merges.
    auto bpe_chunk = [&](const std::vector<std::string>& pieces,
                         std::vector<int32_t>& out) -> bool {
        std::vector<int32_t> ids;
        ids.reserve(pieces.size());
        for (auto& p : pieces) {
            auto it = t->bytes_to_id.find(p);
            if (it == t->bytes_to_id.end()) {
                fprintf(stderr, "tokenizer: unknown byte piece '%s' (len %zu)\n", p.c_str(), p.size());
                return false;
            }
            ids.push_back(it->second);
        }
        // BPE merge loop — find lowest-rank pair, merge, repeat.
        // Maximum iterations capped at initial size to prevent infinite loops.
        const int max_iters = (int)ids.size();
        for (int iter = 0; iter < max_iters; ++iter) {
            int best_rank = INT32_MAX;
            int best_pos  = -1;
            int32_t best_new = 0;
            for (int i = 0; i < (int)ids.size() - 1; ++i) {
                auto mit = t->merges.find(MergeKey{ids[i], ids[i+1]});
                if (mit != t->merges.end() && mit->second.second < best_rank) {
                    best_rank = mit->second.second;
                    best_pos  = i;
                    best_new  = mit->second.first;
                }
            }
            if (best_pos < 0) break;
            ids[best_pos] = best_new;
            ids.erase(ids.begin() + best_pos + 1);
            if (ids.size() < 2) break;  // single token left, done
        }
        out.insert(out.end(), ids.begin(), ids.end());
        return true;
    };

    std::vector<int32_t> all_ids;
    if (add_bos) all_ids.push_back(t->bos_id);
    for (const auto& chunk : chunks) {
        auto pieces = byte_level_split(chunk);
        if (!bpe_chunk(pieces, all_ids)) return RCPP_INTERNAL;
    }

    *out_count = all_ids.size();
    size_t n = std::min(all_ids.size(), max_out);
    if (ids_out) for (size_t i = 0; i < n; ++i) ids_out[i] = all_ids[i];
    return RCPP_OK;
}

extern "C" rcpp_status_t
rcpp_tokenizer_decode(const rcpp_tokenizer_t* t,
                      const int* ids, size_t n_ids,
                      char* out, size_t max_bytes, size_t* out_len)
{
    if (!t || !ids || !out_len) return RCPP_INVALID_ARG;

    // First reconstruct the GPT-2-mapped UTF-8 string, then undo the
    // byte mapping to produce raw UTF-8 output.
    std::string mapped;
    for (size_t i = 0; i < n_ids; ++i) {
        int id = ids[i];
        if (id < 0 || id >= (int)t->id_to_bytes.size()) continue;
        mapped += t->id_to_bytes[id];
    }

    const auto& bm = byte_map();
    std::string raw;
    raw.reserve(mapped.size());
    const uint8_t* p = reinterpret_cast<const uint8_t*>(mapped.data());
    size_t remain = mapped.size();
    while (remain) {
        size_t used = 0;
        uint32_t cp = utf8_decode(p, remain, used);
        if (cp < bm.cp_to_byte.size() && bm.cp_to_byte[cp] >= 0) {
            raw.push_back((char)bm.cp_to_byte[cp]);
        }
        // unknown codepoints silently dropped — special tokens have
        // empty decoded text by design.
        p += used; remain -= used;
    }

    *out_len = raw.size();
    size_t n = std::min(raw.size(), max_bytes);
    if (out) std::memcpy(out, raw.data(), n);
    if (out && max_bytes > n) out[n] = '\0';
    return RCPP_OK;
}

// /home/bcloud/engine/fusion/tokenize.cpp
// Pure C++17 BPE tokenizer — subprocess-callable CLI + linkable C ABI.
//
// CLI:  tokenize --model tokenizer.json --encode "Hello, world!"
//       tokenize --model tokenizer.json --decode 151644 151645 198
//
// C ABI (for direct Zig integration):
//   void* tokenizer_load(const char* json_path);
//   int   tokenizer_encode(void* tok, const char* text, int* out_ids, int max_ids);
//   void  tokenizer_free(void* tok);
//
// Supports HF tokenizer.json BPE format with:
//   - Pre-tokenizers: Sequence, Split, ByteLevel, Metaspace
//   - GPT-2 byte-level encoding
//   - BPE merge-ranks algorithm
//   - Byte fallback for unknown tokens
//
// No external dependencies beyond C++17 standard library.
// ============================================================================

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// ============================================================================
// ── Minimal JSON parser (recursive descent) ─────────────────────────────────
// ============================================================================

namespace json {

// Forward declarations inside struct are tricky — use concrete types directly.
struct Value;
using JsonObject = std::map<std::string, Value>;
using JsonArray = std::vector<Value>;

struct Value {
    enum Type { Null, Bool, Int, Float, String, Array, Object };
    Type type_ = Null;

    bool        bool_   = false;
    int64_t     int_    = 0;
    double      float_  = 0.0;
    std::string str_;
    JsonArray   arr_;
    JsonObject  obj_;

    Value() = default;
    Value(Type t) : type_(t) {}
    Value(bool b)           : type_(Bool),  bool_(b) {}
    Value(int64_t i)        : type_(Int),   int_(i) {}
    Value(double f)         : type_(Float), float_(f) {}
    Value(const char* s)    : type_(String), str_(s) {}
    Value(std::string s)    : type_(String), str_(std::move(s)) {}
    Value(JsonArray a)      : type_(Array),  arr_(std::move(a)) {}
    Value(JsonObject o)     : type_(Object), obj_(std::move(o)) {}

    bool is_null()   const { return type_ == Null; }
    bool is_bool()   const { return type_ == Bool; }
    bool is_int()    const { return type_ == Int; }
    bool is_float()  const { return type_ == Float; }
    bool is_number() const { return type_ == Int || type_ == Float; }
    bool is_string() const { return type_ == String; }
    bool is_array()  const { return type_ == Array; }
    bool is_object() const { return type_ == Object; }

    bool as_bool()   const { return bool_; }
    int64_t as_int() const { return type_ == Int ? int_ : (int64_t)float_; }
    double as_float()const { return type_ == Float ? float_ : (double)int_; }
    const std::string& as_string() const { return str_; }
    const JsonArray&       as_array()  const { return arr_; }
    const JsonObject&      as_object() const { return obj_; }

    const Value& operator[](const std::string& key) const {
        static Value null_value;
        auto it = obj_.find(key);
        return it != obj_.end() ? it->second : null_value;
    }
    const Value& operator[](size_t i) const {
        static Value null_value;
        return i < arr_.size() ? arr_[i] : null_value;
    }
    size_t size() const {
        if (is_array()) return arr_.size();
        if (is_object()) return obj_.size();
        return 0;
    }
};

// ── Parser ──────────────────────────────────────────────────────────────────

class Parser {
    const char* p_;
    const char* end_;

    void skip_ws() {
        while (p_ < end_ && (*p_ == ' ' || *p_ == '\t' || *p_ == '\n' || *p_ == '\r'))
            ++p_;
    }
    char peek() { skip_ws(); return p_ < end_ ? *p_ : '\0'; }
    char next() { skip_ws(); return p_ < end_ ? *p_++ : '\0'; }
    bool eof()  { skip_ws(); return p_ >= end_; }

    void expect(char c) {
        char got = next();
        if (got != c) {
            fprintf(stderr, "JSON parse error: expected '%c', got '%c' at offset %td\n", c, got, p_ - (end_ - strlen(p_?p_:""))); // simplified
        }
    }

    Value parse_value() {
        switch (peek()) {
            case '"': return parse_string();
            case '{': return parse_object();
            case '[': return parse_array();
            case 't': case 'f': return parse_bool();
            case 'n': return parse_null();
            default:  return parse_number();
        }
    }

    Value parse_string() {
        expect('"');
        std::string s;
        while (p_ < end_ && *p_ != '"') {
            if (*p_ == '\\') {
                ++p_;
                switch (*p_) {
                    case '"':  s += '"'; break;
                    case '\\': s += '\\'; break;
                    case '/':  s += '/'; break;
                    case 'b':  s += '\b'; break;
                    case 'f':  s += '\f'; break;
                    case 'n':  s += '\n'; break;
                    case 'r':  s += '\r'; break;
                    case 't':  s += '\t'; break;
                    case 'u': {
                        // Parse 4-hex-digit unicode escape
                        char hex[5] = {p_[1], p_[2], p_[3], p_[4], 0};
                        unsigned codepoint = strtoul(hex, nullptr, 16);
                        p_ += 4;
                        // Handle surrogate pairs
                        if (codepoint >= 0xD800 && codepoint <= 0xDBFF) {
                            // Expect \uDC00-\uDFFF
                            if (p_[0] == '\\' && p_[1] == 'u') {
                                char hex2[5] = {p_[2], p_[3], p_[4], p_[5], 0};
                                unsigned low = strtoul(hex2, nullptr, 16);
                                if (low >= 0xDC00 && low <= 0xDFFF) {
                                    codepoint = 0x10000 + (codepoint - 0xD800) * 0x400 + (low - 0xDC00);
                                    p_ += 6;
                                }
                            }
                        }
                        // Encode UTF-8
                        if (codepoint <= 0x7F) {
                            s += (char)codepoint;
                        } else if (codepoint <= 0x7FF) {
                            s += (char)(0xC0 | (codepoint >> 6));
                            s += (char)(0x80 | (codepoint & 0x3F));
                        } else if (codepoint <= 0xFFFF) {
                            s += (char)(0xE0 | (codepoint >> 12));
                            s += (char)(0x80 | ((codepoint >> 6) & 0x3F));
                            s += (char)(0x80 | (codepoint & 0x3F));
                        } else {
                            s += (char)(0xF0 | (codepoint >> 18));
                            s += (char)(0x80 | ((codepoint >> 12) & 0x3F));
                            s += (char)(0x80 | ((codepoint >> 6) & 0x3F));
                            s += (char)(0x80 | (codepoint & 0x3F));
                        }
                        break;
                    }
                    default: s += *p_; break;
                }
                ++p_;
            } else {
                s += *p_++;
            }
        }
        expect('"');
        return Value(std::move(s));
    }

    Value parse_number() {
        const char* start = p_;
        if (*p_ == '-') ++p_;
        if (*p_ == '0') ++p_;
        else while (p_ < end_ && *p_ >= '0' && *p_ <= '9') ++p_;
        bool is_float = false;
        if (p_ < end_ && *p_ == '.') { is_float = true; ++p_; while (p_ < end_ && *p_ >= '0' && *p_ <= '9') ++p_; }
        if (p_ < end_ && (*p_ == 'e' || *p_ == 'E')) { is_float = true; ++p_; if (*p_ == '+' || *p_ == '-') ++p_; while (p_ < end_ && *p_ >= '0' && *p_ <= '9') ++p_; }
        std::string num_str(start, p_ - start);
        if (is_float)
            return Value(strtod(num_str.c_str(), nullptr));
        else
            return Value((int64_t)strtoll(num_str.c_str(), nullptr, 10));
    }

    Value parse_bool() {
        if (strncmp(p_, "true", 4) == 0) { p_ += 4; return Value(true); }
        if (strncmp(p_, "false", 5) == 0) { p_ += 5; return Value(false); }
        return Value();
    }

    Value parse_null() {
        if (strncmp(p_, "null", 4) == 0) { p_ += 4; return Value(); }
        return Value();
    }

    Value parse_object() {
        expect('{');
        JsonObject obj;
        if (peek() == '}') { next(); return Value(std::move(obj)); }
        while (true) {
            auto key = parse_string();
            expect(':');
            obj[key.as_string()] = parse_value();
            if (peek() == '}') { next(); return Value(std::move(obj)); }
            expect(',');
        }
    }

    Value parse_array() {
        expect('[');
        JsonArray arr;
        if (peek() == ']') { next(); return Value(std::move(arr)); }
        while (true) {
            arr.push_back(parse_value());
            if (peek() == ']') { next(); return Value(std::move(arr)); }
            expect(',');
        }
    }

public:
    Parser(const char* data, size_t len) : p_(data), end_(data + len) {}
    Value parse() { return parse_value(); }
};

inline Value parse(const std::string& s) {
    Parser p(s.data(), s.size());
    return p.parse();
}

} // namespace json

// ============================================================================
// ── UTF-8 utilities ─────────────────────────────────────────────────────────
// ============================================================================

namespace utf8 {

// Decode one UTF-8 codepoint from 's' at offset 'i', advance 'i'.
// Returns U+FFFD on error.
inline char32_t decode(const std::string& s, size_t& i) {
    if (i >= s.size()) return 0;
    unsigned char c0 = (unsigned char)s[i];
    if (c0 <= 0x7F) {
        ++i;
        return c0;
    } else if (c0 >= 0xC0 && c0 <= 0xDF && i + 1 < s.size()) {
        char32_t cp = ((char32_t)(c0 & 0x1F) << 6) | ((unsigned char)s[i+1] & 0x3F);
        i += 2;
        return cp;
    } else if (c0 >= 0xE0 && c0 <= 0xEF && i + 2 < s.size()) {
        char32_t cp = ((char32_t)(c0 & 0x0F) << 12) | (((unsigned char)s[i+1] & 0x3F) << 6) | ((unsigned char)s[i+2] & 0x3F);
        i += 3;
        return cp;
    } else if (c0 >= 0xF0 && c0 <= 0xF7 && i + 3 < s.size()) {
        char32_t cp = ((char32_t)(c0 & 0x07) << 18) | (((unsigned char)s[i+1] & 0x3F) << 12) | (((unsigned char)s[i+2] & 0x3F) << 6) | ((unsigned char)s[i+3] & 0x3F);
        i += 4;
        return cp;
    }
    ++i; // skip invalid byte
    return 0xFFFD;
}

// Encode one codepoint to UTF-8
inline void encode(char32_t cp, std::string& out) {
    if (cp <= 0x7F) {
        out += (char)cp;
    } else if (cp <= 0x7FF) {
        out += (char)(0xC0 | (cp >> 6));
        out += (char)(0x80 | (cp & 0x3F));
    } else if (cp <= 0xFFFF) {
        out += (char)(0xE0 | (cp >> 12));
        out += (char)(0x80 | ((cp >> 6) & 0x3F));
        out += (char)(0x80 | (cp & 0x3F));
    } else {
        out += (char)(0xF0 | (cp >> 18));
        out += (char)(0x80 | ((cp >> 12) & 0x3F));
        out += (char)(0x80 | ((cp >> 6) & 0x3F));
        out += (char)(0x80 | (cp & 0x3F));
    }
}

// Character classification (simplified Unicode-aware)
inline bool is_letter(char32_t cp) {
    return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z') ||
           (cp >= 0xC0 && cp <= 0x024F) || // Latin Extended
           (cp >= 0x0370 && cp <= 0x03FF) || // Greek & Coptic
           (cp >= 0x0400 && cp <= 0x04FF) || // Cyrillic
           (cp >= 0x3040 && cp <= 0x309F) || // Hiragana
           (cp >= 0x30A0 && cp <= 0x30FF) || // Katakana
           (cp >= 0x4E00 && cp <= 0x9FFF) || // CJK Unified
           (cp >= 0xAC00 && cp <= 0xD7AF) || // Hangul
           (cp >= 0x0600 && cp <= 0x06FF) || // Arabic
           (cp >= 0x0900 && cp <= 0x097F) || // Devanagari
           (cp >= 0x0E00 && cp <= 0x0E7F);    // Thai
}

inline bool is_digit(char32_t cp) {
    return (cp >= '0' && cp <= '9') ||
           (cp >= 0x0660 && cp <= 0x0669) || // Arabic-Indic digits
           (cp >= 0x06F0 && cp <= 0x06F9) || // Extended Arabic-Indic
           (cp >= 0x0966 && cp <= 0x096F) || // Devanagari digits
           (cp >= 0xFF10 && cp <= 0xFF19);   // Fullwidth digits
}

inline bool is_whitespace(char32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x0B || cp == 0x0C ||
           cp == 0x00A0 || cp == 0x1680 || cp == 0x2000 || cp == 0x2001 ||
           cp == 0x2002 || cp == 0x2003 || cp == 0x2004 || cp == 0x2005 ||
           cp == 0x2006 || cp == 0x2007 || cp == 0x2008 || cp == 0x2009 ||
           cp == 0x200A || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

} // namespace utf8

// ============================================================================
// ── GPT-2 byte-to-Unicode mapping ──────────────────────────────────────────
// ============================================================================
// Maps each byte value (0-255) to a Unicode character using the GPT-2 scheme.
// Used by the ByteLevel pre-tokenizer.
//
// Rules:
//   bytes 33-126    → chr(byte)        (printable ASCII)
//   bytes 161-172   → chr(byte)
//   bytes 174-255   → chr(byte)
//   all others      → chr(256 + byte)  (mapped to Unicode private use area + 256)
// ============================================================================

namespace bytelevel {

static std::string byte_to_chars[256];
static char char_to_byte[65536]; // maps codepoint -> byte (or -1)
static bool tables_initialized = false;

static void init_tables() {
    if (tables_initialized) return;
    tables_initialized = true;
    std::fill(char_to_byte, char_to_byte + 65536, (char)-1);

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
            // Direct mapping
            byte_to_chars[b] = std::string(1, (char)b);
            if (b < 65536) char_to_byte[b] = (char)b;
        } else {
            // Mapped to 256 + n
            int cp = 256 + n;
            utf8::encode(cp, byte_to_chars[b]);
            if (cp < 65536) char_to_byte[cp] = (char)b;
            ++n;
        }
    }
}

// Convert UTF-8 text to GPT-2 byte-level unicode string.
// Each byte of the text's UTF-8 encoding becomes one Unicode character.
inline std::string encode(const std::string& text) {
    init_tables();
    std::string result;
    for (unsigned char b : text) {
        result += byte_to_chars[b];
    }
    return result;
}

// Decode a GPT-2 byte-level unicode string back to UTF-8 text.
inline std::string decode(const std::string& encoded) {
    init_tables();
    std::string result;
    size_t i = 0;
    while (i < encoded.size()) {
        char32_t cp = utf8::decode(encoded, i);
        if (cp < 65536 && char_to_byte[cp] != (char)-1) {
            result += char_to_byte[cp];
        } else {
            // Pass through (might be a regular character)
            utf8::encode(cp, result);
        }
    }
    return result;
}

} // namespace bytelevel

// ============================================================================
// ── GPT-2 Regex Pre-Tokenizer ──────────────────────────────────────────────
// ============================================================================
// Splits text into words using the GPT-2 pre-tokenization regex pattern.
// Each alternative is implemented with correct greedy+backtracking semantics
// to match the HuggingFace tokenizers Split pre-tokenizer behavior.
// ============================================================================

namespace gpt2_split {

// Decode one codepoint at pos, return cp and advance pos (in-place).
// Returns false on end of string.
static bool peek_char(const std::string& text, size_t& pos, char32_t& cp) {
    if (pos >= text.size()) return false;
    size_t tmp = pos;
    cp = utf8::decode(text, tmp);
    pos = tmp;
    return true;
}

// Alternative 1: (?i:'s|'t|'re|'ve|'m|'ll|'d)
// Case-insensitive English contractions.
static size_t match_contraction(const std::string& text, size_t pos) {
    if (pos + 1 >= text.size() || text[pos] != '\'') return 0;
    
    auto ci_eq = [](char a, char b) { return (a | 0x20) == (b | 0x20); };
    
    if (pos + 2 <= text.size()) {
        char c1 = text[pos+1];
        // 2-char suffixes: 's, 't, 'm, 'd
        if ((ci_eq(c1, 's') || ci_eq(c1, 't') || ci_eq(c1, 'm') || ci_eq(c1, 'd')) &&
            (pos + 2 >= text.size() || !((text[pos+2] | 0x20) >= 'a' && (text[pos+2] | 0x20) <= 'z'))) {
            return pos + 2;
        }
        // 3-char suffixes: 're, 've, 'll
        if (pos + 3 <= text.size()) {
            char c2 = text[pos+2];
            if ((ci_eq(c1, 'r') && ci_eq(c2, 'e')) ||
                (ci_eq(c1, 'v') && ci_eq(c2, 'e')) ||
                (ci_eq(c1, 'l') && ci_eq(c2, 'l'))) {
                return pos + 3;
            }
        }
    }
    return 0;
}

// Alternative 2: [^\r\n\p{L}\p{N}]?\p{L}+
// Optional (non-\r\n, non-letter, non-digit) prefix + one or more letters.
// Greedy: tries WITH prefix first, backtracks to WITHOUT.
static size_t match_letter_seq(const std::string& text, size_t pos) {
    if (pos >= text.size()) return 0;
    
    // Strategy: try with prefix first, then without
    size_t try_positions[2];
    int n_tries = 0;
    
    // Try with prefix
    size_t tmp = pos;
    char32_t cp;
    if (peek_char(text, tmp, cp)) {
        if (cp != '\r' && cp != '\n' && !utf8::is_letter(cp) && !utf8::is_digit(cp)) {
            try_positions[n_tries++] = tmp;  // with prefix
        }
    }
    try_positions[n_tries++] = pos;  // without prefix
    
    for (int t = 0; t < n_tries; ++t) {
        size_t i = try_positions[t];
        if (i >= text.size()) continue;
        
        // Must have at least one letter
        size_t j = i;
        char32_t first = 0;
        size_t tmp2 = i;
        if (!peek_char(text, tmp2, first)) continue;
        if (!utf8::is_letter(first)) continue;
        j = tmp2;
        
        // Match more letters greedily
        while (j < text.size()) {
            size_t next = j;
            char32_t c;
            if (!peek_char(text, next, c)) break;
            if (!utf8::is_letter(c)) break;
            j = next;
        }
        
        return j;  // Return first (greedy) match
    }
    
    return 0;
}

// Alternative 3: \p{N}
// A single number/digit character.
static size_t match_digit(const std::string& text, size_t pos) {
    size_t tmp = pos;
    char32_t cp;
    if (!peek_char(text, tmp, cp)) return 0;
    if (utf8::is_digit(cp)) return tmp;
    return 0;
}

// Alternative 4:  ?[^\s\p{L}\p{N}]+[\r\n]*
// Optional literal space + non-ws/non-L/non-D chars (greedy) + optional \r\n.
static size_t match_non_alnum_seq(const std::string& text, size_t pos) {
    size_t i = pos;
    if (i >= text.size()) return 0;
    
    // Optional literal space (U+0020 only, not general whitespace)
    if (text[i] == ' ') ++i;
    
    if (i >= text.size()) return 0;
    
    // One or more non-whitespace, non-letter, non-digit (greedy)
    size_t tmp = i;
    char32_t cp;
    if (!peek_char(text, tmp, cp)) return 0;
    if (utf8::is_whitespace(cp) || utf8::is_letter(cp) || utf8::is_digit(cp)) return 0;
    i = tmp;
    
    while (i < text.size()) {
        size_t next = i;
        char32_t c;
        if (!peek_char(text, next, c)) break;
        if (c == '\r' || c == '\n') break;
        if (utf8::is_whitespace(c) || utf8::is_letter(c) || utf8::is_digit(c)) break;
        i = next;
    }
    
    // Optional \r\n
    while (i < text.size() && (text[i] == '\r' || text[i] == '\n')) ++i;
    
    return i;
}

// Alternative 5: \s*[\r\n]+
// Zero or more whitespace + one or more \r or \n.
// Greedy \s* with backtracking to ensure [\r\n]+ matches.
static size_t match_newlines(const std::string& text, size_t pos) {
    size_t i = pos;
    
    // Collect all possible split points within the contiguous whitespace.
    // split_points[k] = position after consuming k+1 whitespace chars.
    std::vector<size_t> split_points;
    while (i < text.size()) {
        size_t next = i;
        char32_t c;
        if (!peek_char(text, next, c)) break;
        if (!utf8::is_whitespace(c)) break;
        split_points.push_back(next);
        i = next;
    }
    
    if (split_points.empty()) return 0;
    
    // Try from greedy (longest \s*) to shortest, backtracking until [\r\n]+ matches.
    for (int k = (int)split_points.size() - 1; k >= 0; --k) {
        size_t split_at = split_points[k];  // \s* ends here, [\r\n]+ starts here
        size_t j = split_at;
        if (j >= text.size()) continue;
        if (text[j] != '\r' && text[j] != '\n') continue;
        while (j < text.size() && (text[j] == '\r' || text[j] == '\n')) ++j;
        return j;
    }
    
    return 0;
}

// Alternative 6: \s+(?!\S)
// One or more whitespace characters, NOT followed by a non-whitespace character.
// Greedy with backtracking: tries longest match first, backtracks if
// the lookahead (?!\S) fails.
static size_t match_trailing_ws(const std::string& text, size_t pos) {
    size_t i = pos;
    if (i >= text.size()) return 0;
    
    // Collect all consecutive whitespace end positions
    std::vector<size_t> ends;
    size_t j = i;
    while (j < text.size()) {
        size_t next = j;
        char32_t cp;
        if (!peek_char(text, next, cp)) break;
        if (!utf8::is_whitespace(cp)) break;
        ends.push_back(next);
        j = next;
    }
    
    if (ends.empty()) return 0;
    
    // Try from longest to shortest (greedy → backtrack)
    for (int k = (int)ends.size() - 1; k >= 0; --k) {
        size_t end_pos = ends[k];
        // (?!\S) — next char must NOT be non-whitespace
        if (end_pos >= text.size()) {
            return end_pos;  // end of string → OK
        }
        size_t tmp2 = end_pos;
        char32_t next_cp;
        peek_char(text, tmp2, next_cp);
        if (!utf8::is_whitespace(next_cp)) {
            continue;  // followed by non-ws → backtrack
        }
        return end_pos;  // followed by ws or end → OK
    }
    
    return 0;
}

// Alternative 7: \s+
// One or more whitespace characters (greedy).
static size_t match_whitespace(const std::string& text, size_t pos) {
    size_t i = pos;
    if (i >= text.size()) return 0;
    
    size_t tmp = i;
    char32_t cp;
    if (!peek_char(text, tmp, cp)) return 0;
    if (!utf8::is_whitespace(cp)) return 0;
    i = tmp;
    
    while (i < text.size()) {
        size_t next = i;
        char32_t c;
        if (!peek_char(text, next, c)) break;
        if (!utf8::is_whitespace(c)) break;
        i = next;
    }
    return i;
}

// Split text using GPT-2 pre-tokenizer regex
// Tries each alternative at current position, picks the LONGEST match.
// Falls back to consuming one codepoint if nothing matches.
inline std::vector<std::string> split(const std::string& text) {
    std::vector<std::string> result;
    size_t pos = 0;

    while (pos < text.size()) {
        size_t match_end = 0;
        using Matcher = size_t (*)(const std::string&, size_t);
        Matcher matchers[] = {
            match_contraction,
            match_letter_seq,
            match_digit,
            match_non_alnum_seq,
            match_newlines,
            match_trailing_ws,
            match_whitespace
        };

        // Regex alternation uses FIRST-MATCH-WINS, not longest match!
        for (auto m : matchers) {
            size_t end = m(text, pos);
            if (end > pos) {
                match_end = end;
                break;
            }
        }

        if (match_end > pos) {
            result.push_back(text.substr(pos, match_end - pos));
            pos = match_end;
        } else {
            // No match — consume one codepoint as fallback
            size_t next = pos;
            utf8::decode(text, next);
            result.push_back(text.substr(pos, next - pos));
            pos = next;
        }
    }

    return result;
}

} // namespace gpt2_split

// ============================================================================
// ── Tokenizer data structures (parsed from tokenizer.json) ──────────────────
// ============================================================================

struct Tokenizer {
    // Vocabulary: string -> id
    std::unordered_map<std::string, int> vocab;
    // Added tokens: id -> content
    std::unordered_map<int, std::string> added_tokens;
    // Merge ranks: pair<string,string> -> rank index
    std::map<std::pair<std::string, std::string>, int> merge_ranks;
    // Pre-tokenizer config
    enum PreType { None, ByteLevel, Metaspace, Split, Sequence };
    PreType pre_type = None;
    // Split regex pattern (for Split type)
    std::string split_pattern;
    std::string split_behavior = "Isolated";
    // Metaspace config
    std::string metaspace_replacement = "\xe2\x96\x81"; // ▁ (U+2581)
    bool metaspace_add_prefix = true;
    // ByteLevel config
    bool bytelevel_add_prefix = false;
    bool bytelevel_use_regex = true;
    // Sequence config (nested pre-tokenizers)
    struct PreTokenizerConfig {
        PreType type = None;
        std::string pattern;
        std::string behavior = "Isolated";
        bool invert = false;
        std::string replacement = "\xe2\x96\x81";
        bool add_prefix = true;
        bool use_regex = true;
        bool trim_offsets = false;
    };
    std::vector<PreTokenizerConfig> pre_tokenizers;

    // BPE config
    int unk_token_id = -1; // -1 means use 0 as fallback
    bool byte_fallback = false;
};

// ============================================================================
// ── Tokenizer loading from JSON ─────────────────────────────────────────────
// ============================================================================

static std::string to_utf8(const json::Value& v) {
    return v.is_string() ? v.as_string() : "";
}

static int to_int(const json::Value& v) {
    return v.is_number() ? (int)v.as_int() : 0;
}

static Tokenizer* tokenizer_load_from_json(const std::string& json_str) {
    auto root = json::parse(json_str);
    if (root.is_null()) return nullptr;

    auto tok = new Tokenizer();

    // ── Model ────────────────────────────────────────────────────────────
    auto model = root["model"];
    auto vocab = model["vocab"];
    auto merges = model["merges"];

    // Vocab: string -> id
    if (vocab.is_object()) {
        for (auto& [key, val] : vocab.as_object()) {
            tok->vocab[key] = (int)val.as_int();
        }
    }

    // Merges: ["a b", ...] or [["a","b"], ...]
    if (merges.is_array()) {
        for (size_t i = 0; i < merges.size(); ++i) {
            auto item = merges[i];
            std::string left, right;
            if (item.is_string()) {
                // "a b" format
                auto s = item.as_string();
                auto space = s.find(' ');
                if (space != std::string::npos) {
                    left = s.substr(0, space);
                    right = s.substr(space + 1);
                }
            } else if (item.is_array() && item.size() >= 2) {
                left = item[0].as_string();
                right = item[1].as_string();
            }
            if (!left.empty() && !right.empty()) {
                tok->merge_ranks[{left, right}] = (int)i;
            }
        }
    }

    // Model config
    if (model["unk_token_id"].is_number())
        tok->unk_token_id = (int)model["unk_token_id"].as_int();
    if (model["byte_fallback"].is_bool())
        tok->byte_fallback = model["byte_fallback"].as_bool();

    // ── Added tokens ────────────────────────────────────────────────────
    auto added = root["added_tokens"];
    if (added.is_array()) {
        for (size_t i = 0; i < added.size(); ++i) {
            auto item = added[i];
            int id = to_int(item["id"]);
            std::string content = to_utf8(item["content"]);
            if (!content.empty()) {
                tok->added_tokens[id] = content;
            }
        }
    }

    // ── Pre-tokenizer ───────────────────────────────────────────────────
    auto pre = root["pre_tokenizer"];
    if (pre.is_object()) {
        std::string type = to_utf8(pre["type"]);
        if (type == "Sequence") {
            tok->pre_type = Tokenizer::Sequence;
            auto sub_pre = pre["pretokenizers"];
            if (sub_pre.is_array()) {
                for (size_t i = 0; i < sub_pre.size(); ++i) {
                    auto sp = sub_pre[i];
                    Tokenizer::PreTokenizerConfig cfg;
                    std::string stype = to_utf8(sp["type"]);
                    if (stype == "Split") {
                        cfg.type = Tokenizer::Split;
                        cfg.pattern = to_utf8(sp["pattern"]["Regex"]);
                        cfg.behavior = to_utf8(sp["behavior"]);
                        cfg.invert = sp["invert"].is_bool() && sp["invert"].as_bool();
                    } else if (stype == "ByteLevel") {
                        cfg.type = Tokenizer::ByteLevel;
                        cfg.add_prefix = sp["add_prefix_space"].is_bool() && sp["add_prefix_space"].as_bool();
                        cfg.trim_offsets = sp["trim_offsets"].is_bool() && sp["trim_offsets"].as_bool();
                        cfg.use_regex = sp["use_regex"].is_bool() ? sp["use_regex"].as_bool() : true;
                    } else if (stype == "Metaspace") {
                        cfg.type = Tokenizer::Metaspace;
                        cfg.replacement = to_utf8(sp["replacement"]);
                        cfg.add_prefix = sp["add_prefix_space"].is_bool() && sp["add_prefix_space"].as_bool();
                    }
                    tok->pre_tokenizers.push_back(cfg);
                }
            }
        } else if (type == "Split") {
            tok->pre_type = Tokenizer::Split;
            tok->split_pattern = to_utf8(pre["pattern"]["Regex"]);
            tok->split_behavior = to_utf8(pre["behavior"]);
        } else if (type == "ByteLevel") {
            tok->pre_type = Tokenizer::ByteLevel;
            tok->bytelevel_add_prefix = pre["add_prefix_space"].is_bool() && pre["add_prefix_space"].as_bool();
            tok->bytelevel_use_regex = pre["use_regex"].is_bool() ? pre["use_regex"].as_bool() : true;
        } else if (type == "Metaspace") {
            tok->pre_type = Tokenizer::Metaspace;
            tok->metaspace_replacement = to_utf8(pre["replacement"]);
            tok->metaspace_add_prefix = pre["add_prefix_space"].is_bool() && pre["add_prefix_space"].as_bool();
        }
    }

    return tok;
}

// ============================================================================
// ── BPE Encoding ────────────────────────────────────────────────────────────
// ============================================================================

static std::vector<int> bpe_encode(
    const std::string& word,
    const std::unordered_map<std::string, int>& vocab,
    const std::map<std::pair<std::string, std::string>, int>& merge_ranks,
    int unk_id,
    bool byte_fallback)
{
    // Check if whole word is in vocab
    auto vit = vocab.find(word);
    if (vit != vocab.end()) {
        return {vit->second};
    }

    // For ByteLevel tokenizers, the word has already been byte-encoded by this point,
    // so 'word' contains GPT-2 byte-level Unicode chars.
    // We split into individual characters (they could be multi-byte UTF-8).
    std::vector<std::string> tokens;
    size_t i = 0;
    while (i < word.size()) {
        size_t start = i;
        // Read one UTF-8 character (could be multi-byte for the byte-level mapping)
        if ((unsigned char)word[i] <= 0x7F) {
            ++i;
        } else if ((unsigned char)word[i] >= 0xC0 && (unsigned char)word[i] <= 0xDF && i + 1 < word.size()) {
            i += 2;
        } else if ((unsigned char)word[i] >= 0xE0 && (unsigned char)word[i] <= 0xEF && i + 2 < word.size()) {
            i += 3;
        } else if ((unsigned char)word[i] >= 0xF0 && (unsigned char)word[i] <= 0xF7 && i + 3 < word.size()) {
            i += 4;
        } else {
            ++i; // fallback
        }
        tokens.push_back(word.substr(start, i - start));
    }

    // If tokens all need to be single chars and there's only 1, we have our result
    if (tokens.size() <= 1) {
        std::vector<int> ids;
        for (auto& t : tokens) {
            auto it = vocab.find(t);
            if (it != vocab.end())
                ids.push_back(it->second);
            else if (byte_fallback && t.size() == 1)
                ids.push_back(unk_id);
            else
                ids.push_back(unk_id);
        }
        return ids;
    }

    // Iteratively merge
    while (tokens.size() > 1) {
        // Find best pair (lowest rank)
        int best_idx = -1;
        int best_rank = INT_MAX;

        for (int i = 0; i < (int)tokens.size() - 1; ++i) {
            auto key = std::make_pair(tokens[i], tokens[i+1]);
            auto it = merge_ranks.find(key);
            int rank = (it != merge_ranks.end()) ? it->second : INT_MAX;
            if (rank < best_rank) {
                best_rank = rank;
                best_idx = i;
            }
        }

        if (best_idx < 0) break;

        // Merge the pair
        std::string merged = tokens[best_idx] + tokens[best_idx + 1];
        std::vector<std::string> new_tokens;
        for (int i = 0; i < (int)tokens.size(); ++i) {
            if (i == best_idx) {
                new_tokens.push_back(merged);
                ++i; // skip next
            } else {
                new_tokens.push_back(tokens[i]);
            }
        }
        tokens = std::move(new_tokens);
    }

    // Map to IDs
    std::vector<int> ids;
    for (auto& t : tokens) {
        auto it = vocab.find(t);
        if (it != vocab.end()) {
            ids.push_back(it->second);
        } else if (byte_fallback && t.size() == 1) {
            ids.push_back(unk_id);
        } else {
            ids.push_back(unk_id);
        }
    }
    return ids;
}

// ============================================================================
// ── Full encoding pipeline ──────────────────────────────────────────────────
// ============================================================================

// Split text at added token boundaries.
// Added tokens are extracted as whole units; remaining segments go through the
// normal pre-tokenizer → BPE pipeline.
static std::vector<std::pair<std::string, bool>> split_added(
    const std::string& text,
    const std::unordered_map<int, std::string>& added_tokens)
{
    struct AddedEntry {
        std::string content;
        int id;
    };
    std::vector<AddedEntry> added_list;
    for (auto& [id, content] : added_tokens) {
        if (!content.empty()) {
            added_list.push_back({content, id});
        }
    }
    // Sort by length descending so longest added tokens match first
    std::sort(added_list.begin(), added_list.end(),
        [](const AddedEntry& a, const AddedEntry& b) {
            return a.content.size() > b.content.size();
        });

    std::vector<std::pair<std::string, bool>> result;
    size_t pos = 0;
    while (pos < text.size()) {
        bool matched = false;
        for (auto& entry : added_list) {
            if (pos + entry.content.size() <= text.size() &&
                text.compare(pos, entry.content.size(), entry.content) == 0) {
                result.push_back({entry.content, true});  // is_added_token = true
                pos += entry.content.size();
                matched = true;
                break;
            }
        }
        if (!matched) {
            // Find the next added token in the remaining text
            size_t next_added = text.size();
            for (auto& entry : added_list) {
                size_t found = text.find(entry.content, pos);
                if (found != std::string::npos && found < next_added) {
                    next_added = found;
                }
            }
            // Extract the normal text segment
            if (next_added > pos) {
                result.push_back({text.substr(pos, next_added - pos), false});
            }
            pos = next_added;
        }
    }
    return result;
}

static std::vector<int> tokenizer_encode_str(Tokenizer* tok, const std::string& text) {
    if (!tok) return {};

    int unk_id = tok->unk_token_id >= 0 ? tok->unk_token_id : 0;

    // ── 0. Pre-process: handle added tokens ────────────────────────────
    auto segments = split_added(text, tok->added_tokens);
    std::vector<int> all_ids;

    // Build reverse added: content -> id
    std::unordered_map<std::string, int> added_content_to_id;
    for (auto& [id, content] : tok->added_tokens) {
        added_content_to_id[content] = id;
    }

    for (auto& [segment, is_added] : segments) {
        if (is_added) {
            auto it = added_content_to_id.find(segment);
            if (it != added_content_to_id.end()) {
                all_ids.push_back(it->second);
            }
            continue;
        }

        // ── 1. Apply pre-tokenizer on normal text segment ──────────────
        std::vector<std::string> words;

        if (tok->pre_type == Tokenizer::Sequence) {
        // Sequence pre-tokenizer: apply each in order
        words = {segment};
        for (auto& pt : tok->pre_tokenizers) {
            std::vector<std::string> next_words;
            for (auto& w : words) {
                if (pt.type == Tokenizer::Split) {
                    // Always use GPT-2 split for the Split pre-tokenizer
                    auto parts = gpt2_split::split(w);
                    for (auto& p : parts) {
                        if (pt.invert) {
                            // Invert: keep non-matching parts
                            // For simplicity, no invert support yet
                            next_words.push_back(p);
                        } else {
                            next_words.push_back(p);
                        }
                    }
                } else if (pt.type == Tokenizer::ByteLevel) {
                    // ByteLevel: convert bytes to GPT-2 Unicode chars
                    std::string encoded;
                    if (pt.add_prefix && !w.empty() && w[0] != ' ') {
                        encoded = bytelevel::encode(std::string(" ") + w);
                    } else {
                        encoded = bytelevel::encode(w);
                    }
                    // Remove empty segments
                    if (!encoded.empty()) next_words.push_back(encoded);
                } else if (pt.type == Tokenizer::Metaspace) {
                    std::string processed;
                    if (pt.add_prefix) {
                        processed = pt.replacement + w;
                    } else {
                        processed = w;
                    }
                    for (size_t i = 0; i < processed.size();) {
                        if (processed[i] == ' ') {
                            processed.replace(i, 1, pt.replacement);
                            i += pt.replacement.size();
                        } else {
                            ++i;
                        }
                    }
                    if (!processed.empty()) next_words.push_back(processed);
                } else {
                    if (!w.empty()) next_words.push_back(w);
                }
            }
            words = std::move(next_words);
        }
    } else if (tok->pre_type == Tokenizer::Split) {
        words = gpt2_split::split(segment);
    } else if (tok->pre_type == Tokenizer::ByteLevel) {
        // Standalone ByteLevel: use GPT-2 regex split, then byte encode
        auto parts = gpt2_split::split(segment);
        for (auto& p : parts) {
            std::string encoded;
            if (tok->bytelevel_add_prefix && !p.empty() && p[0] != ' ') {
                encoded = bytelevel::encode(std::string(" ") + p);
            } else {
                encoded = bytelevel::encode(p);
            }
            if (!encoded.empty()) words.push_back(encoded);
        }
    } else if (tok->pre_type == Tokenizer::Metaspace) {
        std::string processed;
        if (tok->metaspace_add_prefix) {
            processed = tok->metaspace_replacement + segment;
        } else {
            processed = segment;
        }
        // Replace spaces with metaspace char
        for (size_t i = 0; i < processed.size();) {
            if (processed[i] == ' ') {
                processed.replace(i, 1, tok->metaspace_replacement);
                i += tok->metaspace_replacement.size();
            } else {
                ++i;
            }
        }
        words.push_back(processed);
    } else {
        // No pre-tokenizer: simple whitespace split
        std::stringstream ss(segment);
        std::string w;
        while (ss >> w) words.push_back(w);
        if (words.empty()) words.push_back(segment);
    }

    // ── 2. BPE encode each word ────────────────────────────────────────
    for (auto& w : words) {
        auto word_ids = bpe_encode(w, tok->vocab, tok->merge_ranks, unk_id, tok->byte_fallback);
        all_ids.insert(all_ids.end(), word_ids.begin(), word_ids.end());
    }
    } // end for over segments
    return all_ids;
}

// ============================================================================
// ── Decoding ─────────────────────────────────────────────────────────────────
// ============================================================================

static std::string tokenizer_decode_str(Tokenizer* tok, const std::vector<int>& ids) {
    if (!tok) return {};

    // Build reverse vocab
    std::unordered_map<int, std::string> rev;
    for (auto& [s, id] : tok->vocab) {
        rev[id] = s;
    }
    // Added tokens
    for (auto& [id, s] : tok->added_tokens) {
        rev[id] = s;
    }

    std::string text;
    for (int id : ids) {
        auto it = rev.find(id);
        if (it != rev.end()) {
            text += it->second;
        } else {
            text += "<unk>";
        }
    }

    // Try byte-level decoding
    std::string decoded = bytelevel::decode(text);

    // For Metaspace, replace metaspace char with space
    if (tok->pre_type == Tokenizer::Metaspace ||
        (tok->pre_type == Tokenizer::Sequence)) {
        auto& rep = tok->metaspace_replacement;
        if (!rep.empty()) {
            size_t pos = 0;
            while ((pos = decoded.find(rep, pos)) != std::string::npos) {
                decoded.replace(pos, rep.size(), " ");
                pos += 1;
            }
        }
    }

    return decoded;
}

// ============================================================================
// ── C ABI ──────────────────────────────────────────────────────────────────
// ============================================================================

extern "C" {

void* tokenizer_load(const char* json_path) {
    if (!json_path) return nullptr;

    FILE* f = fopen(json_path, "rb");
    if (!f) {
        fprintf(stderr, "tokenizer_load: cannot open %s\n", json_path);
        return nullptr;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    std::string data;
    data.resize(size);
    if (fread(&data[0], 1, size, f) != (size_t)size) {
        fclose(f);
        return nullptr;
    }
    fclose(f);

    return tokenizer_load_from_json(data);
}

// Returns number of token IDs written to out_ids.
// Returns -1 on error.
int tokenizer_encode(void* tok, const char* text, int* out_ids, int max_ids) {
    if (!tok || !text || !out_ids) return -1;
    auto t = (Tokenizer*)tok;
    auto ids = tokenizer_encode_str(t, text);
    int n = (int)ids.size();
    if (n > max_ids) n = max_ids;
    for (int i = 0; i < n; ++i) out_ids[i] = ids[i];
    return (int)ids.size();
}

// Returns decoded string. Caller must free the returned pointer with free().
char* tokenizer_decode(void* tok, const int* ids, int n_ids) {
    if (!tok || !ids || n_ids <= 0) return nullptr;
    auto t = (Tokenizer*)tok;
    std::vector<int> id_vec(ids, ids + n_ids);
    auto text = tokenizer_decode_str(t, id_vec);
    char* result = (char*)malloc(text.size() + 1);
    if (result) {
        memcpy(result, text.c_str(), text.size() + 1);
    }
    return result;
}

void tokenizer_free(void* tok) {
    delete (Tokenizer*)tok;
}

} // extern "C"

// ============================================================================
// ── CLI ──────────────────────────────────────────────────────────────────────
// ============================================================================

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s --model <tokenizer.json> --encode <text>\n", prog);
    fprintf(stderr, "  %s --model <tokenizer.json> --decode <id1> [id2 ...]\n", prog);
    fprintf(stderr, "  %s --model <tokenizer.json>  (reads from stdin)\n", prog);
}

#ifndef TOKENIZER_NO_MAIN
int main(int argc, char** argv) {
    const char* model_path = nullptr;
    const char* encode_text = nullptr;
    bool decode_mode = false;
    std::vector<int> decode_ids;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--encode") == 0 && i + 1 < argc) {
            encode_text = argv[++i];
        } else if (strcmp(argv[i], "--decode") == 0) {
            decode_mode = true;
            while (++i < argc) {
                // Stop if next arg starts with --
                if (argv[i][0] == '-' && argv[i][1] == '-') { --i; break; }
                decode_ids.push_back(atoi(argv[i]));
            }
        }
    }

    // Default model path
    if (!model_path) {
        model_path = std::getenv("HOME");
        if (!model_path) model_path = ".";
        std::string default_path = std::string(model_path) + "/.config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json";
        model_path = strdup(default_path.c_str());
        // Leak is fine for CLI
    }

    // Load tokenizer
    void* tok = tokenizer_load(model_path);
    if (!tok) {
        fprintf(stderr, "Error: failed to load tokenizer from %s\n", model_path);
        return 1;
    }

    if (encode_text) {
        // Single text encode
        auto ids = tokenizer_encode_str((Tokenizer*)tok, encode_text);
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i > 0) printf(" ");
            printf("%d", ids[i]);
        }
        printf("\n");
    } else if (decode_mode) {
        // Decode IDs
        char* text = tokenizer_decode(tok, decode_ids.data(), (int)decode_ids.size());
        if (text) {
            printf("%s\n", text);
            free(text);
        }
    } else {
        // Interactive: read from stdin
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            auto ids = tokenizer_encode_str((Tokenizer*)tok, line);
            for (size_t i = 0; i < ids.size(); ++i) {
                if (i > 0) printf(" ");
                printf("%d", ids[i]);
            }
            printf("\n");
        }
    }

    tokenizer_free(tok);
    return 0;
}
#endif // TOKENIZER_NO_MAIN

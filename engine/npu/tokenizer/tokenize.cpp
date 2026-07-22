/**
 * tokenize.cpp — Standalone BPE tokenizer for NPU engine (C++ port).
 * Handles special tokens (added_tokens) correctly before falling through
 * to BPE subword tokenization.
 *
 * Build: g++ -std=c++17 -O3 -o tokenize tokenize.cpp
 *        (also built via CMake: engine/npu/tokenizer/)
 * Usage: echo "Hello" | ./tokenize tokenizer.json
 */
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace {

constexpr int kMaxVocab      = 200000;
constexpr int kMaxTokenLen   = 256;
constexpr int kMaxSpecial    = 64;
constexpr int kMaxInputLen   = 8192;

struct Token {
    int id = -1;
    unsigned char bytes[kMaxTokenLen]{};
    int len = 0;
};

struct SpecialToken {
    int id = -1;
    unsigned char str[kMaxTokenLen]{};
    int len = 0;
};

Token          vocab[kMaxVocab];
int            vocab_size = 0;
SpecialToken   specials[kMaxSpecial];
int            n_specials = 0;
int            bos_id = 151643;
int            eos_id = 151645;
int            byte_fallback[256]{};

// ── JSON unescape ─────────────────────────────────────────────────────
int json_unescape(const unsigned char *src, int slen, unsigned char *dst) {
    int di = 0;
    for (int i = 0; i < slen && di < kMaxTokenLen - 1; i++) {
        if (src[i] == '\\' && i + 1 < slen) {
            i++;
            switch (src[i]) {
            case 'n':  dst[di++] = '\n'; break;
            case 't':  dst[di++] = '\t'; break;
            case 'r':  dst[di++] = '\r'; break;
            case '\\': dst[di++] = '\\'; break;
            case '"':  dst[di++] = '"';  break;
            case 'u':  i += 4; dst[di++] = '?'; break;
            default:   dst[di++] = src[i]; break;
            }
        } else {
            dst[di++] = src[i];
        }
    }
    return di;
}

// ── Special token matching ────────────────────────────────────────────
int match_special(const unsigned char *input, int input_len, int *out_id) {
    for (int s = 0; s < n_specials; s++) {
        if (specials[s].len <= input_len &&
            std::memcmp(specials[s].str, input, specials[s].len) == 0) {
            *out_id = specials[s].id;
            return specials[s].len;
        }
    }
    return 0;
}

// ── Byte-fallback lookup ──────────────────────────────────────────────
void build_byte_fallback() {
    for (int &b : byte_fallback) b = -1;
    for (int i = 0; i < vocab_size; i++) {
        if (vocab[i].len == 2) {
            int b0 = vocab[i].bytes[0];
            int b1 = vocab[i].bytes[1];
            if ((b0 & 0xFE) == 0xC4) {
                unsigned int cp = ((b0 & 0x1F) << 6) | (b1 & 0x3F);
                if (cp >= 0x100 && cp <= 0x1FF) {
                    int b = static_cast<int>(cp - 0x100);
                    if (byte_fallback[b] < 0) byte_fallback[b] = vocab[i].id;
                }
            }
        }
    }
}

// ── Longest matching token ────────────────────────────────────────────
int find_longest_match(const unsigned char *input, int input_len, int *out_id) {
    int best_len = 0, best_id = -1;
    for (int i = 0; i < vocab_size; i++) {
        if (vocab[i].len <= input_len && vocab[i].len > best_len) {
            if (std::memcmp(vocab[i].bytes, input, vocab[i].len) == 0) {
                best_len = vocab[i].len;
                best_id  = vocab[i].id;
            }
        }
    }
    if (best_id >= 0) { *out_id = best_id; return best_len; }
    if (input_len > 0) {
        int fb = byte_fallback[input[0]];
        if (fb >= 0) { *out_id = fb; return 1; }
    }
    return 0;
}

// ── Load tokenizer.json ───────────────────────────────────────────────
int load_tokenizer(const char *path) {
    FILE *f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "Cannot open %s\n", path); return -1; }
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz > 100 * 1024 * 1024) {
        std::fprintf(stderr, "File too large\n");
        std::fclose(f);
        return -1;
    }
    auto *json = static_cast<char *>(std::malloc(sz + 1));
    if (!json) { std::fclose(f); return -1; }
    std::fread(json, 1, sz, f);
    json[sz] = '\0';
    std::fclose(f);

    std::string_view jsv(json, sz);

    // ── Parse added_tokens ──
    auto atok = jsv.find("\"added_tokens\"");
    if (atok != std::string_view::npos) {
        auto arr_start = jsv.find('[', atok);
        auto arr_end   = (arr_start != std::string_view::npos)
                             ? jsv.find(']', arr_start)
                             : std::string_view::npos;
        if (arr_start != std::string_view::npos &&
            arr_end   != std::string_view::npos) {
            size_t p = arr_start;
            while (p < arr_end) {
                auto obrace = jsv.find('{', p);
                if (obrace == std::string_view::npos || obrace >= arr_end) break;
                auto cbrace = jsv.find('}', obrace);
                if (cbrace == std::string_view::npos || cbrace >= arr_end) break;

                // "id"
                auto idf = jsv.find("\"id\"", obrace);
                int  id  = 0;
                if (idf != std::string_view::npos && idf < cbrace) {
                    auto idc = jsv.find(':', idf);
                    if (idc != std::string_view::npos)
                        id = std::atoi(jsv.data() + idc + 1);
                }

                // "content"
                auto ctf = jsv.find("\"content\"", obrace);
                if (ctf != std::string_view::npos && ctf < cbrace) {
                    auto col = jsv.find(':', ctf);
                    if (col != std::string_view::npos) {
                        col++;
                        while (col < jsv.size() && std::isspace(static_cast<unsigned char>(jsv[col])))
                            col++;
                        if (col < jsv.size() && jsv[col] == '"') {
                            col++;
                            unsigned char raw[256];
                            int           ri = 0;
                            while (col < jsv.size() && jsv[col] != '"' && ri < 255) {
                                if (jsv[col] == '\\' && col + 1 < jsv.size() && jsv[col + 1] == 'u') {
                                    col += 6; raw[ri++] = '?';
                                } else if (jsv[col] == '\\' && col + 1 < jsv.size() && jsv[col + 1] == 'n') {
                                    raw[ri++] = '\n'; col += 2;
                                } else if (jsv[col] == '\\' && col + 1 < jsv.size() && jsv[col + 1] == 't') {
                                    raw[ri++] = '\t'; col += 2;
                                } else if (jsv[col] == '\\') {
                                    col += 2; raw[ri++] = '?';
                                } else {
                                    raw[ri++] = static_cast<unsigned char>(jsv[col++]);
                                }
                            }
                            raw[ri] = '\0';

                            if (std::strcmp(reinterpret_cast<char *>(raw), "<|endoftext|>") == 0)
                                bos_id = id;
                            if (std::strcmp(reinterpret_cast<char *>(raw), "<|im_end|>") == 0)
                                eos_id = id;
                            if (std::strcmp(reinterpret_cast<char *>(raw), "</s>") == 0)
                                eos_id = id;

                            if (n_specials < kMaxSpecial && ri > 0) {
                                specials[n_specials].id = id;
                                std::memcpy(specials[n_specials].str, raw, ri);
                                specials[n_specials].len = ri;
                                n_specials++;
                            }
                        }
                    }
                }
                p = cbrace + 1;
            }
        }
    }

    // Sort specials by length descending (longest match first)
    std::sort(specials, specials + n_specials,
              [](const SpecialToken &a, const SpecialToken &b) {
                  return a.len > b.len;
              });

    std::fprintf(stderr, "[tokenize] BOS=%d EOS=%d specials=%d\n",
                 bos_id, eos_id, n_specials);

    // ── Parse main vocab ──
    size_t vocab_start = std::string_view::npos;
    auto   model_sec   = jsv.find("\"model\"");
    if (model_sec != std::string_view::npos) {
        auto mv = jsv.find(':', model_sec);
        if (mv != std::string_view::npos) {
            auto vk = jsv.find("\"vocab\"", mv);
            if (vk != std::string_view::npos) {
                auto vv = jsv.find(':', vk);
                if (vv != std::string_view::npos) {
                    vv++;
                    while (vv < jsv.size() && std::isspace(static_cast<unsigned char>(jsv[vv])))
                        vv++;
                    if (vv < jsv.size() && (jsv[vv] == '{' || jsv[vv] == '['))
                        vocab_start = vv;
                }
            }
        }
    }
    if (vocab_start == std::string_view::npos) {
        std::fprintf(stderr, "Cannot find vocab\n");
        std::free(json);
        return -1;
    }

    int token_count = 0;
    if (jsv[vocab_start] == '{') {
        size_t p = vocab_start + 1;
        while (p < jsv.size() && jsv[p] != '}' && token_count < kMaxVocab) {
            while (p < jsv.size() &&
                   (std::isspace(static_cast<unsigned char>(jsv[p])) || jsv[p] == ','))
                p++;
            if (p >= jsv.size() || jsv[p] == '}') break;
            if (jsv[p] != '"') { p++; continue; }
            p++;
            unsigned char buf[kMaxTokenLen];
            int           blen = 0;
            while (p < jsv.size() && jsv[p] != '"' && blen < kMaxTokenLen - 1) {
                if (jsv[p] == '\\' && p + 1 < jsv.size() && jsv[p + 1] == 'u') {
                    p += 6; buf[blen++] = '?';
                } else if (jsv[p] == '\\' && p + 1 < jsv.size() && jsv[p + 1] == 'n') {
                    buf[blen++] = '\n'; p += 2;
                } else if (jsv[p] == '\\' && p + 1 < jsv.size() && jsv[p + 1] == 't') {
                    buf[blen++] = '\t'; p += 2;
                } else if (jsv[p] == '\\') {
                    p += 2; buf[blen++] = '?';
                } else {
                    buf[blen++] = static_cast<unsigned char>(jsv[p++]);
                }
            }
            if (p < jsv.size() && jsv[p] == '"') p++;
            while (p < jsv.size() &&
                   (std::isspace(static_cast<unsigned char>(jsv[p])) || jsv[p] == ':'))
                p++;
            int id = std::atoi(jsv.data() + p);
            while (p < jsv.size() && jsv[p] >= '0' && jsv[p] <= '9') p++;
            if (id >= 0 && id < kMaxVocab && blen > 0) {
                vocab[id].id = id;
                std::memcpy(vocab[id].bytes, buf, blen);
                vocab[id].len = blen;
                if (id >= vocab_size) vocab_size = id + 1;
            }
        }
    } else if (jsv[vocab_start] == '[') {
        size_t p = vocab_start + 1;
        while (p < jsv.size() && jsv[p] != ']' && token_count < kMaxVocab) {
            auto ck = jsv.find("\"content\"", p);
            auto cb = jsv.find('}', p);
            if (ck == std::string_view::npos || (cb != std::string_view::npos && ck > cb)) break;
            auto col = jsv.find(':', ck);
            if (col == std::string_view::npos) { p = ck + 1; continue; }
            col++;
            while (col < jsv.size() &&
                   std::isspace(static_cast<unsigned char>(jsv[col])))
                col++;
            if (col < jsv.size() && jsv[col] == '"') {
                col++;
                unsigned char buf[kMaxTokenLen];
                int           blen = 0;
                while (col < jsv.size() && jsv[col] != '"' && blen < kMaxTokenLen - 1) {
                    if (jsv[col] == '\\' && col + 1 < jsv.size() && jsv[col + 1] == 'u') {
                        col += 6; buf[blen++] = '?';
                    } else if (jsv[col] == '\\' && col + 1 < jsv.size() && jsv[col + 1] == 'n') {
                        buf[blen++] = '\n'; col += 2;
                    } else if (jsv[col] == '\\') {
                        col += 2; buf[blen++] = '?';
                    } else {
                        buf[blen++] = static_cast<unsigned char>(jsv[col++]);
                    }
                }
                auto ik = jsv.find("\"id\"", p);
                if (ik != std::string_view::npos && (cb == std::string_view::npos || ik < cb)) {
                    auto ic = jsv.find(':', ik);
                    if (ic != std::string_view::npos) {
                        int id = std::atoi(jsv.data() + ic + 1);
                        if (id >= 0 && id < kMaxVocab) {
                            vocab[id].id = id;
                            std::memcpy(vocab[id].bytes, buf, blen);
                            vocab[id].len = blen;
                            if (id >= vocab_size) vocab_size = id + 1;
                        }
                    }
                }
            }
            p = jsv.find('}', p);
            if (p != std::string_view::npos) p++;
        }
    }

    std::free(json);
    std::fprintf(stderr, "[tokenize] loaded %d tokens\n", vocab_size);
    return (vocab_size > 0) ? 0 : -1;
}

// ── Tokenize and print ────────────────────────────────────────────────
int tokenize_and_print(const unsigned char *input, int input_len) {
    bool first = true;
    int  pos   = 0;

    while (pos < input_len) {
        int sid, slen = match_special(input + pos, input_len - pos, &sid);
        if (slen > 0) {
            if (!first) std::printf(",");
            std::printf("%d", sid);
            first = false;
            pos += slen;
            continue;
        }
        int mid, mlen = find_longest_match(input + pos, input_len - pos, &mid);
        if (mlen > 0) {
            if (!first) std::printf(",");
            std::printf("%d", mid);
            first = false;
            pos += mlen;
        } else {
            pos++;
        }
    }
    std::printf("\n");
    return 0;
}

}  // anonymous namespace

// ── main ──────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: echo \"text\" | %s tokenizer.json\n", argv[0]);
        return 1;
    }
    if (load_tokenizer(argv[1]) != 0) {
        std::fprintf(stderr, "Failed to load tokenizer from %s\n", argv[1]);
        return 1;
    }
    std::fprintf(stderr, "[tokenize] loaded %d tokens (bos=%d eos=%d)\n",
                 vocab_size, bos_id, eos_id);

    build_byte_fallback();

    unsigned char input[kMaxInputLen];
    int           len = 0;
    int           c;
    while ((c = std::getchar()) != EOF && len < kMaxInputLen - 1)
        input[len++] = static_cast<unsigned char>(c);
    input[len] = '\0';
    std::fprintf(stderr, "[tokenize] input=%s\n", input);
    return tokenize_and_print(input, len);
}

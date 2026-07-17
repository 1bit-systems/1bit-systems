/**
 * tokenize.c — Standalone BPE tokenizer for NPU engine.
 * Handles special tokens (added_tokens) correctly: matches them FIRST
 * before falling through to BPE subword tokenization.
 *
 * Build: gcc -O3 -o tokenize tokenize.c -lm
 * Usage: echo "Hello" | ./tokenize tokenizer.json
 */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_VOCAB 200000
#define MAX_TOKEN_LEN 256
#define MAX_SPECIAL 64
#define MAX_INPUT_LEN 8192

typedef struct { int id; unsigned char bytes[MAX_TOKEN_LEN]; int len; } Token;
typedef struct { int id; unsigned char str[MAX_TOKEN_LEN]; int len; } SpecialToken;

static Token vocab[MAX_VOCAB];
static int vocab_size = 0;
static SpecialToken specials[MAX_SPECIAL];
static int n_specials = 0;
static int bos_id = 151643, eos_id = 151645;

// Parse JSON escape sequences (handle only common ones)
static int json_unescape(const unsigned char* src, int slen, unsigned char* dst) {
    int di = 0;
    for (int i = 0; i < slen && di < MAX_TOKEN_LEN-1; i++) {
        if (src[i] == '\\' && i+1 < slen) {
            i++;
            switch (src[i]) {
                case 'n': dst[di++] = '\n'; break;
                case 't': dst[di++] = '\t'; break;
                case 'r': dst[di++] = '\r'; break;
                case '\\': dst[di++] = '\\'; break;
                case '"': dst[di++] = '"'; break;
                case 'u': i += 4; dst[di++] = '?'; break;  // skip unicode escapes
                default: dst[di++] = src[i]; break;
            }
        } else {
            dst[di++] = src[i];
        }
    }
    return di;
}

// Check if input starts with special token string
static int match_special(const unsigned char* input, int input_len, int* out_id) {
    for (int s = 0; s < n_specials; s++) {
        if (specials[s].len <= input_len &&
            memcmp(specials[s].str, input, specials[s].len) == 0) {
            *out_id = specials[s].id;
            return specials[s].len;
        }
    }
    return 0;
}

// Build a byte-fallback lookup: for each byte b (0-255), find the
// vocab token id for its byte-fallback representation chr(0x100+b)
static int byte_fallback[256];

static void build_byte_fallback() {
    for (int b = 0; b < 256; b++) byte_fallback[b] = -1;
    for (int i = 0; i < vocab_size; i++) {
        if (vocab[i].len == 2) {
            int b0 = vocab[i].bytes[0], b1 = vocab[i].bytes[1];
            // Check if it's a valid UTF-8 encoding of U+0100+byte
            if ((b0 & 0xFE) == 0xC4) {  // C4-C7 range
                unsigned int cp = ((b0 & 0x1F) << 6) | (b1 & 0x3F);
                if (cp >= 0x100 && cp <= 0x1FF) {
                    int b = cp - 0x100;
                    if (byte_fallback[b] < 0) byte_fallback[b] = vocab[i].id;
                }
            }
        }
    }
}

// Find longest matching token (raw text), with byte-fallback for single bytes
static int find_longest_match(const unsigned char* input, int input_len, int* out_id) {
    int best_len = 0, best_id = -1;
    for (int i = 0; i < vocab_size; i++) {
        if (vocab[i].len <= input_len && vocab[i].len > best_len) {
            if (memcmp(vocab[i].bytes, input, vocab[i].len) == 0) {
                best_len = vocab[i].len; best_id = vocab[i].id;
            }
        }
    }
    if (best_id >= 0) { *out_id = best_id; return best_len; }
    // Fallback: try byte-fallback for single byte
    if (input_len > 0) {
        int fb = byte_fallback[input[0]];
        if (fb >= 0) { *out_id = fb; return 1; }
    }
    return 0;
}

static int load_tokenizer(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return -1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz > 100*1024*1024) { fprintf(stderr, "File too large\n"); fclose(f); return -1; }
    char* json = (char*)malloc(sz+1);
    if (!json) { fclose(f); return -1; }
    fread(json, 1, sz, f); json[sz] = '\0'; fclose(f);

    // Step 1: Parse added_tokens — find each {id, content} object
    const char* atok = strstr(json, "\"added_tokens\"");
    if (atok) {
        const char* arr_start = strchr(atok, '[');
        const char* arr_end = arr_start ? strchr(arr_start, ']') : NULL;
        if (arr_start && arr_end) {
            const char* p = arr_start;
            while (p < arr_end) {
                // Find next opening brace
                const char* obrace = strchr(p, '{');
                if (!obrace || obrace >= arr_end) break;
                const char* cbrace = strchr(obrace, '}');
                if (!cbrace || cbrace >= arr_end) break;

                // Find "id" within this { } block
                const char* idf = strstr(obrace, "\"id\"");
                int id = 0;
                if (idf && idf < cbrace) {
                    const char* idc = strchr(idf, ':');
                    if (idc) id = atoi(idc + 1);
                }

                // Find "content" within this { } block
                const char* ctf = strstr(obrace, "\"content\"");
                if (ctf && ctf < cbrace) {
                    const char* col = strchr(ctf, ':');
                    if (col) { col++; while (*col && isspace((unsigned char)*col)) col++;
                    if (*col == '"') { col++;
                        unsigned char raw[256]; int ri = 0;
                        while (*col && *col != '"' && ri < 255) {
                            if (*col == '\\' && *(col+1) == 'u') { col += 6; raw[ri++] = '?'; }
                            else if (*col == '\\' && *(col+1) == 'n') { raw[ri++] = '\n'; col += 2; }
                            else if (*col == '\\' && *(col+1) == 't') { raw[ri++] = '\t'; col += 2; }
                            else if (*col == '\\') { col += 2; raw[ri++] = '?'; }
                            else raw[ri++] = *col++;
                        }
                        raw[ri] = '\0';

                        if (strcmp((char*)raw, "<|endoftext|>") == 0) bos_id = id;
                        if (strcmp((char*)raw, "<|im_end|>") == 0) eos_id = id;
                        if (strcmp((char*)raw, "</s>") == 0) eos_id = id;

                        if (n_specials < MAX_SPECIAL && ri > 0) {
                            specials[n_specials].id = id;
                            memcpy(specials[n_specials].str, raw, ri);
                            specials[n_specials].len = ri;
                            n_specials++;
                        }
                    }}
                }
                p = cbrace + 1;
            }
        }
    }

    // Sort specials by length descending (longest match first)
    for (int i = 0; i < n_specials; i++)
        for (int j = i+1; j < n_specials; j++)
            if (specials[j].len > specials[i].len) {
                SpecialToken tmp = specials[i]; specials[i] = specials[j]; specials[j] = tmp;
            }

    fprintf(stderr, "[tokenize] BOS=%d EOS=%d specials=%d\n", bos_id, eos_id, n_specials);

    // Step 2: Parse main vocab
    const char* vocab_start = NULL;
    const char* model_sec = strstr(json, "\"model\"");
    if (model_sec) {
        const char* mv = strchr(model_sec, ':');
        if (mv) {
            const char* vk = strstr(mv, "\"vocab\"");
            if (vk) {
                const char* vv = strchr(vk, ':');
                if (vv) { vv++; while (*vv && isspace((unsigned char)*vv)) vv++;
                    if (*vv == '{' || *vv == '[') vocab_start = vv; }
            }
        }
    }
    if (!vocab_start) { fprintf(stderr, "Cannot find vocab\n"); free(json); return -1; }

    int token_count = 0;
    if (*vocab_start == '{') {
        const char* p = vocab_start + 1;
        while (p && *p && *p != '}' && token_count < MAX_VOCAB) {
            while (*p && (isspace((unsigned char)*p) || *p == ',')) p++;
            if (!*p || *p == '}') break;
            if (*p != '"') { p++; continue; }
            p++;
            unsigned char buf[MAX_TOKEN_LEN]; int blen = 0;
            while (*p && *p != '"' && blen < MAX_TOKEN_LEN-1) {
                if (*p == '\\' && *(p+1) == 'u') { p += 6; buf[blen++] = '?'; }
                else if (*p == '\\' && *(p+1) == 'n') { buf[blen++] = '\n'; p += 2; }
                else if (*p == '\\' && *(p+1) == 't') { buf[blen++] = '\t'; p += 2; }
                else if (*p == '\\') { p += 2; buf[blen++] = '?'; }
                else buf[blen++] = *p++;
            }
            if (*p == '"') p++;
            while (*p && (isspace((unsigned char)*p) || *p == ':')) p++;
            int id = atoi(p); while (*p && *p >= '0' && *p <= '9') p++;
            if (id >= 0 && id < MAX_VOCAB && blen > 0) {
                vocab[id].id = id; memcpy(vocab[id].bytes, buf, blen); vocab[id].len = blen;
                if (id >= vocab_size) vocab_size = id + 1;
            }
        }
    } else if (*vocab_start == '[') {
        const char* p = vocab_start + 1;
        while (p && *p && *p != ']' && token_count < MAX_VOCAB) {
            const char* ck = strstr(p, "\"content\"");
            if (!ck || ck > strchr(p, '}')) break;
            const char* col = strchr(ck, ':'); if (!col) { p = ck+1; continue; }
            col++; while (*col && isspace((unsigned char)*col)) col++;
            if (*col == '"') { col++;
                unsigned char buf[MAX_TOKEN_LEN]; int blen = 0;
                while (*col && *col != '"' && blen < MAX_TOKEN_LEN-1) {
                    if (*col == '\\' && *(col+1) == 'u') { col += 6; buf[blen++] = '?'; }
                    else if (*col == '\\' && *(col+1) == 'n') { buf[blen++] = '\n'; col += 2; }
                    else if (*col == '\\') { col += 2; buf[blen++] = '?'; }
                    else buf[blen++] = *col++;
                }
                const char* ik = strstr(p, "\"id\"");
                if (ik && ik < strchr(p, '}')) {
                    const char* ic = strchr(ik, ':');
                    if (ic) { int id = atoi(ic+1);
                        if (id >= 0 && id < MAX_VOCAB) {
                            vocab[id].id = id; memcpy(vocab[id].bytes, buf, blen); vocab[id].len = blen;
                            if (id >= vocab_size) vocab_size = id + 1;
                        }
                    }
                }
            }
            p = strchr(p, '}'); if (p) p++;
        }
    }

    free(json);
    fprintf(stderr, "[tokenize] loaded %d tokens\n", vocab_size);
    return (vocab_size > 0) ? 0 : -1;
}

// Tokenize: special tokens first, then greedy BPE with byte-fallback
static int tokenize_and_print(const unsigned char* input, int input_len) {
    int first = 1, pos = 0;

    while (pos < input_len) {
        // Try special token match first
        int sid, slen = match_special(input + pos, input_len - pos, &sid);
        if (slen > 0) {
            if (!first) printf(","); printf("%d", sid); first = 0;
            pos += slen;
            continue;
        }
        // Greedy BPE longest match with byte-fallback
        int mid, mlen = find_longest_match(input + pos, input_len - pos, &mid);
        if (mlen > 0) {
            if (!first) printf(","); printf("%d", mid); first = 0;
            pos += mlen;
        } else {
            pos++; // skip truly unknown byte
        }
    }
    printf("\n");
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: echo \"text\" | %s tokenizer.json\n", argv[0]); return 1; }
    if (load_tokenizer(argv[1]) != 0) {
        fprintf(stderr, "Failed to load tokenizer from %s\n", argv[1]); return 1; }
    fprintf(stderr, "[tokenize] loaded %d tokens (bos=%d eos=%d)\n", vocab_size, bos_id, eos_id);

    build_byte_fallback();

    unsigned char input[MAX_INPUT_LEN]; int len = 0; int c;
    while ((c = getchar()) != EOF && len < MAX_INPUT_LEN-1) input[len++] = (unsigned char)c;
    input[len] = '\0';
    fprintf(stderr, "[tokenize] input=%d bytes\n", len);
    return tokenize_and_print(input, len);
}

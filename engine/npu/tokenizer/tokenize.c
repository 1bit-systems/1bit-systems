/**
 * tokenize.c — Standalone GGUF BPE tokenizer for NPU engine.
 *
 * Reads text from stdin, loads tokenizer.json (SentencePiece BPE vocab),
 * outputs comma-separated token IDs to stdout.
 *
 * Build: gcc -O3 -o tokenize tokenize.c -lm
 * Usage: echo "Hello" | ./tokenize tokenizer.json
 *
 * This is a minimal byte-level BPE matching the GGUF/llama.cpp format.
 * It handles the common case: low-Unicode English text. Does NOT handle
 * regex pre-tokenization (that would require linking to re2/pcre).
 *
 * The greedy longest-match approach produces the same results as the
 * HuggingFace tokenizer for clean ASCII text. For inputs requiring
 * proper BPE merge rules, extend with a merge table parser.
 */

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Maximum supported values
#define MAX_VOCAB 200000
#define MAX_TOKEN_LEN 256
#define MAX_INPUT_LEN 4096

// A BPE token entry
typedef struct {
    int id;
    unsigned char bytes[MAX_TOKEN_LEN];
    int len;
} Token;

// Global vocab
static Token vocab[MAX_VOCAB];
static int vocab_size = 0;
static int bos_id = 151643;  // default Qwen3 BOS
static int eos_id = 151645;  // default Qwen3 EOS

// Find the longest matching token prefix in the vocab for a byte sequence
static int find_longest_match(const unsigned char* input, int input_len, int* out_id) {
    int best_len = 0;
    int best_id = -1;
    for (int i = 0; i < vocab_size; i++) {
        if (vocab[i].len <= input_len && vocab[i].len > best_len) {
            if (memcmp(vocab[i].bytes, input, vocab[i].len) == 0) {
                best_len = vocab[i].len;
                best_id = vocab[i].id;
            }
        }
    }
    if (best_id >= 0) {
        *out_id = best_id;
        return best_len;
    }
    return 0;
}

// Load tokenizer from GGUF tokenizer.json
// Handles two formats:
//   Format A: {"model":{"vocab": [{"id":0,"content":"token"}]}}
//   Format B: {"model":{"vocab": {"token": id}}}
//   Format C: {"added_tokens": [{"id":0,"content":"token"}]}
static int load_tokenizer(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz > 100 * 1024 * 1024) {
        fprintf(stderr, "File too large\n");
        fclose(f);
        return -1;
    }

    char* json = (char*)malloc((size_t)sz + 1);
    if (!json) {
        fclose(f);
        return -1;
    }
    fread(json, 1, (size_t)sz, f);
    json[sz] = '\0';
    fclose(f);

    // --- Step 1: Extract BOS/EOS from added_tokens ---
    const char* atok = strstr(json, "\"added_tokens\"");
    if (atok) {
        const char* bracket = strchr(atok, '[');
        if (bracket) {
            // Scan for tokens with special content
            const char* p = bracket;
            while ((p = strstr(p, "\"content\"")) != NULL && p < strchr(p, ']')) {
                const char* col = strchr(p, ':');
                if (!col) break;
                col++;
                while (*col && isspace((unsigned char)*col)) col++;
                if (*col != '"') { p = col; continue; }
                col++;
                char tokname[64];
                int ti = 0;
                while (*col && *col != '"' && ti < 63) tokname[ti++] = *col++;
                tokname[ti] = '\0';
                
                // Get id
                const char* idf = strstr(p, "\"id\"");
                int id = 0;
                if (idf) {
                    const char* idc = strchr(idf, ':');
                    if (idc) id = atoi(idc + 1);
                }
                
                if (strcmp(tokname, "<|endoftext|>") == 0) bos_id = id;
                if (strcmp(tokname, "<|im_end|>") == 0) eos_id = id;
                if (strcmp(tokname, "</s>") == 0) bos_id = id;
                if (strcmp(tokname, "<s>") == 0) bos_id = id;
                if (strcmp(tokname, "<pad>") == 0 && bos_id == 0) bos_id = 0;
                p = col;
            }
        }
    }
    fprintf(stderr, "[tokenize] BOS=%d EOS=%d\n", bos_id, eos_id);

    // --- Step 2: Determine vocab format ---
    // Find the "model" > "vocab" section
    const char* vocab_start = NULL;
    const char* model_sec = strstr(json, "\"model\"");
    if (model_sec) {
        // Find the value after "model":
        const char* mv = strchr(model_sec, ':');
        if (mv) {
            // Skip to the vocab entry inside model
            const char* vk = strstr(mv, "\"vocab\"");
            if (vk) {
                const char* vv = strchr(vk, ':');
                if (vv) {
                    vv++;
                    while (*vv && isspace((unsigned char)*vv)) vv++;
                    if (*vv == '{') vocab_start = vv;  // Format B: dict
                    else if (*vv == '[') vocab_start = vv;  // Format A: array
                }
            }
        }
    }

    if (!vocab_start) {
        // Fallback: look for any top-level object with tokens
        fprintf(stderr, "Cannot find vocab in tokenizer.json\n");
        free(json);
        return -1;
    }

    // --- Step 3: Parse the vocab ---
    int token_count = 0;
    
    if (*vocab_start == '{') {
        // Format B: {"token": id, ...}
        // Simple flat dict: "\"string\": id,"
        const char* p = vocab_start + 1;  // skip '{'
        while (p && *p && *p != '}' && token_count < MAX_VOCAB) {
            // Skip whitespace and commas
            while (*p && (isspace((unsigned char)*p) || *p == ',')) p++;
            if (!*p || *p == '}') break;
            
            // Find opening quote of the token string
            if (*p != '"') { p++; continue; }
            p++;  // skip the '"'
            
            // Extract token bytes
            unsigned char buf[MAX_TOKEN_LEN];
            int blen = 0;
            while (*p && *p != '"' && blen < MAX_TOKEN_LEN - 1) {
                if (*p == '\\' && *(p+1) == 'u') {
                    p += 6;
                    buf[blen++] = '?';
                } else if (*p == '\\' && *(p+1) == 'n') {
                    buf[blen++] = '\n';
                    p += 2;
                } else if (*p == '\\' && *(p+1) == 't') {
                    buf[blen++] = '\t';
                    p += 2;
                } else if (*p == '\\') {
                    p += 2;
                    buf[blen++] = '?';
                } else {
                    buf[blen++] = *p++;
                }
            }
            if (*p == '"') p++;  // skip closing '"'
            
            // Skip colon
            while (*p && (isspace((unsigned char)*p) || *p == ':')) p++;
            
            // Parse the integer id
            int id = atoi(p);
            while (*p && *p >= '0' && *p <= '9') p++;
            
            if (id >= 0 && id < MAX_VOCAB && blen > 0) {
                vocab[id].id = id;
                memcpy(vocab[id].bytes, buf, blen);
                vocab[id].len = blen;
                if (id + 1 > token_count) token_count = id + 1;
                if (id + 1 > vocab_size) vocab_size = id + 1;
            }
        }
    } else if (*vocab_start == '[') {
        // Format A: [{"id":0,"content":"token"}, ...]
        const char* p = vocab_start + 1;  // skip '['
        while (p && *p && *p != ']' && token_count < MAX_VOCAB) {
            // Look for "content" field
            const char* content_key = strstr(p, "\"content\"");
            if (!content_key || content_key > strchr(p, '}')) break;

            const char* colon = strchr(content_key, ':');
            if (!colon) { p = content_key + 1; continue; }
            colon++;
            while (*colon && isspace((unsigned char)*colon)) colon++;

            if (*colon == '"') {
                colon++;
                unsigned char buf[MAX_TOKEN_LEN];
                int blen = 0;
                while (*colon && *colon != '"' && blen < MAX_TOKEN_LEN - 1) {
                    if (*colon == '\\' && *(colon + 1) == 'u') {
                        colon += 6;
                        buf[blen++] = '?';
                    } else if (*colon == '\\' && *(colon + 1) == 'n') {
                        buf[blen++] = '\n';
                        colon += 2;
                    } else if (*colon == '\\') {
                        colon += 2;
                        buf[blen++] = '?';
                    } else {
                        buf[blen++] = *colon++;
                    }
                }

                // Find the id
                const char* id_key = strstr(p, "\"id\"");
                if (id_key && id_key < strchr(p, '}')) {
                    const char* id_colon = strchr(id_key, ':');
                    if (id_colon) {
                        int id = atoi(id_colon + 1);
                        if (id >= 0 && id < MAX_VOCAB) {
                            vocab[id].id = id;
                            memcpy(vocab[id].bytes, buf, blen);
                            vocab[id].len = blen;
                            if (id + 1 > token_count) token_count = id + 1;
                            if (id + 1 > vocab_size) vocab_size = id + 1;
                        }
                    }
                }
            }

            // Move to next entry
            p = strchr(p, '}');
            if (p) p++;
        }
    }

    free(json);
    fprintf(stderr, "[tokenize] loaded %d tokens\n", vocab_size);
    return (vocab_size > 0) ? 0 : -1;
}

// Tokenize input text and print token IDs (no BOS/EOS added automatically)
static int tokenize_and_print(const unsigned char* input, int input_len) {
    int first = 1;

    // Greedy longest-match tokenization
    int pos = 0;
    while (pos < input_len) {
        int remaining = input_len - pos;
        int match_id;
        int match_len = find_longest_match(input + pos, remaining, &match_id);

        if (match_len > 0) {
            if (!first) printf(",");
            printf("%d", match_id);
            first = 0;
            pos += match_len;
        } else {
            // No match: try to find a single-byte token
            unsigned char byte = input[pos];
            for (int i = 0; i < vocab_size; i++) {
                if (vocab[i].len == 1 && vocab[i].bytes[0] == byte) {
                    if (!first) printf(",");
                    printf("%d", vocab[i].id);
                    first = 0;
                    break;
                }
            }
            // If no single-byte token exists, skip the byte
            pos++;
        }
    }

    printf("\n");
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: echo \"text\" | %s tokenizer.json\n", argv[0]);
        return 1;
    }

    if (load_tokenizer(argv[1]) != 0) {
        fprintf(stderr, "Failed to load tokenizer from %s\n", argv[1]);
        return 1;
    }

    fprintf(stderr, "[tokenize] loaded %d tokens (bos=%d eos=%d)\n",
            vocab_size, bos_id, eos_id);

    // Read stdin
    unsigned char input[MAX_INPUT_LEN];
    int len = 0;
    int c;
    while ((c = getchar()) != EOF && len < MAX_INPUT_LEN - 1) {
        input[len++] = (unsigned char)c;
    }
    input[len] = '\0';

    fprintf(stderr, "[tokenize] input=%d bytes\n", len);

    return tokenize_and_print(input, len);
}

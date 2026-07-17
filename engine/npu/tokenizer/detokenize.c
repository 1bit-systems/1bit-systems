// detokenize.c — ID→text for NPU engine. Reads comma-sep IDs from stdin.
// gcc -O3 -o detokenize detokenize.c -lm
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#define MAX_VOCAB 200000
#define MAX_TOKEN_LEN 256

static char vocab[MAX_VOCAB][MAX_TOKEN_LEN];
static int vocab_size = 0;

static void load(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    char line[512]; int in_vocab = 0;
    while (fgets(line, sizeof(line), f)) {
        if (!in_vocab) {
            if (strstr(line, "\"vocab\"")) in_vocab = 1;
            continue;
        }
        if (strstr(line, "}") && !strstr(line, "\"")) break;
        // Parse "token_string": id
        char *tok = strchr(line, '"');
        if (!tok) continue;
        tok++; char *te = strchr(tok, '"');
        if (!te) continue;
        int tlen = te - tok;
        if (tlen >= MAX_TOKEN_LEN) tlen = MAX_TOKEN_LEN - 1;
        char *idp = strchr(te + 1, ':');
        if (!idp) continue;
        int id = (int)strtol(idp + 1, NULL, 10);
        if (id < 0 || id >= MAX_VOCAB) continue;
        memcpy(vocab[id], tok, tlen);
        vocab[id][tlen] = 0;
        if (id >= vocab_size) vocab_size = id + 1;
    }
    // Merge added_tokens array
    rewind(f); in_vocab = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, "\"added_tokens\"")) { in_vocab = 1; continue; }
        if (!in_vocab) continue;
        if (strstr(line, "]")) break;
        char *idp = strstr(line, "\"id\"");
        char *ctp = strstr(line, "\"content\"");
        if (!idp || !ctp) continue;
        int id = (int)strtol(strchr(idp, ':') + 1, NULL, 10);
        char *cs = strchr(ctp, '"') + 1;
        char *ce = strchr(cs, '"');
        if (id < 0 || id >= MAX_VOCAB || !ce) continue;
        int clen = ce - cs;
        if (clen >= MAX_TOKEN_LEN) clen = MAX_TOKEN_LEN - 1;
        memcpy(vocab[id], cs, clen);
        vocab[id][clen] = 0;
        if (id >= vocab_size) vocab_size = id + 1;
    }
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: detokenize <tokenizer.json>\n"); return 1; }
    load(argv[1]);
    setvbuf(stdout, NULL, _IONBF, 0);
    char line[65536]; int id;
    while (fgets(line, sizeof(line), stdin)) {
        char *p = line;
        while (*p) {
            while (*p && !isdigit(*p) && *p != '-') p++;
            if (!*p) break;
            id = (int)strtol(p, &p, 10);
            if (id >= 0 && id < vocab_size && vocab[id][0])
                printf("%s", vocab[id]);
            if (*p == ',') p++;
        }
    }
    printf("\n");
    return 0;
}

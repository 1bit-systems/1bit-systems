// detokenize.cpp — ID→text for NPU engine. Reads comma-sep IDs from stdin.
// g++ -std=c++17 -O3 -o detokenize detokenize.cpp
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

constexpr int kMaxVocab    = 200000;
constexpr int kMaxTokenLen = 256;

static char   vocab[kMaxVocab][kMaxTokenLen]{};
static int    vocab_size = 0;

static void load(const char *path) {
    FILE *f = std::fopen(path, "r");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    char line[512];
    bool in_vocab = false;
    while (std::fgets(line, sizeof(line), f)) {
        if (!in_vocab) {
            if (std::strstr(line, "\"vocab\"")) in_vocab = true;
            continue;
        }
        if (std::strstr(line, "}") && !std::strstr(line, "\"")) break;
        char *tok = std::strchr(line, '"');
        if (!tok) continue;
        tok++;
        char *te = std::strchr(tok, '"');
        if (!te) continue;
        int tlen = static_cast<int>(te - tok);
        if (tlen >= kMaxTokenLen) tlen = kMaxTokenLen - 1;
        char *idp = std::strchr(te + 1, ':');
        if (!idp) continue;
        int id = static_cast<int>(std::strtol(idp + 1, nullptr, 10));
        if (id < 0 || id >= kMaxVocab) continue;
        std::memcpy(vocab[id], tok, tlen);
        vocab[id][tlen] = 0;
        if (id >= vocab_size) vocab_size = id + 1;
    }
    // Merge added_tokens array
    std::rewind(f);
    in_vocab = false;
    while (std::fgets(line, sizeof(line), f)) {
        if (std::strstr(line, "\"added_tokens\"")) { in_vocab = true; continue; }
        if (!in_vocab) continue;
        if (std::strstr(line, "]")) break;
        char *idp = std::strstr(line, "\"id\"");
        char *ctp = std::strstr(line, "\"content\"");
        if (!idp || !ctp) continue;
        int id = static_cast<int>(std::strtol(std::strchr(idp, ':') + 1, nullptr, 10));
        char *cs = std::strchr(ctp, '"') + 1;
        char *ce = std::strchr(cs, '"');
        if (id < 0 || id >= kMaxVocab || !ce) continue;
        int clen = static_cast<int>(ce - cs);
        if (clen >= kMaxTokenLen) clen = kMaxTokenLen - 1;
        std::memcpy(vocab[id], cs, clen);
        vocab[id][clen] = 0;
        if (id >= vocab_size) vocab_size = id + 1;
    }
    std::fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: detokenize <tokenizer.json>\n");
        return 1;
    }
    load(argv[1]);
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    char line[65536];
    while (std::fgets(line, sizeof(line), stdin)) {
        char *p = line;
        while (*p) {
            while (*p && !std::isdigit(static_cast<unsigned char>(*p)) && *p != '-') p++;
            if (!*p) break;
            int id = static_cast<int>(std::strtol(p, &p, 10));
            if (id >= 0 && id < vocab_size && vocab[id][0])
                std::printf("%s", vocab[id]);
            if (*p == ',') p++;
        }
    }
    std::printf("\n");
    return 0;
}

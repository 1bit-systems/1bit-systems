// run_albert2.cpp — Pure C++ Albert-MoE-13 inference via flat binary
// Build: python3 tools/dump_albert_flat.py && g++ -std=c++17 -O3 run_albert2.cpp -o run_albert2 && ./run_albert2
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

constexpr int kHidden   = 256;
constexpr int kNHeads   = 4;
constexpr int kHeadDim  = 64;
constexpr int kNExp     = 12;
constexpr int kTopK     = 3;
constexpr int kVocab    = 32000;
constexpr int kFfnDim   = 1024;
constexpr int kMaxSeq   = 256;

static float  *g_weights      = nullptr;
static size_t  g_weight_count = 0;

// Load flat binary: [count:4] then entries[name_len:4,name_off:4,data_off:8,numel:8],
// then name strings, then weight data
static void load_flat(const char *path) {
    FILE *f = std::fopen(path, "rb");
    if (!f) { std::perror(path); std::exit(1); }
    std::fseek(f, 0, SEEK_END);
    size_t sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    g_weights = static_cast<float *>(std::malloc(sz));
    std::fread(g_weights, 1, sz, f);
    std::fclose(f);

    auto *p = reinterpret_cast<unsigned char *>(g_weights);
    g_weight_count = *reinterpret_cast<unsigned int *>(p);
    p += 4;

    int nl = *reinterpret_cast<int *>(p);
    int no = *reinterpret_cast<int *>(p + 4);
    std::printf("Tensors: %zu, first name at offset %d: \"%s\"\n",
                g_weight_count, no,
                reinterpret_cast<char *>(g_weights + no));
}

static float *get_w(const char *name) {
    auto *p = reinterpret_cast<unsigned char *>(g_weights) + 4;
    static long long ds = 0;

    for (size_t i = 0; i < g_weight_count; i++) {
        int        nl       = *reinterpret_cast<int *>(p); p += 4;
        int        no       = *reinterpret_cast<int *>(p); p += 4;
        long long  data_off = *reinterpret_cast<long long *>(p); p += 8;
        long long  numel    = *reinterpret_cast<long long *>(p); p += 8;

        if (std::strcmp(reinterpret_cast<char *>(g_weights) + no, name) == 0) {
            if (i == 0 && !ds) {
                ds = 4 + g_weight_count * 24;
                auto *pp = reinterpret_cast<unsigned char *>(g_weights) + 4;
                for (size_t j = 0; j < g_weight_count; j++) {
                    int nnl = *reinterpret_cast<int *>(pp); pp += 4;
                    pp += 4 + 16;
                    ds += nnl;
                }
            }
            return reinterpret_cast<float *>(
                reinterpret_cast<unsigned char *>(g_weights) + ds + data_off * 4);
        }
    }
    std::fprintf(stderr, "Missing: %s\n", name);
    static float zero_buf[1024 * 1024]{};
    return zero_buf;
}

int main() {
    std::printf("Loading Albert flat binary...\n");
    load_flat("/tmp/albert_flat.bin");

    float *embed = get_w("embed.weight");
    std::printf("embed[0:4]: %.2f %.2f %.2f %.2f\n",
                embed[0], embed[1], embed[2], embed[3]);
    std::printf("Done\n");
    return 0;
}

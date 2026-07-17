// run_albert2.c — Pure C Albert-MoE-13 inference via flat binary
// Build: python3 tools/dump_albert_flat.py && gcc -O3 run_albert2.c -lm -o run_albert2 && ./run_albert2
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define H 256
#define N_HEADS 4
#define HEAD_DIM 64
#define N_EXP 12
#define TOP_K 3
#define VOCAB 32000
#define FFN_HIDDEN 1024
#define MAX_SEQ 256

static float *g_weights = NULL;
static size_t g_weight_count = 0;

// Load flat binary: [count:4] then entries[name_len:4,name_off:4,data_off:8,numel:8],
// then name strings, then weight data
static void load_flat(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
    g_weights = malloc(sz);
    fread(g_weights, 1, sz, f); fclose(f);
    
    unsigned char *p = (unsigned char*)g_weights;
    g_weight_count = *(unsigned int*)p; p += 4;
    
    // Verify by checking first tensor name
    int nl = *(int*)p;
    int no = *(int*)(p+4);
    printf("Tensors: %zu, first name at offset %d: \"%s\"\n", g_weight_count, no, (char*)(g_weights + no));
    
    // Free unused prefix
    // (keep all data in memory)
}

static float *get_w(const char *name) {
    unsigned char *p = (unsigned char*)g_weights + 4; // skip count
    int name_off_base = 4 + g_weight_count * 24; // after all entries
    
    for (size_t i = 0; i < g_weight_count; i++) {
        int nl = *(int*)p; p += 4;
        int no = *(int*)p; p += 4;
        long long data_off = *(long long*)p; p += 8;
        long long numel = *(long long*)p; p += 8;
        
        if (strcmp((char*)g_weights + no, name) == 0) {
            // Data starts at name_off_base + (name table size) + data_off
            // Actually the data is at the very end of the file
            long long data_start = 4 + g_weight_count * 24; // entries
            // names start here too, but we need to skip to data
            // Find the start of the data section (after name strings)
            if (i == 0) {
                // First call: cache data_start
                static long long ds = 0;
                if (!ds) {
                    // Scan all names to find total name size
                    ds = 4 + g_weight_count * 24;
                    unsigned char *pp = (unsigned char*)g_weights + 4;
                    for (size_t j = 0; j < g_weight_count; j++) {
                        int nnl = *(int*)pp; pp += 4;
                        pp += 4; pp += 16;
                        ds += nnl; // name bytes including null
                    }
                }
                return (float*)(g_weights + ds + data_off * 4);
            }
        }
    }
    fprintf(stderr, "Missing: %s\n", name);
    // Return zero buffer
    static float zero_buf[1024*1024]; memset(zero_buf, 0, sizeof(zero_buf));
    return zero_buf;
}

int main() {
    printf("Loading Albert flat binary...\n");
    load_flat("/tmp/albert_flat.bin");
    
    float *embed = get_w("embed.weight");
    printf("embed[0:4]: %.2f %.2f %.2f %.2f\n", embed[0], embed[1], embed[2], embed[3]);
    printf("Done\n");
    return 0;
}

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <dlfcn.h>
#include <unistd.h>

typedef void (*mha_gen_t)(void*, unsigned int, unsigned int, unsigned int, bool, int);
static mha_gen_t real_mha_gen = nullptr;

void scan_for_txn(void* base, size_t max_size, const char* label) {
    if (!base) return;
    uint32_t* p = (uint32_t*)base;
    size_t n = max_size / 4;
    for (size_t i = 0; i < n - 10; i++) {
        if (p[i] == 0x06040100) {
            uint32_t word_count = p[i+3];
            if (word_count > 0 && word_count < 10000) {
                fprintf(stderr, "\n[txn] FOUND at %s offset %zu (%u words):\n", label, i * 4, word_count);
                for (uint32_t j = 0; j < word_count && j < 16; j++)
                    fprintf(stderr, "  [%4u] 0x%08x\n", j, p[i + j]);
                if (word_count > 16)
                    fprintf(stderr, "  ... (%u more words)\n", word_count - 16);
                char fname[256];
                snprintf(fname, sizeof(fname), "/tmp/attn_txn_%lu.bin", (unsigned long)time(NULL));
                FILE* f = fopen(fname, "wb");
                if (f) { fwrite(p + i, 4, word_count, f); fclose(f); fprintf(stderr, "  Saved to %s\n", fname); }
                return;
            }
        }
    }
}

extern "C" void _ZN3MHA21generate_mha_sequenceEP12npu_sequencejjjbi(
    void* seq, unsigned int a, unsigned int b, unsigned int c, bool d, int e) {
    
    if (!real_mha_gen)
        real_mha_gen = (mha_gen_t)dlsym(RTLD_NEXT, "_ZN3MHA21generate_mha_sequenceEP12npu_sequencejjjbi");
    
    fprintf(stderr, "\n[MHA] generate_mha_sequence(seq=%p, %u, %u, %u, %d, %d)\n", seq, a, b, c, d, e);
    scan_for_txn(seq, 65536, "pre-call");
    
    real_mha_gen(seq, a, b, c, d, e);
    fprintf(stderr, "[MHA] call complete\n");
    
    scan_for_txn(seq, 65536, "post-call");
    if (seq) {
        for (size_t off = 0; off < 0x100000; off += 0x1000)
            scan_for_txn((char*)seq + off, 4096, "scan");
    }
}

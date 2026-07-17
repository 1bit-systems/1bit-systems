// Capture attention txn data from npu_sequence after MHA::generate_mha_sequence
// Chases internal buffer pointers to find the generated txn data.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <dlfcn.h>
#include <unistd.h>

typedef void (*mha_gen_t)(void*, unsigned int, unsigned int, unsigned int, bool, int);
static mha_gen_t real_mha_gen = nullptr;

// Dump hex region
void dump_hex(const char* label, void* addr, size_t len) {
    if (!addr) return;
    uint32_t* p = (uint32_t*)addr;
    fprintf(stderr, "  [%s] %p:", label, addr);
    for (size_t i = 0; i < len/4 && i < 12; i++) fprintf(stderr, " %08x", p[i]);
    fprintf(stderr, "\n");
}

// Scan a buffer for txn header
void scan_for_txn(void* base, size_t max_size, const char* label) {
    if (!base || max_size < 16) return;
    uint32_t* p = (uint32_t*)base;
    size_t n = max_size / 4;
    for (size_t i = 0; i < n - 10; i++) {
        if (p[i] == 0x06040100) {
            uint32_t word_count = p[i+3];
            if (word_count > 0 && word_count < 50000) {
                fprintf(stderr, "\n*** FOUND TXN at %s offset %zu (%u words) ***\n", label, i * 4, word_count);
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

// Chase pointers in the sequence object
void chase_pointers(void* seq, size_t max_offset) {
    if (!seq) return;
    uintptr_t* words = (uintptr_t*)seq;
    size_t n_words = max_offset / sizeof(uintptr_t);
    
    fprintf(stderr, "\n=== Chasing pointers in npu_sequence %p ===\n", seq);
    
    for (size_t i = 0; i < n_words && i < 128; i++) {
        uintptr_t val = words[i];
        // Check if value looks like a pointer (in heap range)
        if (val > 0x100000000 && val < 0x800000000000 && val != (uintptr_t)seq) {
            // Try to read from this address
            uint32_t* possible_txn = (uint32_t*)val;
            // Check if it contains the txn header
            if (possible_txn[0] == 0x06040100 || possible_txn[0] == 0x00010406) {
                fprintf(stderr, "  [%4zu] POINTER to txn data: %p (contains 0x%08x)\n", i, (void*)val, possible_txn[0]);
                scan_for_txn((void*)val, 65536, "pointer_chase");
            }
            // Check if it points to another structure with more pointers
            if (possible_txn[0] > 0x10000 && possible_txn[0] < 0xFFFFFFFF) {
                // Might be a buffer object - check its first few words
                uintptr_t* inner = (uintptr_t*)val;
                for (int j = 0; j < 8; j++) {
                    if (inner[j] > 0x100000000 && inner[j] < 0x800000000000) {
                        uint32_t* inner_txn = (uint32_t*)inner[j];
                        if (inner_txn[0] == 0x06040100) {
                            fprintf(stderr, "  [%4zu] NESTED pointer from [%d]: %p\n", i, j, (void*)inner[j]);
                            scan_for_txn((void*)inner[j], 65536, "nested_chase");
                        }
                    }
                }
            }
        }
    }
    
    // Also dump the first 64 bytes of the sequence object itself
    fprintf(stderr, "\n  First 64 bytes of npu_sequence:\n");
    for (size_t i = 0; i < 8; i++) {
        fprintf(stderr, "  [%4zu] 0x%016lx", i, (unsigned long)words[i]);
        if (words[i] > 0x100000000 && words[i] < 0x800000000000)
            fprintf(stderr, " -> %p", (void*)words[i]);
        fprintf(stderr, "\n");
    }
}

// Hook
extern "C" void _ZN3MHA21generate_mha_sequenceEP12npu_sequencejjjbi(
    void* seq, unsigned int a, unsigned int b, unsigned int c, bool d, int e) {
    
    if (!real_mha_gen)
        real_mha_gen = (mha_gen_t)dlsym(RTLD_NEXT, "_ZN3MHA21generate_mha_sequenceEP12npu_sequencejjjbi");
    
    fprintf(stderr, "\n[MHA] generate_mha_sequence(seq=%p, %u, %u, %u, %d, %d)\n", seq, a, b, c, d, e);
    
    // Chase pointers before call (may find pre-allocated txn buffer)
    chase_pointers(seq, 1024);
    
    // Call real function
    real_mha_gen(seq, a, b, c, d, e);
    fprintf(stderr, "[MHA] call complete\n");
    
    // Chase pointers after call
    chase_pointers(seq, 2048);
    
    // Scan various offsets from seq for txn data
    for (size_t off = 0; off < 0x200000; off += 0x1000) {
        scan_for_txn((char*)seq + off, 4096, "offset_scan");
    }
}

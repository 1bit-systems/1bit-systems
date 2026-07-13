// dump_instrs.cpp — Dump NPU instructions from FLM libraries
// Links against libqwen3_npu.so, libgemm.so, libdequant.so
// to call the instruction sequence generators and dump output.
//
// Usage: LD_LIBRARY_PATH=/opt/fastflowlm/lib/flm ./dump_instrs

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <dlfcn.h>
#include <link.h>

// We need LM_Config to construct qwen3_npu_sequence
// From assembly analysis, LM_Config has:
// - std::string[5] at offsets 0, 0x20, 0x40, 0x60, 0x88 (each string is 32 bytes)
// - uint64_t at 0xa8, 0xb0, 0xb8, 0xc0, 0xc8, 0xd0, 0xd8, 0xe0
// - Total size ≈ 0xe8 (232) bytes

// std::string in libstdc++ (SSO):
// - offset 0: pointer to data (or inline buffer if short)
// - offset 8: size
// - offset 16: union { capacity, inline_buffer[16] }
// Total: 32 bytes per string

struct LM_Config {
    // 5 strings: model_path, model_name, cache_dir, xclbin_dir, ??
    std::string str0;      // 0x00
    std::string str1;      // 0x20
    std::string str2;      // 0x40
    std::string str3;      // 0x60
    uint64_t pad_80[1];    // 0x80 - padding?
    std::string str4;      // 0x88
    // Integer/config fields
    uint64_t field_a8;     // 0xa8
    uint64_t field_b0;     // 0xb0
    uint64_t field_b8;     // 0xb8
    uint64_t field_c0;     // 0xc0
    uint64_t field_c8;     // 0xc8
    uint64_t field_d0;     // 0xd0
    uint64_t field_d8;     // 0xd8
    uint64_t field_e0;     // 0xe0
};

// Check sizeof
static_assert(sizeof(LM_Config) <= 256, "LM_Config too large");

int main(int argc, char* argv[]) {
    // Load all FLM libraries
    void* qwen_handle = dlopen("/opt/fastflowlm/lib/flm/libqwen3_npu.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!qwen_handle) {
        fprintf(stderr, "ERROR: dlopen libqwen3_npu.so: %s\n", dlerror());
        return 1;
    }
    
    void* gemm_handle = dlopen("/opt/fastflowlm/lib/flm/libgemm.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!gemm_handle) {
        fprintf(stderr, "ERROR: dlopen libgemm.so: %s\n", dlerror());
        return 1;
    }
    
    void* dequant_handle = dlopen("/opt/fastflowlm/lib/flm/libdequant.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!dequant_handle) {
        fprintf(stderr, "ERROR: dlopen libdequant.so: %s\n", dlerror());
        return 1;
    }
    
    // Get library base addresses for computing weak symbol addresses
    struct link_map* lm_qwen = NULL;
    dlinfo(qwen_handle, RTLD_DI_LINKMAP, &lm_qwen);
    uintptr_t qwen_base = lm_qwen->l_addr;
    
    fprintf(stderr, "libqwen3_npu.so base: 0x%lx\n", qwen_base);
    
    // Compute weak symbol addresses from file offsets (from nm output)
    // These are weak symbols in .symtab but NOT exported in .dynsym
    // cmds2seq @ 0x59a70, clear_cmds @ 0x5a2d0
    // rtp_write @ 0x59dc0, npu_preemption @ 0x5a090
    // npu_dma_memcpy_nd @ 0x69160 (constprop version)
    
    // Use dlsym for exported symbols
    typedef void* (*qwen_seq_ctor_t)(void*, LM_Config, uint32_t);
    typedef void (*qwen_seq_dtor_t)(void*);
    typedef void (*gen_layer_seq_t)(void*, void*, uint32_t);
    typedef void (*gen_lm_head_t)(void*, void*);
    typedef void (*gen_dequant_t)(void*, void*, uint32_t, uint32_t, uint32_t);
    typedef void (*gen_mha_engine_t)(void*, void*, uint32_t, uint32_t);
    typedef void (*gemm_gen_t)(void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, bool, uint32_t, uint32_t);
    typedef void (*dequant_gen_t)(void*, void*, uint32_t, uint32_t, uint32_t, int);
    typedef void (*cmds2seq_t)(void*, std::vector<uint32_t>&);
    typedef void (*clear_cmds_t)(void*);
    
    auto qwen_seq_ctor = (qwen_seq_ctor_t)dlsym(qwen_handle, "_ZN18qwen3_npu_sequenceC1E9LM_Configj");
    auto qwen_seq_dtor = (qwen_seq_dtor_t)dlsym(qwen_handle, "_ZN18qwen3_npu_sequenceD1Ev");
    auto gen_layer_seq = (gen_layer_seq_t)dlsym(qwen_handle, "_ZN18qwen3_npu_sequence4Impl13gen_layer_seqEP12npu_sequencej");
    auto gen_lm_head = (gen_lm_head_t)dlsym(qwen_handle, "_ZN18qwen3_npu_sequence4Impl15gen_lm_head_seqEP12npu_sequence");
    auto gen_mha_engine = (gen_mha_engine_t)dlsym(qwen_handle, "_ZN18qwen3_npu_sequence4Impl18gen_mha_engine_seqEP12npu_sequencejj");
    auto gemm_gen = (gemm_gen_t)dlsym(gemm_handle, "_ZN4Gemm4Impl12generate_seqEP12npu_sequencejjjjbNS_17Activation_Type_tEjj");
    auto dequant_gen = (dequant_gen_t)dlsym(dequant_handle, "_ZN7Dequant4Impl25generate_dequant_q4_1_seqEP12npu_sequencejjjNS0_21dequant_output_mode_tE");
    
    // Weak symbols - compute address from base + file offset
    // From objdump -t: cmds2seq @ 0x59a70, clear_cmds @ 0x5a2d0
    // libqwen3_npu.so .text section starts at 0x25130 (which equals VirtAddr)
    // So file offset = virtual address for .text LOAD segment
    cmds2seq_t cmds2seq_fn = (cmds2seq_t)(qwen_base + 0x59a70);
    clear_cmds_t clear_cmds_fn = (clear_cmds_t)(qwen_base + 0x5a2d0);
    
    fprintf(stderr, "\nResolved symbols:\n");
    fprintf(stderr, "  qwen_seq_ctor: %p\n", (void*)qwen_seq_ctor);
    fprintf(stderr, "  qwen_seq_dtor: %p\n", (void*)qwen_seq_dtor);
    fprintf(stderr, "  gen_layer_seq: %p\n", (void*)gen_layer_seq);
    fprintf(stderr, "  gen_lm_head: %p\n", (void*)gen_lm_head);
    fprintf(stderr, "  gen_mha_engine: %p\n", (void*)gen_mha_engine);
    fprintf(stderr, "  gemm_gen: %p\n", (void*)gemm_gen);
    fprintf(stderr, "  dequant_gen: %p\n", (void*)dequant_gen);
    fprintf(stderr, "  cmds2seq (weak): %p\n", (void*)cmds2seq_fn);
    fprintf(stderr, "  clear_cmds (weak): %p\n", (void*)clear_cmds_fn);
    
    if (!qwen_seq_ctor || !gen_layer_seq || !cmds2seq_fn || !gemm_gen) {
        fprintf(stderr, "ERROR: missing required symbols\n");
        return 1;
    }
    
    // Construct LM_Config with Qwen3-0.6B parameters
    LM_Config config;
    memset(&config, 0, sizeof(config));
    
    // Set model info strings
    config.str0 = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2";
    config.str1 = "qwen3:0.6b";
    config.str2 = "/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2";
    config.str3 = "/opt/fastflowlm/lib/flm";
    
    // Set model dimensions (from config.json)
    // hidden_size=1024, num_layers=28, num_heads=16, num_kv_heads=8
    // intermediate_size=3072, head_dim=128, vocab_size=151936
    config.field_a8 = 1024;      // hidden_size
    config.field_b0 = 28;        // num_layers
    config.field_b8 = 16;        // num_attention_heads
    config.field_c0 = 8;         // num_key_value_heads
    config.field_c8 = 3072;      // intermediate_size
    config.field_d0 = 128;       // head_dim
    config.field_d8 = 151936;    // vocab_size
    config.field_e0 = 40960;     // max_position_embeddings
    
    // Allocate qwen3_npu_sequence on heap
    // From the constructor disassembly, it allocates 0x70 bytes for Impl
    // and qwen3_npu_sequence itself is probably ~16 bytes (just a unique_ptr or pointer to Impl)
    // Let's allocate enough
    void* seq = malloc(512);
    memset(seq, 0, 512);
    
    // Call constructor
    fprintf(stderr, "\nConstructing qwen3_npu_sequence(max_length=4096)...\n");
    qwen_seq_ctor(seq, config, 4096);
    fprintf(stderr, "  Done\n");
    
    // Now get the Impl pointer (first 8 bytes of seq is likely unique_ptr to Impl)
    void** impl_ptr = (void**)seq;
    void* impl = *impl_ptr;
    fprintf(stderr, "  Impl: %p\n", impl);
    
    // Call gen_layer_seq to generate instructions for layer 0
    fprintf(stderr, "\nGenerating layer sequence for layer 0...\n");
    
    // gen_layer_seq signature: Impl::gen_layer_seq(npu_sequence*, layer_idx)
    // The first arg is the Impl's npu_sequence, second is layer index
    // But actually gen_layer_seq is on Impl, so: impl->gen_layer_seq(npu_seq, layer)
    // Wait, looking at the demangled name:
    // _ZN18qwen3_npu_sequence4Impl13gen_layer_seqEP12npu_sequencej
    // Impl::gen_layer_seq(npu_sequence* seq, uint32_t layer)
    // So 'this' is Impl*, first arg is npu_sequence*, second is layer
    
    // But npu_sequence itself - where is it? Looking at the qwen3_npu_sequence class:
    // It likely holds a npu_sequence as a member or the Impl has it
    // From the _send_x method: _ZN18qwen3_npu_sequence4Impl7_send_xEP12npu_sequence
    // This takes npu_sequence* as the first arg (after this)
    
    // So gen_layer_seq(impl, seq, layer_num) where seq is the npu_sequence to fill
    // But wait - in the constructor, it creates an Impl which creates npu_sequence internally
    // Let me check: does qwen3_npu_sequence have its own npu_sequence member?
    
    // The npu_sequence is actually part of qwen3_npu_sequence. Let me look at the
    // constructor more carefully
    // The constructor allocates an Impl (0x70 bytes) and constructs it
    // The Impl constructor takes LM_Config and uint32_t
    // So the Impl creates its own internal npu_sequence
    // Actually no - looking at gen_layer_seq: it takes npu_sequence* as parameter
    // This means the npu_sequence is separate from Impl
    
    // Let me look at what npu_sequence's size is by checking if there's a new/malloc in clear_cmds
    // clear_cmds is at qwen_base + 0x5a2d0
    
    // For now, let me try different call patterns
    
    // Pattern 1: Call gen_layer_seq(impl, seq_pointer, 0)
    // where seq_pointer is the npu_sequence stored inside qwen3_npu_sequence
    // (maybe at offset 0 of qwen3_npu_sequence if the first member IS npu_sequence)
    
    fprintf(stderr, "\nTrying to generate sequences...\n");
    
    // The npu_sequence might be at offset 0 of our seq allocation
    // (since qwen3_npu_sequence inherits from or contains npu_sequence)
    // Or it might be through the Impl pointer
    
    // Let me check if the first member is the Impl pointer
    // If impl_ptr != NULL and != seq, then seq is a wrapper with a pointer
    // If impl_ptr == seq, then the object IS the Impl
    
    // Let me try calling with seq as the npu_sequence pointer
    // (npu_sequence might start at the beginning of the object)
    
    // Actually, the clearest approach: look at what gen_layer_seq does
    // It takes npu_sequence*, so the seq object IS npu_sequence
    // And gen_layer_seq is on Impl, so: impl->gen_layer_seq(npu_seq, layer)
    // But we need Impl to have the config to know tile layout
    
    gen_layer_seq(impl, seq, 0);
    fprintf(stderr, "  gen_layer_seq completed\n");
    
    // Get instructions
    std::vector<uint32_t> instrs;
    cmds2seq_fn(seq, instrs);
    fprintf(stderr, "  Got %zu instructions\n", instrs.size());
    fprintf(stderr, "  cmds2seq: %p, gemm_gen: %p\n", (void*)cmds2seq_fn, (void*)gemm_gen);
    
    if (!instrs.empty()) {
        printf("// Layer 0 instructions (%zu dwords = %zu bytes):\n", 
               instrs.size(), instrs.size() * 4);
        for (size_t i = 0; i < instrs.size(); i++) {
            printf("instrs[%zu] = 0x%08x\n", i, instrs[i]);
        }
    } else {
        fprintf(stderr, "  No instructions generated\n");
    }
    
    // Clean up
    qwen_seq_dtor(seq);
    free(seq);
    
    dlclose(qwen_handle);
    dlclose(gemm_handle);
    dlclose(dequant_handle);
    
    return 0;
}

// dump_instrs2.cpp — Minimal instruction dumper
// Skips qwen3_npu_sequence, directly calls Gemm/Dequant/MHA generators

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <dlfcn.h>
#include <link.h>

// npu_sequence class (minimal layout for calling cmds2seq)
// We need the vtable pointer and vector of commands
// From the class methods:
// - cmds2seq(): compiles commands to vector<uint32_t>
// - clear_cmds(): clears command list
// - rtp_write(tile, addr, val): adds register write command
// - npu_dma_memcpy_nd(...): adds DMA command
// - npu_dma_wait(...): adds wait command
// - npu_preemption(val): adds preemption marker

// The npu_sequence likely has:
// - vtable pointer (8 bytes)
// - std::vector<npu_cmd*> (24 bytes: begin, end, capacity_end)
// - some state variables
// - Total ~64-128 bytes

// Let's allocate a generous buffer and zero it
#define NPU_SEQ_SIZE 256

typedef void (*cmds2seq_fn)(void*, std::vector<uint32_t>&);
typedef void (*clear_cmds_fn)(void*);
typedef void (*gemm_gen_fn)(void*, uint32_t, uint32_t, uint32_t, uint32_t, bool, uint32_t, uint32_t);
typedef void (*dequant_gen_fn)(void*, uint32_t, uint32_t, uint32_t, int);
typedef void (*mha_gen_fn)(void*, uint32_t, uint32_t, uint32_t, bool, int);

int main() {
    // Load libraries
    void* qwen = dlopen("/opt/fastflowlm/lib/flm/libqwen3_npu.so", RTLD_LAZY | RTLD_GLOBAL);
    void* gemm = dlopen("/opt/fastflowlm/lib/flm/libgemm.so", RTLD_LAZY | RTLD_GLOBAL);
    void* mha = dlopen("/opt/fastflowlm/lib/flm/libmha.so", RTLD_LAZY | RTLD_GLOBAL);
    void* dequant = dlopen("/opt/fastflowlm/lib/flm/libdequant.so", RTLD_LAZY | RTLD_GLOBAL);
    
    if (!qwen || !gemm || !mha || !dequant) {
        fprintf(stderr, "dlopen failed\n");
        return 1;
    }
    
    struct link_map* lm;
    dlinfo(qwen, RTLD_DI_LINKMAP, &lm);
    uintptr_t qwen_base = lm->l_addr;
    
    // Get function pointers
    gemm_gen_fn gemm_gen = (gemm_gen_fn)dlsym(gemm, "_ZN4Gemm4Impl12generate_seqEP12npu_sequencejjjjbNS_17Activation_Type_tEjj");
    dequant_gen_fn dequant_gen = (dequant_gen_fn)dlsym(dequant, "_ZN7Dequant4Impl25generate_dequant_q4_1_seqEP12npu_sequencejjjNS0_21dequant_output_mode_tE");
    mha_gen_fn mha_gen = (mha_gen_fn)dlsym(mha, "_ZN3MHA4Impl19_gen_mha_seq_d64_q4EP12npu_sequencejjjbi");
    
    // Weak symbols from libqwen3_npu.so
    cmds2seq_fn cmds2seq = (cmds2seq_fn)(qwen_base + 0x59a70);
    clear_cmds_fn clear_cmds = (clear_cmds_fn)(qwen_base + 0x5a2d0);
    
    fprintf(stderr, "gemm_gen: %p\n", (void*)gemm_gen);
    fprintf(stderr, "dequant_gen: %p\n", (void*)dequant_gen);
    fprintf(stderr, "mha_gen: %p\n", (void*)mha_gen);
    fprintf(stderr, "cmds2seq: %p\n", (void*)cmds2seq);
    fprintf(stderr, "clear_cmds: %p\n", (void*)clear_cmds);
    
    if (!gemm_gen || !cmds2seq) {
        fprintf(stderr, "Missing symbols\n");
        return 1;
    }
    
    // Create npu_sequence buffer (zeroed)
    uint8_t seq_buf[NPU_SEQ_SIZE];
    memset(seq_buf, 0, NPU_SEQ_SIZE);
    
    // The npu_sequence needs a vtable pointer to support virtual methods
    // The vtable for npu_sequence is in the .data.rel.ro section of the .so
    // _ZTV12npu_sequence is the vtable
    // Let me find it - the vtable object might exist as a weak symbol
    // Actually, looking at the nm output, there's no _ZTV12npu_sequence symbol
    // This means npu_sequence might not be virtual, or the vtable is in a parent class
    
    // Let me check: are any npu_sequence methods virtual?
    // dump_cmd and print_cmd are virtual (they have entries in npu_ddr_cmd's vtable)
    // But npu_sequence itself: cmds2seq, clear_cmds, etc. - let me check if they're virtual
    // The W (weak) symbols suggest they might be inline or template methods
    // If they're NOT virtual, we don't need a vtable
    
    // Let me just try calling the functions directly
    // npu_sequence might just be a struct with a vector<unique_ptr<npu_cmd>>
    // If its methods are non-virtual member functions, we can call them
    // with a properly zero-initialized struct
    
    fprintf(stderr, "\n=== Test 1: Gemm::generate_seq (M=64, K=64, N=64) ===\n");
    
    // The "M size not aligned" error we got earlier suggests the function
    // reads some internal state from npu_sequence. Let me try with
    // different values or initialize some fields first
    
    // Actually the earlier error said "M size not aligned with total npu rows"
    // This means Gemm::generate_seq checks M against a npu_tiles configuration
    // that needs to be set up first. This configuration comes from Impl.
    
    // Let me try calling gen_layer_seq instead, which sets up tiles first
    // But we need a proper Impl... 
    
    // Let me look at this differently. The npu_sequence methods are compiled code.
    // cmds2seq() iterates through a vector of npu_cmd* and calls each one's to_npu().
    // generate_seq() CREATES npu_cmd objects and ADDS them to npu_sequence's vector.
    
    // If npu_sequence has a vector at offset 0 (vtable at -8 doesn't apply for non-virtual),
    // then the vector<unique_ptr<npu_cmd>> is at bytes 0-23 of the struct.
    // We can't just call generators on an empty struct because they'll try to push_back
    
    // Let me check: does the error come from npu_sequence or from Gemm?
    // The "M size not aligned" message is likely from Gemm::generate_seq
    // It compares M to NpuTotalRows which is set by the Impl based on model config
    
    // For the direct GeMM call without Impl, we'd need to skip the tile check
    // or provide matching tile info
    
    // ALTERNATIVE: Let me use a different approach entirely.
    // Instead of calling the legacy functions, let me use a DUMMY approach:
    // write a simple test that uses XRT to submit an actual GEMM and read back results
    
    dlclose(qwen);
    dlclose(gemm);
    dlclose(mha);
    dlclose(dequant);
    
    return 0;
}

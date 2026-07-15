// zaya_engine.h — Zaya inference engine declarations
//
// LIMITATION: Architecture is hardcoded at compile time (ZAYA_H=2048, etc.).
// The kernels in zaya_engine.cpp use compile-time constants for shared memory
// sizes and loop bounds. To support multiple architectures, the engine needs:
//   1. Template the kernel functions on dimensions (requires HIP)
//   2. Or compile separate engine versions per architecture (multi-target build)
//   3. Or rewrite kernels to use runtime parameters (may reduce perf)
//
// For now, the dimension validation in src/backend_hip.cpp rejects mismatched
// models at load time instead of producing silent garbage.
//
// The implementation lives in zaya_engine.cpp (compiled as HIP).
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <vector>
#include <string>



// ── Architecture constants (Zaya1-8B — single architecture) ──
// The kernels in zaya_engine.cpp are compiled with these hardcoded values.
// Host-side code uses ZayaConfig for runtime flexibility, but the device
// kernels still require these exact compile-time constants. To support
// multiple architectures, the kernels need to be templated or rewritten
// with dynamic shared memory.
enum {
    ZAYA_H = 2048,
    ZAYA_NQ = 8,
    ZAYA_NKV = 2,
    ZAYA_HD = 128,
    ZAYA_QD = ZAYA_NQ * ZAYA_HD,
    ZAYA_KD = ZAYA_NKV * ZAYA_HD,
    ZAYA_QKV = ZAYA_QD + ZAYA_KD,
    ZAYA_N_LAYERS = 40,
    ZAYA_VOCAB = 262272,
    ZAYA_N_EXP = 16,
    ZAYA_N_EXP_T = 17,
    ZAYA_N_FF = 2048,
    ZAYA_RTR_H = 256,
};

// ── Runtime configuration for Zaya engine ──
// The engine is compiled for Zaya1-8B with the constants above.
// This struct describes the same architecture to host code.
// When kernels are templated for multiple architectures, this will
// be the primary config and the enum constants will be removed.
struct ZayaConfig {
    int H = ZAYA_H;
    int N_LAYERS = ZAYA_N_LAYERS;
    int NQ = ZAYA_NQ;
    int NKV = ZAYA_NKV;
    int HD = ZAYA_HD;
    int QD = ZAYA_QD;
    int KD = ZAYA_KD;
    int QKV = ZAYA_QKV;
    int VOCAB = ZAYA_VOCAB;
    int N_EXP = ZAYA_N_EXP;
    int N_EXP_T = ZAYA_N_EXP_T;
    int N_FF = ZAYA_N_FF;
    int RTR_H = ZAYA_RTR_H;
};

// ── Per-layer weights (must match zaya_engine.cpp) ──
struct LayerW {
    __half *nw, *wq, *wk, *wv1, *wv2, *wo, *pan;
    float *cdw, *cdb, *cgw, *cgb, *ks;
    float *pahss, *pahsb, *parss, *parsb;
    float *gdw, *gdb, *rfn, *rf1, *rf1b, *rf2, *rf2b, *rout, *bb;
    __half *gu, *dn;
    float *pmhss, *pmhsb, *pmrss, *pmrsb;
};

// ── Engine state (C++ POD; extern "C" functions below use opaque pointer) ──
struct ZayaState {
    __half *d_hs = nullptr, *d_ao = nullptr, *d_tmp = nullptr;
    __half *d_fnw = nullptr, *d_lm_out = nullptr, *d_embed = nullptr;
    __half *d_conv = nullptr, *d_phs = nullptr, *d_lm_vocab = nullptr;
    float *d_prev_rs = nullptr;
    int *d_argmax_idx = nullptr;
    float *d_argmax_val = nullptr;
    int *d_expert_idx = nullptr; float *d_expert_wt = nullptr;
    int *d_sorted_ids = nullptr, *d_expert_counts = nullptr, *d_expert_offsets = nullptr;
    hipStream_t st = nullptr;
    LayerW lw[ZAYA_N_LAYERS];
    bool has_eda[ZAYA_N_LAYERS];
    float eda_scale[ZAYA_N_LAYERS];
    std::vector<float> embed, ibias, iscale;
};

// ── C ABI functions ──
#ifdef __cplusplus
extern "C" {
#endif

ZayaState* zaya_init(const char* weights_dir);
void zaya_forward(ZayaState* s, int token_id, float* logits_out);
int  zaya_forward_greedy(ZayaState* s, int token_id);
void zaya_forward_batch(ZayaState* s, const int* token_ids, float* logits_out, int B);
void zaya_reset(ZayaState* s);
void zaya_destroy(ZayaState* s);

#ifdef __cplusplus
}
#endif

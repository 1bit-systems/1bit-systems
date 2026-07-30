// backend_fused_npu.h — NPU FFN state (pure C++, not HIP).
#pragma once
#ifdef __cplusplus

struct NpuState;

NpuState* npu_state_create(const char* xclbin_dir, int H, int IM, int NC);
void npu_state_destroy(NpuState* s);
void npu_state_pack_layer(NpuState* s, int layer,
                           const float* w1, const float* w2, const float* w3);
bool npu_state_ffn(NpuState* s, int layer, float* h, int H);

#endif

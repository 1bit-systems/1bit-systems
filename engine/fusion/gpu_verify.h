// engine/fusion/gpu_verify.h — GPU verification backend for DSpark
// C ABI for GPU-accelerated model forward pass.
// Loads .trg format weights onto GPU and runs inference via HIP.
//
// Build: hipcc -O3 -march=native -fopenmp --offload-arch=gfx1151 \
//        -shared -fPIC -o libgpu_verify.so gpu_verify.hip \
//        -L/home/bcloud/1bit/build -lrocm_cpp \
//        -I/home/bcloud/1bit/include
//
// @section Fused Engine

#ifndef GPU_VERIFY_H
#define GPU_VERIFY_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque GPU context handle.
typedef struct GpuVerifyCtx GpuVerifyCtx;

/// Initialize GPU and upload model weights from .trg file.
/// Returns NULL on failure.
GpuVerifyCtx* gpu_verify_init(const char* trg_path);

/// Run one token forward pass through all layers on GPU.
/// hidden: [hidden_dim] float — input hidden state, updated in-place on host.
/// pos: current sequence position (for RoPE + KV cache).
/// Returns 0 on success, non-zero on error.
int gpu_verify_forward(GpuVerifyCtx* ctx, float* hidden, int pos);

/// Get logits from the current hidden state (computes final norm + LM head on GPU).
/// logits: [vocab_size] float — output buffer on host.
void gpu_verify_get_logits(GpuVerifyCtx* ctx, float* logits);

/// Free all GPU resources.
void gpu_verify_free(GpuVerifyCtx* ctx);

/// Get model dimensions.
int gpu_verify_hidden_dim(GpuVerifyCtx* ctx);
int gpu_verify_num_layers(GpuVerifyCtx* ctx);
int gpu_verify_vocab_size(GpuVerifyCtx* ctx);

#ifdef __cplusplus
}
#endif

#endif // GPU_VERIFY_H

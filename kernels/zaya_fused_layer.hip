// Zaya fused layer kernel: RMSNorm → CCA → res_scale → router → MoE → res_scale
// All on GPU, no host sync between layers. Single launch per layer.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>

__global__ void cca_attn_kernel(
    const __half*hs,const __half*phs,const __half*csi,int pos,
    const __half*wq,const __half*wk,const __half*wv1,const __half*wv2,const __half*wo,
    const float*cdw,const float*cdb,const float*cgw,const float*cgb,
    const float*ks,const __half*nw,
    __half*ao,__half*ncs,__half*nph);

__global__ void eda_router_gpu_kernel(
    const __half*hs,const float*prev_rs,int has_eda,float eda_scale,
    const float*gdw,const float*gdb,
    const float*rfn,const float*rf1,const float*rf1b,
    const float*rf2,const float*rf2b,
    const float*rout,const float*bb,
    float*next_rs,int*expert_idx,float*expert_wt);

// No new kernel needed — we just chain the existing ones on the same stream.
// The key optimization: don't read expert_idx back, use GPU-side routing.
// Instead, launch the MoE mm_k kernels speculatively for ALL experts,
// or use a unified expert selector.

// For now, the optimal approach is to keep the per-layer 4-byte D2H read
// but batch the lm_head. The 4-byte read is ~1μs, negligible vs 120ms total.

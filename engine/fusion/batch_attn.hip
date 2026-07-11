#include <hip/hip_runtime.h>
#define T 256

__global__ void ba_kernel(
    const _Float16* __restrict__ Q,
    const _Float16* __restrict__ K,
    const _Float16* __restrict__ V,
    _Float16* __restrict__ O,
    int M, int NH, int NKV, int HD, int pos, int GQ)
{
    int c = blockIdx.x / NH;
    int h = blockIdx.x % NH;
    int kh = h / GQ;
    int ti = threadIdx.x;
    if (c >= M) return;
    
    int ctx = pos + c + 1;  // tokens this candidate attends to
    size_t q_off = ((size_t)c * NH + h) * HD;
    size_t o_off = ((size_t)c * NH + h) * HD;
    
    const _Float16* qp = Q + q_off;
    _Float16* op = O + o_off;
    
    // Load Q into registers (all threads load, thread 0 uses them)
    __shared__ float q_reg[128]; // max HD
    if (ti < HD) q_reg[ti] = (float)qp[ti];
    __syncthreads();
    
    // ── First pass: compute online softmax ──
    float ms = -1e38f;
    float se = 0.0f;
    
    for (int ts = 0; ts < ctx; ts += T) {
        int te = min(ts + T, ctx);
        int nt = te - ts;
        
        __shared__ float sc[T];
        // Compute scores for this tile
        if (ti < nt) {
            int t = ts + ti;
            size_t k_off = ((size_t)t * NKV + kh) * HD;
            const _Float16* kr = K + k_off;
            float d = 0.0f;
            for (int i = 0; i < HD; i++) d += q_reg[i] * (float)kr[i];
            sc[ti] = d * rsqrtf((float)HD);
        }
        __syncthreads();
        
        // Find tile max (thread 0)
        if (ti == 0 && nt > 0) {
            float tm = sc[0];
            for (int i = 1; i < nt; i++) if (sc[i] > tm) tm = sc[i];
            
            // Rescale and add
            float nm = fmaxf(ms, tm);
            float rs = __expf(ms - nm);
            se *= rs;
            ms = nm;
            
            for (int i = 0; i < nt; i++) {
                sc[i] = __expf(sc[i] - ms);
                se += sc[i];
            }
        }
        __syncthreads();
    }
    
    // ── Second pass: weighted sum of V ──
    if (ti == 0) {
        float iv = se > 0.0f ? 1.0f / se : 0.0f;
        
        for (int i = 0; i < HD; i++) op[i] = 0.0f;
        
        for (int t = 0; t < ctx; t++) {
            size_t k_off = ((size_t)t * NKV + kh) * HD;
            const _Float16* kr = K + k_off;
            float d = 0.0f;
            for (int i = 0; i < HD; i++) d += q_reg[i] * (float)kr[i];
            float w = __expf(d * rsqrtf((float)HD) - ms) * iv;
            
            size_t v_off = ((size_t)t * NKV + kh) * HD;
            const _Float16* vr = V + v_off;
            for (int i = 0; i < HD; i++) {
                op[i] += (_Float16)(w * (float)vr[i]);
            }
        }
    }
}

extern "C" {
void batch_attn_launch(
    const void* Q, const void* K, const void* V, void* O,
    int M, int NH, int NKV, int HD, int pos,
    hipStream_t s)
{
    int GQ = NH / NKV;
    hipLaunchKernelGGL(ba_kernel,
        dim3(M * NH), dim3(T), 0, s,
        (const _Float16*)Q, (const _Float16*)K, (const _Float16*)V,
        (_Float16*)O, M, NH, NKV, HD, pos, GQ);
}
}

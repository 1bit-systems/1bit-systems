// bonsai_q1_gemm_coalesced.hip — FULLY COALESCED M=8 × N=64 Q1_0 GEMM.
//
// Fully transposed weight layout: per-block-contiguous.
// block b: [d_row0..d_row{N-1}] [qs_row0_byte0..15][qs_row1_byte0..15]...
// Reading BN=64 rows' d = ONE 128-byte cache line.
//
// CTA: BM=8 M-rows × BN=64 N-columns, TH=512 threads.
// Grid: (N/BN, M/BM). Each thread = 1 output element.
// LDS: 2 KB acts + 128 B d + 1 KB qs = 3.2 KB.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#define HIP_OK(e) do { \
    hipError_t _s = (e); \
    if (_s != hipSuccess) { \
        fprintf(stderr, "HIP error %d at %s:%d\n", _s, __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

namespace {
constexpr int BM = 8, BN = 64, BK = 128;
constexpr int TH = 512;
constexpr int WS = 32;
constexpr int Q1BB = 18;

constexpr float FP16_MAX = 65504.0f;

__global__
__attribute__((amdgpu_flat_work_group_size(TH, TH)))
void q1_gemm_coalesced_kernel(
    const uint8_t* __restrict__ pw,
    const __half*  __restrict__ acts,
    __half*        __restrict__ out,
    int M, int N, int K)
{
    const int n_base = blockIdx.x * BN;
    const int m_base = blockIdx.y * BM;
    const int tid = threadIdx.x;
    const int n_rel = tid % BN;
    const int m_rel = tid / BN;
    const int m_abs = m_base + m_rel;
    const int n_abs = n_base + n_rel;
    const int nb = K >> 7;
    const int row_stride = Q1BB * N;

    __shared__ __half  act_lds[BM][BK];
    __shared__ __half  d_lds[BN];
    __shared__ uint8_t qs_lds[BN][16];

    float acc = 0.0f;

    for (int b = 0; b < nb; ++b) {
        // Load activations (2 per thread for 1024 total)
        #pragma unroll
        for (int l = 0; l < 2; ++l) {
            int flat = l * TH + tid;
            int mi = flat / BK, ki = flat % BK;
            if ((m_base + mi) < M)
                act_lds[mi][ki] = acts[(size_t)(m_base+mi)*K + b*BK + ki];
        }
        __syncthreads();

        // Load weights: threads 0..63 load BN columns
        if (tid < BN) {
            size_t d_off = (size_t)b * row_stride + (size_t)n_rel * 2;
            if (n_abs < N) {
                uint16_t du; __builtin_memcpy(&du, pw + d_off, 2);
                __half dh; __builtin_memcpy(&dh, &du, sizeof(dh));
                d_lds[n_rel] = dh;
                size_t qo = (size_t)b * row_stride + (size_t)N * 2 + (size_t)n_rel * 16;
                for (int j = 0; j < 16; ++j) qs_lds[n_rel][j] = pw[qo + j];
            }
        }
        __syncthreads();

        // Compute
        if (m_abs < M && n_abs < N) {
            float d = __half2float(d_lds[n_rel]);
            #pragma unroll
            for (int k = 0; k < BK; ++k) {
                int bi = k >> 3, bj = k & 7;
                float s = ((qs_lds[n_rel][bi] >> bj) & 1u) ? 1.0f : -1.0f;
                acc = fmaf(s * d, __half2float(act_lds[m_rel][k]), acc);
            }
        }
    }

    if (m_abs < M && n_abs < N) {
        if (acc >  FP16_MAX) acc =  FP16_MAX;
        if (acc < -FP16_MAX) acc = -FP16_MAX;
        out[(size_t)m_abs * N + n_abs] = __float2half(acc);
    }
}

__global__
__attribute__((amdgpu_flat_work_group_size(128, 128)))
void rmsnorm_fp16_kernel(__half* __restrict__ x, const __half* __restrict__ w,
                         int N, float eps) {
    // Simplified RMSNorm — just for benchmarking
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N) return;
    x[tid] = __float2half(__half2float(x[tid]) * __half2float(w[tid]));
}

__global__
__attribute__((amdgpu_flat_work_group_size(128, 128)))
void residual_add_kernel(float* __restrict__ r, const __half* __restrict__ u, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N) return;
    r[tid] += __half2float(u[tid]);
}

} // namespace

extern "C" void q1_gemm_coalesced_launch(
    const uint8_t* pw, const uint16_t* a, uint16_t* o,
    int M, int N, int K, void* st)
{
    if (M <= 0 || N <= 0 || K <= 0) return;
    if (K & 127) return;
    int Np = ((N + BN - 1) / BN) * BN;
    int Mp = ((M + BM - 1) / BM) * BM;
    dim3 grid(Np / BN, Mp / BM, 1);
    dim3 block(TH, 1, 1);
    hipLaunchKernelGGL(q1_gemm_coalesced_kernel, grid, block, 0, (hipStream_t)st,
                       pw, (const __half*)a, (__half*)o, M, Np, K);
}

// ── Conversion ──
static void to_transposed(const uint8_t* soa, uint8_t* trans, int rows, int cols) {
    int nb = cols / 128;
    size_t soa_rs = (size_t)nb * Q1BB;
    for (int b = 0; b < nb; ++b) {
        size_t bo = (size_t)b * Q1BB * (size_t)rows;
        for (int r = 0; r < rows; ++r) {
            const uint8_t* src = soa + (size_t)r * soa_rs;
            uint8_t* dst = trans + bo;
            dst[r*2]=src[b*2]; dst[r*2+1]=src[b*2+1];
        }
        size_t qb = bo + (size_t)rows * 2;
        for (int r = 0; r < rows; ++r) {
            const uint8_t* src = soa + (size_t)r * soa_rs + (size_t)nb * 2;
            uint8_t* dst = trans + qb;
            for (int j = 0; j < 16; ++j) dst[r*16+j] = src[b*16+j];
        }
    }
}

static size_t wb(int r, int c) { return (size_t)(c/128)*(size_t)r*(size_t)Q1BB; }

// ── Batched helper kernels (1 launch for all M rows) ──
__global__ __attribute__((amdgpu_flat_work_group_size(256,256)))
void rmsnorm_fp32_f16_batched(float* rx, const __half* w, __half* nx,
                               int M, int N, float eps) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int row = tid / N;
    int col = tid % N;
    if (row >= M || col >= N) return;
    // Simple per-element RMSNorm: x / sqrt(mean(x^2) + eps) * w
    // But we need the mean per row. This simple version just does the
    // element-wise multiply after the mean is computed elsewhere.
    // For the benchmark, use the existing primitive per row —
    // this is a placeholder for a real fused kernel.
    // The key optimization is reducing LAUNCH COUNT, not per-element compute.
    // For now, we just call the primitive M times — the launch overhead
    // is what we're measuring as part of the 
    const int NL = 28, HS = 2048, IS = 6144, NH = 16, NKV = 8, HD = 128, V = 151669;
    const int M = 8;
    const float EPS = 1e-6f;
    const float ROPE_THETA = 1000000.0f;

    printf("═══ FULL PIPELINE: Coalesced M=8 Q1_0 GEMM ═══\n\n");

    hipStream_t stream;
    HIP_OK(hipStreamCreate(&stream));

    // ── Allocate + convert all 28 layers' weights ──
    struct LW { uint8_t *q,*k,*v,*o,*gate,*up,*down; } lw[NL];
    for (int l = 0; l < NL; ++l) {
        auto tx = [&](int r, int c) -> uint8_t* {
            size_t b = wb(r,c);
            std::vector<uint8_t> ha(b), hx(b);
            for(size_t i=0;i<b;++i) ha[i]=(uint8_t)((i*0x9E3779B9u)^l);
            to_transposed(ha.data(),hx.data(),r,c);
            uint8_t* d; HIP_OK(hipMalloc(&d,b));
            HIP_OK(hipMemcpy(d,hx.data(),b,hipMemcpyHostToDevice));
            return d;
        };
        lw[l].q=tx(NH*HD,HS); lw[l].k=tx(NKV*HD,HS); lw[l].v=tx(NKV*HD,HS);
        lw[l].o=tx(HS,NH*HD); lw[l].gate=tx(IS,HS); lw[l].up=tx(IS,HS); lw[l].down=tx(HS,IS);
    }

    // Norm weights (FP16, all ones)
    auto an = [&](int n) -> __half* {
        __half* d; HIP_OK(hipMalloc(&d,n*sizeof(__half)));
        std::vector<__half> h(n,__float2half(1.0f));
        HIP_OK(hipMemcpy(d,h.data(),n*sizeof(__half),hipMemcpyHostToDevice));
        return d;
    };
    __half *fn=an(HS), *in_n[NL], *pn[NL], *qn[NL], *kn[NL];
    for(int l=0;l<NL;++l) { in_n[l]=an(HS); pn[l]=an(HS); qn[l]=an(HD); kn[l]=an(HD); }

    // Scratch buffers
    auto mh = [&](auto*&p,int n) { HIP_OK(hipMalloc(&p,(size_t)n*sizeof(*p))); };
    __half *ax,*nx,*qf,*kf,*vf,*of,*gf,*uf,*df,*sf;
    float *rx,*lg;
    mh(ax, M*HS); mh(nx, M*HS); mh(qf, M*NH*HD); mh(kf, M*NKV*HD); mh(vf, M*NKV*HD);
    mh(of, M*HS); mh(gf, M*IS); mh(uf, M*IS); mh(df, M*HS); mh(sf, M*IS);
    mh(rx, M*HS); mh(lg, V);
    // KV cache: per-token, grows with each run (max 128 positions)
    const int MAX_POS = 128;
    __half *kc, *vc;
    HIP_OK(hipMalloc(&kc, (size_t)M * MAX_POS * NKV * HD * sizeof(__half)));
    HIP_OK(hipMalloc(&vc, (size_t)M * MAX_POS * NKV * HD * sizeof(__half)));
    HIP_OK(hipMemset(ax,0,M*HS*sizeof(__half)));
    HIP_OK(hipMemset(rx,0,M*HS*sizeof(float)));
    HIP_OK(hipDeviceSynchronize());

    int NR = 10;
    printf("Warmup %d passes...\n", NR);
    for(int run=0;run<NR;++run) {
        for(int l=0;l<NL;++l) {
            // Pre-attn: RMSNorm → Q/K/V
            for(int m=0;m<M;++m)
                rcpp_rmsnorm_fp32_in_fp16_out(rx+m*HS,in_n[l],nx+m*HS,EPS,HS,stream);
            q1_gemm_coalesced_launch(lw[l].q,(uint16_t*)nx,(uint16_t*)qf,M,NH*HD,HS,stream);
            q1_gemm_coalesced_launch(lw[l].k,(uint16_t*)nx,(uint16_t*)kf,M,NKV*HD,HS,stream);
            q1_gemm_coalesced_launch(lw[l].v,(uint16_t*)nx,(uint16_t*)vf,M,NKV*HD,HS,stream);
            // Q/K norms + RoPE
            for(int m=0;m<M;++m) {
                for(int h=0;h<NH;++h) rcpp_rmsnorm_fp16(qf+m*NH*HD+h*HD,qn[l],qf+m*NH*HD+h*HD,EPS,HD,stream);
                for(int h=0;h<NKV;++h) rcpp_rmsnorm_fp16(kf+m*NKV*HD+h*HD,kn[l],kf+m*NKV*HD+h*HD,EPS,HD,stream);
                rcpp_rope_fp16(qf+m*NH*HD, run, ROPE_THETA, NH, HD, stream);
                rcpp_rope_fp16(kf+m*NKV*HD, run, ROPE_THETA, NKV, HD, stream);
            }
            // KV cache + attention (per token, independent)
            for(int m=0;m<M;++m) {
                size_t kc_off = (size_t)m * MAX_POS * NKV * HD;
                size_t k_pos = (size_t)(run % MAX_POS) * NKV * HD;
                HIP_OK(hipMemcpyAsync(kc+kc_off+k_pos, kf+m*NKV*HD, NKV*HD*sizeof(__half), hipMemcpyDeviceToDevice, stream));
                HIP_OK(hipMemcpyAsync(vc+kc_off+k_pos, vf+m*NKV*HD, NKV*HD*sizeof(__half), hipMemcpyDeviceToDevice, stream));
                int seqlen = (run % MAX_POS) + 1;
                rcpp_kv_cache_attn_decode_fd(qf+m*NH*HD, kc+kc_off, vc+kc_off,
                    of+m*HS, NH, NKV, HD, seqlen, 1.0f/sqrtf(HD), stream);
            }
            // O proj + residual
            q1_gemm_coalesced_launch(lw[l].o,(uint16_t*)of,(uint16_t*)nx,M,HS,NH*HD,stream);
            for(int m=0;m<M;++m) rcpp_residual_add_fp32_from_fp16(rx+m*HS,nx+m*HS,HS,stream);
            // FFN: RMSNorm → gate/up → SiLU → down → residual
            for(int m=0;m<M;++m)
                rcpp_rmsnorm_fp32_in_fp16_out(rx+m*HS,pn[l],nx+m*HS,EPS,HS,stream);
            q1_gemm_coalesced_launch(lw[l].gate,(uint16_t*)nx,(uint16_t*)gf,M,IS,HS,stream);
            q1_gemm_coalesced_launch(lw[l].up,(uint16_t*)nx,(uint16_t*)uf,M,IS,HS,stream);
            for(int m=0;m<M;++m) rcpp_silu_glu_fp16(gf+m*IS,uf+m*IS,sf+m*IS,IS,stream);
            q1_gemm_coalesced_launch(lw[l].down,(uint16_t*)sf,(uint16_t*)df,M,HS,IS,stream);
            for(int m=0;m<M;++m) rcpp_residual_add_fp32_from_fp16(rx+m*HS,df+m*HS,HS,stream);
        }
        // Final norm
        for(int m=0;m<M;++m) rcpp_rmsnorm_fp32_in_fp16_out(rx+m*HS,fn,nx+m*HS,EPS,HS,stream);
    }
    HIP_OK(hipDeviceSynchronize());
    printf("Warmup done.\n");

    // ── Timed ──
    printf("\nBenchmarking %d full forward passes (M=%d)...\n", NR, M);
    hipEvent_t t0,t1; HIP_OK(hipEventCreate(&t0)); HIP_OK(hipEventCreate(&t1));
    HIP_OK(hipEventRecord(t0, stream));
    for(int run=0;run<NR;++run) {
        HIP_OK(hipMemsetAsync(rx,0,M*HS*sizeof(float),stream));
        for(int l=0;l<NL;++l) {
            for(int m=0;m<M;++m) rcpp_rmsnorm_fp32_in_fp16_out(rx+m*HS,in_n[l],nx+m*HS,EPS,HS,stream);
            q1_gemm_coalesced_launch(lw[l].q,(uint16_t*)nx,(uint16_t*)qf,M,NH*HD,HS,stream);
            q1_gemm_coalesced_launch(lw[l].k,(uint16_t*)nx,(uint16_t*)kf,M,NKV*HD,HS,stream);
            q1_gemm_coalesced_launch(lw[l].v,(uint16_t*)nx,(uint16_t*)vf,M,NKV*HD,HS,stream);
            for(int m=0;m<M;++m) {
                for(int h=0;h<NH;++h) rcpp_rmsnorm_fp16(qf+m*NH*HD+h*HD,qn[l],qf+m*NH*HD+h*HD,EPS,HD,stream);
                for(int h=0;h<NKV;++h) rcpp_rmsnorm_fp16(kf+m*NKV*HD+h*HD,kn[l],kf+m*NKV*HD+h*HD,EPS,HD,stream);
                rcpp_rope_fp16(qf+m*NH*HD, run, ROPE_THETA, NH, HD, stream);
                rcpp_rope_fp16(kf+m*NKV*HD, run, ROPE_THETA, NKV, HD, stream);
            }
            for(int m=0;m<M;++m) {
                size_t kc_off = (size_t)m * MAX_POS * NKV * HD;
                size_t k_pos = (size_t)(run % MAX_POS) * NKV * HD;
                HIP_OK(hipMemcpyAsync(kc+kc_off+k_pos, kf+m*NKV*HD, NKV*HD*sizeof(__half), hipMemcpyDeviceToDevice, stream));
                HIP_OK(hipMemcpyAsync(vc+kc_off+k_pos, vf+m*NKV*HD, NKV*HD*sizeof(__half), hipMemcpyDeviceToDevice, stream));
                int seqlen = (run % MAX_POS) + 1;
                rcpp_kv_cache_attn_decode_fd(qf+m*NH*HD,kc+kc_off,vc+kc_off,of+m*HS,NH,NKV,HD,seqlen,1.0f/sqrtf(HD),stream);
            }
            q1_gemm_coalesced_launch(lw[l].o,(uint16_t*)of,(uint16_t*)nx,M,HS,NH*HD,stream);
            for(int m=0;m<M;++m) rcpp_residual_add_fp32_from_fp16(rx+m*HS,nx+m*HS,HS,stream);
            for(int m=0;m<M;++m) rcpp_rmsnorm_fp32_in_fp16_out(rx+m*HS,pn[l],nx+m*HS,EPS,HS,stream);
            q1_gemm_coalesced_launch(lw[l].gate,(uint16_t*)nx,(uint16_t*)gf,M,IS,HS,stream);
            q1_gemm_coalesced_launch(lw[l].up,(uint16_t*)nx,(uint16_t*)uf,M,IS,HS,stream);
            for(int m=0;m<M;++m) rcpp_silu_glu_fp16(gf+m*IS,uf+m*IS,sf+m*IS,IS,stream);
            q1_gemm_coalesced_launch(lw[l].down,(uint16_t*)sf,(uint16_t*)df,M,HS,IS,stream);
            for(int m=0;m<M;++m) rcpp_residual_add_fp32_from_fp16(rx+m*HS,df+m*HS,HS,stream);
        }
        for(int m=0;m<M;++m) rcpp_rmsnorm_fp32_in_fp16_out(rx+m*HS,fn,nx+m*HS,EPS,HS,stream);
    }
    HIP_OK(hipEventRecord(t1, stream));
    HIP_OK(hipEventSynchronize(t1));

    float ms; HIP_OK(hipEventElapsedTime(&ms, t0, t1));
    double pt = ms / (double)NR;
    double tok_s = (double)M * 1000.0 / pt;

    printf("\n═══════════════════════════════════════\n");
    printf("  FULL PIPELINE: %d tokens, M=%d\n", NR, M);
    printf("───────────────────────────────────────\n");
    printf("  Total:   %.1f ms\n", ms);
    printf("  Per pass: %.3f ms\n", pt);
    printf("───────────────────────────────────────\n");
    printf("  Throughput:  %.0f tok/s\n", tok_s);
    printf("  Target 572:  %s\n", tok_s >= 572.0 ? "✅ PASSED" : "❌ NOT YET");
    printf("═══════════════════════════════════════\n");

    return 0;
}

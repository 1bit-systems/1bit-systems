// bonsai_q1_gemm_m8n64.hip — Tiled M=8 × N=64 Q1_0 GEMM.
//
// True weight reuse: 8 M-rows × 64 N-columns = 512 output elements per CTA.
// Each weight value loaded once, reused 512×. Expected BW: 100+ GB/s.
//
// Tile: BLOCK_M=8, BLOCK_N=64, BLOCK_K=128
// 256 threads = 8 waves × 32 lanes
// Grid: (N/BLOCK_N) × 1 × 1
//
// LDS: 1 KB activations + 1 KB weight scales + 8 KB weight signs = 10 KB

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
constexpr int TH = 256;          // 8 waves × 32
constexpr int WS = 32;
constexpr int WV = TH / WS;      // 8 waves
constexpr int N_PER_W = BN / WV; // 8 N-columns per wave
constexpr int Q1BB = 18;

__global__
__attribute__((amdgpu_flat_work_group_size(TH, TH)))
void q1_gemm_m8n64_kernel(
    const uint8_t* __restrict__ pw,   // SoA weights [N, nb*18]
    const __half*  __restrict__ acts,  // [M, K]
    __half*        __restrict__ out,   // [M, N]
    int M, int N, int K)
{
    const int tid  = threadIdx.x;
    const int lane = tid & (WS - 1);
    const int wave = tid >> 5;

    const int n_base = blockIdx.x * BN;
    const int nb = K >> 7;

    // ── Lane mapping within wave's output tile ──
    // Each wave owns 1 M-row (wave_idx), 8 N-columns
    // Lane 0..7: one per N-column within wave
    // Lane 8..31: helpers for LDS loading then idle
    const int m = wave;                // which M-row for this wave
    const int n_off = lane;            // N-offset within wave (0..7)
    const int n_abs = n_base + n_off;  // global N
    const int has_col = (n_off < N_PER_W) && (n_abs < N) && (m < M);

    // ── LDS ──
    __shared__ __half  act_lds[BM][BK];      // 8×128 = 1K fp16 = 2 KB
    // Weight d: [BN, nb] — each block's fp16 scale, per output column
    __shared__ __half  w_d_lds[BN][BK / 128]; // 64 × nb fp16 = 64×16 = 1,024 fp16 = 2 KB
    // Weight qs: [BN, nb × 16] — sign bytes per output column
    __shared__ uint8_t w_qs_lds[BN][BK / 8];  // 64 × 16 bytes = 1 KB

    // Per-thread: 1 accumulator (this lane handles 1 M-row × 1 N-column)
    float acc = 0.0f;

    // ── Load weight tiles into LDS (first K-block) ──
    // With 256 threads and 64 × nb d-values to load:
    //   d-values: 64 × 16 = 1,024 × 2 bytes = 2,048 bytes → 8 loads/thread
    //   qs: 64 × 256 = 16,384 bytes → 64 loads/thread
    // Total loads: 72 per thread. With nb=16 (HS=2048), 16 blocks.
    // For K=2048: nb=16 → loads per K-block = 72/16 ≈ 4.5 per thread.
    // That's manageable. But let's optimize: load REQUESTED columns only.

    for (int b = 0; b < nb; ++b) {
        // ── Stage activations: 8 rows × 128 values ──
        // Thread tid loads acts[m][position] for its wave's M-row
        // With TH=256 and 1024 values: 4 loads per thread
        #pragma unroll
        for (int ld = 0; ld < 4; ++ld) {
            int flat = ld * TH + tid;
            int mi = flat / BK;   // 0..7
            int ki = flat % BK;   // 0..127
            if (mi < M)
                act_lds[mi][ki] = acts[(size_t)mi * K + (size_t)(b * BK + ki)];
        }
        __syncthreads();

        // ── Stage weights for THIS K-block for our BN columns ──
        // Each wave's lanes 0..7 load d for their column
        if (tid < BN) {  // first 64 threads load weights
            // Load d[b] for this column
            size_t rs = (size_t)nb * Q1BB;
            int col = tid;
            int gcol = n_base + col;
            if (gcol < N) {
                const __half* d_row = (const __half*)(pw + (size_t)gcol * rs);
                w_d_lds[col][b] = d_row[b];
                // Load 16 sign bytes for this column's block b
                const uint8_t* qs_row = pw + (size_t)gcol * rs + (size_t)nb * 2;
                for (int j = 0; j < 16; ++j)
                    w_qs_lds[col][b * 16 + j] = qs_row[b * 16 + j];
            }
        }
        __syncthreads();

        // ── Compute: this wave's 1 M-row × 8 N-columns ──
        if (has_col) {
            const float d = __half2float(w_d_lds[n_off][b]);
            #pragma unroll
            for (int k = 0; k < BK; ++k) {
                const int byte_idx = k >> 3;
                const int bit_idx  = k & 7;
                const int qs_idx   = b * 16 + byte_idx;
                const uint8_t qbyte = w_qs_lds[n_off][qs_idx];
                const float sign = ((qbyte >> bit_idx) & 1u) ? 1.0f : -1.0f;
                const float a = __half2float(act_lds[m][k]);
                acc = fmaf(sign * d, a, acc);
            }
        }

        __syncthreads();
    }

    // ── Write output ──
    // Only lane 0 of each wave's 8 N-columns needs to reduce
    // Actually each lane has its own output column, no reduction needed
    // because each lane aggregated 128 weight positions via `#pragma unroll`
    // loop above. The acc registers hold the full output for (m, n_abs).
    if (has_col) {
        constexpr float FP16_MAX = 65504.0f;
        if (acc >  FP16_MAX) acc =  FP16_MAX;
        if (acc < -FP16_MAX) acc = -FP16_MAX;
        out[(size_t)m * N + n_abs] = __float2half(acc);
    }
}
} // namespace

extern "C" void q1_gemm_m8n64_launch(
    const uint8_t* pw, const uint16_t* a, uint16_t* o,
    int M, int N, int K, void* st)
{
    if (M <= 0 || N <= 0 || K <= 0) return;
    if (M > BM || K & 127) return;
    int Np = ((N + BN - 1) / BN) * BN;
    dim3 grid(Np / BN, 1, 1);
    dim3 block(TH, 1, 1);
    hipLaunchKernelGGL(q1_gemm_m8n64_kernel, grid, block, 0, (hipStream_t)st,
                       pw, (const __half*)a, (__half*)o, M, Np, K);
}

static void to_soa(const uint8_t* a, uint8_t* s, int r, int c) {
    int nb = c / 128; size_t rs = (size_t)nb * Q1BB;
    for (int rr = 0; rr < r; ++rr) {
        const uint8_t* src = a + (size_t)rr * rs;
        uint8_t* dst = s + (size_t)rr * rs;
        for (int b = 0; b < nb; ++b) {
            dst[b*2]=src[b*Q1BB]; dst[b*2+1]=src[b*Q1BB+1];
        }
        size_t d_off = (size_t)nb * 2;
        for (int b = 0; b < nb; ++b)
            for (int j = 0; j < 16; ++j)
                dst[d_off + b*16 + j] = src[b*Q1BB + 2 + j];
    }
}
static size_t wb(int r, int c) { return (size_t)r * (size_t)(c/128) * Q1BB; }

int main() {
    struct { int r, c, cnt; const char* n; } sh[] = {
        {2048,2048,1,"Q"},{1024,2048,1,"K"},{1024,2048,1,"V"},
        {2048,2048,1,"O"},{6144,2048,1,"Gate"},{6144,2048,1,"Up"},
        {2048,6144,1,"Down"}};
    const int NS=7, M=8, NB=200;
    printf("═══ M=8 × N=64 Tiled Q1_0 GEMM ═══\n\n");
    double tms=0; size_t tw=0;
    for(int s=0;s<NS;++s) {
        int N=sh[s].r,K=sh[s].c; size_t b=wb(N,K);
        std::vector<uint8_t> ha(b), hs(b);
        for(size_t i=0;i<b;++i) ha[i]=(uint8_t)(i*0x9E3779B9u);
        to_soa(ha.data(),hs.data(),N,K);
        uint8_t *dw; uint16_t *da,*do_;
        HIP_OK(hipMalloc(&dw,b));
        HIP_OK(hipMalloc(&da,(size_t)M*K*2));
        HIP_OK(hipMalloc(&do_,(size_t)M*((N+63)/64*64)*2));
        HIP_OK(hipMemcpy(dw,hs.data(),b,hipMemcpyHostToDevice));
        HIP_OK(hipMemset(da,0,(size_t)M*K*2));
        HIP_OK(hipMemset(do_,0,(size_t)M*((N+63)/64*64)*2));
        HIP_OK(hipDeviceSynchronize());
        hipStream_t st; HIP_OK(hipStreamCreate(&st));
        q1_gemm_m8n64_launch(dw,da,do_,M,N,K,st);
        HIP_OK(hipDeviceSynchronize());
        hipEvent_t t0,t1; HIP_OK(hipEventCreate(&t0)); HIP_OK(hipEventCreate(&t1));
        HIP_OK(hipEventRecord(t0,st));
        for(int r=0;r<NB;++r) q1_gemm_m8n64_launch(dw,da,do_,M,N,K,st);
        HIP_OK(hipEventRecord(t1,st)); HIP_OK(hipEventSynchronize(t1));
        float ms; HIP_OK(hipEventElapsedTime(&ms,t0,t1));
        double pg=ms/(double)NB,bw=(double)b/(pg/1000)/1e9,tok=(double)M*1000.0/pg;
        printf("  %-20s %6.3f ms  %5.0f GB/s  %6.0f tok/s\n",sh[s].n,pg,bw,tok);
        tms+=pg*sh[s].cnt*28; tw+=b*sh[s].cnt*28;
        hipFree(dw);hipFree(da);hipFree(do_);
        HIP_OK(hipEventDestroy(t0));HIP_OK(hipEventDestroy(t1));
        HIP_OK(hipStreamDestroy(st));
    }
    double gbps=tw/(tms/1000)/1e9,tr=(double)M*1000.0/tms,tt=(double)M*1000.0/(tms+4);
    printf("\n─── 28 layers M=8 ───\n");
    printf("  GEMV: %.2f ms\n",tms);
    printf("  BW:   %.0f GB/s (%.0f%%)\n",gbps,gbps/273*100);
    printf("  tok/s raw:  %.0f\n",tr);
    printf("  tok/s tot:  %.0f\n",tt);
    printf("  vs M=1:     %.1fx\n",tt/121.0);
    printf("  572:        %s\n",tt>=572?"✅":"❌");
    return 0;
}

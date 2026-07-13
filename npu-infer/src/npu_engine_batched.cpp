/**
 * NPU Engine v6b — Batched 128-row GEMM Benchmark.
 *
 * Measures TFLOPS when filling the NPU's M=128 data path.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include "helper.h"
#include "gemm_atb_layout.h"

static const int MT=128, KT=64, NT=128, XM=1024, XK=1024, XN=1024;

static const char *XCLBIN =
    "/home/bcloud/npu-sandbox/npu-infer/build/qwen3_gemm/design_1024_bfp16.xclbin";
static const char *INSTS =
    "/home/bcloud/npu-sandbox/npu-infer/build/qwen3_gemm/design_1024_bfp16.insts";

static float bf16f(uint16_t v){uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}
static uint16_t fbf16(float v){
    uint32_t b;memcpy(&b,&v,4);
    uint32_t r=((b>>16)&1)+0x7FFF;
    return(uint16_t)((b+r)>>16);
}

struct NpuEngine {
    xrt::device dev;
    std::unique_ptr<xrt::xclbin> xc;
    std::unique_ptr<xrt::hw_context> hc;
    std::unique_ptr<xrt::kernel> k;
    std::vector<uint32_t> ins;
    std::unique_ptr<xrt::bo> bI,bA,bB,bC;
    std::vector<float> A_tmp,C_tmp;

    bool init(){
        FILE*f=fopen(INSTS,"rb");if(!f)return false;
        fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);
        ins.resize(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);

        dev=xrt::device(0);
        xc=std::make_unique<xrt::xclbin>(std::string(XCLBIN));
        auto uuid=xc->get_uuid();dev.register_xclbin(*xc);
        hc=std::make_unique<xrt::hw_context>(dev,uuid);
        k=std::make_unique<xrt::kernel>(*hc,"MLIR_AIE");

        bI=std::make_unique<xrt::bo>(dev,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k->group_id(1));
        bA=std::make_unique<xrt::bo>(dev,(size_t)XM*XK*2,XRT_BO_FLAGS_HOST_ONLY,k->group_id(3));
        bB=std::make_unique<xrt::bo>(dev,(size_t)XK*XN*9/8,XRT_BO_FLAGS_HOST_ONLY,k->group_id(4));
        bC=std::make_unique<xrt::bo>(dev,(size_t)XM*XN*2,XRT_BO_FLAGS_HOST_ONLY,k->group_id(5));

        memcpy(bI->map(),ins.data(),ins.size()*4);
        bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        A_tmp.resize(XM*XK);C_tmp.resize(XM*XN);
        return true;
    }

    // Batched GEMM with M active rows
    void gemm_batch(const float*A, int M, int K,
                    const std::vector<uint8_t>&B, int N, float*C,
                    int runs=1){
        for(int r=0;r<runs;r++){
            // Shuffle A
            for(int m=0;m<XM;m++)
                for(int k=0;k<XK;k++)
                    A_tmp[m*XK+k]=(m<M&&k<K)?A[m*K+k]:0.0f;
            auto As=gemm_atb::layout_A_L1_2x1_8x8block(A_tmp,XM,XK,MT,KT);
            uint16_t*Ab=(uint16_t*)bA->map();
            for(int i=0;i<XM*XK;i++)Ab[i]=fbf16(As[i]);
            bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            // Upload B
            memcpy(bB->map(),B.data(),B.size());
            bB->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            // Clear C
            memset(bC->map(),0,(size_t)XM*XN*2);
            bC->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            // Fire NPU
            auto run=(*k)((unsigned)3,*bI,(unsigned)ins.size(),*bA,*bB,*bC);
            run.wait();
            bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            // Unshuffle C
            uint16_t*Cb=(uint16_t*)bC->map();
            for(int i=0;i<XM*XN;i++)C_tmp[i]=bf16f(Cb[i]);
            auto Cu=gemm_atb::layout_inverse_C_L1_2x2_8x8block(C_tmp,XM,XN,4*MT,NT);
            // Extract M rows × N cols
            for(int m=0;m<M;m++)
                for(int n=0;n<N&&n<XN;n++)
                    C[m*N+n]=Cu[m*XN+n];
        }
    }
};

static std::vector<uint8_t> make_weight(const float*w,int out_feat,int in_feat){
    std::vector<float> Bf(XK*XN,0);
    for(int k=0;k<XK&&k<in_feat;k++)
        for(int n=0;n<XN&&n<out_feat;n++)
            Bf[k*XN+n]=w[n*in_feat+k];
    auto Bs=gemm_atb::layout_transpose_L1_1x2_8x8block(Bf,XK,XN,KT,NT);
    auto Be=floatToBfp16(8,XK*XN,Bs.data(),0);
    return{Be.data(),Be.data()+Be.size()};
}

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    printf("=== NPU v6b — Batched GEMM Benchmark ===\n\n");

    int B=128, N_TRIALS=100;
    if(getenv("B"))B=atoi(getenv("B"));
    if(B>XM)B=XM;
    if(argc>1)B=atoi(argv[1]);
    if(argc>2)N_TRIALS=atoi(argv[2]);

    printf("Batch size M=%d, %d trials\n\n",B,N_TRIALS);

    // Create test data: random A, random W
    std::vector<float> A(B*XK);
    for(auto&v:A)v=(float)(rand()%1000-500)/500.0f;

    std::vector<float> W(XK*XN);
    for(auto&v:W)v=(float)(rand()%1000-500)/500.0f;

    printf("Packing weight...\n");
    auto Wpack=make_weight(W.data(),XN,XK);

    NpuEngine npu;if(!npu.init()){printf("NPU init failed\n");return 1;}
    std::vector<float> C(B*XN);

    // Warmup
    printf("Warmup...\n");
    npu.gemm_batch(A.data(),B,XK,Wpack,XN,C.data(),1);

    // Benchmark
    printf("Benchmarking...\n");
    auto t0=std::chrono::steady_clock::now();
    npu.gemm_batch(A.data(),B,XK,Wpack,XN,C.data(),N_TRIALS);
    auto t1=std::chrono::steady_clock::now();

    double ms_total=std::chrono::duration<double,std::milli>(t1-t0).count();
    double ms_per_call=ms_total/N_TRIALS;

    // FLOPs per call: M * K * N * 2 (MAC = 2 FLOPs)
    double flops_per_call=(double)B*XK*XN*2;
    double tflops=flops_per_call/(ms_per_call/1000.0)/1e12;

    // Peak throughput: 19.5 TFLOPS (measured at M=128 with one row)
    double peak_tflops=19.5;
    double util=tflops/peak_tflops*100;

    printf("\n=== RESULTS ===\n");
    printf("  Batch M:     %d\n",B);
    printf("  Matrix:      %dx%d @ %dx%d -> %dx%d\n",B,XK,XK,XN,B,XN);
    printf("  Trials:      %d\n",N_TRIALS);
    printf("  Total time:  %.1f ms\n",ms_total);
    printf("  Per call:    %.3f ms (%.1f us)\n",ms_per_call,ms_per_call*1000);
    printf("  FLOPs/call:  %.0f (M*K*N*2 = %d*%d*%d*2)\n",flops_per_call,B,XK,XN);
    printf("  TFLOPS:      %.2f\n",tflops);
    printf("  Peak:        19.5 TFLOPS\n");
    printf("  Utilization: %.1f%%\n",util);

    // Sanity: check first and last values of C
    printf("\n  C[0]=%.4f C[-1]=%.4f\n",C[0],C[B*XN-1]);

    printf("\n=== SWEEP: M=1..%d ===\n",XM);
    printf("  %5s | %10s | %10s | %8s\n","M","ms/call","TFLOPS","Util%");
    std::vector<int> msweep={1,2,4,8,16,32,64,128,256,512,1024};
    for(int m:msweep){
        if(m>XM)break;
        auto t2=std::chrono::steady_clock::now();
        int nt=std::max(1,N_TRIALS*32/m); // more trials for small M
        npu.gemm_batch(A.data(),m,XK,Wpack,XN,C.data(),nt);
        auto t3=std::chrono::steady_clock::now();
        double ms=std::chrono::duration<double,std::milli>(t3-t2).count()/nt;
        double flops=(double)m*XK*XN*2;
        double tf=flops/(ms/1000.0)/1e12;
        printf("  %5d | %8.3f ms | %8.2f | %5.1f%%\n",m,ms,tf,tf/peak_tflops*100);
    }

    printf("\nDone.\n");
    return 0;
}

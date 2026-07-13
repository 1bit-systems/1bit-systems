#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "helper.h"
#include "gemm_atb_layout.h"
static float bf16f(uint16_t v){uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}
static uint16_t fbf16(float v){uint32_t b;memcpy(&b,&v,4);uint32_t r=((b>>16)&1)+0x7FFF;return(uint16_t)((b+r)>>16);}
int main(){
    const int M=128,K=1024,N=2048,mt=128,kt=64,nt=128;
    const char*xp="/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/final_128x1024x2048_128x64x128.xclbin";
    const char*ip="/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/insts_128x1024x2048_128x64x128.txt";
    FILE*f=fopen(ip,"rb");if(!f)return 1;fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);
    std::vector<uint32_t> ins(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);
    auto d=xrt::device(0);auto xc=xrt::xclbin(std::string(xp));d.register_xclbin(xc);
    auto hw=xrt::hw_context(d,xc.get_uuid());auto k=xrt::kernel(hw,"MLIR_AIE");
    srand(42);
    std::vector<float> Af(M*K),Bf(K*N);
    for(int i=0;i<M*K;i++)Af[i]=(float)(rand()%100)/100.0f;
    for(int i=0;i<K*N;i++)Bf[i]=(float)(rand()%100)/100.0f;
    std::vector<float> Ccpu(M*N,0);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){float s=0;for(int k=0;k<K;k++)s+=Af[m*K+k]*Bf[k*N+n];Ccpu[m*N+n]=s;}
    auto As=gemm_atb::layout_A_L1_2x1_8x8block(Af,M,K,mt,kt);
    std::vector<uint16_t> Ab(M*K);for(int i=0;i<M*K;i++)Ab[i]=fbf16(As[i]);
    auto Bs=gemm_atb::layout_transpose_L1_1x2_8x8block(Bf,K,N,kt,nt);
    auto Bp=floatToBfp16(8,K*N,Bs.data(),0);
    auto bi=xrt::bo(d,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k.group_id(1));
    auto bA=xrt::bo(d,M*K*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(3));
    auto bB=xrt::bo(d,Bp.size(),XRT_BO_FLAGS_HOST_ONLY,k.group_id(4));
    auto bC=xrt::bo(d,M*N*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(5));
    memcpy(bi.map(),ins.data(),ins.size()*4);bi.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(bA.map(),Ab.data(),M*K*2);bA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(bB.map(),Bp.data(),Bp.size());bB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memset(bC.map(),0,M*N*2);bC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto start=std::chrono::high_resolution_clock::now();
    auto run=k((unsigned)3,bi,ins.size(),bA,bB,bC);run.wait();
    auto stop=std::chrono::high_resolution_clock::now();
    bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    uint16_t*cb=(uint16_t*)bC.map();
    int err=0;double max_rel=0,max_abs=0;
    for(int i=0;i<M*N;i++){
        float got=bf16f(cb[i]),ref=Ccpu[i];
        double a=fabs(got-ref),r=a/(fabs(ref)+1e-30);
        if(r>max_rel)max_rel=r;if(a>max_abs)max_abs=a;
        if(r>0.5&&a>0.5){if(err<5)printf("  [%d] got=%.3f ref=%.3f\n",i,got,ref);err++;}
    }
    double us=std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count();
    double gf=2.0*M*K*N/(us*1000);
    printf("Max rel: %e max abs: %e errors: %d  %.0fus %.0fGFLOPS\n",max_rel,max_abs,err,us,gf);
    return err?1:0;
}

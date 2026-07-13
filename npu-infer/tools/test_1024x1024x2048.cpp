#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "helper.h"
#include "gemm_atb_layout.h"
static float bf16f(uint16_t v){uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}
static uint16_t fbf16(float v){uint32_t b;memcpy(&b,&v,4);uint32_t r=((b>>16)&1)+0x7FFF;return(uint16_t)((b+r)>>16);}
int main(){
    const int M=1024,K=1024,N=2048,mt=128,kt=64,nt=128,nrows=4;
    const char*xp="/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/final_1024x1024x2048_128x64x128.xclbin";
    const char*ip="/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/insts_1024x1024x2048_128x64x128.txt";
    FILE*f=fopen(ip,"rb");if(!f)return 1;fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);
    std::vector<uint32_t> ins(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);
    printf("Loaded %zu ins\n",ins.size());
    auto d=xrt::device(0);auto xc=xrt::xclbin(std::string(xp));d.register_xclbin(xc);
    auto hw=xrt::hw_context(d,xc.get_uuid());auto k=xrt::kernel(hw,"MLIR_AIE");
    printf("K ready\n");
    // Simple test data
    std::vector<float> Af(M*K,1.0f),Bf(K*N,2.0f);
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
    auto run=k((unsigned)3,bi,ins.size(),bA,bB,bC);
    run.wait();
    bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    uint16_t*cb=(uint16_t*)bC.map();
    std::vector<float> Cr(M*N);for(int i=0;i<M*N;i++)Cr[i]=bf16f(cb[i]);
    auto C=gemm_atb::layout_inverse_C_L1_2x2_8x8block(Cr,M,N,nrows*mt,nt);
    float exp=K*2.0f;int err=0;
    printf("First 4x4:\n");
    for(int m=0;m<4;m++){printf("r%d:",m);for(int n=0;n<4;n++){float g=C[m*N+n];printf(" %.0f",g);if(fabs(g-exp)>2)err++;}printf("\n");}
    printf("row %d first4:",M-1);for(int n=0;n<4;n++)printf(" %.0f",C[(M-1)*N+n]);printf("\n");
    double gf=2.0*M*K*N/1000.0;
    printf("%s (%.0f GFLOPS)\n",err==0?"PASS":"FAIL",gf);
    return err?1:0;
}

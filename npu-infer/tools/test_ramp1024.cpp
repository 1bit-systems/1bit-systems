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
    const int M=1024,K=1024,N=2048,mt=128,kt=64,nt=128;
    const char*xp="/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/final_1024x1024x2048_128x64x128.xclbin";
    const char*ip="/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/insts_1024x1024x2048_128x64x128.txt";
    FILE*f=fopen(ip,"rb");if(!f)return 1;fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);
    std::vector<uint32_t> ins(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);
    auto d=xrt::device(0);auto xc=xrt::xclbin(std::string(xp));d.register_xclbin(xc);
    auto hw=xrt::hw_context(d,xc.get_uuid());auto k=xrt::kernel(hw,"MLIR_AIE");
    // A: row_id (0..1023) + 1, repeated across K
    // B: col_id (0..2047) % 3 - 1  → period 3 values -1,0,1  
    std::vector<float> Af(M*K),Bf(K*N);
    for(int i=0;i<M*K;i++)Af[i]=1.0f+(float)(i/K); // row 0 = 1, row 1 = 2, ...
    for(int i=0;i<K*N;i++)Bf[i]=(float)((i%3)-1);  // -1, 0, 1 repeating
    // CPU ref: C[m,n] = row_val[m] * sum_k B[k,n] = row_val[m] * (K/3)*( -1+0+1 ) = 0? No.
    // sum over 1024: 1024/3=341 remainder 1. Sum of one cycle of B = -1+0+1=0.
    // Over 341 full cycles: 0. Plus 1 remaining (= -1): sum = -1 for each column.
    // But the remaining term depends on n: position in cycle.
    // Actually: for col n, B[k,n] = ((k*N+n)%3)-1, so sum_k = sum over k of ((kN+n)%3)-1
    // This is column-independent! Because k*N advances by N mod 3... N=2048, 2048%3=2.
    // So B[k,n] = ((k*2048+n)%3)-1. For fixed n: k=0..1023. Pattern over k: (n%3, (n+2)%3, (n+4)%3, ...)
    // = stepping by 2 in mod-3 space. Step of 2 in mod 3 = full cycle since gcd(2,3)=1.
    // Over 1024 steps: 1024/3=341 full cycles + 1 extra. Each full cycle sums to 0.
    // Extra 1: B[1023,n] = ((1023*2048+n)%3)-1. 1023*2048 = 2095104. 2095104%3 = 2095104/3=698368 rem 0.
    // So B[1023,n] = (0+n)%3-1 = n%3-1. Extra element = n%3-1 = {-1,0,1}.
    // Sum_k B[k,n] = n%3-1 for each column. So C[m,n] = (m+1) * (n%3-1).
    printf("CPU ref: C[m,n] = (m+1) * (n%%3-1)\n");
    // NPU path
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
    auto run=k((unsigned)3,bi,ins.size(),bA,bB,bC);run.wait();
    bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    uint16_t*cb=(uint16_t*)bC.map();
    int err=0;
    printf("First 4x4 (NPU):\n");
    for(int m=0;m<4;m++){printf("  r%d:",m);for(int n=0;n<4;n++){
        float got=bf16f(cb[m*N+n]);
        float exp=(m+1.0f)*(n%3-1);
        printf(" %.0f(%.0f)",got,exp);
        if(fabs(got-exp)>2.0f)err++;
    }printf("\n");}
    // Check more rows
    for(int m:{127,128,511,512,1023}){
        float exp=(m+1.0f)*(0%3-1); // col 0
        float got=bf16f(cb[m*N]);
        printf("  [%d,0] = %.0f (expect %.0f) %s\n",m,got,exp,fabs(got-exp)<2?"OK":"FAIL");
        if(fabs(got-exp)>2)err++;
    }
    printf("Total errors: %d\n",err);
    return err?1:0;
}

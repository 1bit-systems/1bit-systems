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
    auto d=xrt::device(0);auto xc=xrt::xclbin(std::string(xp));d.register_xclbin(xc);
    auto hw=xrt::hw_context(d,xc.get_uuid());auto k=xrt::kernel(hw,"MLIR_AIE");
    // A: ramp per row = row_id (0..1023) + 1, so each row unique
    std::vector<float> Af(M*K);for(int i=0;i<M*K;i++)Af[i]=1.0f+(float)(i/K);
    std::vector<float> Bf(K*N);for(int i=0;i<K*N;i++)Bf[i]=1.0f;
    // CPU ref
    std::vector<float> Cref(M*N,0);
    for(int m=0;m<M;m++){float a_val=1.0f+m;for(int n=0;n<N;n++)Cref[m*N+n]=a_val*K;}
    // NPU path: A sent as raw BF16
    std::vector<uint16_t> Ab(M*K);for(int i=0;i<M*K;i++)Ab[i]=fbf16(Af[i]);
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
    std::vector<float> C(M*N);for(int i=0;i<M*N;i++)C[i]=bf16f(cb[i]);
    // Check raw output
    printf("Row 0: expect %.0f\n",Cref[0]);
    printf("Row 0 NPU first 5:");
    for(int n=0;n<5;n++)printf(" %.0f",C[n]);
    printf("\nRow 127 NPU first 5:");
    for(int n=0;n<5;n++)printf(" %.0f",C[127*N+n]);
    printf("\nRow 128 NPU first 5:");
    for(int n=0;n<5;n++)printf(" %.0f",C[128*N+n]);
    int err=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++)if(fabs(C[m*N+n]-Cref[m*N+n])>2.0)err++;
    printf("Errors: %d / %d\n%s\n",err,M*N,err==0?"PASS":"FAIL");
    return err?1:0;
}

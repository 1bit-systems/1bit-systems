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
static const int MT=128,KT=64,NT=128;
int main(){
    const int M=256,K=1024,N=4096,mt=128,kt=64,nt=128;
    const char*xp="/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/final_256x1024x4096_128x64x128.xclbin";
    const char*ip="/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/insts_256x1024x4096_128x64x128.txt";
    FILE*f=fopen(ip,"rb");if(!f)return 1;fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);
    std::vector<uint32_t>ins(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);
    auto d=xrt::device(0);auto xc=xrt::xclbin(std::string(xp));d.register_xclbin(xc);
    auto hw=xrt::hw_context(d,xc.get_uuid());auto k=xrt::kernel(hw,"MLIR_AIE");
    
    // Test with M=256 (2 tokens, padded)
    std::vector<float> Af(M*K,1.0f); // All 1.0
    std::vector<float> Bf(K*N,3.0f); // All 3.0
    // Expected output: C[m,n] = sum_k(A[m,k] * B[k,n]) = K*3 = 3072
    
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
    
    printf("Running M=256, K=1024, N=4096 multi-token GEMM...\n");fflush(stdout);
    auto run=k((unsigned)3,bi,ins.size(),bA,bB,bC);
    auto state=run.wait();
    printf("State: %d %s\n",(int)state,state==ERT_CMD_STATE_COMPLETED?"OK":"FAIL");
    if(state!=ERT_CMD_STATE_COMPLETED)return 1;
    
    bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    uint16_t*cb=(uint16_t*)bC.map();
    float exp=3.0f*K;int err=0;
    // Check all M*N elements
    printf("First 8 output values: ");
    for(int i=0;i<8;i++)printf("%.0f ",bf16f(cb[i]));
    printf(" (expected %.0f)\n",exp);
    for(int i=0;i<M*N;i++){float v=bf16f(cb[i]);if(fabs(v-exp)>2.0f)err++;}
    printf("Errors: %d / %d (%.1f%%)\n",err,M*N,100.0*err/(M*N));
    printf("%s\n",err==0?"PASS":"FAIL");
    return err?1:0;
}

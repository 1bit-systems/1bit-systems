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
    
    // Two M=128 shuffles concatenated (proven approach from M=128 xclbins)
    std::vector<float> A0(128*K,1.0f), A1(128*K,2.0f);
    std::vector<float> Bf(K*N,3.0f);
    
    auto As0=gemm_atb::layout_A_L1_2x1_8x8block(A0,128,K,MT,KT);
    auto As1=gemm_atb::layout_A_L1_2x1_8x8block(A1,128,K,MT,KT);
    std::vector<uint16_t> Ab(M*K);
    for(int i=0;i<128*K;i++)Ab[i]=fbf16(As0[i]);
    for(int i=0;i<128*K;i++)Ab[128*K+i]=fbf16(As1[i]);
    
    auto Bs=gemm_atb::layout_transpose_L1_1x2_8x8block(Bf,K,N,KT,NT);
    auto Bp=floatToBfp16(8,K*N,Bs.data(),0);
    
    auto bi=xrt::bo(d,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k.group_id(1));
    auto bA=xrt::bo(d,M*K*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(3));
    auto bB=xrt::bo(d,Bp.size(),XRT_BO_FLAGS_HOST_ONLY,k.group_id(4));
    auto bC=xrt::bo(d,M*N*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(5));
    
    memcpy(bi.map(),ins.data(),ins.size()*4);bi.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(bA.map(),Ab.data(),M*K*2);bA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(bB.map(),Bp.data(),Bp.size());bB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memset(bC.map(),0,M*N*2);bC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    printf("Testing M=256 with 2×M=128 A shuffles...\n");fflush(stdout);
    auto run=k((unsigned)3,bi,ins.size(),bA,bB,bC);run.wait();
    bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    
    // Read raw BF16, decode to float, apply inverse C layout
    uint16_t*cb=(uint16_t*)bC.map();
    std::vector<float> Cf(M*N);
    for(int i=0;i<M*N;i++)Cf[i]=bf16f(cb[i]);
    // Inverse layout: M=256, N=4096, L1_block_m=2*MT=256, NT=128
    auto Cu=gemm_atb::layout_inverse_C_L1_2x2_8x8block(Cf,M,N,2*MT,NT);
    
    float v0=Cu[0*N+0], v1=Cu[1*N+0]; // Row 0 (token 0) and row 1 (token 1)
    float v128=Cu[128*N+0]; // Row 128 (should be token 1's first output row)
    printf("Cu[0][0]=%.0f Cu[1][0]=%.0f Cu[128][0]=%.0f\n",v0,v1,v128);
    printf("Expect: tok0=3072 tok1=6144\n");
    
    int err=0; float exp0=3.0f*K, exp1=6.0f*K;
    for(int m=0;m<M;m++){
        float exp=(m<128)?exp0:exp1;
        for(int n=0;n<N;n++){
            if(fabs(Cu[m*N+n]-exp)>2.0f)err++;
        }
    }
    printf("Errors: %d/%d (%.1f%%)\n",err,M*N,100.0*err/(M*N));
    printf("%s\n",err<M*N*0.1?"PASS":"FAIL");
    return err?1:0;
}

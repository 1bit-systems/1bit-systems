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
    // Use M=1024 xclbin, pad A to 1024 rows (first 512 real, rest 0)
    const int M=1024, M_real=512, K=1024, N=2048, mt=128, kt=64, nt=128, nrows=4;
    const char*xp="/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/final_1024x1024x2048_128x64x128.xclbin";
    const char*ip="/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/insts_1024x1024x2048_128x64x128.txt";
    FILE*f=fopen(ip,"rb");if(!f)return 1;fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);
    std::vector<uint32_t> ins(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);
    auto d=xrt::device(0);auto xc=xrt::xclbin(std::string(xp));d.register_xclbin(xc);
    auto hw=xrt::hw_context(d,xc.get_uuid());auto k=xrt::kernel(hw,"MLIR_AIE");
    // A: first M_real rows = ramp -3..3, rest padded 0
    std::vector<float> Af(M*K,0.0f);
    for(int i=0;i<M_real*K;i++)Af[i]=(float)((i%7)-3);
    std::vector<float> Bf(K*N,2.0f);
    // CPU ref (first M_real rows only)
    std::vector<float> Cref(M_real*N,0);
    for(int m=0;m<M_real;m++)for(int n=0;n<N;n++){float s=0;for(int k=0;k<K;k++)s+=Af[m*K+k]*Bf[k*N+n];Cref[m*N+n]=s;}
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
    std::vector<float> Cr(M*N);for(int i=0;i<M*N;i++)Cr[i]=bf16f(cb[i]);
    auto C=gemm_atb::layout_inverse_C_L1_2x2_8x8block(Cr,M,N,nrows*mt,nt);
    // Check first 4 real rows vs CPU
    printf("First 4 rows of %d (NPU vs CPU):\n",M_real);
    int err=0;float max_rel=0,max_abs=0;
    for(int m=0;m<4;m++){
        printf("  r%d: ",m);for(int n=0;n<4;n++){
            float g=C[m*N+n],r=Cref[m*N+n];
            double a=fabs(g-r),rl=a/(fabs(r)+1e-30);
            if(rl>max_rel)max_rel=rl;if(a>max_abs)max_abs=a;
            printf("%.0f(%.0f) ",g,r);if(rl>0.5&&a>0.5)err++;
        }
        printf("\n");
    }
    printf("  Max rel: %e max abs: %e errors: %d\n",max_rel,max_abs,err);
    double gf=2.0*M_real*K*N/1000.0;
    printf("%s\n",err==0?"PASS":"FAIL");
    return err?1:0;
}

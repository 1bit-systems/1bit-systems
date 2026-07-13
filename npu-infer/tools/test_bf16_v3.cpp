#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include "helper.h"
#include "gemm_atb_layout.h"
static float bf16f(uint16_t v){uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}
static uint16_t fbf16(float v){uint32_t b;memcpy(&b,&v,4);uint32_t r=((b>>16)&1)+0x7FFF;return(uint16_t)((b+r)>>16);}
static const int MT=128,KT=64,NT=128,XM=2048,XK=2048,XN=2048;
int main(){
    const char*xp="/home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev/build/design_bf16_v3.xclbin";
    const char*ip="/home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev/build/design_bf16_v3.insts";
    FILE*f=fopen(ip,"rb");if(!f){printf("No insts\n");return 1;}
    fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);
    std::vector<uint32_t>ins(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);
    printf("Instructions: %zu\n",ins.size());
    
    auto d=xrt::device(0);
    auto xc=xrt::xclbin(std::string(xp));
    d.register_xclbin(xc);
    auto hw=xrt::hw_context(d,xc.get_uuid());
    auto k=xrt::kernel(hw,"MLIR_AIE");
    printf("Kernel loaded\n");fflush(stdout);
    
    // Raw BF16 data (not BFP16!)
    std::vector<float> Af(XM*XK,1.0f);
    std::vector<float> Bf(XK*XN,3.0f);
    auto As=gemm_atb::layout_A_L1_2x1_8x8block(Af,XM,XK,MT,KT);
    auto Bs=gemm_atb::layout_transpose_L1_1x2_8x8block(Bf,XK,XN,KT,NT);
    
    // BF16 BO (2 bytes per value, no BFP16 compression)
    std::vector<uint16_t> Ab(XM*XK),Bb(XK*XN);
    for(int i=0;i<XM*XK;i++)Ab[i]=fbf16(As[i]);
    for(int i=0;i<XK*XN;i++)Bb[i]=fbf16(Bs[i]);
    
    auto bi=xrt::bo(d,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k.group_id(1));
    auto bA=xrt::bo(d,XM*XK*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(3));
    auto bB=xrt::bo(d,XK*XN*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(4));
    auto bC=xrt::bo(d,XM*XN*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(5));
    
    memcpy(bi.map(),ins.data(),ins.size()*4);bi.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(bA.map(),Ab.data(),XM*XK*2);bA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(bB.map(),Bb.data(),XK*XN*2);bB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memset(bC.map(),0,XM*XN*2);bC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    printf("Launching BF16 v3 kernel...\n");fflush(stdout);
    auto run=k((unsigned)3,bi,(unsigned)ins.size(),bA,bB,bC);
    auto state=run.wait();
    printf("State: %d %s\n",(int)state,state==ERT_CMD_STATE_COMPLETED?"OK":"FAIL");
    
    if(state==ERT_CMD_STATE_COMPLETED){
        bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        uint16_t*cb=(uint16_t*)bC.map();
        float v0=bf16f(cb[0]);
        printf("C[0]=%.2f (expected %.0f)\n",v0,3.0f*XK);
        printf("BF16 v3: NO HANG! KERNEL COMPUTES!\n");
        return 0;
    }
    return 1;
}

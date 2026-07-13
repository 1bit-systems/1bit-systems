#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
static const int M=128,K=1024,N=4096;
int main(){
    const char*xp="/home/bcloud/npu-sandbox/npu-infer/build/int8/final_i8_QKV.xclbin";
    const char*ip="/home/bcloud/npu-sandbox/npu-infer/build/int8/insts_i8_QKV.txt";
    FILE*f=fopen(ip,"rb");if(!f){printf("No insts\n");return 1;}
    fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);
    std::vector<uint32_t>ins(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);
    printf("Instructions: %zu\n",ins.size());
    
    auto d=xrt::device(0);auto xc=xrt::xclbin(std::string(xp));
    d.register_xclbin(xc);auto hw=xrt::hw_context(d,xc.get_uuid());
    auto k=xrt::kernel(hw,"MLIR_AIE");
    printf("Kernel loaded\n");
    
    std::vector<int8_t> A(M*K,1);
    std::vector<int8_t> B(K*N,2);
    std::vector<int16_t> C(M*N,0);
    
    auto bi=xrt::bo(d,ins.size()*4,XRT_BO_FLAGS_HOST_ONLY,k.group_id(1));
    auto bA=xrt::bo(d,M*K*1,XRT_BO_FLAGS_HOST_ONLY,k.group_id(3));
    auto bB=xrt::bo(d,K*N*1,XRT_BO_FLAGS_HOST_ONLY,k.group_id(4));
    auto bC=xrt::bo(d,M*N*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(5));
    
    memcpy(bi.map(),ins.data(),ins.size()*4);bi.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(bA.map(),A.data(),M*K*1);bA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(bB.map(),B.data(),K*N*1);bB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memset(bC.map(),0,M*N*2);bC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    printf("Running INT8 GEMM...\n");fflush(stdout);
    auto run=k((unsigned)3,bi,(unsigned)ins.size(),bA,bB,bC);
    auto state=run.wait();
    printf("State: %d %s\n",(int)state,state==ERT_CMD_STATE_COMPLETED?"OK":"FAIL");
    if(state!=ERT_CMD_STATE_COMPLETED)return 1;
    
    bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    int16_t*cb=(int16_t*)bC.map();
    float exp=2.0f*K;int err=0;
    printf("First 8: ");
    for(int i=0;i<8;i++)printf("%d ",cb[i]);
    printf(" (expect %.0f)\n",exp);
    for(int i=0;i<M*N;i++){if(fabs(cb[i]-exp)>2.0f)err++;}
    printf("Errors: %d/%d (%.1f%%)\n",err,M*N,100.0*err/(M*N));
    printf("%s\n",err==0?"PASS":"FAIL");
    return err?1:0;
}

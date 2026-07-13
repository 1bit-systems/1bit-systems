// Minimal test: verify 2-layer QKV xclbin processes N=8192 correctly
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

int main() {
    const int M=128, K=1024, N=8192;
    
    FILE*f=fopen("/home/bcloud/npu-sandbox/npu-infer/build/int8/insts_i8_QKV_2layer.txt","rb");
    if(!f){fprintf(stderr,"FAIL: no insts\n");return 1;}
    fseek(f,0,2);long isz=ftell(f);fseek(f,0,0);
    std::vector<uint32_t>ins(isz/4);
    if(fread(ins.data(),4,ins.size(),f)!=(size_t)ins.size()){fprintf(stderr,"short read\n");return 1;}
    fclose(f);
    fprintf(stderr,"Instructions: %zu\n",ins.size());
    
    xrt::device dev(0);
    
    std::string xp="/home/bcloud/npu-sandbox/npu-infer/build/int8/final_i8_QKV_2layer.xclbin";
    auto xc=xrt::xclbin(xp);
    dev.register_xclbin(xc);
    auto hc=xrt::hw_context(dev,xc.get_uuid());
    auto k=xrt::kernel(hc,"MLIR_AIE");
    
    auto bI=xrt::bo(dev,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k.group_id(1));
    memcpy(bI.map(),ins.data(),ins.size()*4);
    bI.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    auto bA=xrt::bo(dev,(size_t)M*K,XRT_BO_FLAGS_HOST_ONLY,k.group_id(3));
    auto bB=xrt::bo(dev,(size_t)K*N,XRT_BO_FLAGS_HOST_ONLY,k.group_id(4));
    auto bC=xrt::bo(dev,(size_t)M*N*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(5));
    
    auto*Am=(int8_t*)bA.map();
    auto*Bm=(int8_t*)bB.map();
    for(int i=0;i<M*K;i++)Am[i]=(int8_t)(rand()%64-32);
    for(int i=0;i<K*N;i++)Bm[i]=(int8_t)(rand()%64-32);
    bA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    fprintf(stderr,"Running GEMM %dx%d...\n",M,N);
    auto ts=std::chrono::steady_clock::now();
    
    auto r=k((unsigned)3,bI,(unsigned)ins.size(),bA,bB,bC);
    r.wait();
    
    auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ts).count();
    fprintf(stderr,"Done: %.2fms\n",ms);
    
    bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto*Cm=(int16_t*)bC.map();
    
    int64_t sum0=0,sum1=0;
    for(int m=0;m<M;m++){
        for(int n=0;n<N/2;n++)sum0+=Cm[m*N+n];
        for(int n=N/2;n<N;n++)sum1+=Cm[m*N+n];
    }
    fprintf(stderr,"First half  (cols 0..4095): avg=%ld\n",(long)(sum0/(M*N/2)));
    fprintf(stderr,"Second half (cols 4096..8191): avg=%ld\n",(long)(sum1/(M*N/2)));
    
    if(sum0==0||sum1==0){
        fprintf(stderr,"FAIL: all zeros\n");return 1;
    }
    fprintf(stderr,"PASS\n");
    return 0;
}

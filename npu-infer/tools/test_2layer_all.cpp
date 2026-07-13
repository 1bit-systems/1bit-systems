// Test all 2-layer xclbins
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <string>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

static bool test(const char*label,const char*xp,const char*instp,
                 int M,int K,int N,int&out_sum){
    printf("  %s (%dx%dx%d)... ",label,M,K,N);fflush(stdout);
    
    FILE*f=fopen(instp,"rb");if(!f){printf("NO INSTS\n");return false;}
    fseek(f,0,2);long isz=ftell(f);fseek(f,0,0);
    std::vector<uint32_t>ins(isz/4);
    if(fread(ins.data(),4,ins.size(),f)!=(size_t)ins.size()){printf("SHORT\n");fclose(f);return false;}
    fclose(f);
    
    xrt::device dev(0);
    auto xc=xrt::xclbin(std::string(xp));
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
    
    auto ts=std::chrono::steady_clock::now();
    auto r=k((unsigned)3,bI,(unsigned)ins.size(),bA,bB,bC);
    r.wait();
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ts).count();
    
    bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto*Cm=(int16_t*)bC.map();
    int64_t sum=0;
    for(int i=0;i<M*N;i++)sum+=Cm[i];
    out_sum=(int)sum;
    printf("%.2fms sum=%ld %s\n",ms,(long)sum,sum!=0?"OK":"ZERO");
    return sum!=0;
}

int main(){
    int s;
    bool ok=true;
    ok&=test("QKV_2layer","/home/bcloud/npu-sandbox/npu-infer/build/int8/final_i8_QKV_2layer.xclbin",
             "/home/bcloud/npu-sandbox/npu-infer/build/int8/insts_i8_QKV_2layer.txt",
             128,1024,8192,s);
    ok&=test("O_2layer","/home/bcloud/npu-sandbox/npu-infer/build/int8/final_i8_O_2layer.xclbin",
             "/home/bcloud/npu-sandbox/npu-infer/build/int8/insts_i8_O_2layer.txt",
             128,2048,2048,s);
    ok&=test("D_2layer","/home/bcloud/npu-sandbox/npu-infer/build/int8/final_i8_D_2layer.xclbin",
             "/home/bcloud/npu-sandbox/npu-infer/build/int8/insts_i8_D_2layer.txt",
             128,3072,2048,s);
    printf("=== %s ===\n",ok?"ALL PASS":"SOME FAILED");
    return ok?0:1;
}

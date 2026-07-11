// Raw GEMM benchmark — measures INT8 TFLOPS at varying batch sizes
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
int main(){
    auto d=xrt::device(0);
    auto xc=xrt::xclbin(std::string("/home/bcloud/npu-sandbox/npu-infer/build/int8/final_i8_QKV_v.xclbin"));
    d.register_xclbin(xc);auto h=xrt::hw_context(d,xc.get_uuid());auto k=xrt::kernel(h,"MLIR_AIE");
    FILE*f=fopen("/home/bcloud/npu-sandbox/npu-infer/build/int8/insts_i8_QKV_v.txt","rb");
    fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);std::vector<uint32_t>ins(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);
    auto bI=xrt::bo(d,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k.group_id(1));
    memcpy(bI.map(),ins.data(),ins.size()*4);bI.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    int K=1024,N=4096;
    for(int M:{1,4,9,16,32,64}){
        auto bA=xrt::bo(d,(size_t)128*K,XRT_BO_FLAGS_HOST_ONLY,k.group_id(3));
        auto bB=xrt::bo(d,(size_t)K*N,XRT_BO_FLAGS_HOST_ONLY,k.group_id(4));
        auto bC=xrt::bo(d,(size_t)128*N*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(5));
        memset(bA.map(),0,128*K);memset(bB.map(),0,K*N);memset(bC.map(),0,128*N*2);
        int8_t*Am=(int8_t*)bA.map();for(int i=0;i<M*K;i++)Am[i]=1;
        for(int i=0;i<K*N;i++)((int8_t*)bB.map())[i]=1;
        bA.sync(XCL_BO_SYNC_BO_TO_DEVICE);bB.sync(XCL_BO_SYNC_BO_TO_DEVICE);bC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto t0=std::chrono::high_resolution_clock::now();int niter=20;
        for(int it=0;it<niter;it++){auto r=k((unsigned)3,bI,(unsigned)ins.size(),bA,bB,bC);r.wait();}
        auto t1=std::chrono::high_resolution_clock::now();
        double ms=std::chrono::duration<double,std::milli>(t1-t0).count()/niter;
        double ops=2.0*M*K*N,tops=ops/(ms*1e9);
        printf("M=%3d: %6.2fms  %7.1f GFLOPS  %6.1f TOPS  util=%5.1f%%\n",M,ms,ops/(ms*1e6),tops,tops/50.0*100);
    }
}

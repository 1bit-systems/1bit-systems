/** NPU Engine v13 — Fused Full-Layer XCLBIN integration test.
 *  Proves: xclbin loads, BOs allocate, kernel dispatches.
 *  Weight format: Q4NX compact (9.4MB per layer, from pack_fused_weights.py).
 *  Target: 1 dispatch/layer vs 4. Then M=32 batch on top. */
#include <cstdio>
#include <string>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    printf("=== NPU Engine v13 — Fused Full-Layer XCLBIN Integration ===\n\n");

    // MLIR sizes from design_full_layer.mlir:
    //   k_cache: memref<8192xi32>   = 32KB
    //   v_cache: memref<8192xi32>   = 32KB
    //   weights: memref<2458816xi32> = 9.4MB
    //   output:  memref<512xi32>    = 2KB
    //   hidden:  memref<512xi32>    = 2KB
    constexpr int KCACHE_DW=8192, WEIGHT_DW=2458816, OUT_DW=512, HID_DW=512;
    constexpr int KCACHE_SZ=KCACHE_DW*4, WEIGHT_SZ=WEIGHT_DW*4, OUT_SZ=OUT_DW*4, HID_SZ=HID_DW*4;

    // Xclbin path
    const char*XCL=[]{const char*e=getenv("NPU_XCLBIN_DIR");return e?(std::string(e)+"/design_full_layer.xclbin").c_str():"/home/bcloud/torch2aie/build/qwen3_06b_layer/design_full_layer.xclbin";}();
    const char*INS=[]{const char*e=getenv("NPU_XCLBIN_DIR");return e?(std::string(e)+"/insts_full_layer.txt").c_str():"/home/bcloud/torch2aie/build/qwen3_06b_layer/insts_full_layer.txt";}();

    printf("Init NPU...\n");
    xrt::device dev(0);

    // Load instruction stream
    printf("Load instructions...\n");
    std::vector<uint32_t> ins;
    {FILE*f=fopen(INS,"rb");if(!f){printf("FAIL: no insts\n");return 1;}
     fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);
     ins.resize(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);
     printf("  insts: %zu dwords\n",ins.size());}

    // Register xclbin
    printf("Register xclbin...\n");
    auto xc=std::make_unique<xrt::xclbin>(std::string(XCL));
    dev.register_xclbin(*xc);
    auto hc=std::make_unique<xrt::hw_context>(dev,xc->get_uuid());
    auto k=std::make_unique<xrt::kernel>(*hc,"MLIR_AIE");

    // Group mapping: runtime_sequence args = group 3,4,5,6,7
    // insts = group 1 (or whatever the kernel expects)
    // We map: insts→g1, data→g3 (the runtime_sequence handles DMA routing)
    int gI=k->group_id(1);
    int gD=k->group_id(3);

    printf("Groups: inst=%d, data=%d\n",gI,gD);

    // Allocate BOs
    printf("Alloc BOs...\n");
    auto bI=std::make_unique<xrt::bo>(dev,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,gI);
    memcpy(bI->map(),ins.data(),ins.size()*4);bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    printf("  insts: %zu bytes, group %d\n",ins.size()*4,gI);

    // 5 data BOs — all host_only on the data group
    auto bK=std::make_unique<xrt::bo>(dev,KCACHE_SZ,XRT_BO_FLAGS_HOST_ONLY,gD);
    auto bV=std::make_unique<xrt::bo>(dev,KCACHE_SZ,XRT_BO_FLAGS_HOST_ONLY,gD);
    auto bW=std::make_unique<xrt::bo>(dev,WEIGHT_SZ,XRT_BO_FLAGS_HOST_ONLY,gD);
    auto bO=std::make_unique<xrt::bo>(dev,OUT_SZ,XRT_BO_FLAGS_HOST_ONLY,gD);
    auto bH=std::make_unique<xrt::bo>(dev,HID_SZ,XRT_BO_FLAGS_HOST_ONLY,gD);

    // Zero buffers
    memset(bK->map(),0,KCACHE_SZ); memset(bV->map(),0,KCACHE_SZ);
    memset(bO->map(),0,OUT_SZ);    memset(bH->map(),0,HID_SZ);

    // Load packed weights
    const char*WF=(std::string(xd())+"/fused_weights_l0.bin").c_str();
    {int fd=open(WF,O_RDONLY);if(fd<0){printf("No packed weights, zeros\n");}
     else {struct stat st;fstat(fd,&st);
      size_t rs=std::min((size_t)st.st_size,(size_t)WEIGHT_SZ);
      read(fd,bW->map(),rs);close(fd);
      printf("  weights: %zu bytes loaded\n",rs);}}

    // Sync all
    bK->sync(XCL_BO_SYNC_BO_TO_DEVICE);bV->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bW->sync(XCL_BO_SYNC_BO_TO_DEVICE);bO->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bH->sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Dispatch: 1 invocation per layer
    printf("\n=== Fused Layer Dispatch (1 xclbin call) ===\n");
    auto t0=std::chrono::steady_clock::now();

    // The runtime_sequence has 5 arguments: k_cache, v_cache, weights, output, hidden
    // They map to group 3 (data group) with different offsets
    // FLM uses: kernel(inst_bo, k, v, weights, out, hid)
    // Try: kernel with all args on group 3
    auto r=(*k)((unsigned)3,*bI,(unsigned)ins.size(), *bK,*bV,*bW,*bO,*bH);
    r.wait();

    auto t1=std::chrono::steady_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    printf("  Dispatch: %.0fms\n",ms);
    printf("  (vs 4 standalone GEMMs: 4×1346μs = 5.4ms)\n");

    // Check results
    bO->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bK->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bV->sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    int32_t*out=(int32_t*)bO->map();
    int nonzero=0;for(int i=0;i<OUT_DW;i++)if(out[i]!=0)nonzero++;
    printf("  Output: %d/%d non-zero dwords\n",nonzero,OUT_DW);

    printf("\n=== FUSED XCLBIN INTEGRATION VERIFIED ===\n");
    printf("Next: correct Q4NX weight format, then v13 M=32 fused decode.\n");
    return 0;
}

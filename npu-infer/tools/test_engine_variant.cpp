#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "helper.h"
#include "gemm_atb_layout.h"
extern "C" float* dequant_i8_to_float(const uint8_t*,int,int*,int*);
extern "C" float bf16_to_float(uint16_t);
extern "C" uint16_t float_to_bf16(float);
static float bf16f(uint16_t v){uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}
static float bf16_g(uint16_t v){return((v&0x7F80)==0x7F80)?0:bf16f(v);}
static uint16_t fbf16(float v){uint32_t b;memcpy(&b,&v,4);uint32_t r=((b>>16)&1)+0x7FFF;return(uint16_t)((b+r)>>16);}
static const int MT=128, KT=64, NT=128;

struct VarInfo {const char *xclbin,*insts;int XK,XN;};
static VarInfo VARIANTS[]={
    {"/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/final_128x1024x1024_128x64x128.xclbin","/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/insts_128x1024x1024_128x64x128.txt",1024,1024},
    {"/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/final_128x1024x2048_128x64x128.xclbin","/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/insts_128x1024x2048_128x64x128.txt",1024,2048},
    {"/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/final_128x1024x3072_128x64x128.xclbin","/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/insts_128x1024x3072_128x64x128.txt",1024,3072},
    {"/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/final_128x2048x1024_128x64x128.xclbin","/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/insts_128x2048x1024_128x64x128.txt",2048,1024},
    {"/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/final_128x3072x1024_128x64x128.xclbin","/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/insts_128x3072x1024_128x64x128.txt",3072,1024}
};

int test_with_weights(int vi, float *weights, int out_feat, int in_feat, int deq_rows, bool debug=false) {
    VarInfo &va = VARIANTS[vi];
    int XK = va.XK, XN = va.XN;
    auto d = xrt::device(0); auto xc = xrt::xclbin(std::string(va.xclbin)); d.register_xclbin(xc);
    auto hw = xrt::hw_context(d, xc.get_uuid()); auto k = xrt::kernel(hw, "MLIR_AIE");
    FILE *f = fopen(va.insts, "rb"); if(!f) return 1;
    fseek(f,0,2); long sz=ftell(f); fseek(f,0,0);
    std::vector<uint32_t> ins(sz/4); fread(ins.data(),4,ins.size(),f); fclose(f);
    auto bi = xrt::bo(d,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k.group_id(1));
    auto bA = xrt::bo(d,(size_t)128*XK*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(3));
    auto bB = xrt::bo(d,(size_t)XK*XN*9/8,XRT_BO_FLAGS_HOST_ONLY,k.group_id(4));
    auto bC = xrt::bo(d,(size_t)128*XN*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(5));
    memcpy(bi.map(),ins.data(),ins.size()*4); bi.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    // Pack weights using same logic as pre_pack_variant
    std::vector<float> Bf(XK*XN);
    bool trans=(deq_rows==in_feat&&in_feat!=out_feat);
    for(int k=0;k<in_feat&&k<XK;k++)for(int n=0;n<out_feat&&n<XN;n++){
        float v=trans?weights[k*out_feat+n]:weights[n*in_feat+k];
        if(std::isfinite(v))Bf[k*XN+n]=v*16.0f;
    }
    auto Bs=gemm_atb::layout_transpose_L1_1x2_8x8block(Bf,XK,XN,KT,NT);
    auto Bp=floatToBfp16(8,XK*XN,Bs.data(),0);
    memcpy(bB.map(),Bp.data(),Bp.size()); bB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    // A = all 1.0
    std::vector<float> Af(128*XK,1.0f);
    auto As=gemm_atb::layout_A_L1_2x1_8x8block(Af,128,XK,MT,KT);
    uint16_t*Ab=(uint16_t*)bA.map();
    for(int i=0;i<128*XK;i++)Ab[i]=fbf16(As[i]);
    bA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    memset(bC.map(),0,(size_t)128*XN*2); bC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run=k((unsigned)3,bi,(unsigned)ins.size(),bA,bB,bC); run.wait();
    bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    
    uint16_t*Cb=(uint16_t*)bC.map();
    if(debug){
        printf("  First 5 output: ");
        for(int i=0;i<5;i++)printf("%.1f ",bf16f(Cb[i]));
        printf("\n");
    }
    // Check if output has sensible values (not NaN, not 0, not tiny)
    float maxv=0,minv=1e30;
    for(int i=0;i<128*XN;i++){
        float v=bf16f(Cb[i]);
        if(std::isnan(v)) return -1;  // NaN = catastrophic failure
        if(v>maxv)maxv=v;
        if(v<minv)minv=v;
    }
    if(maxv<1e-10f) return -2;  // all zeros
    if(maxv>0)return 0;  // has output
    return -3;
}

int main() {
    const char *mp="/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
    int fd=open(mp,O_RDONLY);struct stat st;fstat(fd,&st);
    uint8_t*d=(uint8_t*)mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    uint64_t hsz;memcpy(&hsz,d,8);uint64_t df=8+hsz;
    auto i8p=[&](uint64_t o){return d+df+o;};
    
    // Test layer 0 weights
    const char*js=(const char*)(d+8);size_t jl=hsz;
    auto joff=[&](const char*nm)->uint64_t{
        size_t nl=strlen(nm);const char*end=js+jl;
        while(js<end){auto q=(const char*)memmem(js,end-js,nm,nl);if(!q)return 0;
        if(q>js&&*(q-1)=='"'&&*(q+nl)=='"'){auto oo=strstr(q,"\"data_offsets\"");if(oo){auto ar=strchr(oo,'[');if(ar)return (uint64_t)strtoull(ar+1,NULL,10);}}js=q+1;}
        return 0;
    };
    char b[128];snprintf(b,sizeof(b),"model.layers.0.self_attn.q_proj.weight");uint64_t qp=joff(b);
    snprintf(b,sizeof(b),"model.layers.0.self_attn.k_proj.weight");uint64_t kp=joff(b);
    snprintf(b,sizeof(b),"model.layers.0.self_attn.v_proj.weight");uint64_t vp=joff(b);
    snprintf(b,sizeof(b),"model.layers.0.self_attn.o_proj.weight");uint64_t op=joff(b);
    snprintf(b,sizeof(b),"model.layers.0.mlp.gate_proj.weight");uint64_t gp=joff(b);
    snprintf(b,sizeof(b),"model.layers.0.mlp.up_proj.weight");uint64_t up=joff(b);
    snprintf(b,sizeof(b),"model.layers.0.mlp.down_proj.weight");uint64_t dp=joff(b);
    
    int r,c;
    float*q=dequant_i8_to_float(i8p(qp),256,&r,&c);printf("Q: rows=%d cols=%d\n",r,c);
    float*k=dequant_i8_to_float(i8p(kp),128,&r,&c);printf("K: rows=%d cols=%d\n",r,c);
    float*v=dequant_i8_to_float(i8p(vp),128,&r,&c);printf("V: rows=%d cols=%d\n",r,c);
    float*o=dequant_i8_to_float(i8p(op),256,&r,&c);printf("O: rows=%d cols=%d\n",r,c);
    float*g=dequant_i8_to_float(i8p(gp),384,&r,&c);printf("G: rows=%d cols=%d\n",r,c);
    float*u=dequant_i8_to_float(i8p(up),384,&r,&c);printf("U: rows=%d cols=%d\n",r,c);
    float*dw=dequant_i8_to_float(i8p(dp),384,&r,&c);printf("D: rows=%d cols=%d\n",r,c);
    
    printf("\nTesting each variant with actual weights:\n");
    struct Test{float*w;int out,in,deq,vi;const char*name;};
    Test tests[]={
        {q,2048,1024,256,1,"Q (1024x2048)"},
        {k,1024,1024,128,0,"K (1024x1024)"},
        {v,1024,1024,128,0,"V (1024x1024)"},
        {o,1024,2048,256,3,"O (2048x1024)"},
        {g,3072,1024,384,2,"G (1024x3072)"},
        {u,3072,1024,384,2,"U (1024x3072)"},
        {dw,1024,3072,384,4,"D (3072x1024)"},
    };
    for(auto&t:tests){
        printf("  %s: ",t.name);
        int res=test_with_weights(t.vi,t.w,t.out,t.in,t.deq,true);
        if(res==0)printf("OK\n");
        else if(res==-1)printf("NAN output!\n");
        else if(res==-2)printf("All zeros!\n");
        else printf("Error %d\n",res);
    }
    free(q);free(k);free(v);free(o);free(g);free(u);free(dw);
    munmap(d,st.st_size);
    return 0;
}

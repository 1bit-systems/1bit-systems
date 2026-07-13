#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "helper.h"
#include "gemm_atb_layout.h"
extern "C" float* dequant_i8_to_float(const uint8_t*,int,int*,int*);
static float bf16f(uint16_t v){uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}
static uint16_t fbf16(float v){uint32_t b;memcpy(&b,&v,4);uint32_t r=((b>>16)&1)+0x7FFF;return(uint16_t)((b+r)>>16);}
static const int MT=128, KT=64, NT=128, MP=128;

static void cpu_matmul(const float*A,const float*B,float*C,int M,int N,int K){
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){double s=0;for(int k=0;k<K;k++)s+=A[m*K+k]*B[k*N+n];C[m*N+n]=(float)s;}
}

int test_gemm(const char*xp,const char*ip,int XK,int XN,const float*A,int M,int K,int N,const float*W,int out_feat,int in_feat) {
    FILE *f = fopen(ip, "rb"); if(!f) return -1;
    fseek(f,0,2); long sz=ftell(f); fseek(f,0,0);
    std::vector<uint32_t> ins(sz/4); fread(ins.data(),4,ins.size(),f); fclose(f);
    auto d=xrt::device(0); auto xc=xrt::xclbin(std::string(xp)); d.register_xclbin(xc);
    auto hw=xrt::hw_context(d,xc.get_uuid()); auto k=xrt::kernel(hw,"MLIR_AIE");
    auto bi=xrt::bo(d,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k.group_id(1));
    auto bA=xrt::bo(d,(size_t)MP*XK*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(3));
    auto bB=xrt::bo(d,(size_t)XK*XN*9/8,XRT_BO_FLAGS_HOST_ONLY,k.group_id(4));
    auto bC=xrt::bo(d,(size_t)MP*XN*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(5));
    memcpy(bi.map(),ins.data(),ins.size()*4); bi.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    // Pack weights
    std::vector<float> Bf(XK*XN);
    bool trans=(in_feat==K && K!=N);
    for(int k=0;k<K&&k<XK;k++)for(int n=0;n<N&&n<XN;n++){
        float v=trans?W[k*N+n]:W[n*K+k];
        if(std::isfinite(v))Bf[k*XN+n]=v*16.0f;
    }
    auto Bs=gemm_atb::layout_transpose_L1_1x2_8x8block(Bf,XK,XN,KT,NT);
    auto Bp=floatToBfp16(8,XK*XN,Bs.data(),0);
    memcpy(bB.map(),Bp.data(),Bp.size()); bB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    // Pad + shuffle A
    std::vector<float> Af(MP*XK);
    for(int m=0;m<M;m++)for(int k=0;k<K;k++)Af[m*XK+k]=A[m*K+k];
    auto As=gemm_atb::layout_A_L1_2x1_8x8block(Af,MP,XK,MT,KT);
    uint16_t*Ab=(uint16_t*)bA.map();
    for(int i=0;i<MP*XK;i++)Ab[i]=fbf16(As[i]);
    bA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    memset(bC.map(),0,(size_t)MP*XN*2); bC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run=k((unsigned)3,bi,(unsigned)ins.size(),bA,bB,bC); run.wait();
    bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    
    uint16_t*Cb=(uint16_t*)bC.map();
    // Check for NaN
    int nan_cnt=0;
    for(int i=0;i<M*N;i++)if(std::isnan(bf16f(Cb[i*XN/XN?1:0])))nan_cnt++; // simple check
    for(int i=0;i<M*N;i++)if(std::isnan(bf16f(Cb[i])))nan_cnt++;
    
    // CPU reference
    std::vector<float> Ccpu(M*N);
    cpu_matmul(A,W,Ccpu.data(),M,N,K);
    
    // Compare
    int err=0; double max_rel=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){
        float got=bf16f(Cb[m*XN+n]), ref=Ccpu[n*K+m];  // W might be transposed
        double r=fabs(got-ref)/(fabs(ref)+1e-30);
        if(r>max_rel)max_rel=r;
        if(r>0.5&&fabs(got-ref)>0.5)err++;
    }
    printf("  NPU: nan=%d/%d  CPU compare: err=%d max_rel=%e  first=%.2f\n", nan_cnt, M*N, err, max_rel, bf16f(Cb[0]));
    return nan_cnt;
}

int main() {
    const char *mp="/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
    int fd=open(mp,O_RDONLY); struct stat st; fstat(fd,&st);
    uint8_t*d=(uint8_t*)mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
    uint64_t hsz; memcpy(&hsz,d,8); uint64_t df=8+hsz;
    auto i8p=[&](uint64_t o){return d+df+o;};
    const char*js=(const char*)(d+8); size_t jl=hsz;
    auto joff=[&](const char*nm)->int{
        size_t nl=strlen(nm);const char*e=js+jl;
        while(js<e){auto q=(const char*)memmem(js,e-js,nm,nl);if(!q)return 0;
        if(q>js&&*(q-1)=='"'&&*(q+nl)=='"'){auto oo=strstr(q,"\"data_offsets\"");if(oo){auto ar=strchr(oo,'[');if(ar)return (int)strtoul(ar+1,NULL,10);}}js=q+1;}
        return 0;
    };
    char b[128]; snprintf(b,sizeof(b),"model.layers.0.self_attn.o_proj.weight"); uint64_t op=joff(b);
    int r,c; float*o=dequant_i8_to_float(i8p(op),256,&r,&c);
    printf("O weight: %dx%d\n", r, c);
    
    // Generate a realistic test input (random)
    srand(42);
    std::vector<float> A(2048);
    for(int i=0;i<2048;i++)A[i]=(float)(rand()%200-100)/100.0f;
    
    printf("Testing O projection (2048x1024):\n");
    test_gemm(
        "/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/final_128x2048x1024_128x64x128.xclbin",
        "/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/insts_128x2048x1024_128x64x128.txt",
        2048, 1024,
        A.data(), 1, 2048, 1024,
        o, 1024, 2048
    );
    
    free(o); munmap(d,st.st_size); return 0;
}

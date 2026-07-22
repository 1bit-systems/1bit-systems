// GPU acceleration test for Laguna 1BP
#include "onebp_loader.h"
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <chrono>
#include <dlfcn.h>
#include <unordered_map>

static uint16_t bf16(const uint8_t* b) { return (uint16_t)b[0]|((uint16_t)b[1]<<8); }
static float f16f(uint16_t bf) { uint32_t bits=(uint32_t)bf<<16; float f; memcpy(&f,&bits,4); return f; }

// CPU matmul
static void mm_cpu(float* y, const float* x, const uint8_t* wd, int M, int K) {
    int tr=32,tc=256,gs=32,ng=tc/gs,rb=ng*4+tc/2,tby=tr*rb;
    int n_tr=(K+tr-1)/tr,n_tc=(M+tc-1)/tc;
    std::fill(y,y+M,0.0f); std::vector<float> tb(tr*tc);
    for(int trr=0;trr<n_tr;trr++){int rs=trr*tr,re=std::min(rs+tr,K),reff=re-rs;
        for(int tcc=0;tcc<n_tc;tcc++){int cs=tcc*tc,ce=std::min(cs+tc,M),ceff=ce-cs;
            const uint8_t*tp=wd+((trr*(size_t)n_tc+tcc)*tby);
            for(int r=0;r<tr;r++){const uint8_t*rp=tp+r*rb;
                for(int g=0;g<ng;g++){float sc=f16f(bf16(rp+g*2)),zp=f16f(bf16(rp+ng*2+g*2));
                    const uint8_t*pk=rp+ng*4;
                    for(int i=0;i<gs;i++){int bi=g*gs/2+i/2,n=(i&1)?(pk[bi]>>4):(pk[bi]&0x0F);
                        float v=(float)n*sc+zp; if(v!=v||v>1e10f||v<-1e10f)v=0;
                        tb[r*(size_t)tc+g*gs+i]=v;}}
            }
            for(int c=0;c<ceff;c++){float s=0;for(int r=0;r<reff;r++)s+=x[rs+r]*tb[r*(size_t)tc+c];y[cs+c]+=s;}}}
}

int main(int argc, char** argv) {
    OnebpModel m; if(!m.load(argv[1])) return 1;
    int H=m.header.hidden_size, V=m.header.vocab_size;
    
    uint8_t* emb=nullptr; size_t eb=0;
    for(auto& t:m.tensors) if(t.name=="token_embd.weight"&&t.ndim==2){emb=m.tensor_data(t);eb=t.bytes;}
    if(!emb) return 1;
    
    // CPU benchmark
    std::vector<float> x(V,0), y(H);
    x[100]=1;
    auto t0=std::chrono::high_resolution_clock::now();
    int w=3,i=5; for(int j=0;j<w;j++) mm_cpu(y.data(),x.data(),emb,H,V);
    t0=std::chrono::high_resolution_clock::now();
    for(int j=0;j<i;j++) mm_cpu(y.data(),x.data(),emb,H,V);
    float cpu_ms=std::chrono::duration<float,std::milli>(std::chrono::high_resolution_clock::now()-t0).count()/i;
    printf("CPU: %.0f ms  GPU: ", cpu_ms);
    
    // GPU test
    void* hl=dlopen("libamdhip64.so",RTLD_NOW);
    if(!hl) hl=dlopen("libamdhip64.so.6",RTLD_NOW);
    if(!hl) { printf("no HIP\\n"); return 0; }
    
    auto M=(int(*)(void**,size_t))dlsym(hl,"hipMalloc");
    auto F=(int(*)(void*))dlsym(hl,"hipFree");
    auto C=(int(*)(void*,const void*,size_t,int))dlsym(hl,"hipMemcpy");
    auto Z=(int(*)(void*,int,size_t))dlsym(hl,"hipMemset");
    auto SY=(int(*)())dlsym(hl,"hipDeviceSynchronize");
    
    void* rl=dlopen("./librocm_cpp.so",RTLD_NOW);
    if(!rl) rl=dlopen("librocm_cpp.so",RTLD_NOW);
    auto KL=(void(*)(const float*,const uint8_t*,float*,int,int,void*))dlsym(rl,"launch_q4nx_gemv");
    
    if(!M||!F||!C||!Z||!SY||!KL) { printf("missing syms\\n"); return 0; }
    
    void *xd,*yd,*wd;
    M(&xd,V*4); M(&yd,H*4); M(&wd,eb);
    
    // Upload weights once
    C(wd, emb, eb, 1);
    
    // GPU benchmark
    C(xd, x.data(), V*4, 1);
    for(int j=0;j<w;j++){Z(yd,0,H*4);KL((const float*)xd,(const uint8_t*)wd,(float*)yd,V,H,nullptr);SY();}
    t0=std::chrono::high_resolution_clock::now();
    for(int j=0;j<i;j++){Z(yd,0,H*4);KL((const float*)xd,(const uint8_t*)wd,(float*)yd,V,H,nullptr);SY();}
    float gpu_ms=std::chrono::duration<float,std::milli>(std::chrono::high_resolution_clock::now()-t0).count()/i;
    
    float gv[5]; C(gv,yd,20,2);
    printf("%.1f ms (%.1fx)\\n  Result: %.4f %.4f (vs CPU: %.4f)\\n", 
           gpu_ms, cpu_ms/gpu_ms, gv[0], gv[1], y[0]);
    
    // Compare
    float cv[2]; cv[0]=y[0]; cv[1]=y[1];
    printf("  Match: %s\\n", fabs(cv[0]-gv[0])<0.01?"YES":"NO");
    
    F(xd); F(yd); F(wd);
    dlclose(rl); dlclose(hl);
    return 0;
}

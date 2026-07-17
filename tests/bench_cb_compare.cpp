#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include "rocm_cpp/bonsai.h"
#define HIP_OK(e) do { hipError_t _s = (e); if(_s!=hipSuccess) { fprintf(stderr,"HIP err %d at %s:%d\n",(int)_s,__FILE__,__LINE__); exit(1); }} while(0)
extern "C" void bonsai_tq2_1024_cb_launch(const uint8_t*,const uint16_t*,uint16_t*,int,int,void*);

constexpr int HS=2048,IS=6144,NL=28,NH=16,NKV=8,HD=128;

int main() {
    hipStream_t s; HIP_OK(hipStreamCreate(&s));
    auto mk=[&](int r,int c){uint8_t*a,*d;
        HIP_OK(hipMalloc(&a,(size_t)r*(size_t)(c/128)*34));
        HIP_OK(hipMemsetAsync(a,0x55,(size_t)r*(size_t)(c/128)*34,s));
        HIP_OK(hipDeviceSynchronize());
        bonsai_tq2_convert_to_1024(a,&d,r,c);HIP_OK(hipFree(a));return d;};
    std::vector<uint8_t*> w;
    for(int l=0;l<NL;l++){w.push_back(mk(NH*HD,HS));w.push_back(mk(NKV*HD,HS));w.push_back(mk(NKV*HD,HS));
        w.push_back(mk(HS,NH*HD));w.push_back(mk(IS,HS));w.push_back(mk(IS,HS));w.push_back(mk(HS,IS));}
    uint16_t*a,*o;HIP_OK(hipMalloc(&a,8192*2));HIP_OK(hipMalloc(&o,16384*2));
    HIP_OK(hipMemsetAsync(a,0x3C,8192*2,s));HIP_OK(hipDeviceSynchronize());

    auto bench=[&](auto launch,const char* name){
        hipEvent_t t0,t1;HIP_OK(hipEventCreate(&t0));HIP_OK(hipEventCreate(&t1));
        HIP_OK(hipEventRecord(t0,s));
        for(int run=0;run<10;run++) for(int l=0;l<NL;l++){int i=l*7;
            launch(w[i],a,o,NH*HD,HS,s);launch(w[i+1],a,o,NKV*HD,HS,s);launch(w[i+2],a,o,NKV*HD,HS,s);
            launch(w[i+3],a,o,HS,NH*HD,s);launch(w[i+4],a,o,IS,HS,s);launch(w[i+5],a,o,IS,HS,s);
            launch(w[i+6],a,o,HS,IS,s);}
        HIP_OK(hipEventRecord(t1,s));HIP_OK(hipEventSynchronize(t1));
        float ms;HIP_OK(hipEventElapsedTime(&ms,t0,t1));double t=1000.0/(ms/10.0);
        HIP_OK(hipEventDestroy(t0));HIP_OK(hipEventDestroy(t1));
        printf("  %s: %.0f tok/s\n",name,t); return t;};

    double t1=bench(+bonsai_tq2_1024_gemv_launch,"Original");
    double t2=bench(+bonsai_tq2_1024_cb_launch,"CacheBlk");
    printf("  Speedup: %.2f×\n",t2/t1);

    for(auto p:w) hipFree(p);hipFree(a);hipFree(o);HIP_OK(hipStreamDestroy(s));
}

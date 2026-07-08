/* Hybrid: GPU Sgemv + CPU math for everything else.
 * Runs on GPU-resident weights. Should produce SAME output as engine_gpu.c. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#define MAX_CTX 4096
#define I8_ROW_B 5120
#define TILE_R 32
#define TILE_C 256
#define DEF_MODEL "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx"
enum { H=1024, NC=28, NH=16, NKV=8, HD=128, IM=3072, NV=151936, GQA=2, QT=NH*HD+2*NKV*HD };
static float bf16(uint16_t v) { uint32_t b=(uint32_t)v<<16;float f;memcpy(&f,&b,4);return f; }

/* CPU math — exact copies from engine_gpu.c */
static void cpu_rms(float *x, const float *w, int n) {
    double ss=0; for(int i=0;i<n;i++){if(!isfinite(x[i]))x[i]=0;ss+=(double)x[i]*(double)x[i];}
    float ir=1.0f/sqrtf((float)(ss/(double)n)+1e-6f); for(int i=0;i<n;i++)x[i]*=ir*w[i];
}
static void cpu_rope(float *x, int p, const float *s, const float *c) {
    for(int d=0;d<HD/2;d++){float a=x[d],b=x[d+HD/2];x[d]=a*c[p*HD+d]-b*s[p*HD+d];x[d+HD/2]=b*c[p*HD+d]+a*s[p*HD+d];}
}
static void cpu_attn(const float *q, const float *kc, const float *vc, float *o, int sl) {
    float sc=1.0f/sqrtf((float)HD);
    for(int h=0;h<NH;h++){int kvh=h/GQA;const float *qh=q+h*HD;float mx=-INFINITY;float sb[4096];
        for(int p=0;p<sl;p++){const float *kr=kc+p*NKV*HD+kvh*HD;double dot=0;
            for(int d=0;d<HD;d++)dot+=(double)qh[d]*(double)kr[d];sb[p]=(float)(dot*sc);if(sb[p]>mx)mx=sb[p];}
        double su=0;for(int p=0;p<sl;p++){sb[p]=expf(sb[p]-mx);su+=sb[p];}
        float iv=su>0?(float)(1.0/su):0;float *oh=o+h*HD;memset(oh,0,(size_t)HD*4);
        for(int p=0;p<sl;p++){float w=sb[p]*iv;const float *vr=vc+p*NKV*HD+kvh*HD;for(int d=0;d<HD;d++)oh[d]+=w*vr[d];}}
}

typedef struct { float *q,*k,*v,*o,*g,*u,*d; } DevW;
static float* deq_i8(const uint8_t *d, int i8r, int id, int *or_, int *oc) {
    int ntc=id/256;if(ntc<1)ntc=1;int ntr=i8r/ntc;
    *or_=ntr*32;*oc=ntc*256;
    float *o=(float*)calloc((size_t)(*or_)*(*oc),4);
    for(int ir=0;ir<i8r;ir++){const uint8_t *rd=d+ir*I8_ROW_B;int tr=ir/ntc,tc=ir%ntc;
        const uint16_t *sc=(const uint16_t*)rd,*zp=(const uint16_t*)(rd+512);const uint8_t *pk=rd+1024;
        for(int lr=0;lr<TILE_R;lr++){int lane=lr/16,lr2=lr%16,bi=lr2/2,ns=lr%2;const uint8_t *ld=pk+lane*(TILE_C*8);
            for(int c=0;c<TILE_C;c++){int g=c/32;float s=bf16(sc[g*32+lr]),z=bf16(zp[g*32+lr]);
                uint8_t bv=ld[c*8+bi];int cd=(ns==0)?(bv&0x0F):((bv>>4)&0x0F);
                o[(tr*TILE_R+lr)*(*oc)+(tc*TILE_C+c)]=(float)cd*s+z;}}}
    return o;
}
static const uint8_t *fk(const uint8_t *j,size_t l,const char *k){
    size_t kl=strlen(k);for(size_t i=0;i+kl+2<l;i++)if(j[i]=='"'&&memcmp(j+i+1,k,kl)==0&&j[i+1+kl]=='"')return j+i;return 0;
}
static int64_t fo(const uint8_t *j,size_t l,const char *k){
    const uint8_t *p=fk(j,l,k);if(!p)return -1;const uint8_t *d=(const uint8_t*)strstr((const char*)p,"\"data_offsets\"");if(!d)return -1;
    const uint8_t *b=(const uint8_t*)strchr((const char*)d,'[');return b?strtoll((const char*)(b+1),0,10):-1;
}
static int si(const uint8_t *j,size_t l,const char *k){
    size_t kl=strlen(k);const char*js=(const char*)j;for(const char*p=js;p<js+(int)l;p++){p=strstr(p,k);if(!p)return 0;
        if((p==js||*(p-1)=='"')&&*(p+kl)=='"'){const char*sh=strstr(p,"\"shape\"");if(!sh)return 0;const char*br=strchr(sh,'[');if(!br)return 0;return (int)strtoul(br+1,0,10);}p+=kl;}return 0;
}

/* GPU Sgemv: din,dout on device, dw on device. Sync after each. */
static void gpu_mm_sync(hipblasHandle_t blas, float *d_one, float *d_zero, 
                        const float *din, int od, int id, float *dout, const float *dw) {
    hipblasSgemv(blas,HIPBLAS_OP_T,id,od,d_one,dw,id,din,1,d_zero,dout,1);
    hipDeviceSynchronize();
}

int main(){
    fprintf(stderr,"=== Hybrid: GPU Sgemv + CPU math ===\n");
    int fd=open(DEF_MODEL,O_RDONLY); struct stat st; fstat(fd,&st);
    uint8_t *map=(uint8_t*)mmap(0,st.st_size,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
    uint64_t hdr; memcpy(&hdr,map,8); size_t ds=8+(size_t)hdr;
    const uint8_t *js=map+8; size_t jl=(size_t)hdr;

    // Load CPU data
    float *emb=(float*)malloc((size_t)NV*H*4);
    const uint16_t *eb=(const uint16_t*)(map+ds+(size_t)fo(js,jl,"model.embed_tokens.weight"));
    for(int i=0;i<NV*H;i++)emb[i]=bf16(eb[i]);
    float *in[NC],*pa[NC],*qn[NC],*kn[NC],*fn;
    for(int l=0;l<NC;l++){char buf[256];
        snprintf(buf,256,"model.layers.%d.input_layernorm.weight",l);in[l]=(float*)malloc(H*4);const uint16_t*s=(const uint16_t*)(map+ds+(size_t)fo(js,jl,buf));for(int i=0;i<H;i++)in[l][i]=bf16(s[i]);
        snprintf(buf,256,"model.layers.%d.post_attention_layernorm.weight",l);pa[l]=(float*)malloc(H*4);s=(const uint16_t*)(map+ds+(size_t)fo(js,jl,buf));for(int i=0;i<H;i++)pa[l][i]=bf16(s[i]);
        snprintf(buf,256,"model.layers.%d.self_attn.q_norm.weight",l);int64_t o=fo(js,jl,buf);if(o>0){qn[l]=(float*)malloc(HD*4);const uint16_t*q=(const uint16_t*)(map+ds+(size_t)o);for(int i=0;i<HD;i++)qn[l][i]=bf16(q[i]);}else qn[l]=0;
        snprintf(buf,256,"model.layers.%d.self_attn.k_norm.weight",l);o=fo(js,jl,buf);if(o>0){kn[l]=(float*)malloc(HD*4);const uint16_t*k=(const uint16_t*)(map+ds+(size_t)o);for(int i=0;i<HD;i++)kn[l][i]=bf16(k[i]);}else kn[l]=0;
    }
    fn=(float*)malloc(H*4);const uint16_t*fs=(const uint16_t*)(map+ds+(size_t)fo(js,jl,"model.norm.weight"));for(int i=0;i<H;i++)fn[i]=bf16(fs[i]);
    float *rs=(float*)malloc((size_t)MAX_CTX*HD*4),*rc=(float*)malloc((size_t)MAX_CTX*HD*4);
    float th=1000000.0f;for(int p=0;p<MAX_CTX;p++)for(int d=0;d<HD/2;d++){float f=1.0f/powf(th,(float)d/(HD/2)),a=(float)p*f;rs[p*HD+d]=sinf(a);rc[p*HD+d]=cosf(a);rs[p*HD+HD/2+d]=sinf(a);rc[p*HD+HD/2+d]=cosf(a);}

    // Init GPU
    hipblasHandle_t blas; hipblasCreate(&blas);
    float *d_one,*d_zero;
    hipMalloc(&d_one,4); hipMalloc(&d_zero,4);
    float one=1.0f,zero=0.0f;
    hipMemcpy(d_one,&one,4,hipMemcpyHostToDevice);
    hipMemcpy(d_zero,&zero,4,hipMemcpyHostToDevice);
    hipblasSetPointerMode(blas,HIPBLAS_POINTER_MODE_DEVICE);

    float *dh,*dout,*datt;  // GPU scratch
    hipMalloc(&dh,H*4);
    hipMalloc(&datt,(size_t)NH*HD*4);
    hipMalloc(&dout,((QT>(2*IM))?QT:(2*IM))*4);

    // Upload weights
    DevW dw[NC];
    for(int l=0;l<NC;l++){char buf[256];
        snprintf(buf,256,"model.layers.%d.self_attn.q_proj.weight",l);int or_,oc;float*c=deq_i8(map+ds+fo(js,jl,buf),si(js,jl,buf),H,&or_,&oc);hipMalloc(&dw[l].q,(size_t)or_*oc*4);hipMemcpy(dw[l].q,c,(size_t)or_*oc*4,hipMemcpyHostToDevice);free(c);
        snprintf(buf,256,"model.layers.%d.self_attn.k_proj.weight",l);c=deq_i8(map+ds+fo(js,jl,buf),si(js,jl,buf),H,&or_,&oc);hipMalloc(&dw[l].k,(size_t)or_*oc*4);hipMemcpy(dw[l].k,c,(size_t)or_*oc*4,hipMemcpyHostToDevice);free(c);
        snprintf(buf,256,"model.layers.%d.self_attn.v_proj.weight",l);c=deq_i8(map+ds+fo(js,jl,buf),si(js,jl,buf),H,&or_,&oc);hipMalloc(&dw[l].v,(size_t)or_*oc*4);hipMemcpy(dw[l].v,c,(size_t)or_*oc*4,hipMemcpyHostToDevice);free(c);
        snprintf(buf,256,"model.layers.%d.self_attn.o_proj.weight",l);c=deq_i8(map+ds+fo(js,jl,buf),si(js,jl,buf),NH*HD,&or_,&oc);hipMalloc(&dw[l].o,(size_t)or_*oc*4);hipMemcpy(dw[l].o,c,(size_t)or_*oc*4,hipMemcpyHostToDevice);free(c);
        snprintf(buf,256,"model.layers.%d.mlp.gate_proj.weight",l);c=deq_i8(map+ds+fo(js,jl,buf),si(js,jl,buf),H,&or_,&oc);hipMalloc(&dw[l].g,(size_t)or_*oc*4);hipMemcpy(dw[l].g,c,(size_t)or_*oc*4,hipMemcpyHostToDevice);free(c);
        snprintf(buf,256,"model.layers.%d.mlp.up_proj.weight",l);c=deq_i8(map+ds+fo(js,jl,buf),si(js,jl,buf),H,&or_,&oc);hipMalloc(&dw[l].u,(size_t)or_*oc*4);hipMemcpy(dw[l].u,c,(size_t)or_*oc*4,hipMemcpyHostToDevice);free(c);
        snprintf(buf,256,"model.layers.%d.mlp.down_proj.weight",l);c=deq_i8(map+ds+fo(js,jl,buf),si(js,jl,buf),IM,&or_,&oc);hipMalloc(&dw[l].d,(size_t)or_*oc*4);hipMemcpy(dw[l].d,c,(size_t)or_*oc*4,hipMemcpyHostToDevice);free(c);
    }
    fprintf(stderr,"Model loaded.\n");

    // CPU buffers
    float h[H],res[H],qkv[QT],at[NH*HD],gt[IM],act[IM];
    float *kc=(float*)calloc((size_t)NC*MAX_CTX*NKV*HD,4),*vc=(float*)calloc((size_t)NC*MAX_CTX*NKV*HD,4);

    // Tokenizer: same as engine_gpu.c
    uint32_t pt[16]; int npt=0;
    pt[npt++]=151644;pt[npt++]=872;pt[npt++]=198;pt[npt++]=9707;pt[npt++]=151645;pt[npt++]=198;pt[npt++]=151644;pt[npt++]=77091;pt[npt++]=198;
    int pos=0;

    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); double t0=ts.tv_sec*1e9+ts.tv_nsec;
    int mx=16; uint32_t out[16]; int gen=0;

    // Prefill
    for(int pi=0;pi<npt;pi++){
        uint32_t tok=pt[pi];
        memcpy(h,emb+(size_t)tok*H,H*4);
        for(int l=0;l<NC;l++){
            /* Layer forward — GPU Sgemv + CPU math */
            memcpy(res,h,H*4);
            cpu_rms(h,in[l],H);

            // Q,K,V on GPU
            hipMemcpy(dh,h,H*4,hipMemcpyHostToDevice);
            gpu_mm_sync(blas,d_one,d_zero,dh,NH*HD,H,dout,dw[l].q);
            hipMemcpy(qkv,dout,(size_t)NH*HD*4,hipMemcpyDeviceToHost);
            gpu_mm_sync(blas,d_one,d_zero,dh,NKV*HD,H,dout+NH*HD,dw[l].k);
            hipMemcpy(qkv+NH*HD,dout+NH*HD,(size_t)NKV*HD*4,hipMemcpyDeviceToHost);
            gpu_mm_sync(blas,d_one,d_zero,dh,NKV*HD,H,dout+NH*HD+NKV*HD,dw[l].v);
            hipMemcpy(qkv+NH*HD+NKV*HD,dout+NH*HD+NKV*HD,(size_t)NKV*HD*4,hipMemcpyDeviceToHost);

            // QK norm + RoPE (CPU)
            for(int hh=0;hh<NH;hh++){float *qh=qkv+hh*HD;double sq=0;for(int d=0;d<HD;d++)sq+=(double)qh[d]*(double)qh[d];
                float iq=1.0f/sqrtf((float)(sq/HD)+1e-6f);for(int d=0;d<HD;d++)qh[d]*=iq*(qn[l]?qn[l][d]:1);cpu_rope(qh,pos,rs,rc);}
            for(int kh=0;kh<NKV;kh++){float *ks=qkv+NH*HD+kh*HD;double sk=0;for(int d=0;d<HD;d++)sk+=(double)ks[d]*(double)ks[d];
                float ik=1.0f/sqrtf((float)(sk/HD)+1e-6f);for(int d=0;d<HD;d++)ks[d]*=ik*(kn[l]?kn[l][d]:1);cpu_rope(ks,pos,rs,rc);
                memcpy(kc+(size_t)l*MAX_CTX*NKV*HD+pos*NKV*HD+kh*HD,ks,HD*4);}
            for(int kh=0;kh<NKV;kh++){float *vs=qkv+NH*HD+NKV*HD+kh*HD;memcpy(vc+(size_t)l*MAX_CTX*NKV*HD+pos*NKV*HD+kh*HD,vs,HD*4);}

            // Attention (CPU)
            cpu_attn(qkv,kc+(size_t)l*MAX_CTX*NKV*HD,vc+(size_t)l*MAX_CTX*NKV*HD,at,pos+1);

            // O projection (GPU) — use datt, not dh!
            hipMemcpy(datt,at,(size_t)NH*HD*4,hipMemcpyHostToDevice);
            gpu_mm_sync(blas,d_one,d_zero,datt,H,NH*HD,dout,dw[l].o);
            hipMemcpy(h,dout,H*4,hipMemcpyDeviceToHost);
            for(int i=0;i<H;i++)h[i]=res[i]+h[i];

            // FFN (CPU math + GPU matmul)
            memcpy(res,h,H*4);
            cpu_rms(h,pa[l],H);

            hipMemcpy(dh,h,H*4,hipMemcpyHostToDevice);
            gpu_mm_sync(blas,d_one,d_zero,dh,IM,H,dout,dw[l].g);
            hipMemcpy(gt,dout,(size_t)IM*4,hipMemcpyDeviceToHost);
            gpu_mm_sync(blas,d_one,d_zero,dh,IM,H,dout+IM,dw[l].u);
            hipMemcpy(act,dout+IM,(size_t)IM*4,hipMemcpyDeviceToHost);

            for(int i=0;i<IM;i++){float gv=gt[i];act[i]=(gv/(1.0f+expf(-gv)))*act[i];}

            hipMemcpy(dh,act,(size_t)IM*4,hipMemcpyHostToDevice);
            gpu_mm_sync(blas,d_one,d_zero,dh,H,IM,dout,dw[l].d);
            hipMemcpy(h,dout,H*4,hipMemcpyDeviceToHost);
            for(int i=0;i<H;i++)h[i]=res[i]+h[i];
        }
        cpu_rms(h,fn,H);
        pos++;
    }

    // Decode
    uint32_t ct=pt[npt-1];
    while(gen<mx){
        memcpy(h,emb+(size_t)ct*H,H*4);
        for(int l=0;l<NC;l++){
            memcpy(res,h,H*4); cpu_rms(h,in[l],H);

            hipMemcpy(dh,h,H*4,hipMemcpyHostToDevice);
            gpu_mm_sync(blas,d_one,d_zero,dh,NH*HD,H,dout,dw[l].q); hipMemcpy(qkv,dout,(size_t)NH*HD*4,hipMemcpyDeviceToHost);
            gpu_mm_sync(blas,d_one,d_zero,dh,NKV*HD,H,dout+NH*HD,dw[l].k); hipMemcpy(qkv+NH*HD,dout+NH*HD,(size_t)NKV*HD*4,hipMemcpyDeviceToHost);
            gpu_mm_sync(blas,d_one,d_zero,dh,NKV*HD,H,dout+NH*HD+NKV*HD,dw[l].v); hipMemcpy(qkv+NH*HD+NKV*HD,dout+NH*HD+NKV*HD,(size_t)NKV*HD*4,hipMemcpyDeviceToHost);

            for(int hh=0;hh<NH;hh++){float *qh=qkv+hh*HD;double sq=0;for(int d=0;d<HD;d++)sq+=(double)qh[d]*(double)qh[d];
                float iq=1.0f/sqrtf((float)(sq/HD)+1e-6f);for(int d=0;d<HD;d++)qh[d]*=iq*(qn[l]?qn[l][d]:1);cpu_rope(qh,pos,rs,rc);}
            for(int kh=0;kh<NKV;kh++){float *ks=qkv+NH*HD+kh*HD;double sk=0;for(int d=0;d<HD;d++)sk+=(double)ks[d]*(double)ks[d];
                float ik=1.0f/sqrtf((float)(sk/HD)+1e-6f);for(int d=0;d<HD;d++)ks[d]*=ik*(kn[l]?kn[l][d]:1);cpu_rope(ks,pos,rs,rc);
                memcpy(kc+(size_t)l*MAX_CTX*NKV*HD+pos*NKV*HD+kh*HD,ks,HD*4);}
            for(int kh=0;kh<NKV;kh++){float *vs=qkv+NH*HD+NKV*HD+kh*HD;memcpy(vc+(size_t)l*MAX_CTX*NKV*HD+pos*NKV*HD+kh*HD,vs,HD*4);}
            cpu_attn(qkv,kc+(size_t)l*MAX_CTX*NKV*HD,vc+(size_t)l*MAX_CTX*NKV*HD,at,pos+1);

            hipMemcpy(datt,at,(size_t)NH*HD*4,hipMemcpyHostToDevice);
            gpu_mm_sync(blas,d_one,d_zero,datt,H,NH*HD,dout,dw[l].o); hipMemcpy(h,dout,H*4,hipMemcpyDeviceToHost);
            for(int i=0;i<H;i++)h[i]=res[i]+h[i];

            memcpy(res,h,H*4); cpu_rms(h,pa[l],H);
            hipMemcpy(dh,h,H*4,hipMemcpyHostToDevice);
            gpu_mm_sync(blas,d_one,d_zero,dh,IM,H,dout,dw[l].g); hipMemcpy(gt,dout,(size_t)IM*4,hipMemcpyDeviceToHost);
            gpu_mm_sync(blas,d_one,d_zero,dh,IM,H,dout+IM,dw[l].u); hipMemcpy(act,dout+IM,(size_t)IM*4,hipMemcpyDeviceToHost);
            for(int i=0;i<IM;i++){float gv=gt[i];act[i]=(gv/(1.0f+expf(-gv)))*act[i];}
            hipMemcpy(dh,act,(size_t)IM*4,hipMemcpyHostToDevice);
            gpu_mm_sync(blas,d_one,d_zero,dh,H,IM,dout,dw[l].d); hipMemcpy(h,dout,H*4,hipMemcpyDeviceToHost);
            for(int i=0;i<H;i++)h[i]=res[i]+h[i];
        }
        cpu_rms(h,fn,H);

        // LM head (CPU)
        float *lg=(float*)malloc((size_t)NV*4);
        for(int n=0;n<NV;n++){double dot=0;const float *r=emb+(size_t)n*H;for(int i=0;i<H;i++)dot+=(double)h[i]*(double)r[i];lg[n]=(float)dot;}
        float bmx=lg[0];uint32_t best=0;for(int i=1;i<NV;i++)if(lg[i]>bmx){bmx=lg[i];best=(uint32_t)i;}
        free(lg);
        out[gen]=best; ct=best; gen++; pos++;
    }

    clock_gettime(CLOCK_MONOTONIC,&ts);
    double ms=(ts.tv_sec*1e9+ts.tv_nsec-t0)/1e6;
    fprintf(stderr,"%d tokens in %.0fms — %.0f tok/s\n",gen,ms,gen/(ms/1000));
    for(int i=0;i<gen;i++)printf("%u ",out[i]);printf("\n");

    return 0;
}

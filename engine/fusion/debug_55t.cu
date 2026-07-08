/* Minimal debug: sync after every Sgemv+kernel, compare with engine_gpu */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
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
#define CEILDIV(a,b) (((a)+(b)-1)/(b))
static float bf16(uint16_t v) { uint32_t b=(uint32_t)v<<16;float f;memcpy(&f,&b,4);return f; }

/* CPU RMS norm — same as engine_gpu.c */
static void cpu_rms(float *x, const float *w, int n) {
    double ss=0; for(int i=0;i<n;i++){if(!isfinite(x[i]))x[i]=0;ss+=(double)x[i]*(double)x[i];}
    float ir=1.0f/sqrtf((float)(ss/(double)n)+1e-6f); for(int i=0;i<n;i++)x[i]*=ir*w[i];
}
static void cpu_rope(float *x, int p, const float *s, const float *c) {
    for(int d=0;d<HD/2;d++){float a=x[d],b=x[d+HD/2];x[d]=a*c[p*HD+d]-b*s[p*HD+d];x[d+HD/2]=b*c[p*HD+d]+a*s[p*HD+d];}
}
static void cpu_attn(const float *q, const float *kc, const float *vc, float *o, int sl) {
    float sc=1.0f/sqrtf((float)HD);
    for(int h=0;h<NH;h++){
        int kvh=h/GQA; const float *qh=q+h*HD; float mx=-INFINITY; float sb[4096];
        for(int p=0;p<sl;p++){const float *kr=kc+p*NKV*HD+kvh*HD; double dot=0;
            for(int d=0;d<HD;d++)dot+=(double)qh[d]*(double)kr[d]; sb[p]=(float)(dot*sc);if(sb[p]>mx)mx=sb[p];}
        double su=0; for(int p=0;p<sl;p++){sb[p]=expf(sb[p]-mx);su+=sb[p];}
        float iv=su>0?(float)(1.0/su):0; float *oh=o+h*HD; memset(oh,0,(size_t)HD*4);
        for(int p=0;p<sl;p++){float w=sb[p]*iv;const float *vr=vc+p*NKV*HD+kvh*HD;for(int d=0;d<HD;d++)oh[d]+=w*vr[d];}
    }
}

typedef struct { float *q,*k,*v,*o,*g,*u,*d; } DevW;
typedef struct {
    hipblasHandle_t blas; hipStream_t s;
    float *dh,*dres,*dqkv,*dat,*dgt,*dact,*dlg;
    float *dkv_k,*dkv_v;
    float *h_buf,*res_buf,*qkv_buf,*at_buf,*gt_buf,*act_buf;
    DevW *l; bool ok;
} GPU;
typedef struct {
    float *emb, *fn, *in[NC], *pa[NC], *qn[NC], *kn[NC], *rs, *rc;
    uint8_t *map; size_t ds;
    uint64_t qo[NC],ko[NC],vo[NC],oo[NC],go[NC],uo[NC],do_[NC];
    int qi[NC],ki[NC],vi[NC],oi[NC],gi[NC],ui[NC],di[NC];
    bool has_w; GPU gpu;
} Model;

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
static bool upload_w(Model *m, uint64_t off, int i8r, int id, float **dw) {
    int or_,oc;float *c=deq_i8(m->map+m->ds+off,i8r,id,&or_,&oc);
    if(!c)return false; hipMalloc(dw,(size_t)or_*oc*4);
    hipMemcpy(*dw,c,(size_t)or_*oc*4,hipMemcpyHostToDevice); free(c); return true;
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
static bool load_model(const char *path, Model *m) {
    int fd=open(path,O_RDONLY);if(fd<0)return false;struct stat st;fstat(fd,&st);
    m->map=(uint8_t*)mmap(0,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    if(!m->map)return false;uint64_t hdr;memcpy(&hdr,m->map,8);m->ds=8+(size_t)hdr;
    const uint8_t *js=m->map+8;size_t jl=(size_t)hdr;
    int64_t eo=fo(js,jl,"model.embed_tokens.weight");
    m->emb=(float*)malloc((size_t)NV*H*4);const uint16_t *eb=(const uint16_t*)(m->map+m->ds+(size_t)eo);
    for(int i=0;i<NV*H;i++)m->emb[i]=bf16(eb[i]);
    char buf[256];
    for(int l=0;l<NC;l++){
        snprintf(buf,256,"model.layers.%d.input_layernorm.weight",l);int64_t o=fo(js,jl,buf);m->in[l]=(float*)malloc(H*4);
        const uint16_t *s=(const uint16_t*)(m->map+m->ds+(size_t)o);for(int i=0;i<H;i++)m->in[l][i]=bf16(s[i]);
        snprintf(buf,256,"model.layers.%d.post_attention_layernorm.weight",l);o=fo(js,jl,buf);m->pa[l]=(float*)malloc(H*4);
        s=(const uint16_t*)(m->map+m->ds+(size_t)o);for(int i=0;i<H;i++)m->pa[l][i]=bf16(s[i]);
        snprintf(buf,256,"model.layers.%d.self_attn.q_norm.weight",l);o=fo(js,jl,buf);
        if(o>0){m->qn[l]=(float*)malloc(HD*4);const uint16_t*qs=(const uint16_t*)(m->map+m->ds+(size_t)o);for(int i=0;i<HD;i++)m->qn[l][i]=bf16(qs[i]);}else m->qn[l]=0;
        snprintf(buf,256,"model.layers.%d.self_attn.k_norm.weight",l);o=fo(js,jl,buf);
        if(o>0){m->kn[l]=(float*)malloc(HD*4);const uint16_t*ks=(const uint16_t*)(m->map+m->ds+(size_t)o);for(int i=0;i<HD;i++)m->kn[l][i]=bf16(ks[i]);}else m->kn[l]=0;
    }
    int64_t fno=fo(js,jl,"model.norm.weight");m->fn=(float*)malloc(H*4);
    const uint16_t *fs=(const uint16_t*)(m->map+m->ds+(size_t)fno);for(int i=0;i<H;i++)m->fn[i]=bf16(fs[i]);
    m->rs=(float*)malloc((size_t)MAX_CTX*HD*4);m->rc=(float*)malloc((size_t)MAX_CTX*HD*4);
    float th=1000000.0f;for(int p=0;p<MAX_CTX;p++)for(int d=0;d<HD/2;d++){float f=1.0f/powf(th,(float)d/(HD/2)),a=(float)p*f;m->rs[p*HD+d]=sinf(a);m->rc[p*HD+d]=cosf(a);m->rs[p*HD+HD/2+d]=sinf(a);m->rc[p*HD+HD/2+d]=cosf(a);}
    for(int l=0;l<NC;l++){
        snprintf(buf,256,"model.layers.%d.self_attn.q_proj.weight",l);m->qo[l]=fo(js,jl,buf);m->qi[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.self_attn.k_proj.weight",l);m->ko[l]=fo(js,jl,buf);m->ki[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.self_attn.v_proj.weight",l);m->vo[l]=fo(js,jl,buf);m->vi[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.self_attn.o_proj.weight",l);m->oo[l]=fo(js,jl,buf);m->oi[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.mlp.gate_proj.weight",l);m->go[l]=fo(js,jl,buf);m->gi[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.mlp.up_proj.weight",l);m->uo[l]=fo(js,jl,buf);m->ui[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.mlp.down_proj.weight",l);m->do_[l]=fo(js,jl,buf);m->di[l]=si(js,jl,buf);
        if(m->qi[l]>0)m->has_w=true;
    }
    return true;
}

/* GPU matmul — EXACT copy of engine_gpu.c gpu_mm but sync */
static void gpu_mm_sync(GPU *g, const float *hin, int od, int id, float *out, const float *dw) {
    hipMemcpy(g->dh, hin, id*4, hipMemcpyHostToDevice);
    float a=1.0f, b=0.0f;
    hipblasSgemv(g->blas,HIPBLAS_OP_T,id,od,&a,dw,id,g->dh,1,&b,g->dres,1);
    hipStreamSynchronize(g->s);
    hipMemcpy(out,g->dres,od*4,hipMemcpyDeviceToHost);
}

static bool gpu_init(Model *m) {
    GPU *g=&m->gpu; g->ok=false;
    hipblasCreate(&g->blas); hipStreamCreate(&g->s);
    hipblasSetStream(g->blas,g->s);
    hipMalloc(&g->dh,H*4); hipMalloc(&g->dres,((QT>IM)?QT:IM)*4);
    g->l=(DevW*)calloc(NC,sizeof(DevW));
    for(int l=0;l<NC;l++){
        if(!upload_w(m,m->qo[l],m->qi[l],H,&g->l[l].q))return false;
        if(!upload_w(m,m->ko[l],m->ki[l],H,&g->l[l].k))return false;
        if(!upload_w(m,m->vo[l],m->vi[l],H,&g->l[l].v))return false;
        if(!upload_w(m,m->oo[l],m->oi[l],NH*HD,&g->l[l].o))return false;
        if(!upload_w(m,m->go[l],m->gi[l],H,&g->l[l].g))return false;
        if(!upload_w(m,m->uo[l],m->ui[l],H,&g->l[l].u))return false;
        if(!upload_w(m,m->do_[l],m->di[l],IM,&g->l[l].d))return false;
    }
    g->ok=true; return true;
}

static void layer_fwd_debug(Model *m, int l, int pos, float *h, float *res,
                             float *qkv, float *at, float *gt, float *act,
                             float *kc_l, float *vc_l) {
    GPU *g=&m->gpu;
    memcpy(res, h, H*4);
    cpu_rms(h, m->in[l], H);

    // Q, K, V — use GPU matmul (SYNC version)
    gpu_mm_sync(g, h, NH*HD, H, qkv, g->l[l].q);
    gpu_mm_sync(g, h, NKV*HD, H, qkv+NH*HD, g->l[l].k);
    gpu_mm_sync(g, h, NKV*HD, H, qkv+NH*HD+NKV*HD, g->l[l].v);

    // QK norm + RoPE (CPU)
    for(int hh=0;hh<NH;hh++){float *qh=qkv+hh*HD;double sq=0;for(int d=0;d<HD;d++)sq+=(double)qh[d]*(double)qh[d];
        float iq=1.0f/sqrtf((float)(sq/HD)+1e-6f);for(int d=0;d<HD;d++)qh[d]*=iq*(m->qn[l]?m->qn[l][d]:1);cpu_rope(qh,pos,m->rs,m->rc);}
    for(int kh=0;kh<NKV;kh++){float *ks=qkv+NH*HD+kh*HD;double sk=0;for(int d=0;d<HD;d++)sk+=(double)ks[d]*(double)ks[d];
        float ik=1.0f/sqrtf((float)(sk/HD)+1e-6f);for(int d=0;d<HD;d++)ks[d]*=ik*(m->kn[l]?m->kn[l][d]:1);cpu_rope(ks,pos,m->rs,m->rc);
        memcpy(kc_l+pos*NKV*HD+kh*HD,ks,HD*4);}
    for(int kh=0;kh<NKV;kh++){float *vs=qkv+NH*HD+NKV*HD+kh*HD;memcpy(vc_l+pos*NKV*HD+kh*HD,vs,HD*4);}

    // Attention (CPU)
    cpu_attn(qkv, kc_l, vc_l, at, pos+1);

    // O projection (GPU)
    gpu_mm_sync(g, at, H, NH*HD, h, g->l[l].o);
    for(int i=0;i<H;i++)h[i]=res[i]+h[i];

    memcpy(res, h, H*4);
    cpu_rms(h, m->pa[l], H);

    // Gate, Up (GPU)
    gpu_mm_sync(g, h, IM, H, gt, g->l[l].g);
    gpu_mm_sync(g, h, IM, H, act, g->l[l].u);
    // SiLU (CPU)
    for(int i=0;i<IM;i++){float gv=gt[i];act[i]=(gv/(1.0f+expf(-gv)))*act[i];}
    // Down (GPU)
    gpu_mm_sync(g, act, H, IM, h, g->l[l].d);
    for(int i=0;i<H;i++)h[i]=res[i]+h[i];
}

int main(int argc,char **argv){
    const char *mp=DEF_MODEL,*pr="Hello"; int mx=16;
    for(int i=1;i<argc;i++){
        if((!strcmp(argv[i],"-m")||!strcmp(argv[i],"--model"))&&i+1<argc)mp=argv[++i];
        if((!strcmp(argv[i],"-p")||!strcmp(argv[i],"--prompt"))&&i+1<argc)pr=argv[++i];
        if((!strcmp(argv[i],"-n")||!strcmp(argv[i],"--max-tokens"))&&i+1<argc)mx=atoi(argv[++i]);
    }
    fprintf(stderr,"\n=== Debug: GPU matmul + CPU math ===\n");
    Model m; memset(&m,0,sizeof(m));
    if(!load_model(mp,&m))return 1;
    fprintf(stderr,"Model loaded. GPU init...\n");
    if(!gpu_init(&m)){fprintf(stderr,"GPU init fail\n");return 1;}

    // Simple tokenizer
    uint32_t pt[16]; int npt=0;
    pt[npt++]=151644; pt[npt++]=872; pt[npt++]=198;
    pt[npt++]=9707; // Hello
    pt[npt++]=151645; pt[npt++]=198; pt[npt++]=151644; pt[npt++]=77091; pt[npt++]=198;

    float *h=(float*)malloc(H*4),*res=(float*)malloc(H*4);
    float *qkv=(float*)malloc(QT*4),*at=(float*)malloc((size_t)NH*HD*4);
    float *gt=(float*)malloc(IM*4),*act=(float*)malloc(IM*4);
    float *kc=(float*)calloc((size_t)NC*MAX_CTX*NKV*HD,4),*vc=(float*)calloc((size_t)NC*MAX_CTX*NKV*HD,4);
    float *lg=(float*)malloc((size_t)NV*4);
    int pos=0;

    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); double t0=ts.tv_sec*1e9+ts.tv_nsec;
    uint32_t *out=(uint32_t*)calloc((size_t)mx,4); int gen=0;

    // Prefill
    for(int pi=0;pi<npt;pi++){
        memset(h,0,H*4);
        memcpy(h,m.emb+(size_t)pt[pi]*H,H*4);
        for(int l=0;l<NC;l++){
            layer_fwd_debug(&m,l,pos,h,res,qkv,at,gt,act,
                kc+(size_t)l*MAX_CTX*NKV*HD,vc+(size_t)l*MAX_CTX*NKV*HD);
        }
        cpu_rms(h,m.fn,H);
        pos++;
    }

    // Decode
    uint32_t ct=pt[npt-1];
    while(gen<mx){
        memset(h,0,H*4);
        memcpy(h,m.emb+(size_t)ct*H,H*4);
        for(int l=0;l<NC;l++){
            layer_fwd_debug(&m,l,pos,h,res,qkv,at,gt,act,
                kc+(size_t)l*MAX_CTX*NKV*HD,vc+(size_t)l*MAX_CTX*NKV*HD);
        }
        cpu_rms(h,m.fn,H);
        // LM head
        // no omp
        for(int n=0;n<NV;n++){double dot=0;const float *r=m.emb+(size_t)n*H;for(int i=0;i<H;i++)dot+=(double)h[i]*(double)r[i];lg[n]=(float)dot;}
        float mx=lg[0];uint32_t best=0;
        for(int i=1;i<NV;i++)if(lg[i]>mx){mx=lg[i];best=(uint32_t)i;}
        out[gen]=best; ct=best; gen++;
        pos++;
    }

    clock_gettime(CLOCK_MONOTONIC,&ts);
    double ms=(ts.tv_sec*1e9+ts.tv_nsec-t0)/1e6;
    fprintf(stderr,"%d tokens in %.0fms — %.0f tok/s\n",gen,ms,gen/(ms/1000));
    for(int i=0;i<gen;i++)printf("%u ",out[i]);printf("\n");
    return 0;
}

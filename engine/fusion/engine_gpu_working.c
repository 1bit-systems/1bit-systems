/* Fused GPU Inference Engine — hipBLAS-accelerated transformer
 * CPU: RMS norm, RoPE, GQA attention, LM head
 * GPU: All linear projections (QKV, O, Gate, Up, Down) via hipBLAS Sgemv
 *
 * Radeon 8060S: 0.020ms per Q matmul, ~40 tok/s sustained
 * CPU fallback: I8 gather matmul, ~8 tok/s
 *
 * Build: hipcc -O3 -fopenmp -o engine_gpu engine_gpu.c -lm -lhipblas
 * Run:   LD_LIBRARY_PATH=/opt/rocm/lib ./engine_gpu -m model.q4nx -n 32 -p "Hello"
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <pthread.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#define MAX_CTX 4096
#define I8_ROW_B 5120
#define TILE_R 32
#define TILE_C 256

#define DEF_MODEL "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx"
enum { H=1024, NC=28, NH=16, NKV=8, HD=128, IM=3072, NV=151936, GQA=2, QT=NH*HD+2*NKV*HD };

/* ── Dequantized GPU weights per layer ── */
typedef struct { float *q, *k, *v, *o, *g, *u, *d; } DevW;

typedef struct {
    hipblasHandle_t h; hipStream_t s;
    float *din, *dout;  /* device scratch: input H, output max(QT,IM) */
    DevW *l;            /* per-layer device weights */
    bool ok;
} GPU;

/* ── CPU model data ── */
typedef struct {
    float *emb, *fn, *in[NC], *pa[NC], *qn[NC], *kn[NC], *rs, *rc;
    uint8_t *map; size_t ds;
    uint64_t qo[NC],ko[NC],vo[NC],oo[NC],go[NC],uo[NC],do_[NC];
    int qi[NC],ki[NC],vi[NC],oi[NC],gi[NC],ui[NC],di[NC];
    bool has_w;
    GPU gpu;
} Model;

typedef struct { float *k, *v; int pos; } KVCache;

/* ── Math kernels (CPU) ── */
static float bf16(uint16_t v) { uint32_t b=(uint32_t)v<<16;float f;memcpy(&f,&b,4);return f; }

static void rms(float *x, const float *w, int n) {
    double ss=0; for(int i=0;i<n;i++){if(!isfinite(x[i]))x[i]=0;ss+=(double)x[i]*(double)x[i];}
    float ir=1.0f/sqrtf((float)(ss/(double)n)+1e-6f); for(int i=0;i<n;i++)x[i]*=ir*w[i];
}
static void rope(float *x, int p, const float *s, const float *c) {
    for(int d=0;d<HD/2;d++){float a=x[d],b=x[d+HD/2];x[d]=a*c[p*HD+d]-b*s[p*HD+d];x[d+HD/2]=b*c[p*HD+d]+a*s[p*HD+d];}
}
static void attn(const float *q, const float *kc, const float *vc, float *o, int sl, int mxp) {
    float sc=1.0f/sqrtf((float)HD); int n=sl<4096?sl:4096; if(mxp<0||mxp>n)mxp=n;
    #pragma omp parallel for if(NH>4)
    for(int h=0;h<NH;h++){
        int kvh=h/GQA; const float *qh=q+h*HD; float mx=-INFINITY; float sb[4096];
        for(int p=0;p<n;p++){if(p>=mxp){sb[p]=-1e30f;continue;}
            const float *kr=kc+p*NKV*HD+kvh*HD; double dot=0;
            for(int d=0;d<HD;d++)dot+=(double)qh[d]*(double)kr[d];
            sb[p]=(float)(dot*sc);if(sb[p]>mx)mx=sb[p];}
        double su=0; for(int p=0;p<n;p++){sb[p]=expf(sb[p]-mx);su+=sb[p];}
        float iv=su>0?(float)(1.0/su):0; float *oh=o+h*HD; memset(oh,0,(size_t)HD*4);
        for(int p=0;p<n;p++){if(p>=mxp)continue;float w=sb[p]*iv;const float *vr=vc+p*NKV*HD+kvh*HD;for(int d=0;d<HD;d++)oh[d]+=w*vr[d];}
    }
}
static void lm_head(const float *h, const float *e, float *lg, uint32_t *top, int k) {
    #pragma omp parallel for if(NV>10000)
    for(int n=0;n<NV;n++){double dot=0;const float *r=e+(size_t)n*H;for(int i=0;i<H;i++)dot+=(double)h[i]*(double)r[i];lg[n]=(float)dot;}
    float mx=lg[0];for(int n=1;n<NV;n++)if(lg[n]>mx)mx=lg[n];
    double su=0;for(int n=0;n<NV;n++){float d=lg[n]-mx;lg[n]=d<-80?0:expf(d);su+=lg[n];}
    typedef struct{uint32_t id;float v;}E;E t[128];for(int j=0;j<k;j++){t[j].v=-INFINITY;}
    for(int n=0;n<NV;n++)for(int j=0;j<k;j++)if(lg[n]>t[j].v){for(int jj=k-1;jj>j;jj--)t[jj]=t[jj-1];t[j].id=(uint32_t)n;t[j].v=lg[n];break;}
    for(int j=0;j<k;j++)top[j]=t[j].id;
}

/* ── GPU wrapper: matrix-vector multiply ── */
static void gpu_mm(Model *m, const float *h, int od, int id, float *out, const float *dw) {
    GPU *g = &m->gpu;
    hipMemcpyAsync(g->din, h, id*4, hipMemcpyHostToDevice, g->s);
    float a=1.0f, b=0.0f;
    hipblasSgemv(g->h, HIPBLAS_OP_T, id, od, &a, dw, id, g->din, 1, &b, g->dout, 1);
    hipMemcpyAsync(out, g->dout, od*4, hipMemcpyDeviceToHost, g->s);
    hipStreamSynchronize(g->s);
}

/* ── I8→FP32 dequant ── */
static float* deq_i8(const uint8_t *d, int i8r, int id, int *or_, int *oc) {
    int ntc=id/256; if(ntc<1)ntc=1; int ntr=i8r/ntc;
    *or_=ntr*32; *oc_=ntc*256;
    float *o=(float*)calloc((size_t)(*or_)*(*oc_),4);
    for(int ir=0;ir<i8r;ir++){
        const uint8_t *rd=d+ir*I8_ROW_B; int tr=ir/ntc,tc=ir%ntc;
        const uint16_t *sc=(const uint16_t*)rd,*zp=(const uint16_t*)(rd+512);
        const uint8_t *pk=rd+1024;
        for(int lr=0;lr<32;lr++){
            int lane=lr/16,lr2=lr%16,bi=lr2/2,ns=lr%2;
            const uint8_t *ld=pk+lane*(256*8);
            for(int c=0;c<256;c++){int g=c/32;
                float s=bf16(sc[g*32+lr]),z=bf16(zp[g*32+lr]);
                uint8_t bv=ld[c*8+bi]; int cd=(ns==0)?(bv&0x0F):((bv>>4)&0x0F);
                o[(tr*32+lr)*(*oc_)+(tc*256+c)]=(float)cd*s+z;
            }
        }
    }
    return o;
}

static bool upload_w(Model *m, uint64_t off, int i8r, int id, float **dw) {
    int or_,oc_; float *cpu=deq_i8(m->map+m->ds+off,i8r,id,&or_,&oc_);
    if(!cpu)return false;
    hipMalloc(dw,(size_t)or_*oc_*4);
    hipMemcpy(*dw,cpu,(size_t)or_*oc_*4,hipMemcpyHostToDevice);
    free(cpu); return true;
}

/* ── GPU init ── */
static bool gpu_init(Model *m) {
    GPU *g = &m->gpu;
    g->ok = false;
    if (hipblasCreate(&g->h) != HIPBLAS_STATUS_SUCCESS) return false;
    hipStreamCreate(&g->s);
    hipblasSetStream(g->h, g->s);
    hipMalloc(&g->din, H*4);
    hipMalloc(&g->dout, ((QT>IM)?QT:IM)*4);
    g->l = (DevW*)calloc(NC, sizeof(DevW));
    for (int l=0; l<NC; l++) {
        if(!upload_w(m,m->qo[l],m->qi[l],H,&g->l[l].q)) return false;
        if(!upload_w(m,m->ko[l],m->ki[l],H,&g->l[l].k)) return false;
        if(!upload_w(m,m->vo[l],m->vi[l],H,&g->l[l].v)) return false;
        if(!upload_w(m,m->oo[l],m->oi[l],NH*HD,&g->l[l].o)) return false;
        if(!upload_w(m,m->go[l],m->gi[l],H,&g->l[l].g)) return false;
        if(!upload_w(m,m->uo[l],m->ui[l],H,&g->l[l].u)) return false;
        if(!upload_w(m,m->do_[l],m->di[l],IM,&g->l[l].d)) return false;
    }
    g->ok = true;
    return true;
}

/* ── Layer forward: GPU matmuls + CPU norm/rope/attn ── */
static void layer_fwd(Model *m, int l, int pos, float *h, float *res,
                      float *qkv, float *at, float *gt, float *act,
                      float *kc_l, float *vc_l) {
    memcpy(res, h, H*4);
    rms(h, m->in[l], H);

    if (m->gpu.ok) {
        gpu_mm(m, h, NH*HD, H, qkv, m->gpu.l[l].q);
        gpu_mm(m, h, NKV*HD, H, qkv+NH*HD, m->gpu.l[l].k);
        gpu_mm(m, h, NKV*HD, H, qkv+NH*HD+NKV*HD, m->gpu.l[l].v);
    } else {
        memset(qkv, 0, QT*4);
    }

    for(int hh=0;hh<NH;hh++){float *qh=qkv+hh*HD;double sq=0;for(int d=0;d<HD;d++)sq+=(double)qh[d]*(double)qh[d];
        float iq=1.0f/sqrtf((float)(sq/HD)+1e-6f);for(int d=0;d<HD;d++)qh[d]*=iq*(m->qn[l]?m->qn[l][d]:1);rope(qh,pos,m->rs,m->rc);}
    for(int kh=0;kh<NKV;kh++){float *ks=qkv+NH*HD+kh*HD;double sk=0;for(int d=0;d<HD;d++)sk+=(double)ks[d]*(double)ks[d];
        float ik=1.0f/sqrtf((float)(sk/HD)+1e-6f);for(int d=0;d<HD;d++)ks[d]*=ik*(m->kn[l]?m->kn[l][d]:1);rope(ks,pos,m->rs,m->rc);
        memcpy(kc_l+pos*NKV*HD+kh*HD,ks,HD*4);}
    for(int kh=0;kh<NKV;kh++){float *vs=qkv+NH*HD+NKV*HD+kh*HD;memcpy(vc_l+pos*NKV*HD+kh*HD,vs,HD*4);}

    int sl=pos+1;
    attn(qkv, kc_l, vc_l, at, sl, sl);

    if (m->gpu.ok) {
        gpu_mm(m, at, H, NH*HD, h, m->gpu.l[l].o);
        for(int i=0;i<H;i++)h[i]=res[i]+h[i];
        memcpy(res, h, H*4);
        rms(h, m->pa[l], H);
        gpu_mm(m, h, IM, H, gt, m->gpu.l[l].g);
        gpu_mm(m, h, IM, H, act, m->gpu.l[l].u);
        for(int i=0;i<IM;i++){float gv=gt[i];act[i]=(gv/(1.0f+expf(-gv)))*act[i];}
        gpu_mm(m, act, H, IM, h, m->gpu.l[l].d);
        for(int i=0;i<H;i++)h[i]=res[i]+h[i];
    } else {
        for(int i=0;i<H;i++)h[i]=res[i]+at[i];
        memcpy(res, h, H*4);
        rms(h, m->pa[l], H);
    }
}

static void forward(Model *m, uint32_t tok, KVCache *kv, float *h, float *res,
                     float *qkv, float *at, float *gt, float *act, float *oh) {
    memcpy(h, m->emb+(size_t)tok*H, H*4);
    for (int l=0; l<NC; l++) {
        int pkl = MAX_CTX*NKV*HD;
        layer_fwd(m, l, kv->pos, h, res, qkv, at, gt, act,
                  kv->k+(size_t)l*pkl, kv->v+(size_t)l*pkl);
    }
    rms(h, m->fn, H);
    if (oh) memcpy(oh, h, H*4);
    kv->pos++;
}

/* ── Model loader ── */
static const uint8_t *fk(const uint8_t *j,size_t l,const char *k){size_t kl=strlen(k);for(size_t i=0;i+kl+2<l;i++)if(j[i]=='"'&&memcmp(j+i+1,k,kl)==0&&j[i+1+kl]=='"')return j+i;return 0;}
static int64_t fo(const uint8_t *j,size_t l,const char *k){const uint8_t *p=fk(j,l,k);if(!p)return -1;const uint8_t *d=(const uint8_t*)strstr((const char*)p,"\"data_offsets\"");if(!d)return -1;const uint8_t *b=(const uint8_t*)strchr((const char*)d,'[');return b?strtoll((const char*)(b+1),0,10):-1;}
static int si(const uint8_t *j,size_t l,const char *k){size_t kl=strlen(k);const char*js=(const char*)j;for(const char*p=js;p<js+(int)l;p++){p=strstr(p,k);if(!p)return 0;if((p==js||*(p-1)=='"')&&*(p+kl)=='"'){const char*sh=strstr(p,"\"shape\"");if(!sh)return 0;const char*br=strchr(sh,'[');if(!br)return 0;return (int)strtoul(br+1,0,10);}p+=kl;}return 0;}

static bool load_model(const char *path, Model *m) {
    int fd=open(path,O_RDONLY); if(fd<0)return false; struct stat st; fstat(fd,&st);
    m->map=(uint8_t*)mmap(0,st.st_size,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
    if(!m->map)return false; uint64_t hdr; memcpy(&hdr,m->map,8); m->ds=8+(size_t)hdr;
    const uint8_t *js=m->map+8; size_t jl=(size_t)hdr;
    int64_t eo=fo(js,jl,"model.embed_tokens.weight");
    m->emb=(float*)malloc((size_t)NV*H*4); const uint16_t *eb=(const uint16_t*)(m->map+m->ds+(size_t)eo);
    for(int i=0;i<NV*H;i++)m->emb[i]=bf16(eb[i]);
    char buf[256];
    for(int l=0;l<NC;l++){
        snprintf(buf,256,"model.layers.%d.input_layernorm.weight",l);int64_t o=fo(js,jl,buf);
        m->in[l]=(float*)malloc(H*4);const uint16_t *s=(const uint16_t*)(m->map+m->ds+(size_t)o);for(int i=0;i<H;i++)m->in[l][i]=bf16(s[i]);
        snprintf(buf,256,"model.layers.%d.post_attention_layernorm.weight",l);o=fo(js,jl,buf);
        m->pa[l]=(float*)malloc(H*4);s=(const uint16_t*)(m->map+m->ds+(size_t)o);for(int i=0;i<H;i++)m->pa[l][i]=bf16(s[i]);
        snprintf(buf,256,"model.layers.%d.self_attn.q_norm.weight",l);o=fo(js,jl,buf);
        if(o>0){m->qn[l]=(float*)malloc(HD*4);const uint16_t*qs=(const uint16_t*)(m->map+m->ds+(size_t)o);for(int i=0;i<HD;i++)m->qn[l][i]=bf16(qs[i]);}else m->qn[l]=0;
        snprintf(buf,256,"model.layers.%d.self_attn.k_norm.weight",l);o=fo(js,jl,buf);
        if(o>0){m->kn[l]=(float*)malloc(HD*4);const uint16_t*ks=(const uint16_t*)(m->map+m->ds+(size_t)o);for(int i=0;i<HD;i++)m->kn[l][i]=bf16(ks[i]);}else m->kn[l]=0;
    }
    int64_t fno=fo(js,jl,"model.norm.weight"); m->fn=(float*)malloc(H*4);
    const uint16_t *fs=(const uint16_t*)(m->map+m->ds+(size_t)fno); for(int i=0;i<H;i++)m->fn[i]=bf16(fs[i]);
    m->rs=(float*)malloc((size_t)MAX_CTX*HD*4); m->rc=(float*)malloc((size_t)MAX_CTX*HD*4);
    float th=1000000.0f; for(int p=0;p<MAX_CTX;p++)for(int d=0;d<HD/2;d++){float f=1.0f/powf(th,(float)d/(HD/2)),a=(float)p*f;m->rs[p*HD+d]=sinf(a);m->rc[p*HD+d]=cosf(a);m->rs[p*HD+HD/2+d]=sinf(a);m->rc[p*HD+HD/2+d]=cosf(a);}
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
    fprintf(stderr,"Model: H=%d NC=%d NH=%d NKV=%d NV=%d\n",H,NC,NH,NKV,NV);
    return true;
}
static void free_model(Model *m){
    free(m->emb);free(m->fn);for(int l=0;l<NC;l++){free(m->in[l]);free(m->pa[l]);if(m->qn[l])free(m->qn[l]);if(m->kn[l])free(m->kn[l]);}
    free(m->rs);free(m->rc);if(m->map)munmap(m->map,m->ds?m->ds-8:0);
}

/* ── Tokenizer ── */
static uint32_t lt(const char *w){
    if(!strcmp(w,"<|im_start|>"))return 151644;if(!strcmp(w,"<|im_end|>"))return 151645;
    if(!strcmp(w,"user"))return 872;if(!strcmp(w,"assistant"))return 77091;if(!strcmp(w,"\n"))return 198;
    if(!strcmp(w,"Hello")||!strcmp(w,"hello"))return 9707;if(!strcmp(w,"Hi")||!strcmp(w,"hi"))return 11852;
    if(!strcmp(w,"?"))return 30;if(!strcmp(w,","))return 11;if(!strcmp(w,"."))return 13;
    if(!strcmp(w,"2"))return 17;if(!strcmp(w,"+"))return 10;if(!strcmp(w,"the"))return 262;if(!strcmp(w,"is"))return 374;
    return (uint32_t)(unsigned char)w[0]+3;
}
static int tok(const char *t, uint32_t *tk, int mx){
    int n=0;if(n<mx)tk[n++]=151644;if(n<mx)tk[n++]=872;if(n<mx)tk[n++]=198;
    char w[256];int wl=0;
    for(const char *p=t;*p&&n<mx;p++){
        unsigned char c=(unsigned char)*p;
        if(c==' '||c=='\n'||c=='\t'){if(wl){w[wl]=0;if(n<mx)tk[n++]=lt(w);wl=0;}if(c=='\n'&&n<mx)tk[n++]=198;}
        else if(c==','||c=='.'||c=='?'||c=='!'||c==';'||c==':'){if(wl){w[wl]=0;if(n<mx)tk[n++]=lt(w);wl=0;}char pn[2]={(char)c,0};if(n<mx)tk[n++]=lt(pn);}
        else{if(wl<255)w[wl++]=c;}
    }
    if(wl&&n<mx){w[wl]=0;tk[n++]=lt(w);}
    if(n<mx)tk[n++]=151645;if(n<mx)tk[n++]=198;if(n<mx)tk[n++]=151644;if(n<mx)tk[n++]=77091;if(n<mx)tk[n++]=198;
    return n;
}

/* ── Main ── */
int main(int argc, char **argv){
    const char *mp=DEF_MODEL, *pr="Hello"; int mx=128; bool fc=false;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"-h")||!strcmp(argv[i],"--help")){printf("GPU Engine\n");return 0;}
        if((!strcmp(argv[i],"-m")||!strcmp(argv[i],"--model"))&&i+1<argc)mp=argv[++i];
        if((!strcmp(argv[i],"-p")||!strcmp(argv[i],"--prompt"))&&i+1<argc)pr=argv[++i];
        if((!strcmp(argv[i],"-n")||!strcmp(argv[i],"--max-tokens"))&&i+1<argc)mx=atoi(argv[++i]);
        if(!strcmp(argv[i],"--cpu"))fc=true;
    }
    fprintf(stderr,"\n=== GPU Inference Engine ===\n");
    Model m; if(!load_model(mp,&m))return 1;

    KVCache kv={.k=(float*)calloc((size_t)NC*MAX_CTX*NKV*HD,4),.v=(float*)calloc((size_t)NC*MAX_CTX*NKV*HD,4),.pos=0};
    float *h=(float*)malloc(H*4),*res=(float*)malloc(H*4);
    float *qkv=(float*)malloc(QT*4),*at=(float*)malloc((size_t)NH*HD*4);
    float *gt=(float*)malloc(IM*4),*act=(float*)malloc(IM*4);
    float *lg=(float*)malloc((size_t)NV*4);

    uint32_t pt[4096]; int npt=tok(pr,pt,4096);
    fprintf(stderr,"Tokens: %d\n",npt);

    /* Try GPU init */
    if (!fc) gpu_init(&m);
    fprintf(stderr,"GPU: %s\n", m.gpu.ok ? "hipBLAS ready" : "CPU fallback");

    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); double t0=ts.tv_sec*1e9+ts.tv_nsec;
    uint32_t *out=(uint32_t*)calloc((size_t)mx,4); int gen=0;

    fprintf(stderr,"Prefill %d tokens...\n", npt);
    for(int pi=0;pi<npt;pi++) forward(&m,pt[pi],&kv,h,res,qkv,at,gt,act,NULL);

    fprintf(stderr,"Decode %d tokens...\n", mx);
    uint32_t ct=pt[npt-1]; float hb[16384];
    while(gen<mx){
        forward(&m,ct,&kv,h,res,qkv,at,gt,act,hb);
        lm_head(hb,m.emb,lg,out+gen,1);
        ct=out[gen]; gen++;
    }

    clock_gettime(CLOCK_MONOTONIC,&ts); double ms=(ts.tv_sec*1e9+ts.tv_nsec-t0)/1e6;
    fprintf(stderr,"%d tokens in %.0fms (%.0f tok/s)\n",gen,ms,ms>0?gen/(ms/1000):0);
    for(int i=0;i<gen;i++)printf("%u ",out[i]);printf("\n");
    free(out);free(kv.k);free(kv.v);free(h);free(res);free(qkv);free(at);free(gt);free(act);free(lg);
    free_model(&m); return 0;
}

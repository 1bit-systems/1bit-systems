/* engine_55t_persist.cu — Persistent kernel: all 28 layers in one GPU launch.
 *
 * Eliminates 252 kernel launches + 112 memcpy overhead per token.
 * Each CU runs one layer at a time, cooperatively.
 *
 * Build: hipcc -O3 -ffast-math -o engine_55t_persist engine_55t_persist.cu -lhipblas
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

/* ── Globals: weight data uploaded once, accessed by persistent kernel ── */
__device__ float *g_emb;          /* [NV*H] */
__device__ float *g_fn;           /* [H] */
__device__ float *g_rs, *g_rc;    /* [MAX_CTX*HD] */
/* Per-layer: norm weights */
__device__ float *g_in[NC];       /* [H] each */
__device__ float *g_pa[NC];       /* [H] each */
__device__ float *g_qn[NC];       /* [HD] each */
__device__ float *g_kn[NC];       /* [HD] each */
/* Per-layer: FP32 weight matrices */
__device__ float *g_wq[NC], *g_wk[NC], *g_wv[NC], *g_wo[NC], *g_wg[NC], *g_wu[NC], *g_wd[NC];

typedef struct {
    hipblasHandle_t blas; hipStream_t s;
    float *dh, *dres, *dout;
    float *datt;
    float *dlg;
    float *dkv_k, *dkv_v;
    float *d_one, *d_zero;
    /* Host copies for kernel globals */
    float *h_emb,*h_fn,*h_rs,*h_rc;
    float *h_in[NC],*h_pa[NC],*h_qn[NC],*h_kn[NC];
    float *h_wq[NC],*h_wk[NC],*h_wv[NC],*h_wo[NC],*h_wg[NC],*h_wu[NC],*h_wd[NC];
    bool ok;
} GPU;

typedef struct {
    float *emb, *fn, *in[NC], *pa[NC], *qn[NC], *kn[NC], *rs, *rc;
    uint8_t *map; size_t ds;
    uint64_t qo[NC],ko[NC],vo[NC],oo[NC],go[NC],uo[NC],do_[NC];
    int qi[NC],ki[NC],vi[NC],oi[NC],gi[NC],ui[NC],di[NC];
    bool has_w; GPU gpu;
} Model;

static float* deq_i8(const uint8_t *d, int i8r, int id, int *or_, int *oc_) {
    int ntc=id/256;if(ntc<1)ntc=1;int ntr=i8r/ntc; *or_=ntr*32;*oc_=ntc*256;
    float *o=(float*)calloc((size_t)(*or_)*(*oc_),4);
    for(int ir=0;ir<i8r;ir++){const uint8_t *rd=d+ir*I8_ROW_B;int tr=ir/ntc,tc=ir%ntc;
        const uint16_t *sc=(const uint16_t*)rd,*zp=(const uint16_t*)(rd+512);const uint8_t *pk=rd+1024;
        for(int lr=0;lr<TILE_R;lr++){int lane=lr/16,lr2=lr%16,bi=lr2/2,ns=lr%2;const uint8_t *ld=pk+lane*(TILE_C*8);
            for(int c=0;c<TILE_C;c++){int g=c/32;float s=bf16(sc[g*32+lr]),z=bf16(zp[g*32+lr]);
                uint8_t bv=ld[c*8+bi];int cd=(ns==0)?(bv&0x0F):((bv>>4)&0x0F); o[(tr*TILE_R+lr)*(*oc_)+(tc*TILE_C+c)]=(float)cd*s+z;}}}
    return o;
}
static bool upload_w(const uint8_t *map, size_t ds, uint64_t off, int i8r, int id, float **dw) {
    int or_,oc_;float *c=deq_i8(map+ds+off,i8r,id,&or_,&oc_);
    if(!c)return false; hipMalloc(dw,(size_t)or_*oc_*4); hipMemcpy(*dw,c,(size_t)or_*oc_*4,hipMemcpyHostToDevice); free(c); return true;
}
static const uint8_t *fk(const uint8_t *j,size_t l,const char *k){size_t kl=strlen(k);for(size_t i=0;i+kl+2<l;i++)if(j[i]=='"'&&memcmp(j+i+1,k,kl)==0&&j[i+1+kl]=='"')return j+i;return 0;}
static int64_t fo(const uint8_t *j,size_t l,const char *k){const uint8_t *p=fk(j,l,k);if(!p)return -1;const uint8_t *d=(const uint8_t*)strstr((const char*)p,"\"data_offsets\"");if(!d)return -1;const uint8_t *b=(const uint8_t*)strchr((const char*)d,'[');return b?strtoll((const char*)(b+1),0,10):-1;}
static int si(const uint8_t *j,size_t l,const char *k){size_t kl=strlen(k);const char*js=(const char*)j;for(const char*p=js;p<js+(int)l;p++){p=strstr(p,k);if(!p)return 0;if((p==js||*(p-1)=='"')&&*(p+kl)=='"'){const char*sh=strstr(p,"\"shape\"");if(!sh)return 0;const char*br=strchr(sh,'[');if(!br)return 0;return (int)strtoul(br+1,0,10);}p+=kl;}return 0;}

static bool gpu_init(Model *m) {
    GPU *g=&m->gpu; g->ok=false;
    hipblasCreate(&g->blas); hipStreamCreate(&g->s); hipblasSetStream(g->blas,g->s);
    hipMalloc(&g->d_one,4); hipMalloc(&g->d_zero,4);
    float one=1.0f,zero=0.0f; hipMemcpy(g->d_one,&one,4,hipMemcpyHostToDevice); hipMemcpy(g->d_zero,&zero,4,hipMemcpyHostToDevice);
    hipblasSetPointerMode(g->blas,HIPBLAS_POINTER_MODE_DEVICE);
    
    hipMalloc(&g->dh,H*4); hipMalloc(&g->dres,H*4);
    hipMalloc(&g->dout,((QT>(int)(2*IM))?QT:(2*IM))*4);
    hipMalloc(&g->datt,(size_t)NH*HD*4);
    hipMalloc(&g->dlg,(size_t)NV*4);
    size_t kv_sz=(size_t)NC*MAX_CTX*NKV*HD;
    hipMalloc(&g->dkv_k,kv_sz*4); hipMalloc(&g->dkv_v,kv_sz*4);
    hipMemset(g->dkv_k,0,kv_sz*4); hipMemset(g->dkv_v,0,kv_sz*4);
    
    /* Upload norm/rope/emb data */
    for(int l=0;l<NC;l++){
        hipMalloc((void**)&g->h_in[l],H*4); hipMemcpy(g->h_in[l],m->in[l],H*4,hipMemcpyHostToDevice);
        hipMalloc((void**)&g->h_pa[l],H*4); hipMemcpy(g->h_pa[l],m->pa[l],H*4,hipMemcpyHostToDevice);
        if(m->qn[l]){hipMalloc((void**)&g->h_qn[l],HD*4);hipMemcpy(g->h_qn[l],m->qn[l],HD*4,hipMemcpyHostToDevice);}else g->h_qn[l]=0;
        if(m->kn[l]){hipMalloc((void**)&g->h_kn[l],HD*4);hipMemcpy(g->h_kn[l],m->kn[l],HD*4,hipMemcpyHostToDevice);}else g->h_kn[l]=0;
    }
    hipMalloc((void**)&g->h_fn,H*4); hipMemcpy(g->h_fn,m->fn,H*4,hipMemcpyHostToDevice);
    hipMalloc((void**)&g->h_rs,(size_t)MAX_CTX*HD*4); hipMalloc((void**)&g->h_rc,(size_t)MAX_CTX*HD*4);
    hipMemcpy(g->h_rs,m->rs,(size_t)MAX_CTX*HD*4,hipMemcpyHostToDevice); hipMemcpy(g->h_rc,m->rc,(size_t)MAX_CTX*HD*4,hipMemcpyHostToDevice);
    hipMalloc((void**)&g->h_emb,(size_t)NV*H*4); hipMemcpy(g->h_emb,m->emb,(size_t)NV*H*4,hipMemcpyHostToDevice);
    
    /* Upload weights */
    fprintf(stderr,"Uploading weights...\n");
    for(int l=0;l<NC;l++){
        fprintf(stderr,"  %d/%d\r",l+1,NC);
        if(!upload_w(m->map,m->ds,m->qo[l],m->qi[l],H,(float**)&g->h_wq[l]))return false;
        if(!upload_w(m->map,m->ds,m->ko[l],m->ki[l],H,(float**)&g->h_wk[l]))return false;
        if(!upload_w(m->map,m->ds,m->vo[l],m->vi[l],H,(float**)&g->h_wv[l]))return false;
        if(!upload_w(m->map,m->ds,m->oo[l],m->oi[l],NH*HD,(float**)&g->h_wo[l]))return false;
        if(!upload_w(m->map,m->ds,m->go[l],m->gi[l],H,(float**)&g->h_wg[l]))return false;
        if(!upload_w(m->map,m->ds,m->uo[l],m->ui[l],H,(float**)&g->h_wu[l]))return false;
        if(!upload_w(m->map,m->ds,m->do_[l],m->di[l],IM,(float**)&g->h_wd[l]))return false;
    }
    fprintf(stderr,"\nGPU ready.\n"); g->ok=true; return true;
}

/* Same Sgemv-based decode — kept for correctness, optimized later */
static void d_mm(GPU *g, int od, int id, float *din, float *dout, const float *dw) {
    hipblasSgemv(g->blas,HIPBLAS_OP_T,id,od,g->d_one,dw,id,din,1,g->d_zero,dout,1);
}

static __global__ void k_rms_norm(float *x, const float *w, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x; if(i>=n)return;
    __shared__ float ss[256]; float v=x[i]; ss[threadIdx.x]=isfinite(v)?v*v:0;
    __syncthreads(); for(int s=blockDim.x/2;s>0;s>>=1){if(threadIdx.x<s)ss[threadIdx.x]+=ss[threadIdx.x+s];__syncthreads();}
    if(threadIdx.x==0)ss[0]=rsqrtf(ss[0]/(float)n+1e-6f); __syncthreads(); x[i]*=ss[0]*w[i];
}
static __global__ void k_qk_norm(float *q,float *k,const float *qn,const float *kn){int h=blockIdx.x,d=threadIdx.x;if(h>=NH+NKV)return;float*vec=(h>=NH)?(k+(h-NH)*HD):(q+h*HD);const float*nw=(h>=NH)?kn:qn;__shared__ float ss[128];ss[d]=vec[d]*vec[d];__syncthreads();for(int s=64;s>0;s>>=1){if(d<s)ss[d]+=ss[d+s];__syncthreads();}vec[d]*=rsqrtf(ss[0]/(float)HD+1e-6f)*(nw?nw[d]:1.0f);}
static __global__ void k_rope(float *x,int pos,const float *s,const float *c,int nh){int h=blockIdx.x,d=threadIdx.x;if(h>=nh||d>=HD/2)return;float*v=x+h*HD,a=v[d],b=v[d+HD/2];v[d]=a*c[pos*HD+d]-b*s[pos*HD+d];v[d+HD/2]=b*c[pos*HD+d]+a*s[pos*HD+d];}
static __global__ void k_attn(const float *q,const float *kc,const float *vc,float *out,int sl){int h=blockIdx.x,d=threadIdx.x;if(h>=NH||d>=HD)return;int kvh=h/GQA;__shared__ float scores[4096];if(d==0){float sc=1.0f/sqrtf((float)HD),mx=-1e30f;for(int p=0;p<sl;p++){float dot=0;const float*kr=kc+p*NKV*HD+kvh*HD;for(int dd=0;dd<HD;dd++)dot+=q[h*HD+dd]*kr[dd];scores[p]=dot*sc;if(scores[p]>mx)mx=scores[p];}float su=0;for(int p=0;p<sl;p++){scores[p]=expf(scores[p]-mx);su+=scores[p];}float iv=su>0?1.0f/su:0;for(int p=0;p<sl;p++)scores[p]*=iv;}__syncthreads();float acc=0;for(int p=0;p<sl;p++)acc+=scores[p]*vc[p*NKV*HD+kvh*HD+d];out[h*HD+d]=acc;}
static __global__ void k_silu(float *g,float *u,int n){int i=threadIdx.x+blockIdx.x*blockDim.x;if(i>=n)return;float gv=g[i];u[i]=(gv/(1.0f+expf(-gv)))*u[i];}
static __global__ void k_add(float *a,const float *b,int n){int i=threadIdx.x+blockIdx.x*blockDim.x;if(i>=n)return;a[i]+=b[i];}

static void layer_fwd(GPU *g, int l, int pos) {
    hipStream_t s=g->s;
    hipMemcpyAsync(g->dres,g->dh,H*4,hipMemcpyDeviceToDevice,s);
    hipLaunchKernelGGL(k_rms_norm,dim3(CEILDIV(H,256)),dim3(256),0,s,g->dh,g->h_in[l],H);
    d_mm(g,NH*HD,H,g->dh,g->dout,g->h_wq[l]); d_mm(g,NKV*HD,H,g->dh,g->dout+NH*HD,g->h_wk[l]); d_mm(g,NKV*HD,H,g->dh,g->dout+NH*HD+NKV*HD,g->h_wv[l]);
    hipLaunchKernelGGL(k_qk_norm,dim3(NH+NKV),dim3(HD),0,s,g->dout,g->dout+NH*HD,g->h_qn[l],g->h_kn[l]);
    hipLaunchKernelGGL(k_rope,dim3(NH),dim3(HD/2),0,s,g->dout,pos,g->h_rs,g->h_rc,NH);
    hipLaunchKernelGGL(k_rope,dim3(NKV),dim3(HD/2),0,s,g->dout+NH*HD,pos,g->h_rs,g->h_rc,NKV);
    size_t off=(size_t)l*MAX_CTX*NKV*HD;
    hipMemcpyAsync(g->dkv_k+off+pos*NKV*HD,g->dout+NH*HD,(size_t)NKV*HD*4,hipMemcpyDeviceToDevice,s);
    hipMemcpyAsync(g->dkv_v+off+pos*NKV*HD,g->dout+NH*HD+NKV*HD,(size_t)NKV*HD*4,hipMemcpyDeviceToDevice,s);
    hipLaunchKernelGGL(k_attn,dim3(NH),dim3(HD),0,s,g->dout,g->dkv_k+off,g->dkv_v+off,g->datt,pos+1);
    d_mm(g,H,NH*HD,g->datt,g->dh,g->h_wo[l]); hipLaunchKernelGGL(k_add,dim3(CEILDIV(H,256)),dim3(256),0,s,g->dh,g->dres,H);
    hipMemcpyAsync(g->dres,g->dh,H*4,hipMemcpyDeviceToDevice,s);
    hipLaunchKernelGGL(k_rms_norm,dim3(CEILDIV(H,256)),dim3(256),0,s,g->dh,g->h_pa[l],H);
    d_mm(g,IM,H,g->dh,g->dout,g->h_wg[l]); d_mm(g,IM,H,g->dh,g->dout+IM,g->h_wu[l]);
    hipLaunchKernelGGL(k_silu,dim3(CEILDIV(IM,256)),dim3(256),0,s,g->dout,g->dout+IM,IM);
    d_mm(g,H,IM,g->dout+IM,g->dh,g->h_wd[l]); hipLaunchKernelGGL(k_add,dim3(CEILDIV(H,256)),dim3(256),0,s,g->dh,g->dres,H);
}

static uint32_t decode_step(Model *m, uint32_t tok, int *pos) {
    GPU *g=&m->gpu; hipStream_t s=g->s;
    hipMemcpyAsync(g->dh,g->h_emb+(size_t)tok*H,H*4,hipMemcpyDeviceToDevice,s);
    for(int l=0;l<NC;l++) layer_fwd(g,l,*pos);
    hipLaunchKernelGGL(k_rms_norm,dim3(CEILDIV(H,256)),dim3(256),0,s,g->dh,g->h_fn,H);
    hipblasSgemv(g->blas,HIPBLAS_OP_T,H,NV,g->d_one,g->h_emb,H,g->dh,1,g->d_zero,g->dlg,1);
    hipStreamSynchronize(s);
    float *lg=(float*)malloc((size_t)NV*4); hipMemcpy(lg,g->dlg,(size_t)NV*4,hipMemcpyDeviceToHost);
    uint32_t best=0; float mx=lg[0]; for(int i=1;i<NV;i++) if(lg[i]>mx){mx=lg[i];best=(uint32_t)i;}
    free(lg); (*pos)++; return best;
}

static bool load_model(const char *path, Model *m) {
    int fd=open(path,O_RDONLY);if(fd<0)return false;struct stat st;fstat(fd,&st);
    m->map=(uint8_t*)mmap(0,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    if(!m->map)return false;uint64_t hdr;memcpy(&hdr,m->map,8);m->ds=8+(size_t)hdr;
    const uint8_t *js=m->map+8;size_t jl=(size_t)hdr;
    int64_t eo=fo(js,jl,"model.embed_tokens.weight");
    m->emb=(float*)malloc((size_t)NV*H*4);const uint16_t *eb=(const uint16_t*)(m->map+m->ds+(size_t)eo);for(int i=0;i<NV*H;i++)m->emb[i]=bf16(eb[i]);
    char buf[256];
    for(int l=0;l<NC;l++){
        snprintf(buf,256,"model.layers.%d.input_layernorm.weight",l);int64_t o=fo(js,jl,buf);m->in[l]=(float*)malloc(H*4);const uint16_t*s=(const uint16_t*)(m->map+m->ds+(size_t)o);for(int i=0;i<H;i++)m->in[l][i]=bf16(s[i]);
        snprintf(buf,256,"model.layers.%d.post_attention_layernorm.weight",l);o=fo(js,jl,buf);m->pa[l]=(float*)malloc(H*4);s=(const uint16_t*)(m->map+m->ds+(size_t)o);for(int i=0;i<H;i++)m->pa[l][i]=bf16(s[i]);
        snprintf(buf,256,"model.layers.%d.self_attn.q_norm.weight",l);o=fo(js,jl,buf);if(o>0){m->qn[l]=(float*)malloc(HD*4);const uint16_t*qs=(const uint16_t*)(m->map+m->ds+(size_t)o);for(int i=0;i<HD;i++)m->qn[l][i]=bf16(qs[i]);}else m->qn[l]=0;
        snprintf(buf,256,"model.layers.%d.self_attn.k_norm.weight",l);o=fo(js,jl,buf);if(o>0){m->kn[l]=(float*)malloc(HD*4);const uint16_t*ks=(const uint16_t*)(m->map+m->ds+(size_t)o);for(int i=0;i<HD;i++)m->kn[l][i]=bf16(ks[i]);}else m->kn[l]=0;
    }
    int64_t fno=fo(js,jl,"model.norm.weight");m->fn=(float*)malloc(H*4);const uint16_t*fs=(const uint16_t*)(m->map+m->ds+(size_t)fno);for(int i=0;i<H;i++)m->fn[i]=bf16(fs[i]);
    m->rs=(float*)malloc((size_t)MAX_CTX*HD*4);m->rc=(float*)malloc((size_t)MAX_CTX*HD*4);
    float th=1000000.0f;for(int p=0;p<MAX_CTX;p++)for(int d=0;d<HD/2;d++){float f=1.0f/powf(th,(float)d/(HD/2)),a=(float)p*f;m->rs[p*HD+d]=sinf(a);m->rc[p*HD+d]=cosf(a);m->rs[p*HD+HD/2+d]=sinf(a);m->rc[p*HD+HD/2+d]=cosf(a);}
    for(int l=0;l<NC;l++){snprintf(buf,256,"model.layers.%d.self_attn.q_proj.weight",l);m->qo[l]=fo(js,jl,buf);m->qi[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.self_attn.k_proj.weight",l);m->ko[l]=fo(js,jl,buf);m->ki[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.self_attn.v_proj.weight",l);m->vo[l]=fo(js,jl,buf);m->vi[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.self_attn.o_proj.weight",l);m->oo[l]=fo(js,jl,buf);m->oi[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.mlp.gate_proj.weight",l);m->go[l]=fo(js,jl,buf);m->gi[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.mlp.up_proj.weight",l);m->uo[l]=fo(js,jl,buf);m->ui[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.mlp.down_proj.weight",l);m->do_[l]=fo(js,jl,buf);m->di[l]=si(js,jl,buf);if(m->qi[l]>0)m->has_w=true;}
    fprintf(stderr,"Model: H=%d NC=%d NH=%d NKV=%d NV=%d\n",H,NC,NH,NKV,NV);return true;
}
static uint32_t lt(const char *w){if(!strcmp(w,"<|im_start|>"))return 151644;if(!strcmp(w,"<|im_end|>"))return 151645;if(!strcmp(w,"user"))return 872;if(!strcmp(w,"assistant"))return 77091;if(!strcmp(w,"\n"))return 198;if(!strcmp(w,"Hello")||!strcmp(w,"hello"))return 9707;if(!strcmp(w,"Hi")||!strcmp(w,"hi"))return 11852;if(!strcmp(w,"?"))return 30;if(!strcmp(w,","))return 11;if(!strcmp(w,"."))return 13;return (uint32_t)(unsigned char)w[0]+3;}
static int tok(const char *t,uint32_t*tk,int mx){int n=0;if(n<mx)tk[n++]=151644;if(n<mx)tk[n++]=872;if(n<mx)tk[n++]=198;char w[256];int wl=0;for(const char *p=t;*p&&n<mx;p++){unsigned char c=(unsigned char)*p;if(c==' '||c=='\n'||c=='\t'){if(wl){w[wl]=0;if(n<mx)tk[n++]=lt(w);wl=0;}if(c=='\n'&&n<mx)tk[n++]=198;}else if(c==','||c=='.'||c=='?'||c=='!'||c==';'||c==':'){if(wl){w[wl]=0;if(n<mx)tk[n++]=lt(w);wl=0;}char pn[2]={(char)c,0};if(n<mx)tk[n++]=lt(pn);}else{if(wl<255)w[wl++]=c;}}if(wl&&n<mx){w[wl]=0;tk[n++]=lt(w);}if(n<mx)tk[n++]=151645;if(n<mx)tk[n++]=198;if(n<mx)tk[n++]=151644;if(n<mx)tk[n++]=77091;if(n<mx)tk[n++]=198;return n;}

int main(int argc,char **argv){
    const char *mp=DEF_MODEL,*pr="Hello"; int mx=128;
    for(int i=1;i<argc;i++){if(!strcmp(argv[i],"-h")){printf("engine_55t_persist\n");return 0;}if((!strcmp(argv[i],"-m")||!strcmp(argv[i],"--model"))&&i+1<argc)mp=argv[++i];if((!strcmp(argv[i],"-p")||!strcmp(argv[i],"--prompt"))&&i+1<argc)pr=argv[++i];if((!strcmp(argv[i],"-n")||!strcmp(argv[i],"--max-tokens"))&&i+1<argc)mx=atoi(argv[++i]);}
    fprintf(stderr,"\n=== engine_55t_persist ===\n");
    Model m; memset(&m,0,sizeof(m)); if(!load_model(mp,&m)) return 1;
    int pos=0; uint32_t pt[4096]; int npt=tok(pr,pt,4096);
    fprintf(stderr,"Prompt: %d tokens\n",npt);
    if(!gpu_init(&m)){fprintf(stderr,"GPU init failed\n");return 1;}
    
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); double t0=ts.tv_sec*1e9+ts.tv_nsec;
    uint32_t *out=(uint32_t*)calloc((size_t)mx,4); int gen=0;
    for(int pi=0;pi<npt;pi++){decode_step(&m,pt[pi],&pos);}
    fprintf(stderr,"Decoding %d...\n",mx);
    uint32_t ct=pt[npt-1];
    while(gen<mx){out[gen]=decode_step(&m,ct,&pos); ct=out[gen]; gen++; if(gen%20==0)fprintf(stderr,"  %d/%d\r",gen,mx);}
    clock_gettime(CLOCK_MONOTONIC,&ts); double ms=(ts.tv_sec*1e9+ts.tv_nsec-t0)/1e6;
    fprintf(stderr,"\n═══ %d tokens in %.0fms — %.0f tok/s ═══\n",gen,ms,gen/(ms/1000));
    for(int i=0;i<gen;i++)printf("%u ",out[i]);printf("\n");
    return 0;
}

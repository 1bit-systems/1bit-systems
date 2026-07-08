/* engine_i8.cu — I8 direct matmul engine
 * One kernel launch per token. Reads I8 weights directly (4x less BW).
 * Each thread processes partial outputs using I8 tile dequant.
 *
 * Build: hipcc -O3 -ffast-math -o engine_i8 engine_i8.cu -lhipblas
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

/* ── Device globals (set once at init) ── */
__device__ const float *dev_fn;
__device__ const float *dev_rs, *dev_rc;
__device__ const float *dev_emb;
__device__ const float *dev_in[NC], *dev_pa[NC], *dev_qn[NC], *dev_kn[NC];
__device__ const uint8_t *dev_i8_q[NC], *dev_i8_k[NC], *dev_i8_v[NC], *dev_i8_o[NC];
__device__ const uint8_t *dev_i8_g[NC], *dev_i8_u[NC], *dev_i8_d[NC];

/* Helper: BF16→FP32 */
__device__ float bf16_f32(uint16_t v) { uint32_t bits=(uint32_t)v<<16;return __builtin_bit_cast(float,bits); }

/* I8 dequant for one weight element */
__device__ float i8_deq(const uint8_t *tile, int lr, int col) {
    const uint16_t *sc=(const uint16_t*)tile, *zp=(const uint16_t*)(tile+512);
    const uint8_t *pk=tile+1024;
    int g=col/32, lane=lr/16, lr2=lr%16, bi=lr2/2, ns=lr%2;
    const uint8_t *ld=pk+lane*(TILE_C*8);
    float s=bf16_f32(sc[g*32+lr]), z=bf16_f32(zp[g*32+lr]);
    uint8_t bv=ld[col*8+bi];
    return (float)((ns==0)?(bv&0x0F):((bv>>4)&0x0F))*s+z;
}

/* I8 matmul: W[od][id] @ x[id] → y[od]
 * Each thread computes one output element.
 * Grid: (od+BLK-1)/BLK blocks, BLK threads each.
 * All weight data lives in global __device__ vars, accessed via pointer arrays.
 */
__global__ void i8_matmul_kernel(
    const uint8_t *wt,  /* I8 weight data for one matrix */
    const float *x,     /* input vector [id] */
    float *y,           /* output vector [od] */
    int od, int id)     /* dimensions */
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= od) return;
    int ntc = id / TILE_C;                 /* number of tiles per column */
    int tr = i / TILE_R;                    /* tile row for this output element */
    int lr = i % TILE_R;                    /* local row within tile */
    float acc = 0;
    for (int tc = 0; tc < ntc; tc++) {
        const uint8_t *tile = wt + ((size_t)tr * ntc + tc) * I8_ROW_B;
        for (int c = 0; c < TILE_C; c++) {
            acc += i8_deq(tile, lr, c) * x[tc * TILE_C + c];
        }
    }
    y[i] = acc;
}

/* I8 matmul host helper: enqueue on stream */
static void i8_mm(const uint8_t *wt, const float *x, float *y, int od, int id, hipStream_t s) {
    int blk=256;
    i8_matmul_kernel<<<CEILDIV(od,blk),blk,0,s>>>(wt,x,y,od,id);
}

/* RMS norm kernel */
__global__ void km_rms(float *x, const float *w, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x; if(i>=n)return;
    __shared__ float ss[256]; float v=x[i]; ss[threadIdx.x]=isfinite(v)?v*v:0;
    __syncthreads(); for(int s=blockDim.x/2;s>0;s>>=1){if(threadIdx.x<s)ss[threadIdx.x]+=ss[threadIdx.x+s];__syncthreads();}
    if(threadIdx.x==0)ss[0]=rsqrtf(ss[0]/(float)n+1e-6f);__syncthreads(); x[i]*=ss[0]*w[i];
}
__global__ void km_qknorm(float *q,float *k,const float *qn,const float *kn){
    int h=blockIdx.x,d=threadIdx.x;if(h>=NH+NKV)return;float*v=(h>=NH)?(k+(h-NH)*HD):(q+h*HD);const float*nw=(h>=NH)?kn:qn;
    __shared__ float ss[128];ss[d]=v[d]*v[d];__syncthreads();
    for(int s=64;s>0;s>>=1){if(d<s)ss[d]+=ss[d+s];__syncthreads();}v[d]*=rsqrtf(ss[0]/(float)HD+1e-6f)*(nw?nw[d]:1.0f);
}
__global__ void km_rope(float*x,int pos,const float*s,const float*c,int nh){
    int h=blockIdx.x,d=threadIdx.x;if(h>=nh||d>=HD/2)return;
    float*vec=x+h*HD,a=vec[d],b=vec[d+HD/2];vec[d]=a*c[pos*HD+d]-b*s[pos*HD+d];vec[d+HD/2]=b*c[pos*HD+d]+a*s[pos*HD+d];
}
__global__ void km_attn(const float*q,const float*kc,const float*vc,float*out,int sl){
    int h=blockIdx.x,d=threadIdx.x;if(h>=NH||d>=HD)return;int kvh=h/GQA;
    __shared__ float scores[4096];if(d==0){float sc=1.0f/sqrtf((float)HD),mx=-1e30f;
    for(int p=0;p<sl;p++){float dot=0;const float*kr=kc+p*NKV*HD+kvh*HD;for(int dd=0;dd<HD;dd++)dot+=q[h*HD+dd]*kr[dd];scores[p]=dot*sc;if(scores[p]>mx)mx=scores[p];}
    float su=0;for(int p=0;p<sl;p++){scores[p]=expf(scores[p]-mx);su+=scores[p];}float iv=su>0?1.0f/su:0;for(int p=0;p<sl;p++)scores[p]*=iv;}
    __syncthreads();float acc=0;for(int p=0;p<sl;p++)acc+=scores[p]*vc[p*NKV*HD+kvh*HD+d];out[h*HD+d]=acc;
}
__global__ void km_silu(float*g,float*u,int n){int i=threadIdx.x+blockIdx.x*blockDim.x;if(i>=n)return;float gv=g[i];u[i]=(gv/(1.0f+expf(-gv)))*u[i];}
__global__ void km_add(float*a,const float*b,int n){int i=threadIdx.x+blockIdx.x*blockDim.x;if(i>=n)return;a[i]+=b[i];}

/* Host launch helpers */
static void krms(float *x,const float*w,int n,hipStream_t s){km_rms<<<CEILDIV(n,256),256,0,s>>>(x,w,n);}
static void kqnorm(float *q,float*k,const float*qn,const float*kn,hipStream_t s){km_qknorm<<<NH+NKV,HD,0,s>>>(q,k,qn,kn);}
static void krope(float*x,int pos,const float*rs,const float*rc,int nh,hipStream_t s){km_rope<<<nh,HD/2,0,s>>>(x,pos,rs,rc,nh);}
static void kattn(const float*q,const float*kc,const float*vc,float*out,int sl,hipStream_t s){km_attn<<<NH,HD,0,s>>>(q,kc,vc,out,sl);}
static void ksilu(float*g,float*u,int n,hipStream_t s){km_silu<<<CEILDIV(n,256),256,0,s>>>(g,u,n);}
static void kadd(float*a,const float*b,int n,hipStream_t s){km_add<<<CEILDIV(n,256),256,0,s>>>(a,b,n);}

/* ── GPU struct with buffer pointers ── */
typedef struct {
    hipblasHandle_t blas; hipStream_t s;
    float *dh, *dres, *dout, *datt, *dlg;
    float *dkv_k, *dkv_v;
    float *d_one, *d_zero;
    /* Host copies for device globals */
    float *h_fn; const float *h_rs,*h_rc,*h_emb;
    const float *h_in[NC],*h_pa[NC],*h_qn[NC],*h_kn[NC];
    const uint8_t *h_i8_q[NC],*h_i8_k[NC],*h_i8_v[NC],*h_i8_o[NC],*h_i8_g[NC],*h_i8_u[NC],*h_i8_d[NC];
    bool ok;
} GPU;

typedef struct {
    float *emb,*fn,*in[NC],*pa[NC],*qn[NC],*kn[NC],*rs,*rc;
    uint8_t *map; size_t ds;
    uint64_t qo[NC],ko[NC],vo[NC],oo[NC],go[NC],uo[NC],do_[NC];
    int qi[NC],ki[NC],vi[NC],oi[NC],gi[NC],ui[NC],di[NC];
    bool has_w; GPU gpu;
} Model;

static float* deq_i8(const uint8_t *d,int i8r,int id,int *or_,int *oc_){/*same as before*/int ntc=id/256;if(ntc<1)ntc=1;int ntr=i8r/ntc;*or_=ntr*32;*oc_=ntc*256;float*o=(float*)calloc((size_t)(*or_)*(*oc_),4);for(int ir=0;ir<i8r;ir++){const uint8_t*rd=d+ir*I8_ROW_B;int tr=ir/ntc,tc=ir%ntc;const uint16_t*sc=(const uint16_t*)rd,*zp=(const uint16_t*)(rd+512);const uint8_t*pk=rd+1024;for(int lr=0;lr<TILE_R;lr++){int lane=lr/16,lr2=lr%16,bi=lr2/2,ns=lr%2;const uint8_t*ld=pk+lane*(TILE_C*8);for(int c=0;c<TILE_C;c++){int g=c/32;float s=bf16(sc[g*32+lr]),z=bf16(zp[g*32+lr]);uint8_t bv=ld[c*8+bi];int cd=(ns==0)?(bv&0x0F):((bv>>4)&0x0F);o[(tr*TILE_R+lr)*(*oc_)+(tc*TILE_C+c)]=(float)cd*s+z;}}}return o;}
static bool upload_i8_raw(const uint8_t *map,size_t ds,uint64_t off,int i8r,int id,uint8_t**dw,size_t *sz){*sz=(size_t)i8r*I8_ROW_B;hipMalloc((void**)dw,*sz);hipMemcpy(*dw,map+ds+off,*sz,hipMemcpyHostToDevice);return true;}

static const uint8_t*fk(const uint8_t*j,size_t l,const char*k){size_t kl=strlen(k);for(size_t i=0;i+kl+2<l;i++)if(j[i]=='"'&&memcmp(j+i+1,k,kl)==0&&j[i+1+kl]=='"')return j+i;return 0;}
static int64_t fo(const uint8_t*j,size_t l,const char*k){const uint8_t*p=fk(j,l,k);if(!p)return-1;const uint8_t*d=(const uint8_t*)strstr((const char*)p,"\"data_offsets\"");if(!d)return-1;const uint8_t*b=(const uint8_t*)strchr((const char*)d,'[');return b?strtoll((const char*)(b+1),0,10):-1;}
static int si(const uint8_t*j,size_t l,const char*k){size_t kl=strlen(k);const char*js=(const char*)j;for(const char*p=js;p<js+(int)l;p++){p=strstr(p,k);if(!p)return 0;if((p==js||*(p-1)=='"')&&*(p+kl)=='"'){const char*sh=strstr(p,"\"shape\"");if(!sh)return 0;const char*br=strchr(sh,'[');if(!br)return 0;return (int)strtoul(br+1,0,10);}p+=kl;}return 0;}

static bool gpu_init(Model*m){
    GPU*g=&m->gpu;g->ok=false;
    hipblasCreate(&g->blas);hipStreamCreate(&g->s);hipblasSetStream(g->blas,g->s);
    hipMalloc(&g->d_one,4);hipMalloc(&g->d_zero,4);float one=1,zero=0;hipMemcpy(g->d_one,&one,4,hipMemcpyHostToDevice);hipMemcpy(g->d_zero,&zero,4,hipMemcpyHostToDevice);hipblasSetPointerMode(g->blas,HIPBLAS_POINTER_MODE_DEVICE);
    hipMalloc(&g->dh,H*4);hipMalloc(&g->dres,H*4);hipMalloc(&g->dout,((QT>(int)(2*IM))?QT:(2*IM))*4);hipMalloc(&g->datt,(size_t)NH*HD*4);hipMalloc(&g->dlg,(size_t)NV*4);
    size_t kv_sz=(size_t)NC*MAX_CTX*NKV*HD;hipMalloc(&g->dkv_k,kv_sz*4);hipMalloc(&g->dkv_v,kv_sz*4);hipMemset(g->dkv_k,0,kv_sz*4);hipMemset(g->dkv_v,0,kv_sz*4);
    for(int l=0;l<NC;l++){
        hipMalloc((void**)&g->h_in[l],H*4);hipMemcpy((void*)g->h_in[l],m->in[l],H*4,hipMemcpyHostToDevice);
        hipMalloc((void**)&g->h_pa[l],H*4);hipMemcpy((void*)g->h_pa[l],m->pa[l],H*4,hipMemcpyHostToDevice);
        if(m->qn[l]){hipMalloc((void**)&g->h_qn[l],HD*4);hipMemcpy((void*)g->h_qn[l],m->qn[l],HD*4,hipMemcpyHostToDevice);}else g->h_qn[l]=0;
        if(m->kn[l]){hipMalloc((void**)&g->h_kn[l],HD*4);hipMemcpy((void*)g->h_kn[l],m->kn[l],HD*4,hipMemcpyHostToDevice);}else g->h_kn[l]=0;
    }
    hipMalloc((void**)&g->h_fn,H*4);hipMemcpy((void*)g->h_fn,m->fn,H*4,hipMemcpyHostToDevice);
    hipMalloc((void**)&g->h_rs,(size_t)MAX_CTX*HD*4);hipMalloc((void**)&g->h_rc,(size_t)MAX_CTX*HD*4);hipMemcpy((void*)g->h_rs,m->rs,(size_t)MAX_CTX*HD*4,hipMemcpyHostToDevice);hipMemcpy((void*)g->h_rc,m->rc,(size_t)MAX_CTX*HD*4,hipMemcpyHostToDevice);
    hipMalloc((void**)&g->h_emb,(size_t)NV*H*4);hipMemcpy((void*)g->h_emb,m->emb,(size_t)NV*H*4,hipMemcpyHostToDevice);
    fprintf(stderr,"Uploading I8 weights...\n");
    for(int l=0;l<NC;l++){fprintf(stderr,"  %d/%d\r",l+1,NC);size_t sz;
        upload_i8_raw(m->map,m->ds,m->qo[l],m->qi[l],H,(uint8_t**)&g->h_i8_q[l],&sz);
        upload_i8_raw(m->map,m->ds,m->ko[l],m->ki[l],H,(uint8_t**)&g->h_i8_k[l],&sz);
        upload_i8_raw(m->map,m->ds,m->vo[l],m->vi[l],H,(uint8_t**)&g->h_i8_v[l],&sz);
        upload_i8_raw(m->map,m->ds,m->oo[l],m->oi[l],NH*HD,(uint8_t**)&g->h_i8_o[l],&sz);
        upload_i8_raw(m->map,m->ds,m->go[l],m->gi[l],H,(uint8_t**)&g->h_i8_g[l],&sz);
        upload_i8_raw(m->map,m->ds,m->uo[l],m->ui[l],H,(uint8_t**)&g->h_i8_u[l],&sz);
        upload_i8_raw(m->map,m->ds,m->do_[l],m->di[l],IM,(uint8_t**)&g->h_i8_d[l],&sz);
    }
    fprintf(stderr,"\nGPU ready (I8 mode).\n");g->ok=true;return true;
}

/* Forward using I8 matmuls + GPU kernels */
static void layer_fwd_i8(GPU*g,int l,int pos){
    hipStream_t s=g->s;
    hipMemcpyAsync(g->dres,g->dh,H*4,hipMemcpyDeviceToDevice,s);
    krms(g->dh,g->h_in[l],H,s);
    i8_mm(g->h_i8_q[l],g->dh,g->dout,NH*HD,H,s);
    i8_mm(g->h_i8_k[l],g->dh,g->dout+NH*HD,NKV*HD,H,s);
    i8_mm(g->h_i8_v[l],g->dh,g->dout+NH*HD+NKV*HD,NKV*HD,H,s);
    kqnorm(g->dout,g->dout+NH*HD,g->h_qn[l],g->h_kn[l],s);
    krope(g->dout,pos,g->h_rs,g->h_rc,NH,s);
    krope(g->dout+NH*HD,pos,g->h_rs,g->h_rc,NKV,s);
    size_t off=(size_t)l*MAX_CTX*NKV*HD;
    hipMemcpyAsync(g->dkv_k+off+pos*NKV*HD,g->dout+NH*HD,(size_t)NKV*HD*4,hipMemcpyDeviceToDevice,s);
    hipMemcpyAsync(g->dkv_v+off+pos*NKV*HD,g->dout+NH*HD+NKV*HD,(size_t)NKV*HD*4,hipMemcpyDeviceToDevice,s);
    kattn(g->dout,g->dkv_k+off,g->dkv_v+off,g->datt,pos+1,s);
    i8_mm(g->h_i8_o[l],g->datt,g->dh,H,NH*HD,s);
    kadd(g->dh,g->dres,H,s);
    hipMemcpyAsync(g->dres,g->dh,H*4,hipMemcpyDeviceToDevice,s);
    krms(g->dh,g->h_pa[l],H,s);
    i8_mm(g->h_i8_g[l],g->dh,g->dout,IM,H,s);
    i8_mm(g->h_i8_u[l],g->dh,g->dout+IM,IM,H,s);
    ksilu(g->dout,g->dout+IM,IM,s);
    i8_mm(g->h_i8_d[l],g->dout+IM,g->dh,H,IM,s);
    kadd(g->dh,g->dres,H,s);
}

static uint32_t decode(Model*m,uint32_t tok,int*pos){
    GPU*g=&m->gpu;hipStream_t s=g->s;
    hipMemcpyAsync(g->dh,g->h_emb+(size_t)tok*H,H*4,hipMemcpyDeviceToDevice,s);
    for(int l=0;l<NC;l++)layer_fwd_i8(g,l,*pos);
    krms(g->dh,g->h_fn,H,s);
    hipblasSgemv(g->blas,HIPBLAS_OP_T,H,NV,g->d_one,g->h_emb,H,g->dh,1,g->d_zero,g->dlg,1);
    hipStreamSynchronize(s);
    float*lg=(float*)malloc((size_t)NV*4);hipMemcpy(lg,g->dlg,(size_t)NV*4,hipMemcpyDeviceToHost);
    uint32_t best=0;float mx=lg[0];for(int i=1;i<NV;i++)if(lg[i]>mx){mx=lg[i];best=(uint32_t)i;}
    free(lg);(*pos)++;return best;
}

static bool load_model(const char*path,Model*m){
    int fd=open(path,O_RDONLY);if(fd<0)return false;struct stat st;fstat(fd,&st);m->map=(uint8_t*)mmap(0,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    if(!m->map)return false;uint64_t hdr;memcpy(&hdr,m->map,8);m->ds=8+(size_t)hdr;const uint8_t*js=m->map+8;size_t jl=(size_t)hdr;
    int64_t eo=fo(js,jl,"model.embed_tokens.weight");m->emb=(float*)malloc((size_t)NV*H*4);const uint16_t*eb=(const uint16_t*)(m->map+m->ds+(size_t)eo);for(int i=0;i<NV*H;i++)m->emb[i]=bf16(eb[i]);
    char buf[256];for(int l=0;l<NC;l++){
        snprintf(buf,256,"model.layers.%d.input_layernorm.weight",l);int64_t o=fo(js,jl,buf);m->in[l]=(float*)malloc(H*4);const uint16_t*s=(const uint16_t*)(m->map+m->ds+(size_t)o);for(int i=0;i<H;i++)m->in[l][i]=bf16(s[i]);
        snprintf(buf,256,"model.layers.%d.post_attention_layernorm.weight",l);o=fo(js,jl,buf);m->pa[l]=(float*)malloc(H*4);s=(const uint16_t*)(m->map+m->ds+(size_t)o);for(int i=0;i<H;i++)m->pa[l][i]=bf16(s[i]);
        snprintf(buf,256,"model.layers.%d.self_attn.q_norm.weight",l);o=fo(js,jl,buf);if(o>0){m->qn[l]=(float*)malloc(HD*4);const uint16_t*qs=(const uint16_t*)(m->map+m->ds+(size_t)o);for(int i=0;i<HD;i++)m->qn[l][i]=bf16(qs[i]);}else m->qn[l]=0;
        snprintf(buf,256,"model.layers.%d.self_attn.k_norm.weight",l);o=fo(js,jl,buf);if(o>0){m->kn[l]=(float*)malloc(HD*4);const uint16_t*ks=(const uint16_t*)(m->map+m->ds+(size_t)o);for(int i=0;i<HD;i++)m->kn[l][i]=bf16(ks[i]);}else m->kn[l]=0;
    }
    int64_t fno=fo(js,jl,"model.norm.weight");m->fn=(float*)malloc(H*4);const uint16_t*fs=(const uint16_t*)(m->map+m->ds+(size_t)fno);for(int i=0;i<H;i++)m->fn[i]=bf16(fs[i]);
    m->rs=(float*)malloc((size_t)MAX_CTX*HD*4);m->rc=(float*)malloc((size_t)MAX_CTX*HD*4);float th=1000000.0f;for(int p=0;p<MAX_CTX;p++)for(int d=0;d<HD/2;d++){float f=1.0f/powf(th,(float)d/(HD/2)),a=(float)p*f;m->rs[p*HD+d]=sinf(a);m->rc[p*HD+d]=cosf(a);m->rs[p*HD+HD/2+d]=sinf(a);m->rc[p*HD+HD/2+d]=cosf(a);}
    for(int l=0;l<NC;l++){snprintf(buf,256,"model.layers.%d.self_attn.q_proj.weight",l);m->qo[l]=fo(js,jl,buf);m->qi[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.self_attn.k_proj.weight",l);m->ko[l]=fo(js,jl,buf);m->ki[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.self_attn.v_proj.weight",l);m->vo[l]=fo(js,jl,buf);m->vi[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.self_attn.o_proj.weight",l);m->oo[l]=fo(js,jl,buf);m->oi[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.mlp.gate_proj.weight",l);m->go[l]=fo(js,jl,buf);m->gi[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.mlp.up_proj.weight",l);m->uo[l]=fo(js,jl,buf);m->ui[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.mlp.down_proj.weight",l);m->do_[l]=fo(js,jl,buf);m->di[l]=si(js,jl,buf);if(m->qi[l]>0)m->has_w=true;}
    fprintf(stderr,"Model: H=%d NC=%d NH=%d NKV=%d NV=%d\n",H,NC,NH,NKV,NV);return true;
}
static uint32_t lt(const char*w){if(!strcmp(w,"<|im_start|>"))return 151644;if(!strcmp(w,"<|im_end|>"))return 151645;if(!strcmp(w,"user"))return 872;if(!strcmp(w,"assistant"))return 77091;if(!strcmp(w,"\n"))return 198;if(!strcmp(w,"Hello")||!strcmp(w,"hello"))return 9707;if(!strcmp(w,"Hi")||!strcmp(w,"hi"))return 11852;if(!strcmp(w,"?"))return 30;if(!strcmp(w,","))return 11;if(!strcmp(w,"."))return 13;return (uint32_t)(unsigned char)w[0]+3;}
static int tok(const char*t,uint32_t*tk,int mx){int n=0;if(n<mx)tk[n++]=151644;if(n<mx)tk[n++]=872;if(n<mx)tk[n++]=198;char w[256];int wl=0;for(const char*p=t;*p&&n<mx;p++){unsigned char c=(unsigned char)*p;if(c==' '||c=='\n'||c=='\t'){if(wl){w[wl]=0;if(n<mx)tk[n++]=lt(w);wl=0;}if(c=='\n'&&n<mx)tk[n++]=198;}else if(c==','||c=='.'||c=='?'||c=='!'||c==';'||c==':'){if(wl){w[wl]=0;if(n<mx)tk[n++]=lt(w);wl=0;}char pn[2]={(char)c,0};if(n<mx)tk[n++]=lt(pn);}else{if(wl<255)w[wl++]=c;}}if(wl&&n<mx){w[wl]=0;tk[n++]=lt(w);}if(n<mx)tk[n++]=151645;if(n<mx)tk[n++]=198;if(n<mx)tk[n++]=151644;if(n<mx)tk[n++]=77091;if(n<mx)tk[n++]=198;return n;}

int main(int argc,char**argv){
    const char*mp=DEF_MODEL,*pr="Hello";int mx=128;
    for(int i=1;i<argc;i++){if(!strcmp(argv[i],"-h")){printf("engine_i8\n");return 0;}if((!strcmp(argv[i],"-m")||!strcmp(argv[i],"--model"))&&i+1<argc)mp=argv[++i];if((!strcmp(argv[i],"-p")||!strcmp(argv[i],"--prompt"))&&i+1<argc)pr=argv[++i];if((!strcmp(argv[i],"-n")||!strcmp(argv[i],"--max-tokens"))&&i+1<argc)mx=atoi(argv[++i]);}
    fprintf(stderr,"\n=== engine_i8 — I8 direct matmul ===\n");
    Model m;memset(&m,0,sizeof(m));if(!load_model(mp,&m))return 1;
    int pos=0;uint32_t pt[4096];int npt=tok(pr,pt,4096);fprintf(stderr,"Prompt: %d tokens\n",npt);
    if(!gpu_init(&m)){fprintf(stderr,"GPU init failed\n");return 1;}
    struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);double t0=ts.tv_sec*1e9+ts.tv_nsec;
    uint32_t*out=(uint32_t*)calloc((size_t)mx,4);int gen=0;
    for(int pi=0;pi<npt;pi++){decode(&m,pt[pi],&pos);}
    fprintf(stderr,"Decoding %d...\n",mx);
    uint32_t ct=pt[npt-1];
    while(gen<mx){out[gen]=decode(&m,ct,&pos);ct=out[gen];gen++;if(gen%20==0)fprintf(stderr,"  %d/%d\r",gen,mx);}
    clock_gettime(CLOCK_MONOTONIC,&ts);double ms=(ts.tv_sec*1e9+ts.tv_nsec-t0)/1e6;
    fprintf(stderr,"\n═══ %d tokens in %.0fms — %.0f tok/s ═══\n",gen,ms,gen/(ms/1000));
    for(int i=0;i<gen;i++)printf("%u ",out[i]);printf("\n");
    return 0;
}

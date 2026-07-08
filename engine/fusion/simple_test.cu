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

#define I8_ROW_B 5120
#define TILE_R 32
#define TILE_C 256
#define DEF_MODEL "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx"
enum { H=1024, NC=28, NH=16, NKV=8, HD=128, IM=3072, NV=151936, GQA=2, QT=NH*HD+2*NKV*HD };
static float bf16(uint16_t v) { uint32_t b=(uint32_t)v<<16;float f;memcpy(&f,&b,4);return f; }

static float* deq_i8(const uint8_t *d, int i8r, int id, int *or_, int *oc) {
    int ntc=id/256;if(ntc<1)ntc=1;int ntr=i8r/ntc;
    *or_=ntr*32;*oc=ntc*256;
    float *o=(float*)calloc((size_t)(*or_)*(*oc),4);
    for(int ir=0;ir<i8r;ir++){
        const uint8_t *rd=d+ir*I8_ROW_B;int tr=ir/ntc,tc=ir%ntc;
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

/* CPU matmul for reference */
static void cpu_mm(const float *h, int od, int id, float *out, const float *w) {
    memset(out,0,(size_t)od*4);
    for(int i=0;i<od;i++){double dot=0;for(int j=0;j<id;j++)dot+=(double)w[i*id+j]*(double)h[j];out[i]=(float)dot;}
}

int main(){
    fprintf(stderr,"Loading model...\n");
    int fd=open(DEF_MODEL,O_RDONLY);
    struct stat st; fstat(fd,&st);
    uint8_t *map=(uint8_t*)mmap(0,st.st_size,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
    uint64_t hdr; memcpy(&hdr,map,8); size_t ds=8+(size_t)hdr;
    const uint8_t *js=map+8; size_t jl=(size_t)hdr;

    fprintf(stderr,"Loading Q weight for layer 0...\n");
    int64_t qoff = fo(js,jl,"model.layers.0.self_attn.q_proj.weight");
    int qi = si(js,jl,"model.layers.0.self_attn.q_proj.weight");
    fprintf(stderr,"  offset=%ld rows=%d\n",(long)qoff,qi);
    int or_,oc;
    float *cpu_q = deq_i8(map+ds+qoff,qi,H,&or_,&oc);
    fprintf(stderr,"  dequantized: %d x %d\n",or_,oc);

    // Init GPU
    hipblasHandle_t blas; hipblasCreate(&blas);
    float *d_q,*d_in,*d_out,*d_one,*d_zero;
    hipMalloc(&d_q,(size_t)or_*oc*4);
    hipMemcpy(d_q,cpu_q,(size_t)or_*oc*4,hipMemcpyHostToDevice);
    hipMalloc(&d_in,H*4);
    hipMalloc(&d_out,(size_t)or_*4);
    hipMalloc(&d_one,4); hipMalloc(&d_zero,4);
    float one=1.0f,zero=0.0f;
    hipMemcpy(d_one,&one,4,hipMemcpyHostToDevice);
    hipMemcpy(d_zero,&zero,4,hipMemcpyHostToDevice);

    // Test input
    float h_in[H];
    for(int i=0;i<H;i++) h_in[i] = (float)(i % 100) / 100.0f;
    hipMemcpy(d_in,h_in,H*4,hipMemcpyHostToDevice);

    // CPU reference
    float cpu_out[NH*HD];
    cpu_mm(h_in,NH*HD,H,cpu_out,cpu_q);

    // GPU with HOST pointer mode
    float gpu_out1[NH*HD];
    float a=1.0f,b=0.0f;
    hipblasSgemv(blas,HIPBLAS_OP_T,H,NH*HD,&a,d_q,H,d_in,1,&b,d_out,1);
    hipDeviceSynchronize();
    hipMemcpy(gpu_out1,d_out,(size_t)NH*HD*4,hipMemcpyDeviceToHost);

    // GPU with DEVICE pointer mode
    float gpu_out2[NH*HD];
    hipblasSetPointerMode(blas,HIPBLAS_POINTER_MODE_DEVICE);
    hipblasSgemv(blas,HIPBLAS_OP_T,H,NH*HD,d_one,d_q,H,d_in,1,d_zero,d_out,1);
    hipDeviceSynchronize();
    hipMemcpy(gpu_out2,d_out,(size_t)NH*HD*4,hipMemcpyDeviceToHost);

    // Compare
    float max_diff1=0,max_diff2=0;
    for(int i=0;i<NH*HD;i++){
        float d1=fabsf(cpu_out[i]-gpu_out1[i]),d2=fabsf(cpu_out[i]-gpu_out2[i]);
        if(d1>max_diff1){max_diff1=d1;}
        if(d2>max_diff2){max_diff2=d2;}
    }
    fprintf(stderr,"CPU vs GPU (HOST ptr):   max_diff=%.8f\n",max_diff1);
    fprintf(stderr,"CPU vs GPU (DEVICE ptr): max_diff=%.8f\n",max_diff2);
    fprintf(stderr,"GPU HOST vs GPU DEVICE:  max_diff=%.8f\n",fabsf(gpu_out1[0]-gpu_out2[0])>0?fabsf(gpu_out1[0]-gpu_out2[0]):0);
    for(int i=0;i<5;i++) fprintf(stderr,"  [%d] cpu=%.6f gpu_host=%.6f gpu_dev=%.6f\n",i,cpu_out[i],gpu_out1[i],gpu_out2[i]);

    // Cleanup
    free(cpu_q); hipFree(d_q); hipFree(d_in); hipFree(d_out); hipFree(d_one); hipFree(d_zero);
    hipblasDestroy(blas);
    munmap(map,st.st_size);
    return 0;
}

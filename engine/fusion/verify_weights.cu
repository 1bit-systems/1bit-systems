/* Verify GPU weights match CPU weights */
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
enum { H=1024, NH=16, NKV=8, HD=128, IM=3072, NV=151936, GQA=2 };
static float bf16(uint16_t v) { uint32_t b=(uint32_t)v<<16;float f;memcpy(&f,&b,4);return f; }

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

int main(){
    int fd=open(DEF_MODEL,O_RDONLY); struct stat st; fstat(fd,&st);
    uint8_t *map=(uint8_t*)mmap(0,st.st_size,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
    uint64_t hdr; memcpy(&hdr,map,8); size_t ds=8+(size_t)hdr;
    const uint8_t *js=map+8; size_t jl=(size_t)hdr;

    char buf[256];
    // Compare Q weight for layer 0: CPU vs GPU upload/readback
    snprintf(buf,256,"model.layers.0.self_attn.q_proj.weight");
    int64_t off = fo(js,jl,buf);
    int rows = si(js,jl,buf);
    fprintf(stderr,"Layer 0 Q: offset=%ld rows=%d\n",(long)off,rows);

    int or_,oc;
    float *cpu_q = deq_i8(map+ds+off, rows, H, &or_, &oc);
    fprintf(stderr,"  CPU dequant: %d x %d, first 5: %.6f %.6f %.6f %.6f %.6f\n",
        or_,oc,cpu_q[0],cpu_q[1],cpu_q[2],cpu_q[3],cpu_q[4]);

    // Upload to GPU and read back
    float *d_q;
    hipMalloc(&d_q,(size_t)or_*oc*4);
    hipMemcpy(d_q,cpu_q,(size_t)or_*oc*4,hipMemcpyHostToDevice);
    float *gpu_q=(float*)malloc((size_t)or_*oc*4);
    hipMemcpy(gpu_q,d_q,(size_t)or_*oc*4,hipMemcpyDeviceToHost);
    fprintf(stderr,"  GPU roundtrip: first 5: %.6f %.6f %.6f %.6f %.6f\n",
        gpu_q[0],gpu_q[1],gpu_q[2],gpu_q[3],gpu_q[4]);

    // Now test with a REAL input from the model
    // Load embeddings
    int64_t eo=fo(js,jl,"model.embed_tokens.weight");
    const uint16_t *eb=(const uint16_t*)(map+ds+(size_t)eo);
    
    // First token: <|im_start|> = 151644
    float h_in[H];
    for(int i=0;i<H;i++) h_in[i] = bf16(eb[151644*H + i]);
    fprintf(stderr,"\nEmbedding[151644] first 5: %.6f %.6f %.6f %.6f %.6f\n",
        h_in[0],h_in[1],h_in[2],h_in[3],h_in[4]);

    // CPU matmul: h_out = W_q @ h_in
    float cpu_out[NH*HD];
    memset(cpu_out,0,sizeof(cpu_out));
    for(int i=0;i<NH*HD;i++){
        double dot=0;
        for(int j=0;j<H;j++) dot += (double)cpu_q[i*H+j] * (double)h_in[j];
        cpu_out[i] = (float)dot;
    }
    fprintf(stderr,"  CPU Q output first 5: %.6f %.6f %.6f %.6f %.6f\n",
        cpu_out[0],cpu_out[1],cpu_out[2],cpu_out[3],cpu_out[4]);

    // GPU Sgemv
    hipblasHandle_t blas; hipblasCreate(&blas);
    float *d_in,*d_out,*d_one,*d_zero;
    hipMalloc(&d_in,H*4); hipMemcpy(d_in,h_in,H*4,hipMemcpyHostToDevice);
    hipMalloc(&d_out,(size_t)NH*HD*4);
    hipMalloc(&d_one,4); hipMalloc(&d_zero,4);
    float one=1.0f,zero=0.0f;
    hipMemcpy(d_one,&one,4,hipMemcpyHostToDevice);
    hipMemcpy(d_zero,&zero,4,hipMemcpyHostToDevice);
    hipblasSetPointerMode(blas,HIPBLAS_POINTER_MODE_DEVICE);

    hipblasSgemv(blas,HIPBLAS_OP_T,H,NH*HD,d_one,d_q,H,d_in,1,d_zero,d_out,1);
    hipDeviceSynchronize();
    float gpu_out[NH*HD];
    hipMemcpy(gpu_out,d_out,(size_t)NH*HD*4,hipMemcpyDeviceToHost);
    fprintf(stderr,"  GPU Q output first 5: %.6f %.6f %.6f %.6f %.6f\n",
        gpu_out[0],gpu_out[1],gpu_out[2],gpu_out[3],gpu_out[4]);

    // Max diff
    float md=0;
    for(int i=0;i<NH*HD;i++){float d=fabsf(cpu_out[i]-gpu_out[i]);if(d>md)md=d;}
    fprintf(stderr,"  Max CPU vs GPU diff: %.10f\n",md);

    // Cleanup
    free(cpu_q); free(gpu_q); hipFree(d_q); hipFree(d_in); hipFree(d_out); hipFree(d_one); hipFree(d_zero);
    hipblasDestroy(blas); munmap(map,st.st_size);
    return 0;
}

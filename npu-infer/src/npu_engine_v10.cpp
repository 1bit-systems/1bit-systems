/** NPU Engine v10 — Single 1024×1024 Chess xclbin, iterated for all projections */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include "helper.h"
#include "gemm_atb_layout.h"
extern "C" float* dequant_i8_to_float(const uint8_t*,int,int*,int*);
extern "C" float bf16_to_float(uint16_t);
extern "C" uint16_t float_to_bf16(float);

static float bf16f(uint16_t v){uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}
static float bf16_g(uint16_t v){return((v&0x7F80)==0x7F80)?0:bf16f(v);}
static uint16_t fbf16(float v){uint32_t b;memcpy(&b,&v,4);uint32_t r=((b>>16)&1)+0x7FFF;return(uint16_t)((b+r)>>16);}
static float fn2(const float*x,int n){double s=0;for(int i=0;i<n;i++)s+=(double)x[i]*x[i];return sqrtf((float)s);}

static constexpr int MT=128,KT=64,NT=128,NROWS=4,XM=1024,XK=1024,XN=1024;
static constexpr int H=1024,NC=28,NH=16,NKV=8,HD=128,IM=3072,NV=151936,GQA=2;
static constexpr float EPS=1e-6f,TEMP=0.7f;

struct Npu {
    std::vector<uint32_t> ins;
    xrt::device dev{0};
    std::unique_ptr<xrt::xclbin> xc;
    std::unique_ptr<xrt::hw_context> hc;
    std::unique_ptr<xrt::kernel> k;
    std::unique_ptr<xrt::bo> bI,bA,bB,bC;
    std::vector<float> Ab,Cb;

    bool init(){
        const char*ip="/home/bcloud/npu-sandbox/npu-infer/build/chess_infer/single/qwen3_single.insts";
        const char*xp="/home/bcloud/npu-sandbox/npu-infer/build/chess_infer/single/qwen3_single.xclbin";
        FILE*f=fopen(ip,"rb");if(!f)return false;
        fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);ins.resize(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);
        xc=std::make_unique<xrt::xclbin>(std::string(xp));dev.register_xclbin(*xc);
        hc=std::make_unique<xrt::hw_context>(dev,xc->get_uuid());
        k=std::make_unique<xrt::kernel>(*hc,"MLIR_AIE");
        bI=std::make_unique<xrt::bo>(dev,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k->group_id(1));
        bA=std::make_unique<xrt::bo>(dev,(size_t)XM*XK*2,XRT_BO_FLAGS_HOST_ONLY,k->group_id(3));
        bB=std::make_unique<xrt::bo>(dev,(size_t)XK*XN*9/8,XRT_BO_FLAGS_HOST_ONLY,k->group_id(4));
        bC=std::make_unique<xrt::bo>(dev,(size_t)XM*XN*2,XRT_BO_FLAGS_HOST_ONLY,k->group_id(5));
        memcpy(bI->map(),ins.data(),ins.size()*4);bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        Ab.resize(XM*XK);Cb.resize(XM*XN);
        return true;
    }

    // Single GEMM on the 1024×1024 xclbin. For M>1 or N>XN, caller iterates.
    void gemm_one(const float*A,int am,int ak,const uint8_t*B,size_t Bsz,int an,float*C){
        int MK=XM*XK;memset(Ab.data(),0,MK*4);
        for(int m=0;m<am;m++)for(int kk=0;kk<ak;kk++)Ab[m*XK+kk]=A[m*ak+kk];
        auto As=gemm_atb::layout_A_L1_2x1_8x8block(Ab,XM,XK,MT,KT);
        uint16_t*Ab2=(uint16_t*)bA->map();for(int i=0;i<MK;i++)Ab2[i]=fbf16(As[i]);bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        memcpy(bB->map(),B,Bsz);bB->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        memset(bC->map(),0,(size_t)XM*XN*2);bC->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto r=(*k)((unsigned)3,*bI,ins.size(),*bA,*bB,*bC);
        if(r.wait()!=ERT_CMD_STATE_COMPLETED){fprintf(stderr,"[NPU] FAIL\n");return;}
        bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        uint16_t*Cc=(uint16_t*)bC->map();int MN=XM*XN;
        for(int i=0;i<MN;i++)Cb[i]=bf16f(Cc[i]);
        auto Cu=gemm_atb::layout_inverse_C_L1_2x2_8x8block(Cb,XM,XN,NROWS*MT,NT);
        for(int m=0;m<am;m++)for(int n=0;n<an;n++)C[m*an+n]=Cu[m*XN+n];
    }

    // Packed GEMM: iterates over N if output dim > XN, and/or over M if > XM
    void gemm(const float*A,int M,int K,const float*W,int w_rows,int N,float*C){
        bool trans=(w_rows==K&&K!=N);
        for(int nc=0;nc<N;nc+=XN){
            int n_sz=(nc+XN<=N)?XN:N-nc;
            // Pack weight slice [K][nc:nc+n_sz]
            std::vector<float>Bf(K*n_sz,0.0f);
            for(int k=0;k<K;k++)for(int n=0;n<n_sz;n++){
                float v=trans?W[(nc+n)*K+k]:W[(nc+n)*w_rows+k];
                if(std::isfinite(v))Bf[k*n_sz+n]=v;
            }
            auto Bs=gemm_atb::layout_transpose_L1_1x2_8x8block(Bf,K,n_sz,KT,NT);
            auto Bp=floatToBfp16(8,K*n_sz,Bs.data(),0);
            // Pad to full xclbin B buffer size (XK*XN*9/8) with zeros
            size_t fullB=(size_t)XK*XN*9/8;
            std::vector<uint8_t>Bpad(fullB,0);
            memcpy(Bpad.data(),Bp.data(),std::min(Bp.size(),fullB));
            gemm_one(A,M,K,Bpad.data(),fullB,n_sz,C+nc);
        }
    }
};

static std::vector<float>rc,rs;
static void ri(int hd,float th,int mp){rc.resize(mp*hd);rs.resize(mp*hd);
    for(int p=0;p<mp;p++)for(int d=0;d<hd;d+=2){float f=1.0f/powf(th,(float)d/hd),a=p*f;
        rc[p*hd+d]=cosf(a);rs[p*hd+d]=sinf(a);rc[p*hd+d+1]=cosf(a);rs[p*hd+d+1]=sinf(a);}}
static void ra(float*x,int hd,int p){for(int d=0;d<hd;d+=2){float a=x[d],b=x[d+1],c=rc[p*hd+d],s=rs[p*hd+d];x[d]=a*c-b*s;x[d+1]=b*c+a*s;}}
static void cn(float*x,int n){for(int i=0;i<n;i++)if(!std::isfinite(x[i]))x[i]=0.0f;}
static void rn(float*x,const uint16_t*w,int n){cn(x,n);double ss=0;for(int i=0;i<n;i++)ss+=(double)x[i]*x[i];
    float ir=1.0f/sqrtf((float)(ss/n)+EPS);for(int i=0;i<n;i++)x[i]=x[i]*ir*bf16_g(w[i]);}
static uint64_t jo(const char*js,size_t jl,const char*nm){
    size_t nl=strlen(nm);const char*p=js,*e=js+jl;
    while(p<e){auto q=(const char*)memmem(p,e-p,nm,nl);if(!q)return 0;
        if(q>js&&*(q-1)=='"'&&*(q+nl)=='"'){auto o=strstr(q,"\"data_offsets\"");
            if(o){auto a=strchr(o,'[');if(a)return strtoull(a+1,NULL,10);}}p=q+1;}return 0;}

int main(){
    setvbuf(stdout,NULL,_IONBF,0);
    printf("=== v10 Single-XCLBIN Chess ===\n\n");
    const char*mp="/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
    int fd=open(mp,O_RDONLY);struct stat st;fstat(fd,&st);
    uint8_t*md=(uint8_t*)mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    uint64_t hsz;memcpy(&hsz,md,8);uint64_t df=8+hsz;
    auto i8p=[&](uint64_t o){return md+df+o;};auto emb=(const uint16_t*)(md+df);
    const char*js=(const char*)(md+8);size_t jl=hsz;

    struct LO{uint64_t qp,kp,vp,op,gp,up,dp,in_off,pa_off,qn_off,kn_off;}lo[28];char b[128];
    for(int l=0;l<NC;l++){
        snprintf(b,sizeof(b),"model.layers.%d.self_attn.q_proj.weight",l);lo[l].qp=jo(js,jl,b);
        snprintf(b,sizeof(b),"model.layers.%d.self_attn.k_proj.weight",l);lo[l].kp=jo(js,jl,b);
        snprintf(b,sizeof(b),"model.layers.%d.self_attn.v_proj.weight",l);lo[l].vp=jo(js,jl,b);
        snprintf(b,sizeof(b),"model.layers.%d.self_attn.o_proj.weight",l);lo[l].op=jo(js,jl,b);
        snprintf(b,sizeof(b),"model.layers.%d.mlp.gate_proj.weight",l);lo[l].gp=jo(js,jl,b);
        snprintf(b,sizeof(b),"model.layers.%d.mlp.up_proj.weight",l);lo[l].up=jo(js,jl,b);
        snprintf(b,sizeof(b),"model.layers.%d.mlp.down_proj.weight",l);lo[l].dp=jo(js,jl,b);
        snprintf(b,sizeof(b),"model.layers.%d.input_layernorm.weight",l);lo[l].in_off=jo(js,jl,b);
        snprintf(b,sizeof(b),"model.layers.%d.post_attention_layernorm.weight",l);lo[l].pa_off=jo(js,jl,b);
        snprintf(b,sizeof(b),"model.layers.%d.self_attn.q_norm.weight",l);lo[l].qn_off=jo(js,jl,b);
        snprintf(b,sizeof(b),"model.layers.%d.self_attn.k_norm.weight",l);lo[l].kn_off=jo(js,jl,b);
    }
    uint64_t no=jo(js,jl,"model.norm.weight"),lo_=jo(js,jl,"lm_head.weight");

    printf("Init NPU...\n");Npu npu;if(!npu.init()){fprintf(stderr,"NPU FAIL\n");return 1;}
    printf("NPU ready.\n\nDequant+pack...\n");auto t0=std::chrono::steady_clock::now();

    // Dequantize all weights (keep as float, NO BFP16 pre-packing)
    // Q: [2048][1024]  K: [1024][1024]  V: [1024][1024]
    // O: [2048][1024]  G: [3072][1024]  U: [3072][1024]  D: [3072][1024]
    // We pack to BFP16 at GEMM time, one column-slice at a time
    struct FW{std::vector<float>q,k,v,o,g,u,d;}fw[28];
    float*lf;int lr,lc;
    for(int l=0;l<NC;l++){
        int rows,unused;
        #define DQ(dst,offset,nrows) do{ \
            float*p=dequant_i8_to_float(i8p(offset),nrows,&rows,&unused); \
            dst.assign(p,p+rows*H);free(p);}while(0)
        DQ(fw[l].q,lo[l].qp,256);  // [2048][1024]
        DQ(fw[l].k,lo[l].kp,128);  // [1024][1024]
        DQ(fw[l].v,lo[l].vp,128);  // [1024][1024]
        DQ(fw[l].o,lo[l].op,256);  // [2048][1024]
        DQ(fw[l].g,lo[l].gp,384);  // [3072][1024]
        DQ(fw[l].u,lo[l].up,384);  // [3072][1024]
        DQ(fw[l].d,lo[l].dp,384);  // [3072][1024]
        if(l%7==0)printf("  L%d\n",l);
    }
    lf=dequant_i8_to_float(i8p(lo_),18992,&lr,&lc);
    printf("Ready %.1fs\n\n",std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count());

    ri(HD,1000000.0f,4096);

    auto gemm=[&](const float*A,int K,int N,const float*W,int w_rows,float*C){
        npu.gemm(A,1,K,W,w_rows,N,C);
    };

    struct KV{std::vector<float>k,v;int n;KV():k(4096*NKV*HD),v(4096*NKV*HD),n(0){}};
    std::vector<KV>kv(NC);
    std::vector<float>h(H),qo(2048),ko(1024),vo(1024),at(2048),oo(H),gt(3072),su(3072),dw(H),lg(NV),sb(H);

    int pt[]={151643,872,198,11852,151644,198,151643,77091,198};int npt=9,sp=0;
    printf("=== Prefill %d ===\n",npt);
    for(int pi=0;pi<npt;pi++){
        for(int i=0;i<H;i++)h[i]=bf16_g(emb[pt[pi]*H+i]);
        for(int l=0;l<NC;l++){
            memcpy(sb.data(),h.data(),H*4);rn(h.data(),(const uint16_t*)(md+df+lo[l].in_off),H);
            gemm(h.data(),H,2048,fw[l].q.data(),H,qo.data());
            gemm(h.data(),H,1024,fw[l].k.data(),H,ko.data());
            gemm(h.data(),H,1024,fw[l].v.data(),H,vo.data());
            auto qn=(const uint16_t*)(md+df+lo[l].qn_off),kn=(const uint16_t*)(md+df+lo[l].kn_off);
            for(int hh=0;hh<NH;hh++){double sq=0;for(int d=0;d<HD;d++)sq+=qo[hh*HD+d]*qo[hh*HD+d];
                float iq=1.0f/sqrtf((float)(sq/HD)+EPS);for(int d=0;d<HD;d++)qo[hh*HD+d]*=iq*bf16_g(qn[d]);ra(&qo[hh*HD],HD,sp);
                if(hh%GQA==0){int kvh=hh/GQA;double sk=0;for(int d=0;d<HD;d++)sk+=ko[kvh*HD+d]*ko[kvh*HD+d];
                    float ik=1.0f/sqrtf((float)(sk/HD)+EPS);for(int d=0;d<HD;d++)ko[kvh*HD+d]*=ik*bf16_g(kn[d]);ra(&ko[kvh*HD],HD,sp);
                    memcpy(&kv[l].k[sp*NKV*HD+kvh*HD],&ko[kvh*HD],HD*4);memcpy(&kv[l].v[sp*NKV*HD+kvh*HD],&vo[kvh*HD],HD*4);}}
            kv[l].n=sp+1;int cl=kv[l].n;
            for(int hh=0;hh<NH;hh++){int kvh=hh/GQA;std::vector<float>sc(cl);
                for(int p=0;p<cl;p++){double s=0;for(int d=0;d<HD;d++)s+=qo[hh*HD+d]*kv[l].k[p*NKV*HD+kvh*HD+d];sc[p]=(float)(s/sqrtf(HD));}
                float mx=sc[0];for(int i=1;i<cl;i++)if(sc[i]>mx)mx=sc[i];double sum=0;for(int i=0;i<cl;i++){sc[i]=expf(sc[i]-mx);sum+=sc[i];}
                float is=1.0f/(float)sum;for(int i=0;i<cl;i++)sc[i]*=is;
                for(int d=0;d<HD;d++){float s=0;for(int p=0;p<cl;p++)s+=sc[p]*kv[l].v[p*NKV*HD+kvh*HD+d];at[hh*HD+d]=s;}}
            gemm(at.data(),NH*HD,H,fw[l].o.data(),NH*HD,oo.data());
            for(int i=0;i<H;i++)h[i]=sb[i]+oo[i];
            memcpy(sb.data(),h.data(),H*4);rn(h.data(),(const uint16_t*)(md+df+lo[l].pa_off),H);
            gemm(h.data(),H,IM,fw[l].g.data(),H,gt.data());
            gemm(h.data(),H,IM,fw[l].u.data(),H,su.data());
            for(int i=0;i<IM;i++)su[i]=(gt[i]/(1.0f+expf(-gt[i])))*su[i];
            gemm(su.data(),IM,H,fw[l].d.data(),IM,dw.data());
            for(int i=0;i<H;i++)h[i]=sb[i]+dw[i];
        }
        sp++;if(pi%3==0)printf("  %d/%d\n",pi+1,npt);
    }
    printf("Done (%d tokens, pos=%d) hn=%.2f\n\n",npt,sp,fn2(h.data(),H));

    printf("=== Generate ===\n");int ng=8;auto tgs=std::chrono::steady_clock::now();
    for(int st=0;st<ng;st++){auto ts=std::chrono::steady_clock::now();
        for(int l=0;l<NC;l++){
            memcpy(sb.data(),h.data(),H*4);rn(h.data(),(const uint16_t*)(md+df+lo[l].in_off),H);
            gemm(h.data(),H,2048,fw[l].q.data(),H,qo.data());
            gemm(h.data(),H,1024,fw[l].k.data(),H,ko.data());
            gemm(h.data(),H,1024,fw[l].v.data(),H,vo.data());
            auto qn=(const uint16_t*)(md+df+lo[l].qn_off),kn=(const uint16_t*)(md+df+lo[l].kn_off);
            for(int hh=0;hh<NH;hh++){double sq=0;for(int d=0;d<HD;d++)sq+=qo[hh*HD+d]*qo[hh*HD+d];
                float iq=1.0f/sqrtf((float)(sq/HD)+EPS);for(int d=0;d<HD;d++)qo[hh*HD+d]*=iq*bf16_g(qn[d]);ra(&qo[hh*HD],HD,sp);
                if(hh%GQA==0){int kvh=hh/GQA;double sk=0;for(int d=0;d<HD;d++)sk+=ko[kvh*HD+d]*ko[kvh*HD+d];
                    float ik=1.0f/sqrtf((float)(sk/HD)+EPS);for(int d=0;d<HD;d++)ko[kvh*HD+d]*=ik*bf16_g(kn[d]);ra(&ko[kvh*HD],HD,sp);
                    memcpy(&kv[l].k[sp*NKV*HD+kvh*HD],&ko[kvh*HD],HD*4);memcpy(&kv[l].v[sp*NKV*HD+kvh*HD],&vo[kvh*HD],HD*4);}}
            kv[l].n=sp+1;int cl=kv[l].n;
            for(int hh=0;hh<NH;hh++){int kvh=hh/GQA;std::vector<float>sc(cl);
                for(int p=0;p<cl;p++){double s=0;for(int d=0;d<HD;d++)s+=qo[hh*HD+d]*kv[l].k[p*NKV*HD+kvh*HD+d];sc[p]=(float)(s/sqrtf(HD));}
                float mx=sc[0];for(int i=1;i<cl;i++)if(sc[i]>mx)mx=sc[i];double sum=0;for(int i=0;i<cl;i++){sc[i]=expf(sc[i]-mx);sum+=sc[i];}
                float is=1.0f/(float)sum;for(int i=0;i<cl;i++)sc[i]*=is;
                for(int d=0;d<HD;d++){float s=0;for(int p=0;p<cl;p++)s+=sc[p]*kv[l].v[p*NKV*HD+kvh*HD+d];at[hh*HD+d]=s;}}
            gemm(at.data(),NH*HD,H,fw[l].o.data(),NH*HD,oo.data());
            for(int i=0;i<H;i++)h[i]=sb[i]+oo[i];
            memcpy(sb.data(),h.data(),H*4);rn(h.data(),(const uint16_t*)(md+df+lo[l].pa_off),H);
            gemm(h.data(),H,IM,fw[l].g.data(),H,gt.data());
            gemm(h.data(),H,IM,fw[l].u.data(),H,su.data());
            for(int i=0;i<IM;i++)su[i]=(gt[i]/(1.0f+expf(-gt[i])))*su[i];
            gemm(su.data(),IM,H,fw[l].d.data(),IM,dw.data());
            for(int i=0;i<H;i++)h[i]=sb[i]+dw[i];
        }
        memcpy(sb.data(),h.data(),H*4);auto fnw=(const uint16_t*)(md+df+no);rn(sb.data(),fnw,H);
        for(int n=0;n<NV;n++){double s=0;for(int k=0;k<H;k++){uint16_t r=emb[n*H+k];if((r&0x7F80)!=0x7F80)s+=sb[k]*bf16f(r);}lg[n]=(float)s;}
        float mx=lg[0],mn=lg[0];for(int i=1;i<NV;i++){if(lg[i]>mx)mx=lg[i];if(lg[i]<mn)mn=lg[i];}
        double sum=0;for(int i=0;i<NV;i++){lg[i]=expf((lg[i]-mx)/TEMP);sum+=lg[i];}
        float rr=(float)rand()/RAND_MAX*(float)sum,acc=0;int tok=0;for(int i=0;i<NV;i++){acc+=lg[i];if(acc>=rr){tok=i;break;}}
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ts).count();
        printf("  [%d] %d (%.0fms) hn=%.2f\n",st,tok,ms,fn2(h.data(),H));
        for(int i=0;i<H;i++)h[i]=bf16_g(emb[tok*H+i]);sp++;
    }
    double ts=std::chrono::duration<double>(std::chrono::steady_clock::now()-tgs).count();
    printf("\n=== %.0f ms/tok ===\n",ts*1000/ng);
    free(lf);munmap(md,st.st_size);return 0;
}

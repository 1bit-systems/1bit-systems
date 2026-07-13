/** NPU Engine GGUF — loads Q8_0-extracted INT8 weights directly into 4-live contexts */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

static float bf16f(uint16_t v){uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}
static float bf16g(uint16_t v){return(v&0x7F80)==0x7F80?0.0f:bf16f(v);}
static constexpr int H=1024,NC=28,NH=16,NKV=8,HD=128,IM=3072,NV=151936,GQA=2;
static constexpr float EPS=1e-6f;static constexpr int XM=128;

static inline void cn(float*x,int n){for(int i=0;i<n;i++)if(!std::isfinite(x[i]))x[i]=0.0f;}
static inline void sm(float*sc,int n){if(n<=0)return;cn(sc,n);float mx=sc[0];for(int i=1;i<n;i++)if(sc[i]>mx)mx=sc[i];double s=0;for(int i=0;i<n;i++){float d=sc[i]-mx;if(d>80)d=80;else if(d<-80)d=-80;sc[i]=expf(d);s+=sc[i];}if(s<=0){float iv=1.0f/n;for(int i=0;i<n;i++)sc[i]=iv;return;}float is=1.0f/(float)s;for(int i=0;i<n;i++)sc[i]*=is;}
static std::vector<float>rc,rs;static void ri(int hd,float th,int mp){rc.resize(mp*hd);rs.resize(mp*hd);for(int p=0;p<mp;p++)for(int d=0;d<hd;d+=2){float f=1.0f/powf(th,(float)d/hd),a=p*f;rc[p*hd+d]=cosf(a);rs[p*hd+d]=sinf(a);rc[p*hd+d+1]=cosf(a);rs[p*hd+d+1]=sinf(a);}}
static inline void ra(float*x,int hd,int p){for(int d=0;d<hd;d+=2){float a=x[d],b=x[d+1],c=rc[p*hd+d],s=rs[p*hd+d];x[d]=a*c-b*s;x[d+1]=b*c+a*s;}}
static inline void rn_c(float*x,const float*w,int n){cn(x,n);double ss=0;for(int i=0;i<n;i++)if(std::isfinite(x[i]))ss+=(double)x[i]*x[i];float ir=1.0f/sqrtf((float)(ss/n)+EPS);for(int i=0;i<n;i++)x[i]=std::isfinite(x[i])?x[i]*ir*w[i]:0.0f;}

// I8Ctx with pre-loaded per-layer weight BOs (4-live, no swapping)
struct I8Ctx{
    const char*name;int MD,KD,ND;
    std::unique_ptr<xrt::xclbin>xc;std::unique_ptr<xrt::hw_context>hc;std::unique_ptr<xrt::kernel>k;
    std::vector<uint32_t>ins;std::unique_ptr<xrt::bo>bI,bA,bC,layerB[NC];int8_t*Am;int16_t*Cm;
    bool init(xrt::device&d,const char*xp,const char*ip,int gid_B){
        FILE*f=fopen(ip,"rb");if(!f)return false;fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);ins.resize(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);
        xc=std::make_unique<xrt::xclbin>(std::string(xp));d.register_xclbin(*xc);hc=std::make_unique<xrt::hw_context>(d,xc->get_uuid());k=std::make_unique<xrt::kernel>(*hc,"MLIR_AIE");
        bI=std::make_unique<xrt::bo>(d,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k->group_id(1));memcpy(bI->map(),ins.data(),ins.size()*4);bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bA=std::make_unique<xrt::bo>(d,(size_t)MD*KD,XRT_BO_FLAGS_HOST_ONLY,k->group_id(3));bC=std::make_unique<xrt::bo>(d,(size_t)MD*ND*2,XRT_BO_FLAGS_HOST_ONLY,k->group_id(5));
        Am=(int8_t*)bA->map();Cm=(int16_t*)bC->map();
        for(int l=0;l<NC;l++)layerB[l]=std::make_unique<xrt::bo>(d,(size_t)KD*ND,XRT_BO_FLAGS_HOST_ONLY,k->group_id(gid_B));
        return true;
    }
    void loadB(int l,const char*fmt,int K,int N){
        char path[512];snprintf(path,512,fmt,l);
        size_t expect=(size_t)K*N;
        FILE*f=fopen(path,"rb");if(!f){fprintf(stderr,"FAIL: %s\n",path);return;}
        fseek(f,0,2);size_t sz=ftell(f);fseek(f,0,0);
        auto*Bm=(int8_t*)layerB[l]->map();
        memset(Bm,0,(size_t)KD*ND);
        if(sz>=expect)fread(Bm,1,expect,f);else fread(Bm,1,sz,f);
        fclose(f);layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    inline void go(int l,const float*A,int am,int ak,float ascale,float Bscale,float*C,int an){
        float ais=1.0f/ascale;memset(Am,0,(size_t)MD*KD);
        for(int m=0;m<am;m++)for(int k=0;k<ak;k++){float v=A[m*ak+k];if(!std::isfinite(v))v=0;int q=(int)roundf(v*ais);if(q>127)q=127;else if(q<-127)q=-127;Am[m*KD+k]=(int8_t)q;}
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto r=(*k)((unsigned)3,*bI,(unsigned)ins.size(),*bA,*layerB[l],*bC);r.wait();bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        float cs=ascale*Bscale;for(int m=0;m<am;m++)for(int n=0;n<an;n++){float val=(float)Cm[m*ND+n]*cs;if(!std::isfinite(val))val=0;C[m*an+n]=val;}
    }
};

static float load_scale_fixed(const char*path){FILE*f=fopen(path,"r");if(!f)return 1.0f/127.0f;float s;fscanf(f,"%f",&s);fclose(f);return s;}

int main(){
    setvbuf(stdout,NULL,_IONBF,0);
    #define E "/home/bcloud/npu-sandbox/models/gguf_extracted"
    printf("=== NPU Engine GGUF (extracted) ===\n\n");

    // Load embedding table (on-the-fly row dequant for token lookup)
    printf("Loading embedding...\n");
    FILE*f_emb=fopen(E"/embd.i8","rb");if(!f_emb){perror("embd");return 1;}
    fseek(f_emb,0,2);size_t emb_sz=ftell(f_emb);fseek(f_emb,0,0);
    uint8_t*embd=(uint8_t*)mmap(NULL,emb_sz,PROT_READ,MAP_PRIVATE,fileno(f_emb),0);
    float emb_scale=load_scale_fixed(E"/embd.i8.scale");
    printf("  %zu bytes, scale=%.6f\n",emb_sz,emb_scale);
    size_t row_size=emb_sz/1024; // 151936 bytes per row

    // Init 4 NPU contexts
    printf("Init 4 INT8 contexts...\n");xrt::device dev(0);
    #define D "/home/bcloud/npu-sandbox/npu-infer/build/int8"
    I8Ctx cq{"QKV",XM,H,4096},co{"O",XM,NH*HD,H},cg{"GU",XM,H,6144},cd{"D",XM,IM,H};
    cq.init(dev,D"/final_i8_QKV_v.xclbin",D"/insts_i8_QKV_v.txt",4);
    co.init(dev,D"/final_i8_O_v.xclbin",  D"/insts_i8_O_v.txt",  4);
    cg.init(dev,D"/final_i8_GU_v.xclbin", D"/insts_i8_GU_v.txt", 4);
    cd.init(dev,D"/final_i8_D_v.xclbin",  D"/insts_i8_D_v.txt",  4);
    printf("All 4 alive.\n\n");

    // Load pre-extracted weights + scales into BOs
    printf("Loading weights into BOs...\n");auto tp=std::chrono::steady_clock::now();
    struct WS{float qk,o_,g_,d_;}wsc[NC];
    char path[512];
    for(int l=0;l<NC;l++){
        // QKV fused: Q(2M) + K(1M) + V(1M) = 4M elems
        auto*Bq=(int8_t*)cq.layerB[l]->map();
        snprintf(path,512,E"/L%d_attn_q.i8",l);{FILE*f=fopen(path,"rb");fread(Bq,1,2097152,f);fclose(f);}
        snprintf(path,512,E"/L%d_attn_k.i8",l);{FILE*f=fopen(path,"rb");fread(Bq+2097152,1,1048576,f);fclose(f);}
        snprintf(path,512,E"/L%d_attn_v.i8",l);{FILE*f=fopen(path,"rb");fread(Bq+3145728,1,1048576,f);fclose(f);}
        cq.layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        snprintf(path,512,E"/L%d_attn_q.i8.scale",l);wsc[l].qk=load_scale_fixed(path);

        co.loadB(l,E"/L%d_attn_output.i8",2048,1024);
        snprintf(path,512,E"/L%d_attn_output.i8.scale",l);wsc[l].o_=load_scale_fixed(path);

        auto*Bg=(int8_t*)cg.layerB[l]->map();
        snprintf(path,512,E"/L%d_ffn_gate.i8",l);{FILE*f=fopen(path,"rb");fread(Bg,1,3145728,f);fclose(f);}
        snprintf(path,512,E"/L%d_ffn_up.i8",l);{FILE*f=fopen(path,"rb");fread(Bg+3145728,1,3145728,f);fclose(f);}
        cg.layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        snprintf(path,512,E"/L%d_ffn_gate.i8.scale",l);wsc[l].g_=load_scale_fixed(path);

        cd.loadB(l,E"/L%d_ffn_down.i8",3072,1024);
        snprintf(path,512,E"/L%d_ffn_down.i8.scale",l);wsc[l].d_=load_scale_fixed(path);

        if(l%7==0)printf("  L%d\n",l);
    }
    printf("  %.0fms\n\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tp).count());

    // Load norm weights
    float in_n[NC][H],pa_n[NC][H],qn_w[NC][HD],kn_w[NC][HD],fin[H];
    for(int l=0;l<NC;l++){
        snprintf(path,512,E"/L%d_attn_norm.f32",l);{FILE*f=fopen(path,"rb");fread(in_n[l],4,H,f);fclose(f);}
        snprintf(path,512,E"/L%d_ffn_norm.f32",l);{FILE*f=fopen(path,"rb");fread(pa_n[l],4,H,f);fclose(f);}
        snprintf(path,512,E"/L%d_attn_q_norm.f32",l);{FILE*f=fopen(path,"rb");fread(qn_w[l],4,HD,f);fclose(f);}
        snprintf(path,512,E"/L%d_attn_k_norm.f32",l);{FILE*f=fopen(path,"rb");fread(kn_w[l],4,HD,f);fclose(f);}
    }
    {FILE*f=fopen(E"/final_norm.f32","rb");fread(fin,4,H,f);fclose(f);}
    printf("Norms loaded.\n\n");

    ri(HD,1000000.0f,4096);
    struct KV{std::vector<float>k,v;int n;KV():k(4096*NKV*HD),v(4096*NKV*HD),n(0){}};std::vector<KV>kv(NC);
    std::vector<float>h(H),qo(4096),ko(1024),vo(1024),at(2048),oo(H),gt(6144),su(IM),dwo(H),lg(NV),sb(H),sc(4096);
    int sp=0;

    auto emb_row=[&](int tok,float*dst){
        const uint8_t*s=embd+tok*row_size;
        for(int i=0;i<H;i++)dst[i]=(float)((int8_t)(s[i]-128))*emb_scale;
    };

    auto layer=[&](int l){
        memcpy(sb.data(),h.data(),H*4);rn_c(h.data(),in_n[l],H);
        cq.go(l,h.data(),1,H,5.0f/127.0f,wsc[l].qk,qo.data(),4096);cn(qo.data(),4096);
        memcpy(ko.data(),&qo[2048],4096);memcpy(vo.data(),&qo[3072],4096);
        float*qn=qn_w[l],*kn=kn_w[l];
        for(int hh=0;hh<NH;hh++){double sq=0;for(int d=0;d<HD;d++)sq+=qo[hh*HD+d]*qo[hh*HD+d];float iq=1.0f/sqrtf((float)(sq/HD)+EPS);for(int d=0;d<HD;d++)qo[hh*HD+d]*=iq*qn[d];ra(&qo[hh*HD],HD,sp);
            if(hh%GQA==0){int kvh=hh/GQA;double sk=0;for(int d=0;d<HD;d++)sk+=ko[kvh*HD+d]*ko[kvh*HD+d];float ik=1.0f/sqrtf((float)(sk/HD)+EPS);for(int d=0;d<HD;d++)ko[kvh*HD+d]*=ik*kn[d];ra(&ko[kvh*HD],HD,sp);memcpy(&kv[l].k[sp*NKV*HD+kvh*HD],&ko[kvh*HD],HD*4);memcpy(&kv[l].v[sp*NKV*HD+kvh*HD],&vo[kvh*HD],HD*4);}}
        kv[l].n=sp+1;int cl=kv[l].n;
        for(int hh=0;hh<NH;hh++){int kvh=hh/GQA;float*ss=sc.data();
            for(int p=0;p<cl;p++){double s=0;for(int d=0;d<HD;d++)s+=qo[hh*HD+d]*kv[l].k[p*NKV*HD+kvh*HD+d];ss[p]=(float)(s/sqrtf(HD));}
            sm(ss,cl);for(int d=0;d<HD;d++){float s=0;for(int p=0;p<cl;p++)s+=ss[p]*kv[l].v[p*NKV*HD+kvh*HD+d];at[hh*HD+d]=s;}}
        co.go(l,at.data(),1,NH*HD,5.0f/127.0f,wsc[l].o_,oo.data(),H);cn(oo.data(),H);for(int i=0;i<H;i++)h[i]=sb[i]+oo[i];
        memcpy(sb.data(),h.data(),H*4);rn_c(h.data(),pa_n[l],H);
        cg.go(l,h.data(),1,H,5.0f/127.0f,wsc[l].g_,gt.data(),6144);cn(gt.data(),6144);
        for(int i=0;i<IM;i++){float gv=gt[i];if(!std::isfinite(gv))gv=0;su[i]=(gv/(1.0f+expf(-gv)))*gt[IM+i];}
        cd.go(l,su.data(),1,IM,5.0f/127.0f,wsc[l].d_,dwo.data(),H);cn(dwo.data(),H);for(int i=0;i<H;i++)h[i]=sb[i]+dwo[i];
    };

    int pt[]={151643,872,198,11852,151644,198,151643,77091,198};int npt=9;
    printf("=== Prefill %d ===\n",npt);
    for(int pi=0;pi<npt;pi++){emb_row(pt[pi],h.data());for(int l=0;l<NC;l++)layer(l);sp++;if(pi%3==0)printf("  %d/%d\n",pi+1,npt);}
    printf("Done\n\n");
    printf("=== Generate ===\n");int ng=8;auto tgs=std::chrono::steady_clock::now();
    for(int st=0;st<ng;st++){auto ts=std::chrono::steady_clock::now();
        for(int l=0;l<NC;l++)layer(l);
        memcpy(sb.data(),h.data(),H*4);rn_c(sb.data(),fin,H);
        for(int n=0;n<NV;n++){double s=0;const uint8_t*er=embd+n*row_size;for(int k=0;k<H;k++){float v=(float)((int8_t)(er[k]-128))*emb_scale;if(std::isfinite(v))s+=sb[k]*v;}lg[n]=(float)s;}
        float mx=lg[0];for(int i=1;i<NV;i++)if(lg[i]>mx)mx=lg[i];double sum=0;for(int i=0;i<NV;i++){float d=lg[i]-mx;if(d<-80)d=-80;lg[i]=expf(d);sum+=lg[i];}
        float rr=(float)rand()/RAND_MAX*(float)sum,acc=0;int tok=0;for(int i=0;i<NV;i++){acc+=lg[i];if(acc>=rr){tok=i;break;}}
        double mss=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ts).count();
        printf("  [%d] %d (%.0fms)\n",st,tok,mss);emb_row(tok,h.data());sp++;}
    double tts=std::chrono::duration<double>(std::chrono::steady_clock::now()-tgs).count();
    printf("\n=== %.0f ms/tok ===\n",tts*1000/ng);
    munmap(embd,emb_sz);return 0;
}

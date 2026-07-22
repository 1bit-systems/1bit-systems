#define _GNU_SOURCE
#include <stdint.h>
#include <stdarg.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>

static int H=1024,NC=28,NH=16,NKV=8,HD=128,IM=3072,NV=151936,GQA,QD,KD;
static pid_t wp=0; static int wsi=-1,wso=-1,resp_r=-1,resp_w=-1;
static volatile int ready=0; static float EPS=1e-6f;

// Q4NX loader
static float *em,*fn,*lme; static float *inw[256],*paw[256];
static uint8_t*md=NULL; static uint64_t ds=0;
static const char*mh=NULL; static size_t mhlen=0;
static float bf(uint16_t v){uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}
static uint64_t fo(const char*h,size_t hl,const char*n){
    size_t nl=strlen(n);const char*p=h,*e=h+hl;
    while(p+nl<=e){const char*q=(const char*)memmem(p,e-p,n,nl);if(!q)return UINT64_MAX;
    if((q==h||*(q-1)=='"')&&*(q+nl)=='"'){const char*o=(const char*)memmem(q,e-q,"\"data_offsets\"",14);
    if(o){const char*b=(const char*)memchr(o,'[',e-o);if(b)return strtoull(b+1,NULL,10);}}p=q+1;}return UINT64_MAX;}
static float*rd(const char*n,int c){if(!mh)return NULL;uint64_t o=fo(mh,mhlen,n);
    if(o==UINT64_MAX)return NULL;float*d=calloc(c,4);uint16_t*s=(uint16_t*)(md+ds+o);for(int i=0;i<c;i++)d[i]=bf(s[i]);return d;}
static int load(const char*p){
    int f=open(p,O_RDONLY);struct stat st;fstat(f,&st);md=mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,f,0);close(f);
    if(!md)return -1;uint64_t hz;memcpy(&hz,md,8);ds=8+hz;mh=(const char*)(md+8);mhlen=hz;
    if(getenv("NPU_H"))H=atoi(getenv("NPU_H"));if(getenv("NPU_NC"))NC=atoi(getenv("NPU_NC"));
    if(getenv("NPU_NH"))NH=atoi(getenv("NPU_NH"));if(getenv("NPU_NKV"))NKV=atoi(getenv("NPU_NKV"));
    if(getenv("NPU_HD"))HD=atoi(getenv("NPU_HD"));if(getenv("NPU_IM"))IM=atoi(getenv("NPU_IM"));
    if(getenv("NPU_NV"))NV=atoi(getenv("NPU_NV"));GQA=NH/NKV;QD=NH*HD;KD=NKV*HD;
    em=rd("model.embed_tokens.weight",NV*H);fn=rd("model.norm.weight",H);lme=rd("lm_head.weight",NV*H);if(!lme)lme=em;
    char bn[128];for(int l=0;l<NC&&l<256;l++){snprintf(bn,sizeof(bn),"model.layers.%d.input_layernorm.weight",l);inw[l]=rd(bn,H);
    snprintf(bn,sizeof(bn),"model.layers.%d.post_attention_layernorm.weight",l);paw[l]=rd(bn,H);}return em?0:-1;}

// Worker
static void* drain(void*a){
    int fd=*(int*)a;free(a);int m=1048576;fcntl(fd,F_SETPIPE_SZ,&m);
    char buf[131072];ssize_t n;int inited=0;
    while((n=read(fd,buf,sizeof(buf)))>0){
        if(!inited){
            char* rp = strstr(buf,"READY\n");
            if(rp){
                ready=1;inited=1;
                // Forward any data after READY marker
                size_t skip = rp - buf + 6;
                if((size_t)n > skip){
                    if(write(resp_w,buf+skip,n-skip)!=n-skip)break;
                }
            }
        }else{
            if(write(resp_w,buf,n)!=n)break;
        }
    }
    close(fd);close(resp_w);return NULL;}
static void* rdyst(void*a){
    int fd=*(int*)a;free(a);char buf[4096];FILE*f=fdopen(fd,"r");
    if(!f)return NULL;while(fgets(buf,sizeof(buf),f))if(strstr(buf,"WORKER_READY"))break;fclose(f);return NULL;}

static int spawn(const char*mo,const char*ta){
    const char*bin=getenv("NPU_ENGINE_BIN");char def[1024];
    if(!bin){snprintf(def,sizeof(def),"%s/1bit-systems/engine/npu/build/npu_engine_universal",getenv("HOME"));bin=def;}
    int tc[2],fc[2],ec[2],rp[2];
    if(pipe(tc)<0||pipe(fc)<0||pipe(ec)<0||pipe(rp)<0)return -1;
    resp_r=rp[0];resp_w=rp[1];
    wp=fork();
    if(wp<0)return -1;
    if(wp==0){close(tc[1]);dup2(tc[0],0);close(tc[0]);close(fc[0]);dup2(fc[1],1);close(fc[1]);close(ec[0]);dup2(ec[1],2);close(ec[1]);close(rp[0]);close(rp[1]);execlp(bin,bin,mo,"--model-tag",ta,"--worker",NULL);_exit(1);}
    close(tc[0]);close(fc[1]);close(ec[1]);wsi=tc[1];wso=fc[0];
    int*df=malloc(sizeof(int));*df=fc[0];pthread_t dt;
    if(pthread_create(&dt,NULL,drain,df)!=0){free(df);close(fc[0]);}
    else pthread_detach(dt);
    int*ef=malloc(sizeof(int));*ef=ec[0];pthread_t et;
    if(pthread_create(&et,NULL,rdyst,ef)!=0){free(ef);close(ec[0]);}
    else pthread_detach(et);
    return 0;}

static int gemm(int op,int l,int b,int id,const float*in,float*out,int*od){
    uint32_t h[4]={(uint32_t)op,(uint32_t)l,(uint32_t)b,(uint32_t)id};
    if(write(wsi,h,16)!=16){fprintf(stderr,"W1FAIL\n");return -1;}
    if(write(wsi,in,(size_t)b*id*4)!=(ssize_t)(b*id*4)){fprintf(stderr,"W2FAIL\n");return -1;}
    uint32_t r[2];size_t rr=0;while(rr<8){ssize_t n=read(resp_r,(char*)r+rr,8-rr);if(n<=0){if(n<0&&errno==EINTR)continue;fprintf(stderr,"RFAIL(%zd)\n",n);return -1;}rr+=n;}
    if(r[0]!=0)return -1;*od=(int)r[1];
    size_t dr=0;while(dr<(size_t)(*od)*4){ssize_t n=read(resp_r,(char*)out+dr,(size_t)(*od)*4-dr);if(n<=0){if(n<0&&errno==EINTR)continue;return -1;}dr+=n;}
    return 0;}

// Math + inference (same as before)
static void cn(float*x,int n){for(int i=0;i<n;i++)if(!isfinite(x[i]))x[i]=0;}
static float silu(float x){return x/(1.0f+expf(-x));}
static void rn(float*x,const float*w,int n){double ss=0;for(int i=0;i<n;i++)ss+=(double)x[i]*x[i];
    float ir=1.0f/sqrtf((float)(ss/n)+EPS);for(int i=0;i<n;i++)x[i]*=ir*w[i];}
static float*cc,*sc;
static void mk_rope(int ms){cc=calloc((size_t)ms*HD,4);sc=calloc((size_t)ms*HD,4);int h2=HD/2;
    for(int p=0;p<ms;p++)for(int d=0;d<h2;d++){float a=p*powf(1000000.0f,-2.0f*d/HD);
    cc[p*HD+d]=cosf(a);sc[p*HD+d]=sinf(a);cc[p*HD+d+h2]=cosf(a);sc[p*HD+d+h2]=sinf(a);}}
static void rp(float*x,int pos){int h2=HD/2;for(int d=0;d<h2;d++){float a=x[d],b=x[d+h2];
    x[d]=a*cc[pos*HD+d]-b*sc[pos*HD+d];x[d+h2]=b*cc[pos*HD+d]+a*sc[pos*HD+d];}}
static void attn(float*q,float*o,int cl,const float*kk,const float*vv,int st){
    #pragma omp parallel for
    for(int h=0;h<NH;h++){int kvh=h/GQA;float sc[16384],mx=-1e30f;
        for(int p=0;p<cl;p++){double s=0;for(int d=0;d<HD;d++)s+=(double)q[h*HD+d]*kk[p*st+kvh*HD+d];
        sc[p]=(float)(s/sqrtf((float)HD));if(sc[p]>mx)mx=sc[p];}
        double sw=0;for(int p=0;p<cl;p++){sc[p]=expf(sc[p]-mx);sw+=sc[p];}float iw=sw>0?1.0f/(float)sw:1.0f/cl;
        for(int d=0;d<HD;d++){double acc=0;for(int p=0;p<cl;p++)acc+=sc[p]*vv[p*st+kvh*HD+d];o[h*HD+d]=(float)acc*iw;}}}
static float*kc[256],*vc[256];static int sl[256],tp=0;
static int fwd(int tok){
    float h[8192],sb[8192],qkv[32768],at[8192],oo[8192],gu[32768],dw[8192];
    memcpy(h,&em[(size_t)tok*H],H*4);
    for(int l=0;l<NC;l++){memcpy(sb,h,H*4);if(inw[l])rn(h,inw[l],H);int od;
        if(gemm(1,l,1,H,h,qkv,&od))return -2;cn(qkv,od);
        for(int hh=0;hh<NH;hh++){float*q=qkv+hh*HD;double sq=0;for(int d=0;d<HD;d++)sq+=(double)q[d]*q[d];
        float iq=1.0f/sqrtf((float)(sq/HD)+EPS);for(int d=0;d<HD;d++)q[d]*=iq;rp(q,tp);}
        for(int kvh=0;kvh<NKV;kvh++){float*k=qkv+QD+kvh*HD;double sk=0;for(int d=0;d<HD;d++)sk+=(double)k[d]*k[d];
        float ik=1.0f/sqrtf((float)(sk/HD)+EPS);for(int d=0;d<HD;d++)k[d]*=ik;rp(k,tp);}
        int sp=sl[l];for(int kvh=0;kvh<NKV;kvh++){memcpy(&kc[l][(size_t)sp*NKV*HD+(size_t)kvh*HD],qkv+QD+kvh*HD,HD*4);
        memcpy(&vc[l][(size_t)sp*NKV*HD+(size_t)kvh*HD],qkv+QD+KD+kvh*HD,HD*4);}
        sl[l]=sp+1;attn(qkv,at,sl[l],kc[l],vc[l],NKV*HD);
        if(gemm(2,l,1,QD,at,oo,&od))return -2;cn(oo,od);
        for(int i=0;i<H;i++)h[i]=sb[i]+oo[i];memcpy(sb,h,H*4);if(paw[l])rn(h,paw[l],H);
        if(gemm(3,l,1,H,h,gu,&od))return -2;cn(gu,od);
        for(int i=0;i<IM;i++)gu[i]=silu(gu[i])*gu[IM+i];
        if(gemm(5,l,1,IM,gu,dw,&od))return -2;cn(dw,od);
        for(int i=0;i<H;i++)h[i]=sb[i]+dw[i];}
    double ss=0;for(int i=0;i<H;i++)ss+=(double)h[i]*h[i];float ir=1.0f/sqrtf((float)(ss/H)+EPS);
    for(int i=0;i<H;i++)h[i]*=ir*fn[i];
    float*lg=malloc((size_t)NV*4);
    #pragma omp parallel for
    for(int n=0;n<NV;n++){double s=0;for(int k=0;k<H;k++)s+=(double)h[k]*lme[(size_t)n*H+k];lg[n]=(float)s;}
    int bst=0;float bv=lg[0];for(int n=1;n<NV;n++)if(lg[n]>bv){bv=lg[n];bst=n;}free(lg);tp++;return bst;}

// HTTP
static void sj(int fd,const char*j,...){char b[65536];va_list ap;va_start(ap,j);int jl=vsnprintf(b,sizeof(b),j,ap);va_end(ap);
    char h[4096];int n=snprintf(h,sizeof(h),"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",jl);
    write(fd,h,n);write(fd,b,jl);close(fd);}
static void*hc(void*a){
    int fd=(int)(long)a;char b[65536];ssize_t n=read(fd,b,sizeof(b)-1);if(n<=0){close(fd);return NULL;}b[n]=0;
    char m[16],p[256];if(sscanf(b,"%15s %255s",m,p)<2){close(fd);return NULL;}
    if(strcmp(m,"GET")==0){sj(fd,"{\"ready\":%d}",ready);return NULL;}
    if(*m=='P'&&strstr(p,"chat/completions")){
        char*bb=strstr(b,"\r\n\r\n");if(!bb){close(fd);return NULL;}bb+=4;
        char pr[4096]={};char*c=strstr(bb,"\"content\"");
        if(c){c=strchr(c,':');if(c){c++;while(*c==' '||*c=='"')c++;char*e=strchr(c,'"');if(e)*e=0;strncpy(pr,c,sizeof(pr)-1);}}
        if(!pr[0])strcpy(pr,"hello");
        if(!ready){sj(fd,"{\"error\":\"not ready\"}");return NULL;}
        int it[512],nt=0;for(int i=0;pr[i]&&nt<512;i++)it[nt++]=(unsigned char)pr[i];
        tp=0;for(int l=0;l<NC;l++)sl[l]=0;int last=it[0]>0?it[0]:0;
        for(int i=0;i<nt;i++)last=it[i];tp=0;for(int l=0;l<NC;l++)sl[l]=0;
        int ot[1024],no=0;for(int i=0;i<16;i++){int nx=fwd(last);if(nx<0)break;ot[no++]=nx;if(nx==2)break;last=nx;}
        char rp[16384];int rl=0;
        rl+=snprintf(rp+rl,sizeof(rp)-rl,"{\"id\":\"1\",\"object\":\"chat.completion\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"[%d tokens:",no);
        for(int i=0;i<no&&i<16;i++)rl+=snprintf(rp+rl,sizeof(rp)-rl,"%d ",ot[i]);
        rl+=snprintf(rp+rl,sizeof(rp)-rl,"]\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d}}",nt,no);
        sj(fd,"%s",rp);return NULL;}
    close(fd);return NULL;}

int main(int argc,char**argv){
    int port=argc>1?atoi(argv[1]):9091;
    const char*mo=getenv("NPU_MODEL_PATH");const char*ta=getenv("NPU_MODEL_TAG")?:"qwen3_0_6b";
    if(!mo){fprintf(stderr,"Set NPU_MODEL_PATH\n");return 1;}
    signal(SIGCHLD,SIG_IGN);signal(SIGPIPE,SIG_IGN);
    fprintf(stderr,"Loading...\n");if(load(mo)<0){fprintf(stderr,"Model fail\n");return 1;}
    fprintf(stderr,"  %dx%d H=%d\n",NC,H,H);
    for(int l=0;l<NC&&l<256;l++){kc[l]=calloc((size_t)4096*NKV*HD,4);vc[l]=calloc((size_t)4096*NKV*HD,4);}
    mk_rope(4096);
    fprintf(stderr,"Starting worker...\n");spawn(mo,ta);
    int fd=socket(AF_INET,SOCK_STREAM,0);int opt=1;
    setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&opt,4);
    struct sockaddr_in a={.sin_family=AF_INET,.sin_port=htons(port),.sin_addr={INADDR_ANY}};
    if(bind(fd,(struct sockaddr*)&a,sizeof(a))<0||listen(fd,16)<0){perror("bind");return 1;}
    fprintf(stderr,"\nNPU chat: http://127.0.0.1:%d (zero FLM)\n",port);
    while(1){struct sockaddr_in ca;socklen_t cl=sizeof(ca);int cf=accept(fd,(struct sockaddr*)&ca,&cl);
        if(cf<0)break;pthread_t t;pthread_create(&t,NULL,hc,(void*)(long)cf);pthread_detach(t);}
    return 0;}

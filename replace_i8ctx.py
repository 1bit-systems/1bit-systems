#!/usr/bin/env python3
with open('engine/npu/src/npu_engine_universal.cpp') as f:
    content = f.read()

# Find the start and end of I8Ctx struct (up to the go() function)
idx = content.find('struct I8Ctx{int MD,KD,ND,NL;bool use_bf16=false;')
if idx < 0:
    print("Not found")
    exit(1)

# Find the end of the I8Ctx (the go function closing brace)
go_marker = 'inline bool go('
go_idx = content.find(go_marker, idx)
# Find the closing of go
end_of_go = content.find('return true;}', go_idx)
# Find the newline after return true;}
end_brace = content.find('\n', end_of_go)
if end_brace < 0:
    end_brace = end_of_go + len('return true;}')

# Extract the old I8Ctx def
old_text = content[idx:end_brace]

new_text = '''struct I8Ctx{int MD,KD,ND,NL;
    std::unique_ptr<xrt::xclbin>xc;
    std::unique_ptr<xrt::hw_context>hc;
    std::unique_ptr<xrt::kernel>k;
    std::unique_ptr<xrt::bo>bI,bA,bB,bC;
    int8_t*Am;int16_t*Cm;
    bool initialized=false;
    ~I8Ctx(){}
    bool isReady(){return initialized&&k&&bA&&bC;}
    size_t a_size() const { return (size_t)MD*KD; }
    size_t b_size() const { return (size_t)KD*ND; }
    size_t c_size() const { return (size_t)MD*ND*2; }
    bool init(xrt::device&d,const char*xp,int nlayers){
        NL=nlayers;
        try{
        fprintf(stderr,"  I8Ctx: loading xclbin %s\\n",xp);
        xc=std::make_unique<xrt::xclbin>(std::string(xp));
        d.register_xclbin(*xc);
        hc=std::make_unique<xrt::hw_context>(d,xc->get_uuid());
        k=std::make_unique<xrt::kernel>(*hc,"MLIR_AIE");
        size_t max_inst = 32768;
        bI=std::make_unique<xrt::bo>(d,max_inst,XCL_BO_FLAGS_CACHEABLE,k->group_id(1));
        bA=std::make_unique<xrt::bo>(d,a_size(),XCL_BO_FLAGS_CACHEABLE,k->group_id(3));
        bB=std::make_unique<xrt::bo>(d,b_size(),XRT_BO_FLAGS_HOST_ONLY,k->group_id(4));
        bC=std::make_unique<xrt::bo>(d,c_size(),XCL_BO_FLAGS_CACHEABLE,k->group_id(5));
        Am=(int8_t*)bA->map();Cm=(int16_t*)bC->map();
        }catch(std::exception&e){fprintf(stderr,"  I8Ctx::init: %s (%s)\\n",e.what(),xp);return false;}
        initialized=true;return true;}
    void packB(const float*w,int K,int N,float&sout){
        auto Bm = (int8_t*)bB->map();
        float amax=0;
        for(int i=0;i<K*N;i++){float a=fabsf(w[i]);if(std::isfinite(a)&&a>amax)amax=a;}
        if(amax<1e-12f)amax=1.0f;sout=amax/127.0f;float is=127.0f/amax;
        for(int i=0;i<K*N;i++){float v=w[i];if(!std::isfinite(v))v=0;
            int x=(int)roundf(v*is);if(x>127)x=127;else if(x<-127)x=-127;Bm[i]=(int8_t)x;}}
    inline int8_t* quantize_async(const float*A,int am,int ak,float ascale){
        float ais=1.0f/ascale;memset(Am,0,(size_t)am*KD);
        for(int m=0;m<am;m++)for(int k=0;k<ak;k++){
            float v=A[m*ak+k];if(!std::isfinite(v))v=0;
            int q=(int)roundf(v*ais);if(q>127)q=127;else if(q<-127)q=-127;
            Am[m*KD+k]=(int8_t)q;}
        return Am;}
    inline xrt::run launch(){return (*k)(3ULL,*bI,(unsigned)32768,*bA,*bB,*bC);}
    inline xrt::run sync_and_launch(){bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);bC->sync(XCL_BO_SYNC_BO_TO_DEVICE);return (*k)(3ULL,*bI,(unsigned)32768,*bA,*bB,*bC);}
    inline void dequantize(xrt::run& r,float*C,int am,int an,float ascale,float Bscale){
        r.wait();bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        float cs=ascale*Bscale;
        for(int m=0;m<am;m++)for(int n=0;n<an;n++){
            float val=(float)Cm[m*ND+n]*cs;if(!std::isfinite(val))val=0;C[m*an+n]=val;}}
    inline bool go(const float*A,int am,int ak,float ascale,float Bscale,float*C,int an){
        quantize_async(A,am,ak,ascale);auto r=sync_and_launch();r.wait();
        dequantize(r,C,am,an,ascale,Bscale);return true;}'''

content = content[:idx] + new_text + content[end_brace:]

with open('engine/npu/src/npu_engine_universal.cpp', 'w') as f:
    f.write(content)
print("I8Ctx replaced")

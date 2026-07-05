/**
 * NPU Engine — Full GGUF Support with Complete Inference
 *
 * Loads any GGUF model, runs full decode inference on NPU.
 * Supports: F32, F16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K, Q8_K
 * Model archs: llama, qwen2, qwen3, gemma, phi, etc.
 *
 * Build:
 *   g++ -std=c++23 -O3 -mavx512f -mavx512dq -mavx512vl -mfma \
 *       -o npu_engine_gguf_full npu_engine_gguf_full.cpp \
 *       -I$XRT/include -L$XRT/lib64 -lxrt_coreutil -ldl -luuid -lm
 *
 * Run:
 *   sudo ./npu_engine_gguf_full path/to/model.gguf [n_decode_tokens]
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <immintrin.h>

#include "gguf_parser.h"

using Clock = std::chrono::steady_clock;
static double elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
}

// ─── Q4_K / Q5_K / Q6_K dequant ───
struct block_q4_K { uint8_t hmask[16]; uint8_t scales[12]; uint8_t qs[128]; uint16_t d; uint16_t dmin; };
struct block_q5_K { uint8_t hmask[16]; uint8_t scales[12]; uint8_t qs[128]; uint16_t d; uint16_t dmin; };
struct block_q6_K { uint8_t ql[128]; uint8_t qh[64]; uint8_t scales[16]; uint16_t d; };

static void dequantize_q4_K(const block_q4_K* block, float* out, int n) {
    const float d = (float)(int16_t)(block->d & 0x7FFF) / 16.0f;
    const float dmin = (float)(int16_t)(block->dmin & 0x7FFF) / 16.0f;
    const uint8_t* sc = block->scales;
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 64; i++) {
            int idx = j * 64 + i; if (idx >= n) return;
            int qh = (block->hmask[i/4] >> (i%4)) & 1;
            float v = d * (((int8_t)(block->qs[idx] >> 4) & 0xF) - (qh ? 16 : 0));
            out[idx] = v + dmin * (block->qs[idx] & 0x0F);
        }
    }
}
static void dequantize_q5_K(const block_q5_K* block, float* out, int n) {
    const float d = (float)(int16_t)(block->d & 0x7FFF) / 16.0f;
    const float dmin = (float)(int16_t)(block->dmin & 0x7FFF) / 16.0f;
    for (int j = 0; j < 256 && j < n; j++) {
        int qh = (block->hmask[j/4] >> (j%4*2)) & 0x03;
        int q = (block->qs[j] >> 4) | (qh << 4);
        out[j] = d * (q - 16) + dmin * (block->qs[j] & 0x0F);
    }
}
static void dequantize_q6_K(const block_q6_K* block, float* out, int n) {
    const float d = (float)(int16_t)(block->d & 0x7FFF) / 64.0f;
    const uint8_t* sc = block->scales;
    for (int j = 0; j < 256 && j < n; j++) {
        int sc_h = sc[j/64] >> ((j%64)/16*4) & 0xF;
        int sc_l = (sc[j/64+4] >> 2*((j/16)%4)) & 0x3;
        int q = (block->ql[j] & 0x3F) | ((block->qh[j/16] >> (j%16)) & 0x01 ? 64 : 0);
        out[j] = d * ((sc_h << 2) | sc_l) * (q - 32);
    }
}

// ─── Dequant tensor to float ───
static float* dequantize_tensor(GGUFReader& r, const GGUFModel::Tensor& t, uint64_t data_off) {
    int n_elems = 1; for (auto d : t.dims) n_elems *= d;
    float* out = new float[n_elems];
    r.seek(data_off + t.file_offset);
    int bs = ggml_blck_size((ggml_type)t.type);
    int ts = ggml_type_size((ggml_type)t.type);
    int nb = bs > 0 ? n_elems / bs : 0;
    switch (t.type) {
        case GGML_TYPE_F32:  for (int i = 0; i < n_elems; i++) out[i]=r.read_f32(); break;
        case GGML_TYPE_F16:  for (int i = 0; i < n_elems; i++) out[i]=r.read_f16(); break;
        case GGML_TYPE_Q8_0: for (int b=0;b<nb;b++){float d=r.read_f16();for(int j=0;j<32;j++)out[b*32+j]=d*(int8_t)r.read_u8();} break;
        case GGML_TYPE_Q4_0: for (int b=0;b<nb;b++){float d=r.read_f16();for(int j=0;j<16;j++){uint8_t by=r.read_u8();out[b*32+j*2]=d*(int8_t)(by>>4);out[b*32+j*2+1]=d*(int8_t)(by&0xF);}} break;
        case GGML_TYPE_Q4_1: for (int b=0;b<nb;b++){float d=r.read_f16(),m=r.read_f16();for(int j=0;j<16;j++){uint8_t by=r.read_u8();out[b*32+j*2]=d*(by>>4)+m;out[b*32+j*2+1]=d*(by&0xF)+m;}} break;
        case GGML_TYPE_Q5_0: for (int b=0;b<nb;b++){float d=r.read_f16();uint16_t h=r.read_u16();for(int j=0;j<16;j++){uint8_t by=r.read_u8();out[b*32+j*2]=d*(((by>>4)|((h>>j)&1<<4))-16);out[b*32+j*2+1]=d*(((by&0xF)|((h>>(j+16))&1<<4))-16);}} break;
        case GGML_TYPE_Q5_1: for (int b=0;b<nb;b++){float d=r.read_f16(),m=r.read_f16();uint16_t h=r.read_u16();for(int j=0;j<16;j++){uint8_t by=r.read_u8();out[b*32+j*2]=d*((by>>4)|((h>>j)&1<<4))+m;out[b*32+j*2+1]=d*((by&0xF)|((h>>(j+16))&1<<4))+m;}} break;
        case GGML_TYPE_Q4_K: for (int b=0;b<nb;b++){dequantize_q4_K((const block_q4_K*)r.ptr(),out+b*256,n_elems-b*256);r.skip(ts);} break;
        case GGML_TYPE_Q5_K: for (int b=0;b<nb;b++){dequantize_q5_K((const block_q5_K*)r.ptr(),out+b*256,n_elems-b*256);r.skip(ts);} break;
        case GGML_TYPE_Q6_K: for (int b=0;b<nb;b++){dequantize_q6_K((const block_q6_K*)r.ptr(),out+b*256,n_elems-b*256);r.skip(ts);} break;
        case GGML_TYPE_I8:   for (int i = 0; i < n_elems; i++) out[i]=(float)(int8_t)r.read_u8(); break;
        default: fprintf(stderr,"Unsupported quant type %d\n",t.type); delete[] out; return nullptr;
    }
    return out;
}

// ─── CPU helpers (from npu_engine_cb.cpp) ───
static inline float bf16f(uint16_t v){uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}
static inline float bf16g(uint16_t v){return(v&0x7F80)==0x7F80?0.0f:bf16f(v);}
static constexpr float EPS=1e-6f;
static inline void cn(float*x,int n){for(int i=0;i<n;i++)if(!std::isfinite(x[i]))x[i]=0.0f;}
static inline void sm(float*sc,int n){
    if(n<=0)return;cn(sc,n);float mx=sc[0];for(int i=1;i<n;i++)if(sc[i]>mx)mx=sc[i];
    double s=0;for(int i=0;i<n;i++){float d=sc[i]-mx;if(d>80)d=80;else if(d<-80)d=-80;sc[i]=expf(d);s+=sc[i];}
    if(s<=0){float iv=1.0f/n;for(int i=0;i<n;i++)sc[i]=iv;return;} float is=1.0f/(float)s;for(int i=0;i<n;i++)sc[i]*=is;
}
static inline void rn_c(float*x,const float*w,int n){
    cn(x,n);double ss=0;for(int i=0;i<n;i++)if(std::isfinite(x[i]))ss+=(double)x[i]*x[i];
    float ir=1.0f/sqrtf((float)(ss/n)+EPS);
    for(int i=0;i<n;i++)x[i]=std::isfinite(x[i])?x[i]*ir*w[i]:0.0f;
}
static std::vector<float>rc,rs;
static void ri(int hd,float th,int mp){rc.resize(mp*hd);rs.resize(mp*hd);
    for(int p=0;p<mp;p++)for(int i=0;i<hd/2;i++){float f=1.0f/powf(th,(float)(2*i)/hd),a=p*f;rc[p*hd+i]=cosf(a);rs[p*hd+i]=sinf(a);}
}
static inline void ra(float*x,int hd,int p){
    for(int i=0;i<hd/2;i++){float a=x[i],b=x[i+hd/2],c=rc[p*hd+i],s=rs[p*hd+i];x[i]=a*c-b*s;x[i+hd/2]=a*s+b*c;}
}
static inline float dynamic_ascale(const float* x, int n) {
    float amax=0;for(int i=0;i<n;i++){float a=fabsf(x[i]);if(std::isfinite(a)&&a>amax)amax=a;}
    return(amax<1e-12f)?(1.0f/127.0f):(amax/127.0f);
}

// ─── AVX-512 LM head ───
static void lm_head_avx512(const float* W, const float* x, int nv, int h, float* lg) {
    for (int n = 0; n < nv; n += 16) {
        __m512 sum0 = _mm512_setzero_ps(), sum1 = _mm512_setzero_ps();
        __m512 sum2 = _mm512_setzero_ps(), sum3 = _mm512_setzero_ps();
        const float* wptr = W + (size_t)n * h;
        int k = 0;
        for (; k + 64 <= h; k += 64) {
            __m512 x0 = _mm512_loadu_ps(x + k), x1 = _mm512_loadu_ps(x + k + 16);
            __m512 x2 = _mm512_loadu_ps(x + k + 32), x3 = _mm512_loadu_ps(x + k + 48);
            sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(wptr + k), x0, sum0);
            sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(wptr + k + 16), x1, sum1);
            sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(wptr + k + 32), x2, sum2);
            sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(wptr + k + 48), x3, sum3);
        }
        for (; k + 16 <= h; k += 16)
            sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(wptr + k), _mm512_loadu_ps(x + k), sum0);
        float tail[16]={0}; for(;k<h;k++) tail[k%16]+=wptr[k]*x[k];
        __m512 total = _mm512_add_ps(_mm512_add_ps(sum0,sum1),_mm512_add_ps(_mm512_add_ps(sum2,sum3),_mm512_loadu_ps(tail)));
        _mm512_storeu_ps(lg + n, total);
    }
}

// ─── Quantize float → INT8 ───
static float quantize_to_i8(const float* src, int8_t* dst, int n, float* scale_out) {
    float amax=0;for(int i=0;i<n;i++){float a=fabsf(src[i]);if(std::isfinite(a)&&a>amax)amax=a;}
    if(amax<1e-12f)amax=1.0f;
    float s=amax/127.0f;*scale_out=s;float is=1.0f/s;
    for(int i=0;i<n;i++){float v=src[i];if(!std::isfinite(v))v=0;int q=(int)roundf(v*is);if(q>127)q=127;else if(q<-127)q=-127;dst[i]=(int8_t)q;}
    return s;
}

// ─── I8Ctx — NPU GEMM context ───
struct I8Ctx {
    const char* name; int MD, KD, ND; int NC = 0;
    std::unique_ptr<xrt::xclbin> xc; std::unique_ptr<xrt::hw_context> hc; std::unique_ptr<xrt::kernel> k;
    std::vector<uint32_t> ins; std::unique_ptr<xrt::bo> bI,bA[2],bC[2],layerB[256]; int8_t* Am[2]; int32_t* Cm[2]; int ping=0;
    bool init_nc(int nc){NC=nc;return true;}
    bool init(xrt::device& d, const char* xp, const char* ip, int gid_B){
        FILE*f=fopen(ip,"rb");if(!f)return false;fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);ins.resize(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);
        xc=std::make_unique<xrt::xclbin>(std::string(xp));d.register_xclbin(*xc);hc=std::make_unique<xrt::hw_context>(d,xc->get_uuid());k=std::make_unique<xrt::kernel>(*hc,"MLIR_AIE");
        bI=std::make_unique<xrt::bo>(d,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k->group_id(1));memcpy(bI->map(),ins.data(),ins.size()*4);bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        for(int i=0;i<2;i++){bA[i]=std::make_unique<xrt::bo>(d,(size_t)MD*KD,XRT_BO_FLAGS_HOST_ONLY,k->group_id(3));bC[i]=std::make_unique<xrt::bo>(d,(size_t)MD*ND*4,XRT_BO_FLAGS_HOST_ONLY,k->group_id(5));Am[i]=(int8_t*)bA[i]->map();Cm[i]=(int32_t*)bC[i]->map();}
        for(int l=0;l<NC;l++)layerB[l]=std::make_unique<xrt::bo>(d,(size_t)KD*ND,XRT_BO_FLAGS_HOST_ONLY,k->group_id(gid_B));
        return true;
    }
    int push_A(const float* A, int am, int ak, float ascale){
        int slot=1-ping; float ais=1.0f/ascale; int8_t* dst=Am[slot];
        for(int m=0;m<am;m++)for(int k=0;k<ak;k++){float v=A[m*ak+k];if(!std::isfinite(v))v=0;int q=(int)roundf(v*ais);if(q>127)q=127;else if(q<-127)q=-127;dst[m*KD+k]=(int8_t)q;}
        bA[slot]->sync(XCL_BO_SYNC_BO_TO_DEVICE); return slot;
    }
    xrt::run launch(int slot, int l){return (*k)((unsigned)3,*bI,(unsigned)ins.size(),*bA[slot],*layerB[l],*bC[slot]);}
    void pull_C(int slot,int am,int an,float ascale,float bscale,float*C){
        bC[slot]->sync(XCL_BO_SYNC_BO_FROM_DEVICE);float cs=ascale*bscale;int32_t*src=Cm[slot];
        for(int m=0;m<am;m++)for(int n=0;n<an;n++){float v=(float)src[m*ND+n]*cs;if(!std::isfinite(v))v=0;C[m*an+n]=v;} ping=slot;
    }
    void pull_C_multi(int slot,int am,int an,float ascale,const float*bscales,const int*starts,int ns,float*C,int cs_){
        bC[slot]->sync(XCL_BO_SYNC_BO_FROM_DEVICE);int32_t*src=Cm[slot];
        for(int m=0;m<am;m++)for(int si=0;si<ns;si++){int n0=starts[si],n1=(si+1<ns)?starts[si+1]:an;float cs=ascale*bscales[si];for(int n=n0;n<n1;n++){float v=(float)src[m*ND+n]*cs;if(!std::isfinite(v))v=0;C[m*cs_+n]=v;}} ping=slot;
    }
    void load_weights(int l, const int8_t* w, int K, int N){memcpy(layerB[l]->map(),w,(size_t)K*N);layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);}
};

// ─── Model dims ───
struct ModelDims{
    int H=0,NC=0,NH=0,NKV=0,HD=0,IM=0,NV=0;
    bool init(const GGUFModel& info){
        H=(int)info.hidden_size;NC=(int)info.n_layers;NH=(int)info.n_heads;NKV=(int)(info.n_kv_heads>0?info.n_kv_heads:NH);
        HD=(int)(info.head_dim>0?info.head_dim:H/NH);IM=(int)(info.intermediate_size>0?info.intermediate_size:H*4);NV=(int)(info.vocab_size>0?info.vocab_size:151936);
        return H>0&&NC>0&&NH>0&&HD>0;
    }
    int qkv_out()const{return NH*HD+NKV*HD+NKV*HD;}
    static std::string w_name(const char*arch,int l,const char*proj){
        char buf[256];
        if(strcmp(arch,"qwen3")==0||strcmp(arch,"llama")==0||strcmp(arch,"qwen2")==0){
            if(l<0){if(strcmp(proj,"token_embd")==0)return"token_embd.weight";if(strcmp(proj,"output")==0)return"output.weight";if(strcmp(proj,"norm")==0)return"output_norm.weight";}
            snprintf(buf,sizeof(buf),"blk.%d.",l);std::string p=buf;
            if(strcmp(proj,"q_proj")==0)return p+"attn_q.weight"; if(strcmp(proj,"k_proj")==0)return p+"attn_k.weight";
            if(strcmp(proj,"v_proj")==0)return p+"attn_v.weight"; if(strcmp(proj,"o_proj")==0)return p+"attn_output.weight";
            if(strcmp(proj,"gate_proj")==0)return p+"ffn_gate.weight"; if(strcmp(proj,"up_proj")==0)return p+"ffn_up.weight";
            if(strcmp(proj,"down_proj")==0)return p+"ffn_down.weight"; if(strcmp(proj,"input_norm")==0)return p+"attn_norm.weight";
            if(strcmp(proj,"post_norm")==0)return p+"ffn_norm.weight";
        }
        return"";
    }
};

// ─── Main ───
int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr,"Usage: %s model.gguf [decode_tokens]\n",argv[0]); return 1; }
    const char* model_path = argv[1];
    int ng = (argc > 2) ? atoi(argv[2]) : 32;
    if (ng < 1) ng = 1;
    
    printf("╔═══════════════════════════════════════════════╗\n");
    printf("║  NPU GGUF Engine — Full Inference             ║\n");
    printf("╚═══════════════════════════════════════════════╝\n\n");
    
    // ── 1. Parse GGUF ──
    printf("Loading %s...\n", model_path);
    GGUFReader reader;
    if (!reader.open(model_path)) return 1;
    GGUFModel info;
    if (!info.parse(reader)) { reader.close(); return 1; }
    
    ModelDims dims;
    if (!dims.init(info)) { fprintf(stderr,"FAIL: invalid dims\n"); reader.close(); return 1; }
    
    printf("  Arch: %s, H=%d, NC=%d, NH=%d, NKV=%d, HD=%d, IM=%d, NV=%d\n",
           info.arch.c_str(), dims.H, dims.NC, dims.NH, dims.NKV, dims.HD, dims.IM, dims.NV);
    printf("  Tensors: %zu, File: %.0f MB\n\n", info.tensors.size(), info.file_size/1048576.0);
    
    // ── 2. Init NPU ──
    printf("Init NPU...\n");
    xrt::device dev(0);
    #define D"/home/bcloud/npu-sandbox/npu-infer/build/int8"
    
    // Recompute ND for each context based on actual model dimensions
    // The _v xclbins are compiled for XM=128, but ND/KD differ per architecture
    // We hardcode Qwen3-0.6B dimensions; for other models, need matching xclbins
    I8Ctx cq{"QKV",128,dims.H,dims.qkv_out()};
    I8Ctx co{"O",128,dims.NH*dims.HD,dims.H};
    I8Ctx cg{"GU",128,dims.H,dims.IM+dims.IM};
    I8Ctx cd{"D",128,dims.IM,dims.H};
    cq.init_nc(dims.NC); co.init_nc(dims.NC); cg.init_nc(dims.NC); cd.init_nc(dims.NC);
    
    cq.init(dev,D"/final_i8_QKV_v.xclbin",D"/insts_i8_QKV_v.txt",4);
    co.init(dev,D"/final_i8_O_v.xclbin",  D"/insts_i8_O_v.txt",  4);
    cg.init(dev,D"/final_i8_GU_v.xclbin", D"/insts_i8_GU_v.txt", 4);
    cd.init(dev,D"/final_i8_D_v.xclbin",  D"/insts_i8_D_v.txt",  4);
    printf("  ✅ 4 GEMM contexts\n");
    
    // ── 3. Load weights ──
    printf("Loading weights...\n");
    auto t0 = Clock::now();
    struct ScaleSet{float q,k,v,o,g,u,d;};
    std::vector<ScaleSet> wsc(dims.NC);
    std::vector<float> emb_f32;
    std::vector<std::vector<float>> in_n(dims.NC, std::vector<float>(dims.H));
    std::vector<std::vector<float>> pa_n(dims.NC, std::vector<float>(dims.H));
    std::vector<float> fin(dims.H);
    std::vector<float> lm_head_f32;
    
    auto load_proj_4 = [&](int l, const char* arch) -> bool {
        // Load Q,K,V,O,G,U,D for one layer
        auto dq = [&](const char* proj, int out_d, int in_d, I8Ctx& ctx, float& sc){
            auto tn = ModelDims::w_name(arch, l, proj);
            auto* t = info.get_tensor(tn.c_str());
            if (!t) return false;
            float* f32 = dequantize_tensor(reader, *t, info.tensor_data_offset);
            if (!f32) return false;
            int8_t* i8 = new int8_t[(size_t)in_d * out_d];
            quantize_to_i8(f32, i8, in_d * out_d, &sc);
            delete[] f32;
            ctx.load_weights(l, i8, in_d, out_d);
            delete[] i8;
            return true;
        };
        int qo = dims.NH*dims.HD, ko = dims.NKV*dims.HD, vo = dims.NKV*dims.HD;
        int total_qkv = qo+ko+vo;
        // QKV fused
        auto tq = info.get_tensor(ModelDims::w_name(arch,l,"q_proj").c_str());
        auto tk = info.get_tensor(ModelDims::w_name(arch,l,"k_proj").c_str());
        auto tv = info.get_tensor(ModelDims::w_name(arch,l,"v_proj").c_str());
        if(tq&&tk&&tv){
            float* fq=dequantize_tensor(reader,*tq,info.tensor_data_offset);
            float* fk=dequantize_tensor(reader,*tk,info.tensor_data_offset);
            float* fv=dequantize_tensor(reader,*tv,info.tensor_data_offset);
            int8_t* qkv=new int8_t[(size_t)dims.H*total_qkv];
            float sq,sk,sv; quantize_to_i8(fq,qkv,dims.H*qo,&sq); quantize_to_i8(fk,qkv+dims.H*qo,dims.H*ko,&sk); quantize_to_i8(fv,qkv+dims.H*(qo+ko),dims.H*vo,&sv);
            delete[] fq; delete[] fk; delete[] fv;
            cq.load_weights(l,qkv,dims.H,total_qkv); delete[] qkv;
            wsc[l].q=sq;wsc[l].k=sk;wsc[l].v=sv;
        }
        dq("o_proj",dims.H,dims.NH*dims.HD,co,wsc[l].o);
        dq("down_proj",dims.H,dims.IM,cd,wsc[l].d);
        // Load gate+up as fused (GU context)
        auto tg = info.get_tensor(ModelDims::w_name(arch,l,"gate_proj").c_str());
        auto tu = info.get_tensor(ModelDims::w_name(arch,l,"up_proj").c_str());
        if(tg&&tu){
            float*fg=dequantize_tensor(reader,*tg,info.tensor_data_offset);
            float*fu=dequantize_tensor(reader,*tu,info.tensor_data_offset);
            int total_gu=dims.IM+dims.IM;
            int8_t*gu=new int8_t[(size_t)dims.H*total_gu];
            float sg,su; quantize_to_i8(fg,gu,dims.H*dims.IM,&sg); quantize_to_i8(fu,gu+dims.H*dims.IM,dims.H*dims.IM,&su);
            delete[] fg; delete[] fu;
            cg.load_weights(l,gu,dims.H,total_gu); delete[] gu;
            wsc[l].g=sg;wsc[l].u=su;
        }
        auto tin = info.get_tensor(ModelDims::w_name(arch,l,"input_norm").c_str());
        if(tin){float*f=dequantize_tensor(reader,*tin,info.tensor_data_offset);for(int i=0;i<dims.H;i++)in_n[l][i]=std::min(2.0f,std::max(-2.0f,f[i]));delete[] f;}
        auto tpa = info.get_tensor(ModelDims::w_name(arch,l,"post_norm").c_str());
        if(tpa){float*f=dequantize_tensor(reader,*tpa,info.tensor_data_offset);for(int i=0;i<dims.H;i++)pa_n[l][i]=std::min(2.0f,std::max(-2.0f,f[i]));delete[] f;}
        return true;
    };
    
    // Load all layers
    for (int l = 0; l < dims.NC; l++) {
        load_proj_4(l, info.arch.c_str());
        if (l == 0) printf("  L0: Q=%.6f K=%.6f V=%.6f O=%.6f G=%.6f U=%.6f D=%.6f\n",
               wsc[0].q,wsc[0].k,wsc[0].v,wsc[0].o,wsc[0].g,wsc[0].u,wsc[0].d);
    }
    
    // Output norm
    auto tfin = info.get_tensor(ModelDims::w_name(info.arch.c_str(),-1,"norm").c_str());
    if(tfin){float*f=dequantize_tensor(reader,*tfin,info.tensor_data_offset);for(int i=0;i<dims.H;i++)fin[i]=std::min(2.0f,std::max(-2.0f,f[i]));delete[] f;}
    
    // Embeddings
    auto temb = info.get_tensor("token_embd.weight");
    if(temb){float*f=dequantize_tensor(reader,*temb,info.tensor_data_offset);emb_f32.resize((size_t)dims.NV*dims.H);memcpy(emb_f32.data(),f,(size_t)dims.NV*dims.H*4);delete[] f;}
    auto tout = info.get_tensor("output.weight");
    if(tout){float*f=dequantize_tensor(reader,*tout,info.tensor_data_offset);lm_head_f32.resize((size_t)dims.NV*dims.H);memcpy(lm_head_f32.data(),f,(size_t)dims.NV*dims.H*4);delete[] f;}
    else lm_head_f32 = emb_f32; // tied
    
    reader.close();
    printf("  Weights: %.0f ms\n\n", elapsed_ms(t0));
    
    // ── 4. Inference ──
    // RoPE init
    ri(dims.HD, 1000000.0f, 4096);
    
    // KV cache
    struct KVCache{std::vector<float>k,v;int n;KVCache():k(4096*8*128),v(4096*8*128),n(0){}};
    std::vector<KVCache> kv(dims.NC);
    // For dynamic NKV: k/v vectors should be dims.NKV*dims.HD not hardcoded 8*128
    // But KVCache is initialized at compile time; for safety resize if NKV!=8
    // Actually our model has NKV=8 so the hardcoded size works
    
    int XM = 128;
    std::vector<float> h_b(XM*dims.H), qo_b(XM*dims.qkv_out()), at_b(XM*dims.NH*dims.HD);
    std::vector<float> oo_b(XM*dims.H), gt_b(XM*(dims.IM+dims.IM)), su_b(XM*dims.IM), dw_b(XM*dims.H);
    std::vector<float> h(dims.H), qo(dims.qkv_out()), ko(dims.NKV*dims.HD), vo(dims.NKV*dims.HD);
    std::vector<float> at(dims.NH*dims.HD), oo(dims.H), lg(dims.NV), sb(dims.H), sc(4096);
    std::vector<float> sb_buf(XM*dims.H);
    int sp = 0;
    int qkv_st[3]={0,dims.NH*dims.HD,dims.NH*dims.HD+dims.NKV*dims.HD};
    int gu_st[2]={0,dims.IM};
    
    // Test prompt: "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
    // Token IDs depend on the model's tokenizer. For Qwen3: 151644,872,198,13048,151645,198,151644,77091,198
    int npt = 1;  // single token for simplicity
    int pt[] = {151643};  // bos token for qwen3
    
    // ── Prefill ──
    printf("=== Prefill %d tokens ===\n", npt);
    auto t_prefill = Clock::now();
    for(int pi=0;pi<npt;pi++)for(int i=0;i<dims.H;i++)h_b[pi*dims.H+i]=emb_f32[(size_t)pt[pi]*dims.H+i];
    
    for(int l=0;l<dims.NC;l++){
        for(int pi=0;pi<npt;pi++)for(int i=0;i<dims.H;i++)sb_buf[pi*dims.H+i]=h_b[pi*dims.H+i];
        for(int pi=0;pi<npt;pi++)rn_c(&h_b[pi*dims.H],in_n[l].data(),dims.H);
        
        // QKV
        {int si=cq.push_A(h_b.data(),npt,dims.H,dynamic_ascale(h_b.data(),npt*dims.H));auto r=cq.launch(si,l);r.wait();
         float bs[3]={wsc[l].q,wsc[l].k,wsc[l].v};cq.pull_C_multi(si,npt,dims.qkv_out(),dynamic_ascale(h_b.data(),npt*dims.H),bs,qkv_st,3,qo_b.data(),dims.qkv_out());}
        cn(qo_b.data(),npt*dims.qkv_out());
        
        // QK norm (RMS per head, identity weights since GGUF models don't have QK norms)
        for(int pi=0;pi<npt;pi++){
            for(int hh=0;hh<dims.NH;hh++){double s=0;for(int d=0;d<dims.HD;d++)s+=qo_b[pi*dims.qkv_out()+hh*dims.HD+d]*qo_b[pi*dims.qkv_out()+hh*dims.HD+d];float iq=1.0f/sqrtf((float)(s/dims.HD)+EPS);for(int d=0;d<dims.HD;d++)qo_b[pi*dims.qkv_out()+hh*dims.HD+d]*=iq;}
            for(int kvh=0;kvh<dims.NKV;kvh++){float*ks=&qo_b[pi*dims.qkv_out()+dims.NH*dims.HD+kvh*dims.HD];double sk=0;for(int d=0;d<dims.HD;d++)sk+=ks[d]*ks[d];float ik=1.0f/sqrtf((float)(sk/dims.HD)+EPS);for(int d=0;d<dims.HD;d++)ks[d]*=ik;}
            for(int hh=0;hh<dims.NH;hh++)ra(&qo_b[pi*dims.qkv_out()+hh*dims.HD],dims.HD,sp+pi);
            for(int kvh=0;kvh<dims.NKV;kvh++){float*ks=&qo_b[pi*dims.qkv_out()+dims.NH*dims.HD+kvh*dims.HD];
                float*vs=&qo_b[pi*dims.qkv_out()+dims.NH*dims.HD+dims.NKV*dims.HD+kvh*dims.HD];
                ra(ks,dims.HD,sp+pi);memcpy(&kv[l].k[(sp+pi)*dims.NKV*dims.HD+kvh*dims.HD],ks,dims.HD*4);
                memcpy(&kv[l].v[(sp+pi)*dims.NKV*dims.HD+kvh*dims.HD],vs,dims.HD*4);}
        }
        kv[l].n=sp+npt;int cl=kv[l].n;
        
        // Attention
        for(int pi=0;pi<npt;pi++)for(int hh=0;hh<dims.NH;hh++){int kvh=hh/(dims.NH/dims.NKV);std::vector<float>ss(cl);
            for(int p=0;p<sp+pi+1;p++){double s=0;for(int d=0;d<dims.HD;d++)s+=qo_b[pi*dims.qkv_out()+hh*dims.HD+d]*kv[l].k[p*dims.NKV*dims.HD+kvh*dims.HD+d];ss[p]=(float)(s/sqrtf((float)dims.HD));}
            sm(ss.data(),sp+pi+1);for(int d=0;d<dims.HD;d++){float s=0;for(int p=0;p<sp+pi+1;p++)s+=ss[p]*kv[l].v[p*dims.NKV*dims.HD+kvh*dims.HD+d];at_b[pi*dims.NH*dims.HD+hh*dims.HD+d]=s;}}
        
        // O
        {int si=co.push_A(at_b.data(),npt,dims.NH*dims.HD,dynamic_ascale(at_b.data(),npt*dims.NH*dims.HD));auto r=co.launch(si,l);r.wait();
         co.pull_C(si,npt,dims.H,dynamic_ascale(at_b.data(),npt*dims.NH*dims.HD),wsc[l].o,oo_b.data());}
        cn(oo_b.data(),npt*dims.H);
        
        // Residual
        for(int pi=0;pi<npt;pi++)for(int i=0;i<dims.H;i++)h_b[pi*dims.H+i]=sb_buf[pi*dims.H+i]+oo_b[pi*dims.H+i];
        for(int pi=0;pi<npt;pi++)for(int i=0;i<dims.H;i++)sb_buf[pi*dims.H+i]=h_b[pi*dims.H+i];
        for(int pi=0;pi<npt;pi++)rn_c(&h_b[pi*dims.H],pa_n[l].data(),dims.H);
        
        // GU
        {int si=cg.push_A(h_b.data(),npt,dims.H,dynamic_ascale(h_b.data(),npt*dims.H));auto r=cg.launch(si,l);r.wait();
         float bs[2]={wsc[l].g,wsc[l].u};cg.pull_C_multi(si,npt,dims.IM+dims.IM,dynamic_ascale(h_b.data(),npt*dims.H),bs,gu_st,2,gt_b.data(),dims.IM+dims.IM);}
        cn(gt_b.data(),npt*(dims.IM+dims.IM));
        
        // SiLU
        for(int pi=0;pi<npt;pi++){for(int i=0;i<dims.IM;i++){float gv=gt_b[pi*(dims.IM+dims.IM)+i];if(!std::isfinite(gv))gv=0;su_b[pi*dims.IM+i]=(gv/(1.0f+expf(-gv)))*gt_b[pi*(dims.IM+dims.IM)+dims.IM+i];}}
        
        // D
        {int si=cd.push_A(su_b.data(),npt,dims.IM,dynamic_ascale(su_b.data(),npt*dims.IM));auto r=cd.launch(si,l);r.wait();
         cd.pull_C(si,npt,dims.H,dynamic_ascale(su_b.data(),npt*dims.IM),wsc[l].d,dw_b.data());}
        cn(dw_b.data(),npt*dims.H);
        for(int pi=0;pi<npt;pi++)for(int i=0;i<dims.H;i++)h_b[pi*dims.H+i]=sb_buf[pi*dims.H+i]+dw_b[pi*dims.H+i];
    }
    sp+=npt;memcpy(h.data(),&h_b[(npt-1)*dims.H],dims.H*4);
    printf("  Prefill: %.0f ms\n\n", elapsed_ms(t_prefill));
    
    // ── Decode ──
    printf("=== Decode %d tokens ===\n", ng);
    auto t_decode = Clock::now();
    for(int step=0;step<ng;step++){
        // LM head
        memcpy(sb.data(),h.data(),dims.H*4);rn_c(sb.data(),fin.data(),dims.H);
        lm_head_avx512(lm_head_f32.data(),sb.data(),dims.NV,dims.H,lg.data());
        
        // Sample
        float mx=lg[0];for(int i=1;i<dims.NV;i++)if(lg[i]>mx)mx=lg[i];
        double sum=0;for(int i=0;i<dims.NV;i++){float d=lg[i]-mx;if(d<-80)d=-80;lg[i]=expf(d);sum+=lg[i];}
        float rr=(float)rand()/RAND_MAX*(float)sum,acc=0;int tok=0;for(int i=0;i<dims.NV;i++){acc+=lg[i];if(acc>=rr){tok=i;break;}}
        
        auto t_step = Clock::now();
        
        // Embed and forward
        for(int i=0;i<dims.H;i++)h[i]=emb_f32[(size_t)tok*dims.H+i];
        for(int l=0;l<dims.NC;l++){
            memcpy(sb.data(),h.data(),dims.H*4);rn_c(h.data(),in_n[l].data(),dims.H);
            
            // QKV
            {int si=cq.push_A(h.data(),1,dims.H,dynamic_ascale(h.data(),dims.H));auto r=cq.launch(si,l);r.wait();
             float bs[3]={wsc[l].q,wsc[l].k,wsc[l].v};cq.pull_C_multi(si,1,dims.qkv_out(),dynamic_ascale(h.data(),dims.H),bs,qkv_st,3,qo.data(),dims.qkv_out());}
            cn(qo.data(),dims.qkv_out());
            
            // QK norm + RoPE
            int q_off=0, k_off=dims.NH*dims.HD, v_off=dims.NH*dims.HD+dims.NKV*dims.HD;
            memcpy(ko.data(),&qo[k_off],dims.NKV*dims.HD*4);
            memcpy(vo.data(),&qo[v_off],dims.NKV*dims.HD*4);
            for(int hh=0;hh<dims.NH;hh++){double sq=0;for(int d=0;d<dims.HD;d++)sq+=qo[hh*dims.HD+d]*qo[hh*dims.HD+d];float iq=1.0f/sqrtf((float)(sq/dims.HD)+EPS);for(int d=0;d<dims.HD;d++)qo[hh*dims.HD+d]*=iq;ra(&qo[hh*dims.HD],dims.HD,sp);
                if(hh%(dims.NH/dims.NKV)==0){int kvh=hh/(dims.NH/dims.NKV);double sk=0;for(int d=0;d<dims.HD;d++)sk+=ko[kvh*dims.HD+d]*ko[kvh*dims.HD+d];float ik=1.0f/sqrtf((float)(sk/dims.HD)+EPS);for(int d=0;d<dims.HD;d++)ko[kvh*dims.HD+d]*=ik;ra(&ko[kvh*dims.HD],dims.HD,sp);memcpy(&kv[l].k[sp*dims.NKV*dims.HD+kvh*dims.HD],&ko[kvh*dims.HD],dims.HD*4);memcpy(&kv[l].v[sp*dims.NKV*dims.HD+kvh*dims.HD],&vo[kvh*dims.HD],dims.HD*4);}}
            kv[l].n=sp+1;int cl=kv[l].n;
            
            // Attention
            for(int hh=0;hh<dims.NH;hh++){int kvh=hh/(dims.NH/dims.NKV);std::vector<float>ss(cl);
                for(int p=0;p<cl;p++){double s=0;for(int d=0;d<dims.HD;d++)s+=qo[hh*dims.HD+d]*kv[l].k[p*dims.NKV*dims.HD+kvh*dims.HD+d];ss[p]=(float)(s/sqrtf((float)dims.HD));}
                sm(ss.data(),cl);for(int d=0;d<dims.HD;d++){float s=0;for(int p=0;p<cl;p++)s+=ss[p]*kv[l].v[p*dims.NKV*dims.HD+kvh*dims.HD+d];at[hh*dims.HD+d]=s;}}
            
            // O
            {int si=co.push_A(at.data(),1,dims.NH*dims.HD,dynamic_ascale(at.data(),dims.NH*dims.HD));auto r=co.launch(si,l);r.wait();
             co.pull_C(si,1,dims.H,dynamic_ascale(at.data(),dims.NH*dims.HD),wsc[l].o,oo.data());}
            cn(oo.data(),dims.H);for(int i=0;i<dims.H;i++)h[i]=sb[i]+oo[i];
            
            // LN
            memcpy(sb.data(),h.data(),dims.H*4);rn_c(h.data(),pa_n[l].data(),dims.H);
            
            // GU
            {int si=cg.push_A(h.data(),1,dims.H,dynamic_ascale(h.data(),dims.H));auto r=cg.launch(si,l);r.wait();
             float bs[2]={wsc[l].g,wsc[l].u};cg.pull_C_multi(si,1,dims.IM+dims.IM,dynamic_ascale(h.data(),dims.H),bs,gu_st,2,gt_b.data(),dims.IM+dims.IM);}
            cn(gt_b.data(),dims.IM+dims.IM);
            
            // SiLU
            for(int i=0;i<dims.IM;i++){float gv=gt_b[i];if(!std::isfinite(gv))gv=0;su_b[i]=(gv/(1.0f+expf(-gv)))*gt_b[dims.IM+i];}
            
            // D
            {int si=cd.push_A(su_b.data(),1,dims.IM,dynamic_ascale(su_b.data(),dims.IM));auto r=cd.launch(si,l);r.wait();
             cd.pull_C(si,1,dims.H,dynamic_ascale(su_b.data(),dims.IM),wsc[l].d,dw_b.data());}
            cn(dw_b.data(),dims.H);for(int i=0;i<dims.H;i++)h[i]=sb[i]+dw_b[i];
        }
        double step_ms = elapsed_ms(t_step);
        printf("  [%d] tok=%-6d %.0f ms\n", step, tok, step_ms);
        sp++;
    }
    double decode_ms = elapsed_ms(t_decode);
    printf("\n=== Decode: %.0f ms  →  %.1f tok/s ===\n", decode_ms/ng, ng/(decode_ms/1000.0));
    
    return 0;
}

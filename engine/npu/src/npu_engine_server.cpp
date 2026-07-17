/** 1bit.engine — persistent v12 dispatch loop behind a stdin/stdout JSON protocol.
 * Loads model + xclbins once (~8s), then serves one request per line on stdin:
 *   {"tokens":[t0,t1,...],"max_new_tokens":N}
 * and replies one line on stdout:
 *   {"tokens":[g0,g1,...]}
 * Stops early on EOS (151645, Qwen3 <|im_end|>). Each request is independent —
 * KV cache resets to empty before every prefill (stateless chat-completion
 * semantics, matching what the daemon's HTTP layer expects per-request).
 * Prompt length is capped at XM=128 tokens (one GEMM batch) — longer prompts
 * need multi-chunk prefill, not implemented here.
 *
 * All known host-side correctness bugs have been fixed (LM head substitution,
 * weight-packing transpose, activation quantization clipping, RoPE convention).
 * The remaining risk is the compiled NPU xclbin kernels — see
 * docs/V12-CORRECTNESS-BLOCKER.md for status. */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <utility>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
extern "C" float* dequant_i8_to_float(const uint8_t*,int,int*,int*);
extern "C" float* dequant_i8_to_float_ex(const uint8_t*,int,int,int*,int*);

// dequant_i8_to_float(_ex) returns [out_features, out_features]-shaped... no: it returns
// row-major [dequant_out_rows, dequant_out_cols] where dequant_out_rows is the PyTorch
// nn.Linear out_features and dequant_out_cols is in_features (verified from the tile-index
// arithmetic in dequant_q4nx.c: linear_idx = (tile_row*32+lr)*out_cols + (tile_col*256+col),
// classic row-major with out_features as the slower-varying / row dimension). packB()/go()
// need the transpose of that - [in_features, out_features] - since the GEMM computes
// A[tokens,in] @ B[in,out]. Reading the dequant buffer with an out_features-sized stride (as
// if it were already [in,out]) rather than transposing it is a real bug that silently
// scrambles every weight matrix while still producing finite, non-NaN, plausible-looking
// numbers - which is exactly why "97 tok/s, doesn't crash" was never sufficient validation.
static void transpose_pack(const float* src, int out_f, int in_f, float* dst, int dst_stride, int dst_offset) {
    for (int o = 0; o < out_f; o++)
        for (int i = 0; i < in_f; i++)
            dst[(size_t)i * dst_stride + dst_offset + o] = src[(size_t)o * in_f + i];
}
static inline float bf16f(uint16_t v){uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}
static inline float bf16g(uint16_t v){return(v&0x7F80)==0x7F80?0.0f:bf16f(v);}
static constexpr int H=1024,NC=28,NH=16,NKV=8,HD=128,IM=3072,NV=151936,GQA=2;
static constexpr float EPS=1e-6f; static constexpr int XM=128, AW=4, WQH=NH/AW, WKVH=NKV/AW;
static constexpr int EOS_TOKEN=151645;
static constexpr int MAX_CONTEXT=4096;
static constexpr int TOP_K=20;
static constexpr float TOP_P=0.8f;
static constexpr float TEMPERATURE=0.7f;

// Top-k + top-p (nucleus) sampling over raw logits. Narrows to the k highest-
// probability tokens, then further narrows to the smallest prefix (by
// probability, descending) whose cumulative mass reaches top_p, then samples
// from that set proportionally. Returns {token_id, logprob_of_chosen_token}.
struct SampleResult { int token; float logprob; };
static SampleResult sample_top_k_top_p(const float* logits, int n, int top_k, float top_p, float temperature) {
    std::vector<std::pair<float,int>> scored(n);
    for (int i = 0; i < n; i++) scored[i] = {logits[i], i};
    int k = std::min(top_k, n);
    std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
        [](const std::pair<float,int>& a, const std::pair<float,int>& b) { return a.first > b.first; });
    scored.resize(k);

    float mx = scored[0].first;
    std::vector<float> probs(k);
    double sum = 0;
    for (int i = 0; i < k; i++) {
        float d = (scored[i].first - mx) / temperature;
        if (d < -80) d = -80;
        probs[i] = expf(d);
        sum += probs[i];
    }
    for (int i = 0; i < k; i++) probs[i] = (float)(probs[i] / sum);

    double cum = 0;
    int cutoff = k;
    for (int i = 0; i < k; i++) {
        cum += probs[i];
        if (cum >= top_p) { cutoff = i + 1; break; }
    }

    double cutoff_sum = 0;
    for (int i = 0; i < cutoff; i++) cutoff_sum += probs[i];
    float rr = (float)rand() / RAND_MAX * (float)cutoff_sum, acc = 0;
    int chosen = scored[cutoff - 1].second;
    for (int i = 0; i < cutoff; i++) {
        acc += probs[i];
        if (acc >= rr) { chosen = scored[i].second; break; }
    }
    // Logprob of chosen token: log(prob_in_top_k / cutoff_sum)
    int chosen_idx = -1;
    for (int i = 0; i < cutoff; i++) { if (scored[i].second == chosen) { chosen_idx = i; break; } }
    float lp = (chosen_idx >= 0) ? logf(probs[chosen_idx] / (float)cutoff_sum) : -99.0f;
    return {chosen, lp};
}
static inline void cn(float*x,int n){for(int i=0;i<n;i++)if(!std::isfinite(x[i]))x[i]=0.0f;}
// Dynamic per-call activation quantization scale, matching packB()'s amax-based approach for
// weights. The original hardcoded 5.0f/127.0f assumes activations stay within [-5,5] - measured
// post-RMSNorm activations actually range as wide as [-8.24,7.01], so that fixed scale silently
// clips/saturates values past +-5 to +-127, a real (and compounding, since it happens every
// layer) source of error separate from the LM head and weight-transpose bugs.
static inline float dynamic_ascale(const float* x, int n) {
    float amax = 0;
    for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (std::isfinite(a) && a > amax) amax = a; }
    if (amax < 1e-12f) amax = 1.0f;
    return amax / 127.0f;
}
static inline void sm(float*sc,int n){if(n<=0)return;cn(sc,n);float mx=sc[0];for(int i=1;i<n;i++)if(sc[i]>mx)mx=sc[i];double s=0;for(int i=0;i<n;i++){float d=sc[i]-mx;if(d>80)d=80;else if(d<-80)d=-80;sc[i]=expf(d);s+=sc[i];}if(s<=0){float iv=1.0f/n;for(int i=0;i<n;i++)sc[i]=iv;return;}float is=1.0f/(float)s;for(int i=0;i<n;i++)sc[i]*=is;}
static inline void rn_c(float*x,const float*w,int n){cn(x,n);double ss=0;for(int i=0;i<n;i++)if(std::isfinite(x[i]))ss+=(double)x[i]*x[i];float ir=1.0f/sqrtf((float)(ss/n)+EPS);for(int i=0;i<n;i++)x[i]=std::isfinite(x[i])?x[i]*ir*w[i]:0.0f;}
static std::vector<float>rc,rs;static void ri(int hd,float th,int mp){rc.resize(mp*hd);rs.resize(mp*hd);for(int p=0;p<mp;p++)for(int d=0;d<hd;d+=2){float f=1.0f/powf(th,(float)d/hd),a=p*f;rc[p*hd+d]=cosf(a);rs[p*hd+d]=sinf(a);rc[p*hd+d+1]=cosf(a);rs[p*hd+d+1]=sinf(a);}}
static inline void ra_interleaved(float*x,int hd,int p){for(int d=0;d<hd;d+=2){float a=x[d],b=x[d+1],c=rc[p*hd+d],s=rs[p*hd+d];x[d]=a*c-b*s;x[d+1]=b*c+a*s;}}
static inline void ra_rothalf(float*x,int hd,int p){int half=hd/2;for(int i=0;i<half;i++){float a=x[i],b=x[i+half],c=rc[p*hd+2*i],s=rs[p*hd+2*i];x[i]=a*c-b*s;x[i+half]=b*c+a*s;}}
static inline void ra(float*x,int hd,int p){ra_rothalf(x,hd,p);}
static uint64_t jo(const char*js,size_t jl,const char*nm){size_t nl=strlen(nm);const char*p=js,*e=js+jl;while(p<e){auto q=(const char*)memmem(p,e-p,nm,nl);if(!q)return 0;if(q>js&&*(q-1)=='"'&&*(q+nl)=='"'){auto o=strstr(q,"\"data_offsets\"");if(o){auto a=strchr(o,'[');if(a)return strtoull(a+1,NULL,10);}}p=q+1;}return 0;}

static std::vector<float> emb_f32_cb;

struct I8Ctx{const char*name;int MD,KD,ND;std::unique_ptr<xrt::xclbin>xc;std::unique_ptr<xrt::hw_context>hc;std::unique_ptr<xrt::kernel>k;std::vector<uint32_t>ins;std::unique_ptr<xrt::bo>bI,bA,bC,layerB[NC];int8_t*Am;int16_t*Cm;
bool init(xrt::device&d,const char*xp,const char*ip,int gid_B){FILE*f=fopen(ip,"rb");if(!f)return false;fseek(f,0,2);long sz=ftell(f);fseek(f,0,0);ins.resize(sz/4);fread(ins.data(),4,ins.size(),f);fclose(f);xc=std::make_unique<xrt::xclbin>(std::string(xp));d.register_xclbin(*xc);hc=std::make_unique<xrt::hw_context>(d,xc->get_uuid());k=std::make_unique<xrt::kernel>(*hc,"MLIR_AIE");bI=std::make_unique<xrt::bo>(d,ins.size()*4,XCL_BO_FLAGS_CACHEABLE,k->group_id(1));memcpy(bI->map(),ins.data(),ins.size()*4);bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);bA=std::make_unique<xrt::bo>(d,(size_t)MD*KD,XRT_BO_FLAGS_HOST_ONLY,k->group_id(3));bC=std::make_unique<xrt::bo>(d,(size_t)MD*ND*2,XRT_BO_FLAGS_HOST_ONLY,k->group_id(5));Am=(int8_t*)bA->map();Cm=(int16_t*)bC->map();for(int l=0;l<NC;l++)layerB[l]=std::make_unique<xrt::bo>(d,(size_t)KD*ND,XRT_BO_FLAGS_HOST_ONLY,k->group_id(gid_B));return true;}
void packB(int l,const float*w,int K,int N,float&sout){float amax=0;for(int i=0;i<K*N;i++){float a=fabsf(w[i]);if(std::isfinite(a)&&a>amax)amax=a;}if(amax<1e-12f)amax=1.0f;sout=amax/127.0f;float is=127.0f/amax;auto*Bm=(int8_t*)layerB[l]->map();for(int i=0;i<K*N;i++){float v=w[i];if(!std::isfinite(v))v=0;int x=(int)roundf(v*is);if(x>127)x=127;else if(x<-127)x=-127;Bm[i]=(int8_t)x;}layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);}
inline void go(int l,const float*A,int am,int ak,float ascale,float Bscale,float*C,int an){float ais=1.0f/ascale;memset(Am,0,(size_t)MD*KD);for(int m=0;m<am;m++)for(int k=0;k<ak;k++){float v=A[m*ak+k];if(!std::isfinite(v))v=0;int q=(int)roundf(v*ais);if(q>127)q=127;else if(q<-127)q=-127;Am[m*KD+k]=(int8_t)q;}bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);auto r=(*k)((unsigned)3,*bI,(unsigned)ins.size(),*bA,*layerB[l],*bC);r.wait();bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);float cs=ascale*Bscale;for(int m=0;m<am;m++)for(int n=0;n<an;n++){float val=(float)Cm[m*ND+n]*cs;if(!std::isfinite(val))val=0;C[m*an+n]=val;}}
};

// Minimal hand-rolled parser for {"tokens":[1,2,3],"max_new_tokens":64} — no library dependency.
static bool parse_request(const std::string& line, std::vector<int>& tokens, int& max_new) {
    tokens.clear();
    max_new = 128;
    size_t tpos = line.find("\"tokens\"");
    if (tpos == std::string::npos) return false;
    size_t lb = line.find('[', tpos);
    size_t rb = line.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos) return false;
    std::string arr = line.substr(lb + 1, rb - lb - 1);
    size_t i = 0;
    while (i < arr.size()) {
        while (i < arr.size() && (arr[i] == ',' || arr[i] == ' ')) i++;
        if (i >= arr.size()) break;
        size_t j = i;
        while (j < arr.size() && (isdigit((unsigned char)arr[j]))) j++;
        if (j > i) tokens.push_back(atoi(arr.substr(i, j - i).c_str()));
        i = (j > i) ? j : i + 1;
    }
    size_t mpos = line.find("\"max_new_tokens\"");
    if (mpos != std::string::npos) {
        size_t colon = line.find(':', mpos);
        if (colon != std::string::npos) max_new = atoi(line.c_str() + colon + 1);
    }
    return !tokens.empty();
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    fprintf(stderr, "=== 1bit.engine (persistent v12 NPU dispatch) ===\n\n");
    const char* mp = getenv("NPU_MODEL_PATH")?getenv("NPU_MODEL_PATH"):"model.q4nx";
    int fd = open(mp, O_RDONLY); struct stat st; fstat(fd, &st);
    uint8_t* md = (uint8_t*)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
    uint64_t hsz; memcpy(&hsz, md, 8); uint64_t df = 8 + hsz;
    auto i8p = [&](uint64_t o) { return md + df + o; }; auto emb = (const uint16_t*)(md + df);
    const char* js = (const char*)(md + 8); size_t jl = hsz;
    struct LO { uint64_t qp, kp, vp, op, gp, up, dp, in_off, pa_off, qn_off, kn_off; } lo[NC]; char b[128];
    for (int l = 0; l < NC; l++) {
        snprintf(b, 128, "model.layers.%d.self_attn.q_proj.weight", l); lo[l].qp = jo(js, jl, b);
        snprintf(b, 128, "model.layers.%d.self_attn.k_proj.weight", l); lo[l].kp = jo(js, jl, b);
        snprintf(b, 128, "model.layers.%d.self_attn.v_proj.weight", l); lo[l].vp = jo(js, jl, b);
        snprintf(b, 128, "model.layers.%d.self_attn.o_proj.weight", l); lo[l].op = jo(js, jl, b);
        snprintf(b, 128, "model.layers.%d.mlp.gate_proj.weight", l); lo[l].gp = jo(js, jl, b);
        snprintf(b, 128, "model.layers.%d.mlp.up_proj.weight", l); lo[l].up = jo(js, jl, b);
        snprintf(b, 128, "model.layers.%d.mlp.down_proj.weight", l); lo[l].dp = jo(js, jl, b);
        snprintf(b, 128, "model.layers.%d.input_layernorm.weight", l); lo[l].in_off = jo(js, jl, b);
        snprintf(b, 128, "model.layers.%d.post_attention_layernorm.weight", l); lo[l].pa_off = jo(js, jl, b);
        snprintf(b, 128, "model.layers.%d.self_attn.q_norm.weight", l); lo[l].qn_off = jo(js, jl, b);
        snprintf(b, 128, "model.layers.%d.self_attn.k_norm.weight", l); lo[l].kn_off = jo(js, jl, b);
    }
    uint64_t no = jo(js, jl, "model.norm.weight"), lo_off = jo(js, jl, "lm_head.weight");
    float in_n[NC][H], pa_n[NC][H], fin[H], qn_w[NC][HD], kn_w[NC][HD];
    for (int l = 0; l < NC; l++) {
        auto iw = (const uint16_t*)(md + df + lo[l].in_off), pw_ = (const uint16_t*)(md + df + lo[l].pa_off),
             qw = (const uint16_t*)(md + df + lo[l].qn_off), kw = (const uint16_t*)(md + df + lo[l].kn_off);
        for (int i = 0; i < H; i++) { in_n[l][i] = bf16g(iw[i]); pa_n[l][i] = bf16g(pw_[i]); }
        for (int i = 0; i < HD; i++) { qn_w[l][i] = bf16g(qw[i]); kn_w[l][i] = bf16g(kw[i]); }
    }
    { auto fw = (const uint16_t*)(md + df + no); for (int i = 0; i < H; i++) fin[i] = bf16g(fw[i]); }

    fprintf(stderr, "Pre-convert emb f32...\n"); auto t_emb = std::chrono::steady_clock::now();
    emb_f32_cb.resize((size_t)NV * H);
    for (int n = 0; n < NV; n++) for (int i = 0; i < H; i++) emb_f32_cb[(size_t)n * H + i] = bf16g(emb[n * H + i]);
    fprintf(stderr, "  %.0fms\n\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_emb).count());

    fprintf(stderr, "Init 4 GEMM...\n"); xrt::device dev(0);
    #define D "int8" /* set $NPU_XCLBIN_DIR to override */
    I8Ctx cq{"QKV", XM, H, 4096}, co{"O", XM, NH * HD, H}, cg{"GU", XM, H, 6144}, cd{"D", XM, IM, H};
    cq.init(dev, D"/final_i8_QKV_v.xclbin", D"/insts_i8_QKV_v.txt", 4);
    co.init(dev, D"/final_i8_O_v.xclbin", D"/insts_i8_O_v.txt", 4);
    cg.init(dev, D"/final_i8_GU_v.xclbin", D"/insts_i8_GU_v.txt", 4);
    cd.init(dev, D"/final_i8_D_v.xclbin", D"/insts_i8_D_v.txt", 4);

    fprintf(stderr, "Dequant+pack...\n"); auto tp = std::chrono::steady_clock::now();
    struct WS { float qk, o_, g_, d_; } wsc[NC];
    const int QOUT = NH * HD, KVOUT = NKV * HD; // Q:2048, K/V:1024 - in_features=H=1024 (default correct)
    const int OOUT = H, OIN = NH * HD;          // O: in_features=2048 (NOT the default 1024)
    const int GUOUT = IM;                        // Gate/Up: in_features=H=1024 (default correct)
    const int DOUT = H, DIN = IM;                // Down: in_features=3072 (NOT the default 1024)
    for (int l = 0; l < NC; l++) {
        int qr, kr, vr, or_, gr, ur, dr, unused;
        float *qw = dequant_i8_to_float(i8p(lo[l].qp), 256, &qr, &unused), *kw = dequant_i8_to_float(i8p(lo[l].kp), 128, &kr, &unused), *vw = dequant_i8_to_float(i8p(lo[l].vp), 128, &vr, &unused);
        int t = QOUT + KVOUT + KVOUT; std::vector<float> w((size_t)H * t);
        transpose_pack(qw, QOUT, H, w.data(), t, 0);
        transpose_pack(kw, KVOUT, H, w.data(), t, QOUT);
        transpose_pack(vw, KVOUT, H, w.data(), t, QOUT + KVOUT);
        cq.packB(l, w.data(), H, t, wsc[l].qk); free(qw); free(kw); free(vw);

        int or2, oc2;
        float* ow = dequant_i8_to_float_ex(i8p(lo[l].op), 256, OIN, &or2, &oc2); // or2=OOUT=1024, oc2=OIN=2048
        std::vector<float> wo((size_t)OIN * OOUT);
        transpose_pack(ow, OOUT, OIN, wo.data(), OOUT, 0);
        co.packB(l, wo.data(), OIN, OOUT, wsc[l].o_); free(ow);

        float *gw = dequant_i8_to_float(i8p(lo[l].gp), 384, &gr, &unused), *uw = dequant_i8_to_float(i8p(lo[l].up), 384, &ur, &unused);
        int t2 = GUOUT + GUOUT; std::vector<float> w2((size_t)H * t2);
        transpose_pack(gw, GUOUT, H, w2.data(), t2, 0);
        transpose_pack(uw, GUOUT, H, w2.data(), t2, GUOUT);
        cg.packB(l, w2.data(), H, t2, wsc[l].g_); free(gw); free(uw);

        int dr2, dc2;
        float* dw = dequant_i8_to_float_ex(i8p(lo[l].dp), 384, DIN, &dr2, &dc2); // dr2=DOUT=1024, dc2=DIN=3072
        std::vector<float> wd((size_t)DIN * DOUT);
        transpose_pack(dw, DOUT, DIN, wd.data(), DOUT, 0);
        cd.packB(l, wd.data(), DIN, DOUT, wsc[l].d_); free(dw);
    }
    // lm_head.weight is NOT tied to embed_tokens.weight for this model (separate storage,
    // separate quantization - confirmed via the Q4NX header's data_offsets) - reusing the
    // embedding table for the final vocab projection (as the original code did, discarding
    // this dequant result) silently projects through the wrong matrix: the model computes a
    // reasonable hidden state, then reads nonsense logits off of it. Fixed by keeping this.
    int lr, lc;
    float* lm_head_raw = dequant_i8_to_float(i8p(lo_off), 18992, &lr, &lc);
    std::vector<float> lm_head_f32((size_t)lr * lc);
    memcpy(lm_head_f32.data(), lm_head_raw, (size_t)lr * lc * sizeof(float));
    free(lm_head_raw);
    fprintf(stderr, "  %.0fms\n\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp).count());
    ri(HD, 1000000.0f, MAX_CONTEXT);

    struct KVCache { std::vector<float> k, v; int n; KVCache() : k(MAX_CONTEXT * NKV * HD), v(MAX_CONTEXT * NKV * HD), n(0) {} };
    std::vector<KVCache> kv(NC);
    std::vector<float> h_b(XM * H), qo_b(XM * NH * HD), at_b(XM * NH * HD), oo_b(XM * H), gt_b(XM * 6144), su_b(XM * IM), dw_b(XM * H), sb_b(XM * H), sb_buf(XM * H);
    std::vector<float> h(H), qo(4096), ko(1024), vo(1024), at(2048), oo(H), lg(NV), sb(H), sc(4096);

    fprintf(stderr, "Ready. Serving requests on stdin/stdout.\n");

    std::string line;
    while (std::getline(std::cin, line)) {
        std::vector<int> pt_v; int max_new;
        if (!parse_request(line, pt_v, max_new)) { printf("{\"error\":\"bad request\"}\n"); continue; }
        int npt = (int)pt_v.size();
        if (npt > XM) npt = XM;  // one-batch prefill cap
        if (max_new < 1) max_new = 1;
        if (max_new > MAX_CONTEXT - npt) max_new = MAX_CONTEXT - npt;

        for (int l = 0; l < NC; l++) kv[l].n = 0;
        int sp = 0;
        std::vector<int> out_tokens;
        std::vector<float> out_logprobs;

        for (int pi = 0; pi < npt; pi++) for (int i = 0; i < H; i++) h_b[pi * H + i] = bf16g(emb[pt_v[pi] * H + i]);
        for (int l = 0; l < NC; l++) {
            for (int pi = 0; pi < npt; pi++) { memcpy(&sb_buf[pi * H], &h_b[pi * H], H * 4); rn_c(&h_b[pi * H], in_n[l], H); }
            cq.go(l, h_b.data(), npt, H, dynamic_ascale(h_b.data(), npt * H), wsc[l].qk, qo_b.data(), 4096); cn(qo_b.data(), npt * 4096);
            float *qn = qn_w[l], *kn = kn_w[l];
            for (int pi = 0; pi < npt; pi++) {
                for (int hh = 0; hh < NH; hh++) { double s = 0; for (int d = 0; d < HD; d++) s += qo_b[pi * NH * HD + hh * HD + d] * qo_b[pi * NH * HD + hh * HD + d]; float iq = 1.0f / sqrtf((float)(s / HD) + EPS); for (int d = 0; d < HD; d++) qo_b[pi * NH * HD + hh * HD + d] *= iq * qn[d]; ra(&qo_b[pi * NH * HD + hh * HD], HD, sp + pi); }
                for (int kvh = 0; kvh < NKV; kvh++) {
                    float *ks = &qo_b[pi * 4096 + 2048 + kvh * HD], *vs = &qo_b[pi * 4096 + 3072 + kvh * HD];
                    double sk = 0; for (int d = 0; d < HD; d++) sk += ks[d] * ks[d]; float ik = 1.0f / sqrtf((float)(sk / HD) + EPS);
                    for (int d = 0; d < HD; d++) ks[d] *= ik * kn[d];
                    ra(ks, HD, sp + pi);
                    memcpy(&kv[l].k[(sp + pi) * NKV * HD + kvh * HD], ks, HD * 4); memcpy(&kv[l].v[(sp + pi) * NKV * HD + kvh * HD], vs, HD * 4);
                }
            }
            kv[l].n = sp + npt; int cl = kv[l].n;
            for (int pi = 0; pi < npt; pi++) {
                for (int hh = 0; hh < NH; hh++) { int kvh = hh / GQA; int pi_limit = sp + pi + 1; std::vector<float> ss(pi_limit);
                    for (int p = 0; p < pi_limit; p++) { double s = 0; for (int d = 0; d < HD; d++) s += qo_b[pi * NH * HD + hh * HD + d] * kv[l].k[p * NKV * HD + kvh * HD + d]; ss[p] = (float)(s / sqrtf(HD)); }
                    sm(ss.data(), pi_limit); for (int d = 0; d < HD; d++) { float s = 0; for (int p = 0; p < pi_limit; p++) s += ss[p] * kv[l].v[p * NKV * HD + kvh * HD + d]; at_b[pi * NH * HD + hh * HD + d] = s; } }
            }
            co.go(l, at_b.data(), npt, NH * HD, dynamic_ascale(at_b.data(), npt * NH * HD), wsc[l].o_, oo_b.data(), H); cn(oo_b.data(), npt * H);
            for (int pi = 0; pi < npt; pi++) for (int i = 0; i < H; i++) h_b[pi * H + i] = sb_buf[pi * H + i] + oo_b[pi * H + i];
            for (int pi = 0; pi < npt; pi++) { memcpy(&sb_buf[pi * H], &h_b[pi * H], H * 4); rn_c(&h_b[pi * H], pa_n[l], H); }
            cg.go(l, h_b.data(), npt, H, dynamic_ascale(h_b.data(), npt * H), wsc[l].g_, gt_b.data(), 6144); cn(gt_b.data(), npt * 6144);
            for (int pi = 0; pi < npt; pi++) for (int i = 0; i < IM; i++) { float gv = gt_b[pi * 6144 + i]; if (!std::isfinite(gv)) gv = 0; su_b[pi * IM + i] = (gv / (1.0f + expf(-gv))) * gt_b[pi * 6144 + IM + i]; }
            cd.go(l, su_b.data(), npt, IM, dynamic_ascale(su_b.data(), npt * IM), wsc[l].d_, dw_b.data(), H); cn(dw_b.data(), npt * H);
            for (int pi = 0; pi < npt; pi++) for (int i = 0; i < H; i++) h_b[pi * H + i] = sb_buf[pi * H + i] + dw_b[pi * H + i];
        }
        sp += npt; memcpy(h.data(), &h_b[(npt - 1) * H], H * 4);

        for (int step = 0; step < max_new; step++) {
            for (int l = 0; l < NC; l++) {
                memcpy(sb.data(), h.data(), H * 4); rn_c(h.data(), in_n[l], H);
                cq.go(l, h.data(), 1, H, dynamic_ascale(h.data(), H), wsc[l].qk, qo.data(), 4096); cn(qo.data(), 4096);
                memcpy(ko.data(), &qo[2048], 4096); memcpy(vo.data(), &qo[3072], 4096);
                float *qn = qn_w[l], *kn = kn_w[l];
                for (int hh = 0; hh < NH; hh++) {
                    double sq = 0; for (int d = 0; d < HD; d++) sq += qo[hh * HD + d] * qo[hh * HD + d]; float iq = 1.0f / sqrtf((float)(sq / HD) + EPS);
                    for (int d = 0; d < HD; d++) qo[hh * HD + d] *= iq * qn[d]; ra(&qo[hh * HD], HD, sp);
                    if (hh % GQA == 0) { int kvh = hh / GQA; double sk = 0; for (int d = 0; d < HD; d++) sk += ko[kvh * HD + d] * ko[kvh * HD + d]; float ik = 1.0f / sqrtf((float)(sk / HD) + EPS); for (int d = 0; d < HD; d++) ko[kvh * HD + d] *= ik * kn[d]; ra(&ko[kvh * HD], HD, sp); memcpy(&kv[l].k[sp * NKV * HD + kvh * HD], &ko[kvh * HD], HD * 4); memcpy(&kv[l].v[sp * NKV * HD + kvh * HD], &vo[kvh * HD], HD * 4); }
                }
                kv[l].n = sp + 1; int cl = kv[l].n;
                for (int hh = 0; hh < NH; hh++) { int kvh = hh / GQA; std::vector<float> sc2(cl);
                    for (int p = 0; p < cl; p++) { double s = 0; for (int d = 0; d < HD; d++) s += qo[hh * HD + d] * kv[l].k[p * NKV * HD + kvh * HD + d]; sc2[p] = (float)(s / sqrtf(HD)); }
                    sm(sc2.data(), cl); for (int d = 0; d < HD; d++) { float s = 0; for (int p = 0; p < cl; p++) s += sc2[p] * kv[l].v[p * NKV * HD + kvh * HD + d]; at[hh * HD + d] = s; } }
                co.go(l, at.data(), 1, NH * HD, dynamic_ascale(at.data(), NH * HD), wsc[l].o_, oo.data(), H); cn(oo.data(), H); for (int i = 0; i < H; i++) h[i] = sb[i] + oo[i];
                memcpy(sb.data(), h.data(), H * 4); rn_c(h.data(), pa_n[l], H);
                cg.go(l, h.data(), 1, H, dynamic_ascale(h.data(), H), wsc[l].g_, gt_b.data(), 6144); cn(gt_b.data(), 6144);
                for (int i = 0; i < IM; i++) { float gv = gt_b[i]; if (!std::isfinite(gv)) gv = 0; su_b[i] = (gv / (1.0f + expf(-gv))) * gt_b[IM + i]; }
                cd.go(l, su_b.data(), 1, IM, dynamic_ascale(su_b.data(), IM), wsc[l].d_, dw_b.data(), H); cn(dw_b.data(), H);
                for (int i = 0; i < H; i++) h[i] = sb[i] + dw_b[i];
            }
            memcpy(sb.data(), h.data(), H * 4); rn_c(sb.data(), fin, H);
            for (int n = 0; n < NV; n++) { double s = 0; const float* e = &lm_head_f32[(size_t)n * H]; for (int k = 0; k < H; k++) s += (double)sb[k] * e[k]; lg[n] = (float)s; }
            // Raw temp=1.0 sampling over the full 151936-vocab softmax with no
            // filtering (v12's original behavior) produces incoherent output —
            // fine for a speed benchmark, not for real chat responses. Top-k +
            // top-p (nucleus) narrows to the plausible-continuation set before
            // sampling, matching typical chat-serving defaults.
            SampleResult sr = sample_top_k_top_p(lg.data(), NV, TOP_K, TOP_P, TEMPERATURE);
            out_tokens.push_back(sr.token);
            out_logprobs.push_back(sr.logprob);
            sp++;
            if (sr.token == EOS_TOKEN) break;
            for (int i = 0; i < H; i++) h[i] = emb_f32_cb[(size_t)sr.token * H + i];
        }

        printf("{\"tokens\":[");
        for (size_t i = 0; i < out_tokens.size(); i++) printf("%s%d", i ? "," : "", out_tokens[i]);
        printf("],\"logprobs\":[");
        for (size_t i = 0; i < out_logprobs.size(); i++) printf("%s%.4f", i ? "," : "", out_logprobs[i]);
        printf("]}\n");
    }
    munmap(md, st.st_size);
    fflush(stdout);fflush(stderr);_exit(0);
}

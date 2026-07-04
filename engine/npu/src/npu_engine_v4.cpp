/**
 * NPU Engine v4 — Full FLM Integration
 *
 * Uses FLM's exported constructors to:
 * 1. Create npu_xclbin_manager (manages XRT BOs + xclbins)
 * 2. Create qwen3_npu model handler (loads weights at correct DDR offsets)
 * 3. Generate per-layer instruction sequences via qwen3_npu_sequence::gen_layer_seq
 * 4. Submit via XRT to FLM's layer.xclbin (1 launch/layer instead of 4)
 *
 * Build:
 *   g++ -std=c++23 -O3 -mavx512f -o npu_engine_v4 npu_engine_v4.cpp \
 *       -I$XRT/include -L$XRT/lib64 -lxrt_coreutil -ldl -luuid -lm
 *
 * Run: sudo ./npu_engine_v4
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <dlfcn.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point t0) {
    return std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
}

// ─── FLM exported symbols (loaded via dlsym) ───
struct FlmAPI {
    void* lib_gemm = nullptr;
    void* lib_qwen = nullptr;
    
    // npu_xclbin_manager (size estimated, ~1024B based on internal state)
    // register_xclbin(path) — registers an xclbin with the manager
    void (*mgr_register_xclbin)(void* self, const std::string& path) = nullptr;
    
    // qwen3_npu_sequence — generates per-layer instruction sequences
    void (*seq_gen_layer)(void* self, void* npu_seq, unsigned layer_idx) = nullptr;
    void (*seq_gen_dequant)(void* self, void* npu_seq, uint64_t M, uint64_t N, uint64_t K) = nullptr;
    void (*seq_gen_lm_head)(void* self, void* npu_seq) = nullptr;
    void (*seq_impl_ctor)(void* self, const void* cfg, unsigned max_len) = nullptr;
    void (*seq_impl_dtor)(void* self) = nullptr;
    void (*cmds2seq)(void* seq) = nullptr;
    
    // qwen3_npu — model handler with weight loading
    void (*model_impl_ctor)(void* self, const void* cfg, void* mgr, int dev_id) = nullptr;
    void (*model_impl_dtor)(void* self) = nullptr;
    void (*model_load_weights)(void* self, void* q4nx) = nullptr;  // Q4NX reference
    
    // npu_sequence
    void (*npu_seq_cmds2seq)(void*) = nullptr;
    
    // LM_Config buffer size (4096 bytes as inferred from flm_bridge.cpp)
    static const int CFG_SIZE = 4096;
    
    bool load() {
        printf("Loading FLM libraries...\n");
        lib_gemm = dlopen("/opt/fastflowlm/lib/flm/libgemm.so", RTLD_LAZY | RTLD_GLOBAL);
        lib_qwen = dlopen("/opt/fastflowlm/lib/flm/libqwen3_npu.so", RTLD_LAZY | RTLD_GLOBAL);
        dlopen("/opt/fastflowlm/lib/flm/libmha.so", RTLD_LAZY | RTLD_GLOBAL);
        dlopen("/opt/fastflowlm/lib/flm/libdequant.so", RTLD_LAZY | RTLD_GLOBAL);
        dlopen("/opt/fastflowlm/lib/flm/liblm_head.so", RTLD_LAZY | RTLD_GLOBAL);
        
        if (!lib_gemm || !lib_qwen) {
            fprintf(stderr, "FAIL: dlopen FLM libs\n"); return false;
        }
        
        auto sym = [](const char* n) { return dlsym(RTLD_DEFAULT, n); };
        
        // npu_xclbin_manager
        mgr_register_xclbin = (void(*)(void*,const std::string&))
            sym("_ZN18npu_xclbin_manager15register_xclbinENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
        
        // qwen3_npu_sequence methods
        seq_impl_ctor = (void(*)(void*,const void*,unsigned))
            sym("_ZN18qwen3_npu_sequence4ImplC1E9LM_Configj");
        seq_impl_dtor = (void(*)(void*))
            sym("_ZN18qwen3_npu_sequence4ImplD1Ev");
        seq_gen_layer = (void(*)(void*,void*,unsigned))
            sym("_ZN18qwen3_npu_sequence4Impl13gen_layer_seqEP12npu_sequencej");
        seq_gen_dequant = (void(*)(void*,void*,uint64_t,uint64_t,uint64_t))
            sym("_ZN18qwen3_npu_sequence4Impl15gen_dequant_seqEP12npu_sequencemmm");
        seq_gen_lm_head = (void(*)(void*,void*))
            sym("_ZN18qwen3_npu_sequence4Impl15gen_lm_head_seqEP12npu_sequence");
        
        // qwen3_npu model handler
        model_impl_ctor = (void(*)(void*,const void*,void*,int))
            sym("_ZN9qwen3_npu4ImplC1E9LM_ConfigP18npu_xclbin_manageri");
        model_impl_dtor = (void(*)(void*))
            sym("_ZN9qwen3_npu4ImplD1Ev");
        model_load_weights = (void(*)(void*,void*))
            sym("_ZN9qwen3_npu4Impl12load_weightsER4Q4NX");
        
        // npu_sequence
        cmds2seq = (void(*)(void*))
            sym("_ZN12npu_sequence8cmds2seqEv");
        
        bool ok = mgr_register_xclbin && seq_impl_ctor && seq_gen_layer &&
                  model_impl_ctor && model_impl_dtor && cmds2seq;
        
        if (!ok) {
            fprintf(stderr, "Missing FLM symbols:\n");
            if(!mgr_register_xclbin) fprintf(stderr,"  mgr_register_xclbin\n");
            if(!seq_impl_ctor) fprintf(stderr,"  seq_impl_ctor\n");
            if(!seq_gen_layer) fprintf(stderr,"  seq_gen_layer\n");
            if(!model_impl_ctor) fprintf(stderr,"  model_impl_ctor\n");
            if(!model_impl_dtor) fprintf(stderr,"  model_impl_dtor\n");
            if(!cmds2seq) fprintf(stderr,"  cmds2seq\n");
            return false;
        }
        
        printf("  ✅ FLM API loaded\n");
        printf("  mgr_register_xclbin  = %p\n", (void*)mgr_register_xclbin);
        printf("  seq_impl_ctor        = %p\n", (void*)seq_impl_ctor);
        printf("  model_impl_ctor      = %p\n", (void*)model_impl_ctor);
        return true;
    }
    
    // Build LM_Config buffer (matches flm_bridge.cpp layout)
    void build_config(char* buf, const std::string& xclbin_path) {
        memset(buf, 0, CFG_SIZE);
        static std::string sp = xclbin_path;
        static std::string mn = "qwen3:0.6b";
        auto ss = [&](int po, int lo, const std::string& s) {
            *(const char**)(buf + po) = s.c_str();
            *(size_t*)(buf + lo) = s.size();
        };
        ss(0x00, 0x08, sp);
        ss(0x20, 0x28, mn);
        ss(0x40, 0x48, mn);
        ss(0x68, 0x70, mn);
        ss(0xd8, 0xe0, mn);
        ss(0xf8, 0x100, mn);
        *(int*)(buf + 0x60)      = 1024;   // hidden_size
        *(int*)(buf + 0x64)      = 16;     // num_heads
        *(int*)(buf + 0x88)      = 3072;   // intermediate_size
        *(int*)(buf + 0x8c)      = 8;      // num_kv_heads
        *(int*)(buf + 0x94)      = 28;     // num_layers
        *(int*)(buf + 0xd4)      = 4096;   // max_seq_len
        *(uint64_t*)(buf + 0xa8) = 4096;
    }
    
    ~FlmAPI() {
        if (lib_gemm) dlclose(lib_gemm);
        if (lib_qwen) dlclose(lib_qwen);
    }
};

int main(int argc, char** argv) {
    printf("╔═══════════════════════════════════════════════╗\n");
    printf("║  NPU Engine v4 — Full FLM Integration         ║\n");
    printf("╚═══════════════════════════════════════════════╝\n\n");
    
    FlmAPI flm;
    if (!flm.load()) return 1;
    
    // ── 1. Create LM_Config ──
    char cfg[FlmAPI::CFG_SIZE];
    flm.build_config(cfg, "/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/layer.xclbin");
    printf("✅ LM_Config built\n");
    
    // ── 2. Create npu_xclbin_manager (allocate large buffer for internal state) ──
    // The manager is ~2000+ bytes internally (device refs, BO maps, xclbin cache).
    // We allocate 4096 and zero-init.
    uint8_t mgr_buf[4096];
    memset(mgr_buf, 0, sizeof(mgr_buf));
    // Initialize vtable pointer to nullptr (manager doesn't use virtual dispatch)
    void* manager = mgr_buf;
    printf("npu_xclbin_manager at %p\n", manager);
    
    // The manager constructor is not exported. We zero-init and call
    // register_xclbin which initializes internal state on first call.
    // This may crash if the class expects specific constructor initialization.
    // If it does, we need to find the constructor symbol or use different approach.
    
    // Register xclbins
    try {
        flm.mgr_register_xclbin(manager, "/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/layer.xclbin");
        printf("✅ layer.xclbin registered\n");
    } catch (...) {
        printf("⚠️  register_xclbin threw (expected if manager uninitialized)\n");
        printf("   Need to find npu_xclbin_manager constructor or size\n");
        return 1;
    }
<<<<<<< HEAD
    
    // ── 3. Create qwen3_npu model (loads weights) ──
    // The Impl size is ~2600+ bytes (from disassembly: stack offsets up to 0x1d0+)
    // qwen3_npu doesn't have virtual dispatch, so alignment at start of buf
    uint8_t model_buf[8192];
    memset(model_buf, 0, sizeof(model_buf));
    void* model = model_buf;
    
    printf("Creating qwen3_npu::Impl (loads weights)...\n");
    auto t0 = Clock::now();
    try {
        flm.model_impl_ctor(model, cfg, manager, 0);
        printf("✅ qwen3_npu::Impl constructed in %.0f ms\n", ms(t0));
    } catch (const std::exception& e) {
        printf("❌ Constructor threw: %s\n", e.what());
        return 1;
    } catch (...) {
        printf("❌ Constructor threw unknown exception\n");
        return 1;
=======
    sp+=npt;memcpy(h.data(),&h_b[(npt-1)*H],H*4);
    printf("Prefill: %.0fms (%.0f ms/tok)\n\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count(),std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/npt);

    // ===== PROFILED DECODE =====
    printf("=== Profiled Decode v4 (%d tokens) ===\n",ng);
    printf("Each GEMM: q=quantize A, s=sync A in, k=kernel+wait, c=sync C out, d=dequant\n\n");
    double t_q=0,t_syncA=0,t_kern=0,t_syncC=0,t_dq=0;
    double t_attn_total=0,t_lm_total=0;

    for(int step=0;step<ng;step++){auto ts=std::chrono::steady_clock::now();
        for(int l=0;l<NC;l++){
            memcpy(sb.data(),h.data(),H*4);rn_c(h.data(),in_n[l],H);

            // === QKV GEMM ===
            {auto t0g=std::chrono::steady_clock::now();
            float ascale_q=dynamic_ascale(h.data(),H);float ais=1.0f/ascale_q;float*iA=h.data();
            memset(cq.Am,0,(size_t)1*cq.KD);
            for(int k=0;k<H;k++){float v=iA[k];if(!std::isfinite(v))v=0;int q=(int)roundf(v*ais);if(q>127)q=127;else if(q<-127)q=-127;cq.Am[k]=(int8_t)q;}
            t_q+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0g).count();
            auto t1g=std::chrono::steady_clock::now();cq.bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            t_syncA+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t1g).count();
            auto t2g=std::chrono::steady_clock::now();auto r=(*cq.k)((unsigned)3,*cq.bI,(unsigned)cq.ins.size(),*cq.bA,*cq.layerB[l],*cq.bC);r.wait();
            t_kern+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t2g).count();
            auto t3g=std::chrono::steady_clock::now();cq.bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            t_syncC+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t3g).count();
            auto t4g=std::chrono::steady_clock::now();float cs=ascale_qkv*wsc[l].qk;for(int n=0;n<4096;n++){float val=(float)cq.Cm[n]*cs;if(!std::isfinite(val))val=0;qo[n]=val;}
            t_dq+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t4g).count();}

            cn(qo.data(),4096);memcpy(ko.data(),&qo[2048],4096);memcpy(vo.data(),&qo[3072],4096);
            float*qn=qn_w[l],*kn=kn_w[l];

            auto ta0=std::chrono::steady_clock::now();
            for(int hh=0;hh<NH;hh++){double sq=0;for(int d=0;d<HD;d++)sq+=qo[hh*HD+d]*qo[hh*HD+d];float iq=1.0f/sqrtf((float)(sq/HD)+EPS);for(int d=0;d<HD;d++)qo[hh*HD+d]*=iq*qn[d];ra(&qo[hh*HD],HD,sp);
                if(hh%GQA==0){int kvh=hh/GQA;double sk=0;for(int d=0;d<HD;d++)sk+=ko[kvh*HD+d]*ko[kvh*HD+d];float ik=1.0f/sqrtf((float)(sk/HD)+EPS);for(int d=0;d<HD;d++)ko[kvh*HD+d]*=ik*kn[d];ra(&ko[kvh*HD],HD,sp);memcpy(&kv[l].k[sp*NKV*HD+kvh*HD],&ko[kvh*HD],HD*4);memcpy(&kv[l].v[sp*NKV*HD+kvh*HD],&vo[kvh*HD],HD*4);}}
            kv[l].n=sp+1;int cl=kv[l].n;
            for(int hh=0;hh<NH;hh++){int kvh=hh/GQA;std::vector<float>sc(sp+pi+1);
                for(int p=0;p<sp+pi+1;p++){double s=0;for(int d=0;d<HD;d++)s+=qo[hh*HD+d]*kv[l].k[p*NKV*HD+kvh*HD+d];sc[p]=(float)(s/sqrtf(HD));}
                sm(sc.data(),sp+pi+1);for(int d=0;d<HD;d++){float s=0;for(int p=0;p<sp+pi+1;p++)s+=sc[p]*kv[l].v[p*NKV*HD+kvh*HD+d];at[hh*HD+d]=s;}
            }
            t_attn_total+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-ta0).count();

            // === O GEMM ===
            {auto t0g=std::chrono::steady_clock::now();
            float ascale_o=dynamic_ascale(at.data(),NH*HD);float ais=1.0f/ascale_o;float*iA=at.data();
            memset(co.Am,0,(size_t)1*co.KD);
            for(int k=0;k<NH*HD;k++){float v=iA[k];if(!std::isfinite(v))v=0;int q=(int)roundf(v*ais);if(q>127)q=127;else if(q<-127)q=-127;co.Am[k]=(int8_t)q;}
            t_q+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0g).count();
            auto t1g=std::chrono::steady_clock::now();co.bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            t_syncA+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t1g).count();
            auto t2g=std::chrono::steady_clock::now();auto r=(*co.k)((unsigned)3,*co.bI,(unsigned)co.ins.size(),*co.bA,*co.layerB[l],*co.bC);r.wait();
            t_kern+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t2g).count();
            auto t3g=std::chrono::steady_clock::now();co.bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            t_syncC+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t3g).count();
            auto t4g=std::chrono::steady_clock::now();float cs=ascale_o*wsc[l].o_;for(int n=0;n<H;n++){float val=(float)co.Cm[n]*cs;if(!std::isfinite(val))val=0;oo[n]=val;}
            t_dq+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t4g).count();}

            cn(oo.data(),H);for(int i=0;i<H;i++)h[i]=sb[i]+oo[i];

            memcpy(sb.data(),h.data(),H*4);rn_c(h.data(),pa_n[l],H);

            // === GU GEMM ===
            {auto t0g=std::chrono::steady_clock::now();
            float ascale_g=dynamic_ascale(h.data(),H);float ais=1.0f/ascale_g;float*iA=h.data();
            memset(cg.Am,0,(size_t)1*cg.KD);
            for(int k=0;k<H;k++){float v=iA[k];if(!std::isfinite(v))v=0;int q=(int)roundf(v*ais);if(q>127)q=127;else if(q<-127)q=-127;cg.Am[k]=(int8_t)q;}
            t_q+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0g).count();
            auto t1g=std::chrono::steady_clock::now();cg.bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            t_syncA+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t1g).count();
            auto t2g=std::chrono::steady_clock::now();auto r=(*cg.k)((unsigned)3,*cg.bI,(unsigned)cg.ins.size(),*cg.bA,*cg.layerB[l],*cg.bC);r.wait();
            t_kern+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t2g).count();
            auto t3g=std::chrono::steady_clock::now();cg.bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            t_syncC+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t3g).count();
            auto t4g=std::chrono::steady_clock::now();float cs=ascale_g*wsc[l].g_;for(int n=0;n<6144;n++){float val=(float)cg.Cm[n]*cs;if(!std::isfinite(val))val=0;gt[n]=val;}
            t_dq+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t4g).count();}

            cn(gt.data(),6144);for(int i=0;i<IM;i++){float gv=gt[i];if(!std::isfinite(gv))gv=0;su[i]=(gv/(1.0f+expf(-gv)))*gt[IM+i];}

            // === D GEMM ===
            {auto t0g=std::chrono::steady_clock::now();
            float ascale_d=dynamic_ascale(su.data(),IM);float ais=1.0f/ascale_d;float*iA=su.data();
            memset(cd.Am,0,(size_t)1*cd.KD);
            for(int k=0;k<IM;k++){float v=iA[k];if(!std::isfinite(v))v=0;int q=(int)roundf(v*ais);if(q>127)q=127;else if(q<-127)q=-127;cd.Am[k]=(int8_t)q;}
            t_q+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0g).count();
            auto t1g=std::chrono::steady_clock::now();cd.bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            t_syncA+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t1g).count();
            auto t2g=std::chrono::steady_clock::now();auto r=(*cd.k)((unsigned)3,*cd.bI,(unsigned)cd.ins.size(),*cd.bA,*cd.layerB[l],*cd.bC);r.wait();
            t_kern+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t2g).count();
            auto t3g=std::chrono::steady_clock::now();cd.bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            t_syncC+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t3g).count();
            auto t4g=std::chrono::steady_clock::now();float cs=ascale_d*wsc[l].d_;for(int n=0;n<H;n++){float val=(float)cd.Cm[n]*cs;if(!std::isfinite(val))val=0;dwo[n]=val;}
            t_dq+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t4g).count();}

            cn(dwo.data(),H);for(int i=0;i<H;i++)h[i]=sb[i]+dwo[i];
        }

        auto tlm0=std::chrono::steady_clock::now();
        memcpy(sb.data(),h.data(),H*4);rn_c(sb.data(),fin,H);
        float mx=-1e30f;
        for(int n=0;n<NV;n++){double s=0;const float*e=&lm_head_f32[(size_t)n*H];
            for(int k=0;k<H;k++)s+=(double)sb[k]*e[k];lg[n]=std::isfinite((float)s)?(float)s:-1e30f;if(lg[n]>mx)mx=lg[n];}
        double sum=0;for(int i=0;i<NV;i++){float d=lg[i]-mx;if(d<-80)d=-80;lg[i]=expf(d);sum+=lg[i];}
        float rr=(float)rand()/RAND_MAX*(float)sum,acc=0;int tok=0;for(int i=0;i<NV;i++){acc+=lg[i];if(acc>=rr){tok=i;break;}}
        t_lm_total+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tlm0).count();

        double mss=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ts).count();
        printf("  [%d] %d (%.0fms)\n",step,tok,mss);
        for(int i=0;i<H;i++)h[i]=emb_f32[(size_t)tok*H+i];sp++;
>>>>>>> d34a5b3d (fix(npu): finish v4.cpp lm_head untied-embeddings fix + decode quant bug)
    }
    
    // ── 4. Generate instruction sequences ──
    // We have the model with weights loaded. Now generate per-layer seqs.
    // qwen3_npu_sequence::Impl generates instruction sequences for each layer.
    // Its constructor and gen_layer_seq() produce the fused instruction stream.
    
    // Create npu_sequence buffer (~4096 bytes from earlier analysis)
    uint8_t seq_buf[4096];
    memset(seq_buf, 0, sizeof(seq_buf));
    ((uint32_t*)(seq_buf + 0x20))[0] = 4;  // op_line_count = 4
    
    // Create qwen3_npu_sequence::Impl
    uint8_t seq_impl_buf[4096];
    memset(seq_impl_buf, 0, sizeof(seq_impl_buf));
    
    printf("\nGenerating layer instruction sequences...\n");
    t0 = Clock::now();
    try {
        flm.seq_impl_ctor(seq_impl_buf, cfg, (unsigned)4096);
        printf("  seq_impl constructed\n");
    } catch (...) {
        printf("  ⚠️  seq_impl ctor failed (non-critical, trying gen_layer directly)\n");
    }
    
    // Generate layer 0's instruction sequence
    for (int l = 0; l < 28; l++) {
        try {
            flm.seq_gen_layer(seq_impl_buf, seq_buf, (unsigned)l);
            flm.cmds2seq(seq_buf);
            
            // Extract instructions
            uint32_t** vb = (uint32_t**)(seq_buf + 0x40);
            uint32_t** ve = (uint32_t**)(seq_buf + 0x50);
            size_t ninstr = 0;
            if (vb && ve && *vb && *ve && *ve > *vb) {
                ninstr = *ve - *vb;
            }
            printf("  Layer %2d: %zu instructions\n", l, ninstr);
        } catch (const std::exception& e) {
            printf("  Layer %2d: FAILED — %s\n", l, e.what());
            break;
        }
    }
    printf("Generation done: %.0f ms\n\n", ms(t0));
    
    // ── 5. Submit via XRT ──
    printf("Opening XRT device for submission...\n");
    xrt::device dev(0);
    auto xc = xrt::xclbin([]()->std::vector<char>{
        std::string xp = "/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/layer.xclbin";
        FILE* f = fopen(xp.c_str(), "rb"); fseek(f,0,2); long sz = ftell(f); fseek(f,0,0);
        std::vector<char> d(sz); fread(d.data(),1,sz,f); fclose(f); return d;
    }());
    dev.register_xclbin(xc);
    auto kern = xrt::kernel(dev, xc.get_uuid(), "MLIR_AIE");
    printf("✅ Kernel ready (data gid=%d, inst gid=%d)\n",
           kern.group_id(3), kern.group_id(1));
    
    // The BOs are managed by npu_xclbin_manager. In FLM's architecture,
    // the manager allocates and owns the BOs. We need to retrieve them.
    // Since we allocated the manager ourselves, the BOs are created inside it.
    // We need to access them through the same manager interface.
    // 
    // For now: the manager constructor handles BO creation internally.
    // If the constructor succeeded, the BOs exist in the manager's state.
    // We can access them by reverse-engineering the manager's internal layout
    // or by using a separate XRT kernel submission with our own BOs.
    
    printf("\n✅ Engine ready. Need npu_xclbin_manager accessor methods\n");
    printf("   to retrieve BOs for instruction submission.\n");
    printf("   Current BOs are held inside the manager at %p\n", manager);
    
    // Cleanup
    if (flm.model_impl_dtor) flm.model_impl_dtor(model);
    
    return 0;
}

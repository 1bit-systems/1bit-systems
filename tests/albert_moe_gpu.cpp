// albert_moe_gpu.cpp — Albert-MoE-13 GPU inference (HIP)
// Pre-loads all weights, runs full 31-block pipeline on GPU
//
// Build: cmake --build . --target albert_moe_gpu -j8
// Run:   ./build/albert_moe_gpu "prompt"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <sys/stat.h>

#define HIP_OK(e) do{auto _s=(e);if(_s!=hipSuccess){fprintf(stderr,"HIP Error %d at %s:%d\n",_s,__FILE__,__LINE__);abort();}}while(0)

// ── Architecture ──
constexpr int H=256, N_HEADS=4, HD=64, N_EXP=12, TOP_K=3;
constexpr int FFN_HID=H*4, VOCAB=32000, MAX_SEQ=256, N_BLOCKS=31, N_STREAMS=2;
constexpr int BLK=256;

// ── Pre-computed RoPE cache (CPU, uploaded to GPU) ──
static float cos_cache[MAX_SEQ * HD/2];
static float sin_cache[MAX_SEQ * HD/2];
static void init_rope() {
    for (int pos=0; pos<MAX_SEQ; pos++)
        for (int d=0; d<HD/2; d++) {
            float th = pos * powf(10000.0f, -2.0f*d/(float)HD);
            cos_cache[pos*HD/2+d] = cosf(th);
            sin_cache[pos*HD/2+d] = sinf(th);
        }
}

// ── Weight loading ──
static std::vector<float> load_bin(const std::string& p){
    std::ifstream f(p,std::ios::binary|std::ios::ate);
    if(!f){fprintf(stderr,"Missing: %s\n",p.c_str());exit(1);}
    size_t n=f.tellg()/sizeof(float);f.seekg(0);
    std::vector<float> d(n);f.read((char*)d.data(),n*sizeof(float));return d;
}
static std::string L(int i){return std::to_string(i);}
#define W(N) load_bin(std::string("/tmp/albert_weights/")+N)

static void upf32(const std::vector<float>& s,float*d,hipStream_t h=0){
    HIP_OK(hipMemcpyAsync(d,s.data(),s.size()*4,hipMemcpyHostToDevice,h));
}

// ── Kernel declarations (from kernels/albert_moe.hip) ──
__global__ void layernorm_k(float*x,const float*w,const float*b,int n);
__global__ void mm_k(float*out,const float*in,const float*wt,int M,int K);
__global__ void mm_t_k(float*out,const float*in,const float*wt,int M,int K);
__global__ void gelu_k(float*x,int n);
__global__ void copy_k(float*dst,const float*src,int n);
__global__ void add_k(float*dst,const float*a,const float*b,int n);
__global__ void embed_k(float*h,const float*embed,int token);
__global__ void attn_k(float*attn_out,float*q,float*k,float*v,
    float*k_cache,float*v_cache,int pos,const float*cos_,const float*sin_);
__global__ void moe_k(float*moe_out,const float*h,
    const float*gate_w,const float*fc_w,const float*fc_b,
    const float*proj_w,const float*proj_b);
__global__ void lm_head_k(float*logits,int*argmax,const float*h,const float*lm_head_w);

// ── Per-stream weight struct (stays on GPU) ──
struct StreamGPU {
    float *ln1_w, *ln1_b;
    float *q_w, *k_w, *v_w, *o_w;
    float *ln2_w, *ln2_b;
    float *gate_w;
};
// ── Per-expert-block weight struct (blocks 3-30 have experts) ──
struct ExpertGPU {
    float *fc_w[N_EXP], *fc_b[N_EXP];
    float *proj_w[N_EXP], *proj_b[N_EXP];
};

int main(int argc, char** argv) {
    setvbuf(stdout,NULL,_IONBF,0);
    printf("=== Albert-MoE-13 GPU Inference ===\n");
    if(argc<2){printf("Usage: %s \"prompt\"\n",argv[0]);return 1;}

    init_rope();

    // ── Load base weights ──
    auto embed   = W("embed_weight.bin");
    auto lm_head = W("lm_head_weight.bin");
    auto ln_f_w  = W("ln_f_weight.bin");
    auto ln_f_b  = W("ln_f_bias.bin");
    printf("Base weights: embed=%zu, lm_head=%zu\n", embed.size(), lm_head.size());

    // ── GPU memory: base ──
    float *d_embed, *d_lm_head, *d_ln_f_w, *d_ln_f_b, *d_h, *d_attn_out, *d_moe_out, *d_buf;
    float *d_cos, *d_sin;
    float *d_k_cache, *d_v_cache;
    float *d_logits; int *d_argmax;
    float *d_qkv; // [3*H] for Q, K, V projections
    HIP_OK(hipMalloc(&d_embed,   (size_t)VOCAB*H*4));
    HIP_OK(hipMalloc(&d_lm_head, (size_t)VOCAB*H*4));
    HIP_OK(hipMalloc(&d_ln_f_w,  H*4));
    HIP_OK(hipMalloc(&d_ln_f_b,  H*4));
    HIP_OK(hipMalloc(&d_h,       H*4));
    HIP_OK(hipMalloc(&d_attn_out,H*4));
    HIP_OK(hipMalloc(&d_moe_out, H*4));
    HIP_OK(hipMalloc(&d_buf,     H*4));
    HIP_OK(hipMalloc(&d_qkv,     3*H*4));
    HIP_OK(hipMalloc(&d_cos,     (size_t)MAX_SEQ*HD/2*4));
    HIP_OK(hipMalloc(&d_sin,     (size_t)MAX_SEQ*HD/2*4));
    HIP_OK(hipMalloc(&d_k_cache, (size_t)N_BLOCKS*N_STREAMS*MAX_SEQ*N_HEADS*HD*4));
    HIP_OK(hipMalloc(&d_v_cache, (size_t)N_BLOCKS*N_STREAMS*MAX_SEQ*N_HEADS*HD*4));
    HIP_OK(hipMalloc(&d_logits,  (size_t)VOCAB*4));
    HIP_OK(hipMalloc(&d_argmax,  4));

    hipStream_t st; HIP_OK(hipStreamCreate(&st));
    upf32(embed,   d_embed,   st);
    upf32(lm_head, d_lm_head, st);
    upf32(ln_f_w,  d_ln_f_w, st);
    upf32(ln_f_b,  d_ln_f_b, st);
    HIP_OK(hipMemcpyAsync(d_cos, cos_cache, MAX_SEQ*HD/2*4, hipMemcpyHostToDevice, st));
    HIP_OK(hipMemcpyAsync(d_sin, sin_cache, MAX_SEQ*HD/2*4, hipMemcpyHostToDevice, st));
    HIP_OK(hipMemsetAsync(d_k_cache, 0, (size_t)N_BLOCKS*N_STREAMS*MAX_SEQ*N_HEADS*HD*4, st));
    HIP_OK(hipMemsetAsync(d_v_cache, 0, (size_t)N_BLOCKS*N_STREAMS*MAX_SEQ*N_HEADS*HD*4, st));

    // ── Load stream weights ──
    std::vector<StreamGPU> sg(N_BLOCKS*N_STREAMS);
    auto A=[&](float*&p,int n){HIP_OK(hipMalloc(&p,n*4));};
    printf("Loading %d blocks × %d streams...\n", N_BLOCKS, N_STREAMS);
    for(int blk=0;blk<N_BLOCKS;blk++){
        for(int stm=0;stm<N_STREAMS;stm++){
            int idx=blk*N_STREAMS+stm; char s=stm?'b':'a';
            auto& g=sg[idx]; char key[256];
            auto L=[&](const char*fmt){snprintf(key,sizeof(key),fmt,blk,s);return std::string(key);};
            A(g.ln1_w,H); upf32(W(L("blocks_%d_stream_%c_ln1_weight.bin")),g.ln1_w,st);
            A(g.ln1_b,H); upf32(W(L("blocks_%d_stream_%c_ln1_bias.bin")),g.ln1_b,st);
            A(g.q_w,H*H); upf32(W(L("blocks_%d_stream_%c_attn_q_proj_weight.bin")),g.q_w,st);
            A(g.k_w,H*H); upf32(W(L("blocks_%d_stream_%c_attn_k_proj_weight.bin")),g.k_w,st);
            A(g.v_w,H*H); upf32(W(L("blocks_%d_stream_%c_attn_v_proj_weight.bin")),g.v_w,st);
            A(g.o_w,H*H); upf32(W(L("blocks_%d_stream_%c_attn_o_proj_weight.bin")),g.o_w,st);
            A(g.ln2_w,H); upf32(W(L("blocks_%d_stream_%c_ln2_weight.bin")),g.ln2_w,st);
            A(g.ln2_b,H); upf32(W(L("blocks_%d_stream_%c_ln2_bias.bin")),g.ln2_b,st);
            A(g.gate_w,N_EXP*H); upf32(W(L("blocks_%d_stream_%c_moe_gate_weight.bin")),g.gate_w,st);
        }
    }

    // ── Load expert weights (blocks 3-30, each has 12 experts) ──
    const int EXP_BLK_START=3, EXP_BLK_END=30;
    std::vector<ExpertGPU> eg(EXP_BLK_END-EXP_BLK_START+1);
    printf("Loading experts (blocks %d-%d)...\n", EXP_BLK_START, EXP_BLK_END);
    int exp_total=0;
    for(int eb=EXP_BLK_START;eb<=EXP_BLK_END;eb++){
        auto& e=eg[eb-EXP_BLK_START]; char key[256];
        for(int ex=0;ex<N_EXP;ex++){
            snprintf(key,sizeof(key),"blocks_%d_experts_%d_c_fc_weight.bin",eb,ex);
            auto w=W(key); if(w.empty()){e.fc_w[ex]=nullptr;e.fc_b[ex]=nullptr;e.proj_w[ex]=nullptr;e.proj_b[ex]=nullptr;continue;}
            A(e.fc_w[ex],H*FFN_HID); upf32(w,e.fc_w[ex],st);
            snprintf(key,sizeof(key),"blocks_%d_experts_%d_c_fc_bias.bin",eb,ex);
            A(e.fc_b[ex],FFN_HID); upf32(W(key),e.fc_b[ex],st);
            snprintf(key,sizeof(key),"blocks_%d_experts_%d_c_proj_weight.bin",eb,ex);
            A(e.proj_w[ex],FFN_HID*H); upf32(W(key),e.proj_w[ex],st);
            snprintf(key,sizeof(key),"blocks_%d_experts_%d_c_proj_bias.bin",eb,ex);
            A(e.proj_b[ex],H); upf32(W(key),e.proj_b[ex],st);
            exp_total++;
        }
    }
    HIP_OK(hipStreamSynchronize(st));
    printf("Loaded %d experts total\n", exp_total);

    // ── Tokenize (simple char mapping) ──
    int tokens[MAX_SEQ], seq=0;
    tokens[seq++]=1; // BOS
    for(const char*p=argv[1];*p&&seq<MAX_SEQ-1;p++)
        tokens[seq++]=(unsigned char)*p+3;
    printf("Prompt: %s (%d tokens)\n", argv[1], seq);

    // ── Forward pass ──
    int g1=(H+BLK-1)/BLK, gV=(VOCAB+BLK-1)/BLK;
    auto t0=std::chrono::high_resolution_clock::now();

    for(int pos=0;pos<seq;pos++){
        // Embed
        embed_k<<<1,BLK,0,st>>>(d_h,d_embed,tokens[pos]);

        for(int blk=0;blk<N_BLOCKS;blk++){
            for(int stm=0;stm<N_STREAMS;stm++){
                int idx=blk*N_STREAMS+stm;
                int cid=idx; // cache ID = stream index
                auto& g=sg[idx];
                float *kc=d_k_cache+(size_t)cid*MAX_SEQ*N_HEADS*HD;
                float *vc=d_v_cache+(size_t)cid*MAX_SEQ*N_HEADS*HD;

                // LN1
                copy_k<<<g1,BLK,0,st>>>(d_buf,d_h,H);
                layernorm_k<<<1,BLK,0,st>>>(d_buf,g.ln1_w,g.ln1_b,H);

                // QKV projections
                float *d_q=d_qkv, *d_k=d_qkv+H, *d_v=d_qkv+2*H;
                mm_t_k<<<g1,BLK,0,st>>>(d_q, d_buf, g.q_w, H, H);
                mm_t_k<<<g1,BLK,0,st>>>(d_k, d_buf, g.k_w, H, H);
                mm_t_k<<<g1,BLK,0,st>>>(d_v, d_buf, g.v_w, H, H);

                // Attention (with RoPE + KV cache)
                attn_k<<<1,BLK,0,st>>>(d_attn_out, d_q, d_k, d_v, kc, vc, pos, d_cos, d_sin);

                // O projection + residual
                mm_t_k<<<g1,BLK,0,st>>>(d_buf, d_attn_out, g.o_w, H, H);
                add_k<<<g1,BLK,0,st>>>(d_h, d_h, d_buf, H);

                // LN2
                copy_k<<<g1,BLK,0,st>>>(d_buf,d_h,H);
                layernorm_k<<<1,BLK,0,st>>>(d_buf,g.ln2_w,g.ln2_b,H);

                // MoE: select expert block
                // For blocks 0-2, use expert block = 3 + blk%3; for blocks 3+, use same block
                int eb=(blk<3)?3+(blk%3):blk;
                if(eb>=EXP_BLK_START&&eb<=EXP_BLK_END){
                    auto& ex=eg[eb-EXP_BLK_START];
                    // Use expert 0 weights for simplicity; real routing needs top-3 dispatch
                    moe_k<<<1,BLK,0,st>>>(d_moe_out,d_buf,
                        g.gate_w,
                        ex.fc_w[0],ex.fc_b[0],
                        ex.proj_w[0],ex.proj_b[0]);
                }else{
                    // No experts — skip MoE
                    HIP_OK(hipMemsetAsync(d_moe_out,0,H*4,st));
                }

                // Residual
                add_k<<<g1,BLK,0,st>>>(d_h, d_h, d_moe_out, H);
            }
        }
    }

    // ── Generation loop ──
    int n_gen=0, output[32];
    for(int gen=0;gen<10;gen++){
        // Final LN
        copy_k<<<g1,BLK,0,st>>>(d_buf,d_h,H);
        layernorm_k<<<1,BLK,0,st>>>(d_buf,d_ln_f_w,d_ln_f_b,H);

        // lm_head
        lm_head_k<<<1,gV,0,st>>>(d_logits,d_argmax,d_buf,d_lm_head);
        HIP_OK(hipStreamSynchronize(st));

        int next; HIP_OK(hipMemcpy(&next,d_argmax,4,hipMemcpyDeviceToHost));
        output[n_gen++]=next;
        if(next==2||n_gen>=10)break;

        // Embed next token
        embed_k<<<1,BLK,0,st>>>(d_h,d_embed,next);
        int pos=seq+n_gen-1;

        for(int blk=0;blk<N_BLOCKS;blk++){
            for(int stm=0;stm<N_STREAMS;stm++){
                int idx=blk*N_STREAMS+stm;
                int cid=idx;
                auto& g=sg[idx];
                float *kc=d_k_cache+(size_t)cid*MAX_SEQ*N_HEADS*HD;
                float *vc=d_v_cache+(size_t)cid*MAX_SEQ*N_HEADS*HD;

                copy_k<<<g1,BLK,0,st>>>(d_buf,d_h,H);
                layernorm_k<<<1,BLK,0,st>>>(d_buf,g.ln1_w,g.ln1_b,H);

                float *d_q=d_qkv, *d_k=d_qkv+H, *d_v=d_qkv+2*H;
                mm_t_k<<<g1,BLK,0,st>>>(d_q, d_buf, g.q_w, H, H);
                mm_t_k<<<g1,BLK,0,st>>>(d_k, d_buf, g.k_w, H, H);
                mm_t_k<<<g1,BLK,0,st>>>(d_v, d_buf, g.v_w, H, H);

                attn_k<<<1,BLK,0,st>>>(d_attn_out, d_q, d_k, d_v, kc, vc, pos, d_cos, d_sin);

                mm_t_k<<<g1,BLK,0,st>>>(d_buf, d_attn_out, g.o_w, H, H);
                add_k<<<g1,BLK,0,st>>>(d_h, d_h, d_buf, H);

                copy_k<<<g1,BLK,0,st>>>(d_buf,d_h,H);
                layernorm_k<<<1,BLK,0,st>>>(d_buf,g.ln2_w,g.ln2_b,H);

                // MoE: use appropriate expert block
                int eb=(blk<3)?3+(blk%3):blk;
                if(eb>=EXP_BLK_START&&eb<=EXP_BLK_END){
                    auto& ex=eg[eb-EXP_BLK_START];
                    // Use expert 0 for single-expert test (top-3 dispatch needs 3 launches)
                    moe_k<<<1,BLK,0,st>>>(d_moe_out,d_buf,
                        g.gate_w,
                        ex.fc_w[0],ex.fc_b[0],
                        ex.proj_w[0],ex.proj_b[0]);
                }else{
                    HIP_OK(hipMemsetAsync(d_moe_out,0,H*4,st));
                }
                add_k<<<g1,BLK,0,st>>>(d_h, d_h, d_moe_out, H);
            }
        }
    }

    float ms=std::chrono::duration<float,std::milli>(std::chrono::high_resolution_clock::now()-t0).count();
    printf("Generated %d tokens in %.0f ms (%.1f tok/s)\n",n_gen,ms,n_gen/(ms/1000.0f));
    printf("Tokens:");
    for(int i=0;i<n_gen;i++)printf(" %d",output[i]);
    printf("\n");

    // ── Cleanup ──
    hipFree(d_embed);hipFree(d_lm_head);hipFree(d_ln_f_w);hipFree(d_ln_f_b);
    hipFree(d_h);hipFree(d_attn_out);hipFree(d_moe_out);hipFree(d_buf);hipFree(d_qkv);
    hipFree(d_cos);hipFree(d_sin);hipFree(d_k_cache);hipFree(d_v_cache);
    hipFree(d_logits);hipFree(d_argmax);
    for(auto& g:sg){hipFree(g.ln1_w);hipFree(g.ln1_b);hipFree(g.q_w);hipFree(g.k_w);hipFree(g.v_w);hipFree(g.o_w);hipFree(g.ln2_w);hipFree(g.ln2_b);hipFree(g.gate_w);}
    for(auto& e:eg)for(int x=0;x<N_EXP;x++){if(e.fc_w[x]){hipFree(e.fc_w[x]);hipFree(e.fc_b[x]);hipFree(e.proj_w[x]);hipFree(e.proj_b[x]);}}
    hipStreamDestroy(st);
    printf("Done.\n");
    return 0;
}

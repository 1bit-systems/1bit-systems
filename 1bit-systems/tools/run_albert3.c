// run_albert3.c — Albert-MoE-13 inference from individual .bin files
// Build: gcc -O3 run_albert3.c -lm -o run_albert3 && ./run_albert3
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#define H 256
#define N_HEADS 4
#define HEAD_DIM 64
#define N_EXP 12
#define TOP_K 3
#define VOCAB 32000
#define FFN_HIDDEN 1024
#define MAX_SEQ 256
#define MAX_BLOCKS 33

static float *load_bin(const char *name, int64_t *n) {
    char path[256]; snprintf(path, sizeof(path), "/tmp/albert_weights/%s", name);
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr,"Missing: %s\n",name); *n=0; return NULL; }
    fseek(f, 0, SEEK_END); int64_t sz = ftell(f); fseek(f, 0, SEEK_SET);
    float *d = malloc(sz); fread(d, 1, sz, f); fclose(f);
    *n = sz / 4; return d;
}

// Remove trailing .weight or .bias from key to get the base name
static void strip_suffix(char *dst, const char *src) {
    strcpy(dst, src);
    char *p = strstr(dst, ".weight"); if (p) { *p = 0; return; }
    p = strstr(dst, ".bias"); if (p) { *p = 0; return; }
}

static float gelu(float x) {
    float t = tanhf(0.79788456f * (x + 0.044715f * x * x * x));
    return 0.5f * x * (1.0f + t);
}

int main(int argc, char **argv) {
    printf("Albert-MoE-13 C Inference (v3)\n");
    
    // Load embed + lm_head
    int64_t n; float *embed_w = load_bin("embed_weight.bin", &n);
    float *lm_head_w = load_bin("lm_head_weight.bin", &n);
    float *ln_f_w = load_bin("ln_f_weight.bin", &n);
    float *ln_f_b = load_bin("ln_f_bias.bin", &n);
    if (!embed_w) return 1;
    printf("Embed: %ld elements\n", n);
    
    // Tokenize prompt
    int tokens[MAX_SEQ], n_tokens = 1;
    tokens[0] = 1; // BOS
    const char *prompt = argc > 1 ? argv[1] : "Hello";
    for (const char *p = prompt; *p && n_tokens < MAX_SEQ-1; p++)
        tokens[n_tokens++] = (unsigned char)*p % 32000; // crude mapping
    printf("Prompt: '%s' → %d tokens\n", prompt, n_tokens);
    
    float x[MAX_SEQ * H], buf[MAX_SEQ * H];
    
    // Generate
    clock_t t0 = clock();
    for (int gen = 0; gen < 8; gen++) {
        // Embed
        for (int i = 0; i < n_tokens; i++)
            memcpy(&x[i*H], &embed_w[tokens[i]*H], H*sizeof(float));
        
        // Run all blocks (single stream simplified)
        for (int blk = 0; blk < MAX_BLOCKS; blk++) {
            char key[256], key_w[256], key_b[256];
            
            for (int stream = 0; stream < 2; stream++) {
                char s = 'a' + stream;
                
                // LN1
                snprintf(key, sizeof(key), "blocks_%d_stream_%c_ln1", blk, s);
                snprintf(key_w, sizeof(key), "%s_weight.bin", key);
                snprintf(key_b, sizeof(key), "%s_bias.bin", key);
                float *ln1_w = load_bin(key_w, &n); 
                float *ln1_b = load_bin(key_b, &n);
                
                // Attention weights
                snprintf(key_w, sizeof(key_w), "blocks_%d_stream_%c_attn_q_proj_weight.bin", blk, s);
                float *q_w = load_bin(key_w, &n);
                snprintf(key_w, sizeof(key_w), "blocks_%d_stream_%c_attn_k_proj_weight.bin", blk, s);
                float *k_w = load_bin(key_w, &n);
                snprintf(key_w, sizeof(key_w), "blocks_%d_stream_%c_attn_v_proj_weight.bin", blk, s);
                float *v_w = load_bin(key_w, &n);
                snprintf(key_w, sizeof(key_w), "blocks_%d_stream_%c_attn_o_proj_weight.bin", blk, s);
                float *o_w = load_bin(key_w, &n);
                
                // MoE gate
                snprintf(key_w, sizeof(key_w), "blocks_%d_stream_%c_moe_gate_weight.bin", blk, s);
                float *gate_w = load_bin(key_w, &n);
                
                if (!ln1_w || !q_w) continue; // skip missing blocks
                
                // LN1
                memcpy(buf, x, n_tokens*H*sizeof(float));
                for (int i = 0; i < n_tokens; i++) {
                    float mean=0,var=0;
                    for (int j=0;j<H;j++) mean+=buf[i*H+j];
                    mean/=H; for (int j=0;j<H;j++){float d=buf[i*H+j]-mean;var+=d*d;}
                    var=sqrtf(var/H+1e-5f);
                    for (int j=0;j<H;j++) buf[i*H+j]=(buf[i*H+j]-mean)/var*ln1_w[j]+ln1_b[j];
                }
                
                // Attention
                float q[n_tokens*H],k[n_tokens*H],v[n_tokens*H],attn[n_tokens*n_tokens];
                for (int i=0;i<n_tokens;i++) for (int j=0;j<H;j++){float s=0;for(int t=0;t<H;t++)s+=buf[i*H+t]*q_w[j*H+t];q[i*H+j]=s;}
                for (int i=0;i<n_tokens;i++) for (int j=0;j<H;j++){float s=0;for(int t=0;t<H;t++)s+=buf[i*H+t]*k_w[j*H+t];k[i*H+j]=s;}
                for (int i=0;i<n_tokens;i++) for (int j=0;j<H;j++){float s=0;for(int t=0;t<H;t++)s+=buf[i*H+t]*v_w[j*H+t];v[i*H+j]=s;}
                
                float scale=1.0f/sqrtf(HEAD_DIM);
                for (int h=0;h<N_HEADS;h++) {
                    for (int i=0;i<n_tokens;i++) for (int j=0;j<n_tokens;j++){float s=0;for(int d=0;d<HEAD_DIM;d++)s+=q[i*H+h*HEAD_DIM+d]*k[j*H+h*HEAD_DIM+d];attn[i*n_tokens+j]=s*scale;}
                    for (int i=0;i<n_tokens;i++){float mv=attn[i*n_tokens];for(int j=1;j<n_tokens;j++)if(attn[i*n_tokens+j]>mv)mv=attn[i*n_tokens+j];float su=0;for(int j=0;j<n_tokens;j++){attn[i*n_tokens+j]=expf(attn[i*n_tokens+j]-mv);su+=attn[i*n_tokens+j];}for(int j=0;j<n_tokens;j++)attn[i*n_tokens+j]/=su;}
                    for (int i=0;i<n_tokens;i++){float oh[HEAD_DIM]={0};for(int j=0;j<n_tokens;j++)for(int d=0;d<HEAD_DIM;d++)oh[d]+=attn[i*n_tokens+j]*v[j*H+h*HEAD_DIM+d];for(int d=0;d<HEAD_DIM;d++)buf[i*H+h*HEAD_DIM+d]=oh[d];}
                }
                // O proj + residual
                float tmp[n_tokens*H];
                for (int i=0;i<n_tokens;i++) for (int j=0;j<H;j++){float s=0;for(int t=0;t<H;t++)s+=buf[i*H+t]*o_w[j*H+t];tmp[i*H+j]=s;}
                for (int i=0;i<n_tokens*H;i++) x[i]+=tmp[i];
                
                // LN2
                snprintf(key_w, sizeof(key_w), "blocks_%d_stream_%c_ln2_weight.bin", blk, s);
                snprintf(key_b, sizeof(key_b), "blocks_%d_stream_%c_ln2_bias.bin", blk, s);
                float *ln2_w = load_bin(key_w, &n);
                float *ln2_b = load_bin(key_b, &n);
                
                memcpy(buf, x, n_tokens*H*sizeof(float));
                for (int i = 0; i < n_tokens; i++) {
                    float mean=0,var=0;
                    for (int j=0;j<H;j++) mean+=buf[i*H+j];
                    mean/=H; for (int j=0;j<H;j++){float d=buf[i*H+j]-mean;var+=d*d;}
                    var=sqrtf(var/H+1e-5f);
                    for (int j=0;j<H;j++) buf[i*H+j]=(buf[i*H+j]-mean)/var*ln2_w[j]+ln2_b[j];
                }
                
                // MoE with averaged experts
                if (gate_w) {
                    float gs[n_tokens*N_EXP];
                    for (int i=0;i<n_tokens;i++) for (int e=0;e<N_EXP;e++){float s=0;for(int j=0;j<H;j++)s+=buf[i*H+j]*gate_w[e*H+j];gs[i*N_EXP+e]=s;}
                    
                    // Average expert weights from blocks 3-32
                    float fc_w[H*FFN_HIDDEN]={0},fc_b[FFN_HIDDEN]={0},pj_w[FFN_HIDDEN*H]={0},pj_b[H]={0};
                    int ec=0;
                    for (int eb=3;eb<MAX_BLOCKS;eb++) for (int e=0;e<N_EXP;e++){
                        snprintf(key_w, sizeof(key_w), "blocks_%d_experts_%d_c_fc_weight.bin", eb, e);
                        float *w=load_bin(key_w,&n); if(!w)continue;
                        snprintf(key_b, sizeof(key_b), "blocks_%d_experts_%d_c_fc_bias.bin", eb, e);
                        float *bi=load_bin(key_b,&n);
                        snprintf(key_w, sizeof(key_w), "blocks_%d_experts_%d_c_proj_weight.bin", eb, e);
                        float *pw=load_bin(key_w,&n);
                        snprintf(key_b, sizeof(key_b), "blocks_%d_experts_%d_c_proj_bias.bin", eb, e);
                        float *pb=load_bin(key_b,&n);
                        for(int j=0;j<H*FFN_HIDDEN;j++)fc_w[j]+=w[j]; for(int j=0;j<FFN_HIDDEN;j++)fc_b[j]+=bi[j];
                        for(int j=0;j<FFN_HIDDEN*H;j++)pj_w[j]+=pw[j]; for(int j=0;j<H;j++)pj_b[j]+=pb[j];
                        ec++;
                    }
                    if(ec){float iv=1.0f/ec;for(int j=0;j<H*FFN_HIDDEN;j++)fc_w[j]*=iv;for(int j=0;j<FFN_HIDDEN;j++)fc_b[j]*=iv;for(int j=0;j<FFN_HIDDEN*H;j++)pj_w[j]*=iv;for(int j=0;j<H;j++)pj_b[j]*=iv;}
                    
                    for (int i=0;i<n_tokens;i++) {
                        float mv=gs[i*N_EXP];for(int e=1;e<N_EXP;e++)if(gs[i*N_EXP+e]>mv)mv=gs[i*N_EXP+e];
                        float su=0;for(int e=0;e<N_EXP;e++){gs[i*N_EXP+e]=expf(gs[i*N_EXP+e]-mv);su+=gs[i*N_EXP+e];}
                        for(int e=0;e<N_EXP;e++)gs[i*N_EXP+e]/=su;
                        
                        int ti[TOP_K];float tv[TOP_K];
                        for(int k=0;k<TOP_K;k++){ti[k]=0;tv[k]=-1e9f;}
                        for(int e=0;e<N_EXP;e++)for(int k=0;k<TOP_K;k++)if(gs[i*N_EXP+e]>tv[k]){for(int k2=TOP_K-1;k2>k;k2--){ti[k2]=ti[k2-1];tv[k2]=tv[k2-1];}ti[k]=e;tv[k]=gs[i*N_EXP+e];break;}
                        
                        float moe_out[H]={0};
                        for(int k=0;k<TOP_K;k++){float fc[FFN_HIDDEN];for(int j=0;j<FFN_HIDDEN;j++){float s=fc_b[j];for(int d=0;d<H;d++)s+=buf[i*H+d]*fc_w[j*H+d];fc[j]=gelu(s);}float ph[H];for(int j=0;j<H;j++){float s=pj_b[j];for(int d=0;d<FFN_HIDDEN;d++)s+=fc[d]*pj_w[j*FFN_HIDDEN+d];ph[j]=s;}for(int j=0;j<H;j++)moe_out[j]+=tv[k]*ph[j];}
                        for(int j=0;j<H;j++)x[i*H+j]+=moe_out[j];
                    }
                }
                free(ln1_w); free(ln1_b); free(q_w); free(k_w); free(v_w); free(o_w);
                free(ln2_w); free(ln2_b);
            }
        }
        
        // Final LN + lm_head
        for (int i = 0; i < n_tokens; i++) {
            float mean=0,var=0;
            for (int j=0;j<H;j++) mean+=x[i*H+j];
            mean/=H; for (int j=0;j<H;j++){float d=x[i*H+j]-mean;var+=d*d;}
            var=sqrtf(var/H+1e-5f);
            for (int j=0;j<H;j++) x[i*H+j]=(x[i*H+j]-mean)/var*ln_f_w[j]+ln_f_b[j];
        }
        
        float logits[VOCAB];
        for (int v=0;v<VOCAB;v++){float s=0;for(int j=0;j<H;j++)s+=x[(n_tokens-1)*H+j]*lm_head_w[v*H+j];logits[v]=s;}
        
        int next=0;for(int v=1;v<VOCAB;v++)if(logits[v]>logits[next])next=v;
        if (n_tokens < MAX_SEQ) tokens[n_tokens++] = next;
        if (next == 0 || next == 2) break;
    }
    
    float ms = (float)(clock()-t0)/CLOCKS_PER_SEC*1000;
    printf("Generated %d tokens in %.0fms (%.1f tok/s)\n", n_tokens, ms, n_tokens/(ms/1000.0f));
    printf("Tokens:");
    for (int i=0;i<n_tokens;i++) printf(" %d",tokens[i]);
    printf("\n");
    return 0;
}

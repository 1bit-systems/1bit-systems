/**
 * GGUF → Q4NX Converter
 *
 * Converts any GGUF model to FLM's Q4NX format for 94 tok/s inference.
 *
#include "../src/gguf_parser.h"
#include <iostream>
 *   g++ -std=c++23 -O3 -o gguf_to_q4nx gguf_to_q4nx.cpp \
 *       -I.. -I../src -ldl -lm
 *
 * Run:
 *   ./gguf_to_q4nx input.gguf output.q4nx
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <map>

#include "../src/gguf_parser.h"

#define QK_K_BITNET 256

static float bitnet_half_to_float(uint16_t h) {
    uint32_t b = (h & 0x7FFF) << 13 | (h & 0x8000) << 16;
    if ((h & 0x7C00) == 0x7C00) b |= 0x7F800000;
    float f; memcpy(&f, &b, 4); return f;
}

static void dequant_tq1_0(const uint8_t* data, float* out, int n) {
    int nb = n / QK_K_BITNET;
    const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    const int qs_bytes = (QK_K_BITNET - 4 * QK_K_BITNET / 64) / 5;
    const int qh_bytes = QK_K_BITNET / 64;
    for (int i = 0; i < nb; i++) {
        const uint8_t* block = data + i * (qs_bytes + qh_bytes + 2);
        uint16_t dh; memcpy(&dh, block + qs_bytes + qh_bytes, 2);
        float d = bitnet_half_to_float(dh);
        const uint8_t* qs = block;
        const uint8_t* qh = block + qs_bytes;
        for (int j = 0; j < 32; j += 32)
            for (int nn = 0; nn < 5; nn++)
                for (int m = 0; m < 32; m++) {
                    uint8_t q = qs[j + m] * pow3[nn];
                    *out++ = (float)((int)(((uint16_t)q * 3) >> 8) - 1) * d;
                }
        for (int j = 32; j < qs_bytes; j += 16)
            for (int nn = 0; nn < 5; nn++)
                for (int m = 0; m < 16; m++) {
                    uint8_t q = qs[j + m] * pow3[nn];
                    *out++ = (float)((int)(((uint16_t)q * 3) >> 8) - 1) * d;
                }
        for (int nn = 0; nn < 4; nn++)
            for (int j = 0; j < qh_bytes; ++j) {
                uint8_t q = qh[j] * pow3[nn];
                *out++ = (float)((int)(((uint16_t)q * 3) >> 8) - 1) * d;
            }
    }
}

// ─── Tensor name mapping ───
static std::string tn(const char* arch, int l, const char* proj) {
    char buf[256];
    auto a = [&](const char* s) { return strcmp(arch, s) == 0; };
    auto prefix = [&]() { snprintf(buf,sizeof(buf),"blk.%d.",l); return std::string(buf); };
    if (l < 0) {
        if (!strcmp(proj,"token_embd")) return "token_embd.weight";
        if (!strcmp(proj,"output")) return "output.weight";
        if (!strcmp(proj,"norm")) return "output_norm.weight";
    }
    std::string p = prefix();
    if (!strcmp(proj,"q_proj")) return p+"attn_q.weight";
    if (!strcmp(proj,"k_proj")) return p+"attn_k.weight";
    if (!strcmp(proj,"v_proj")) return p+"attn_v.weight";
    if (!strcmp(proj,"o_proj")) return p+"attn_output.weight";
    if (!strcmp(proj,"gate_proj")) return p+"ffn_gate.weight";
    if (!strcmp(proj,"up_proj")) return p+"ffn_up.weight";
    if (!strcmp(proj,"down_proj")) return p+"ffn_down.weight";
    if (!strcmp(proj,"input_norm")) return p+"attn_norm.weight";
    if (!strcmp(proj,"post_norm")) return p+"ffn_norm.weight";
    return "";
}

// ─── Dequant (simplified, supports Q8_0 + F32 + F16) ───
static float* deq(GGUFReader& r, const GGUFModel::Tensor& t, uint64_t data_off) {
    int n=1;for(auto d:t.dims)n*=d;float*o=new float[n];
    r.seek(data_off+t.file_offset);
    int bs=t.type==36?QK_K_BITNET:ggml_blck_size((ggml_type)t.type),ts=t.type==36?12:ggml_type_size((ggml_type)t.type),nb=bs>0?n/bs:0;
    fprintf(stderr,"deq type=%d n=%d bs=%d ts=%d\n",t.type,n,bs,ts);switch(t.type){
        case 0: for(int i=0;i<n;i++)o[i]=r.read_f32();break;
        case 1: for(int i=0;i<n;i++)o[i]=r.read_f16();break;
        case 8: for(int b=0;b<nb;b++){float d=r.read_f16();for(int j=0;j<32;j++)o[b*32+j]=d*(int8_t)r.read_u8();}break;
        case 36: { r.seek(data_off+t.file_offset); size_t tqb=(size_t)n*12/QK_K_BITNET; uint8_t* buf=new uint8_t[tqb]; for(size_t bi=0;bi<tqb;bi++)buf[bi]=r.read_u8(); dequant_tq1_0(buf, o, n); delete[] buf; } break;
        default: fprintf(stderr,"Unsupported type %d for dequant\n",t.type);delete[]o;return nullptr;
    }
    return o;
}

static uint16_t f32bf16(float f){uint32_t u;memcpy(&u,&f,4);return(uint16_t)(u>>16);}

int main(int argc,char**argv){fprintf(stderr,"START\n");
    if(argc<3){fprintf(stderr,"Usage: %s input.gguf output.q4nx\n",argv[0]);return 1;}
    printf("GGUF → Q4NX Converter\n\n");
    
    GGUFReader r;if(!r.open(argv[1]))return 1;
    GGUFModel info;if(!info.parse(r)){r.close();return 1;}
    
    int H=info.hidden_size,NC=info.n_layers,NH=info.n_heads;
    int NKV=info.n_kv_heads?info.n_kv_heads:(NH>0?NH:1);
    int HD=info.head_dim?info.head_dim:(NH>0?H/NH:128);
    int IM=info.intermediate_size,NV=info.vocab_size;
    // Try to get vocab from embedding tensor shape
    if(NV==0){auto te=info.get_tensor("token_embd.weight");if(te&&te->dims.size()>=2)NV=(int)te->dims[1];}
    // For most models, dims are [in, out] = [H, NV]
    if(NV==0){auto te=info.get_tensor("token_embd.weight");if(te&&te->dims.size()>=2)NV=(int)te->dims[0];}
    // Try output.weight
    if(NV==0){auto to=info.get_tensor("output.weight");if(to&&to->dims.size()>=2)NV=(int)to->dims[1];}
    printf("Arch=%s H=%d NC=%d NH=%d NKV=%d IM=%d NV=%d\n",info.arch.c_str(),H,NC,NH,NKV,IM,NV);
    
    // Build output data buffer + JSON header manually
    std::vector<uint8_t> data;
    std::string json = "{";
    
    auto add = [&](const std::string& name, const std::string& dtype,
                   const std::vector<int>& shape, const void* d, size_t n) {
        if (json.size() > 1) json += ",";
        uint64_t off = data.size();
        data.resize(data.size() + n);
        memcpy(data.data() + off, d, n);
        char buf[4096];
        snprintf(buf,sizeof(buf),"\"%s\":{\"dtype\":\"%s\",\"shape\":[%d,%d],\"data_offsets\":[%zu,%zu]}",
                 name.c_str(),dtype.c_str(),
                 shape.size()>=1?shape[0]:0,shape.size()>=2?shape[1]:0,
                 (size_t)off,(size_t)(off+n));
        json += buf;
    };
    
    auto get_i8 = [&](const char* proj, int l, int out_f, int in_f) {
        auto t = info.get_tensor(tn(info.arch.c_str(),l,proj).c_str());
        if(!t)return std::vector<int8_t>();
        float*f=deq(r,*t,info.tensor_data_offset);if(!f)return std::vector<int8_t>();
        float am=0;for(int i=0;i<out_f*in_f;i++){float a=fabsf(f[i]);if(a>am)am=a;}
        if(am<1e-12f)am=1;float s=127.0f/am;
        std::vector<int8_t> i8(out_f*in_f);
        for(int i=0;i<out_f*in_f;i++){float v=f[i];if(!std::isfinite(v))v=0;int q=(int)roundf(v*s);if(q>127)q=127;else if(q<-127)q=-127;i8[i]=(int8_t)q;}
        delete[]f;printf("  L%d %s → I8 [%d,%d]\n",l,proj,out_f,in_f);return i8;
    };
    
    auto get_bf16 = [&](const char* proj, int l) {
        auto t = info.get_tensor(tn(info.arch.c_str(),l,proj).c_str());
        if(!t)return std::vector<uint16_t>();
        float*f=deq(r,*t,info.tensor_data_offset);if(!f)return std::vector<uint16_t>();
        std::vector<uint16_t> b((size_t)H);for(int i=0;i<H;i++)b[i]=f32bf16(f[i]);
        delete[]f;printf("  L%d %s → BF16 [%d]\n",l,proj,H);return b;
    };
    
    printf("\nConverting...\n");
    
    // 1. Embeddings
    auto temb=info.get_tensor("token_embd.weight");
    if(temb){
        float*f=deq(r,*temb,info.tensor_data_offset);
        if(f){std::vector<uint16_t>b((size_t)NV*H);for(size_t i=0;i<(size_t)NV*H;i++)b[i]=f32bf16(f[i]);
            add("model.embed_tokens.weight","BF16",{NV,H},b.data(),b.size()*2);delete[]f;}
    }
    
    // 2. Layer norms
    for(int l=0;l<NC;l++){
        auto b1=get_bf16("input_norm",l);if(!b1.empty()){char buf[256];snprintf(buf,sizeof(buf),"model.layers.%d.input_layernorm.weight",l);add(buf,"BF16",{H},b1.data(),b1.size()*2);}
        auto b2=get_bf16("post_norm",l);if(!b2.empty()){char buf[256];snprintf(buf,sizeof(buf),"model.layers.%d.post_attention_layernorm.weight",l);add(buf,"BF16",{H},b2.data(),b2.size()*2);}
    }
    
    // 3. Output norm
    auto tfn=info.get_tensor(tn(info.arch.c_str(),-1,"norm").c_str());
    if(!tfn)tfn=info.get_tensor("output_norm.weight");
    if(tfn){float*f=deq(r,*tfn,info.tensor_data_offset);if(f){std::vector<uint16_t>b(H);for(int i=0;i<H;i++)b[i]=f32bf16(f[i]);add("model.norm.weight","BF16",{H},b.data(),b.size()*2);delete[]f;}}
    
    // 4. LM head
    auto tlm=info.get_tensor("output.weight");
    if(tlm){auto i8=get_i8("output",-1,NV,H);if(!i8.empty())add("lm_head.weight","I8",{NV,H},i8.data(),i8.size());}
    
    // 5. Per-layer projections
    int QOUT=NH*HD,KVOUT=NKV*HD;
    for(int l=0;l<NC;l++){
        auto iq=get_i8("q_proj",l,QOUT,H);auto ik=get_i8("k_proj",l,KVOUT,H);auto iv=get_i8("v_proj",l,KVOUT,H);
        auto io=get_i8("o_proj",l,H,QOUT);auto ig=get_i8("gate_proj",l,IM,H);auto iu=get_i8("up_proj",l,IM,H);auto id=get_i8("down_proj",l,H,IM);
        char buf[256];
        if(!iq.empty()){snprintf(buf,sizeof(buf),"model.layers.%d.self_attn.q_proj.weight",l);add(buf,"I8",{QOUT,H},iq.data(),iq.size());}
        if(!ik.empty()){snprintf(buf,sizeof(buf),"model.layers.%d.self_attn.k_proj.weight",l);add(buf,"I8",{KVOUT,H},ik.data(),ik.size());}
        if(!iv.empty()){snprintf(buf,sizeof(buf),"model.layers.%d.self_attn.v_proj.weight",l);add(buf,"I8",{KVOUT,H},iv.data(),iv.size());}
        if(!io.empty()){snprintf(buf,sizeof(buf),"model.layers.%d.self_attn.o_proj.weight",l);add(buf,"I8",{H,QOUT},io.data(),io.size());}
        if(!ig.empty()){snprintf(buf,sizeof(buf),"model.layers.%d.mlp.gate_proj.weight",l);add(buf,"I8",{IM,H},ig.data(),ig.size());}
        if(!iu.empty()){snprintf(buf,sizeof(buf),"model.layers.%d.mlp.up_proj.weight",l);add(buf,"I8",{IM,H},iu.data(),iu.size());}
        if(!id.empty()){snprintf(buf,sizeof(buf),"model.layers.%d.mlp.down_proj.weight",l);add(buf,"I8",{H,IM},id.data(),id.size());}
    }
    
    json += "}";
    
    // Write Q4NX
    printf("\nWriting %s...\n",argv[2]);
    FILE*out=fopen(argv[2],"wb");if(!out){fprintf(stderr,"FAIL\n");return 1;}
    uint64_t hs=json.size();fwrite(&hs,8,1,out);
    fwrite(json.data(),1,json.size(),out);
    fwrite(data.data(),1,data.size(),out);
    fclose(out);
    
    printf("\n✅ %s  (JSON: %zu bytes, Data: %.0f KB, Total: %.0f MB)\n",
           argv[2],json.size(),data.size()/1024.0,(8+json.size()+data.size())/1048576.0);
    r.close();
    return 0;
}

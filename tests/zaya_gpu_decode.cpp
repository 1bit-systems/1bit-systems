// zaya_gpu_decode.cpp — Zaya1-8B GPU Decode Engine (Q4NX model loader)
// Build: cmake --build . --target zaya_gpu_decode -j8
// Run:   ./zaya_gpu_decode model.q4nx [--prompt N] [--tokens N]
//
// Loads a Q4NX-format model, copies weights to GPU, runs auto-regressive
// generation with the kernel-decomposed CCA attention + MoE from zaya_server.
//
// Q4NX format: [8-byte header_size][JSON metadata][raw binary weight data]
//   JSON keys: hidden_size, num_hidden_layers, vocab_size, tensor_names, data_offsets
//   Weight dtypes: BF16 (1D norms/biases), I4_CUSTOM (2D matrices with group quantization)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <fstream>
#include <sstream>
#include <chrono>

#define HIP_OK(e) do { auto _s=(e); if(_s!=hipSuccess){fprintf(stderr,"HIP Error %d at %s:%d\n",_s,__FILE__,__LINE__); abort();}} while(0)

// ── Zaya Architecture Constants ──
constexpr int H=2048, NQ=8, NKV=2, HD=128, QD=NQ*HD, KD=NKV*HD, QKV=QD+KD;
constexpr int N_LAYERS=40, VOCAB=262272, N_EXP=16, N_EXP_T=17, N_FF=2048, RTR_H=256;
constexpr float RMD_EPS=1e-5f;
constexpr int BLK=256;

// ── Q4NX Format ──
// Header: [8-byte header_size][JSON metadata of that size][raw binary data]
// JSON keys: hidden_size, num_hidden_layers, vocab_size, num_experts, intermediate_size,
//            tensor_names, data_offsets (map of name -> [start, end] within data section)
// Weight dtypes: BF16 (1D), I4_CUSTOM (2D group-quantized), I8 (2D int8)

#pragma pack(push, 1)
struct Q4nxTensor {
    std::string dtype;       // BF16, I4_CUSTOM, I8
    std::vector<int64_t> shape;
    int64_t offset = 0;      // offset within file data section
    int64_t size = 0;        // bytes in file
    bool is_quantized = false;
};
#pragma pack(pop)

struct Q4nxModel {
    std::map<std::string, Q4nxTensor> tensors;
    int64_t hidden_size = H;
    int64_t num_hidden_layers = N_LAYERS;
    int64_t vocab_size = VOCAB;
    int64_t num_experts = N_EXP;
    int64_t intermediate_size = N_FF;
    
    // Raw file data (kept alive for tensor access)
    std::vector<uint8_t> file_data;
    int64_t data_section_offset = 0;
    
    bool load(const char* path) {
        // Read file
        FILE* f = fopen(path, "rb");
        if (!f) { fprintf(stderr, "Cannot open %s\n", path); return false; }
        fseek(f, 0, SEEK_END);
        int64_t file_size = ftell(f);
        fseek(f, 0, SEEK_SET);
        file_data.resize(file_size);
        if (fread(file_data.data(), 1, file_size, f) != (size_t)file_size) {
            fprintf(stderr, "Failed to read %s\n", path); fclose(f); return false;
        }
        fclose(f);
        
        // Parse header: first 8 bytes = header_size
        uint64_t header_size = 0;
        memcpy(&header_size, file_data.data(), 8);
        if (header_size > file_size - 8) {
            fprintf(stderr, "Header too large: %zu\n", (size_t)header_size);
            return false;
        }
        
        // Parse JSON manifest
        std::string manifest((const char*)file_data.data() + 8, (size_t)header_size);
        
        // Very simple JSON parser - extract key fields
        auto json_str = [&](const std::string& key) -> std::string {
            auto pos = manifest.find("\"" + key + "\"");
            if (pos == std::string::npos) return "";
            pos = manifest.find(':', pos);
            if (pos == std::string::npos) return "";
            pos = manifest.find_first_of("\"", pos);
            if (pos == std::string::npos || manifest[pos] != '"') return "";
            auto end = manifest.find('"', pos + 1);
            if (end == std::string::npos) return "";
            return manifest.substr(pos + 1, end - pos - 1);
        };
        auto json_int = [&](const std::string& key, int64_t def=0) -> int64_t {
            auto pos = manifest.find("\"" + key + "\"");
            if (pos == std::string::npos) return def;
            pos = manifest.find(':', pos);
            if (pos == std::string::npos) return def;
            pos = manifest.find_first_of("0123456789-", pos);
            if (pos == std::string::npos) return def;
            char* end = nullptr;
            return strtoll(&manifest[pos], &end, 10);
        };
        
        hidden_size = json_int("hidden_size", H);
        num_hidden_layers = json_int("num_hidden_layers", N_LAYERS);
        vocab_size = json_int("vocab_size", VOCAB);
        num_experts = json_int("num_experts", N_EXP);
        intermediate_size = json_int("intermediate_size", N_FF);
        
        // Parse tensor_names array
        std::vector<std::string> tensor_names;
        auto tn_pos = manifest.find("\"tensor_names\"");
        if (tn_pos != std::string::npos) {
            auto arr_start = manifest.find('[', tn_pos);
            auto arr_end = manifest.find(']', arr_start);
            if (arr_start != std::string::npos && arr_end != std::string::npos) {
                std::string arr = manifest.substr(arr_start + 1, arr_end - arr_start - 1);
                size_t p = 0;
                while (p < arr.size()) {
                    // Skip whitespace and commas
                    while (p < arr.size() && (arr[p] == ' ' || arr[p] == ',' || arr[p] == '\n' || arr[p] == '\r')) p++;
                    if (p >= arr.size()) break;
                    if (arr[p] == '"') {
                        auto eq = arr.find('"', p + 1);
                        if (eq != std::string::npos) {
                            tensor_names.push_back(arr.substr(p + 1, eq - p - 1));
                            p = eq + 1;
                        }
                    }
                }
            }
        }
        
        // Parse data_offsets for each tensor
        data_section_offset = 8 + header_size;
        
        auto find_offset = [&](const std::string& name) -> std::pair<int64_t, int64_t> {
            // Find "name": { ... "data_offsets": [start, end] }
            auto pos = manifest.find("\"" + name + "\"");
            if (pos == std::string::npos) return {-1, -1};
            auto do_pos = manifest.find("\"data_offsets\"", pos);
            if (do_pos == std::string::npos) return {-1, -1};
            auto bracket = manifest.find('[', do_pos);
            if (bracket == std::string::npos) return {-1, -1};
            char* end = nullptr;
            int64_t start = strtoll(&manifest[bracket + 1], &end, 10);
            if (!end) return {-1, -1};
            int64_t end_val = strtoll(end + 1, &end, 10);
            return {start, end_val};
        };
        
        auto find_dtype = [&](const std::string& name) -> std::string {
            auto pos = manifest.find("\"" + name + "\"");
            if (pos == std::string::npos) return "";
            auto dt_pos = manifest.find("\"dtype\"", pos);
            if (dt_pos == std::string::npos) return "";
            auto colon = manifest.find(':', dt_pos);
            if (colon == std::string::npos) return "";
            auto q = manifest.find_first_of("\"", colon + 1);
            if (q == std::string::npos || manifest[q] != '"') return "";
            auto eq = manifest.find('"', q + 1);
            if (eq == std::string::npos) return "";
            return manifest.substr(q + 1, eq - q - 1);
        };
        
        auto find_shape = [&](const std::string& name) -> std::vector<int64_t> {
            auto pos = manifest.find("\"" + name + "\"");
            if (pos == std::string::npos) return {};
            auto s_pos = manifest.find("\"shape\"", pos);
            if (s_pos == std::string::npos) return {};
            auto bracket = manifest.find('[', s_pos);
            if (bracket == std::string::npos) return {};
            auto end_bracket = manifest.find(']', bracket);
            if (end_bracket == std::string::npos) return {};
            std::string arr = manifest.substr(bracket + 1, end_bracket - bracket - 1);
            std::vector<int64_t> shape;
            char* end = nullptr;
            int64_t v = strtoll(arr.c_str(), &end, 10);
            shape.push_back(v);
            if (end && *end != '\0') {
                v = strtoll(end + 1, &end, 10);
                shape.push_back(v);
            }
            return shape;
        };
        
        for (const auto& name : tensor_names) {
            Q4nxTensor t;
            t.dtype = find_dtype(name);
            t.shape = find_shape(name);
            auto off = find_offset(name);
            t.offset = off.first;
            t.size = off.second - off.first;
            t.is_quantized = (t.dtype == "I4_CUSTOM" || t.dtype == "I8");
            tensors[name] = t;
        }
        
        printf("[Q4NX] %zu tensors, V=%ld H=%ld L=%ld\n",
               tensors.size(), vocab_size, hidden_size, num_hidden_layers);
        return true;
    }
    
    // Get tensor data pointer in file
    const uint8_t* tensor_data(const std::string& name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) return nullptr;
        return file_data.data() + data_section_offset + it->second.offset;
    }
    
    // Get tensor size in bytes
    int64_t tensor_size(const std::string& name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) return 0;
        return it->second.size;
    }
    
    // Dequantize an I4_CUSTOM quantized tensor to FP16 on CPU
    // Groups of 8: each group = [2B scale_bf16][2B zp_bf16][4B I4 data (8 values)]
    std::vector<__half> dequant_i8(const std::string& name, int rows, int cols) {
        const auto* raw = tensor_data(name);
        if (!raw) {
            fprintf(stderr, "Missing tensor: %s\n", name.c_str());
            return {};
        }
        
        int groups = (cols + 7) / 8;  // groups per row
        std::vector<__half> result((size_t)rows * cols);
        
        for (int r = 0; r < rows; r++) {
            for (int g = 0; g < groups; g++) {
                int group_idx = r * groups + g;
                const uint8_t* group_data = raw + group_idx * 8;
                
                // Read scale (BF16)
                uint16_t scale_bits;
                memcpy(&scale_bits, group_data, 2);
                uint32_t scale_f32 = ((uint32_t)scale_bits) << 16;
                float scale;
                memcpy(&scale, &scale_f32, 4);
                
                // Read zero-point (BF16)
                uint16_t zp_bits;
                memcpy(&zp_bits, group_data + 2, 2);
                uint32_t zp_f32 = ((uint32_t)zp_bits) << 16;
                float zp;
                memcpy(&zp, &zp_f32, 4);
                
                // Read 4 bytes of I4 data -> 8 values
                uint8_t packed[4];
                memcpy(packed, group_data + 4, 4);
                
                for (int i = 0; i < 8; i++) {
                    int col = g * 8 + i;
                    if (col >= cols) break;
                    uint8_t nibble = (packed[i / 2] >> (i % 2 ? 4 : 0)) & 0x0F;
                    float val = ((int)nibble - 8 + zp) * scale;  // center around 0
                    result[(size_t)r * cols + col] = __float2half(val);
                }
            }
        }
        return result;
    }
    
    // Read a BF16 tensor directly
    std::vector<__half> read_bf16(const std::string& name, int count) {
        const auto* raw = tensor_data(name);
        if (!raw) {
            fprintf(stderr, "Missing tensor: %s\n", name.c_str());
            return {};
        }
        std::vector<__half> result(count);
        memcpy(result.data(), raw, count * 2);
        return result;
    }
    
    // Read a float tensor (stored as BF16)
    std::vector<float> read_f32(const std::string& name, int count) {
        const auto* raw = tensor_data(name);
        if (!raw) {
            fprintf(stderr, "Missing tensor: %s\n", name.c_str());
            return {};
        }
        std::vector<float> result(count);
        const uint16_t* src = (const uint16_t*)raw;
        for (int i = 0; i < count; i++) {
            uint32_t bits = ((uint32_t)src[i]) << 16;
            memcpy(&result[i], &bits, 4);
        }
        return result;
    }
};

// ── Layer GPU weights ──
struct LayerGPU {
    // Weight pointers (GPU)
    __half *nw;     // input_layernorm.weight
    __half *wq;     // attn_q.weight
    __half *wk;     // attn_k.weight
    __half *wv1;    // cca_val_proj1.weight (current)
    __half *wv2;    // cca_val_proj2.weight (delayed)
    __half *wo;     // attn_output.weight
    __half *pan;    // attn_norm.weight
    float *cdw;     // cca_conv_grp.weight (depthwise)
    float *cdb;     // cca_conv_grp.bias
    float *cgw;     // cca_conv_grp.weight (grouped)
    float *cgb;     // cca_conv_grp.bias
    float *ks;      // cca_k_scale.weight
    float *pahss, *pahsb;   // post-attention residual scale hs
    float *parss, *parsb;   // post-attention residual scale res
    float *gdw, *gdb;       // ffn_gate.weight/bias (router gate down)
    float *rfn;             // ffn_norm.weight (router norm)
    float *rf1, *rf1b;      // zaya_router_mlp2.weight/bias (router fc1)
    float *rf2, *rf2b;      // zaya_router_mlp4.weight/bias (router fc2)
    float *rout;            // zaya_router_biases.weight (router out)
    float *bb;              // router balancing biases
    __half *gu;             // ffn_gate_up_exps.weight (MoE gate_up)
    __half *dn;             // ffn_down_exps.weight (MoE down)
    float *pmhss, *pmhsb;   // post-MLP residual scale hs
    float *pmrss, *pmrsb;   // post-MLP residual scale res
};

// ── Forward declarations for kernels ──
#include "../kernels/zaya_moe_tiled_gemv.hip"
#include "../kernels/zaya_cca_custom.hip"
#include "../kernels/v_interleave_kernel.hip"
#include "../kernels/zaya_gpu_router.hip"
#include "../kernels/zaya_router_moe.hip"
#include "../kernels/zaya_moe_expert_ffn.hip"
#include "../kernels/argmax_kernel.hip"

// ── Utility kernels (inlined from zaya_server) ──
__global__ void rmsnorm_k(__half*x,const __half*w,int n){__shared__ float r[32];int tx=threadIdx.x,wid=tx/32,l=tx%32;float ss=0;for(int i=tx;i<n;i+=blockDim.x)ss+=(float)x[i]*(float)x[i];for(int o=16;o>0;o>>=1)ss+=__shfl_xor(ss,o);if(l==0)r[wid]=ss;__syncthreads();if(wid==0){ss=(l<(256/32))?r[l]:0;for(int o=16;o>0;o>>=1)ss+=__shfl_xor(ss,o);if(l==0)r[0]=ss;}__syncthreads();float iv=1.0f/sqrtf(r[0]/n+RMD_EPS);for(int i=tx;i<n;i+=blockDim.x)x[i]=__float2half((float)x[i]*iv*(float)w[i]);}
__global__ void copy_k(__half*d,const __half*s,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;d[i]=s[i];}
__global__ void mm_k(__half*out,const __half*in,const __half*wt,int M,int K){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=M)return;float s=0;for(int k=0;k<K;k++)s+=(float)in[k]*(float)wt[k*(size_t)M+i];out[i]=__float2half(s);}
__global__ void silu_mul_k(__half*out,const __half*g,const __half*u,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float v=(float)g[i];out[i]=__float2half((v/(1.0f+expf(-v)))*(float)u[i]);}
__global__ void residual_scale_k(__half*out,const __half*res,const float*hs_s,const float*hs_b,const float*res_s,const float*res_b,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float o=(float)out[i]*hs_s[i]+hs_b[i];float r=(float)res[i]*res_s[i]+res_b[i];out[i]=__float2half(o+r);}

// ── Host helper: upload float vector as fp16 to GPU ──
static void upload_f16(const std::vector<float>& src, __half* dst, int n, hipStream_t stream = 0) {
    std::vector<__half> buf(n);
    for (int i = 0; i < n; i++) buf[i] = __float2half(src[i]);
    HIP_OK(hipMemcpyAsync(dst, buf.data(), (size_t)n * 2, hipMemcpyHostToDevice, stream));
}

// ── Host helper: upload fp16 vector to GPU ──
static void upload_h16(const std::vector<__half>& src, __half* dst, int n, hipStream_t stream = 0) {
    HIP_OK(hipMemcpyAsync(dst, src.data(), (size_t)n * 2, hipMemcpyHostToDevice, stream));
}

// ── Tokenizer (simple byte-level) ──
struct Tokenizer {
    std::vector<std::string> id_to_token;
    
    void load(const std::string& vocab_path) {
        // Placeholder — for full tokenization, use external BPE
        id_to_token.resize(32000);
    }
    
    std::vector<int> encode(const std::string& text) {
        std::vector<int> r = {2}; // BOS
        for (char c : text) {
            if (c >= ' ' && c <= '~') r.push_back((unsigned char)c + 100);
        }
        return r;
    }
    
    std::string decode(const std::vector<int>& tokens) {
        std::string r;
        for (int v : tokens) {
            if (v == 2 || v == 106) continue; // BOS/EOS
            if (v > 100 && v < 200) r += (char)(v - 100);
            else { r += '['; r += std::to_string(v); r += ']'; }
        }
        return r;
    }
};

// ── Main ──
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s model.q4nx [--prompt N] [--tokens N]\n", argv[0]);
        return 1;
    }
    
    const char* model_path = argv[1];
    int prompt_tokens = 2;   // BOS
    int gen_tokens = 64;
    
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc)
            prompt_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc)
            gen_tokens = atoi(argv[++i]);
    }
    
    printf("Zaya1-8B GPU Decode Engine\n");
    printf("Loading: %s\n", model_path);
    
    // ── Load Q4NX model ──
    Q4nxModel model;
    auto t0 = std::chrono::high_resolution_clock::now();
    if (!model.load(model_path)) return 1;
    
    // ── Allocate GPU memory ──
    printf(" Allocating GPU memory\n");
    __half *d_hs, *d_ao, *d_tmp, *d_fnw, *d_lm_out, *d_embed, *d_all_logits = nullptr;
    float *d_fbuf;
    HIP_OK(hipMalloc(&d_hs, (size_t)H * 2));
    HIP_OK(hipMalloc(&d_ao, (size_t)H * 2));
    HIP_OK(hipMalloc(&d_tmp, (size_t)H * 2));
    HIP_OK(hipMalloc(&d_fnw, (size_t)H * 2));
    HIP_OK(hipMalloc(&d_lm_out, (size_t)4096 * 2));
    HIP_OK(hipMalloc(&d_embed, (size_t)VOCAB * H * 2));
    HIP_OK(hipMalloc(&d_fbuf, (size_t)std::max(H * 2, RTR_H * 8) * 4));
    
    // Conv state + per-layer hidden states + router state
    __half *d_conv, *d_phs;
    float *d_prev_rs;
    int *d_expert_idx;
    float *d_expert_wt;
    HIP_OK(hipMalloc(&d_conv, (size_t)N_LAYERS * 2 * QKV * 2));
    HIP_OK(hipMalloc(&d_phs, (size_t)N_LAYERS * H * 2));
    HIP_OK(hipMalloc(&d_prev_rs, (size_t)N_LAYERS * RTR_H * 4));
    HIP_OK(hipMalloc(&d_expert_idx, 4));
    HIP_OK(hipMalloc(&d_expert_wt, 4));
    
    hipStream_t stream;
    HIP_OK(hipStreamCreate(&stream));
    
    // ── Load embedding ──
    printf(" Loading weights...\n");
    
    // Embedding: model.embed_tokens.weight [VOCAB, H] stored as I4_CUSTOM
    {
        auto it = model.tensors.find("token_embd.weight");
        if (it == model.tensors.end())
            it = model.tensors.find("model.embed_tokens.weight");
        if (it != model.tensors.end()) {
            auto deq = model.dequant_i8(it->first, VOCAB, H);
            printf("  embed: %s %dx%d -> %zu fp16\n", it->first.c_str(), VOCAB, H, deq.size());
            HIP_OK(hipMemcpyAsync(d_embed, deq.data(), (size_t)VOCAB * H * 2, hipMemcpyHostToDevice, stream));
        } else {
            // Try BF16 embed
            auto bf16 = model.read_bf16("token_embd.weight", VOCAB * H);
            if (bf16.empty()) bf16 = model.read_bf16("model.embed_tokens.weight", VOCAB * H);
            if (!bf16.empty()) {
                printf("  embed: BF16 %dx%d\n", VOCAB, H);
                HIP_OK(hipMemcpyAsync(d_embed, bf16.data(), (size_t)VOCAB * H * 2, hipMemcpyHostToDevice, stream));
            } else {
                fprintf(stderr, "No embed_tokens\n");
                return 1;
            }
        }
    }
    
    // Final norm: model.norm.weight
    {
        auto fn = model.read_bf16("output_norm.weight", H);
        if (fn.empty()) fn = model.read_bf16("model.norm.weight", H);
        if (fn.empty()) {
            fprintf(stderr, "Missing output_norm.weight\n");
            return 1;
        }
        HIP_OK(hipMemcpyAsync(d_fnw, fn.data(), (size_t)H * 2, hipMemcpyHostToDevice, stream));
    }
    
    // Input hidden states scale/bias (if present)
    float input_scale = 1.0f, input_bias = 0.0f;
    {
        auto s = model.read_f32("input_hidden_states_scale.weight", 1);
        auto b = model.read_f32("input_hidden_states_scale.bias", 1);
        if (!s.empty()) input_scale = s[0];
        if (!b.empty()) input_bias = b[0];
    }
    
    // ── Load per-layer weights ──
    std::vector<LayerGPU> layers(N_LAYERS);
    int total_layers = (int)model.num_hidden_layers;
    printf(" Loading %d layers\n", total_layers);
    
    for (int il = 0; il < total_layers; il++) {
        auto& l = layers[il];
        std::string prefix = "model.layers." + std::to_string(il) + ".";
        
        auto load_half = [&](const std::string& key, __half*& gpu_ptr, int count) {
            // Try both naming conventions
            auto data = model.read_bf16(prefix + key, count);
            if (data.empty()) data = model.read_bf16(key, count);
            if (data.empty()) {
                fprintf(stderr, "  WARN: no %s (%d bf16)\n", key.c_str(), count);
                gpu_ptr = nullptr;
                return;
            }
            HIP_OK(hipMalloc(&gpu_ptr, (size_t)count * 2));
            HIP_OK(hipMemcpyAsync(gpu_ptr, data.data(), (size_t)count * 2, hipMemcpyHostToDevice, stream));
        };
        
        auto load_float = [&](const std::string& key, float*& gpu_ptr, int count) {
            auto data = model.read_f32(prefix + key, count);
            if (data.empty()) data = model.read_f32(key, count);
            if (data.empty()) {
                gpu_ptr = nullptr;
                return;
            }
            HIP_OK(hipMalloc(&gpu_ptr, (size_t)count * 4));
            HIP_OK(hipMemcpyAsync(gpu_ptr, data.data(), (size_t)count * 4, hipMemcpyHostToDevice, stream));
        };
        
        auto load_quant = [&](const std::string& key, __half*& gpu_ptr, int rows, int cols) {
            auto deq = model.dequant_i8(prefix + key, rows, cols);
            if (deq.empty()) deq = model.dequant_i8(key, rows, cols);
            if (deq.empty()) {
                gpu_ptr = nullptr;
                return;
            }
            HIP_OK(hipMalloc(&gpu_ptr, (size_t)rows * cols * 2));
            HIP_OK(hipMemcpyAsync(gpu_ptr, deq.data(), (size_t)rows * cols * 2, hipMemcpyHostToDevice, stream));
        };
        
        // Attention weights
        load_quant("attn_q.weight", l.wq, QD, H);
        load_quant("attn_k.weight", l.wk, KD, H);
        load_quant("cca_val_proj1.weight", l.wv1, KD/2, H);
        load_quant("cca_val_proj2.weight", l.wv2, KD/2, H);
        load_quant("attn_output.weight", l.wo, H, QD);
        
        // Layer norms
        load_half("input_layernorm.weight", l.nw, H);
        load_half("attn_norm.weight", l.pan, H);
        load_half("attn_norm_2.weight", l.pan, H);  // fallback
        
        // CCA conv weights (float)
        load_float("cca_conv_grp.weight", l.cdw, QKV * 2);
        load_float("cca_conv_grp.bias", l.cdb, QKV);
        // grouped weights are [QKV, 128, 2]
        load_float("cca_conv_grp.weight", l.cgw, QKV * 128 * 2);
        load_float("cca_conv_grp.bias", l.cgb, QKV);
        load_float("cca_k_scale.weight", l.ks, NKV);
        
        // Post-attention residual scale
        load_float("res_scale_hs.weight", l.pahss, H);
        load_float("res_scale_hs.bias", l.pahsb, H);
        load_float("res_scale_res.weight", l.parss, H);
        load_float("res_scale_res.bias", l.parsb, H);
        
        // Router
        load_float("ffn_gate.weight", l.gdw, H * RTR_H);
        load_float("ffn_gate.bias", l.gdb, RTR_H);
        {
            auto router_norm = model.read_f32("ffn_gate_inp.weight", RTR_H);
            if (router_norm.empty()) router_norm = model.read_f32("ffn_norm.weight", RTR_H);
            if (!router_norm.empty()) {
                HIP_OK(hipMalloc(&l.rfn, (size_t)RTR_H * 4));
                HIP_OK(hipMemcpyAsync(l.rfn, router_norm.data(), (size_t)RTR_H * 4, hipMemcpyHostToDevice, stream));
            } else {
                l.rfn = nullptr;
            }
        }
        load_float("zaya_router_mlp2.weight", l.rf1, RTR_H * RTR_H);
        load_float("zaya_router_mlp2.bias", l.rf1b, RTR_H);
        load_float("zaya_router_mlp4.weight", l.rf2, RTR_H * RTR_H);
        load_float("zaya_router_mlp4.bias", l.rf2b, RTR_H);
        load_float("zaya_router_biases.weight", l.rout, N_EXP_T * RTR_H);
        load_float("zaya_router_biases.weight", l.bb, N_EXP_T);
        
        // MoE experts
        load_quant("ffn_gate_up_exps.weight", l.gu, N_EXP * 2 * N_FF, H);
        load_quant("ffn_down_exps.weight", l.dn, N_EXP * H, N_FF);
        
        // Post-MLP residual scale
        load_float("res_scale_hs.mlp.weight", l.pmhss, H);
        load_float("res_scale_hs.mlp.bias", l.pmhsb, H);
        load_float("res_scale_res.mlp.weight", l.pmrss, H);
        load_float("res_scale_res.mlp.bias", l.pmrsb, H);
        
        printf("  Layer %d/%d loaded\n", il + 1, total_layers);
    }
    
    HIP_OK(hipStreamSynchronize(stream));
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("  Load time: %.1f ms\n", std::chrono::duration<float, std::milli>(t1 - t0).count());
    
    // ── Router state (per layer: has_eda flag for EDynamic Attention) ──
    struct RouterState { float eda_scale[1]; bool has_eda; };
    std::vector<RouterState> router_states(N_LAYERS);
    for (int il = 0; il < N_LAYERS; il++) {
        router_states[il].has_eda = false;
        router_states[il].eda_scale[0] = 0.0f;
    }
    
    // ── Generate tokens ──
    printf(" Generating %d tokens (prompt=%d)\n", gen_tokens, prompt_tokens);
    
    // Zero out conv state and per-layer hidden states
    HIP_OK(hipMemsetAsync(d_conv, 0, (size_t)N_LAYERS * 2 * QKV * 2, stream));
    HIP_OK(hipMemsetAsync(d_phs, 0, (size_t)N_LAYERS * H * 2, stream));
    HIP_OK(hipMemsetAsync(d_prev_rs, 0, (size_t)N_LAYERS * RTR_H * 4, stream));
    
    std::vector<int> output;
    output.push_back(2); // BOS
    
    // Copy prompt token embedding
    {
        int token_id = 2; // BOS
        std::vector<__half> hh(H);
        // Grab embedding row
        HIP_OK(hipMemcpy(hh.data(), d_embed + (size_t)token_id * H, (size_t)H * 2, hipMemcpyDeviceToHost));
        for (int i = 0; i < H; i++) {
            float v = __half2float(hh[i]);
            hh[i] = __float2half(v * input_scale + input_bias);
        }
        HIP_OK(hipMemcpyAsync(d_hs, hh.data(), (size_t)H * 2, hipMemcpyHostToDevice, stream));
    }
    
    int last_token = 2;
    auto gen_start = std::chrono::high_resolution_clock::now();
    int g1 = (H + BLK - 1) / BLK;
    
    for (int pos = 0; pos < gen_tokens; pos++) {
        // ── Forward through all layers ──
        for (int il = 0; il < total_layers; il++) {
            auto& l = layers[il];
            
            // ── A) CCA: tiled QKV + custom conv+attn + O_proj ──
            copy_k<<<g1, BLK, 0, stream>>>(d_phs + (size_t)il * H, d_hs, H);
            rmsnorm_k<<<1, BLK, 0, stream>>>(d_hs, l.nw, H);
            moe_tiled_gemv<<<QD/16, 128, 0, stream>>>(d_tmp, d_hs, l.wq, QD, H);
            moe_tiled_gemv<<<KD/16, 128, 0, stream>>>(d_tmp+QD, d_hs, l.wk, KD, H);
            moe_tiled_gemv<<<KD/2/16, 128, 0, stream>>>(d_tmp+QD+KD, d_hs, l.wv1, KD/2, H);
            moe_tiled_gemv<<<KD/2/16, 128, 0, stream>>>(d_tmp+QD+KD+KD/2, d_phs+(size_t)il*H, l.wv2, KD/2, H);
            v_interleave_kernel<<<(KD/2+BLK-1)/BLK, BLK, 0, stream>>>(d_tmp+QD, d_tmp+QD+KD, d_tmp+QD+KD+KD/2, KD/2);
            cca_custom_kernel<<<1, 256, 0, stream>>>(d_tmp, d_tmp+QD, d_tmp+QD, d_phs+(size_t)il*H, d_conv+(size_t)il*2*QKV, l.cdw, l.cdb, l.cgw, l.cgb, l.ks, d_ao, d_conv+(size_t)il*2*QKV, d_phs+(size_t)il*H, il, 1);
            moe_tiled_gemv<<<H/16, 128, 0, stream>>>(d_ao, d_ao, l.wo, H, QD);
            residual_scale_k<<<g1, BLK, 0, stream>>>(d_ao, d_hs, l.pahss, l.pahsb, l.parss, l.parsb, H);
            copy_k<<<g1, BLK, 0, stream>>>(d_hs, d_ao, H);
            rmsnorm_k<<<1, BLK, 0, stream>>>(d_hs, l.pan, H);
            
            // ── B) EDA Router + MoE Expert (GPU) ──
            if (l.gu && l.dn) {
                eda_router_gpu_kernel<<<1, RTR_H, 0, stream>>>(
                    d_hs, d_prev_rs + (size_t)il * RTR_H,
                    router_states[il].has_eda ? 1 : 0, router_states[il].eda_scale[0],
                    l.gdw, l.gdb, l.rfn, l.rf1, l.rf1b, l.rf2, l.rf2b, l.rout, l.bb,
                    d_prev_rs + (size_t)il * RTR_H, d_expert_idx, d_expert_wt);
                encode_expert_cache_kernel<<<1, 32, 0, stream>>>(d_prev_rs + (size_t)il * RTR_H, d_expert_idx, RTR_H);
                {
                    const int gb = (2 * N_FF + 15) / 16, db = (H + 15) / 16, sb = (N_FF + BLK - 1) / BLK;
                    wmma_gateup_kernel<<<gb, 128, 0, stream>>>(d_tmp, d_hs, l.gu, d_expert_idx);
                    silu_mul_k<<<sb, BLK, 0, stream>>>(d_ao, d_tmp, d_tmp + N_FF, N_FF);
                    wmma_down_kernel<<<db, 128, 0, stream>>>(d_tmp, d_ao, l.dn, d_expert_idx);
                }
                // Post-MLP residual scale
                residual_scale_k<<<g1, BLK, 0, stream>>>(d_tmp, d_hs, l.pmhss, l.pmhsb, l.pmrss, l.pmrsb, H);
                copy_k<<<g1, BLK, 0, stream>>>(d_hs, d_tmp, H);
            }
        }
        
        // ── Final RMSNorm + lm_head ──
        rmsnorm_k<<<1, BLK, 0, stream>>>(d_hs, d_fnw, H);
        
        // GPU lm_head: logits[v] = hs @ embed[v]^T for all vocab
        if (!d_all_logits) HIP_OK(hipMalloc(&d_all_logits, (size_t)VOCAB * 2));
        for (int v = 0; v < VOCAB; v += 4096) {
            int todo = std::min(4096, VOCAB - v);
            mm_k<<<todo, BLK, 0, stream>>>(d_all_logits + (size_t)v, d_hs, d_embed + (size_t)v * H, todo, H);
        }
        
        HIP_OK(hipStreamSynchronize(stream));
        
        // ── Read logits and sample ──
        std::vector<__half> logits_half(VOCAB);
        HIP_OK(hipMemcpy(logits_half.data(), d_all_logits, (size_t)VOCAB * 2, hipMemcpyDeviceToHost));
        
        int best = 0;
        float best_val = -1e30f;
        for (int v = 0; v < VOCAB; v++) {
            float val = __half2float(logits_half[v]);
            if (val > best_val) { best_val = val; best = v; }
        }
        
        last_token = best;
        output.push_back(last_token);
        
        // ── Copy next token embedding for next iteration ──
        if (pos + 1 < gen_tokens && last_token != 106) { // EOS check
            std::vector<__half> hh(H);
            HIP_OK(hipMemcpy(hh.data(), d_embed + (size_t)last_token * H, (size_t)H * 2, hipMemcpyDeviceToHost));
            for (int i = 0; i < H; i++) {
                float v = __half2float(hh[i]);
                hh[i] = __float2half(v * input_scale + input_bias);
            }
            HIP_OK(hipMemcpyAsync(d_hs, hh.data(), (size_t)H * 2, hipMemcpyHostToDevice, stream));
        }
        
        if (last_token == 106) break; // EOS
    }
    
    auto gen_end = std::chrono::high_resolution_clock::now();
    float gen_ms = std::chrono::duration<float, std::milli>(gen_end - gen_start).count();
    int gen_count = (int)output.size() - 1; // exclude BOS
    
    printf("  Generated: %d tokens\n", gen_count);
    printf("  Total time: %.1f ms (%.1f tok/s)\n", gen_ms, (float)gen_count / (gen_ms / 1000.0f));
    printf("  Tokens: ");
    for (size_t i = 0; i < output.size(); i++) {
        if (i > 0) printf(" ");
        printf("%d", output[i]);
    }
    printf("\n");
    
    // ── Cleanup ──
    for (auto& l : layers) {
        if (l.nw) HIP_OK(hipFree(l.nw));
        if (l.wq) HIP_OK(hipFree(l.wq));
        if (l.wk) HIP_OK(hipFree(l.wk));
        if (l.wv1) HIP_OK(hipFree(l.wv1));
        if (l.wv2) HIP_OK(hipFree(l.wv2));
        if (l.wo) HIP_OK(hipFree(l.wo));
        if (l.pan) HIP_OK(hipFree(l.pan));
        if (l.cdw) HIP_OK(hipFree(l.cdw));
        if (l.cdb) HIP_OK(hipFree(l.cdb));
        if (l.cgw) HIP_OK(hipFree(l.cgw));
        if (l.cgb) HIP_OK(hipFree(l.cgb));
        if (l.ks) HIP_OK(hipFree(l.ks));
        if (l.pahss) HIP_OK(hipFree(l.pahss));
        if (l.pahsb) HIP_OK(hipFree(l.pahsb));
        if (l.parss) HIP_OK(hipFree(l.parss));
        if (l.parsb) HIP_OK(hipFree(l.parsb));
        if (l.gdw) HIP_OK(hipFree(l.gdw));
        if (l.gdb) HIP_OK(hipFree(l.gdb));
        if (l.rfn) HIP_OK(hipFree(l.rfn));
        if (l.rf1) HIP_OK(hipFree(l.rf1));
        if (l.rf1b) HIP_OK(hipFree(l.rf1b));
        if (l.rf2) HIP_OK(hipFree(l.rf2));
        if (l.rf2b) HIP_OK(hipFree(l.rf2b));
        if (l.rout) HIP_OK(hipFree(l.rout));
        if (l.bb) HIP_OK(hipFree(l.bb));
        if (l.gu) HIP_OK(hipFree(l.gu));
        if (l.dn) HIP_OK(hipFree(l.dn));
        if (l.pmhss) HIP_OK(hipFree(l.pmhss));
        if (l.pmhsb) HIP_OK(hipFree(l.pmhsb));
        if (l.pmrss) HIP_OK(hipFree(l.pmrss));
        if (l.pmrsb) HIP_OK(hipFree(l.pmrsb));
    }
    HIP_OK(hipFree(d_hs));
    HIP_OK(hipFree(d_ao));
    HIP_OK(hipFree(d_tmp));
    HIP_OK(hipFree(d_fnw));
    HIP_OK(hipFree(d_lm_out));
    HIP_OK(hipFree(d_embed));
    HIP_OK(hipFree(d_fbuf));
    HIP_OK(hipFree(d_conv));
    HIP_OK(hipFree(d_phs));
    HIP_OK(hipFree(d_prev_rs));
    HIP_OK(hipFree(d_expert_idx));
    HIP_OK(hipFree(d_expert_wt));
    if (d_all_logits) HIP_OK(hipFree(d_all_logits));
    
    return 0;
}

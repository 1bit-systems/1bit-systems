/** Q4NX Weight Stream Packer for Fused XCLBIN.
 *  Builds the exact weights+AUX BO that the full-layer xclbin expects.
 *
 *  Layout: [AUX prefix (1216 dwords = 4864 bytes)] [weight stream (2,457,600 dwords)]
 *
 *  AUX: input_norm(H bf16) + post_norm(H bf16) + q_norm(HD bf16) + k_norm(HD bf16) + cos(HD/2 bf16) + sin(HD/2 bf16)
 *
 *  Weight stream: 4 columns × 2 patches per column
 *    Each patch: Q/K/V chunks, O chunks, UPGATE span0, UPGATE span1, DOWN
 *    Each chunk: 5120 bytes = 2560 bf16 (512 scale + 512 zero + 4096 u4 data)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

// Qwen3-0.6B constants
constexpr int H=1024, HD=128, NH=16, NKV=8, IM=3072;
constexpr int CHUNK_BF16=2560, CHUNK_BYTES=5120;
constexpr int ROWS_PER_PATCH=2, PATCHES=2, COLS=4;
constexpr int M_PER_TILE=32, K_CHUNK=256, GROUP_SIZE=32;
constexpr int GROUPS_PER_CHUNK=M_PER_TILE;
constexpr int Q4_SCALE=512, Q4_DATA=4096;

// AUX constants
// HIDDEN_DWORDS = H/2 = 512
// RMS_NORM_DWORDS = HIDDEN_DWORDS*2 = 1024
// QK_ROPE_BF16 = HD+HD+HD = 384 → 192 dwords
// AUX_DWORDS = 1024 + 192 = 1216
constexpr int HID_DW=H/2, AUX_DW=1216;
// TOTAL_WEIGHT_AND_AUX_I32 = 2458816
constexpr int TOTAL_DW=2458816;

// Simple JSON parser helpers
static const char* find_json_key(const char*js, size_t jl, const char*key){
    size_t kl=strlen(key);
    const char*q=(const char*)memmem(js,jl,key,kl);
    return q;
}
static int64_t get_data_offset(const char*js, size_t jl, const char*key){
    const char*q=find_json_key(js,jl,key);
    if(!q)return -1;
    auto o=(const char*)memmem(q,jl-(q-js),"\"data_offsets\"",14);
    if(!o)return -1;
    auto a=(const char*)memchr(o,'[',30);
    if(!a)return -1;
    return strtoll(a+1,NULL,10);
}
static void get_shape(const char*js, size_t jl, const char*key, int&d0,int&d1){
    d0=d1=0;
    const char*q=find_json_key(js,jl,key);
    if(!q)return;
    auto s=(const char*)memmem(q,jl-(q-js),"\"shape\"",7);
    if(!s)return;
    auto a=(const char*)memchr(s,'[',30);
    if(!a)return;
    d0=strtol(a+1,NULL,10);
    auto c=(const char*)memchr(a,',',30);
    if(c)d1=strtol(c+1,NULL,10);
}

// Read bf16 weight array from model data section
static void read_bf16(const uint8_t*data, int64_t data_off, int64_t off, float*dst, int n){
    auto src=(const uint16_t*)(data+data_off+off);
    for(int i=0;i<n;i++){
        uint32_t b=((uint32_t)src[i])<<16;
        float f;memcpy(&f,&b,4);
        dst[i]=(src[i]&0x7F80)==0x7F80?0.0f:f;
    }
}

// Write bf16 from float to byte stream
static void write_bf16(std::vector<uint8_t>&out,const float*src,int n){
    for(int i=0;i<n;i++){
        uint32_t fb;memcpy(&fb,&src[i],4);
        uint16_t b=(uint16_t)(fb>>16);
        out.push_back((uint8_t)(b&0xFF));
        out.push_back((uint8_t)(b>>8));
    }
}

// Build the fused weight stream
bool build_fused_weights(const char*model_path, const char*output_path){
    printf("Loading model: %s\n",model_path);

    // Read entire model file
    FILE*f=fopen(model_path,"rb");if(!f){printf("Cannot open model\n");return false;}
    fseek(f,0,2);size_t fsz=ftell(f);fseek(f,0,0);
    std::vector<uint8_t> model_data(fsz);
    fread(model_data.data(),1,fsz,f);fclose(f);

    uint64_t hsz;memcpy(&hsz,model_data.data(),8);
    int64_t data_off=8+hsz;
    const char*js=(const char*)(model_data.data()+8);
    size_t jl=(size_t)hsz;

    printf("JSON header: %zu bytes, data at offset %ld\n",jl,data_off);

    // === Phase 1: Build AUX prefix ===
    printf("\n=== Building AUX prefix ===\n");
    std::vector<uint8_t> aux_bytes;

    // input_norm.weight: H bf16s
    {
        char key[256];snprintf(key,sizeof(key),"model.layers.0.input_layernorm.weight");
        int64_t off=get_data_offset(js,jl,key);
        std::vector<float> w(H);
        if(off>=0){read_bf16(model_data.data(),data_off,off,w.data(),H);
                   printf("  input_norm: offset=%ld, w[0]=%f\n",off,w[0]);}
        else{for(int i=0;i<H;i++)w[i]=1.0f;printf("  input_norm: defaults\n");}
        write_bf16(aux_bytes,w.data(),H);
    }

    // post_attention_layernorm.weight: H bf16s
    {
        char key[256];snprintf(key,sizeof(key),"model.layers.0.post_attention_layernorm.weight");
        int64_t off=get_data_offset(js,jl,key);
        std::vector<float> w(H);
        if(off>=0){read_bf16(model_data.data(),data_off,off,w.data(),H);
                   printf("  post_norm: offset=%ld\n",off);}
        else{for(int i=0;i<H;i++)w[i]=1.0f;}
        write_bf16(aux_bytes,w.data(),H);
    }

    // q_norm.weight: HD bf16s (if present)
    {
        char key[256];snprintf(key,sizeof(key),"model.layers.0.self_attn.q_norm.weight");
        int64_t off=get_data_offset(js,jl,key);
        std::vector<float> w(HD);
        if(off>=0){read_bf16(model_data.data(),data_off,off,w.data(),HD);
                   printf("  q_norm: offset=%ld\n",off);}
        else{for(int i=0;i<HD;i++)w[i]=1.0f;printf("  q_norm: defaults\n");}
        write_bf16(aux_bytes,w.data(),HD);
    }

    // k_norm.weight: HD bf16s
    {
        char key[256];snprintf(key,sizeof(key),"model.layers.0.self_attn.k_norm.weight");
        int64_t off=get_data_offset(js,jl,key);
        std::vector<float> w(HD);
        if(off>=0){read_bf16(model_data.data(),data_off,off,w.data(),HD);
                   printf("  k_norm: offset=%ld\n",off);}
        else{for(int i=0;i<HD;i++)w[i]=1.0f;}
        write_bf16(aux_bytes,w.data(),HD);
    }

    // RoPE cos/sin angles for position 0
    printf("  RoPE: theta=1000000.0\n");
    float theta=1000000.0f;
    for(int pass=0;pass<2;pass++){ // cos then sin
        for(int d=0;d<HD/2;d++){
            int di=d*2;
            float inv=1.0f/powf(theta,(float)di/(float)HD);
            float ang=0.0f*inv; // position 0
            float val=pass==0?cosf(ang):sinf(ang);
            uint32_t fb;memcpy(&fb,&val,4);
            aux_bytes.push_back((uint8_t)(fb>>16));
            aux_bytes.push_back((uint8_t)(fb>>24));
        }
    }

    // Pad AUX to exactly 1216 dwords (4864 bytes)
    while(aux_bytes.size()<AUX_DW*4) aux_bytes.push_back(0);
    printf("  AUX: %zu bytes (%zu dwords, expected %d)\n",aux_bytes.size(),aux_bytes.size()/4,AUX_DW);

    // === Phase 2: Build weight stream ===
    printf("\n=== Building Q4NX weight stream ===\n");

    // Phase ordering for streaming
    const char* phases[]={
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    };
    const char* phase_names[]={"Q","K","V","O","UP","GATE","DOWN"};
    // Fixed Qwen3-0.6B projection dimensions (out_dim, in_dim) — hardcoded rather than parsed
    // from the model JSON's "shape" field, which this file's naive get_shape() misreads (it
    // picked up garbage like [256,5120] during testing). Matches pack_fused_v3.py's phase table.
    const int phase_out_dim[]={NH*HD, NKV*HD, NKV*HD, H,   IM,  IM,  H};
    const int phase_in_dim[] ={H,     H,      H,      NH*HD, H, H,   IM};

    // Collect all chunks from each phase. Chunks are read sequentially from the model file in
    // row_chunk-major, input_chunk-minor order (row_chunk = output tile row 0..tile_rows-1,
    // input_chunk = input tile col 0..n_tile_cols-1) — this is the order FLM's own file format
    // uses, and it's what the schedule below indexes as `row_chunk * chunks + input_chunk`.
    struct ChunkInfo{const char*name;std::vector<uint8_t>data;int count;int blocks;int chunks;};
    std::vector<ChunkInfo> all_phases;
    int total_chunks=0;

    for(int p=0;p<7;p++){
        ChunkInfo ci;ci.name=phase_names[p];ci.count=0;ci.blocks=0;ci.chunks=0;
        int64_t off=get_data_offset(js,jl,phases[p]);
        int s0=phase_in_dim[p],s1=phase_out_dim[p];
        if(off>=0){
            // Compute chunk count from tile dimensions
            int in_d=s0, out_d=s1;
            int n_tile_cols=(in_d+255)/256; // ceil(in/256) = input chunks per tile row
            int tile_rows=(out_d+31)/32; // ceil(out/32) = output tile rows (blocks*16, see schedule)
            int nchunks=n_tile_cols*tile_rows;
            // "block" = one group×patch×row_in_patch sweep (16 row_chunks); tile_rows must be
            // a multiple of 16 by construction (4 columns × 2 patches × 2 rows_per_patch).
            ci.blocks=tile_rows/16; ci.chunks=n_tile_cols;

            // Read chunks from model
            for(int i=0;i<nchunks;i++){
                int64_t chunk_off=off+i*CHUNK_BYTES;
                ci.data.insert(ci.data.end(),
                    model_data.begin()+data_off+chunk_off,
                    model_data.begin()+data_off+chunk_off+CHUNK_BYTES);
                ci.count++;
            }
            printf("  %s: %d chunks (shape=[%d,%d], %d tile_cols × %d tile_rows)\n",
                   phase_names[p],ci.count,s0,s1,n_tile_cols,tile_rows);
        }else{
            printf("  %s: NOT FOUND in model JSON\n",phase_names[p]);
        }
        all_phases.push_back(ci);
        total_chunks+=ci.count;
    }

    printf("  Total: %d chunks (%d bytes each = %.0f KB total)\n",
           total_chunks,CHUNK_BYTES,(double)total_chunks*CHUNK_BYTES/1024);

    // Build the FLM schedule: (phase_index, block, input_chunk) triples in
    // Q,K,V -> O -> interleaved(UP,GATE) per block -> DOWN order. This matches
    // qwen3_model.py::layer_weight_stream / _projection_stream_from_schedule exactly.
    enum PhaseIdx{Q=0,K=1,V=2,O=3,UP=4,GATE=5,DOWN=6};
    struct SchedEntry{int phase;int block;int input_chunk;};
    std::vector<SchedEntry> schedule;
    auto append_phase=[&](int p){
        for(int b=0;b<all_phases[p].blocks;b++)
            for(int c=0;c<all_phases[p].chunks;c++)
                schedule.push_back({p,b,c});
    };
    append_phase(Q); append_phase(K); append_phase(V);
    append_phase(O);
    for(int b=0;b<all_phases[UP].blocks;b++){
        for(int c=0;c<all_phases[UP].chunks;c++) schedule.push_back({UP,b,c});
        for(int c=0;c<all_phases[GATE].chunks;c++) schedule.push_back({GATE,b,c});
    }
    append_phase(DOWN);
    printf("  Schedule: %zu entries\n",schedule.size());

    // Emit chunks in group(column) -> patch -> schedule -> row_in_patch order. Each AIE core
    // (group, row=patch*2+row_in_patch) owns a distinct 32-row slice of every projection matrix,
    // selected via row_chunk = block*16 + group*4 + patch*2 + row_in_patch.
    std::vector<uint8_t> weight_stream;
    weight_stream.reserve((size_t)COLS*PATCHES*schedule.size()*ROWS_PER_PATCH*CHUNK_BYTES);

    for(int group=0;group<COLS;group++){
        for(int patch=0;patch<PATCHES;patch++){
            for(auto&e:schedule){
                auto&ci=all_phases[e.phase];
                if(ci.chunks==0) continue;
                for(int row_in_patch=0;row_in_patch<ROWS_PER_PATCH;row_in_patch++){
                    int row_chunk=e.block*16+group*4+patch*2+row_in_patch;
                    int source=row_chunk*ci.chunks+e.input_chunk;
                    auto start=ci.data.begin()+(size_t)source*CHUNK_BYTES;
                    auto end=start+CHUNK_BYTES;
                    weight_stream.insert(weight_stream.end(),start,end);
                }
            }
        }
    }

    printf("\n  Weight stream: %zu bytes (%zu dwords)\n",weight_stream.size(),weight_stream.size()/4);
    printf("  Expected: %d dwords\n",TOTAL_DW-AUX_DW);

    // === Phase 3: Combine ===
    printf("\n=== Writing output ===\n");
    FILE*of=fopen(output_path,"wb");
    if(!of){printf("Cannot create output\n");return false;}

    // Write AUX + weights
    fwrite(aux_bytes.data(),1,aux_bytes.size(),of);
    fwrite(weight_stream.data(),1,weight_stream.size(),of);

    // Pad to exact total
    size_t written=aux_bytes.size()+weight_stream.size();
    size_t target=(size_t)TOTAL_DW*4;
    while(written<target){fputc(0,of);written++;}

    long fsz_out=ftell(of);
    fclose(of);
    printf("  Output: %ld bytes (%ld dwords, target %d)\n",fsz_out,fsz_out/4,TOTAL_DW);
    printf("  Diff from target: %ld bytes\n",(long)(fsz_out-(long)target*4));

    return true;
}

int main(int argc,char**argv){
    const char*model=argc>1?argv[1]:"/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
    const char*out=argc>2?argv[2]:"/home/bcloud/npu-sandbox/npu-infer/build/int8/fused_weights_l0.bin";
    printf("=== Q4NX Fused Weight Stream Packer ===\n\n");
    if(!build_fused_weights(model,out))return 1;
    printf("\n✅ DONE\n");
    return 0;
}

// Minimal ONNX weight extractor — protobuf wire format, zero dependencies.
// Reads .onnx files and extracts weight tensors into rcpp_bitnet_model_t.
//
// ONNX is protobuf. We decode the wire format directly (no libprotobuf needed)
// to extract tensor data from the GraphProto initializer list.

#include "rocm_cpp/bitnet_model.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>

#define HIP_CHECK(e) do { auto _s=(e); if(_s!=hipSuccess){fprintf(stderr,"HIP %d %s:%d\\n",_s,__FILE__,__LINE__); return RCPP_HIP_ERROR;}} while(0)

namespace {

// Minimal protobuf wire format decoder — handles the ONNX subset
struct PbReader {
    const uint8_t* data;
    size_t len;
    size_t pos = 0;

    PbReader(const uint8_t* d, size_t l) : data(d), len(l) {}

    bool ok() const { return pos < len; }

    // Read a varint (protobuf variable-length integer)
    uint64_t varint() {
        uint64_t val = 0;
        int shift = 0;
        while (pos < len) {
            uint8_t byte = data[pos++];
            val |= uint64_t(byte & 0x7F) << shift;
            shift += 7;
            if (!(byte & 0x80)) return val;
        }
        return val;
    }

    // Read a length-delimited field value (wire type 2)
    std::vector<uint8_t> bytes() {
        uint64_t sz = varint();
        if (pos + sz > len) sz = len - pos;
        std::vector<uint8_t> result(data + pos, data + pos + sz);
        pos += sz;
        return result;
    }

    // Read a fixed 32-bit value (wire type 5)
    uint32_t fixed32() {
        if (pos + 4 > len) return 0;
        uint32_t v;
        memcpy(&v, data + pos, 4);
        pos += 4;
        return v;
    }

    // Read a fixed 64-bit value (wire type 1)
    uint64_t fixed64() {
        if (pos + 8 > len) return 0;
        uint64_t v;
        memcpy(&v, data + pos, 8);
        pos += 8;
        return v;
    }

    // Skip current field — with bounds check to prevent OOB reads
    // from crafted/malformed ONNX files (issue #960).
    void skip_field(uint32_t wire_type) {
        if (wire_type == 0) { varint(); }
        else if (wire_type == 1) { if (pos + 8 > len) { pos = len; return; } pos += 8; }
        else if (wire_type == 2) { uint64_t sz = varint(); if (pos + sz > len) sz = len - pos; pos += sz; }
        else if (wire_type == 5) { if (pos + 4 > len) { pos = len; return; } pos += 4; }
    }
};

// ONNX tensor data types
enum OnnxDataType {
    ONNX_FLOAT = 1,
    ONNX_UINT8 = 2,
    ONNX_INT8 = 3,
    ONNX_UINT16 = 4,
    ONNX_INT16 = 5,
    ONNX_INT32 = 6,
    ONNX_INT64 = 7,
    ONNX_FLOAT16 = 10,
    ONNX_BFLOAT16 = 16,
};

struct OnnxTensor {
    std::string name;
    std::vector<int64_t> dims;
    int32_t data_type = 0;
    std::vector<float> float_data;  // dequantized
};

// Recursively find all initializer tensors in an ONNX protobuf
static void find_initializers(PbReader& pb, std::vector<OnnxTensor>& tensors, int depth = 0) {
    if (depth > 20 || !pb.ok()) return;

    while (pb.ok()) {
        size_t field_start = pb.pos;
        if (field_start >= pb.len) break;

        uint8_t key_byte = pb.data[pb.pos++];
        uint32_t field_num = key_byte >> 3;
        uint32_t wire_type = key_byte & 0x7;

        if (field_num == 0) break; // shouldn't happen

        // GraphProto.initializer (field 12 in GraphProto)
        // We need to find the graph (field 7 in ModelProto) and then
        // look for initializer sub-messages (field 12 in GraphProto)
        if (wire_type == 2) {
            auto content = pb.bytes();
            PbReader sub(content.data(), content.size());

            if (field_num == 7) {
                // Could be GraphProto — recurse to find initializers
                find_initializers(sub, tensors, depth + 1);
            } else if (field_num == 12) {
                // Could be an initializer TensorProto — parse it
                OnnxTensor t;
                PbReader tp(content.data(), content.size());
                while (tp.ok()) {
                    size_t tk = tp.pos;
                    if (tk >= content.size()) break;
                    uint8_t tkb = tp.data[tp.pos++];
                    uint32_t tf = tkb >> 3;
                    uint32_t tw = tkb & 0x7;

                    if (tf == 1 && tw == 2) { // dims (packed varint, field 1, wire type 2)
                        auto dim_bytes = tp.bytes();
                        PbReader dim_pb(dim_bytes.data(), dim_bytes.size());
                        while (dim_pb.ok()) t.dims.push_back((int64_t)dim_pb.varint());
                    } else if (tf == 1 && tw == 0) { // dims (non-packed varint, rare)
                        t.dims.push_back((int64_t)tp.varint());
                    } else if (tf == 2) { // data_type (int32, field 2)
                        t.data_type = (int32_t)tp.varint();
                    } else if (tf == 3) { // segment (field 3) — skip
                        auto seg = tp.bytes();
                    } else if (tf == 4) { // float_data (float, repeated, field 4)
                        if (tw == 2) { auto b = tp.bytes();
                            for (size_t i = 0; i + 4 <= b.size(); i += 4) {
                                float v; memcpy(&v, &b[i], 4); t.float_data.push_back(v);
                            }
                        }
                    } else if (tf == 5) { // int32_data (field 5)
                        if (tw == 2) { auto b = tp.bytes();
                            for (size_t i = 0; i + 4 <= b.size(); i += 4) {
                                int32_t v; memcpy(&v, &b[i], 4); t.float_data.push_back((float)v);
                            }
                        }
                    } else if (tf == 7) { // strings (field 7) — skip
                        tp.skip_field(tw);
                    } else if (tf == 8) { // double_data (field 8)
                        if (tw == 2) { auto b = tp.bytes();
                            for (size_t i = 0; i + 8 <= b.size(); i += 8) {
                                double v; memcpy(&v, &b[i], 8); t.float_data.push_back((float)v);
                            }
                        }
                    } else if (tf == 9) { // int64_data (field 9)
                        if (tw == 2) { auto b = tp.bytes();
                            for (size_t i = 0; i + 8 <= b.size(); i += 8) {
                                int64_t v; memcpy(&v, &b[i], 8); t.float_data.push_back((float)v);
                            }
                        }
                    } else if (tf == 10) { // uint64_data (field 10) — skip
                        auto bd = tp.bytes();
                    } else if (tf == 14) { // raw_data (bytes, field 14)
                        auto raw = tp.bytes();
                        if (t.data_type == ONNX_FLOAT16) {
                            // Proper IEEE float16 → float32 (not bfloat16 shift trick)
                            t.float_data.resize(raw.size() / 2);
                            for (size_t i = 0; i + 2 <= raw.size(); i += 2) {
                                uint16_t f16; memcpy(&f16, &raw[i], 2);
                                uint32_t s = (f16 >> 15) & 1, e = (f16 >> 10) & 0x1f, m = f16 & 0x3ff;
                                float sign = s ? -1.0f : 1.0f;
                                float v;
                                if (e == 0)
                                    v = sign * (float)m * 5.9604644775390625e-08f;
                                else if (e == 31)
                                    v = m ? NAN : sign * INFINITY;
                                else
                                    v = sign * (1.0f + (float)m / 1024.0f) * powf(2.0f, (float)((int)e - 15));
                                t.float_data[i / 2] = v;
                            }
                        } else if (t.data_type == ONNX_BFLOAT16) {
                            // bfloat16: upper 16 bits of float32
                            t.float_data.resize(raw.size() / 2);
                            for (size_t i = 0; i + 2 <= raw.size(); i += 2) {
                                uint16_t bf16; memcpy(&bf16, &raw[i], 2);
                                uint32_t bits = (uint32_t)bf16 << 16;
                                float v; memcpy(&v, &bits, 4);
                                t.float_data[i / 2] = v;
                            }
                        } else if (t.data_type == ONNX_INT8 || t.data_type == ONNX_UINT8) {
                            // INT8/UINT8: expand 1 byte per value to float
                            t.float_data.resize(raw.size());
                            float scale = (t.data_type == ONNX_UINT8) ? 1.0f : 1.0f;
                            for (size_t i = 0; i < raw.size(); i++) {
                                int8_t byte = (int8_t)raw[i];
                                t.float_data[i] = (float)byte;
                            }
                        } else {
                            // Default: F32 (4 bytes per value)
                            size_t n_floats = raw.size() / 4;
                            t.float_data.resize(n_floats);
                            for (size_t i = 0; i + 4 <= raw.size(); i += 4) {
                                float v; memcpy(&v, &raw[i], 4);
                                t.float_data[i / 4] = v;
                            }
                        }
                    } else if (tf == 12) { // name (string, field 12)
                        auto nm = tp.bytes();
                        if (!nm.empty()) t.name.assign((char*)nm.data(), nm.size());
                    } else {
                        tp.skip_field(tw);
                    }
                }
                if (!t.name.empty() && !t.dims.empty() && !t.float_data.empty()) {
                    tensors.push_back(std::move(t));
                }
            } else {
                // Skip other length-delimited fields
            }
        } else if (wire_type == 0) {
            pb.varint();
        } else if (wire_type == 1) {
            pb.pos += 8;
        } else if (wire_type == 5) {
            pb.pos += 4;
        } else {
            break;
        }
    }
}

} // anonymous namespace

extern "C" {

rcpp_status_t rcpp_bitnet_load_onnx(const char* path, rcpp_bitnet_model_t* out_model) {
    if (!path || !out_model) return RCPP_INVALID_ARG;
    memset(out_model, 0, sizeof(*out_model));

    fprintf(stderr, "[onnx] Loading: %s\\n", path);

    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return RCPP_INVALID_ARG;
    size_t file_size = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);

    std::vector<uint8_t> file_data(file_size);
    f.read((char*)file_data.data(), file_size);
    if (!f) return RCPP_INVALID_ARG;

    // Parse protobuf to find initializer tensors
    PbReader pb(file_data.data(), file_size);
    std::vector<OnnxTensor> tensors;
    find_initializers(pb, tensors);

    fprintf(stderr, "[onnx] Found %zu tensors\\n", tensors.size());

    // ── Build name → tensor lookup ──────────────────────────────────────────
    std::unordered_map<std::string, OnnxTensor*> name_map;
    for (auto& t : tensors) name_map[t.name] = &t;

    auto lookup = [&](const std::string& name) -> OnnxTensor* {
        auto it = name_map.find(name);
        return (it != name_map.end()) ? it->second : nullptr;
    };

    // ── Determine number of layers ──────────────────────────────────────────
    int n_layers = 0;
    for (auto& t : tensors) {
        int lidx = -1;
        if (sscanf(t.name.c_str(), "model.layers.%d.", &lidx) == 1 && lidx >= 0) {
            if (lidx + 1 > n_layers) n_layers = lidx + 1;
        }
    }
    if (n_layers == 0) {
        fprintf(stderr, "[onnx] ERROR: no layers found\\n");
        return RCPP_INVALID_ARG;
    }

    // ── Determine model dimensions from tensor shapes ───────────────────────
    auto* emb_t = lookup("model.embed_tokens.weight");
    if (!emb_t || emb_t->dims.size() < 2) {
        fprintf(stderr, "[onnx] ERROR: missing embed_tokens.weight\\n");
        return RCPP_INVALID_ARG;
    }
    int hidden_size  = (int)emb_t->dims[1];
    int vocab_size   = (int)emb_t->dims[0];

    auto* gate0 = lookup("model.layers.0.mlp.gate_proj.weight");
    int intermediate_size = gate0 ? (int)gate0->dims[0] : hidden_size;

    // Detect dtype of weights (F32 or F16)
    int weight_dtype = emb_t->data_type;

    // ── Helper: allocate device memory and copy tensor data ─────────────────
    auto tensor_to_dev = [&](OnnxTensor* t, size_t* out_bytes = nullptr) -> void* {
        if (!t) {
            if (out_bytes) *out_bytes = 0;
            return nullptr;
        }
        size_t n_elems = t->float_data.size();
        void* dev_ptr = nullptr;
        size_t bytes = 0;

        if (t->data_type == ONNX_FLOAT16) {
            // Convert f32 back to f16 for device storage
            bytes = n_elems * sizeof(uint16_t);
            std::vector<uint16_t> f16_buf(n_elems);
            for (size_t i = 0; i < n_elems; i++) {
                float v = t->float_data[i];
                uint32_t f32_bits; memcpy(&f32_bits, &v, 4);
                uint32_t sign = (f32_bits >> 16) & 0x8000;
                int32_t exp = ((int32_t)(f32_bits >> 23) & 0xff) - 127 + 15;
                uint32_t mant = (f32_bits >> 13) & 0x3ff;
                if (exp <= 0) { f16_buf[i] = (uint16_t)sign; }
                else if (exp >= 31) { f16_buf[i] = (uint16_t)(sign | 0x7c00); }
                else { f16_buf[i] = (uint16_t)(sign | (exp << 10) | mant); }
            }
            if (hipMalloc(&dev_ptr, bytes) != hipSuccess) return nullptr;
            if (hipMemcpy(dev_ptr, f16_buf.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) { hipFree(dev_ptr); return nullptr; }
        } else if (t->data_type == ONNX_BFLOAT16) {
            // BF16 was already converted to F32 by the raw_data parser
            bytes = (size_t)n_elems * sizeof(float);
            if (hipMalloc(&dev_ptr, bytes) != hipSuccess) return nullptr;
            if (hipMemcpy(dev_ptr, t->float_data.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) { hipFree(dev_ptr); return nullptr; }
        } else if (t->data_type == ONNX_INT8 || t->data_type == ONNX_UINT8) {
            // INT8/UINT8 was already dequantized to F32 by the raw_data parser
            bytes = (size_t)n_elems * sizeof(float);
            if (hipMalloc(&dev_ptr, bytes) != hipSuccess) return nullptr;
            if (hipMemcpy(dev_ptr, t->float_data.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) { hipFree(dev_ptr); return nullptr; }
        } else {
            // F32 (ONNX_FLOAT) or fallback
            bytes = (size_t)n_elems * sizeof(float);
            if (hipMalloc(&dev_ptr, bytes) != hipSuccess) return nullptr;
            if (hipMemcpy(dev_ptr, t->float_data.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) { hipFree(dev_ptr); return nullptr; }
        }

        if (out_bytes) *out_bytes = bytes;
        return dev_ptr;
    };

    // ── Allocate embedding and final norm ──────────────────────────────────
    out_model->embedding_dev = tensor_to_dev(emb_t);

    auto* norm_t = lookup("model.norm.weight");
    out_model->final_norm_weight_dev = tensor_to_dev(norm_t);

    // ── Allocate per-layer weights ──────────────────────────────────────────
    out_model->layers = (rcpp_bitnet_layer_t*)calloc(n_layers, sizeof(rcpp_bitnet_layer_t));
    if (!out_model->layers) return RCPP_INTERNAL;

    for (int i = 0; i < n_layers; i++) {
        char buf[256];
        auto& L = out_model->layers[i];

        snprintf(buf, sizeof(buf), "model.layers.%d.input_layernorm.weight", i);
        L.input_norm_dev = tensor_to_dev(lookup(buf));

        snprintf(buf, sizeof(buf), "model.layers.%d.post_attention_layernorm.weight", i);
        L.post_attn_norm_dev = tensor_to_dev(lookup(buf));

        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.q_proj.weight", i);
        L.q_packed_dev = tensor_to_dev(lookup(buf));

        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.k_proj.weight", i);
        L.k_packed_dev = tensor_to_dev(lookup(buf));

        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.v_proj.weight", i);
        L.v_packed_dev = tensor_to_dev(lookup(buf));

        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.o_proj.weight", i);
        L.o_packed_dev = tensor_to_dev(lookup(buf));

        snprintf(buf, sizeof(buf), "model.layers.%d.mlp.gate_proj.weight", i);
        L.gate_packed_dev = tensor_to_dev(lookup(buf));

        snprintf(buf, sizeof(buf), "model.layers.%d.mlp.up_proj.weight", i);
        L.up_packed_dev = tensor_to_dev(lookup(buf));

        snprintf(buf, sizeof(buf), "model.layers.%d.mlp.down_proj.weight", i);
        L.down_packed_dev = tensor_to_dev(lookup(buf));
    }

    // ── Fill model metadata ─────────────────────────────────────────────────
    out_model->hidden_size       = hidden_size;
    out_model->intermediate_size = intermediate_size;
    out_model->num_layers        = n_layers;
    out_model->num_heads         = 0;      // caller should set from config
    out_model->num_kv_heads      = 0;      // caller should set from config
    out_model->vocab_size        = vocab_size;
    out_model->max_seq_len       = 2048;
    out_model->tie_embeddings    = 0;
    out_model->rope_theta        = 10000.0f;
    out_model->rms_norm_eps      = 1e-5f;
    out_model->format_version    = 0;
    out_model->flags             = 0;
    out_model->weight_format     = RCPP_WEIGHT_FORMAT_HALO_V2;
    out_model->is_qwen3          = 0;
    out_model->arch              = RCPP_ARCH_BITNET;

    fprintf(stderr, "[onnx] Model built: %d layers, hidden=%d, intermediate=%d, vocab=%d, dtype=%s\\n",
            n_layers, hidden_size, intermediate_size, vocab_size,
            weight_dtype == ONNX_FLOAT16 ? "F16" : "F32");

    return RCPP_OK;
}

} // extern "C"

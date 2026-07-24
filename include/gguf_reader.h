#pragma once
// gguf_reader.h — the single, verified GGUF format parser for this codebase.
//
// Before this existed, four different files independently re-implemented
// GGUF tensor-info parsing and dequantization, and three of them had the
// same class of bug (wrong dtype enum values, or tensor-info fields read in
// the wrong order) — see gguf_loader.cpp's fix history, the Q8_0/Q5_0/Q5_1
// enum fix in backend_hip.cpp, and the deleted dead code that used to live
// in backend_generic.cpp (issue #490). Every GGUF-consuming file should use
// this instead of writing its own parser.
//
// Dtype enum values and dequantization math are verified against
// ~/llama.cpp/ggml/include/ggml.h and byte-exact (5 random seeds each)
// against the Python `gguf` package's reference dequantizers for all 11
// supported dtypes (F32/F16 trivially, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K-Q8_K),
// 2026-07-19. Two real bugs were caught by this verification and are not
// hypothetical: the initial Q6_K port only ever wrote 64 of every 256
// elements per super-block, and the initial Q4_0/Q4_1/Q5_0/Q5_1 port used
// the wrong nibble-to-element mapping (interleaved pairs instead of
// low-nibbles-then-high-nibbles) — both silently produced wrong weights for
// every real tensor of those types, and neither had ever been checked
// against anything but "does it crash" before this verification pass.

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

// Real ggml_type values (0=F32, 1=F16, 2=Q4_0, 3=Q4_1, [4,5 removed],
// 6=Q5_0, 7=Q5_1, 8=Q8_0, 9=Q8_1, 10..15=K-quants,
// 16=IQ2_XXS, 17=IQ3_XXS, 18=IQ1_S, 19=IQ4_NL, 20=IQ3_S, 21=IQ2_S,
// 22=IQ4_XS, 23=IQ1_M, 24=BF16, 26=Q4_0_4_4, 27=Q4_0_4_8, 28=Q4_0_8_8).
// Only F32/F16/BF16 and the standard quant types listed below are supported
// for dequant; IQ-format tensors are dequantized via a shared inline helper.
enum GgufDtype : uint32_t {
    GGUF_DTYPE_F32    = 0,
    GGUF_DTYPE_F16    = 1,
    GGUF_DTYPE_Q4_0   = 2,
    GGUF_DTYPE_Q4_1   = 3,
    GGUF_DTYPE_Q5_0   = 6,
    GGUF_DTYPE_Q5_1   = 7,
    GGUF_DTYPE_Q8_0   = 8,
    GGUF_DTYPE_Q8_1   = 9,
    GGUF_DTYPE_Q2_K   = 10,
    GGUF_DTYPE_Q3_K   = 11,
    GGUF_DTYPE_Q4_K   = 12,
    GGUF_DTYPE_Q5_K   = 13,
    GGUF_DTYPE_Q6_K   = 14,
    GGUF_DTYPE_Q8_K   = 15,
    GGUF_DTYPE_IQ2_XXS = 16,
    GGUF_DTYPE_IQ3_XXS = 17,
    GGUF_DTYPE_IQ1_S   = 18,
    GGUF_DTYPE_IQ4_NL  = 19,
    GGUF_DTYPE_IQ3_S   = 20,
    GGUF_DTYPE_IQ2_S   = 21,
    GGUF_DTYPE_IQ4_XS  = 22,
    GGUF_DTYPE_IQ1_M   = 23,
    GGUF_DTYPE_BF16   = 24,
    GGUF_DTYPE_Q4_0_4_4 = 26,
    GGUF_DTYPE_Q4_0_4_8 = 27,
    GGUF_DTYPE_Q4_0_8_8 = 28,
    // Custom / project-specific dtypes (collide with real ggml_type enums
    // but are used by this project's weight formats)
    GGUF_DTYPE_TQ2_0_G128 = 42,  // Ternary Q2_0, group size 128 (h1b format)
    GGUF_DTYPE_Q1_0_G128 = 41,   // Binary Q1_0, group size 128 (h1b format)
    // llama.cpp native ternary block types (GGML_TYPE values from ggml.h)
    GGUF_DTYPE_TQ1_0_LLAMA = 34,  // GGML_TYPE_TQ1_0 — 1.6875 bpw base-3 ternary
    GGUF_DTYPE_TQ2_0_LLAMA = 35,  // GGML_TYPE_TQ2_0 — 2.0625 bpw 2-bit ternary
};

// Block size (elements) and bytes-per-block for a dtype. Returns {0,0} for
// an unrecognized dtype.
struct GgufBlockInfo { int block_size; int block_bytes; };
GgufBlockInfo gguf_block_info(uint32_t dtype);

// Dequantize `count` elements of raw block data (as read straight off disk)
// to float32. `data` must hold enough whole blocks to cover `count` elements
// (i.e. ceil(count / block_size) blocks). Returns false for an unrecognized
// dtype; F32/F16 are handled here too for a uniform call site.
bool gguf_dequant(uint32_t dtype, const uint8_t* data, float* out, int count);

struct GgufTensorInfo {
    std::vector<uint64_t> shape;  // GGUF native order, shape[0] fastest-varying
    uint32_t dtype = 0;
    uint64_t abs_offset = 0;      // absolute byte offset of this tensor's data in the file
    uint64_t numel = 0;
};

// A single open GGUF file: header + KV metadata + tensor-info table parsed
// up front; tensor data itself is read (and dequantized) lazily on request.
class GgufReader {
public:
    GgufReader() = default;
    ~GgufReader();
    GgufReader(const GgufReader&) = delete;
    GgufReader& operator=(const GgufReader&) = delete;

    bool open(const std::string& path);
    bool is_open() const { return f_ != nullptr; }

    const std::string& architecture() const { return arch_; }
    bool has_tensor(const std::string& name) const;
    const GgufTensorInfo* tensor_info(const std::string& name) const;
    // Names of every tensor in the file, in the order GGUF declares them.
    const std::vector<std::string>& tensor_names() const { return tensor_order_; }

    // KV metadata accessors. Return false if the key is absent or holds a
    // different type than requested (arrays/strings for scalar accessors,
    // etc). Integer accessors accept any of GGUF's int/uint scalar types.
    bool get_u32(const std::string& key, uint32_t& out) const;
    bool get_f32(const std::string& key, float& out) const;
    bool get_string(const std::string& key, std::string& out) const;
    bool get_string_array(const std::string& key, std::vector<std::string>& out) const;
    // All KV metadata key names present in the file, for architecture-
    // agnostic callers that need to match keys by suffix (e.g. any
    // "<arch>.attention.head_count") rather than a known exact name.
    std::vector<std::string> kv_keys() const;

    // Dequantize a named tensor to float32. Returns false if not found.
    bool get_tensor_f32(const std::string& name, std::vector<float>& out, size_t* out_n = nullptr);

    // Raw block bytes for a tensor, undequantized — for callers with their
    // own dequantization for a dtype this module doesn't know about (e.g. a
    // project-specific block-scaled-ternary format sharing a real ggml_type
    // numeric value). Caller supplies the block geometry since it's not one
    // of GgufDtype's real types. Returns false if the tensor isn't found.
    bool get_tensor_raw(const std::string& name, int block_size, int block_bytes,
                        std::vector<uint8_t>& out, uint64_t* out_numel = nullptr);

private:
    struct KV {
        uint32_t vtype = 0;
        uint64_t u = 0;
        double f = 0;
        std::string s;
        std::vector<std::string> arr_str;
    };

    FILE* f_ = nullptr;
    std::string arch_;
    std::unordered_map<std::string, GgufTensorInfo> tensors_;
    std::vector<std::string> tensor_order_;
    std::unordered_map<std::string, KV> kv_;

    std::string read_string();
    bool read_kv_value(uint32_t vtype, KV& out);
    void skip_kv_value(uint32_t vtype);
};

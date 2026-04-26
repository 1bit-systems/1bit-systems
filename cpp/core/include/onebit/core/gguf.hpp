#pragma once

#include "onebit/core/error.hpp"
#include "onebit/core/types.hpp"

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace onebit::core::gguf {

inline constexpr std::array<std::uint8_t, 4> MAGIC = {'G', 'G', 'U', 'F'};
inline constexpr std::uint32_t MIN_VERSION = 3;

enum class ValueType : std::uint32_t {
    UInt8   = 0,
    Int8    = 1,
    UInt16  = 2,
    Int16   = 3,
    UInt32  = 4,
    Int32   = 5,
    Float32 = 6,
    Bool    = 7,
    String  = 8,
    Array   = 9,
    UInt64  = 10,
    Int64   = 11,
    Float64 = 12,
};

enum class TensorType : std::uint32_t {
    F32    = 0,
    F16    = 1,
    Q4_0   = 2,
    Q4_1   = 3,
    Q5_0   = 6,
    Q5_1   = 7,
    Q8_0   = 8,
    Q8_1   = 9,
    Q2_K   = 10,
    Q3_K   = 11,
    Q4_K   = 12,
    Q5_K   = 13,
    Q6_K   = 14,
    Q8_K   = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    IQ1_M   = 29,
    BF16    = 30,
    TQ1_0   = 34,
    TQ2_0   = 35,
};

struct Array;
using Value = std::variant<
    std::uint8_t, std::int8_t,
    std::uint16_t, std::int16_t,
    std::uint32_t, std::int32_t,
    float,
    bool,
    std::string,
    std::shared_ptr<Array>,
    std::uint64_t, std::int64_t,
    double>;

struct Array {
    ValueType          element_type = ValueType::UInt8;
    std::vector<Value> items;
};

struct TensorInfo {
    std::string                 name;
    std::vector<std::uint64_t>  dims;
    TensorType                  type   = TensorType::F32;
    std::uint64_t               offset = 0;
};

// Halo-side derived view of a BitNet GGUF: vocab, hidden, etc, looked up
// from the metadata KV map.
struct BitnetHeader {
    std::int32_t hidden_dim       = 0;
    std::int32_t intermediate_dim = 0;
    std::int32_t num_heads        = 0;
    std::int32_t num_kv_heads     = 0;
    std::int32_t num_layers       = 0;
    std::int32_t vocab_size       = 0;
    std::int32_t max_seq_len      = 0;
    float        rope_theta       = DEFAULT_ROPE_THETA;
    float        rms_norm_eps     = DEFAULT_RMS_NORM_EPS;
};

class GgufFile {
public:
    // C.66: not noexcept because metadata_ (unordered_map<string, Value>)
    // and tensors_ (vector<TensorInfo>) move via allocator-aware paths
    // that don't promise nothrow on all stdlibs.
    GgufFile();
    GgufFile(const GgufFile&)            = delete;
    GgufFile& operator=(const GgufFile&) = delete;
    GgufFile(GgufFile&&);
    GgufFile& operator=(GgufFile&&);
    ~GgufFile();

    [[nodiscard]] static std::expected<GgufFile, HaloError>
    open(const std::filesystem::path& path);

    [[nodiscard]] std::uint32_t                                version()   const noexcept { return version_; }
    [[nodiscard]] const std::unordered_map<std::string, Value>& metadata() const noexcept { return metadata_; }
    [[nodiscard]] const std::vector<TensorInfo>&               tensors()  const noexcept { return tensors_; }

    [[nodiscard]] std::expected<std::span<const std::uint8_t>, HaloError>
    tensor_bytes(const TensorInfo& info) const noexcept;

    [[nodiscard]] std::expected<BitnetHeader, HaloError> bitnet_header() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl>                            impl_;
    std::uint32_t                                    version_ = 0;
    std::unordered_map<std::string, Value>           metadata_{};
    std::vector<TensorInfo>                          tensors_{};
};

} // namespace onebit::core::gguf

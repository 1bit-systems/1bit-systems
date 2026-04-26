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
#include <vector>

namespace onebit::core::h1b {

// Magic = "H1B\0"
inline constexpr std::array<std::uint8_t, 4> MAGIC = {'H', '1', 'B', 0};

// Format flags (bitmask in header).
inline constexpr std::uint32_t FLAG_HADAMARD_ROTATED = 1u << 0;
inline constexpr std::uint32_t FLAG_SHERRY_FP16      = 1u << 1;
inline constexpr std::uint32_t FLAG_BONSAI_TQ2       = 1u << 2;
inline constexpr std::uint32_t FLAG_BONSAI_Q1        = 1u << 3;

inline constexpr std::uint32_t BONSAI_GROUP_SIZE = 32;

enum class WeightFormat : std::uint32_t {
    Ternary2bpw       = 0,
    HadamardRotated   = FLAG_HADAMARD_ROTATED,
    SherryFp16Sparse  = FLAG_SHERRY_FP16,
    BonsaiTq2         = FLAG_BONSAI_TQ2,
    BonsaiQ1          = FLAG_BONSAI_Q1,
};

struct Config {
    std::int32_t version          = 0;
    std::int32_t hidden_dim       = 0;
    std::int32_t intermediate_dim = 0;
    std::int32_t num_heads        = 0;
    std::int32_t num_kv_heads     = 0;
    std::int32_t num_layers       = 0;
    std::int32_t vocab_size       = 0;
    std::int32_t max_seq_len      = 0;
    std::uint32_t flags           = 0;
    float        rope_theta       = DEFAULT_ROPE_THETA;
    float        rms_norm_eps     = DEFAULT_RMS_NORM_EPS;
};

struct LayerOffsets {
    std::uint64_t attn_q     = 0;
    std::uint64_t attn_k     = 0;
    std::uint64_t attn_v     = 0;
    std::uint64_t attn_o     = 0;
    std::uint64_t ffn_gate   = 0;
    std::uint64_t ffn_up     = 0;
    std::uint64_t ffn_down   = 0;
    std::uint64_t attn_norm  = 0;
    std::uint64_t ffn_norm   = 0;
};

// Memory-mapped .h1b file. Raw byte access only — kernels in rocm-cpp
// pull tensors out by offset.
class File {
public:
    // C.66: not marked noexcept because std::filesystem::path move is
    // not nothrow on libstdc++ (small-string buffer can allocate).
    File();
    File(const File&)            = delete;
    File& operator=(const File&) = delete;
    File(File&&);
    File& operator=(File&&);
    ~File();

    [[nodiscard]] static std::expected<File, HaloError>
    open(const std::filesystem::path& path);

    [[nodiscard]] const Config&                     config() const noexcept { return config_; }
    [[nodiscard]] const std::vector<LayerOffsets>&  layers() const noexcept { return layers_; }
    [[nodiscard]] std::span<const std::uint8_t>     bytes()  const noexcept;
    [[nodiscard]] WeightFormat                      format() const noexcept;
    [[nodiscard]] const std::filesystem::path&      path()   const noexcept { return path_; }

    // Tensor lookup by absolute byte offset; bounds-checked.
    [[nodiscard]] std::expected<std::span<const std::uint8_t>, HaloError>
    slice(std::uint64_t offset, std::size_t bytes) const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl>      impl_;
    std::filesystem::path      path_;
    Config                     config_{};
    std::vector<LayerOffsets>  layers_{};
};

} // namespace onebit::core::h1b

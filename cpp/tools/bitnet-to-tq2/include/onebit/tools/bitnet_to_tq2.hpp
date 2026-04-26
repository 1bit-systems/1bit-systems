// bitnet-to-tq2 — header-only kernels + minimal safetensors reader.
//
// Faithful C++23 port of `tools/bitnet-to-tq2/src/lib.rs` (the production
// converter behind RyzenAI's bf16 master → halo .h1b v4 path). Layout
// references:
//   * `tools/bitnet-to-tq2/src/lib.rs`
//   * `docs/wiki/Bonsai-Kernel-Spec.md` §TQ2_0_g128
//
// Header-only by design — same TU is shared between CLI + test suite.

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace onebit::tools::bitnet_to_tq2 {

// ----- Public constants ------------------------------------------------------

inline constexpr std::size_t TQ2_GROUP_SIZE  = 128;
inline constexpr std::size_t TQ2_BLOCK_BYTES = 34; // 2 B fp16 d + 32 B codes
inline constexpr std::int32_t H1B_FLAG_BONSAI_TQ2 = 0x4;
inline constexpr std::array<std::uint8_t, 4> H1B_MAGIC = {'H', '1', 'B', 0};

// ----- Errors ----------------------------------------------------------------

struct ConvertError {
    std::string what;
};

// ----- bf16 ↔ fp32 -----------------------------------------------------------

[[nodiscard]] inline float bf16_bits_to_f32(std::uint16_t bits) noexcept
{
    // bf16 = upper 16 bits of fp32; lower 16 zero.
    const std::uint32_t v = static_cast<std::uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &v, sizeof(f));
    return f;
}

[[nodiscard]] inline std::uint16_t f32_to_f16_bits(float v) noexcept
{
    // Standard IEEE 754 fp32 → fp16 with round-to-nearest-even, handling
    // overflow → inf, underflow → subnormal/zero, NaN → quiet NaN.
    std::uint32_t x;
    std::memcpy(&x, &v, sizeof(x));
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    std::int32_t  exp_  = static_cast<std::int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t mant  = x & 0x7FFFFFu;
    std::uint16_t out;
    if (exp_ >= 0x1F) {
        // Inf / NaN.
        if ((x & 0x7FFFFFFFu) > 0x7F800000u) {
            out = static_cast<std::uint16_t>(sign | 0x7E00u); // qNaN
        } else {
            out = static_cast<std::uint16_t>(sign | 0x7C00u); // inf
        }
    } else if (exp_ <= 0) {
        if (exp_ < -10) {
            out = static_cast<std::uint16_t>(sign);
        } else {
            mant |= 0x800000u;
            const std::uint32_t shift = static_cast<std::uint32_t>(14 - exp_);
            const std::uint32_t low = mant & ((1u << shift) - 1u);
            std::uint16_t m = static_cast<std::uint16_t>(mant >> shift);
            // Round-to-nearest-even.
            const std::uint32_t halfway = 1u << (shift - 1);
            if (low > halfway || (low == halfway && (m & 1u) != 0u)) ++m;
            out = static_cast<std::uint16_t>(sign | m);
        }
    } else {
        std::uint16_t m = static_cast<std::uint16_t>(mant >> 13);
        std::uint16_t e = static_cast<std::uint16_t>(exp_);
        const std::uint32_t low = mant & 0x1FFFu;
        if (low > 0x1000u || (low == 0x1000u && (m & 1u) != 0u)) {
            ++m;
            if (m == 0x400u) {
                m = 0;
                ++e;
                if (e >= 0x1F) {
                    return static_cast<std::uint16_t>(sign | 0x7C00u);
                }
            }
        }
        out = static_cast<std::uint16_t>(sign | (e << 10) | m);
    }
    return out;
}

[[nodiscard]] inline float f16_bits_to_f32(std::uint16_t bits) noexcept
{
    const std::uint32_t sign = (static_cast<std::uint32_t>(bits) & 0x8000u) << 16;
    std::int32_t  exp_  = static_cast<std::int32_t>((bits >> 10) & 0x1Fu);
    std::uint32_t mant  = bits & 0x3FFu;
    std::uint32_t out;
    if (exp_ == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            // Subnormal → normalize. exp_ is signed so we don't underflow.
            while ((mant & 0x400u) == 0u) {
                mant <<= 1;
                --exp_;
            }
            ++exp_;
            mant &= ~0x400u;
            out = sign
                | (static_cast<std::uint32_t>(exp_ + (127 - 15)) << 23)
                | (mant << 13);
        }
    } else if (exp_ == 0x1F) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign
            | (static_cast<std::uint32_t>(exp_ + (127 - 15)) << 23)
            | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

// ----- TQ2_0_g128 quantize / dequantize -------------------------------------

// Quantize one 128-element fp32 group into a 34-byte block using a
// caller-supplied scale `d_override`. Layout (matches PrismML +
// `bonsai_tq2_gemv.hip`):
//
//   block[0..2]   = fp16 d
//   block[2..34]  = 32 B codes, 4 lanes/byte LSB-first.
//   Codes: 00 → -d, 01 → 0, 10 → +d, 11 → unused (treated as 0).
[[nodiscard]] inline std::array<std::uint8_t, TQ2_BLOCK_BYTES>
quantize_group_with_scale(const std::array<float, TQ2_GROUP_SIZE>& w,
                          float                                    d_override) noexcept
{
    std::array<std::uint8_t, TQ2_BLOCK_BYTES> block{};
    if (d_override == 0.0f) {
        // All-zero scale → emit decoder-zero codes (0b01) on every lane.
        for (std::size_t i = 2; i < TQ2_BLOCK_BYTES; ++i) block[i] = 0x55;
        return block;
    }
    const float threshold = 0.5f * d_override;
    for (std::size_t j = 0; j < TQ2_GROUP_SIZE; ++j) {
        std::uint8_t code;
        if      (w[j] >=  threshold) code = 0b10;
        else if (w[j] <= -threshold) code = 0b00;
        else                         code = 0b01;
        const std::size_t byte_idx = 2 + j / 4;
        const std::uint32_t shift  = static_cast<std::uint32_t>((j % 4) * 2);
        block[byte_idx] = static_cast<std::uint8_t>(
            block[byte_idx] | static_cast<std::uint8_t>(code << shift));
    }
    const std::uint16_t d_bits = f32_to_f16_bits(d_override);
    block[0] = static_cast<std::uint8_t>(d_bits & 0xFFu);
    block[1] = static_cast<std::uint8_t>((d_bits >> 8) & 0xFFu);
    return block;
}

// Per-block absmean policy.
[[nodiscard]] inline std::array<std::uint8_t, TQ2_BLOCK_BYTES>
quantize_group_tq2(const std::array<float, TQ2_GROUP_SIZE>& w) noexcept
{
    double sum = 0.0;
    for (float v : w) sum += static_cast<double>(v < 0 ? -v : v);
    const float absmean = static_cast<float>(sum / static_cast<double>(TQ2_GROUP_SIZE));
    return quantize_group_with_scale(w, absmean);
}

// Inverse decoder for round-trip tests.
[[nodiscard]] inline std::array<float, TQ2_GROUP_SIZE>
dequantize_group_tq2(const std::array<std::uint8_t, TQ2_BLOCK_BYTES>& block) noexcept
{
    std::array<float, TQ2_GROUP_SIZE> out{};
    const std::uint16_t bits = static_cast<std::uint16_t>(
        static_cast<std::uint16_t>(block[0]) |
        (static_cast<std::uint16_t>(block[1]) << 8));
    const float d = f16_bits_to_f32(bits);
    for (std::size_t j = 0; j < TQ2_GROUP_SIZE; ++j) {
        const std::size_t byte_idx = 2 + j / 4;
        const std::uint32_t shift  = static_cast<std::uint32_t>((j % 4) * 2);
        const std::uint8_t code = static_cast<std::uint8_t>(
            (block[byte_idx] >> shift) & 0b11u);
        out[j] = (code == 0b00) ? -d : (code == 0b10) ? d : 0.0f;
    }
    return out;
}

// ----- Scale mode -----------------------------------------------------------

enum class ScaleMode { PerTensor, PerBlock };

// ----- Model config (parsed from HF config.json) ----------------------------

struct ModelConfig {
    std::uint32_t hidden_size              = 0;
    std::uint32_t intermediate_size        = 0;
    std::uint32_t num_hidden_layers        = 0;
    std::uint32_t num_attention_heads      = 0;
    std::uint32_t num_key_value_heads      = 0;
    std::uint32_t vocab_size               = 0;
    std::uint32_t max_position_embeddings  = 0;
    bool          tie_word_embeddings      = false;
    float         rope_theta               = 0.0f;
    float         rms_norm_eps             = 1e-5f;

    [[nodiscard]] static std::expected<ModelConfig, ConvertError>
    parse_json(std::span<const std::uint8_t> bytes)
    {
        nlohmann::json v;
        try {
            v = nlohmann::json::parse(bytes.begin(), bytes.end());
        } catch (const std::exception& e) {
            return std::unexpected(ConvertError{std::string{"json: "} + e.what()});
        }
        auto req_u = [&](const char* k) -> std::expected<std::uint32_t, ConvertError> {
            if (!v.contains(k) || !v[k].is_number())
                return std::unexpected(ConvertError{std::string{"missing "} + k});
            return static_cast<std::uint32_t>(v[k].get<std::int64_t>());
        };
        auto req_f = [&](const char* k) -> std::expected<float, ConvertError> {
            if (!v.contains(k) || !v[k].is_number())
                return std::unexpected(ConvertError{std::string{"missing "} + k});
            return static_cast<float>(v[k].get<double>());
        };
        ModelConfig c;
        if (auto r = req_u("hidden_size");             r) c.hidden_size             = *r; else return std::unexpected(r.error());
        if (auto r = req_u("intermediate_size");       r) c.intermediate_size       = *r; else return std::unexpected(r.error());
        if (auto r = req_u("num_hidden_layers");       r) c.num_hidden_layers       = *r; else return std::unexpected(r.error());
        if (auto r = req_u("num_attention_heads");     r) c.num_attention_heads     = *r; else return std::unexpected(r.error());
        if (auto r = req_u("num_key_value_heads");     r) c.num_key_value_heads     = *r; else return std::unexpected(r.error());
        if (auto r = req_u("vocab_size");              r) c.vocab_size              = *r; else return std::unexpected(r.error());
        if (auto r = req_u("max_position_embeddings"); r) c.max_position_embeddings = *r; else return std::unexpected(r.error());
        if (auto r = req_f("rope_theta");              r) c.rope_theta              = *r; else return std::unexpected(r.error());
        c.tie_word_embeddings =
            v.contains("tie_word_embeddings") && v["tie_word_embeddings"].is_boolean()
                ? v["tie_word_embeddings"].get<bool>()
                : false;
        if (v.contains("rms_norm_eps") && v["rms_norm_eps"].is_number()) {
            c.rms_norm_eps = static_cast<float>(v["rms_norm_eps"].get<double>());
        }
        return c;
    }
};

// ----- Minimal safetensors v0 reader ----------------------------------------

enum class StDtype { F32, F16, BF16, I8, U8, I32, I64, F64, Other };

struct StTensorView {
    StDtype                      dtype = StDtype::Other;
    std::vector<std::size_t>     shape;
    std::span<const std::uint8_t> data; // view into mmap'd buffer
};

class SafeTensors {
public:
    [[nodiscard]] static std::expected<SafeTensors, ConvertError>
    parse(std::span<const std::uint8_t> file)
    {
        if (file.size() < 8) {
            return std::unexpected(ConvertError{"safetensors: file < 8 bytes"});
        }
        std::uint64_t hdr_len = 0;
        std::memcpy(&hdr_len, file.data(), 8);
        if (hdr_len > file.size() - 8) {
            return std::unexpected(ConvertError{"safetensors: header overflows file"});
        }
        nlohmann::json hdr;
        try {
            hdr = nlohmann::json::parse(file.data() + 8, file.data() + 8 + hdr_len);
        } catch (const std::exception& e) {
            return std::unexpected(ConvertError{std::string{"safetensors json: "} + e.what()});
        }
        SafeTensors st;
        st.data_start_ = 8 + static_cast<std::size_t>(hdr_len);
        st.bytes_      = file;
        for (auto it = hdr.begin(); it != hdr.end(); ++it) {
            if (it.key() == "__metadata__") continue;
            const auto& obj = it.value();
            StTensorView t;
            const std::string dt = obj.at("dtype").get<std::string>();
            if      (dt == "BF16") t.dtype = StDtype::BF16;
            else if (dt == "F16")  t.dtype = StDtype::F16;
            else if (dt == "F32")  t.dtype = StDtype::F32;
            else if (dt == "I8")   t.dtype = StDtype::I8;
            else if (dt == "U8")   t.dtype = StDtype::U8;
            else if (dt == "I32")  t.dtype = StDtype::I32;
            else if (dt == "I64")  t.dtype = StDtype::I64;
            else if (dt == "F64")  t.dtype = StDtype::F64;
            else                   t.dtype = StDtype::Other;
            for (const auto& d : obj.at("shape"))
                t.shape.push_back(static_cast<std::size_t>(d.get<std::uint64_t>()));
            const auto& off = obj.at("data_offsets");
            const std::size_t a = static_cast<std::size_t>(off[0].get<std::uint64_t>());
            const std::size_t b = static_cast<std::size_t>(off[1].get<std::uint64_t>());
            if (b < a || st.data_start_ + b > file.size()) {
                return std::unexpected(ConvertError{"safetensors: tensor offsets out of range"});
            }
            t.data = std::span<const std::uint8_t>{
                file.data() + st.data_start_ + a, b - a};
            st.tensors_.emplace(it.key(), std::move(t));
        }
        return st;
    }

    [[nodiscard]] std::expected<const StTensorView*, ConvertError>
    tensor(const std::string& name) const noexcept
    {
        auto it = tensors_.find(name);
        if (it == tensors_.end())
            return std::unexpected(ConvertError{"safetensors: missing tensor " + name});
        return &it->second;
    }

    [[nodiscard]] const std::unordered_map<std::string, StTensorView>&
    map() const noexcept { return tensors_; }

private:
    std::span<const std::uint8_t>                  bytes_{};
    std::size_t                                    data_start_ = 0;
    std::unordered_map<std::string, StTensorView>  tensors_{};
};

// Decode a bf16 tensor view to fp32 vector. Caller validates dtype/shape.
[[nodiscard]] inline std::expected<std::vector<float>, ConvertError>
bf16_view_to_f32(const StTensorView& v, const std::string& name)
{
    if (v.dtype != StDtype::BF16) {
        return std::unexpected(ConvertError{
            "tensor " + name + ": expected BF16"});
    }
    const std::size_t n = v.data.size() / 2;
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint16_t bits = static_cast<std::uint16_t>(
            static_cast<std::uint16_t>(v.data[2 * i]) |
            (static_cast<std::uint16_t>(v.data[2 * i + 1]) << 8));
        out[i] = bf16_bits_to_f32(bits);
    }
    return out;
}

[[nodiscard]] inline std::vector<std::uint8_t>
f32_vec_to_le_bytes(std::span<const float> v)
{
    std::vector<std::uint8_t> b(v.size() * 4);
    for (std::size_t i = 0; i < v.size(); ++i) {
        std::memcpy(b.data() + i * 4, &v[i], 4);
    }
    return b;
}

// ----- Per-tensor reports + final ConvertStats ------------------------------

struct TensorReport {
    std::string  name;
    std::size_t  rows         = 0;
    std::size_t  cols         = 0;
    std::size_t  block_count  = 0;
    std::size_t  packed_bytes = 0;
};

struct ConvertStats {
    ModelConfig                              config{};
    std::vector<std::vector<TensorReport>>   per_layer{};
    std::uint64_t                            embedding_bytes        = 0;
    std::uint64_t                            final_norm_bytes       = 0;
    std::uint64_t                            packed_ternary_bytes   = 0;
    std::uint64_t                            output_bytes           = 0;
    std::filesystem::path                    output_path{};
    std::int32_t                             h1b_reserved_flags     = 0;
    std::vector<std::string>                 fp32_passthrough_names{};
    std::vector<std::string>                 unmatched_tensors{};
};

// ----- Top-level converter --------------------------------------------------

[[nodiscard]] inline std::expected<std::vector<std::uint8_t>, ConvertError>
read_whole_file(const std::filesystem::path& p)
{
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return std::unexpected(ConvertError{"cannot open " + p.string()});
    const std::streamsize n = f.tellg();
    if (n < 0) return std::unexpected(ConvertError{"tellg: " + p.string()});
    std::vector<std::uint8_t> buf(static_cast<std::size_t>(n));
    f.seekg(0, std::ios::beg);
    f.read(reinterpret_cast<char*>(buf.data()), n);
    if (!f) return std::unexpected(ConvertError{"read: " + p.string()});
    return buf;
}

namespace detail {

[[nodiscard]] inline std::expected<std::vector<std::uint8_t>, ConvertError>
pass_through_fp32_1d(const SafeTensors&        st,
                     const std::string&        name,
                     std::size_t               expected_len,
                     std::vector<std::string>& recorder)
{
    auto vp = st.tensor(name);
    if (!vp) return std::unexpected(vp.error());
    const StTensorView& v = **vp;
    if (v.shape.size() != 1 || v.shape[0] != expected_len) {
        return std::unexpected(ConvertError{
            "tensor " + name + ": shape mismatch"});
    }
    auto fp32 = bf16_view_to_f32(v, name);
    if (!fp32) return std::unexpected(fp32.error());
    recorder.push_back(name);
    return f32_vec_to_le_bytes(*fp32);
}

inline void put(std::vector<std::uint8_t>& out, std::span<const std::uint8_t> s)
{
    out.insert(out.end(), s.begin(), s.end());
}

template <typename T>
inline void put_le(std::vector<std::uint8_t>& out, T v)
{
    static_assert(std::is_trivially_copyable_v<T>);
    std::array<std::uint8_t, sizeof(T)> bytes{};
    std::memcpy(bytes.data(), &v, sizeof(T));
    out.insert(out.end(), bytes.begin(), bytes.end());
}

} // namespace detail

[[nodiscard]] inline std::expected<ConvertStats, ConvertError>
convert_with_mode(const std::filesystem::path& input_dir,
                  const std::filesystem::path& output_path,
                  ScaleMode                    scale_mode)
{
    const auto cfg_path = input_dir / "config.json";
    auto cfg_buf = read_whole_file(cfg_path);
    if (!cfg_buf) return std::unexpected(cfg_buf.error());
    auto cfg_or = ModelConfig::parse_json(*cfg_buf);
    if (!cfg_or) return std::unexpected(cfg_or.error());
    const ModelConfig cfg = *cfg_or;

    const auto st_path = input_dir / "model.safetensors";
    auto st_buf = read_whole_file(st_path);
    if (!st_buf) return std::unexpected(st_buf.error());
    auto st_or = SafeTensors::parse(*st_buf);
    if (!st_or) return std::unexpected(st_or.error());
    const SafeTensors st = std::move(*st_or);

    const std::int32_t reserved = H1B_FLAG_BONSAI_TQ2;
    const std::int32_t version  = 4;
    if (cfg.num_attention_heads == 0) {
        return std::unexpected(ConvertError{"num_attention_heads == 0"});
    }
    const std::size_t head_dim =
        static_cast<std::size_t>(cfg.hidden_size) / cfg.num_attention_heads;
    const std::int32_t tie_i32 = cfg.tie_word_embeddings ? 1 : 0;

    std::vector<std::uint8_t> out;
    out.reserve(64 * 1024);
    detail::put(out, std::span<const std::uint8_t>{H1B_MAGIC.data(), 4});
    detail::put_le(out, version);
    const std::int32_t cfg_arr[9] = {
        static_cast<std::int32_t>(cfg.hidden_size),
        static_cast<std::int32_t>(cfg.intermediate_size),
        static_cast<std::int32_t>(cfg.num_hidden_layers),
        static_cast<std::int32_t>(cfg.num_attention_heads),
        static_cast<std::int32_t>(cfg.num_key_value_heads),
        static_cast<std::int32_t>(cfg.vocab_size),
        static_cast<std::int32_t>(cfg.max_position_embeddings),
        tie_i32,
        reserved,
    };
    for (auto v : cfg_arr) detail::put_le(out, v);
    detail::put_le(out, cfg.rope_theta);
    detail::put_le(out, cfg.rms_norm_eps);

    std::vector<std::string> fp32_passthrough;

    // -- Embedding + final norm: bf16 → fp32 raw passthrough --
    auto emb_view = st.tensor("model.embed_tokens.weight");
    if (!emb_view) return std::unexpected(emb_view.error());
    const StTensorView& emb = **emb_view;
    if (emb.shape.size() != 2 ||
        emb.shape[0] != cfg.vocab_size ||
        emb.shape[1] != cfg.hidden_size) {
        return std::unexpected(ConvertError{"embed_tokens shape mismatch"});
    }
    auto emb_fp32 = bf16_view_to_f32(emb, "model.embed_tokens.weight");
    if (!emb_fp32) return std::unexpected(emb_fp32.error());
    auto emb_bytes = f32_vec_to_le_bytes(*emb_fp32);
    detail::put(out, emb_bytes);
    fp32_passthrough.emplace_back("model.embed_tokens.weight");
    const std::uint64_t embedding_bytes = emb_bytes.size();

    auto fnorm_view = st.tensor("model.norm.weight");
    if (!fnorm_view) return std::unexpected(fnorm_view.error());
    const StTensorView& fnorm = **fnorm_view;
    if (fnorm.shape.size() != 1 || fnorm.shape[0] != cfg.hidden_size) {
        return std::unexpected(ConvertError{"model.norm shape mismatch"});
    }
    auto fnorm_fp32 = bf16_view_to_f32(fnorm, "model.norm.weight");
    if (!fnorm_fp32) return std::unexpected(fnorm_fp32.error());
    auto fnorm_bytes = f32_vec_to_le_bytes(*fnorm_fp32);
    detail::put(out, fnorm_bytes);
    fp32_passthrough.emplace_back("model.norm.weight");
    const std::uint64_t final_norm_bytes = fnorm_bytes.size();

    // -- Per-layer payload --
    std::vector<std::vector<TensorReport>> per_layer_reports;
    per_layer_reports.reserve(cfg.num_hidden_layers);
    std::uint64_t packed_ternary_bytes = 0;
    for (std::uint32_t l = 0; l < cfg.num_hidden_layers; ++l) {
        auto in_norm = detail::pass_through_fp32_1d(
            st, "model.layers." + std::to_string(l) + ".input_layernorm.weight",
            cfg.hidden_size, fp32_passthrough);
        if (!in_norm) return std::unexpected(in_norm.error());
        auto post_norm = detail::pass_through_fp32_1d(
            st, "model.layers." + std::to_string(l) + ".post_attention_layernorm.weight",
            cfg.hidden_size, fp32_passthrough);
        if (!post_norm) return std::unexpected(post_norm.error());
        auto attn_sub = detail::pass_through_fp32_1d(
            st, "model.layers." + std::to_string(l) + ".self_attn.attn_sub_norm.weight",
            cfg.hidden_size, fp32_passthrough);
        if (!attn_sub) return std::unexpected(attn_sub.error());
        auto ffn_sub = detail::pass_through_fp32_1d(
            st, "model.layers." + std::to_string(l) + ".mlp.ffn_sub_norm.weight",
            cfg.intermediate_size, fp32_passthrough);
        if (!ffn_sub) return std::unexpected(ffn_sub.error());

        // Norm block: hs * (1+1+4+2) + is*1 — matches onebit_core::h1b::serialize.
        detail::put(out, *in_norm);
        detail::put(out, *post_norm);
        for (int i = 0; i < 4; ++i) detail::put(out, *attn_sub);
        const std::size_t trunc_bytes = static_cast<std::size_t>(cfg.hidden_size) * 4;
        detail::put(out, std::span<const std::uint8_t>{ffn_sub->data(), trunc_bytes});
        detail::put(out, std::span<const std::uint8_t>{ffn_sub->data(), trunc_bytes});
        detail::put(out, *ffn_sub);

        // Seven ternary tensors.
        struct Slot { const char* hf_infix; std::size_t rows; std::size_t cols; };
        const std::size_t hs  = cfg.hidden_size;
        const std::size_t is_ = cfg.intermediate_size;
        const std::size_t nh  = cfg.num_attention_heads;
        const std::size_t nkv = cfg.num_key_value_heads;
        const Slot slots[7] = {
            {"self_attn.q_proj",  nh  * head_dim, hs},
            {"self_attn.k_proj",  nkv * head_dim, hs},
            {"self_attn.v_proj",  nkv * head_dim, hs},
            {"self_attn.o_proj",  hs,             nh * head_dim},
            {"mlp.gate_proj",     is_,            hs},
            {"mlp.up_proj",       is_,            hs},
            {"mlp.down_proj",     hs,             is_},
        };
        std::vector<TensorReport> layer_reports;
        layer_reports.reserve(7);
        for (const Slot& s : slots) {
            const std::string name =
                "model.layers." + std::to_string(l) + "." + s.hf_infix + ".weight";
            auto tv = st.tensor(name);
            if (!tv) return std::unexpected(tv.error());
            const StTensorView& view = **tv;
            if (view.shape.size() != 2 ||
                view.shape[0] != s.rows ||
                view.shape[1] != s.cols) {
                return std::unexpected(ConvertError{
                    "tensor " + name + ": shape mismatch"});
            }
            if (s.cols % TQ2_GROUP_SIZE != 0) {
                return std::unexpected(ConvertError{
                    "tensor " + name + ": cols not multiple of "
                    + std::to_string(TQ2_GROUP_SIZE)});
            }
            auto fp32_or = bf16_view_to_f32(view, name);
            if (!fp32_or) return std::unexpected(fp32_or.error());
            const std::vector<float>& fp32 = *fp32_or;

            float s_tensor = 0.0f;
            if (scale_mode == ScaleMode::PerTensor) {
                double sum = 0.0;
                for (float x : fp32) sum += static_cast<double>(x < 0 ? -x : x);
                if (!fp32.empty()) {
                    s_tensor = static_cast<float>(sum / static_cast<double>(fp32.size()));
                }
            }

            const std::size_t n_blocks_per_row = s.cols / TQ2_GROUP_SIZE;
            const std::size_t row_packed      = n_blocks_per_row * TQ2_BLOCK_BYTES;
            std::vector<std::uint8_t> packed(s.rows * row_packed, 0);
            for (std::size_t r = 0; r < s.rows; ++r) {
                for (std::size_t b = 0; b < n_blocks_per_row; ++b) {
                    const std::size_t base = r * s.cols + b * TQ2_GROUP_SIZE;
                    std::array<float, TQ2_GROUP_SIZE> grp{};
                    std::memcpy(grp.data(), fp32.data() + base,
                                TQ2_GROUP_SIZE * sizeof(float));
                    std::array<std::uint8_t, TQ2_BLOCK_BYTES> blk;
                    if (scale_mode == ScaleMode::PerTensor)
                        blk = quantize_group_with_scale(grp, s_tensor);
                    else
                        blk = quantize_group_tq2(grp);
                    std::memcpy(packed.data() + r * row_packed + b * TQ2_BLOCK_BYTES,
                                blk.data(), TQ2_BLOCK_BYTES);
                }
            }
            detail::put(out, packed);
            packed_ternary_bytes += packed.size();
            layer_reports.push_back(TensorReport{
                name, s.rows, s.cols, s.rows * n_blocks_per_row, packed.size()});
        }
        per_layer_reports.push_back(std::move(layer_reports));
    }

    // Persist.
    {
        std::ofstream f(output_path, std::ios::binary | std::ios::trunc);
        if (!f) return std::unexpected(ConvertError{"cannot open " + output_path.string()});
        f.write(reinterpret_cast<const char*>(out.data()),
                static_cast<std::streamsize>(out.size()));
        if (!f) return std::unexpected(ConvertError{"short write " + output_path.string()});
    }

    // Compute "unmatched" set (diagnostic only).
    std::unordered_map<std::string, bool> consumed;
    consumed["model.embed_tokens.weight"] = true;
    consumed["model.norm.weight"]         = true;
    for (const auto& n : fp32_passthrough) consumed[n] = true;
    for (std::uint32_t l = 0; l < cfg.num_hidden_layers; ++l) {
        for (const char* tail : {"self_attn.q_proj", "self_attn.k_proj",
                                 "self_attn.v_proj", "self_attn.o_proj",
                                 "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}) {
            consumed[std::string{"model.layers."} + std::to_string(l)
                     + "." + tail + ".weight"] = true;
        }
    }
    std::vector<std::string> unmatched;
    for (const auto& [k, _] : st.map()) {
        if (consumed.find(k) == consumed.end()) unmatched.push_back(k);
    }
    std::sort(unmatched.begin(), unmatched.end());

    ConvertStats stats;
    stats.config                  = cfg;
    stats.per_layer               = std::move(per_layer_reports);
    stats.embedding_bytes         = embedding_bytes;
    stats.final_norm_bytes        = final_norm_bytes;
    stats.packed_ternary_bytes    = packed_ternary_bytes;
    stats.output_bytes            = out.size();
    stats.output_path             = output_path;
    stats.h1b_reserved_flags      = reserved;
    stats.fp32_passthrough_names  = std::move(fp32_passthrough);
    stats.unmatched_tensors       = std::move(unmatched);
    return stats;
}

[[nodiscard]] inline std::expected<ConvertStats, ConvertError>
convert(const std::filesystem::path& input_dir,
        const std::filesystem::path& output_path)
{
    return convert_with_mode(input_dir, output_path, ScaleMode::PerTensor);
}

// Read back the reserved flags word from a freshly-written .h1b. Mirrors
// the Rust helper used by the integration tests.
[[nodiscard]] inline std::expected<std::int32_t, ConvertError>
read_reserved_flags(const std::filesystem::path& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::unexpected(ConvertError{"cannot open " + path.string()});
    std::array<std::uint8_t, 48> buf{};
    f.read(reinterpret_cast<char*>(buf.data()), buf.size());
    if (!f) return std::unexpected(ConvertError{"short read " + path.string()});
    if (std::memcmp(buf.data(), H1B_MAGIC.data(), 4) != 0) {
        return std::unexpected(ConvertError{"not a .h1b file"});
    }
    // magic(4) + version(4) + cfg[0..9]; reserved is cfg[8] → offset 4+4+8*4 = 40.
    std::int32_t v = 0;
    std::memcpy(&v, buf.data() + 40, 4);
    return v;
}

} // namespace onebit::tools::bitnet_to_tq2

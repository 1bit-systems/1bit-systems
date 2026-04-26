// gguf-to-h1b — header-only converter + helpers.
//
// Faithful C++23 port of `tools/gguf-to-h1b/src/{lib,htok_export}.rs`. Frames
// a PrismML Bonsai GGUF (`Q1_0_g128` dtype 41 / `TQ2_0_g128` dtype 42) into a
// halo `.h1b` v2 with the appropriate `H1B_FLAG_BONSAI_*` bit set.
//
// Reuses `onebit::core::gguf::GgufFile` for the heavy lifting (KV parse,
// tensor directory). We only add:
//   * Bonsai dtype recognition (41/42 — not in core's `TensorType` enum).
//   * Streaming `.h1b` writer — the streaming layout matches the Rust
//     `onebit-core::h1b::serialize` exactly.
//   * `.htok` sidecar emission from the GGUF tokenizer KV arrays.

#pragma once

#include "onebit/core/gguf.hpp"
#include "onebit/core/htok.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace onebit::tools::gguf_to_h1b {

// ----- Constants -------------------------------------------------------------

inline constexpr std::array<std::uint8_t, 4> H1B_MAGIC = {'H', '1', 'B', 0};
inline constexpr std::int32_t H1B_FLAG_BONSAI_TQ2 = 0x4;
inline constexpr std::int32_t H1B_FLAG_BONSAI_Q1  = 0x8;
inline constexpr std::array<std::uint8_t, 4> HTOK_MAGIC = {'H', 'T', 'O', 'K'};
inline constexpr std::size_t BONSAI_GROUP_SIZE = 128;

// ----- Error type ------------------------------------------------------------

struct ConvertError {
    std::string what;
};

// ----- Bonsai dtype tag ------------------------------------------------------

enum class BonsaiDtype : std::uint32_t {
    Q1G128  = 41,
    TQ2G128 = 42,
};

[[nodiscard]] constexpr std::optional<BonsaiDtype> bonsai_from_u32(std::uint32_t v) noexcept
{
    if (v == 41) return BonsaiDtype::Q1G128;
    if (v == 42) return BonsaiDtype::TQ2G128;
    return std::nullopt;
}

[[nodiscard]] constexpr std::size_t bonsai_block_bytes(BonsaiDtype d) noexcept
{
    return d == BonsaiDtype::Q1G128 ? 18 : 34;
}

[[nodiscard]] constexpr std::int32_t bonsai_h1b_flag(BonsaiDtype d) noexcept
{
    return d == BonsaiDtype::Q1G128 ? H1B_FLAG_BONSAI_Q1 : H1B_FLAG_BONSAI_TQ2;
}

// ----- BitnetHeader-equivalent extracted from a Bonsai GGUF ------------------

struct BonsaiHeader {
    std::string                  architecture;
    std::uint32_t                block_count               = 0;
    std::uint32_t                embedding_length          = 0;
    std::uint32_t                feed_forward_length       = 0;
    std::uint32_t                attention_head_count      = 0;
    std::uint32_t                attention_head_count_kv   = 0;
    float                        rope_freq_base            = 1e4f;
    float                        rms_norm_eps              = 1e-6f;
    std::string                  tokenizer_model;
    std::vector<std::string>     tokens;
    std::vector<std::string>     merges;
    std::optional<std::int64_t>  bos_token_id;
    std::optional<std::int64_t>  eos_token_id;
};

namespace detail {

template <typename T>
[[nodiscard]] inline std::optional<T>
get_md(const std::unordered_map<std::string, onebit::core::gguf::Value>& md,
       const std::string& key)
{
    auto it = md.find(key);
    if (it == md.end()) return std::nullopt;
    if (auto* p = std::get_if<T>(&it->second)) return *p;
    return std::nullopt;
}

[[nodiscard]] inline std::optional<std::uint32_t>
get_u32(const std::unordered_map<std::string, onebit::core::gguf::Value>& md,
        const std::string& key)
{
    if (auto v = get_md<std::uint32_t>(md, key)) return *v;
    if (auto v = get_md<std::int32_t>(md, key))  return static_cast<std::uint32_t>(*v);
    if (auto v = get_md<std::uint64_t>(md, key)) return static_cast<std::uint32_t>(*v);
    if (auto v = get_md<std::int64_t>(md, key))  return static_cast<std::uint32_t>(*v);
    return std::nullopt;
}

[[nodiscard]] inline std::optional<float>
get_f32(const std::unordered_map<std::string, onebit::core::gguf::Value>& md,
        const std::string& key)
{
    if (auto v = get_md<float>(md, key))  return *v;
    if (auto v = get_md<double>(md, key)) return static_cast<float>(*v);
    return std::nullopt;
}

[[nodiscard]] inline std::optional<std::int64_t>
get_i64(const std::unordered_map<std::string, onebit::core::gguf::Value>& md,
        const std::string& key)
{
    if (auto v = get_md<std::int64_t>(md, key))  return *v;
    if (auto v = get_md<std::uint64_t>(md, key)) return static_cast<std::int64_t>(*v);
    if (auto v = get_md<std::int32_t>(md, key))  return static_cast<std::int64_t>(*v);
    if (auto v = get_md<std::uint32_t>(md, key)) return static_cast<std::int64_t>(*v);
    return std::nullopt;
}

[[nodiscard]] inline std::optional<std::vector<std::string>>
get_string_array(const std::unordered_map<std::string, onebit::core::gguf::Value>& md,
                 const std::string& key)
{
    auto it = md.find(key);
    if (it == md.end()) return std::nullopt;
    auto* arr_p = std::get_if<std::shared_ptr<onebit::core::gguf::Array>>(&it->second);
    if (!arr_p || !*arr_p) return std::nullopt;
    const auto& arr = **arr_p;
    if (arr.element_type != onebit::core::gguf::ValueType::String) return std::nullopt;
    std::vector<std::string> out;
    out.reserve(arr.items.size());
    for (const auto& v : arr.items) {
        if (auto* s = std::get_if<std::string>(&v)) out.push_back(*s);
    }
    return out;
}

template <typename T>
inline void put_le(std::vector<std::uint8_t>& out, T v)
{
    static_assert(std::is_trivially_copyable_v<T>);
    std::array<std::uint8_t, sizeof(T)> bytes{};
    std::memcpy(bytes.data(), &v, sizeof(T));
    out.insert(out.end(), bytes.begin(), bytes.end());
}

inline void write_zeros_to_file(std::ofstream& f, std::uint64_t count)
{
    static constexpr std::array<std::uint8_t, 64 * 1024> chunk{};
    while (count > 0) {
        const std::size_t n = static_cast<std::size_t>(
            std::min<std::uint64_t>(count, chunk.size()));
        f.write(reinterpret_cast<const char*>(chunk.data()),
                static_cast<std::streamsize>(n));
        count -= n;
    }
}

} // namespace detail

[[nodiscard]] inline std::expected<BonsaiHeader, ConvertError>
read_bonsai_header(const onebit::core::gguf::GgufFile& gguf)
{
    BonsaiHeader h;
    const auto& md = gguf.metadata();

    if (auto a = detail::get_md<std::string>(md, "general.architecture"))
        h.architecture = *a;
    if (auto v = detail::get_u32(md, "qwen3.block_count"))
        h.block_count = *v;
    if (auto v = detail::get_u32(md, "qwen3.embedding_length"))
        h.embedding_length = *v;
    if (auto v = detail::get_u32(md, "qwen3.feed_forward_length"))
        h.feed_forward_length = *v;
    if (auto v = detail::get_u32(md, "qwen3.attention.head_count"))
        h.attention_head_count = *v;
    if (auto v = detail::get_u32(md, "qwen3.attention.head_count_kv"))
        h.attention_head_count_kv = *v;
    if (auto v = detail::get_f32(md, "qwen3.rope.freq_base"))
        h.rope_freq_base = *v;
    if (auto v = detail::get_f32(md, "qwen3.attention.layer_norm_rms_epsilon"))
        h.rms_norm_eps = *v;

    if (auto m = detail::get_md<std::string>(md, "tokenizer.ggml.model"))
        h.tokenizer_model = *m;
    if (auto t = detail::get_string_array(md, "tokenizer.ggml.tokens"))
        h.tokens = std::move(*t);
    if (auto m = detail::get_string_array(md, "tokenizer.ggml.merges"))
        h.merges = std::move(*m);
    h.bos_token_id = detail::get_i64(md, "tokenizer.ggml.bos_token_id");
    h.eos_token_id = detail::get_i64(md, "tokenizer.ggml.eos_token_id");

    return h;
}

// ----- Convert stats ---------------------------------------------------------

struct ConvertStats {
    BonsaiDtype           dtype           = BonsaiDtype::TQ2G128;
    std::uint32_t         hidden_size       = 0;
    std::uint32_t         intermediate_size = 0;
    std::uint32_t         num_layers        = 0;
    std::uint32_t         num_heads         = 0;
    std::uint32_t         num_kv_heads      = 0;
    std::uint32_t         head_dim          = 0;
    std::uint32_t         vocab_size        = 0;
    std::uint32_t         context_length    = 0;
    float                 rope_theta        = 1e4f;
    float                 rms_norm_eps      = 1e-6f;
    std::uint64_t         ternary_bytes_carried = 0;
    std::uint64_t         output_bytes      = 0;
    std::filesystem::path output_path{};
    std::int32_t          h1b_reserved_flags = 0;
};

struct HtokStats {
    std::uint32_t         vocab_size      = 0;
    std::uint32_t         num_merges      = 0;
    std::int32_t          bos_id          = 0;
    std::int32_t          eos_id          = 0;
    std::uint64_t         output_bytes    = 0;
    std::filesystem::path output_path{};
    std::uint32_t         dropped_merges  = 0;
};

// ----- Per-layer ternary tensor names ---------------------------------------

inline const std::array<const char*, 7> PER_LAYER_TERNARY = {
    "attn_q", "attn_k", "attn_v", "attn_output",
    "ffn_gate", "ffn_up", "ffn_down",
};

// ----- Serialize an .htok blob ---------------------------------------------
//
// Byte layout (Rust `HtokFile::to_bytes` — what the rocm-cpp tokenizer.cpp
// loader actually consumes today):
//
//   "HTOK" + u32 vocab + u32 nmerges + u32 bos + u32 eos
//   then vocab × (u16 len + bytes)
//   then nmerges × (u32 a_id, u32 b_id, u32 merged_id)
//
// The C++ `onebit::core::htok::File::parse` in cpp/core/src/htok.cpp is a
// DIFFERENT format with its own header shape (i32-based + pad_id + 8 B
// reserved). The runtime today reads the Rust format, so this writer
// emits that format for cross-stack compatibility.

struct MergeTriple {
    std::uint32_t a_id;
    std::uint32_t b_id;
    std::uint32_t merged_id;
};

[[nodiscard]] inline std::vector<std::uint8_t>
serialize_htok(std::uint32_t bos, std::uint32_t eos,
               const std::vector<std::vector<std::uint8_t>>& pieces,
               const std::vector<MergeTriple>& merges)
{
    std::vector<std::uint8_t> out;
    out.insert(out.end(), HTOK_MAGIC.begin(), HTOK_MAGIC.end());
    detail::put_le<std::uint32_t>(out, static_cast<std::uint32_t>(pieces.size()));
    detail::put_le<std::uint32_t>(out, static_cast<std::uint32_t>(merges.size()));
    detail::put_le<std::uint32_t>(out, bos);
    detail::put_le<std::uint32_t>(out, eos);
    for (const auto& p : pieces) {
        detail::put_le<std::uint16_t>(out, static_cast<std::uint16_t>(p.size()));
        out.insert(out.end(), p.begin(), p.end());
    }
    for (const auto& m : merges) {
        detail::put_le<std::uint32_t>(out, m.a_id);
        detail::put_le<std::uint32_t>(out, m.b_id);
        detail::put_le<std::uint32_t>(out, m.merged_id);
    }
    return out;
}

// Build pieces + merges from a Bonsai header's tokenizer arrays.
//
// Rust port semantics:
//   * Surface form → id map built from `tokens[]`.
//   * Each merge string "A B" splits on the single space; A_id, B_id, merged_id
//     are looked up in the surface map.
//   * Merges with any unresolved side are dropped (incremented in `dropped`).
//   * The C++ htok writer here doesn't carry a numeric `(a_id, b_id, merged_id)`
//     triple; the in-tree htok parser is shape-agnostic and just stores raw
//     piece bytes. That matches what BPE encode/decode actually need.
[[nodiscard]] inline std::expected<std::pair<HtokStats, std::vector<std::uint8_t>>,
                                    ConvertError>
build_htok_blob(const BonsaiHeader& hdr)
{
    if (hdr.tokens.empty()) {
        return std::unexpected(ConvertError{
            "GGUF contains no tokenizer.ggml.tokens — cannot emit .htok"});
    }
    std::vector<std::vector<std::uint8_t>> pieces;
    pieces.reserve(hdr.tokens.size());
    std::unordered_map<std::string, std::int32_t> surface_to_id;
    surface_to_id.reserve(hdr.tokens.size());
    for (std::size_t i = 0; i < hdr.tokens.size(); ++i) {
        const std::string& s = hdr.tokens[i];
        pieces.emplace_back(s.begin(), s.end());
        surface_to_id.emplace(s, static_cast<std::int32_t>(i));
    }
    std::vector<MergeTriple> merges;
    merges.reserve(hdr.merges.size());
    std::uint32_t dropped = 0;
    for (std::size_t rank = 0; rank < hdr.merges.size(); ++rank) {
        const std::string& m = hdr.merges[rank];
        const auto sp = m.find(' ');
        if (sp == std::string::npos || sp == 0 || sp == m.size() - 1) {
            return std::unexpected(ConvertError{
                "merge " + std::to_string(rank) + " `" + m
                + "` does not contain exactly one space separator"});
        }
        const std::string a = m.substr(0, sp);
        const std::string b = m.substr(sp + 1);
        if (b.find(' ') != std::string::npos) {
            return std::unexpected(ConvertError{
                "merge " + std::to_string(rank) + " `" + m
                + "` contains multiple spaces"});
        }
        const std::string merged = a + b;
        auto a_it = surface_to_id.find(a);
        auto b_it = surface_to_id.find(b);
        auto m_it = surface_to_id.find(merged);
        if (a_it == surface_to_id.end() || b_it == surface_to_id.end()
            || m_it == surface_to_id.end()) {
            ++dropped;
            continue;
        }
        merges.push_back(MergeTriple{
            static_cast<std::uint32_t>(a_it->second),
            static_cast<std::uint32_t>(b_it->second),
            static_cast<std::uint32_t>(m_it->second)});
    }
    const std::uint32_t bos = hdr.bos_token_id ? static_cast<std::uint32_t>(*hdr.bos_token_id) : 0u;
    const std::uint32_t eos = hdr.eos_token_id ? static_cast<std::uint32_t>(*hdr.eos_token_id) : 0u;

    auto blob = serialize_htok(bos, eos, pieces, merges);
    HtokStats stats;
    stats.vocab_size      = static_cast<std::uint32_t>(pieces.size());
    stats.num_merges      = static_cast<std::uint32_t>(merges.size());
    stats.bos_id          = static_cast<std::int32_t>(bos);
    stats.eos_id          = static_cast<std::int32_t>(eos);
    stats.output_bytes    = blob.size();
    stats.dropped_merges  = dropped;
    return std::make_pair(std::move(stats), std::move(blob));
}

// ----- htok sidecar export --------------------------------------------------

[[nodiscard]] inline std::expected<HtokStats, ConvertError>
export_htok_sidecar(const std::filesystem::path& gguf_path,
                    const std::filesystem::path& out_path)
{
    auto g = onebit::core::gguf::GgufFile::open(gguf_path);
    if (!g) return std::unexpected(ConvertError{"failed to open GGUF"});
    auto hdr_or = read_bonsai_header(*g);
    if (!hdr_or) return std::unexpected(hdr_or.error());
    auto built = build_htok_blob(*hdr_or);
    if (!built) return std::unexpected(built.error());
    auto& [stats, blob] = *built;
    std::ofstream f(out_path, std::ios::binary | std::ios::trunc);
    if (!f) return std::unexpected(ConvertError{"cannot open " + out_path.string()});
    f.write(reinterpret_cast<const char*>(blob.data()),
            static_cast<std::streamsize>(blob.size()));
    if (!f) return std::unexpected(ConvertError{"short write " + out_path.string()});
    stats.output_path = out_path;
    return stats;
}

// ----- Top-level Bonsai → .h1b framing converter ----------------------------

[[nodiscard]] inline std::expected<ConvertStats, ConvertError>
convert_file(const std::filesystem::path& input,
             const std::filesystem::path& output)
{
    auto g_or = onebit::core::gguf::GgufFile::open(input);
    if (!g_or) return std::unexpected(ConvertError{"failed to open GGUF"});
    const auto& g = *g_or;
    auto hdr_or = read_bonsai_header(g);
    if (!hdr_or) return std::unexpected(hdr_or.error());
    const BonsaiHeader hdr = *hdr_or;
    if (hdr.architecture != "qwen3") {
        return std::unexpected(ConvertError{
            "input is not a Bonsai GGUF: architecture = " + hdr.architecture});
    }
    const std::uint32_t n_layers = hdr.block_count;
    const std::uint32_t hs       = hdr.embedding_length;
    const std::uint32_t is_      = hdr.feed_forward_length;
    const std::uint32_t nh       = hdr.attention_head_count;
    const std::uint32_t nkv      = hdr.attention_head_count_kv;
    const std::uint32_t hd =
        detail::get_u32(g.metadata(), "qwen3.attention.key_length")
            .value_or(nh != 0 ? hs / nh : 0);
    const std::uint32_t vocab =
        detail::get_u32(g.metadata(), "qwen3.vocab_size")
            .value_or(static_cast<std::uint32_t>(hdr.tokens.size()));
    const std::uint32_t ctx_len =
        detail::get_u32(g.metadata(), "qwen3.context_length").value_or(0);

    // Build a fast tensor name → info lookup.
    std::unordered_map<std::string, const onebit::core::gguf::TensorInfo*> by_name;
    for (const auto& t : g.tensors()) by_name.emplace(t.name, &t);

    // Detect Q1 vs TQ2 by scanning all per-layer ternary tensors.
    std::optional<BonsaiDtype> detected;
    for (std::uint32_t l = 0; l < n_layers; ++l) {
        for (const char* tail : PER_LAYER_TERNARY) {
            const std::string name =
                "blk." + std::to_string(l) + "." + tail + ".weight";
            auto it = by_name.find(name);
            if (it == by_name.end()) {
                return std::unexpected(ConvertError{"missing tensor " + name});
            }
            const std::uint32_t dt =
                static_cast<std::uint32_t>(it->second->type);
            auto bd = bonsai_from_u32(dt);
            if (!bd) {
                return std::unexpected(ConvertError{
                    "tensor " + name + " dtype=" + std::to_string(dt)
                    + " is not Bonsai (41=Q1_0_g128 or 42=TQ2_0_g128)"});
            }
            if (detected && *detected != *bd) {
                return std::unexpected(ConvertError{
                    "mixed Bonsai dtypes — both Q1 and TQ2 present"});
            }
            detected = bd;
            if (it->second->dims.size() != 2) {
                return std::unexpected(ConvertError{
                    "tensor " + name + " has !=2 dims"});
            }
            const std::uint64_t cols = it->second->dims[0];
            if (cols % BONSAI_GROUP_SIZE != 0) {
                return std::unexpected(ConvertError{
                    "tensor " + name + " cols=" + std::to_string(cols)
                    + " not multiple of " + std::to_string(BONSAI_GROUP_SIZE)});
            }
        }
    }
    if (!detected) {
        return std::unexpected(ConvertError{
            "no ternary tensors found in input GGUF"});
    }
    const BonsaiDtype dtype = *detected;
    const std::int32_t reserved = bonsai_h1b_flag(dtype);

    // Open output and stream into it.
    std::ofstream f(output, std::ios::binary | std::ios::trunc);
    if (!f) return std::unexpected(ConvertError{"cannot open " + output.string()});

    std::uint64_t output_bytes = 0;
    {
        std::vector<std::uint8_t> hdr_buf;
        hdr_buf.reserve(64);
        hdr_buf.insert(hdr_buf.end(), H1B_MAGIC.begin(), H1B_MAGIC.end());
        const std::int32_t version = 2; // v2 carries rope/eps; Bonsai bit overrides format
        detail::put_le<std::int32_t>(hdr_buf, version);
        const std::int32_t cfg9[9] = {
            static_cast<std::int32_t>(hs),
            static_cast<std::int32_t>(is_),
            static_cast<std::int32_t>(n_layers),
            static_cast<std::int32_t>(nh),
            static_cast<std::int32_t>(nkv),
            static_cast<std::int32_t>(vocab),
            static_cast<std::int32_t>(ctx_len),
            0, // tie_embeddings — unused in Bonsai framing
            reserved,
        };
        for (auto v : cfg9) detail::put_le<std::int32_t>(hdr_buf, v);
        detail::put_le<float>(hdr_buf, hdr.rope_freq_base);
        detail::put_le<float>(hdr_buf, hdr.rms_norm_eps);
        f.write(reinterpret_cast<const char*>(hdr_buf.data()),
                static_cast<std::streamsize>(hdr_buf.size()));
        output_bytes += hdr_buf.size();
    }

    // Embedding + final_norm placeholders (zeroed — runtime reads real
    // tensors from the GGUF directly when the loader sees the Bonsai bit).
    const std::uint64_t embedding_bytes  = static_cast<std::uint64_t>(vocab) * hs * 4u;
    const std::uint64_t final_norm_bytes = static_cast<std::uint64_t>(hs) * 4u;
    detail::write_zeros_to_file(f, embedding_bytes);
    detail::write_zeros_to_file(f, final_norm_bytes);
    output_bytes += embedding_bytes + final_norm_bytes;

    // Per-layer.
    std::uint64_t ternary_bytes_carried = 0;
    for (std::uint32_t l = 0; l < n_layers; ++l) {
        // Norm slots — zeros (runtime looks them up by name from GGUF).
        detail::write_zeros_to_file(f, static_cast<std::uint64_t>(hs) * 4u); // input_norm
        detail::write_zeros_to_file(f, static_cast<std::uint64_t>(hs) * 4u); // post_attn_norm
        for (int i = 0; i < 4; ++i) {
            detail::write_zeros_to_file(f, static_cast<std::uint64_t>(hs) * 4u);
        }
        for (int i = 0; i < 2; ++i) {
            detail::write_zeros_to_file(f, static_cast<std::uint64_t>(hs) * 4u);
        }
        detail::write_zeros_to_file(f, static_cast<std::uint64_t>(is_) * 4u); // ffn_sub_norm
        output_bytes +=
            static_cast<std::uint64_t>(hs) * 4u * (1 + 1 + 4 + 2)
            + static_cast<std::uint64_t>(is_) * 4u;

        for (const char* tail : PER_LAYER_TERNARY) {
            const std::string name =
                "blk." + std::to_string(l) + "." + tail + ".weight";
            const auto* info = by_name.at(name);
            if (info->dims.size() != 2) {
                return std::unexpected(ConvertError{name + " bad ndim"});
            }
            const std::uint64_t cols = info->dims[0];
            const std::uint64_t rows = info->dims[1];
            if (cols % BONSAI_GROUP_SIZE != 0) {
                return std::unexpected(ConvertError{name + " cols not g128"});
            }
            const std::uint64_t expected = rows
                * (cols / BONSAI_GROUP_SIZE)
                * static_cast<std::uint64_t>(bonsai_block_bytes(dtype));
            auto v = g.tensor_bytes(*info);
            if (!v) {
                return std::unexpected(ConvertError{
                    name + ": tensor_bytes failed"});
            }
            const std::span<const std::uint8_t> all_after = *v;
            if (all_after.size() < expected) {
                return std::unexpected(ConvertError{
                    name + ": payload short (got "
                    + std::to_string(all_after.size())
                    + " want " + std::to_string(expected) + ")"});
            }
            f.write(reinterpret_cast<const char*>(all_after.data()),
                    static_cast<std::streamsize>(expected));
            output_bytes += expected;
            ternary_bytes_carried += expected;
            // Bonsai dtypes embed scales inline — no trailing scale bytes.
        }
    }
    f.flush();

    ConvertStats stats;
    stats.dtype                 = dtype;
    stats.hidden_size           = hs;
    stats.intermediate_size     = is_;
    stats.num_layers            = n_layers;
    stats.num_heads             = nh;
    stats.num_kv_heads          = nkv;
    stats.head_dim              = hd;
    stats.vocab_size            = vocab;
    stats.context_length        = ctx_len;
    stats.rope_theta            = hdr.rope_freq_base;
    stats.rms_norm_eps          = hdr.rms_norm_eps;
    stats.ternary_bytes_carried = ternary_bytes_carried;
    stats.output_bytes          = output_bytes;
    stats.output_path           = output;
    stats.h1b_reserved_flags    = reserved;
    return stats;
}

// Cheap check — read first 4 bytes of a file to confirm the GGUF magic.
[[nodiscard]] inline std::expected<bool, ConvertError>
is_gguf_magic(const std::filesystem::path& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::unexpected(ConvertError{"cannot open " + path.string()});
    std::array<std::uint8_t, 4> m{};
    f.read(reinterpret_cast<char*>(m.data()), 4);
    if (!f) return std::unexpected(ConvertError{"short read " + path.string()});
    return m == std::array<std::uint8_t, 4>{'G', 'G', 'U', 'F'};
}

[[nodiscard]] inline std::expected<std::uint32_t, ConvertError>
read_gguf_version(const std::filesystem::path& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::unexpected(ConvertError{"cannot open " + path.string()});
    std::array<std::uint8_t, 8> b{};
    f.read(reinterpret_cast<char*>(b.data()), 8);
    if (!f) return std::unexpected(ConvertError{"short read " + path.string()});
    if (b[0] != 'G' || b[1] != 'G' || b[2] != 'U' || b[3] != 'F') return 0u;
    std::uint32_t v = 0;
    std::memcpy(&v, b.data() + 4, 4);
    return v;
}

} // namespace onebit::tools::gguf_to_h1b

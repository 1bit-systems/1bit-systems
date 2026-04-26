#pragma once

// Minimal CBOR (RFC 8949) writer + reader. Supports just the types we
// emit/consume in `.1bl` manifests: bool, null, uint, neg-int, text
// string, byte string, array, map. No tags, no float subtypes other
// than CBOR float64. No indefinite-length items.
//
// This is intentionally small. The original Rust code uses `ciborium`
// + `serde` reflection — we don't need that flexibility, only round-trip
// for our schema.

#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace onebit::ingest::detail::cbor {

class Value;

using Array  = std::vector<Value>;
using Object = std::vector<std::pair<std::string, Value>>;

class Value {
public:
    using Variant = std::variant<
        std::monostate,         // null
        bool,                   // bool
        std::int64_t,           // int (uint or neg-int)
        std::uint64_t,          // big positive uint
        double,                 // float64
        std::string,            // text string
        std::vector<std::uint8_t>, // byte string
        Array,
        Object>;

    Value() = default;
    Value(std::monostate) {}
    Value(bool b) : v_{b} {}
    Value(int v) : v_{static_cast<std::int64_t>(v)} {}
    Value(std::int64_t v) : v_{v} {}
    Value(std::uint64_t v) : v_{v} {}
    Value(double v) : v_{v} {}
    Value(const char* s) : v_{std::string{s}} {}
    Value(std::string s) : v_{std::move(s)} {}
    Value(std::string_view s) : v_{std::string{s}} {}
    Value(std::vector<std::uint8_t> b) : v_{std::move(b)} {}
    Value(Array a) : v_{std::move(a)} {}
    Value(Object o) : v_{std::move(o)} {}

    [[nodiscard]] bool is_null() const noexcept
    {
        return std::holds_alternative<std::monostate>(v_);
    }
    [[nodiscard]] bool is_bool() const noexcept { return std::holds_alternative<bool>(v_); }
    [[nodiscard]] bool is_text() const noexcept { return std::holds_alternative<std::string>(v_); }
    [[nodiscard]] bool is_array() const noexcept { return std::holds_alternative<Array>(v_); }
    [[nodiscard]] bool is_object() const noexcept { return std::holds_alternative<Object>(v_); }
    [[nodiscard]] bool is_int() const noexcept
    {
        return std::holds_alternative<std::int64_t>(v_) ||
               std::holds_alternative<std::uint64_t>(v_);
    }

    [[nodiscard]] const Variant& variant() const noexcept { return v_; }

    [[nodiscard]] const std::string&             as_text() const { return std::get<std::string>(v_); }
    [[nodiscard]] const Array&                   as_array() const { return std::get<Array>(v_); }
    [[nodiscard]] const Object&                  as_object() const { return std::get<Object>(v_); }
    [[nodiscard]] bool                           as_bool() const { return std::get<bool>(v_); }
    [[nodiscard]] const std::vector<std::uint8_t>& as_bytes() const
    {
        return std::get<std::vector<std::uint8_t>>(v_);
    }

    // Lossy widening: returns int as int64 even if the wire was uint64.
    // For our manifests, sizes/counts always fit in int64.
    [[nodiscard]] std::int64_t as_int() const
    {
        if (auto p = std::get_if<std::int64_t>(&v_)) {
            return *p;
        }
        if (auto p = std::get_if<std::uint64_t>(&v_)) {
            return static_cast<std::int64_t>(*p);
        }
        return 0;
    }

    [[nodiscard]] const Value* find(std::string_view key) const
    {
        if (!is_object()) {
            return nullptr;
        }
        for (const auto& [k, v] : as_object()) {
            if (k == key) {
                return &v;
            }
        }
        return nullptr;
    }

    [[nodiscard]] std::string text_or(std::string_view key, std::string_view fallback) const
    {
        if (auto* v = find(key); v != nullptr && v->is_text()) {
            return v->as_text();
        }
        return std::string{fallback};
    }
    [[nodiscard]] bool bool_or(std::string_view key, bool fallback) const
    {
        if (auto* v = find(key); v != nullptr && v->is_bool()) {
            return v->as_bool();
        }
        return fallback;
    }

private:
    Variant v_{};
};

// Encode a Value to a fresh byte vector.
[[nodiscard]] std::vector<std::uint8_t> encode(const Value& v);

// Append-encode into an existing buffer.
void encode_into(std::vector<std::uint8_t>& out, const Value& v);

struct DecodeError {
    std::string message;
    std::size_t offset{0};
};

// Decode the first CBOR item from `data`. Reads at most data.size() bytes;
// trailing data is ignored. The byte count consumed is set in `consumed`
// when supplied and decoding succeeds.
[[nodiscard]] std::expected<Value, DecodeError> decode(std::span<const std::uint8_t> data,
                                                       std::size_t* consumed = nullptr);

} // namespace onebit::ingest::detail::cbor

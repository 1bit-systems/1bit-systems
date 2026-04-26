#include "cbor.hpp"

#include <bit>
#include <cstring>

namespace onebit::ingest::detail::cbor {

namespace {

// ----------------------------------------------------------------------
// Encode

void put_u8(std::vector<std::uint8_t>& out, std::uint8_t v)
{
    out.push_back(v);
}

void put_u16_be(std::vector<std::uint8_t>& out, std::uint16_t v)
{
    out.push_back(static_cast<std::uint8_t>(v >> 8));
    out.push_back(static_cast<std::uint8_t>(v));
}
void put_u32_be(std::vector<std::uint8_t>& out, std::uint32_t v)
{
    out.push_back(static_cast<std::uint8_t>(v >> 24));
    out.push_back(static_cast<std::uint8_t>(v >> 16));
    out.push_back(static_cast<std::uint8_t>(v >> 8));
    out.push_back(static_cast<std::uint8_t>(v));
}
void put_u64_be(std::vector<std::uint8_t>& out, std::uint64_t v)
{
    for (int i = 7; i >= 0; --i) {
        out.push_back(static_cast<std::uint8_t>(v >> (i * 8)));
    }
}

// Major type in high 3 bits; argument follows per spec.
void put_head(std::vector<std::uint8_t>& out, std::uint8_t major, std::uint64_t arg)
{
    const auto m = static_cast<std::uint8_t>(major << 5);
    if (arg < 24) {
        put_u8(out, m | static_cast<std::uint8_t>(arg));
    } else if (arg <= 0xFF) {
        put_u8(out, m | 24);
        put_u8(out, static_cast<std::uint8_t>(arg));
    } else if (arg <= 0xFFFF) {
        put_u8(out, m | 25);
        put_u16_be(out, static_cast<std::uint16_t>(arg));
    } else if (arg <= 0xFFFFFFFFULL) {
        put_u8(out, m | 26);
        put_u32_be(out, static_cast<std::uint32_t>(arg));
    } else {
        put_u8(out, m | 27);
        put_u64_be(out, arg);
    }
}

void encode_text(std::vector<std::uint8_t>& out, std::string_view s)
{
    put_head(out, 3, s.size());
    out.insert(out.end(),
               reinterpret_cast<const std::uint8_t*>(s.data()),
               reinterpret_cast<const std::uint8_t*>(s.data() + s.size()));
}

void encode_bytes(std::vector<std::uint8_t>& out, std::span<const std::uint8_t> b)
{
    put_head(out, 2, b.size());
    out.insert(out.end(), b.begin(), b.end());
}

void encode_uint(std::vector<std::uint8_t>& out, std::uint64_t v)
{
    put_head(out, 0, v);
}
void encode_negint(std::vector<std::uint8_t>& out, std::int64_t v)
{
    // CBOR negative int is encoded as -1 - n with major type 1.
    const std::uint64_t arg = static_cast<std::uint64_t>(-(v + 1));
    put_head(out, 1, arg);
}

// ----------------------------------------------------------------------
// Decode

class Reader {
public:
    explicit Reader(std::span<const std::uint8_t> data) noexcept : data_{data} {}

    [[nodiscard]] std::size_t pos() const noexcept { return pos_; }
    [[nodiscard]] bool        eof() const noexcept { return pos_ >= data_.size(); }

    [[nodiscard]] std::expected<std::uint8_t, DecodeError> peek() const noexcept
    {
        if (eof()) {
            return std::unexpected(DecodeError{"unexpected eof", pos_});
        }
        return data_[pos_];
    }
    [[nodiscard]] std::expected<std::uint8_t, DecodeError> read_u8() noexcept
    {
        if (eof()) {
            return std::unexpected(DecodeError{"unexpected eof", pos_});
        }
        return data_[pos_++];
    }
    [[nodiscard]] std::expected<std::uint16_t, DecodeError> read_u16_be() noexcept
    {
        if (pos_ + 2 > data_.size()) {
            return std::unexpected(DecodeError{"truncated u16", pos_});
        }
        const auto v = static_cast<std::uint16_t>(
            (static_cast<std::uint16_t>(data_[pos_]) << 8) | data_[pos_ + 1]);
        pos_ += 2;
        return v;
    }
    [[nodiscard]] std::expected<std::uint32_t, DecodeError> read_u32_be() noexcept
    {
        if (pos_ + 4 > data_.size()) {
            return std::unexpected(DecodeError{"truncated u32", pos_});
        }
        std::uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            v = (v << 8) | data_[pos_ + i];
        }
        pos_ += 4;
        return v;
    }
    [[nodiscard]] std::expected<std::uint64_t, DecodeError> read_u64_be() noexcept
    {
        if (pos_ + 8 > data_.size()) {
            return std::unexpected(DecodeError{"truncated u64", pos_});
        }
        std::uint64_t v = 0;
        for (int i = 0; i < 8; ++i) {
            v = (v << 8) | data_[pos_ + i];
        }
        pos_ += 8;
        return v;
    }
    [[nodiscard]] std::expected<std::span<const std::uint8_t>, DecodeError>
    take(std::size_t n) noexcept
    {
        if (pos_ + n > data_.size()) {
            return std::unexpected(DecodeError{"truncated payload", pos_});
        }
        auto s = data_.subspan(pos_, n);
        pos_ += n;
        return s;
    }

private:
    std::span<const std::uint8_t> data_;
    std::size_t                   pos_{0};
};

[[nodiscard]] std::expected<std::uint64_t, DecodeError>
read_arg(Reader& r, std::uint8_t info)
{
    if (info < 24) {
        return static_cast<std::uint64_t>(info);
    }
    if (info == 24) {
        auto v = r.read_u8();
        if (!v) {
            return std::unexpected(v.error());
        }
        return static_cast<std::uint64_t>(*v);
    }
    if (info == 25) {
        auto v = r.read_u16_be();
        if (!v) {
            return std::unexpected(v.error());
        }
        return static_cast<std::uint64_t>(*v);
    }
    if (info == 26) {
        auto v = r.read_u32_be();
        if (!v) {
            return std::unexpected(v.error());
        }
        return static_cast<std::uint64_t>(*v);
    }
    if (info == 27) {
        return r.read_u64_be();
    }
    return std::unexpected(DecodeError{"indefinite or reserved length",
                                       r.pos()});
}

[[nodiscard]] std::expected<Value, DecodeError> decode_one(Reader& r)
{
    auto first = r.read_u8();
    if (!first) {
        return std::unexpected(first.error());
    }
    const std::uint8_t major = (*first) >> 5;
    const std::uint8_t info  = (*first) & 0x1F;

    switch (major) {
    case 0: { // unsigned int
        auto a = read_arg(r, info);
        if (!a) {
            return std::unexpected(a.error());
        }
        if (*a <= static_cast<std::uint64_t>(INT64_MAX)) {
            return Value{static_cast<std::int64_t>(*a)};
        }
        return Value{*a};
    }
    case 1: { // negative int = -1 - arg
        auto a = read_arg(r, info);
        if (!a) {
            return std::unexpected(a.error());
        }
        const auto v = -(static_cast<std::int64_t>(*a) + 1);
        return Value{v};
    }
    case 2: { // byte string
        auto a = read_arg(r, info);
        if (!a) {
            return std::unexpected(a.error());
        }
        auto b = r.take(static_cast<std::size_t>(*a));
        if (!b) {
            return std::unexpected(b.error());
        }
        std::vector<std::uint8_t> bytes(b->begin(), b->end());
        return Value{std::move(bytes)};
    }
    case 3: { // text string
        auto a = read_arg(r, info);
        if (!a) {
            return std::unexpected(a.error());
        }
        auto b = r.take(static_cast<std::size_t>(*a));
        if (!b) {
            return std::unexpected(b.error());
        }
        return Value{std::string(reinterpret_cast<const char*>(b->data()),
                                 b->size())};
    }
    case 4: { // array
        auto a = read_arg(r, info);
        if (!a) {
            return std::unexpected(a.error());
        }
        Array arr;
        arr.reserve(static_cast<std::size_t>(*a));
        for (std::uint64_t i = 0; i < *a; ++i) {
            auto e = decode_one(r);
            if (!e) {
                return std::unexpected(e.error());
            }
            arr.push_back(std::move(*e));
        }
        return Value{std::move(arr)};
    }
    case 5: { // map
        auto a = read_arg(r, info);
        if (!a) {
            return std::unexpected(a.error());
        }
        Object obj;
        obj.reserve(static_cast<std::size_t>(*a));
        for (std::uint64_t i = 0; i < *a; ++i) {
            auto k = decode_one(r);
            if (!k) {
                return std::unexpected(k.error());
            }
            // Manifest keys are text strings. Skip non-string keys.
            std::string key;
            if (k->is_text()) {
                key = k->as_text();
            }
            auto val = decode_one(r);
            if (!val) {
                return std::unexpected(val.error());
            }
            obj.emplace_back(std::move(key), std::move(*val));
        }
        return Value{std::move(obj)};
    }
    case 7: { // simple / float
        if (info == 20) {
            return Value{false};
        }
        if (info == 21) {
            return Value{true};
        }
        if (info == 22 || info == 23) {
            return Value{std::monostate{}};
        }
        if (info == 25) { // half-float — not used by us, skip cleanly
            auto a = r.read_u16_be();
            if (!a) {
                return std::unexpected(a.error());
            }
            return Value{0.0};
        }
        if (info == 26) {
            auto a = r.read_u32_be();
            if (!a) {
                return std::unexpected(a.error());
            }
            float    f = 0.0F;
            std::uint32_t bits = *a;
            std::memcpy(&f, &bits, sizeof(f));
            return Value{static_cast<double>(f)};
        }
        if (info == 27) {
            auto a = r.read_u64_be();
            if (!a) {
                return std::unexpected(a.error());
            }
            double         d    = 0.0;
            std::uint64_t  bits = *a;
            std::memcpy(&d, &bits, sizeof(d));
            return Value{d};
        }
        return std::unexpected(DecodeError{"unsupported simple value",
                                           r.pos()});
    }
    default:
        return std::unexpected(DecodeError{"unsupported major type",
                                           r.pos()});
    }
}

} // namespace

void encode_into(std::vector<std::uint8_t>& out, const Value& v)
{
    std::visit(
        [&](const auto& x) {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                put_u8(out, 0xF6); // null
            } else if constexpr (std::is_same_v<T, bool>) {
                put_u8(out, x ? 0xF5 : 0xF4);
            } else if constexpr (std::is_same_v<T, std::int64_t>) {
                if (x >= 0) {
                    encode_uint(out, static_cast<std::uint64_t>(x));
                } else {
                    encode_negint(out, x);
                }
            } else if constexpr (std::is_same_v<T, std::uint64_t>) {
                encode_uint(out, x);
            } else if constexpr (std::is_same_v<T, double>) {
                put_u8(out, 0xFB);
                std::uint64_t bits = 0;
                std::memcpy(&bits, &x, sizeof(bits));
                put_u64_be(out, bits);
            } else if constexpr (std::is_same_v<T, std::string>) {
                encode_text(out, x);
            } else if constexpr (std::is_same_v<T, std::vector<std::uint8_t>>) {
                encode_bytes(out, x);
            } else if constexpr (std::is_same_v<T, Array>) {
                put_head(out, 4, x.size());
                for (const auto& e : x) {
                    encode_into(out, e);
                }
            } else if constexpr (std::is_same_v<T, Object>) {
                put_head(out, 5, x.size());
                for (const auto& [k, val] : x) {
                    encode_text(out, k);
                    encode_into(out, val);
                }
            }
        },
        v.variant());
}

std::vector<std::uint8_t> encode(const Value& v)
{
    std::vector<std::uint8_t> out;
    encode_into(out, v);
    return out;
}

std::expected<Value, DecodeError> decode(std::span<const std::uint8_t> data,
                                         std::size_t* consumed)
{
    Reader r{data};
    auto   v = decode_one(r);
    if (!v) {
        return v;
    }
    if (consumed != nullptr) {
        *consumed = r.pos();
    }
    return v;
}

} // namespace onebit::ingest::detail::cbor

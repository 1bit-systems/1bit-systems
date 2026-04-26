#include "onebit/core/gguf.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <utility>

namespace onebit::core::gguf {

struct GgufFile::Impl {
    int          fd       = -1;
    void*        addr     = MAP_FAILED;
    std::size_t  size     = 0;
    std::size_t  data_off = 0;

    ~Impl()
    {
        if (addr != MAP_FAILED && size > 0) ::munmap(addr, size);
        if (fd >= 0) ::close(fd);
    }
};

GgufFile::GgufFile()                          = default;
GgufFile::GgufFile(GgufFile&&)                = default;
GgufFile& GgufFile::operator=(GgufFile&&)     = default;
GgufFile::~GgufFile()                         = default;

namespace {

[[nodiscard]] std::error_code last_errno()
{
    return std::error_code(errno, std::generic_category());
}

template <typename T>
[[nodiscard]] T read_le(const std::uint8_t* p) noexcept
{
    T v{};
    std::memcpy(&v, p, sizeof(T));
    return v;
}

class Reader {
public:
    Reader(const std::uint8_t* p, std::size_t size) noexcept
        : p_(p), size_(size) {}

    [[nodiscard]] std::expected<void, HaloError> require(std::size_t n) noexcept
    {
        if (off_ + n > size_) {
            return std::unexpected(HaloError::truncated(off_, n, size_ - off_));
        }
        return {};
    }

    template <typename T>
    [[nodiscard]] std::expected<T, HaloError> read() noexcept
    {
        if (auto r = require(sizeof(T)); !r) return std::unexpected(r.error());
        T v = read_le<T>(p_ + off_);
        off_ += sizeof(T);
        return v;
    }

    [[nodiscard]] std::expected<std::string, HaloError> read_string()
    {
        auto len = read<std::uint64_t>();
        if (!len) return std::unexpected(len.error());
        if (auto r = require(static_cast<std::size_t>(*len)); !r)
            return std::unexpected(r.error());
        std::string s(reinterpret_cast<const char*>(p_ + off_),
                      static_cast<std::size_t>(*len));
        off_ += static_cast<std::size_t>(*len);
        return s;
    }

    [[nodiscard]] std::size_t offset() const noexcept { return off_; }
    void set_offset(std::size_t o) noexcept { off_ = o; }

private:
    const std::uint8_t* p_;
    std::size_t         size_;
    std::size_t         off_ = 0;
};

[[nodiscard]] std::expected<Value, HaloError>
read_value(Reader& r, ValueType t);

[[nodiscard]] std::expected<std::shared_ptr<Array>, HaloError>
read_array(Reader& r)
{
    auto et = r.read<std::uint32_t>();
    if (!et) return std::unexpected(et.error());
    auto n = r.read<std::uint64_t>();
    if (!n) return std::unexpected(n.error());

    auto arr = std::make_shared<Array>();
    arr->element_type = static_cast<ValueType>(*et);
    arr->items.reserve(static_cast<std::size_t>(*n));
    for (std::uint64_t i = 0; i < *n; ++i) {
        auto v = read_value(r, arr->element_type);
        if (!v) return std::unexpected(v.error());
        arr->items.emplace_back(std::move(*v));
    }
    return arr;
}

template <typename T>
[[nodiscard]] std::expected<Value, HaloError>
read_value_as(Reader& r)
{
    auto v = r.read<T>();
    if (!v) return std::unexpected(v.error());
    return Value{*v};
}

[[nodiscard]] std::expected<Value, HaloError>
read_value(Reader& r, ValueType t)
{
    switch (t) {
    case ValueType::UInt8:   return read_value_as<std::uint8_t>(r);
    case ValueType::Int8:    return read_value_as<std::int8_t>(r);
    case ValueType::UInt16:  return read_value_as<std::uint16_t>(r);
    case ValueType::Int16:   return read_value_as<std::int16_t>(r);
    case ValueType::UInt32:  return read_value_as<std::uint32_t>(r);
    case ValueType::Int32:   return read_value_as<std::int32_t>(r);
    case ValueType::Float32: return read_value_as<float>(r);
    case ValueType::UInt64:  return read_value_as<std::uint64_t>(r);
    case ValueType::Int64:   return read_value_as<std::int64_t>(r);
    case ValueType::Float64: return read_value_as<double>(r);
    case ValueType::Bool: {
        auto v = r.read<std::uint8_t>();
        if (!v) return std::unexpected(v.error());
        return Value{static_cast<bool>(*v)};
    }
    case ValueType::String: {
        auto v = r.read_string();
        if (!v) return std::unexpected(v.error());
        return Value{std::move(*v)};
    }
    case ValueType::Array: {
        auto v = read_array(r);
        if (!v) return std::unexpected(v.error());
        return Value{std::move(*v)};
    }
    }
    return std::unexpected(HaloError::invalid_config("unknown gguf value type"));
}

template <typename T>
[[nodiscard]] std::optional<T> get_metadata(
    const std::unordered_map<std::string, Value>& md, const std::string& key)
{
    auto it = md.find(key);
    if (it == md.end()) return std::nullopt;
    if (auto* p = std::get_if<T>(&it->second)) return *p;
    return std::nullopt;
}

} // namespace

std::expected<GgufFile, HaloError>
GgufFile::open(const std::filesystem::path& path)
{
    GgufFile f;
    f.impl_     = std::make_unique<Impl>();

    f.impl_->fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (f.impl_->fd < 0) return std::unexpected(HaloError::io(path, last_errno()));

    struct stat st{};
    if (::fstat(f.impl_->fd, &st) != 0)
        return std::unexpected(HaloError::io(path, last_errno()));
    f.impl_->size = static_cast<std::size_t>(st.st_size);

    f.impl_->addr =
        ::mmap(nullptr, f.impl_->size, PROT_READ, MAP_PRIVATE, f.impl_->fd, 0);
    if (f.impl_->addr == MAP_FAILED)
        return std::unexpected(HaloError::io(path, last_errno()));

    Reader r(static_cast<const std::uint8_t*>(f.impl_->addr), f.impl_->size);
    if (auto v = r.require(8); !v) return std::unexpected(v.error());

    const std::uint8_t* p = static_cast<const std::uint8_t*>(f.impl_->addr);
    std::array<std::uint8_t, 4> got{p[0], p[1], p[2], p[3]};
    if (got != MAGIC) return std::unexpected(HaloError::bad_magic(MAGIC, got));
    r.set_offset(4);

    auto ver = r.read<std::uint32_t>();
    if (!ver) return std::unexpected(ver.error());
    if (*ver < MIN_VERSION)
        return std::unexpected(HaloError::unsupported_version(
            static_cast<std::int32_t>(*ver),
            static_cast<std::int32_t>(MIN_VERSION),
            std::numeric_limits<std::int32_t>::max()));
    f.version_ = *ver;

    auto n_tensors = r.read<std::uint64_t>();
    if (!n_tensors) return std::unexpected(n_tensors.error());
    auto n_kv = r.read<std::uint64_t>();
    if (!n_kv) return std::unexpected(n_kv.error());

    for (std::uint64_t i = 0; i < *n_kv; ++i) {
        auto k = r.read_string();
        if (!k) return std::unexpected(k.error());
        auto vt = r.read<std::uint32_t>();
        if (!vt) return std::unexpected(vt.error());
        auto v = read_value(r, static_cast<ValueType>(*vt));
        if (!v) return std::unexpected(v.error());
        f.metadata_.emplace(std::move(*k), std::move(*v));
    }

    f.tensors_.reserve(static_cast<std::size_t>(*n_tensors));
    for (std::uint64_t i = 0; i < *n_tensors; ++i) {
        TensorInfo ti;
        auto name = r.read_string();
        if (!name) return std::unexpected(name.error());
        ti.name = std::move(*name);

        auto n_dims = r.read<std::uint32_t>();
        if (!n_dims) return std::unexpected(n_dims.error());
        ti.dims.resize(*n_dims);
        for (std::uint32_t d = 0; d < *n_dims; ++d) {
            auto v = r.read<std::uint64_t>();
            if (!v) return std::unexpected(v.error());
            ti.dims[d] = *v;
        }

        auto tt = r.read<std::uint32_t>();
        if (!tt) return std::unexpected(tt.error());
        ti.type = static_cast<TensorType>(*tt);

        auto off = r.read<std::uint64_t>();
        if (!off) return std::unexpected(off.error());
        ti.offset = *off;

        f.tensors_.emplace_back(std::move(ti));
    }

    // Tensor data starts at the next alignment boundary (32 by default;
    // overridden by general.alignment KV if present).
    std::uint64_t alignment = 32;
    if (auto a = get_metadata<std::uint32_t>(f.metadata_, "general.alignment")) {
        alignment = *a;
    }
    const std::size_t cur = r.offset();
    const std::size_t pad = (alignment - (cur % alignment)) % alignment;
    f.impl_->data_off = cur + pad;

    return f;
}

std::expected<std::span<const std::uint8_t>, HaloError>
GgufFile::tensor_bytes(const TensorInfo& info) const noexcept
{
    if (!impl_) return std::unexpected(HaloError::truncated(0, 0, 0));
    // Caller provides byte length via dims × dtype size; we just bounds-check
    // that the offset is within the file.
    const std::size_t base = impl_->data_off + static_cast<std::size_t>(info.offset);
    if (base > impl_->size) {
        return std::unexpected(HaloError::truncated(base, 0, impl_->size - std::min(impl_->size, base)));
    }
    return std::span<const std::uint8_t>(
        static_cast<const std::uint8_t*>(impl_->addr) + base,
        impl_->size - base);
}

std::expected<BitnetHeader, HaloError>
GgufFile::bitnet_header() const noexcept
{
    BitnetHeader h{};
    auto get_i32 = [&](const std::string& key) -> std::optional<std::int32_t> {
        if (auto v = get_metadata<std::uint32_t>(metadata_, key)) return static_cast<std::int32_t>(*v);
        if (auto v = get_metadata<std::int32_t>(metadata_, key))  return *v;
        if (auto v = get_metadata<std::uint64_t>(metadata_, key)) return static_cast<std::int32_t>(*v);
        if (auto v = get_metadata<std::int64_t>(metadata_, key))  return static_cast<std::int32_t>(*v);
        return std::nullopt;
    };
    auto get_f32 = [&](const std::string& key) -> std::optional<float> {
        if (auto v = get_metadata<float>(metadata_, key))  return *v;
        if (auto v = get_metadata<double>(metadata_, key)) return static_cast<float>(*v);
        return std::nullopt;
    };

    if (auto v = get_i32("bitnet.embedding_length"))             h.hidden_dim = *v;
    if (auto v = get_i32("bitnet.feed_forward_length"))          h.intermediate_dim = *v;
    if (auto v = get_i32("bitnet.attention.head_count"))         h.num_heads = *v;
    if (auto v = get_i32("bitnet.attention.head_count_kv"))      h.num_kv_heads = *v;
    if (auto v = get_i32("bitnet.block_count"))                  h.num_layers = *v;
    if (auto v = get_i32("bitnet.context_length"))               h.max_seq_len = *v;
    if (auto v = get_i32("tokenizer.ggml.token_count"))          h.vocab_size = *v;
    if (auto v = get_f32("bitnet.rope.freq_base"))               h.rope_theta = *v;
    if (auto v = get_f32("bitnet.attention.layer_norm_rms_epsilon")) h.rms_norm_eps = *v;

    return h;
}

} // namespace onebit::core::gguf

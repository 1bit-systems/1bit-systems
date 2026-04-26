#include "onebit/core/h1b.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <utility>

namespace onebit::core::h1b {

struct File::Impl {
    int          fd       = -1;
    void*        addr     = MAP_FAILED;
    std::size_t  size     = 0;
    WeightFormat fmt      = WeightFormat::Ternary2bpw;

    ~Impl()
    {
        if (addr != MAP_FAILED && size > 0) {
            ::munmap(addr, size);
        }
        if (fd >= 0) {
            ::close(fd);
        }
    }
};

File::File()                           = default;
File::File(File&&)                     = default;
File& File::operator=(File&&)          = default;
File::~File()                          = default;

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

} // namespace

std::expected<File, HaloError> File::open(const std::filesystem::path& path)
{
    File f;
    f.path_ = path;
    f.impl_ = std::make_unique<Impl>();

    f.impl_->fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (f.impl_->fd < 0) {
        return std::unexpected(HaloError::io(path, last_errno()));
    }
    struct stat st{};
    if (::fstat(f.impl_->fd, &st) != 0) {
        return std::unexpected(HaloError::io(path, last_errno()));
    }
    f.impl_->size = static_cast<std::size_t>(st.st_size);
    if (f.impl_->size < 64) {
        return std::unexpected(HaloError::truncated(0, 64, f.impl_->size));
    }
    f.impl_->addr =
        ::mmap(nullptr, f.impl_->size, PROT_READ, MAP_PRIVATE, f.impl_->fd, 0);
    if (f.impl_->addr == MAP_FAILED) {
        return std::unexpected(HaloError::io(path, last_errno()));
    }

    const auto* p = static_cast<const std::uint8_t*>(f.impl_->addr);
    std::array<std::uint8_t, 4> got{p[0], p[1], p[2], p[3]};
    if (got != MAGIC) {
        return std::unexpected(HaloError::bad_magic(MAGIC, got));
    }

    Config c{};
    c.version          = read_le<std::int32_t>(p + 4);
    if (c.version < MIN_SUPPORTED_VERSION || c.version > MAX_SUPPORTED_VERSION) {
        return std::unexpected(HaloError::unsupported_version(
            c.version, MIN_SUPPORTED_VERSION, MAX_SUPPORTED_VERSION));
    }
    c.hidden_dim       = read_le<std::int32_t>(p + 8);
    c.intermediate_dim = read_le<std::int32_t>(p + 12);
    c.num_heads        = read_le<std::int32_t>(p + 16);
    c.num_kv_heads     = read_le<std::int32_t>(p + 20);
    c.num_layers       = read_le<std::int32_t>(p + 24);
    c.vocab_size       = read_le<std::int32_t>(p + 28);
    c.max_seq_len      = read_le<std::int32_t>(p + 32);
    c.flags            = read_le<std::uint32_t>(p + 36);
    c.rope_theta       = (c.version >= 2) ? read_le<float>(p + 40) : DEFAULT_ROPE_THETA;
    c.rms_norm_eps     = (c.version >= 2) ? read_le<float>(p + 44) : DEFAULT_RMS_NORM_EPS;
    f.config_ = c;

    if (c.flags & FLAG_HADAMARD_ROTATED) f.impl_->fmt = WeightFormat::HadamardRotated;
    else if (c.flags & FLAG_SHERRY_FP16) f.impl_->fmt = WeightFormat::SherryFp16Sparse;
    else if (c.flags & FLAG_BONSAI_TQ2)  f.impl_->fmt = WeightFormat::BonsaiTq2;
    else if (c.flags & FLAG_BONSAI_Q1)   f.impl_->fmt = WeightFormat::BonsaiQ1;
    else                                 f.impl_->fmt = WeightFormat::Ternary2bpw;

    // Layer offset table starts at byte 64; 9 u64s per layer.
    constexpr std::size_t HEADER_BYTES = 64;
    constexpr std::size_t LAYER_BYTES  = 9 * sizeof(std::uint64_t);
    const std::size_t layers_start = HEADER_BYTES;
    const std::size_t layers_end =
        layers_start + static_cast<std::size_t>(c.num_layers) * LAYER_BYTES;
    if (layers_end > f.impl_->size) {
        return std::unexpected(HaloError::truncated(layers_start, layers_end - layers_start, f.impl_->size - layers_start));
    }
    f.layers_.resize(static_cast<std::size_t>(c.num_layers));
    for (int i = 0; i < c.num_layers; ++i) {
        const std::uint8_t* lp = p + layers_start + i * LAYER_BYTES;
        LayerOffsets& lo = f.layers_[i];
        lo.attn_q    = read_le<std::uint64_t>(lp + 0  * 8);
        lo.attn_k    = read_le<std::uint64_t>(lp + 1  * 8);
        lo.attn_v    = read_le<std::uint64_t>(lp + 2  * 8);
        lo.attn_o    = read_le<std::uint64_t>(lp + 3  * 8);
        lo.ffn_gate  = read_le<std::uint64_t>(lp + 4  * 8);
        lo.ffn_up    = read_le<std::uint64_t>(lp + 5  * 8);
        lo.ffn_down  = read_le<std::uint64_t>(lp + 6  * 8);
        lo.attn_norm = read_le<std::uint64_t>(lp + 7  * 8);
        lo.ffn_norm  = read_le<std::uint64_t>(lp + 8  * 8);
    }

    return f;
}

std::span<const std::uint8_t> File::bytes() const noexcept
{
    if (!impl_ || impl_->addr == MAP_FAILED) return {};
    return {static_cast<const std::uint8_t*>(impl_->addr), impl_->size};
}

WeightFormat File::format() const noexcept
{
    return impl_ ? impl_->fmt : WeightFormat::Ternary2bpw;
}

std::expected<std::span<const std::uint8_t>, HaloError>
File::slice(std::uint64_t offset, std::size_t bytes_) const noexcept
{
    if (!impl_) {
        return std::unexpected(HaloError::truncated(0, bytes_, 0));
    }
    if (offset > impl_->size || bytes_ > impl_->size - offset) {
        return std::unexpected(HaloError::truncated(
            static_cast<std::size_t>(offset), bytes_,
            impl_->size - std::min<std::size_t>(impl_->size, offset)));
    }
    return std::span<const std::uint8_t>(
        static_cast<const std::uint8_t*>(impl_->addr) + offset, bytes_);
}

} // namespace onebit::core::h1b

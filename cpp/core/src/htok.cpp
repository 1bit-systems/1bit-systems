#include "onebit/core/htok.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace onebit::core::htok {

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
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return std::unexpected(HaloError::io(path, last_errno()));
    }
    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        return std::unexpected(HaloError::io(path, last_errno()));
    }
    const std::size_t size = static_cast<std::size_t>(st.st_size);
    void* addr = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (addr == MAP_FAILED) {
        return std::unexpected(HaloError::io(path, last_errno()));
    }

    auto bytes = std::span<const std::uint8_t>(
        static_cast<const std::uint8_t*>(addr), size);
    auto result = parse(bytes);
    ::munmap(addr, size);
    return result;
}

std::expected<File, HaloError> File::parse(std::span<const std::uint8_t> bytes)
{
    if (bytes.size() < 32) {
        return std::unexpected(HaloError::truncated(0, 32, bytes.size()));
    }
    const std::uint8_t* p = bytes.data();
    std::array<std::uint8_t, 4> got{p[0], p[1], p[2], p[3]};
    if (got != MAGIC) {
        return std::unexpected(HaloError::bad_magic(MAGIC, got));
    }

    File f;
    f.bos_id_ = read_le<std::int32_t>(p + 4);
    f.eos_id_ = read_le<std::int32_t>(p + 8);
    f.pad_id_ = read_le<std::int32_t>(p + 12);
    const std::int32_t n_pieces = read_le<std::int32_t>(p + 16);
    const std::int32_t n_merges = read_le<std::int32_t>(p + 20);

    std::size_t off = 32;
    f.pieces_.reserve(static_cast<std::size_t>(n_pieces));
    for (std::int32_t i = 0; i < n_pieces; ++i) {
        if (off + 4 > bytes.size()) {
            return std::unexpected(HaloError::truncated(off, 4, bytes.size() - off));
        }
        const std::uint32_t len = read_le<std::uint32_t>(p + off);
        off += 4;
        if (off + len > bytes.size()) {
            return std::unexpected(HaloError::truncated(off, len, bytes.size() - off));
        }
        std::vector<std::uint8_t> piece(p + off, p + off + len);
        std::string key(reinterpret_cast<const char*>(piece.data()), piece.size());
        f.piece_to_id_.emplace(std::move(key), i);
        f.pieces_.emplace_back(std::move(piece));
        off += len;
    }

    f.merges_.reserve(static_cast<std::size_t>(n_merges));
    for (std::int32_t i = 0; i < n_merges; ++i) {
        if (off + 8 > bytes.size()) {
            return std::unexpected(HaloError::truncated(off, 8, bytes.size() - off));
        }
        const std::uint32_t la = read_le<std::uint32_t>(p + off);
        const std::uint32_t lb = read_le<std::uint32_t>(p + off + 4);
        off += 8;
        if (off + la + lb > bytes.size()) {
            return std::unexpected(HaloError::truncated(off, la + lb, bytes.size() - off));
        }
        Merge m;
        m.a.assign(p + off,        p + off + la);
        m.b.assign(p + off + la,   p + off + la + lb);
        m.rank = i;
        f.merges_.emplace_back(std::move(m));
        off += la + lb;
    }

    return f;
}

std::expected<std::vector<TokenId>, HaloError>
File::encode(std::span<const std::uint8_t> bytes) const
{
    // Byte-fallback BPE: start with one piece per UTF-8 byte, repeatedly
    // apply the lowest-rank merge that covers an adjacent pair until no
    // further merges apply. Linear in input * merges in the worst case;
    // good enough for the small contexts the host-side path uses.
    std::vector<std::vector<std::uint8_t>> tokens;
    tokens.reserve(bytes.size());
    for (std::uint8_t b : bytes) {
        tokens.push_back({b});
    }

    for (;;) {
        std::int32_t  best_rank = -1;
        std::size_t   best_idx  = 0;
        for (std::size_t i = 0; i + 1 < tokens.size(); ++i) {
            for (const Merge& m : merges_) {
                if (tokens[i] == m.a && tokens[i + 1] == m.b) {
                    if (best_rank < 0 || m.rank < best_rank) {
                        best_rank = m.rank;
                        best_idx  = i;
                    }
                    break;
                }
            }
        }
        if (best_rank < 0) break;
        tokens[best_idx].insert(tokens[best_idx].end(),
                                tokens[best_idx + 1].begin(),
                                tokens[best_idx + 1].end());
        tokens.erase(tokens.begin() + best_idx + 1);
    }

    std::vector<TokenId> ids;
    ids.reserve(tokens.size());
    for (const auto& tok : tokens) {
        std::string key(reinterpret_cast<const char*>(tok.data()), tok.size());
        auto it = piece_to_id_.find(key);
        if (it == piece_to_id_.end()) {
            return std::unexpected(HaloError::unknown_byte_piece(tok));
        }
        ids.push_back(it->second);
    }
    return ids;
}

std::vector<std::uint8_t>
File::decode(std::span<const TokenId> ids) const
{
    std::vector<std::uint8_t> out;
    for (TokenId id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= pieces_.size()) continue;
        const auto& p = pieces_[static_cast<std::size_t>(id)];
        out.insert(out.end(), p.begin(), p.end());
    }
    return out;
}

} // namespace onebit::core::htok

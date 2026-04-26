#include "onebit/stream/container.hpp"

#include "onebit/ingest/sha256.hpp"

#include <fstream>

namespace onebit::stream {

std::expected<Catalog, onebit::ingest::IngestError>
open_catalog(const std::filesystem::path& path)
{
    auto rep = onebit::ingest::validate(path);
    if (!rep) {
        return std::unexpected(rep.error());
    }
    Catalog cat;
    cat.path        = path;
    cat.manifest    = std::move(rep->manifest);
    cat.total_bytes = rep->total_bytes;
    cat.footer_offset =
        (rep->total_bytes >= 32) ? rep->total_bytes - 32 : 0;
    cat.sections.reserve(rep->sections.size());
    for (const auto& s : rep->sections) {
        cat.sections.push_back(Section{s.tag, s.offset, s.length});
    }
    return cat;
}

std::expected<std::vector<std::uint8_t>, onebit::ingest::IngestError>
build_lossy_bytes(const Catalog& cat)
{
    std::ifstream in(cat.path, std::ios::binary);
    if (!in) {
        return std::unexpected(onebit::ingest::IngestError::io(cat.path, "open"));
    }
    // Read the entire header block — magic (4) + u32 header_len + CBOR.
    std::array<std::uint8_t, 4> magic{};
    in.read(reinterpret_cast<char*>(magic.data()), 4);
    if (!in.good()) {
        return std::unexpected(onebit::ingest::IngestError::io(cat.path, "read magic"));
    }
    std::array<std::uint8_t, 4> hlen_le{};
    in.read(reinterpret_cast<char*>(hlen_le.data()), 4);
    if (!in.good()) {
        return std::unexpected(onebit::ingest::IngestError::io(cat.path, "read hlen"));
    }
    const std::uint32_t header_len =
        static_cast<std::uint32_t>(hlen_le[0]) |
        (static_cast<std::uint32_t>(hlen_le[1]) << 8) |
        (static_cast<std::uint32_t>(hlen_le[2]) << 16) |
        (static_cast<std::uint32_t>(hlen_le[3]) << 24);
    std::vector<std::uint8_t> header_buf(header_len);
    if (header_len > 0) {
        in.read(reinterpret_cast<char*>(header_buf.data()), header_len);
        if (!in.good()) {
            return std::unexpected(onebit::ingest::IngestError::io(cat.path, "read header"));
        }
    }

    if (magic[0] != '1' || magic[1] != 'B' || magic[2] != 'L') {
        return std::unexpected(onebit::ingest::IngestError::bad_magic(magic));
    }
    if (magic[3] != onebit::ingest::VERSION) {
        return std::unexpected(onebit::ingest::IngestError::bad_version(magic[3]));
    }

    std::vector<std::uint8_t> out;
    out.reserve(static_cast<std::size_t>(cat.total_bytes));
    out.insert(out.end(), magic.begin(), magic.end());
    out.insert(out.end(), hlen_le.begin(), hlen_le.end());
    out.insert(out.end(), header_buf.begin(), header_buf.end());

    constexpr std::size_t buf_sz = 64 * 1024;
    std::vector<char>     buf(buf_sz);
    for (const auto& s : cat.sections) {
        if (!s.is_lossy_tier()) {
            continue;
        }
        out.push_back(s.tag);
        const auto len = s.length;
        for (int i = 0; i < 8; ++i) {
            out.push_back(static_cast<std::uint8_t>(len >> (i * 8)));
        }
        in.seekg(static_cast<std::streamoff>(s.offset), std::ios::beg);
        std::uint64_t remaining = s.length;
        while (remaining > 0) {
            const auto want = static_cast<std::streamsize>(
                std::min<std::uint64_t>(remaining, buf.size()));
            in.read(buf.data(), want);
            const auto got = in.gcount();
            if (got <= 0) {
                return std::unexpected(
                    onebit::ingest::IngestError::io(cat.path, "short read"));
            }
            out.insert(out.end(), buf.begin(), buf.begin() + got);
            remaining -= static_cast<std::uint64_t>(got);
        }
    }

    const auto digest = onebit::ingest::detail::sha256(out);
    out.insert(out.end(), digest.begin(), digest.end());
    return out;
}

} // namespace onebit::stream

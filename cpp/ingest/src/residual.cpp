#include "onebit/ingest/ingest.hpp"

#include "onebit/ingest/sha256.hpp"

#include <fstream>
#include <vector>

namespace onebit::ingest {

namespace {

namespace fs = std::filesystem;

[[nodiscard]] std::expected<std::vector<std::uint8_t>, IngestError>
read_all(const fs::path& p)
{
    std::ifstream in(p, std::ios::binary);
    if (!in) {
        return std::unexpected(IngestError::io(p, "open"));
    }
    in.seekg(0, std::ios::end);
    const auto sz = in.tellg();
    in.seekg(0, std::ios::beg);
    if (sz < 0) {
        return std::unexpected(IngestError::io(p, "tellg"));
    }
    std::vector<std::uint8_t> buf(static_cast<std::size_t>(sz));
    if (sz > 0) {
        in.read(reinterpret_cast<char*>(buf.data()),
                static_cast<std::streamsize>(sz));
        if (!in.good() && !in.eof()) {
            return std::unexpected(IngestError::io(p, "read"));
        }
    }
    return buf;
}

void append_section(std::vector<std::uint8_t>& out,
                    std::uint8_t               tag,
                    std::span<const std::uint8_t> bytes)
{
    out.push_back(tag);
    const auto len = static_cast<std::uint64_t>(bytes.size());
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<std::uint8_t>(len >> (i * 8)));
    }
    out.insert(out.end(), bytes.begin(), bytes.end());
}

} // namespace

std::expected<ResidualSummary, IngestError>
add_residual(const fs::path& input,
             const fs::path& residual_path,
             const fs::path& index_path,
             const fs::path& out_path)
{
    auto rep = validate(input);
    if (!rep) {
        return std::unexpected(rep.error());
    }
    if (!rep->footer_ok) {
        return std::unexpected(IngestError::invalid(
            "input footer hash did not verify; refusing to upgrade"));
    }
    auto base = read_all(input);
    if (!base) {
        return std::unexpected(base.error());
    }
    if (base->size() < 32) {
        return std::unexpected(IngestError::invalid("input too small to have a footer"));
    }
    base->resize(base->size() - 32); // drop old footer

    auto residual_bytes = read_all(residual_path);
    if (!residual_bytes) {
        return std::unexpected(residual_bytes.error());
    }
    auto index_bytes = read_all(index_path);
    if (!index_bytes) {
        return std::unexpected(index_bytes.error());
    }

    append_section(*base, tag::RESIDUAL_BLOB, *residual_bytes);
    append_section(*base, tag::RESIDUAL_INDEX, *index_bytes);
    const auto digest = detail::sha256(*base);
    base->insert(base->end(), digest.begin(), digest.end());

    std::ofstream of(out_path, std::ios::binary | std::ios::trunc);
    if (!of) {
        return std::unexpected(IngestError::io(out_path, "create"));
    }
    of.write(reinterpret_cast<const char*>(base->data()),
             static_cast<std::streamsize>(base->size()));
    if (!of.good()) {
        return std::unexpected(IngestError::io(out_path, "write"));
    }
    of.flush();

    return ResidualSummary{
        static_cast<std::uint64_t>(residual_bytes->size()),
        static_cast<std::uint64_t>(index_bytes->size()),
    };
}

} // namespace onebit::ingest

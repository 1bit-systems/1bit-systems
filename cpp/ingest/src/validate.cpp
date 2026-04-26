#include "onebit/ingest/ingest.hpp"

#include "cbor.hpp"
#include "onebit/ingest/sha256.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>

namespace onebit::ingest {

namespace {

namespace fs = std::filesystem;

[[nodiscard]] Manifest cbor_to_manifest(const detail::cbor::Value& v)
{
    Manifest m;
    if (!v.is_object()) {
        return m;
    }
    auto str_or = [&](std::string_view key, std::string_view fb) -> std::string {
        return v.text_or(key, fb);
    };
    m.v               = str_or("v", "");
    m.catalog         = str_or("catalog", "");
    m.title           = str_or("title", "");
    m.artist          = str_or("artist", "");
    m.license         = str_or("license", "");
    m.created         = str_or("created", "");
    m.tier            = str_or("tier", "");

    if (auto* p = v.find("license_url"); p != nullptr && p->is_text()) {
        m.license_url = p->as_text();
    }
    if (auto* p = v.find("attribution"); p != nullptr && p->is_text()) {
        m.attribution = p->as_text();
    }
    if (auto* p = v.find("source"); p != nullptr && p->is_text()) {
        m.source = p->as_text();
    }

    if (auto* codec = v.find("codec"); codec != nullptr && codec->is_object()) {
        m.codec.audio = codec->text_or("audio", "");
        if (auto* sr = codec->find("sample_rate"); sr != nullptr && sr->is_int()) {
            m.codec.sample_rate = static_cast<std::uint32_t>(sr->as_int());
        }
        if (auto* ch = codec->find("channels"); ch != nullptr && ch->is_int()) {
            m.codec.channels = static_cast<std::uint32_t>(ch->as_int());
        }
    }
    if (auto* model = v.find("model"); model != nullptr && model->is_object()) {
        m.model.arch = model->text_or("arch", "");
        if (auto* p = model->find("params"); p != nullptr && p->is_int()) {
            m.model.params = static_cast<std::uint64_t>(p->as_int());
        }
        if (auto* p = model->find("bpw"); p != nullptr) {
            if (auto* d = std::get_if<double>(&p->variant())) {
                m.model.bpw = *d;
            } else if (p->is_int()) {
                m.model.bpw = static_cast<double>(p->as_int());
            }
        }
        m.model.sha256 = model->text_or("sha256", "");
    }
    if (auto* tracks = v.find("tracks");
        tracks != nullptr && tracks->is_array()) {
        for (const auto& el : tracks->as_array()) {
            if (!el.is_object()) {
                continue;
            }
            Track t;
            t.id        = el.text_or("id", "");
            t.title     = el.text_or("title", "");
            t.sha256    = el.text_or("sha256", "");
            if (auto* p = el.find("length_ms"); p != nullptr && p->is_int()) {
                t.length_ms = static_cast<std::uint64_t>(p->as_int());
            }
            m.tracks.push_back(std::move(t));
        }
    }
    m.residual_present = v.bool_or("residual_present", false);
    if (auto* p = v.find("residual_sha256"); p != nullptr && p->is_text()) {
        m.residual_sha256 = p->as_text();
    }
    return m;
}

[[nodiscard]] std::uint32_t read_u32_le(const std::uint8_t* p) noexcept
{
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
}
[[nodiscard]] std::uint64_t read_u64_le(const std::uint8_t* p) noexcept
{
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<std::uint64_t>(p[i]) << (i * 8);
    }
    return v;
}

} // namespace

std::expected<ValidateReport, IngestError>
parse_bytes(std::span<const std::uint8_t> buf)
{
    constexpr std::size_t MIN_SIZE     = 4 + 4 + 32;
    constexpr std::size_t HEADER_LIMIT = 16ULL * 1024 * 1024;

    if (buf.size() < MIN_SIZE) {
        return std::unexpected(
            IngestError::truncated("file too small for magic+header+footer"));
    }

    std::array<std::uint8_t, 4> magic{};
    std::copy_n(buf.begin(), 4, magic.begin());
    if (magic[0] != MAGIC[0] || magic[1] != MAGIC[1] || magic[2] != MAGIC[2]) {
        return std::unexpected(IngestError::bad_magic(magic));
    }
    const auto version = magic[3];
    if (version != VERSION) {
        return std::unexpected(IngestError::bad_version(version));
    }

    const auto header_len = read_u32_le(buf.data() + 4);
    if (header_len > HEADER_LIMIT) {
        return std::unexpected(IngestError::header_too_large(header_len));
    }
    const std::size_t header_start = 8;
    const std::size_t header_end   = header_start + header_len;
    if (header_end + 32 > buf.size()) {
        return std::unexpected(IngestError::truncated("header extends past EOF"));
    }
    auto cbor_decode = detail::cbor::decode(
        std::span<const std::uint8_t>{buf.data() + header_start, header_len});
    if (!cbor_decode) {
        return std::unexpected(IngestError::cbor(cbor_decode.error().message));
    }

    const std::size_t          sections_end = buf.size() - 32;
    std::vector<SectionRecord> sections;
    std::size_t                cursor = header_end;
    while (cursor < sections_end) {
        if (sections_end - cursor < 1 + 8) {
            return std::unexpected(IngestError::truncated("partial section header"));
        }
        const auto    tag       = buf[cursor];
        const auto    len       = read_u64_le(buf.data() + cursor + 1);
        const std::size_t payload_start = cursor + 9;
        if (len > sections_end - payload_start) {
            return std::unexpected(
                IngestError::truncated("section payload past sections_end"));
        }
        sections.push_back(SectionRecord{tag,
                                          static_cast<std::uint64_t>(payload_start),
                                          len});
        cursor = payload_start + static_cast<std::size_t>(len);
    }
    if (cursor != sections_end) {
        return std::unexpected(IngestError::truncated("leftover bytes before footer"));
    }

    detail::Sha256 h;
    h.update(std::span<const std::uint8_t>{buf.data(), sections_end});
    const auto expected = h.finalize();
    const bool footer_ok =
        std::equal(expected.begin(), expected.end(), buf.data() + sections_end);
    if (!footer_ok) {
        return std::unexpected(IngestError::footer_mismatch());
    }

    ValidateReport rep;
    rep.total_bytes = static_cast<std::uint64_t>(buf.size());
    rep.version     = version;
    rep.manifest    = cbor_to_manifest(*cbor_decode);
    rep.sections    = std::move(sections);
    rep.footer_ok   = footer_ok;
    return rep;
}

std::expected<ValidateReport, IngestError> validate(const fs::path& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return std::unexpected(IngestError::io(path, "open"));
    }
    in.seekg(0, std::ios::end);
    const auto sz = in.tellg();
    in.seekg(0, std::ios::beg);
    if (sz < 0) {
        return std::unexpected(IngestError::io(path, "tellg"));
    }
    std::vector<std::uint8_t> buf(static_cast<std::size_t>(sz));
    if (sz > 0) {
        in.read(reinterpret_cast<char*>(buf.data()),
                static_cast<std::streamsize>(sz));
        if (!in.good() && !in.eof()) {
            return std::unexpected(IngestError::io(path, "read"));
        }
    }
    return parse_bytes(buf);
}

std::string format_report(const ValidateReport& r)
{
    std::ostringstream o;
    o << "1bl container: " << r.total_bytes << " bytes, version 0x"
      << std::hex << static_cast<int>(r.version) << std::dec << '\n';
    o << "  catalog  : " << r.manifest.catalog << '\n';
    o << "  title    : " << r.manifest.title << '\n';
    o << "  artist   : " << r.manifest.artist << '\n';
    o << "  license  : " << r.manifest.license << '\n';
    o << "  tier     : " << r.manifest.tier << '\n';
    o << "  tracks   : " << r.manifest.tracks.size() << '\n';
    o << "  model    : " << r.manifest.model.arch << " (" << r.manifest.model.params
      << " params, " << r.manifest.model.bpw << " bpw)\n";
    o << "  footer   : " << (r.footer_ok ? "OK" : "MISMATCH") << '\n';
    o << "  sections : " << r.sections.size() << '\n';
    for (const auto& s : r.sections) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
                      "    tag 0x%02x @ offset %10llu  len %12llu\n",
                      static_cast<unsigned>(s.tag),
                      static_cast<unsigned long long>(s.offset),
                      static_cast<unsigned long long>(s.length));
        o << buf;
    }
    return o.str();
}

} // namespace onebit::ingest

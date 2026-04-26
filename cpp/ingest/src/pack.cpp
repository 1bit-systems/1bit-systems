#include "onebit/ingest/ingest.hpp"

#include "cbor.hpp"
#include "onebit/ingest/sha256.hpp"

#include <toml++/toml.hpp>

#include <fstream>
#include <sstream>
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

[[nodiscard]] std::expected<std::string, IngestError> read_text(const fs::path& p)
{
    auto bytes = read_all(p);
    if (!bytes) {
        return std::unexpected(bytes.error());
    }
    return std::string(reinterpret_cast<const char*>(bytes->data()), bytes->size());
}

void write_section(std::vector<std::uint8_t>& out,
                   std::uint8_t               tag,
                   std::span<const std::uint8_t> payload)
{
    out.push_back(tag);
    const auto len = static_cast<std::uint64_t>(payload.size());
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<std::uint8_t>(len >> (i * 8)));
    }
    out.insert(out.end(), payload.begin(), payload.end());
}

[[nodiscard]] detail::cbor::Value manifest_to_cbor(const Manifest& m)
{
    using onebit::ingest::detail::cbor::Array;
    using onebit::ingest::detail::cbor::Object;
    using onebit::ingest::detail::cbor::Value;

    Object obj;
    obj.emplace_back("v", Value{m.v});
    obj.emplace_back("catalog", Value{m.catalog});
    obj.emplace_back("title", Value{m.title});
    obj.emplace_back("artist", Value{m.artist});
    obj.emplace_back("license", Value{m.license});
    if (m.license_url) {
        obj.emplace_back("license_url", Value{*m.license_url});
    }
    if (m.attribution) {
        obj.emplace_back("attribution", Value{*m.attribution});
    }
    if (m.source) {
        obj.emplace_back("source", Value{*m.source});
    }
    obj.emplace_back("created", Value{m.created});
    obj.emplace_back("tier", Value{m.tier});

    Object codec;
    codec.emplace_back("audio", Value{m.codec.audio});
    codec.emplace_back("sample_rate",
                       Value{static_cast<std::int64_t>(m.codec.sample_rate)});
    codec.emplace_back("channels",
                       Value{static_cast<std::int64_t>(m.codec.channels)});
    obj.emplace_back("codec", Value{std::move(codec)});

    Object model;
    model.emplace_back("arch", Value{m.model.arch});
    model.emplace_back("params", Value{static_cast<std::int64_t>(m.model.params)});
    model.emplace_back("bpw", Value{m.model.bpw});
    model.emplace_back("sha256", Value{m.model.sha256});
    obj.emplace_back("model", Value{std::move(model)});

    Array tracks;
    for (const auto& t : m.tracks) {
        Object tt;
        tt.emplace_back("id", Value{t.id});
        tt.emplace_back("title", Value{t.title});
        tt.emplace_back("length_ms", Value{static_cast<std::int64_t>(t.length_ms)});
        tt.emplace_back("sha256", Value{t.sha256});
        tracks.emplace_back(Value{std::move(tt)});
    }
    obj.emplace_back("tracks", Value{std::move(tracks)});

    obj.emplace_back("residual_present", Value{m.residual_present});
    if (m.residual_sha256) {
        obj.emplace_back("residual_sha256", Value{*m.residual_sha256});
    }
    return Value{std::move(obj)};
}

[[nodiscard]] std::optional<std::string>
toml_str(const toml::table& t, std::string_view key)
{
    if (auto* v = t.get_as<std::string>(key)) {
        return v->get();
    }
    return std::nullopt;
}

} // namespace

// Public — also used by tests / future readers that need to interpret a
// hand-written toml file.
std::expected<CatalogToml, IngestError> parse_catalog_toml(std::string_view text)
{
    toml::table tbl;
    try {
        tbl = toml::parse(text);
    } catch (const toml::parse_error& e) {
        return std::unexpected(IngestError::toml(std::string{e.description()}));
    }
    CatalogToml c;
    auto require_str = [&](std::string_view key) -> std::expected<std::string, IngestError> {
        if (auto v = toml_str(tbl, key)) {
            return *v;
        }
        return std::unexpected(IngestError::toml(
            std::string{"missing required string key: "} + std::string{key}));
    };

    if (auto v = require_str("catalog"); v) c.catalog = std::move(*v); else return std::unexpected(v.error());
    if (auto v = require_str("title"); v) c.title = std::move(*v); else return std::unexpected(v.error());
    if (auto v = require_str("artist"); v) c.artist = std::move(*v); else return std::unexpected(v.error());
    if (auto v = require_str("license"); v) c.license = std::move(*v); else return std::unexpected(v.error());
    if (auto v = require_str("created"); v) c.created = std::move(*v); else return std::unexpected(v.error());

    c.license_url      = toml_str(tbl, "license_url");
    c.attribution      = toml_str(tbl, "attribution");
    c.source           = toml_str(tbl, "source");
    c.license_txt      = toml_str(tbl, "license_txt");
    c.license_txt_path = toml_str(tbl, "license_txt_path");
    c.attribution_txt  = toml_str(tbl, "attribution_txt");
    if (auto t = toml_str(tbl, "tier")) {
        c.tier = *t;
    }

    if (auto* codec = tbl.get_as<toml::table>("codec")) {
        if (auto v = toml_str(*codec, "audio")) {
            c.codec.audio = *v;
        }
        if (auto v = codec->get_as<std::int64_t>("sample_rate")) {
            c.codec.sample_rate = static_cast<std::uint32_t>(v->get());
        }
        if (auto v = codec->get_as<std::int64_t>("channels")) {
            c.codec.channels = static_cast<std::uint32_t>(v->get());
        }
    } else {
        return std::unexpected(IngestError::toml("missing [codec] table"));
    }

    if (auto* mdl = tbl.get_as<toml::table>("model")) {
        if (auto v = toml_str(*mdl, "arch")) {
            c.model.arch = *v;
        }
        if (auto v = mdl->get_as<std::int64_t>("params")) {
            c.model.params = static_cast<std::uint64_t>(v->get());
        }
        if (auto v = mdl->get_as<double>("bpw")) {
            c.model.bpw = v->get();
        } else if (auto v = mdl->get_as<std::int64_t>("bpw")) {
            c.model.bpw = static_cast<double>(v->get());
        }
    } else {
        return std::unexpected(IngestError::toml("missing [model] table"));
    }

    if (auto* tracks = tbl.get_as<toml::array>("tracks")) {
        for (auto&& el : *tracks) {
            if (auto* tt = el.as_table()) {
                Track t;
                if (auto v = toml_str(*tt, "id")) {
                    t.id = *v;
                }
                if (auto v = toml_str(*tt, "title")) {
                    t.title = *v;
                }
                if (auto v = tt->get_as<std::int64_t>("length_ms")) {
                    t.length_ms = static_cast<std::uint64_t>(v->get());
                }
                if (auto v = toml_str(*tt, "sha256")) {
                    t.sha256 = *v;
                }
                c.tracks.push_back(std::move(t));
            }
        }
    }

    return c;
}

std::expected<PackSummary, IngestError>
pack(const fs::path&                                     model_path,
     const fs::path&                                     manifest_toml_path,
     const std::optional<fs::path>&                      cover_path,
     const std::optional<fs::path>&                      lyrics_path,
     const fs::path&                                     out_path)
{
    auto gguf_bytes = read_all(model_path);
    if (!gguf_bytes) {
        return std::unexpected(gguf_bytes.error());
    }
    auto toml_text = read_text(manifest_toml_path);
    if (!toml_text) {
        return std::unexpected(toml_text.error());
    }
    auto cat = parse_catalog_toml(*toml_text);
    if (!cat) {
        return std::unexpected(cat.error());
    }

    // Resolve license text — inline first, sidecar second.
    std::string license_txt;
    if (cat->license_txt) {
        license_txt = *cat->license_txt;
    } else if (cat->license_txt_path) {
        const auto base = manifest_toml_path.parent_path();
        const auto p    = base / *cat->license_txt_path;
        auto       t    = read_text(p);
        if (!t) {
            return std::unexpected(t.error());
        }
        license_txt = *t;
    } else {
        return std::unexpected(IngestError::invalid(
            "catalog.toml must set license_txt (inline) or license_txt_path (sidecar)"));
    }

    const std::string attribution_txt =
        cat->attribution_txt ? *cat->attribution_txt
                             : (cat->attribution ? *cat->attribution : cat->artist);

    Manifest m;
    m.v                = "0.1";
    m.catalog          = cat->catalog;
    m.title            = cat->title;
    m.artist           = cat->artist;
    m.license          = cat->license;
    m.license_url      = cat->license_url;
    m.attribution      = cat->attribution;
    m.source           = cat->source;
    m.created          = cat->created;
    m.tier             = cat->tier;
    m.codec            = cat->codec;
    m.model            = cat->model;
    m.model.sha256     = detail::sha256_hex(*gguf_bytes);
    m.tracks           = cat->tracks;
    m.residual_present = false;

    std::optional<std::vector<std::uint8_t>> cover_bytes;
    if (cover_path) {
        auto b = read_all(*cover_path);
        if (!b) {
            return std::unexpected(b.error());
        }
        cover_bytes = std::move(*b);
    }
    std::optional<std::vector<std::uint8_t>> lyrics_bytes;
    if (lyrics_path) {
        auto b = read_all(*lyrics_path);
        if (!b) {
            return std::unexpected(b.error());
        }
        lyrics_bytes = std::move(*b);
    }

    // Build the file in memory; the sizes here are bounded by the
    // GGUF (single-digit GB upper bound for our use). Streaming would
    // be a follow-up if curators ship multi-GB models.
    std::vector<std::uint8_t> out;
    out.reserve(gguf_bytes->size() + 4096);
    out.insert(out.end(), MAGIC.begin(), MAGIC.end());

    auto cbor_bytes = detail::cbor::encode(manifest_to_cbor(m));
    if (cbor_bytes.size() > 0xFFFFFFFFULL) {
        return std::unexpected(IngestError::header_too_large(cbor_bytes.size()));
    }
    const auto h_len = static_cast<std::uint32_t>(cbor_bytes.size());
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<std::uint8_t>(h_len >> (i * 8)));
    }
    out.insert(out.end(), cbor_bytes.begin(), cbor_bytes.end());

    std::size_t section_count = 0;
    write_section(out, tag::MODEL_GGUF, *gguf_bytes);
    ++section_count;
    write_section(out, tag::ATTRIBUTION_TXT,
                  std::span<const std::uint8_t>{
                      reinterpret_cast<const std::uint8_t*>(attribution_txt.data()),
                      attribution_txt.size()});
    ++section_count;
    write_section(out, tag::LICENSE_TXT,
                  std::span<const std::uint8_t>{
                      reinterpret_cast<const std::uint8_t*>(license_txt.data()),
                      license_txt.size()});
    ++section_count;
    if (cover_bytes) {
        write_section(out, tag::COVER, *cover_bytes);
        ++section_count;
    }
    if (lyrics_bytes) {
        write_section(out, tag::TRACK_LYRICS, *lyrics_bytes);
        ++section_count;
    }

    // Footer — SHA-256 over everything written so far.
    const auto digest = detail::sha256(out);
    out.insert(out.end(), digest.begin(), digest.end());

    std::ofstream of(out_path, std::ios::binary | std::ios::trunc);
    if (!of) {
        return std::unexpected(IngestError::io(out_path, "create"));
    }
    of.write(reinterpret_cast<const char*>(out.data()),
             static_cast<std::streamsize>(out.size()));
    if (!of.good()) {
        return std::unexpected(IngestError::io(out_path, "write"));
    }
    of.flush();

    return PackSummary{section_count, static_cast<std::uint64_t>(out.size())};
}

std::string IngestError::what() const
{
    return std::visit(
        [](const auto& e) -> std::string {
            using T = std::decay_t<decltype(e)>;
            if constexpr (std::is_same_v<T, ErrIo>) {
                return "io: " + e.path.string() + ": " + e.message;
            } else if constexpr (std::is_same_v<T, ErrBadMagic>) {
                std::ostringstream o;
                o << "bad magic bytes (got";
                for (auto b : e.got) {
                    o << ' ' << std::hex << static_cast<int>(b);
                }
                o << ')';
                return o.str();
            } else if constexpr (std::is_same_v<T, ErrBadVersion>) {
                std::ostringstream o;
                o << "unsupported version byte: " << std::hex
                  << static_cast<int>(e.version);
                return o.str();
            } else if constexpr (std::is_same_v<T, ErrFooterHashMismatch>) {
                return "footer hash mismatch (file corrupt or truncated)";
            } else if constexpr (std::is_same_v<T, ErrTruncated>) {
                return std::string{"truncated file: "} + e.what;
            } else if constexpr (std::is_same_v<T, ErrHeaderTooLarge>) {
                return "header too large: " + std::to_string(e.bytes) +
                       " bytes (cap is 16 MiB)";
            } else if constexpr (std::is_same_v<T, ErrCbor>) {
                return "cbor decode: " + e.message;
            } else if constexpr (std::is_same_v<T, ErrToml>) {
                return "toml: " + e.message;
            } else if constexpr (std::is_same_v<T, ErrInvalid>) {
                return e.message;
            }
        },
        v_);
}

} // namespace onebit::ingest

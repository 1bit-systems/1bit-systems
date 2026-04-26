#include "onebit/ingest/ingest.hpp"

#include "onebit/ingest/sha256.hpp"
#include "tar.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>

namespace onebit::ingest {

namespace {

namespace fs = std::filesystem;

[[nodiscard]] std::uint64_t unix_now() noexcept
{
    using namespace std::chrono;
    return static_cast<std::uint64_t>(
        duration_cast<seconds>(system_clock::now().time_since_epoch()).count());
}

[[nodiscard]] std::expected<std::pair<std::uint64_t, std::string>, IngestError>
hash_file(const fs::path& p)
{
    std::ifstream in(p, std::ios::binary);
    if (!in) {
        return std::unexpected(IngestError::io(p, "open flac"));
    }
    detail::Sha256 h;
    std::vector<std::uint8_t> buf(64 * 1024);
    std::uint64_t total = 0;
    while (in) {
        in.read(reinterpret_cast<char*>(buf.data()),
                static_cast<std::streamsize>(buf.size()));
        const auto got = in.gcount();
        if (got > 0) {
            h.update(std::span<const std::uint8_t>{buf.data(),
                                                   static_cast<std::size_t>(got)});
            total += static_cast<std::uint64_t>(got);
        }
        if (got <= 0) {
            break;
        }
    }
    if (!in.eof() && in.fail()) {
        return std::unexpected(IngestError::io(p, "read flac"));
    }
    return std::make_pair(total, detail::to_hex(h.finalize()));
}

[[nodiscard]] std::string serialize_manifest_json(const CorpusManifest& m)
{
    // Tiny hand-rolled JSON writer with deterministic key order. The
    // schema is fixed; no need for nlohmann here.
    std::ostringstream o;
    auto               escape = [](const std::string& s) {
        std::string out;
        out.reserve(s.size() + 2);
        for (char c : s) {
            switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char tmp[8];
                    std::snprintf(tmp, sizeof(tmp), "\\u%04x",
                                  static_cast<unsigned>(c));
                    out += tmp;
                } else {
                    out += c;
                }
            }
        }
        return out;
    };
    o << "{\n";
    o << "  \"version\": \"" << escape(m.version) << "\",\n";
    o << "  \"created_unix\": " << m.created_unix << ",\n";
    o << "  \"tool\": \"" << escape(m.tool) << "\",\n";
    o << "  \"entries\": [";
    for (std::size_t i = 0; i < m.entries.size(); ++i) {
        const auto& e = m.entries[i];
        o << (i == 0 ? "\n    " : ",\n    ");
        o << "{\"rel_path\": \"" << escape(e.rel_path)
          << "\", \"size_bytes\": " << e.size_bytes
          << ", \"sha256\": \"" << escape(e.sha256) << "\"}";
    }
    if (!m.entries.empty()) {
        o << "\n  ";
    }
    o << "]\n}";
    return o.str();
}

} // namespace

std::expected<PrepareSummary, IngestError>
prepare(const fs::path& src_dir, const fs::path& out_path)
{
    std::error_code ec;
    const auto      canon = fs::canonical(src_dir, ec);
    if (ec) {
        return std::unexpected(
            IngestError::io(src_dir, "canonicalize: " + ec.message()));
    }

    // Collect *.flac (case-insensitive) and sort. Use increment(ec) so
    // a permission glitch in a subdir doesn't crash the walk.
    std::vector<fs::path> flacs;
    auto                  it = fs::recursive_directory_iterator(
        canon, fs::directory_options::skip_permission_denied, ec);
    if (ec) {
        return std::unexpected(IngestError::io(src_dir, ec.message()));
    }
    const auto end = fs::recursive_directory_iterator{};
    while (it != end) {
        std::error_code step_ec;
        if (it->is_regular_file(step_ec)) {
            const auto& p   = it->path();
            const auto  ext = p.extension().string();
            std::string lc;
            lc.reserve(ext.size());
            for (char c : ext) {
                lc.push_back(static_cast<char>(
                    std::tolower(static_cast<unsigned char>(c))));
            }
            if (lc == ".flac") {
                flacs.push_back(p);
            }
        }
        it.increment(step_ec);
        if (step_ec) {
            break;
        }
    }
    std::sort(flacs.begin(), flacs.end());

    std::vector<FlacEntry> entries;
    entries.reserve(flacs.size());
    std::uint64_t total_bytes = 0;
    for (const auto& p : flacs) {
        auto h = hash_file(p);
        if (!h) {
            return std::unexpected(h.error());
        }
        const auto rel_path = fs::relative(p, canon, ec);
        std::string rel = (ec ? p.string() : rel_path.generic_string());
        // Strip any leading ./
        if (rel.starts_with("./")) {
            rel.erase(0, 2);
        }
        total_bytes += h->first;
        entries.push_back(FlacEntry{std::move(rel), h->first, std::move(h->second)});
    }

    CorpusManifest manifest;
    manifest.version      = "0.1";
    manifest.created_unix = unix_now();
    manifest.tool         = "1bit-ingest/0.1.0";
    manifest.entries      = entries;

    const auto manifest_json = serialize_manifest_json(manifest);

    std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return std::unexpected(IngestError::io(out_path, "create out tar"));
    }
    detail::tar::Writer tw{out};
    tw.append_blob("manifest.json",
                   std::span<const std::uint8_t>{
                       reinterpret_cast<const std::uint8_t*>(manifest_json.data()),
                       manifest_json.size()});
    if (!tw.ok()) {
        return std::unexpected(IngestError::io(out_path, tw.error()));
    }
    for (std::size_t i = 0; i < flacs.size(); ++i) {
        if (!tw.append_path_with_name(flacs[i], entries[i].rel_path)) {
            return std::unexpected(IngestError::io(flacs[i], tw.error()));
        }
    }
    tw.finish();
    if (!tw.ok()) {
        return std::unexpected(IngestError::io(out_path, tw.error()));
    }
    out.flush();

    return PrepareSummary{entries.size(), total_bytes};
}

} // namespace onebit::ingest

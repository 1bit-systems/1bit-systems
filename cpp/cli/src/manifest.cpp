#include "onebit/cli/registry.hpp"

#include <toml++/toml.hpp>

#include <cstring>
#include <fstream>
#include <sstream>
#include <utility>

namespace onebit::cli {

namespace {

[[nodiscard]] std::string toml_string_at(const toml::table& tbl, std::string_view key,
                                          std::string_view dflt = "")
{
    const auto* node = tbl.get(key);
    if (node == nullptr) return std::string(dflt);
    if (const auto* sv = node->as_string()) return std::string(sv->get());
    return std::string(dflt);
}

[[nodiscard]] std::vector<std::string>
toml_string_array(const toml::table& tbl, std::string_view key)
{
    std::vector<std::string> out;
    const auto* node = tbl.get(key);
    if (node == nullptr) return out;
    const auto* arr = node->as_array();
    if (arr == nullptr) return out;
    out.reserve(arr->size());
    for (const auto& el : *arr) {
        if (const auto* s = el.as_string()) {
            out.emplace_back(s->get());
        }
    }
    return out;
}

[[nodiscard]] std::vector<std::vector<std::string>>
toml_string_array_array(const toml::table& tbl, std::string_view key)
{
    std::vector<std::vector<std::string>> out;
    const auto* node = tbl.get(key);
    if (node == nullptr) return out;
    const auto* arr = node->as_array();
    if (arr == nullptr) return out;
    out.reserve(arr->size());
    for (const auto& el : *arr) {
        const auto* sub = el.as_array();
        if (sub == nullptr) continue;
        std::vector<std::string> step;
        step.reserve(sub->size());
        for (const auto& tok : *sub) {
            if (const auto* s = tok.as_string()) {
                step.emplace_back(s->get());
            }
        }
        out.push_back(std::move(step));
    }
    return out;
}

[[nodiscard]] std::vector<FileEntry>
parse_files_array(const toml::table& tbl, std::string_view key, Error& err_sink)
{
    std::vector<FileEntry> out;
    const auto* node = tbl.get(key);
    if (node == nullptr) return out;
    const auto* arr = node->as_array();
    if (arr == nullptr) return out;

    for (const auto& el : *arr) {
        // Two accepted shapes — pair-array or table.
        if (const auto* pair = el.as_array()) {
            if (pair->size() < 2) {
                err_sink = Error::schema(
                    std::string("files entry pair must be [src, dest]"));
                return {};
            }
            const auto* s_node = pair->get(0);
            const auto* d_node = pair->get(1);
            if (s_node == nullptr || d_node == nullptr) continue;
            const auto* s_str = s_node->as_string();
            const auto* d_str = d_node->as_string();
            if (s_str == nullptr || d_str == nullptr) {
                err_sink = Error::schema("files pair must be string,string");
                return {};
            }
            out.push_back(FileEntry{
                std::string(s_str->get()),
                std::string(d_str->get()),
                {},
            });
            continue;
        }
        if (const auto* tab = el.as_table()) {
            FileEntry e;
            e.src = toml_string_at(*tab, "src");
            e.dst = toml_string_at(*tab, "dst");
            if (e.src.empty() || e.dst.empty()) {
                err_sink = Error::schema("files table must have src + dst");
                return {};
            }
            if (const auto* sub_node = tab->get("substitute")) {
                if (const auto* sub_tab = sub_node->as_table()) {
                    for (const auto& [k, v] : *sub_tab) {
                        if (const auto* vs = v.as_string()) {
                            e.substitute.emplace(std::string(k.str()),
                                                 std::string(vs->get()));
                        }
                    }
                }
            }
            out.push_back(std::move(e));
            continue;
        }
        err_sink = Error::schema("files entry must be array or table");
        return {};
    }
    return out;
}

}  // namespace

std::expected<Manifest, Error>
parse_manifest_str(std::string_view toml_str, PackageOrigin origin)
{
    toml::table root;
    try {
        root = toml::parse(toml_str);
    } catch (const toml::parse_error& e) {
        std::ostringstream os;
        os << "parse error: " << e.description();
        if (const auto& src = e.source(); src.begin.line != 0) {
            os << " at line " << src.begin.line;
        }
        return std::unexpected(Error::parse(os.str()));
    }

    Manifest m;

    if (const auto* comp_node = root.get("component")) {
        const auto* comp_tab = comp_node->as_table();
        if (comp_tab == nullptr) {
            return std::unexpected(Error::schema("[component] must be a table"));
        }
        for (const auto& [name, sub] : *comp_tab) {
            const auto* st = sub.as_table();
            if (st == nullptr) continue;

            Component c;
            c.name        = std::string(name.str());
            c.description = toml_string_at(*st, "description");
            c.deps        = toml_string_array(*st, "deps");
            c.build       = toml_string_array_array(*st, "build");
            c.units       = toml_string_array(*st, "units");
            c.packages    = toml_string_array(*st, "packages");
            Error file_err;
            c.files       = parse_files_array(*st, "files", file_err);
            if (!file_err.message.empty()) {
                return std::unexpected(file_err);
            }
            c.check       = toml_string_at(*st, "check");
            c.origin      = origin;

            m.components.emplace(c.name, std::move(c));
        }
    }

    if (const auto* model_node = root.get("model")) {
        const auto* model_tab = model_node->as_table();
        if (model_tab == nullptr) {
            return std::unexpected(Error::schema("[model] must be a table"));
        }
        for (const auto& [id, sub] : *model_tab) {
            const auto* st = sub.as_table();
            if (st == nullptr) continue;

            Model mm;
            mm.id          = std::string(id.str());
            mm.description = toml_string_at(*st, "description");
            mm.hf_repo     = toml_string_at(*st, "hf_repo");
            mm.hf_file     = toml_string_at(*st, "hf_file");
            mm.sha256      = toml_string_at(*st, "sha256");
            mm.license     = toml_string_at(*st, "license");
            mm.requires_   = toml_string_array(*st, "requires");
            if (const auto* sm = st->get("size_mb")) {
                if (const auto* iv = sm->as_integer()) {
                    const std::int64_t n = iv->get();
                    mm.size_mb = n < 0 ? 0 : static_cast<std::uint64_t>(n);
                }
            }
            mm.origin = origin;
            m.models.emplace(mm.id, std::move(mm));
        }
    }

    return m;
}

std::expected<Manifest, Error>
parse_manifest_file(const std::filesystem::path& path, PackageOrigin origin)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return std::unexpected(
            Error::io("cannot open manifest: " + path.string()));
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return parse_manifest_str(ss.str(), origin);
}

}  // namespace onebit::cli

#include "onebit/power/profile.hpp"

#include <fstream>
#include <sstream>
#include <utility>

#include <toml++/toml.hpp>

namespace onebit::power {

namespace {

// Load a single inner profile table, e.g. [balanced]. Returns false +
// fills `err` if any field is non-integer or out-of-range.
bool load_one(const toml::table& t, Profile& out, std::string& err)
{
    auto pull = [&](std::string_view k, std::optional<std::uint32_t>& dst) -> bool {
        const toml::node* n = t.get(k);
        if (!n) return true;
        if (auto v = n->value<std::int64_t>()) {
            if (*v < 0 || *v > std::numeric_limits<std::uint32_t>::max()) {
                err = std::string{"value for `"} + std::string{k} + "` out of u32 range";
                return false;
            }
            dst = static_cast<std::uint32_t>(*v);
            return true;
        }
        err = std::string{"`"} + std::string{k} + "` must be an integer";
        return false;
    };

    if (!pull("stapm_limit",       out.stapm_limit))       return false;
    if (!pull("fast_limit",        out.fast_limit))        return false;
    if (!pull("slow_limit",        out.slow_limit))        return false;
    if (!pull("tctl_temp",         out.tctl_temp))         return false;
    if (!pull("vrm_current",       out.vrm_current))       return false;
    if (!pull("vrmmax_current",    out.vrmmax_current))    return false;
    if (!pull("vrmsoc_current",    out.vrmsoc_current))    return false;
    if (!pull("vrmsocmax_current", out.vrmsocmax_current)) return false;

    // Mimic the Rust deny_unknown_fields behaviour at the inner level:
    // any extra key is flagged.
    static constexpr std::array<std::string_view, 8> KNOWN{
        "stapm_limit", "fast_limit", "slow_limit", "tctl_temp",
        "vrm_current", "vrmmax_current", "vrmsoc_current", "vrmsocmax_current",
    };
    for (auto&& [k, _] : t) {
        bool ok = false;
        for (auto kn : KNOWN) {
            if (k.str() == kn) { ok = true; break; }
        }
        if (!ok) {
            err = std::string{"unknown knob `"} + std::string{k.str()} + "`";
            return false;
        }
    }
    return true;
}

} // namespace

Result<Profiles> Profiles::parse(std::string_view src)
{
    toml::table tbl;
    try {
        tbl = toml::parse(src);
    } catch (const toml::parse_error& e) {
        std::ostringstream os;
        os << "toml parse: " << e.description();
        return Status::fail(Error::ParseError, os.str());
    }

    Profiles out;
    for (auto&& [name, node] : tbl) {
        const toml::table* inner = node.as_table();
        if (!inner) {
            return Status::fail(Error::ParseError,
                std::string{"top-level entry `"} + std::string{name.str()}
                    + "` is not a table");
        }
        Profile p{};
        std::string err;
        if (!load_one(*inner, p, err)) {
            return Status::fail(Error::ParseError,
                std::string{"in [" } + std::string{name.str()} + "]: " + err);
        }
        out.map_.emplace(std::string{name.str()}, std::move(p));
    }
    return out;
}

Result<Profiles> Profiles::load(std::string_view path)
{
    std::ifstream f{std::string{path}};
    if (!f) {
        return Status::fail(Error::IoError,
            std::string{"reading profiles from "} + std::string{path});
    }
    std::ostringstream buf;
    buf << f.rdbuf();
    return parse(buf.str());
}

const Profile* Profiles::get(std::string_view name) const noexcept
{
    auto it = map_.find(name);
    return it == map_.end() ? nullptr : &it->second;
}

std::vector<std::string> Profiles::names() const
{
    std::vector<std::string> out;
    out.reserve(map_.size());
    for (const auto& [k, _] : map_) out.push_back(k);
    return out;
}

} // namespace onebit::power

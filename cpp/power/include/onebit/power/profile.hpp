#pragma once

// On-disk TOML profile table loader.
//
// Mirrors the Rust crate's `profiles.rs`: each profile is a struct of
// std::optional<std::uint32_t> knobs; an absent knob means "RyzenAdj
// leaves this MSR alone". The TOML at /etc/halo-power/profiles.toml is
// a flat map of profile-name → table-of-knobs.

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "onebit/power/result.hpp"

namespace onebit::power {

struct Profile {
    std::optional<std::uint32_t> stapm_limit;
    std::optional<std::uint32_t> fast_limit;
    std::optional<std::uint32_t> slow_limit;
    std::optional<std::uint32_t> tctl_temp;
    std::optional<std::uint32_t> vrm_current;
    std::optional<std::uint32_t> vrmmax_current;
    std::optional<std::uint32_t> vrmsoc_current;
    std::optional<std::uint32_t> vrmsocmax_current;

    [[nodiscard]] bool empty() const noexcept
    {
        return !stapm_limit && !fast_limit && !slow_limit && !tctl_temp
            && !vrm_current && !vrmmax_current
            && !vrmsoc_current && !vrmsocmax_current;
    }
};

class Profiles {
public:
    // Load + parse a profiles.toml from disk.
    static Result<Profiles> load(std::string_view path);

    // Parse from an in-memory string (used by tests + load()).
    static Result<Profiles> parse(std::string_view toml_src);

    [[nodiscard]] const Profile* get(std::string_view name) const noexcept;
    [[nodiscard]] std::vector<std::string> names() const;
    [[nodiscard]] bool empty() const noexcept { return map_.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return map_.size(); }

private:
    // BTreeMap<String, Profile> in Rust → std::map for stable iteration.
    std::map<std::string, Profile, std::less<>> map_;
};

} // namespace onebit::power

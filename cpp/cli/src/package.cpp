#include "onebit/cli/package.hpp"

#include "onebit/cli/paths.hpp"

#include <array>
#include <cstdlib>
#include <string>
#include <string_view>

namespace onebit::cli {

bool is_placeholder_seed(std::string_view raw) noexcept
{
    return raw == "$USER" || raw == "$HOME";
}

namespace {

// Strict substring substitution table. Each entry maps a literal placeholder
// token to a callable that returns the resolved value. Strict semantics:
//
//   * No shell-style fallback (`${VAR:-default}`) — every match is a
//     plain substring replace; the literal `:-` would be treated as part
//     of a non-matching tail.
//   * No command substitution (`$(cmd)`) — never spawn a subprocess.
//   * No globbing — `*`, `?`, brace-expansion are passthrough literals.
//   * Longest-prefix-first ordering: `${HOME}` matches before `$HOME`
//     so `$HOMEY` is left alone (no false-prefix on `$HOME`).
//
// The audit (cpp/cli/src/package.cpp:13-28) called out `${HOME}` substring
// expansion as the missing piece — every install recipe in packages.toml
// uses `${HOME}/.local/bin/...` and the literal slipped through to
// `install -Dm755` verbatim on a fresh box.
struct Token {
    std::string_view  needle;
    const char*       env_key;
    bool              fall_back_to_home;  ///< for $XDG_* tokens
    std::string_view  home_suffix;        ///< only used when fall_back_to_home
};

[[nodiscard]] std::string resolve(const Token& t)
{
    if (const char* v = std::getenv(t.env_key); v != nullptr && *v != '\0') {
        return v;
    }
    if (t.fall_back_to_home) {
        // XDG_CONFIG_HOME defaults to $HOME/.config; XDG_DATA_HOME to
        // $HOME/.local/share. `paths.cpp` already encodes the same
        // fallback for filesystem paths; we mirror it on the string side.
        return (home_dir() / std::filesystem::path(t.home_suffix)).string();
    }
    return {};
}

[[nodiscard]] bool replace_all(std::string& s,
                                std::string_view needle,
                                std::string_view value)
{
    if (needle.empty()) return false;
    bool any = false;
    std::size_t pos = 0;
    while ((pos = s.find(needle, pos)) != std::string::npos) {
        s.replace(pos, needle.size(), value);
        pos += value.size();
        any  = true;
    }
    return any;
}

}  // namespace

std::string expand_placeholder(std::string_view raw)
{
    // Longest braced form first so `${HOME}` is consumed before `$HOME`
    // would match its prefix. `$USER` / `$HOME` keep their original
    // whole-string-equality semantics for back-compat with the existing
    // seed-resolution callers (substitute = { USER = "$USER" }).
    static constexpr std::array<Token, 6> kTokens = {{
        {"${XDG_CONFIG_HOME}", "XDG_CONFIG_HOME", true,  ".config"},
        {"${XDG_DATA_HOME}",   "XDG_DATA_HOME",   true,  ".local/share"},
        {"${HOME}",            "HOME",            false, {}},
        {"${USER}",            "USER",            false, {}},
        {"$HOME",              "HOME",            false, {}},
        {"$USER",              "USER",            false, {}},
    }};

    std::string out(raw);
    for (const auto& t : kTokens) {
        // Skip the costly resolve() if the needle isn't present at all.
        if (out.find(t.needle) == std::string::npos) continue;
        const std::string value = resolve(t);
        replace_all(out, t.needle, value);
    }
    return out;
}

}  // namespace onebit::cli

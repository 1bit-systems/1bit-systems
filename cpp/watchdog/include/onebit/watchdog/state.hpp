#pragma once

// Persistent state for 1bit-watchdog. Per watch entry we record the SHA we
// saw on the most recent poll, when we first saw a SHA different from the
// last_merged one, and the last SHA we propagated. Logic mirrors the Rust
// crate verbatim: dwell timer arms on first divergence, fires after
// soak_hours.

#include <chrono>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

namespace onebit::watchdog {

using Clock     = std::chrono::system_clock;
using TimePoint = Clock::time_point;

struct EntryState {
    std::optional<std::string> last_seen_sha;
    std::optional<TimePoint>   first_seen_at;
    std::optional<std::string> last_merged_sha;
    std::optional<TimePoint>   last_merged_at;
};

// Discriminated union of poll outcomes. Tag is the active variant; only
// `Soaking` carries a payload.
struct Transition {
    enum class Kind {
        NoChange,
        SeenNew,
        Soaking,
        SoakComplete,
    };
    Kind         kind            = Kind::NoChange;
    std::int64_t remaining_hours = 0; // valid when kind == Soaking
};

[[nodiscard]] std::string_view to_string(Transition::Kind k) noexcept;

enum class StateError {
    ReadFailed,
    ParseFailed,
    WriteFailed,
};

class State {
public:
    State() = default;

    // Load from JSON file. Returns nullopt on failure.
    static std::optional<State> load(std::string_view path,
                                     StateError* err = nullptr);

    // Persist to JSON file (creating parents). Returns false on failure.
    bool save(std::string_view path, StateError* err = nullptr) const;

    [[nodiscard]] const std::map<std::string, EntryState>& entries() const noexcept
    {
        return entries_;
    }

    void reset(std::string_view id);
    void mark_merged(std::string_view id, TimePoint now);

    // Records `latest` as the last_seen_sha for `id` and returns the
    // resulting transition, using `now` as the wall clock.
    Transition observe(std::string_view id,
                       std::string_view latest,
                       std::uint32_t    soak_hours,
                       TimePoint        now);

    // Convenience overload: uses Clock::now().
    Transition observe(std::string_view id,
                       std::string_view latest,
                       std::uint32_t    soak_hours);

    // Serialize the full state to pretty JSON (used by `status` subcommand).
    [[nodiscard]] std::string to_json_pretty() const;

private:
    std::map<std::string, EntryState> entries_;
};

// Default state-file path: $XDG_STATE_HOME/1bit-watchdog/state.json,
// falling back to $HOME/.local/state/1bit-watchdog/state.json, then
// /tmp/.local/state/1bit-watchdog/state.json.
[[nodiscard]] std::string default_state_path();

// Default manifest path: $XDG_CONFIG_HOME/1bit/packages.toml, falling back
// to $HOME/.config/1bit/packages.toml, then ./packages.toml.
[[nodiscard]] std::string default_manifest_path();

// RFC3339 helpers (chrono <-> string).
[[nodiscard]] std::string                to_iso8601(TimePoint t);
[[nodiscard]] std::optional<TimePoint>   from_iso8601(std::string_view s);

} // namespace onebit::watchdog

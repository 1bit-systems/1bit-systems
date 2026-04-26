#include "onebit/watchdog/state.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

namespace onebit::watchdog {

namespace {

using json = nlohmann::json;
namespace fs = std::filesystem;

const char* getenv_or(const char* k, const char* fallback)
{
    const char* v = std::getenv(k);
    return (v && *v) ? v : fallback;
}

} // namespace

std::string_view to_string(Transition::Kind k) noexcept
{
    switch (k) {
        case Transition::Kind::NoChange:     return "NoChange";
        case Transition::Kind::SeenNew:      return "SeenNew";
        case Transition::Kind::Soaking:      return "Soaking";
        case Transition::Kind::SoakComplete: return "SoakComplete";
    }
    return "?";
}

std::string default_state_path()
{
    if (const char* xdg = std::getenv("XDG_STATE_HOME"); xdg && *xdg) {
        return (fs::path(xdg) / "1bit-watchdog" / "state.json").string();
    }
    const char* home = getenv_or("HOME", "/tmp");
    return (fs::path(home) / ".local" / "state" / "1bit-watchdog" / "state.json").string();
}

std::string default_manifest_path()
{
    if (const char* xdg = std::getenv("XDG_CONFIG_HOME"); xdg && *xdg) {
        return (fs::path(xdg) / "1bit" / "packages.toml").string();
    }
    if (const char* home = std::getenv("HOME"); home && *home) {
        return (fs::path(home) / ".config" / "1bit" / "packages.toml").string();
    }
    return "packages.toml";
}

std::string to_iso8601(TimePoint t)
{
    using namespace std::chrono;
    const auto secs    = time_point_cast<seconds>(t);
    const auto fract   = duration_cast<microseconds>(t - secs);
    const auto tt      = Clock::to_time_t(secs);
    std::tm tm_utc{};
#if defined(_WIN32)
    gmtime_s(&tm_utc, &tt);
#else
    gmtime_r(&tt, &tm_utc);
#endif
    char buf[64];
    std::snprintf(buf, sizeof(buf),
                  "%04d-%02d-%02dT%02d:%02d:%02d.%06lldZ",
                  tm_utc.tm_year + 1900,
                  tm_utc.tm_mon + 1,
                  tm_utc.tm_mday,
                  tm_utc.tm_hour,
                  tm_utc.tm_min,
                  tm_utc.tm_sec,
                  static_cast<long long>(fract.count()));
    return std::string(buf);
}

std::optional<TimePoint> from_iso8601(std::string_view s)
{
    // Accept YYYY-MM-DDTHH:MM:SS[.ffffff][Z|+HH:MM|-HH:MM]
    if (s.size() < 19) return std::nullopt;
    int Y = 0, M = 0, D = 0, h = 0, m = 0, sec = 0;
    long long us = 0;
    if (std::sscanf(std::string(s.substr(0, 19)).c_str(),
                    "%4d-%2d-%2dT%2d:%2d:%2d", &Y, &M, &D, &h, &m, &sec) != 6) {
        return std::nullopt;
    }
    std::size_t pos = 19;
    if (pos < s.size() && s[pos] == '.') {
        ++pos;
        std::size_t start = pos;
        while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) {
            ++pos;
        }
        std::string frac(s.substr(start, pos - start));
        // pad/truncate to microseconds (6 digits)
        if (frac.size() > 6) frac.resize(6);
        while (frac.size() < 6) frac.push_back('0');
        try { us = std::stoll(frac); } catch (...) { us = 0; }
    }
    int tz_offset_min = 0;
    if (pos < s.size()) {
        char z = s[pos];
        if (z == 'Z') {
            // utc
        } else if (z == '+' || z == '-') {
            if (s.size() < pos + 6) return std::nullopt;
            int hh = 0, mm = 0;
            if (std::sscanf(std::string(s.substr(pos + 1, 5)).c_str(),
                            "%2d:%2d", &hh, &mm) != 2) {
                return std::nullopt;
            }
            tz_offset_min = (hh * 60 + mm) * (z == '+' ? 1 : -1);
        }
    }

    std::tm tm_utc{};
    tm_utc.tm_year = Y - 1900;
    tm_utc.tm_mon  = M - 1;
    tm_utc.tm_mday = D;
    tm_utc.tm_hour = h;
    tm_utc.tm_min  = m;
    tm_utc.tm_sec  = sec;

#if defined(_WIN32)
    auto tt = _mkgmtime(&tm_utc);
#else
    auto tt = timegm(&tm_utc);
#endif
    if (tt == static_cast<std::time_t>(-1)) return std::nullopt;
    auto tp = Clock::from_time_t(tt) + std::chrono::microseconds(us)
              - std::chrono::minutes(tz_offset_min);
    return tp;
}

namespace {

json entry_to_json(const EntryState& e)
{
    json j = json::object();
    j["last_seen_sha"]   = e.last_seen_sha   ? json(*e.last_seen_sha)   : json(nullptr);
    j["first_seen_at"]   = e.first_seen_at   ? json(to_iso8601(*e.first_seen_at)) : json(nullptr);
    j["last_merged_sha"] = e.last_merged_sha ? json(*e.last_merged_sha) : json(nullptr);
    j["last_merged_at"]  = e.last_merged_at  ? json(to_iso8601(*e.last_merged_at))  : json(nullptr);
    return j;
}

EntryState entry_from_json(const json& j)
{
    EntryState e;
    if (j.contains("last_seen_sha") && j["last_seen_sha"].is_string()) {
        e.last_seen_sha = j["last_seen_sha"].get<std::string>();
    }
    if (j.contains("first_seen_at") && j["first_seen_at"].is_string()) {
        e.first_seen_at = from_iso8601(j["first_seen_at"].get<std::string>());
    }
    if (j.contains("last_merged_sha") && j["last_merged_sha"].is_string()) {
        e.last_merged_sha = j["last_merged_sha"].get<std::string>();
    }
    if (j.contains("last_merged_at") && j["last_merged_at"].is_string()) {
        e.last_merged_at = from_iso8601(j["last_merged_at"].get<std::string>());
    }
    return e;
}

} // namespace

std::optional<State> State::load(std::string_view path, StateError* err)
{
    std::ifstream ifs{std::string(path)};
    if (!ifs.is_open()) {
        if (err) *err = StateError::ReadFailed;
        return std::nullopt;
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();

    json j;
    try {
        j = json::parse(oss.str());
    } catch (const json::parse_error&) {
        if (err) *err = StateError::ParseFailed;
        return std::nullopt;
    }
    State s;
    if (j.contains("entries") && j["entries"].is_object()) {
        for (auto it = j["entries"].begin(); it != j["entries"].end(); ++it) {
            s.entries_.emplace(it.key(), entry_from_json(it.value()));
        }
    }
    return s;
}

bool State::save(std::string_view path, StateError* err) const
{
    fs::path p{std::string(path)};
    if (p.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(p.parent_path(), ec);
        if (ec) {
            if (err) *err = StateError::WriteFailed;
            return false;
        }
    }
    std::ofstream ofs{p, std::ios::trunc};
    if (!ofs.is_open()) {
        if (err) *err = StateError::WriteFailed;
        return false;
    }
    ofs << to_json_pretty();
    if (!ofs) {
        if (err) *err = StateError::WriteFailed;
        return false;
    }
    return true;
}

std::string State::to_json_pretty() const
{
    json out = json::object();
    json entries = json::object();
    for (const auto& [id, e] : entries_) {
        entries[id] = entry_to_json(e);
    }
    out["entries"] = std::move(entries);
    return out.dump(2);
}

void State::reset(std::string_view id)
{
    auto it = entries_.find(std::string(id));
    if (it != entries_.end()) {
        it->second.first_seen_at.reset();
        it->second.last_seen_sha.reset();
    }
}

void State::mark_merged(std::string_view id, TimePoint now)
{
    auto& e = entries_[std::string(id)];
    e.last_merged_sha = e.last_seen_sha;
    e.last_merged_at  = now;
    e.first_seen_at.reset();
}

Transition State::observe(std::string_view id,
                          std::string_view latest,
                          std::uint32_t    soak_hours,
                          TimePoint        now)
{
    auto& e = entries_[std::string(id)];
    e.last_seen_sha = std::string(latest);

    if (e.last_merged_sha && *e.last_merged_sha == latest) {
        e.first_seen_at.reset();
        return Transition{Transition::Kind::NoChange, 0};
    }

    if (!e.first_seen_at) {
        e.first_seen_at = now;
        return Transition{Transition::Kind::SeenNew, 0};
    }

    const auto dwell  = now - *e.first_seen_at;
    const auto target = std::chrono::hours(static_cast<std::int64_t>(soak_hours));
    if (dwell >= target) {
        return Transition{Transition::Kind::SoakComplete, 0};
    }
    const auto remaining =
        std::chrono::duration_cast<std::chrono::hours>(target - dwell).count();
    return Transition{Transition::Kind::Soaking,
                      remaining < 0 ? 0 : remaining};
}

Transition State::observe(std::string_view id,
                          std::string_view latest,
                          std::uint32_t    soak_hours)
{
    return observe(id, latest, soak_hours, Clock::now());
}

} // namespace onebit::watchdog

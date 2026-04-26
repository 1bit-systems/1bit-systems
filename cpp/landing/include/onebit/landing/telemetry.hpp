#pragma once

// Mirrors crates/1bit-landing/src/telemetry.rs.
//
// Aggregates lemond + rocm-smi + xrt-smi + systemctl + shadow-burnin
// into a single `Stats` snapshot serialized on `/_live/stats`. Sources
// are injected via `Sources` — production wiring uses the real binaries,
// tests point them at /nonexistent paths to exercise the offline-degrade
// path. Every source must degrade silently to a sentinel (the endpoint
// must never 5xx — invariant 5).

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::landing {

// User-scope systemd units probed on /_live/services. Mirrors
// crates/1bit-landing/src/telemetry.rs::TRACKED_SERVICES verbatim.
inline constexpr std::array<std::string_view, 10> TRACKED_SERVICES = {
    "1bit-halo-lemonade",
    "1bit-halo-bitnet",
    "1bit-halo-sd",
    "1bit-halo-whisper",
    "1bit-halo-kokoro",
    "1bit-halo-agent",
    "strix-landing",
    "strix-echo",
    "strix-burnin",
    "strix-cloudflared",
};

struct ServiceState {
    std::string name;
    bool        active{false};

    [[nodiscard]] bool operator==(const ServiceState&) const = default;
};

struct Stats {
    std::string                loaded_model{};
    float                      tok_s_decode{0.0F};
    float                      gpu_temp_c{0.0F};
    std::uint8_t               gpu_util_pct{0};
    bool                       npu_up{false};
    float                      shadow_burn_exact_pct{0.0F};
    std::vector<ServiceState>  services{};
    bool                       stale{true};

    [[nodiscard]] static Stats empty() { return Stats{}; }

    // Field order matches the Rust serde_json output exactly. Monitoring
    // scripts depend on this — do not rearrange.
    [[nodiscard]] nlohmann::json to_json() const
    {
        nlohmann::json j;
        j["loaded_model"]          = loaded_model;
        j["tok_s_decode"]          = tok_s_decode;
        j["gpu_temp_c"]             = gpu_temp_c;
        j["gpu_util_pct"]          = gpu_util_pct;
        j["npu_up"]                = npu_up;
        j["shadow_burn_exact_pct"] = shadow_burn_exact_pct;
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& s : services) {
            arr.push_back(nlohmann::json{ {"name", s.name}, {"active", s.active} });
        }
        j["services"] = std::move(arr);
        j["stale"]    = stale;
        return j;
    }
};

// Pluggable source bundle. Same shape as the Rust `Sources` struct so the
// test surface is recognisable. Default-constructed values reproduce the
// Rust `Default::default()` impl.
struct Sources {
    // HTTP base URL for the lemond probe. Tests aim this at TEST-NET-1
    // (192.0.2.1) so probes fail fast without network.
    std::string             onebit_server_base{"http://127.0.0.1:8180"};
    std::filesystem::path   rocm_smi_bin{"rocm-smi"};
    std::filesystem::path   xrt_smi_bin{"xrt-smi"};
    std::filesystem::path   accel_dev{"/dev/accel/accel0"};
    std::filesystem::path   shadow_burnin_jsonl{}; // populated in default factory
    std::filesystem::path   systemctl_bin{"systemctl"};
    std::vector<std::string> services{};            // populated in default factory

    // Returns a Sources prepopulated with TRACKED_SERVICES + the canonical
    // ~/claude output/shadow-burnin.jsonl path (HOME-aware).
    [[nodiscard]] static Sources defaults();
};

// Pure helper — emit just the rows whose `active` flipped between `prev`
// and `next`, or std::nullopt if nothing changed. Exposed for tests.
[[nodiscard]] std::optional<std::vector<ServiceState>>
service_delta(const std::vector<ServiceState>& prev,
              const std::vector<ServiceState>& next);

// Parse a rocm-smi 6.x JSON blob and pull (edge_temp_c, gpu_use_pct).
// Sentinel (0.0F, 0) on any shape mismatch.
[[nodiscard]] std::pair<float, std::uint8_t>
parse_rocm_smi_json(const nlohmann::json& v);

// ~1 s TTL cache wrapping the source fan-out.
class Telemetry {
public:
    explicit Telemetry(Sources sources,
                       std::chrono::milliseconds ttl = std::chrono::milliseconds{1000});

    // Cache-aware snapshot — cheap if called within `ttl` of the last
    // collect.
    [[nodiscard]] Stats snapshot();

    // Bypass the cache. Used by tests + (in Rust) the SSE poll path,
    // which already throttles externally.
    [[nodiscard]] Stats collect();

    [[nodiscard]] const Sources& sources() const noexcept { return sources_; }

private:
    Sources                                      sources_;
    std::chrono::milliseconds                    ttl_;
    mutable std::mutex                           mu_;
    Stats                                        cached_{};
    std::optional<std::chrono::steady_clock::time_point> fetched_at_{};
};

// ---- Source-level probe entry points (free functions for testability) ----

// HTTP probes (lemond):
[[nodiscard]] std::optional<std::string> probe_model(const Sources& s);
[[nodiscard]] std::optional<float>       probe_tokps(const Sources& s);

// Subprocess + filesystem probes:
[[nodiscard]] std::pair<float, std::uint8_t> probe_rocm_smi(const Sources& s);
[[nodiscard]] bool                            probe_npu(const Sources& s);
[[nodiscard]] float                           probe_shadow_burn(const std::filesystem::path& p);
[[nodiscard]] std::vector<ServiceState>       probe_services(const Sources& s);

} // namespace onebit::landing

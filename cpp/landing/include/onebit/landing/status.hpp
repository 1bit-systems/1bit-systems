#pragma once

// Mirrors crates/1bit-landing/src/status.rs.
//
// Live-status probe for the gen-2 lemond server on :8180. The shape
// returned to the browser is byte-stable with the Rust port — monitoring
// scripts depend on field names + ordering.
//
// `/_live/status` never 5xx's. On any backend failure we serialize an
// `offline()` snapshot: v2_up=false, model="", numeric fields 0.

#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>

namespace onebit::landing {

struct LiveStatus {
    bool        v2_up{false};
    bool        v1_up{false};
    std::string model{};
    double      tokps{0.0};
    double      p50_ms{0.0};
    double      p95_ms{0.0};
    std::uint64_t requests{0};
    std::uint64_t generated_tokens{0};

    [[nodiscard]] static LiveStatus offline() noexcept { return LiveStatus{}; }

    // Field order matches the Rust serde_json output exactly.
    [[nodiscard]] nlohmann::json to_json() const
    {
        nlohmann::json j;
        j["v2_up"]            = v2_up;
        j["v1_up"]            = v1_up;
        j["model"]            = model;
        j["tokps"]            = tokps;
        j["p50_ms"]           = p50_ms;
        j["p95_ms"]           = p95_ms;
        j["requests"]         = requests;
        j["generated_tokens"] = generated_tokens;
        return j;
    }
};

// Probe injection point. Default impl issues HTTP GET /v1/models +
// /metrics against `lemond_url` with a 2 s timeout. Tests pass a stub.
class LemondProbe {
public:
    virtual ~LemondProbe() = default;
    [[nodiscard]] virtual LiveStatus probe() const = 0;
};

class HttpLemondProbe final : public LemondProbe {
public:
    explicit HttpLemondProbe(std::string base_url)
        : base_url_(std::move(base_url)) {}
    [[nodiscard]] LiveStatus probe() const override;

private:
    std::string base_url_;
};

} // namespace onebit::landing

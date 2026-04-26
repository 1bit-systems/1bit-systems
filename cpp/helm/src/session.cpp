#include "onebit/helm/session.hpp"

namespace onebit::helm {

nlohmann::json to_json(const SessionConfig& c)
{
    nlohmann::json j = {
        {"server_url", c.server_url},
        {"default_model", c.default_model},
    };
    if (c.bearer)        j["bearer"]        = *c.bearer;
    if (c.system_prompt) j["system_prompt"] = *c.system_prompt;
    return j;
}

SessionConfig from_json(const nlohmann::json& j)
{
    SessionConfig c;
    c.server_url    = j.value("server_url",    std::string{});
    c.default_model = j.value("default_model", std::string{});
    if (j.contains("bearer") && j["bearer"].is_string()) {
        c.bearer = j["bearer"].get<std::string>();
    }
    if (j.contains("system_prompt") && j["system_prompt"].is_string()) {
        c.system_prompt = j["system_prompt"].get<std::string>();
    }
    return c;
}

} // namespace onebit::helm

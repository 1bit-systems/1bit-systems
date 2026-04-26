#include "onebit/helm/conversation.hpp"

#include <chrono>
#include <stdexcept>

namespace onebit::helm {

std::string_view role_to_string(Role r) noexcept
{
    switch (r) {
        case Role::System:    return "system";
        case Role::User:      return "user";
        case Role::Assistant: return "assistant";
    }
    return "user";
}

Role role_from_string(std::string_view s)
{
    if (s == "system")    return Role::System;
    if (s == "user")      return Role::User;
    if (s == "assistant") return Role::Assistant;
    throw std::runtime_error("unknown role: " + std::string(s));
}

void Conversation::push(Role r, std::string content)
{
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto sec = std::chrono::duration_cast<std::chrono::seconds>(now);
    turns_.push_back(ChatTurn{r, std::move(content),
                              static_cast<std::uint64_t>(sec.count())});
}

nlohmann::json Conversation::to_openai_messages() const
{
    auto arr = nlohmann::json::array();
    for (const auto& t : turns_) {
        arr.push_back({{"role", std::string(role_to_string(t.role))},
                       {"content", t.content}});
    }
    return arr;
}

} // namespace onebit::helm

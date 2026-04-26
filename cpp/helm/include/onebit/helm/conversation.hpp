// 1bit-helm — chat turn buffer + OpenAI messages wire shape.
//
// Mirrors crates/1bit-helm/src/conversation.rs. Pure types — no Qt,
// no I/O, fully testable headlessly.

#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::helm {

enum class Role : std::uint8_t { System, User, Assistant };

[[nodiscard]] std::string_view role_to_string(Role r) noexcept;
[[nodiscard]] Role             role_from_string(std::string_view s);

struct ChatTurn {
    Role        role;
    std::string content;
    std::uint64_t ts; // unix seconds
};

class Conversation {
public:
    Conversation() = default;

    void push_system(std::string s)    { push(Role::System,    std::move(s)); }
    void push_user(std::string s)      { push(Role::User,      std::move(s)); }
    void push_assistant(std::string s) { push(Role::Assistant, std::move(s)); }

    [[nodiscard]] const std::vector<ChatTurn>& turns() const noexcept
    {
        return turns_;
    }
    [[nodiscard]] std::vector<ChatTurn>& turns() noexcept
    {
        return turns_;
    }
    [[nodiscard]] bool empty() const noexcept { return turns_.empty(); }

    // Render as the array shape OpenAI /v1/chat/completions expects.
    [[nodiscard]] nlohmann::json to_openai_messages() const;

private:
    void push(Role r, std::string content);

    std::vector<ChatTurn> turns_;
};

} // namespace onebit::helm

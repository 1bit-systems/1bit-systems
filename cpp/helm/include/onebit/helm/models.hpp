// 1bit-helm — Models pane transport: GET /v1/models off the lemonade
// gateway. Mirrors crates/1bit-helm/src/models.rs.

#pragma once

#include <expected>
#include <string>
#include <string_view>
#include <vector>
#include <cstdint>

namespace onebit::helm {

// One row from /v1/models. Drops every field the UI doesn't render.
struct ModelCard {
    std::string   id;
    std::string   owned_by;
    std::uint64_t created{0};
};

// Parse the OpenAI-shaped envelope. Returns the failure reason on
// malformed input.
[[nodiscard]] std::expected<std::vector<ModelCard>, std::string>
parse_models(std::string_view body);

} // namespace onebit::helm

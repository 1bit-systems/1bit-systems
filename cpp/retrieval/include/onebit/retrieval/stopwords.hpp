#pragma once

// Standard English stopword list (~80 words). Short on purpose — we want
// to keep technical terms that look common in English but are
// discriminative in our domain (e.g. "build", "install", "test" stay in).

#include <string_view>

namespace onebit::retrieval {

[[nodiscard]] bool is_stopword(std::string_view s) noexcept;

} // namespace onebit::retrieval

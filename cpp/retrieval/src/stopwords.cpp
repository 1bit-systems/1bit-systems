#include "onebit/retrieval/stopwords.hpp"

#include <string>
#include <unordered_set>

namespace onebit::retrieval {

namespace {

const std::unordered_set<std::string>& stopword_set()
{
    // Built once on first call. `std::string` keys so heterogeneous lookup
    // with std::string_view works via std::hash<std::string_view> on C++20+.
    static const std::unordered_set<std::string> set{
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
        "can", "could", "did", "do", "does", "doing", "for", "from", "had",
        "has", "have", "he", "her", "hers", "him", "his", "how", "i", "if",
        "in", "into", "is", "it", "its", "just", "me", "my", "no", "nor",
        "not", "of", "off", "on", "or", "our", "out", "over", "own", "same",
        "she", "so", "some", "such", "than", "that", "the", "their", "them",
        "then", "there", "these", "they", "this", "those", "through", "to",
        "too", "under", "up", "was", "we", "were", "what", "when", "where",
        "which", "while", "who", "whom", "why", "will", "with", "would",
        "you", "your", "yours",
    };
    return set;
}

} // namespace

bool is_stopword(std::string_view s) noexcept
{
    // unordered_set has heterogeneous lookup since C++20.
    return stopword_set().find(std::string{s}) != stopword_set().end();
}

} // namespace onebit::retrieval

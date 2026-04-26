#include <doctest/doctest.h>

#include "onebit/retrieval/chunker.hpp"

#include <string>
#include <vector>

using onebit::retrieval::chunk_markdown;
using onebit::retrieval::Chunk;

TEST_CASE("heading resets chunk")
{
    const std::string src = "# One\n\nalpha alpha alpha\n\n# Two\n\nbeta beta beta\n";
    auto              chunks = chunk_markdown(src);
    REQUIRE(chunks.size() == 2);
    CHECK(chunks[0].heading == "One");
    CHECK(chunks[0].text.find("alpha") != std::string::npos);
    CHECK(chunks[1].heading == "Two");
    CHECK(chunks[1].text.find("beta") != std::string::npos);
}

TEST_CASE("non-heading hash lines are content")
{
    // `#foo` (no space) is not a heading.
    const std::string src    = "# Title\n\n#notaheading and some text\n";
    auto              chunks = chunk_markdown(src);
    REQUIRE(chunks.size() == 1);
    CHECK(chunks[0].heading == "Title");
    CHECK(chunks[0].text.find("#notaheading") != std::string::npos);
}

TEST_CASE("nested heading levels tracked")
{
    const std::string src =
        "# A\n\ntext A\n\n## A.1\n\ntext A.1\n\n### A.1.a\n\ntext deep\n";
    auto chunks = chunk_markdown(src);
    REQUIRE(chunks.size() == 3);
    CHECK(chunks[0].heading == "A");
    CHECK(chunks[1].heading == "A.1");
    CHECK(chunks[2].heading == "A.1.a");
}

TEST_CASE("multibyte glyphs do not panic at cap")
{
    // U+2500 box-drawing char is 3 bytes in UTF-8. A run that lands the
    // cap in the middle of one used to panic in the Rust impl — regression.
    std::string line;
    for (int i = 0; i < 400; ++i) {
        line.append("\xe2\x94\x80"); // ─
    }
    const auto src    = "# H\n\n" + line + "\n";
    auto       chunks = chunk_markdown(src);
    CHECK_FALSE(chunks.empty());
}

TEST_CASE("content before first heading gets empty heading")
{
    const std::string src    = "preamble text\n\nmore preamble\n\n# Real\n\nbody\n";
    auto              chunks = chunk_markdown(src);
    REQUIRE(chunks.size() == 2);
    CHECK(chunks[0].heading.empty());
    CHECK(chunks[0].text.find("preamble") != std::string::npos);
    CHECK(chunks[1].heading == "Real");
}

TEST_CASE("long section splits on word boundary")
{
    std::string lorem;
    for (int i = 0; i < 200; ++i) {
        lorem.append("word ");
    }
    const auto src    = "# Big\n\n" + lorem;
    auto       chunks = chunk_markdown(src);
    REQUIRE(chunks.size() >= 2);
    for (const auto& c : chunks) {
        // Last whitespace-delimited word should be "word", never a half-cut.
        const auto pos = c.text.find_last_not_of(" \n\t");
        if (pos == std::string::npos) {
            continue;
        }
        const auto start = c.text.find_last_of(" \n\t", pos);
        const auto last  = (start == std::string::npos)
                              ? c.text.substr(0, pos + 1)
                              : c.text.substr(start + 1, pos - start);
        CHECK((last == "word" || last.empty()));
    }
}

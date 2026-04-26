#include <doctest/doctest.h>

#include "onebit/retrieval/stopwords.hpp"

using onebit::retrieval::is_stopword;

TEST_CASE("common english words are stopwords")
{
    CHECK(is_stopword("the"));
    CHECK(is_stopword("a"));
    CHECK(is_stopword("is"));
    CHECK(is_stopword("and"));
}

TEST_CASE("technical words are not stopwords")
{
    CHECK_FALSE(is_stopword("install"));
    CHECK_FALSE(is_stopword("build"));
    CHECK_FALSE(is_stopword("ternary"));
    CHECK_FALSE(is_stopword("amdgpu"));
    CHECK_FALSE(is_stopword("gfx1151"));
}

TEST_CASE("empty string is not a stopword")
{
    CHECK_FALSE(is_stopword(""));
}

TEST_CASE("string_view lookup works without allocation surface")
{
    const char* word = "the";
    CHECK(is_stopword(std::string_view{word, 3}));
}

// (main lives in test_main.cpp)
#include <doctest/doctest.h>

#include "onebit/agent/memory.hpp"

using namespace onebit::agent;

namespace {

Memory open_mem()
{
    auto m = Memory::open(":memory:");
    REQUIRE(m.has_value());
    return std::move(*m);
}

} // namespace

TEST_CASE("Memory::open: in-memory database opens and exposes sqlite version")
{
    auto m = open_mem();
    auto v = Memory::sqlite_version();
    CHECK_FALSE(v.empty());
    // version triplet should contain at least one '.'
    CHECK(v.find('.') != std::string::npos);
}

TEST_CASE("Memory: append + recent round-trips messages in chronological order")
{
    auto m = open_mem();
    REQUIRE(m.append_message("c1", "u1", "user", "hi", "", 1000).has_value());
    REQUIRE(m.append_message("c1", "", "assistant", "yo", "", 1001).has_value());
    REQUIRE(m.append_message("c1", "u1", "user", "again", "", 1002).has_value());

    auto rows = m.recent_messages("c1", 10);
    REQUIRE(rows.has_value());
    REQUIRE_EQ(rows->size(), 3u);
    CHECK_EQ((*rows)[0].content, "hi");
    CHECK_EQ((*rows)[1].content, "yo");
    CHECK_EQ((*rows)[2].content, "again");
    CHECK_EQ((*rows)[0].role,    "user");
    CHECK_EQ((*rows)[1].role,    "assistant");
}

TEST_CASE("Memory: recent_messages limits to N most recent")
{
    auto m = open_mem();
    for (int i = 0; i < 5; ++i) {
        REQUIRE(m.append_message("c1", "u", "user", std::to_string(i), "", 100 + i).has_value());
    }
    auto rows = m.recent_messages("c1", 3);
    REQUIRE(rows.has_value());
    REQUIRE_EQ(rows->size(), 3u);
    CHECK_EQ((*rows)[0].content, "2");
    CHECK_EQ((*rows)[2].content, "4");
}

TEST_CASE("Memory: channels are isolated")
{
    auto m = open_mem();
    REQUIRE(m.append_message("c1", "u", "user", "from-c1", "", 1).has_value());
    REQUIRE(m.append_message("c2", "u", "user", "from-c2", "", 2).has_value());
    auto a = m.recent_messages("c1", 10);
    auto b = m.recent_messages("c2", 10);
    REQUIRE(a.has_value());
    REQUIRE(b.has_value());
    REQUIRE_EQ(a->size(), 1u);
    REQUIRE_EQ(b->size(), 1u);
    CHECK_EQ((*a)[0].content, "from-c1");
    CHECK_EQ((*b)[0].content, "from-c2");
}

TEST_CASE("Memory: trim_messages drops everything before the most recent N")
{
    auto m = open_mem();
    for (int i = 0; i < 10; ++i) {
        REQUIRE(m.append_message("c1", "u", "user", std::to_string(i), "", 100 + i).has_value());
    }
    auto trimmed = m.trim_messages(3);
    REQUIRE(trimmed.has_value());
    CHECK_EQ(*trimmed, 7);
    auto rows = m.recent_messages("c1", 10);
    REQUIRE(rows.has_value());
    REQUIRE_EQ(rows->size(), 3u);
    CHECK_EQ((*rows)[0].content, "7");
}

TEST_CASE("Memory: tool_calls_json column round-trips arbitrary text")
{
    auto m = open_mem();
    const std::string j = R"([{"id":"c","name":"x","args":{"a":1}}])";
    REQUIRE(m.append_message("c1", "", "assistant", "", j, 5).has_value());
    auto rows = m.recent_messages("c1", 1);
    REQUIRE(rows.has_value());
    REQUIRE_EQ(rows->size(), 1u);
    CHECK_EQ((*rows)[0].tool_calls_json, j);
}

TEST_CASE("Memory::upsert_fact + get_fact: insert then update")
{
    auto m = open_mem();
    REQUIRE(m.upsert_fact("k", "v1", 100).has_value());
    auto v = m.get_fact("k");
    REQUIRE(v.has_value());
    REQUIRE(v->has_value());
    CHECK_EQ(**v, "v1");
    REQUIRE(m.upsert_fact("k", "v2", 200).has_value());
    v = m.get_fact("k");
    REQUIRE(v.has_value());
    REQUIRE(v->has_value());
    CHECK_EQ(**v, "v2");
    auto count = m.fact_count();
    REQUIRE(count.has_value());
    CHECK_EQ(*count, 1);
}

TEST_CASE("Memory::get_fact: missing key returns nullopt, not an error")
{
    auto m = open_mem();
    auto v = m.get_fact("nope");
    REQUIRE(v.has_value());
    CHECK_FALSE(v->has_value());
}

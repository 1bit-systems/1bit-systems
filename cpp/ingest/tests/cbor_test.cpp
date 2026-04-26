#include <doctest/doctest.h>

#include "cbor.hpp"

using onebit::ingest::detail::cbor::Array;
using onebit::ingest::detail::cbor::decode;
using onebit::ingest::detail::cbor::encode;
using onebit::ingest::detail::cbor::Object;
using onebit::ingest::detail::cbor::Value;

TEST_CASE("uint round-trip across boundary widths")
{
    for (std::int64_t v : {0, 1, 23, 24, 255, 256, 65535, 65536, 1'000'000}) {
        auto bytes = encode(Value{v});
        auto back  = decode(bytes);
        REQUIRE(back.has_value());
        CHECK(back->as_int() == v);
    }
}

TEST_CASE("text and bool and null round-trip")
{
    Object obj;
    obj.emplace_back("hello", Value{std::string{"world"}});
    obj.emplace_back("flag", Value{true});
    obj.emplace_back("nope", Value{});
    auto bytes = encode(Value{std::move(obj)});

    auto back = decode(bytes);
    REQUIRE(back.has_value());
    REQUIRE(back->is_object());
    auto* h = back->find("hello");
    REQUIRE(h);
    CHECK(h->as_text() == "world");
    auto* f = back->find("flag");
    REQUIRE(f);
    CHECK(f->as_bool());
    auto* n = back->find("nope");
    REQUIRE(n);
    CHECK(n->is_null());
}

TEST_CASE("array round-trip")
{
    Array a;
    a.emplace_back(Value{std::int64_t{1}});
    a.emplace_back(Value{std::string{"two"}});
    a.emplace_back(Value{false});
    auto  bytes = encode(Value{std::move(a)});
    auto  back  = decode(bytes);
    REQUIRE(back.has_value());
    REQUIRE(back->is_array());
    REQUIRE(back->as_array().size() == 3);
    CHECK(back->as_array()[0].as_int() == 1);
    CHECK(back->as_array()[1].as_text() == "two");
    CHECK(back->as_array()[2].as_bool() == false);
}

TEST_CASE("double round-trip")
{
    auto bytes = encode(Value{1.58});
    auto back  = decode(bytes);
    REQUIRE(back.has_value());
    auto* d = std::get_if<double>(&back->variant());
    REQUIRE(d);
    CHECK(*d == doctest::Approx(1.58));
}

TEST_CASE("nested object")
{
    Object inner;
    inner.emplace_back("x", Value{std::int64_t{42}});
    Object outer;
    outer.emplace_back("inner", Value{std::move(inner)});
    auto bytes = encode(Value{std::move(outer)});

    auto back = decode(bytes);
    REQUIRE(back.has_value());
    auto* p = back->find("inner");
    REQUIRE(p);
    REQUIRE(p->is_object());
    auto* x = p->find("x");
    REQUIRE(x);
    CHECK(x->as_int() == 42);
}

TEST_CASE("decode returns error on truncated input")
{
    std::vector<std::uint8_t> bad{0x65, 'h', 'i'}; // text(5) but only 2 bytes
    auto                      back = decode(bad);
    CHECK_FALSE(back.has_value());
}

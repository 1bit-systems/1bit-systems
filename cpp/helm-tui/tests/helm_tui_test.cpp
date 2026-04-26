#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "onebit/helm_tui/layout.hpp"
#include "onebit/helm_tui/pane.hpp"
#include "onebit/helm_tui/state.hpp"
#include "onebit/helm_tui/widgets.hpp"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>

using namespace onebit::helm_tui;

namespace {

std::filesystem::path scratch_path(const char* leaf)
{
    auto base = std::filesystem::temp_directory_path()
              / "onebit-helm-tui-test";
    std::error_code ec;
    std::filesystem::create_directories(base, ec);
    return base / leaf;
}

} // namespace

TEST_CASE("default layout has three leaves")
{
    auto n = Node::default_layout();
    CHECK(n.leaf_count() == 3);
}

TEST_CASE("pane tree round-trips through json")
{
    auto n  = Node::default_layout();
    auto j  = node_to_json(n);
    auto n2 = node_from_json(j);
    CHECK(n.leaf_count() == n2.leaf_count());

    // Top-level shape is preserved.
    CHECK(j["kind"] == "vertical_split");
    CHECK(j["top"]["kind"] == "horizontal_split");
    CHECK(j["bottom"]["kind"] == "leaf");
    CHECK(j["bottom"]["widget"] == "logs");
}

TEST_CASE("layout: load_or_default on missing path returns built-in")
{
    auto n = load_or_default("/nonexistent/1bit/tui-layout.json");
    CHECK(n.leaf_count() == 3);
}

TEST_CASE("layout: save+load round-trips on disk")
{
    auto path = scratch_path("layout.json");
    std::filesystem::remove(path);
    auto n   = Node::default_layout();
    auto res = save(path, n);
    REQUIRE(res.has_value());
    CHECK(std::filesystem::exists(path));

    auto loaded = load_or_default(path);
    CHECK(loaded.leaf_count() == n.leaf_count());
    std::filesystem::remove(path);
}

TEST_CASE("layout: malformed json falls back to default")
{
    auto path = scratch_path("garbage.json");
    {
        std::ofstream f(path, std::ios::trunc);
        f << "not json at all";
    }
    auto n = load_or_default(path);
    CHECK(n.leaf_count() == 3);
    std::filesystem::remove(path);
}

TEST_CASE("widget keys are stable + non-empty")
{
    CHECK(WIDGET_KEYS.size() == 7);
    CHECK(WIDGET_KEYS[0] == "status");
    CHECK(WIDGET_KEYS[1] == "logs");
    CHECK(WIDGET_KEYS[2] == "gpu");
    CHECK(WIDGET_KEYS[3] == "power");
    CHECK(WIDGET_KEYS[4] == "kv");
    CHECK(WIDGET_KEYS[5] == "bench");
    CHECK(WIDGET_KEYS[6] == "repl");
    CHECK(is_known_widget("status"));
    CHECK(is_known_widget("repl"));
    CHECK_FALSE(is_known_widget("not-a-widget"));
    CHECK_FALSE(is_known_widget(""));
}

TEST_CASE("default state has the documented banner")
{
    auto s = default_app_state();
    CHECK_FALSE(s.quit);
    CHECK(s.focused == 0);
    CHECK(s.status_line.find("press Ctrl-q to quit") != std::string::npos);
}

TEST_CASE("ctrl-q + ctrl-c flip the quit flag, plain keys do not")
{
    AppState s = default_app_state();
    CHECK_FALSE(handle_key(s, 'q', /*ctrl=*/false));
    CHECK_FALSE(s.quit);
    CHECK(handle_key(s, 'q', /*ctrl=*/true));
    CHECK(s.quit);

    AppState s2 = default_app_state();
    CHECK(handle_key(s2, 'c', /*ctrl=*/true));
    CHECK(s2.quit);

    AppState s3 = default_app_state();
    CHECK_FALSE(handle_key(s3, 'a', /*ctrl=*/true));
    CHECK_FALSE(s3.quit);
}

TEST_CASE("body_for routes by widget key")
{
    AppState s = default_app_state();
    CHECK(body_for("status", s).find("decode") != std::string::npos);
    CHECK(body_for("gpu", s).find("util") != std::string::npos);
    CHECK(body_for("logs", s).find("journalctl") != std::string::npos);
    CHECK(body_for("nope", s).find("unknown") != std::string::npos);
}

TEST_CASE("pane builders return the matching variant kind")
{
    auto leaf = Node::leaf("status");
    CHECK(leaf.leaf_count() == 1);
    auto v = node_to_json(leaf);
    CHECK(v["kind"] == "leaf");
    CHECK(v["widget"] == "status");

    auto split = Node::vsplit(0.4F, Node::leaf("a"), Node::leaf("b"));
    CHECK(split.leaf_count() == 2);
    auto sj = node_to_json(split);
    CHECK(sj["kind"] == "vertical_split");
    CHECK(sj["ratio"] == doctest::Approx(0.4F));
}

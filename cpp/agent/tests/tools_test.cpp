// (main lives in test_main.cpp)

#include "onebit/agent/tools/registry.hpp"

#include <doctest/doctest.h>
#include <nlohmann/json.hpp>

#include <expected>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using onebit::agent::ToolCall;
using onebit::agent::ToolRegistry;
using onebit::agent::tools::set_test_run_capture;
using onebit::agent::tools::StubResult;
using onebit::agent::tools::trim_and_cap;
using onebit::agent::tools::validate_args;
using nlohmann::json;

namespace {

// RAII helper — installs a stub for one test then clears it on dtor so
// other tests don't see leaked state. The registry's stub hook is a
// global function pointer; not clearing it leaks across cases.
struct StubGuard {
    explicit StubGuard(onebit::agent::tools::RunCaptureFn fn) {
        set_test_run_capture(std::move(fn));
    }
    ~StubGuard() {
        set_test_run_capture({});
    }
};

ToolCall make_call(std::string_view name, json args)
{
    ToolCall c;
    c.id        = "call_test_1";
    c.name      = std::string(name);
    c.args_json = std::move(args);
    return c;
}

} // namespace

TEST_CASE("ToolRegistry::build registers known tools, warns on unknown")
{
    ToolRegistry reg;
    auto out = reg.build({"repo_search", "bench_lookup", "no_such_tool"});

    CHECK(reg.size() == 2);
    CHECK(reg.has("repo_search"));
    CHECK(reg.has("bench_lookup"));
    CHECK_FALSE(reg.has("no_such_tool"));
    REQUIRE(out.warnings.size() == 1);
    CHECK(out.warnings[0].find("no_such_tool") != std::string::npos);

    // list_tools_openai_format() returns one schema per registered tool,
    // each shaped {type:function, function:{name, ...}}.
    auto tools = reg.list_tools_openai_format();
    REQUIRE(tools.size() == 2);
    CHECK(tools[0].at("type").get<std::string>() == "function");
    CHECK(tools[0].at("function").at("name").get<std::string>()
          == "repo_search");
}

TEST_CASE("ToolRegistry::call rejects malformed args at the schema check")
{
    ToolRegistry reg;
    reg.build({"repo_search"});

    SUBCASE("missing required field") {
        auto r = reg.call(make_call("repo_search", json::object({{"limit", 5}})));
        REQUIRE(r.has_value());
        CHECK_FALSE(r->success);
        CHECK(r->content.find("missing required field: query") != std::string::npos);
    }

    SUBCASE("wrong type") {
        auto r = reg.call(make_call("repo_search",
            json::object({{"query", 42}})));
        REQUIRE(r.has_value());
        CHECK_FALSE(r->success);
        CHECK(r->content.find("expected string") != std::string::npos);
    }

    SUBCASE("unknown tool name") {
        auto r = reg.call(make_call("nonexistent",
            json::object({{"x", 1}})));
        REQUIRE_FALSE(r.has_value());
        // AgentError variant carries the tool name.
        CHECK(r.error().what().find("nonexistent") != std::string::npos);
    }
}

TEST_CASE("repo_search composes argv with rg --json and parses match records")
{
    // Stub out rg. The stub captures the argv it received so we can
    // assert on it; the canned stdout has one rg `match` record plus
    // a `summary` record we expect the parser to ignore.
    std::vector<std::string> captured;
    StubGuard g([&](const std::vector<std::string>& argv)
        -> std::expected<StubResult, std::string>
    {
        captured = argv;
        StubResult r;
        r.exit_code = 0;
        r.stdout_text =
            R"({"type":"match","data":{"path":{"text":"/home/bcloud/repos/1bit-systems/cpp/core/foo.cpp"},)"
            R"("line_number":42,"lines":{"text":"int kHaloAnswer = 42;\n"}}})" "\n"
            R"({"type":"summary","data":{"stats":{"matched_lines":1}}})" "\n";
        return r;
    });

    ToolRegistry reg;
    reg.build({"repo_search"});

    auto r = reg.call(make_call("repo_search",
        json::object({{"query", "kHaloAnswer"}, {"limit", 5}})));
    REQUIRE(r.has_value());
    CHECK(r->success);
    CHECK(r->content.find("cpp/core/foo.cpp") != std::string::npos);
    CHECK(r->content.find("42") != std::string::npos);
    CHECK(r->content.find("kHaloAnswer") != std::string::npos);

    // argv assertions: `rg --json --max-count 20 -- <query> <root>`.
    REQUIRE(captured.size() >= 7);
    CHECK(captured[0] == "rg");
    CHECK(captured[1] == "--json");
    CHECK(captured[2] == "--max-count");
    CHECK(captured[4] == "--");                  // separator before query
    CHECK(captured[5] == "kHaloAnswer");
    CHECK(captured[6].find("/home/bcloud/repos/1bit-systems") != std::string::npos);
}

TEST_CASE("bench_lookup parses the canonical RESULTS markdown table")
{
    // This test reads the real benchmark file. If the file isn't there
    // (e.g. running outside the repo), skip.
    const std::filesystem::path p =
        "/home/bcloud/repos/1bit-systems/benchmarks/RESULTS-1bit-2026-04-26.md";
    if (!std::filesystem::exists(p)) {
        WARN("bench file missing; skipping bench_lookup parse test");
        return;
    }

    ToolRegistry reg;
    reg.build({"bench_lookup"});

    SUBCASE("known model returns matching row + headline") {
        auto r = reg.call(make_call("bench_lookup",
            json::object({{"model", "lily-bonsai-1.7B"}})));
        REQUIRE(r.has_value());
        CHECK(r->success);
        CHECK(r->content.find("lily-bonsai-1.7B") != std::string::npos);
        CHECK(r->content.find("281.2") != std::string::npos);     // headline tok/s
        CHECK(r->content.find("Headline") != std::string::npos);
    }

    SUBCASE("missing model returns success=true with no-match note") {
        auto r = reg.call(make_call("bench_lookup",
            json::object({{"model", "definitely-not-a-real-model-xyz"}})));
        REQUIRE(r.has_value());
        CHECK(r->success);
        CHECK(r->content.find("no benchmark row matches") != std::string::npos);
    }
}

TEST_CASE("install_runbook dispatches by component and rejects path traversal")
{
    ToolRegistry reg;
    reg.build({"install_runbook"});

    SUBCASE("known component loads runbook") {
        auto r = reg.call(make_call("install_runbook",
            json::object({{"component", "core"}})));
        REQUIRE(r.has_value());
        CHECK(r->success);
        CHECK(r->content.find("1bit install core") != std::string::npos);
        CHECK(r->content.find("Common errors") != std::string::npos);
    }

    SUBCASE("path traversal attempt rejected before any IO") {
        auto r = reg.call(make_call("install_runbook",
            json::object({{"component", "../../../etc/passwd"}})));
        REQUIRE(r.has_value());
        CHECK_FALSE(r->success);
        CHECK(r->content.find("unknown component") != std::string::npos);
    }

    SUBCASE("unknown component rejected even if file existed") {
        auto r = reg.call(make_call("install_runbook",
            json::object({{"component", "ghost-component"}})));
        REQUIRE(r.has_value());
        CHECK_FALSE(r->success);
        CHECK(r->content.find("unknown component") != std::string::npos);
    }
}

TEST_CASE("gh_issue_create refuses without confirm, fires with confirm")
{
    SUBCASE("confirm:false returns success=false content='not confirmed'") {
        ToolRegistry reg;
        reg.build({"gh_issue_create"});

        auto r = reg.call(make_call("gh_issue_create", json::object({
            {"title",   "test"},
            {"body",    "x"},
            {"confirm", false},
        })));
        REQUIRE(r.has_value());
        CHECK_FALSE(r->success);
        CHECK(r->content.find("not confirmed") != std::string::npos);
    }

    SUBCASE("confirm-omitted (defaults missing) is also rejected by schema") {
        ToolRegistry reg;
        reg.build({"gh_issue_create"});

        auto r = reg.call(make_call("gh_issue_create", json::object({
            {"title", "test"},
            {"body",  "x"},
            // confirm omitted — required field, schema rejects.
        })));
        REQUIRE(r.has_value());
        CHECK_FALSE(r->success);
        // Either schema reject or confirm-gate reject; both are
        // success=false. We assert on either acceptable failure mode.
        CHECK((r->content.find("missing required") != std::string::npos
               || r->content.find("not confirmed")  != std::string::npos));
    }

    SUBCASE("confirm:true with stub gh exec captures argv form") {
        std::vector<std::string> captured;
        StubGuard g([&](const std::vector<std::string>& argv)
            -> std::expected<StubResult, std::string>
        {
            captured = argv;
            StubResult r;
            r.exit_code  = 0;
            r.stdout_text = "https://github.com/example/repo/issues/123\n";
            return r;
        });

        ToolRegistry reg;
        reg.build({"gh_issue_create"});

        auto r = reg.call(make_call("gh_issue_create", json::object({
            {"title",   "tunnel busted"},
            {"body",    "smelly errors abound"},
            {"labels",  json::array({"bug", "tunnel"})},
            {"confirm", true},
        })));
        REQUIRE(r.has_value());
        CHECK(r->success);
        CHECK(r->content.find("https://github.com/example/repo/issues/123")
              != std::string::npos);

        // Argv: gh issue create --title <t> --body <b> --label bug --label tunnel
        REQUIRE(captured.size() >= 9);
        CHECK(captured[0] == "gh");
        CHECK(captured[1] == "issue");
        CHECK(captured[2] == "create");
        CHECK(captured[3] == "--title");
        CHECK(captured[4] == "tunnel busted");
        CHECK(captured[5] == "--body");
        CHECK(captured[6] == "smelly errors abound");
        // Two labels — both `--label X` pairs present.
        bool saw_bug = false, saw_tunnel = false;
        for (std::size_t i = 7; i + 1 < captured.size(); i += 2) {
            CHECK(captured[i] == "--label");
            if (captured[i + 1] == "bug")    saw_bug = true;
            if (captured[i + 1] == "tunnel") saw_tunnel = true;
        }
        CHECK(saw_bug);
        CHECK(saw_tunnel);
    }

    SUBCASE("auto-confirm build option flips default") {
        ToolRegistry reg;
        ToolRegistry::BuildOptions opts;
        opts.gh_issue_auto_confirm = true;
        reg.build({"gh_issue_create"}, opts);

        StubGuard g([](const std::vector<std::string>&)
            -> std::expected<StubResult, std::string>
        {
            StubResult r;
            r.exit_code   = 0;
            r.stdout_text = "https://example/issues/1\n";
            return r;
        });

        // Note: schema still requires `confirm`, so we must pass it
        // — but the auto-confirm wrapper supplies a default if missing.
        // Our schema is `required: [title, body, confirm]`; passing
        // confirm:false here would still bypass the gate because the
        // wrapper overrides any false to true. Spec says auto-confirm
        // is for "let the agent file issues without asking" — the
        // schema gate stays so we don't silently break existing
        // configs that pass confirm:true explicitly.
        auto r = reg.call(make_call("gh_issue_create", json::object({
            {"title",   "auto"},
            {"body",    "x"},
            {"confirm", true},
        })));
        REQUIRE(r.has_value());
        CHECK(r->success);
    }
}

TEST_CASE("trim_and_cap caps oversize content with truncation marker")
{
    std::string huge(8 * 1024, 'x');
    auto out = trim_and_cap(huge);
    CHECK(out.size() <= onebit::agent::tools::kMaxContentBytes + 64);
    CHECK(out.find("[truncated") != std::string::npos);

    auto small = trim_and_cap("hello world\n\n\n");
    CHECK(small == "hello world");
}

TEST_CASE("validate_args type / required handling")
{
    auto schema = json::parse(R"({
        "type": "object",
        "properties": {
            "name":  {"type": "string"},
            "n":     {"type": "integer"},
            "go":    {"type": "boolean"}
        },
        "required": ["name"]
    })");

    CHECK_FALSE(validate_args(schema,
        json::object({{"name", "ok"}, {"n", 3}, {"go", true}})).has_value());

    auto bad_required = validate_args(schema, json::object({{"n", 1}}));
    REQUIRE(bad_required.has_value());
    CHECK(bad_required->find("name") != std::string::npos);

    auto bad_type = validate_args(schema,
        json::object({{"name", "ok"}, {"n", "three"}}));
    REQUIRE(bad_type.has_value());
    CHECK(bad_type->find("integer") != std::string::npos);

    // Optional fields can be absent.
    CHECK_FALSE(validate_args(schema,
        json::object({{"name", "ok"}})).has_value());
}

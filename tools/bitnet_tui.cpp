// bitnet_tui — FTXUI frontend for rocm-cpp.
//
// Talks to a running `bitnet_decode --server` over HTTP (OpenAI-compat
// /v1/chat/completions). Keeps inference + UI cleanly separated; the
// TUI is a pure UX layer. No model loading in this process.
//
// Usage:
//   # terminal 1:
//   bitnet_decode model.h1b --server 8080
//   # terminal 2:
//   bitnet_tui [--url http://127.0.0.1:8080]
//
// Layout (Main page, the only page wired today — see docs/15-tui-spec.md
// for the Man Cave page spec that contributors can fill in next):
//
//   ┌ rocm-cpp // bitnet_decode ───────────────────── 85 tok/s ─┐
//   │ CHAT                                 │ STATS             │
//   │   User: hello                        │   tok/s: 85.1     │
//   │   Assistant: hi, how can I help?     │   last ms: 245    │
//   │                                      │   prompt tok: 14  │
//   │                                      │   total tok: 21   │
//   │                                      │                   │
//   ├──────────────────────────────────────┴───────────────────┤
//   │ > _                                                      │
//   └ Enter send · Ctrl-C quit ────────────────────────────────┘

#define CPPHTTPLIB_OPENSSL_SUPPORT  // harmless if not linked; we won't use https

#include <ftxui/component/captured_mouse.hpp>
#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace ftxui;

struct Stats {
    std::atomic<double> tok_per_s{0.0};
    std::atomic<double> last_latency_ms{0.0};
    std::atomic<int>    prompt_tokens{0};
    std::atomic<int>    total_tokens{0};
    std::atomic<bool>   busy{false};
};

int main(int argc, char** argv) {
    std::string server_url = "http://127.0.0.1:8080";
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--url" && i + 1 < argc) server_url = argv[++i];
    }

    struct Turn { std::string role; std::string content; };
    std::vector<Turn> history;
    std::mutex        history_mu;
    std::string       input;
    Stats             stats;

    auto chat_view = Renderer([&] {
        Elements lines;
        {
            std::lock_guard<std::mutex> lk(history_mu);
            for (const auto& t : history) {
                std::string prefix = (t.role == "user") ? "> " : "  ";
                Color c = (t.role == "user") ? Color::Cyan : Color::White;
                lines.push_back(paragraph(prefix + t.content) | color(c));
                lines.push_back(separatorEmpty());
            }
            if (stats.busy.load()) {
                lines.push_back(text("  ...") | dim | blink);
            }
        }
        return vbox(std::move(lines)) | yframe | flex;
    });

    auto stats_view = Renderer([&] {
        auto row = [](const std::string& l, const std::string& r) {
            return hbox({ text(l) | color(Color::GrayDark), filler(),
                          text(r) | color(Color::Yellow) });
        };
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.1f", stats.tok_per_s.load());
        std::string tps = buf;
        std::snprintf(buf, sizeof(buf), "%.0f ms", stats.last_latency_ms.load());
        std::string lat = buf;
        return vbox({
            text("STATS") | bold | color(Color::GrayLight),
            separator(),
            row("tok/s",      tps),
            row("last",       lat),
            row("prompt tok", std::to_string(stats.prompt_tokens.load())),
            row("total tok",  std::to_string(stats.total_tokens.load())),
            separatorEmpty(),
            text(stats.busy.load() ? "thinking..." : "idle")
                | color(stats.busy.load() ? Color::Green : Color::GrayDark),
        }) | size(WIDTH, EQUAL, 22);
    });

    auto screen = ScreenInteractive::Fullscreen();

    auto submit_prompt = [&](std::string prompt) {
        if (prompt.empty()) return;
        {
            std::lock_guard<std::mutex> lk(history_mu);
            history.push_back({"user", prompt});
        }
        stats.busy.store(true);

        std::thread([&stats, &history, &history_mu, &screen, server_url, prompt] {
            // Build OpenAI request from full history.
            nlohmann::json body;
            body["model"] = "bitnet-b1.58-2b-4t";
            body["max_tokens"] = 256;
            body["messages"] = nlohmann::json::array();
            {
                std::lock_guard<std::mutex> lk(history_mu);
                for (const auto& t : history)
                    body["messages"].push_back({{"role", t.role}, {"content", t.content}});
            }

            // Parse scheme://host:port out of server_url.
            std::string url = server_url;
            std::string scheme = "http";
            if (url.rfind("http://", 0) == 0)  { scheme = "http";  url.erase(0, 7); }
            if (url.rfind("https://", 0) == 0) { scheme = "https"; url.erase(0, 8); }
            std::string host = url;
            int port = (scheme == "https") ? 443 : 80;
            auto cpos = url.find(':');
            if (cpos != std::string::npos) {
                host = url.substr(0, cpos);
                port = std::atoi(url.c_str() + cpos + 1);
            }
            httplib::Client cli(host, port);
            cli.set_connection_timeout(5);
            cli.set_read_timeout(120);
            auto res = cli.Post("/v1/chat/completions", body.dump(), "application/json");

            std::string reply;
            if (!res) {
                reply = "[error] server not reachable at " + server_url;
            } else if (res->status != 200) {
                reply = "[error] HTTP " + std::to_string(res->status) + ": " + res->body;
            } else {
                try {
                    auto j = nlohmann::json::parse(res->body);
                    reply = j["choices"][0]["message"]["content"].get<std::string>();
                    auto u = j["usage"];
                    double lat = u.value("latency_ms", 0.0);
                    int ct = u.value("completion_tokens", 0);
                    int pt = u.value("prompt_tokens", 0);
                    stats.last_latency_ms.store(lat);
                    stats.prompt_tokens.store(pt);
                    stats.total_tokens.store(u.value("total_tokens", 0));
                    if (lat > 0 && ct > 0)
                        stats.tok_per_s.store(1000.0 * ct / lat);
                } catch (const std::exception& e) {
                    reply = std::string("[parse error] ") + e.what();
                }
            }

            {
                std::lock_guard<std::mutex> lk(history_mu);
                history.push_back({"assistant", reply});
            }
            stats.busy.store(false);
            screen.PostEvent(Event::Custom);
        }).detach();
    };

    InputOption in_opt;
    in_opt.on_enter = [&] {
        if (input.empty() || stats.busy.load()) return;
        std::string p = input;
        input.clear();
        submit_prompt(p);
    };
    auto input_box = Input(&input, "type a message…", in_opt);

    auto main_area = Container::Horizontal({chat_view, stats_view});
    auto page = Container::Vertical({main_area, input_box});

    auto main_page = Renderer(page, [&] {
        return vbox({
            hbox({
                text("rocm-cpp") | bold | color(Color::Cyan),
                text(" // bitnet_decode ") | color(Color::GrayDark),
                filler(),
                text(server_url) | color(Color::GrayDark),
            }),
            separator(),
            hbox({
                chat_view->Render() | border | flex,
                stats_view->Render() | border,
            }) | flex,
            hbox({ text("> "), input_box->Render() | flex }) | border,
            text("Enter: send │ Ctrl-C: quit") | color(Color::GrayDark) | hcenter,
        });
    });

    screen.Loop(main_page);
    return 0;
}

// 1bit-helm-tui — interactive tmux-style operator TUI (FTXUI port).
//
// Entry point: parses CLI flags, sets up the FTXUI screen, runs the
// interactive event loop. Widgets + layout + theme live in sibling
// modules. Mirrors crates/1bit-helm-tui/src/main.rs flag-for-flag.

#include "onebit/helm_tui/state.hpp"
#include "onebit/helm_tui/widgets.hpp"

#include <CLI/CLI.hpp>

#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>

#include <string>

using onebit::helm_tui::AppState;
using onebit::helm_tui::default_app_state;
using onebit::helm_tui::render_root_panes;
using onebit::helm_tui::render_status_bar;

int main(int argc, char** argv)
{
    CLI::App app{"Interactive operator TUI for the 1bit stack",
                 "1bit-helm-tui"};
    app.set_version_flag("--version", "1bit-helm-tui 0.1.0");

    std::string layout_path  = "~/.config/1bit/tui-layout.json";
    std::string server_url   = "http://127.0.0.1:8180";
    std::string landing_url  = "http://127.0.0.1:8190";
    app.add_option("--layout",      layout_path,
                   "Path to a saved layout JSON (defaults to built-in)")
        ->capture_default_str();
    app.add_option("--server-url",  server_url,
                   "1bit-server base URL for probes")
        ->capture_default_str();
    app.add_option("--landing-url", landing_url,
                   "1bit-landing base URL for /metrics probe")
        ->capture_default_str();

    CLI11_PARSE(app, argc, argv);

    AppState state = default_app_state();

    auto screen = ftxui::ScreenInteractive::Fullscreen();

    // Renderer pulls a fresh element tree each frame from the
    // pure-function `render_root_panes` + `render_status_bar`. No
    // mutable widget tree to manage — the AppState is the source of
    // truth.
    auto renderer = ftxui::Renderer([&] {
        return ftxui::vbox({
            render_root_panes(state) | ftxui::flex,
            render_status_bar(state),
        });
    });

    // Top-level event handler — Ctrl-q / Ctrl-c quit the loop.
    // FTXUI 5.0 reports control characters as Event::Special({byte});
    // Ctrl-Q = 0x11, Ctrl-C = 0x03.
    static const auto kCtrlQ = ftxui::Event::Special(std::string(1, '\x11'));
    static const auto kCtrlC = ftxui::Event::Special(std::string(1, '\x03'));
    auto with_quit = ftxui::CatchEvent(renderer, [&](ftxui::Event ev) {
        if (ev == kCtrlQ || ev == kCtrlC) {
            state.quit = true;
            screen.ExitLoopClosure()();
            return true;
        }
        return false;
    });

    screen.Loop(with_quit);
    return 0;
}

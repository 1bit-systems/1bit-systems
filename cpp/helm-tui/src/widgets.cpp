#include "onebit/helm_tui/widgets.hpp"

#include "onebit/helm_tui/state.hpp"
#include "onebit/helm_tui/theme.hpp"

#include <ftxui/dom/elements.hpp>

#include <algorithm>
#include <sstream>
#include <vector>

namespace onebit::helm_tui {

bool is_known_widget(std::string_view key) noexcept
{
    return std::ranges::find(WIDGET_KEYS, key) != WIDGET_KEYS.end();
}

std::string state_status(const AppState& /*s*/)
{
    return "decode: 80.8 tok/s · gfx1151 · halo-1bit-2b\n"
           "PPL wikitext-103: 9.16 chunk-1024\n"
           "ternary GEMV: 92% LPDDR5 peak\n"
           "attn: 10.25× @ L=2048\n"
           "NPU: toolchain validated, not serve-path";
}

std::string state_gpu(const AppState& /*s*/)
{
    return "util: ── %\n"
           "edge: ── °C\n"
           "power: ── W\n"
           "mem: ── MiB / 128 GB";
}

std::string state_logs(const AppState& /*s*/)
{
    return "[journalctl tail] — widget stub; v2 pipes strix-server";
}

std::string body_for(std::string_view widget, const AppState& s)
{
    if (widget == "status") return state_status(s);
    if (widget == "gpu")    return state_gpu(s);
    if (widget == "logs")   return state_logs(s);
    if (widget == "power")  return std::string{"power: stub"};
    if (widget == "kv")     return std::string{"kv-cache: stub"};
    if (widget == "bench")  return std::string{"bench: stub"};
    if (widget == "repl")   return std::string{"> "};
    return std::string{"unknown widget: "} + std::string(widget);
}

namespace {

ftxui::Elements paragraph_lines(std::string_view body)
{
    ftxui::Elements out;
    std::string buf;
    for (char c : body) {
        if (c == '\n') {
            out.push_back(ftxui::text(buf));
            buf.clear();
        } else {
            buf.push_back(c);
        }
    }
    if (!buf.empty()) out.push_back(ftxui::text(buf));
    if (out.empty())  out.push_back(ftxui::text(""));
    return out;
}

} // namespace

ftxui::Element render_pane(std::string_view title,
                           std::string_view body,
                           bool focused)
{
    auto title_color = focused ? theme::accent() : theme::dim();
    auto title_el = ftxui::text(std::string{" "} + std::string(title) + " ")
                  | ftxui::bold | ftxui::color(title_color);
    auto body_el = ftxui::vbox(paragraph_lines(body));
    auto framed  = ftxui::window(title_el, body_el);
    if (focused) {
        framed = framed | ftxui::color(theme::accent());
    } else {
        framed = framed | ftxui::color(theme::dim());
    }
    return framed;
}

ftxui::Element render_status_bar(const AppState& s)
{
    return ftxui::hbox({
        ftxui::text(" 1bit-helm-tui ") | ftxui::bold
            | ftxui::color(theme::accent()),
        ftxui::text("· ") | ftxui::color(theme::dim()),
        ftxui::text(s.status_line) | ftxui::color(theme::bright()),
    });
}

ftxui::Element render_root_panes(const AppState& s)
{
    using namespace ftxui;
    // v1: hardcoded status / gpu top row, logs full-bottom — mirrors
    // the Rust skeleton in widgets/mod.rs::draw_root_panes.
    auto top = hbox({
                   render_pane("status", state_status(s), s.focused == 0)
                       | flex,
                   render_pane("gpu", state_gpu(s), s.focused == 1) | flex,
               })
             | flex;
    auto bottom = render_pane("logs", state_logs(s), s.focused == 2) | flex;
    return vbox({top, bottom}) | flex;
}

} // namespace onebit::helm_tui

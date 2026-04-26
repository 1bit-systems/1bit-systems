// 1bit-helm-tui — widget keys + FTXUI Element renderers.
//
// Mirrors crates/1bit-helm-tui/src/widgets/mod.rs. The TUI renderer
// composes the bottom banner + pane tree with FTXUI's `vbox` /
// `hbox` / `border` primitives.

#pragma once

#include <ftxui/dom/elements.hpp>

#include <array>
#include <string>
#include <string_view>

namespace onebit::helm_tui {

struct AppState;

// Widget keys recognised by the pane tree leaf renderer. Order +
// content match the Rust `WIDGET_KEYS` array.
inline constexpr std::array<std::string_view, 7> WIDGET_KEYS = {
    "status", // 1bit-server /v1/models + /metrics probe
    "logs",   // journalctl tail
    "gpu",    // rocm-smi util + edge temp
    "power",  // ryzenadj + TDP gauges
    "kv",     // KV-cache occupancy
    "bench",  // tok/s over time chart
    "repl",   // command input line
};

// Returns true when `key` is in WIDGET_KEYS.
[[nodiscard]] bool is_known_widget(std::string_view key) noexcept;

// Per-widget body copy — placeholder data providers, mirroring the
// Rust `state_status / state_gpu / state_logs` strings. Pure (no I/O,
// no async probes); v2 swaps these for live readouts.
[[nodiscard]] std::string state_status(const AppState& s);
[[nodiscard]] std::string state_gpu(const AppState& s);
[[nodiscard]] std::string state_logs(const AppState& s);
[[nodiscard]] std::string body_for(std::string_view widget, const AppState& s);

// Build the FTXUI element for one bordered pane.
[[nodiscard]] ftxui::Element render_pane(std::string_view title,
                                         std::string_view body,
                                         bool focused);

// Build the bottom one-line status banner.
[[nodiscard]] ftxui::Element render_status_bar(const AppState& s);

// Build the v1 hardcoded three-pane layout (status / gpu top row,
// logs full-bottom). Mirrors the Rust `draw_root_panes` skeleton —
// once the recursive pane-tree walker lands we'll thread `Node`
// through here too.
[[nodiscard]] ftxui::Element render_root_panes(const AppState& s);

} // namespace onebit::helm_tui

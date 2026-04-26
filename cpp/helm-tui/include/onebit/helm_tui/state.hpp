// 1bit-helm-tui — global app state.
//
// Owned by the event loop; widgets pull const refs each frame. Mirrors
// crates/1bit-helm-tui/src/main.rs::AppState.

#pragma once

#include <cstddef>
#include <string>

namespace onebit::helm_tui {

// Global TUI state. Mutations happen in the event handler / probe
// tasks; widgets only read. Keeping this a plain struct (no MVC
// signals) so headless tests can drive it directly.
struct AppState {
    // Quit flag — set on Ctrl-q / Ctrl-c.
    bool quit{false};
    // Focused pane index. Pane tree's recursive walker reads this.
    std::size_t focused{0};
    // Bottom-banner status line.
    std::string status_line{"ready — press Ctrl-q to quit, F1 for help"};
};

// Default-constructed state used at startup. Pulled out for tests
// that want to assert the initial banner without poking at AppState
// internals.
[[nodiscard]] AppState default_app_state();

// Test-friendly key handler. Returns true when the key triggers quit.
// Mirrors the Rust event loop's `(CONTROL, q|c) -> quit`.
[[nodiscard]] bool handle_key(AppState& state, char ch, bool ctrl);

} // namespace onebit::helm_tui

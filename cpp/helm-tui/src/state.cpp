#include "onebit/helm_tui/state.hpp"

namespace onebit::helm_tui {

AppState default_app_state()
{
    return AppState{};
}

bool handle_key(AppState& state, char ch, bool ctrl)
{
    if (ctrl && (ch == 'q' || ch == 'c')) {
        state.quit = true;
        return true;
    }
    return false;
}

} // namespace onebit::helm_tui

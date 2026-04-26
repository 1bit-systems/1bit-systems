// 1bit-helm-tui — turquoise + magenta palette matching the 1bit.systems site.
// Pure constants + FTXUI helpers, no I/O.

#pragma once

#include <ftxui/screen/color.hpp>

namespace onebit::helm_tui::theme {

// Primary accent — turquoise (#00e5d1). Borders + highlights.
inline ftxui::Color accent() { return ftxui::Color::RGB(0x00, 0xe5, 0xd1); }
// Secondary — deep magenta (#ff3d9a). Alerts + over-threshold.
inline ftxui::Color magenta() { return ftxui::Color::RGB(0xff, 0x3d, 0x9a); }
// Dim text — subdued body copy.
inline ftxui::Color dim() { return ftxui::Color::RGB(0x8a, 0x93, 0x97); }
// Bright text — labels + headers.
inline ftxui::Color bright() { return ftxui::Color::RGB(0xe6, 0xed, 0xef); }

} // namespace onebit::helm_tui::theme

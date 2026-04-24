//! Theme — turquoise + magenta palette matching the `1bit.systems` site.

#![allow(dead_code)]

use ratatui::style::{Color, Modifier, Style};

/// Primary accent — turquoise (#00e5d1). Used for borders + highlights.
pub const ACCENT: Color = Color::Rgb(0x00, 0xe5, 0xd1);
/// Secondary — deep magenta (#ff3d9a). Used for alerts + gauges over threshold.
pub const MAGENTA: Color = Color::Rgb(0xff, 0x3d, 0x9a);
/// Dim text — subdued body copy.
pub const DIM: Color = Color::Rgb(0x8a, 0x93, 0x97);
/// Bright text — labels + headers.
pub const BRIGHT: Color = Color::Rgb(0xe6, 0xed, 0xef);

/// Focused pane border style.
pub fn focused_border() -> Style {
    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
}

/// Unfocused pane border.
pub fn unfocused_border() -> Style {
    Style::default().fg(DIM)
}

/// Header label style (pane title, column header).
pub fn header() -> Style {
    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
}

/// Alert (over-threshold metrics, critical log lines).
pub fn alert() -> Style {
    Style::default().fg(MAGENTA).add_modifier(Modifier::BOLD)
}

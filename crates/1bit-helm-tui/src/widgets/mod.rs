//! Widget implementations. Skeleton — only the three v1 panes landed.

#![allow(dead_code)]

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;

use crate::theme;
use crate::AppState;

/// Widget keys recognised by the pane tree. Keep in sync with
/// [`crate::pane::Node::Leaf::widget`] accepted values.
pub const WIDGET_KEYS: &[&str] = &[
    "status", // 1bit-server /v1/models + /metrics probe
    "logs",   // journalctl tail
    "gpu",    // rocm-smi util + edge temp
    "power",  // ryzenadj + TDP gauges
    "kv",     // KV-cache occupancy
    "bench",  // tok/s over time chart
    "repl",   // command input line
];

/// Draw the root pane area. v1 = hardcoded three-pane split; v2 will
/// walk the [`crate::pane::Node`] tree.
pub fn draw_root_panes(f: &mut Frame<'_>, area: Rect, state: &AppState) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(area);

    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(rows[0]);

    draw_pane(f, top[0], "status", &state_status(state), 0 == state.focused);
    draw_pane(f, top[1], "gpu", &state_gpu(state), 1 == state.focused);
    draw_pane(f, rows[1], "logs", &state_logs(state), 2 == state.focused);
}

/// Bottom single-line status banner.
pub fn draw_status_bar(f: &mut Frame<'_>, area: Rect, state: &AppState) {
    let text = Line::from(vec![
        Span::styled(" 1bit-helm-tui ", theme::header()),
        Span::styled("· ", Style::default().fg(theme::DIM)),
        Span::styled(&state.status_line, Style::default().fg(theme::BRIGHT)),
    ]);
    f.render_widget(Paragraph::new(text), area);
}

fn draw_pane(f: &mut Frame<'_>, area: Rect, title: &str, body: &str, focused: bool) {
    let border = if focused {
        theme::focused_border()
    } else {
        theme::unfocused_border()
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border)
        .title(Span::styled(format!(" {title} "), theme::header()));
    let paragraph = Paragraph::new(body).block(block);
    f.render_widget(paragraph, area);
}

// Placeholder data providers — v2 pulls from async probes.
fn state_status(_s: &AppState) -> String {
    "decode: 80.8 tok/s · gfx1151 · halo-1bit-2b\n\
     PPL wikitext-103: 9.16 chunk-1024\n\
     ternary GEMV: 92% LPDDR5 peak\n\
     attn: 10.25× @ L=2048\n\
     NPU: toolchain validated, not serve-path"
        .into()
}

fn state_gpu(_s: &AppState) -> String {
    "util: ── %\nedge: ── °C\npower: ── W\nmem: ── MiB / 128 GB".into()
}

fn state_logs(_s: &AppState) -> String {
    "[journalctl tail] — widget stub; v2 pipes strix-server + strix-watch-discord".into()
}

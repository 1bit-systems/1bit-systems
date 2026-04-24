//! 1bit-helm-tui — interactive tmux-style operator TUI.
//!
//! Entry point: sets up crossterm raw mode + alt screen + mouse, runs the
//! event loop. Widgets + layout + theme live in sibling modules.
//!
//! Current shape is a skeleton — three panes (status / logs / repl) with a
//! hardcoded split, minimal key bindings. Split-tree + resize + preset
//! layouts land in follow-up commits.
//!
//! See `docs/wiki/Helm-TUI.md` for the design brief (binding table, layout
//! JSON schema, widget list).

mod layout;
mod pane;
mod theme;
mod widgets;

use std::io::{self, Stdout};
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use crossterm::event::{DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::Terminal;

/// CLI.
#[derive(Parser, Debug)]
#[command(name = "1bit-helm-tui", version, about = "Interactive operator TUI")]
struct Cli {
    /// Path to a saved layout JSON (see `src/layout.rs`). Falls back to
    /// built-in default on miss.
    #[arg(long, default_value = "~/.config/1bit/tui-layout.json")]
    layout: String,

    /// 1bit-server base URL for probes.
    #[arg(long, default_value = "http://127.0.0.1:8180")]
    server_url: String,

    /// 1bit-landing base URL for /metrics probe.
    #[arg(long, default_value = "http://127.0.0.1:8190")]
    landing_url: String,
}

/// Global application state.
///
/// Owned by the event loop; widgets pull read-only references each frame.
/// Mutations come from async probe tasks via `tokio::sync::watch`
/// channels (added in v2 — skeleton uses synchronous placeholders).
#[derive(Debug, Default)]
pub struct AppState {
    /// Quit flag — set on Ctrl-q or `:q` from the repl.
    quit: bool,
    /// Focused pane index. Split tree lives in `pane::Root`.
    focused: usize,
    /// Last status line for the bottom banner.
    status_line: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(io::stderr)
        .init();

    let args = Cli::parse();

    let mut terminal = setup_terminal().context("failed to set up terminal")?;
    let result = run(&mut terminal, &args).await;
    restore_terminal(&mut terminal).context("failed to restore terminal")?;
    result
}

/// Enter raw mode + alt screen + mouse.
fn setup_terminal() -> Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    Ok(Terminal::new(CrosstermBackend::new(stdout))?)
}

/// Tear down raw mode + alt screen + mouse.
fn restore_terminal(term: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
    disable_raw_mode()?;
    execute!(term.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    term.show_cursor()?;
    Ok(())
}

/// Main render + event loop. Polls crossterm events with a short timeout
/// so the frame rate stays ~30 Hz; probes feed async.
async fn run(term: &mut Terminal<CrosstermBackend<Stdout>>, _args: &Cli) -> Result<()> {
    let mut state = AppState {
        status_line: "ready — press Ctrl-q to quit, F1 for help".into(),
        ..Default::default()
    };

    loop {
        term.draw(|f| {
            let outer = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Min(1),    // pane tree area
                    Constraint::Length(1), // bottom status banner
                ])
                .split(f.area());

            widgets::draw_root_panes(f, outer[0], &state);
            widgets::draw_status_bar(f, outer[1], &state);
        })?;

        if crossterm::event::poll(Duration::from_millis(30))?
            && let Event::Key(k) = crossterm::event::read()?
            && matches!(
                (k.modifiers, k.code),
                (KeyModifiers::CONTROL, KeyCode::Char('q' | 'c'))
            )
        {
            state.quit = true;
        }

        if state.quit {
            break;
        }
    }
    Ok(())
}

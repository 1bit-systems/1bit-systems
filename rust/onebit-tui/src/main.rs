//! 1bit TUI — terminal UI for the 1bit.systems NPU inference stack.
//!
//! A full-screen terminal user interface using ratatui + crossterm.
//! Connects to the NPU API (default http://127.0.0.1:9090/v1) for chat
//! completions.
//!
//! Layout:
//!   ┌─ Header (status) ────────────────────────────┐
//!   │                                              │
//!   │  Chat messages (scrollable list)             │
//!   │                                              │
//!   ├─ Input bar ──────────────────────────────────┤
//!   └──────────────────────────────────────────────┘

use anyhow::Result;
use chrono::Local;
use clap::Parser;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    Frame,
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, List, ListItem, Paragraph},
    Terminal,
};
use std::{
    io::stdout,
    time::Duration,
};
use tokio::sync::mpsc;

const VERSION: &str = "0.1.0";
const APP_NAME: &str = "1bit TUI";

// ── CLI ─────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(name = "1bit-tui", version = VERSION, about = "Terminal UI for 1bit NPU agent")]
struct Args {
    /// NPU API endpoint
    #[arg(short, long, default_value = "http://127.0.0.1:9090/v1")]
    endpoint: String,

    /// Model to use
    #[arg(short, long, default_value = "qwen3-0.6b-FLM")]
    model: String,
}

// ── Message types ───────────────────────────────────────────────

#[derive(Debug, Clone)]
enum Msg {
    User(String),
    Assistant(String),
    Error(String),
    System(String),
}

impl Msg {
    fn role_style(&self) -> Style {
        match self {
            Msg::User(_) => Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            Msg::Assistant(_) => Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            Msg::Error(_) => Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            Msg::System(_) => Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
        }
    }

    fn role_label(&self) -> &str {
        match self {
            Msg::User(_) => "You",
            Msg::Assistant(_) => "1bit",
            Msg::Error(_) => "Error",
            Msg::System(_) => "System",
        }
    }

    fn content(&self) -> &str {
        match self {
            Msg::User(s) | Msg::Assistant(s) | Msg::Error(s) | Msg::System(s) => s,
        }
    }
}

// ── API client ──────────────────────────────────────────────────

#[derive(Clone)]
struct ApiClient {
    client: reqwest::Client,
    base_url: String,
    model: String,
}

impl ApiClient {
    fn new(base_url: &str, model: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
        }
    }

    async fn health_check(&self) -> bool {
        let health_url = self.base_url.replace("/v1", "/health");
        if let Ok(resp) = self.client.get(&health_url).timeout(Duration::from_secs(3)).send().await {
            if resp.status().is_success() { return true; }
        }
        let models_url = format!("{}/models", self.base_url);
        self.client.get(&models_url).timeout(Duration::from_secs(3)).send().await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    async fn chat(&self, messages: Vec<serde_json::Value>) -> Result<String> {
        let url = format!("{}/chat/completions", self.base_url);
        let body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "stream": false,
        });

        let resp = self.client.post(&url).json(&body)
            .timeout(Duration::from_secs(120)).send().await?;
        let status = resp.status();

        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("API error ({}): {}", status, text);
        }

        let chat: serde_json::Value = resp.json().await?;
        Ok(chat["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("[no response]")
            .to_string())
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/models", self.base_url);
        let resp = self.client.get(&url).timeout(Duration::from_secs(5)).send().await?;
        let parsed: serde_json::Value = resp.json().await?;
        Ok(parsed["data"].as_array()
            .map(|arr| arr.iter().filter_map(|m| m["id"].as_str().map(String::from)).collect())
            .unwrap_or_default())
    }
}

// ── TUI app state ───────────────────────────────────────────────

enum InputMode {
    Insert,
    Command,
}

enum PendingAction {
    None,
    Chat(Vec<serde_json::Value>),
    FetchModels,
    Reconnect,
}

struct TuiApp {
    messages: Vec<Msg>,
    input: String,
    cmd_buffer: String,
    mode: InputMode,
    scroll: usize,
    connected: bool,
    api: ApiClient,
    models: Vec<String>,
    status: String,
    pending: PendingAction,
    quit: bool,
    thinking: bool,
}

impl TuiApp {
    fn new(api: ApiClient) -> Self {
        Self {
            messages: vec![
                Msg::System(format!("{APP_NAME} v{VERSION} — NPU-native coding agent. Type /help for commands.")),
            ],
            input: String::new(),
            cmd_buffer: String::new(),
            mode: InputMode::Insert,
            scroll: 0,
            connected: false,
            api,
            models: vec![],
            status: "Connecting...".into(),
            pending: PendingAction::None,
            quit: false,
            thinking: false,
        }
    }

    fn push(&mut self, msg: Msg) {
        self.messages.push(msg);
        self.scroll = 0; // auto-scroll
    }

    fn send_msg(&mut self, content: String) {
        self.push(Msg::User(content.clone()));
        let history: Vec<serde_json::Value> = self.messages.iter()
            .filter_map(|m| match m {
                Msg::User(s) => Some(serde_json::json!({"role": "user", "content": s})),
                Msg::Assistant(s) => Some(serde_json::json!({"role": "assistant", "content": s})),
                _ => None,
            })
            .collect();
        self.pending = PendingAction::Chat(history);
        self.thinking = true;
        self.status = "Thinking...".into();
    }
}

// ── Rendering ───────────────────────────────────────────────────

fn render(app: &mut TuiApp, frame: &mut Frame) {
    let area = frame.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(3),
        ])
        .split(area);

    // Status bar
    let status = format!(
        " {} {} | {} | {} ",
        if app.connected { "●" } else { "○" },
        if app.connected { "NPU" } else { "OFFLINE" },
        app.api.model,
        app.status,
    );
    let status_color = if app.connected { Color::Green } else { Color::Red };
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(&status, Style::default().fg(status_color))))
            .block(Block::default().borders(Borders::NONE)),
        chunks[0],
    );

    // Messages
    let max_w = chunks[1].width.saturating_sub(4) as usize;
    let items: Vec<ListItem> = app.messages.iter().rev().map(|m| {
        let content = if max_w > 20 {
            wrap_text(m.content(), max_w)
        } else {
            m.content().to_string()
        };
        let lines = vec![
            Line::from(Span::styled(format!(" {} ", m.role_label()), m.role_style())),
            Line::from(Span::raw(content)),
            Line::from(""),
        ];
        ListItem::new(lines)
    }).collect();

    frame.render_widget(
        List::new(items)
            .block(Block::default()
                .borders(Borders::TOP)
                .border_type(BorderType::Plain)
                .title(" Chat ")
                .title_alignment(Alignment::Left)),
        chunks[1],
    );

    // Input
    let (prefix, text) = match app.mode {
        InputMode::Command => (":", &app.cmd_buffer),
        InputMode::Insert => (">", &app.input),
    };
    let input_style = match app.mode {
        InputMode::Insert => Style::default().fg(Color::Cyan),
        InputMode::Command => Style::default().fg(Color::Yellow),
    };

    let input_display = if text.is_empty() {
        format!("{} ", prefix)
    } else {
        format!("{} {}", prefix, text)
    };

    frame.render_widget(
        Paragraph::new(input_display.as_str())
            .style(input_style)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Input ")
                .title_alignment(Alignment::Left)),
        chunks[2],
    );

    // Cursor
    let cursor_x = chunks[2].x + 2 + input_display.len().saturating_sub(1) as u16;
    let cursor_y = chunks[2].y + 1;
    frame.set_cursor_position((cursor_x.min(area.width.saturating_sub(1)), cursor_y));
}

fn wrap_text(text: &str, max_width: usize) -> String {
    let mut result = String::new();
    let mut line_len = 0;
    for word in text.split_whitespace() {
        if line_len + word.len() + 1 > max_width && line_len > 0 {
            result.push('\n');
            line_len = 0;
        }
        if line_len > 0 { result.push(' '); line_len += 1; }
        result.push_str(word);
        line_len += word.len();
    }
    result
}

// ── Event handling ──────────────────────────────────────────────

fn handle_key(app: &mut TuiApp, key: KeyCode, _mods: KeyModifiers) {
    match app.mode {
        InputMode::Command => match key {
            KeyCode::Enter => {
                let cmd = app.cmd_buffer.trim().to_string();
                app.cmd_buffer.clear();
                app.mode = InputMode::Insert;
                exec_cmd(app, &cmd);
            }
            KeyCode::Esc => {
                app.cmd_buffer.clear();
                app.mode = InputMode::Insert;
            }
            KeyCode::Char(c) => app.cmd_buffer.push(c),
            KeyCode::Backspace => { app.cmd_buffer.pop(); }
            _ => {}
        },
        InputMode::Insert => match key {
            KeyCode::Enter => {
                let msg = app.input.trim().to_string();
                if !msg.is_empty() {
                    app.input.clear();
                    if msg.starts_with('/') {
                        exec_cmd(app, &msg);
                    } else {
                        app.send_msg(msg);
                    }
                }
            }
            KeyCode::Char(c) => app.input.push(c),
            KeyCode::Backspace => { app.input.pop(); }
            KeyCode::Esc => {}  // stay in insert mode
            KeyCode::Tab => {
                app.mode = InputMode::Command;
            }
            _ => {}
        },
    }
}

fn exec_cmd(app: &mut TuiApp, cmd: &str) {
    let parts: Vec<&str> = cmd.trim().split_whitespace().collect();
    let verb = parts.first().copied().unwrap_or("");
    match verb {
        "/help" | "help" => {
            app.push(Msg::System(
                "\
Commands:
  /help          Show this help
  /quit          Quit
  /clear         Clear chat
  /models        List models
  /model <name>  Switch model
  /connect       Reconnect
  /status        Show status
  /version       Show version

Keys:
  Enter  Send message
  Tab    Command mode
  Esc    Back to insert".into()));
        }
        "/quit" | "quit" | "/exit" | "exit" => app.quit = true,
        "/clear" | "clear" => { app.messages.clear(); app.push(Msg::System("Chat cleared.".into())); }
        "/models" | "models" => { app.pending = PendingAction::FetchModels; app.status = "Fetching models...".into(); }
        "/model" | "model" => {
            if let Some(name) = parts.get(1) {
                app.api.model = name.to_string();
                app.push(Msg::System(format!("Model: {name}")));
            } else {
                app.push(Msg::System(format!("Current model: {}", app.api.model)));
            }
        }
        "/connect" | "connect" => { app.pending = PendingAction::Reconnect; app.status = "Reconnecting...".into(); }
        "/status" | "status" => {
            if app.connected {
                app.push(Msg::System(format!("✅ Connected to {}", app.api.base_url)));
            } else {
                app.push(Msg::System(format!("❌ Not connected to {}", app.api.base_url)));
            }
        }
        "/version" | "version" => app.push(Msg::System(format!("{APP_NAME} v{VERSION}"))),
        _ => app.push(Msg::System(format!("Unknown: {verb}. /help"))),
    }
}

// ── Main ────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let api = ApiClient::new(&args.endpoint, &args.model);
    let mut app = TuiApp::new(api);
    app.connected = app.api.health_check().await;
    app.status = if app.connected { "Ready".into() } else { "Disconnected".into() };
    app.push(Msg::System(
        if app.connected { "✅ Connected to NPU API.".into() } else { "⚠️  NPU offline. /connect to retry.".into() }
    ));

    let (tx, mut rx) = mpsc::channel::<Result<String>>(64);

    loop {
        // Process pending actions
        match std::mem::replace(&mut app.pending, PendingAction::None) {
            PendingAction::Chat(history) => {
                let api = app.api.clone();
                let tx = tx.clone();
                tokio::spawn(async move {
                    let result = api.chat(history).await;
                    tx.send(result).await.ok();
                });
            }
            PendingAction::FetchModels => {
                let api = app.api.clone();
                let tx2 = tx.clone();
                tokio::spawn(async move {
                    let result = api.list_models().await;
                    let msg = match result {
                        Ok(models) => format!("Models: {}", models.join(", ")),
                        Err(e) => format!("Error: {e}"),
                    };
                    tx2.send(Ok(msg)).await.ok();
                });
            }
            PendingAction::Reconnect => {
                let api = app.api.clone();
                let tx = tx.clone();
                tokio::spawn(async move {
                    let ok = api.health_check().await;
                    tx.send(Ok(if ok { "✅ Reconnected.".into() } else { "❌ Still offline.".into() })).await.ok();
                });
            }
            PendingAction::None => {}
        }

        // Check for async results
        match rx.try_recv() {
            Ok(result) => {
                match result {
                    Ok(text) => {
                        if app.thinking {
                            app.push(Msg::Assistant(text));
                            app.thinking = false;
                            app.status = "Ready".into();
                        } else {
                            app.push(Msg::System(text));
                            app.status = "Ready".into();
                        }
                    }
                    Err(e) => {
                        app.push(Msg::Error(format!("{}", e)));
                        app.thinking = false;
                        app.status = "Error".into();
                    }
                }
            }
            Err(_) => {} // No message available yet or channel closed
        }

        // Draw
        terminal.draw(|f| render(&mut app, f))?;

        if app.quit { break; }

        // Input with timeout
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    handle_key(&mut app, key.code, key.modifiers);
                }
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

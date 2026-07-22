//! 1bit — terminal coding agent for the 1bit.systems NPU+GPU+CPU inference stack.
//!
//! A single static binary that replaces the old TypeScript wrapper (which depended
//! on pi's npm packages).  Handles NPU stack management, interactive chat with tool
//! execution via MCP, config, and engine builds — all in one Rust binary.
//!
//! Architecture:
//!   1bit chat        → interactive REPL → NPU API + MCP tool execution
//!   1bit up          → spawn onebitd + bitnet_decode daemon
//!   1bit down        → kill NPU processes
//!   1bit status      → check NPU stack health
//!   1bit build       → compile NPU engine from source
//!   1bit config      → view / set settings

mod config;
mod mcp;
mod npu;
mod provider;
mod secrets;
mod server;
mod session;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::process::{Command, Stdio};
use std::time::Duration;
use tracing::error;

// ── CLI Definition ───────────────────────────────────────────────

const VERSION: &str = "0.1.0";
const BANNER: &str = r"
  ╔══════════════════════════════════════════╗
  ║                                          ║
  ║    ██   ██████╗  ██╗  ████████╗         ║
  ║    ██   ██╔══██╗  ██║  ╚══██╔══╝        ║
  ║    ██   ██████╔╝  ██║     ██║           ║
  ║    ██   ██╔══██╗  ██║     ██║           ║
  ║   ██████ ██████╔╝  ██║     ██║          ║
  ║   ╚═════ ╚═════╝   ╚═╝     ╚═╝          ║
  ║                                          ║
  ║       NPU-native coding agent            ║
  ║    50 TOPS · 94 tok/s · 0 cloud          ║
  ║              vVERSION                     ║
  ╚══════════════════════════════════════════╝
";

#[derive(Parser, Debug)]
#[command(
    name = "1bit",
    version = VERSION,
    about = "NPU-native coding agent for 1bit.systems",
    long_about = "Terminal coding agent for the 1bit.systems NPU+GPU+CPU inference stack.
Zero cloud, zero Python — runs entirely on your AMD Strix Halo NPU."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Prompt to send directly (non-interactive)
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    prompt: Vec<String>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start interactive agent session (default)
    Chat {
        /// Model to use
        #[arg(short, long)]
        model: Option<String>,
    },

    /// Start NPU stack (onebitd + bitnet_decode daemon)
    Up {
        /// Path to model file
        #[arg(short, long)]
        model: Option<String>,

        /// Path to bitnet_decode binary
        #[arg(long)]
        bitnet_decode: Option<String>,
    },

    /// Stop NPU stack
    Down,

    /// Show NPU stack status
    Status,

    /// Build NPU engine from source
    Build {
        /// Build directory
        #[arg(short, long, default_value = "engine/npu")]
        dir: String,
    },

    /// View or set configuration
    Config {
        /// Key to get or set (e.g. "theme", "npu.api_port")
        key: Option<String>,

        /// Value to set (omit to get current value)
        value: Option<String>,
    },

    /// Serve the agent runtime over HTTP/SSE
    Serve {
        /// Port to bind
        #[arg(short, long, default_value_t = 7878)]
        port: u16,

        /// Host to bind (default 127.0.0.1; use 0.0.0.0 for LAN)
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Model to use for chat completions
        #[arg(short, long)]
        model: Option<String>,
    },

    /// Manage API keys for providers
    Auth {
        /// Provider name (e.g. "deepseek", "openai")
        provider: Option<String>,

        /// Set API key (reads from stdin if not provided)
        #[arg(short, long)]
        set: Option<String>,

        /// Remove stored API key
        #[arg(short, long)]
        remove: bool,

        /// List all configured providers
        #[arg(short, long)]
        list: bool,
    },

    /// Update 1bit to the latest version
    Update {
        /// Check for updates without installing
        #[arg(short, long)]
        check: bool,
    },
}

// ── Main ─────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "onebit=info".into()),
        )
        .init();

    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Chat { model }) => cmd_chat(model.as_deref()).await?,
        Some(Commands::Up { model, bitnet_decode }) => {
            cmd_up(model.as_deref(), bitnet_decode.as_deref()).await?
        }
        Some(Commands::Down) => cmd_down().await?,
        Some(Commands::Status) => cmd_status().await?,
        Some(Commands::Build { dir }) => cmd_build(dir)?,
        Some(Commands::Config { key, value }) => cmd_config(key.as_deref(), value.as_deref())?,
        Some(Commands::Serve { port, host, model }) => cmd_serve(*port, host, model.as_deref()).await?,
        Some(Commands::Auth { provider, set, remove, list }) => {
            cmd_auth(provider.as_deref(), set.as_deref(), *remove, *list)?;
        }
        Some(Commands::Update { check }) => cmd_update(*check).await?,
        None => {
            if !cli.prompt.is_empty() {
                let prompt = cli.prompt.join(" ");
                cmd_chat_once(&prompt).await?;
            } else {
                cmd_chat(None).await?;
            }
        }
    }

    Ok(())
}

// ── Command: chat (interactive REPL with MCP tools) ──────────────

async fn cmd_chat(model_override: Option<&str>) -> Result<()> {
    let settings = config::Settings::load()?;
    let model = model_override
        .map(|s| s.to_string())
        .unwrap_or(settings.default_model.clone());

    // Initialize MCP tools
    let mut mcp_manager = mcp::McpManager::load()?;
    let tools = mcp_manager.all_tools();
    if !tools.is_empty() {
        println!("  🔌 MCP tools loaded: {}", tools.len());
        for (server, tool) in &tools {
            println!("     • {}/{}", server, tool.name);
        }
        println!();
    }

    println!("{}", BANNER.replace("VERSION", VERSION));

    // Check NPU health
    let client = npu::NpuClient::new(&settings.npu_endpoint)?;
    match client.health_check().await {
        Ok(true) => println!("  ✅ NPU stack is online — ask me anything.\n"),
        Ok(false) => {
            println!("  ℹ️  NPU stack is not running. Type /up to start it.\n");
        }
        Err(e) => {
            println!("  ⚠️  Could not reach NPU: {e}\n");
        }
    }

    let mut rl = rustyline::DefaultEditor::new()
        .context("Failed to initialize readline editor")?;

    // Conversation history for tool-aware chat
    let mut history: Vec<npu::Message> = Vec::new();

    println!("  Type /help for commands, /exit to quit.\n");

    loop {
        let readline = rl.readline("1bit> ");
        match readline {
            Ok(line) => {
                let trimmed = line.trim().to_string();
                if trimmed.is_empty() {
                    continue;
                }
                rl.add_history_entry(&trimmed).ok();

                match handle_chat_command(&trimmed, &client, &model).await {
                    Action::Continue => continue,
                    Action::Break => break,
                    Action::Send => {
                        if !client.health_check().await.unwrap_or(false) {
                            println!("  ⚠️  NPU stack not running. Type /up to start.\n");
                            continue;
                        }

                        // Add user message to history
                        history.push(npu::Message::user(&trimmed));

                        // Prepare tool definitions for the model
                        let tool_defs = if !tools.is_empty() {
                            Some(
                                tools
                                    .iter()
                                    .map(|(_server, t)| npu::ToolDefinition {
                                        tool_type: "function".into(),
                                        function: npu::ToolFunction {
                                            name: t.name.clone(),
                                            description: t
                                                .description
                                                .clone()
                                                .unwrap_or_default(),
                                            parameters: Some(t.input_schema.clone()),
                                        },
                                    })
                                    .collect(),
                            )
                        } else {
                            None
                        };

                        println!("  🤔 Thinking...");

                        // Run the tool loop: model may request tool calls, we execute them
                        match run_tool_loop(&client, &model, &mut history, tool_defs, &mut mcp_manager).await {
                            Ok(final_text) => {
                                println!("\n  {}\n", final_text);
                            }
                            Err(e) => {
                                println!("  ⚠️  Error: {e}\n");
                            }
                        }
                    }
                }
            }
            Err(rustyline::error::ReadlineError::Interrupted)
            | Err(rustyline::error::ReadlineError::Eof) => {
                println!("\n  Goodbye.\n");
                break;
            }
            Err(e) => {
                error!("Readline error: {e}");
                break;
            }
        }
    }

    Ok(())
}

/// Run the model ↔ tool loop until the model produces a text response.
async fn run_tool_loop(
    client: &npu::NpuClient,
    model: &str,
    history: &mut Vec<npu::Message>,
    tool_defs: Option<Vec<npu::ToolDefinition>>,
    mcp_manager: &mut mcp::McpManager,
) -> Result<String> {
    let max_iterations = 10;
    for _i in 0..max_iterations {
        match client.chat(model, history.clone(), tool_defs.clone(), None).await? {
            npu::ChatResult::Text(text) => {
                history.push(npu::Message::assistant(&text));
                return Ok(text);
            }
            npu::ChatResult::ToolCalls(tool_calls) => {
                // Record the assistant's tool calls in history
                history.push(npu::Message::assistant_with_tools(tool_calls.clone()));

                for tc in &tool_calls {
                    println!("  🛠️  Calling {}()...", tc.function.name);

                    // Parse arguments
                    let args: serde_json::Value = match serde_json::from_str(&tc.function.arguments) {
                        Ok(a) => a,
                        Err(e) => {
                            let err = format!("Failed to parse arguments: {e}");
                            history.push(npu::Message::tool_result(&tc.id, &err));
                            continue;
                        }
                    };

                    // Execute via MCP
                    match mcp_manager.call_tool_by_name(&tc.function.name, args) {
                        Ok(result) => {
                            let mut output = String::new();
                            for content in &result.content {
                                if let Some(text) = &content.text {
                                    output.push_str(text);
                                    output.push('\n');
                                }
                            }
                            if output.trim().is_empty() {
                                output = "[tool returned no output]".to_string();
                            }
                            if result.is_error {
                                println!("  ⚠️  Tool {} failed: {}", tc.function.name, output.trim());
                            } else {
                                let preview: String = output.chars().take(200).collect();
                                if output.len() > 200 {
                                    println!("  ✅ {} → {}... ({} chars)", tc.function.name, preview, output.len());
                                } else {
                                    println!("  ✅ {} → {}", tc.function.name, preview.trim());
                                }
                            }
                            history.push(npu::Message::tool_result(&tc.id, output.trim().to_string()));
                        }
                        Err(e) => {
                            let err = format!("Tool execution error: {e}");
                            println!("  ⚠️  {}", err);
                            history.push(npu::Message::tool_result(&tc.id, err));
                        }
                    }
                }
                // Loop back — let the model respond after tool results
            }
        }
    }

    anyhow::bail!("Tool loop exceeded max iterations ({max_iterations}) — possible infinite tool call loop");
}

/// One-shot chat (for `1bit "prompt"` or `1bit -- "prompt"`)
async fn cmd_chat_once(prompt: &str) -> Result<()> {
    let settings = config::Settings::load()?;
    let client = npu::NpuClient::new(&settings.npu_endpoint)?;

    if !client.health_check().await.unwrap_or(false) {
        anyhow::bail!("NPU stack is not running. Start it with `1bit up`");
    }

    let history = vec![npu::Message::user(prompt)];
    match client.chat(&settings.default_model, history, None, None).await? {
        npu::ChatResult::Text(text) => {
            println!("{}", text);
        }
        npu::ChatResult::ToolCalls(tcs) => {
            for tc in &tcs {
                println!("  🛠️  Tool call: {}({})", tc.function.name, tc.function.arguments);
            }
        }
    }

    Ok(())
}

enum Action {
    Continue,
    Break,
    Send,
}

async fn handle_chat_command(line: &str, _client: &npu::NpuClient, _model: &str) -> Action {
    let parts: Vec<&str> = line.trim().split_whitespace().collect();
    let verb = parts.first().copied().unwrap_or("");

    match verb {
        "/help" => {
            println!(
                "
  Commands:
    /help              Show this help
    /status            Check NPU stack status
    /up                Start NPU stack
    /down              Stop NPU stack
    /clear             Clear the screen
    /models            List available models
    /tools             List available MCP tools
    /exit              Exit 1bit chat
  "
            );
            Action::Continue
        }
        "/status" => {
            match cmd_status().await {
                Ok(()) => {}
                Err(e) => println!("  ⚠️  Status error: {e}"),
            }
            Action::Continue
        }
        "/up" => {
            match cmd_up(None, None).await {
                Ok(()) => println!("  ✅ NPU stack started"),
                Err(e) => println!("  ⚠️  Error: {e}"),
            }
            Action::Continue
        }
        "/down" => {
            match cmd_down().await {
                Ok(()) => println!("  ✅ NPU stack stopped"),
                Err(e) => println!("  ⚠️  Error: {e}"),
            }
            Action::Continue
        }
        "/clear" => {
            print!("\x1B[2J\x1B[1;1H");
            println!("{}", BANNER.replace("VERSION", VERSION));
            Action::Continue
        }
        "/models" => {
            let settings = config::Settings::load().ok();
            let endpoint = settings
                .as_ref()
                .map(|s| s.npu_endpoint.clone())
                .unwrap_or_else(|| "http://127.0.0.1:9090/v1".to_string());
            match npu::NpuClient::new(&endpoint) {
                Ok(client) => match client.list_models().await {
                    Ok(models) => {
                        if models.is_empty() {
                            println!("  ℹ️  No models available");
                        } else {
                            println!("  Available models:");
                            for m in models {
                                println!("    • {m}");
                            }
                        }
                    }
                    Err(e) => println!("  ⚠️  {e}"),
                },
                Err(e) => println!("  ⚠️  {e}"),
            }
            Action::Continue
        }
        "/tools" => {
            match mcp::McpManager::load() {
                Ok(manager) => {
                    let all_tools = manager.all_tools();
                    if all_tools.is_empty() {
                        println!("  ℹ️  No MCP tools available");
                        println!("     Configure servers in ~/.1bit/mcp.json");
                    } else {
                        println!("  Available MCP tools:");
                        for (server, tool) in &all_tools {
                            let desc = tool
                                .description
                                .as_ref()
                                .filter(|d| !d.is_empty())
                                .map(|d| format!(" — {d}"))
                                .unwrap_or_default();
                            println!("    • {}/{}", server, tool.name);
                            println!("      {desc}");
                        }
                    }
                }
                Err(e) => {
                    println!("  ⚠️  Could not load MCP config: {e}");
                    println!("     Create ~/.1bit/mcp.json with MCP server definitions");
                }
            }
            Action::Continue
        }
        "/exit" | "/quit" => {
            println!("  Goodbye.\n");
            Action::Break
        }
        _ if verb.starts_with('/') => {
            println!("  Unknown command: {verb}. Try /help");
            Action::Continue
        }
        _ => Action::Send,
    }
}

// ── Command: up ──────────────────────────────────────────────────

async fn cmd_up(model: Option<&str>, bitnet_decode: Option<&str>) -> Result<()> {
    let settings = config::Settings::load()?;
    let api_port = settings.npu.api_port;

    println!("  🚀 Starting 1bit NPU stack...\n");

    // Check if port is already in use
    if is_port_in_use(api_port).await {
        println!("  ✅ onebitd already running on port {api_port}");
    } else {
        let decode_path = bitnet_decode
            .map(|s| s.to_string())
            .unwrap_or(settings.npu.bitnet_decode_path.clone());

        let model_path = model
            .map(|s| s.to_string())
            .unwrap_or_else(|| "./model.h1b".to_string());

        println!("  Starting onebitd (bitnet_decode proxy)...");
        let mut cmd = Command::new(&decode_path);
        cmd.arg(&model_path)
            .arg("--server")
            .arg(api_port.to_string())
            .arg("--bind")
            .arg("127.0.0.1")
            .stdout(Stdio::null())
            .stderr(Stdio::null());

        if settings.npu.tune_prefill {
            cmd.arg("--tune-prefill");
        }
        if let Some(v) = settings.npu.prefill_variant {
            cmd.arg("--prefill-variant").arg(v.to_string());
        }
        if settings.npu.fp16_weights {
            cmd.arg("--fp16-weights");
        }

        cmd.env("HSA_OVERRIDE_GFX_VERSION", "11.5.1");
        cmd.env("HSA_ENABLE_SDMA", "0");

        match cmd.spawn() {
            Ok(mut child) => {
                child.stdin = None;
                println!("  ✅ bitnet_decode started (pid {}) on port {api_port}", child.id());
            }
            Err(e) => {
                println!("  ⚠️  Failed to start bitnet_decode: {e}");
                println!("       Is rocm-cpp installed? Try `1bit build`");
            }
        }
    }

    tokio::time::sleep(Duration::from_secs(2)).await;

    let client = npu::NpuClient::new(&settings.npu_endpoint)?;
    match client.health_check().await {
        Ok(true) => {
            let models = client.list_models().await.unwrap_or_default();
            println!(
                "  ✅ NPU API responding ({} model{})",
                models.len(),
                if models.len() == 1 { "" } else { "s" }
            );
        }
        Ok(false) => {
            println!("  ⚠️  NPU API not yet ready (still starting?)");
        }
        Err(e) => {
            println!("  ⚠️  Health check error: {e}");
        }
    }

    println!("\n  📍 API:       http://127.0.0.1:{api_port}/v1");
    println!("  📍 Health:    http://127.0.0.1:{api_port}/health\n");

    Ok(())
}

// ── Command: down ────────────────────────────────────────────────

async fn cmd_down() -> Result<()> {
    println!("  🛑 Stopping 1bit NPU stack...\n");

    let ports = [9090, 13305, 9000];
    let mut any_killed = false;

    for port in ports {
        let output = Command::new("fuser")
            .args(["-k", &format!("{port}/tcp")])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();

        if let Ok(status) = output {
            if status.success() {
                println!("  ✅ Killed process on port {port}");
                any_killed = true;
            }
        }

        for proc in &["bitnet_decode", "onebitd"] {
            let _ = Command::new("pkill")
                .args(["-f", proc])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
    }

    if any_killed {
        println!("\n  ✅ NPU stack stopped");
    } else {
        println!("  ℹ️  No NPU processes were running");
    }

    Ok(())
}

// ── Command: status ──────────────────────────────────────────────

async fn cmd_status() -> Result<()> {
    let settings = config::Settings::load()?;

    println!("  ┌─ 1bit NPU Stack Status ──────────────────────────┐\n");

    let client = npu::NpuClient::new(&settings.npu_endpoint)?;
    match client.health_check().await {
        Ok(true) => {
            println!("  ✅  NPU API (port {})     — running", settings.npu.api_port);
            match client.list_models().await {
                Ok(models) => {
                    if !models.is_empty() {
                        let display: Vec<&str> = models.iter().map(|s| s.as_str()).take(5).collect();
                        let suffix = if models.len() > 5 { "..." } else { "" };
                        println!("      Models: {}{}", display.join(", "), suffix);
                    }
                }
                Err(_) => {}
            }
        }
        Ok(false) => {
            println!("  ❌  NPU API (port {})     — not running", settings.npu.api_port);
        }
        Err(e) => {
            println!("  ❌  NPU API (port {})     — error: {e}", settings.npu.api_port);
        }
    }

    for &(port, name) in &[(13305, "Lemond (Chat UI)"), (9000, "Lemond WebSocket")] {
        if is_port_in_use(port).await {
            println!("  ✅  {name:<27} port {port} — running");
        } else {
            println!("  ❌  {name:<27} port {port} — not running");
        }
    }

    // Show MCP tools status
    match mcp::McpManager::load() {
        Ok(manager) => {
            let all_tools = manager.all_tools();
            if !all_tools.is_empty() {
                println!();
                for (server, tool) in &all_tools {
                    println!("  🔌  MCP: {}/{}", server, tool.name);
                }
            }
        }
        Err(e) => {
            println!("  ⚠️  MCP: {e}");
        }
    }

    println!("\n  └──────────────────────────────────────────────────┘");
    println!("  📍 API:   http://127.0.0.1:{}/v1", settings.npu.api_port);
    println!("  📍 Chat:  http://127.0.0.1:13305/\n");

    Ok(())
}

// ── Command: build ───────────────────────────────────────────────

fn cmd_build(dir: &str) -> Result<()> {
    println!("  🔨 Building NPU engine from source...\n");

    let build_sh = format!("{dir}/build_npu.sh");
    let cmake_dir = dir;

    if std::path::Path::new(&build_sh).exists() {
        println!("  Found build_npu.sh — running...");
        let status = Command::new("bash")
            .arg(&build_sh)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .context("Failed to run build script")?;

        if status.success() {
            println!("\n  ✅ NPU engine build complete");
        } else {
            anyhow::bail!("Build script exited with status {status}");
        }
    } else if std::path::Path::new(cmake_dir).join("CMakeLists.txt").exists() {
        println!("  Running cmake...");
        let status = Command::new("cmake")
            .args(["-B", "build", "-G", "Ninja"])
            .current_dir(cmake_dir)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .context("cmake failed")?;

        if !status.success() {
            anyhow::bail!("cmake configuration failed");
        }

        println!("  Running ninja...");
        let status = Command::new("ninja")
            .arg("-j")
            .arg(num_cpus().to_string())
            .current_dir(format!("{cmake_dir}/build"))
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .context("ninja build failed")?;

        if status.success() {
            println!("\n  ✅ NPU engine build complete");
        } else {
            anyhow::bail!("Build failed with status {status}");
        }
    } else {
        anyhow::bail!("No build script or CMakeLists.txt found in {dir}");
    }

    Ok(())
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

// ── Command: config ──────────────────────────────────────────────

fn cmd_config(key: Option<&str>, value: Option<&str>) -> Result<()> {
    let mut settings = config::Settings::load()?;

    match (key, value) {
        (None, None) => {
            println!("  1bit Configuration\n");
            println!("  Theme:                {}", settings.theme);
            println!("  Default provider:     {}", settings.default_provider);
            println!("  Default model:        {}", settings.default_model);
            println!("  NPU endpoint:         {}", settings.npu_endpoint);
            println!("  Thinking level:       {}", settings.thinking_level);
            println!(
                "  Packages:             {}",
                if settings.packages.is_empty() {
                    "none".to_string()
                } else {
                    settings.packages.join(", ")
                }
            );
            println!(
                "  Fallback providers:   {}",
                if settings.fallback_providers.is_empty() {
                    "none".to_string()
                } else {
                    settings.fallback_providers.join(", ")
                }
            );
            println!("\n  NPU settings:");
            println!("    bitnet_decode:      {}", settings.npu.bitnet_decode_path);
            println!("    daemon path:        {}", settings.npu.daemon_path);
            println!("    tune prefill:       {}", settings.npu.tune_prefill);
            println!(
                "    prefill variant:    {}",
                settings
                    .npu
                    .prefill_variant
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "auto".to_string())
            );
            println!("    fp16 weights:       {}", settings.npu.fp16_weights);
            println!("    API port:           {}", settings.npu.api_port);
            println!("    Lemond port:        {}", settings.npu.lemond_port);
            println!(
                "\n  Config file: {}",
                config::settings_path().unwrap_or_default().display()
            );
        }
        (Some(k), None) => {
            match settings.get(k) {
                Ok(val) => println!("{k} = {val}"),
                Err(e) => println!("  ⚠️  {e}"),
            }
        }
        (Some(k), Some(v)) => {
            settings.set(k, v)?;
            println!("  ✅ {k} = {v}");
        }
        (None, Some(v)) => {
            println!("  ⚠️  Cannot set value without a key: {v}");
            println!("       Usage: 1bit config <key> <value>");
        }
    }

    Ok(())
}

// ── Command: serve (HTTP/SSE app server) ─────────────────────────

async fn cmd_serve(port: u16, host: &str, model_override: Option<&str>) -> Result<()> {
    let settings = config::Settings::load()?;
    let model = model_override
        .map(|s| s.to_string())
        .unwrap_or(settings.default_model.clone());

    println!("  🚀 Starting 1bit API server...\n");

    let sessions = session::SessionStore::open()?;
    let npu_client = npu::NpuClient::new(&settings.npu_endpoint)?;

    let provider_router = std::sync::Arc::new(tokio::sync::Mutex::new(
        provider::ProviderRouter::new(&settings.default_provider, vec![], vec![])?,
    ));

    let app = server::build_router(sessions, npu_client, provider_router, model);

    let addr: std::net::SocketAddr = format!("{host}:{port}")
        .parse()
        .context("Invalid host:port")?;

    println!("  📍 API:     http://{addr}/v1");
    println!("  📍 Health:  http://{addr}/health");
    println!("  📍 Models:  http://{addr}/v1/models");
    println!("  📍 Sessions: http://{addr}/v1/sessions");
    println!();

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// ── Command: auth (secrets management) ───────────────────────────

fn cmd_auth(
    provider: Option<&str>,
    set: Option<&str>,
    remove: bool,
    list: bool,
) -> Result<()> {
    if list || (provider.is_none() && !remove && set.is_none()) {
        // List all providers
        let secrets = secrets::SecretsStore::open()?;
        let stored = secrets.list_providers()?;
        let router = provider::ProviderRouter::new("npu", vec![], vec![])?;
        let all_providers = router.list_providers();

        println!("  Provider API Keys\n");
        println!("  {:<20} {:<10} {}", "Provider", "Status", "Source");
        println!("  {}", "-".repeat(50));

        for p in &all_providers {
            let has_env = p.env_var.as_ref().map_or(false, |env| {
                std::env::var(env).is_ok_and(|v| !v.is_empty())
            });
            let has_stored = stored.contains(&p.id);
            let (status, source) = if has_env {
                ("✅", "env")
            } else if has_stored {
                ("✅", "file")
            } else {
                ("⬜", "not set")
            };
            println!("  {:<20} {:<10} {}", p.id, status, source);
        }

        println!();
        println!("  Commands:");
        println!("    1bit auth <provider> --set <key>    Store API key");
        println!("    1bit auth <provider> --remove       Remove stored key");
        println!("    1bit auth --list                    Show all providers");
        println!("    Or set {}_API_KEY env var", all_providers.first().map(|p| p.env_var.as_deref().unwrap_or("PROVIDER")).unwrap_or("PROVIDER"));

        return Ok(());
    }

    let secrets = secrets::SecretsStore::open()?;

    if let Some(prov) = provider {
        if remove {
            secrets.remove_key(prov)?;
            println!("  ✅ Removed API key for '{prov}'");
        } else if let Some(key) = set {
            secrets.set_key(prov, key)?;
            println!("  ✅ Saved API key for '{prov}'");
        } else {
            // Show status for this provider
            if secrets.has_key(prov) {
                println!("  ✅ '{prov}' has a key configured");
            } else {
                println!("  ⬜ No key configured for '{prov}'");
                if let Some(def) = provider::ProviderRouter::new("npu", vec![], vec![])?.get_provider(prov) {
                    if let Some(env) = def.env_var {
                        println!("     Set ${env} or use: 1bit auth {prov} --set <key>");
                    }
                }
            }
        }
    }

    Ok(())
}

// ── Command: update (auto-update) ────────────────────────────────

async fn cmd_update(check: bool) -> Result<()> {
    println!("  🔄 Checking for updates...\n");

    // Fetch latest release from GitHub
    let client = reqwest::Client::new();
    let resp = client
        .get("https://api.github.com/repos/bong-water-water-bong/1bit-systems/releases/latest")
        .header("User-Agent", "1bit-cli")
        .header("Accept", "application/vnd.github.v3+json")
        .timeout(Duration::from_secs(10))
        .send()
        .await;

    match resp {
        Ok(resp) if resp.status().is_success() => {
            let data: serde_json::Value = resp.json().await.unwrap_or_default();
            let latest_tag = data
                .get("tag_name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let html_url = data
                .get("html_url")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            println!("  Current version:  v{VERSION}");
            println!("  Latest release:   {latest_tag}");

            if check {
                println!();
                if html_url.is_empty() {
                    println!("  📍 https://github.com/bong-water-water-bong/1bit-systems/releases");
                } else {
                    println!("  📍 {html_url}");
                }
                return Ok(());
            }

            // Check if we need to update: compare versions
            let latest_ver = latest_tag.trim_start_matches('v');
            if latest_ver != VERSION || latest_tag.contains("nightly") {
                println!("\n  📥 A new version is available!");
                println!();

                // Try to find a release asset matching our platform
                if let Some(assets) = data.get("assets").and_then(|a| a.as_array()) {
                    // Look for linux-amd64 or x86_64-linux binary
                    let asset = assets.iter().find(|a| {
                        a.get("name")
                            .and_then(|n| n.as_str())
                            .map(|n| {
                                n.contains("x86_64") || n.contains("amd64") || n.contains("linux")
                            })
                            .unwrap_or(false)
                    });

                    if let Some(asset) = asset {
                        let download_url = asset
                            .get("browser_download_url")
                            .and_then(|u| u.as_str())
                            .unwrap_or("");

                        if !download_url.is_empty() {
                            println!("  Downloading update...");
                            match download_and_install(download_url).await {
                                Ok(()) => {
                                    println!("  ✅ Update complete! Restart 1bit to use the new version.");
                                    return Ok(());
                                }
                                Err(e) => {
                                    println!("  ⚠️  Auto-update failed: {e}");
                                }
                            }
                        }
                    }
                }

                println!("  Manual update:");
                println!("    cd /path/to/1bit-systems && git pull && cargo build --release");
                println!("    Or download from:");
                if !html_url.is_empty() {
                    println!("    {html_url}");
                }
            } else {
                println!("\n  ✅ You're up to date!");
            }
        }
        Ok(resp) => {
            println!("  ⚠️  GitHub API returned {}", resp.status());
            println!("       Manual: https://github.com/bong-water-water-bong/1bit-systems/releases");
        }
        Err(e) => {
            println!("  ⚠️  Could not check for updates: {e}");
            println!("       Manual: https://github.com/bong-water-water-bong/1bit-systems/releases");
        }
    }

    Ok(())
}

/// Download a binary release and replace the current executable.
async fn download_and_install(url: &str) -> Result<()> {
    let client = reqwest::Client::new();
    let resp = client
        .get(url)
        .header("User-Agent", "1bit-cli")
        .timeout(Duration::from_secs(120))
        .send()
        .await
        .context("Failed to download update")?;

    let bytes = resp
        .bytes()
        .await
        .context("Failed to read update data")?;

    // Get current executable path
    let current_exe = std::env::current_exe()?;
    let backup_path = current_exe.with_extension("bak");

    // Backup current binary
    std::fs::rename(&current_exe, &backup_path)?;

    // Write new binary
    std::fs::write(&current_exe, &bytes)?;

    // Make executable
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&current_exe, std::fs::Permissions::from_mode(0o755))?;
    }

    // Remove backup on success
    std::fs::remove_file(&backup_path).ok();

    Ok(())
}

// ── Helpers ──────────────────────────────────────────────────────

async fn is_port_in_use(port: u16) -> bool {
    tokio::net::TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .is_ok()
}

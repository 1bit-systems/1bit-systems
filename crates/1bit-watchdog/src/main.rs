// 1bit-watchdog — 24h soak upstream tracker for the 1bit-systems C++ media
// stack. Polls GitHub and HuggingFace Hub, dwells a configurable number of
// hours on new commits or releases, then triggers fork-merge + rebuild +
// redeploy per on_merge / on_bump hooks declared in packages.toml.
//
// Intended to run as a systemd --user timer firing hourly. State persists
// at ~/.local/state/1bit-watchdog/state.json so restarts don't reset the
// dwell clock. Each state transition fires a Discord notification via the
// 1bit-mcp-discord bridge.
//
// Subcommands:
//   1bit-watchdog check       — one poll cycle (called by the timer)
//   1bit-watchdog status      — print current state table
//   1bit-watchdog force <id>  — bypass dwell, trigger merge/bump now
//   1bit-watchdog reset <id>  — forget the seen-new SHA, re-arm dwell

use anyhow::{Context, Result};
use chrono::Utc;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process::Command;
use tracing::{info, warn};

mod config;
mod state;

use config::{Manifest, WatchEntry, WatchKind};
use state::{State, Transition};

// Resolved at runtime via xdg config paths in main(). Constant retained as
// a documented placeholder — never read directly. Was hardcoded to
// /home/bcloud/repos/1bit-halo-workspace/packages.toml which broke on every
// non-bcloud box AND on bcloud's box after the workspace rename to 1bit-systems.
const DEFAULT_MANIFEST_FILENAME: &str = "packages.toml";

#[derive(Parser, Debug)]
#[command(
    name = "1bit-watchdog",
    version,
    about = "24h soak upstream tracker for 1bit-systems C++ media stack"
)]
struct Cli {
    /// Alternate packages.toml path.
    #[arg(long, global = true)]
    manifest: Option<String>,

    /// Alternate state file.
    #[arg(long, global = true)]
    state_file: Option<String>,

    /// Dry-run: poll + log, don't run on_merge/on_bump.
    #[arg(long, global = true)]
    dry_run: bool,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// One poll cycle (systemd timer calls this).
    Check,
    /// Print current state table as JSON.
    Status,
    /// Bypass the dwell timer, fire on_merge/on_bump now.
    Force { id: String },
    /// Forget the seen-new SHA, re-arm dwell.
    Reset { id: String },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .compact()
        .init();

    let cli = Cli::parse();
    // Resolve manifest: CLI override -> $XDG_CONFIG_HOME/1bit/packages.toml ->
    // $HOME/.config/1bit/packages.toml -> CWD. Old build hardcoded
    // /home/bcloud/repos/1bit-halo-workspace/packages.toml which broke on
    // every other box.
    let default_manifest = std::env::var("XDG_CONFIG_HOME").ok()
        .map(PathBuf::from)
        .or_else(|| std::env::var("HOME").ok().map(|h| PathBuf::from(h).join(".config")))
        .map(|d| d.join("1bit").join(DEFAULT_MANIFEST_FILENAME))
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MANIFEST_FILENAME));
    let manifest_path: PathBuf = cli.manifest.map(PathBuf::from).unwrap_or(default_manifest);
    let manifest_path_str = manifest_path.to_string_lossy().into_owned();
    let state_path = cli
        .state_file
        .map(PathBuf::from)
        .unwrap_or_else(state::default_path);

    let manifest = Manifest::load(&manifest_path_str)
        .with_context(|| format!("loading watch entries from {manifest_path_str}"))?;
    let mut state = State::load(&state_path).unwrap_or_default();

    match cli.cmd {
        Cmd::Status => {
            let snapshot = serde_json::json!({
                "manifest_path": manifest_path_str,
                "state_path": state_path.display().to_string(),
                "entries": state.entries(),
            });
            println!("{}", serde_json::to_string_pretty(&snapshot)?);
        }
        Cmd::Check => {
            let client = reqwest::Client::builder()
                .user_agent("1bit-watchdog/0.1")
                .build()?;
            for entry in manifest.watch.values() {
                if let Err(e) = poll_entry(&client, entry, &mut state, cli.dry_run).await {
                    warn!(id = %entry.id, error = %e, "poll failed");
                }
            }
            state.save(&state_path)?;
        }
        Cmd::Force { id } => {
            let entry = manifest
                .watch
                .get(&id)
                .with_context(|| format!("no watch entry `{id}` in manifest"))?;
            info!(id = %id, "forcing merge/bump (dwell bypassed)");
            run_hooks(entry, cli.dry_run)?;
            state.mark_merged(&id, Utc::now());
            state.save(&state_path)?;
        }
        Cmd::Reset { id } => {
            state.reset(&id);
            state.save(&state_path)?;
            info!(id = %id, "state cleared; dwell re-armed on next check");
        }
    }
    Ok(())
}

async fn poll_entry(
    client: &reqwest::Client,
    entry: &WatchEntry,
    state: &mut State,
    dry_run: bool,
) -> Result<()> {
    let latest = match entry.kind {
        WatchKind::Github => poll_github(client, &entry.repo, entry.branch.as_deref()).await?,
        WatchKind::Huggingface => poll_huggingface(client, &entry.repo).await?,
    };

    let transition = state.observe(&entry.id, &latest, entry.soak_hours);
    info!(id = %entry.id, latest = %latest, ?transition, "polled");

    match transition {
        Transition::NoChange => {}
        Transition::SeenNew => {
            notify(
                entry,
                &format!("new upstream ref {latest} — dwell {}h", entry.soak_hours),
            );
        }
        Transition::Soaking { remaining_hours } => {
            info!(id = %entry.id, remaining_hours, "still dwelling");
        }
        Transition::SoakComplete => {
            notify(
                entry,
                &format!("soak clean — triggering on_merge/on_bump for {latest}"),
            );
            if !dry_run {
                run_hooks(entry, dry_run)?;
            }
            state.mark_merged(&entry.id, Utc::now());
        }
    }
    Ok(())
}

async fn poll_github(client: &reqwest::Client, repo: &str, branch: Option<&str>) -> Result<String> {
    let branch = branch.unwrap_or("main");
    let url = format!("https://api.github.com/repos/{repo}/commits/{branch}");
    let mut req = client
        .get(&url)
        .header("Accept", "application/vnd.github+json");
    if let Ok(tok) = std::env::var("GH_TOKEN") {
        req = req.bearer_auth(tok);
    }
    let v: serde_json::Value = req.send().await?.error_for_status()?.json().await?;
    let sha = v
        .get("sha")
        .and_then(|s| s.as_str())
        .context("github response missing sha")?;
    Ok(sha.to_string())
}

async fn poll_huggingface(client: &reqwest::Client, repo: &str) -> Result<String> {
    // HF Hub "models" endpoint exposes a sha for the default revision.
    let url = format!("https://huggingface.co/api/models/{repo}");
    let v: serde_json::Value = client
        .get(&url)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let sha = v
        .get("sha")
        .and_then(|s| s.as_str())
        .context("hf response missing sha")?;
    Ok(sha.to_string())
}

fn run_hooks(entry: &WatchEntry, dry_run: bool) -> Result<()> {
    let hooks: &[Vec<String>] = match entry.kind {
        WatchKind::Github => &entry.on_merge,
        WatchKind::Huggingface => &entry.on_bump,
    };
    for argv in hooks {
        if argv.is_empty() {
            continue;
        }
        if dry_run {
            info!(?argv, "dry-run: would run hook");
            continue;
        }
        let status = Command::new(&argv[0])
            .args(&argv[1..])
            .status()
            .with_context(|| format!("spawning {:?}", argv))?;
        if !status.success() {
            anyhow::bail!("hook {argv:?} exit {:?}", status.code());
        }
    }
    Ok(())
}

fn notify(entry: &WatchEntry, msg: &str) {
    // Minimal stdout notification for now. Full Discord wire-up lands once
    // 1bit-mcp-discord exposes a simple `halo notify <channel> <text>` CLI
    // surface; shelling out to that will keep this crate free of reqwest
    // Discord plumbing.
    info!(id = %entry.id, notify = %entry.notify, "{msg}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_parses() {
        Cli::command().debug_assert();
    }

    #[test]
    fn parses_check_subcommand() {
        let cli = Cli::try_parse_from(["1bit-watchdog", "check"]).unwrap();
        matches!(cli.cmd, Cmd::Check);
    }

    #[test]
    fn parses_force_subcommand() {
        let cli = Cli::try_parse_from(["1bit-watchdog", "force", "qwen3-tts-cpp"]).unwrap();
        match cli.cmd {
            Cmd::Force { id } => assert_eq!(id, "qwen3-tts-cpp"),
            _ => panic!("wrong subcommand"),
        }
    }
}

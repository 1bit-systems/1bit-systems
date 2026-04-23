// halo — strix-ai-rs unified CLI.
//
// Rust port of bin/1bit (bash). First crate in the 1bit-systems to prove
// the cargo workspace + tokio + clap + reqwest stack before we touch kernels
// or agents.

use anyhow::Result;
use clap::{Parser, Subcommand};

mod bench;
mod budget;
mod burnin;
mod chat;
mod doctor;
mod install;
mod install_model;
mod logs;
mod memory;
mod npu;
mod oobe_error;
mod power;
mod ppl;
mod preflight;
mod restart;
mod rollback;
mod say;
mod skill;
mod status;
mod update;
mod version;

/// halo — one command, all 1bit systems ops
#[derive(Parser, Debug)]
#[command(name = "halo", about, version, propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// One-line-per-service state snapshot
    Status,
    /// Tail systemd journal for a 1bit service
    Logs {
        /// Service short name (bitnet, sd, whisper, kokoro, lemonade, agent, ...)
        service: String,
        /// Follow
        #[arg(short = 'f', long)]
        follow: bool,
        /// Last N lines
        #[arg(short = 'n', long, default_value_t = 50)]
        lines: u32,
    },
    /// Restart a 1bit service
    Restart { service: String },
    /// Comprehensive health check across the stack
    Doctor,
    /// Check for / install a signed release (default: `--check`).
    ///
    /// Legacy git-rebuild path is still reachable via `--legacy-rebuild`
    /// (or the transitional `--no-build` / `--no-restart` flags) until
    /// the signed atomic-install ceremony lands.
    Update {
        /// Probe the release feed and report available updates. Exits
        /// 0 if up to date, 1 if an update is available, 2 on feed error.
        #[arg(long)]
        check: bool,
        /// Download + sha256+minisign-verify the latest artifact, then
        /// stop short of overwriting the running binary. Real install
        /// atomics come in a later pass.
        #[arg(long)]
        install: bool,
        /// Legacy flag (git-rebuild path): skip cargo build phase.
        #[arg(long)]
        no_build: bool,
        /// Legacy flag (git-rebuild path): skip systemctl restart.
        #[arg(long)]
        no_restart: bool,
        /// Explicitly select the legacy git-pull + rebuild + restart
        /// path. Equivalent to the pre-release-feed `1bit update`
        /// behaviour; kept so build-box flows don't break mid-transition.
        #[arg(long)]
        legacy_rebuild: bool,
    },
    /// 1bit stack version + component SHAs
    Version,
    /// Install a component from packages.toml (core, agents, voice, sd, ...)
    Install {
        /// Component name; omit with --list to see the full catalogue
        component: Option<String>,
        /// List available components
        #[arg(long)]
        list: bool,
        /// Run the fresh-box OOBE flow: preflight gates, diagnostic
        /// errors, sensible-default component (`core`). Non-interactive.
        #[arg(long)]
        oobe: bool,
        /// Skip the cargo build / install phase. Useful when the binary
        /// is already in `target/release` (CI) or when the operator
        /// only wants to run preflight.
        #[arg(long)]
        skip_build: bool,
        /// Anchor #9 (non-interactive): auto-answer yes to every
        /// confirmation prompt. Required for ansible / GitHub Actions.
        #[arg(long)]
        yes: bool,
        /// Anchor #7 escape hatch: skip the tail-end `1bit doctor`
        /// probe. CI uses this because CI hosts have no gfx1151 + no
        /// user systemd bus.
        #[arg(long)]
        doctor_skip: bool,
    },
    /// Anchor #6 — rollback to a snapper snapshot (defaults to the
    /// latest `.halo-preinstall` if no number is given).
    Rollback {
        /// Explicit snapshot number. Omit for auto-pick of the latest
        /// `.halo-preinstall` snapshot.
        snapshot: Option<u32>,
        /// Skip the confirmation prompt (anchor #9).
        #[arg(long)]
        yes: bool,
    },
    /// Speak text through 1bit-halo-kokoro :8083 + the host's audio player
    Say {
        /// The text to synthesize. Quote it for multi-word input.
        text: Vec<String>,
        /// Voice id (see `curl :8083/voices` for the list)
        #[arg(short = 'v', long)]
        voice: Option<String>,
        /// Playback speed (0 < x ≤ 4)
        #[arg(short = 's', long, default_value_t = 1.0)]
        speed: f32,
    },
    /// Interactive one-shot REPL against 1bit-server :8180
    Chat {
        #[arg(long)]
        url: Option<String>,
        #[arg(long)]
        model: Option<String>,
        #[arg(long, default_value_t = 128)]
        max_tokens: u32,
    },
    /// Shadow-burnin summary (parity vs gen-1 C++ :8080)
    Bench {
        /// If set, run N new rounds before printing the summary
        #[arg(long)]
        rounds: Option<u32>,
        /// Only count rounds after this ISO-8601 timestamp
        #[arg(long)]
        since: Option<String>,
    },
    /// Perplexity against gen-1 wikitext baseline (9.1607)
    Ppl {
        #[arg(long)]
        url: Option<String>,
        #[arg(long, default_value_t = 1024)]
        stride: u32,
        #[arg(long, default_value_t = 1024)]
        max_tokens: u32,
        /// Bytes of wikitext to send (from the start of the file)
        #[arg(long, default_value_t = 6000)]
        bytes: usize,
    },
    /// Apply / query Ryzen APU power profile (wraps FlyGoat/ryzenadj)
    Power {
        /// Profile to apply. Omit to print current state.
        profile: Option<String>,
        /// Print what would run without executing
        #[arg(long)]
        dry_run: bool,
        /// List available profiles and exit
        #[arg(long)]
        list: bool,
    },
    /// Manage SKILL.md files under ~/.halo/skills/ (operator-facing CRUD).
    Skill {
        #[command(subcommand)]
        cmd: skill::SkillCmd,
    },
    /// Manage MEMORY.md + USER.md under ~/.halo/memories/ (operator-facing).
    Memory {
        #[command(subcommand)]
        cmd: memory::MemoryCmd,
    },
    /// XDNA 2 NPU diagnostics — xrt-smi wrapper + firmware / memlock probes.
    Npu {
        #[command(subcommand)]
        cmd: npu::NpuCmd,
    },
    /// Shadow-burnin log analyzer (byte-exact rate + drift patterns).
    ///
    /// With no subcommand prints a one-line summary and exits 0 if the
    /// byte-exact rate is ≥ 95%, 1 otherwise.
    Burnin {
        #[command(subcommand)]
        cmd: Option<burnin::BurninCmd>,
    },
    /// GTT + RAM budget audit for concurrent-model scaling
    Budget,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "halo=info".into()),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Status => status::run().await,
        Cmd::Logs {
            service,
            follow,
            lines,
        } => logs::run(&service, follow, lines).await,
        Cmd::Restart { service } => restart::run(&service).await,
        Cmd::Doctor => doctor::run().await,
        Cmd::Update {
            check,
            install,
            no_build,
            no_restart,
            legacy_rebuild,
        } => update::run(check, install, no_build, no_restart, legacy_rebuild).await,
        Cmd::Version => version::run().await,
        Cmd::Install {
            component,
            list,
            oobe,
            skip_build,
            yes,
            doctor_skip,
        } => {
            if oobe {
                let defaults = install::OobeDefaults {
                    component: component.unwrap_or_else(|| "core".into()),
                    skip_build,
                    yes,
                    doctor_skip,
                };
                install::run_oobe(defaults).await
            } else {
                match (list, component) {
                    (true, _) | (false, None) => install::list().await,
                    (false, Some(name)) => install::run_install(&name).await,
                }
            }
        }
        Cmd::Rollback { snapshot, yes } => rollback::run(snapshot, yes).await,
        Cmd::Say { text, voice, speed } => {
            let phrase = text.join(" ");
            let voice_owned: String = voice.unwrap_or_else(|| say::default_voice().into());
            say::run(&phrase, &voice_owned, speed).await
        }
        Cmd::Chat {
            url,
            model,
            max_tokens,
        } => chat::run(url, model, max_tokens).await,
        Cmd::Bench { rounds, since } => bench::run(rounds, since).await,
        Cmd::Ppl {
            url,
            stride,
            max_tokens,
            bytes,
        } => ppl::run(url, stride, max_tokens, bytes).await,
        Cmd::Power {
            profile,
            dry_run,
            list,
        } => power::run(profile, dry_run, list),
        Cmd::Skill { cmd } => skill::run(cmd),
        Cmd::Memory { cmd } => memory::run(cmd),
        Cmd::Npu { cmd } => npu::run(cmd),
        Cmd::Burnin { cmd } => burnin::run(cmd).await,
        Cmd::Budget => budget::run().await,
    }
}

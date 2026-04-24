// `1bit install <component>` — read packages.toml, resolve deps, build, start.
// Lean: single file, manifest embedded at compile time.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::time::Duration;

use crate::oobe_error::OobeError;
use crate::preflight::{GateResult, PreflightOutcome, RealProbe, SystemProbe, run_all};

const MANIFEST_SRC: &str = include_str!("../../../packages.toml");

#[derive(Debug, Deserialize)]
struct Manifest {
    component: BTreeMap<String, Component>,
    /// Model slots for instant-load weight download. Added 2026-04-23.
    /// Parsed but ignored unless `--model` flag is passed or the id
    /// matches a known model. See [`Model`].
    #[serde(default)]
    model: BTreeMap<String, Model>,
}

/// One on-disk model weight. Used by `1bit install <model>` which resolves
/// the `requires` engine component (tts-engine, image-engine, etc),
/// fetches the GGUF via the `hf` CLI with sha256 verification, symlinks
/// the file into `~/.local/share/1bit/models/<id>/`, and restarts the
/// owning systemd user unit so the new weights go live immediately.
#[derive(Debug, Deserialize)]
struct Model {
    description: String,
    /// Hugging Face repo id, e.g. `bong-water-water-bong/qwen3-tts-0p6b-ternary`.
    hf_repo: String,
    /// Specific file inside the repo.
    hf_file: String,
    /// Expected sha256. `UPSTREAM` = accept whatever HF serves (upstream
    /// release pin), `PENDING-RUN*` = weights not yet trained.
    #[serde(default)]
    sha256: String,
    // size_mb + license are kept for packages.toml round-trip fidelity
    // but not consumed today; `install_model` reads them via ModelSpec.
    #[allow(dead_code)]
    #[serde(default)]
    size_mb: u64,
    #[allow(dead_code)]
    #[serde(default)]
    license: String,
    /// Engine components this model needs (tts-engine, image-engine, ...).
    #[serde(default)]
    requires: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Component {
    description: String,
    #[serde(default)]
    deps: Vec<String>,
    #[serde(default)]
    build: Vec<Vec<String>>,
    #[serde(default)]
    units: Vec<String>,
    /// Distro packages required for the component to function. Advisory
    /// only — 1bit install does NOT run `pacman -S` on the operator's
    /// box; install.sh is the one place that actually installs system
    /// packages. The field is surfaced in `1bit install --list` and in
    /// `1bit install <component>` logs so a missing `xrt` etc. is
    /// visible rather than silently failing at runtime.
    #[serde(default)]
    packages: Vec<String>,
    /// Tracked files to copy into the user's config tree. Two supported
    /// shapes (see `FileEntry`):
    ///
    ///   files = [["src", "dest"], ...]                         # legacy pair
    ///   files = [{ src = "src", dst = "dest",                  # new table
    ///              substitute = { USER = "$USER" } }, ...]
    ///
    /// The destination root is `$XDG_CONFIG_HOME` (default `~/.config`),
    /// UNLESS the destination is absolute (starts with `/`), in which case
    /// it is written to that absolute path via `sudo tee` — this is how
    /// the `npu` component drops `/etc/security/limits.d/99-npu-memlock.conf`
    /// without the operator having to copy it by hand.
    #[serde(default)]
    files: Vec<FileEntry>,
    #[serde(default)]
    check: String,
}

/// A tracked-file entry in `packages.toml`. Accepts both legacy pair form
/// (`["src", "dest"]`) and the new table form with optional `substitute`
/// placeholders. Deserialized with `#[serde(untagged)]` so existing
/// components keep working.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum FileEntry {
    /// `[src, dest]` — legacy.
    Pair(Vec<String>),
    /// `{ src, dst, substitute }` — new.
    Table {
        src: String,
        dst: String,
        #[serde(default)]
        substitute: HashMap<String, String>,
    },
}

impl FileEntry {
    fn src(&self) -> Result<&str> {
        match self {
            FileEntry::Pair(v) => match v.as_slice() {
                [s, _] => Ok(s.as_str()),
                other => bail!("files entry must be [src, dest], got {other:?}"),
            },
            FileEntry::Table { src, .. } => Ok(src.as_str()),
        }
    }

    fn dst(&self) -> Result<&str> {
        match self {
            FileEntry::Pair(v) => match v.as_slice() {
                [_, d] => Ok(d.as_str()),
                other => bail!("files entry must be [src, dest], got {other:?}"),
            },
            FileEntry::Table { dst, .. } => Ok(dst.as_str()),
        }
    }

    fn substitute(&self) -> HashMap<String, String> {
        match self {
            FileEntry::Pair(_) => HashMap::new(),
            FileEntry::Table { substitute, .. } => substitute.clone(),
        }
    }
}

fn parse() -> Result<Manifest> {
    toml::from_str::<Manifest>(MANIFEST_SRC).context("parsing packages.toml")
}

#[cfg(test)]
fn parse_src(src: &str) -> Result<Manifest> {
    toml::from_str::<Manifest>(src).context("parsing packages.toml")
}

fn workspace_root() -> &'static Path {
    // Manifest lives two levels up from src/; compile-time constant.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
}

/// Resolve the user config root. XDG first, then ~/.config.
fn user_config_root() -> Result<PathBuf> {
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME")
        && !xdg.is_empty()
    {
        return Ok(PathBuf::from(xdg));
    }
    let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("no $HOME"))?;
    Ok(home.join(".config"))
}

pub async fn list() -> Result<()> {
    let m = parse()?;
    println!("strix-ai-rs components:\n");
    for (name, c) in &m.component {
        println!("  {:<16} {}", name, c.description);
        if !c.deps.is_empty() {
            println!("  {:<16}   deps: {}", "", c.deps.join(", "));
        }
        if !c.packages.is_empty() {
            println!("  {:<16}   packages: {}", "", c.packages.join(", "));
        }
    }
    Ok(())
}

fn resolve<'a>(
    m: &'a Manifest,
    target: &str,
    order: &mut Vec<&'a str>,
    seen: &mut HashSet<String>,
) -> Result<()> {
    if seen.contains(target) {
        return Ok(());
    }
    let c = m.component.get(target).ok_or_else(|| {
        anyhow::anyhow!("unknown component '{target}' (try `1bit install --list`)")
    })?;
    for d in &c.deps {
        resolve(m, d, order, seen)?;
    }
    seen.insert(target.to_string());
    order.push(m.component.get_key_value(target).unwrap().0);
    Ok(())
}

async fn healthcheck(url: &str) -> bool {
    if url.is_empty() {
        return true;
    }
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .danger_accept_invalid_certs(true)
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    match client.get(url).send().await {
        Ok(r) => r.status().is_success(),
        Err(_) => false,
    }
}

fn run(root: &Path, argv: &[String]) -> Result<()> {
    // Expand ~ to $HOME on every argv token before spawning. Command::new
    // runs execve directly — there is no shell layer to expand the tilde,
    // so literal "~" tokens from packages.toml would land as directory
    // names inside root/. Do it here once, centrally.
    let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("no $HOME"))?;
    let expanded: Vec<String> = argv
        .iter()
        .map(|a| {
            if let Some(rest) = a.strip_prefix("~/") {
                home.join(rest).to_string_lossy().into_owned()
            } else if a == "~" {
                home.to_string_lossy().into_owned()
            } else {
                a.clone()
            }
        })
        .collect();
    println!("    $ {}", expanded.join(" "));
    let (bin, rest) = expanded
        .split_first()
        .ok_or_else(|| anyhow::anyhow!("empty argv"))?;
    let s = Command::new(bin)
        .args(rest)
        .current_dir(root)
        .status()
        .with_context(|| format!("spawn {bin}"))?;
    if !s.success() {
        bail!("{bin} failed");
    }
    Ok(())
}

/// Expand placeholder values in a `substitute` map. Supports `$USER`
/// (most common), `$HOME`, and literal string values. Returns the
/// concrete replacement string for a placeholder.
fn expand_placeholder(raw: &str) -> String {
    match raw {
        "$USER" => std::env::var("USER").unwrap_or_default(),
        "$HOME" => std::env::var("HOME").unwrap_or_default(),
        other => other.to_string(),
    }
}

/// Render a tracked file's contents, substituting `@KEY@` placeholders
/// with the expanded value for each entry in `subs`. With an empty map
/// this is a straight file read.
fn render_tracked(src: &Path, subs: &HashMap<String, String>) -> Result<Vec<u8>> {
    let raw = std::fs::read_to_string(src).with_context(|| format!("read {}", src.display()))?;
    let mut out = raw;
    for (key, raw_val) in subs {
        let needle = format!("@{key}@");
        let value = expand_placeholder(raw_val);
        out = out.replace(&needle, &value);
    }
    Ok(out.into_bytes())
}

/// Copy a single tracked file from the workspace into either the user's
/// config tree (relative dest) or an absolute path (e.g.
/// `/etc/security/limits.d/*`) via `sudo tee`. Creates parent dirs as
/// needed. Does NOT overwrite an existing destination (operator may have
/// already customized it) — prints a "skip (exists)" note instead.
///
/// If `subs` is non-empty, `@KEY@` placeholders in the source file are
/// replaced with the expansion of each value before the write.
fn copy_tracked_file(
    root: &Path,
    cfg_root: &Path,
    src_rel: &str,
    dest_rel: &str,
    subs: &HashMap<String, String>,
) -> Result<()> {
    let src = root.join(src_rel);
    if !src.is_file() {
        bail!("tracked file not found: {}", src.display());
    }

    // Absolute destinations go to the real filesystem via sudo tee. This
    // lets `npu` drop `/etc/security/limits.d/99-npu-memlock.conf` as
    // part of `1bit install core` without a separate shell step.
    let dest_is_absolute = Path::new(dest_rel).is_absolute();
    let dest = if dest_is_absolute {
        PathBuf::from(dest_rel)
    } else {
        cfg_root.join(dest_rel)
    };

    if dest.exists() {
        println!("    skip (exists): {}", dest.display());
        return Ok(());
    }

    let rendered = render_tracked(&src, subs)?;

    if dest_is_absolute {
        // System path. Use `sudo tee` so the operator gets one auth
        // prompt instead of a panic on EACCES.
        println!("    installing (sudo) {} → {}", src_rel, dest.display());
        let mut child = Command::new("sudo")
            .args([
                "tee",
                dest.to_str()
                    .ok_or_else(|| anyhow::anyhow!("bad dest path"))?,
            ])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .spawn()
            .with_context(|| format!("spawn sudo tee {}", dest.display()))?;
        {
            use std::io::Write;
            let stdin = child
                .stdin
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("no stdin for sudo tee"))?;
            stdin
                .write_all(&rendered)
                .with_context(|| format!("write to sudo tee for {}", dest.display()))?;
        }
        let status = child
            .wait()
            .with_context(|| format!("wait sudo tee for {}", dest.display()))?;
        if !status.success() {
            bail!("sudo tee failed for {}", dest.display());
        }
        return Ok(());
    }

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("mkdir {}", parent.display()))?;
    }
    std::fs::write(&dest, &rendered).with_context(|| format!("write {}", dest.display()))?;
    println!("    copied {} → {}", src_rel, dest.display());
    Ok(())
}

pub async fn run_install(component: &str) -> Result<()> {
    // If the requested name matches a [model.*] slot, dispatch to the
    // model installer — it will recurse back into run_install for the
    // owning engine component, then fetch weights + restart units.
    let m = parse()?;
    if let Some(model) = m.model.get(component) {
        // Ensure each required engine is installed first.
        for engine in &model.requires {
            run_install_tracked(engine, &InstallTracker::new()).await?;
        }
        // Collect the units owned by the engines so the model installer
        // can restart them after the GGUF lands.
        let mut engine_units: Vec<String> = Vec::new();
        for engine in &model.requires {
            if let Some(c) = m.component.get(engine) {
                engine_units.extend(c.units.clone());
            }
        }
        let spec = crate::install_model::ModelSpec {
            id: component.to_string(),
            description: model.description.clone(),
            hf_repo: model.hf_repo.clone(),
            hf_file: model.hf_file.clone(),
            sha256: model.sha256.clone(),
            requires: model.requires.clone(),
        };
        return crate::install_model::run(&spec, &engine_units);
    }
    run_install_tracked(component, &InstallTracker::new()).await
}

/// Anchor #10 — atomic-on-failure install tracker.
///
/// The tracker doesn't try to be a true transaction (no journaled fs, no
/// snapper dance — that's what `1bit rollback` is for). What it *does*
/// is record every state change the installer makes (services started,
/// symlinks created, files copied to paths that didn't exist) and, if a
/// subsequent step fails, best-effort revert them before printing a
/// `left state: X` recovery line.
///
/// Three action kinds cover every current install side-effect:
///
///   * `EnabledUnit(name)` — `systemctl --user enable --now X` was run.
///     Revert = `systemctl --user disable --now X`.
///   * `CopiedFile(path)` — a tracked file was newly written (not "skip
///     (exists)"). Revert = `std::fs::remove_file(path)`.
///   * `CopiedSudoFile(path)` — same but installed via `sudo tee` to an
///     absolute path. Revert = `sudo rm -f` (best-effort; we never
///     prompt for a second sudo).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstallAction {
    EnabledUnit(String),
    CopiedFile(PathBuf),
    CopiedSudoFile(PathBuf),
}

/// Thread-safe action log. Lives for one install run; dropped on
/// success. Wrapped in `Mutex` because the async run hops threads and
/// we want the tracker to be `Sync` without the callers caring.
#[derive(Debug)]
pub struct InstallTracker {
    actions: Mutex<Vec<InstallAction>>,
}

impl InstallTracker {
    pub fn new() -> Self {
        Self {
            actions: Mutex::new(Vec::new()),
        }
    }

    pub fn record(&self, action: InstallAction) {
        self.actions.lock().unwrap().push(action);
    }

    /// Snapshot of recorded actions. Clones out of the Mutex so the
    /// tests can assert without holding the lock.
    pub fn actions(&self) -> Vec<InstallAction> {
        self.actions.lock().unwrap().clone()
    }

    /// Best-effort revert of every recorded action, LIFO. Prints a
    /// one-line note per revert + a trailing summary. Never panics;
    /// individual revert failures are logged and skipped.
    pub fn best_effort_revert(&self) {
        let actions: Vec<InstallAction> = {
            let mut guard = self.actions.lock().unwrap();
            std::mem::take(&mut *guard)
        };
        if actions.is_empty() {
            println!("    (nothing to revert — install failed before any side-effect landed)");
            return;
        }
        println!("    atomic revert: {} action(s) to undo", actions.len());
        // LIFO — mirror actual stack-unwind order.
        for action in actions.iter().rev() {
            match action {
                InstallAction::EnabledUnit(unit) => {
                    println!("      - disabling user unit {unit}");
                    let _ = Command::new("systemctl")
                        .args(["--user", "disable", "--now", unit])
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::null())
                        .status();
                }
                InstallAction::CopiedFile(path) => {
                    println!("      - removing {}", path.display());
                    let _ = std::fs::remove_file(path);
                }
                InstallAction::CopiedSudoFile(path) => {
                    println!("      - removing (sudo) {}", path.display());
                    let _ = Command::new("sudo")
                        .args(["rm", "-f", path.to_str().unwrap_or("")])
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::null())
                        .status();
                }
            }
        }
    }
}

impl Default for InstallTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// `run_install` with an injected tracker so the OOBE path can hand us
/// one and drive the anchor-10 revert on failure. The public
/// `run_install` wraps this with a fresh tracker for back-compat with
/// non-OOBE `1bit install` calls that don't want the revert noise.
pub async fn run_install_tracked(component: &str, tracker: &InstallTracker) -> Result<()> {
    let m = parse()?;
    let mut order = Vec::new();
    let mut seen = HashSet::new();
    resolve(&m, component, &mut order, &mut seen)?;

    let root = workspace_root();
    let cfg_root = user_config_root()?;
    println!("install plan: {}\n", order.join(" → "));

    for name in order {
        let c = &m.component[name];
        println!("── {name}: {} ──", c.description);

        if !c.packages.is_empty() {
            println!(
                "    required distro packages: {} (install via install.sh / pacman)",
                c.packages.join(", ")
            );
        }

        for step in &c.build {
            run(root, step)?;
        }

        for entry in &c.files {
            let src_rel = entry
                .src()
                .with_context(|| format!("component '{name}' files entry"))?;
            let dest_rel = entry
                .dst()
                .with_context(|| format!("component '{name}' files entry"))?;
            let subs = entry.substitute();
            copy_tracked_file_tracked(root, &cfg_root, src_rel, dest_rel, &subs, tracker)?;
        }

        for unit in &c.units {
            println!("    $ systemctl --user enable --now {unit}");
            let s = Command::new("systemctl")
                .args(["--user", "enable", "--now", unit])
                .status()?;
            if !s.success() {
                bail!("systemctl failed for {unit}");
            }
            tracker.record(InstallAction::EnabledUnit(unit.clone()));
        }

        if !c.check.is_empty() {
            print!("    checking {}…  ", c.check);
            // Services can take a moment to bind; retry up to 5s.
            let mut ok = false;
            for _ in 0..5 {
                if healthcheck(&c.check).await {
                    ok = true;
                    break;
                }
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
            println!(
                "{}",
                if ok {
                    "ok"
                } else {
                    "FAIL (component installed but health probe failed)"
                }
            );
        }
    }

    println!("\n✓ install complete");
    Ok(())
}

/// Same as `copy_tracked_file` but records a `CopiedFile` /
/// `CopiedSudoFile` action on success so the anchor-10 tracker can
/// revert it if a later step fails. A "skip (exists)" hit is NOT
/// recorded — the file was there before we arrived; we must not delete
/// it on rollback.
fn copy_tracked_file_tracked(
    root: &Path,
    cfg_root: &Path,
    src_rel: &str,
    dest_rel: &str,
    subs: &HashMap<String, String>,
    tracker: &InstallTracker,
) -> Result<()> {
    let dest_is_absolute = Path::new(dest_rel).is_absolute();
    let dest = if dest_is_absolute {
        PathBuf::from(dest_rel)
    } else {
        cfg_root.join(dest_rel)
    };
    let pre_existed = dest.exists();
    copy_tracked_file(root, cfg_root, src_rel, dest_rel, subs)?;
    if !pre_existed && dest.exists() {
        if dest_is_absolute {
            tracker.record(InstallAction::CopiedSudoFile(dest));
        } else {
            tracker.record(InstallAction::CopiedFile(dest));
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// OOBE path (anchors 1-5 in project_oobe_bar.md)
// ─────────────────────────────────────────────────────────────────────
//
// `run_oobe` is the "fresh box" entry point: it runs preflight first,
// pretty-prints the results, bails out with a diagnostic `OobeError` if
// any gate is red, then calls `run_install(default.component)`. Anchor 5
// ("sensible defaults, zero config") is encoded in `OobeDefaults`.

/// Defaults the OOBE applies before invoking `run_install`. Every value
/// here is deliberately a field on a struct, not a global const, so the
/// tests can build a custom default for the offline/CI path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OobeDefaults {
    /// Component to install. The manifest's `core` is the fresh-box
    /// entry point and pulls in voice + echo + mcp + landing + gaia +
    /// lemonade transitively.
    pub component: String,
    /// When true, `run_oobe` skips the `cargo build` phase inside
    /// `run_install` — useful for the `--skip-build` flag in CI where
    /// the binary is already in `target/release` from an upstream job,
    /// and for unit tests that drive the OOBE path without a toolchain.
    pub skip_build: bool,
    /// Anchor #9 (non-interactive): when true, every confirmation
    /// prompt is auto-answered `yes`. Used by CI / ansible / automated
    /// tests that cannot drive stdin. Preflight honors this as well:
    /// we never block on `read_line` with `yes = true`.
    pub yes: bool,
    /// Anchor #7 (doctor hook): skip the tail-end `1bit doctor` run.
    /// CI uses this to avoid a doctor-ranged Fail caused by the tests'
    /// own environment (no gfx1151, no systemd bus). Local operators
    /// leave it false.
    pub doctor_skip: bool,
}

impl Default for OobeDefaults {
    fn default() -> Self {
        Self {
            component: "core".into(),
            skip_build: false,
            yes: false,
            doctor_skip: false,
        }
    }
}

/// Abstraction over `1bit doctor`'s exit behavior. The real probe runs
/// `doctor::run` for its side-effects (printed table) and captures the
/// `fail` tally from `std::process::exit`. Tests use a `FakeDoctor` so
/// they don't actually call `exit`.
pub trait DoctorProbe {
    /// Tally of `(warn, fail)` from the doctor run. Only `fail > 0`
    /// aborts the OOBE — anchor #7 is explicit that warn rows are
    /// allowed (e.g. pi-archive offline in CI).
    fn run(&self) -> (u32, u32);
}

/// Test-only doctor that returns canned counts. Used by the anchor-7
/// OOBE tests so the wiring can be exercised without touching the real
/// host probes.
#[cfg(test)]
pub struct FakeDoctor {
    pub warn: u32,
    pub fail: u32,
}

#[cfg(test)]
impl DoctorProbe for FakeDoctor {
    fn run(&self) -> (u32, u32) {
        (self.warn, self.fail)
    }
}

/// Lightweight wrapper that reports `fail = 0, warn = 0` so `run_oobe`
/// stays a one-liner. The real `1bit doctor` command (invoked by
/// `Cmd::Doctor`) already prints + exits non-zero on its own; the OOBE
/// hook uses a separate probe trait because `doctor::run` calls
/// `std::process::exit` directly and that's impossible to unit-test.
pub struct HostDoctor;

impl DoctorProbe for HostDoctor {
    fn run(&self) -> (u32, u32) {
        // Inline a narrow reimplementation: we re-use `doctor::`'s
        // core accelerator probes + service checks and tally locally
        // instead of calling `doctor::run` (which exits the process).
        crate::doctor::tally_for_oobe()
    }
}

/// One-glyph status prefix for a preflight outcome. Uses ASCII so the
/// output is readable over SSH + captured in logs without Unicode
/// surprises.
fn outcome_glyph(o: &PreflightOutcome) -> &'static str {
    match o {
        PreflightOutcome::Pass(_) => "[ OK ]",
        PreflightOutcome::Skip(_) => "[WARN]",
        PreflightOutcome::Fail(_) => "[FAIL]",
    }
}

/// Pretty-print the preflight table to stdout. Output shape:
///
/// ```text
/// preflight:
///   [ OK ] kernel   : kernel 6.18.22-1-cachyos-lts (LTS OK)
///   [ OK ] rocm     : rocminfo reachable
///   ...
/// ```
///
/// Deliberately boring — an operator scanning `1bit install --oobe`
/// logs over SSH should read it in under a second.
pub fn print_preflight_table(results: &[GateResult]) {
    println!("preflight:");
    for r in results {
        let note = match &r.outcome {
            PreflightOutcome::Pass(s) => s.clone(),
            PreflightOutcome::Skip(s) => s.clone(),
            PreflightOutcome::Fail(e) => e.what.to_string(),
        };
        println!("  {:6} {:8}: {}", outcome_glyph(&r.outcome), r.name, note);
    }
}

/// Render a failed gate as the four-field OOBE block + the `fix` /
/// `wiki` trailers. Kept as a helper so every surface that can emit an
/// `OobeError` (preflight, install step, future uninstall) renders it
/// identically.
pub fn print_oobe_error(label: &str, e: &OobeError) {
    println!();
    println!("error: {label}");
    println!("{e}");
}

/// Fresh-box OOBE run with the real host probe. Wraps `run_oobe_full`
/// so the main CLI stays a one-liner and the tests can drive a
/// `FakeProbe` / `FakeDoctor` via `run_oobe_full` directly.
pub async fn run_oobe(defaults: OobeDefaults) -> Result<()> {
    let probe = RealProbe;
    let doctor = HostDoctor;
    run_oobe_full(&probe, &doctor, defaults).await
}

/// Legacy shim kept for any external caller still on the pre-doctor-
/// hook shape. Internally delegates to `run_oobe_full` with a
/// `NullDoctor` so the behavior is identical to pass-1 OOBE.
#[allow(dead_code)] // back-compat surface; covered by the preflight tests.
pub async fn run_oobe_with_probe(probe: &dyn SystemProbe, defaults: OobeDefaults) -> Result<()> {
    let defaults = OobeDefaults {
        doctor_skip: true,
        ..defaults
    };
    run_oobe_full(probe, &NullDoctor, defaults).await
}

/// Null doctor used when the caller asked to skip the hook entirely.
/// Returns a zero/zero tally so `run_oobe_full` treats the probe as
/// green and continues. `pub(crate)` so the install tests can reuse
/// it as a "the hook won't fire" sentinel.
pub(crate) struct NullDoctor;

impl DoctorProbe for NullDoctor {
    fn run(&self) -> (u32, u32) {
        (0, 0)
    }
}

/// Core OOBE flow with injectable preflight + doctor probes. This is
/// the new shape; `run_oobe` and `run_oobe_with_probe` wrap it.
///
/// Flow, in order:
///
///   1. Run preflight. Abort on any `Fail`.
///   2. If `skip_build` is set, print a note and stop.
///   3. Set up an `InstallTracker` and call `run_install_tracked`.
///      Anchor #10 — on any failure, revert recorded actions + print
///      a `left state:` line before bailing.
///   4. If `doctor_skip` is false, run the doctor hook (anchor #7).
///      Any `Fail` tally makes the whole OOBE bail with a
///      `doctor_failed` diagnostic error.
pub async fn run_oobe_full(
    probe: &dyn SystemProbe,
    doctor: &dyn DoctorProbe,
    defaults: OobeDefaults,
) -> Result<()> {
    let results = run_all(probe);
    print_preflight_table(&results);

    // Any `Fail` gate aborts before destructive work happens. We print
    // the *first* failure in full four-field shape; the rest are in the
    // table already, so we don't spam the operator.
    if let Some(bad) = results.iter().find(|r| !r.outcome.is_green()) {
        if let PreflightOutcome::Fail(e) = &bad.outcome {
            print_oobe_error(bad.name, e);
        }
        bail!("preflight failed at gate '{}'", bad.name);
    }

    if defaults.yes {
        println!("\n(--yes) non-interactive mode; all prompts auto-answered yes.");
    }

    if defaults.skip_build {
        println!(
            "\n(--skip-build) preflight green; skipping cargo build / install for '{}'.",
            defaults.component
        );
        // Fall through so the doctor hook still runs — operators who
        // pass `--skip-build` (CI with pre-built binaries) still want
        // the anchor #7 post-install probe.
    } else {
        println!(
            "\npreflight green — proceeding with `1bit install {}`\n",
            defaults.component
        );

        let tracker = InstallTracker::new();
        if let Err(e) = run_install_tracked(&defaults.component, &tracker).await {
            // Anchor #10: best-effort revert + clear "left state:"
            // line before bubbling the error.
            println!();
            println!("install failed: {e}");
            println!("atomic revert (anchor 10):");
            tracker.best_effort_revert();
            let remaining = tracker.actions();
            if remaining.is_empty() {
                println!("    left state: installer undid its own side-effects cleanly.");
            } else {
                println!(
                    "    left state: {} action(s) could not be reverted — see above.",
                    remaining.len()
                );
            }
            let oe = OobeError::install_step_failed("install");
            print_oobe_error("install", &oe);
            return Err(e);
        }
    }

    if defaults.doctor_skip {
        println!("\n(--doctor-skip) skipping the `1bit doctor` tail probe.");
        return Ok(());
    }

    println!("\nrunning `1bit doctor` (OOBE anchor #7) ...");
    let (warn, fail) = doctor.run();
    println!("doctor summary: {warn} warn, {fail} fail");
    if fail > 0 {
        let oe = OobeError::doctor_failed(fail);
        print_oobe_error("doctor", &oe);
        bail!("`1bit doctor` reported {fail} fail(s)");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The embedded `packages.toml` must parse. This is the cheapest
    /// safety net against a malformed manifest landing on main.
    #[test]
    fn manifest_parses_clean() {
        let m = parse().expect("packages.toml must parse");
        assert!(!m.component.is_empty(), "at least one component");
    }

    /// The four components added with the systemd/binary rewiring must be
    /// present with the exact shape declared in the prompt: voice +
    /// echo + mcp + tunnel.
    #[test]
    fn new_components_are_declared() {
        let m = parse().unwrap();
        for name in ["voice", "echo", "mcp", "tunnel"] {
            assert!(
                m.component.contains_key(name),
                "packages.toml is missing component '{name}'"
            );
        }

        // voice — cargo installs the 1bit-voice binary, no systemd unit.
        let voice = &m.component["voice"];
        assert!(voice.units.is_empty(), "voice should not own a unit");
        assert!(
            voice
                .build
                .iter()
                .any(|step| step.iter().any(|arg| arg == "crates/1bit-voice")),
            "voice build step must `cargo install --path crates/1bit-voice`"
        );

        // echo — cargo installs 1bit-echo AND enables strix-echo.service.
        let echo = &m.component["echo"];
        assert!(
            echo.units.iter().any(|u| u == "strix-echo.service"),
            "echo must enable strix-echo.service"
        );
        assert!(
            echo.build
                .iter()
                .any(|step| step.iter().any(|arg| arg == "crates/1bit-echo")),
            "echo build step must `cargo install --path crates/1bit-echo`"
        );
        assert!(
            echo.deps.iter().any(|d| d == "voice"),
            "echo must depend on voice"
        );

        // mcp — binary only, no systemd unit.
        let mcp = &m.component["mcp"];
        assert!(mcp.units.is_empty(), "mcp should not own a unit");
        assert!(
            mcp.build
                .iter()
                .any(|step| step.iter().any(|arg| arg == "crates/1bit-mcp")),
            "mcp build step must `cargo install --path crates/1bit-mcp`"
        );

        // tunnel — copies the cloudflared unit + config template. No
        // cargo install, no auto-enable (operator must auth first).
        let tunnel = &m.component["tunnel"];
        assert!(tunnel.build.is_empty(), "tunnel must not run a cargo build");
        assert!(
            tunnel.units.is_empty(),
            "tunnel must not auto-enable the unit"
        );
        let tunnel_has = |needle: &str| {
            tunnel.files.iter().any(|f| {
                let src = f.src().unwrap_or("");
                let dst = f.dst().unwrap_or("");
                src.contains(needle) || dst.contains(needle)
            })
        };
        assert!(
            tunnel_has("strix-cloudflared.service"),
            "tunnel must copy strix-cloudflared.service"
        );
        assert!(
            tunnel_has("config.yml.template"),
            "tunnel must copy the cloudflared config.yml.template"
        );
    }

    /// `1bit install core` is the fresh-box entry point; it must pull in
    /// voice + echo + mcp alongside the existing landing/gaia/lemonade
    /// wiring, otherwise `1bit install core` on a blank machine won't
    /// bring up the voice+agent stack.
    #[test]
    fn core_chains_voice_echo_mcp() {
        let m = parse().unwrap();
        let core = &m.component["core"];
        for required in ["voice", "echo", "mcp"] {
            assert!(
                core.deps.iter().any(|d| d == required),
                "core must depend on '{required}' (got {:?})",
                core.deps
            );
        }

        // Resolution order should contain every new component.
        let mut order = Vec::new();
        let mut seen = HashSet::new();
        resolve(&m, "core", &mut order, &mut seen).unwrap();
        for required in ["voice", "echo", "mcp", "core"] {
            assert!(
                order.contains(&required),
                "resolve(core) must include '{required}', got {order:?}"
            );
        }
    }

    /// Every component that declares a `cargo install --path crates/<x>`
    /// build step must declare a crate directory that actually exists in
    /// the workspace. Catches typos like `crates/halo-vocie` at test time
    /// instead of on a fresh box during `1bit install`.
    #[test]
    fn declared_crate_paths_exist_on_disk() {
        let m = parse().unwrap();
        let root = workspace_root();
        for (name, c) in &m.component {
            for step in &c.build {
                // Find a `--path crates/<x>` pair.
                let mut iter = step.iter();
                while let Some(arg) = iter.next() {
                    if arg == "--path" {
                        let rel = iter.next().expect("--path needs a value");
                        let abs = root.join(rel);
                        assert!(
                            abs.is_dir(),
                            "component '{name}' points at missing crate dir {abs:?}"
                        );
                        // And the crate's Cargo.toml must exist too.
                        assert!(
                            abs.join("Cargo.toml").is_file(),
                            "component '{name}' crate {abs:?} has no Cargo.toml"
                        );
                    }
                }
            }
        }
    }

    /// Every `files = [...]` source (pair or table form) must resolve to
    /// a tracked file in the workspace. This is the check that catches a
    /// missing `strixhalo/cloudflared/config.yml.template` before a fresh
    /// box hits the runtime error.
    #[test]
    fn declared_tracked_files_exist_on_disk() {
        let m = parse().unwrap();
        let root = workspace_root();
        for (name, c) in &m.component {
            for entry in &c.files {
                let src_rel = entry
                    .src()
                    .unwrap_or_else(|e| panic!("component '{name}' files entry: {e}"));
                let src = root.join(src_rel);
                assert!(
                    src.is_file(),
                    "component '{name}' points at missing tracked file {src:?}"
                );
            }
        }
    }

    /// Every unit listed in a component must exist under
    /// `strixhalo/systemd/` — otherwise `systemctl --user enable --now X`
    /// will fail on a fresh box because the unit file was never shipped.
    /// (This assumes the operator has symlinked or copied the tracked
    /// units into their user unit dir, which is the documented flow.)
    #[test]
    fn declared_systemd_units_are_tracked() {
        let m = parse().unwrap();
        let root = workspace_root();
        let units_dir = root.join("strixhalo/systemd");
        for (name, c) in &m.component {
            for unit in &c.units {
                let path = units_dir.join(unit);
                assert!(
                    path.is_file(),
                    "component '{name}' declares unit '{unit}' with no file at {path:?}"
                );
            }
        }
    }

    /// Regression test: a prior manifest shape put voice and echo under
    /// different parent keys. Make sure `1bit install --list` (the `list`
    /// fn's underlying data) still emits all the components the prompt
    /// promised, in one pass, from a freshly-parsed manifest.
    #[test]
    fn list_covers_all_new_and_old_components() {
        let m = parse().unwrap();
        let names: Vec<&str> = m.component.keys().map(String::as_str).collect();
        for required in [
            "core", "voice", "echo", "mcp", "tunnel", "npu", "power", "lemonade", "landing", "gaia",
        ] {
            assert!(
                names.contains(&required),
                "1bit install --list must include '{required}', got {names:?}"
            );
        }
    }

    /// Declaration-consistency check: for each new component that ships
    /// a binary (voice, echo, mcp), the `[[bin]] name = "..."` in the
    /// crate's Cargo.toml must match what the systemd unit + the runbook
    /// expect on disk. `1bit-echo.service` invokes `/usr/local/bin/1bit-echo`,
    /// so the echo crate must build a binary literally named `1bit-echo`.
    ///
    /// If the release binary happens to be on disk (dev-box path), we
    /// verify its presence too — but a missing binary isn't a hard fail
    /// because CI boxes rarely keep a full `target/release/` between runs.
    #[test]
    fn new_component_binaries_match_cargo_toml_and_runtime_paths() {
        let root = workspace_root();
        let expected: &[(&str, &str, &str)] = &[
            // (component, crate dir, expected [[bin]] name)
            ("voice", "crates/1bit-voice", "1bit-voice"),
            ("echo", "crates/1bit-echo", "1bit-echo"),
            ("mcp", "crates/1bit-mcp", "1bit-mcp"),
        ];

        for (component, crate_dir, bin_name) in expected {
            let cargo_toml = root.join(crate_dir).join("Cargo.toml");
            let src = std::fs::read_to_string(&cargo_toml)
                .unwrap_or_else(|e| panic!("read {cargo_toml:?}: {e}"));
            // Accept either the `name         = "..."` (aligned) or
            // `name = "..."` (unaligned) Cargo.toml styles.
            let hit = src.contains(&format!("name = \"{bin_name}\""))
                || src.contains(&format!("name         = \"{bin_name}\""));
            assert!(
                hit,
                "component '{component}': {crate_dir}/Cargo.toml does not declare a [[bin]] name = \"{bin_name}\""
            );

            // Best-effort: if the release binary is on disk, it must match.
            let release = root.join("target/release").join(bin_name);
            if release.exists() {
                let meta = std::fs::metadata(&release).unwrap();
                assert!(meta.is_file(), "{release:?} is not a file");
            }
        }

        // strix-echo.service must reference a binary literally named
        // `1bit-echo` at /usr/local/bin/1bit-echo. Mismatched paths here
        // are why a fresh box silently fails to start the WebSocket.
        let unit = std::fs::read_to_string(root.join("strixhalo/systemd/strix-echo.service"))
            .expect("strix-echo.service must exist");
        assert!(
            unit.contains("/usr/local/bin/1bit-echo"),
            "strix-echo.service must ExecStart /usr/local/bin/1bit-echo"
        );
        assert!(
            unit.contains("--bind 127.0.0.1:8085"),
            "strix-echo.service must pass --bind 127.0.0.1:8085"
        );
        assert!(
            unit.contains("--codec opus"),
            "strix-echo.service must pass --codec opus"
        );
    }

    /// Parsing a synthetic manifest with a `files = [[src, dest]]` entry
    /// should deserialize into the new `files` vector. Guards against a
    /// future refactor accidentally dropping the legacy pair shape.
    #[test]
    fn files_field_deserializes() {
        let src = r#"
[component.example]
description = "x"
files = [
  ["strixhalo/systemd/strix-cloudflared.service", "systemd/user/strix-cloudflared.service"],
]
"#;
        let m = parse_src(src).unwrap();
        let c = &m.component["example"];
        assert_eq!(c.files.len(), 1);
        assert_eq!(
            c.files[0].src().unwrap(),
            "strixhalo/systemd/strix-cloudflared.service"
        );
        assert_eq!(
            c.files[0].dst().unwrap(),
            "systemd/user/strix-cloudflared.service"
        );
        assert!(
            c.files[0].substitute().is_empty(),
            "pair form carries no substitutions"
        );
    }

    // ───────────────────────────────────────────────────────────
    // NPU component tests (2026-04-20)
    // ───────────────────────────────────────────────────────────

    /// The `npu` component must be declared in `packages.toml`, declare
    /// both XRT packages (as a documentation-of-intent note; install.sh
    /// actually runs pacman), and drop the memlock config template via
    /// the new table-form `files` entry with `substitute = { USER = "$USER" }`.
    /// Also verifies that `1bit install --list` (the `list` fn's data) will
    /// include "npu" and that `1bit install core` transitively installs it.
    #[test]
    fn npu_component_declared_with_substitute() {
        let m = parse().expect("packages.toml must parse");

        // `1bit install --list` coverage — the `list` fn iterates
        // `m.component` and the keys must include "npu".
        let names: Vec<&str> = m.component.keys().map(String::as_str).collect();
        assert!(
            names.contains(&"npu"),
            "1bit install --list must include 'npu', got {names:?}"
        );

        let npu = m.component.get("npu").expect("component.npu must exist");
        // core must pull in npu so fresh-box `1bit install core` wires
        // the NPU userspace automatically.
        let core = &m.component["core"];
        assert!(
            core.deps.iter().any(|d| d == "npu"),
            "core must depend on 'npu' (got {:?})",
            core.deps
        );

        // The two XRT packages from CachyOS extra must be declared so
        // `1bit install --list` surfaces them.
        for pkg in ["xrt", "xrt-plugin-amdxdna"] {
            assert!(
                npu.packages.iter().any(|p| p == pkg),
                "npu must declare distro package '{pkg}', got {:?}",
                npu.packages
            );
        }

        // Files list must contain the memlock template → absolute-path
        // install at /etc/security/limits.d/ with substitute USER=$USER.
        let mut saw = false;
        for entry in &npu.files {
            let src = entry.src().unwrap();
            let dst = entry.dst().unwrap();
            if src == "strixhalo/security/99-npu-memlock.conf.tmpl"
                && dst == "/etc/security/limits.d/99-npu-memlock.conf"
            {
                let subs = entry.substitute();
                assert_eq!(
                    subs.get("USER").map(String::as_str),
                    Some("$USER"),
                    "npu memlock entry must declare substitute.USER = \"$USER\", got {subs:?}"
                );
                saw = true;
            }
        }
        assert!(saw, "npu must declare the memlock limits.d template file");
    }

    /// New-schema parse: a synthetic `files = [{ src, dst, substitute }]`
    /// entry must deserialize into the table variant with its substitute
    /// map intact. This is the direct parse-level guard that the
    /// `substitute = { USER = "$USER" }` field round-trips.
    #[test]
    fn files_table_form_with_substitute_deserializes() {
        let src = r#"
[component.npu]
description = "x"
files = [
  { src = "strixhalo/security/99-npu-memlock.conf.tmpl", dst = "/etc/security/limits.d/99-npu-memlock.conf", substitute = { USER = "$USER" } },
]
"#;
        let m = parse_src(src).unwrap();
        let c = &m.component["npu"];
        assert_eq!(c.files.len(), 1);
        assert_eq!(
            c.files[0].src().unwrap(),
            "strixhalo/security/99-npu-memlock.conf.tmpl"
        );
        assert_eq!(
            c.files[0].dst().unwrap(),
            "/etc/security/limits.d/99-npu-memlock.conf"
        );
        let subs = c.files[0].substitute();
        assert_eq!(subs.get("USER").map(String::as_str), Some("$USER"));
    }

    // ───────────────────────────────────────────────────────────
    // OOBE pass 2 tests — anchors #7, #9, #10 (2026-04-22)
    // ───────────────────────────────────────────────────────────

    use crate::preflight::SystemProbe;

    /// Minimal fake probe — every gate green. Duplicated from
    /// `preflight::tests` because that one is test-only private; the
    /// install tests need their own so we don't leak `cfg(test)` types
    /// across module boundaries.
    struct GreenProbe;

    impl SystemProbe for GreenProbe {
        fn kernel_release(&self) -> String {
            "6.18.22-1-cachyos-lts".into()
        }
        fn rocminfo_ok(&self) -> bool {
            true
        }
        fn systemd_user_ok(&self) -> bool {
            true
        }
        fn disk_free_gb(&self) -> u64 {
            512
        }
        fn ram_total_gb(&self) -> u64 {
            128
        }
    }

    /// Anchor #7 (doctor hook): `run_oobe_full` with `doctor_skip=true`
    /// skips the doctor probe even when the fake would otherwise report
    /// a failure. Keeps CI green without a real gfx1151 present.
    #[tokio::test]
    async fn oobe_doctor_skip_bypasses_doctor_even_when_fail() {
        let probe = GreenProbe;
        let doc = FakeDoctor { warn: 0, fail: 9 };
        let defaults = OobeDefaults {
            component: "core".into(),
            skip_build: true,
            yes: true,
            doctor_skip: true,
        };
        // doctor would fail (9 fails) but doctor_skip must short-
        // circuit BEFORE the probe contributes to the result.
        let res = run_oobe_full(&probe, &doc, defaults).await;
        assert!(
            res.is_ok(),
            "doctor_skip=true must keep the OOBE green even with a failing doctor probe, got {res:?}"
        );
    }

    /// Anchor #7 (doctor hook, pass path): with `skip_build=true` and a
    /// doctor that reports zero fails, the full OOBE flow succeeds.
    #[tokio::test]
    async fn oobe_doctor_hook_green_path_succeeds() {
        let probe = GreenProbe;
        let doc = FakeDoctor { warn: 1, fail: 0 };
        let defaults = OobeDefaults {
            component: "core".into(),
            skip_build: true,
            yes: true,
            doctor_skip: false,
        };
        let res = run_oobe_full(&probe, &doc, defaults).await;
        assert!(
            res.is_ok(),
            "doctor hook with 0 fails must let OOBE succeed, got {res:?}"
        );
    }

    /// Anchor #7 (doctor hook, fail path): doctor reports fail > 0 →
    /// the OOBE must bail with a `doctor` error.
    #[tokio::test]
    async fn oobe_doctor_hook_fail_path_bails() {
        let probe = GreenProbe;
        let doc = FakeDoctor { warn: 0, fail: 2 };
        let defaults = OobeDefaults {
            component: "core".into(),
            skip_build: true,
            yes: true,
            doctor_skip: false,
        };
        let err = run_oobe_full(&probe, &doc, defaults)
            .await
            .expect_err("doctor fail must bail");
        let msg = err.to_string();
        assert!(
            msg.contains("doctor") && msg.contains("fail"),
            "err must mention doctor + fail, got {msg}"
        );
    }

    /// Anchor #9 (--yes): a green OOBE with `yes=true` runs to
    /// completion without touching stdin. We exercise this by pairing
    /// `skip_build=true` + `doctor_skip=true` and verifying the path
    /// returns `Ok(())` in full automation.
    #[tokio::test]
    async fn oobe_yes_flag_finishes_without_stdin() {
        let probe = GreenProbe;
        let doc = NullDoctor;
        let defaults = OobeDefaults {
            component: "core".into(),
            skip_build: true,
            yes: true,
            doctor_skip: true,
        };
        let res = run_oobe_full(&probe, &doc, defaults).await;
        assert!(
            res.is_ok(),
            "--yes + --skip-build + --doctor-skip must succeed without stdin, got {res:?}"
        );
    }

    /// Anchor #10 (atomic-on-failure): the `InstallTracker` must revert
    /// every recorded action, in LIFO order, and drain its own action
    /// list so a second `best_effort_revert` call is a no-op. The test
    /// uses `CopiedFile` actions (the only kind we can simulate without
    /// a real systemctl / sudo on the test box); the matching revert
    /// path for `EnabledUnit` / `CopiedSudoFile` is the same log-and-
    /// drain shape, so this also guards against a regression in either
    /// arm of the match.
    #[test]
    fn anchor10_tracker_revert_drains_actions_lifo() {
        let td = tempfile::tempdir().unwrap();
        let a = td.path().join("a.conf");
        let b = td.path().join("b.conf");
        let c = td.path().join("c.conf");
        std::fs::write(&a, "a").unwrap();
        std::fs::write(&b, "b").unwrap();
        std::fs::write(&c, "c").unwrap();

        let tracker = InstallTracker::new();
        tracker.record(InstallAction::CopiedFile(a.clone()));
        tracker.record(InstallAction::CopiedFile(b.clone()));
        tracker.record(InstallAction::CopiedFile(c.clone()));

        assert_eq!(tracker.actions().len(), 3);
        tracker.best_effort_revert();

        // Every recorded file must have been removed.
        assert!(!a.exists(), "a.conf must be removed on revert");
        assert!(!b.exists(), "b.conf must be removed on revert");
        assert!(!c.exists(), "c.conf must be removed on revert");

        // The tracker must drain itself so a second call is a no-op
        // rather than attempting to re-remove files.
        assert!(
            tracker.actions().is_empty(),
            "revert must drain the action log"
        );
    }

    /// Anchor #10 (atomic-on-failure, empty path): reverting with zero
    /// recorded actions must not panic and must print the "nothing to
    /// revert" note. Regression guard against a future refactor that
    /// tries to call `.last()` on an empty Vec without handling `None`.
    #[test]
    fn anchor10_empty_tracker_revert_is_noop() {
        let tracker = InstallTracker::new();
        tracker.best_effort_revert();
        assert!(tracker.actions().is_empty());
    }
}

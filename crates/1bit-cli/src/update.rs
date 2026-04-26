//! `1bit update` — signed release feed + legacy git-rebuild path.
//!
//! Two modes coexist during the transition:
//!
//! * **New (default)** — `--check` / `--install` talk to the signed
//!   release feed at [`DEFAULT_FEED_URL`]. Compares `env!("CARGO_PKG_VERSION")`
//!   against `releases.json`, downloads the matching artifact, verifies
//!   sha256 + minisign signature against a compiled-in pubkey
//!   ([`HALO_RELEASE_PUBKEY_MINISIGN`]), then stops short of installing
//!   (atomic install comes in a later pass — see PR checklist).
//!
//! * **Legacy** — if the operator still passes `--no-build` or
//!   `--no-restart` (or the new `--legacy-rebuild` flag), we run the
//!   pre-release-feed path: git pull the workspace + rocm-cpp, rebuild
//!   release binaries, optionally delegate kernel rebuild to anvil,
//!   restart `strix-server.service`. This keeps `bin/1bit update` from
//!   breaking while we ramp the signed path.
//!
//! ### Exit codes (`--check`)
//! * `0` — already on the latest channel pin
//! * `1` — update available (stdout carries the hint)
//! * `2` — feed unreachable / unparseable
//!
//! ### Signature scheme
//! Minisign (EdDSA Ed25519, prehashed). Real release signing happens
//! **offline** with a key the project lead manages. The compiled-in
//! pubkey in this file is a **sample-only** keypair generated during
//! scaffolding (see `tests/fixtures/update/sample.pub`); swap it for
//! the real pubkey before cutting a production build. See the PR body
//! section "Sample keypair" for the matching private key.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// ─── release-feed constants ────────────────────────────────────────────

/// Canonical release-feed URL. Fronted by the `1bit-site` Worker; the
/// sample `releases.json` at `1bit-site/releases.json` is the format of
/// record. Override for offline tests / staging via `HALO_RELEASE_FEED`.
pub const DEFAULT_FEED_URL: &str = "https://1bit.systems/releases.json";

/// Compile-time minisign pubkey for release-artifact verification.
///
/// **SAMPLE ONLY.** This key was generated during scaffolding (Gap P1
/// #15) so the verify path can be wired end-to-end before a real
/// signing ceremony. The matching private key lives offline and is
/// referenced in the PR body; do NOT bake that priv key into any
/// service runtime.
pub const HALO_RELEASE_PUBKEY_MINISIGN: &str =
    "RWSLBfI8Kvx1y0hh48uFk6cyGEv9JVAWYcPOBG1cLF8Q0IWG5/nloFWQ";

/// HTTP timeout on the feed fetch. 5 s matches the OOBE preflight budget.
const FEED_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Hard ceiling on artifact download size (prevent a compromised feed
/// from streaming gigabytes into `/tmp`). 500 MB is well above the
/// current AppImage target (~40 MB for the CLI) with headroom for the
/// all-in-one Helm bundle.
const MAX_ARTIFACT_BYTES: u64 = 500 * 1024 * 1024;

// ─── feed schema ───────────────────────────────────────────────────────

/// Top-level `releases.json`.
///
/// Lives at a stable URL so we don't have to version-out the feed
/// itself when we rev the CLI. `latest` is the operator-facing pin
/// (usually == `channels.stable`); `channels` lets us ship preview /
/// nightly opt-ins without mutating `latest`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Feed {
    pub latest: String,
    #[serde(default)]
    pub channels: std::collections::BTreeMap<String, String>,
    pub releases: Vec<Release>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Release {
    pub version: String,
    #[serde(default)]
    pub date: String,
    pub artifacts: Vec<Artifact>,
    /// Lowest CLI version that can safely auto-update to this release.
    /// The current CLI must be `>=` this value; otherwise the update
    /// is flagged as requiring a manual install (breaking change).
    #[serde(default)]
    pub min_compatible: String,
    #[serde(default)]
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Artifact {
    /// `x86_64-linux-gnu`, `aarch64-linux-gnu`, …  Matches Rust target
    /// triples where possible so existing tooling can grep.
    pub platform: String,
    /// `appimage`, `tarball`, `deb`, `model`, `log`. Only `primary`
    /// install artifacts get sha256+minisign verified; `log` / `doc`
    /// attachments ride along as metadata.
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
    /// Lowercase hex-encoded SHA-256 over the artifact bytes. Required
    /// for install artifacts; omitted on auxiliary attachments like logs.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub sha256: String,
    /// Base64 minisign signature block (the full `.minisig` file
    /// contents — untrusted+trusted comments + sig lines). We inline
    /// it into the feed so verify is one HTTP GET away.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub minisign_sig: String,
    /// True when this artifact is the one the CLI installs by default.
    /// Defaults to true for backward compatibility with existing feeds
    /// that don't declare this.
    #[serde(default = "default_primary")]
    pub primary: bool,
}

fn default_primary() -> bool {
    true
}

// ─── version compare (semver-lite) ─────────────────────────────────────

/// Best-effort semver parser sufficient for `major.minor.patch[-rcN]`.
/// We deliberately avoid pulling in the `semver` crate for this: the
/// feed pins we publish are under our own control.
fn parse_version(v: &str) -> (u32, u32, u32, i32) {
    // `i32` pre-release tier: non-negative values are pre-releases
    // (lower => earlier), `i32::MAX` means "release, no pre".
    let (core, pre) = match v.split_once('-') {
        Some((c, p)) => (c, p),
        None => (v, ""),
    };
    let mut parts = core.split('.');
    let major = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let minor = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let patch = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let pre_rank = if pre.is_empty() {
        i32::MAX
    } else {
        // `rc1` < `rc2` < release. Strip non-digits to pull the number
        // out. Anything unparseable sorts before `rc0`.
        pre.chars()
            .filter(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse::<i32>()
            .unwrap_or(-1)
    };
    (major, minor, patch, pre_rank)
}

/// Strictly-greater version compare. Returns `true` iff `newer > older`.
pub fn is_newer(newer: &str, older: &str) -> bool {
    parse_version(newer) > parse_version(older)
}

// ─── feed fetch + parse ────────────────────────────────────────────────

fn feed_url() -> String {
    std::env::var("HALO_RELEASE_FEED").unwrap_or_else(|_| DEFAULT_FEED_URL.into())
}

async fn fetch_feed(url: &str) -> Result<Feed> {
    let client = reqwest::Client::builder()
        .timeout(FEED_TIMEOUT)
        .user_agent(concat!("1bit-cli/", env!("CARGO_PKG_VERSION")))
        .build()
        .context("build reqwest client")?;
    let resp = client
        .get(url)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;
    if !resp.status().is_success() {
        bail!("feed {} returned HTTP {}", url, resp.status());
    }
    let body = resp.bytes().await.context("read feed body")?;
    parse_feed(&body)
}

/// Pure-function parse so unit tests can hit it without HTTP.
pub fn parse_feed(bytes: &[u8]) -> Result<Feed> {
    serde_json::from_slice(bytes).context("parse releases.json")
}

/// Given a feed + current version, return the artifact matching the
/// current platform IF an update is available. Platform is fixed to
/// the compile-time target triple; cross-platform support lands when
/// we ship non-linux artifacts.
pub fn pick_update<'a>(feed: &'a Feed, current: &str) -> Option<(&'a Release, &'a Artifact)> {
    if !is_newer(&feed.latest, current) {
        return None;
    }
    let release = feed.releases.iter().find(|r| r.version == feed.latest)?;
    let platform = current_platform();
    let artifact = release.artifacts.iter().find(|a| a.platform == platform)?;
    Some((release, artifact))
}

/// Return the running binary's target triple in the same format the
/// feed uses. Keep in sync with `Artifact::platform` strings.
pub fn current_platform() -> &'static str {
    // Only linux-gnu x86_64 ships today; extend when we pack arm64.
    if cfg!(all(
        target_arch = "x86_64",
        target_os = "linux",
        target_env = "gnu"
    )) {
        "x86_64-linux-gnu"
    } else if cfg!(all(target_arch = "aarch64", target_os = "linux")) {
        "aarch64-linux-gnu"
    } else {
        "unknown"
    }
}

// ─── verification ──────────────────────────────────────────────────────

/// Recompute sha256 over a path, compare hex-lowercase against `expect`.
pub fn verify_sha256(path: &Path, expect: &str) -> Result<()> {
    let mut file =
        std::fs::File::open(path).with_context(|| format!("open {} for hash", path.display()))?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).context("hash read")?;
    let got = format!("{:x}", hasher.finalize());
    if !got.eq_ignore_ascii_case(expect.trim()) {
        bail!(
            "sha256 mismatch for {}: got {}, expected {}",
            path.display(),
            got,
            expect
        );
    }
    Ok(())
}

/// Verify a minisign `.minisig` block against `pubkey_b64` over the
/// contents of `path`. On failure returns an error that names the
/// reason without leaking bytes.
pub fn verify_minisign(path: &Path, sig_block: &str, pubkey_b64: &str) -> Result<()> {
    use minisign_verify::{PublicKey, Signature};

    let pk = PublicKey::from_base64(pubkey_b64).context("parse minisign pubkey")?;
    let sig = Signature::decode(sig_block).context("parse minisign signature block")?;
    let data = std::fs::read(path)
        .with_context(|| format!("read {} for signature verify", path.display()))?;
    // `allow_legacy = false` forces prehashed signatures (our signing
    // flow always uses `-H`). Refuse legacy raw signatures — they're
    // usable only for files small enough to fit in RAM twice and offer
    // no security benefit.
    pk.verify(&data, &sig, false)
        .map_err(|e| anyhow::anyhow!("minisign verify failed: {e}"))
}

// ─── subcommand entry points ───────────────────────────────────────────

/// Result of `--check`, surfaced so tests can assert without parsing stdout.
// `FeedError` is reserved for a future `--json` output (structured feed
// error); the current `--check` CLI exits 2 with stderr instead of
// constructing this variant. Keep it so the enum grows forward without an
// API-break when `--json` lands.
// TODO(gap-p2): box `Artifact` inside `Available` to calm
// `clippy::large_enum_variant`; touches every match site in this file.
#[allow(dead_code, clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckOutcome {
    UpToDate {
        current: String,
        latest: String,
    },
    Available {
        current: String,
        release: Release,
        artifact: Artifact,
    },
    FeedError(String),
}

impl CheckOutcome {
    pub fn exit_code(&self) -> i32 {
        match self {
            CheckOutcome::UpToDate { .. } => 0,
            CheckOutcome::Available { .. } => 1,
            CheckOutcome::FeedError(_) => 2,
        }
    }
}

/// Pure-core for tests: given a feed + current version, classify.
pub fn classify_check(feed: &Feed, current: &str) -> CheckOutcome {
    match pick_update(feed, current) {
        Some((rel, art)) => CheckOutcome::Available {
            current: current.into(),
            release: rel.clone(),
            artifact: art.clone(),
        },
        None => CheckOutcome::UpToDate {
            current: current.into(),
            latest: feed.latest.clone(),
        },
    }
}

async fn cmd_check() -> Result<i32> {
    let url = feed_url();
    let feed = match fetch_feed(&url).await {
        Ok(f) => f,
        Err(e) => {
            eprintln!("feed unreachable: {e:#}");
            return Ok(2);
        }
    };
    let outcome = classify_check(&feed, env!("CARGO_PKG_VERSION"));
    print_check(&outcome);
    Ok(outcome.exit_code())
}

fn print_check(outcome: &CheckOutcome) {
    match outcome {
        CheckOutcome::UpToDate { current, latest } => {
            println!("1bit {current} — up to date (feed latest: {latest})");
        }
        CheckOutcome::Available {
            current,
            release,
            artifact,
        } => {
            println!("update available: {current} → {}", release.version);
            println!("  released: {}", release.date);
            println!("  artifact: {}", artifact.url);
            println!("  sha256:   {}", artifact.sha256);
            if !release.notes.is_empty() {
                println!("  notes:    {}", release.notes);
            }
            println!("run `1bit update --install` to fetch + verify");
        }
        CheckOutcome::FeedError(msg) => {
            eprintln!("feed error: {msg}");
        }
    }
}

async fn cmd_install() -> Result<i32> {
    let url = feed_url();
    let feed = fetch_feed(&url).await.context("fetch feed")?;
    let current = env!("CARGO_PKG_VERSION");
    let (release, artifact) = match pick_update(&feed, current) {
        Some(pair) => pair,
        None => {
            println!("1bit {current} — already up to date, nothing to install");
            return Ok(0);
        }
    };

    println!("fetching {} ({})", artifact.url, release.version);
    let dest = PathBuf::from(format!(
        "/tmp/1bit-update-{}.{}",
        release.version,
        match artifact.kind.as_str() {
            "appimage" => "AppImage",
            other => other,
        }
    ));
    download_to(&artifact.url, &dest, MAX_ARTIFACT_BYTES).await?;

    println!("verifying sha256...");
    verify_sha256(&dest, &artifact.sha256)?;
    println!("verifying minisign signature...");
    verify_minisign(&dest, &artifact.minisign_sig, HALO_RELEASE_PUBKEY_MINISIGN)?;

    println!("\n✓ verified {}", dest.display());
    println!(
        "\nnext step (manual, real install lands in a later pass):\n  \
         mv {} ~/.local/bin/1bit && chmod +x ~/.local/bin/1bit",
        dest.display()
    );
    Ok(0)
}

async fn download_to(url: &str, dest: &Path, max_bytes: u64) -> Result<()> {
    use tokio::io::AsyncWriteExt;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .user_agent(concat!("1bit-cli/", env!("CARGO_PKG_VERSION")))
        .build()?;
    let mut resp = client
        .get(url)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;
    if !resp.status().is_success() {
        bail!("artifact {} returned HTTP {}", url, resp.status());
    }
    if let Some(len) = resp.content_length()
        && len > max_bytes
    {
        bail!("artifact too large: {len} bytes > cap {max_bytes}");
    }

    let mut file = tokio::fs::File::create(dest)
        .await
        .with_context(|| format!("create {}", dest.display()))?;
    let mut total: u64 = 0;
    while let Some(chunk) = resp.chunk().await.context("stream artifact")? {
        total = total.saturating_add(chunk.len() as u64);
        if total > max_bytes {
            // Nuke the partial file so a compromised feed can't leave
            // verified-looking bytes lying around.
            drop(file);
            let _ = std::fs::remove_file(dest);
            bail!("artifact exceeded cap {max_bytes} during stream");
        }
        file.write_all(&chunk).await.context("write artifact")?;
    }
    file.flush().await.ok();
    Ok(())
}

/// Entry point matched to `Cmd::Update { check, install, no_build, no_restart, legacy }`.
///
/// Selection priority:
/// 1. `--check` → feed probe, exit code is update-state
/// 2. `--install` → feed fetch + verify (no-op install)
/// 3. legacy (`--legacy-rebuild` OR `--no-build` OR `--no-restart`) → git-pull rebuild path
/// 4. default with no flags → `--check` behavior (safest default)
pub async fn run(
    check: bool,
    install: bool,
    no_build: bool,
    no_restart: bool,
    legacy: bool,
) -> Result<()> {
    if check && install {
        bail!("--check and --install are mutually exclusive");
    }
    let use_legacy = legacy || no_build || no_restart;
    if use_legacy && (check || install) {
        bail!(
            "--legacy-rebuild / --no-build / --no-restart are for the legacy git path; drop --check/--install"
        );
    }
    if use_legacy {
        return legacy_rebuild::run(no_build, no_restart).await;
    }
    if install {
        let code = cmd_install().await?;
        if code != 0 {
            std::process::exit(code);
        }
        return Ok(());
    }
    // --check is the safe default: no side effects, exits nonzero only
    // when something's actionable for the operator.
    let code = cmd_check().await?;
    if code != 0 {
        std::process::exit(code);
    }
    Ok(())
}

// ─── legacy git-rebuild path (pre-release-feed behaviour) ──────────────

mod legacy_rebuild {
    //! Preserved verbatim from the pre-Gap-P1-#15 implementation so
    //! `bin/1bit update` keeps working during the transition. Deprecate
    //! once the signed-install atomic swap lands and the build-box flow
    //! switches to the feed.

    use super::*;

    fn home() -> std::path::PathBuf {
        dirs::home_dir().unwrap_or_else(|| ".".into())
    }
    fn workspace_dir() -> std::path::PathBuf {
        std::env::var_os("HALO_WORKSPACE")
            .map(Into::into)
            .unwrap_or_else(|| home().join("repos/1bit-halo-workspace"))
    }
    fn rocm_cpp_dir() -> std::path::PathBuf {
        std::env::var_os("HALO_ROCM_CPP")
            .map(Into::into)
            .unwrap_or_else(|| home().join("repos/rocm-cpp"))
    }
    fn anvil_path() -> std::path::PathBuf {
        std::env::var_os("HALO_ANVIL")
            .map(Into::into)
            .unwrap_or_else(|| home().join("bin/1bit-anvil.sh"))
    }

    fn step(title: &str) {
        println!("\n── {title} ──");
    }

    fn run_in(dir: &Path, bin: &str, args: &[&str]) -> Result<()> {
        println!("  $ {bin} {}", args.join(" "));
        let s = Command::new(bin)
            .args(args)
            .current_dir(dir)
            .status()
            .with_context(|| format!("spawn {bin}"))?;
        if !s.success() {
            bail!("{bin} {} failed in {}", args.join(" "), dir.display());
        }
        Ok(())
    }

    pub async fn run(no_build: bool, no_restart: bool) -> Result<()> {
        let ws = workspace_dir();
        let rc = rocm_cpp_dir();
        let anvil = anvil_path();

        step("pull");
        run_in(&ws, "git", &["pull", "--ff-only"])?;
        if rc.exists() {
            run_in(&rc, "git", &["pull", "--ff-only"]).ok();
        }

        if !no_build {
            step("build rust workspace");
            run_in(&ws, "cargo", &["build", "--release", "--workspace"])?;
            run_in(
                &ws,
                "cargo",
                &["install", "--path", "crates/1bit-cli", "--force", "--quiet"],
            )?;
            run_in(
                &ws,
                "cargo",
                &[
                    "install",
                    "--path",
                    "crates/1bit-server",
                    "--force",
                    "--quiet",
                    "--features",
                    "real-backend",
                ],
            )?;

            if anvil.exists() {
                step("delegate rocm-cpp rebuild to anvil");
                run_in(&ws, anvil.to_str().unwrap_or("1bit-anvil.sh"), &[])?;
            } else {
                println!(
                    "  (anvil missing at {}, skip kernel rebuild)",
                    anvil.display()
                );
            }
        }

        if !no_restart {
            step("restart gen-2 server");
            run_in(
                &ws,
                "systemctl",
                &["--user", "restart", "strix-server.service"],
            )?;
        }

        println!("\n✓ update complete");
        Ok(())
    }
}

// ─── tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Sample keypair pubkey (matches `tests/fixtures/update/sample.pub`).
    /// Identical to `HALO_RELEASE_PUBKEY_MINISIGN`; duplicated here so
    /// a future pubkey bump doesn't silently break the fixture tests.
    const FIXTURE_PUBKEY: &str = "RWSLBfI8Kvx1y0hh48uFk6cyGEv9JVAWYcPOBG1cLF8Q0IWG5/nloFWQ";

    fn fixture_path(rel: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/update")
            .join(rel)
    }

    fn sample_feed_json() -> String {
        // Hand-rolled fixture that mirrors the real releases.json schema.
        // Keep in sync with `1bit-site/releases.json`. If that file
        // diverges from this fixture, `feed_parses_example_json` will
        // catch it.
        let site_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../1bit-site/releases.json");
        std::fs::read_to_string(&site_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", site_path.display()))
    }

    #[test]
    fn feed_parses_example_json() {
        let body = sample_feed_json();
        let feed = parse_feed(body.as_bytes()).expect("parse");
        assert!(!feed.latest.is_empty(), "latest pin must be set");
        assert!(
            feed.releases.iter().any(|r| r.version == feed.latest),
            "releases[] must include the version named in `latest`"
        );
        let rel = feed
            .releases
            .iter()
            .find(|r| r.version == feed.latest)
            .unwrap();
        assert!(
            !rel.artifacts.is_empty(),
            "pinned release must have artifacts"
        );
        assert!(
            feed.channels.contains_key("stable"),
            "stable channel pin is expected"
        );
    }

    #[test]
    fn version_compare_newer_triggers_update_hint() {
        assert!(is_newer("0.3.0", "0.2.9"));
        assert!(is_newer("0.3.1", "0.3.0"));
        assert!(is_newer("0.3.0", "0.3.0-rc1"));
        assert!(!is_newer("0.3.0", "0.3.0"));
        assert!(!is_newer("0.2.9", "0.3.0"));

        // Wire the version-compare into classify_check so we exercise
        // the full `--check` logic (minus the HTTP hop). Use a current
        // version that's older than ANY plausible feed.latest — 0.0.0
        // makes the test resilient when releases.json gets bumped or
        // the placeholder/sample fixture is removed (as in 2026-04-25
        // when latest flipped 0.3.0 -> real 0.1.0).
        let feed = parse_feed(sample_feed_json().as_bytes()).unwrap();
        let stale = "0.0.0";
        assert!(is_newer(&feed.latest, stale),
            "feed.latest {} should be newer than {}", feed.latest, stale);
        let outcome = classify_check(&feed, stale);
        match outcome {
            CheckOutcome::Available {
                current, release, ..
            } => {
                assert_eq!(current, stale);
                assert_eq!(release.version, feed.latest);
            }
            other => panic!("expected Available, got {other:?}"),
        }
    }

    #[test]
    fn check_up_to_date_exit_zero() {
        let feed = parse_feed(sample_feed_json().as_bytes()).unwrap();
        let outcome = classify_check(&feed, &feed.latest.clone());
        assert!(matches!(outcome, CheckOutcome::UpToDate { .. }));
        assert_eq!(outcome.exit_code(), 0);
    }

    #[test]
    fn sha256_mismatch_rejects_artifact() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"hello world").unwrap();
        // Real sha256 of "hello world" is b94d27b9... — pass a bogus one.
        let bogus = "0000000000000000000000000000000000000000000000000000000000000000";
        let err = verify_sha256(tmp.path(), bogus).unwrap_err();
        assert!(
            err.to_string().contains("sha256 mismatch"),
            "expected mismatch error, got: {err}"
        );

        // And the positive path for completeness: real hash passes.
        let real = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
        verify_sha256(tmp.path(), real).expect("real hash should verify");
    }

    #[test]
    fn minisign_signature_valid_accepts() {
        let artifact = fixture_path("sample_artifact.bin");
        let sig = std::fs::read_to_string(fixture_path("sample_artifact.bin.minisig"))
            .expect("read fixture sig");
        verify_minisign(&artifact, &sig, FIXTURE_PUBKEY).expect("sig should verify");
    }

    #[test]
    fn minisign_signature_invalid_rejects() {
        // Tamper with the signed payload: copy to tempfile, flip a byte,
        // verify against the real sig -> must fail.
        let orig = fixture_path("sample_artifact.bin");
        let sig = std::fs::read_to_string(fixture_path("sample_artifact.bin.minisig"))
            .expect("read fixture sig");
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mut bytes = std::fs::read(&orig).unwrap();
        bytes[0] ^= 0xff;
        std::fs::write(tmp.path(), &bytes).unwrap();

        let err = verify_minisign(tmp.path(), &sig, FIXTURE_PUBKEY)
            .expect_err("tampered payload must fail signature verify");
        assert!(
            err.to_string().to_lowercase().contains("verify"),
            "expected verify-failure error, got: {err}"
        );
    }

    #[test]
    fn compiled_in_pubkey_matches_fixture() {
        // Guard against a merge conflict or accidental pubkey bump:
        // the scaffold test fixtures were signed with the key whose
        // pub half is compiled into HALO_RELEASE_PUBKEY_MINISIGN. If
        // they diverge, swap the fixture AND the const at the same time.
        assert_eq!(HALO_RELEASE_PUBKEY_MINISIGN, FIXTURE_PUBKEY);
    }
}

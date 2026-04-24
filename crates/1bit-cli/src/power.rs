// `1bit power [profile]` — Ryzen APU power-profile CLI.
//
// Tier (b) shell-out to FlyGoat/ryzenadj (see docs/halo-power-design.md).
// We intentionally do NOT link a Rust ryzenadj crate; we just resolve the
// profile name → a fixed ryzenadj argv and exec it under `sudo`. That keeps
// us forwards-compatible with AMD family/stepping changes (ryzenadj catches
// those upstream and we inherit the fix for free via `pacman -Syu`).
//
// Rule A clean: Rust talking to a local C binary, no Python in sight.

use anyhow::{Context, Result, anyhow, bail};
use std::fmt;
use std::process::{Command, Stdio};
use std::str::FromStr;

/// Power profile — a named envelope of ryzenadj limits.
///
/// Values come from `docs/halo-power-design.md` (profile mapping table).
/// Units: stapm/fast/slow in milliwatts, tctl in °C.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// Max sustained decode tok/s. 65/80/75 W, 95 °C.
    Inference,
    /// Interactive, balanced, low fan. 45/65/55 W, 90 °C. Default after boot.
    Chat,
    /// Watchdog-triggered, quiet-closet mode. 20/35/25 W, 80 °C.
    Idle,
}

/// One row of the profile table — the numbers that flow into ryzenadj.
#[derive(Debug, Clone, Copy)]
pub struct Envelope {
    pub stapm_mw: u32,
    pub fast_mw: u32,
    pub slow_mw: u32,
    pub tctl_c: u32,
}

impl Profile {
    /// Resolve a profile to the concrete limits we hand to ryzenadj.
    pub fn envelope(self) -> Envelope {
        match self {
            Profile::Inference => Envelope {
                stapm_mw: 65_000,
                fast_mw: 80_000,
                slow_mw: 75_000,
                tctl_c: 95,
            },
            Profile::Chat => Envelope {
                stapm_mw: 45_000,
                fast_mw: 65_000,
                slow_mw: 55_000,
                tctl_c: 90,
            },
            Profile::Idle => Envelope {
                stapm_mw: 20_000,
                fast_mw: 35_000,
                slow_mw: 25_000,
                tctl_c: 80,
            },
        }
    }

    /// One-line description for `--list`.
    pub fn description(self) -> &'static str {
        match self {
            Profile::Inference => "Max sustained decode tok/s — all headroom to package.",
            Profile::Chat => "Interactive, balanced, low fan — default after boot.",
            Profile::Idle => "Watchdog-triggered quiet-closet mode — no active requests.",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Profile::Inference => "inference",
            Profile::Chat => "chat",
            Profile::Idle => "idle",
        }
    }

    /// Build the ryzenadj argv for this profile. Ordering per design doc:
    /// tctl-temp first (raises thermal ceiling before raising power),
    /// then slow, fast, stapm. Interrupted mid-set leaves "headroom raised
    /// but power not yet raised" — safe, just wasteful.
    pub fn ryzenadj_argv(self) -> Vec<String> {
        let e = self.envelope();
        vec![
            format!("--tctl-temp={}", e.tctl_c),
            format!("--slow-limit={}", e.slow_mw),
            format!("--fast-limit={}", e.fast_mw),
            format!("--stapm-limit={}", e.stapm_mw),
        ]
    }
}

impl fmt::Display for Profile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Profile {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "inference" | "decode" | "perf" | "performance" => Ok(Profile::Inference),
            "chat" | "balanced" | "default" => Ok(Profile::Chat),
            "idle" | "silent" | "save" => Ok(Profile::Idle),
            other => bail!("unknown power profile '{other}' (try `1bit power --list`)"),
        }
    }
}

/// Static list of profiles, stable ordering for display.
pub fn list_profiles() -> Vec<Profile> {
    vec![Profile::Inference, Profile::Chat, Profile::Idle]
}

/// Is `ryzenadj` on PATH? Shell-out guard; mirrors the pattern in `say.rs`.
fn which_ryzenadj() -> bool {
    Command::new("which")
        .arg("ryzenadj")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

const INSTALL_HINT: &str = "ryzenadj not found on PATH. Install on CachyOS with:\n\
     \n\
     \tsudo pacman -S ryzenadj\n\
     \n\
     (upstream: https://github.com/FlyGoat/RyzenAdj)";

/// Apply a profile by shelling out to `sudo ryzenadj <args>`. If `dry_run`
/// is true we print the argv and return `Ok(())` without executing.
pub fn apply(profile: Profile, dry_run: bool) -> Result<()> {
    let args = profile.ryzenadj_argv();

    if dry_run {
        println!("1bit power {profile} --dry-run");
        println!("    would exec: sudo ryzenadj {}", args.join(" "));
        return Ok(());
    }

    if !which_ryzenadj() {
        bail!("{INSTALL_HINT}");
    }

    println!("1bit power {profile}");
    println!("    $ sudo ryzenadj {}", args.join(" "));

    // Pass args to `sudo ryzenadj` directly — sudo owns the TTY for the
    // password prompt, we never touch the credential path ourselves.
    let status = Command::new("sudo")
        .arg("ryzenadj")
        .args(&args)
        .status()
        .with_context(|| "spawn sudo ryzenadj")?;

    if !status.success() {
        bail!("ryzenadj returned {status}");
    }
    println!("    ok — profile '{profile}' applied");
    Ok(())
}

/// Read current state via `ryzenadj --info`. Returns a one-line summary
/// (stapm / fast / slow / tctl) suitable for `1bit power` with no args.
pub fn current() -> Result<String> {
    if !which_ryzenadj() {
        return Err(anyhow!("{INSTALL_HINT}"));
    }
    let out = Command::new("ryzenadj")
        .arg("--info")
        .output()
        .with_context(|| "spawn ryzenadj --info")?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        bail!("ryzenadj --info failed: {stderr}");
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    Ok(summarize_info(&stdout))
}

/// Extract stapm / fast / slow / tctl values from `ryzenadj --info` output.
/// Upstream prints a two-column ASCII table; we just grep the labels so we
/// don't have to pin exact column widths across ryzenadj versions.
// Keep the branch-ladder shape intact: each `else if` maps to one ryzenadj
// label, and collapsing the `thm limit core` arm into the else-if makes the
// table-lookup intent harder to eyeball. Known-left, net-worse to collapse.
#[allow(clippy::collapsible_if)]
fn summarize_info(info: &str) -> String {
    let mut stapm = String::from("?");
    let mut fast = String::from("?");
    let mut slow = String::from("?");
    let mut tctl = String::from("?");

    for line in info.lines() {
        // Normalize both spaces and hyphens to a single form so we catch both
        // ryzenadj's dashed flag-style labels and its space-separated --info
        // headings.
        let norm = line.to_ascii_lowercase().replace('-', " ");
        if norm.contains("stapm value") { /* live reading — ignore */
        } else if norm.contains("stapm limit") {
            stapm = pick_value(line);
        } else if norm.contains("ppt limit fast") {
            fast = pick_value(line);
        } else if norm.contains("ppt limit slow") {
            slow = pick_value(line);
        } else if norm.contains("tctl temp") {
            tctl = pick_value(line);
        } else if norm.contains("thm limit core") {
            if tctl == "?" {
                tctl = pick_value(line);
            }
        }
    }
    format!("stapm={stapm} fast={fast} slow={slow} tctl={tctl}")
}

/// Pull the first whitespace-separated numeric-ish field after the `|`
/// column separator that ryzenadj --info uses. Falls back to the last
/// whitespace token on lines without a `|`.
fn pick_value(line: &str) -> String {
    if let Some(rest) = line.split('|').nth(1)
        && let Some(tok) = rest.split_whitespace().next()
    {
        return tok.trim().to_string();
    }
    line.split_whitespace().last().unwrap_or("?").to_string()
}

/// Pretty-print the static profile table for `1bit power --list`.
pub fn print_list() {
    println!("1bit power — available profiles:\n");
    for p in list_profiles() {
        let e = p.envelope();
        println!("  {:<10} {}", p.name(), p.description());
        println!(
            "  {:<10}   stapm={} W  fast={} W  slow={} W  tctl={} °C",
            "",
            e.stapm_mw / 1000,
            e.fast_mw / 1000,
            e.slow_mw / 1000,
            e.tctl_c
        );
    }
    println!("\nApply with: 1bit power <profile>   (add --dry-run to preview)");
}

/// Entry point wired from `main.rs`. Handles the three call-shapes
/// documented in the design doc:
///   1bit power                 -> print current state (or warn if missing)
///   1bit power --list          -> print_list()
///   1bit power <profile>       -> apply()
pub fn run(profile: Option<String>, dry_run: bool, list: bool) -> Result<()> {
    if list {
        print_list();
        return Ok(());
    }
    match profile {
        Some(name) => {
            let p: Profile = name.parse()?;
            apply(p, dry_run)
        }
        None => {
            if !which_ryzenadj() {
                // Safe-defaults: warn + no-op, don't break the CLI.
                eprintln!("warning: {INSTALL_HINT}");
                return Ok(());
            }
            match current() {
                Ok(summary) => {
                    println!("{summary}");
                    Ok(())
                }
                Err(e) => {
                    eprintln!("warning: {e}");
                    Ok(())
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_from_str_parses_all_three() {
        assert_eq!("inference".parse::<Profile>().unwrap(), Profile::Inference);
        assert_eq!("chat".parse::<Profile>().unwrap(), Profile::Chat);
        assert_eq!("idle".parse::<Profile>().unwrap(), Profile::Idle);
    }

    #[test]
    fn profile_from_str_is_case_insensitive_and_has_aliases() {
        assert_eq!("INFERENCE".parse::<Profile>().unwrap(), Profile::Inference);
        assert_eq!("Decode".parse::<Profile>().unwrap(), Profile::Inference);
        assert_eq!("balanced".parse::<Profile>().unwrap(), Profile::Chat);
        assert_eq!("silent".parse::<Profile>().unwrap(), Profile::Idle);
    }

    #[test]
    fn profile_from_str_rejects_bogus() {
        let err = "bogus".parse::<Profile>().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unknown power profile"), "got: {msg}");
    }

    #[test]
    fn ryzenadj_argv_matches_design_doc() {
        // Design doc says inference = stapm 65 W / fast 80 W / slow 75 W / tctl 95 °C.
        let argv = Profile::Inference.ryzenadj_argv();
        assert_eq!(
            argv,
            vec![
                "--tctl-temp=95".to_string(),
                "--slow-limit=75000".to_string(),
                "--fast-limit=80000".to_string(),
                "--stapm-limit=65000".to_string(),
            ]
        );

        // Idle = 20/35/25 W, tctl 80.
        let idle = Profile::Idle.ryzenadj_argv();
        assert!(idle.contains(&"--stapm-limit=20000".to_string()));
        assert!(idle.contains(&"--tctl-temp=80".to_string()));
    }

    #[test]
    fn list_profiles_is_stable_and_nonempty() {
        let list = list_profiles();
        assert_eq!(list.len(), 3);
        assert_eq!(list[0], Profile::Inference);
        assert_eq!(list[1], Profile::Chat);
        assert_eq!(list[2], Profile::Idle);
    }

    #[test]
    fn summarize_info_extracts_values() {
        // Sample shaped like ryzenadj --info: "label | value unit"
        let sample = "\
            CPU Family        |      Strix Halo\n\
            STAPM LIMIT       |     45.000\n\
            PPT LIMIT FAST    |     65.000\n\
            PPT LIMIT SLOW    |     55.000\n\
            THM LIMIT CORE    |     90.000\n";
        let s = summarize_info(sample);
        // Each field is parsed best-effort; ensure at least stapm is populated.
        assert!(s.contains("stapm=45.000"), "got: {s}");
        assert!(s.contains("fast=65.000"), "got: {s}");
    }
}

// rollback.rs — OOBE anchor #6 (one-command rollback).
//
// Fresh-box contract: if a user types `1bit rollback`, we either
//
//   - use the snapshot number they passed,
//   - or auto-pick the most recent snapper snapshot whose description
//     contains `.halo-preinstall`,
//
// confirm interactively (unless `--yes`), and then invoke
// `sudo snapper -c root rollback <N>`. If snapper isn't installed we
// bail out through the OOBE diagnostic error surface so the operator
// sees the same four-field block every other OOBE error uses.
//
// Everything that touches the host goes through a `Snapper` trait so
// the unit tests can drive a `FakeSnapper` that records calls without
// shelling out.

use anyhow::{Result, bail};
use std::io::{self, BufRead, Write};
use std::process::Command;

use crate::oobe_error::OobeError;

/// Label used by `install.sh` (and the OOBE flow) when it takes a
/// preinstall snapshot. The auto-pick path filters `snapper list` on
/// this substring; the `.` prefix mirrors how snapper conventionally
/// stores tags in the description field.
pub const HALO_PREINSTALL_LABEL: &str = ".halo-preinstall";

/// Injection seam. `RealSnapper` shells out to the `snapper` binary;
/// tests use `FakeSnapper` to drive the rollback flow without touching
/// the host. Methods are sync because snapper itself is sync + slow-enough
/// that adding tokio gives us no win.
pub trait Snapper {
    /// True iff `snapper` is on $PATH. Test probes return a bool field;
    /// the real probe runs `snapper --version` and checks exit status.
    fn available(&self) -> bool;

    /// Parsed rows of `snapper -c root list --output json` or a
    /// best-effort text parse. One entry per snapshot, most-recent last.
    /// Return `Ok(vec![])` rather than erroring when snapper itself is
    /// missing — the caller already asked `available()`.
    fn list(&self) -> Result<Vec<SnapperEntry>>;

    /// Execute `sudo snapper -c root rollback <N>`. Return `Ok(())` on
    /// success and bail with a readable error on failure.
    fn rollback(&self, number: u32) -> Result<()>;
}

/// A single row from `snapper list`. We deliberately keep only the
/// fields the OOBE needs: number + description (the label lives there).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapperEntry {
    pub number: u32,
    pub description: String,
}

/// Real-host snapper. Shells out to `snapper` + `sudo snapper rollback`.
pub struct RealSnapper;

impl Snapper for RealSnapper {
    fn available(&self) -> bool {
        Command::new("snapper")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    fn list(&self) -> Result<Vec<SnapperEntry>> {
        // Text parse is cheaper than pulling a JSON flag that older
        // snapper builds don't support. The table format is stable
        // across 0.10+ and parse-failures are skipped (not fatal).
        let out = match Command::new("snapper")
            .args(["-c", "root", "list"])
            .output()
        {
            Ok(o) => o,
            Err(_) => return Ok(Vec::new()),
        };
        if !out.status.success() {
            return Ok(Vec::new());
        }
        let stdout = String::from_utf8_lossy(&out.stdout);
        Ok(parse_snapper_list(&stdout))
    }

    fn rollback(&self, number: u32) -> Result<()> {
        let status = Command::new("sudo")
            .args(["snapper", "-c", "root", "rollback", &number.to_string()])
            .status()?;
        if !status.success() {
            bail!("snapper rollback {number} exited {status}");
        }
        Ok(())
    }
}

/// Parse the stdout of `snapper -c root list`. Best-effort: we accept
/// any row whose first `|`-separated cell is an integer, and pick up
/// the description from the cell containing `.halo-` markers or a
/// known-label substring. Rows we can't parse are skipped silently.
pub fn parse_snapper_list(stdout: &str) -> Vec<SnapperEntry> {
    let mut out = Vec::new();
    for line in stdout.lines() {
        // Skip header + separator lines.
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('-') {
            continue;
        }
        // Snapper's default table output has `|` column separators.
        let cols: Vec<&str> = line.split('|').map(|c| c.trim()).collect();
        if cols.len() < 2 {
            continue;
        }
        let number: u32 = match cols[0].parse() {
            Ok(n) => n,
            Err(_) => continue,
        };
        // Description is the last column in the default table shape;
        // we scan all columns and pick the one that actually has text,
        // defaulting to the last cell when no halo label is present.
        let desc = cols
            .iter()
            .find(|c| c.contains(HALO_PREINSTALL_LABEL))
            .copied()
            .unwrap_or_else(|| cols.last().copied().unwrap_or(""));
        out.push(SnapperEntry {
            number,
            description: desc.to_string(),
        });
    }
    out
}

/// Auto-pick the most recent snapshot whose description contains the
/// halo preinstall label. Returns `None` if no match was found — the
/// caller turns that into an `OobeError::no_rollback_candidate()`.
pub fn pick_latest_preinstall(entries: &[SnapperEntry]) -> Option<u32> {
    entries
        .iter()
        .filter(|e| e.description.contains(HALO_PREINSTALL_LABEL))
        .map(|e| e.number)
        .max()
}

/// Interactive confirmation prompt. Returns `Ok(true)` when the operator
/// types `y` / `Y`, `Ok(false)` otherwise. `--yes` (anchor #9) skips this
/// wholesale — we never block on stdin when `yes = true`.
fn confirm(prompt: &str, yes: bool) -> Result<bool> {
    if yes {
        return Ok(true);
    }
    print!("{prompt} [y/N]: ");
    io::stdout().flush().ok();
    let stdin = io::stdin();
    let mut line = String::new();
    stdin.lock().read_line(&mut line)?;
    let ans = line.trim().to_ascii_lowercase();
    Ok(ans == "y" || ans == "yes")
}

/// Pretty-print the rollback plan to stdout. Kept as a helper so the
/// fake-snapper test and the real caller render identically.
fn print_plan(entry: &SnapperEntry) {
    println!("rollback plan:");
    println!("  snapshot : {}", entry.number);
    println!("  label    : {}", entry.description);
    println!(
        "  command  : sudo snapper -c root rollback {}",
        entry.number
    );
}

/// Top-level `1bit rollback` entry point with a live snapper probe.
pub async fn run(snapshot: Option<u32>, yes: bool) -> Result<()> {
    run_with_snapper(&RealSnapper, snapshot, yes)
}

/// Testable core of `run`. Keeps the decision logic (gate / pick /
/// confirm / invoke) out of `run` so unit tests can wire a `FakeSnapper`
/// without going through tokio / stdin.
pub fn run_with_snapper(snapper: &dyn Snapper, snapshot: Option<u32>, yes: bool) -> Result<()> {
    // Gate 1: snapper must be installed.
    if !snapper.available() {
        let e = OobeError::snapper_absent();
        println!();
        println!("error: rollback");
        println!("{e}");
        bail!("snapper not installed");
    }

    // Gate 2: pick a snapshot (explicit or auto).
    let entries = snapper.list()?;
    let number = match snapshot {
        Some(n) => n,
        None => match pick_latest_preinstall(&entries) {
            Some(n) => n,
            None => {
                let e = OobeError::no_rollback_candidate();
                println!();
                println!("error: rollback");
                println!("{e}");
                bail!("no .halo-preinstall snapshot found");
            }
        },
    };

    // Render the plan the same way every OOBE surface does.
    let entry = entries
        .iter()
        .find(|e| e.number == number)
        .cloned()
        .unwrap_or(SnapperEntry {
            number,
            description: "(no description — explicit snapshot)".into(),
        });
    print_plan(&entry);

    // Gate 3: confirm (unless --yes).
    if !confirm(&format!("proceed with rollback to #{number}?"), yes)? {
        println!("rollback aborted by operator.");
        return Ok(());
    }

    // Gate 4: invoke. Failure bubbles up as anyhow, which the CLI turns
    // into a non-zero exit.
    snapper.rollback(number)?;
    println!("\n✓ rollback to #{number} submitted — reboot to apply.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// Records invocations so the tests can assert what the run flow
    /// did (or didn't) call.
    struct FakeSnapper {
        available: bool,
        entries: Vec<SnapperEntry>,
        rolled: RefCell<Vec<u32>>,
    }

    impl Snapper for FakeSnapper {
        fn available(&self) -> bool {
            self.available
        }
        fn list(&self) -> Result<Vec<SnapperEntry>> {
            Ok(self.entries.clone())
        }
        fn rollback(&self, n: u32) -> Result<()> {
            self.rolled.borrow_mut().push(n);
            Ok(())
        }
    }

    fn entry(n: u32, desc: &str) -> SnapperEntry {
        SnapperEntry {
            number: n,
            description: desc.into(),
        }
    }

    /// `1bit rollback` on a box without snapper must fail loudly with a
    /// diagnostic pointing at `sudo pacman -S snapper` rather than
    /// panicking. Anchor 6 explicitly requires "graceful fail if
    /// snapper not installed".
    #[test]
    fn snapper_absent_fails_with_diagnostic() {
        let s = FakeSnapper {
            available: false,
            entries: vec![],
            rolled: RefCell::new(vec![]),
        };
        let err = run_with_snapper(&s, None, true).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("snapper"), "err must mention snapper: {msg}");
        assert!(
            s.rolled.borrow().is_empty(),
            "must not invoke rollback when snapper is absent"
        );
    }

    /// Auto-pick (no explicit number, `--yes`) takes the highest-
    /// numbered `.halo-preinstall` row.
    #[test]
    fn auto_pick_takes_latest_preinstall() {
        let s = FakeSnapper {
            available: true,
            entries: vec![
                entry(3, "boot"),
                entry(6, "7.00 with claude .halo-preinstall"),
                entry(11, "random snapshot"),
                entry(14, "pre-install .halo-preinstall"),
                entry(17, "manual"),
            ],
            rolled: RefCell::new(vec![]),
        };
        run_with_snapper(&s, None, true).unwrap();
        assert_eq!(s.rolled.borrow().as_slice(), &[14]);
    }

    /// Auto-pick with zero matching rows surfaces
    /// `no_rollback_candidate` and never invokes rollback.
    #[test]
    fn auto_pick_with_no_candidate_bails() {
        let s = FakeSnapper {
            available: true,
            entries: vec![entry(3, "boot"), entry(6, "manual")],
            rolled: RefCell::new(vec![]),
        };
        let err = run_with_snapper(&s, None, true).unwrap_err();
        assert!(
            err.to_string().contains(".halo-preinstall")
                || err.to_string().contains("no .halo-preinstall"),
            "err must mention the missing label, got {err}"
        );
        assert!(s.rolled.borrow().is_empty());
    }

    /// Explicit snapshot number wins even when auto-pick would find a
    /// different one. The operator is always right.
    #[test]
    fn explicit_number_overrides_auto_pick() {
        let s = FakeSnapper {
            available: true,
            entries: vec![
                entry(6, "7.00 with claude .halo-preinstall"),
                entry(14, "pre-install .halo-preinstall"),
            ],
            rolled: RefCell::new(vec![]),
        };
        run_with_snapper(&s, Some(6), true).unwrap();
        assert_eq!(s.rolled.borrow().as_slice(), &[6]);
    }

    /// Parser must handle the plain `|`-separated table shape snapper
    /// uses by default. Header rows are skipped; data rows with a
    /// non-integer first column are skipped; halo-labelled rows land
    /// in the output.
    #[test]
    fn parse_snapper_list_picks_up_halo_rows() {
        let sample = "\
# | Type   | Pre # | Date                     | User | Cleanup  | Description           | Userdata
--+--------+-------+--------------------------+------+----------+-----------------------+---------
0 | single |       | 2026-04-18 10:00:00 UTC  | root |          | current               |
6 | single |       | 2026-04-18 10:01:00 UTC  | root | number   | 7.00 with claude .halo-preinstall |
14 | pre    |       | 2026-04-22 02:00:00 UTC | root |          | pre-install .halo-preinstall       |
";
        let parsed = parse_snapper_list(sample);
        // 0, 6, 14 should all parse; only 6 + 14 should carry the halo
        // label in the description cell.
        assert!(parsed.iter().any(|e| e.number == 6));
        assert!(parsed.iter().any(|e| e.number == 14));
        let halo: Vec<u32> = parsed
            .iter()
            .filter(|e| e.description.contains(HALO_PREINSTALL_LABEL))
            .map(|e| e.number)
            .collect();
        assert_eq!(halo, vec![6, 14]);
    }
}

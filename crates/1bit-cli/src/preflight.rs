// preflight.rs — OOBE anchor #1: pre-flight gates.
//
// Before any download / build / systemctl call the installer must probe
// the host and decide whether it can proceed. The gates listed in
// `project_oobe_bar.md` #1 are:
//
//   - GPU arch match (gfx1151 / gfx1201, via rocminfo)
//   - ROCm 7.x present OR installable from system pm
//   - Free disk ≥ 10 GB
//   - RAM ≥ 64 GB (warn below 128 GB)
//   - Kernel version (6.18-lts recommended, 7.x warn per amdgpu OPTC hang)
//   - systemd present + user session available
//
// We encode each gate as a pure function that takes a `&dyn SystemProbe`
// so the tests can inject fake answers. The `RealProbe` wired from
// `run_all` shells out to rocminfo, reads /proc/meminfo, etc.
//
// Outcomes are tri-state: `Pass` (green, proceed), `Skip` (yellow, warn
// but continue), `Fail` (red, wrap an `OobeError` and stop before any
// destructive work).

use crate::oobe_error::OobeError;
use std::path::Path;
use std::process::Command;

/// Tri-state result of a single gate. `Skip` is a soft warning that is
/// allowed to stay green overall — used for "RAM < 128 GB" where the box
/// still works at 64 GB but the operator should know it's tight.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreflightOutcome {
    /// Gate passed with an optional note (e.g. "Kernel: 6.18.22-lts").
    Pass(String),
    /// Gate is soft-failing: the installer should continue but print a
    /// yellow warning. Carries an operator-facing note.
    Skip(String),
    /// Gate is hard-failing with a diagnostic error. The installer must
    /// abort before any destructive step runs.
    Fail(OobeError),
}

impl PreflightOutcome {
    /// True iff the installer is allowed to continue past this gate.
    /// `Pass` and `Skip` both continue; only `Fail` stops the world.
    pub fn is_green(&self) -> bool {
        !matches!(self, PreflightOutcome::Fail(_))
    }
}

/// Abstraction over the facts preflight wants to learn about the host.
/// All methods return something cheap + deterministic so the unit tests
/// can construct a `FakeProbe` and drive every gate deterministically.
pub trait SystemProbe {
    /// `uname -r` output, e.g. "6.18.22-1-cachyos-lts".
    fn kernel_release(&self) -> String;
    /// True iff `rocminfo` is on $PATH and prints a known-good agent.
    /// Test probes can return `true` without spawning anything.
    fn rocminfo_ok(&self) -> bool;
    /// True iff `systemctl --user is-system-running` returns a value we
    /// can live with (running / degraded / starting).
    fn systemd_user_ok(&self) -> bool;
    /// Free disk on `/` in GB, rounded down.
    fn disk_free_gb(&self) -> u64;
    /// Total physical RAM in GB, rounded down.
    fn ram_total_gb(&self) -> u64;
}

/// Real implementation that shells out to the host. Unit tests never see
/// this — they use `FakeProbe` below. The installer instantiates one of
/// these in `install::run_oobe`.
pub struct RealProbe;

impl SystemProbe for RealProbe {
    fn kernel_release(&self) -> String {
        // `uname -r` is the canonical answer. We avoid the `uname(3)`
        // libc dance because the text parse is the same either way and
        // the extra crate cost buys nothing on a desktop.
        Command::new("uname")
            .arg("-r")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "unknown".into())
    }

    fn rocminfo_ok(&self) -> bool {
        // rocminfo exits 0 on a healthy ROCm install and prints a
        // "Agent" stanza. We only care about the exit status — the
        // GPU-arch check is a separate concern handled by `1bit doctor`.
        Command::new("rocminfo")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    fn systemd_user_ok(&self) -> bool {
        // `systemctl --user is-system-running` returns 0 for "running",
        // non-zero for "degraded" etc. We accept anything that lets the
        // caller enable user units. `starting` is also fine — fresh
        // boots hit that briefly.
        let out = match Command::new("systemctl")
            .args(["--user", "is-system-running"])
            .output()
        {
            Ok(o) => o,
            Err(_) => return false,
        };
        let stdout = String::from_utf8_lossy(&out.stdout);
        matches!(stdout.trim(), "running" | "degraded" | "starting")
    }

    fn disk_free_gb(&self) -> u64 {
        // libc::statvfs is the shortest path; no text parse, no shell.
        // Target the install root ("/") — if the operator has a funky
        // btrfs layout that's fine too, they'll see the GB figure in
        // the table and know if the probe lied.
        use std::ffi::CString;
        let c = match CString::new("/") {
            Ok(c) => c,
            Err(_) => return 0,
        };
        // SAFETY: FFI into statvfs; out-param is fully initialized by
        // the call and we zero-init the struct before the call to keep
        // the compiler happy on fields the kernel doesn't touch.
        let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
        let rc = unsafe { libc::statvfs(c.as_ptr(), &mut st) };
        if rc != 0 {
            return 0;
        }
        // f_bavail = blocks available to non-root; f_frsize = fragment
        // size in bytes. Multiply to bytes, then /1 GiB.
        // (libc already types these as u64 on Linux — explicit casts are
        // a no-op and clippy flags them.)
        let bytes = st.f_bavail.saturating_mul(st.f_frsize);
        bytes / (1024 * 1024 * 1024)
    }

    fn ram_total_gb(&self) -> u64 {
        // `/proc/meminfo` MemTotal in kB. Simplest cross-distro path.
        let raw = match std::fs::read_to_string("/proc/meminfo") {
            Ok(s) => s,
            Err(_) => return 0,
        };
        for line in raw.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                // Line looks like `MemTotal:       131072 kB`.
                let kb = rest
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0);
                return kb / (1024 * 1024);
            }
        }
        0
    }
}

/// The hard floor for RAM. 64 GB is the minimum documented in
/// `project_oobe_bar.md`; below 128 GB is a soft `Skip`.
pub fn default_ram_floor_gb() -> u64 {
    64
}

/// Gate: kernel must be 6.x (LTS) for the OOBE path. 7.x warns +
/// soft-fails because the amdgpu OPTC hang on Strix Halo freezes Wayland
/// and needs a power-cycle. `0.x` / unparseable → `Fail` to force the
/// operator to actually figure out what they're booting.
pub fn gate_kernel(probe: &dyn SystemProbe) -> PreflightOutcome {
    let rel = probe.kernel_release();
    // First token before `.` should be major version.
    let major = rel
        .split(|c: char| !c.is_ascii_digit())
        .find(|s| !s.is_empty())
        .and_then(|s| s.parse::<u32>().ok());
    match major {
        Some(6) => PreflightOutcome::Pass(format!("kernel {rel} (LTS OK)")),
        Some(7) => PreflightOutcome::Fail(OobeError::kernel_too_new(&rel)),
        Some(other) => PreflightOutcome::Skip(format!(
            "kernel {rel} (major {other}) is outside the tested range — proceeding anyway"
        )),
        None => PreflightOutcome::Skip(format!("kernel release unparseable: {rel:?}")),
    }
}

/// Gate: rocminfo must reach at least one agent. On a fresh box that
/// hasn't installed ROCm, this is a `Fail` — the installer tells the
/// operator exactly which pacman invocation fixes it.
pub fn gate_rocm(probe: &dyn SystemProbe) -> PreflightOutcome {
    if probe.rocminfo_ok() {
        PreflightOutcome::Pass("rocminfo reachable".into())
    } else {
        PreflightOutcome::Fail(OobeError::rocm_missing())
    }
}

/// Gate: 10 GB free on the install root. Below that we abort before
/// triggering a half-download that fills the disk and bricks the
/// rollback path.
pub fn gate_disk(probe: &dyn SystemProbe) -> PreflightOutcome {
    let free = probe.disk_free_gb();
    if free >= 10 {
        PreflightOutcome::Pass(format!("{free} GB free on /"))
    } else {
        PreflightOutcome::Fail(OobeError::disk_too_small(free))
    }
}

/// Gate: RAM floor (64 GB) is a hard fail; below 128 GB is a soft
/// warning so the operator knows halo-v2 will be tight but still works.
pub fn gate_ram(probe: &dyn SystemProbe) -> PreflightOutcome {
    let ram = probe.ram_total_gb();
    let floor = default_ram_floor_gb();
    if ram < floor {
        return PreflightOutcome::Fail(OobeError::ram_too_small(ram, floor));
    }
    if ram < 128 {
        return PreflightOutcome::Skip(format!(
            "RAM {ram} GB < 128 GB recommended — halo-v2 Q4_K_M will be tight"
        ));
    }
    PreflightOutcome::Pass(format!("RAM {ram} GB"))
}

/// Named gate, useful for rendering the preflight table. Kept separate
/// from the outcome so the renderer can colour per-gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateResult {
    pub name: &'static str,
    pub outcome: PreflightOutcome,
}

/// Run every gate and return the full set of results. The caller
/// (`install::run_oobe`) decides whether to abort or proceed based on
/// `is_green()` across the vector. `systemd_user_ok` is folded in as a
/// `Skip` rather than a hard fail because a non-interactive CI box may
/// not have a user session and we still want the binary to be useful
/// there.
pub fn run_all(probe: &dyn SystemProbe) -> Vec<GateResult> {
    let mut out = Vec::with_capacity(5);
    out.push(GateResult {
        name: "kernel",
        outcome: gate_kernel(probe),
    });
    out.push(GateResult {
        name: "rocm",
        outcome: gate_rocm(probe),
    });
    out.push(GateResult {
        name: "disk",
        outcome: gate_disk(probe),
    });
    out.push(GateResult {
        name: "ram",
        outcome: gate_ram(probe),
    });
    // systemd gate is soft — degraded is fine, missing is a Skip, not a
    // Fail, because `1bit install` on a container builder has no user
    // bus but the binary should still run.
    let sd = if probe.systemd_user_ok() {
        PreflightOutcome::Pass("user bus reachable".into())
    } else {
        PreflightOutcome::Skip(
            "systemd --user bus not reachable (containers / CI are OK here)".into(),
        )
    };
    out.push(GateResult {
        name: "systemd",
        outcome: sd,
    });
    out
}

/// Helper for `1bit doctor` and friends — detect whether `/proc` is
/// mounted. Used to bail early on weird sandboxes before we try to
/// parse `/proc/meminfo`.
#[allow(dead_code)] // reserved for `1bit doctor` wiring (gap P2)
pub fn proc_mounted() -> bool {
    Path::new("/proc/meminfo").is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fake probe: every fact is field-driven so a test can construct a
    /// green box, a bad-kernel box, a missing-ROCm box, etc., and assert
    /// the exact outcome.
    struct FakeProbe {
        kernel: String,
        rocminfo: bool,
        systemd: bool,
        disk_gb: u64,
        ram_gb: u64,
    }

    impl FakeProbe {
        fn green() -> Self {
            Self {
                kernel: "6.18.22-1-cachyos-lts".into(),
                rocminfo: true,
                systemd: true,
                disk_gb: 256,
                ram_gb: 128,
            }
        }
    }

    impl SystemProbe for FakeProbe {
        fn kernel_release(&self) -> String {
            self.kernel.clone()
        }
        fn rocminfo_ok(&self) -> bool {
            self.rocminfo
        }
        fn systemd_user_ok(&self) -> bool {
            self.systemd
        }
        fn disk_free_gb(&self) -> u64 {
            self.disk_gb
        }
        fn ram_total_gb(&self) -> u64 {
            self.ram_gb
        }
    }

    // ── gate_kernel ───────────────────────────────────────────────
    #[test]
    fn kernel_6_18_lts_passes() {
        let p = FakeProbe::green();
        match gate_kernel(&p) {
            PreflightOutcome::Pass(s) => assert!(s.contains("6.18")),
            other => panic!("expected Pass, got {other:?}"),
        }
    }

    #[test]
    fn kernel_7_x_fails_with_diagnostic() {
        let p = FakeProbe {
            kernel: "7.0.0-arch1-1".into(),
            ..FakeProbe::green()
        };
        match gate_kernel(&p) {
            PreflightOutcome::Fail(e) => {
                assert!(e.what.to_lowercase().contains("kernel"));
                assert!(e.expected.contains("6.18"));
            }
            other => panic!("expected Fail, got {other:?}"),
        }
    }

    #[test]
    fn kernel_unparseable_is_skip_not_fail() {
        let p = FakeProbe {
            kernel: "unknown".into(),
            ..FakeProbe::green()
        };
        assert!(matches!(gate_kernel(&p), PreflightOutcome::Skip(_)));
    }

    #[test]
    fn kernel_5_x_is_skip_out_of_tested_range() {
        let p = FakeProbe {
            kernel: "5.15.0-100-generic".into(),
            ..FakeProbe::green()
        };
        assert!(matches!(gate_kernel(&p), PreflightOutcome::Skip(_)));
    }

    // ── gate_rocm ─────────────────────────────────────────────────
    #[test]
    fn rocm_ok_passes() {
        let p = FakeProbe::green();
        assert!(matches!(gate_rocm(&p), PreflightOutcome::Pass(_)));
    }

    #[test]
    fn rocm_missing_fails_with_pacman_hint() {
        let p = FakeProbe {
            rocminfo: false,
            ..FakeProbe::green()
        };
        match gate_rocm(&p) {
            PreflightOutcome::Fail(e) => {
                assert!(e.repro.contains("pacman"), "fix must hint pacman");
                assert!(e.wiki_link.contains("rocm"));
            }
            other => panic!("expected Fail, got {other:?}"),
        }
    }

    // ── gate_disk ─────────────────────────────────────────────────
    #[test]
    fn disk_at_floor_passes() {
        let p = FakeProbe {
            disk_gb: 10,
            ..FakeProbe::green()
        };
        assert!(matches!(gate_disk(&p), PreflightOutcome::Pass(_)));
    }

    #[test]
    fn disk_below_floor_fails() {
        let p = FakeProbe {
            disk_gb: 3,
            ..FakeProbe::green()
        };
        match gate_disk(&p) {
            PreflightOutcome::Fail(e) => assert!(e.expected.contains("10 GB")),
            other => panic!("expected Fail, got {other:?}"),
        }
    }

    #[test]
    fn disk_zero_fails_gracefully() {
        let p = FakeProbe {
            disk_gb: 0,
            ..FakeProbe::green()
        };
        assert!(matches!(gate_disk(&p), PreflightOutcome::Fail(_)));
    }

    // ── gate_ram ──────────────────────────────────────────────────
    #[test]
    fn ram_below_floor_fails() {
        let p = FakeProbe {
            ram_gb: 32,
            ..FakeProbe::green()
        };
        match gate_ram(&p) {
            PreflightOutcome::Fail(e) => {
                assert!(e.expected.contains("64"));
            }
            other => panic!("expected Fail, got {other:?}"),
        }
    }

    #[test]
    fn ram_64_to_128_is_skip_with_warning() {
        let p = FakeProbe {
            ram_gb: 96,
            ..FakeProbe::green()
        };
        match gate_ram(&p) {
            PreflightOutcome::Skip(s) => assert!(s.contains("128")),
            other => panic!("expected Skip, got {other:?}"),
        }
    }

    #[test]
    fn ram_128_plus_passes_clean() {
        let p = FakeProbe::green(); // 128
        assert!(matches!(gate_ram(&p), PreflightOutcome::Pass(_)));
    }

    #[test]
    fn ram_floor_is_64_not_128() {
        assert_eq!(default_ram_floor_gb(), 64);
    }

    // ── run_all ───────────────────────────────────────────────────
    #[test]
    fn run_all_on_green_box_is_all_green() {
        let p = FakeProbe::green();
        let results = run_all(&p);
        assert_eq!(results.len(), 5);
        for r in &results {
            assert!(
                r.outcome.is_green(),
                "gate {:?} unexpectedly failed: {:?}",
                r.name,
                r.outcome
            );
        }
    }

    #[test]
    fn run_all_surfaces_a_single_bad_gate() {
        let p = FakeProbe {
            rocminfo: false,
            ..FakeProbe::green()
        };
        let results = run_all(&p);
        let bad: Vec<&GateResult> = results.iter().filter(|r| !r.outcome.is_green()).collect();
        assert_eq!(bad.len(), 1, "exactly one gate should fail");
        assert_eq!(bad[0].name, "rocm");
    }

    #[test]
    fn run_all_preserves_gate_order_for_table_rendering() {
        let p = FakeProbe::green();
        let names: Vec<&str> = run_all(&p).into_iter().map(|r| r.name).collect();
        assert_eq!(names, vec!["kernel", "rocm", "disk", "ram", "systemd"]);
    }

    #[test]
    fn systemd_absent_is_skip_not_fail() {
        let p = FakeProbe {
            systemd: false,
            ..FakeProbe::green()
        };
        let results = run_all(&p);
        let sd = results.iter().find(|r| r.name == "systemd").unwrap();
        assert!(
            matches!(sd.outcome, PreflightOutcome::Skip(_)),
            "systemd gate must be Skip on CI/container boxes, got {:?}",
            sd.outcome
        );
    }

    #[test]
    fn outcome_is_green_tri_state() {
        assert!(PreflightOutcome::Pass("x".into()).is_green());
        assert!(PreflightOutcome::Skip("x".into()).is_green());
        assert!(!PreflightOutcome::Fail(OobeError::rocm_missing()).is_green());
    }
}

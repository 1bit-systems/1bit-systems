// oobe_error.rs — user-facing error shape for OOBE (anchors #4 + #8).
//
// OOBE hard-bar element 4 says every error must have: what broke, likely
// cause / expected state, an exact repro or fix command, and a link to the
// wiki `#troubleshooting` section. We encode that shape as a struct so
// callers cannot forget a field and so the Display block is consistent
// across every gate and step.
//
// Anchor 8 (discoverability) piggy-backs on the same struct via the
// `next_step` field — every error must print a one-line "next step"
// pointer so the operator knows the next command to run without having
// to click through to the wiki. `what` answers *what broke*, `expected`
// answers *what we wanted*, `repro` is the exact fix command, `wiki_link`
// is the deep-link, and `next_step` is the shortest "now do this" pointer
// (example: "Run `1bit doctor` to verify the stack state.").
//
// Kept intentionally tiny: no `thiserror`, no `anyhow` wrapping — the
// whole point is the contract, and mixing error-chain noise into the
// user face defeats the purpose. `install::print_oobe_error` renders
// this into the terminal with the same shape every time.

use std::fmt;

/// A diagnostic, user-facing error for the OOBE path. Every field is
/// required; the whole point of the struct is that you cannot construct
/// one without the contract pieces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OobeError {
    /// One short sentence describing what failed. Not a stack trace.
    /// Example: "Kernel 7.0.x is not supported on Strix Halo yet."
    pub what: &'static str,

    /// What the installer expected / what the operator needs to have.
    /// Example: "Linux 6.18-lts (amdgpu OPTC hang fixed on LTS)."
    pub expected: String,

    /// Exact command or concrete action the operator should run.
    /// Example: "sudo limine-snapshot rollback 6 && reboot"
    pub repro: String,

    /// Deep-link into the GitHub wiki `#troubleshooting` section so the
    /// user can copy-paste and land on the right anchor.
    pub wiki_link: &'static str,

    /// Anchor 8 (discoverability): one-line "next step" pointer rendered
    /// alongside the error block. Operators read the error table top-to-
    /// bottom; the next_step is the pointer they see BEFORE they click
    /// the wiki link — a shortest-path command like "Run `1bit doctor` to
    /// probe the stack state.". Every constructor on this struct MUST
    /// populate it; the test suite enforces non-empty.
    pub next_step: &'static str,
}

impl OobeError {
    /// Kernel is too new (7.x); the amdgpu OPTC CRTC hang freezes Wayland
    /// on Strix Halo on anything newer than the 6.18-lts baseline.
    pub fn kernel_too_new(current: &str) -> Self {
        Self {
            what: "Kernel is too new for Strix Halo OOBE.",
            expected: format!("Linux 6.18-lts is the recommended baseline. Detected: {current}."),
            repro: "Boot snapper snapshot #6 or install linux-lts and reboot.".to_string(),
            wiki_link: "https://github.com/bong-water-water-bong/1bit-systems/wiki/Troubleshooting#kernel-too-new",
            next_step: "Next: `1bit rollback` to pick a tested snapshot, then reboot.",
        }
    }

    /// ROCm 7.x userspace missing. We don't attempt to install it from
    /// Rust — that's install.sh's job — but we tell the operator exactly
    /// how to.
    pub fn rocm_missing() -> Self {
        Self {
            what: "ROCm 7.x userspace not detected.",
            expected: "rocminfo reachable on $PATH with a gfx1151 or gfx1201 agent.".to_string(),
            repro: "sudo pacman -S rocm-hip-runtime rocminfo".to_string(),
            wiki_link: "https://github.com/bong-water-water-bong/1bit-systems/wiki/Troubleshooting#rocm-missing",
            next_step: "Next: install ROCm via install.sh or pacman, then re-run `1bit install --oobe`.",
        }
    }

    /// Disk free is below the 10 GB OOBE floor (binaries + weights).
    pub fn disk_too_small(free_gb: u64) -> Self {
        Self {
            what: "Free disk is below the 10 GB OOBE floor.",
            expected: format!("≥ 10 GB free on the install target. Detected: {free_gb} GB."),
            repro: "df -h / ; remove old caches under ~/.cargo/registry and ~/.halo/logs"
                .to_string(),
            wiki_link: "https://github.com/bong-water-water-bong/1bit-systems/wiki/Troubleshooting#disk-too-small",
            next_step: "Next: free up space, then re-run `1bit install --oobe`.",
        }
    }

    /// RAM is below the 64 GB OOBE floor. Warn below 128 GB is handled
    /// separately in `preflight` as a `Skip` outcome, not an error.
    pub fn ram_too_small(have_gb: u64, floor_gb: u64) -> Self {
        Self {
            what: "RAM is below the OOBE minimum.",
            expected: format!("≥ {floor_gb} GB RAM for halo-v2 at Q4_K_M. Detected: {have_gb} GB."),
            repro: "Close other tenants or choose a smaller default model (Qwen3-4B Q4)."
                .to_string(),
            wiki_link: "https://github.com/bong-water-water-bong/1bit-systems/wiki/Troubleshooting#ram-too-small",
            next_step: "Next: pick a smaller model with `1bit install --oobe core` on a Q4 profile.",
        }
    }

    /// `1bit doctor` ran at the tail of `--oobe` and reported one or more
    /// `Fail` rows. We don't swallow the exit code — we re-emit the
    /// doctor output URL and tell the operator the ONE command that
    /// reprints the full table.
    pub fn doctor_failed(fail_count: u32) -> Self {
        Self {
            what: "`1bit doctor` reported one or more failing probes after install.",
            expected: format!(
                "All `1bit doctor` probes green (or WARN). Detected: {fail_count} FAIL."
            ),
            repro: "1bit doctor".to_string(),
            wiki_link: "https://github.com/bong-water-water-bong/1bit-systems/wiki/Troubleshooting#doctor-failed",
            next_step: "Next: run `1bit doctor` to see which rows are red and follow each row's fix.",
        }
    }

    /// Snapper is required for `1bit rollback`. If the CLI can't find
    /// snapper on $PATH we bail early — the rollback feature only makes
    /// sense on btrfs + snapper hosts (CachyOS default).
    pub fn snapper_absent() -> Self {
        Self {
            what: "`snapper` is not installed or not on $PATH.",
            expected: "snapper ≥ 0.10 available on $PATH (btrfs + snapper on CachyOS).".to_string(),
            repro: "sudo pacman -S snapper && sudo snapper -c root create-config /".to_string(),
            wiki_link: "https://github.com/bong-water-water-bong/1bit-systems/wiki/Troubleshooting#snapper-absent",
            next_step: "Next: install snapper, or pass a manual snapshot number via `1bit rollback <N>`.",
        }
    }

    /// No rollback candidate was found. Auto-pick scans for the
    /// `.halo-preinstall` label and returns this if the list is empty.
    pub fn no_rollback_candidate() -> Self {
        Self {
            what: "No `.halo-preinstall` snapper snapshot found.",
            expected: "At least one snapshot labelled `.halo-preinstall` in `snapper list`."
                .to_string(),
            repro: "sudo snapper -c root list | grep .halo-preinstall".to_string(),
            wiki_link: "https://github.com/bong-water-water-bong/1bit-systems/wiki/Troubleshooting#no-rollback-candidate",
            next_step: "Next: pass a snapshot number explicitly, e.g. `1bit rollback 6`.",
        }
    }

    /// Anchor 10: a step inside `run_install` failed. We wrap the step
    /// name + the best-effort recovery note so the user knows what got
    /// rolled back and what (if anything) was left behind.
    pub fn install_step_failed(step: &'static str) -> Self {
        Self {
            what: "An install step failed; best-effort rollback was attempted.",
            expected: format!("Clean completion of step `{step}`."),
            repro: "Re-run `1bit install --oobe` after addressing the cause printed above."
                .to_string(),
            wiki_link: "https://github.com/bong-water-water-bong/1bit-systems/wiki/Troubleshooting#install-step-failed",
            next_step: "Next: read the `left state:` line above, then re-run `1bit install --oobe`.",
        }
    }
}

impl fmt::Display for OobeError {
    /// Fixed five-line block (four contract lines + the anchor-8 next
    /// step pointer). The preflight renderer relies on this shape so a
    /// human eye can scan `what → expected → fix → wiki → next` in any
    /// error the OOBE emits.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "  what     : {}", self.what)?;
        writeln!(f, "  expected : {}", self.expected)?;
        writeln!(f, "  fix      : {}", self.repro)?;
        writeln!(f, "  wiki     : {}", self.wiki_link)?;
        write!(f, "  next     : {}", self.next_step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every constructor must populate all five fields — the whole
    /// contract is that the OOBE error has `what / expected / fix /
    /// wiki / next` and the Display block therefore never loses a line.
    #[test]
    fn constructors_fill_all_five_fields() {
        let errs = vec![
            OobeError::kernel_too_new("7.1.0"),
            OobeError::rocm_missing(),
            OobeError::disk_too_small(3),
            OobeError::ram_too_small(32, 64),
            OobeError::doctor_failed(2),
            OobeError::snapper_absent(),
            OobeError::no_rollback_candidate(),
            OobeError::install_step_failed("cargo build"),
        ];
        for e in &errs {
            assert!(!e.what.is_empty(), "what must be populated");
            assert!(!e.expected.is_empty(), "expected must be populated");
            assert!(!e.repro.is_empty(), "repro must be populated");
            assert!(
                e.wiki_link.starts_with("https://"),
                "wiki_link must be an https URL, got {:?}",
                e.wiki_link
            );
            assert!(
                e.wiki_link.contains("#"),
                "wiki_link must deep-link to a #section anchor, got {:?}",
                e.wiki_link
            );
            assert!(
                !e.next_step.is_empty(),
                "next_step (anchor 8) must be populated, got {:?}",
                e.next_step
            );
            assert!(
                e.next_step.starts_with("Next:"),
                "next_step must start with `Next:` for a consistent face, got {:?}",
                e.next_step
            );
        }
    }

    /// The Display block is the user-facing surface; it must emit all
    /// five fields in the documented order. Any future refactor that
    /// drops a line or reorders them breaks the OOBE contract.
    #[test]
    fn display_block_emits_five_lines_in_order() {
        let e = OobeError::kernel_too_new("7.1.0");
        let out = format!("{e}");
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 5, "must be exactly 5 lines, got: {out:?}");
        assert!(lines[0].trim_start().starts_with("what"), "line 0 = what");
        assert!(
            lines[1].trim_start().starts_with("expected"),
            "line 1 = expected"
        );
        assert!(lines[2].trim_start().starts_with("fix"), "line 2 = fix");
        assert!(lines[3].trim_start().starts_with("wiki"), "line 3 = wiki");
        assert!(lines[4].trim_start().starts_with("next"), "line 4 = next");
    }
}

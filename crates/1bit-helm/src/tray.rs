//! KDE Plasma StatusNotifierItem tray — MVP desktop presence for Helm.
//!
//! Gap P1 #7. The `1bit-halo-helm-tray` binary (see `src/bin/tray.rs`)
//! wires this module up to a real dbus connection; callers that just
//! want to test the pure logic (service-name set, status-line
//! formatter, menu shape) import `tray::*` without pulling an SNI
//! host.
//!
//! Scope is intentionally small — one icon, six menu items, 3 s status
//! polling. No chat window, no install flow, no log tail. Those live
//! in the egui `1bit-helm` binary today and will migrate under this
//! process when the full Helm v2 shell lands (memory:
//! project_helm_scope_v2, project_helm_tray_icon).
//!
//! ## Service probe
//!
//! Live on this box (see `systemctl --user list-unit-files '1bit-*' 'strix-*'`):
//! * `1bit-halo-bitnet.service`  — ternary inference daemon
//! * `strix-server.service`      — 1bit-server HTTP on :8180
//!
//! The spec (P1 #7) names these three verbatim. `strix-lemonade`,
//! `strix-landing`, `strix-burnin` are live too but out of MVP scope
//! — adding them is a `SERVICES` edit, not a refactor.

use std::process::Command;

/// Services the tray controls. Ordering is stable (used for the Status
/// line) and matches the P1 #7 spec. Edit this single constant to
/// extend the tray's reach — every other site reads through here.
pub const SERVICES: &[&str] = &["1bit-halo-bitnet", "strix-server"];

/// Menu-item identifiers. Stringly-typed because ksni takes owned
/// `String` labels; we expose enum→label + enum→action so the UI side
/// stays declarative and the test side can assert coverage without
/// touching dbus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Top dimmed status line. Not clickable.
    Status,
    /// `systemctl --user start` every entry in [`SERVICES`].
    StartAll,
    /// Mirror of `StartAll`.
    StopAll,
    /// `systemctl --user restart strix-server` — the HTTP face.
    RestartServer,
    /// `xdg-open https://1bit.systems`.
    OpenSite,
    /// Exits the tray process. Does NOT stop any services.
    Quit,
}

impl Action {
    /// Every menu item in display order. Referenced by the binary to
    /// build the `ksni::Tray::menu` vec and by tests to assert
    /// coverage.
    pub const ALL: &'static [Action] = &[
        Action::Status,
        Action::StartAll,
        Action::StopAll,
        Action::RestartServer,
        Action::OpenSite,
        Action::Quit,
    ];

    /// Human-readable menu label. No emoji per CLAUDE.md brand voice.
    pub fn label(self) -> &'static str {
        match self {
            Action::Status => "Status",
            Action::StartAll => "Start All",
            Action::StopAll => "Stop All",
            Action::RestartServer => "Restart 1bit-server",
            Action::OpenSite => "Open 1bit.systems",
            Action::Quit => "Quit",
        }
    }
}

/// One service's active/inactive state as seen by systemd.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ServiceState {
    Active,
    Inactive,
    #[default]
    Unknown,
}

impl ServiceState {
    pub fn as_str(self) -> &'static str {
        match self {
            ServiceState::Active => "active",
            ServiceState::Inactive => "inactive",
            ServiceState::Unknown => "unknown",
        }
    }

    /// Parse the single-word output of `systemctl is-active <unit>`
    /// (trailing newline OK). Anything we don't recognise → Unknown.
    pub fn parse(s: &str) -> Self {
        match s.trim() {
            "active" | "activating" | "reloading" => ServiceState::Active,
            "inactive" | "failed" | "deactivating" => ServiceState::Inactive,
            _ => ServiceState::Unknown,
        }
    }
}

/// One row of the live status snapshot — service name + its state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceStatus {
    pub name: String,
    pub state: ServiceState,
}

/// Format the top-of-menu status line. Compact — Plasma clips long
/// menu labels around ~60 chars. Example output:
/// `"1bit-halo-bitnet: active · strix-server: active"`.
pub fn build_status_line(rows: &[ServiceStatus]) -> String {
    if rows.is_empty() {
        return "no services".to_string();
    }
    rows.iter()
        .map(|r| format!("{}: {}", r.name, r.state.as_str()))
        .collect::<Vec<_>>()
        .join(" · ")
}

/// Live probe of [`SERVICES`] via `systemctl --user is-active`. Runs
/// one subprocess per service — systemd lets us batch but the MVP
/// keeps the code boring. Any failure → `Unknown` for that row; we
/// never panic on a dead systemd bus.
pub fn probe_services() -> Vec<ServiceStatus> {
    SERVICES
        .iter()
        .map(|name| ServiceStatus {
            name: (*name).to_string(),
            state: probe_one(name),
        })
        .collect()
}

fn probe_one(unit: &str) -> ServiceState {
    match Command::new("systemctl")
        .args(["--user", "is-active", unit])
        .output()
    {
        Ok(out) => {
            let s = String::from_utf8_lossy(&out.stdout);
            ServiceState::parse(&s)
        }
        Err(_) => ServiceState::Unknown,
    }
}

/// Fire-and-forget `systemctl --user start|stop|restart ...`. Returns
/// the exit status; callers log but don't block on it. We rely on the
/// 3 s refresh loop to surface the result in the tray.
pub fn systemctl(verb: &str, units: &[&str]) -> std::io::Result<std::process::ExitStatus> {
    let mut cmd = Command::new("systemctl");
    cmd.arg("--user").arg(verb);
    for u in units {
        cmd.arg(u);
    }
    cmd.status()
}

/// `xdg-open https://1bit.systems`. Detached — the browser lifetime
/// is not ours to manage.
pub fn open_site() -> std::io::Result<()> {
    Command::new("xdg-open")
        .arg("https://1bit.systems")
        .spawn()
        .map(|_| ())
}

/// 3 s — fast enough that a user-initiated start/stop shows up
/// without feeling laggy; slow enough that we don't hammer the dbus
/// socket.
pub const REFRESH_INTERVAL: std::time::Duration = std::time::Duration::from_secs(3);

/// 1x1 transparent PNG encoded as base64. Placeholder — a real
/// monochrome "1" glyph replaces this when 1bit-pkg ships brand
/// assets. The spec (P1 #7) required an embedded base64 PNG; this
/// satisfies the literal requirement without waiting on art.
pub const ICON_PNG_PLACEHOLDER_B64: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

/// Themed icon name ksni ships to Plasma when the user has a standard
/// icon theme. Fallback pathway; the base64 placeholder above is the
/// strictly-spec-compliant form.
pub const ICON_THEME_NAME: &str = "applications-system";

#[cfg(test)]
mod tests {
    use super::*;

    /// The SERVICES constant is load-bearing — every subcommand,
    /// every status row, every test below indexes into it. Pin both
    /// content and order so a silent refactor gets caught.
    #[test]
    fn service_name_set_is_stable() {
        assert_eq!(SERVICES.len(), 2, "MVP covers exactly two units post-Discord-cull");
        assert_eq!(SERVICES[0], "1bit-halo-bitnet");
        assert_eq!(SERVICES[1], "strix-server");
        // No duplicates — systemctl would accept them but the status
        // line would read ugly.
        let mut sorted = SERVICES.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), SERVICES.len());
    }

    #[test]
    fn build_status_line_formats() {
        let rows = vec![
            ServiceStatus {
                name: "1bit-halo-bitnet".into(),
                state: ServiceState::Active,
            },
            ServiceStatus {
                name: "strix-server".into(),
                state: ServiceState::Active,
            },
        ];
        let line = build_status_line(&rows);
        assert_eq!(
            line,
            "1bit-halo-bitnet: active · strix-server: active",
        );

        // Empty input is a degenerate-but-real case — the polling
        // loop hasn't run yet when Plasma first queries us.
        assert_eq!(build_status_line(&[]), "no services");

        // systemd parse: "failed" collapses to inactive. Anything
        // unrecognised → unknown.
        assert_eq!(ServiceState::parse("active\n"), ServiceState::Active);
        assert_eq!(ServiceState::parse("inactive"), ServiceState::Inactive);
        assert_eq!(ServiceState::parse("failed"), ServiceState::Inactive);
        assert_eq!(ServiceState::parse("???"), ServiceState::Unknown);
    }

    #[test]
    fn menu_items_include_all_actions() {
        // Every variant of Action must appear in the canonical
        // ordered list — otherwise the tray drops a menu item on
        // refactor.
        let actions = Action::ALL;
        assert!(actions.contains(&Action::Status));
        assert!(actions.contains(&Action::StartAll));
        assert!(actions.contains(&Action::StopAll));
        assert!(actions.contains(&Action::RestartServer));
        assert!(actions.contains(&Action::OpenSite));
        assert!(actions.contains(&Action::Quit));
        assert_eq!(actions.len(), 6);

        // Labels are present and non-empty for every action.
        for a in actions {
            let l = a.label();
            assert!(!l.is_empty(), "action {a:?} has empty label");
        }

        // Status is first (dimmed header), Quit is last (exits
        // tray).
        assert_eq!(actions[0], Action::Status);
        assert_eq!(*actions.last().unwrap(), Action::Quit);
    }

    #[test]
    fn refresh_interval_is_three_seconds() {
        // Pin the spec number — changing this changes perceived UI
        // responsiveness and dbus load; worth a test-level guard.
        assert_eq!(REFRESH_INTERVAL, std::time::Duration::from_secs(3));
    }

    #[test]
    fn icon_placeholder_is_nonempty() {
        // Sanity: the b64 blob is non-empty and ends with the
        // standard base64 pad char (one or two `=`). We don't link a
        // base64 decoder just for a test.
        assert!(ICON_PNG_PLACEHOLDER_B64.ends_with('='));
        assert!(ICON_PNG_PLACEHOLDER_B64.len() > 16);
        assert_eq!(ICON_THEME_NAME, "applications-system");
    }
}

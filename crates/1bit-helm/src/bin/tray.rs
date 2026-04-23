//! `1bit-halo-helm-tray` — MVP Plasma tray binary.
//!
//! Gap P1 #7. Spawns a KDE StatusNotifierItem via the `ksni` crate on a
//! `tokio` current-thread runtime, polls `systemctl --user is-active`
//! every 3 s, and surfaces six menu items: Status (dimmed), Start All,
//! Stop All, Restart 1bit-server, Open 1bit.systems, Quit. See
//! `onebit_helm::tray` for the pure-logic side.
//!
//! Not a replacement for `1bit-helm` (the egui window) — they run as
//! separate processes. Intended deployment is one `systemctl --user`
//! unit (`1bit-halo-helm-tray.service`, not shipped in this MVP) that
//! auto-starts at Plasma login.
//!
//! Exit: clicking "Quit" drops the `handle` future and returns from
//! `main`. It does NOT stop any of the tracked services — that's what
//! "Stop All" is for.

use ksni::TrayMethods;
use onebit_helm::tray::{
    Action, ICON_THEME_NAME, REFRESH_INTERVAL, SERVICES, ServiceStatus, build_status_line,
    open_site, probe_services, systemctl,
};
use std::sync::atomic::{AtomicBool, Ordering};

/// Tray model. Held inside the ksni handle on its own task.
struct HelmTray {
    /// Latest snapshot from the 3 s polling loop. Rendered verbatim
    /// into the top-dimmed menu line.
    status: Vec<ServiceStatus>,
    /// Set by the Quit menu item; checked by the main task to exit.
    quit: std::sync::Arc<AtomicBool>,
}

impl ksni::Tray for HelmTray {
    fn id(&self) -> String {
        "1bit-halo-helm-tray".into()
    }

    fn title(&self) -> String {
        "1bit Helm".into()
    }

    fn icon_name(&self) -> String {
        // Theme fallback — Plasma finds this in every standard icon
        // theme. The base64 PNG placeholder in `tray::ICON_PNG_*` is
        // the strict-spec artifact; we prefer themed rendering in
        // practice.
        ICON_THEME_NAME.into()
    }

    fn menu(&self) -> Vec<ksni::MenuItem<Self>> {
        use ksni::menu::*;
        let status_line = build_status_line(&self.status);
        let mut items: Vec<ksni::MenuItem<Self>> = Vec::with_capacity(Action::ALL.len() + 1);
        for a in Action::ALL {
            match a {
                Action::Status => {
                    items.push(
                        StandardItem {
                            label: format!("{}: {}", Action::Status.label(), status_line),
                            enabled: false,
                            ..Default::default()
                        }
                        .into(),
                    );
                    items.push(ksni::MenuItem::Separator);
                }
                Action::StartAll => items.push(
                    StandardItem {
                        label: Action::StartAll.label().into(),
                        activate: Box::new(|_this: &mut Self| {
                            let _ = systemctl("start", SERVICES);
                        }),
                        ..Default::default()
                    }
                    .into(),
                ),
                Action::StopAll => items.push(
                    StandardItem {
                        label: Action::StopAll.label().into(),
                        activate: Box::new(|_this: &mut Self| {
                            let _ = systemctl("stop", SERVICES);
                        }),
                        ..Default::default()
                    }
                    .into(),
                ),
                Action::RestartServer => items.push(
                    StandardItem {
                        label: Action::RestartServer.label().into(),
                        activate: Box::new(|_this: &mut Self| {
                            let _ = systemctl("restart", &["strix-server"]);
                        }),
                        ..Default::default()
                    }
                    .into(),
                ),
                Action::OpenSite => items.push(
                    StandardItem {
                        label: Action::OpenSite.label().into(),
                        activate: Box::new(|_this: &mut Self| {
                            let _ = open_site();
                        }),
                        ..Default::default()
                    }
                    .into(),
                ),
                Action::Quit => items.push(
                    StandardItem {
                        label: Action::Quit.label().into(),
                        icon_name: "application-exit".into(),
                        activate: Box::new(|this: &mut Self| {
                            this.quit.store(true, Ordering::SeqCst);
                        }),
                        ..Default::default()
                    }
                    .into(),
                ),
            }
        }
        items
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let quit = std::sync::Arc::new(AtomicBool::new(false));
    let tray = HelmTray {
        status: probe_services(),
        quit: quit.clone(),
    };
    let handle = tray.spawn().await?;

    // Refresh loop. 3 s cadence per spec. Each tick re-probes every
    // service and asks ksni to re-render the menu via `update`.
    let refresh_handle = handle.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(REFRESH_INTERVAL).await;
            let fresh = probe_services();
            refresh_handle
                .update(move |t: &mut HelmTray| {
                    t.status = fresh;
                })
                .await;
        }
    });

    // Main task parks until Quit flips the flag. We check at
    // REFRESH_INTERVAL cadence — no busy loop, no channels needed for
    // an MVP.
    while !quit.load(Ordering::SeqCst) {
        tokio::time::sleep(REFRESH_INTERVAL).await;
    }
    Ok(())
}

//! helm-tauri host crate.
//!
//! Hosts the Tauri 2 process and registers the in-tree IPC commands
//! defined in [`commands`]. The frontend (React + react-mosaic) talks
//! to lemonade :8180 directly via `fetch`; the Tauri IPC bridge is
//! reserved for things the browser can't do — reading runbooks off
//! disk and querying systemd unit state.

mod commands;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            commands::read_runbook,
            commands::list_runbooks,
            commands::service_status,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

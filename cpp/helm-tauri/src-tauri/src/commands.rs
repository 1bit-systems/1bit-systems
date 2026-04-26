//! IPC commands exposed to the React frontend via `invoke()`.
//!
//! The frontend addresses lemonade and halo-agent over plain HTTP
//! (allowlisted by `tauri.conf.json` CSP). Anything that needs disk
//! or process access lives here. Each command is `#[tauri::command]`,
//! returns `Result<T, String>` so errors marshal cleanly into JS, and
//! is wired into the `invoke_handler!` list in `lib.rs`.
//!
//! Path safety: `read_runbook` and `list_runbooks` are anchored to the
//! in-tree `cpp/agent/configs/runbooks/` directory and reject any
//! relative path that escapes via `..` or absolute prefix.

use std::path::{Path, PathBuf};

use serde::Serialize;

/// Resolve the in-tree runbook root relative to the source checkout.
///
/// Layout:  `<repo>/cpp/agent/configs/runbooks/`
/// helm-tauri lives at: `<repo>/cpp/helm-tauri/src-tauri/`
fn runbook_root() -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR"); // .../cpp/helm-tauri/src-tauri
    PathBuf::from(manifest)
        .join("..")
        .join("..")
        .join("agent")
        .join("configs")
        .join("runbooks")
}

fn sanitize(rel: &str) -> Result<PathBuf, String> {
    let p = Path::new(rel);
    if p.is_absolute() {
        return Err("absolute paths rejected".into());
    }
    for c in p.components() {
        if matches!(c, std::path::Component::ParentDir) {
            return Err("parent traversal rejected".into());
        }
    }
    Ok(runbook_root().join(p))
}

/// Read a markdown runbook, anchored to `cpp/agent/configs/runbooks/`.
///
/// `rel_path` is relative; `..` and absolute paths are rejected.
#[tauri::command]
pub async fn read_runbook(rel_path: String) -> Result<String, String> {
    let abs = sanitize(&rel_path)?;
    tokio::fs::read_to_string(&abs)
        .await
        .map_err(|e| format!("read {abs:?}: {e}"))
}

/// List markdown runbooks under `cpp/agent/configs/runbooks/`.
///
/// Returns paths relative to the runbook root, sorted lexicographically.
#[tauri::command]
pub async fn list_runbooks() -> Result<Vec<String>, String> {
    let root = runbook_root();
    let mut out = Vec::new();
    let mut stack = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        let mut rd = match tokio::fs::read_dir(&dir).await {
            Ok(rd) => rd,
            Err(_) => continue, // missing root in dev tree is non-fatal
        };
        while let Ok(Some(entry)) = rd.next_entry().await {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|s| s.to_str()) == Some("md") {
                if let Ok(rel) = path.strip_prefix(&root) {
                    if let Some(s) = rel.to_str() {
                        out.push(s.to_string());
                    }
                }
            }
        }
    }
    out.sort();
    Ok(out)
}

/// Status of a 1bit-* systemd --user unit. Reports active state + pid.
#[derive(Debug, Serialize)]
pub struct ServiceStatus {
    pub name: String,
    pub active: bool,
    pub pid: Option<u32>,
}

/// Query `systemctl --user is-active` + `MainPID` for a unit name.
///
/// Restricted to units prefixed with `1bit-` so the bridge cannot be
/// abused as a generic systemd query channel from the frontend.
#[tauri::command]
pub async fn service_status(unit: String) -> Result<ServiceStatus, String> {
    if !unit.starts_with("1bit-") {
        return Err("only 1bit-* units allowed".into());
    }
    let active = tokio::process::Command::new("systemctl")
        .args(["--user", "is-active", &unit])
        .output()
        .await
        .map_err(|e| format!("spawn systemctl: {e}"))?;
    let active_str = String::from_utf8_lossy(&active.stdout).trim().to_string();
    let is_active = active_str == "active";

    let pid_out = tokio::process::Command::new("systemctl")
        .args(["--user", "show", "-p", "MainPID", "--value", &unit])
        .output()
        .await
        .map_err(|e| format!("spawn systemctl show: {e}"))?;
    let pid: Option<u32> = String::from_utf8_lossy(&pid_out.stdout)
        .trim()
        .parse()
        .ok()
        .filter(|p: &u32| *p != 0);

    Ok(ServiceStatus {
        name: unit,
        active: is_active,
        pid,
    })
}

//! 1bit-mcp-linuxgsm — MCP stdio bridge to LinuxGSM.
//!
//! Each LinuxGSM install exposes a `<game>server` bash driver under a
//! dedicated unix user (e.g. `mcserver`, `csgoserver`, `arkserver`).
//! This server spawns those drivers via `tokio::process::Command` (no
//! shell, arguments never expanded) and returns their stdout/exit code
//! as MCP tool results. Per the LinuxGSM docs we only call known-safe
//! subcommands: `details`, `status`, `start`, `stop`, `restart`,
//! `update`, `backup`. `console` and `sendcommand` are deliberately
//! excluded (interactive / arbitrary command injection risk).
//!
//! Discovery: by default we trust `$HALO_LINUXGSM_ROOT` (one directory
//! per server, each containing the `<name>server` driver). Falls back
//! to `$HOME/linuxgsm`.

use std::path::PathBuf;
use std::process::Stdio;

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

const ALLOWED_SUBCOMMANDS: &[&str] = &[
    "details", "status", "start", "stop", "restart", "update", "backup",
];

fn gsm_root() -> PathBuf {
    if let Ok(r) = std::env::var("HALO_LINUXGSM_ROOT") {
        return PathBuf::from(r);
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join("linuxgsm");
    }
    PathBuf::from("/var/lib/linuxgsm")
}

fn tools() -> Value {
    json!([
        {
            "name": "linuxgsm_list",
            "description": "List detected LinuxGSM servers under HALO_LINUXGSM_ROOT (one dir per server).",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "linuxgsm_run",
            "description": "Run an allowlisted <game>server subcommand. Returns stdout. Allowed: details, status, start, stop, restart, update, backup.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "server": { "type": "string", "description": "<game>server driver name (e.g. mcserver)" },
                    "subcommand": { "type": "string", "enum": ALLOWED_SUBCOMMANDS }
                },
                "required": ["server", "subcommand"]
            }
        }
    ])
}

fn text_result(s: &str, is_error: bool) -> Value {
    json!({ "content": [{ "type": "text", "text": s }], "isError": is_error })
}

async fn list_servers() -> Value {
    let root = gsm_root();
    let mut found = Vec::new();
    if let Ok(mut entries) = tokio::fs::read_dir(&root).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let name = match path.file_name().and_then(|s| s.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };
            let driver = path.join(format!("{name}server"));
            if driver.exists() {
                found.push(name);
            }
        }
    }
    text_result(&found.join("\n"), false)
}

async fn run_driver(server: &str, subcommand: &str) -> Value {
    if !ALLOWED_SUBCOMMANDS.contains(&subcommand) {
        return text_result(&format!("subcommand not allowed: {subcommand}"), true);
    }
    if !is_safe_server_name(server) {
        return text_result(&format!("invalid server name: {server}"), true);
    }
    let driver = gsm_root().join(server).join(format!("{server}server"));
    if !driver.exists() {
        return text_result(&format!("driver missing: {}", driver.display()), true);
    }
    let output = Command::new(&driver)
        .arg(subcommand)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await;
    match output {
        Ok(o) => {
            let stdout = String::from_utf8_lossy(&o.stdout).to_string();
            let stderr = String::from_utf8_lossy(&o.stderr).to_string();
            let combined = if stderr.is_empty() {
                stdout
            } else {
                format!("{stdout}\n---stderr---\n{stderr}")
            };
            text_result(&combined, !o.status.success())
        }
        Err(e) => text_result(&format!("spawn failed: {e}"), true),
    }
}

/// LinuxGSM server names are lowercase ASCII alphanumerics plus `-`/`_`.
/// Reject anything else to keep `..`, pipes, and shell metachars out of
/// the path we construct from the name.
pub fn is_safe_server_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
}

async fn handle(req: Value) -> Value {
    let id = req.get("id").cloned().unwrap_or(Value::Null);
    let method = req.get("method").and_then(Value::as_str).unwrap_or("");
    match method {
        "initialize" => json!({
            "jsonrpc": "2.0", "id": id,
            "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": { "tools": { "listChanged": false } },
                "serverInfo": { "name": "1bit-mcp-linuxgsm", "version": env!("CARGO_PKG_VERSION") }
            }
        }),
        "tools/list" => json!({
            "jsonrpc": "2.0", "id": id,
            "result": { "tools": tools() }
        }),
        "tools/call" => {
            let name = req
                .pointer("/params/name")
                .and_then(Value::as_str)
                .unwrap_or("");
            let args = req
                .pointer("/params/arguments")
                .cloned()
                .unwrap_or(json!({}));
            let result = match name {
                "linuxgsm_list" => list_servers().await,
                "linuxgsm_run" => {
                    let server = args.get("server").and_then(Value::as_str).unwrap_or("");
                    let sub = args.get("subcommand").and_then(Value::as_str).unwrap_or("");
                    run_driver(server, sub).await
                }
                other => text_result(&format!("unknown tool: {other}"), true),
            };
            json!({ "jsonrpc": "2.0", "id": id, "result": result })
        }
        _ => json!({
            "jsonrpc": "2.0", "id": id,
            "error": { "code": -32601, "message": format!("method not found: {method}") }
        }),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let stdin = tokio::io::stdin();
    let mut stdin = BufReader::new(stdin);
    let mut stdout = tokio::io::stdout();
    let mut line = String::new();
    loop {
        line.clear();
        let n = stdin.read_line(&mut line).await?;
        if n == 0 {
            break;
        }
        let req: Value = match serde_json::from_str(line.trim_end()) {
            Ok(v) => v,
            Err(e) => {
                let err = json!({
                    "jsonrpc": "2.0", "id": Value::Null,
                    "error": { "code": -32700, "message": format!("parse error: {e}") }
                });
                writeln_frame(&mut stdout, &err).await?;
                continue;
            }
        };
        let resp = handle(req).await;
        writeln_frame(&mut stdout, &resp).await?;
    }
    Ok(())
}

async fn writeln_frame<W: tokio::io::AsyncWriteExt + Unpin>(
    w: &mut W,
    v: &Value,
) -> std::io::Result<()> {
    let mut s = serde_json::to_string(v).map_err(std::io::Error::other)?;
    s.push('\n');
    w.write_all(s.as_bytes()).await?;
    w.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowed_subcommands_do_not_include_interactive_ones() {
        assert!(!ALLOWED_SUBCOMMANDS.contains(&"console"));
        assert!(!ALLOWED_SUBCOMMANDS.contains(&"sendcommand"));
        assert!(ALLOWED_SUBCOMMANDS.contains(&"status"));
    }

    #[test]
    fn safe_server_name_rejects_traversal() {
        assert!(!is_safe_server_name(".."));
        assert!(!is_safe_server_name("mc/server"));
        assert!(!is_safe_server_name("mc server"));
        assert!(!is_safe_server_name(""));
        assert!(is_safe_server_name("mcserver"));
        assert!(is_safe_server_name("cs2-server"));
        assert!(is_safe_server_name("ark_server"));
    }

    #[tokio::test]
    async fn run_driver_rejects_disallowed_subcommand() {
        let v = run_driver("mcserver", "console").await;
        assert!(v["isError"].as_bool().unwrap_or(false));
        assert!(
            v["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("not allowed")
        );
    }

    #[tokio::test]
    async fn run_driver_rejects_unsafe_server_name() {
        let v = run_driver("../etc", "status").await;
        assert!(v["isError"].as_bool().unwrap_or(false));
    }

    #[tokio::test]
    async fn initialize_advertises_server_name() {
        let req = json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize" });
        let resp = handle(req).await;
        assert_eq!(resp["result"]["serverInfo"]["name"], "1bit-mcp-linuxgsm");
    }
}

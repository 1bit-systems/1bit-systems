//! 1bit-mcp — JSON-RPC 2.0 stdio MCP server skeleton.
//!
//! This crate previously wrapped `onebit_agents::Registry` to expose the
//! 17-specialist surface as MCP tools. After the 2026-04-25 cull
//! (`onebit-agents` deleted; superseded by GAIA agent-core in
//! `1bit-services/agent-core/`), the registry-backed implementation was
//! retired here. The Rust crate is held as a slot for a future
//! re-targeting at the new GAIA agent-core surface; the canonical port
//! target is `1bit-services/mcp/` (C++).
//!
//! What ships today: an empty tool registry + a no-op `StdioServer` that
//! returns the empty list on `tools/list` and an error on `tools/call`.
//! This keeps the workspace green and preserves the on-disk crate so
//! cargo references / packages.toml `[component.mcp]` do not break.

pub const PROTOCOL_VERSION: &str = "2024-11-05";
pub const SERVER_NAME: &str = "1bit-mcp";
pub const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Placeholder registry. Empty until re-pointed at GAIA agent-core.
#[derive(Debug, Default, Clone)]
pub struct ToolRegistry;

impl ToolRegistry {
    pub fn new() -> Self {
        Self
    }
    pub fn len(&self) -> usize {
        0
    }
    pub fn is_empty(&self) -> bool {
        true
    }
}

/// Stdio JSON-RPC server skeleton. Reads `\n`-delimited JSON-RPC objects
/// from `reader`, writes responses to `writer`. Tool surface is empty —
/// `tools/list` returns `[]` and `tools/call` returns method-not-found.
#[derive(Debug, Default)]
pub struct StdioServer {
    registry: ToolRegistry,
}

impl StdioServer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compatibility shim for old call sites; no agents are wired in.
    pub fn with_default_agents() -> Self {
        Self::default()
    }

    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }

    /// Drive the JSON-RPC loop until EOF on `reader`. Always returns
    /// `Ok(())` on clean EOF.
    pub async fn run<R, W>(self, mut reader: R, mut writer: W) -> std::io::Result<()>
    where
        R: tokio::io::AsyncRead + Unpin,
        W: tokio::io::AsyncWrite + Unpin,
    {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        let mut lines = BufReader::new(&mut reader).lines();
        while let Some(line) = lines.next_line().await? {
            if line.trim().is_empty() {
                continue;
            }
            // Parse just enough to extract the request id; any malformed
            // input gets a generic parse error.
            let val: serde_json::Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(_) => {
                    let resp = serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": null,
                        "error": { "code": -32700, "message": "parse error" },
                    });
                    let mut bytes = serde_json::to_vec(&resp)?;
                    bytes.push(b'\n');
                    writer.write_all(&bytes).await?;
                    continue;
                }
            };
            let id = val.get("id").cloned().unwrap_or(serde_json::Value::Null);
            let method = val.get("method").and_then(|m| m.as_str()).unwrap_or("");
            let resp = match method {
                "initialize" => serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "protocolVersion": PROTOCOL_VERSION,
                        "capabilities": { "tools": {} },
                        "serverInfo": { "name": SERVER_NAME, "version": SERVER_VERSION },
                    },
                }),
                "tools/list" => serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": { "tools": [] },
                }),
                _ => serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": { "code": -32601, "message": "method not found" },
                }),
            };
            let mut bytes = serde_json::to_vec(&resp)?;
            bytes.push(b'\n');
            writer.write_all(&bytes).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_is_empty() {
        let r = ToolRegistry::new();
        assert_eq!(r.len(), 0);
        assert!(r.is_empty());
    }

    #[test]
    fn server_default_constructs() {
        let s = StdioServer::new();
        assert_eq!(s.registry().len(), 0);
    }

    #[tokio::test]
    async fn tools_list_returns_empty() {
        let req = b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}\n";
        let mut out = Vec::new();
        StdioServer::new().run(&req[..], &mut out).await.unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("\"tools\":[]"), "got: {s}");
    }
}

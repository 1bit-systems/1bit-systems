//! Spawn a child process that speaks MCP over stdio and drive it via
//! newline-delimited JSON-RPC 2.0.
//!
//! Used for servers that ship as a local binary — e.g. our own
//! `1bit-mcp-discord`, `1bit-mcp-linuxgsm`, or third-party Node/Python
//! implementations. Child is spawned with piped stdin/stdout; stderr is
//! inherited so the parent can see server-side logs.

use std::ffi::OsStr;
use std::process::Stdio;

use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

use crate::error::McpError;
use crate::protocol::{Response, Tool, ToolCallResult, build_request, initialize_params};

pub struct StdioClient {
    child: Mutex<Child>,
    stdin: Mutex<ChildStdin>,
    stdout: Mutex<BufReader<ChildStdout>>,
    next_id: Mutex<u64>,
}

impl StdioClient {
    /// Spawn `program` with `args` and hold onto its stdio.
    pub async fn spawn<S, I>(program: S, args: I) -> Result<Self, McpError>
    where
        S: AsRef<OsStr>,
        I: IntoIterator<Item = S>,
    {
        let mut cmd = Command::new(program);
        for a in args {
            cmd.arg(a);
        }
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        let mut child = cmd.spawn()?;
        let stdin = child.stdin.take().ok_or(McpError::Closed)?;
        let stdout = child.stdout.take().ok_or(McpError::Closed)?;
        Ok(Self {
            child: Mutex::new(child),
            stdin: Mutex::new(stdin),
            stdout: Mutex::new(BufReader::new(stdout)),
            next_id: Mutex::new(1),
        })
    }

    async fn next_id(&self) -> u64 {
        let mut g = self.next_id.lock().await;
        let id = *g;
        *g += 1;
        id
    }

    async fn send(&self, frame: &Value) -> Result<(), McpError> {
        let mut line = serde_json::to_string(frame)?;
        line.push('\n');
        let mut stdin = self.stdin.lock().await;
        stdin.write_all(line.as_bytes()).await?;
        stdin.flush().await?;
        Ok(())
    }

    async fn recv(&self) -> Result<Response, McpError> {
        let mut buf = String::new();
        let mut stdout = self.stdout.lock().await;
        let n = stdout.read_line(&mut buf).await?;
        if n == 0 {
            return Err(McpError::Closed);
        }
        let resp: Response = serde_json::from_str(buf.trim_end())?;
        if let Some(err) = resp.error.as_ref() {
            return Err(McpError::Rpc {
                code: err.code,
                message: err.message.clone(),
            });
        }
        Ok(resp)
    }

    pub async fn initialize(
        &self,
        client_name: &str,
        client_version: &str,
    ) -> Result<Value, McpError> {
        let id = self.next_id().await;
        let req = build_request(
            id,
            "initialize",
            Some(initialize_params(client_name, client_version)),
        );
        self.send(&req).await?;
        let resp = self.recv().await?;
        resp.result
            .ok_or_else(|| McpError::Protocol("initialize returned no result".into()))
    }

    pub async fn list_tools(&self) -> Result<Vec<Tool>, McpError> {
        let id = self.next_id().await;
        let req = build_request(id, "tools/list", Some(serde_json::json!({})));
        self.send(&req).await?;
        let resp = self.recv().await?;
        let result = resp
            .result
            .ok_or_else(|| McpError::Protocol("no result".into()))?;
        let tools = result
            .get("tools")
            .cloned()
            .unwrap_or_else(|| serde_json::json!([]));
        Ok(serde_json::from_value(tools)?)
    }

    pub async fn call_tool(
        &self,
        name: &str,
        arguments: Value,
    ) -> Result<ToolCallResult, McpError> {
        let id = self.next_id().await;
        let req = build_request(
            id,
            "tools/call",
            Some(serde_json::json!({ "name": name, "arguments": arguments })),
        );
        self.send(&req).await?;
        let resp = self.recv().await?;
        let result = resp
            .result
            .ok_or_else(|| McpError::Protocol("no result".into()))?;
        Ok(serde_json::from_value(result)?)
    }

    pub async fn shutdown(&self) -> Result<(), McpError> {
        let mut child = self.child.lock().await;
        let _ = child.start_kill();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spawn an echo-style fake MCP server (bash + printf). Verifies the
    /// framing: request written to stdin → response line read from stdout.
    #[tokio::test]
    async fn stdio_round_trips_a_canned_response() {
        // Minimal fake server: read one line, discard it, print a canned
        // initialize response, then exit. Uses /bin/sh which is always
        // available on the strix box.
        let canned = r#"{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-06-18","capabilities":{},"serverInfo":{"name":"fake","version":"0"}}}"#;
        let script = format!(
            "read -r _; printf '%s\\n' '{}'",
            canned.replace('\'', "'\\''")
        );
        let client = StdioClient::spawn("/bin/sh", ["-c", &script].iter().copied())
            .await
            .expect("spawn");
        let result = client.initialize("halo-test", "0.0.1").await.expect("init");
        assert_eq!(result["protocolVersion"], "2025-06-18");
    }

    #[tokio::test]
    async fn stdio_propagates_rpc_error() {
        let canned =
            r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"method not found"}}"#;
        let script = format!(
            "read -r _; printf '%s\\n' '{}'",
            canned.replace('\'', "'\\''")
        );
        let client = StdioClient::spawn("/bin/sh", ["-c", &script].iter().copied())
            .await
            .expect("spawn");
        let err = client.initialize("halo-test", "0.0.1").await.unwrap_err();
        match err {
            McpError::Rpc { code, .. } => assert_eq!(code, -32601),
            other => panic!("expected Rpc error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stdio_closed_stream_returns_closed() {
        // Exit immediately → no response → Closed.
        let client = StdioClient::spawn("/bin/true", std::iter::empty::<&str>())
            .await
            .expect("spawn");
        let err = client.initialize("halo-test", "0.0.1").await.unwrap_err();
        assert!(matches!(err, McpError::Closed));
    }
}

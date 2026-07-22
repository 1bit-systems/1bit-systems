//! MCP (Model Context Protocol) client for tool execution.
//!
//! Connects to MCP servers over stdio transport, lists available tools,
//! and calls them.  This is what gives the agent its ability to read files,
//! run bash commands, edit code, and use other tools — without depending
//! on pi's Node.js runtime for tool execution.
//!
//! # Protocol
//!
//! MCP uses JSON-RPC 2.0 over stdio.  Each server is a child process that
//! speaks JSON-RPC on stdin/stdout.  The key methods:
//!
//!   - `initialize`          — handshake
//!   - `tools/list`          — get available tool definitions
//!   - `tools/call`          — invoke a tool with arguments

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::io::{BufRead, BufReader, Write};
use std::sync::atomic::{AtomicU64, Ordering};

// ── JSON-RPC types ──────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: u64,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct JsonRpcResponse {
    #[allow(dead_code)]
    id: Option<u64>,
    result: Option<Value>,
    error: Option<JsonRpcError>,
}

#[derive(Debug, Deserialize)]
struct JsonRpcError {
    #[allow(dead_code)]
    code: i64,
    message: String,
}

// ── MCP server config ───────────────────────────────────────────

/// A single MCP server entry, matching the codewhale/pi mcp.json format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    pub url: Option<String>,
    #[serde(default)]
    pub disabled: bool,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub enabled_tools: Vec<String>,
    #[serde(default)]
    pub disabled_tools: Vec<String>,
}

/// Top-level MCP config (the `servers` map).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfig {
    #[serde(default)]
    pub servers: HashMap<String, McpServerConfig>,
}

/// A tool definition returned by `tools/list`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: Option<String>,
    #[serde(default)]
    pub input_schema: Value,
}

/// Result of a `tools/call`.
#[derive(Debug, Deserialize)]
pub struct ToolCallResult {
    #[serde(default)]
    pub content: Vec<ToolContent>,
    #[serde(default)]
    pub is_error: bool,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct ToolContent {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: Option<String>,
    pub data: Option<String>,
    pub mime_type: Option<String>,
}

// ── Stdio MCP client ────────────────────────────────────────────

/// A connected MCP server over stdio transport.
pub struct McpClient {
    name: String,
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    reader: Option<BufReader<ChildStdout>>,
    next_id: AtomicU64,
    tools: Vec<ToolDefinition>,
}

impl McpClient {
    /// Connect to an MCP server by spawning its command.
    pub fn connect(name: &str, config: &McpServerConfig) -> Result<Self> {
        let cmd = config
            .command
            .as_ref()
            .context("MCP server has no command (HTTP transport not yet supported)")?;

        let mut child = Command::new(cmd);
        child.args(&config.args);
        child.envs(&config.env);
        child.stdin(Stdio::piped());
        child.stdout(Stdio::piped());
        child.stderr(Stdio::inherit());

        let mut child = child.spawn().context("Failed to spawn MCP server")?;

        let stdin = child.stdin.take().context("Failed to open MCP stdin")?;
        let stdout = child.stdout.take().context("Failed to open MCP stdout")?;
        let reader = BufReader::new(stdout);

        let mut client = Self {
            name: name.to_string(),
            child: Some(child),
            stdin: Some(stdin),
            reader: Some(reader),
            next_id: AtomicU64::new(1),
            tools: vec![],
        };

        // Initialize
        client.send_request("initialize", Some(serde_json::json!({
            "protocolVersion": "0.1.0",
            "capabilities": {},
            "clientInfo": {
                "name": "onebit-cli",
                "version": "0.1.0"
            }
        })))?;

        Ok(client)
    }

    /// Initialize the tools list by calling `tools/list`.
    pub fn init_tools(&mut self) -> Result<Vec<ToolDefinition>> {
        let result = self.send_request("tools/list", None)?;
        let tools = result
            .get("tools")
            .and_then(|v| serde_json::from_value::<Vec<ToolDefinition>>(v.clone()).ok())
            .unwrap_or_default();
        self.tools = tools.clone();
        Ok(tools)
    }

    /// Get the list of available tools.
    pub fn tools(&self) -> &[ToolDefinition] {
        &self.tools
    }

    /// Call a tool by name with the given arguments.
    pub fn call_tool(&mut self, name: &str, args: Value) -> Result<ToolCallResult> {
        let result = self.send_request("tools/call", Some(serde_json::json!({
            "name": name,
            "arguments": args
        })))?;
        serde_json::from_value::<ToolCallResult>(result)
            .context("Failed to parse tool call result")
    }

    /// Send a JSON-RPC request and read the response.
    fn send_request(&mut self, method: &str, params: Option<Value>) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);

        let request = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id,
            method: method.to_string(),
            params,
        };

        let mut request_bytes = serde_json::to_vec(&request)?;
        request_bytes.push(b'\n');

        let stdin = self.stdin.as_mut().context("MCP stdin closed")?;
        stdin.write_all(&request_bytes)?;
        stdin.flush()?;

        let reader = self.reader.as_mut().context("MCP stdout closed")?;
        let mut line = String::new();
        reader.read_line(&mut line)?;

        if line.trim().is_empty() {
            anyhow::bail!("MCP server '{}' returned empty response", self.name);
        }

        let response: JsonRpcResponse = serde_json::from_str(&line)?;

        if let Some(err) = response.error {
            anyhow::bail!("MCP error from '{}': {} (code {})", self.name, err.message, err.code);
        }

        response.result.ok_or_else(|| anyhow::anyhow!("MCP response missing result"))
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Manager for multiple MCP servers.
pub struct McpManager {
    pub clients: HashMap<String, McpClient>,
}

impl McpManager {
    /// Load MCP config from the standard config path and connect all enabled servers.
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path()?;
        let mut manager = Self {
            clients: HashMap::new(),
        };

        if !config_path.exists() {
            return Ok(manager);
        }

        let raw = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read MCP config from {}", config_path.display()))?;
        let config: McpConfig = serde_json::from_str(&raw)?;

        for (name, server_config) in &config.servers {
            if server_config.disabled {
                continue;
            }
            if server_config.command.is_none() {
                tracing::debug!("Skipping MCP server '{name}': no command (HTTP transport)");
                continue;
            }

            match McpClient::connect(name, server_config) {
                Ok(client) => {
                    tracing::info!("Connected to MCP server: {name}");
                    manager.clients.insert(name.clone(), client);
                }
                Err(e) => {
                    tracing::warn!("Failed to connect to MCP server '{name}': {e}");
                }
            }
        }

        // Initialize tools for all connected servers
        let server_names: Vec<String> = manager.clients.keys().cloned().collect();
        for name in server_names {
            if let Some(client) = manager.clients.get_mut(&name) {
                match client.init_tools() {
                    Ok(tools) => {
                        tracing::info!("MCP '{name}' offers {} tool(s)", tools.len());
                    }
                    Err(e) => {
                        tracing::warn!("Failed to initialize tools for '{name}': {e}");
                    }
                }
            }
        }

        Ok(manager)
    }

    /// Path to the MCP config file (~/.1bit/mcp.json).
    fn config_path() -> Result<PathBuf> {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .context("Cannot find home directory")?;
        Ok(PathBuf::from(home).join(".1bit").join("mcp.json"))
    }

    /// Get all tools across all servers.
    pub fn all_tools(&self) -> Vec<(String, ToolDefinition)> {
        let mut tools = Vec::new();
        for (server_name, client) in &self.clients {
            for tool in client.tools() {
                tools.push((server_name.clone(), tool.clone()));
            }
        }
        tools
    }

    /// Call a tool on a specific server.
    pub fn call_tool(&mut self, server_name: &str, tool_name: &str, args: Value) -> Result<ToolCallResult> {
        let client = self
            .clients
            .get_mut(server_name)
            .with_context(|| format!("MCP server '{server_name}' not connected"))?;
        client.call_tool(tool_name, args)
    }

    /// Find which server hosts a tool and call it.
    pub fn call_tool_by_name(&mut self, tool_name: &str, args: Value) -> Result<ToolCallResult> {
        // First pass: find which server hosts this tool
        let server_name = {
            let mut found = None;
            for (server_name, client) in &self.clients {
                for tool in client.tools() {
                    if tool.name == tool_name {
                        found = Some(server_name.clone());
                        break;
                    }
                }
            }
            found
        };

        match server_name {
            Some(name) => self.call_tool(&name, tool_name, args),
            None => anyhow::bail!("No MCP server provides tool '{tool_name}'"),
        }
    }
}

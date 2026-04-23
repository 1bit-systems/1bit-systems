//! JSON-RPC 2.0 wire types for the subset of MCP we speak on the client
//! side: `initialize`, `tools/list`, `tools/call`.
//!
//! We intentionally do not implement the full MCP spec — no prompts,
//! no resources, no sampling — because halo's outbound use case is
//! tool-calling only. If that changes, add methods here and bump
//! `PROTOCOL_VERSION`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// MCP protocol version we advertise in `initialize`. Matches the
/// `2025-06-18` revision.
pub const PROTOCOL_VERSION: &str = "2025-06-18";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request<'a> {
    pub jsonrpc: &'a str,
    pub id: u64,
    pub method: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Response {
    #[allow(dead_code)]
    pub jsonrpc: String,
    pub id: Option<u64>,
    #[serde(default)]
    pub result: Option<Value>,
    #[serde(default)]
    pub error: Option<RpcErrorPayload>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RpcErrorPayload {
    pub code: i64,
    pub message: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct Tool {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default, rename = "inputSchema")]
    pub input_schema: Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCallResult {
    #[serde(default)]
    pub content: Vec<ContentBlock>,
    #[serde(default, rename = "isError")]
    pub is_error: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    #[serde(other)]
    Other,
}

impl ContentBlock {
    pub fn as_text(&self) -> Option<&str> {
        match self {
            ContentBlock::Text { text } => Some(text),
            _ => None,
        }
    }
}

pub fn build_request(id: u64, method: &str, params: Option<Value>) -> Value {
    serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
        "params": params,
    })
}

pub fn initialize_params(client_name: &str, client_version: &str) -> Value {
    serde_json::json!({
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {},
        "clientInfo": { "name": client_name, "version": client_version }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_version_is_known_revision() {
        assert_eq!(PROTOCOL_VERSION, "2025-06-18");
    }

    #[test]
    fn build_request_shape_matches_jsonrpc_2() {
        let v = build_request(7, "tools/list", None);
        assert_eq!(v["jsonrpc"], "2.0");
        assert_eq!(v["id"], 7);
        assert_eq!(v["method"], "tools/list");
    }

    #[test]
    fn tool_deserializes_loose_schema() {
        let raw = r#"{ "name":"x","description":"d" }"#;
        let t: Tool = serde_json::from_str(raw).unwrap();
        assert_eq!(t.name, "x");
        assert_eq!(t.description, "d");
        assert!(t.input_schema.is_null());
    }

    #[test]
    fn content_block_text_extracts() {
        let raw = r#"{"type":"text","text":"hello"}"#;
        let b: ContentBlock = serde_json::from_str(raw).unwrap();
        assert_eq!(b.as_text(), Some("hello"));
    }

    #[test]
    fn content_block_unknown_is_other() {
        let raw = r#"{"type":"image","data":"..."}"#;
        let b: ContentBlock = serde_json::from_str(raw).unwrap();
        assert!(b.as_text().is_none());
    }
}

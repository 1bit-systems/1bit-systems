//! Streamable-HTTP transport. MCP 2025-06-18 permits a single HTTP
//! endpoint that accepts JSON-RPC 2.0 POSTs and returns either a single
//! JSON response or a Server-Sent-Events stream. We only speak the
//! single-response form here — that covers GitHub, Semgrep, DeepWiki,
//! Linear, Sentry, and every other "remote MCP" on mcpservers.org.

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;
use tokio::sync::Mutex;

use crate::error::McpError;
use crate::protocol::{Response, Tool, ToolCallResult, build_request, initialize_params};

#[derive(Debug)]
pub struct HttpClient {
    endpoint: String,
    client: reqwest::Client,
    headers: HeaderMap,
    next_id: Mutex<u64>,
}

impl HttpClient {
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            client: reqwest::Client::new(),
            headers: HeaderMap::new(),
            next_id: Mutex::new(1),
        }
    }

    /// Add a header applied to every outgoing request — e.g. `Authorization:
    /// Bearer <pat>` for GitHub.
    pub fn header(mut self, name: &str, value: &str) -> Result<Self, McpError> {
        let k: HeaderName = name
            .parse()
            .map_err(|_| McpError::Protocol(format!("invalid header name: {name}")))?;
        let v: HeaderValue = value
            .parse()
            .map_err(|_| McpError::Protocol(format!("invalid header value: {value}")))?;
        self.headers.insert(k, v);
        Ok(self)
    }

    async fn next_id(&self) -> u64 {
        let mut g = self.next_id.lock().await;
        let id = *g;
        *g += 1;
        id
    }

    async fn round_trip(&self, method: &str, params: Option<Value>) -> Result<Value, McpError> {
        let id = self.next_id().await;
        let req = build_request(id, method, params);
        let mut builder = self
            .client
            .post(&self.endpoint)
            .header("Accept", "application/json, text/event-stream")
            .header("Content-Type", "application/json")
            .json(&req);
        for (k, v) in self.headers.iter() {
            builder = builder.header(k, v);
        }
        let http_resp = builder.send().await?;
        if !http_resp.status().is_success() {
            return Err(McpError::Protocol(format!(
                "http status {}",
                http_resp.status()
            )));
        }
        let body: Response = http_resp.json().await?;
        if let Some(err) = body.error {
            return Err(McpError::Rpc {
                code: err.code,
                message: err.message,
            });
        }
        body.result
            .ok_or_else(|| McpError::Protocol("empty result".into()))
    }

    pub async fn initialize(
        &self,
        client_name: &str,
        client_version: &str,
    ) -> Result<Value, McpError> {
        self.round_trip(
            "initialize",
            Some(initialize_params(client_name, client_version)),
        )
        .await
    }

    pub async fn list_tools(&self) -> Result<Vec<Tool>, McpError> {
        let result = self
            .round_trip("tools/list", Some(serde_json::json!({})))
            .await?;
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
        let result = self
            .round_trip(
                "tools/call",
                Some(serde_json::json!({ "name": name, "arguments": arguments })),
            )
            .await?;
        Ok(serde_json::from_value(result)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_rejects_invalid_name() {
        let c = HttpClient::new("http://localhost:9/mcp");
        let err = c.header("bad name\n", "v").unwrap_err();
        assert!(matches!(err, McpError::Protocol(_)));
    }

    #[test]
    fn header_accepts_bearer() {
        let c = HttpClient::new("http://localhost:9/mcp")
            .header("Authorization", "Bearer xyz")
            .expect("ok");
        assert_eq!(
            c.headers.get("Authorization").unwrap().to_str().unwrap(),
            "Bearer xyz"
        );
    }

    #[test]
    fn endpoint_preserved() {
        let c = HttpClient::new("https://api.example/mcp");
        assert_eq!(c.endpoint, "https://api.example/mcp");
    }
}

//! NPU API client — communicates with the 1bit inference engine.
//!
//! Talks to the OpenAI-compatible endpoint (default `http://127.0.0.1:9090/v1`)
//! served by the `onebitd` proxy in front of `bitnet_decode`.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

/// NPU API client wrapping an HTTP client + base URL.
#[derive(Debug, Clone)]
pub struct NpuClient {
    client: reqwest::Client,
    base_url: String,
}

// ── Message types ───────────────────────────────────────────────

/// A chat message in the OpenAI API format.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Message {
    /// Standard user/assistant/system message.
    Text {
        role: String,
        content: String,
    },
    /// Assistant message with tool calls.
    AssistantWithTools {
        role: String,
        content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ToolCall>>,
    },
    /// Tool result message.
    ToolResult {
        role: String,
        tool_call_id: String,
        content: String,
    },
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Message::Text {
            role: "user".into(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Message::Text {
            role: "assistant".into(),
            content: content.into(),
        }
    }

    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Message::ToolResult {
            role: "tool".into(),
            tool_call_id: tool_call_id.into(),
            content: content.into(),
        }
    }

    pub fn assistant_with_tools(tool_calls: Vec<ToolCall>) -> Self {
        Message::AssistantWithTools {
            role: "assistant".into(),
            content: None,
            tool_calls: Some(tool_calls),
        }
    }
}

/// A tool call from the assistant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

/// A tool definition to advertise to the model.
#[derive(Debug, Clone, Serialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
}

// ── API request/response types ──────────────────────────────────

/// Request body for `/v1/chat/completions`.
#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDefinition>>,
}

/// Response body for a non-streaming `/v1/chat/completions`.
#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Option<Vec<Choice>>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: Option<ChoiceMessage>,
}

#[derive(Debug, Deserialize)]
struct ChoiceMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
}

/// Response from `/v1/models`
#[derive(Debug, Deserialize)]
pub struct ModelsResponse {
    pub data: Option<Vec<ModelInfo>>,
}

#[derive(Debug, Deserialize)]
pub struct ModelInfo {
    pub id: String,
}

/// Result of a chat completion — either a text response or tool calls.
#[derive(Debug)]
pub enum ChatResult {
    Text(String),
    ToolCalls(Vec<ToolCall>),
}

impl NpuClient {
    /// Create a new NPU API client targeting the given base URL.
    pub fn new(base_url: &str) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()?;
        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
        })
    }

    /// Health check — GET /health (or /v1/models as fallback).
    pub async fn health_check(&self) -> Result<bool> {
        let health_url = self.base_url.replace("/v1", "/health");
        if let Ok(resp) = self.client.get(&health_url).timeout(Duration::from_secs(3)).send().await {
            if resp.status().is_success() {
                return Ok(true);
            }
        }

        let models_url = format!("{}/models", self.base_url);
        match self.client.get(&models_url).timeout(Duration::from_secs(3)).send().await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    /// List available models.
    pub async fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/models", self.base_url);
        let resp = self
            .client
            .get(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .context("Failed to connect to NPU API")?;

        if !resp.status().is_success() {
            anyhow::bail!("NPU API returned status {}", resp.status());
        }

        let body: ModelsResponse = resp
            .json()
            .await
            .context("Failed to parse models response")?;

        Ok(body
            .data
            .unwrap_or_default()
            .into_iter()
            .map(|m| m.id)
            .collect())
    }

    /// Send a chat completion, returning either text or tool calls.
    pub async fn chat(
        &self,
        model: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        max_tokens: Option<u32>,
    ) -> Result<ChatResult> {
        let url = format!("{}/chat/completions", self.base_url);

        let body = ChatRequest {
            model: model.to_string(),
            messages,
            stream: false,
            max_tokens,
            tools,
        };

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .timeout(Duration::from_secs(120))
            .send()
            .await
            .context("Failed to send chat request to NPU API")?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("NPU API error ({}): {}", status, text);
        }

        let chat_resp: ChatResponse = resp
            .json()
            .await
            .context("Failed to parse chat response")?;

        let choice = chat_resp
            .choices
            .and_then(|c| c.into_iter().next())
            .context("Empty response from NPU API")?;

        let msg = choice.message.context("No message in response")?;

        if let Some(tool_calls) = msg.tool_calls {
            if !tool_calls.is_empty() {
                return Ok(ChatResult::ToolCalls(tool_calls));
            }
        }

        Ok(ChatResult::Text(
            msg.content.unwrap_or_else(|| "[no response]".to_string()),
        ))
    }
}

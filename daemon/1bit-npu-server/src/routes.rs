//! HTTP route handlers for the 1bit NPU Chat Server.

use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::Event, IntoResponse, Response, Sse},
    routing::{get, post},
    Json, Router,
};
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokenizers::Tokenizer;

use crate::inference::Inference;
use tracing::{error, info};

/// Shared state across all handlers.
pub struct AppState {
    pub inference: Arc<Mutex<Inference>>,
    pub tokenizer: Tokenizer,
}

/// Build the router.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/models", get(handle_models))
        .route("/health", get(handle_health))
        .route("/v1/chat/completions", post(handle_chat_completions))
        .route("/api/generate", post(handle_chat_completions))
        .route("/api/chat", post(handle_chat_completions))
        .with_state(state)
}

async fn handle_models() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [{"id": "default", "object": "model"}]
    }))
}

async fn handle_health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}



async fn handle_chat_completions(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> Response {
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let max_tokens = body.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(256) as usize;
    let prompt = extract_prompt(&body);

    // Tokenize
    let ids = {
        let tok = &state.tokenizer;
        let encoded = tok.encode(prompt.as_str(), false).map_err(|e| {
            (StatusCode::BAD_REQUEST, format!("Tokenization error: {e}"))
        });
        match encoded {
            Ok(e) => {
                let ids: Vec<u32> = e.get_ids().iter().take(512).copied().collect();
                ids
            }
            Err(e) => return (e.0, e.1).into_response(),
        }
    };

    if ids.is_empty() {
        return (StatusCode::BAD_REQUEST, "No tokens to process").into_response();
    }

    // Generate
    let out_ids = {
        let mut inf = state.inference.lock().await;
        inf.generate(&ids, max_tokens)
    };

    // Decode
    let text = state.tokenizer.decode(&out_ids, false).unwrap_or_default();

    if stream {
        // SSE streaming
        let words: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
        let stream = tokio_stream::iter(words.into_iter().enumerate().map(move |(_i, word)| {
            let chunk = serde_json::json!({
                "choices": [{
                    "delta": {"content": format!("{} ", word)},
                    "index": 0
                }]
            });
            Ok::<_, Infallible>(Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()))
        }));
        Sse::new(stream)
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response()
    } else {
        let resp = serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": ids.len(),
                "completion_tokens": out_ids.len(),
                "total_tokens": ids.len() + out_ids.len(),
            },
        });
        Json(resp).into_response()
    }
}

/// Extract prompt text from the request body, supporting multiple formats.
fn extract_prompt(body: &serde_json::Value) -> String {
    // Check for messages array (OpenAI format)
    if let Some(messages) = body.get("messages").and_then(|v| v.as_array()) {
        let parts: Vec<String> = messages
            .iter()
            .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
            .map(|s| s.to_string())
            .collect();
        return parts.join("\n");
    }

    // Check for "prompt" field (Ollama format)
    if let Some(prompt) = body.get("prompt").and_then(|v| v.as_str()) {
        return prompt.to_string();
    }

    // Check for "input" field
    if let Some(input) = body.get("input").and_then(|v| v.as_str()) {
        return input.to_string();
    }

    String::new()
}

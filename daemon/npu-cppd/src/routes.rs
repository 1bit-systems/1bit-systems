//! HTTP route handlers for the NPU C++ Engine Daemon.

use crate::backend::NpuBackend;
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use std::sync::Arc;
use tracing::error;

pub struct AppState {
    pub backend: Arc<NpuBackend>,
    pub model_name: String,
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(handle_health))
        .route("/v1/models", get(handle_models))
        .route("/v1/chat/completions", post(handle_chat_completions))
        .with_state(state)
}

async fn handle_health(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let ready = state.backend.is_ready().await;
    Json(serde_json::json!({
        "status": "ok",
        "ready": ready,
    }))
}

async fn handle_models(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "data": [{
            "id": state.model_name,
            "object": "model",
        }]
    }))
}

#[derive(Deserialize)]
struct ChatBody {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
    #[serde(flatten)]
    #[allow(dead_code)]
    extra: std::collections::HashMap<String, serde_json::Value>,
}

fn default_max_tokens() -> u32 {
    64
}

#[derive(Deserialize)]
struct ChatMessage {
    #[serde(default)]
    role: String,
    #[serde(default)]
    content: String,
}

async fn handle_chat_completions(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ChatBody>,
) -> impl IntoResponse {
    let model = body.model.unwrap_or_else(|| state.model_name.clone());

    // Build ChatML prompt
    let mut prompt = String::new();
    for msg in &body.messages {
        prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content));
    }
    prompt.push_str("<|im_start|>assistant\n");

    // Tokenize
    let prompt_tokens = match state.backend.tokenize(&prompt).await {
        Ok(tokens) if !tokens.is_empty() => tokens,
        Ok(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Empty prompt after tokenization"})),
            )
                .into_response();
        }
        Err(e) => {
            error!("Tokenization error: {e}");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("Tokenization failed: {e}")})),
            )
                .into_response();
        }
    };

    // Cap max_new_tokens
    let max_new = body.max_tokens.min(256);

    // Run inference
    match state.backend.chat(&model, &prompt_tokens, max_new).await {
        Ok(resp) => Json(resp).into_response(),
        Err(e) => {
            error!("Inference error: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("Inference failed: {e}")})),
            )
                .into_response()
        }
    }
}

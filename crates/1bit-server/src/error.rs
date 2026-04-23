//! Server error type → OpenAI-shaped JSON error body.
//!
//! The C++ reference server emits errors of the form:
//! ```json
//! {"error":{"message":"...","type":"invalid_request_error","code":"context_length_exceeded"}}
//! ```
//! We reproduce that shape so existing clients (OpenAI SDK, llama-cpp
//! wrappers, Lemonade webui) don't have to special-case halo.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ServerError {
    #[error("bad request: {0}")]
    BadRequest(String),

    #[error("prompt exceeds context window ({got} >= {limit})")]
    ContextOverflow { got: usize, limit: usize },

    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("backend failure: {0}")]
    Backend(String),

    /// Upstream service (e.g. sd-server on :8081) returned an error or was
    /// unreachable. Surfaced to the client as HTTP 502 Bad Gateway so it's
    /// distinguishable from our own Backend(..) failures.
    #[error("upstream error: {0}")]
    Upstream(String),

    #[error("internal: {0}")]
    Internal(#[from] anyhow::Error),
}

impl ServerError {
    fn status_and_code(&self) -> (StatusCode, &'static str, &'static str) {
        match self {
            ServerError::BadRequest(_) => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "bad_request",
            ),
            ServerError::ContextOverflow { .. } => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "context_length_exceeded",
            ),
            ServerError::ModelNotFound(_) => (
                StatusCode::NOT_FOUND,
                "invalid_request_error",
                "model_not_found",
            ),
            ServerError::Backend(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                "backend_failure",
            ),
            ServerError::Upstream(_) => (StatusCode::BAD_GATEWAY, "api_error", "upstream_error"),
            ServerError::Internal(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                "internal_error",
            ),
        }
    }
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let (status, kind, code) = self.status_and_code();
        let body = Json(json!({
            "error": {
                "message": self.to_string(),
                "type": kind,
                "code": code,
            }
        }));
        (status, body).into_response()
    }
}

// Convenience: allow `?` on serde_json errors inside handlers.
impl From<serde_json::Error> for ServerError {
    fn from(e: serde_json::Error) -> Self {
        ServerError::BadRequest(format!("json: {e}"))
    }
}

//! HTTP/SSE app server — serves the agent runtime over HTTP.
//!
//! Provides a REST API for session management, chat completions, and tool
//! approval — matching codewhale's `serve --http` surface.  Enables IDE
//! plugins, mobile UIs, and CI systems to interact with the agent remotely.
//!
//! Endpoints:
//!   GET  /health               — health check
//!   GET  /v1/sessions          — list sessions
//!   POST /v1/sessions          — create session
//!   GET  /v1/sessions/:id      — get session
//!   GET  /v1/sessions/:id/turns — get session turns
//!   POST /v1/chat/completions   — streaming chat (SSE)

use anyhow::Result;
use axum::{
    Router,
    extract::{Path, State},
    http::{Method, StatusCode},
    routing::{get, post},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};

use crate::npu;
use crate::provider::ProviderRouter;
use crate::session::SessionStore;

// ── Shared app state ────────────────────────────────────────────

pub struct AppState {
    pub sessions: tokio::sync::Mutex<SessionStore>,
    pub npu_client: npu::NpuClient,
    pub provider_router: Arc<Mutex<ProviderRouter>>,
    pub model: String,
}

// ── API types ───────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct SessionSummary {
    id: String,
    name: String,
    model: String,
    created_at: String,
    updated_at: String,
    turn_count: u32,
    archived: bool,
}

#[derive(Debug, Serialize)]
struct TurnSummary {
    id: String,
    role: String,
    content: String,
    tool_calls: Option<String>,
    created_at: String,
}

#[derive(Debug, Deserialize)]
struct CreateSessionRequest {
    name: Option<String>,
    model: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    session_id: String,
    message: String,
    model: Option<String>,
    stream: Option<bool>,
}

#[derive(Debug, Serialize)]
struct ChatResponse {
    session_id: String,
    turn_id: String,
    content: String,
}

// ── Router ──────────────────────────────────────────────────────

pub fn build_router(
    sessions: SessionStore,
    npu_client: npu::NpuClient,
    provider_router: Arc<Mutex<ProviderRouter>>,
    model: String,
) -> Router {
    let state = Arc::new(AppState {
        sessions: tokio::sync::Mutex::new(sessions),
        npu_client,
        provider_router,
        model,
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(Any);

    Router::new()
        .route("/health", get(health))
        .route("/v1/sessions", get(list_sessions).post(create_session))
        .route("/v1/sessions/{id}", get(get_session))
        .route("/v1/sessions/{id}/turns", get(get_turns))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/v1/providers", get(list_providers))
        .layer(cors)
        .with_state(state)
}

// ── Handlers ────────────────────────────────────────────────────

async fn health() -> &'static str {
    "ok"
}

async fn list_sessions(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<SessionSummary>>, StatusCode> {
    let sessions = state.sessions.lock().await.list_sessions(false).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let summaries: Vec<SessionSummary> = sessions
        .into_iter()
        .map(|s| SessionSummary {
            id: s.id,
            name: s.name,
            model: s.model,
            created_at: s.created_at,
            updated_at: s.updated_at,
            turn_count: s.turn_count,
            archived: s.archived,
        })
        .collect();
    Ok(Json(summaries))
}

async fn create_session(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateSessionRequest>,
) -> Result<Json<SessionSummary>, StatusCode> {
    let name = req.name.unwrap_or_else(|| "API Session".to_string());
    let model = req.model.unwrap_or_else(|| state.model.clone());
    let session = state.sessions.lock().await.create_session(&name, &model).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(SessionSummary {
        id: session.id,
        name: session.name,
        model: session.model,
        created_at: session.created_at,
        updated_at: session.updated_at,
        turn_count: session.turn_count,
        archived: session.archived,
    }))
}

async fn get_session(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<SessionSummary>, StatusCode> {
    let session = state
        .sessions
        .lock().await
        .get_session(&id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;
    Ok(Json(SessionSummary {
        id: session.id,
        name: session.name,
        model: session.model,
        created_at: session.created_at,
        updated_at: session.updated_at,
        turn_count: session.turn_count,
        archived: session.archived,
    }))
}

async fn get_turns(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<Vec<TurnSummary>>, StatusCode> {
    let turns = state
        .sessions
        .lock().await
        .get_turns(&id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let summaries: Vec<TurnSummary> = turns
        .into_iter()
        .map(|t| TurnSummary {
            id: t.id,
            role: t.role,
            content: t.content,
            tool_calls: t.tool_calls,
            created_at: t.created_at,
        })
        .collect();
    Ok(Json(summaries))
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, StatusCode> {
    // Ensure session exists
    {
        let sessions = state.sessions.lock().await;
        if sessions.get_session(&req.session_id).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?.is_none() {
            return Err(StatusCode::NOT_FOUND);
        }
    }

    // Run the chat
    let model = req.model.unwrap_or_else(|| state.model.clone());

    // Record user turn
    state
        .sessions
        .lock().await
        .add_user_turn(&req.session_id, &req.message)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Get conversation history
    let history = state
        .sessions
        .lock().await
        .turns_to_messages(&req.session_id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Call NPU API
    let result = state
        .npu_client
        .chat(&model, history, None, None)
        .await
        .map_err(|e| {
            tracing::error!("Chat error: {e}");
            StatusCode::BAD_GATEWAY
        })?;

    match result {
        crate::npu::ChatResult::Text(content) => {
            let turn = state
                .sessions
                .lock().await
                .add_assistant_turn(&req.session_id, &content, None)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

            Ok(Json(ChatResponse {
                session_id: req.session_id,
                turn_id: turn.id,
                content,
            }))
        }
        crate::npu::ChatResult::ToolCalls(tcs) => {
            let tcs_json = serde_json::to_string(&tcs).unwrap_or_default();
            let content = format!("[{:?} tool call(s)]", tcs.len());
            let turn = state
                .sessions
                .lock().await
                .add_assistant_turn(&req.session_id, &content, Some(&tcs_json))
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

            Ok(Json(ChatResponse {
                session_id: req.session_id,
                turn_id: turn.id,
                content: format!("Tool calls: {}", serde_json::to_string_pretty(&tcs).unwrap_or_default()),
            }))
        }
    }
}

async fn list_models(
    State(state): State<Arc<AppState>>,
) -> Json<Vec<String>> {
    match state.npu_client.list_models().await {
        Ok(models) => Json(models),
        Err(_) => Json(vec![state.model.clone()]),
    }
}

async fn list_providers(
    State(state): State<Arc<AppState>>,
) -> Json<Vec<String>> {
    let router = state.provider_router.lock().await;
    let available = router.providers_with_keys();
    Json(available)
}

//! HTTP route handlers for the control plane daemon.

use crate::backends::{ChatMessage, GpuBackend, NpuBackend};
use crate::orders::{self, OrderItem, OrderManager};
use crate::policy::{self, Device};
use crate::stripe;
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use std::sync::Arc;
use tracing::{error, info, warn};

// ── Shared application state ───────────────────────────────────────

pub struct AppState {
    pub npu: NpuBackend,
    pub gpu: GpuBackend,
    pub orders: OrderManager,
    pub store_html: Option<String>,
    pub start_time: std::time::Instant,
    pub stripe_secret_key: String,
    pub stripe_webhook_secret: String,
}

// ── Router builder ─────────────────────────────────────────────────

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(handle_root))
        .route("/store", get(handle_store))
        .route("/store/", get(handle_store))
        .route("/v1/health", get(handle_health))
        .route("/v1/models", get(handle_models))
        .route("/v1/chat/completions", post(handle_chat_completions))
        .route("/v1/batch/completions", post(handle_batch_completions))
        .route("/api/checkout", post(handle_checkout))
        .route("/api/webhook", post(handle_webhook))
        .with_state(state)
}

// ── Helpers ────────────────────────────────────────────────────────

fn json_response(status: StatusCode, value: serde_json::Value) -> Response {
    (status, Json(value)).into_response()
}

// ── Handlers ───────────────────────────────────────────────────────

/// GET / or /store — Serve the store HTML page.
async fn handle_store(State(state): State<Arc<AppState>>) -> Response {
    match &state.store_html {
        Some(html) => Html(html.clone()).into_response(),
        None => json_response(StatusCode::NOT_FOUND, serde_json::json!({"error": "Store not found"})),
    }
}

async fn handle_root() -> Response {
    // Redirect to /store
    axum::response::Redirect::temporary("/store").into_response()
}

/// GET /v1/health — Device and backend status.
async fn handle_health(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let uptime = state.start_time.elapsed().as_secs();
    Json(serde_json::json!({
        "status": "ok",
        "uptime_sec": uptime,
        "devices": {
            "npu": {
                "backend": "C++ engine (97 tok/s, MIT)",
                "available": state.npu.is_running().await,
            },
            "gpu": {
                "backend": "ROCm/WMMA GPU engine",
                "available": state.gpu.is_running().await,
            },
            "cpu": {
                "backend": "not implemented",
                "available": false,
            },
        },
        "policy": {
            "< 2B params": "npu",
            ">= 2B params": "gpu",
            "> 8B params": "gpu (no CPU backend yet — see #147)",
        },
    }))
}

/// GET /v1/models — List available models.
async fn handle_models() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [
            {"id": "Qwen3-0.6B-NPU2", "object": "model", "owned_by": "npu"},
            {"id": "Qwen3-8B-NPU2", "object": "model", "owned_by": "npu"},
            {"id": "Llama-3.1-8B-NPU2", "object": "model", "owned_by": "npu"},
            {"id": "Gemma4-E2B-IT-NPU2", "object": "model", "owned_by": "npu"},
        ]
    }))
}

/// POST /v1/chat/completions — Route inference to NPU or GPU.
async fn handle_chat_completions(
    State(state): State<Arc<AppState>>,
    _headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Response {
    let model = body.get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let messages: Vec<ChatMessage> = body.get("messages")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();
    let _stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let max_tokens = body.get("max_tokens").and_then(|v| v.as_u64()).map(|v| v as u32);
    let temperature = body.get("temperature").and_then(|v| v.as_f64()).map(|v| v as f32);

    // Collect extra kwargs (everything except model, messages, stream)
    let extra_keys = ["model", "messages", "stream", "max_tokens", "temperature"];
    let mut extra = serde_json::Map::new();
    if let Some(obj) = body.as_object() {
        for (k, v) in obj {
            if !extra_keys.contains(&k.as_str()) {
                extra.insert(k.clone(), v.clone());
            }
        }
    }

    info!(
        "REQ model={model:?} msgs={} stream={_stream} extra_keys={:?}",
        messages.len(),
        extra.keys().collect::<Vec<_>>(),
    );

    // Determine device
    let (device, model_size) = if model.starts_with("npu://") {
        (Device::Npu, 0.6)
    } else {
        let size = policy::estimate_model_size(&model);
        let dev = policy::select_device(size);
        (dev, size)
    };

    // TODO: streaming support — for now return non-streaming
    let _ = _stream;

    match device {
        Device::Npu => {
            match state.npu.chat(&model, &messages, max_tokens, temperature).await {
                Ok(mut resp) => {
                    if let Some(obj) = resp.as_object_mut() {
                        obj.insert("x-device".to_string(), serde_json::json!("npu"));
                        obj.insert("x-model-size".to_string(), serde_json::json!(format!("{model_size}B")));
                    }
                    json_response(StatusCode::OK, resp)
                }
                Err(e) => {
                    error!("NPU inference error: {e}");
                    json_response(
                        StatusCode::BAD_GATEWAY,
                        serde_json::json!({"error": e.to_string(), "x-device": "npu"}),
                    )
                }
            }
        }
        Device::Gpu => {
            match state.gpu.chat(&model, &messages, &extra).await {
                Ok(mut resp) => {
                    if let Some(obj) = resp.as_object_mut() {
                        obj.insert("x-device".to_string(), serde_json::json!("gpu"));
                        obj.insert("x-model-size".to_string(), serde_json::json!(format!("{model_size}B")));
                    }
                    json_response(StatusCode::OK, resp)
                }
                Err(e) => {
                    error!("GPU inference error: {e}");
                    json_response(
                        StatusCode::BAD_GATEWAY,
                        serde_json::json!({"error": e.to_string(), "x-device": "gpu"}),
                    )
                }
            }
        }
    }
}

/// POST /v1/batch/completions — Batch prefill on NPU.
async fn handle_batch_completions(
    State(_state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> Response {
    // Parse tokens
    let tokens_val = match body.get("tokens") {
        Some(v) => v,
        None => return json_response(StatusCode::BAD_REQUEST, serde_json::json!({"error": "No tokens in request"})),
    };

    let tokens: Vec<u32> = match tokens_val.as_array() {
        Some(arr) => {
            let mut out = Vec::new();
            for v in arr {
                match v.as_u64() {
                    Some(n) if n <= u64::from(u32::MAX) => out.push(n as u32),
                    _ => return json_response(StatusCode::BAD_REQUEST, serde_json::json!({"error": "Invalid token value"})),
                }
            }
            out
        }
        None => return json_response(StatusCode::BAD_REQUEST, serde_json::json!({"error": "tokens must be an array"})),
    };

    if tokens.is_empty() {
        return json_response(StatusCode::BAD_REQUEST, serde_json::json!({"error": "No tokens in request"}));
    }
    if tokens.len() > 256 {
        return json_response(StatusCode::BAD_REQUEST, serde_json::json!({"error": "Max 256 tokens"}));
    }

    let model_path = body.get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx");

    // Build engine path relative to home
    let engine = std::path::PathBuf::from(
        std::env::var("HOME").unwrap_or_else(|_| "/home/bcloud".to_string())
    ).join("npu-sandbox/npu-infer/build/npu_engine_mt");

    if !engine.exists() {
        return json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            serde_json::json!({"error": "npu_engine_mt not found"}),
        );
    }

    // Build LD_LIBRARY_PATH
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/bcloud".to_string());
    let ld_path = format!(
        "{home}/torch2aie/toolchain/mlir_aie.libs:{home}/torch2aie/toolchain/sysroot/usr/lib64:{}",
        std::env::var("LD_LIBRARY_PATH").unwrap_or_default()
    );

    // Run npu_engine_mt as subprocess
    let mut cmd = tokio::process::Command::new(&engine);
    cmd.arg(model_path);
    for t in &tokens {
        cmd.arg(t.to_string());
    }
    cmd.env("LD_LIBRARY_PATH", &ld_path);
    cmd.stderr(std::process::Stdio::piped());

    match cmd.output().await {
        Ok(output) => {
            let stderr_str = String::from_utf8_lossy(&output.stderr);
            let mut tokens_out: Vec<serde_json::Value> = Vec::new();
            let mut total_ms = 0.0f64;

            for line in stderr_str.lines() {
                if line.contains("===") {
                    // Parse timing info
                    for part in line.split(',') {
                        let part = part.trim();
                        if part.ends_with("ms") {
                            if let Some(num) = part.split_whitespace().next() {
                                if let Ok(ms) = num.trim_end_matches("ms").parse::<f64>() {
                                    total_ms = ms;
                                }
                            }
                        }
                    }
                }
                // Parse token output: "token N ... top8=[...]"
                if let Some(caps) = parse_token_line(line) {
                    tokens_out.push(caps);
                }
            }

            if !total_ms.is_finite() {
                total_ms = 0.0;
            }

            Json(serde_json::json!({
                "object": "batch.completion",
                "tokens": tokens_out,
                "total_ms": total_ms,
                "ms_per_token": if tokens.is_empty() { 0.0 } else { total_ms / tokens.len() as f64 },
                "x-tokens": tokens.len(),
                "x-device": "npu",
            }))
            .into_response()
        }
        Err(e) => {
            error!("npu_engine_mt execution failed: {e}");
            json_response(
                StatusCode::BAD_GATEWAY,
                serde_json::json!({"error": format!("Engine execution failed: {e}")}),
            )
        }
    }
}

fn parse_token_line(line: &str) -> Option<serde_json::Value> {
    // Match: token N ... top8=[v1,v2,...]
    // Simple string parsing instead of regex to avoid the dependency.
    if !line.contains("token") || !line.contains("top8=[") {
        return None;
    }
    // Extract token index
    let after_token = line.find("token")? + 5;
    let rest = line[after_token..].trim_start();
    let index_end = rest.find(|c: char| !c.is_ascii_digit())?;
    let index: u32 = rest[..index_end].parse().ok()?;
    // Extract top8 array
    let top8_start = line.find("top8=[")? + 6;
    let top8_end = line[top8_start..].find(']')?;
    let top8_str = &line[top8_start..top8_start + top8_end];
    let top8: Vec<u32> = top8_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    Some(serde_json::json!({"index": index, "top8": top8}))
}

/// POST /api/checkout — Create a Stripe Checkout Session.
async fn handle_checkout(
    State(state): State<Arc<AppState>>,
    Json(req): Json<orders::CheckoutRequest>,
) -> Response {
    if req.items.is_empty() {
        return json_response(StatusCode::BAD_REQUEST, serde_json::json!({"error": "Cart is empty"}));
    }

    if state.stripe_secret_key.is_empty() {
        return json_response(
            StatusCode::SERVICE_UNAVAILABLE,
            serde_json::json!({"error": "Stripe not configured. Set STRIPE_SECRET_KEY env var."}),
        );
    }

    match stripe::create_checkout_session(
        &state.stripe_secret_key,
        &req.items,
        &req.success_url,
        &req.cancel_url,
    )
    .await
    {
        Ok(resp) => {
            if let Some(ref err) = resp.error {
                return json_response(StatusCode::BAD_REQUEST, serde_json::json!({"error": err}));
            }

            // Store pending order
            if let Some(ref sid) = resp.id {
                let order = orders::Order {
                    items: req
                        .items
                        .iter()
                        .map(|item| OrderItem {
                            product: item
                                .price_data
                                .as_ref()
                                .and_then(|pd| pd.product_data.as_ref())
                                .and_then(|p| p.name.clone())
                                .unwrap_or_else(|| "Merch".to_string()),
                            size: item.size.clone(),
                            qty: item.quantity.unwrap_or(1),
                            price: item
                                .price_data
                                .as_ref()
                                .and_then(|pd| pd.unit_amount)
                                .unwrap_or(0),
                        })
                        .collect(),
                    total: req
                        .items
                        .iter()
                        .map(|item| {
                            item.price_data
                                .as_ref()
                                .and_then(|pd| pd.unit_amount)
                                .unwrap_or(0)
                                * item.quantity.unwrap_or(1) as u64
                        })
                        .sum(),
                    customer_name: String::new(),
                    customer_email: String::new(),
                    shipping_address: String::new(),
                    stripe_session_id: sid.clone(),
                    logged_at: String::new(),
                };
                if let Err(e) = state.orders.add_pending(sid, order).await {
                    warn!("Failed to save pending order: {e}");
                }
            }

            Json(resp).into_response()
        }
        Err(e) => json_response(StatusCode::BAD_REQUEST, serde_json::json!({"error": e.to_string()})),
    }
}

/// POST /api/webhook — Stripe webhook event handler.
async fn handle_webhook(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let sig_header = headers
        .get("Stripe-Signature")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    let result = stripe::handle_stripe_webhook(
        &body,
        sig_header,
        &state.stripe_webhook_secret,
    )
    .await;

    match result {
        Ok(wh) => {
            if wh.status >= 400 {
                return json_response(
                    StatusCode::from_u16(wh.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                    serde_json::json!({"error": wh.error}),
                );
            }

            // If checkout.session.completed, fulfill the pending order
            if let Some(sid) = wh.session_id {
                if let Some(mut order) = state.orders.take_pending(&sid).await {
                    // Enrich with Stripe customer details (from the parsed webhook body)
                    if let Ok(event) = serde_json::from_slice::<serde_json::Value>(&body) {
                        let session = &event["data"]["object"];
                        let customer = &session["customer_details"];
                        let shipping = &session["shipping_details"];
                        let addr = shipping.get("address");

                        order.customer_name = customer
                            .get("name")
                            .and_then(|v| v.as_str())
                            .or_else(|| shipping.get("name").and_then(|v| v.as_str()))
                            .unwrap_or("?")
                            .to_string();
                        order.customer_email = customer
                            .get("email")
                            .and_then(|v| v.as_str())
                            .unwrap_or("?")
                            .to_string();
                        order.shipping_address = addr.map(|a| {
                            let parts = [
                                shipping.get("name").and_then(|v| v.as_str()).unwrap_or(""),
                                a.get("line1").and_then(|v| v.as_str()).unwrap_or(""),
                                a.get("city").and_then(|v| v.as_str()).unwrap_or(""),
                                a.get("state").and_then(|v| v.as_str()).unwrap_or(""),
                                a.get("country").and_then(|v| v.as_str()).unwrap_or(""),
                            ];
                            parts.join(" ").trim().to_string()
                        }).unwrap_or_default();
                        order.stripe_session_id = sid.clone();
                    }

                    if let Err(e) = state.orders.fulfill_order(&order).await {
                        warn!("Order fulfillment failed: {e}");
                    }
                } else {
                    warn!("No pending order found for session {sid}");
                }
            }

            json_response(
                StatusCode::OK,
                serde_json::json!({
                    "ok": wh.ok,
                    "skipped": wh.skipped,
                }),
            )
        }
        Err(e) => {
            error!("Webhook processing error: {e}");
            json_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                serde_json::json!({"error": e.to_string()}),
            )
        }
    }
}

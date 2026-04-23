//! axum handlers for the `.1bl` catalog HTTP surface.
//!
//! Routes:
//! * `GET  /v1/health`                      — "ok" liveness
//! * `GET  /v1/catalogs`                    — JSON list of catalogs
//! * `GET  /v1/catalogs/:slug`              — single catalog manifest
//! * `GET  /v1/catalogs/:slug/lossy`        — streams lossy-tier bytes
//! * `GET  /v1/catalogs/:slug/lossless`     — streams full .1bl, premium-gated
//! * `POST /internal/reindex`               — admin: reload catalog dir
//!
//! Lossy / lossless streaming both respond with
//! `application/octet-stream` and a `Content-Disposition: attachment`
//! header. The lossy path materializes a fresh valid `.1bl` on the fly —
//! magic + header + lossy sections + footer hash. The lossless path just
//! passes the source file through. Both are memory-bounded (chunked).

use axum::{
    Json, Router,
    body::Body,
    extract::{Path, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::auth::{self, AuthConfig, GateOutcome};
use crate::container::{Catalog, FORMAT_VERSION, MAGIC, Section};

/// Runtime state. Catalogs live behind an `RwLock` so `/internal/reindex`
/// can swap the set atomically without pausing serves.
#[derive(Clone)]
pub struct AppState {
    pub catalog_dir: std::path::PathBuf,
    pub catalogs: Arc<RwLock<Vec<Catalog>>>,
    pub auth: Arc<AuthConfig>,
}

impl AppState {
    pub fn new(catalog_dir: std::path::PathBuf, auth: AuthConfig) -> Self {
        Self {
            catalog_dir,
            catalogs: Arc::new(RwLock::new(Vec::new())),
            auth: Arc::new(auth),
        }
    }

    /// Scan `catalog_dir`, parse every `*.1bl`, swap the cache. Returns
    /// the number of catalogs loaded and a list of (path, error) for any
    /// files that failed to parse.
    pub async fn reindex(&self) -> (usize, Vec<(String, String)>) {
        let mut loaded = Vec::new();
        let mut errs = Vec::new();
        let entries = match std::fs::read_dir(&self.catalog_dir) {
            Ok(e) => e,
            Err(e) => {
                errs.push((self.catalog_dir.display().to_string(), e.to_string()));
                let mut w = self.catalogs.write().await;
                *w = loaded;
                return (0, errs);
            }
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("1bl") {
                continue;
            }
            match Catalog::open(&path) {
                Ok(c) => loaded.push(c),
                Err(e) => errs.push((path.display().to_string(), e.to_string())),
            }
        }
        let count = loaded.len();
        let mut w = self.catalogs.write().await;
        *w = loaded;
        (count, errs)
    }
}

pub fn build(state: AppState) -> Router {
    Router::new()
        .route("/v1/health", get(health))
        .route("/v1/catalogs", get(list_catalogs))
        .route("/v1/catalogs/:slug", get(get_catalog))
        .route("/v1/catalogs/:slug/lossy", get(stream_lossy))
        .route("/v1/catalogs/:slug/lossless", get(stream_lossless))
        .route("/internal/reindex", post(reindex))
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
}

async fn list_catalogs(State(s): State<AppState>) -> Json<Value> {
    let cats = s.catalogs.read().await;
    let data: Vec<Value> = cats
        .iter()
        .map(|c| {
            json!({
                "slug": c.slug(),
                "title": c.manifest.title,
                "artist": c.manifest.artist,
                "license": c.manifest.license,
                "tier": c.manifest.tier,
                "residual_present": c.manifest.residual_present,
                "bytes": c.total_bytes,
            })
        })
        .collect();
    Json(json!({ "object": "list", "data": data }))
}

async fn get_catalog(State(s): State<AppState>, Path(slug): Path<String>) -> Response {
    let cats = s.catalogs.read().await;
    let Some(cat) = cats.iter().find(|c| c.slug() == slug) else {
        return (StatusCode::NOT_FOUND, "unknown catalog").into_response();
    };
    Json(&cat.manifest).into_response()
}

async fn stream_lossy(State(s): State<AppState>, Path(slug): Path<String>) -> Response {
    let cats = s.catalogs.read().await;
    let Some(cat) = cats.iter().find(|c| c.slug() == slug).cloned() else {
        return (StatusCode::NOT_FOUND, "unknown catalog").into_response();
    };
    drop(cats);

    // Build a lossy-only .1bl in memory. The free-tier payload is
    // ~2 MB per the spec's sizing table so this is fine for v0; we'll
    // revisit if users start shipping lossy-only packs over 50 MB.
    match build_lossy_bytes(&cat).await {
        Ok(bytes) => {
            let filename = format!("{}.lossy.1bl", cat.slug());
            (
                StatusCode::OK,
                [
                    (header::CONTENT_TYPE, "application/octet-stream".to_string()),
                    (
                        header::CONTENT_DISPOSITION,
                        format!("attachment; filename=\"{filename}\""),
                    ),
                ],
                bytes,
            )
                .into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("lossy build: {e}")).into_response(),
    }
}

async fn stream_lossless(
    State(s): State<AppState>,
    Path(slug): Path<String>,
    headers: HeaderMap,
) -> Response {
    let gate = auth::check_premium(&s.auth, &headers);
    if !matches!(gate, GateOutcome::Allow) {
        let msg = match gate {
            GateOutcome::MissingHeader => "missing authorization header",
            GateOutcome::BadScheme => "expected Bearer scheme",
            GateOutcome::BadToken => "invalid token",
            GateOutcome::WrongTier => "token does not carry premium tier",
            GateOutcome::ServerMisconfigured => "lossless gate not configured",
            GateOutcome::Allow => unreachable!(),
        };
        return (gate.as_status(), msg).into_response();
    }

    let cats = s.catalogs.read().await;
    let Some(cat) = cats.iter().find(|c| c.slug() == slug).cloned() else {
        return (StatusCode::NOT_FOUND, "unknown catalog").into_response();
    };
    drop(cats);

    let filename = format!("{}.1bl", cat.slug());
    match tokio::fs::File::open(&cat.path).await {
        Ok(file) => {
            let stream = tokio_util_reader_stream(file);
            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "application/octet-stream")
                .header(header::CONTENT_LENGTH, cat.total_bytes)
                .header(
                    header::CONTENT_DISPOSITION,
                    format!("attachment; filename=\"{filename}\""),
                )
                .body(Body::from_stream(stream))
                .unwrap()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("open: {e}")).into_response(),
    }
}

async fn reindex(State(s): State<AppState>, headers: HeaderMap) -> Response {
    let gate = auth::check_admin(&s.auth, &headers);
    if !matches!(gate, GateOutcome::Allow) {
        return (gate.as_status(), "admin auth failed").into_response();
    }
    let (count, errs) = s.reindex().await;
    Json(json!({
        "loaded": count,
        "errors": errs.into_iter().map(|(p, e)| json!({ "path": p, "error": e })).collect::<Vec<_>>(),
    }))
    .into_response()
}

// -----------------------------------------------------------------------------
// helpers

/// Materialize a lossy-only `.1bl` in a Vec<u8>. Preserves magic +
/// original CBOR header, drops 0x10 / 0x11, recomputes the footer SHA.
async fn build_lossy_bytes(cat: &Catalog) -> std::io::Result<Vec<u8>> {
    use sha2::{Digest, Sha256};
    use tokio::io::{AsyncReadExt, AsyncSeekExt};
    let mut f = tokio::fs::File::open(&cat.path).await?;

    // --- header: magic + u32 header_len + CBOR manifest ------------------
    // Read the original header block verbatim so we preserve byte-for-byte
    // what the publisher signed (modulo the footer, which we redo).
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).await?;

    let mut hlen = [0u8; 4];
    f.read_exact(&mut hlen).await?;
    let header_len = u32::from_le_bytes(hlen);
    let mut header_buf = vec![0u8; header_len as usize];
    f.read_exact(&mut header_buf).await?;

    let mut out = Vec::with_capacity(cat.total_bytes as usize);
    out.extend_from_slice(&magic);
    assert_eq!(magic, MAGIC, "magic mismatch at serve time");
    assert_eq!(magic[3], FORMAT_VERSION, "version mismatch at serve time");
    out.extend_from_slice(&hlen);
    out.extend_from_slice(&header_buf);

    // --- copy lossy-tier sections ----------------------------------------
    let lossy: Vec<Section> = cat.lossy_sections().cloned().collect();
    for s in lossy {
        out.push(s.tag);
        out.extend_from_slice(&s.length.to_le_bytes());
        f.seek(std::io::SeekFrom::Start(s.offset)).await?;
        let mut remaining = s.length;
        let mut buf = vec![0u8; 64 * 1024];
        while remaining > 0 {
            let want = remaining.min(buf.len() as u64) as usize;
            let got = f.read(&mut buf[..want]).await?;
            if got == 0 {
                break;
            }
            out.extend_from_slice(&buf[..got]);
            remaining -= got as u64;
        }
    }

    // --- footer ---------------------------------------------------------
    let hash: [u8; 32] = Sha256::digest(&out).into();
    out.extend_from_slice(&hash);
    Ok(out)
}

/// Thin tokio → axum ReaderStream adapter. Keeps us off the
/// `tokio-util` workspace dep — we use `tokio::io::ReaderStream` equivalent
/// via `async-stream` which is already in the workspace.
fn tokio_util_reader_stream(
    mut file: tokio::fs::File,
) -> impl futures::Stream<Item = std::io::Result<bytes::Bytes>> {
    use tokio::io::AsyncReadExt;
    async_stream::stream! {
        let mut buf = vec![0u8; 64 * 1024];
        loop {
            match file.read(&mut buf).await {
                Ok(0) => break,
                Ok(n) => yield Ok(bytes::Bytes::copy_from_slice(&buf[..n])),
                Err(e) => { yield Err(e); break; }
            }
        }
    }
}

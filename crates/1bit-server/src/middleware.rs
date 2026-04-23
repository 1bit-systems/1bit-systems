//! Axum middleware: request-id injection + per-IP rate limiting.
//!
//! Wired in from [`crate::routes::build_router_with_state`]:
//!
//! * [`request_id`] — applied globally via `axum::middleware::from_fn`.
//!   Reads any inbound `x-request-id` (preserved for trace correlation when
//!   Caddy already stamped one), or mints a fresh UUID. The id is stored as
//!   a request extension so downstream layers / handlers can log against it
//!   and copied back onto the response headers so the client sees it.
//!
//! * [`rate_limit`] — applied only to `/v{1,2}/chat/completions` via
//!   `route_layer(from_fn_with_state(...))`. Token-bucket per client IP:
//!   capacity = rpm, refill = rpm/60 tokens per second, computed lazily on
//!   each request from monotonic wall-clock. Cost is 1.0 / request. Under
//!   starvation we emit `429 Too Many Requests` with an OpenAI-shaped error
//!   body and a `Retry-After` header (seconds, rounded up). The map is
//!   [`DashMap`]-backed and swept every 10 minutes to evict idle IPs.
//!
//! Keeping these in one module avoids adding yet another top-level file for
//! 200 lines of plumbing. Unit tests live at the bottom; HTTP-level tests
//! live under `tests/middleware.rs` + `tests/middleware_tracing.rs`.

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Json;
use axum::body::Body;
use axum::extract::{ConnectInfo, Request, State};
use axum::http::{HeaderName, HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use dashmap::DashMap;
use serde_json::json;
use uuid::Uuid;

/// Header name we stamp on both the request extension and the outgoing
/// response. Matches Caddy's `{http.request.uuid}` convention so a request
/// flowing Client → Caddy → 1bit-server → logs can be traced end-to-end.
pub const REQUEST_ID_HEADER: &str = "x-request-id";

/// Stored as a request extension for handlers / log layers to pick up.
#[derive(Debug, Clone)]
pub struct RequestId(pub String);

/// Axum middleware: mint or preserve `x-request-id`, store as extension,
/// echo on the response.
pub async fn request_id(mut req: Request<Body>, next: Next) -> Response {
    let id = req
        .headers()
        .get(REQUEST_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty() && s.len() <= 128)
        .unwrap_or_else(|| Uuid::new_v4().simple().to_string());

    // Make the id visible to handlers + downstream layers.
    req.extensions_mut().insert(RequestId(id.clone()));

    let mut resp = next.run(req).await;
    if let Ok(hv) = HeaderValue::from_str(&id) {
        resp.headers_mut()
            .insert(HeaderName::from_static(REQUEST_ID_HEADER), hv);
    }
    resp
}

// ─── Rate limiter ────────────────────────────────────────────────────────

/// Per-IP token bucket.
///
/// `tokens` is `f64` so partial refill across sub-second intervals is exact
/// (an `u32` would force us to round down and under-count allowed
/// requests). `last_refill` is a monotonic [`Instant`] — we never want a
/// wall-clock jump to hand out a burst of tokens.
#[derive(Debug, Clone, Copy)]
struct Bucket {
    tokens: f64,
    last_refill: Instant,
    last_seen: Instant,
}

/// Token-bucket rate limiter, keyed by client IP.
///
/// * `rpm` = capacity (max burst), 0 disables the limiter entirely.
/// * refill rate = `rpm / 60` tokens / second.
/// * Buckets older than [`Self::EVICT_AFTER`] are swept out on each
///   `check()` call, at most once per [`Self::SWEEP_INTERVAL`].
#[derive(Debug)]
pub struct RateLimit {
    rpm: u32,
    /// Pre-computed for the hot path: `rpm as f64 / 60.0`.
    refill_per_sec: f64,
    buckets: DashMap<IpAddr, Bucket>,
    last_sweep: parking_lot_like::Atomic<Instant>,
}

impl RateLimit {
    /// Idle buckets this old are evicted on the next sweep.
    const EVICT_AFTER: Duration = Duration::from_secs(10 * 60);
    /// Don't scan the entire map more than once every 10 minutes.
    const SWEEP_INTERVAL: Duration = Duration::from_secs(10 * 60);

    /// Construct a limiter with the given requests-per-minute ceiling.
    /// `rpm = 0` disables the limiter (every `check()` returns `Ok`).
    pub fn new(rpm: u32) -> Self {
        Self {
            rpm,
            refill_per_sec: rpm as f64 / 60.0,
            buckets: DashMap::new(),
            last_sweep: parking_lot_like::Atomic::new(Instant::now()),
        }
    }

    /// True when the limiter is off — caller can skip the IP-extraction
    /// path. Exposed for the middleware fast-path.
    pub fn disabled(&self) -> bool {
        self.rpm == 0
    }

    /// Configured cap (requests per 60 s).
    pub fn rpm(&self) -> u32 {
        self.rpm
    }

    /// Current bucket-map size. Exposed for tests + /metrics if we ever
    /// want to publish it.
    pub fn active_ips(&self) -> usize {
        self.buckets.len()
    }

    /// Consume one token for `ip`. Returns [`Ok`] on allow, [`Err`] with the
    /// suggested `Retry-After` duration on deny.
    pub fn check(&self, ip: IpAddr) -> Result<(), Duration> {
        if self.disabled() {
            return Ok(());
        }
        self.check_with_now(ip, Instant::now())
    }

    /// `check()` with an injectable `now` — drives the unit tests without a
    /// `tokio::time::sleep` race.
    fn check_with_now(&self, ip: IpAddr, now: Instant) -> Result<(), Duration> {
        // Lazy sweep: at most every SWEEP_INTERVAL, walk the map and evict
        // anything stale. Kept inside check() so we don't need a background
        // task (and therefore don't need to wire a cancel token for
        // graceful shutdown).
        self.maybe_sweep(now);

        let capacity = self.rpm as f64;
        let refill = self.refill_per_sec;

        let mut entry = self.buckets.entry(ip).or_insert(Bucket {
            tokens: capacity,
            last_refill: now,
            last_seen: now,
        });

        // Refill based on elapsed time since last check. Saturates at `capacity`.
        let elapsed = now
            .saturating_duration_since(entry.last_refill)
            .as_secs_f64();
        entry.tokens = (entry.tokens + elapsed * refill).min(capacity);
        entry.last_refill = now;
        entry.last_seen = now;

        if entry.tokens >= 1.0 {
            entry.tokens -= 1.0;
            Ok(())
        } else {
            // Deficit → time-to-one-token. refill > 0 because rpm > 0 (we
            // returned early when disabled). Round up so we never suggest a
            // retry that would still 429.
            let deficit = 1.0 - entry.tokens;
            let secs = (deficit / refill).ceil().max(1.0) as u64;
            Err(Duration::from_secs(secs))
        }
    }

    fn maybe_sweep(&self, now: Instant) {
        let prev = self.last_sweep.load();
        if now.saturating_duration_since(prev) < Self::SWEEP_INTERVAL {
            return;
        }
        if !self.last_sweep.compare_exchange(prev, now) {
            // Another thread is sweeping.
            return;
        }
        self.buckets
            .retain(|_, b| now.saturating_duration_since(b.last_seen) < Self::EVICT_AFTER);
    }
}

/// Tiny local shim around `parking_lot::Mutex<Instant>` — we only need
/// load + CAS-style swap and didn't want to pull parking_lot into the
/// dependency graph for one atomic. std `Mutex<Instant>` is fine because
/// the critical section is ~5 ns.
mod parking_lot_like {
    use std::sync::Mutex;
    use std::time::Instant;

    #[derive(Debug)]
    pub struct Atomic<T>(Mutex<T>);

    impl Atomic<Instant> {
        pub fn new(v: Instant) -> Self {
            Self(Mutex::new(v))
        }
        pub fn load(&self) -> Instant {
            *self.0.lock().unwrap()
        }
        /// Set to `new` only if current equals `cur`. Returns true on swap.
        pub fn compare_exchange(&self, cur: Instant, new: Instant) -> bool {
            let mut g = self.0.lock().unwrap();
            if *g == cur {
                *g = new;
                true
            } else {
                false
            }
        }
    }
}

/// Axum middleware that consults the shared [`RateLimit`].
///
/// Expects `axum::serve(..).into_make_service_with_connect_info::<SocketAddr>()`
/// at the bind site so that [`ConnectInfo<SocketAddr>`] is available. If the
/// extractor misses (which only happens when a test or embedded caller
/// forgot the `with_connect_info` wiring), we fall through to allow — a
/// rate-limiter that rejects on an internal wiring bug would be worse than
/// one that silently lets traffic through.
pub async fn rate_limit(
    State(limiter): State<Arc<RateLimit>>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    req: Request<Body>,
    next: Next,
) -> Response {
    if limiter.disabled() {
        return next.run(req).await;
    }
    let Some(ConnectInfo(addr)) = connect_info else {
        tracing::debug!("rate_limit: no ConnectInfo — skipping");
        return next.run(req).await;
    };
    match limiter.check(addr.ip()) {
        Ok(()) => next.run(req).await,
        Err(retry_after) => too_many_requests(retry_after),
    }
}

/// OpenAI-shaped 429 with `Retry-After: <seconds>`.
fn too_many_requests(retry_after: Duration) -> Response {
    let secs = retry_after.as_secs().max(1);
    let body = Json(json!({
        "error": {
            "message": format!(
                "rate limit exceeded; retry in {secs}s"
            ),
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded",
        }
    }));
    let mut resp = (StatusCode::TOO_MANY_REQUESTS, body).into_response();
    if let Ok(hv) = HeaderValue::from_str(&secs.to_string()) {
        resp.headers_mut().insert("retry-after", hv);
    }
    resp
}

// ─── Unit tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    fn ip(n: u8) -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(10, 0, 0, n))
    }

    #[test]
    fn disabled_rpm_zero_always_allows() {
        let rl = RateLimit::new(0);
        assert!(rl.disabled());
        for _ in 0..1000 {
            assert!(rl.check(ip(1)).is_ok());
        }
    }

    #[test]
    fn bucket_allows_up_to_capacity_then_denies() {
        let rl = RateLimit::new(5);
        let t0 = Instant::now();
        for i in 0..5 {
            assert!(
                rl.check_with_now(ip(2), t0).is_ok(),
                "token #{i} should allow"
            );
        }
        // Sixth call within the same instant must 429.
        let err = rl.check_with_now(ip(2), t0).expect_err("should 429");
        assert!(err.as_secs() >= 1, "retry-after should be >= 1s");
    }

    #[test]
    fn bucket_refills_with_time() {
        let rl = RateLimit::new(60); // 1 tok/s
        let t0 = Instant::now();
        // Drain all 60 tokens.
        for _ in 0..60 {
            assert!(rl.check_with_now(ip(3), t0).is_ok());
        }
        assert!(rl.check_with_now(ip(3), t0).is_err());
        // 2 s later → 2 more tokens available.
        let t1 = t0 + Duration::from_secs(2);
        assert!(rl.check_with_now(ip(3), t1).is_ok());
        assert!(rl.check_with_now(ip(3), t1).is_ok());
        assert!(
            rl.check_with_now(ip(3), t1).is_err(),
            "third call at t+2s should 429 (only 2 tokens refilled)"
        );
    }

    #[test]
    fn buckets_are_per_ip() {
        let rl = RateLimit::new(2);
        let t0 = Instant::now();
        assert!(rl.check_with_now(ip(4), t0).is_ok());
        assert!(rl.check_with_now(ip(4), t0).is_ok());
        assert!(rl.check_with_now(ip(4), t0).is_err());
        // Different IP — fresh bucket.
        assert!(rl.check_with_now(ip(5), t0).is_ok());
        assert!(rl.check_with_now(ip(5), t0).is_ok());
        assert!(rl.check_with_now(ip(5), t0).is_err());
        assert_eq!(rl.active_ips(), 2);
    }

    #[test]
    fn sweep_evicts_idle_buckets() {
        let rl = RateLimit::new(10);
        let t0 = Instant::now();
        // Populate a handful of IPs.
        for n in 0..5 {
            assert!(rl.check_with_now(ip(n + 10), t0).is_ok());
        }
        assert_eq!(rl.active_ips(), 5);
        // Jump forward past EVICT_AFTER + SWEEP_INTERVAL.
        let t1 = t0 + Duration::from_secs(25 * 60);
        // One more hit forces the sweep. Use a novel IP so we can tell the
        // difference between "swept + re-added" and "never touched".
        assert!(rl.check_with_now(ip(99), t1).is_ok());
        assert_eq!(
            rl.active_ips(),
            1,
            "idle IPs should have been evicted, leaving only the new one"
        );
    }
}

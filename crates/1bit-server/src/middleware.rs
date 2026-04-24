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

/// Header the XFF-aware rate-limiter reads when `HALO_TRUST_XFF=1`.
///
/// Caddy stamps this in the strixhalo edge config (see
/// `strixhalo/caddy/Caddyfile`). When the server is exposed directly we
/// keep `HALO_TRUST_XFF` unset so nobody can spoof `X-Forwarded-For` into
/// a different bucket.
pub const FORWARDED_FOR_HEADER: &str = "x-forwarded-for";

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
///
/// **XFF note.** When `HALO_TRUST_XFF=1` is set in the environment and the
/// request carries an `X-Forwarded-For` header, we use the *left-most*
/// parseable IP from that header as the bucket key instead of the peer
/// address. This is the canonical "real client IP" shape for the Caddy-
/// fronted strixhalo deploy (Caddy prepends the trust-chain). The env
/// flag is opt-in on purpose — a direct-exposed server must keep it
/// unset so attackers can't spoof into a different bucket.
pub async fn rate_limit(
    State(limiter): State<Arc<RateLimit>>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    req: Request<Body>,
    next: Next,
) -> Response {
    if limiter.disabled() {
        return next.run(req).await;
    }
    let peer_ip = connect_info.as_ref().map(|ConnectInfo(a)| a.ip());
    let xff = req
        .headers()
        .get(FORWARDED_FOR_HEADER)
        .and_then(|h| h.to_str().ok());
    let Some(key_ip) = resolve_client_ip(peer_ip, xff, trust_xff_enabled()) else {
        tracing::debug!("rate_limit: no IP available (no ConnectInfo, no usable XFF) — skipping");
        return next.run(req).await;
    };
    match limiter.check(key_ip) {
        Ok(()) => next.run(req).await,
        Err(retry_after) => too_many_requests(retry_after),
    }
}

/// True iff `HALO_TRUST_XFF=1` (any other value, unset, or parse error
/// means "don't trust"). Split out so the unit tests can drive the logic
/// without touching the process environment.
fn trust_xff_enabled() -> bool {
    std::env::var("HALO_TRUST_XFF")
        .map(|v| v.trim() == "1")
        .unwrap_or(false)
}

/// Pick the bucket-key IP for a given request.
///
/// * `peer_ip` — the TCP peer, from `ConnectInfo<SocketAddr>`.
/// * `xff` — raw `X-Forwarded-For` header value (comma-separated).
/// * `trust_xff` — whether to honor XFF. Only ever `true` when the
///   operator set `HALO_TRUST_XFF=1`.
///
/// Precedence when `trust_xff` is on: left-most parseable IP from XFF,
/// then fall back to peer. When `trust_xff` is off we ignore XFF
/// entirely. Returns `None` only when both sources are absent / unusable
/// — caller is expected to pass the request through in that case (a
/// misconfigured limiter rejecting on internal wiring would be worse
/// than one that lets a handful of requests slip).
pub(crate) fn resolve_client_ip(
    peer_ip: Option<IpAddr>,
    xff: Option<&str>,
    trust_xff: bool,
) -> Option<IpAddr> {
    if trust_xff
        && let Some(raw) = xff
        && let Some(ip) = leftmost_xff_ip(raw)
    {
        return Some(ip);
    }
    peer_ip
}

/// Parse the left-most valid IP from an `X-Forwarded-For` value.
///
/// Accepts IPv4 + IPv6 (RFC-7239 allows bracketed `[…]:port` for v6;
/// we tolerate that too). Whitespace around each entry is trimmed, and
/// an entry with a trailing `:port` has the port stripped. Returns
/// `None` when no entry parses as an IP.
fn leftmost_xff_ip(raw: &str) -> Option<IpAddr> {
    for entry in raw.split(',') {
        let trimmed = entry.trim();
        if trimmed.is_empty() {
            continue;
        }
        // RFC-7239 / draft bracketed v6: "[::1]:8080" or "[::1]".
        if let Some(rest) = trimmed.strip_prefix('[')
            && let Some((host, _)) = rest.split_once(']')
            && let Ok(ip) = host.parse::<IpAddr>()
        {
            return Some(ip);
        }
        // Plain IPv6 without brackets — one or more colons, no port
        // delimiter. Parse the whole thing.
        if trimmed.matches(':').count() > 1
            && let Ok(ip) = trimmed.parse::<IpAddr>()
        {
            return Some(ip);
        }
        // IPv4 or IPv4:port.
        let host = trimmed.split(':').next().unwrap_or(trimmed);
        if let Ok(ip) = host.parse::<IpAddr>() {
            return Some(ip);
        }
    }
    None
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

    // ─── XFF resolution ─────────────────────────────────────────────
    //
    // The rate-limiter's bucket key is the most load-bearing security
    // input on `/v{1,2}/chat/completions`. Pin the precedence at the
    // data layer so a regression doesn't need an HTTP dance to surface.

    #[test]
    fn resolve_client_ip_prefers_peer_when_xff_untrusted() {
        let peer: IpAddr = "203.0.113.10".parse().unwrap();
        // XFF present but `HALO_TRUST_XFF=0` → key off peer.
        let got = super::resolve_client_ip(Some(peer), Some("198.51.100.1, 10.0.0.1"), false);
        assert_eq!(got, Some(peer));
    }

    #[test]
    fn resolve_client_ip_prefers_xff_leftmost_when_trusted() {
        let peer: IpAddr = "203.0.113.10".parse().unwrap();
        let client: IpAddr = "198.51.100.1".parse().unwrap();
        // With trust on, left-most parseable entry wins over the peer.
        let got = super::resolve_client_ip(
            Some(peer),
            Some("198.51.100.1, 10.0.0.1, 10.0.0.2"),
            true,
        );
        assert_eq!(got, Some(client));
    }

    #[test]
    fn resolve_client_ip_falls_back_to_peer_when_xff_missing_or_empty() {
        let peer: IpAddr = "203.0.113.10".parse().unwrap();
        assert_eq!(
            super::resolve_client_ip(Some(peer), None, true),
            Some(peer),
            "trust-on but no header → peer"
        );
        assert_eq!(
            super::resolve_client_ip(Some(peer), Some(""), true),
            Some(peer),
            "trust-on but empty header → peer"
        );
        assert_eq!(
            super::resolve_client_ip(Some(peer), Some("   ,  , "), true),
            Some(peer),
            "trust-on but all-whitespace entries → peer"
        );
    }

    #[test]
    fn resolve_client_ip_handles_v6_brackets_and_ports() {
        let client_v4: IpAddr = "198.51.100.1".parse().unwrap();
        let client_v6: IpAddr = "2001:db8::1".parse().unwrap();
        // Bracketed v6 with port (draft RFC-7239 shape).
        assert_eq!(
            super::resolve_client_ip(None, Some("[2001:db8::1]:8080, 10.0.0.1"), true),
            Some(client_v6),
        );
        // Plain v6, no brackets, no port.
        assert_eq!(
            super::resolve_client_ip(None, Some("2001:db8::1"), true),
            Some(client_v6),
        );
        // v4:port — port stripped.
        assert_eq!(
            super::resolve_client_ip(None, Some("198.51.100.1:55555"), true),
            Some(client_v4),
        );
    }

    #[test]
    fn resolve_client_ip_skips_garbage_and_returns_none_when_both_missing() {
        // Trust on, header is all garbage, no peer → None (caller will
        // fall through to allow).
        assert_eq!(
            super::resolve_client_ip(None, Some("not-an-ip, still-not-an-ip"), true),
            None,
        );
        // Even without trust, no peer means no key.
        assert_eq!(super::resolve_client_ip(None, Some("1.2.3.4"), false), None);
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

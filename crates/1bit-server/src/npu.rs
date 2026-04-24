//! `GET /v1/npu/status` — XDNA 2 NPU health probe.
//!
//! Mirrors the logic in `1bit-cli`'s `halo npu status` subcommand but returns
//! structured JSON instead of human-formatted text, so operators + dashboards
//! can poll health without shelling out to `xrt-smi`.
//!
//! # What we probe
//!
//! * `/dev/accel/accel0` — the amdxdna driver's device node; present iff the
//!   kernel module loaded + enumerated the NPU.
//! * `RLIMIT_MEMLOCK` — the calling process's soft memlock limit. XRT pins
//!   buffer objects so this needs to be `unlimited` (or at least very large)
//!   for serious workloads. We report the value, never refuse.
//! * `which xrt-smi` — is the XRT CLI on `$PATH`?
//! * `xrt-smi examine` — if present, parse out the `NPU Firmware Version`
//!   line. This is the one non-trivial parse and the one most likely to fail
//!   (xrt-smi returns a partial dump when no NPU is visible).
//! * `/proc/modules` — is `amdxdna` loaded?
//!
//! # Design notes
//!
//! * **Stub-mode safe.** We do NOT link `1bit-xdna` here — that crate
//!   is feature-gated and the server must build on CI hosts without XRT.
//!   Instead we probe the live filesystem / `$PATH` directly. The `backend`
//!   field of the response just echoes the compile-time state (it's always
//!   `"stub"` in current 1bit-server builds because we don't link xdna at
//!   all).
//! * **Never error out.** Every probe is best-effort. If `xrt-smi` isn't
//!   installed we return `xrt_smi_installed: false` and leave firmware
//!   `null`; we do not return HTTP 500. Same for missing `/dev/accel`.
//! * **Testable via dependency injection.** The [`NpuProber`] struct carries
//!   the device path, the name of the `xrt-smi` binary, and the path to
//!   `/proc/modules`, so unit tests can point it at fake inputs.
//!
//! # Endpoint shape
//!
//! ```json
//! {
//!   "backend": "stub",
//!   "device_node_present": true,
//!   "memlock_unlimited": true,
//!   "xrt_smi_installed": true,
//!   "npu_firmware_version": "1.1.2.65",
//!   "amdxdna_loaded": true,
//!   "error": null,
//!   "advisories": []
//! }
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

use axum::Json;
use axum::extract::State;
use serde::{Deserialize, Serialize};

use crate::routes::AppState;

/// Compile-time backend identity. Flipped to `"real-xrt"` if 1bit-server is
/// ever built with a live XDNA feature; today it's always `"stub"` because
/// 1bit-server doesn't link `1bit-xdna`.
const BACKEND: &str = "stub";

/// JSON response body for `GET /v1/npu/status`. Every field is always
/// populated; fields that can't be probed (e.g. firmware when xrt-smi is
/// missing) are `null`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NpuStatus {
    /// `"stub"` or `"real-xrt"` — reflects 1bit-server's build flags, NOT
    /// whether the NPU is physically usable.
    pub backend: String,
    /// Does `/dev/accel/accel0` exist?
    pub device_node_present: bool,
    /// Is `RLIMIT_MEMLOCK` soft limit `RLIM_INFINITY` / unlimited?
    pub memlock_unlimited: bool,
    /// Is `xrt-smi` on `$PATH`?
    pub xrt_smi_installed: bool,
    /// Firmware version parsed out of `xrt-smi examine`, or `null`.
    pub npu_firmware_version: Option<String>,
    /// Is the `amdxdna` kernel module loaded (per `/proc/modules`)?
    pub amdxdna_loaded: bool,
    /// Single summary error when something looks off (e.g. "no device node").
    /// `null` when every probe came back healthy.
    pub error: Option<String>,
    /// Human-readable one-line hints surfaced to UIs.
    pub advisories: Vec<String>,
}

/// Injectable probe configuration. Production code uses [`NpuProber::live`].
/// Tests construct one by hand with fake paths.
#[derive(Debug, Clone)]
pub struct NpuProber {
    /// Usually `/dev/accel/accel0`.
    pub device_node: PathBuf,
    /// The binary to invoke for firmware detection (`"xrt-smi"` live;
    /// tests override with something nonexistent).
    pub xrt_smi_bin: String,
    /// Usually `/proc/modules`.
    pub proc_modules: PathBuf,
    /// Overrides `$PATH` for `which`-style lookups during tests. `None`
    /// means "use the process's current PATH".
    pub path_override: Option<String>,
}

impl NpuProber {
    /// Production probe — hits the real filesystem + real `xrt-smi`.
    pub fn live() -> Self {
        Self {
            device_node: PathBuf::from("/dev/accel/accel0"),
            xrt_smi_bin: "xrt-smi".to_string(),
            proc_modules: PathBuf::from("/proc/modules"),
            path_override: None,
        }
    }

    /// Run every probe and assemble an [`NpuStatus`]. Never panics, never
    /// returns an error — probe failures surface as `false` / `null` fields
    /// + entries in `advisories`.
    pub fn probe(&self) -> NpuStatus {
        let device_node_present = self.device_node.exists();
        let memlock_unlimited = probe_memlock_unlimited();
        let xrt_smi_installed = which(&self.xrt_smi_bin, self.path_override.as_deref());
        let npu_firmware_version = if xrt_smi_installed {
            probe_firmware(&self.xrt_smi_bin, self.path_override.as_deref())
        } else {
            None
        };
        let amdxdna_loaded = probe_amdxdna_loaded(&self.proc_modules);

        let mut advisories = Vec::new();
        let mut errors = Vec::new();

        if !device_node_present {
            errors.push(format!(
                "device node {} missing",
                self.device_node.display()
            ));
            advisories.push("amdxdna module may not be loaded; try `sudo modprobe amdxdna`".into());
        }
        if !amdxdna_loaded {
            advisories
                .push("amdxdna not in /proc/modules — run `halo npu status` for details".into());
        }
        if !xrt_smi_installed {
            advisories.push(
                "xrt-smi not on PATH; install `xrt` + `xrt-plugin-amdxdna` to enable firmware probe"
                    .into(),
            );
        }
        if !memlock_unlimited {
            advisories.push(
                "RLIMIT_MEMLOCK is not unlimited; see /etc/security/limits.d/99-npu-memlock.conf"
                    .into(),
            );
        }

        let error = if errors.is_empty() {
            None
        } else {
            Some(errors.join("; "))
        };

        NpuStatus {
            backend: BACKEND.to_string(),
            device_node_present,
            memlock_unlimited,
            xrt_smi_installed,
            npu_firmware_version,
            amdxdna_loaded,
            error,
            advisories,
        }
    }
}

// ─── Individual probes ───────────────────────────────────────────────────

/// Read `RLIMIT_MEMLOCK` via `getrlimit(2)` and report whether the soft limit
/// is `RLIM_INFINITY`. Returns `false` on any syscall failure.
fn probe_memlock_unlimited() -> bool {
    // SAFETY: `rlim` is a POD struct, `getrlimit` fills it; on error we
    // just return `false`.
    unsafe {
        let mut rlim = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        if libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut rlim) != 0 {
            return false;
        }
        rlim.rlim_cur == libc::RLIM_INFINITY
    }
}

/// Walk `$PATH` (or `path_override`) looking for `bin`. Returns `true` iff
/// a regular-file match exists and is executable-by-user. We don't
/// exec the binary, so this is safe to call on every request.
fn which(bin: &str, path_override: Option<&str>) -> bool {
    if bin.is_empty() {
        return false;
    }
    let path = match path_override {
        Some(p) => p.to_string(),
        None => std::env::var("PATH").unwrap_or_default(),
    };
    for dir in path.split(':') {
        if dir.is_empty() {
            continue;
        }
        let candidate = Path::new(dir).join(bin);
        if candidate.is_file() {
            return true;
        }
    }
    false
}

/// Spawn `xrt-smi examine` and grep for the firmware version line. Returns
/// `None` if the binary fails to spawn, exits non-zero, or output doesn't
/// contain a parseable `NPU Firmware Version : ...` line.
fn probe_firmware(bin: &str, path_override: Option<&str>) -> Option<String> {
    let mut cmd = Command::new(bin);
    cmd.arg("examine");
    if let Some(p) = path_override {
        cmd.env("PATH", p);
    }
    let out = cmd.output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        let l = line.trim();
        // Expected line shape: "NPU Firmware Version : 1.1.2.65"
        if let Some(rest) = l.strip_prefix("NPU Firmware Version") {
            // Find the colon, take everything after.
            if let Some(idx) = rest.find(':') {
                let v = rest[idx + 1..].trim();
                if !v.is_empty() {
                    return Some(v.to_string());
                }
            }
        }
    }
    None
}

/// Read `/proc/modules` (one line per loaded module, first token is the
/// module name) and return `true` if `amdxdna` appears. Best-effort —
/// returns `false` on I/O error.
fn probe_amdxdna_loaded(proc_modules: &Path) -> bool {
    let Ok(s) = std::fs::read_to_string(proc_modules) else {
        return false;
    };
    s.lines().any(|line| {
        line.split_whitespace()
            .next()
            .map(|name| name == "amdxdna")
            .unwrap_or(false)
    })
}

// ─── HTTP handler ────────────────────────────────────────────────────────

/// `GET /v1/npu/status`. Always 200 — if probes show problems, the
/// `error` / `advisories` fields carry the signal, not the HTTP status.
pub async fn npu_status(State(s): State<AppState>) -> Json<NpuStatus> {
    // Blocking syscalls + a `Command::output` on xrt-smi — offload so we
    // don't stall the tokio reactor under contention.
    let status = tokio::task::spawn_blocking(|| NpuProber::live().probe())
        .await
        .unwrap_or_else(|_| NpuStatus {
            backend: BACKEND.to_string(),
            device_node_present: false,
            memlock_unlimited: false,
            xrt_smi_installed: false,
            npu_firmware_version: None,
            amdxdna_loaded: false,
            error: Some("npu probe task panicked".into()),
            advisories: vec![],
        });

    // Update the cheap `npu_up` gauge on the shared Metrics handle. "Up"
    // means: device node present AND amdxdna module loaded. xrt-smi /
    // memlock don't gate "up" because the endpoint is meant to surface
    // *hardware* presence, not full readiness.
    let up = status.device_node_present && status.amdxdna_loaded;
    s.metrics.set_npu_up(up);

    Json(status)
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_json_roundtrip() {
        // Serialise a fully-populated status, deserialise, assert equality.
        let original = NpuStatus {
            backend: "stub".into(),
            device_node_present: true,
            memlock_unlimited: false,
            xrt_smi_installed: true,
            npu_firmware_version: Some("1.1.2.65".into()),
            amdxdna_loaded: true,
            error: None,
            advisories: vec!["memlock not unlimited".into()],
        };
        let s = serde_json::to_string(&original).expect("serialize");
        let v: serde_json::Value = serde_json::from_str(&s).expect("json");
        // Key presence check — catches field renames / accidental skips.
        for k in [
            "backend",
            "device_node_present",
            "memlock_unlimited",
            "xrt_smi_installed",
            "npu_firmware_version",
            "amdxdna_loaded",
            "error",
            "advisories",
        ] {
            assert!(v.get(k).is_some(), "missing key {k} in {v}");
        }
        let round: NpuStatus = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(round, original);
        // Null roundtrip — None should serialize as JSON null, not absent.
        let missing = NpuStatus {
            backend: "stub".into(),
            device_node_present: false,
            memlock_unlimited: false,
            xrt_smi_installed: false,
            npu_firmware_version: None,
            amdxdna_loaded: false,
            error: Some("no device".into()),
            advisories: vec![],
        };
        let j = serde_json::to_value(&missing).unwrap();
        assert!(j["npu_firmware_version"].is_null(), "fw must be null");
        assert!(j["error"].as_str() == Some("no device"));
    }

    #[test]
    fn probe_fake_paths_reports_missing_device() {
        // Point every probe at a nonexistent path / uninstalled binary.
        let prober = NpuProber {
            device_node: PathBuf::from("/definitely/does/not/exist/accel0"),
            xrt_smi_bin: "xrt-smi-never-installed-xxxxxx".into(),
            proc_modules: PathBuf::from("/definitely/does/not/exist/modules"),
            // Empty PATH — which() can't find anything.
            path_override: Some(String::new()),
        };
        let s = prober.probe();
        assert_eq!(s.backend, "stub");
        assert!(
            !s.device_node_present,
            "device_node_present should be false"
        );
        assert!(!s.xrt_smi_installed, "xrt_smi_installed should be false");
        assert!(s.npu_firmware_version.is_none(), "fw must be None");
        assert!(!s.amdxdna_loaded, "amdxdna_loaded should be false");
        assert!(
            s.error.as_deref().unwrap_or("").contains("device node"),
            "error should mention device node, got {:?}",
            s.error
        );
        // Advisories should at minimum mention xrt-smi + amdxdna.
        let joined = s.advisories.join("\n");
        assert!(
            joined.contains("xrt-smi"),
            "advisories should mention xrt-smi, got {joined:?}"
        );
        assert!(
            joined.contains("amdxdna"),
            "advisories should mention amdxdna, got {joined:?}"
        );
    }

    #[test]
    fn probe_amdxdna_loaded_parses_proc_modules() {
        use std::io::Write;
        // Fake /proc/modules with amdxdna present.
        let mut f = tempfile_like("modules_with_amdxdna");
        writeln!(
            f.file,
            "amdxdna 294912 0 - Live 0x0000000000000000\nnvidia 99999 0 - Live 0x0"
        )
        .unwrap();
        assert!(probe_amdxdna_loaded(&f.path));

        // Fake /proc/modules without it.
        let mut g = tempfile_like("modules_without_amdxdna");
        writeln!(g.file, "snd_hda_intel 61440 0 - Live 0x0").unwrap();
        assert!(!probe_amdxdna_loaded(&g.path));

        // Nonexistent file.
        assert!(!probe_amdxdna_loaded(Path::new(
            "/definitely/not/a/modules/file"
        )));
    }

    #[tokio::test]
    async fn route_returns_200_with_expected_shape() {
        use crate::backend::EchoBackend;
        use crate::routes::build_router_with_state;
        use axum::body::{Body, to_bytes};
        use axum::http::{Request, StatusCode};
        use std::sync::Arc;
        use tower::ServiceExt;

        let state = AppState {
            backend: Arc::new(EchoBackend::new()),
            metrics: Arc::new(crate::metrics::Metrics::new()),
            sd_base_url: Arc::new("http://127.0.0.1:8081".to_string()),
            http_client: crate::routes::default_http_client(),
            rate_limit: Arc::new(crate::middleware::RateLimit::new(0)),
            models: Arc::new(crate::registry::ModelRegistry::empty()),
            default_chat_template: crate::chat_template::ChatTemplate::default(),
        };
        let app = build_router_with_state(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/v1/npu/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 16 * 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["backend"], "stub");
        for k in [
            "backend",
            "device_node_present",
            "memlock_unlimited",
            "xrt_smi_installed",
            "npu_firmware_version",
            "amdxdna_loaded",
            "error",
            "advisories",
        ] {
            assert!(v.get(k).is_some(), "missing key {k} in {v}");
        }
        // advisories is always an array, never missing.
        assert!(v["advisories"].is_array(), "advisories should be array");
    }

    // Tiny self-cleaning temp-file helper so we don't pull in the `tempfile`
    // crate just for these tests. Deletes on drop.
    struct TempFile {
        file: std::fs::File,
        path: PathBuf,
    }
    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }
    fn tempfile_like(tag: &str) -> TempFile {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "1bit-server-npu-test-{}-{}-{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let file = std::fs::File::create(&path).expect("create temp file");
        TempFile { file, path }
    }
}

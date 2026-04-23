//! Hardware detection.
//!
//! At router startup we need to pick exactly one execution backend for the
//! lifetime of the process:
//!
//! * **HIP (gfx1151)** — preferred on Strix Halo / AMD boxes. Detected by
//!   the presence of `/opt/rocm/bin/rocminfo` **or** a DRM node whose
//!   `device/vendor` is `0x1002` (the AMD PCI vendor ID).
//! * **MLX (Apple Silicon)** — macOS fallback. Not actually wired in this
//!   session; we return the variant but the implementation path is
//!   `unimplemented!()` on first call (see `backend_impl.rs`).
//! * **None** — everything else. The router constructor loudly fails
//!   rather than silently falling back to a CPU path that would take
//!   minutes per token.
//!
//! The detection is cheap (a handful of filesystem stat calls), so we just
//! re-run it every time the router boots rather than caching.

use std::fs;
use std::path::Path;

/// The backend family selected by `detect()`. Carries no state — it is
/// just a routing tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// AMD ROCm / HIP on gfx1151 (Strix Halo) or compatible.
    Hip,
    /// Apple MLX — macOS ARM64.
    Mlx,
    /// No supported accelerator found. Router construction must fail.
    None,
}

impl BackendKind {
    /// Human-readable label used in log lines and model cards.
    pub fn label(self) -> &'static str {
        match self {
            BackendKind::Hip => "hip (gfx1151)",
            BackendKind::Mlx => "mlx (apple)",
            BackendKind::None => "none",
        }
    }
}

/// Run detection. Order matters: we prefer HIP on any host that has it,
/// because on a dual-stack CI box (AMD iGPU + macOS-in-VM) the "real"
/// accelerator is the one with ROCm installed.
pub fn detect() -> BackendKind {
    if has_hip() {
        BackendKind::Hip
    } else if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
        BackendKind::Mlx
    } else {
        BackendKind::None
    }
}

fn has_hip() -> bool {
    // Signal 1: /opt/rocm/bin/rocminfo (installed alongside the runtime
    // on every ROCm-packaged distro — CachyOS, Ubuntu's rocm-dev, Fedora,
    // etc.). This is the cheapest positive indicator.
    if Path::new("/opt/rocm/bin/rocminfo").exists() {
        return true;
    }

    // Signal 2: look for an AMD DRM node. `/sys/class/drm/card*/device/vendor`
    // reads `0x1002\n` when the card is AMD. We don't need the exact model;
    // 1bit-hip's FFI layer will fail loudly if the GPU isn't gfx1151.
    if let Ok(entries) = fs::read_dir("/sys/class/drm") {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = match name.to_str() {
                Some(s) => s,
                None => continue,
            };
            // Match `card0`, `card1`, ... — skip `cardN-HDMI-...` output
            // connectors which also live under /sys/class/drm.
            if !name.starts_with("card") || name.contains('-') {
                continue;
            }
            let vendor_path = entry.path().join("device").join("vendor");
            if let Ok(s) = fs::read_to_string(&vendor_path)
                && s.trim().eq_ignore_ascii_case("0x1002")
            {
                return true;
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_is_stable() {
        assert_eq!(BackendKind::Hip.label(), "hip (gfx1151)");
        assert_eq!(BackendKind::Mlx.label(), "mlx (apple)");
        assert_eq!(BackendKind::None.label(), "none");
    }

    #[test]
    fn detection_returns_one_variant() {
        // We cannot assert which variant — that depends on the host —
        // but detection itself must not panic and must return a sane enum.
        let _ = detect();
    }
}

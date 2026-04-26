//! 1bit-mlx — Apple Silicon ternary backend for 1bit systems.
//!
//! Sibling of `1bit-hip` (AMD gfx1151). Both expose the same `Backend`
// (deleted: was 1bit-router orphan crate)
//! librocm_cpp.so. Cargo features pick at build time.
//!
//! Build on Apple Silicon:  `cargo build -p 1bit-mlx --features mlx-apple`
//! Build anywhere else:     `cargo build -p 1bit-mlx`  (stub, no engine)

#[cfg(feature = "mlx-apple")]
pub mod apple;

#[cfg(not(feature = "mlx-apple"))]
pub mod stub;

/// Status returned by every halo-ternary backend call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendStatus {
    Ok,
    NotCompiled,
    InvalidArg,
    RuntimeFailed,
}

/// The shape every halo ternary backend must expose. Both -hip and -mlx
// (deleted: was 1bit-router orphan crate)
pub trait TernaryBackend {
    fn name(&self) -> &'static str;
    fn supports_shape(&self, n_heads: usize, head_dim: usize) -> bool;
    fn load_h1b(&mut self, path: &str) -> anyhow::Result<()>;
    fn generate(&mut self, prompt_tokens: &[u32], max_new: usize) -> anyhow::Result<Vec<u32>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `BackendStatus` is a plain enum; ensure each variant is constructible
    /// and that `Debug` stays stable — downstream log consumers grep on it.
    #[test]
    fn backend_status_debug_format_is_stable() {
        assert_eq!(format!("{:?}", BackendStatus::Ok), "Ok");
        assert_eq!(format!("{:?}", BackendStatus::NotCompiled), "NotCompiled");
        assert_eq!(format!("{:?}", BackendStatus::InvalidArg), "InvalidArg");
        assert_eq!(
            format!("{:?}", BackendStatus::RuntimeFailed),
            "RuntimeFailed"
        );
    }

    /// Every variant must be distinct; PartialEq is load-bearing for the
    /// router when it compares returned status against its expectations.
    #[test]
    fn backend_status_variants_are_distinct() {
        let all = [
            BackendStatus::Ok,
            BackendStatus::NotCompiled,
            BackendStatus::InvalidArg,
            BackendStatus::RuntimeFailed,
        ];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                assert_eq!(i == j, a == b, "{a:?} vs {b:?}");
            }
        }
    }

    /// Copy + Clone must be preserved so callers can stash the status
    /// without giving up ownership of the originating result.
    #[test]
    fn backend_status_is_copy_and_clone() {
        let s = BackendStatus::Ok;
        let _copy = s; // Copy
        let cloned = s.clone();
        assert_eq!(s, cloned);
        // After copy, original still usable — would not compile if !Copy.
        assert_eq!(s, BackendStatus::Ok);
    }

    /// On non-Apple hosts (the default build), the crate compiles the
    /// stub module and its backend refuses every call. Make sure both
    /// surface methods panic-free and report the stub status.
    #[cfg(not(feature = "mlx-apple"))]
    #[test]
    fn stub_backend_advertises_stub_and_refuses_load() {
        use crate::stub::{MlxBackend, status};
        let mut b = MlxBackend;
        assert_eq!(status(), BackendStatus::NotCompiled);
        assert!(
            b.name().contains("stub"),
            "stub backend name must say 'stub': {}",
            b.name()
        );
        assert!(!b.supports_shape(4, 64));
        assert!(b.load_h1b("/nope").is_err());
        assert!(b.generate(&[1, 2, 3], 4).is_err());
    }
}

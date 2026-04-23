//! 1bit-stream — `.1bl` catalog HTTP server.
//!
//! Serves the read side of the 1bit-systems streaming catalog. Lossy
//! tier is open to the world; lossless tier is gated on a premium JWT
//! claim verified here. Upload, BTCPay, and arithmetic decoding live in
//! sibling crates.
//!
//! See `docs/wiki/1bl-container-spec.md` for the container format this
//! crate reads + trims.

pub mod auth;
pub mod container;
pub mod handlers;

pub use auth::AuthConfig;
pub use container::{Catalog, Manifest};
pub use handlers::{AppState, build};

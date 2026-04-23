//! Outbound MCP (Model Context Protocol) client library.
//!
//! halo specialists (Librarian, Magistrate, Quartermaster, Warden…) use
//! this crate to call out to external MCP servers — GitHub, Semgrep,
//! Discord, LinuxGSM — via JSON-RPC 2.0 over either stdio (child process)
//! or streamable HTTP.
//!
//! Caller-side code by design: nothing here is in the inference hot path,
//! so Rule A is satisfied by the tokio::process child model. No Python
//! deps pulled in transitively.

pub mod error;
pub mod github;
pub mod http;
pub mod protocol;
pub mod semgrep;
pub mod stdio;

pub use error::McpError;
pub use http::HttpClient;
pub use protocol::{ContentBlock, Tool, ToolCallResult};
pub use stdio::StdioClient;

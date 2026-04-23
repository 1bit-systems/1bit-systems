//! Outbound MCP facade for specialists.
//!
//! Librarian, Magistrate, and Quartermaster all need to talk to GitHub;
//! Warden wants Semgrep for its CVG static-analysis gate. Rather than
//! let each specialist spin up its own transport, this module owns a
//! small set of lazy-initialised clients and hands out references.
//!
//! Gated behind the `mcp-out` cargo feature so callers that only need
//! the registry don't pay for reqwest/tokio-process indirections.

use anyhow::{Context, Result};
use onebit_mcp_clients::{github::GitHub, semgrep::Semgrep};

/// Construct a GitHub MCP client from the ambient `GITHUB_TOKEN`. Returns
/// `Ok(None)` when the env var is missing — callers treat that as
/// "GitHub tools unavailable in this deployment" rather than a hard
/// failure.
pub fn github_from_env() -> Result<Option<GitHub>> {
    match GitHub::from_env() {
        Some(c) => Ok(Some(c)),
        None => Ok(None),
    }
}

/// Construct a Semgrep MCP client against the public hosted endpoint.
/// Always succeeds — no auth required — but still returns a Result for
/// symmetry with `github_from_env`.
pub fn semgrep_default() -> Result<Semgrep> {
    Ok(Semgrep::new())
}

/// Helper used by Librarian / Magistrate / Quartermaster: call
/// `tools/list` against the GitHub MCP and return the tool names the
/// specialist can legitimately invoke. Useful for the `@halo-bot status`
/// path where we want to report which surface is live without actually
/// calling a tool.
pub async fn github_tool_names() -> Result<Vec<String>> {
    let Some(gh) = github_from_env()? else {
        return Ok(Vec::new());
    };
    let tools = gh.inner().list_tools().await.context("github tools/list")?;
    Ok(tools.into_iter().map(|t| t.name).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn github_from_env_none_when_token_missing() {
        let prev = std::env::var("GITHUB_TOKEN").ok();
        unsafe {
            std::env::remove_var("GITHUB_TOKEN");
        }
        let r = github_from_env().expect("no error path");
        assert!(r.is_none());
        if let Some(v) = prev {
            unsafe {
                std::env::set_var("GITHUB_TOKEN", v);
            }
        }
    }

    #[test]
    fn semgrep_default_builds() {
        let s = semgrep_default().expect("semgrep builds");
        // Compile-time smoke: we got a client. Any real call would need
        // network, so just confirm we can reach the inner type.
        let _ = s.inner();
    }
}

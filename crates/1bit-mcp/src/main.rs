// 1bit-mcp — minimal stdio JSON-RPC entry point.
//
// The agents-backed registry was retired in the 2026-04-25 cull (see
// lib.rs). This binary still launches and accepts JSON-RPC traffic so
// existing systemd / packages.toml plumbing keeps working; tool list is
// empty until the crate is re-pointed at GAIA agent-core (or until the
// canonical C++ port at 1bit-services/mcp/ supersedes it).

use anyhow::Result;
use tokio::io;
use tracing::info;

use onebit_mcp::StdioServer;

#[tokio::main]
async fn main() -> Result<()> {
    // Logs go to stderr so they never corrupt the JSON-RPC stream on
    // stdout. Clients talking to us over stdio rely on this.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "onebit_mcp=info".into()),
        )
        .with_writer(std::io::stderr)
        .with_target(false)
        .init();

    let server = StdioServer::new();
    info!(
        version = %onebit_mcp::SERVER_VERSION,
        tools = server.registry().len(),
        "1bit-mcp starting (post-agents-cull stub — empty tool list)"
    );

    server.run(io::stdin(), io::stdout()).await?;

    info!("1bit-mcp stdin closed, exiting 0");
    Ok(())
}

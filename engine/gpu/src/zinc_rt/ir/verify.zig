//! ZINC_RT IR verifier entrypoints.
//! Verification stays separate from graph construction so future passes can
//! reject malformed shapes and bindings before any backend executes them.
//! @section Decode Planning
const graph_mod = @import("graph.zig");

/// Run structural verification over an IR graph.
/// Thin wrapper around `Graph.verify` so callers depend on this module
/// rather than reaching into the graph type directly; future passes will
/// add shape and type checks here without touching graph construction.
/// @param ir Graph to inspect; not mutated.
/// @returns Propagates any verification error from `Graph.verify`.
pub fn graph(ir: *const graph_mod.Graph) !void {
    try ir.verify();
}

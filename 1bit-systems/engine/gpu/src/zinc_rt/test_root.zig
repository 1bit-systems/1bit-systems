//! ZINC_RT standalone unit-test root.
//! Keeps the M0 scaffold tests rooted at `src/zinc_rt` so internal modules can
//! be imported without escaping the module path.
//! @section Inference Runtime

comptime {
    _ = @import("zinc_rt");
    _ = @import("forward_zinc_rt");
}

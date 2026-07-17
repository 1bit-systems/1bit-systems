//! Backend-selected model manager for the HTTP server.
//! @section API Server
//!
//! This thin shim keeps the HTTP server code importing one stable manager type
//! while build-time backend selection decides whether that implementation comes
//! from the Vulkan runtime or the Apple Silicon Metal runtime.
const gpu = @import("../gpu/interface.zig");

const impl = if (gpu.is_metal)
    @import("model_manager_metal.zig")
else
    @import("model_manager.zig");

/// Specification describing which model to load: a filesystem path, an optional managed-catalog ID, and an optional context-length override.
pub const LoadSpec = impl.LoadSpec;
/// Thread-safe manager for the currently active model and inference engine; handles loading, hot-swapping, catalog queries, and VRAM budget enforcement.
pub const ModelManager = impl.ModelManager;

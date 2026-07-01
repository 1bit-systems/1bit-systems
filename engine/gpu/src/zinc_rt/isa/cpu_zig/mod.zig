//! Pure Zig T-CPU opcode implementations.
//! The modules exported here are clarity-first reference kernels used by the
//! CPU ring and by future cross-tier validation tests.
//! @section Inference Runtime

/// Scalar RMS normalization with learned per-channel weight, used as the CPU oracle for layer norm.
pub const rms_norm = @import("rms_norm.zig");
/// Scalar SwiGLU activation (`silu(gate) * up`) used by dense and MoE MLP paths.
pub const swiglu = @import("swiglu.zig");
/// Deterministic argmax over a logits vector; picks the lowest index on ties.
pub const argmax = @import("argmax.zig");
/// Shared GGML row dequantization and quantized row dot-product helpers consumed by other CPU ops.
pub const dequant = @import("dequant.zig");
/// Token embedding lookup: dequantize one row of a GGUF embedding matrix into f32 hidden state.
pub const embed = @import("embed.zig");
/// LM head projection: multiply a GGUF output matrix by the final hidden state to produce vocab logits.
pub const lm_head = @import("lm_head.zig");

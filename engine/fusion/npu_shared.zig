//! NPU shared types — re-exports from npu/src for use by the fused engine.
// This avoids module-path conflicts with multiple npu/src files.
const std = @import("std");

pub const NpuPageTable = @import("../npu/src/npu_page_table.zig").NpuPageTable;
pub const PageMapping = @import("../npu/src/npu_page_table.zig").PageMapping;

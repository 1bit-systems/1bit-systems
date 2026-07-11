//! Source-level regression guards for bugs that are hard to cover with unit-only GPU tests.
const builtin = @import("builtin");
const std = @import("std");

fn expectContains(haystack: []const u8, needle: []const u8) !void {
    try std.testing.expect(std.mem.indexOf(u8, haystack, needle) != null);
}

fn expectNotContains(haystack: []const u8, needle: []const u8) !void {
    try std.testing.expect(std.mem.indexOf(u8, haystack, needle) == null);
}

fn expectContainsNear(haystack: []const u8, marker: []const u8, needle: []const u8, window: usize) !void {
    const start = std.mem.indexOf(u8, haystack, marker) orelse return error.TestExpectedEqual;
    const end = @min(start + window, haystack.len);
    try std.testing.expect(std.mem.indexOf(u8, haystack[start..end], needle) != null);
}

fn expectMultiSubgroupFallback(shader_src: []const u8, reduce_name: []const u8) !void {
    try expectContains(shader_src, "gl_NumSubgroups > 1u");
    try expectContains(shader_src, "gl_SubgroupInvocationID == 0u");
    try expectContains(shader_src, reduce_name);
    try expectContains(shader_src, "barrier();");
}

test "decode loop keeps deinterleave split for packed Q and gate" {
    const src = @embedFile("compute/forward.zig");
    try expectContains(src, "Qwen3Next packs per-head [Q(head_dim), gate(head_dim)] blocks.");
    try expectContains(src, "Deinterleave Q+gate using compute shader");
    try expectContains(src, "pipeline_deinterleave");
}

test "decode loop applies packed attention gate after flash attention" {
    const src = @embedFile("compute/forward.zig");
    try expectContains(src, "// Flash attention");
    try expectContains(src, "self.writeDescSet3(gds, self.attn_out_buf.handle");
}

test "decode loop keeps compute-to-transfer barrier before KV cache writes" {
    const src = @embedFile("compute/forward.zig");
    const marker = std.mem.indexOf(u8, src, "Transfer fallback: Q RoPE before barrier (original order preserved)") orelse return error.TestExpectedEqual;
    const fallback_src = src[marker..@min(marker + 1600, src.len)];
    try expectContains(fallback_src, "self.decode_cmd.computeAndTransferBarrier();");
    try expectContains(fallback_src, "vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.k_buf.handle, self.kv_k_cache[layer_idx].handle");
    try expectContains(fallback_src, "self.decode_cmd.transferToComputeBarrier();");
}

test "decode loop keeps layer-boundary compute barrier after FFN residual" {
    const src = @embedFile("compute/forward.zig");
    try expectContainsNear(src, "// FFN residual: hidden_buf += down_buf", "self.decode_cmd.computeBarrier();", 2400);
}

test "prefill resets per-request state before processing prompt tokens" {
    const src = @embedFile("compute/forward.zig");
    try expectContainsNear(src, "pub fn prefillBatch(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "try self.resetRequestState(target_context_tokens);", 900);
}

test "Metal prefill preserves cached prefixes instead of resetting unconditionally" {
    const src = @embedFile("compute/forward_metal.zig");
    try expectContainsNear(src, "pub fn prefillBatch(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "if (state.position == 0 and state.generated_tokens.items.len == 0)", 900);
    try expectContainsNear(src, "pub fn prefillBatch(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "return error.KvStateNotAvailable;", 1400);
}

test "Metal prefillBatched gates on env flag and supported architecture" {
    const src = @embedFile("compute/forward_metal.zig");
    try expectContains(src, "ZINC_BATCHED_PREFILL");
    try expectContainsNear(src, "pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "batchedPrefillMode()", 600);
    try expectContainsNear(src, "pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "canUseBatchedPrefill(self)", 1800);
    try expectContainsNear(src, "if (mode == .off or !can_batched_prefill) {", "return self.prefillBatch(state, prompt_tokens);", 200);
}

test "Metal prefillBatched validate path diffs last-token logits within 1e-3" {
    const src = @embedFile("compute/forward_metal.zig");
    try expectContainsNear(src, "pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "if (mode == .validate)", 20000);
    try expectContainsNear(src, "if (mode == .validate)", "const tol: f32 = 1e-3;", 1500);
    try expectContainsNear(src, "if (mode == .validate)", "try self.prefillBatch(state, prompt_tokens);", 1500);
}

test "Metal Gemma MoE batched prefill honors Gemma env and Q8 GEMM" {
    const src = @embedFile("compute/forward_metal.zig");
    try expectContains(src, "ZINC_GEMMA_BATCHED_PREFILL");
    try expectContainsNear(src, "pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "gemmaBatchedPrefillMode()", 900);
    try expectContainsNear(src, "fn supportsBatchedGemmQuant", ".q8_0 => engine.gemm_q8_0_pipe.handle != null", 600);
    try expectContainsNear(src, "fn dispatchGemmBatchedOnCmd", ".q8_0 => dispatchGemmQ8_0OnCmd", 800);
}

test "Metal Gemma MoE validation is env gated and fails above 1e-3" {
    const src = @embedFile("compute/forward_metal.zig");
    try expectContains(src, "ZINC_GEMMA_MOE_VALIDATE");
    try expectContainsNear(src, "fn shouldValidateGemmaMoe", "engine.position == 0", 500);
    try expectContainsNear(src, "fn shouldValidateGemmaMoe", "layer_idx == 0", 500);
    try expectContainsNear(src, "fn validateGemmaMoePostVector", "const tol: f32 = 1e-3;", 5000);
    try expectContainsNear(src, "fn validateGemmaMoePostVector", "return error.GemmaMoeValidationFailed;", 8000);
    try expectContains(src, "try validateGemmaMoePostVector(");
}

test "Metal prefillBatched uses gemm/rope batched dispatch helpers" {
    const src = @embedFile("compute/forward_metal.zig");
    try expectContainsNear(src, "pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "dispatchGemmBatchedOnCmd", 12000);
    try expectContainsNear(src, "pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "dispatchRopeBatchedOnCmd", 12000);
    try expectContainsNear(src, "pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "dispatchFlashAttnBatchedOnCmd", 12000);
}

test "Metal prefillBatched routes Q8 KV cache through flash_attn_batched_q8" {
    const src = @embedFile("compute/forward_metal.zig");
    try expectContainsNear(src, "pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "if (self.kv_cache_q8)", 12000);
    try expectContainsNear(src, "pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "dispatchFlashAttnBatchedQ8OnCmd", 12000);
    try expectContainsNear(src, "pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "dispatchKvCacheWriteBatchedQ8OnCmd", 12000);
}

test "Metal prefillBatched supports prefix reuse by extending KV at state.position" {
    const src = @embedFile("compute/forward_metal.zig");
    try expectContainsNear(src, "pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "const position_base: u32 = state.position;", 2600);
    try expectContainsNear(src, "pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "return error.KvStateNotAvailable;", 2000);
    try expectContainsNear(src, "pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "const kv_len = position_base + n_tokens;", 12000);
    try expectContainsNear(src, "pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "self.position = position_base + n_tokens;", 20000);
}

test "Vulkan prefillBatched gates on env flag + canUseBatchedPrefillRdna" {
    const src = @embedFile("compute/forward.zig");
    const fn_marker = "fn prefillBatchedImpl(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {";
    try expectContains(src, "ZINC_BATCHED_PREFILL");
    try expectContains(src, "fn canUseBatchedPrefillRdna(engine: *const InferenceEngine) bool {");
    try expectContainsNear(src, fn_marker, "canUseBatchedPrefillRdna(self)", 2000);
    try expectContainsNear(src, fn_marker, "ensureBatchedScratchCapacity", 3500);
}

test "Vulkan batched prefill keeps RDNA default and Intel dense Gemma guard" {
    const src = @embedFile("compute/forward.zig");
    const fn_marker = "fn canUseBatchedPrefillRdna(engine: *const InferenceEngine) bool {";
    try expectContainsNear(src, fn_marker, "vendor == .amd_rdna3", 900);
    try expectContainsNear(src, fn_marker, "vendor == .amd_rdna4", 900);
    try expectContainsNear(src, fn_marker, "vendor == .amd_rdna4_apu", 900);
    try expectContainsNear(src, "fn isIntelGpuVendor", "vendor == .intel_arc_xe2", 200);
    try expectContainsNear(src, fn_marker, "ZINC_INTEL_BATCHED_PREFILL", 1200);
    try expectContainsNear(src, "if (is_intel) {", "intel_dense_gemma_default", 700);
    try expectContainsNear(src, "const intel_dense_gemma_default", "cfg.architecture == .gemma", 250);
    try expectContainsNear(src, "const intel_dense_gemma_default", "cfg.n_experts == 0", 300);
    try expectContainsNear(src, "const intel_dense_gemma_default", "cfg.ssm_d_inner == 0", 350);
    try expectContainsNear(src, "if (is_intel) {", "return false;", 900);
}

test "Vulkan batched projection chunk size matches selected shader family" {
    const src = @embedFile("compute/forward.zig");
    const fn_marker = "fn dispatchProjectionBatched(";
    // Window widened by the mul_mm_q4k fast-path block prepended in
    // effort-6 Step 5; the SERIAL_MAX_COLS/KPAR_MAX_COLS constants stay
    // co-located with the kpar/serial chunk loop further down.
    try expectContainsNear(src, fn_marker, "const SERIAL_MAX_COLS: u32 = 32;", 3200);
    try expectContainsNear(src, fn_marker, "const KPAR_MAX_COLS: u32 = 40;", 3200);
    try expectContainsNear(src, fn_marker, "if (kpar_pipeline != null) KPAR_MAX_COLS else SERIAL_MAX_COLS", 5000);
}

test "Vulkan batched projection kpar is allowed on Intel wave32" {
    const src = @embedFile("compute/forward.zig");
    const marker = "const q4k_batch_kpar_enabled =";
    const start = std.mem.indexOf(u8, src, marker) orelse return error.TestExpectedEqual;
    const end = @min(start + 300, src.len);
    try expectContains(src[start..end], "dmmv.pipeline_q4k_batch_kpar != null");
    try expectNotContains(src[start..end], "gpu_config.wave_size == 64");
}

test "Vulkan Intel Gemma decode uses split-K at short sequence lengths" {
    const src = @embedFile("compute/forward.zig");
    const marker = "const intel_gemma_split_k_short_seq =";
    try expectContains(src, "for all Intel Gemma decode lengths");
    try expectContainsNear(src, marker, "config.architecture == .gemma", 200);
    try expectContainsNear(src, marker, "isIntelGpuVendor(self.gpu_config.vendor)", 200);
    try expectContainsNear(src, marker, "self.fa_split_k_forced or intel_gemma_split_k_short_seq or attn_seq_len", 400);
}

test "Vulkan Qwen dense prefill padding covers short dense-hybrid DP4a shapes" {
    const src = @embedFile("compute/forward.zig");
    const marker = "fn qwen36DensePrefillPaddedTokenCount";
    try expectContainsNear(src, marker, "isQwenDenseHybridLayerMajorPrefillModel", 500);
    try expectContainsNear(src, marker, "const min_dp4a_tokens: u32 = 32;", 500);
    try expectContains(src, "fn qwen36DenseFfnPrefillPaddedTokenCount");
    try expectContains(src, "pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n40");
    try expectContains(src, "fn qwen36SsmPrefillPaddedTokenCount");
    try expectContains(src, "pipeline_mul_mm_q6k_full_dp4a_q8_1_k5120_n40");
    try expectContains(src, "pipeline_mul_mm_q5k_full_dp4a_k6144_n40");
    try expectContains(src, "return 40;");
}

test "Vulkan Qwen dense gate-up DP4a keeps K5120 specializations" {
    const src = @embedFile("compute/dmmv.zig");
    try expectContains(src, "const spec_k_5120 = [_]pipeline_mod.SpecConst{.{ .id = 0, .value = 5120 }};");
    try expectContains(src, "const spec_k_5120_n40 = [_]pipeline_mod.SpecConst{");
    try expectContains(src, "const spec_k_5120_n64_ragged = [_]pipeline_mod.SpecConst{");
    try expectContains(src, "pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64");
    try expectContains(src, "pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_ragged");
    try expectContains(src, "pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n40");
    try expectContains(src, "pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64");
    try expectContains(src, "pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_ragged");
    try expectContains(src, "pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n40");
    try expectContains(src, "K == 5120 and n_tile == 40");
    try expectContains(src, "K == 5120 and n_tile == 64");
    try expectContainsNear(src, "pub fn recordMulMmQ4KGateUpSwigluFullDp4aQ8(", "N >= 64 and (N & 63) == 0", 2200);
    try expectContainsNear(src, "pub fn recordMulMmQ4KGateUpSwigluFullDp4aQ8_1(", "N >= 64 and (N & 63) == 0", 2200);
    try expectContainsNear(src, "pub fn recordMulMmQ4KGateUpSwigluFullDp4aQ8(", "use_ragged_n64", 2200);
    try expectContainsNear(src, "pub fn recordMulMmQ4KGateUpSwigluFullDp4aQ8_1(", "use_ragged_n64", 2200);
}

test "Vulkan Gemma dense decode keeps fused GEGLU gate-up pair path" {
    const src = @embedFile("compute/forward.zig");
    const marker = "const gemma_dense_geglu_pair_eligible =";
    try expectContainsNear(src, marker, "pipeline_q4k_fused_gate_up_geglu_pair != null", 500);
    try expectContainsNear(src, marker, "config.architecture == .gemma", 500);
    try expectContainsNear(src, marker, "gate_tensor.info.type_ == .q4_k", 600);
    try expectContainsNear(src, marker, "try self.dispatchDmmvFusedGateUpGegluPair", 2200);
}

test "Vulkan Gemma dense decode keeps BN8 DP4a packed GEGLU path" {
    const src = @embedFile("compute/forward.zig");
    try expectContains(src, "ZINC_GEMMA_DENSE_DECODE_DP4A");
    try expectContainsNear(src, "fn gemmaDenseDecodeDp4aSupported", "hidden_dim != 5376 or inter_dim != 21504", 1200);
    try expectContainsNear(src, "fn gemmaDenseDecodeDp4aSupported", "cfg.architecture != .gemma or cfg.n_experts != 0 or cfg.ssm_d_inner != 0", 900);
    try expectContainsNear(src, "fn dispatchGemmaDenseDecodeGateUpDp4a", "try self.dmmv.recordQuantizeActQ8_1(", 2200);
    try expectContainsNear(src, "fn dispatchGemmaDenseDecodeGateUpDp4a", "try self.dmmv.recordMulMmQ4KGateUpGegluFullDp4aQ8(", 5200);
    try expectContainsNear(src, "fn dispatchGemmaDenseDecodeGateUpDp4a", "try self.dmmv.recordMulMmQ4KGateUpGegluFullDp4aQ8_1(", 7600);
    try expectContainsNear(src, "fn dispatchGemmaDenseDecodeDownDp4a", "try self.dmmv.recordMulMmQ6KTail8Dp4a(", 2200);
    try expectContainsNear(src, "fn dispatchGemmaDenseDecodeDownDp4a", "try self.dmmv.recordMulMmQ4KTail8Dp4a(", 3800);
    try expectContainsNear(src, "var gemma_decode_dp4a_activation: GemmaDecodeDp4aActivation = .none;", "try self.dispatchGemmaDenseDecodeGateUpDp4a(", 800);
    try expectContainsNear(src, "else if (use_fused_pfn_decode)", "try self.dispatchGemmaDenseDecodeDownDp4a(", 1800);
}

test "Vulkan Gemma prefill top-k cap can be tested without decode top-k cap" {
    const src = @embedFile("compute/forward.zig");
    const marker = "const gemma_prefill_topk_env =";
    try expectContainsNear(src, marker, "ZINC_GEMMA_MOE_PREFILL_TOPK", 200);
    try expectContainsNear(src, marker, "gemma_prefill_base_topk_limit", 900);
    try expectContainsNear(src, marker, "else if (gemma_topk_env != null)", 900);
    try expectContainsNear(src, "Gemma non-terminal prefill MoE top-k cap disabled by default on RDNA", "ZINC_GEMMA_MOE_PREFILL_TOPK", 200);
}

test "Vulkan Gemma grouped MoE prefill keeps exact top-k route buffers separate" {
    const src = @embedFile("compute/forward.zig");
    const marker = "fn prefillGemmaGroupedMoeExact";
    try expectContains(src, "ZINC_GEMMA_MOE_GROUPED_PREFILL");
    try expectContains(src, "fn prefillGemmaRunBatchedAttentionToFfnNorm");
    try expectContainsNear(src, "fn gemmaGroupedMoePrefillEnvEnabled", "orelse return true", 200);
    try expectContainsNear(src, "fn gemmaGroupedMoePrefillEnvEnabled", "std.ascii.eqlIgnoreCase(env, \"off\")", 500);
    try expectContainsNear(src, "fn gemmaGroupedMoePrefillEnabled", "isIntelGpuVendor(self.gpu_config.vendor)", 900);
    try expectContainsNear(src, "fn gemmaShortMoePrefixPrefillEnabled", "!self.isAmdRdna()", 700);
    try expectContainsNear(src, marker, "try self.prefillGemmaRunBatchedAttentionToFfnNorm", 18000);
    try expectContainsNear(src, marker, "try self.ensureGemmaMoePrefillDp4aScratchCapacity", 4200);
    try expectContainsNear(src, "fn ensureGemmaMoePrefillDp4aScratchCapacity", "batched_scratch_norm_q8_scale", 2400);
    try expectContainsNear(src, "fn prefillGemmaRunBatchedAttentionToFfnNorm", "try self.dispatchGemmaQkvProjectionsBatched", 9000);
    try expectContainsNear(src, "fn prefillGemmaRunBatchedAttentionToFfnNorm", "try self.dispatchFlashAttnBatched", 22000);
    try expectContains(src, "fn gemmaProjectionPrefillPaddedTokenCount");
    try expectContainsNear(src, "fn gemmaDenseProjectionDp4aEnabled", "cfg.n_experts != 0 and !gemmaGroupedMoePrefillEnvEnabled()", 900);
    try expectContainsNear(src, "fn gemmaDenseProjectionDp4aEnabled", "isIntelGpuVendor(self.gpu_config.vendor)", 400);
    try expectContainsNear(src, "fn gemmaDenseGegluDp4aEnabled", "isIntelGpuVendor(self.gpu_config.vendor)", 400);
    try expectContainsNear(src, "fn gemmaDenseDownDp4aEnabled", "isIntelGpuVendor(self.gpu_config.vendor)", 400);
    try expectContainsNear(src, "fn gemmaProjectionPrefillPaddedTokenCount", "isIntelGpuVendor(self.gpu_config.vendor)", 500);
    try expectContainsNear(src, "fn gemmaDensePrefillPaddedTokenCount", "isIntelGpuVendor(self.gpu_config.vendor)", 500);
    try expectContainsNear(src, "fn gemmaDenseProjectionDp4aSupported", ".q8_0 => (K & 31) == 0", 1400);
    try expectContainsNear(src, "fn gemmaDenseProjectionDp4aSupported", "self.batched_scratch_norm_q8", 1500);
    try expectContainsNear(src, "fn dispatchGemmaProjectionBatchedDp4a", "recordMulMmQ8_0FullDp4a", 7000);
    try expectContainsNear(src, "fn gemmaDenseGegluDp4aEnabled", "cfg.n_experts != 0 and !gemmaGroupedMoePrefillEnvEnabled()", 900);
    try expectContainsNear(src, marker, "const scratch_route_ids = scratch_shared_up;", 2600);
    try expectContainsNear(src, marker, "if (scratch_route_ids.size < route_pack_ids_bytes) return error.BufferTooSmall;", 11200);
    try expectContainsNear(src, marker, "scratch_route_ids.handle", 15500);
    try expectContainsNear(src, marker, "try self.dispatchMoeWeightedAccScaledBatch", 20000);
    try expectContainsNear(src, marker, "try self.gemmaPrepareProjectionQ8(scratch_shared_norm", 26000);
    try expectContainsNear(src, marker, "try self.gemmaPrepareProjectionQ8(scratch_swiglu", 30000);
    try expectContainsNear(src, marker, "const enable_gpu_phase_timing =", 9200);
    try expectContainsNear(src, marker, "self.resetTimestamps();", 13600);
    try expectContainsNear(src, "fn prefillBatchedImpl", "return self.prefillGemmaGroupedMoeExact(state, prompt_tokens);", 1800);
}

test "Vulkan Gemma grouped MoE prefill wires Q5_1 route-column down projection" {
    const dmmv = @embedFile("compute/dmmv.zig");
    try expectContains(dmmv, "pipeline_q5_1_moe_cols");
    try expectContains(dmmv, "dmmv_q5_1_moe_cols.spv");
    try expectContainsNear(dmmv, "recordMoeColsDispatchIndirect", ".q5_1 => if (self.pipeline_q5_1_moe_cols)", 1800);

    const build = try std.fs.cwd().readFileAlloc(std.testing.allocator, "build.zig", 1024 * 1024);
    defer std.testing.allocator.free(build);
    try expectContains(build, "\"dmmv_q5_1_moe_cols\"");
    try expectContains(build, "\"moe_weighted_acc_scaled_batch\"");

    const q5_cols = @embedFile("shaders/dmmv_q5_1_moe_cols.comp");
    try expectContains(q5_cols, "Q5_1_BYTES = 24u");
    try expectContains(q5_cols, "ROWS_PER_WG = 8u");
    try expectContains(q5_cols, "LANES_PER_ROW = 8u");
    try expectContains(q5_cols, "const float w0 = d * float(lo | (bit_lo << 4)) + m;");
    try expectContains(q5_cols, "x_route_divisor");
}

test "Vulkan Qwen dense-down DP4a keeps K17408 BN40 and BN64 specializations" {
    const src = @embedFile("compute/dmmv.zig");
    try expectContains(src, "const spec_k_17408_n40_bk2 = [_]pipeline_mod.SpecConst{");
    try expectContains(src, "const spec_k_17408_n64 = [_]pipeline_mod.SpecConst{");
    try expectContains(src, "const spec_k_17408_n64_bk2 = [_]pipeline_mod.SpecConst{");
    try expectContains(src, "const spec_k_17408_n64_bk2_acc = [_]pipeline_mod.SpecConst{");
    try expectContains(src, "const spec_k_17408_n64_ragged = [_]pipeline_mod.SpecConst{");
    try expectContains(src, "const spec_k_17408_n64_bk2_ragged = [_]pipeline_mod.SpecConst{");
    try expectContainsNear(src, "const spec_k_17408_n64_bk2 = [_]pipeline_mod.SpecConst{", ".{ .id = 3, .value = 2 },", 260);
    try expectContainsNear(src, "const spec_k_17408_n64_bk2_acc = [_]pipeline_mod.SpecConst{", ".{ .id = 4, .value = 1 },", 320);
    try expectContainsNear(src, "const spec_k_17408_n64_bk2_ragged = [_]pipeline_mod.SpecConst{", ".{ .id = 2, .value = 1 },", 320);
    try expectContainsNear(src, "const spec_k_17408_n64_bk2_ragged = [_]pipeline_mod.SpecConst{", ".{ .id = 3, .value = 2 },", 360);
    try expectContains(src, "pipeline_mul_mm_q6k_full_dp4a_k17408_n40");
    try expectContains(src, "pipeline_mul_mm_q6k_full_dp4a_k17408_n64");
    try expectContains(src, "pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bk2");
    try expectContains(src, "pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bm64");
    try expectContains(src, "pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bm64_acc");
    try expectContains(src, "pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bk2_acc");
    try expectContains(src, "pipeline_mul_mm_q6k_full_dp4a_k17408_n64_ragged");
    try expectContains(src, "pipeline_mul_mm_q6k_full_dp4a_k17408_n64_ragged_bm64");
    try expectContains(src, "pipeline_mul_mm_q4k_full_dp4a_k17408_n40");
    try expectContains(src, "pipeline_mul_mm_q4k_full_dp4a_k17408_n64");
    try expectContains(src, "pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bk2");
    try expectContains(src, "pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bm64");
    try expectContains(src, "pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bm64_acc");
    try expectContains(src, "pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bk2_acc");
    try expectContains(src, "pipeline_mul_mm_q4k_full_dp4a_k17408_n64_ragged");
    try expectContains(src, "pipeline_mul_mm_q4k_full_dp4a_k17408_n64_ragged_bm64");
    try expectContains(src, "K == 17408 and n_tile == 40");
    try expectContains(src, "K == 17408 and n_tile == 64");
    try expectContains(src, "K == 17408 and N == 64");
    try expectContains(src, "K == 17408 and N >= 64 and (N & 63) == 0");
    try expectContains(src, "K == 17408 and N > 64 and (N & 63) != 0");
    try expectContains(src, "N / n_tile");
    try expectContains(src, "(N + n_tile - 1) / n_tile");
    try expectContains(src, "use_n64_bm64");
    try expectContains(src, "use_exact_n64_bm64_acc");
    try expectContains(src, "use_ragged_n64_bm64");
    try expectContains(src, "if (use_n64_bm64 or use_k21504_n64_bm64 or use_exact_n64_mmq64_acc or use_exact_n64_bm64_acc or use_ragged_n64_bm64) M / 64 else M / 32");
    try expectContainsNear(src, "pub fn recordMulMmQ6KFullDp4a(", "use_exact_n64_bk2", 2200);
    try expectContainsNear(src, "pub fn recordMulMmQ6KFullDp4a(", "use_exact_n64_acc", 2200);
    try expectContainsNear(src, "pub fn recordMulMmQ6KFullDp4a(", "use_ragged_n64", 2200);
    try expectContainsNear(src, "pub fn recordMulMmQ4KFullDp4a(", "use_exact_n64_bk2", 2400);
    try expectContainsNear(src, "pub fn recordMulMmQ4KFullDp4a(", "use_exact_n64_acc", 2400);
    try expectContainsNear(src, "pub fn recordMulMmQ4KFullDp4a(", "use_ragged_n64", 2400);

    const forward = @embedFile("compute/forward.zig");
    try expectContains(forward, "fn qwenDenseDownDp4aAccEligible(");
    try expectContains(forward, "inter_dim != 17408 or n_tokens != 64 or full_cols != n_tokens");
    try expectContains(forward, ".q4_k => has_q4_k17408_acc");
    try expectContains(forward, ".q6_k => has_q6_k17408_acc");
    try expectContains(forward, "self.dmmv.pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bk2_acc != null");
    try expectContains(forward, "self.dmmv.pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bk2_acc != null");
    try expectContains(forward, "const down_out = if (accumulate_down) accum_target.? else scratch_down;");
    try expectContains(forward, "return accumulate_down;");
}

test "Vulkan Qwen SSM DP4a keeps BN40 and BN64 specializations" {
    const src = @embedFile("compute/dmmv.zig");
    try expectContains(src, "pipeline_mul_mm_q6k_full_dp4a_q8_1_k5120_n64_ragged");
    try expectContains(src, "pipeline_mul_mm_q6k_full_dp4a_q8_1_k5120_n40");
    try expectContains(src, "pipeline_mul_mm_q4k_full_dp4a_k5120_n64_ragged");
    try expectContains(src, "pipeline_mul_mm_q4k_full_dp4a_k5120_n40");
    try expectContains(src, "pipeline_mul_mm_q5k_full_dp4a_k6144_n40");
    try expectContains(src, "K == 5120 and n_tile == 64");
    try expectContains(src, "K == 5120 and N > 64 and (N & 63) != 0");
    try expectContains(src, "K == 5120 and n_tile == 40");
    try expectContains(src, "K == 6144 and n_tile == 40");
    try expectContainsNear(src, "pub fn recordMulMmQ6KFullDp4aQ8_1(", "use_ragged_n64", 1800);
    try expectContainsNear(src, "pub fn recordMulMmQ4KFullDp4a(", "use_k5120_ragged_n64", 2400);

    const q6_q8_1 = @embedFile("shaders/mul_mm_q6k_full_dp4a_q8_1.comp");
    try expectContains(q6_q8_1, "layout(constant_id = 2) const uint SPEC_RAGGED_N = 0u;");
    try expectContains(q6_q8_1, "const bool col_ok = SPEC_RAGGED_N == 0u || col_global < N;");
    try expectContains(q6_q8_1, "col_ok ? b_packed[src + p] : 0u");
    try expectContains(q6_q8_1, "if (SPEC_RAGGED_N == 0u || col_g < N)");
}

test "Vulkan Gemma dense-down DP4a keeps K21504 short-prompt specializations" {
    const dmmv = @embedFile("compute/dmmv.zig");
    try expectContains(dmmv, "pipeline_mul_mm_q6k_full_dp4a_k21504_n64_bm64");
    try expectContains(dmmv, "pipeline_mul_mm_q6k_full_dp4a_k21504_n64");
    try expectContains(dmmv, "pipeline_mul_mm_q6k_full_dp4a_k21504_n72");
    try expectContains(dmmv, "pipeline_mul_mm_q4k_full_dp4a_k21504_n64");
    try expectContains(dmmv, "pipeline_mul_mm_q4k_full_dp4a_k21504_n8");
    try expectContains(dmmv, "const pipeline_mul_mm_q6k_full_dp4a_k21504_n64_bm64");
    try expectContains(dmmv, "const pipeline_mul_mm_q6k_full_dp4a_k21504_n64");
    try expectContains(dmmv, "const pipeline_mul_mm_q6k_full_dp4a_k21504_n72");
    try expectContains(dmmv, "const pipeline_mul_mm_q4k_full_dp4a_k21504_n64");
    try expectContains(dmmv, "const use_k21504_n64_bm64");
    try expectContains(dmmv, "K == 21504 and n_tile == 64");

    const forward = @embedFile("compute/forward.zig");
    try expectContains(forward, "self.gemmaDenseQ4RaggedTailDp4aEnabled(down_t, n_tokens)");
    try expectContains(forward, "try self.dmmv.recordMulMmQ4KTail8Dp4a(");
}

test "Vulkan Gemma Q4_K LM-head DP4a path stays opt-in" {
    const dmmv = @embedFile("compute/dmmv.zig");
    try expectContains(dmmv, "pipeline_mul_mm_q4k_full_dp4a_k2816_n8");
    try expectContains(dmmv, "pipeline_mul_mm_q4k_full_dp4a_k5376_n8");
    try expectContains(dmmv, "const spec_k_2816_n8");
    try expectContains(dmmv, "const spec_k_5376_n8");
    try expectContainsNear(dmmv, "pub fn recordMulMmQ4KTail8Dp4a", "K == 2816", 1200);
    try expectContainsNear(dmmv, "pub fn recordMulMmQ4KTail8Dp4a", "K == 5376", 1400);

    const forward = @embedFile("compute/forward.zig");
    try expectContains(forward, "ZINC_Q4K_LM_HEAD_DP4A");
    try expectContains(forward, "q8_1_act_packed_buf");
    try expectContainsNear(forward, "fn dispatchQ4KLmHeadDp4a", "K != 2816 and K != 5376", 900);
    try expectContainsNear(forward, "fn dispatchQ4KLmHeadDp4a", "pipeline_mul_mm_q4k_full_dp4a_k5376_n8", 1400);
    try expectContainsNear(forward, "fn dispatchQ4KLmHeadDp4a", "try self.dmmv.recordQuantizeActQ8_1(", 2400);
    try expectContainsNear(forward, "fn dispatchQ4KLmHeadDp4a", "try self.dmmv.recordMulMmQ4KTail8Dp4a(", 4200);
    try expectContainsNear(forward, "const use_q8_1_lm_path =", "try self.dispatchQ4KLmHeadDp4a(", 700);
}

test "Vulkan full-DP4a wide shaders load every activation half tile safely" {
    const q4 = @embedFile("shaders/mul_mm_q4k_full_dp4a.comp");
    const q6 = @embedFile("shaders/mul_mm_q6k_full_dp4a.comp");
    const q6_q8_1 = @embedFile("shaders/mul_mm_q6k_full_dp4a_q8_1.comp");
    const q5 = @embedFile("shaders/mul_mm_q5k_full_dp4a.comp");
    const gate_q8 = @embedFile("shaders/mul_mm_q4k_gate_up_swiglu_full_dp4a_q8.comp");
    const gate_q8_1 = @embedFile("shaders/mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1.comp");
    for ([_][]const u8{ q4, q6, q6_q8_1, q5, gate_q8, gate_q8_1 }) |src| {
        try expectContains(src, "layout(constant_id = 1) const uint SPEC_BN = 32u;");
        try expectContains(src, "for (uint cbase = 0u; cbase < BN; cbase += 32u)");
        try expectContains(src, "const uint col_local = cbase + tid / 2u;");
        try expectContains(src, "if (col_local < BN)");
    }
}

test "Vulkan BM64 DP4a down shaders use 128-thread column loaders" {
    const q4 = @embedFile("shaders/mul_mm_q4k_full_dp4a_bm64_n64_acc.comp");
    const q6 = @embedFile("shaders/mul_mm_q6k_full_dp4a_bm64_n64_acc.comp");
    for ([_][]const u8{ q4, q6 }) |src| {
        try expectContains(src, "const uint BM = 64u;");
        try expectContains(src, "const uint WG_SIZE = 128u;");
        try expectContains(src, "layout(constant_id = 1) const uint SPEC_BN = 64u;");
        try expectContains(src, "layout(constant_id = 3) const uint SPEC_BK_STEP = 2u;");
        try expectContains(src, "layout(constant_id = 4) const uint SPEC_ACCUMULATE = 0u;");
        try expectContains(src, "const uint col_local = tid / 2u;");
        try expectNotContains(src, "for (uint cbase = 0u; cbase < BN; cbase += 32u)");
        try expectContains(src, "if (SPEC_ACCUMULATE != 0u)");
        try expectContains(src, "d_data[out_idx] += sums[m][n];");
        try expectContains(src, "d_data[out_idx] = sums[m][n];");
    }
}

test "Vulkan BM64 gate-up producers keep per-32 Q8 output blocks" {
    const q8 = @embedFile("shaders/mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_bm64.comp");
    const q8_1 = @embedFile("shaders/mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_bm64.comp");
    for ([_][]const u8{ q8, q8_1 }) |src| {
        try expectContains(src, "const uint BM = 64u;");
        try expectContains(src, "const uint WG_SIZE = 128u;");
        try expectContains(src, "const uint Q8_BLOCK_THREADS = 8u;");
        try expectContains(src, "const uint col_local = tid / 2u;");
        try expectNotContains(src, "for (uint cbase = 0u; cbase < BN; cbase += 32u)");
        try expectContains(src, "const uint q8_block = tiwr / Q8_BLOCK_THREADS;");
        try expectContains(src, "const uint q8_lane = tiwr & (Q8_BLOCK_THREADS - 1u);");
        try expectContains(src, "subgroupClusteredMax(local_max, Q8_BLOCK_THREADS)");
        try expectContains(src, "ir * (BM / 4u) + tiwr");
        try expectContains(src, "ir * (BM / 32u) + q8_block");
    }
    try expectContains(q8_1, "subgroupClusteredAdd(local_isum, Q8_BLOCK_THREADS)");
}

test "Vulkan Qwen gate-up BM64 path is isolated to K5120 N64 dispatches" {
    const dmmv = @embedFile("compute/dmmv.zig");
    try expectContains(dmmv, "pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_bm64");
    try expectContains(dmmv, "pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_ragged_bm64");
    try expectContains(dmmv, "pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_bm64");
    try expectContains(dmmv, "pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_ragged_bm64");
    try expectContains(dmmv, "const m_tile: u32 = if (use_bm64_n64 or use_bm64_ragged_n64) 64 else 32;");
    try expectContains(dmmv, "M / m_tile");
}

test "Vulkan Qwen DP4a ragged BN64 shaders guard inactive columns" {
    const q4 = @embedFile("shaders/mul_mm_q4k_full_dp4a.comp");
    const q6 = @embedFile("shaders/mul_mm_q6k_full_dp4a.comp");
    const gate_q8 = @embedFile("shaders/mul_mm_q4k_gate_up_swiglu_full_dp4a_q8.comp");
    const gate_q8_1 = @embedFile("shaders/mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1.comp");

    try expectContains(q4, "layout(constant_id = 2) const uint SPEC_RAGGED_N = 0u;");
    try expectContains(q6, "layout(constant_id = 2) const uint SPEC_RAGGED_N = 0u;");
    for ([_][]const u8{ gate_q8, gate_q8_1 }) |src| {
        try expectContains(src, "layout(constant_id = 3) const uint SPEC_RAGGED_N = 0u;");
    }

    for ([_][]const u8{ q4, q6, gate_q8, gate_q8_1 }) |src| {
        try expectContains(src, "const bool col_ok = SPEC_RAGGED_N == 0u || col_global < N;");
        try expectContains(src, "col_ok ? b_packed[src +");
        try expectContains(src, "SPEC_RAGGED_N == 0u");
    }

    for ([_][]const u8{ q4, gate_q8, gate_q8_1 }) |src| {
        try expectContains(src, "if ((BN >= 32u && SPEC_RAGGED_N == 0u) || col_g < N)");
    }
    try expectContains(q6, "if (SPEC_RAGGED_N == 0u || col_g < N)");
}

test "Vulkan full-DP4a prefill shaders expose guarded two-slice K staging" {
    const q4 = @embedFile("shaders/mul_mm_q4k_full_dp4a.comp");
    const q5 = @embedFile("shaders/mul_mm_q5k_full_dp4a.comp");
    const q6 = @embedFile("shaders/mul_mm_q6k_full_dp4a.comp");
    const q6_q8_1 = @embedFile("shaders/mul_mm_q6k_full_dp4a_q8_1.comp");
    for ([_][]const u8{ q4, q5, q6, q6_q8_1 }) |src| {
        try expectContains(src, "layout(constant_id = 3) const uint SPEC_BK_STEP = 1u;");
        try expectContains(src, "const uint BK_STEP = SPEC_BK_STEP;");
        try expectContains(src, "for (uint k_outer = 0u; k_outer < k_limit; k_outer += BK * BK_STEP)");
        try expectContains(src, "shared uint  buf_a[BK_STEP * BM * SPACK]");
        try expectContains(src, "shared uint  buf_b[BK_STEP * BN * SPACK]");
        try expectContains(src, "[[unroll]] for (uint ks = 0u; ks < BK_STEP; ks++)");
    }
    for ([_][]const u8{ q4, q6 }) |src| {
        try expectContains(src, "layout(constant_id = 4) const uint SPEC_ACCUMULATE = 0u;");
        try expectContains(src, "if (SPEC_ACCUMULATE != 0u)");
        try expectContains(src, "d_data[out_idx] += sums[m][n];");
    }

    const dispatch = @embedFile("compute/dmmv.zig");
    try expectContains(dispatch, "const spec_k_5120_n40_bk2 = [_]pipeline_mod.SpecConst{");
    try expectContains(dispatch, "const spec_k_6144_n40_bk2 = [_]pipeline_mod.SpecConst{");
    try expectContains(dispatch, "const spec_k_17408_n40_bk2 = [_]pipeline_mod.SpecConst{");
    try expectContains(dispatch, ".{ .id = 3, .value = 2 },");
    try expectContainsNear(dispatch, "const pipeline_mul_mm_q6k_full_dp4a_q8_1_k5120_n40", "&spec_k_5120_n40_bk2", 300);
    try expectContainsNear(dispatch, "const pipeline_mul_mm_q4k_full_dp4a_k5120_n40", "&spec_k_5120_n40_bk2", 300);
    try expectContainsNear(dispatch, "const pipeline_mul_mm_q5k_full_dp4a_k6144_n40", "&spec_k_6144_n40_bk2", 300);
    try expectContainsNear(dispatch, "const pipeline_mul_mm_q6k_full_dp4a_k17408_n40", "&spec_k_17408_n40_bk2", 300);
    try expectContainsNear(dispatch, "const pipeline_mul_mm_q4k_full_dp4a_k17408_n40", "&spec_k_17408_n40_bk2", 300);
}

test "Vulkan batched kpar shaders merge cross-subgroup partials" {
    const q4 = @embedFile("shaders/dmmv_q4k_batch_kpar.comp");
    const q6 = @embedFile("shaders/dmmv_q6k_batch_kpar.comp");
    for ([_][]const u8{ q4, q6 }) |src| {
        try expectContains(src, "shared float s_sg_sums[4];");
        try expectContains(src, "gl_NumSubgroups > 1u");
        try expectContains(src, "subgroupElect()");
        try expectContains(src, "s_sg_sums[gl_SubgroupID]");
        try expectContains(src, "barrier();");
    }
}

test "Vulkan batched kpar pipelines use non-wave64 options on Intel" {
    const src = @embedFile("compute/dmmv.zig");
    try expectContainsNear(src, "const q4k_batch_kpar_path", "effective_wave64_options", 700);
    try expectContainsNear(src, "const q6k_batch_kpar_path", "effective_wave64_options", 700);
}

test "Vulkan Intel batched prefill keeps chunk override for fallback debugging" {
    const src = @embedFile("compute/forward.zig");
    try expectContains(src, "ZINC_INTEL_BATCHED_PREFILL_CHUNK");
    try expectContainsNear(src, "fn intelBatchedPrefillChunkLimit", "orelse return 96;", 500);
    try expectContainsNear(src, "fn intelBatchedPrefillChunkLimit", "if (std.mem.eql(u8, raw, \"0\")) return 0;", 700);
    try expectContainsNear(src, "pub fn prefillBatched(self: *InferenceEngine", "intelBatchedPrefillChunkLimit", 1200);
    try expectContainsNear(src, "Intel batched prefill chunking ENABLED", "prefillBatchedImpl(state, prompt_tokens[offset..end])", 1200);
}

test "Vulkan Intel SSM fast paths are default-off with env opt-in" {
    const src = @embedFile("compute/forward.zig");
    try expectContains(src, "const intel_ssm_fast_paths_default_off = isIntelGpuVendor(gpu_config.vendor) and config.ssm_d_inner > 0;");
    try expectContainsNear(src, "const fused_ssm_ab_policy_enabled", "fused_ssm_ab_forced_on", 500);
    try expectContainsNear(src, "const ssm_delta_cols8_policy_enabled", "ssm_delta_cols8_forced_on", 500);
    try expectContainsNear(src, "Fused SSM pre-norm DISABLED by default on Intel SSM path", "ZINC_FUSED_SSM_AB=1", 200);
    try expectContainsNear(src, "SSM delta cols8 DISABLED by default on Intel SSM path", "ZINC_SSM_DELTA_COLS8=1", 200);
}

test "Vulkan Qwen 3.6 MoE top-k caps default to full metadata top-k on Intel" {
    const src = @embedFile("compute/forward.zig");
    try expectContains(src, "const qwen36_moe_intel_safe_defaults = qwen36_like_f32_ssm and isIntelGpuVendor(gpu_config.vendor);");
    try expectContains(src, "const qwen36_topk_default: u32 = if (qwen36_moe_intel_safe_defaults) 0 else 3;");
    try expectContains(src, "const qwen36_prefill_topk_default: u32 = if (qwen36_moe_intel_safe_defaults) 0 else 1;");
    try expectContains(src, "Qwen 3.6 MoE top-k cap disabled by default on Intel");
    try expectContains(src, "Qwen 3.6 non-terminal prefill MoE top-k cap disabled by default on Intel");
}

test "Vulkan prefillBatched uses all batched primitives in the per-layer loop" {
    const src = @embedFile("compute/forward.zig");
    const fn_marker = "fn prefillBatchedImpl(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {";
    try expectContainsNear(src, fn_marker, "dispatchProjectionBatched", 24000);
    try expectContainsNear(src, fn_marker, "dispatchRopeBatched", 24000);
    try expectContainsNear(src, fn_marker, "dispatchKvCacheWriteBatched", 24000);
    try expectContainsNear(src, fn_marker, "dispatchFlashAttnBatched", 24000);
    try expectContainsNear(src, fn_marker, "dispatchResidualRmsNorm", 30000);
    try expectContainsNear(src, fn_marker, "dispatchFfnActivation", 24000);
    try expectContainsNear(src, fn_marker, "dispatchDmmvInner", 24000);
}

test "Vulkan prefillBatched threads base_token through RoPE, KV write, flash attn" {
    const src = @embedFile("compute/forward.zig");
    const fn_marker = "fn prefillBatchedImpl(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {";
    try expectContainsNear(src, fn_marker, "const base_token: u32 = state.position;", 6000);
    try expectContainsNear(src, fn_marker, "state.position = base_token + n_tokens;", 24000);
}

test "Vulkan prefillBatched avoids full logits copy for greedy GPU argmax" {
    const src = @embedFile("compute/forward.zig");
    const marker = "// Read back only what the sampler needs.";
    try expectContainsNear(src, marker, "vkCmdCopyBuffer(self.decode_cmd.handle, self.argmax_result_buf.handle", 1200);
    try expectContainsNear(src, marker, "const need_logits_readback = validate_mode or self.logits_readback_enabled or self.validation_diagnostics_enabled or !have_gpu_argmax;", 1600);
    try expectContainsNear(src, marker, "if (need_logits_readback) {", 1800);
    try expectContainsNear(src, marker, "vkCmdCopyBuffer(self.decode_cmd.handle, self.logits_buf.handle", 2200);
}

test "Vulkan batched KV write shader uses page_table with base_token offset" {
    const src = @embedFile("shaders/kv_cache_write_batched.comp");
    try expectContains(src, "uint base_token;");
    try expectContains(src, "uint logical_token = base_token + tok;");
    try expectContains(src, "page_ids[page_idx]");
}

test "Vulkan residual_rms_norm shader matches Metal's fused semantics" {
    const src = @embedFile("shaders/residual_rms_norm.comp");
    try expectContains(src, "hidden[base + i] = h");
    try expectContains(src, "subgroupAdd");
    try expectContains(src, "norm_out[base + i] = weights[i] * hidden[base + i] * rms_inv;");
}

test "Vulkan Gemma decode keeps post-norm residual next-norm fusions wired" {
    const forward = @embedFile("compute/forward.zig");
    try expectContains(forward, "use_fused_pan_ffn_norm_decode");
    try expectContainsNear(forward, "} else if (use_fused_pan_ffn_norm_decode)", "try self.dispatchPostNormResidualRmsNorm", 900);
    try expectContainsNear(forward, "} else if (use_fused_pan_ffn_norm_decode)", "ffn_norm_ready = true", 1400);
    try expectContains(forward, "gemma_dense_next_attn_norm_ready");
    try expectContainsNear(forward, "can_fuse_dense_tail_next_attn_norm", "self.layer_output_scales[layer]", 2400);
    try expectContainsNear(forward, "if (gemma_dense_tail_wrote_next_attn_norm)", "self.norm_buf.handle", 700);
}

test "Vulkan post_norm_residual_rms_norm shader folds Gemma layer scale" {
    const elementwise = @embedFile("compute/elementwise.zig");
    try expectContains(elementwise, "hidden_scale: f32 = 1.0");

    const shader = @embedFile("shaders/post_norm_residual_rms_norm.comp");
    try expectContains(shader, "float hidden_scale;");
    try expectContains(shader, "vec4 h_store = apply_hidden_scale ? (h * hidden_scale) : h;");
    try expectContains(shader, "hidden[base + i + 0u] = h_store.x;");
    try expectContains(shader, "if (apply_hidden_scale) h = h / hidden_scale;");
}

test "Vulkan Gemma norm-add vec4 path stays wired and wave64" {
    const build = try std.fs.cwd().readFileAlloc(std.testing.allocator, "build.zig", 1024 * 1024);
    defer std.testing.allocator.free(build);
    try expectContains(build, "\"rms_norm_add_vec4\"");
    try expectNotContains(build, "dmmv_q4k_q8_1_fused_gate_up_geglu_pair");

    const elementwise = @embedFile("compute/elementwise.zig");
    try expectContains(elementwise, "pipeline_rms_norm_add_vec4: ?Pipeline");
    try expectContains(elementwise, "rms_norm_add_vec4.spv");
    try expectContains(elementwise, ".pipeline_rms_norm_add_vec4 = pipeline_rms_norm_add_vec4");
    try expectContains(elementwise, "if (self.pipeline_rms_norm_add_vec4) |*p| p.deinit();");

    const forward = @embedFile("compute/forward.zig");
    const marker = "fn dispatchRmsNormAdd(";
    try expectContainsNear(forward, marker, "hidden_dim & 3", 1200);
    try expectContainsNear(forward, marker, "pipeline_rms_norm_add_vec4", 1200);
    try expectContainsNear(forward, marker, "pipeline_rms_norm_add orelse", 1200);
    try expectNotContains(forward, "dmmv_q4k_q8_1_fused_gate_up_geglu_pair");

    const shader = @embedFile("shaders/rms_norm_add_vec4.comp");
    try expectContains(shader, "layout(local_size_x = 64) in;");
    try expectContains(shader, "vec4 hidden_v4[]");
    try expectContains(shader, "const uint n4 = n >> 2u;");
    try expectContains(shader, "subgroupAdd");
    try expectContains(shader, "gl_NumSubgroups > 1u");
    try expectContains(shader, "hidden_v4[idx] +=");
}

test "Vulkan Q4_K Q8_1 DMMV path stays gated and does not route Q6_K" {
    const build = try std.fs.cwd().readFileAlloc(std.testing.allocator, "build.zig", 1024 * 1024);
    defer std.testing.allocator.free(build);
    try expectContains(build, "\"dmmv_q4k_q8_1\"");
    try expectNotContains(build, "\"dmmv_q6k_q8_1\"");
    try expectNotContains(build, "\"dmmv_q4k_pair_geglu_q8_1\"");

    const dmmv = @embedFile("compute/dmmv.zig");
    try expectContains(dmmv, "pipeline_q4k_q8_1: ?Pipeline");
    try expectContains(dmmv, "dmmv_q4k_q8_1.spv");
    try expectNotContains(dmmv, "pipeline_q6k_q8_1");
    try expectNotContains(dmmv, "pipeline_q4k_pair_geglu_q8_1");

    const forward = @embedFile("compute/forward.zig");
    try expectContains(forward, "ZINC_Q4K_Q8_1_DMMV");
    try expectContains(forward, "q4k_q8_1_explicitly_off");
    try expectContains(forward, "config.architecture == .gemma");
    try expectContains(forward, "config.n_experts == 0");
    try expectContains(forward, "config.hidden_dim == 5376");
    try expectContains(forward, "gpu_config.vendor == .amd_rdna4");
    try expectContains(forward, "std.ascii.eqlIgnoreCase(env, \"off\")");
    try expectContains(forward, "std.ascii.eqlIgnoreCase(env, \"false\")");
    try expectContains(forward, "std.ascii.eqlIgnoreCase(env, \"no\")");
    try expectContainsNear(forward, "if (self.use_q4k_q8_1_dmmv", "qt == .q4_k", 300);
    try expectNotContains(forward, "ZINC_GEMMA_Q4K_GEGLU_Q8_1");
    try expectNotContains(forward, "dispatchDmmvFusedGateUpGegluPairQ8_1");

    const shader = @embedFile("shaders/dmmv_q4k_q8_1.comp");
    try expectContains(shader, "layout(local_size_x = 64) in;");
    try expectContains(shader, "dotPacked4x8AccSatEXT");
    try expectContains(shader, "Q8_1_U32_PER_BLOCK = 9u");
}

test "Vulkan Gemma dense decode uses true single-token GEGLU producer" {
    const build = try std.fs.cwd().readFileAlloc(std.testing.allocator, "build.zig", 1024 * 1024);
    defer std.testing.allocator.free(build);
    try expectContains(build, "\"mul_mm_q4k_gate_up_geglu_n1_dp4a_q8\"");

    const dmmv = @embedFile("compute/dmmv.zig");
    try expectContains(dmmv, "pipeline_mul_mm_q4k_gate_up_geglu_n1_dp4a_q8: ?Pipeline");
    try expectContains(dmmv, "mul_mm_q4k_gateup_geglu_n1_dp4a_q8_path_buf");
    try expectNotContains(dmmv, "std.fmt.bufPrint(&path_buf, \"{s}/mul_mm_q4k_gate_up_geglu_n1_dp4a_q8.spv\"");
    try expectContains(dmmv, "mul_mm_q4k_gate_up_geglu_n1_dp4a_q8.spv");
    try expectContains(dmmv, "recordMulMmQ4KGateUpGegluN1Dp4aQ8");
    try expectContains(dmmv, ".pipeline_mul_mm_q4k_gate_up_geglu_n1_dp4a_q8 = pipeline_mul_mm_q4k_gate_up_geglu_n1_dp4a_q8");
    try expectContains(dmmv, "if (self.pipeline_mul_mm_q4k_gate_up_geglu_n1_dp4a_q8) |*p| p.deinit();");

    const forward = @embedFile("compute/forward.zig");
    const route = "fn dispatchGemmaDenseDecodeGateUpDp4a(";
    try expectContainsNear(forward, route, "pipeline_mul_mm_q4k_gate_up_geglu_n1_dp4a_q8", 5200);
    try expectContainsNear(forward, route, "recordMulMmQ4KGateUpGegluN1Dp4aQ8", 6400);

    const shader = @embedFile("shaders/mul_mm_q4k_gate_up_geglu_n1_dp4a_q8.comp");
    try expectContains(shader, "layout(local_size_x = 64) in;");
    try expectContains(shader, "const uint row_local = tid >> 1u;");
    try expectContains(shader, "const uint tile_half = tid & 1u;");
    try expectContains(shader, "subgroupMax(local_max)");
    try expectContains(shader, "d_packed[d_packed_offset + row_block * 8u + pack_idx] = packed;");
    try expectNotContains(shader, "const uint BN");
}

test "softmax_topk shader keeps RADV-safe shared-memory winner scan" {
    const src = @embedFile("shaders/softmax_topk.comp");
    try expectContains(src, "shared float s_local_val[64];");
    try expectContains(src, "shared uint  s_local_idx[64];");
    try expectContains(src, "for (uint t = 0; t < 64; t++)");
    try expectContains(src, "s_logits[global_idx] = -1.0 / 0.0;");
    try expectNotContains(src, "GL_KHR_shader_subgroup_ballot");
    try expectNotContains(src, "subgroupBroadcast(");
}

test "softmax_topk shader renormalizes only selected logits" {
    const src = @embedFile("shaders/softmax_topk.comp");
    try expectContains(src, "shared float s_logits[256];");
    try expectContains(src, "float max_logit = -1.0 / 0.0;");
    try expectContains(src, "exp(uintBitsToFloat(output_data[k + i]) - max_logit)");
    try expectNotContains(src, "shared float s_probs[256];");
}

test "flash_attn shader keeps multi-subgroup fallback" {
    const src = @embedFile("shaders/flash_attn.comp");
    try expectContains(src, "subgroupMax");
    try expectContains(src, "subgroupAdd");
    try expectMultiSubgroupFallback(src, "s_reduce_scalar");
}

test "ssm_delta_net shader keeps multi-subgroup fallback" {
    const src = @embedFile("shaders/ssm_delta_net.comp");
    try expectContains(src, "subgroupAdd");
    try expectMultiSubgroupFallback(src, "s_reduce_scalar");
}

test "Q5_K shader keeps separate low/high nibble unpack helpers" {
    // The k-parallel rewrite dropped the explicit `x_grp` indexing in favor of
    // two small unpack functions. The correctness invariant is the same: the
    // lo-nibble bytes and hi-nibble bytes of each qs uint32 are expanded
    // independently before being paired with qh-bit promotion, not multiplied
    // together in a single mis-shifted path. Guard the two unpack functions
    // and verify both end up in the hot loop.
    const src = @embedFile("shaders/dmmv_q5k.comp");
    try expectContains(src, "vec4 unpack_nibbles_lo(uint v)");
    try expectContains(src, "vec4 unpack_nibbles_hi(uint v)");
    try expectContains(src, "(v & 0xFu)");
    try expectContains(src, "(v >> 4) & 0xFu");
    try expectContains(src, "unpack_qh_bits");
}

test "Q5_K shader uses wave64 K-parallel layout with 2 rows per workgroup" {
    // Regression guard for the K-parallel shape: 64-thread workgroups, each
    // covering 2 output rows, with the super-block sized at 44 uint32 (176
    // bytes / 4). The previous non-k-parallel path used slice*4 loops and
    // silently dropped half the qs bytes; the new path sweeps the full block.
    const src = @embedFile("shaders/dmmv_q5k.comp");
    try expectContains(src, "layout(local_size_x = 64) in;");
    try expectContains(src, "const uint NUM_ROWS = 2u;");
    try expectContains(src, "const uint Q5K_BLOCK_U32 = 44u;");
    try expectContains(src, "subgroupAdd");
    try expectNotContains(src, "slice * 4u");
}

test "Q5_K MoE shader keeps GGML contiguous half ordering" {
    const src = @embedFile("shaders/dmmv_q5k_moe.comp");
    try expectContains(src, "x[x_grp + e]");
    try expectContains(src, "x[x_grp + 32u + e]");
    try expectNotContains(src, "2u * e");
}

test "Q5_0 and Q5_1 DMMV launch with 2 rows per workgroup" {
    const src = @embedFile("compute/dmmv.zig");
    // The single-column (non-batch, non-k-parallel) DMMV path must keep
    // Q5_0, Q5_1, MXFP4, Q8_0, and F16 on the 2-rows-per-workgroup layout
    // (M+1)/2 workgroups, each covering 2 output rows via simdgroup reduce.
    try expectContains(src, ".q5_0, .q5_1, .mxfp4, .q8_0, .f16 => (M + 1) / 2");
    // K-parallel branch is taken for Q4_K / Q5_K / Q6_K and also uses 2 rows
    // per workgroup, so the same (M+1)/2 shape applies.
    try expectContains(src, ".q4_k, .q5_k, .q6_k => (M + 1) / 2");
}

test "GPT-OSS routing keeps SOFTMAX_WEIGHT expert selection path" {
    const src = @embedFile("compute/forward.zig");
    try expectContains(src, "GPT-OSS uses this SOFTMAX_WEIGHT routing rule instead of softmax-over-all-experts.");
    try expectContains(src, "topKSoftmaxWeight(router_logits, n_used, expert_ids[0..n_used], expert_weights[0..n_used]);");
    try expectContains(src, "topKSoftmax(router_logits, n_used, expert_ids[0..n_used], expert_weights[0..n_used]);");
}

test "GPT-OSS FFN keeps OAI SwiGLU and bias-add dispatches" {
    const forward_src = @embedFile("compute/forward.zig");
    const elementwise_src = @embedFile("compute/elementwise.zig");
    try expectContainsNear(forward_src, "if (self.model.config.architecture == .gpt_oss) {", "return self.dispatchSwigluOai(gate_buf, gate_size, up_buf, up_size, output_buf, output_size, n_elements);", 200);
    try expectContains(forward_src, "try self.dispatchBiasAddSlice(self.gate_buf.handle, self.gate_buf.size, bias, eid * inter_dim, inter_dim);");
    try expectContains(forward_src, "try self.dispatchBiasAddSlice(self.down_buf.handle, hidden_size, bias, eid * hidden_dim, hidden_dim);");
    try expectContains(elementwise_src, "pub fn recordSwigluOai(");
    try expectContains(elementwise_src, "pub fn recordBiasAdd(");
}

test "IMROPE frequency uses global pair index, not per-section reset" {
    // Regression: IMROPE precomputation used per-section independent exponents,
    // resetting to 0 at each section boundary. For text IMROPE (all position IDs
    // equal), frequencies must use a single global progression: freq[k] = 1/base^(2k/rope_dim).
    // The per-section code caused pairs at section boundaries (11, 22) to get freq=1.0
    // instead of the correct monotonically decreasing values.
    const src = @embedFile("compute/forward.zig");
    // Must use total_pairs (global), not sec_pairs (per-section)
    try expectContains(src, "total_pairs = config.rope_sections[0] + config.rope_sections[1]");
    try expectContains(src, "for (0..total_pairs)");
    // Must NOT have per-section loop that resets exponents
    try expectNotContains(src, "for (0..sec_pairs)");
}

test "Metal Gemma embedding scaling applied before debug logging" {
    // Regression: Metal backend was missing sqrt(hidden_dim) embedding scaling
    // for Gemma models, causing ~62x smaller initial hidden states.
    const src = @embedFile("compute/forward_metal.zig");
    try expectContains(src, "Gemma models scale embeddings by sqrt(hidden_dim).");
    try expectContains(src, "config.architecture == .gemma");
}

test "Metal FFN norm prefers ffn_norm over post_attention_norm" {
    // Regression: Metal used post_attention_norm.weight as FFN norm (wrong for Gemma
    // where both exist). Must prefer ffn_norm.weight, falling back to post_attention_norm.
    const src = @embedFile("compute/forward_metal.zig");
    // The ffn_norm_bufs init should try ffn_norm FIRST
    try expectContainsNear(src, "FFN norm: prefer ffn_norm.weight", "findLayerTensor(model, layer, \"ffn_norm.weight\")", 200);
}

test "Metal supports Gemma post-attention and post-FFN norms" {
    // Regression: Metal was missing post_attention_norm and post_ffw_norm dispatches,
    // which Gemma requires for correctness.
    const src = @embedFile("compute/forward_metal.zig");
    try expectContains(src, "post_attn_norm_bufs");
    try expectContains(src, "post_ffw_norm_bufs");
    try expectContains(src, "post_ffw_norm.weight");
}

test "softmax_topk shader uses -inf for global_best init, not -1.0" {
    // Regression: softmax_topk used -1.0 as the initial value for the global
    // winner search. When router logits are all < -1.0, this silently selects
    // expert 0 instead of the actual best expert, corrupting MoE routing.
    const src = @embedFile("shaders/softmax_topk.comp");
    // global_best must be -inf
    try expectContains(src, "float global_best = -1.0 / 0.0;");
    // Must NOT use -1.0 as init
    try expectNotContains(src, "float global_best = -1.0;");
}

test "Metal loads GEGLU pipeline for Gemma activation" {
    // Regression: Metal used SwiGLU for all models, but Gemma requires GEGLU.
    const src = @embedFile("compute/forward_metal.zig");
    try expectContains(src, "geglu_pipe");
    try expectContains(src, "cfg.architecture == .gemma");
}

test "router_logits_buf sized for max(n_experts, ssm_dt_rank)" {
    // Regression: router_logits_buf was sized for n_experts=1 (non-MoE) but SSM alpha
    // projection writes dt_rank floats. Buffer overflow corrupted alpha[1..15].
    const src = @embedFile("compute/forward.zig");
    try expectContains(src, "@max(if (config.n_experts > 0) config.n_experts else @as(u32, 1), config.ssm_dt_rank)");
}

test "GPU SSM path enabled when all three shaders are available" {
    // The GPU SSM path requires conv1d + delta-net + gated_norm shaders.
    // Must NOT gate on architecture enum (qwen35 vs qwen2_moe confusion).
    const src = @embedFile("compute/forward.zig");
    try expectContains(src, "pipeline_ssm_conv1d != null");
    try expectContains(src, "pipeline_ssm_delta_net != null");
    try expectContains(src, "pipeline_ssm_gated_norm != null");
    try expectNotContains(src, "config.architecture != .qwen35");
    try expectNotContains(src, "!has_delta_net");
}

test "Vulkan SSM gated norm keeps long prefill token-parallel" {
    // Long layer-major prefills need the token-batched grid; the fused
    // token-loop shader is only a short-prompt dispatch-overhead optimization.
    const src = @embedFile("compute/forward.zig");
    try expectContains(src, "const prefer_fused_gnorm = n_tokens <= 128;");
    try expectContainsNear(src, "const prefer_fused_gnorm = n_tokens <= 128;", "pipeline_ssm_gated_norm_batch_tok) |*batch_pip|", 1800);
    try expectContainsNear(src, "pipeline_ssm_gated_norm_batch_tok) |*batch_pip|", "dt_rank,\n                    n_tokens,\n                    1,", 1400);
}

test "Vulkan FFN norm prefers ffn_norm over post_attention_norm" {
    // Same fix as Metal — Vulkan must also prefer ffn_norm.weight first.
    const src = @embedFile("compute/forward.zig");
    try expectContainsNear(src, "FFN norm: prefer ffn_norm.weight", "const ffn_norm_tensor = lt.ffn_norm orelse", 500);
    try expectContains(src, "lt.post_attention_norm orelse return error.TensorNotFound;");
}

test "Vulkan Gemma embedding scaling matches Metal" {
    // Both backends must scale Gemma embeddings by sqrt(hidden_dim).
    const vulkan_src = @embedFile("compute/forward.zig");
    try expectContains(vulkan_src, "Gemma models scale embeddings by sqrt(hidden_dim).");
}

test "Vulkan post-attention norm applied before attn residual" {
    // Gemma requires RMS norm on o_proj output before residual add.
    const src = @embedFile("compute/forward.zig");
    try expectContains(src, "Gemma post-attention norm: RMS norm on o_proj output before residual add");
}

test "Vulkan post-FFN norm applied before FFN residual" {
    // Gemma requires RMS norm on down_proj output before residual add.
    const src = @embedFile("compute/forward.zig");
    try expectContains(src, "Gemma post-FFN norm: RMS norm on down_proj output before residual add");
}

test "rope_sections loaded from GGUF metadata" {
    // IMROPE requires rope.dimension_sections from GGUF for Qwen3.5 models.
    const src = @embedFile("model/loader.zig");
    try expectContains(src, "rope.dimension_sections");
}

test "RoPE shader supports freq buffer path for IMROPE" {
    // When freq_base_bits=0, the RoPE shader reads precomputed frequencies from
    // binding 2 instead of computing from freq_base. This supports IMROPE and
    // proportional RoPE (Gemma 4).
    const src = @embedFile("shaders/rope_fused.comp");
    try expectContains(src, "freq_base_bits == 0u");
    try expectContains(src, "inv_freq[i]");
}

test "YaRN RoPE attention scale stays wired through Vulkan RoPE dispatch" {
    const forward_src = @embedFile("compute/forward.zig");
    const rope_src = @embedFile("shaders/rope_fused.comp");
    const norm_rope_src = @embedFile("shaders/norm_rope.comp");
    try expectContains(forward_src, "const rope_attn_scale = if (use_yarn_rope) effectiveRopeAttnScale(config) else 1.0;");
    try expectContains(forward_src, "const push = RopePush{\n                .stride = stride,\n                .rope_dim = rope_dim,\n                .n_heads = n_heads,\n                .position = position,\n                .freq_base_bits = @bitCast(freq_base),\n                .attn_scale_bits = @bitCast(attn_scale),");
    try expectContains(forward_src, "const push = NormRopePush{\n            .head_dim = head_dim,\n            .rope_dim = rope_dim,\n            .n_heads = n_heads,\n            .position = position,\n            .freq_base_bits = @bitCast(freq_base),\n            .attn_scale_bits = @bitCast(attn_scale),");
    try expectContains(rope_src, "float attn_scale = attn_scale_bits != 0u ? uintBitsToFloat(attn_scale_bits) : 1.0;");
    try expectContains(rope_src, "float cos_t = cos(theta) * attn_scale;");
    try expectContains(norm_rope_src, "float attn_scale = attn_scale_bits != 0u ? uintBitsToFloat(attn_scale_bits) : 1.0;");
    try expectContains(norm_rope_src, "float cos_t = cos(theta) * attn_scale;");
}

test "flash attention sink buffer stays in final normalization" {
    const src = @embedFile("shaders/flash_attn.comp");
    try expectContains(src, "layout(set = 0, binding = 5) readonly  buffer Sinks");
    // Sink indexing grew a per-layer offset when the multi-layer sink buffer
    // pattern landed; the invariant is that sink_val still feeds final_sum.
    try expectContains(src, "float sink_val = sink_data[sink_offset + head];");
    try expectContains(src, "final_sum = s_sum_old * rescale + exp(sink_val - sink_max);");
    try expectContains(src, "o_data_v4[o_base_v4 + d4] = s_out_v4[d4] * rescale * inv_sum;");
}

test "F32 DMMV uses K-parallel reduction via subgroupAdd" {
    // Performance: F32 DMMV must use K-parallel (1 row per workgroup, 64 threads
    // collaborate via subgroupAdd) instead of M-parallel (64 rows per workgroup,
    // 1 thread per row). K-parallel gives M workgroups instead of ceil(M/64),
    // dramatically improving GPU utilization for small M (MoE router, M=256).
    const src = @embedFile("shaders/dmmv_f32.comp");
    try expectContains(src, "subgroupAdd");
    try expectContains(src, "shared float s_x[SPEC_K]");
    try expectContains(src, "row = gl_WorkGroupID.x");
    // Must NOT use gl_GlobalInvocationID (old M-parallel pattern)
    try expectNotContains(src, "gl_GlobalInvocationID");
}

test "Q4_K MoE shader uses packed uint32 reads, not byte access" {
    // Performance: Q4_K MoE DMMV must use uint32 packed reads (36 u32 per block)
    // instead of uint8_t byte access (144 individual reads). The packed path
    // gives 4x fewer memory transactions and enables vec4 dot products.
    const src = @embedFile("shaders/dmmv_q4k_moe.comp");
    // Must use uint buffer, not uint8_t
    try expectContains(src, "uint a_u32[]");
    try expectNotContains(src, "uint8_t a_data[]");
    // Must use vec4 dot products
    try expectContains(src, "unpack_nibbles_lo");
    try expectContains(src, "unpack_nibbles_hi");
    try expectContains(src, "dot(vec4(factor_lo)");
    // Must NOT have individual byte reads
    try expectNotContains(src, "a_data[");
}

test "Q5_K MoE shader processes all 32 elements per sub-block pair" {
    // The MoE Q5_K shader must iterate e from 0 to 31 (not 0..15 like the
    // old dense Q5_K bug). Each sub-block pair has 32 bytes of qs data.
    const src = @embedFile("shaders/dmmv_q5k_moe.comp");
    try expectContains(src, "for (uint e = 0; e < 32; e++)");
}

test "chat UI derives the model link from the reported model name" {
    const src = @embedFile("server/chat.html");
    try expectContains(src, "const chatStateKey='zinc.chat.state.v3';");
    try expectContains(src, "function restoreChatState()");
    try expectContains(src, "function clearConversation()");
    try expectContains(src, "id=\"cb\" class=\"btn btn-clear\"");
    try expectContains(src, "@media (max-width:720px)");
    try expectContains(src, "function modelHrefForName(name)");
    try expectContains(src, "function switchableModels()");
    try expectContains(src, "function activeModel()");
    try expectContains(src, "function scheduleHealthRefresh(delay)");
    try expectContains(src, "function refreshHealth()");
    try expectContains(src, "setModelTag(d.model)");
    try expectContains(src, "setGpuMemory(d);");
    try expectContains(src, "restoreChatState();");
    try expectContains(src, "CB.addEventListener('click',clearConversation);");
    try expectContains(src, "setCurrentModel(current);");
    try expectContains(src, "await Promise.allSettled([refreshHealth(),refreshModels()]);");
    try expectContains(src, "fetch(base+'/models/activate'");
    try expectContains(src, "m.managed&&m.installed&&m.supported_on_current_gpu&&m.fits_current_gpu");
    try expectNotContains(src, "setCurrentModel(selectedModel());");
}

test "Metal flash_attn supports head_dim=512 for Gemma 4 global attention layers" {
    // Regression: FLASH_MAX_HEAD_DIM was 256. Gemma 4 has mixed attention where
    // SWA layers use head_dim=256 but global layers (every 6th) use head_dim=512.
    // The threadgroup arrays were too small and per-thread loops used `if (tid < vec4_dim)`
    // instead of strided loops, leaving the second half of Q/acc uninitialized → NaN.
    const src = @embedFile("shaders/metal/flash_attn.metal");
    try expectContains(src, "FLASH_MAX_HEAD_DIM = 512");
    // Must use strided loops, not single-pass `if (tid < vec4_dim)`
    try expectContains(src, "for (uint i = tid; i < vec4_dim; i += FLASH_TG_SIZE)");
    try expectContains(src, "for (uint vi = tid; vi < vec4_dim; vi += FLASH_TG_SIZE)");
    try expectNotContains(src, "if (tid < vec4_dim)");
}

test "Metal flash_attn_q8 supports head_dim=512" {
    const src = @embedFile("shaders/metal/flash_attn_q8.metal");
    try expectContains(src, "FLASH_MAX_HEAD_DIM = 512");
    try expectContains(src, "for (uint i = tid; i < vec4_dim; i += FLASH_TG_SIZE)");
    try expectContains(src, "for (uint vi = tid; vi < vec4_dim; vi += FLASH_TG_SIZE)");
    try expectNotContains(src, "if (tid < vec4_dim)");
}

test "Metal flash_attn_batched supports head_dim=512" {
    const src = @embedFile("shaders/metal/flash_attn_batched.metal");
    try expectContains(src, "FLASH_MAX_HEAD_DIM = 512");
    try expectContains(src, "for (uint i = tid; i < vec4_dim; i += FLASH_TG_SIZE)");
    try expectContains(src, "for (uint vi = tid; vi < vec4_dim; vi += FLASH_TG_SIZE)");
    try expectNotContains(src, "if (tid < vec4_dim)");
}

test "Metal Q8_0 DMMV uses float dot products, not half (overflow at large norm values)" {
    // Regression: Q8_0 shader converted float input to half4 for dot products.
    // Gemma 4 attn_norm weights up to ~300 produce norm_buf values up to ~3000;
    // int8(127) × half(3000) = 381,000 overflows f16 max (65504) → -inf.
    // Fix: use float4 dot products in all Q8_0 DMMV variants.
    //
    // Also: quants must be read via packed_char4 (not int* cast) because Q8_0
    // quants start at byte offset 2 within 34-byte blocks — misaligned for int*.
    const src = @embedFile("shaders/metal/dmmv_q8_0.metal");
    try expectContains(src, "packed_char4");
    try expectContains(src, "dot(float4(");
    // Must NOT convert input to half
    try expectNotContains(src, "half4 x = half4(");
    try expectNotContains(src, "half4 q_half");
    // Must NOT use misaligned int* cast for quant reads
    try expectNotContains(src, "device const int*)(blk");
}

test "Metal Q8_0 k2048 DMMV uses float dot products" {
    const src = @embedFile("shaders/metal/dmmv_q8_0_k2048.metal");
    try expectContains(src, "packed_char4");
    try expectContains(src, "dot(float4(");
    try expectNotContains(src, "half4 x = half4(");
    try expectNotContains(src, "half4 q_half");
    try expectNotContains(src, "device const int*)(blk");
}

test "Metal Q8_0 dual DMMV uses float dot products" {
    const src = @embedFile("shaders/metal/dmmv_q8_0_dual.metal");
    try expectContains(src, "float4 q_f = float4(q)");
    try expectContains(src, "dot(q_f, x_f)");
    try expectNotContains(src, "half4 x = half4(");
    try expectNotContains(src, "half4 q_half");
}

test "Metal Q5_1 DMMV shader exists and uses factored d*sum(q*x)+m*sum(x)" {
    // Q5_1 expert down projections in Gemma 4 26B-A4B MoE were falling back to CPU.
    const src = @embedFile("shaders/metal/dmmv_q5_1.metal");
    // Q5_1 block: 24 bytes (d=f16 + m=f16 + qh=u32 + qs=16 bytes)
    try expectContains(src, "bpb = 24");
    // Factored dot product: d * sum(q*x) + m * sum(x)
    try expectContains(src, "d * sum_qx + m * sum_x");
    // Must read min value from bytes 2-3
    try expectContains(src, "half*)(block + 2)");
    // Must read qh from bytes 4-7
    try expectContains(src, "block[4]");
}

test "Vulkan Q5_1 DMMV shader exists and uses factored dot product" {
    const src = @embedFile("shaders/dmmv_q5_1.comp");
    try expectContains(src, "Q5_1_BYTES      = 24");
    try expectContains(src, "d * sum_qx + m * sum_x");
}

test "Vulkan flash_attn supports head_dim up to 512" {
    // Gemma 4 global attention layers use head_dim=512.
    // The Vulkan shader uses vec4-packed shared memory sized for 512 (128 vec4)
    // and strided loops over head_dim_v4 = head_dim/4.
    const src = @embedFile("shaders/flash_attn.comp");
    try expectContains(src, "shared vec4 s_out_v4[128]");
    // Uses strided loop (tid increments by 64) that naturally handles any
    // head_dim by iterating over head_dim_v4 vec4 chunks.
    try expectContains(src, "for (uint d4 = tid; d4 < head_dim_v4; d4 += 64u)");
}

test "Metal forward derives per-layer head_dim from attn_q_norm tensor" {
    // Regression: Gemma 4 has mixed head_dim per layer (256 for SWA, 512 for global).
    // The Metal forward must derive head_dim from attn_q_norm or attn_k_norm tensors,
    // not use the global config.head_dim for all layers.
    const src = @embedFile("compute/forward_metal.zig");
    try expectContains(src, "if (lt.attn_q_norm) |qn|");
    try expectContainsNear(src, "if (lt.attn_q_norm) |qn|", "head_dim = @intCast(qn.info.numElements())", 200);
}

test "Metal forward handles use_k_as_v for Gemma global attention layers" {
    // Regression: Gemma 4 global attention layers have no attn_v tensor — they share K as V.
    const src = @embedFile("compute/forward_metal.zig");
    try expectContains(src, "use_k_as_v = lt.attn_v == null and cfg.architecture == .gemma");
}

test "Vulkan forward handles use_k_as_v for Gemma global attention layers" {
    const src = @embedFile("compute/forward.zig");
    try expectContains(src, "use_k_as_v = lt.attn_v == null and config.architecture == .gemma");
}

test "Vulkan forward derives per-layer head_dim from attn_q_norm tensor" {
    const src = @embedFile("compute/forward.zig");
    try expectContains(src, "layer_head_dim");
    try expectContains(src, "layer_kv_dim");
    try expectContains(src, "layer_n_kv_heads");
}

test "Vulkan forward handles fused ffn_gate_up_exps for Gemma 4 MoE" {
    // Gemma 4 26B-A4B uses fused ffn_gate_up_exps instead of separate gate/up tensors.
    const src = @embedFile("compute/forward.zig");
    try expectContains(src, "fused_gate_up = lt.ffn_gate_up_exps");
    try expectContains(src, "up_base_offset");
}

test "Vulkan forward uses GEGLU activation for Gemma architecture" {
    // Gemma models use GEGLU, not SwiGLU. The dispatchFfnActivation helper
    // selects the right shader based on architecture.
    const src = @embedFile("compute/forward.zig");
    try expectContains(src, "fn dispatchFfnActivation(");
    try expectContains(src, "fn dispatchGeglu(");
    try expectContains(src, "self.model.config.architecture == .gemma");
    // All MoE/FFN activation calls must use dispatchFfnActivation, not dispatchSwiglu directly
    try expectNotContains(src, "try self.dispatchSwiglu(");
}

test "Gemma 4 26B-A4B MoE catalog entry has correct download URL with UD prefix" {
    const src = @embedFile("model/catalog.zig");
    // The Unsloth Dynamic quantization uses UD- prefix in filenames
    try expectContains(src, "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf");
    // Must NOT have the old incorrect filename without UD-
    try expectNotContains(src, "gemma-4-26B-A4B-it-Q4_K_M.gguf");
}

test "Q5_0 shader reads qh via byte assembly, not unaligned uint32 cast" {
    // Regression guard: the Q5_0 block stores qh at byte offset 2 within a 22-byte block.
    // Reading via *((device const uint*)&block[2]) silently returns wrong values on Apple
    // Silicon for non-4-byte-aligned addresses. The fix reads bytes individually.
    const src = @embedFile("shaders/metal/dmmv_q5_0.metal");
    // Must NOT contain the broken unaligned cast pattern
    try expectNotContains(src, "uint*)&block[2]");
    try expectNotContains(src, "uint*)(block + 2)");
    // Must contain the safe byte-by-byte assembly
    try expectContains(src, "uint(block[2])");
    try expectContains(src, "uint(block[3])");
    try expectContains(src, "uint(block[4])");
    try expectContains(src, "uint(block[5])");
}

test "Q5_0 dequantRow matches expected values for known block" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;

    const forward_metal = @import("compute/forward_metal.zig");
    // Build a Q5_0 block: d=0.5, qh=0x0000FFFF (bits 0-15 set), qs all 0x53 (lo=3, hi=5)
    // Element j (0-15): lo=3, bit_lo=1 → quant=3|(1<<4)=19 → value=0.5*(19-16)=1.5
    // Element 16+j:     hi=5, bit_hi=0 → quant=5|(0<<4)=5  → value=0.5*(5-16)=-5.5
    var block: [22]u8 = undefined;
    const d_bits: u16 = @bitCast(@as(f16, 0.5));
    block[0] = @truncate(d_bits);
    block[1] = @truncate(d_bits >> 8);
    block[2] = 0xFF;
    block[3] = 0xFF;
    block[4] = 0x00;
    block[5] = 0x00; // qh = 0x0000FFFF
    @memset(block[6..22], 0x53); // lo=3, hi=5
    var output: [32]f32 = undefined;
    forward_metal.dequantRow(&block, 0, 32, .q5_0, &output);
    for (0..16) |j| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.5), output[j], 0.001);
    }
    for (16..32) |j| {
        try std.testing.expectApproxEqAbs(@as(f32, -5.5), output[j], 0.001);
    }
}

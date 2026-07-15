//! Tenant-aware batch planning for ZINC_RT.
//! @section Inference Runtime
//! This module owns admission, quotas, and prefill/decode batch selection.
//! It is intentionally independent of the current host-assisted
//! `forward_zinc_rt` execution path so it can be validated before the M3
//! continuous-batching executor consumes it.
// SPDX-FileCopyrightText: ZINC Authors
const std = @import("std");

/// Stable tenant identifier supplied by the API/server layer.
pub const TenantId = u32;
/// Monotonic request identifier assigned at admission.
pub const RequestId = u64;

/// Per-tenant admission and scheduling limits.
pub const TenantLimits = struct {
    /// Maximum live requests admitted for this tenant.
    max_active_requests: u32 = 1,
    /// Maximum prompt tokens this tenant may occupy inside one prefill batch.
    max_prefill_tokens_per_batch: u32 = 1024,
    /// Maximum decode rows this tenant may occupy inside one batch.
    max_decode_slots: u32 = 1,
};

/// Request metadata needed by the ZINC_RT batch planner.
pub const RequestConfig = struct {
    /// Tenant that owns the request.
    tenant_id: TenantId,
    /// Number of prompt tokens that must be prefetched before decode.
    prompt_tokens: u32,
    /// Maximum number of decode tokens to emit.
    max_new_tokens: u32,
};

/// Lifecycle tracked by the ZINC_RT batch planner.
pub const RequestState = enum {
    /// Waiting for prefill work.
    queued_prefill,
    /// Prompt prefill has started but not all prompt tokens are consumed.
    prefilling,
    /// Ready for one-token-at-a-time decode batching.
    decoding,
    /// Finished normally.
    completed,
    /// Cancelled by the caller.
    cancelled,
};

/// One request slot in the multitenant planner.
pub const Slot = struct {
    /// Request ID assigned by the scheduler.
    id: RequestId,
    /// Tenant that owns the request.
    tenant_id: TenantId,
    /// Current request state.
    state: RequestState,
    /// Total prompt tokens to prefill.
    prompt_tokens_total: u32,
    /// Prompt tokens already consumed by prefill.
    prompt_tokens_consumed: u32 = 0,
    /// Decode tokens already generated.
    generated_tokens: u32 = 0,
    /// Maximum decode tokens to emit.
    max_new_tokens: u32,

    /// Remaining prompt tokens.
    pub fn promptRemaining(self: Slot) u32 {
        return self.prompt_tokens_total - self.prompt_tokens_consumed;
    }

    /// Whether decode has reached the configured maximum.
    pub fn decodeDone(self: Slot) bool {
        return self.generated_tokens >= self.max_new_tokens;
    }
};

/// Type of work represented by a batch entry.
pub const BatchKind = enum {
    prefill,
    decode,
};

/// One request inside a selected prefill or decode batch.
pub const BatchEntry = struct {
    /// Slot index inside `BatchScheduler.slots`.
    slot_id: u32,
    /// Request identifier for logging and result routing.
    request_id: RequestId,
    /// Tenant that owns the entry.
    tenant_id: TenantId,
    /// Number of tokens granted to this entry. Decode batches use 1.
    tokens: u32,
};

const Tenant = struct {
    id: TenantId,
    limits: TenantLimits,
    active_requests: u32 = 0,
};

/// Fixed-capacity multitenant scheduler for ZINC_RT prefill/decode batches.
pub const BatchScheduler = struct {
    allocator: std.mem.Allocator,
    slots: []?Slot,
    tenants: []?Tenant,
    next_request_id: RequestId = 1,
    next_prefill_cursor: usize = 0,
    next_decode_cursor: usize = 0,

    /// Initialize a fixed-capacity scheduler.
    /// @param allocator Allocator for slot and tenant arrays.
    /// @param max_slots Maximum live requests across all tenants.
    /// @param max_tenants Maximum registered tenants.
    pub fn init(allocator: std.mem.Allocator, max_slots: u32, max_tenants: u32) !BatchScheduler {
        if (max_slots == 0 or max_tenants == 0) return error.InvalidCapacity;

        const slots = try allocator.alloc(?Slot, max_slots);
        errdefer allocator.free(slots);
        @memset(slots, null);

        const tenants = try allocator.alloc(?Tenant, max_tenants);
        errdefer allocator.free(tenants);
        @memset(tenants, null);

        return .{
            .allocator = allocator,
            .slots = slots,
            .tenants = tenants,
        };
    }

    /// Register or update limits for a tenant.
    /// @param self Scheduler to mutate.
    /// @param tenant_id Tenant identifier.
    /// @param limits New tenant limits.
    pub fn registerTenant(self: *BatchScheduler, tenant_id: TenantId, limits: TenantLimits) !void {
        try validateTenantLimits(limits);

        if (self.tenantIndex(tenant_id)) |idx| {
            if (self.tenants[idx].?.active_requests > limits.max_active_requests) {
                return error.LimitBelowActiveRequests;
            }
            self.tenants[idx].?.limits = limits;
            return;
        }

        for (self.tenants) |*tenant_slot| {
            if (tenant_slot.* == null) {
                tenant_slot.* = .{
                    .id = tenant_id,
                    .limits = limits,
                };
                return;
            }
        }
        return error.TooManyTenants;
    }

    /// Submit a request into the first free slot.
    /// @returns Assigned slot ID.
    pub fn submit(self: *BatchScheduler, config: RequestConfig) !u32 {
        if (config.prompt_tokens == 0 or config.max_new_tokens == 0) return error.InvalidRequest;

        const tenant_idx = self.tenantIndex(config.tenant_id) orelse return error.UnknownTenant;
        const owner = &self.tenants[tenant_idx].?;
        if (owner.active_requests >= owner.limits.max_active_requests) {
            return error.TenantRequestQuotaExceeded;
        }

        for (self.slots, 0..) |*slot, i| {
            if (slot.* == null) {
                const id = self.next_request_id;
                self.next_request_id += 1;
                slot.* = .{
                    .id = id,
                    .tenant_id = config.tenant_id,
                    .state = .queued_prefill,
                    .prompt_tokens_total = config.prompt_tokens,
                    .max_new_tokens = config.max_new_tokens,
                };
                owner.active_requests += 1;
                return @intCast(i);
            }
        }

        return error.AllSlotsBusy;
    }

    /// Select queued/prefilling requests for prompt prefill.
    /// @param out Caller-owned scratch for entries.
    /// @param max_prompt_tokens Total prompt-token budget for this batch.
    /// @returns A slice of `out` with selected prefill entries.
    pub fn selectPrefillBatch(self: *BatchScheduler, out: []BatchEntry, max_prompt_tokens: u32) []BatchEntry {
        if (self.slots.len == 0 or max_prompt_tokens == 0 or out.len == 0) return out[0..0];

        var count: usize = 0;
        var used_tokens: u32 = 0;
        var visited: usize = 0;
        var last_selected: ?usize = null;
        const start = self.next_prefill_cursor % self.slots.len;

        while (visited < self.slots.len and count < out.len and used_tokens < max_prompt_tokens) : (visited += 1) {
            const idx = (start + visited) % self.slots.len;
            const slot = self.slots[idx];
            const req = slot orelse continue;
            if (req.state != .queued_prefill and req.state != .prefilling) continue;

            const owner = self.tenantInfo(req.tenant_id) orelse continue;
            const tenant_used = selectedTokensForTenant(out[0..count], req.tenant_id);
            if (tenant_used >= owner.limits.max_prefill_tokens_per_batch) continue;

            const remaining = req.promptRemaining();
            if (remaining == 0) continue;

            const global_budget = max_prompt_tokens - used_tokens;
            const tenant_budget = owner.limits.max_prefill_tokens_per_batch - tenant_used;
            const grant = @min(remaining, @min(global_budget, tenant_budget));
            if (grant == 0) continue;

            out[count] = .{
                .slot_id = @intCast(idx),
                .request_id = req.id,
                .tenant_id = req.tenant_id,
                .tokens = grant,
            };
            count += 1;
            used_tokens += grant;
            last_selected = idx;
        }

        if (last_selected) |idx| {
            self.next_prefill_cursor = (idx + 1) % self.slots.len;
        }
        return out[0..count];
    }

    /// Account for completed prompt work.
    /// @param slot_id Slot that consumed prompt tokens.
    /// @param tokens Number of prompt tokens consumed.
    pub fn advancePrefill(self: *BatchScheduler, slot_id: u32, tokens: u32) !void {
        const slot = try self.slotPtr(slot_id);
        if (slot.state != .queued_prefill and slot.state != .prefilling) return error.InvalidState;
        const remaining = slot.promptRemaining();
        if (tokens == 0 or tokens > remaining) return error.InvalidTokenCount;

        slot.prompt_tokens_consumed += tokens;
        slot.state = if (slot.promptRemaining() == 0) .decoding else .prefilling;
    }

    /// Select active decode requests fairly across slots and tenant limits.
    /// @param out Caller-owned scratch for entries.
    /// @param max_slots Maximum entries to select.
    /// @returns A slice of `out` with selected decode entries.
    pub fn selectDecodeBatch(self: *BatchScheduler, out: []BatchEntry, max_slots: u32) []BatchEntry {
        if (self.slots.len == 0 or out.len == 0 or max_slots == 0) return out[0..0];

        const limit: usize = @min(out.len, max_slots);
        var count: usize = 0;
        var visited: usize = 0;
        var last_selected: ?usize = null;
        const start = self.next_decode_cursor % self.slots.len;

        while (visited < self.slots.len and count < limit) : (visited += 1) {
            const idx = (start + visited) % self.slots.len;
            const req = self.slots[idx] orelse continue;
            if (req.state != .decoding or req.decodeDone()) continue;

            const owner = self.tenantInfo(req.tenant_id) orelse continue;
            const already_selected = selectedCountForTenant(out[0..count], req.tenant_id);
            if (already_selected >= owner.limits.max_decode_slots) continue;

            out[count] = .{
                .slot_id = @intCast(idx),
                .request_id = req.id,
                .tenant_id = req.tenant_id,
                .tokens = 1,
            };
            count += 1;
            last_selected = idx;
        }

        if (last_selected) |idx| {
            self.next_decode_cursor = (idx + 1) % self.slots.len;
        }
        return out[0..count];
    }

    /// Account for one generated decode token.
    /// @param slot_id Slot that emitted a token.
    pub fn recordDecodeToken(self: *BatchScheduler, slot_id: u32) !void {
        const slot = try self.slotPtr(slot_id);
        if (slot.state != .decoding) return error.InvalidState;
        if (slot.decodeDone()) return error.InvalidState;

        slot.generated_tokens += 1;
        if (slot.decodeDone()) {
            slot.state = .completed;
        }
    }

    /// Mark a request as cancelled.
    pub fn cancel(self: *BatchScheduler, slot_id: u32) !void {
        const slot = try self.slotPtr(slot_id);
        slot.state = .cancelled;
    }

    /// Release a completed or cancelled slot.
    pub fn release(self: *BatchScheduler, slot_id: u32) void {
        if (slot_id >= self.slots.len) return;
        const idx: usize = @intCast(slot_id);
        const slot = self.slots[idx] orelse return;

        if (self.tenantIndex(slot.tenant_id)) |tenant_idx| {
            if (self.tenants[tenant_idx]) |*owner| {
                if (owner.active_requests > 0) owner.active_requests -= 1;
            }
        }
        self.slots[idx] = null;
    }

    /// Number of occupied request slots.
    pub fn activeCount(self: *const BatchScheduler) u32 {
        var count: u32 = 0;
        for (self.slots) |slot| {
            if (slot != null) count += 1;
        }
        return count;
    }

    /// Number of live requests owned by `tenant_id`.
    pub fn tenantActiveCount(self: *const BatchScheduler, tenant_id: TenantId) u32 {
        const idx = self.tenantIndex(tenant_id) orelse return 0;
        return self.tenants[idx].?.active_requests;
    }

    /// Free all planner storage.
    pub fn deinit(self: *BatchScheduler) void {
        self.allocator.free(self.slots);
        self.allocator.free(self.tenants);
        self.* = undefined;
    }

    fn tenantIndex(self: *const BatchScheduler, tenant_id: TenantId) ?usize {
        for (self.tenants, 0..) |tenant_slot, i| {
            if (tenant_slot) |entry| {
                if (entry.id == tenant_id) return i;
            }
        }
        return null;
    }

    fn tenantInfo(self: *const BatchScheduler, tenant_id: TenantId) ?Tenant {
        const idx = self.tenantIndex(tenant_id) orelse return null;
        return self.tenants[idx].?;
    }

    fn slotPtr(self: *BatchScheduler, slot_id: u32) !*Slot {
        if (slot_id >= self.slots.len) return error.InvalidSlot;
        const idx: usize = @intCast(slot_id);
        if (self.slots[idx]) |*slot| return slot;
        return error.InvalidSlot;
    }
};

fn validateTenantLimits(limits: TenantLimits) !void {
    if (limits.max_active_requests == 0 or
        limits.max_prefill_tokens_per_batch == 0 or
        limits.max_decode_slots == 0)
    {
        return error.InvalidTenantLimits;
    }
}

fn selectedTokensForTenant(entries: []const BatchEntry, tenant_id: TenantId) u32 {
    var tokens: u32 = 0;
    for (entries) |entry| {
        if (entry.tenant_id == tenant_id) tokens += entry.tokens;
    }
    return tokens;
}

fn selectedCountForTenant(entries: []const BatchEntry, tenant_id: TenantId) u32 {
    var count: u32 = 0;
    for (entries) |entry| {
        if (entry.tenant_id == tenant_id) count += 1;
    }
    return count;
}

test "batch scheduler enforces per-tenant admission quota" {
    const allocator = std.testing.allocator;
    var sched = try BatchScheduler.init(allocator, 4, 2);
    defer sched.deinit();

    try sched.registerTenant(7, .{ .max_active_requests = 1, .max_decode_slots = 1 });

    _ = try sched.submit(.{ .tenant_id = 7, .prompt_tokens = 4, .max_new_tokens = 8 });
    try std.testing.expectError(
        error.TenantRequestQuotaExceeded,
        sched.submit(.{ .tenant_id = 7, .prompt_tokens = 4, .max_new_tokens = 8 }),
    );
    try std.testing.expectEqual(@as(u32, 1), sched.tenantActiveCount(7));
}

test "batch scheduler requires explicit tenant registration" {
    const allocator = std.testing.allocator;
    var sched = try BatchScheduler.init(allocator, 4, 2);
    defer sched.deinit();

    try std.testing.expectError(
        error.UnknownTenant,
        sched.submit(.{ .tenant_id = 7, .prompt_tokens = 4, .max_new_tokens = 8 }),
    );
}

test "batch scheduler rejects zero tenant limits" {
    const allocator = std.testing.allocator;
    var sched = try BatchScheduler.init(allocator, 4, 2);
    defer sched.deinit();

    try std.testing.expectError(
        error.InvalidTenantLimits,
        sched.registerTenant(7, .{ .max_active_requests = 0 }),
    );
    try std.testing.expectError(
        error.InvalidTenantLimits,
        sched.registerTenant(7, .{ .max_prefill_tokens_per_batch = 0 }),
    );
    try std.testing.expectError(
        error.InvalidTenantLimits,
        sched.registerTenant(7, .{ .max_decode_slots = 0 }),
    );
}

test "batch scheduler selects and advances prompt prefill batches" {
    const allocator = std.testing.allocator;
    var sched = try BatchScheduler.init(allocator, 4, 4);
    defer sched.deinit();

    try sched.registerTenant(1, .{});
    try sched.registerTenant(2, .{});

    const slot0 = try sched.submit(.{ .tenant_id = 1, .prompt_tokens = 5, .max_new_tokens = 2 });
    const slot1 = try sched.submit(.{ .tenant_id = 2, .prompt_tokens = 7, .max_new_tokens = 2 });

    var entries: [4]BatchEntry = undefined;
    const batch = sched.selectPrefillBatch(&entries, 8);
    try std.testing.expectEqual(@as(usize, 2), batch.len);
    try std.testing.expectEqual(slot0, batch[0].slot_id);
    try std.testing.expectEqual(@as(u32, 5), batch[0].tokens);
    try std.testing.expectEqual(slot1, batch[1].slot_id);
    try std.testing.expectEqual(@as(u32, 3), batch[1].tokens);

    try sched.advancePrefill(batch[0].slot_id, batch[0].tokens);
    try sched.advancePrefill(batch[1].slot_id, batch[1].tokens);
    try std.testing.expectEqual(RequestState.decoding, sched.slots[slot0].?.state);
    try std.testing.expectEqual(RequestState.prefilling, sched.slots[slot1].?.state);
}

test "batch scheduler caps prefill tokens per tenant" {
    const allocator = std.testing.allocator;
    var sched = try BatchScheduler.init(allocator, 4, 4);
    defer sched.deinit();

    try sched.registerTenant(1, .{ .max_active_requests = 2, .max_prefill_tokens_per_batch = 5 });
    try sched.registerTenant(2, .{ .max_active_requests = 1, .max_prefill_tokens_per_batch = 7 });

    const a = try sched.submit(.{ .tenant_id = 1, .prompt_tokens = 9, .max_new_tokens = 2 });
    _ = try sched.submit(.{ .tenant_id = 1, .prompt_tokens = 9, .max_new_tokens = 2 });
    const c = try sched.submit(.{ .tenant_id = 2, .prompt_tokens = 9, .max_new_tokens = 2 });

    var entries: [4]BatchEntry = undefined;
    const batch = sched.selectPrefillBatch(&entries, 32);
    try std.testing.expectEqual(@as(usize, 2), batch.len);
    try std.testing.expectEqual(a, batch[0].slot_id);
    try std.testing.expectEqual(@as(u32, 5), selectedTokensForTenant(batch, 1));
    try std.testing.expectEqual(c, batch[1].slot_id);
    try std.testing.expectEqual(@as(u32, 7), selectedTokensForTenant(batch, 2));
}

test "batch scheduler selects decode batch with tenant caps" {
    const allocator = std.testing.allocator;
    var sched = try BatchScheduler.init(allocator, 4, 4);
    defer sched.deinit();

    try sched.registerTenant(1, .{ .max_active_requests = 2, .max_decode_slots = 1 });
    try sched.registerTenant(2, .{ .max_active_requests = 2, .max_decode_slots = 2 });

    const a = try sched.submit(.{ .tenant_id = 1, .prompt_tokens = 1, .max_new_tokens = 2 });
    const b = try sched.submit(.{ .tenant_id = 1, .prompt_tokens = 1, .max_new_tokens = 2 });
    const c = try sched.submit(.{ .tenant_id = 2, .prompt_tokens = 1, .max_new_tokens = 2 });

    try sched.advancePrefill(a, 1);
    try sched.advancePrefill(b, 1);
    try sched.advancePrefill(c, 1);

    var entries: [4]BatchEntry = undefined;
    const batch = sched.selectDecodeBatch(&entries, 4);
    try std.testing.expectEqual(@as(usize, 2), batch.len);
    try std.testing.expectEqual(@as(u32, 1), selectedCountForTenant(batch, 1));
    try std.testing.expectEqual(@as(u32, 1), selectedCountForTenant(batch, 2));
}

test "batch scheduler completes requests after max decode tokens" {
    const allocator = std.testing.allocator;
    var sched = try BatchScheduler.init(allocator, 2, 2);
    defer sched.deinit();

    try sched.registerTenant(1, .{});

    const slot = try sched.submit(.{ .tenant_id = 1, .prompt_tokens = 1, .max_new_tokens = 2 });
    try sched.advancePrefill(slot, 1);
    try sched.recordDecodeToken(slot);
    try std.testing.expectEqual(RequestState.decoding, sched.slots[slot].?.state);
    try sched.recordDecodeToken(slot);
    try std.testing.expectEqual(RequestState.completed, sched.slots[slot].?.state);

    sched.release(slot);
    try std.testing.expectEqual(@as(u32, 0), sched.activeCount());
    try std.testing.expectEqual(@as(u32, 0), sched.tenantActiveCount(1));
}

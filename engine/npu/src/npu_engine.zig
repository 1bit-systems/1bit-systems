//! Main NPU inference engine — fuses the unified KV cache scheduler (KvPagePool,
//! H2O eviction) with XRT xclbin GEMM kernels on the NPU.
const std = @import("std");

const xrt = @import("xrt.zig");
const model_reader = @import("model_reader.zig");
const ModelConfig = model_reader.ModelConfig;
const cpu_ops = @import("cpu_ops.zig");
const cdequant = @import("cdequant.zig");
const XclbinKernel = @import("npu_kernels.zig").XclbinKernel;
const NpuPageTable = @import("npu_page_table.zig").NpuPageTable;
const PageMapping = @import("npu_page_table.zig").PageMapping;

// Unified scheduler — imported via build.zig as single module
const Scheduler = @import("sched").Scheduler;
const KvPagePool = @import("sched").KvPagePool;
const EvictionPolicy = @import("sched").EvictionPolicy;

const log = std.log.scoped(.npu_engine);

pub const NpuEngine = struct {
    allocator: std.mem.Allocator,
    config: ModelConfig,

    scheduler: Scheduler,
    page_pool: KvPagePool,
    page_table: NpuPageTable,

    device: xrt.XrtDevice,
    kernel_qkv: XclbinKernel,
    kernel_o: XclbinKernel,
    kernel_gu: XclbinKernel,
    kernel_d: XclbinKernel,
    kernel_u: ?XclbinKernel,

    model_map: []align(4096) const u8,

    emb_f32: []f32,
    lm_head_f32: []f32,

    in_norm: [][]f32,
    pa_norm: [][]f32,
    q_norm: [][]f32,
    k_norm: [][]f32,
    final_norm: []f32,

    rope_rc: []f32,
    rope_rs: []f32,

    qkv_scales: []f32,
    o_scales: []f32,
    gu_scales: []f32,
    d_scales: []f32,
    u_scales: ?[]f32,

    h_buf: []f32,
    qkv_buf: []f32,
    attn_buf: []f32,
    o_buf: []f32,
    gate_buf: []f32,
    silu_buf: []f32,
    d_buf: []f32,
    scratch: []f32,
    logits_buf: []f32,
    tok_ids: []u32,

    kv_page_size: u32,
    eos_token_id: u32,

    // ============================================================
    // init
    // ============================================================

    pub fn init(
        allocator: std.mem.Allocator,
        model_path: []const u8,
        xclbin_dir: []const u8,
        model_tag: []const u8,
        max_parallel: u32,
        total_kv_pages: u32,
    ) !NpuEngine {
        log.info("Init: model={s} tag={s}", .{ model_path, model_tag });

        const cfg = try model_reader.parseQ4nxHeader(model_path, model_tag);
        if (!cfg.valid()) return error.InvalidModelConfig;

        const H = cfg.H; const NC = cfg.NC; const NH = cfg.NH; const NKV = cfg.NKV;
        const HD = cfg.HD; const IM = cfg.IM; const NV = cfg.NV; const XM = cfg.XM;
        const qkv_total = cfg.qkv_total;
        const mlp_out: usize = if (cfg.gu_split) IM else 2 * IM;

        log.info("Model: H={d} NC={d} NH={d} NKV={d} HD={d} IM={d} NV={d} GQA={d} gu_split={}",
            .{ H, NC, NH, NKV, HD, IM, NV, cfg.GQA, cfg.gu_split });

        // ---- Mmap model file ----
        const fd = try std.fs.openFileAbsolute(model_path, .{});
        defer fd.close();
        const file_size = try fd.getEndPos();
        const model_map = try std.posix.mmap(
            null, file_size, std.posix.PROT.READ,
            std.posix.MAP.PRIVATE, fd.handle, 0,
        );
        errdefer std.posix.munmap(model_map);

        const hdr_size = std.mem.readInt(u64, model_map[0..8].*, .little);
        const data_start: usize = 8 + hdr_size;
        const js = model_map[8 .. 8 + hdr_size];
        const weight_data = model_map[data_start..];

        // ---- Open XRT device ----
        var device = try xrt.XrtDevice.open(0);
        errdefer device.close();

        // ---- Load xclbin kernels ----
        var kq = XclbinKernel.init(allocator, "QKV");
        var ko = XclbinKernel.init(allocator, "O");
        var kg = XclbinKernel.init(allocator, if (cfg.gu_split) "G" else "GU");
        var kd = XclbinKernel.init(allocator, "D");
        var ku: ?XclbinKernel = if (cfg.gu_split) XclbinKernel.init(allocator, "U") else null;

        const tag_sfx = model_tag;
        var xp: [512]u8 = undefined;
        var ip: [512]u8 = undefined;

        {
            const x = try std.fmt.bufPrint(&xp, "{s}/final_i8_QKV_{s}.xclbin", .{ xclbin_dir, tag_sfx });
            const i = try std.fmt.bufPrint(&ip, "{s}/insts_i8_QKV_{s}.txt", .{ xclbin_dir, tag_sfx });
            try kq.load(&device, x, i, XM, cfg.xclbin_qkv_k, cfg.xclbin_qkv_n, NC, 4);
        }
        {
            const x = try std.fmt.bufPrint(&xp, "{s}/final_i8_O_{s}.xclbin", .{ xclbin_dir, tag_sfx });
            const i = try std.fmt.bufPrint(&ip, "{s}/insts_i8_O_{s}.txt", .{ xclbin_dir, tag_sfx });
            try ko.load(&device, x, i, XM, cfg.xclbin_o_k, cfg.xclbin_o_n, NC, 4);
        }
        {
            const sfx = if (cfg.gu_split) "G" else "GU";
            const x = try std.fmt.bufPrint(&xp, "{s}/final_i8_{s}_{s}.xclbin", .{ xclbin_dir, sfx, tag_sfx });
            const i = try std.fmt.bufPrint(&ip, "{s}/insts_i8_{s}_{s}.txt", .{ xclbin_dir, sfx, tag_sfx });
            try kg.load(&device, x, i, XM, cfg.xclbin_gu_k, cfg.xclbin_gu_n, NC, 4);
        }
        {
            const x = try std.fmt.bufPrint(&xp, "{s}/final_i8_D_{s}.xclbin", .{ xclbin_dir, tag_sfx });
            const i = try std.fmt.bufPrint(&ip, "{s}/insts_i8_D_{s}.txt", .{ xclbin_dir, tag_sfx });
            try kd.load(&device, x, i, XM, cfg.xclbin_d_k, cfg.xclbin_d_n, NC, 4);
        }
        if (cfg.gu_split) {
            const x = try std.fmt.bufPrint(&xp, "{s}/final_i8_U_{s}.xclbin", .{ xclbin_dir, tag_sfx });
            const i = try std.fmt.bufPrint(&ip, "{s}/insts_i8_U_{s}.txt", .{ xclbin_dir, tag_sfx });
            try ku.?.load(&device, x, i, XM, cfg.xclbin_u_k, cfg.xclbin_u_n, NC, 4);
        }

        // ---- JSON offsets per layer ----
        const LayerOffsets = struct {
            q_off: u32, q_i8: u32,
            k_off: u32, k_i8: u32,
            v_off: u32, v_i8: u32,
            o_off: u32, o_i8: u32,
            g_off: u32, g_i8: u32,
            u_off: u32, u_i8: u32,
            d_off: u32, d_i8: u32,
            in_off: u32, pa_off: u32,
            qn_off: u32, kn_off: u32,
        };
        var lo = try allocator.alloc(LayerOffsets, NC);
        defer allocator.free(lo);

        const q_i8 = model_reader.findTileRows(js, "model.layers.0.self_attn.q_proj.weight") orelse 0;
        const k_i8 = model_reader.findTileRows(js, "model.layers.0.self_attn.k_proj.weight") orelse 0;
        const v_i8 = model_reader.findTileRows(js, "model.layers.0.self_attn.v_proj.weight") orelse 0;
        const o_i8 = model_reader.findTileRows(js, "model.layers.0.self_attn.o_proj.weight") orelse 0;
        const g_i8 = model_reader.findTileRows(js, "model.layers.0.mlp.gate_proj.weight") orelse 0;
        const u_i8 = model_reader.findTileRows(js, "model.layers.0.mlp.up_proj.weight") orelse 0;
        const d_i8 = model_reader.findTileRows(js, "model.layers.0.mlp.down_proj.weight") orelse 0;

        var bn: [128]u8 = undefined;
        for (0..NC) |l| {

            lo[l] = .{
                .q_off = findOff(js, &bn, "model.layers.{d}.self_attn.q_proj.weight", .{l}) orelse 0,
                .q_i8 = q_i8,
                .k_off = findOff(js, &bn, "model.layers.{d}.self_attn.k_proj.weight", .{l}) orelse 0,
                .k_i8 = k_i8,
                .v_off = findOff(js, &bn, "model.layers.{d}.self_attn.v_proj.weight", .{l}) orelse 0,
                .v_i8 = v_i8,
                .o_off = findOff(js, &bn, "model.layers.{d}.self_attn.o_proj.weight", .{l}) orelse 0,
                .o_i8 = o_i8,
                .g_off = findOff(js, &bn, "model.layers.{d}.mlp.gate_proj.weight", .{l}) orelse 0,
                .g_i8 = g_i8,
                .u_off = findOff(js, &bn, "model.layers.{d}.mlp.up_proj.weight", .{l}) orelse 0,
                .u_i8 = u_i8,
                .d_off = findOff(js, &bn, "model.layers.{d}.mlp.down_proj.weight", .{l}) orelse 0,
                .d_i8 = d_i8,
                .in_off = findOff(js, &bn, "model.layers.{d}.input_layernorm.weight", .{l}) orelse 0,
                .pa_off = findOff(js, &bn, "model.layers.{d}.post_attention_layernorm.weight", .{l}) orelse 0,
                .qn_off = if (cfg.has_q_norm) findOff(js, &bn, "model.layers.{d}.self_attn.q_norm.weight", .{l}) orelse 0 else 0,
                .kn_off = if (cfg.has_k_norm) findOff(js, &bn, "model.layers.{d}.self_attn.k_norm.weight", .{l}) orelse 0 else 0,
            };
        }

        // ---- Dequant + pack ----
        log.info("Dequant+pack...", .{});

        const qkv_scales = try allocator.alloc(f32, NC);
        const o_scales = try allocator.alloc(f32, NC);
        const gu_scales = try allocator.alloc(f32, NC);
        const d_scales = try allocator.alloc(f32, NC);
        const usc = if (cfg.gu_split) try allocator.alloc(f32, NC) else null;

        const QOUT = NH * HD; const KVOUT = NKV * HD;
        const OOUT = H; const OIN = NH * HD;
        const GUOUT = IM; const DIN = IM; const DOUT = H;

        for (0..NC) |l| {
            // QKV
            {
                const qd = try dequantToSliceEx(weight_data[lo[l].q_off..], lo[l].q_i8, H);
                defer allocator.free(qd);
                const kd_ = try dequantToSliceEx(weight_data[lo[l].k_off..], lo[l].k_i8, H);
                defer allocator.free(kd_);
                const vd = try dequantToSliceEx(weight_data[lo[l].v_off..], lo[l].v_i8, H);
                defer allocator.free(vd);

                const t = QOUT + KVOUT + KVOUT;
                const fused = try allocator.alloc(f32, H * t);
                defer allocator.free(fused);
                @memset(fused, 0);

                transposePack(qd, QOUT, H, fused, t, 0);
                transposePack(kd_, KVOUT, H, fused, t, QOUT);
                transposePack(vd, KVOUT, H, fused, t, QOUT + KVOUT);

                qkv_scales[l] = try kq.packWeight(@intCast(l), fused, H, t);
            }
            // O
            {
                const od = try dequantToSliceEx(weight_data[lo[l].o_off..], lo[l].o_i8, OIN);
                defer allocator.free(od);
                const buf_o = try allocator.alloc(f32, OIN * OOUT);
                defer allocator.free(buf_o);
                transposePack(od, OOUT, OIN, buf_o, OOUT, 0);
                o_scales[l] = try ko.packWeight(@intCast(l), buf_o, OIN, OOUT);
            }
            // Gate/Up
            {
                const gd = try dequantToSliceEx(weight_data[lo[l].g_off..], lo[l].g_i8, H);
                defer allocator.free(gd);
                if (cfg.gu_split) {
                    const buf_g = try allocator.alloc(f32, H * GUOUT);
                    defer allocator.free(buf_g);
                    transposePack(gd, GUOUT, H, buf_g, GUOUT, 0);
                    gu_scales[l] = try kg.packWeight(@intCast(l), buf_g, H, GUOUT);

                    const ud = try dequantToSliceEx(weight_data[lo[l].u_off..], lo[l].u_i8, H);
                    defer allocator.free(ud);
                    const buf_u = try allocator.alloc(f32, H * GUOUT);
                    defer allocator.free(buf_u);
                    transposePack(ud, GUOUT, H, buf_u, GUOUT, 0);
                    usc.?[l] = try ku.?.packWeight(@intCast(l), buf_u, H, GUOUT);
                } else {
                    const ud = try dequantToSliceEx(weight_data[lo[l].u_off..], lo[l].u_i8, H);
                    defer allocator.free(ud);
                    const t2 = GUOUT + GUOUT;
                    const buf_gu = try allocator.alloc(f32, H * t2);
                    defer allocator.free(buf_gu);
                    transposePack(gd, GUOUT, H, buf_gu, t2, 0);
                    transposePack(ud, GUOUT, H, buf_gu, t2, GUOUT);
                    gu_scales[l] = try kg.packWeight(@intCast(l), buf_gu, H, t2);
                }
            }
            // Down
            {
                const dd = try dequantToSliceEx(weight_data[lo[l].d_off..], lo[l].d_i8, DIN);
                defer allocator.free(dd);
                const buf_d = try allocator.alloc(f32, DIN * DOUT);
                defer allocator.free(buf_d);
                transposePack(dd, DOUT, DIN, buf_d, DOUT, 0);
                d_scales[l] = try kd.packWeight(@intCast(l), buf_d, DIN, DOUT);
            }
        }

        // ---- Embeddings ----
        log.info("Pre-convert embeddings f32...", .{});
        const emb_f32 = try allocator.alloc(f32, NV * H);
        {
            const emb_bf16: [*]const u16 = @ptrCast(@alignCast(weight_data.ptr));
            for (0..NV) |n| {
                for (0..H) |i| {
                    emb_f32[n * H + i] = model_reader.bf16ToF32Safe(emb_bf16[n * H + i]);
                }
            }
        }

        // ---- LM head ----
        const lm_head_f32 = if (cfg.has_lm_head) blk: {
            const lm_off = model_reader.findTensorInfo(js, "lm_head.weight") orelse 0;
            if (lm_off > 0) {
                const lm_i8 = model_reader.findTileRows(js, "lm_head.weight") orelse 0;
                if (lm_i8 > 0) {
                    log.info("Loading lm_head.weight separately", .{});
                    break :blk try dequantToSliceEx(weight_data[lm_off..], lm_i8, H);
                }
            }
            log.info("Using tied embeddings for LM head", .{});
            const cpy = try allocator.alloc(f32, NV * H);
            @memcpy(cpy, emb_f32);
            break :blk cpy;
        } else blk: {
            log.info("Using tied embeddings for LM head", .{});
            const cpy = try allocator.alloc(f32, NV * H);
            @memcpy(cpy, emb_f32);
            break :blk cpy;
        };

        // ---- Norm weights ----
        log.info("Loading norm weights...", .{});
        const in_norm = try allocator.alloc([]f32, NC);
        const pa_norm = try allocator.alloc([]f32, NC);
        // Clamp norm weights to [-2,2] to prevent bf16 overflow and
        // residual-stream collapse from large norm weights (Qwen3-0.6B
        // norm weights grow to 106x+ by layer 26).
        const clip: f32 = 2.0;

        const q_norm = try allocator.alloc([]f32, NC);
        const k_norm = try allocator.alloc([]f32, NC);

        for (0..NC) |l| {
            in_norm[l] = loadBf16Weights(allocator, weight_data, lo[l].in_off, H) catch blk: {
                const id = try allocator.alloc(f32, H);
                @memset(id, 1.0);
                break :blk id;
            };
            pa_norm[l] = loadBf16Weights(allocator, weight_data, lo[l].pa_off, H) catch blk: {
                const id = try allocator.alloc(f32, H);
                @memset(id, 1.0);
                break :blk id;
            };
            for (0..H) |i| {
                in_norm[l][i] = @max(-clip, @min(clip, in_norm[l][i]));
                pa_norm[l][i] = @max(-clip, @min(clip, pa_norm[l][i]));
            }
            if (cfg.has_q_norm and lo[l].qn_off > 0) {
                q_norm[l] = loadBf16Weights(allocator, weight_data, lo[l].qn_off, HD) catch blk: {
                    const id = try allocator.alloc(f32, HD);
                    @memset(id, 1.0);
                    break :blk id;
                };
            } else {
                q_norm[l] = try allocator.alloc(f32, HD);
                @memset(q_norm[l], 1.0);
            }
            if (cfg.has_k_norm and lo[l].kn_off > 0) {
                k_norm[l] = loadBf16Weights(allocator, weight_data, lo[l].kn_off, HD) catch blk: {
                    const id = try allocator.alloc(f32, HD);
                    @memset(id, 1.0);
                    break :blk id;
                };
            } else {
                k_norm[l] = try allocator.alloc(f32, HD);
                @memset(k_norm[l], 1.0);
            }
            // Also clip QK-norm weights — same growth pathology
            for (0..HD) |i| {
                q_norm[l][i] = @max(-clip, @min(clip, q_norm[l][i]));
                k_norm[l][i] = @max(-clip, @min(clip, k_norm[l][i]));
            }
        }

        const final_norm = blk: {
            const fn_off = model_reader.findTensorInfo(js, "model.norm.weight") orelse 0;
            if (fn_off > 0) {
                const fw = loadBf16Weights(allocator, weight_data, fn_off, H) catch {
                    const id = try allocator.alloc(f32, H);
                    @memset(id, 1.0);
                    break :blk id;
                };
                for (0..H) |i| fw[i] = @max(-clip, @min(clip, fw[i]));
                break :blk fw;
            } else {
                const id = try allocator.alloc(f32, H);
                @memset(id, 1.0);
                break :blk id;
            }
        };

        // ---- RoPE ----
        const rope_tables = try cpu_ops.precomputeRoPE(allocator, HD, 4096, cfg.rope_theta);

        // ---- KV cache ----
        const kv_page_size: u32 = 16;
        var page_pool = try KvPagePool.initWithEviction(
            allocator, total_kv_pages, kv_page_size, EvictionPolicy.h2o_attention_score,
        );
        var page_table = NpuPageTable.init(kv_page_size, NKV, NC, HD);
        try page_table.allocKVBO(&device, page_pool.usablePageCount());

        // ---- Scheduler ----
        const scheduler = try Scheduler.init(allocator, max_parallel, &page_pool, 2);

        // ---- Workspace ----
        const h_buf = try allocator.alloc(f32, XM * H);
        const qkv_buf = try allocator.alloc(f32, XM * qkv_total);
        const attn_buf = try allocator.alloc(f32, XM * NH * HD);
        const o_buf = try allocator.alloc(f32, XM * H);
        const gate_buf = try allocator.alloc(f32, XM * mlp_out);
        const silu_buf = try allocator.alloc(f32, XM * IM);
        const d_buf = try allocator.alloc(f32, XM * H);
        const scratch = try allocator.alloc(f32, XM * H);
        const logits_buf = try allocator.alloc(f32, NV);
        const tok_ids = try allocator.alloc(u32, XM);

        return NpuEngine{
            .allocator = allocator,
            .config = cfg,
            .scheduler = scheduler,
            .page_pool = page_pool,
            .page_table = page_table,
            .device = device,
            .kernel_qkv = kq,
            .kernel_o = ko,
            .kernel_gu = kg,
            .kernel_d = kd,
            .kernel_u = ku,
            .model_map = model_map,
            .emb_f32 = emb_f32,
            .lm_head_f32 = lm_head_f32,
            .in_norm = in_norm,
            .pa_norm = pa_norm,
            .q_norm = q_norm,
            .k_norm = k_norm,
            .final_norm = final_norm,
            .rope_rc = rope_tables.rc,
            .rope_rs = rope_tables.rs,
            .qkv_scales = qkv_scales,
            .o_scales = o_scales,
            .gu_scales = gu_scales,
            .d_scales = d_scales,
            .u_scales = usc,
            .h_buf = h_buf,
            .qkv_buf = qkv_buf,
            .attn_buf = attn_buf,
            .o_buf = o_buf,
            .gate_buf = gate_buf,
            .silu_buf = silu_buf,
            .d_buf = d_buf,
            .scratch = scratch,
            .logits_buf = logits_buf,
            .tok_ids = tok_ids,
            .kv_page_size = kv_page_size,
            .eos_token_id = 151645,
        };
    }

    pub fn deinit(self: *NpuEngine) void {
        const a = self.allocator;
        self.kernel_qkv.deinit();
        self.kernel_o.deinit();
        self.kernel_gu.deinit();
        self.kernel_d.deinit();
        if (self.kernel_u) |*ku| ku.deinit();
        self.page_table.deinit();
        self.page_pool.deinit();
        a.free(self.h_buf); a.free(self.qkv_buf); a.free(self.attn_buf);
        a.free(self.o_buf); a.free(self.gate_buf); a.free(self.silu_buf);
        a.free(self.d_buf); a.free(self.scratch);
        a.free(self.logits_buf); a.free(self.tok_ids);
        for (self.in_norm) |n| a.free(n); a.free(self.in_norm);
        for (self.pa_norm) |n| a.free(n); a.free(self.pa_norm);
        for (self.q_norm) |n| a.free(n); a.free(self.q_norm);
        for (self.k_norm) |n| a.free(n); a.free(self.k_norm);
        a.free(self.final_norm);
        a.free(self.emb_f32); a.free(self.lm_head_f32);
        a.free(self.rope_rc); a.free(self.rope_rs);
        a.free(self.qkv_scales); a.free(self.o_scales);
        a.free(self.gu_scales); a.free(self.d_scales);
        if (self.u_scales) |us| a.free(us);
        std.posix.munmap(self.model_map);
        self.device.close();
    }

    // ============================================================
    // runSimple
    // ============================================================

    pub fn runSimple(self: *NpuEngine, prompt_tokens: []const u32, max_tokens: u32) ![]u32 {
        const a = self.allocator;
        var generated = std.ArrayList(u32).init(a);
        errdefer generated.deinit();

        const H = self.config.H; const NC = self.config.NC;
        const NH = self.config.NH; const NKV = self.config.NKV;
        const HD = self.config.HD; const IM = self.config.IM;
        const NV = self.config.NV; _ = self.config.XM;
        const qkv_total = self.config.qkv_total;
        const mlp_out: usize = if (self.config.gu_split) IM else 2 * IM;
        const npt = @as(u32, @intCast(prompt_tokens.len));
        const eps: f32 = 1e-6;

        const max_pages = (npt + max_tokens + self.kv_page_size - 1) / self.kv_page_size;
        var mapping = try PageMapping.init(a, @max(max_pages, 4));
        defer mapping.deinit(a);

        const rid: u64 = 1;

        // ================================================================
        // PREFILL
        // ================================================================
        log.info("=== Prefill {d} tokens ===", .{npt});

        for (0..npt) |pi| {
            const tok = prompt_tokens[pi];
            @memcpy(self.h_buf[pi * H .. (pi + 1) * H], self.emb_f32[tok * H .. (tok + 1) * H]);
        }

        const init_pages = (npt + self.kv_page_size - 1) / self.kv_page_size;
        {
            const new_pages = try self.page_pool.allocOrEvict(rid, init_pages);
            for (new_pages) |pid| _ = try mapping.addPage(pid);
        }

        for (0..NC) |l| {
            // Save residuals
            for (0..npt) |pi| {
                @memcpy(self.scratch[pi * H .. (pi + 1) * H], self.h_buf[pi * H .. (pi + 1) * H]);
            }
            // Pre-attention RMS norm
            for (0..npt) |pi| {
                cpu_ops.rmsNorm(self.h_buf[pi * H .. (pi + 1) * H], self.in_norm[l], eps);
            }
            // QKV GEMM
            try self.kernel_qkv.run(
                @intCast(l), self.h_buf[0 .. npt * H],
                npt, self.config.xclbin_qkv_k,
                dynamicScale(self.h_buf[0 .. npt * H]), self.qkv_scales[l],
                self.qkv_buf[0 .. npt * qkv_total], qkv_total,
            );
            cpu_ops.clipAndClean(self.qkv_buf[0 .. npt * qkv_total]);

            // Q/K norm + RoPE + store KV
            const koff = self.config.qkv_k_offset;
            const voff = self.config.qkv_v_offset;
            for (0..npt) |pi| {
                const qb = self.qkv_buf[pi * qkv_total ..];
                for (0..NH) |hh| {
                    const qh = qb[hh * HD .. (hh + 1) * HD];
                    var sq: f64 = 0;
                    for (qh) |v| sq += @as(f64, v) * @as(f64, v);
                    const iq = 1.0 / @sqrt(sq / @as(f64, HD) + eps);
                    for (qh) |*v| v.* = @as(f32, @floatCast(@as(f64, v.*) * iq));
                    if (self.config.has_q_norm) {
                        for (0..HD) |d| qh[d] *= self.q_norm[l][d];
                    }
                    cpu_ops.rope(qh, HD, pi, self.rope_rc, self.rope_rs);
                }
                for (0..NKV) |kvh| {
                    const kh = qb[koff + kvh * HD .. koff + (kvh + 1) * HD];
                    var sk: f64 = 0;
                    for (kh) |v| sk += @as(f64, v) * @as(f64, v);
                    const ik = 1.0 / @sqrt(sk / @as(f64, HD) + eps);
                    for (kh) |*v| v.* = @as(f32, @floatCast(@as(f64, v.*) * ik));
                    if (self.config.has_k_norm) {
                        for (0..HD) |d| kh[d] *= self.k_norm[l][d];
                    }
                    cpu_ops.rope(kh, HD, pi, self.rope_rc, self.rope_rs);

                    const vh = qb[voff + kvh * HD .. voff + (kvh + 1) * HD];
                    const page_idx = pi / self.kv_page_size;
                    const t_in_page = pi % self.kv_page_size;
                    self.page_table.writeKV(@intCast(l), mapping.page_ids[page_idx], t_in_page, kh, vh);
                }
            }

            // CPU attention
            const cl = npt;
            var kl_scratch: [1024 * 8 * 128]f32 = undefined;
            var vl_scratch: [1024 * 8 * 128]f32 = undefined;
            const kctx = kl_scratch[0 .. cl * NKV * HD];
            const vctx = vl_scratch[0 .. cl * NKV * HD];
            self.page_table.readAllKV(@intCast(l), mapping.page_ids[0..mapping.pageCount()], cl, kctx, vctx);

            for (0..npt) |pi| {
                const qb_ = self.qkv_buf[pi * qkv_total ..];
                const ab = self.attn_buf[pi * NH * HD ..];
                for (0..NH) |hh| {
                    const qh = qb_[hh * HD .. (hh + 1) * HD];
                    const at = ab[hh * HD .. (hh + 1) * HD];
                    cpu_ops.attentionHead(qh, kctx, vctx, at, cl, hh, NH, NKV, HD, pi + 1);
                }
            }

            // O GEMM
            try self.kernel_o.run(
                @intCast(l), self.attn_buf[0 .. npt * NH * HD],
                npt, self.config.xclbin_o_k,
                dynamicScale(self.attn_buf[0 .. npt * NH * HD]), self.o_scales[l],
                self.o_buf[0 .. npt * H], H,
            );
            cpu_ops.clipAndClean(self.o_buf[0 .. npt * H]);
            // Residual add
            for (0..npt) |pi| {
                for (0..H) |i| {
                    self.h_buf[pi * H + i] = self.scratch[pi * H + i] + self.o_buf[pi * H + i];
                }
            }

            // ---- MLP ----
            for (0..npt) |pi| {
                @memcpy(self.scratch[pi * H .. (pi + 1) * H], self.h_buf[pi * H .. (pi + 1) * H]);
            }
            for (0..npt) |pi| {
                cpu_ops.rmsNorm(self.h_buf[pi * H .. (pi + 1) * H], self.pa_norm[l], eps);
            }
            // Gate/Up GEMM
            try self.kernel_gu.run(
                @intCast(l), self.h_buf[0 .. npt * H],
                npt, self.config.xclbin_gu_k,
                dynamicScale(self.h_buf[0 .. npt * H]), self.gu_scales[l],
                self.gate_buf[0 .. npt * mlp_out], @intCast(mlp_out),
            );
            cpu_ops.clipAndClean(self.gate_buf[0 .. npt * mlp_out]);

            if (self.config.gu_split and self.kernel_u != null) {
                try self.kernel_u.?.run(
                    @intCast(l), self.h_buf[0 .. npt * H],
                    npt, self.config.xclbin_u_k,
                    dynamicScale(self.h_buf[0 .. npt * H]), self.u_scales.?[l],
                    self.silu_buf[0 .. npt * IM], @intCast(IM),
                );
                cpu_ops.clipAndClean(self.silu_buf[0 .. npt * IM]);
                for (0..npt) |pi| {
                    for (0..IM) |i| {
                        self.silu_buf[pi * IM + i] = cpu_ops.silu(self.gate_buf[pi * IM + i]) * self.silu_buf[pi * IM + i];
                    }
                }
            } else {
                for (0..npt) |pi| {
                    for (0..IM) |i| {
                        self.silu_buf[pi * IM + i] = cpu_ops.silu(self.gate_buf[pi * mlp_out + i]) * self.gate_buf[pi * mlp_out + IM + i];
                    }
                }
            }

            // Down GEMM
            try self.kernel_d.run(
                @intCast(l), self.silu_buf[0 .. npt * IM],
                npt, self.config.xclbin_d_k,
                dynamicScale(self.silu_buf[0 .. npt * IM]), self.d_scales[l],
                self.d_buf[0 .. npt * H], H,
            );
            cpu_ops.clipAndClean(self.d_buf[0 .. npt * H]);
            // Residual add
            for (0..npt) |pi| {
                for (0..H) |i| {
                    self.h_buf[pi * H + i] = self.scratch[pi * H + i] + self.d_buf[pi * H + i];
                }
            }
        }

        mapping.token_count = npt;

        // Final norm + LM head
        {
            const last_h = self.h_buf[(npt - 1) * H .. npt * H];
            cpu_ops.rmsNorm(last_h, self.final_norm, eps);
            cpu_ops.lmHeadTopK(last_h, self.lm_head_f32, self.tok_ids, 32, NV, H);
        }
        try generated.append(self.tok_ids[0]);
        self.recordH2OScores(&mapping);

        // ================================================================
        // DECODE
        // ================================================================
        log.info("=== Decode {d} tokens ===", .{max_tokens});

        var pos = npt;
        while (pos < npt + max_tokens - 1) {
            const last_tok = generated.items[generated.items.len - 1];
            @memcpy(self.h_buf[0..H], self.emb_f32[last_tok * H .. (last_tok + 1) * H]);

            // Allocate new page if needed
            if (pos % self.kv_page_size == 0 and pos > 0) {
                const new_pages = try self.page_pool.allocOrEvict(rid, 1);
                for (new_pages) |pid| _ = try mapping.addPage(pid);
            }

            const page_idx = pos / self.kv_page_size;
            const t_in_page = pos % self.kv_page_size;

            for (0..NC) |l| {
                // Save residual
                for (0..H) |i| self.scratch[i] = self.h_buf[i];
                // Pre-attention RMS norm
                cpu_ops.rmsNorm(self.h_buf[0..H], self.in_norm[l], eps);
                // QKV GEMM
                try self.kernel_qkv.run(
                    @intCast(l), self.h_buf[0..H], 1, self.config.xclbin_qkv_k,
                    dynamicScale(self.h_buf[0..H]), self.qkv_scales[l],
                    self.qkv_buf[0..qkv_total], qkv_total,
                );
                cpu_ops.clipAndClean(self.qkv_buf[0..qkv_total]);

                // Q norm + RoPE
                for (0..NH) |hh| {
                    const qh = self.qkv_buf[hh * HD .. (hh + 1) * HD];
                    var sq: f64 = 0;
                    for (qh) |v| sq += @as(f64, v) * @as(f64, v);
                    const iq = 1.0 / @sqrt(sq / @as(f64, HD) + eps);
                    for (qh) |*v| v.* = @as(f32, @floatCast(@as(f64, v.*) * iq));
                    if (self.config.has_q_norm) {
                        for (0..HD) |d| qh[d] *= self.q_norm[l][d];
                    }
                    cpu_ops.rope(qh, HD, pos, self.rope_rc, self.rope_rs);
                }

                // K norm + RoPE + store KV
                {
                    const koff = self.config.qkv_k_offset;
                    const voff = self.config.qkv_v_offset;
                    for (0..NKV) |kvh| {
                        const kh = self.qkv_buf[koff + kvh * HD .. koff + (kvh + 1) * HD];
                        var sk: f64 = 0;
                        for (kh) |v| sk += @as(f64, v) * @as(f64, v);
                        const ik = 1.0 / @sqrt(sk / @as(f64, HD) + eps);
                        for (kh) |*v| v.* = @as(f32, @floatCast(@as(f64, v.*) * ik));
                        if (self.config.has_k_norm) {
                            for (0..HD) |d| kh[d] *= self.k_norm[l][d];
                        }
                        cpu_ops.rope(kh, HD, pos, self.rope_rc, self.rope_rs);
                        const vh = self.qkv_buf[voff + kvh * HD .. voff + (kvh + 1) * HD];
                        self.page_table.writeKV(@intCast(l), mapping.page_ids[page_idx], t_in_page, kh, vh);
                    }
                }

                // CPU attention
                const cl = pos + 1;
                var kl_scratch: [1024 * 8 * 128]f32 = undefined;
                var vl_scratch: [1024 * 8 * 128]f32 = undefined;
                const kctx = kl_scratch[0 .. cl * NKV * HD];
                const vctx = vl_scratch[0 .. cl * NKV * HD];
                self.page_table.readAllKV(@intCast(l), mapping.page_ids[0..mapping.pageCount()], cl, kctx, vctx);

                for (0..NH) |hh| {
                    const qh = self.qkv_buf[hh * HD .. (hh + 1) * HD];
                    const at = self.attn_buf[hh * HD .. (hh + 1) * HD];
                    cpu_ops.attentionHead(qh, kctx, vctx, at, cl, hh, NH, NKV, HD, null);
                }

                // O GEMM
                try self.kernel_o.run(
                    @intCast(l), self.attn_buf[0 .. NH * HD], 1, self.config.xclbin_o_k,
                    dynamicScale(self.attn_buf[0 .. NH * HD]), self.o_scales[l],
                    self.o_buf[0..H], H,
                );
                cpu_ops.clipAndClean(self.o_buf[0..H]);
                for (0..H) |i| self.h_buf[i] = self.scratch[i] + self.o_buf[i];

                // ---- MLP ----
                for (0..H) |i| self.scratch[i] = self.h_buf[i];
                cpu_ops.rmsNorm(self.h_buf[0..H], self.pa_norm[l], eps);

                try self.kernel_gu.run(
                    @intCast(l), self.h_buf[0..H], 1, self.config.xclbin_gu_k,
                    dynamicScale(self.h_buf[0..H]), self.gu_scales[l],
                    self.gate_buf[0..mlp_out], @intCast(mlp_out),
                );
                cpu_ops.clipAndClean(self.gate_buf[0..mlp_out]);

                if (self.config.gu_split and self.kernel_u != null) {
                    try self.kernel_u.?.run(
                        @intCast(l), self.h_buf[0..H], 1, self.config.xclbin_u_k,
                        dynamicScale(self.h_buf[0..H]), self.u_scales.?[l],
                        self.silu_buf[0..IM], @intCast(IM),
                    );
                    cpu_ops.clipAndClean(self.silu_buf[0..IM]);
                    for (0..IM) |i| {
                        self.silu_buf[i] = cpu_ops.silu(self.gate_buf[i]) * self.silu_buf[i];
                    }
                } else {
                    for (0..IM) |i| {
                        self.silu_buf[i] = cpu_ops.silu(self.gate_buf[i]) * self.gate_buf[IM + i];
                    }
                }

                try self.kernel_d.run(
                    @intCast(l), self.silu_buf[0..IM], 1, self.config.xclbin_d_k,
                    dynamicScale(self.silu_buf[0..IM]), self.d_scales[l],
                    self.d_buf[0..H], H,
                );
                cpu_ops.clipAndClean(self.d_buf[0..H]);
                for (0..H) |i| self.h_buf[i] = self.scratch[i] + self.d_buf[i];
            }

            // Final norm + LM head
            cpu_ops.rmsNorm(self.h_buf[0..H], self.final_norm, eps);
            cpu_ops.lmHeadTopK(self.h_buf[0..H], self.lm_head_f32, self.tok_ids, 32, NV, H);

            const next_tok = self.tok_ids[0];
            try generated.append(next_tok);
            mapping.token_count = pos + 1;
            self.recordH2OScores(&mapping);

            pos += 1;
            if (next_tok == self.eos_token_id) break;
        }

        log.info("Generated {d} tokens", .{generated.items.len});
        self.page_pool.freePages(rid);
        return try generated.toOwnedSlice(a);
    }

    fn recordH2OScores(self: *NpuEngine, mapping: *const PageMapping) void {
        if (self.page_pool.eviction_policy != .h2o_attention_score) return;
        const pc = mapping.pageCount();
        if (pc == 0) return;
        self.page_pool.recordScores(mapping.page_ids[0..pc], null);
    }
};

// ---- Helpers ----

fn dequantToSliceEx(data: []const u8, i8_rows: u32, in_features: u32) ![]f32 {
    const result = try cdequant.dequantToSlice(std.heap.page_allocator, data, i8_rows, in_features);
    return result.data;
}

fn transposePack(src: []const f32, out_f: u32, in_f: u32, dst: []f32, dst_stride: u32, dst_offset: u32) void {
    for (0..out_f) |o| {
        for (0..in_f) |i| {
            dst[i * dst_stride + dst_offset + o] = src[o * in_f + i];
        }
    }
}

fn dynamicScale(x: []const f32) f32 {
    return cpu_ops.findMaxAbs(x) / 127.0;
}

fn loadBf16Weights(allocator: std.mem.Allocator, data: []const u8, offset: u32, count: u32) ![]f32 {
    if (offset == 0 or offset + count * 2 > data.len) return error.OffsetOutOfRange;
    const bf16_slice: [*]const u16 = @ptrCast(@alignCast(&data[offset]));
    const result = try allocator.alloc(f32, count);
    for (0..count) |i| result[i] = model_reader.bf16ToF32Safe(bf16_slice[i]);
    return result;
}

fn findOff(js: []const u8, buf: []u8, comptime fmt_str: []const u8, args: anytype) ?u32 {
    const key = std.fmt.bufPrint(buf, fmt_str, args) catch return null;
    const null_idx = std.mem.indexOfScalar(u8, key, 0) orelse key.len;
    return model_reader.findTensorInfo(js, key[0..null_idx]);
}

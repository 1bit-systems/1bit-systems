.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
.text
.globl zinc_rt_dmmv_q4_0_resident_grid
.type zinc_rt_dmmv_q4_0_resident_grid,@function

// Grid-over-rows Q4_0 dequant-matvec. One wave (64 lanes) per workgroup; each
// lane owns one output row. workgroup_id_x (ttmp9 on gfx11/gfx12) selects the
// 64-row block, so a grid of ceil(total_rows/64) workgroups covers all rows in
// ONE submit and spreads across all CUs.
//
// ABI:
//   s[0:1] = input f32 vector pointer (length cols)
//   s[2:3] = output f32 pointer (length total_rows), indexed by GLOBAL row
//   s[4:5] = Q4_0 weight base pointer (all rows, row-major; may be VRAM-resident)
//   s6     = cols (multiple of 32)
//   s7     = total_rows
//   ttmp9  = workgroup_id_x ; v0 = workitem_id_x (lane 0..63)
//   COMPUTE_PGM_RSRC2 must enable workgroup_id_x (bit 7) and VGPR workitem id.
//
// global_row = ttmp9*64 + lane; lanes with global_row >= total_rows are masked
// off via EXEC so they neither load weights nor store. Q4_0 block = 18 bytes
// (f16 scale + 16 nibble bytes = 32 weights); y[row] = sum_k (nibble-8)*scale*x.
zinc_rt_dmmv_q4_0_resident_grid:
    s_lshl_b32 s8, ttmp9, 6              // s8 = workgroup_id_x * 64 (block base row)
    v_add_nc_u32_e32 v8, s8, v0          // v8 = global_row = block_base + lane
    v_mov_b32_e32 v15, s7
    v_cmpx_lt_u32_e32 v8, v15            // EXEC = lanes with global_row < total_rows
    s_cbranch_execz done                 // whole wave out of range -> nothing to do

    v_mov_b32_e32 v1, 0                  // v1 = accumulator
    s_lshr_b32 s10, s6, 5                // s10 = num_blocks = cols/32
    v_mul_lo_u32 v9, s10, v8             // v9 = global_row * num_blocks  (32-bit: rows can be >64k)
    v_mul_lo_u32 v9, 18, v9             // v9 = row byte offset into the weight base
    s_mov_b32 s11, 0                     // s11 = block index

block_loop:
    s_cmp_ge_u32 s11, s10
    s_cbranch_scc1 store_row

    s_mul_i32 s13, s11, 18              // block byte offset within the row
    v_add_nc_u32_e32 v10, s13, v9        // v10 = byte offset of this block's scale
    global_load_ushort v2, v10, s[4:5]   // f16 block scale

    s_mul_i32 s14, s11, 32              // first column of this block
    s_mov_b32 s15, 0                     // s15 = j (0..15)
    s_waitcnt vmcnt(0)
    v_cvt_f32_f16_e32 v2, v2             // scale -> f32

j_loop:
    s_cmp_ge_u32 s15, 16
    s_cbranch_scc1 next_block

    s_add_u32 s16, s13, 2
    s_add_u32 s16, s16, s15             // nibble byte offset within row = s13 + 2 + j
    v_add_nc_u32_e32 v11, s16, v9
    global_load_ubyte v3, v11, s[4:5]    // packed nibble pair

    s_add_u32 s17, s14, s15
    s_lshl_b32 s17, s17, 2
    v_mov_b32_e32 v12, s17
    global_load_b32 v6, v12, s[0:1]      // input[col]

    s_add_u32 s18, s14, s15
    s_add_u32 s18, s18, 16
    s_lshl_b32 s18, s18, 2
    v_mov_b32_e32 v13, s18
    global_load_b32 v7, v13, s[0:1]      // input[col+16]

    s_waitcnt vmcnt(0)
    v_and_b32_e32 v4, 0x0f, v3
    v_lshrrev_b32_e32 v5, 4, v3
    v_add_nc_u32_e32 v4, -8, v4
    v_add_nc_u32_e32 v5, -8, v5
    v_cvt_f32_i32_e32 v4, v4
    v_cvt_f32_i32_e32 v5, v5
    v_mul_f32_e32 v4, v2, v4
    v_fmac_f32_e32 v1, v4, v6
    v_mul_f32_e32 v5, v2, v5
    v_fmac_f32_e32 v1, v5, v7

    s_add_u32 s15, s15, 1
    s_branch j_loop

next_block:
    s_add_u32 s11, s11, 1
    s_branch block_loop

store_row:
    v_lshlrev_b32_e32 v14, 2, v8         // output byte offset = global_row * 4
    global_store_b32 v14, v1, s[2:3]
    s_branch done

done:
    s_nop 0
    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
    s_endpgm

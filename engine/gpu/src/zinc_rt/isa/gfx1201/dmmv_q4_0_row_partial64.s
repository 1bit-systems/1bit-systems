.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
.text
.globl zinc_rt_dmmv_q4_0_row_partial64
.type zinc_rt_dmmv_q4_0_row_partial64,@function

// ABI:
//   s[0:1] = input f32 vector pointer
//   s[2:3] = output f32 partial sums pointer (64 floats)
//   s[4:5] = one Q4_0 weight row pointer
//   s6     = cols, multiple of 32
//   s7     = unused
//   v0     = workitem_id_x / lane id
//
// Lanes 0..31 each accumulate one element from every Q4_0 block. Lanes 32..63
// write zero so the host can reduce a fixed 64-float partial buffer.
zinc_rt_dmmv_q4_0_row_partial64:
    v_mov_b32_e32 v8, v0
    v_mov_b32_e32 v1, 0
    v_and_b32_e32 v21, 31, v8
    v_and_b32_e32 v22, 15, v8
    s_lshr_b32 s10, s6, 5
    s_mov_b32 s11, 0

block_loop:
    s_cmp_ge_u32 s11, s10
    s_cbranch_scc1 store_partial

    s_mul_i32 s13, s11, 18
    v_mov_b32_e32 v10, s13
    global_load_ushort v2, v10, s[4:5]

    s_mul_i32 s14, s11, 32
    s_lshl_b32 s17, s14, 2
    v_lshlrev_b32_e32 v12, 2, v21
    v_add_nc_u32_e32 v12, s17, v12
    global_load_b32 v6, v12, s[0:1]

    s_add_u32 s16, s13, 2
    v_add_nc_u32_e32 v11, s16, v22
    global_load_ubyte v3, v11, s[4:5]

    s_waitcnt vmcnt(0)
    v_cvt_f32_f16_e32 v2, v2
    v_and_b32_e32 v4, 0x0f, v3
    v_lshrrev_b32_e32 v5, 4, v3
    v_mov_b32_e32 v30, 16
    v_cmp_lt_u32_e32 v21, v30
    v_cndmask_b32_e32 v4, v5, v4
    v_add_nc_u32_e32 v4, -8, v4
    v_cvt_f32_i32_e32 v4, v4
    v_mul_f32_e32 v4, v2, v4
    v_fmac_f32_e32 v1, v4, v6

    s_add_u32 s11, s11, 1
    s_branch block_loop

store_partial:
    v_mov_b32_e32 v30, 32
    v_cmp_lt_u32_e32 v8, v30
    v_cndmask_b32_e32 v1, 0, v1
    v_lshlrev_b32_e32 v14, 2, v8
    global_store_b32 v14, v1, s[2:3]

    s_nop 0
    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
    s_endpgm

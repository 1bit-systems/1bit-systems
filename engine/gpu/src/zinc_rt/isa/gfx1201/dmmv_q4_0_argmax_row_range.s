.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
.text
.globl zinc_rt_dmmv_q4_0_argmax_row_range
.type zinc_rt_dmmv_q4_0_argmax_row_range,@function

// ABI:
//   s[0:1] = input f32 vector pointer
//   s[2:3] = output pointer: u32 local row, f32 score bits
//   s[4:5] = Q4_0 weight rows pointer
//   s6     = cols, multiple of 32
//   s7     = rows
//
// One workitem serially evaluates a compact Q4_0 row range and keeps the
// best row in-register. This is still a correctness-oriented M1 slice, but it
// lets the LM-head prefix/window path consume a GPU-selected model value
// without a separate argmax dispatch over GPU-written logits.
zinc_rt_dmmv_q4_0_argmax_row_range:
    s_mov_b32 s9, 0
    s_lshr_b32 s10, s6, 5
    v_mov_b32_e32 v8, 0xff800000
    v_mov_b32_e32 v9, 0

row_loop:
    s_cmp_ge_u32 s9, s7
    s_cbranch_scc1 done

    v_mov_b32_e32 v1, 0
    s_mul_i32 s12, s9, s10
    s_mul_i32 s12, s12, 18
    s_mov_b32 s11, 0

block_loop:
    s_cmp_ge_u32 s11, s10
    s_cbranch_scc1 offer_row

    s_mul_i32 s13, s11, 18
    s_add_u32 s13, s12, s13
    v_mov_b32_e32 v0, s13
    global_load_ushort v2, v0, s[4:5]

    s_mul_i32 s14, s11, 32
    s_mov_b32 s15, 0
    s_waitcnt vmcnt(0)
    v_cvt_f32_f16_e32 v2, v2

j_loop:
    s_cmp_ge_u32 s15, 16
    s_cbranch_scc1 next_block

    s_add_u32 s16, s13, 2
    s_add_u32 s16, s16, s15
    v_mov_b32_e32 v0, s16
    global_load_ubyte v3, v0, s[4:5]

    s_add_u32 s17, s14, s15
    s_lshl_b32 s17, s17, 2
    v_mov_b32_e32 v0, s17
    global_load_b32 v6, v0, s[0:1]

    s_add_u32 s18, s14, s15
    s_add_u32 s18, s18, 16
    s_lshl_b32 s18, s18, 2
    v_mov_b32_e32 v0, s18
    global_load_b32 v7, v0, s[0:1]

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

offer_row:
    v_cmp_gt_f32_e32 v1, v8
    v_mov_b32_e32 v10, s9
    v_cndmask_b32_e32 v8, v8, v1
    v_cndmask_b32_e32 v9, v9, v10

    s_add_u32 s9, s9, 1
    s_branch row_loop

done:
    v_mov_b32_e32 v0, 0
    global_store_b32 v0, v9, s[2:3]
    v_mov_b32_e32 v0, 4
    global_store_b32 v0, v8, s[2:3]

    s_nop 0
    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
    s_endpgm

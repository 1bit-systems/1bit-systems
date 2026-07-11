.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
.text
.globl zinc_rt_sgpr_tgid_dump
.type zinc_rt_sgpr_tgid_dump,@function

// ABI:
//   s[0:1] = output u32 dump pointer
//   s[2:3] = shader-written signal pointer
//   s4     = dispatch groups
//   s5     = signal low dword
//   s6     = signal high dword
//   s7     = unused
//   s8..   = candidate workgroup-id / system SGPRs
//
// Each candidate value < groups is used as a row id and stores its raw value at:
//   output[(candidate * 16) + candidate_index]
// This distinguishes "TGID is in s8/s16/etc." from "all workgroups collide at
// row 0" without trusting any one candidate for address selection.
zinc_rt_sgpr_tgid_dump:
    // candidate 0: s8
    s_cmp_ge_u32 s8, s4
    s_cbranch_scc1 cand1
    s_lshl_b32 s20, s8, 6
    v_mov_b32_e32 v0, s20
    v_mov_b32_e32 v1, s8
    global_store_b32 v0, v1, s[0:1]

cand1:
    s_cmp_ge_u32 s9, s4
    s_cbranch_scc1 cand2
    s_lshl_b32 s20, s9, 6
    s_add_u32 s20, s20, 4
    v_mov_b32_e32 v0, s20
    v_mov_b32_e32 v1, s9
    global_store_b32 v0, v1, s[0:1]

cand2:
    s_cmp_ge_u32 s10, s4
    s_cbranch_scc1 cand3
    s_lshl_b32 s20, s10, 6
    s_add_u32 s20, s20, 8
    v_mov_b32_e32 v0, s20
    v_mov_b32_e32 v1, s10
    global_store_b32 v0, v1, s[0:1]

cand3:
    s_cmp_ge_u32 s11, s4
    s_cbranch_scc1 cand4
    s_lshl_b32 s20, s11, 6
    s_add_u32 s20, s20, 12
    v_mov_b32_e32 v0, s20
    v_mov_b32_e32 v1, s11
    global_store_b32 v0, v1, s[0:1]

cand4:
    s_cmp_ge_u32 s12, s4
    s_cbranch_scc1 cand5
    s_lshl_b32 s20, s12, 6
    s_add_u32 s20, s20, 16
    v_mov_b32_e32 v0, s20
    v_mov_b32_e32 v1, s12
    global_store_b32 v0, v1, s[0:1]

cand5:
    s_cmp_ge_u32 s13, s4
    s_cbranch_scc1 cand6
    s_lshl_b32 s20, s13, 6
    s_add_u32 s20, s20, 20
    v_mov_b32_e32 v0, s20
    v_mov_b32_e32 v1, s13
    global_store_b32 v0, v1, s[0:1]

cand6:
    s_cmp_ge_u32 s14, s4
    s_cbranch_scc1 cand7
    s_lshl_b32 s20, s14, 6
    s_add_u32 s20, s20, 24
    v_mov_b32_e32 v0, s20
    v_mov_b32_e32 v1, s14
    global_store_b32 v0, v1, s[0:1]

cand7:
    s_cmp_ge_u32 s15, s4
    s_cbranch_scc1 cand8
    s_lshl_b32 s20, s15, 6
    s_add_u32 s20, s20, 28
    v_mov_b32_e32 v0, s20
    v_mov_b32_e32 v1, s15
    global_store_b32 v0, v1, s[0:1]

cand8:
    s_cmp_ge_u32 s16, s4
    s_cbranch_scc1 cand9
    s_lshl_b32 s20, s16, 6
    s_add_u32 s20, s20, 32
    v_mov_b32_e32 v0, s20
    v_mov_b32_e32 v1, s16
    global_store_b32 v0, v1, s[0:1]

cand9:
    s_cmp_ge_u32 s17, s4
    s_cbranch_scc1 cand10
    s_lshl_b32 s20, s17, 6
    s_add_u32 s20, s20, 36
    v_mov_b32_e32 v0, s20
    v_mov_b32_e32 v1, s17
    global_store_b32 v0, v1, s[0:1]

cand10:
    s_cmp_ge_u32 s18, s4
    s_cbranch_scc1 cand11
    s_lshl_b32 s20, s18, 6
    s_add_u32 s20, s20, 40
    v_mov_b32_e32 v0, s20
    v_mov_b32_e32 v1, s18
    global_store_b32 v0, v1, s[0:1]

cand11:
    s_cmp_ge_u32 s19, s4
    s_cbranch_scc1 signal
    s_lshl_b32 s20, s19, 6
    s_add_u32 s20, s20, 44
    v_mov_b32_e32 v0, s20
    v_mov_b32_e32 v1, s19
    global_store_b32 v0, v1, s[0:1]

signal:
    v_mov_b32_e32 v0, 3840
    v_mov_b32_e32 v1, 0x53475052
    global_store_b32 v0, v1, s[0:1]

    v_mov_b32_e32 v0, 0
    v_mov_b32_e32 v1, s5
    global_store_b32 v0, v1, s[2:3]
    v_mov_b32_e32 v0, 4
    v_mov_b32_e32 v1, s6
    global_store_b32 v0, v1, s[2:3]

    s_waitcnt vmcnt(0)
    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
    s_endpgm

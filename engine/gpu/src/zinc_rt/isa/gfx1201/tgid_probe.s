.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
.text
.globl zinc_rt_tgid_probe
.type zinc_rt_tgid_probe,@function

// ABI:
//   s[0:1] = output u32 pointer (one slot per workgroup)
//   workgroup_id_x is delivered in ttmp9 on gfx11/gfx12 (RDNA3/RDNA4) — NOT in
//   an s8-style SGPR as on gfx6-10. Enabled by COMPUTE_PGM_RSRC2 bit 7
//   (ENABLE_SGPR_WORKGROUP_ID_X); the CP writes the id into ttmp9.
//   num_thread_x = 1, grid = (groups,1,1).
//
// Each workgroup stores its own id at output[workgroup_id_x]. A correct
// multi-workgroup dispatch yields output = [0,1,2,...,groups-1].
zinc_rt_tgid_probe:
    v_mov_b32_e32 v1, ttmp9          // v1 = workgroup_id_x
    v_lshlrev_b32_e32 v0, 2, v1      // v0 = workgroup_id_x * 4 (byte offset)
    global_store_b32 v0, v1, s[0:1]
    s_waitcnt vmcnt(0)
    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
    s_endpgm

#!/usr/bin/env python3
"""Apply optimizations #1, #3, #4, #5 to zaya_engine.cpp (skipping #2 graph capture)."""

import os, sys
base = '/home/bcloud/1bit-systems'

# ─── Step 1: Kernel files ───
# fixup_skip_expert.hip: new signature, no buffer zeroing
with open(f'{base}/kernels/zaya_skip_fixup.hip','w') as f:
    f.write('''// fixup_skip_expert_kernel -- GPU-side skip handling for MoE routing (#1).
// When router selects the "skip" expert (idx == n_exp), this kernel:
// 1. Maps expert_idx to 0 (valid expert for gateup/down to use)
// 2. Sets skip_flag = 1 so gateup/down kernels know to output zeros
// 3. Does NOT zero any buffers -- gateup/down handle the skip
__global__ void fixup_skip_expert_kernel(int* __restrict__ expert_idx, int* __restrict__ skip_flag, int n_exp, int n_exp_plus1){
    int idx=*expert_idx;
    if(idx==n_exp||idx>=n_exp_plus1){*expert_idx=0; *skip_flag=1;}
    else{*skip_flag=0;}
}
''')
print('1: fixup_skip_expert.hip updated')

# moe_expert_ffn.hip: add skip_flag to gateup/down
with open(f'{base}/kernels/zaya_moe_expert_ffn.hip') as f:
    orig = f.read()

# Update gateup: add skip_flag param and early-out
orig = orig.replace(
    'const int*    __restrict__ d_expert_idx) // GPU pointer to selected expert',
    'const int*    __restrict__ d_expert_idx, // GPU pointer to selected expert\n    const int*    __restrict__ d_skip_flag)  // GPU pointer: 1 = skip FFN, output zeros')
orig = orig.replace(
    '    int expert = *d_expert_idx;\n    if (expert >= N_EXP) return;  // passthrough handled by caller\n\n    int row_block = blockIdx.x;\n    int row = row_block * WMMA_M;\n    if (row >= 2 * N_FF) return;',
    '    int row_block = blockIdx.x;\n    int row = row_block * WMMA_M;\n\n    if (*d_skip_flag) {\n        if (row < 2 * N_FF) {\n            for (int j = 0; j < WMMA_M; j++)\n                if (row + j < 2 * N_FF)\n                    tmp[row + j] = __float2half(0.0f);\n        }\n        return;\n    }\n\n    int expert = *d_expert_idx;\n    if (expert >= N_EXP) return;')

# Update down: add skip_flag param and early-out
orig = orig.replace(
    'const int*    __restrict__ d_expert_idx) // GPU pointer to selected expert',
    'const int*    __restrict__ d_expert_idx, // GPU pointer to selected expert\n    const int*    __restrict__ d_skip_flag)  // GPU pointer: 1 = skip FFN, output zeros')
orig = orig.replace(
    '    int expert = *d_expert_idx;\n    if (expert >= N_EXP) return;\n\n    int row_block = blockIdx.x;\n    int row = row_block * WMMA_M;\n    if (row >= H) return;',
    '    int row_block = blockIdx.x;\n    int row = row_block * WMMA_M;\n\n    if (*d_skip_flag) {\n        if (row < H) {\n            for (int j = 0; j < WMMA_M; j++)\n                if (row + j < H)\n                    out[row + j] = __float2half(0.0f);\n        }\n        return;\n    }\n\n    int expert = *d_expert_idx;\n    if (expert >= N_EXP) return;')

with open(f'{base}/kernels/zaya_moe_expert_ffn.hip','w') as f:
    f.write(orig)
print('2: moe_expert_ffn.hip updated')

# ─── Step 2: zaya_engine.h ───
with open(f'{base}/src/zaya_engine.h') as f:
    h = f.read()

# Add d_ibias, d_iscale, use_linear_kv
old_h = '    __half *d_conv = nullptr, *d_phs = nullptr, *d_lm_vocab = nullptr;\n    // Paged KV cache: flat pooled allocation'
new_h = '    __half *d_conv = nullptr, *d_phs = nullptr, *d_lm_vocab = nullptr;\n    // Device-side embeddings: eliminates per-token H2D copy (#5)\n    __half *d_ibias = nullptr, *d_iscale = nullptr;\n    // Linear KV cache: contiguous [n_layers, max_seq, NKV, HD] (#3)\n    bool use_linear_kv = true;\n    // Paged KV cache: flat pooled allocation'
h = h.replace(old_h, new_h)

# Update d_skip_flag comment
h = h.replace("int *d_skip_flag = nullptr;  // fixup_skip_expert flag",
               "int *d_skip_flag = nullptr;  // fixup_skip_expert flag (GPU-side, no sync #1)")

with open(f'{base}/src/zaya_engine.h','w') as f:
    f.write(h)
print('3: zaya_engine.h updated')

# ─── Step 3: zaya_engine.cpp ───
with open(f'{base}/src/zaya_engine.cpp') as f:
    c = f.read()

# --- #5: Add embed_lookup_k kernel ---
old_helpers = '// ── Helper kernels ──\n__global__ void rmsnorm_k'
new_helpers = '''// ── Embedding lookup kernel (#5): avoids per-token H2D copy ──
__global__ void embed_lookup_k(__half* out, const __half* embed, const __half* ibias, const __half* iscale, int token_id, int h){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h) return;
    float raw = (float)embed[(size_t)token_id * h + i];
    out[i] = __float2half((raw + (float)ibias[i]) * (float)iscale[i]);
}

// ── Helper kernels ──
__global__ void rmsnorm_k'''
c = c.replace(old_helpers, new_helpers)
print('4: embed_lookup_k added')

# --- #4: Add fused_qkv include ---
c = c.replace('#include "zaya_moe_tiled_gemv.hip"\n#include "zaya_moe_expert_ffn.hip"',
              '#include "zaya_moe_tiled_gemv.hip"\n#include "zaya_fused_qkv.hip"\n#include "zaya_moe_expert_ffn.hip"')
print('5: fused_qkv include added')

# --- #5: Allocate d_ibias/d_iscale ---
c = c.replace('ALLOC_OR_FAIL(s, alloc_f16, s->d_embed, eng.vocab * eng.h);\n    ALLOC_OR_FAIL(s, alloc_f16, s->d_conv,',
              'ALLOC_OR_FAIL(s, alloc_f16, s->d_embed, eng.vocab * eng.h);\n    ALLOC_OR_FAIL(s, alloc_f16, s->d_ibias, eng.h);\n    ALLOC_OR_FAIL(s, alloc_f16, s->d_iscale, eng.h);\n    ALLOC_OR_FAIL(s, alloc_f16, s->d_conv,')
print('6: d_ibias/d_iscale allocated')

# --- #5: Upload ibias/iscale to GPU ---
c = c.replace('''    upf16(s->embed,s->d_embed,eng.vocab*eng.h,s->st);
    std::vector<__half>hf(eng.h);for(int i=0;i<eng.h;i++)hf[i]=__float2half(fnorm[i]);
    HIP_OK_R(hipMemcpy(s->d_fnw,hf.data(),eng.h*2,hipMemcpyHostToDevice), nullptr);''',
              '''    upf16(s->embed,s->d_embed,eng.vocab*eng.h,s->st);
    // Upload ibias/iscale to GPU for device-side embedding lookup (#5)
    std::vector<__half> h_ibias(eng.h), h_iscale(eng.h);
    for(int i=0;i<eng.h;i++){h_ibias[i]=__float2half(s->ibias[i]);h_iscale[i]=__float2half(s->iscale[i]);}
    HIP_OK_R(hipMemcpy(s->d_ibias,h_ibias.data(),eng.h*2,hipMemcpyHostToDevice), nullptr);
    HIP_OK_R(hipMemcpy(s->d_iscale,h_iscale.data(),eng.h*2,hipMemcpyHostToDevice), nullptr);
    // Final norm
    std::vector<__half>hf(eng.h);for(int i=0;i<eng.h;i++)hf[i]=__float2half(fnorm[i]);
    HIP_OK_R(hipMemcpy(s->d_fnw,hf.data(),eng.h*2,hipMemcpyHostToDevice), nullptr);''')
print('7: ibias/iscale uploaded')

# --- #3: Linear KV cache ---
old_kv = '''    s->max_seq = eng.max_seq_len > 0 ? eng.max_seq_len : 4096;
    s->page_size = KV_PAGE_SIZE;
    s->n_kv_pages = (s->max_seq + s->page_size - 1) / s->page_size;
    s->kv_pool_pages = eng.kv_pool_pages > 0 ?
        std::min(eng.kv_pool_pages, s->n_kv_pages) :
        std::min(s->n_kv_pages, KV_DEFAULT_PAGES);
    if (s->kv_pool_pages < 1) s->kv_pool_pages = 1;
    int kv_pool_elems = eng.n_layers * s->kv_pool_pages * s->page_size * eng.nkv * eng.hd;
    ALLOC_OR_FAIL(s, alloc_f16, s->d_kcache, kv_pool_elems);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_vcache, kv_pool_elems);
    fprintf(stderr, "  KV cache: %d pages (%d tok/page, %d pool, %d max_seq) = %.1f MB\\n",
            s->n_kv_pages, s->page_size, s->kv_pool_pages, s->max_seq,
            (double)kv_pool_elems * 2 / (1024*1024));'''

new_kv = '''    s->max_seq = eng.max_seq_len > 0 ? eng.max_seq_len : 4096;
    if (s->use_linear_kv) {
        int kv_elems = eng.n_layers * s->max_seq * eng.nkv * eng.hd;
        ALLOC_OR_FAIL(s, alloc_f16, s->d_kcache, kv_elems);
        ALLOC_OR_FAIL(s, alloc_f16, s->d_vcache, kv_elems);
        fprintf(stderr, "  KV cache: linear contiguous %d tok x %d layers = %.1f MB\\n",
                s->max_seq, eng.n_layers, (double)kv_elems * 2 / (1024*1024));
    } else {
        s->page_size = KV_PAGE_SIZE;
        s->n_kv_pages = (s->max_seq + s->page_size - 1) / s->page_size;
        s->kv_pool_pages = eng.kv_pool_pages > 0 ?
            std::min(eng.kv_pool_pages, s->n_kv_pages) :
            std::min(s->n_kv_pages, KV_DEFAULT_PAGES);
        if (s->kv_pool_pages < 1) s->kv_pool_pages = 1;
        int kv_pool_elems = eng.n_layers * s->kv_pool_pages * s->page_size * eng.nkv * eng.hd;
        ALLOC_OR_FAIL(s, alloc_f16, s->d_kcache, kv_pool_elems);
        ALLOC_OR_FAIL(s, alloc_f16, s->d_vcache, kv_pool_elems);
        fprintf(stderr, "  KV cache: %d pages (%d tok/page, %d pool, %d max_seq) = %.1f MB\\n",
                s->n_kv_pages, s->page_size, s->kv_pool_pages, s->max_seq,
                (double)kv_pool_elems * 2 / (1024*1024));
    }'''
c = c.replace(old_kv, new_kv)
print('8: linear KV cache')

# --- #3: Conditional page table ---
old_page = '''    // Initialize page table: all pages start unallocated.
    // page_map[layer][logical_page] = pool_page (or -1 if evicted/unused)
    s->page_alloc.resize(eng.n_layers);
    s->page_map.resize(eng.n_layers);
    s->page_lru.resize(eng.n_layers);
    s->page_next_evict.resize(eng.n_layers);
    for (int il = 0; il < eng.n_layers; il++) {
        s->page_alloc[il].assign(s->n_kv_pages, false);
        s->page_map[il].assign(s->n_kv_pages, -1);
        s->page_lru[il].assign(s->kv_pool_pages, -1);
        s->page_next_evict[il] = 0;
    }
    // Gather scratch buffer for non-contiguous page assembly
    ALLOC_OR_FAIL(s, alloc_f16, s->d_k_gather, s->max_seq * eng.nkv * eng.hd);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_v_gather, s->max_seq * eng.nkv * eng.hd);
    ALLOC_OR_FAIL(s, alloc_f32, s->d_page_map, s->n_kv_pages);
    s->gather_seq_len = 0;'''

new_page = '''    // Page table / gather buffers only needed for paged KV cache (#3)
    if (!s->use_linear_kv) {
        s->page_alloc.resize(eng.n_layers);
        s->page_map.resize(eng.n_layers);
        s->page_lru.resize(eng.n_layers);
        s->page_next_evict.resize(eng.n_layers);
        for (int il = 0; il < eng.n_layers; il++) {
            s->page_alloc[il].assign(s->n_kv_pages, false);
            s->page_map[il].assign(s->n_kv_pages, -1);
            s->page_lru[il].assign(s->kv_pool_pages, -1);
            s->page_next_evict[il] = 0;
        }
        ALLOC_OR_FAIL(s, alloc_f16, s->d_k_gather, s->max_seq * eng.nkv * eng.hd);
        ALLOC_OR_FAIL(s, alloc_f16, s->d_v_gather, s->max_seq * eng.nkv * eng.hd);
        ALLOC_OR_FAIL(s, alloc_f32, s->d_page_map, s->n_kv_pages);
    }
    s->gather_seq_len = 0;'''
c = c.replace(old_page, new_page)
print('9: conditional page table')

# --- #1 + #4 + #5: Replace forward pass in both functions ---
# Replace the entire attention prep + MoE block in zaya_forward and zaya_forward_greedy
old_attn = '''        rmsnorm_k<<<1,BLK,0,s->st>>>(s->d_hs,l.nw,eng.h);
        HIP_CHECK(hipGetLastError());
        moe_tiled_gemv<<<eng.qd/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_hs,l.wq,eng.qd,eng.h);                           // q_raw
        HIP_CHECK(hipGetLastError());
        moe_tiled_gemv<<<eng.kd/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp+eng.qd,s->d_hs,l.wk,eng.kd,eng.h);                     // k_raw
        HIP_CHECK(hipGetLastError());
        moe_tiled_gemv<<<eng.kd/2/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp+eng.qd+eng.kd,s->d_hs,l.wv1,eng.kd/2,eng.h);        // v_cur
        HIP_CHECK(hipGetLastError());
        moe_tiled_gemv<<<eng.kd/2/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp+eng.qd+eng.kd+eng.kd/2,s->d_hs,l.wv2,eng.kd/2,eng.h); // v_del
        HIP_CHECK(hipGetLastError());'''

new_attn = '''        // Fused RMSNorm + Q/K/V1/V2: 5 kernel launches -> 1 (#4)
        {
            int total_blocks = (eng.qd + WMMA_M - 1) / WMMA_M
                             + (eng.kd + WMMA_M - 1) / WMMA_M
                             + ((eng.kd/2) + WMMA_M - 1) / WMMA_M
                             + ((eng.kd/2) + WMMA_M - 1) / WMMA_M;
            fused_rmsnorm_qkv_kernel<<<total_blocks, WMMA_THREADS, 0, s->st>>>(
                s->d_hs, l.nw, s->d_tmp,
                l.wq, l.wk, l.wv1, l.wv2,
                eng.h, eng.qd, eng.kd);
            HIP_CHECK(hipGetLastError());
        }'''
c = c.replace(old_attn, new_attn)
print(f'10: fused QKV ({c.count(new_attn)} occurrences)')

# Replace paged KV with linear KV in forward/greedy
old_kv_block = '''        // Paged KV: allocate page, write, gather for attention
        {
            size_t kv_off = zaya_kv_page_write(s, il, s->pos);
            __half* layer_k = s->d_kcache + (size_t)il * s->kv_pool_pages * s->page_size * eng.nkv * eng.hd;
            __half* layer_v = s->d_vcache + (size_t)il * s->kv_pool_pages * s->page_size * eng.nkv * eng.hd;
            copy_k<<<(eng.kd+BLK-1)/BLK,BLK,0,s->st>>>(layer_k + kv_off, s->d_kout, eng.kd);
            HIP_CHECK(hipGetLastError());
            copy_k<<<(eng.kd+BLK-1)/BLK,BLK,0,s->st>>>(layer_v + kv_off, s->d_vout, eng.kd);
            HIP_CHECK(hipGetLastError());
            if (s->pos > 0) {
                zaya_kv_gather(s, il, s->pos + 1);
            } else {
                copy_k<<<(eng.kd+BLK-1)/BLK,BLK,0,s->st>>>(s->d_k_gather, s->d_kout, eng.kd);
                HIP_CHECK(hipGetLastError());
                copy_k<<<(eng.kd+BLK-1)/BLK,BLK,0,s->st>>>(s->d_v_gather, s->d_vout, eng.kd);
                HIP_CHECK(hipGetLastError());
            }
            rcpp_kv_cache_attn_decode_fd(s->d_qout, s->d_k_gather, s->d_v_gather,
                s->d_ao, eng.nq, eng.nkv, eng.hd, s->pos+1, 1.0f/sqrtf((float)eng.hd), (void*)s->st);
        }'''

new_kv_block = '''        // KV cache: linear contiguous write (#3) -- no gather overhead
        {
            __half* layer_k = s->d_kcache + ((size_t)il * s->max_seq + s->pos) * eng.nkv * eng.hd;
            __half* layer_v = s->d_vcache + ((size_t)il * s->max_seq + s->pos) * eng.nkv * eng.hd;
            copy_k<<<(eng.kd+BLK-1)/BLK,BLK,0,s->st>>>(layer_k, s->d_kout, eng.kd);
            HIP_CHECK(hipGetLastError());
            copy_k<<<(eng.kd+BLK-1)/BLK,BLK,0,s->st>>>(layer_v, s->d_vout, eng.kd);
            HIP_CHECK(hipGetLastError());
            rcpp_kv_cache_attn_decode_fd(s->d_qout,
                s->d_kcache + (size_t)il * s->max_seq * eng.nkv * eng.hd,
                s->d_vcache + (size_t)il * s->max_seq * eng.nkv * eng.hd,
                s->d_ao, eng.nq, eng.nkv, eng.hd, s->pos+1, 1.0f/sqrtf((float)eng.hd), (void*)s->st);
        }'''
c = c.replace(old_kv_block, new_kv_block)
print(f'11: linear KV block ({c.count(new_kv_block)} occurrences)')

# Replace MoE sync with always-run (#1) for zaya_forward (HIP_OK_V)
old_moe_fwd = '''            fixup_skip_expert_kernel<<<1,256,0,s->st>>>(s->d_expert_idx,s->d_tmp,s->d_skip_flag,eng.n_exp,eng.n_exp_t,eng.h);
            HIP_CHECK(hipGetLastError());
            HIP_OK_V(hipStreamSynchronize(s->st));
            int was_skip; HIP_OK_V(hipMemcpy(&was_skip,s->d_skip_flag,4,hipMemcpyDeviceToHost));
            if(!was_skip){
                const int gb=(2*eng.n_ff+WMMA_M-1)/WMMA_M;
                const int db=(eng.h+WMMA_M-1)/WMMA_M;
                const int sb=(eng.n_ff+BLK-1)/BLK;
                wmma_gateup_kernel<<<gb,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_hs,l.gu,s->d_expert_idx);
                HIP_CHECK(hipGetLastError());
                silu_mul_k<<<sb,BLK,0,s->st>>>(s->d_ao,s->d_tmp,s->d_tmp+eng.n_ff,eng.n_ff);
                HIP_CHECK(hipGetLastError());
                wmma_down_kernel<<<db,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_ao,l.dn,s->d_expert_idx);
            HIP_CHECK(hipGetLastError());
            }
            residual_scale_k<<<g1,BLK,0,s->st>>>(s->d_tmp,s->d_hs,l.pmhss,l.pmhsb,l.pmrss,l.pmrsb,eng.h);
            HIP_CHECK(hipGetLastError());
            copy_k<<<g1,BLK,0,s->st>>>(s->d_hs,s->d_tmp,eng.h);
        HIP_CHECK(hipGetLastError());
        }else{
            copy_k<<<g1,BLK,0,s->st>>>(s->d_tmp,s->d_hs,eng.h);
        HIP_CHECK(hipGetLastError());'''

new_moe_fwd = '''            fixup_skip_expert_kernel<<<1,256,0,s->st>>>(s->d_expert_idx,s->d_skip_flag,eng.n_exp,eng.n_exp_t);
            HIP_CHECK(hipGetLastError());
            // #1: No host sync - skip_flag read by gateup/down on GPU
            const int gb=(2*eng.n_ff+WMMA_M-1)/WMMA_M;
            const int db=(eng.h+WMMA_M-1)/WMMA_M;
            const int sb=(eng.n_ff+BLK-1)/BLK;
            wmma_gateup_kernel<<<gb,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_hs,l.gu,s->d_expert_idx,s->d_skip_flag);
            HIP_CHECK(hipGetLastError());
            silu_mul_k<<<sb,BLK,0,s->st>>>(s->d_ao,s->d_tmp,s->d_tmp+eng.n_ff,eng.n_ff);
            HIP_CHECK(hipGetLastError());
            wmma_down_kernel<<<db,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_ao,l.dn,s->d_expert_idx,s->d_skip_flag);
            HIP_CHECK(hipGetLastError());
            residual_scale_k<<<g1,BLK,0,s->st>>>(s->d_tmp,s->d_hs,l.pmhss,l.pmhsb,l.pmrss,l.pmrsb,eng.h);
            HIP_CHECK(hipGetLastError());
            copy_k<<<g1,BLK,0,s->st>>>(s->d_hs,s->d_tmp,eng.h);
        HIP_CHECK(hipGetLastError());
        }else{
            copy_k<<<g1,BLK,0,s->st>>>(s->d_tmp,s->d_hs,eng.h);
        HIP_CHECK(hipGetLastError());'''
c = c.replace(old_moe_fwd, new_moe_fwd)
print(f'12: MoE sync forward ({c.count(new_moe_fwd)} occurrence)')

# Replace MoE sync with always-run (#1) for zaya_forward_greedy (HIP_OK_R)
old_moe_greedy = '''            fixup_skip_expert_kernel<<<1,256,0,s->st>>>(s->d_expert_idx,s->d_tmp,s->d_skip_flag,eng.n_exp,eng.n_exp_t,eng.h);
            HIP_CHECK(hipGetLastError());
            HIP_OK_R(hipStreamSynchronize(s->st), -1);
            int was_skip; HIP_OK_R(hipMemcpy(&was_skip,s->d_skip_flag,4,hipMemcpyDeviceToHost), -1);
            if(!was_skip){
                const int gb=(2*eng.n_ff+WMMA_M-1)/WMMA_M;
                const int db=(eng.h+WMMA_M-1)/WMMA_M;
                const int sb=(eng.n_ff+BLK-1)/BLK;
                wmma_gateup_kernel<<<gb,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_hs,l.gu,s->d_expert_idx);
                HIP_CHECK(hipGetLastError());
                silu_mul_k<<<sb,BLK,0,s->st>>>(s->d_ao,s->d_tmp,s->d_tmp+eng.n_ff,eng.n_ff);
                HIP_CHECK(hipGetLastError());
                wmma_down_kernel<<<db,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_ao,l.dn,s->d_expert_idx);
            HIP_CHECK(hipGetLastError());
            }
            residual_scale_k<<<g1,BLK,0,s->st>>>(s->d_tmp,s->d_hs,l.pmhss,l.pmhsb,l.pmrss,l.pmrsb,eng.h);
            HIP_CHECK(hipGetLastError());
            copy_k<<<g1,BLK,0,s->st>>>(s->d_hs,s->d_tmp,eng.h);
        HIP_CHECK(hipGetLastError());
        }else{copy_k<<<g1,BLK,0,s->st>>>(s->d_tmp,s->d_hs,eng.h);}
    HIP_CHECK(hipGetLastError());'''

new_moe_greedy = '''            fixup_skip_expert_kernel<<<1,256,0,s->st>>>(s->d_expert_idx,s->d_skip_flag,eng.n_exp,eng.n_exp_t);
            HIP_CHECK(hipGetLastError());
            // #1: No host sync - skip_flag read by gateup/down on GPU
            const int gb=(2*eng.n_ff+WMMA_M-1)/WMMA_M;
            const int db=(eng.h+WMMA_M-1)/WMMA_M;
            const int sb=(eng.n_ff+BLK-1)/BLK;
            wmma_gateup_kernel<<<gb,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_hs,l.gu,s->d_expert_idx,s->d_skip_flag);
            HIP_CHECK(hipGetLastError());
            silu_mul_k<<<sb,BLK,0,s->st>>>(s->d_ao,s->d_tmp,s->d_tmp+eng.n_ff,eng.n_ff);
            HIP_CHECK(hipGetLastError());
            wmma_down_kernel<<<db,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_ao,l.dn,s->d_expert_idx,s->d_skip_flag);
            HIP_CHECK(hipGetLastError());
            residual_scale_k<<<g1,BLK,0,s->st>>>(s->d_tmp,s->d_hs,l.pmhss,l.pmhsb,l.pmrss,l.pmrsb,eng.h);
            HIP_CHECK(hipGetLastError());
            copy_k<<<g1,BLK,0,s->st>>>(s->d_hs,s->d_tmp,eng.h);
        HIP_CHECK(hipGetLastError());
        }else{copy_k<<<g1,BLK,0,s->st>>>(s->d_tmp,s->d_hs,eng.h);}
    HIP_CHECK(hipGetLastError());'''
c = c.replace(old_moe_greedy, new_moe_greedy)
print(f'13: MoE sync greedy ({c.count(new_moe_greedy)} occurrence)')

# Replace embedding in both forward and greedy (#5)
old_embed = '    std::vector<__half> hh(eng.h);\n    for(int i=0;i<eng.h;i++){float raw=s->embed[token_id*(size_t)eng.h+i];hh[i]=__float2half((raw+s->ibias[i])*s->iscale[i]);}\n    HIP_OK_V(hipMemcpyAsync(s->d_hs,hh.data(),eng.h*2,hipMemcpyHostToDevice,s->st));'
new_embed = '    // Device-side embedding lookup (#5): no H2D copy\n    embed_lookup_k<<<g1,BLK,0,s->st>>>(s->d_hs, s->d_embed, s->d_ibias, s->d_iscale, token_id, eng.h);\n    HIP_CHECK(hipGetLastError());'
c = c.replace(old_embed, new_embed)
# Also the HIP_OK_R variant in greedy
old_embed2 = '    std::vector<__half> hh(eng.h);\n    for(int i=0;i<eng.h;i++){float raw=s->embed[token_id*(size_t)eng.h+i];hh[i]=__float2half((raw+s->ibias[i])*s->iscale[i]);}\n    HIP_OK_R(hipMemcpyAsync(s->d_hs,hh.data(),eng.h*2,hipMemcpyHostToDevice,s->st), -1);'
new_embed2 = '    // Device-side embedding lookup (#5): no H2D copy\n    embed_lookup_k<<<g1,BLK,0,s->st>>>(s->d_hs, s->d_embed, s->d_ibias, s->d_iscale, token_id, eng.h);\n    HIP_CHECK(hipGetLastError());'
c = c.replace(old_embed2, new_embed2)
print('14: embedding replaced')

# --- #3: Update zaya_reset for linear KV ---
old_reset = '''    size_t kv_pool_bytes = (size_t)eng.n_layers * s->kv_pool_pages * s->page_size * eng.nkv * eng.hd * 2;
    HIP_OK_V(hipMemsetAsync(s->d_kcache, 0, kv_pool_bytes, s->st));
    HIP_OK_V(hipMemsetAsync(s->d_vcache, 0, kv_pool_bytes, s->st));
    // Reset page table
    for (int il = 0; il < eng.n_layers; il++) {
        s->page_alloc[il].assign(s->n_kv_pages, false);
        s->page_map[il].assign(s->n_kv_pages, -1);
        s->page_next_evict[il] = 0;
    }'''
new_reset = '''    if (s->use_linear_kv) {
        size_t kv_bytes = (size_t)eng.n_layers * s->max_seq * eng.nkv * eng.hd * 2;
        HIP_OK_V(hipMemsetAsync(s->d_kcache, 0, kv_bytes, s->st));
        HIP_OK_V(hipMemsetAsync(s->d_vcache, 0, kv_bytes, s->st));
    } else {
        size_t kv_pool_bytes = (size_t)eng.n_layers * s->kv_pool_pages * s->page_size * eng.nkv * eng.hd * 2;
        HIP_OK_V(hipMemsetAsync(s->d_kcache, 0, kv_pool_bytes, s->st));
        HIP_OK_V(hipMemsetAsync(s->d_vcache, 0, kv_pool_bytes, s->st));
        for (int il = 0; il < eng.n_layers; il++) {
            s->page_alloc[il].assign(s->n_kv_pages, false);
            s->page_map[il].assign(s->n_kv_pages, -1);
            s->page_next_evict[il] = 0;
        }
    }'''
c = c.replace(old_reset, new_reset)
print('15: reset updated')

# --- #5: Update zaya_destroy for d_ibias/d_iscale ---
c = c.replace('safe(s->d_hs); safe(s->d_ao); safe(s->d_tmp); safe(s->d_fnw);\n    safe(s->d_lm_out); safe(s->d_embed); safe(s->d_conv); safe(s->d_phs);',
              'safe(s->d_hs); safe(s->d_ao); safe(s->d_tmp); safe(s->d_fnw);\n    safe(s->d_lm_out); safe(s->d_embed); safe(s->d_ibias); safe(s->d_iscale);\n    safe(s->d_conv); safe(s->d_phs);')
print('16: destroy updated')

# Write back
with open(f'{base}/src/zaya_engine.cpp','w') as f:
    f.write(c)
print(f'\nAll changes applied ({len(c)} bytes written)')

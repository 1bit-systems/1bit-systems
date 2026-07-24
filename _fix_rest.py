c = open('/home/bcloud/1bit-systems/src/zaya_engine.cpp').read()
ocs = len(c)

# 1. Linear KV block - read the actual text from file
# Find "Paged KV: allocate page" block
old_start = c.find('// Paged KV: allocate page, write, gather for attention')
assert old_start > 0, 'KV block not found'
old_end = c.find('        }\n        moe_tiled_gemv', old_start)
old_end += len('        }\n')
old_kv = c[old_start:old_end]

new_kv = '''        // KV cache: linear contiguous write (#3) -- no gather overhead
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

c = c.replace(old_kv, new_kv)
print(f'1: KV linear ({c.count(new_kv)} occurrences)')

# 2. MoE sync forward - read actual text
# Find fixup_skip_expert_kernel call in forward path (HIP_OK_V variant)
old_start2 = c.find('            fixup_skip_expert_kernel<<<1,256,0,s->st>>>')
assert old_start2 > 0, 'MoE forward not found'
old_end2 = c.find('        }else{\n            copy_k', old_start2)
old_end2 = c.find('\n        HIP_CHECK(hipGetLastError());\n        }', old_end2)
old_end2 += len('\n        HIP_CHECK(hipGetLastError());\n        }')
old_moe_fwd = c[old_start2:old_end2]

new_moe_fwd = '''            fixup_skip_expert_kernel<<<1,256,0,s->st>>>(s->d_expert_idx,s->d_skip_flag,eng.n_exp,eng.n_exp_t);
            HIP_CHECK(hipGetLastError());
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
print(f'2: MoE forward ({c.count(new_moe_fwd)} occurrence)')

# 3. MoE sync greedy - read actual text
old_start3 = c.find('            fixup_skip_expert_kernel<<<1,256,0,s->st>>>', old_start2 + len(new_moe_fwd))
assert old_start3 > 0, 'MoE greedy not found'
old_end3 = c.find('        }else{copy_k', old_start3)
old_end3 = c.find('\n    HIP_CHECK(hipGetLastError());\n    }', old_end3)
old_end3 += len('\n    HIP_CHECK(hipGetLastError());\n    }')
old_moe_greedy = c[old_start3:old_end3]

new_moe_greedy = '''            fixup_skip_expert_kernel<<<1,256,0,s->st>>>(s->d_expert_idx,s->d_skip_flag,eng.n_exp,eng.n_exp_t);
            HIP_CHECK(hipGetLastError());
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
print(f'3: MoE greedy ({c.count(new_moe_greedy)} occurrence)')

open('/home/bcloud/1bit-systems/src/zaya_engine.cpp','w').write(c)
print(f'Done ({len(c)} bytes)')

#!/usr/bin/env python3
"""GGUF → 1BP converter with Laguna architecture support.
Uses the gguf Python package for tensor reading + numpy for quant.

Supports:
  - Dense transformers (Qwen3, Llama, Phi, etc.)  -> ONEBP_DENSE
  - Generic MoE (Zaya1, DeepSeek, etc.)           -> ONEBP_MOE
  - Poolside Laguna (sigmoid-routed MoE, hybrid    -> ONEBP_LAGUNA
    SWA/global attention, softplus output gate)
  - Ternary models (Bonsai, BitNet)                -> ONEBP_TERNARY (TQ2)
"""
import struct, sys, os, numpy as np
from gguf import GGUFReader, dequantize

# ─── Helpers ────────────────────────────────────────────────────────────────

def f32b(v):
    """float32 -> bf16 with round-to-nearest-even."""
    x = int(np.float32(v).view(np.uint32))
    return np.uint16((x + 0x7FFF + ((x >> 16) & 1)) >> 16)

def quant_tile(data, tr=32, tc=256, gs=32):
    """Q4NX tile quantization: asymmetric 4-bit, bf16 scales/zero_points per group of 32."""
    r, c = data.shape
    pr, pc = tr, tc
    grps = pc // gs
    padded = np.zeros((pr, pc), dtype=np.float32)
    padded[:r, :c] = data

    # Compute min/max on original data only — zero padding for partial
    # edge tiles must not pull min toward zero (issue #639).
    mn = np.zeros((pr, grps), dtype=np.float32)
    mx = np.zeros((pr, grps), dtype=np.float32)
    n_full_grps = c // gs
    if n_full_grps > 0:
        full = data[:, :n_full_grps * gs].reshape(r, n_full_grps, gs)
        mn[:r, :n_full_grps] = full.min(axis=2)
        mx[:r, :n_full_grps] = full.max(axis=2)
    rem = c % gs
    if rem > 0:
        part = data[:, n_full_grps * gs:]
        mn[:r, n_full_grps] = part.min(axis=1)
        mx[:r, n_full_grps] = part.max(axis=1)
    # padded rows (r..pr) stay at 0.0 min/max

    grouped = padded.reshape(pr, grps, gs)
    rng = mx - mn
    flat_range = rng < 1e-10
    scale = np.where(flat_range, 0.0, rng / 15.0)
    zp_mn = np.where(flat_range, 0.0, mn)
    flat_scale = scale < 1e-10
    scale = np.where(flat_scale, 1.0, scale).astype(np.float32)
    zp_mn = np.where(flat_scale, 0.0, zp_mn).astype(np.float32)

    sc = f32b(scale).astype(np.uint16)
    zp = f32b(zp_mn).astype(np.uint16)
    inv = 1.0 / scale
    qi = np.clip(np.round((grouped - zp_mn[:, :, None]) * inv[:, :, None]), 0, 15).astype(np.uint8)
    qi_flat = qi.reshape(pr, pc)
    pk = (qi_flat[:, 1::2] << 4) | qi_flat[:, 0::2]
    return sc.tobytes() + zp.tobytes() + pk.tobytes()

def quant_tile_tq2(data, tr=32, tc=256, gs=32):
    """TQ2 symmetric ternary: values round to -scale, 0, +scale, packed 2-bit."""
    r, c = data.shape
    pr, pc = tr, tc
    grps = pc // gs
    padded = np.zeros((pr, pc), dtype=np.float32)
    padded[:r, :c] = data
    
    # Compute max abs on original data only (same #639 fix for TQ2).
    mx = np.zeros((pr, grps), dtype=np.float32)
    n_full_grps = c // gs
    if n_full_grps > 0:
        full = data[:, :n_full_grps * gs].reshape(r, n_full_grps, gs)
        mx[:r, :n_full_grps] = np.abs(full).max(axis=2)
    rem = c % gs
    if rem > 0:
        part = data[:, n_full_grps * gs:]
        mx[:r, n_full_grps] = np.abs(part).max(axis=1)

    grouped = padded.reshape(pr, grps, gs)
    scale = np.where(mx < 1e-10, 1.0, mx).astype(np.float32)
    sc = f32b(scale).astype(np.uint16)
    inv = 1.0 / scale
    signed = np.clip(np.round(grouped * inv[:, :, None]), -1, 1).astype(np.int8)
    code = (signed + 1).astype(np.uint8).reshape(pr, pc)
    c0, c1, c2, c3 = code[:, 0::4], code[:, 1::4], code[:, 2::4], code[:, 3::4]
    pk = (c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)).astype(np.uint8)
    return sc.tobytes() + pk.tobytes()

def tiled_size(rows, cols, tr=32, tc=256, gs=32):
    ntr = (rows + tr - 1) // tr
    ntc = (cols + tc - 1) // tc
    return ntr * ntc * (tr * (tc // gs) * 4 + tr * tc // 2)

def tiled_size_tq2(rows, cols, tr=32, tc=256, gs=32):
    ntr = (rows + tr - 1) // tr
    ntc = (cols + tc - 1) // tc
    groups_per_row = tc // gs
    return ntr * ntc * (tr * groups_per_row * 2 + tr * tc // 4)

def to_f32(ten):
    """Dequantize a GGUF tensor to flat float32 array."""
    if ten.tensor_type <= 1:
        dt = np.float32 if ten.tensor_type == 0 else np.float16
        return np.frombuffer(ten.data, dtype=dt).astype(np.float32)
    return dequantize(ten.data, ten.tensor_type).astype(np.float32).reshape(-1)

# ─── GGUF metadata readers ─────────────────────────────────────────────────

def _gf_val(arr):
    """Extract a scalar value from a numpy memmap or array."""
    if hasattr(arr, 'item'):
        return arr.item()
    if hasattr(arr, '__len__'):
        return arr[0] if len(arr) > 0 else 0
    return arr

def _gf(rd, fields, alt=None):
    """Read a uint32 GGUF metadata field with fallback names."""
    for fn in [fields] if isinstance(fields, str) else fields:
        if not fn:
            continue
        v = rd.fields.get(fn)
        if v is None or len(v.parts) < 4:
            continue
        try:
            return int(_gf_val(v.parts[3]))
        except Exception:
            pass
    return 0

def _gf_str(rd, field):
    """Read a string GGUF metadata field."""
    v = rd.fields.get(field)
    if v is None or len(v.parts) < 4:
        return ''
    # parts[0] = key length, parts[1] = key bytes
    # For string: parts[2] = value type (8), parts[3] = string length, parts[4] = string data
    for pi in range(2, len(v.parts)):
        try:
            raw = v.parts[pi]
            if hasattr(raw, 'tobytes'):
                s = bytes(raw.tobytes()).decode('utf-8', errors='replace')
                if s.isprintable() and len(s) > 0:
                    return s
            elif hasattr(raw, 'item'):
                continue  # scalar, not string
        except:
            pass
    # Fallback: try parts[4] directly (common for string-typed keys)
    if len(v.parts) > 4:
        raw = v.parts[4]
        if hasattr(raw, 'tobytes'):
            try:
                return bytes(raw.tobytes()).decode('utf-8', errors='replace')
            except:
                pass
    return ''

def _gf_f32(rd, field):
    """Read a float32 GGUF metadata field."""
    v = rd.fields.get(field)
    if v is None or len(v.parts) < 4:
        return None
    try:
        return float(_gf_val(v.parts[3]))
    except Exception:
        return None

def _get_arch_fields(arch):
    """Return list of arch-qualified field prefixes to try."""
    base = f"{arch}"
    return [base, "llm"]  # Try architecture-specific first, then generic 'llm'

def _read_metadata(rd):
    """Read all model metadata from GGUF. Returns dict."""
    arch = _gf_str(rd, "general.architecture") or "unknown"
    prefixes = _get_arch_fields(arch)

    def r(field, alt_field=None):
        """Read uint32 trying all prefixes."""
        for p in prefixes:
            v = _gf(rd, f"{p}.{field}")
            if v:
                return v
        # Fall back to unqualified names
        if alt_field:
            return _gf(rd, alt_field)
        return _gf(rd, field)

    def r_str(field):
        for p in prefixes:
            v = _gf_str(rd, f"{p}.{field}")
            if v:
                return v
        return ""

    def r_f32(field):
        for p in prefixes:
            v = _gf_f32(rd, f"{p}.{field}")
            if v is not None:
                return v
        return 0.0

    def r_direct(key):
        """Read a uint32 field by its exact GGUF key name."""
        return _gf(rd, key)

    def r_f32_direct(key):
        """Read a float32 field by its exact GGUF key name."""
        return _gf_f32(rd, key)

    def r_array_max(field):
        """Read an array field and return the max value (for per-layer arrays like head_count).
        GGUF array format: parts[2]=type(9=array), parts[3]=elem_type, parts[4]=count, parts[5:]=values"""
        for p in prefixes:
            k = f"{p}.{field}"
            v = rd.fields.get(k)
            if v is not None and hasattr(v, 'parts') and len(v.parts) >= 6:
                try:
                    # parts[2] = value type marker
                    vt = int(_gf_val(v.parts[2]))
                    if vt != 9:  # not an array
                        continue
                    # parts[3] = element type
                    # parts[4] = element count
                    count = int(_gf_val(v.parts[4]))
                    # parts[5:] = actual values
                    vals = []
                    for i in range(count):
                        if 5 + i < len(v.parts):
                            vals.append(int(_gf_val(v.parts[5 + i])))
                    if vals:
                        return max(vals)
                except Exception as e:
                    print(f"  DEBUG r_array_max({field}): {e}")
                    pass
        return 0

    def r_f32_p(field):
        """Read f32 by trying both qualified and exact keys."""
        # First try with prefix
        for p in prefixes:
            v = _gf_f32(rd, f"{p}.{field}")
            if v is not None:
                return v
        # Then try exact key
        v = r_f32_direct(field)
        if v is not None:
            return v
        return 0.0

    # Read head_count as array (per-layer), take max
    n_heads = r_array_max("attention.head_count")
    if not n_heads:
        n_heads = r("attention.head_count", "num_attention_heads")

    meta = {
        'arch': arch,
        'hidden_size': r("embedding_length", "hidden_size"),
        'num_layers': r("block_count", "num_hidden_layers"),
        'num_attention_heads': n_heads,
        'num_kv_heads': r("attention.head_count_kv", "num_key_value_heads"),
        'head_dim': r("attention.key_length", "head_dim"),
        'intermediate_size': r("feed_forward_length", "intermediate_size"),
        'vocab_size': r("vocab_size"),
        'max_seq_len': r("context_length", "max_seq_len"),
        'rope_theta': r_f32_p("rope.freq_base"),
        'rms_norm_eps': r_f32("attention.layer_norm_rms_epsilon"),
        'has_q_norm': 1 if r("attention.q_norm_has_rms_norm", "has_q_norm") else 0,
        'has_k_norm': 1 if r("attention.k_norm_has_rms_norm", "has_k_norm") else 0,
        'bos_token_id': r_direct("tokenizer.ggml.bos_token_id") or r("bos_token_id"),
        'eos_token_id': r_direct("tokenizer.ggml.eos_token_id") or r("eos_token_id"),
    }

    # Laguna-specific - use direct key names when prefix fails.
    # Note: GGUF keys can use dots for namespacing (attention.sliding_window)
    # OR flat underscores (expert_feed_forward_length). Try both.
    meta['num_experts'] = r("expert_count", "num_experts")
    meta['n_expert_used'] = r("expert_used_count", "num_experts_used")
    meta['n_ff_exp'] = r("expert_feed_forward_length") or r("expert.feed_forward_length")
    meta['n_ff_shexp'] = r("expert_shared_feed_forward_length") or r("expert.shared_feed_forward_length")
    meta['n_layer_dense_lead'] = r("leading_dense_block_count", "n_layer_dense_lead")
    meta['sliding_window'] = r("attention.sliding_window", "sliding_window")
    meta['swa_period'] = r("attention.sliding_window_pattern", "swa_period")
    if meta['swa_period'] == 0 and meta['sliding_window'] > 0:
        meta['swa_period'] = 4  # default for Laguna: 1 FULL : 3 SWA pattern
    meta['expert_weights_norm'] = r("expert_weights_norm") or r("expert.weights_norm")
    meta['expert_weights_scale'] = _gf_f32(rd, "laguna.expert_weights_scale")
    meta['rope_freq_base_swa'] = r_f32_p("rope.freq_base_swa")
    meta['n_rot_swa'] = r("rope.dimension_count_swa")
    meta['n_rot_full'] = r("rope.dimension_count")

    # Expert gating function - read as uint32 value
    gate_val = r_direct(f"{arch}.expert_gating_func")
    if gate_val == 2:
        meta['expert_gating_func'] = 0  # SIGMOID
    elif gate_val == 1:
        meta['expert_gating_func'] = 1  # SOFTMAX
    else:
        gate_str = r_str("expert.gating_func")
        if gate_str in ("sigmoid", "SIGMOID"):
            meta['expert_gating_func'] = 0
        elif gate_str in ("softmax", "SOFTMAX"):
            meta['expert_gating_func'] = 1
        else:
            meta['expert_gating_func'] = 0  # default sigmoid

    # Attention gate type - detect from gate tensor shapes
    # Laguna S 2.1: per-head (gate_out == layer's n_heads)
    # Laguna M.1: per-element (gate_out == n_heads * head_dim)
    gate_type_str = r_str("attention.gating_type")
    if gate_type_str in ("per_element", "per-element"):
        meta['attn_gate_type'] = 1
    elif gate_type_str in ("per_head", "per-head"):
        meta['attn_gate_type'] = 0
    else:
        # Auto-detect from gate tensor shapes
        # Get per-layer head counts
        head_counts = []
        v = rd.fields.get(f"{arch}.attention.head_count")
        if v and len(v.parts) >= 6:
            try:
                count = int(_gf_val(v.parts[4]))
                for i in range(count):
                    if 5 + i < len(v.parts):
                        head_counts.append(int(_gf_val(v.parts[5 + i])))
            except:
                pass
        
        if head_counts:
            # Check a SWA layer (index 1, which has 72 heads) for the gate
            for tn in rd.tensors:
                if "attn_gate" in tn.name and len(tn.shape) >= 2:
                    # GGUF shape is [rows, cols] = [hidden, gate_out]
                    # gate_out is the second dimension (shape[1])
                    gate_out = int(tn.shape[1])  # number of heads (per-head) or heads*dim (per-element)
                    parts = tn.name.split('.')
                    layer_idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                    n_head_il = head_counts[layer_idx] if layer_idx < len(head_counts) else 0
                    if gate_out == n_head_il:
                        meta['attn_gate_type'] = 0  # per-head (gate dim = num heads)
                    elif gate_out == n_head_il * meta['head_dim']:
                        meta['attn_gate_type'] = 1  # per-element (gate dim = heads*head_dim)
                    else:
                        meta['attn_gate_type'] = 0  # default per-head
                    break
        else:
            meta['attn_gate_type'] = 0

    # Derive missing values
    if not meta['num_kv_heads']:
        meta['num_kv_heads'] = meta['num_attention_heads']
    if not meta['head_dim'] and meta['num_attention_heads']:
        meta['head_dim'] = meta['hidden_size'] // meta['num_attention_heads']
    if not meta['max_seq_len']:
        meta['max_seq_len'] = 4096
    if not meta['rope_theta']:
        meta['rope_theta'] = 10000.0
    if not meta['rms_norm_eps']:
        meta['rms_norm_eps'] = 1e-6
    if meta['n_rot_swa'] == 0:
        # If SWA RoPE dim not specified, use head_dim (default for SWA layers)
        meta['n_rot_swa'] = meta['head_dim']

    return meta

# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    argv = sys.argv[1:]
    tq2 = '--tq2' in argv
    if tq2:
        argv.remove('--tq2')
    if len(argv) < 2:
        print(f"Usage: {sys.argv[0]} input.gguf output.1bp [max_tensors] [--tq2]")
        print("  --tq2: symmetric 2-bit ternary quant instead of 4-bit Q4NX.")
        sys.exit(1)

    print(f"Reading {argv[0]}...")
    rd = GGUFReader(argv[0])
    max_t = int(argv[2]) if len(argv) > 2 else 0
    by_name = {t.name: t for t in rd.tensors}

    meta = _read_metadata(rd)

    H = meta['hidden_size']
    L = meta['num_layers']
    NH = meta['num_attention_heads']
    NKV = meta['num_kv_heads']
    HD = meta['head_dim']
    IM = meta['intermediate_size']
    V = meta['vocab_size']
    arch = meta['arch']

    # Determine arch enum
    is_laguna = (arch == "laguna")
    is_moe = (meta['num_experts'] > 0)
    is_ternary = any(t.tensor_type == 42 for t in rd.tensors)  # BST dtype
    
    if is_laguna:
        arch_enum = 6  # ONEBP_LAGUNA
    elif arch in ("falcon", "falcon3", "falcon_h1"):
        arch_enum = 0  # ONEBP_DENSE (dense transformer)
        print(f"  Falcon arch detected — using dense forward pass")
    elif arch in ("olmo", "olmo2", "olmoe"):
        arch_enum = 0  # ONEBP_DENSE
        if "olmoe" in arch:
            arch_enum = 1  # ONEBP_MOE (MoE version)
        print(f"  OLMo arch detected — using {'MoE' if arch_enum else 'dense'} forward pass")
    elif is_ternary:
        arch_enum = 4  # ONEBP_TERNARY
    elif is_moe:
        arch_enum = 1  # ONEBP_MOE
    else:
        arch_enum = 0  # ONEBP_DENSE

    print(f"Model: {arch} arch_enum={arch_enum} "
          f"H={H} L={L} NH={NH} NKV={NKV} HD={HD} IM={IM} V={V}")
    if is_laguna:
        print(f"  Experts: {meta['num_experts']} routed, top-{meta['n_expert_used']}, "
              f"shared_ff={meta['n_ff_shexp']}, dense_lead={meta['n_layer_dense_lead']}")
        print(f"  SWA: window={meta['sliding_window']}, period={meta['swa_period']}")
        print(f"  Attn gate: {'per-element' if meta['attn_gate_type'] else 'per-head'}")
        print(f"  RoPE: base={meta['rope_theta']}, swa_base={meta['rope_freq_base_swa']}, "
              f"swa_rot={meta['n_rot_swa']}")

    if not H or not L or not V:
        # Fallback: get vocab size from embedding tensor
        emb = by_name.get("token_embd.weight")
        if emb is not None and len(emb.shape) == 2:
            V = int(emb.shape[1])
            meta['vocab_size'] = V
            print(f"  vocab_size from tensor shape: {V}")
        if not H or not L or not V:
            print("ERROR: could not read model config")
            sys.exit(1)

    # Build header
    tr, tc, gs = 32, 256, 32
    quant_id = 3 if tq2 else 0  # ONEBP_TQ2 : ONEBP_Q4NX
    scale_type = 0              # ONEBP_SCALE_BF16

    rope_theta_f = int(meta['rope_theta'] * 1000)
    rope_swa_f = int(meta['rope_freq_base_swa'] * 1000) if meta['rope_freq_base_swa'] else 0
    exp_scale_f = int(meta.get('expert_weights_scale', 1.0) * 1000)

    # Pack the common header fields + Laguna-specific extensions
    # Standard header fields (5 uint32 + 8 int32 + 10 uint32 = 23 * 4 = 92 bytes)
    hdr_parts = [
        0x00504231, 1, arch_enum, quant_id, scale_type,        # magic, version, arch, quant, scale
        H, L, NH, NKV, HD, IM, V, meta.get('max_seq_len', 4096),  # model dims
        tr, tc, gs,                                            # tile config
        meta['has_q_norm'], meta['has_k_norm'], 0,             # has_q_norm, has_k_norm, has_bias
        rope_theta_f,                                          # rope_theta * 1000
        meta['bos_token_id'], meta['eos_token_id'],             # token ids
        0,                                                     # tensor_count (filled later)
    ]

    hdr = struct.pack('<5I8i10I', *hdr_parts)
    hdr = bytearray(hdr.ljust(92, b'\x00'))

    # Laguna-specific fields start at offset 92 (right after the standard 23 fields)
    laguna_fields = [
        meta.get('num_experts', 0),
        meta.get('n_expert_used', 0),
        meta.get('n_ff_exp', 0),
        meta.get('n_ff_shexp', 0),
        meta.get('n_layer_dense_lead', 0),
        meta.get('sliding_window', 0),
        meta.get('swa_period', 0),
        meta.get('expert_gating_func', 0),
        meta.get('expert_weights_norm', 0),
        exp_scale_f,
        meta.get('attn_gate_type', 0),
        rope_swa_f,
        meta.get('n_rot_swa', 0),
        meta.get('n_rot_full', HD if HD else 64),
    ]

    # Pad Laguna fields to 56 bytes (14 uint32), then 44 reserved, then 64 model_tag
    hdr += struct.pack(f'<{len(laguna_fields)}I', *laguna_fields)
    hdr += b'\x00' * (256 - len(hdr))
    hdr = bytearray(hdr[:256])

    # Model tag for identification
    tag = f"gguf:{arch}"
    hdr[192:192+len(tag)] = tag.encode()

    print(f"Quant: {'TQ2 (2-bit symmetric ternary)' if tq2 else 'Q4NX (4-bit)'}")
    print(f"Arch enum: {arch_enum} ({arch}), header size: {len(hdr)} bytes")

    # Collect tensors
    tlist = []  # (name, ndim, dims:list[int], file_off, byte_size)
    total = 0
    n_skipped_other = 0

    for tn in rd.tensors:
        shape = tn.shape
        if len(shape) == 1:
            length = int(shape[0])
            sz = length * 4  # raw f32
            tlist.append((tn.name, 1, [length], total, sz))
            total += sz
        elif len(shape) == 2:
            # GGUF shape is [cols, rows] — we store as [rows, cols]
            cols, rows = int(shape[0]), int(shape[1])
            sz = (tiled_size_tq2 if tq2 else tiled_size)(rows, cols, tr, tc, gs)
            tlist.append((tn.name, 2, [rows, cols], total, sz))
            total += sz
        elif len(shape) == 3:
            # MoE experts: GGUF shape is [cols, rows, n_experts]
            cols, rows, n_experts = int(shape[0]), int(shape[1]), int(shape[2])
            per_expert = (tiled_size_tq2 if tq2 else tiled_size)(rows, cols, tr, tc, gs)
            sz = per_expert * n_experts
            tlist.append((tn.name, 3, [n_experts, rows, cols], total, sz))
            total += sz
        else:
            print(f"  SKIP unsupported ndim={len(shape)}: {tn.name} {shape}")
            n_skipped_other += 1

    print(f"Tensors: {len(tlist)} ({n_skipped_other} skipped), data: {total/1e6:.1f} MB")

    # Write tensor count into header
    struct.pack_into('<I', hdr, 88, len(tlist))

    # Write output file
    fout = open(argv[1], 'wb')
    fout.write(bytes(hdr))

    # Write tensor index
    for name, ndim, dims, off, sz in tlist:
        nb = len(name)
        fout.write(struct.pack('<I', nb))
        fout.write(name.encode())
        fout.write(b'\0')
        fout.write(struct.pack('<I', ndim))
        fout.write(struct.pack(f'<{ndim}I', *dims))
        fout.write(struct.pack('<QQ', off, sz))

    # Quantize and write tensor data
    print(f"Writing {len(tlist)} tensors (this may take a while)...")
    done = 0
    qfn = quant_tile_tq2 if tq2 else quant_tile

    for name, ndim, dims, off, sz in tlist:
        done += 1
        if max_t and done > max_t:
            break

        ten = by_name.get(name)
        if ten is None:
            print(f"  WARN: tensor {name} not found, skipping")
            continue
        flat = to_f32(ten)

        if ndim == 1:
            # 1D tensors: norms, biases — store as raw f32
            fout.write(flat.tobytes())
        elif ndim == 2:
            nr, nc = dims
            w = flat.reshape(nr, nc)
            ntr = (nr + tr - 1) // tr
            ntc = (nc + tc - 1) // tc
            for rr in range(ntr):
                for cc in range(ntc):
                    td = w[rr*tr:rr*tr+tr, cc*tc:cc*tc+tc]
                    fout.write(qfn(td, tr, tc, gs))
        elif ndim == 3:
            ne, nr, nc = dims
            # GGUF stores expert-stacked as [cols, rows, n_experts]
            # flat.reshape(ne, nr, nc) gives [n_experts, rows, cols]
            w = flat.reshape(ne, nr, nc)
            ntr = (nr + tr - 1) // tr
            ntc = (nc + tc - 1) // tc
            for e in range(ne):
                we = w[e]
                for rr in range(ntr):
                    for cc in range(ntc):
                        td = we[rr*tr:rr*tr+tr, cc*tc:cc*tc+tc]
                        fout.write(qfn(td, tr, tc, gs))

        if done <= 5 or done % 200 == 0:
            mb_done = fout.tell() / 1e6
            print(f"  [{done}/{len(tlist)}] {name}: ndim={ndim} dims={dims} "
                  f"({mb_done:.0f} MB written)")

    fout.close()
    mb = os.path.getsize(argv[1]) / 1e6
    print(f"\nDone: {argv[1]} ({mb:.1f} MB)")

if __name__ == '__main__':
    main()

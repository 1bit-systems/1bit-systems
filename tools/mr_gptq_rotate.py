#!/usr/bin/env python3
"""
MR-GPTQ: block-Hadamard GGUF rotation preprocessor for 1bit.systems.
Applies H_128 rotation to each weight tensor row before ternary quantization.
Zero runtime cost (rotation fused into weights).
Based on arXiv:2509.23202 (Egiazarian et al.).
"""
import struct, numpy as np, os
from pathlib import Path

def h128():
    h = np.array([[1]], dtype=np.float32)
    for _ in range(7): h = np.block([[h,h],[h,-h]])
    return h / np.sqrt(128)
_H128 = h128()

# ---- GGUF value types ----
T_UINT8   = 0
T_INT8    = 1
T_UINT16  = 2
T_INT16   = 3
T_UINT32  = 4
T_INT32   = 5
T_FLOAT32 = 6
T_BOOL    = 7
T_STRING  = 8
T_ARRAY   = 9
T_UINT64  = 10
T_INT64   = 11
T_FLOAT64 = 12

def _read_str(f):
    """Read a GGUF string (length-prefixed UTF-8)."""
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8', errors='replace')

def _read_val(f, vtype):
    """Read a GGUF value of given type."""
    if vtype == T_UINT8:   return struct.unpack('<B', f.read(1))[0]
    if vtype == T_INT8:    return struct.unpack('<b', f.read(1))[0]
    if vtype == T_UINT16:  return struct.unpack('<H', f.read(2))[0]
    if vtype == T_INT16:   return struct.unpack('<h', f.read(2))[0]
    if vtype in (T_UINT32, T_INT32): return struct.unpack('<I', f.read(4))[0]
    if vtype == T_FLOAT32: return struct.unpack('<f', f.read(4))[0]
    if vtype == T_BOOL:    return bool(struct.unpack('<B', f.read(1))[0])
    if vtype == T_STRING:  return _read_str(f)
    if vtype == T_ARRAY:
        at = struct.unpack('<I', f.read(4))[0]
        an = struct.unpack('<Q', f.read(8))[0]
        return [_read_val(f, at) for _ in range(an)]
    if vtype in (T_UINT64, T_INT64): return struct.unpack('<q', f.read(8))[0]
    if vtype == T_FLOAT64: return struct.unpack('<d', f.read(8))[0]
    raise ValueError(f'Unknown GGUF value type {vtype}')

def _write_val(f, vtype, val):
    """Write a GGUF value."""
    if vtype in (T_STRING, 8):  # string
        data = val.encode('utf-8') if isinstance(val, str) else val
        f.write(struct.pack('<Q', len(data)))
        f.write(data)
        return
    if vtype in (T_ARRAY, 5, 9):
        # Write array: element type + count + elements
        # For simplicity, assume string array
        f.write(struct.pack('<I', T_STRING))
        f.write(struct.pack('<Q', len(val)))
        for item in val:
            _write_val(f, T_STRING, item)
        return
    # Primitive types
    fmt_map = {
        T_UINT8: '<B', T_INT8: '<b', T_UINT16: '<H', T_INT16: '<h',
        T_UINT32: '<I', T_INT32: '<I', T_FLOAT32: '<f', T_BOOL: '<B',
        T_UINT64: '<Q', T_INT64: '<q', T_FLOAT64: '<d',
    }
    fmt = fmt_map.get(vtype, '<I')
    f.write(struct.pack(fmt, val))

def read_gguf(path):
    """Read GGUF file, return (f, kv_dict, tensor_list).
    File handle `f` is left positioned at the start of tensor data."""
    f = open(path, 'rb')
    if f.read(4) != b'GGUF':
        raise ValueError('Not a GGUF file')
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]
    
    kv = {}
    for _ in range(n_kv):
        key = _read_str(f)
        vtype = struct.unpack('<I', f.read(4))[0]
        kv[key] = _read_val(f, vtype)
    
    alignment = max(kv.get('general.alignment', 32), 32)
    
    tensors = []
    for _ in range(n_tensors):
        name = _read_str(f)
        ndim = struct.unpack('<I', f.read(4))[0]
        shape = list(struct.unpack(f'<{ndim}Q', f.read(8 * ndim)))
        dtype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        tensors.append((name, shape, dtype, offset))
    
    # Compute data start position
    data_start = f.tell()
    rem = data_start % alignment
    if rem:
        data_start += alignment - rem
    
    # Convert offsets to absolute file positions
    tensors = [(n, s, d, o + data_start) for n, s, d, o in tensors]
    return f, kv, tensors


def write_gguf(path, kv, tensors_data):
    """
    Write a GGUF file with proper alignment and offset computation.
    
    Args:
        path: output file path
        kv: dict of metadata key-value pairs
        tensors_data: list of (name, shape, dtype, data_bytes)
            where data_bytes is the raw tensor data
    """
    out = open(path, 'wb')
    out.write(b'GGUF')
    out.write(struct.pack('<I', 3))           # version = 3
    out.write(struct.pack('<Q', len(tensors_data)))  # n_tensors
    out.write(struct.pack('<Q', len(kv)))             # n_kv
    
    # Write KV metadata
    for key, val in kv.items():
        _write_val(out, T_STRING, key)  # write key as string
        if isinstance(val, str):
            out.write(struct.pack('<I', T_STRING))
            _write_val(out, T_STRING, val)
        elif isinstance(val, int):
            out.write(struct.pack('<I', T_UINT32))
            out.write(struct.pack('<I', val))
        elif isinstance(val, float):
            out.write(struct.pack('<I', T_FLOAT64))
            out.write(struct.pack('<d', val))
        elif isinstance(val, list):
            out.write(struct.pack('<I', T_ARRAY))
            out.write(struct.pack('<I', T_STRING))
            out.write(struct.pack('<Q', len(val)))
            for item in val:
                _write_val(out, T_STRING, item)
        elif isinstance(val, bool):
            out.write(struct.pack('<I', T_BOOL))
            out.write(struct.pack('<B', int(val)))
        else:
            out.write(struct.pack('<I', T_STRING))
            _write_val(out, T_STRING, str(val))
    
    # Compute alignment
    alignment = max(kv.get('general.alignment', 32), 32)
    
    # Write tensor info with OFFSET = running cumulative bytes from data_start
    offset = 0
    for name, shape, dtype, _ in tensors_data:
        _write_val(out, T_STRING, name)
        out.write(struct.pack('<I', len(shape)))
        for dim in shape:
            out.write(struct.pack('<Q', dim))
        out.write(struct.pack('<I', dtype))  # always F16 (type 1) for output
        out.write(struct.pack('<Q', offset))
        # Accumulate bytes (F16 = 2 bytes per element)
        nbytes = int(np.prod(shape)) * 2
        offset += nbytes
    
    # Pad to alignment boundary
    header_end = out.tell()
    rem = header_end % alignment
    if rem:
        out.write(b'\x00' * (alignment - rem))
    
    # Write tensor data (F16, already rotated)
    for _, _, _, data in tensors_data:
        out.write(data.tobytes() if hasattr(data, 'tobytes') else data)
    
    out.close()


def rotate_gguf(inpath, outpath):
    """Read a GGUF, rotate all F16/F32 weight tensors with block-Hadamard,
    write a new GGUF with the same headers but rotated weights."""
    f, kv, tensors = read_gguf(inpath)
    print(f'Opened {inpath}: {len(tensors)} tensors')
    
    out_tensors = []
    rotated_count = 0
    skipped_count = 0
    
    for name, shape, dtype, file_offset in tensors:
        # Skip non-weight tensors (norms, embeddings, biases)
        skip = any(k in name for k in ['norm', 'embed', 'bias', 'tok_embeddings',
                                         'output.weight', 'token_embd'])
        skip = skip or len(shape) < 2 or int(np.prod(shape)) < 128
        
        # Read tensor data
        elem_size = {0: 4, 1: 2}.get(dtype, 2)
        nbytes = int(np.prod(shape)) * elem_size
        
        f.seek(file_offset)
        raw = f.read(nbytes)
        if len(raw) < nbytes:
            print(f'  WARN: short read on {name}')
            continue
        
        if skip or dtype not in (0, 1):  # not F16 or F32, or not a weight
            print(f'  COPY {name} {shape}')
            out_tensors.append((name, shape, 1, raw))  # write as F16
            skipped_count += 1
            continue
        
        # Decode to float32
        dt = np.float16 if dtype == 1 else np.float32
        arr = np.frombuffer(raw, dtype=dt).reshape(shape).astype(np.float32)
        
        # Apply H_128 per row (axis=1 = columns)
        print(f'  ROTATE {name} {shape}', end=' ', flush=True)
        w = np.ascontiguousarray(arr)
        for r in range(w.shape[0]):
            row = w[r]
            for i in range(row.shape[0] // 128):
                seg = row[i * 128:(i + 1) * 128]
                row[i * 128:(i + 1) * 128] = seg @ _H128.T
        
        # Write as F16
        out_tensors.append((name, shape, 1, w.astype(np.float16)))
        rotated_count += 1
        print(f'done [{w.shape[0]}x{w.shape[1]}]')
    
    f.close()
    
    print(f'\nRotated {rotated_count}, copied {skipped_count}')
    write_gguf(outpath, kv, out_tensors)
    print(f'Wrote {outpath}')


if __name__ == '__main__':
    import sys
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else inp.replace('.gguf', '_rotated.gguf')
    rotate_gguf(inp, out)
    print(f'After rotation, run: llama-quantize {out} model.gguf IQ1_S')

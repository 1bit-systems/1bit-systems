#!/usr/bin/env python3
"""
MR-GPTQ: block-Hadamard GGUF rotation preprocessor for 1bit.systems.
Applies H_128 rotation to each weight tensor row before ternary quantization.
Zero runtime cost (rotation fused into weights).
Based on arXiv:2509.23202 (Egiazarian et al.).
"""
import struct, numpy as np
from pathlib import Path

def h128():
    h = np.array([[1]], dtype=np.float32)
    for _ in range(7): h = np.block([[h,h],[h,-h]])
    return h / np.sqrt(128)
_H128 = h128()

def read_gguf(path):
    f = open(path, 'rb')
    if f.read(4) != b'GGUF': raise ValueError('not GGUF')
    ver = struct.unpack('<I', f.read(4))[0]
    nt  = struct.unpack('<Q', f.read(8))[0]
    nkv = struct.unpack('<Q', f.read(8))[0]
    kv = {}
    def rstr(): l = struct.unpack('<Q', f.read(8))[0]; return f.read(l).decode()
    def rval(t):
        if t==0: return struct.unpack('<f', f.read(4))[0]
        if t==4: return rstr()
        if t==5: at=struct.unpack('<I',f.read(4))[0]; an=struct.unpack('<Q',f.read(8))[0]; return [rval(at) for _ in range(an)]
        if t==6: return struct.unpack('<Q',f.read(8))[0]
        return struct.unpack('<I',f.read(4))[0]
    for _ in range(nkv):
        k = rstr()
        vt = struct.unpack('<I', f.read(4))[0]
        kv[k] = rval(vt) if vt in (0,4,5,6) else struct.unpack('<f',f.read(4))[0] if vt==2 else None
    tensors = []
    for _ in range(nt):
        name = rstr()
        ndim = struct.unpack('<I', f.read(4))[0]
        shape = list(struct.unpack(f'<{ndim}Q', f.read(8*ndim)))
        dtype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        tensors.append((name, shape, dtype, offset))
    align = max(kv.get('general.alignment',32),32)
    ds = f.tellg() if hasattr(f,'tellg') else f.tell()
    rem = ds % align
    if rem: ds += align - rem
    tensors = [(n,s,d,o+ds) for n,s,d,o in tensors]
    return f, kv, tensors

def rotate_gguf(inpath, outpath):
    f, kv, tensors = read_gguf(inpath)
    out = open(outpath, 'wb')
    out.write(b'GGUF'); out.write(struct.pack('<I',3))
    out.write(struct.pack('<Q',len(tensors))); out.write(struct.pack('<Q',len(kv)))
    seen = {}
    for name, shape, dtype, offset in tensors:
        skip = any(k in name for k in ['norm','embed','bias','tok_embeddings','output.weight','token_embd'])
        skip = skip or len(shape) < 2 or np.prod(shape) < 128
        if skip:
            f.seek(offset); raw = f.read(np.prod(shape)*(2 if dtype==1 else 4))
            seen[name] = (shape, raw); continue
        f.seek(offset)
        dt = {0:np.float32,1:np.float16}.get(dtype,np.float16)
        arr = np.frombuffer(f.read(np.prod(shape)*dt().itemsize), dtype=dt).reshape(shape).astype(np.float32)
        print(f'  rotate {name} {shape}')
        # Apply H_128 per row (axis=1 = columns)
        w = np.ascontiguousarray(arr)
        for r in range(w.shape[0]):
            row = w[r]; dim = row.shape[0]
            for i in range(dim // 128):
                seg = row[i*128:(i+1)*128]
                row[i*128:(i+1)*128] = seg @ _H128.T
        seen[name] = (shape, w.astype(np.float16).tobytes())
    # Write output
    offset = 0
    tensor_entries = []
    for name, shape, dtype, _ in tensors:
        out.write(struct.pack('<Q',len(name))+name.encode())
        out.write(struct.pack('<I',len(shape)))
        for d in shape: out.write(struct.pack('<Q',d))
        out.write(struct.pack('<I',1))  # F16
        tensor_entries.append((name, offset))
        out.write(struct.pack('<Q',offset))
        nbytes = np.prod(shape) * 2
        offset += nbytes
    pad = (32 - out.tell() % 32) % 32
    out.write(b'\x00'*pad)
    for name, _, _, _ in tensors:
        shp, data = seen.get(name, (None,b''))
        if not data:
            f.seek(offset); data = f.read(np.prod(shape)*2)
        out.write(data)
    f.close(); out.close()
    print(f'Wrote {outpath}')

if __name__ == '__main__':
    import sys; rotate_gguf(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else sys.argv[1].replace('.gguf','_rotated.gguf'))

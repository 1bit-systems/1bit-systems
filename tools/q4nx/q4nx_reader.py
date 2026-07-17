#!/usr/bin/env python3
"""
q4nx_reader.py — Read, dump, and extract FLM Q4NX model files.

Q4NX format:
  [4B magic 0x00008718][4B flags][JSON manifest][raw binary weights]

Usage:
  python q4nx_reader.py info model.q4nx              # Show model summary
  python q4nx_reader.py dump model.q4nx               # List all tensors
  python q4nx_reader.py extract model.q4nx --name lm_head.weight -o head.bin
  python q4nx_reader.py extract model.q4nx --all -o /tmp/weights/
"""

import struct, json, os, sys
import numpy as np

MAGIC = 0x00008718

class Q4NXReader:
    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            raw = f.read()
        
        self.magic = struct.unpack("<I", raw[0:4])[0]
        self.flags = struct.unpack("<I", raw[4:8])[0]
        
        if self.magic != MAGIC:
            raise ValueError(f"Bad magic: 0x{self.magic:08X} (expected 0x{MAGIC:08X})")
        
        # Parse JSON manifest
        depth = 0; self.json_end = 8; in_str = escape = False
        for i in range(8, len(raw)):
            c = chr(raw[i])
            if escape: escape = False; continue
            if c == '\\': escape = True; continue
            if c == '"' and not escape: in_str = not in_str; continue
            if in_str: continue
            if c == '{': depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0: self.json_end = i + 1; break
        
        self.raw = raw
        self.json_bytes = raw[8:self.json_end]
        self.manifest = json.loads(self.json_bytes.decode('utf-8'))
        self.data_start = self.json_end
        
        # Build offset-sorted access list
        self.tensors = []
        for name, info in self.manifest.items():
            start, end = info['data_offsets']
            self.tensors.append({
                'name': name,
                'dtype': info['dtype'],
                'shape': tuple(info['shape']),
                'offset': start,
                'size': end - start,
            })
    
    @property
    def file_size(self):
        return len(self.raw)
    
    @property
    def header_size(self):
        return self.json_end
    
    @property
    def data_size(self):
        return len(self.raw) - self.json_end
    
    def get_tensor(self, name: str) -> np.ndarray:
        """Read a tensor's raw bytes from the file."""
        if name not in self.manifest:
            raise KeyError(f"Tensor '{name}' not found. Available: {[t['name'] for t in self.tensors[:5]]}...")
        
        info = self.manifest[name]
        start, end = info['data_offsets']
        data_start = self.data_start + start
        data_end = self.data_start + end
        raw_bytes = self.raw[data_start:data_end]
        
        dtype = info['dtype']
        shape = tuple(info['shape'])
        
        if dtype == 'BF16':
            arr = np.frombuffer(raw_bytes, dtype=np.uint16).reshape(shape).copy()
            # View as bfloat16 via float32 hack
            i32 = arr.astype(np.uint32) << 16
            return i32.view(np.float32)
        elif dtype == 'I8':
            return np.frombuffer(raw_bytes, dtype=np.int8).reshape(shape).copy()
        elif dtype == 'F32':
            return np.frombuffer(raw_bytes, dtype=np.float32).reshape(shape).copy()
        elif dtype == 'F16':
            return np.frombuffer(raw_bytes, dtype=np.float16).reshape(shape).copy()
        else:
            return np.frombuffer(raw_bytes, dtype=np.uint8).reshape(shape).copy()
    
    def get_tensor_raw(self, name: str) -> bytes:
        info = self.manifest[name]
        start, end = info['data_offsets']
        return bytes(self.raw[self.data_start + start : self.data_start + end])
    
    def summary(self) -> dict:
        """Return summary stats."""
        fp16_total = 0
        actual_total = 0
        dtype_sizes = {}
        layer_count = 0
        layer_names = set()
        
        for name, info in self.manifest.items():
            el = 1
            for d in info['shape']: el *= d
            fp16_total += el * 2
            sz = info['data_offsets'][1] - info['data_offsets'][0]
            actual_total += sz
            dt = info['dtype']
            dtype_sizes[dt] = dtype_sizes.get(dt, 0) + sz
            
            if name.startswith("model.layers."):
                lidx = name.split(".")[2]
                if lidx.isdigit():
                    layer_count = max(layer_count, int(lidx) + 1)
                    layer_names.add(name.split("model.layers.")[1].split(".")[0])
        
        return {
            'file_size_gb': self.file_size / (1024**3),
            'data_size_gb': self.data_size / (1024**3),
            'tensor_count': len(self.tensors),
            'layer_count': layer_count,
            'fp16_equiv_gb': fp16_total / (1024**3),
            'actual_gb': actual_total / (1024**3),
            'ratio_pct': actual_total / fp16_total * 100,
            'dtype_sizes_gb': {k: v/(1024**3) for k, v in sorted(dtype_sizes.items(), key=lambda x: -x[1])},
        }
    
    def print_summary(self):
        s = self.summary()
        print(f"File:      {self.path}")
        print(f"Size:      {s['file_size_gb']:.2f} GB ({s['data_size_gb']:.2f} GB weights)")
        print(f"Tensors:   {s['tensor_count']} ({s['layer_count']} layers)")
        print(f"FP16 eq:   {s['fp16_equiv_gb']:.2f} GB")
        print(f"Actual:    {s['actual_gb']:.2f} GB ({s['ratio_pct']:.1f}% of FP16)")
        print(f"Dtypes:    " + ", ".join(f"{k}={v:.2f} GB" for k, v in s['dtype_sizes_gb'].items()))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Q4NX model reader")
    parser.add_argument("command", choices=["info", "dump", "extract", "validate"])
    parser.add_argument("path", help="Path to model.q4nx file")
    parser.add_argument("--name", help="Tensor name to extract")
    parser.add_argument("--all", action="store_true", help="Extract all tensors")
    parser.add_argument("-o", "--output", default=".", help="Output path")
    parser.add_argument("--raw", action="store_true", help="Skip BF16 conversion, dump raw bytes")
    args = parser.parse_args()
    
    reader = Q4NXReader(args.path)
    
    if args.command == "info":
        reader.print_summary()
    
    elif args.command == "dump":
        reader.print_summary()
        print(f"\nTensors:")
        for t in sorted(reader.tensors, key=lambda x: x['offset']):
            mb = t['size'] / (1024*1024)
            print(f"  {t['dtype']:4s} {str(t['shape']):30s} {t['offset']:>10,}-{t['offset']+t['size']:>10,}  {mb:7.1f} MB  {t['name']}")
    
    elif args.command == "extract":
        os.makedirs(args.output, exist_ok=True)
        if args.name:
            names = [args.name]
        elif args.all:
            names = [t['name'] for t in reader.tensors]
        else:
            parser.error("Specify --name or --all")
        
        for name in names:
            data = reader.get_tensor_raw(name) if args.raw else reader.get_tensor(name)
            if args.raw:
                out_path = os.path.join(args.output, name.replace("/", "_").replace(".", "_") + ".bin")
                with open(out_path, "wb") as f:
                    f.write(data)
            else:
                out_path = os.path.join(args.output, name.replace("/", "_").replace(".", "_") + ".npy")
                np.save(out_path, data)
            print(f"Extracted {name} -> {out_path}  ({len(data)} bytes)")


if __name__ == "__main__":
    main()

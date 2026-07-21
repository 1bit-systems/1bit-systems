#!/usr/bin/env python3
"""Inspect GGUF metadata keys and types."""
import struct, sys

with open(sys.argv[1], "rb") as f:
    magic, ver, nt, nm = struct.unpack("<IIQQ", f.read(24))
    print(f"GGUF v{ver}: {nt} tensors, {nm} metadata keys")
    for i in range(nm):
        klen = struct.unpack("<Q", f.read(8))[0]
        key = f.read(klen).decode()
        dt = struct.unpack("<I", f.read(4))[0]
        if dt == 4:  # uint32
            val = struct.unpack("<I", f.read(4))[0]
            print(f"  [{dt}] {key} = {val}")
        elif dt == 6:  # float32
            val = struct.unpack("<f", f.read(4))[0]
            print(f"  [{dt}] {key} = {val}")
        elif dt == 8:  # string
            sl = struct.unpack("<Q", f.read(8))[0]
            print(f"  [{dt}] {key} = \"{f.read(sl).decode()}\"")
        elif dt == 9:  # array
            at = struct.unpack("<I", f.read(4))[0]
            al = struct.unpack("<Q", f.read(8))[0]
            print(f"  [{dt}] {key} = array[{al}] type={at}")
            for _ in range(al):
                sl = struct.unpack("<Q", f.read(8))[0]
                f.read(sl)
        else:
            print(f"  [{dt}] {key} (unknown type)")
            f.read(4)

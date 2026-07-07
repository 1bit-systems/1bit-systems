#!/usr/bin/env python3
# Verify dequant_q4nx.c signed-nibble vs unsigned-nibble against HF safetensors.
# Run: python3 verify_dequant_unsigned.py
import torch, struct, mmap, numpy as np
from safetensors import safe_open
HF="/home/bcloud/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
MP="/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx"
def bf16(v):
    bits=(v<<16)&0xFFFFFFFF
    return struct.unpack('<f',struct.pack('<I',bits))[0]
def dequant(signed):
    fp=open(MP,'rb');hsz=struct.unpack('<Q',fp.read(8))[0];js=fp.read(hsz).decode()
    p=js.find('"model.layers.0.self_attn.q_proj.weight"');o=js.find('"data_offsets"',p);a=js.find('[',o);b=js.find(']',a)
    qo=int(js[a+1:b].split(',')[0])
    mm=mmap.mmap(fp.fileno(),0,access=mmap.ACCESS_READ); df=8+hsz
    out=np.zeros((2048,1024),dtype=np.float32)
    for ir in range(64*4):
        tr=ir//4; tc=ir%4
        rd=mm[df+qo+ir*5120: df+qo+(ir+1)*5120]
        for lr in range(32):
            lane=lr//16; lane_row=lr%16; byte_idx=lane_row//2; nib_sel=lr%2
            for col in range(256):
                g=col//32
                sc=bf16(int.from_bytes(rd[g*32+lr*2:g*32+lr*2+2],'little'))
                zp=bf16(int.from_bytes(rd[512+g*32+lr*2:512+g*32+lr*2+2],'little'))
                b=rd[1024+ lane*256*8 + col*8 + byte_idx]
                code=b&0x0F if nib_sel==0 else (b>>4)&0x0F
                if signed and code>=8: code-=16
                out[tr*32+lr, tc*256+col]=code*sc+zp
    return out
with safe_open(HF+"/model.safetensors","pt") as f: hf=f.get_tensor("model.layers.0.self_attn.q_proj.weight").float().numpy()
for name,dq in [("SIGNED   (old)",dequant(True)),("UNSIGNED (fix)",dequant(False))]:
    d=np.abs(dq-hf).max(); r=(dq[hf!=0]/hf[hf!=0]).mean()
    print(f"{name}: max_abs_diff={d:.4f}  ratio_mean={r:.3f}  norm={np.linalg.norm(dq):.2f} (HF={np.linalg.norm(hf):.2f})")

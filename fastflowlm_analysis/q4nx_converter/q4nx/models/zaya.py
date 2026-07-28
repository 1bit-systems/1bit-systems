"""Zaya1-8B → Q4NX converter"""
from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch
from gguf import GGUFReader, dequantize
import torch
import numpy as np

class Zaya(__Q4NX_Converter, model_arch=ModelArch.ZAYA):
    """Convert Zaya1-8B GGUF to Q4NX format for NPU inference.
    
    Architecture: CCA attention + TQ1 MoE
    - hidden=2048, heads=8, kv_heads=2, head_dim=128
    - 40 layers, 16 experts, top-2, FFN=2048
    - vocab=262272
    """
    
    def __init__(self, gguf_reader: GGUFReader):
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()
    
    def initialize(self):
        super().initialize()
    
    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        self.q4nx_tensors = {}
        
        if not self._has_lm_head():
            print("[INFO] Using embedding weights as lm_head")
            unpacked = self.gguf_tensors["token_embd.weight"].unpack(self.default_tensor_type)
            self.q4nx_tensors["lm_head.weight"] = self._pack_q4nx(*unpacked)
        
        for key, gguf_tensor in self.gguf_tensors.items():
            name = gguf_tensor.name
            
            # Embedding stays as BF16
            if "token_embd.weight" in name:
                w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
                w = torch.from_numpy(w).contiguous().to(torch.bfloat16)
                self.q4nx_tensors[self.forward_name_map.get(name, name)] = w
                continue
            
            # Layer norm weights stay as BF16
            if "weight" in name and ("norm" in name or "layernorm" in name):
                w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
                w = torch.from_numpy(w).contiguous().to(torch.bfloat16)
                self.q4nx_tensors[self.forward_name_map.get(name, name)] = w
                continue
            
            # All other weights: INT4 Q4NX quantize
            unpacked = gguf_tensor.unpack(self.default_tensor_type)
            self.q4nx_tensors[self.forward_name_map.get(gguf_tensor.name, gguf_tensor.name)] = self._pack_q4nx(*unpacked)
        
        self._export_q4nx_tensors(q4nx_path)
        self._extract_tokenizer_json(q4nx_path)
        print(f"Zaya Q4NX written to {q4nx_path}")

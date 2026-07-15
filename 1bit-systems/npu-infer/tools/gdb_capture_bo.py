#!/usr/bin/env python3
"""
GDB Python capture script for FLM BO protocol analysis.

Usage (attach to running FLM):
  gdb -batch -x gdb_capture_bo.py -p $(pgrep -f "flm serve")

Or (launch FLM under GDB):
  gdb -batch -x gdb_capture_bo.py --args /opt/fastflowlm/bin/flm serve --host 127.0.0.1 --port 8080
"""

import gdb
import struct
import os
import sys
import json
import re

# ==========================================================
# Configuration
# ==========================================================
CAPTURE_DIR = "/tmp/gdb_flm_capture"
os.makedirs(CAPTURE_DIR, exist_ok=True)

# Counters
bp_counters = {
    "gen_layer_seq": 0,
    "gen_mha_engine_seq": 0,
    "gen_dequant_seq": 0,
    "gen_lm_head_seq": 0,
    "cmds2seq": 0,
    "send_x": 0,
    "move_weights": 0,
    "send_rms_weights": 0,
    "move_kv_cache": 0,
}

# ==========================================================
# Output Files
# ==========================================================
log_file = open(f"{CAPTURE_DIR}/gdb_capture.log", "w", buffering=1)
instr_file = open(f"{CAPTURE_DIR}/gdb_capture_instrs.bin", "wb", buffering=0)
bo_file = open(f"{CAPTURE_DIR}/gdb_bo_info.txt", "w", buffering=1)


def l(msg):
    """Log to file and stdout"""
    log_file.write(f"[GDB] {msg}\n")
    print(f"[GDB CAPTURE] {msg}", flush=True)


def dump_mem(inferior, addr, size, desc=""):
    """Read memory and return hex dump string"""
    try:
        data = inferior.read_memory(addr, size)
        result = f"--- {desc} @ 0x{int(addr):x} ({size} bytes) ---\n"
        for i in range(0, len(data), 16):
            chunk = data[i:i+16]
            hex_str = ' '.join(f'{b:02x}' for b in chunk)
            ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
            result += f"  0x{i:04x}: {hex_str:<48s} {ascii_str}\n"
        return result
    except Exception as e:
        return f"  Error reading at 0x{int(addr):x}: {e}\n"


def read_qword(addr):
    """Read a 64-bit value from inferior memory"""
    try:
        inf = gdb.selected_inferior()
        data = inf.read_memory(addr, 8)
        return struct.unpack('<Q', bytes(data))[0]
    except:
        return 0


def read_dword(addr):
    """Read a 32-bit value from inferior memory"""
    try:
        inf = gdb.selected_inferior()
        data = inf.read_memory(addr, 4)
        return struct.unpack('<I', bytes(data))[0]
    except:
        return 0


def read_word(addr):
    """Read a 16-bit value from inferior memory"""
    try:
        inf = gdb.selected_inferior()
        data = inf.read_memory(addr, 2)
        return struct.unpack('<H', bytes(data))[0]
    except:
        return 0


def read_string(addr, max_len=256):
    """Read a null-terminated string from inferior memory"""
    try:
        inf = gdb.selected_inferior()
        data = b""
        offset = 0
        while offset < max_len:
            byte = inf.read_memory(addr + offset, 1)
            if byte[0] == 0:
                break
            data += bytes(byte)
            offset += 1
        return data.decode('utf-8', errors='replace')
    except:
        return ""
    
# ==========================================================
# Breakpoints
# ==========================================================

class GenLayerSeqBP(gdb.Breakpoint):
    """Break on gen_layer_seq to capture layer-level state"""
    def __init__(self):
        super().__init__("qwen3_npu_sequence::Impl::gen_layer_seq")
    
    def stop(self):
        bp_counters["gen_layer_seq"] += 1
        n = bp_counters["gen_layer_seq"]
        frame = gdb.selected_frame()
        inferior = gdb.selected_inferior()
        
        # AMD64 ABI: rdi=this, rsi=npu_seq*, rdx=layer_index
        this_ptr = int(gdb.parse_and_eval("$rdi"))
        npu_seq_ptr = int(gdb.parse_and_eval("$rsi"))
        layer = int(gdb.parse_and_eval("$edx")) & 0xFFFFFFFF
        
        l(f"\n{'='*60}")
        l(f"gen_layer_seq #{n}: this=0x{this_ptr:x}, npu_seq=0x{npu_seq_ptr:x}, layer={layer}")
        l(f"{'='*60}")
        
        # Dump npu_sequence object header
        seq_vtable = read_qword(npu_seq_ptr)
        l(f"  npu_sequence vtable=0x{seq_vtable:x}")
        
        # The npu_sequence layout (64 bytes+):
        # vtable(8) + vector<command*>(24) + state_vars
        l(f"\n  npu_sequence raw dump:")
        l(dump_mem(inferior, npu_seq_ptr, 128, "npu_sequence"))
        
        # Dump command vector (std::vector is at offset 8: begin, end, capacity)
        cmd_begin = read_qword(npu_seq_ptr + 8)
        cmd_end = read_qword(npu_seq_ptr + 16)
        cmd_cap = read_qword(npu_seq_ptr + 24)
        cmd_count = (cmd_end - cmd_begin) // 8  # each is a pointer (8 bytes)
        l(f"\n  command vector: begin=0x{cmd_begin:x}, end=0x{cmd_end:x}, capacity=0x{cmd_cap:x}")
        l(f"  command count: {cmd_count}")
        
        # Dump each command pointer
        if cmd_count > 0 and cmd_count < 100:
            for i in range(min(cmd_count, 50)):
                cmd_ptr = read_qword(cmd_begin + i * 8)
                if cmd_ptr:
                    l(f"    cmd[{i}] = 0x{cmd_ptr:x}")
        
        # Dump the qwen3_npu_sequence::Impl object (this)
        l(f"\n  Impl (this=0x{this_ptr:x}) raw dump:")
        l(dump_mem(inferior, this_ptr, 256, "qwen3_npu_sequence::Impl"))
        
        # Look for tile arrays within Impl
        # proj_tiles, attn_kv_tiles, attn_qk_tiles, mvm_tiles are weak symbols
        # Their addresses might be in the data section or embedded in the Impl
        for name, symbol in [
            ("proj_tiles", "_ZN18qwen3_npu_sequence4Impl10proj_tilesE"),
            ("attn_kv_tiles", "_ZN18qwen3_npu_sequence4Impl13attn_kv_tilesE"),
            ("attn_qk_tiles", "_ZN18qwen3_npu_sequence4Impl13attn_qk_tilesE"),
            ("mvm_tiles", "_ZN18qwen3_npu_sequence4Impl9mvm_tilesE"),
        ]:
            try:
                addr = int(gdb.parse_and_eval(f"(void*){symbol}"))
                if addr:
                    # Read tile data (likely 4 uint32_t values per tile)
                    l(f"\n  {name} @ 0x{addr:x}:")
                    for row in range(4):
                        vals = [read_dword(addr + row*16 + c*4) for c in range(4)]
                        l(f"    [{row}]: {vals}")
            except:
                pass
        
        # Check the Impl's string members (model paths, etc.)
        # std::string is 32 bytes: ptr(8) + size(8) + union{capacity, ptr}(8) + ?(8)
        l(f"\n  Checking for string members in Impl (offset 0-192):")
        for soff in range(0, 192, 32):
            sptr = read_qword(this_ptr + soff)
            ssize = read_qword(this_ptr + soff + 8)
            if sptr and 0 < ssize < 1000 and sptr > 0x100000:
                sstr = read_string(sptr if ssize > 15 else this_ptr + soff, min(ssize+1, 256))
                if sstr and len(sstr) > 2 and not all(c == '\x00' for c in sstr):
                    l(f"    string @ offset 0x{soff:x}: \"{sstr}\" (ptr=0x{sptr:x}, size={ssize})")
        
        return False  # Don't stop execution


class Cmds2seqBP(gdb.FinishBreakpoint):
    """Break after cmds2seq() returns — capture the compiled DMA instructions"""
    def __init__(self, seq_num):
        super().__init__(internal=True)
        self.seq_num = seq_num
    
    def stop(self):
        bp_counters["cmds2seq"] += 1
        n = bp_counters["cmds2seq"]
        l(f"\n  [cmds2seq RETURN #{n}] (seq #{self.seq_num})")
        
        # The return value is in RAX - it's std::vector<uint32_t> by value
        # RAX points to the returned vector
        vec_ptr = int(gdb.parse_and_eval("$rax"))
        l(f"  returned vector @ 0x{vec_ptr:x}")
        
        inferior = gdb.selected_inferior()
        vec_begin = read_qword(vec_ptr)
        vec_end = read_qword(vec_ptr + 8)
        vec_count = (vec_end - vec_begin) // 4
        
        l(f"  instructions: begin=0x{vec_begin:x}, count={vec_count}")
        
        # Always save the first cmds2seq output per layer
        if vec_count > 0 and vec_count < 65536:
            l(f"  first 64 values:")
            for i in range(min(vec_count, 64)):
                val = read_dword(vec_begin + i * 4)
                l(f"    [{i:4d}] = 0x{val:08x} ({val})")
        
        return False  # Don't stop


class GenLayerSeqExceptionBP(gdb.Breakpoint):
    """Catch any crashes/exceptions in gen_layer_seq"""
    def __init__(self):
        super().__init__("__cxx_throw_reference", internal=True)
    
    def stop(self):
        l("\n*** C++ EXCEPTION THROWN ***")
        frame = gdb.selected_frame()
        # Try to get exception message
        try:
            exc_ptr = int(gdb.parse_and_eval("$rdi"))
            l(f"  exception pointer: 0x{exc_ptr:x}")
        except:
            pass
        return False


class GenMhaEngineSeqBP(gdb.Breakpoint):
    """Break to capture MHA engine sequence generation"""
    def __init__(self):
        super().__init__("qwen3_npu_sequence::Impl::gen_mha_engine_seq")
    
    def stop(self):
        bp_counters["gen_mha_engine_seq"] += 1
        n = bp_counters["gen_mha_engine_seq"]
        frame = gdb.selected_frame()
        inferior = gdb.selected_inferior()
        
        this_ptr = int(gdb.parse_and_eval("$rdi"))
        npu_seq_ptr = int(gdb.parse_and_eval("$rsi"))
        
        l(f"\n--- gen_mha_engine_seq #{n}: this=0x{this_ptr:x}, npu_seq=0x{npu_seq_ptr:x} ---")
        return False


class GenDequantSeqBP(gdb.Breakpoint):
    """Break to capture dequant sequence generation"""
    def __init__(self):
        super().__init__("qwen3_npu_sequence::Impl::gen_dequant_seq")
    
    def stop(self):
        bp_counters["gen_dequant_seq"] += 1
        n = bp_counters["gen_dequant_seq"]
        frame = gdb.selected_frame()
        inferior = gdb.selected_inferior()
        
        this_ptr = int(gdb.parse_and_eval("$rdi"))
        npu_seq_ptr = int(gdb.parse_and_eval("$rsi"))
        
        l(f"\n--- gen_dequant_seq #{n}: this=0x{this_ptr:x}, npu_seq=0x{npu_seq_ptr:x} ---")
        return False


class GenLmHeadSeqBP(gdb.Breakpoint):
    """Break to capture LM head sequence generation"""
    def __init__(self):
        super().__init__("qwen3_npu_sequence::Impl::gen_lm_head_seq")
    
    def stop(self):
        bp_counters["gen_lm_head_seq"] += 1
        n = bp_counters["gen_lm_head_seq"]
        frame = gdb.selected_frame()
        inferior = gdb.selected_inferior()
        
        this_ptr = int(gdb.parse_and_eval("$rdi"))
        npu_seq_ptr = int(gdb.parse_and_eval("$rsi"))
        
        l(f"\n--- gen_lm_head_seq #{n}: this=0x{this_ptr:x}, npu_seq=0x{npu_seq_ptr:x} ---")
        
        # Dump the returned vector
        return False


class SendXBP(gdb.Breakpoint):
    """Break on _send_x — input activation preparation"""
    def __init__(self):
        super().__init__("qwen3_npu_sequence::Impl::_send_x")
    
    def stop(self):
        bp_counters["send_x"] += 1
        n = bp_counters["send_x"]
        frame = gdb.selected_frame()
        inferior = gdb.selected_inferior()
        
        this_ptr = int(gdb.parse_and_eval("$rdi"))
        npu_seq_ptr = int(gdb.parse_and_eval("$rsi"))
        
        l(f"\n--- _send_x #{n}: this=0x{this_ptr:x}, npu_seq=0x{npu_seq_ptr:x} ---")
        
        # Check args - _send_x is:  _ZN18qwen3_npu_sequence4Impl7_send_xEP12npu_sequence
        # It needs to know what data to send. This data is in the Impl object
        # The input activations come from the previous layer output
        
        # Dump more of the Impl to find BO addresses
        l(dump_mem(inferior, this_ptr, 256, "Impl in _send_x"))
        
        return False


class MoveWeightsBP(gdb.Breakpoint):
    """Break on _move_weights — weight buffer setup"""
    def __init__(self):
        super().__init__("qwen3_npu_sequence::Impl::_move_weights")
    
    def stop(self):
        bp_counters["move_weights"] += 1
        n = bp_counters["move_weights"]
        frame = gdb.selected_frame()
        inferior = gdb.selected_inferior()
        
        this_ptr = int(gdb.parse_and_eval("$rdi"))
        npu_seq_ptr = int(gdb.parse_and_eval("$rsi"))
        
        l(f"\n--- _move_weights #{n}: this=0x{this_ptr:x}, npu_seq=0x{npu_seq_ptr:x} ---")
        
        return False


class SendRmsWeightsBP(gdb.Breakpoint):
    """Break on _send_rms_weights — RMS norm weights"""
    def __init__(self):
        super().__init__("qwen3_npu_sequence::Impl::_send_rms_weights")
    
    def stop(self):
        bp_counters["send_rms_weights"] += 1
        n = bp_counters["send_rms_weights"]
        frame = gdb.selected_frame()
        inferior = gdb.selected_inferior()
        
        this_ptr = int(gdb.parse_and_eval("$rdi"))
        npu_seq_ptr = int(gdb.parse_and_eval("$rsi"))
        
        l(f"\n--- _send_rms_weights #{n}: this=0x{this_ptr:x}, npu_seq=0x{npu_seq_ptr:x} ---")
        
        return False


# ==========================================================
# GDB Session Setup
# ==========================================================

def setup():
    l("=" * 60)
    l("GDB FLM BO Protocol Capture Script")
    l(f"Capture directory: {CAPTURE_DIR}")
    l("=" * 60)
    
    # Set up GDB
    gdb.execute("set pagination off")
    gdb.execute("set print thread-events off")
    gdb.execute("set breakpoint condition-evaluation host")
    
    # Register breakpoints
    l("\nRegistering breakpoints...")
    
    GenLayerSeqBP()
    l("  ✓ gen_layer_seq (will NOT stop)")
    
    SendXBP()
    l("  ✓ _send_x (will NOT stop)")
    
    MoveWeightsBP()
    l("  ✓ _move_weights (will NOT stop)")
    
    SendRmsWeightsBP()
    l("  ✓ _send_rms_weights (will NOT stop)")
    
    GenMhaEngineSeqBP()
    l("  ✓ gen_mha_engine_seq (will NOT stop)")
    
    GenDequantSeqBP()
    l("  ✓ gen_dequant_seq (will NOT stop)")
    
    GenLmHeadSeqBP()
    l("  ✓ gen_lm_head_seq (will NOT stop)")
    
    l("\nBreakpoints registered. Continuing execution...")
    l("Make an inference request to FLM to trigger captures.")
    l(f"Output: {CAPTURE_DIR}/gdb_capture.log")


# ==========================================================
# Run
# ==========================================================

setup()

#!/usr/bin/env python3
"""
GDB capture focused on:
1. XRT BO creation/handles (via xrt::bo::sync or xrt::kernel::operator())
2. DMA instruction dump from cmds2seq return value
3. The npu_sequence command list BEFORE cmds2seq (to see the DMA commands)
"""

import gdb
import struct
import os
import sys

# ==========================================================
# Configuration
# ==========================================================
CAPTURE_DIR = "/tmp/gdb_flm_capture_v4"
os.makedirs(CAPTURE_DIR, exist_ok=True)

log_file = open(f"{CAPTURE_DIR}/gdb_capture.log", "w", buffering=1)
instr_file = open(f"{CAPTURE_DIR}/gdb_instructions.bin", "wb", buffering=0)
seq_cmds_file = open(f"{CAPTURE_DIR}/gdb_seq_commands.txt", "w", buffering=1)

counters = {"gen_layer": 0, "cmds2seq": 0, "move_w": 0}

def l(msg):
    log_file.write(msg + "\n")
    log_file.flush()
    print(msg, flush=True)

def read_qword(addr):
    try:
        return int.from_bytes(bytes(gdb.selected_inferior().read_memory(addr, 8)), 'little')
    except:
        return 0

def read_dword(addr):
    try:
        return int.from_bytes(bytes(gdb.selected_inferior().read_memory(addr, 4)), 'little')
    except:
        return 0

# Register storage in gdb convenience variables
# We'll store them in a module-level dict
session = {
    'current_impl_ptr': 0,
    'current_seq_ptr': 0,
    'current_layer': 0,
}


class GenLayerSeqBP_final(gdb.Breakpoint):
    """Track layer sequence and save pointers for other breakpoints"""
    def __init__(self):
        super().__init__("qwen3_npu_sequence::Impl::gen_layer_seq")

    def stop(self):
        counters["gen_layer"] += 1
        n = counters["gen_layer"]
        
        # AMD64: rdi=this, rsi=npu_seq, rdx=layer
        impl_ptr = int(gdb.parse_and_eval("$rdi"))
        seq_ptr = int(gdb.parse_and_eval("$rsi"))
        layer = int(gdb.parse_and_eval("$edx")) & 0xFFFFFFFF
        
        session['current_impl_ptr'] = impl_ptr
        session['current_seq_ptr'] = seq_ptr
        session['current_layer'] = layer
        
        l(f"\n{'#'*60}")
        l(f"gen_layer_seq #{n}: layer={layer}")
        l(f"  Impl=0x{impl_ptr:012x}, seq=0x{seq_ptr:012x}")
        l(f"{'#'*60}")
        
        return False


class MoveWeightsBP_final(gdb.Breakpoint):
    """Capture weight dimensions. rdi=this, rsi=seq, rdx=m, rcx=n, r8=k"""
    def __init__(self):
        super().__init__("qwen3_npu_sequence::Impl::_move_weights")
    
    def stop(self):
        counters["move_w"] += 1
        n = counters["move_w"]
        
        m = int(gdb.parse_and_eval("$rdx"))
        nval = int(gdb.parse_and_eval("$rcx"))
        k = int(gdb.parse_and_eval("$r8"))
        
        # Determine which projection based on m,n,k pattern
        # Group the 4 _move_weights calls per layer
        layer = session.get('current_layer', '?')
        
        proj = ""
        if n == 1024:
            proj = "???"
            if k == 0: proj = "Q_proj"
            elif k == 1310720: proj = "K_proj"
            elif k == 1966080: proj = "V_proj" 
            elif k == 3932160: proj = "O_proj"
        
        l(f"\n  _move_weights #{n}: L{layer} {proj} m={m:6d} (0x{m:x}) n={nval:6d} (0x{nval:x}) k={k:8d} (0x{k:x})")
        l(f"    bytes_per_row_ratio: {m/nval:.1f} (m/n ratio)")
        
        return False


class Cmds2seqBP_final(gdb.Breakpoint):
    """Capture the command list BEFORE cmds2seq compiles it.
    Also trace the return value."""
    def __init__(self):
        super().__init__("npu_sequence::cmds2seq")
    
    def stop(self):
        counters["cmds2seq"] += 1
        n = counters["cmds2seq"]
        
        seq_ptr = int(gdb.parse_and_eval("$rdi"))
        layer = session.get('current_layer', '?')
        
        l(f"\n--- cmds2seq #{n}: layer={layer}, seq=0x{seq_ptr:012x} ---")
        
        # Dump npu_sequence command vector
        inferior = gdb.selected_inferior()
        cmd_begin = read_qword(seq_ptr + 8)
        cmd_end = read_qword(seq_ptr + 16)
        cmd_count = (cmd_end - cmd_begin) // 8
        
        l(f"  cmd_begin=0x{cmd_begin:012x}, cmd_end=0x{cmd_end:012x}, count={cmd_count}")
        
        if cmd_count > 0 and cmd_count < 2000:
            # Save all commands for analysis
            seq_cmds_file.write(f"\n=== cmds2seq #{n} (layer {layer}, {cmd_count} commands) ===\n")
            
            for i in range(min(cmd_count, 1000)):
                cmd_ptr = read_qword(cmd_begin + i*8)
                if cmd_ptr:
                    # Read first 64 bytes of each command
                    try:
                        cmd_data = bytes(inferior.read_memory(cmd_ptr, 64))
                        # Classify command by first DWORD
                        cmd_type = struct.unpack('<I', cmd_data[:4])[0]
                        
                        # Print compact form
                        hex_str = ' '.join(f'{cmd_data[j]:02x}' for j in range(min(32, len(cmd_data))))
                        seq_cmds_file.write(f"  cmd[{i:4d}] ptr=0x{cmd_ptr:012x} type={cmd_type:4d}: {hex_str}\n")
                    except Exception as e:
                        seq_cmds_file.write(f"  cmd[{i:4d}] ptr=0x{cmd_ptr:012x} ERROR: {e}\n")
            
            seq_cmds_file.flush()
            
            # Summary
            l(f"  Saved {min(cmd_count, 1000)} commands to seq_commands file")
        
        # Set up a FinishBreakpoint to capture the return value
        class Cmds2seqFinish(gdb.FinishBreakpoint):
            def stop(self2):
                n2 = counters["cmds2seq"]
                
                # Return value is in $rax - pointer to the returned vector
                ret_vec = int(gdb.parse_and_eval("$rax"))
                
                l(f"\n--- cmds2seq RETURN #{n2} ---")
                l(f"  Returned vector @ 0x{ret_vec:012x}")
                
                inf = gdb.selected_inferior()
                vec_begin = read_qword(ret_vec)
                vec_end = read_qword(ret_vec + 8)
                vec_count = (vec_end - vec_begin) // 4
                
                l(f"  instructions: {vec_count} u32 values")
                
                # Save ALL instructions
                if vec_count > 0 and vec_count < 65536:
                    for j in range(vec_count):
                        val = read_dword(vec_begin + j*4)
                        instr_file.write(struct.pack('<I', val))
                    
                    # Print first 32
                    l(f"  First 32 instructions:")
                    for j in range(min(vec_count, 32)):
                        val = read_dword(vec_begin + j*4)
                        l(f"    [{j:4d}] = 0x{val:08x}")
                    
                    instr_file.flush()
                    l(f"  All {vec_count} instructions saved to gdb_instructions.bin")
                else:
                    l(f"  Suspicious count={vec_count}, skipping")
                
                return False
        
        Cmds2seqFinish()
        return False


class LmHeadSeqBP_final(gdb.Breakpoint):
    """Capture lm_head weight dimensions"""
    def __init__(self):
        super().__init__("qwen3_npu_sequence::Impl::gen_lm_head_seq")
    
    def stop(self):
        impl_ptr = int(gdb.parse_and_eval("$rdi"))
        seq_ptr = int(gdb.parse_and_eval("$rsi"))
        l(f"\n{'#'*60}")
        l(f"gen_lm_head_seq")
        l(f"  Impl=0x{impl_ptr:012x}, seq=0x{seq_ptr:012x}")
        l(f"{'#'*60}")
        
        session['current_impl_ptr'] = impl_ptr
        session['current_seq_ptr'] = seq_ptr
        session['current_layer'] = 'LM_HEAD'
        
        return False


class SendRmsBP_final(gdb.Breakpoint):
    """Just track that RMS weights are sent"""
    def __init__(self):
        super().__init__("qwen3_npu_sequence::Impl::_send_rms_weights")
    
    def stop(self):
        l(f"  _send_rms_weights")
        return False


class SendXBP_final(gdb.Breakpoint):
    """Track input x being sent"""
    def __init__(self):
        super().__init__("qwen3_npu_sequence::Impl::_send_x")
    
    def stop(self):
        l(f"  _send_x")
        return False


# ==========================================================
# Setup
# ==========================================================

def setup():
    gdb.execute("set pagination off")
    gdb.execute("set print thread-events off")
    gdb.execute("set breakpoint pending on")
    gdb.execute("set confirm off")
    gdb.execute("set print symbol-loading off")
    
    l("=" * 60)
    l("GDB Final BO Protocol Capture v4")
    l(f"Capture dir: {CAPTURE_DIR}")
    l("=" * 60)
    
    GenLayerSeqBP_final()
    SendXBP_final()
    SendRmsBP_final()
    MoveWeightsBP_final()
    Cmds2seqBP_final()
    LmHeadSeqBP_final()
    
    l("\nAll breakpoints registered. Running FLM...")
    l("Send a request to trigger captures.\n")

setup()

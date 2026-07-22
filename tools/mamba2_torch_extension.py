"""
mamba2_torch_extension.py — PyTorch extension wrapping Mamba2 HIP kernels

Loads librocm_cpp.so and exposes the Mamba2 selective scan + conv1d
as torch.autograd Functions for training on AMD MI300X.

Usage:
  from mamba2_torch_extension import mamba2_selective_scan, mamba2_conv1d
  y = mamba2_selective_scan(x, dt, A, B, C, D)
"""
import ctypes
import torch
import os

_lib = None

def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    # Search paths for librocm_cpp.so
    paths = [
        "librocm_cpp.so",
        os.path.join(os.path.dirname(__file__), "..", "build", "librocm_cpp.so"),
        os.path.join(os.getcwd(), "build", "librocm_cpp.so"),
    ]
    for p in paths:
        absp = os.path.abspath(p)
        if os.path.exists(absp):
            _lib = ctypes.cdll.LoadLibrary(absp)
            break
    if _lib is None:
        raise RuntimeError(
            "librocm_cpp.so not found. Build it first:\n"
            "  cd 1bit-systems && source env.sh && cmake --build build"
        )
    return _lib

def _setup_functions(lib):
    """Set argument/return types for the C functions."""
    # launch_mamba2_selective_scan(x, dt, A, B, C, D, y, final_state,
    #   B_dim, L, d_inner, d_state, n_head, n_group, head_dim, stream)
    lib.launch_mamba2_selective_scan.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # x, dt, A
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # B, C, D
        ctypes.c_void_p, ctypes.c_void_p,                    # y, final_state
        ctypes.c_int, ctypes.c_int, ctypes.c_int,           # B_dim, L, d_inner
        ctypes.c_int, ctypes.c_int, ctypes.c_int,           # d_state, n_head, n_group
        ctypes.c_int, ctypes.c_void_p,                       # head_dim, stream
    ]
    lib.launch_mamba2_selective_scan.restype = None

    # launch_mamba2_conv1d(x, w, b, y, state, B_dim, L, d_conv, conv_dim, stream)
    lib.launch_mamba2_conv1d.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # x, w, b
        ctypes.c_void_p, ctypes.c_void_p,                    # y, state
        ctypes.c_int, ctypes.c_int, ctypes.c_int,           # B_dim, L, d_conv
        ctypes.c_int, ctypes.c_void_p,                       # conv_dim, stream
    ]
    lib.launch_mamba2_conv1d.restype = None

# ── Selective Scan (no grad for now — inference only) ──
def mamba2_selective_scan(x, dt, A, B, C, D):
    """Mamba2 selective scan.
    Args:
        x: [B, L, d_inner] float32
        dt: [B, L, n_head] float32
        A: [n_head] float32
        B: [B, L, n_group, d_state] float32
        C: [B, L, n_group, d_state] float32
        D: [n_head] float32
    Returns:
        y: [B, L, d_inner] float32
    """
    lib = _load_lib()
    _setup_functions(lib)

    B_dim, L, d_inner = x.shape
    n_head = dt.shape[-1]
    n_group = B.shape[-2]
    d_state = B.shape[-1]
    head_dim = d_inner // n_head

    y = torch.empty_like(x)
    final_state = torch.zeros(B_dim, d_state, d_inner, device=x.device, dtype=torch.float32)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    lib.launch_mamba2_selective_scan(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(dt.data_ptr()),
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_void_p(B.data_ptr()),
        ctypes.c_void_p(C.data_ptr()),
        ctypes.c_void_p(D.data_ptr()),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_void_p(final_state.data_ptr()),
        B_dim, L, d_inner, d_state, n_head, n_group, head_dim,
        stream,
    )
    return y, final_state

def mamba2_conv1d(x, w, b, state):
    """Mamba2 conv1d.
    Args:
        x: [B, L, conv_dim] float32
        w: [d_conv, conv_dim] float32
        b: [conv_dim] float32
        state: [B, d_conv-1, conv_dim] float32 (in-place updated)
    Returns:
        y: [B, L, conv_dim] float32
    """
    lib = _load_lib()
    _setup_functions(lib)

    B_dim, L, conv_dim = x.shape
    d_conv = w.shape[0]

    y = torch.empty_like(x)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    lib.launch_mamba2_conv1d(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(w.data_ptr()),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_void_p(state.data_ptr()),
        B_dim, L, d_conv, conv_dim,
        stream,
    )
    return y

# ── Testing ──
if __name__ == "__main__":
    print("Mamba2 Torch Extension — testing on AMD MI300X")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device: {torch.cuda.get_device_name(0)}")

    # Quick smoke test
    B, L, d_inner, n_head, n_group, d_state = 2, 4, 64, 4, 2, 16
    head_dim = d_inner // n_head

    x = torch.randn(B, L, d_inner, device="cuda", dtype=torch.float32)
    dt = torch.randn(B, L, n_head, device="cuda", dtype=torch.float32)
    A = torch.randn(n_head, device="cuda", dtype=torch.float32)
    B_ = torch.randn(B, L, n_group, d_state, device="cuda", dtype=torch.float32)
    C = torch.randn(B, L, n_group, d_state, device="cuda", dtype=torch.float32)
    D = torch.randn(n_head, device="cuda", dtype=torch.float32)

    try:
        y, state = mamba2_selective_scan(x, dt, A, B_, C, D)
        print(f"  ✓ Selective scan: y.shape={y.shape} state.shape={state.shape}")
        print(f"    y.mean={y.mean().item():.4f} y.std={y.std().item():.4f}")
    except Exception as e:
        print(f"  ✗ Selective scan failed: {e}")

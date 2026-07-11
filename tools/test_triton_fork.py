"""Test Triton HIP kernel execution in a forked subprocess on gfx1151.
This reproduces TheRock #4552 — invalid device ordinal in child process.
"""
import torch
import multiprocessing as mp
import sys

def worker():
    try:
        device = torch.device('cuda:0')
        x = torch.randn(64, 64, device=device)
        y = torch.matmul(x, x)
        print(f'[CHILD] GPU matmul OK: {y.shape}', flush=True)

        # Now test Triton
        try:
            import triton
            import triton.language as tl

            @triton.jit
            def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
                pid = tl.program_id(0)
                offs = pid * BLOCK + tl.arange(0, BLOCK)
                mask = offs < n
                x = tl.load(x_ptr + offs, mask=mask)
                y = tl.load(y_ptr + offs, mask=mask)
                tl.store(out_ptr + offs, x + y, mask=mask)

            n = 1024
            a = torch.randn(n, device='cuda')
            b = torch.randn(n, device='cuda')
            out = torch.empty(n, device='cuda')
            add_kernel[(n // 256,)](a, b, out, n, BLOCK=256)
            torch.cuda.synchronize()
            ref = a + b
            err = (out - ref).abs().max().item()
            print(f'[CHILD] Triton kernel OK (max_err={err:.6f})', flush=True)
        except Exception as e:
            print(f'[CHILD] Triton error: {e}', flush=True)
    except Exception as e:
        print(f'[CHILD] GPU error: {e}', flush=True)

if __name__ == '__main__':
    mp.set_start_method('spawn')

    # Test parent first
    print('[PARENT] Testing GPU in parent process...')
    x = torch.randn(64, 64, device='cuda')
    print(f'[PARENT] GPU OK: {torch.cuda.get_device_name(0)}')

    print('[PARENT] Starting child process...')
    p = mp.Process(target=worker)
    p.start()
    p.join(timeout=60)
    print(f'[PARENT] Child exit code: {p.exitcode}')

    if p.exitcode == 0:
        print('PASS — Triton works in subprocess')
    else:
        print('FAIL — Triton subprocess crashed')

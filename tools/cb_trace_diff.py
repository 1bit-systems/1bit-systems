#!/usr/bin/env python3
"""
Compare NPU engine trace dumps against Python reference traces.

Usage:
    python3 tools/cb_trace_diff.py /tmp/trace_ref/ /tmp/trace_npu/ [--threshold 1e-2]

Output: per-layer, per-tensor cosine similarity + max absolute error.
If any tensor exceeds threshold, prints detailed breakdown.
"""
import argparse, os, sys
from pathlib import Path
import numpy as np

def load_f32(path):
    """Load a .f32 binary file as float32 numpy array."""
    data = np.fromfile(str(path), dtype=np.float32)
    if data.size == 0:
        return None, 0
    return data, data.size

def compare_tensor(ref_path, npu_path, name=""):
    """Compare reference vs NPU binary. Returns (cos_sim, max_abs_err, rmse, ref_norm, npu_norm)."""
    ref, n_ref = load_f32(ref_path)
    npu, n_npu = load_f32(npu_path)
    
    if ref is None or npu is None:
        return None, None, None, None, None, "MISSING"
    if n_ref != n_npu:
        return None, None, None, None, None, f"SIZE_MISMATCH ref={n_ref} npu={n_npu}"
    
    # Convert to float64 for precision
    ref_f64 = ref.astype(np.float64)
    npu_f64 = npu.astype(np.float64)
    
    ref_norm = np.linalg.norm(ref_f64)
    npu_norm = np.linalg.norm(npu_f64)
    
    if ref_norm < 1e-30 and npu_norm < 1e-30:
        return 1.0, 0.0, 0.0, 0.0, 0.0, "OK_ZERO"
    if ref_norm < 1e-30:
        return 0.0, float(np.max(np.abs(npu_f64))), float(np.sqrt(np.mean(npu_f64**2))), ref_norm, npu_norm, "REF_ZERO"
    
    cos_sim = float(np.dot(ref_f64, npu_f64) / (ref_norm * npu_norm))
    max_abs = float(np.max(np.abs(ref_f64 - npu_f64)))
    rmse = float(np.sqrt(np.mean((ref_f64 - npu_f64) ** 2)))
    
    status = "OK" if cos_sim > 0.999 and max_abs < 1e-2 else "DIVERGE" if cos_sim < 0.9 else "CLOSE"
    
    return cos_sim, max_abs, rmse, ref_norm, npu_norm, status


def main():
    ap = argparse.ArgumentParser(description="Compare NPU engine traces vs Python reference")
    ap.add_argument("ref_dir", help="Reference trace directory (from full_ref_forward.py)")
    ap.add_argument("npu_dir", help="NPU trace directory (from npu_engine_universal --trace)")
    ap.add_argument("--threshold", type=float, default=0.99, 
                    help="Cosine similarity threshold for warnings (default: 0.99)")
    ap.add_argument("--layers", default=None, help="Comma-separated layer indices (default: all)")
    ap.add_argument("--verbose", "-v", action="store_true", help="Print first 10 elements on divergence")
    ap.add_argument("--csv", default=None, help="Save results to CSV file")
    args = ap.parse_args()
    
    ref_root = Path(args.ref_dir)
    npu_root = Path(args.npu_dir)
    
    if not ref_root.is_dir():
        print(f"ERROR: reference dir not found: {ref_root}")
        sys.exit(1)
    if not npu_root.is_dir():
        print(f"ERROR: NPU trace dir not found: {npu_root}")
        sys.exit(1)
    
    # Parse layer list
    if args.layers:
        layer_list = [int(x) for x in args.layers.split(",")]
    else:
        # Auto-detect from directory listing
        layer_list = sorted(int(d.name.split("_")[1]) for d in ref_root.iterdir() 
                           if d.is_dir() and d.name.startswith("layer_"))
    
    tensor_names = ["h_in", "h_ln1", "q_flat", "k_flat", "v_flat", 
                    "q_heads", "k_heads", "v_heads",
                    "q_normed", "k_normed", "q_rope", "k_rope",
                    "attn_flat", "attn_proj", "h_after_attn",
                    "h_ln2", "ffn_gate", "ffn_up", "ffn_hidden", "ffn_out", "h_out"]
    
    print(f"{'Layer':<8} {'Tensor':<18} {'Cos Sim':<10} {'Max Abs':<12} {'RMSE':<12} {'|ref|':<10} {'|npu|':<10} {'Status'}")
    print("-" * 90)
    
    all_results = []
    any_diverged = False
    
    for l in layer_list:
        ref_layer = ref_root / f"layer_{l:02d}"
        npu_layer = npu_root / f"layer_{l:02d}"
        
        if not ref_layer.is_dir():
            print(f"  Layer {l}: ref directory not found")
            continue
        if not npu_layer.is_dir():
            print(f"  Layer {l}: NPU directory not found — all tensors MISSING")
            for tn in tensor_names:
                all_results.append((l, tn, 0, 0, 0, 0, 0, "MISSING"))
            continue
        
        for tn in tensor_names:
            ref_path = ref_layer / f"{tn}.f32"
            npu_path = npu_layer / f"{tn}.f32"
            
            if not ref_path.exists():
                continue  # ref doesn't dump this tensor
            
            cos_sim, max_abs, rmse, rn, nn, status = compare_tensor(ref_path, npu_path, tn)
            
            if status is None:
                continue
            
            cos_str = f"{cos_sim:.6f}" if cos_sim is not None else "N/A"
            ma_str = f"{max_abs:.6e}" if max_abs is not None else "N/A"
            rmse_str = f"{rmse:.6e}" if rmse is not None else "N/A"
            rn_str = f"{rn:.4f}" if rn is not None else "N/A"
            nn_str = f"{nn:.4f}" if nn is not None else "N/A"
            
            flag = ""
            if cos_sim is not None and cos_sim < args.threshold:
                flag = " ⚠️"
                any_diverged = True
            
            print(f"  L{l:<4}  {tn:<18} {cos_str:<10} {ma_str:<12} {rmse_str:<12} {rn_str:<10} {nn_str:<10} {status}{flag}")
            all_results.append((l, tn, cos_sim, max_abs, rmse, rn, nn, status))
            
            if args.verbose and flag:
                ref_data, _ = load_f32(ref_path)
                npu_data, _ = load_f32(npu_path)
                print(f"          ref[:10]: {ref_data[:10].tolist()}")
                print(f"          npu[:10]: {npu_data[:10].tolist()}")
    
    # Also compare final norm and logits
    for fname in ["final_norm_output.f32", "logits.f32"]:
        ref_path = ref_root / fname
        npu_path = npu_root / fname
        if ref_path.exists() and npu_path.exists():
            cos_sim, max_abs, rmse, rn, nn, status = compare_tensor(ref_path, npu_path, fname)
            if cos_sim is not None:
                print(f"  {'':<6}  {fname:<18} {cos_sim:.6f}  {max_abs:.6e}  {rmse:.6e}  {rn:.4f}  {nn:.4f}  {status}")
                all_results.append(("final", fname, cos_sim, max_abs, rmse, rn, nn, status))
                if cos_sim < args.threshold:
                    any_diverged = True
    
    # Summary
    print(f"\n{'='*90}")
    n_total = len(all_results)
    n_ok = sum(1 for r in all_results if r[7] == "OK" or r[7] == "OK_ZERO")
    n_close = sum(1 for r in all_results if r[7] == "CLOSE")
    n_diverge = sum(1 for r in all_results if r[7] == "DIVERGE")
    n_missing = sum(1 for r in all_results if r[7] == "MISSING")
    n_size = sum(1 for r in all_results if r[7] == "SIZE_MISMATCH")
    
    print(f"Total: {n_total} tensors")
    print(f"  OK:      {n_ok}")
    print(f"  CLOSE:   {n_close} (cos_sim < {args.threshold} but >= 0.9)")
    print(f"  DIVERGE: {n_diverge} (cos_sim < 0.9)")
    print(f"  MISSING: {n_missing}")
    print(f"  SIZE:    {n_size}")
    
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer", "tensor", "cos_sim", "max_abs", "rmse", "ref_norm", "npu_norm", "status"])
            for r in all_results:
                w.writerow(r)
        print(f"Saved to: {args.csv}")
    
    sys.exit(1 if any_diverged else 0)


if __name__ == "__main__":
    main()

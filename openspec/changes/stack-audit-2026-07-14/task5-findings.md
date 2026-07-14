# Task 5 findings (2026-07-14)

## rocm_hip (113 tok/s, status: reported)

Checked `site/benchmarks.json` — every other entry has a "note" field with
full methodology (hardware, command, run count, what was fixed). `rocm_hip`
has none:

    "rocm_hip": { "tok_s": 113, "status": "reported", "label": "GPU ROCm HIP (reported)" }

Searched `benchmarks/`, `tools/`, `tests/` for a script that produces this
specific number — found ~15 candidate bench_*.cpp/bench-*.sh files
(bench_gemm.cpp, bench_ternary.cpp, run_bench.sh, bench-sweep.sh, etc.) but
none is labeled or referenced as the source of the 113 figure. This is not
"needs re-verification" so much as "the original measurement was never
documented" — there's nothing to re-run yet.

**Real next step for whoever picks this up:** pick one canonical HIP kernel
bench path (probably `tools/run_bench.sh` or `tools/bench-sweep.sh` given
the naming — worth confirming which one actually builds+runs an
end-to-end HIP decode loop rather than a microbenchmark), compile with
`-DCMAKE_HIP_ARCHITECTURES=gfx1151`, run it clean, and write a note field
matching the rigor of the `npu_v12` entry (hardware, run count, what
if anything needed tuning). Budget real time for this — it's a from-scratch
measurement, not a re-run.

## dspark (0.8 tok/s, status: unresolved)

Already has full methodology in its note field (2026-07-11 fixes: checkpoint
wiring bug + oversized global_batch_size, both confirmed fixed, 420-step
training run showed real learning: loss 26.5 -> ~7.5). Re-measuring won't
change the number — the note already states the root cause is insufficient
training data (perplexity ~1800 from 343 examples/420 steps). This one
doesn't need re-benchmarking, it needs a decision: invest in more training
data for the draft head, or deprioritize DSpark. Not an engineering task,
a product-priority call.

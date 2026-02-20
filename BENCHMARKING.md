# Benchmarking for "Speeding Up Sum-Check Proving"

This document explains how to re-run and verify the experiments used in the paper evaluation.

It corresponds to the paper-side files:
- `eval_plan_v2.md`
- `sections/8_eval.tex`

If you keep the paper repo side-by-side with this repo, those files are typically at:
- `../speeding-up-sumcheck/eval_plan_v2.md`
- `../speeding-up-sumcheck/sections/8_eval.tex`

## Scope

This branch includes benchmark harnesses and instrumentation for:
- high-degree kernel microbenchmarks,
- Spartan outer ablations (runtime + memory),
- streaming vs linear-space comparisons,
- degree-2 delayed-reduction micros,
- RA virtualization benchmarking in Shout,
- end-to-end SHA2-chain sweeps.

Notes:
- Exp 1 (`bench-ops`) and Exp 2 (`bench-lin-combo`) are listed in the paper plan, but there is no dedicated standalone harness for those exact tables in this branch.
- All reported measurements were run on Apple M4 Max (12 P-cores + 4 E-cores, 64 GB RAM).

## Prerequisites

- Use the pinned toolchain from `rust-toolchain.toml`.
- Build once before benchmarking:

```bash
cargo build --release -p jolt-core
```

- For memory runs, this guide uses macOS `time`:
  - runtime-only RSS snapshots: `/usr/bin/time -p`
  - peak RSS: `/usr/bin/time -l`

## Results Locations

Primary outputs are written under:
- `benchmark-runs/results/`
- `benchmark-runs/perfetto_traces/`
- `target/criterion/` (Criterion raw outputs)

## Experiment Matrix (matching paper/eval plan)

### Exp 3: High-degree product kernel (`bench-high-d`)

```bash
cargo bench -p jolt-core --bench mles_product_sum
```

Source harness:
- `jolt-core/benches/mles_product_sum.rs`

---

### Exp 4 and Exp 5: Spartan outer ablations + streaming runtime

```bash
cargo bench -p jolt-core --bench spartan_outer --features host
```

This single bench emits all key runtime variants used in the paper:
- `outer-baseline`
- `outer-split-eq`
- `outer-delayed-reduction`
- `outer-round-batched`
- `outer-uni-skip`
- `outer-streaming`
- `outer-streaming-coeff-basis`

Source harness:
- `jolt-core/benches/spartan_outer.rs`

---

### Exp 6: RA virtualization in Shout (`bench-shout`)

Optimized kernel (for each `RA_VIRTUAL_POLYS` in `8 4 2 1`):

```bash
TRACE=1 BENCH_RUNS=1 SCALE=22 RA_VIRTUAL_POLYS=8 \
  cargo run --release -p sha2-chain --bin sha2-chain-bench
```

Naive kernel (same settings, add `NAIVE_RA_KERNEL=1`):

```bash
TRACE=1 BENCH_RUNS=1 SCALE=22 RA_VIRTUAL_POLYS=8 NAIVE_RA_KERNEL=1 \
  cargo run --release -p sha2-chain --bin sha2-chain-bench
```

Convenient sweep:

```bash
for n in 8 4 2 1; do
  TRACE=1 BENCH_RUNS=1 SCALE=22 RA_VIRTUAL_POLYS=$n \
    cargo run --release -p sha2-chain --bin sha2-chain-bench
  TRACE=1 BENCH_RUNS=1 SCALE=22 RA_VIRTUAL_POLYS=$n NAIVE_RA_KERNEL=1 \
    cargo run --release -p sha2-chain --bin sha2-chain-bench
done
```

---

### Exp 7: Degree-2 delayed-reduction micro

```bash
RAYON_NUM_THREADS=1 cargo bench -p jolt-core --bench degree2_sumcheck
```

Source harness:
- `jolt-core/benches/degree2_sumcheck.rs`

---

### Exp 8: Degree-2 + eq-poly delayed-reduction micro

```bash
RAYON_NUM_THREADS=1 cargo bench -p jolt-core --bench degree2_with_eq_sumcheck
```

Source harness:
- `jolt-core/benches/degree2_with_eq_sumcheck.rs`

---

### Exp 9: End-to-end SHA2-chain sweep (`bench-e2e`)

```bash
for s in 22 23 24 25; do
  TRACE=1 BENCH_RUNS=3 SCALE=$s \
    cargo run --release -p sha2-chain --bin sha2-chain-bench
done
```

This is the run shape used for scale 22--25 summary tables.

## Memory Benchmarking Methodology (paper Section 8)

Memory numbers in the paper use three complementary methods.

### 1) Isolated Spartan outer object-level heap (16k/32k)

```bash
/usr/bin/time -p env OUTER_MODE=<streaming|streaming-coeff-basis|outer-uni-skip> \
  SHA2_ITERS=<16000|32000> \
  cargo run --release --features allocative -p jolt-core --bin spartan_outer_memory
```

### 2) Setup-normalized masked peak RSS (streaming sweep 8--8192)

```bash
/usr/bin/time -l env OUTER_MODE=<streaming|streaming-coeff-basis|outer-uni-skip> \
  SHA2_ITERS=<8|16|32|64|128|256|512|1024|2048|4096|8192> \
  cargo run --release --features allocative -p jolt-core --bin spartan_outer_memory
```

Definition used in paper:
- `Δpeak RSS = (maximum resident set size) - (rss_mb_after_setup)`

### 3) Linear-space ablation isolated peak RSS (128--8192)

```bash
/usr/bin/time -l env OUTER_MODE=<outer-baseline|outer-split-eq|outer-delayed-reduction|outer-round-batched|outer-uni-skip> \
  SHA2_ITERS=<128|256|512|1024|2048|4096|8192> \
  cargo run --release --features allocative -p jolt-core --bin spartan_outer_memory
```

### Optional: phase-isolated uni-skip profiling

```bash
PHASE_MEM_PROFILE=1 OUTER_MODE=outer-uni-skip SHA2_ITERS=<128|256|512> \
  cargo run --release -p jolt-core --bin spartan_outer_memory
```

## End-to-end monitor/perfetto memory traces

For process-level peaks in end-to-end runs, use monitor traces:

```bash
cargo run --release --features monitor -p jolt-core benchmark \
  --name sha2-chain --format chrome --scale 23 --target-trace-size 6955008
```

(Repeat for scales of interest; outputs are written to `benchmark-runs/perfetto_traces/`.)

## Verification Checklist

- Use the same machine class (M4 Max, 64 GB) for closest reproduction.
- Keep Rayon thread defaults consistent with paper runs (except Exp 7/8, which are explicitly single-threaded).
- Compare generated numbers against:
  - the paper tables in `sections/8_eval.tex`, and
  - the detailed run ledger in `eval_plan_v2.md`.


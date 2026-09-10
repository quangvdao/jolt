# Artifact for “Speeding Up Sum-Check Proving”

This branch contains the Rust implementations and benchmark harnesses used for
the evaluation in the ACM CCS 2026 paper “Speeding Up Sum-Check Proving.” The
paper's assigned DOI is
[`10.1145/3830454.3846581`](https://doi.org/10.1145/3830454.3846581).

This is a research artifact built on a development snapshot of Jolt. It is
intended for reproducing the paper's experiments, not for production use.

## What is included

- high-degree multilinear-product benchmarks;
- degree-2 sum-check benchmarks with and without an equality polynomial;
- the Spartan outer-sum-check ablation and streaming variants;
- the RA-virtualization integration used in Jolt's Shout lookup argument;
- an end-to-end SHA2-chain benchmark and memory-measurement driver; and
- processed values used in the camera-ready paper under
  [`paper-artifact/results/`](paper-artifact/results/).

The historical raw console logs and profiling traces from the February 2026
measurement campaign are not included. The processed tables are preserved, and
the commands below regenerate new measurements from this revision.

## Tested environment

The reported paper results were collected on a MacBook Pro with an Apple M4 Max
(16 CPU cores: 12 performance and 4 efficiency cores) and 64 GB RAM. The
benchmarks use BN254. Microbenchmarks were either single-threaded with
`RAYON_NUM_THREADS=1` or used the 12 performance cores, as identified below.

The repository pins Rust 1.88 in [`rust-toolchain.toml`](rust-toolchain.toml)
and pins Rust dependencies in `Cargo.lock`. A recent macOS or Linux system with
Git, Rustup, and a C/C++ build toolchain is required. Running the first command
may download the pinned Rust toolchain and dependencies.

## Obtain the artifact

Clone the paper branch and record the revision used for the run:

```bash
git clone --branch ccs-26-benchmarks https://github.com/quangvdao/jolt.git
cd jolt
git rev-parse HEAD
```

Use `--locked` on all Cargo commands so that Cargo does not change the resolved
dependency versions.

## Fast validation

Check that the main Spartan benchmark and memory driver compile:

```bash
cargo check --locked -p jolt-core --bench spartan_outer --bin spartan_outer_memory
```

Then run one small Spartan measurement. The anchors in the filter are
important: without them, Criterion also matches names such as
`sha2-chain-8192`.

```bash
RAYON_NUM_THREADS=1 cargo bench --locked -p jolt-core \
  --bench spartan_outer -- \
  '^Spartan Sumcheck/outer-uni-skip/sha2-chain-8$'
```

Criterion times the prover subprotocol; this microbenchmark does not verify a
complete proof. The end-to-end benchmark below includes verification.

## Reproduce the experiments

Run benchmark commands from the repository root. Criterion writes new output
under `target/criterion/`.

### High-degree product

This reproduces the baseline and optimized rows in the paper's high-degree
product table. The paper used one Rayon thread.

```bash
RAYON_NUM_THREADS=1 cargo bench --locked -p jolt-core --bench mles_product_sum
```

### Degree-2 sum-check

These commands reproduce the delayed-reduction microbenchmarks, without and
with an equality polynomial. The paper used one Rayon thread.

```bash
RAYON_NUM_THREADS=1 cargo bench --locked -p jolt-core --bench degree2_sumcheck
RAYON_NUM_THREADS=1 cargo bench --locked -p jolt-core --bench degree2_with_eq_sumcheck
```

### Spartan outer sum-check

This command runs the complete ablation and streaming suite. The reported paper
run used 12 Rayon threads.

```bash
RAYON_NUM_THREADS=12 cargo bench --locked -p jolt-core --bench spartan_outer
```

To select a particular size safely, use an anchored Criterion filter as shown
in [Fast validation](#fast-validation). Benchmark names and their paper mapping
are listed in [`benchmark-runs/BENCHMARK_GUIDE.md`](benchmark-runs/BENCHMARK_GUIDE.md).

### Spartan memory

The dedicated driver accepts a variant through `OUTER_MODE` and a workload
through `SHA2_ITERS`. For example:

```bash
OUTER_MODE=outer-uni-skip SHA2_ITERS=128 \
  cargo run --locked --release --features allocative \
  -p jolt-core --bin spartan_outer_memory
```

The paper measured process peak RSS externally with `/usr/bin/time -l` on
macOS. On Linux, use `/usr/bin/time -v` and convert the reported maximum RSS
from KiB. The driver also prints setup and phase memory samples.

### RA virtualization and end-to-end proving

The SHA2-chain benchmark verifies each proof and can emit Chrome/Perfetto trace
data. The following is a small end-to-end run; the paper used three runs per
scale and 12 Rayon threads.

```bash
RAYON_NUM_THREADS=12 RUST_LOG=info SCALE=22 BENCH_RUNS=1 TRACE=1 \
  cargo run --locked --release -p sha2-chain --bin sha2-chain-bench
```

Set `RA_VIRTUAL_POLYS` to `8`, `4`, `2`, or `1` to select effective degree 5,
9, 17, or 33. Set `NAIVE_RA_KERNEL=1` to run the baseline kernel. For example:

```bash
RAYON_NUM_THREADS=12 SCALE=22 BENCH_RUNS=1 TRACE=1 \
  RA_VIRTUAL_POLYS=1 NAIVE_RA_KERNEL=1 \
  cargo run --locked --release -p sha2-chain --bin sha2-chain-bench
```

## Resource requirements

Start with the fast validation command. Runtime and memory depend substantially
on the machine, operating system, background load, and Rayon thread count, so a
new run need not reproduce the paper values exactly.

The largest dense Spartan variants are intentionally not quick-start examples.
In the paper's `SHA2_ITERS=8192` measurements, they took up to about 121 seconds
and 50 GB peak RSS. The optimized univariate-skip variant still reached about
11.5 GB peak RSS at that size. Use a smaller size unless the machine has ample
free memory; an out-of-memory termination is possible.

## Result map

| Paper result | Implementation or driver | Processed values |
|---|---|---|
| High-degree product | `jolt-core/benches/mles_product_sum.rs` | `high_degree_product_us.csv` |
| Degree-2 delayed reduction | `jolt-core/benches/degree2_sumcheck.rs` | `degree2_sumcheck_ms.csv` |
| Degree-2 plus equality polynomial | `jolt-core/benches/degree2_with_eq_sumcheck.rs` | `degree2_sumcheck_ms.csv` |
| Spartan runtime ablation | `jolt-core/benches/spartan_outer.rs` | `spartan_runtime_ms.csv` |
| Spartan peak RSS | `jolt-core/src/bin/spartan_outer_memory.rs` | `spartan_peak_rss_gb.csv` |
| Streaming memory tradeoff | `jolt-core/src/bin/spartan_outer_memory.rs` | `spartan_streaming_delta_peak_rss_gb.csv` |
| RA virtualization | `examples/sha2-chain/src/bench.rs` | `ra_virtualization_ms.csv` |
| End-to-end proving | `examples/sha2-chain/src/bench.rs` | `e2e_sha2_chain.csv` |

All processed files are in [`paper-artifact/results/`](paper-artifact/results/).
See that directory's README for units, provenance, and interpretation.

## Known limitations

- Historical raw logs and traces are unavailable in this branch. The CSV files
  contain the processed values reported in the paper, not newly reconstructed
  raw evidence.
- Criterion benchmark timings cover the named prover component and do not by
  themselves establish end-to-end proof correctness.
- Exact performance is machine-dependent. Compare relative behavior only after
  holding the revision, build profile, thread count, and workload constant.
- `scripts/setup_machine.sh` and `scripts/optimize_machine.sh` are legacy Linux
  administration helpers. They are not required and should be reviewed before
  use because they change host-level settings.

## License

This artifact uses the repository's existing dual MIT/Apache-2.0 license. See
[`LICENSE-MIT`](LICENSE-MIT) and [`LICENSE-APACHE`](LICENSE-APACHE).

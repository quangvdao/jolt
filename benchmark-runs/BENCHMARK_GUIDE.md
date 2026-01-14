# Jolt Benchmark Guide

This guide documents the benchmark setup for reproducible performance measurements of Jolt's prover. These benchmarks are used for the paper evaluation.

## Quick Start

```bash
# Scale 24 benchmark (matches profile command, ~15M cycles)
RUST_LOG=info SCALE=24 BENCH_RUNS=3 cargo run --release -p sha2-chain --bin sha2-chain-bench

# Scale 22 benchmark (faster iteration, ~3.4M cycles)
RUST_LOG=info SCALE=22 BENCH_RUNS=5 cargo run --release -p sha2-chain --bin sha2-chain-bench
```

---

## Benchmark Programs

### SHA2-Chain (`sha2-chain`)

Repeatedly hashes a 32-byte input using SHA-256. This is the primary benchmark for measuring prover performance.

**Location:** `examples/sha2-chain/`

**Binary:** `sha2-chain-bench`

---

## Configuration

The benchmark is controlled via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `SCALE` | Target trace size as 2^SCALE (power of 2) | 22 |
| `SHA2_ITERS` | Manual SHA2 iteration count (overrides SCALE) | — |
| `BENCH_RUNS` | Number of proving runs | 3 |
| `RUST_LOG` | Logging level (`info` for detailed output) | — |
| `RA_VIRTUAL_POLYS` | Number of virtual RA polynomials (1, 2, 4, 8); controls RA sumcheck degree | — |
| `HIGH_DEGREE_RA` | Shortcut for `RA_VIRTUAL_POLYS=1` (set to `1` or `true`) | — |
| `NAIVE_RA_KERNEL` | If set, use the O(D²) RA kernel (benchmarking only) | — |

### Scale to Cycles Mapping

The benchmark targets 90% of the padded trace size to avoid exceeding `max_trace_length`.

| Scale | Target Cycles | SHA2 Iters | Padded Size |
|-------|---------------|------------|-------------|
| 20 | ~0.94M | ~277 | 1,048,576 |
| 22 | ~3.77M | ~1,111 | 4,194,304 |
| 24 | ~15.1M | ~4,446 | 16,777,216 |
| 26 | ~60.4M | ~17,784 | 67,108,864 |

**Formula:**
```
target_cycles = 2^SCALE × 0.9
sha2_iters = target_cycles / 3396  (cycles per SHA256)
```

---

## Running Benchmarks

### Basic Usage

```bash
cd /path/to/jolt-spartan

# Run at scale 24 with 3 repetitions
RUST_LOG=info SCALE=24 BENCH_RUNS=3 cargo run --release -p sha2-chain --bin sha2-chain-bench
```

### Example Output

```
=== SHA2-Chain Benchmark ===
  Scale:           2^24 (~15.1M target cycles, 90% of 2^24)
  SHA2 iterations: 4446
  Est. cycles:     15.10M
  Benchmark runs:  3

>>> Preprocessing (one-time)...
>>> Preprocessing done in 5.201 s

--- Run 1/3 ---
  14614620 total cycles
  Proved in 37.3s (391.8 kHz / padded 449.7 kHz)
  Verification: PASSED

--- Run 2/3 ---
  Proved in 35.7s (409.1 kHz)

--- Run 3/3 ---
  Proved in 33.9s (431.2 kHz)

========== Proving Time ==========
  Runs:    3
  Mean:    37.506 s
  Std Dev: 0.930 s
  Min:     36.262 s
  Max:     38.497 s
  All runs: ["38.497s", "37.759s", "36.262s"]

Preprocessing time (one-time): 5.201 s
```

---

## Profiling with Chrome Tracing

For detailed performance analysis with flame graphs:

```bash
cargo run --release -p jolt-core profile --name sha2-chain --format chrome
```

Output: `benchmark-runs/perfetto_traces/sha2_chain_YYYYMMDD-HHMM.json`

View in: [Perfetto UI](https://ui.perfetto.dev/) or `chrome://tracing`

---

## Machine Setup (Optional)

For more consistent benchmarks, consider:

### 1. CPU Frequency Scaling

```bash
# Disable frequency scaling (Linux)
sudo cpupower frequency-set -g performance
```

### 2. Process Priority

```bash
# Run with real-time priority (Linux)
sudo chrt -f 80 cargo run --release -p sha2-chain --bin sha2-chain-bench
```

### 3. Compiler Optimizations

```bash
RUSTFLAGS="-C target-cpu=native -C opt-level=3" cargo run --release -p sha2-chain --bin sha2-chain-bench
```

### 4. Full Optimization Script

See `scripts/optimize_machine.sh` for comprehensive setup.

---

## Experiment Configurations

### Baseline Measurement

Standard configuration for paper results:

```bash
# Scale 24, 5 runs for statistical significance
RUST_LOG=info SCALE=24 BENCH_RUNS=5 cargo run --release -p sha2-chain --bin sha2-chain-bench
```

### Scaling Experiment

Measure performance across different trace sizes:

```bash
for scale in 20 22 24; do
  echo "=== Scale $scale ==="
  RUST_LOG=info SCALE=$scale BENCH_RUNS=3 cargo run --release -p sha2-chain --bin sha2-chain-bench 2>&1 | tee "benchmark-runs/results/sha2_chain_scale_${scale}.log"
done
```

### A/B Comparison

When comparing two implementations:

```bash
# Checkout baseline
git checkout main
SCALE=24 BENCH_RUNS=5 cargo run --release -p sha2-chain --bin sha2-chain-bench 2>&1 | tee baseline.log

# Checkout experiment
git checkout feature-branch
SCALE=24 BENCH_RUNS=5 cargo run --release -p sha2-chain --bin sha2-chain-bench 2>&1 | tee experiment.log
```

---

## Key Metrics

| Metric | Description |
|--------|-------------|
| **Proving Time** | Total time for `prove()` call (excludes preprocessing) |
| **Throughput (kHz)** | `cycles / proving_time` — raw cycles processed per second |
| **Padded Throughput** | `padded_cycles / proving_time` — accounts for power-of-2 padding |
| **Preprocessing Time** | One-time setup (generators, bytecode processing) |

---

## Files Modified

- `examples/sha2-chain/src/bench.rs` — Benchmark harness
- `examples/sha2-chain/guest/src/lib.rs` — `max_trace_length` set to 2^24
- `examples/sha2-chain/Cargo.toml` — Added `sha2-chain-bench` binary

---

## TODO

- [ ] Add memory usage tracking
- [ ] Add proof size measurement
- [ ] Support other benchmarks (fibonacci, btreemap, sha3-chain)
- [ ] CSV output for automated analysis
- [ ] Warmup runs option

---

## References

- Jolt paper: https://eprint.iacr.org/2023/1217
- Internal profiling: `jolt-core/benches/e2e_profiling.rs`

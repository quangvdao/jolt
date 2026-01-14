# RA Virtual (instruction lookups) — Naive vs Optimized Kernel Bench
**Date:** 2026-01-14  
**Benchmark:** `sha2-chain-bench` (host benchmark in `examples/sha2-chain/`)  
**Tracing:** Chrome trace JSON via `tracing-chrome`  
**Analysis:** `scripts/analyze_trace.py`  
**Goal:** quantify (a) impact of RA virtualization degree (via `RA_VIRTUAL_POLYS`) and (b) impact of the RA compute kernel (`NAIVE_RA_KERNEL`) on total proving time and RA Virtual time.

---

## Summary (what this run establishes)

- **Behavior is monotone and sensible:** decreasing the number of virtual RA polynomials (N=8 → 1) increases the RA Virtual sumcheck degree and increases RA time.
- **Naive kernel amplifies the gap:** `NAIVE_RA_KERNEL=1` makes high-degree (N=1) substantially slower than low-degree (N=8), matching the expected \(O(D^2)\) vs \(O(D \log D)\) effect.
- **Root cause of earlier “RA % collapsed to ~3%” confusion:** traces contain multiple spans named `prove` (notably `jolt_core::zkvm::prover::prove` and `dory_pcs::prove`). The analyzer must compute total proving time from the **zkVM prover `prove`** span only; otherwise the denominator gets inflated and percentages become misleading. The analyzer was updated accordingly (see “Trace analysis details”).

---

## Method (reproducible)

### 1) Configuration knobs being tested

#### **RA virtualization (controls degree)**

The one-hot config supports:
- `RA_VIRTUAL_POLYS=N` where \(N \in \{1,2,4,8\}\).  
  This controls the number of *virtual* RA polynomials in the instruction lookup RA sumcheck (and thus the sumcheck degree).  
  See the documentation in `OneHotConfig::new()`:

```178:224:jolt-core/src/zkvm/config.rs
impl OneHotConfig {
    /// Environment variable overrides:
    /// - `RA_VIRTUAL_POLYS=N`: Set number of virtual RA polynomials (1, 2, 4, or 8).
    ///   This controls the sumcheck degree: degree = (LOG_K / N) / log_k_chunk + 1
    ///   - N=8 → 8 × degree-5 (default for small traces)
    ///   - N=4 → 4 × degree-9
    ///   - N=2 → 2 × degree-17
    ///   - N=1 → 1 × degree-33
    /// - `HIGH_DEGREE_RA=1`: Shortcut for N=1 (single high-degree sumcheck)
```

#### **RA compute kernel**

The RA sumcheck prover’s `compute_message` dispatches between optimized kernels and a naive \(O(D^2)\) kernel based on `NAIVE_RA_KERNEL` (see `InstructionRaSumcheckProver::compute_message` in `jolt-core/src/zkvm/instruction_lookups/ra_virtual.rs`).

### 2) Running the benchmark (fresh traces)

All runs used:
- `TRACE=1` to emit a Chrome trace JSON file,
- `BENCH_RUNS=1` (single proving run per trace),
- `SCALE ∈ {20, 21}`,
- `RA_VIRTUAL_POLYS ∈ {8,4,2,1}`,
- kernel ∈ {optimized, `NAIVE_RA_KERNEL=1`}.

Command pattern:

```bash
# from repo root
TRACE=1 BENCH_RUNS=1 SCALE=<20|21> RA_VIRTUAL_POLYS=<8|4|2|1> \
  cargo run --release -p sha2-chain --bin sha2-chain-bench

# naive kernel variant
TRACE=1 BENCH_RUNS=1 SCALE=<20|21> RA_VIRTUAL_POLYS=<8|4|2|1> NAIVE_RA_KERNEL=1 \
  cargo run --release -p sha2-chain --bin sha2-chain-bench
```

### 3) Trace filenames (avoid stale/mixed traces)

The harness now embeds RA config into the trace filename, so traces are self-describing:

```146:163:examples/sha2-chain/src/bench.rs
let timestamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
let ra_virtual_polys = std::env::var("RA_VIRTUAL_POLYS").ok();
let high_degree_ra = std::env::var("HIGH_DEGREE_RA")
    .map(|v| v == "1" || v.to_lowercase() == "true")
    .unwrap_or(false);
let naive_ra_kernel = std::env::var("NAIVE_RA_KERNEL").is_ok();
...
let trace_name = format!("sha2_chain_scale{scale}_{ra_label}_{kernel_label}_{timestamp}");
```

This produces traces like:
- `benchmark-runs/traces/sha2_chain_scale21_raN1_naive_YYYYMMDD-HHMMSS.json`
- `benchmark-runs/traces/sha2_chain_scale20_raN8_opt_YYYYMMDD-HHMMSS.json`

### 4) Trace analysis details (how “RA Virtual” is computed)

The analysis script:
- parses Chrome trace JSON into completed spans (supports `ph: "B"/"E"` and `ph: "X"`),
- filters to Jolt spans (`cat` starts with `jolt_core` or `.file` under `jolt-core/`),
- computes **total proving time** from `jolt_core::zkvm::prover::prove` only (avoids collisions with `dory_pcs::prove`),
- computes “RA Virtual Total” as the sum of:
  - `InstructionRaSumcheckProver::initialize`
  - `InstructionRaSumcheckProver::compute_message`
  - `InstructionRaSumcheckProver::ingest_challenge`

Key implementation points:

```43:82:scripts/analyze_trace.py
def parse_trace(trace_path: str) -> List[Span]:
    ...
    # Match B/E pairs by thread and (name, cat) so nested spans with the same
    # name from different crates (e.g. jolt_core::...::prove vs dory_pcs::prove)
    # don't collide.
```

```167:180:scripts/analyze_trace.py
def get_total_proving_time(spans: List[Span]) -> float:
    ...
    # compute total proving time from the zkVM prover's 'prove' span only
    if s.cat == "jolt_core::zkvm::prover" or (s.file == "jolt-core/src/zkvm/prover.rs"):
        total += s.dur_ms
```

---

## Results (fresh runs)

Numbers below are taken from `scripts/analyze_trace.py` output for each newly generated trace.

### Scale 20

| N (`RA_VIRTUAL_POLYS`) | Kernel | Total prove (ms) | RA Virtual (ms) | RA % of prove |
|---:|:--|---:|---:|---:|
| 8 | opt | 4298.4 | 261.3 | 6.1% |
| 8 | naive | 4431.0 | 363.9 | 8.2% |
| 4 | opt | 4271.2 | 341.4 | 8.0% |
| 4 | naive | 4406.3 | 495.6 | 11.2% |
| 2 | opt | 4574.0 | 539.9 | 11.8% |
| 2 | naive | 5034.8 | 903.3 | 17.9% |
| 1 | opt | 4860.3 | 828.1 | 17.0% |
| 1 | naive | 5787.8 | 1600.8 | 27.7% |

### Scale 21

| N (`RA_VIRTUAL_POLYS`) | Kernel | Total prove (ms) | RA Virtual (ms) | RA % of prove |
|---:|:--|---:|---:|---:|
| 8 | opt | 6792.6 | 498.5 | 7.3% |
| 8 | naive | 6604.5 | 678.8 | 10.3% |
| 4 | opt | 6726.9 | 680.1 | 10.1% |
| 4 | naive | 7174.1 | 1045.8 | 14.6% |
| 2 | opt | 7046.7 | 1042.4 | 14.8% |
| 2 | naive | 7930.7 | 1764.1 | 22.2% |
| 1 | opt | 7428.8 | 1569.7 | 21.1% |
| 1 | naive | 9416.9 | 3343.1 | 35.5% |

---

## Notes / pitfalls (avoid reintroducing the earlier inconsistency)

- **Do not use stale traces**: always generate fresh traces for each config. The trace filenames now encode `(scale, N, kernel)` to make this easy to audit.
- **Use the correct env var name**: the knob is `RA_VIRTUAL_POLYS`, not `RA_VIRTUAL_N_POLYS`.
- **Percentages depend on correct denominator**: total prove time must be taken from the **zkVM prover** `prove` span; other crates emit a span named `prove` too.


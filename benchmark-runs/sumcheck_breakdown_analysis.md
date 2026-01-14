# Sumcheck Breakdown Analysis

This document records the timing breakdown of sumchecks in Jolt's prover, with focus on the RA Virtual sumcheck.

## Methodology

- **Benchmark:** SHA2-Chain
- **Tracing:** Chrome trace format via `tracing-chrome`
- **Analysis:** `scripts/analyze_trace.py`
- **Config:** `OuterStreamingScheduleKind::LinearOnly` (no streaming windows)

Run command:
```bash
TRACE=1 SCALE=<N> BENCH_RUNS=1 cargo run --release -p sha2-chain --bin sha2-chain-bench
python3 scripts/analyze_trace.py benchmark-runs/traces/sha2_chain_scale<N>_*.json
```

---

## Scale Sweep Results (LinearOnly Mode)

**Date:** 2026-01-14  
**Config:** `ONEHOT_CHUNK_THRESHOLD_LOG_T = 26` (uniform config for scales 20-25)

| Scale | Prove Time (s) | All Sumchecks % | RA Virtual % of Prove | RA Virtual % of SC | RA Virtual (ms) |
|------:|---------------:|----------------:|----------------------:|-------------------:|----------------:|
| 20 | 4.79 | 27.7% | 5.5% | 19.9% | 261 |
| 21 | 7.85 | 33.2% | 6.3% | 19.0% | 496 |
| 22 | 12.76 | 38.9% | 6.8% | 17.5% | 864 |
| 23 | 21.93 | 45.0% | 8.3% | 18.4% | 1,831 |
| 24 | 38.42 | 51.2% | 8.8% | 17.2% | 3,376 |
| 25 | 67.18 | 59.1% | 9.7% | 16.4% | 6,501 |

---

## Observations

1. **RA Virtual percentage** ranges from **5.5% to 9.7%** of total proving time (increasing with scale)
2. **All sumchecks** account for **27.7% to 59.1%** of proving time (increasing with scale)
3. **RA Virtual % of sumchecks** is stable at **16-20%** across all scales
4. **RA Virtual absolute time** roughly doubles with each scale increase (as expected for O(n) work)
5. **Sumcheck dominance increases with scale:** At scale 25, sumchecks are 59% of proving time

### RA Virtual Time Scaling

| Scale | RA Virtual (ms) | Ratio to Previous |
|------:|----------------:|------------------:|
| 20 | 261 | — |
| 21 | 496 | 1.90x |
| 22 | 864 | 1.74x |
| 23 | 1,831 | 2.12x |
| 24 | 3,376 | 1.84x |
| 25 | 6,501 | 1.93x |

RA Virtual time scales approximately **2x per scale** (as expected for O(n) work).

---

## Scale 22 Detailed Breakdown

**Total Cycles:** 3,652,475 (~3.65M)  
**Total Proving Time:** 12,755 ms (12.8 s)

### Sumcheck / Prover Component Breakdown

| Component | Time (ms) | % of Total SC | % of Prove |
|-----------|----------:|--------------:|-----------:|
| Instruction Read RAF | 1,413.8 | 25.8% | 11.4% |
| Polynomial Binding | 1,006.2 | 18.4% | 8.1% |
| Spartan Outer | 858.3 | 15.7% | 6.9% |
| **RA Virtual** | **864** | **~15%** | **6.8%** |
| Booleanity | 697.5 | 12.7% | 5.6% |
| Registers RW | 409.8 | 7.5% | 3.3% |
| Matrix Binding | 214.6 | 3.9% | 1.7% |
| RAM RW | 63.8 | 1.2% | 0.5% |

### RA Virtual Sumcheck Details

| Span | Total (ms) | Count | Avg (ms) |
|------|----------:|------:|----------:|
| `initialize` | ~60 | 1 | ~60 |
| `compute_message` | ~600 | 22 | ~27 |
| `ingest_challenge` | ~160 | 22 | ~7 |
| **Total** | **~864** | — | **6.8% of prove** |

---

## Scale Sweep Results (HIGH_DEGREE_RA Mode)

**Date:** 2026-01-14  
**Config:** `HIGH_DEGREE_RA=1` — Single virtual RA polynomial for true high-degree sumcheck (degree 33)

This mode sets `lookups_ra_virtual_log_k_chunk = LOG_K = 128`, so instead of splitting
the RA sumcheck into 8 degree-5 terms, we have a single degree-33 sumcheck.

| Scale | Prove Time (s) | All Sumchecks % | RA Virtual % of Prove | RA Virtual % of SC | RA Virtual (ms) |
|------:|---------------:|----------------:|----------------------:|-------------------:|----------------:|
| 20 | 5.40 | 32.1% | 14.8% | 45.9% | 797 |
| 21 | 7.97 | 40.1% | 18.5% | 46.1% | 1,475 |
| 22 | 14.32 | 46.4% | 19.9% | 42.9% | 2,847 |
| 23 | 23.21 | 51.2% | 23.9% | 46.7% | 5,547 |
| 24 | 42.91 | 57.1% | 25.3% | 44.3% | 10,855 |
| 25 | 76.29 | 67.4% | 27.6% | 40.9% | 21,043 |

### Comparison: Default vs HIGH_DEGREE_RA

| Scale | Default RA % | High-Degree RA % | Slowdown |
|------:|-------------:|-----------------:|---------:|
| 20 | 5.5% | 14.8% | 3.1x (797 vs 261 ms) |
| 21 | 6.3% | 18.5% | 3.0x (1475 vs 496 ms) |
| 22 | 6.8% | 19.9% | 3.3x (2847 vs 864 ms) |
| 23 | 8.3% | 23.9% | 3.0x (5547 vs 1831 ms) |
| 24 | 8.8% | 25.3% | 3.2x (10855 vs 3376 ms) |
| 25 | 9.7% | 27.6% | 3.2x (21043 vs 6501 ms) |

### Key Observations (HIGH_DEGREE_RA)

1. **RA Virtual = 14.8% - 27.6%** of proving time (vs 5.5% - 9.7% default)
2. **RA Virtual = 40.9% - 46.7%** of all sumcheck time (vs 16-20% default)
3. **~3x slowdown** in RA Virtual sumcheck due to higher degree
4. **Sumcheck degree:** 33 (vs 5 in default mode)
5. **Total prove time overhead:** ~10-15% slower overall

---

## Machine Info

- **Date:** 2026-01-14
- **OS:** macOS Darwin 24.6.0
- **Build:** `cargo run --release`
- **Spartan Config:** `OuterStreamingScheduleKind::LinearOnly`
- **RA Config:** Configurable via `HIGH_DEGREE_RA=1` env var

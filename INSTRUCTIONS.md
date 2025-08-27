# Jolt Benchmark Instructions

This document provides instructions for running the benchmarks needed to reproduce the results from our paper "Speeding Up Sumcheck". These benchmarks measure the performance of various field operations, polynomial evaluations, and sumcheck protocols as described in Section 8 (Evaluation).

## Prerequisites & Setup Used

- Rust nightly toolchain (automatically installed via rustup)
- **Hardware**: MacBook Air with 10-core Apple M4 processor and 24GB RAM (as used in paper)
- **Memory**: At least 8GB RAM recommended for larger benchmarks
- **Runtime**: 30-60 minutes depending on your hardware
- **Threading**: Multi-threaded experiments use 10 threads (matching paper setup)

## Individual Benchmark Suites

### 1. Field Operations (`field_ops`)

Benchmarks primitive finite field operations as described in Section 8.1 "The cost of finite field operations":
- **Baselines**: Addition and multiplication operations
- **Small-small multiplication**: u64 × u64 operations
- **Small-big multiplication**: Using JoltField optimizations (our $\sbig$ operations)
- **Power-of-2 multiplication**: Optimized power-of-2 operations
- **Field conversions**: From u64/i64/u128/i128

```bash
cargo bench -p jolt-core --bench field_ops
```

**Expected output**: Performance measurements showing the advantage of prioritizing $\ssmall$ and $\sbig$ multiplications over $\bbig$ multiplications. Results correspond to Table 1 (bench-ops) in the paper.

### 2. Linear Combinations (`linear_combinations`)

Benchmarks linear combination computations with small coefficients as described in Section 8.1:
- **Naive approach**: Using $\bbig$ operations (decompose into additions/multiplications)
- **Small-big approach**: Using our $\sbig$ operations (mul_u64 operations)
- **Single-reduction approach**: Using linear_combination_u64 (our single-reduction algorithm)
- **Signed combinations**: With positive/negative terms
- **Optimized versions**: For 2-term and 3-term combinations

```bash
cargo bench -p jolt-core --bench linear_combinations
```

**Test sizes**: 2, 4, 8, 16, and 32 terms
**Expected output**: Performance comparison showing that times are not easily computable from field operations alone, implying even better savings than microbenchmarks suggest. Results correspond to Table 2 (lin-combo) in the paper.

### 3. Univariate Product (`univariate_product`)

Benchmarks univariate product evaluation for multilinear polynomials:
- **Naive O(D²) accumulate approach**: Traditional polynomial evaluation
- **Fast accumulate using optimized kernels**: Our optimized approach
- **Full evaluation including g(0)**: Complete polynomial evaluation
- **Specialized optimizations**: For small sizes (2, 3 terms)

```bash
cargo bench -p jolt-core --bench univariate_product
```

**Test degrees**: 2, 4, 8, 16, and 32
**Expected output**: Performance scaling with polynomial degree, showing the effectiveness of our optimized kernels.

### 4. Multivariate Product (`multivariate_product`)

Benchmarks multivariate product evaluation as described in Section 8.1 "The cost of finite field operations":
- **Naive approach**: `multivariate_product_evaluations_naive` (extrapolate multilinear polynomials first, then multiply)
- **Optimized approach**: `multi_product_eval` (our new algorithm from Algorithm 1 with and without extrapolation technique)
- **Comprehensive coverage**: Tests v=1,2,3,4 with d=4,8,16,32 combinations
- **Smart filtering**: Automatically skips overly large cases for benchmarking efficiency

```bash
cargo bench -p jolt-core --bench multivariate_product
```

**Test cases**: 16 combinations covering (v=1-4, d=4,8,16,32)
**Expected output**: Performance comparison showing speedups that roughly match those expected from analyzing $\bbig$ operation counts. Results correspond to Table 3 (bench-high-d) in the paper.
**Key benefits**: Eliminates code duplication, provides single source of truth for polynomial generation, easier maintenance.

### 5. RA Sumcheck (`ra_sumcheck`)

Benchmarks the RA (Random Access) sumcheck protocol used in the Shout lookup argument:
- **Round kernel performance**: For different polynomial counts (d = 16 currently)
- **Full prover performance**: Across multiple rounds with multithreading enabled
- **Comparison**: Between optimized and naive implementations
- **End-to-end impact**: On the read-address virtualization sum-check protocol

```bash
cargo bench -p jolt-core --bench ra_sumcheck
```

**Test parameters**: log₂(T) = 18, 20, 22, 24, 26 (where T is the table size)
**Expected output**: Performance scaling showing consistent 2.5× improvement for d=16, jumping to 3× improvement for d=32. Results demonstrate the impact of our high-degree linear-time prover techniques.

## Benchmark Configuration

### Memory Profiling

To enable memory profiling with allocative:

```bash
RUST_LOG=debug cargo bench --release --features allocative -p jolt-core
```

This will generate SVG flamegraphs showing memory usage patterns.

### Custom Parameters

You can modify benchmark parameters in the source files:
- `SAMPLES`: Number of test samples (default: 100)
- `log_ts`: Table sizes for RA sumcheck (default: [18, 20, 22, 24, 26])
- Test sizes for linear combinations and polynomial products

## End-to-End Evaluation

Our techniques are implemented in two key places in Jolt, as described in Section 8.2:

### 1. Spartan Component
- **Purpose**: Sum-check protocol benefiting from our small-value and equality polynomial techniques
- **Implementation**: Requires careful handling of small-value arithmetic to track potential overflows
- **Expected improvement**: Consistent 2-3× improvement, up to 16× improvement for exceptionally large computations
- **Benchmark**: Compare original Spartan implementation vs. modified version using our techniques

### 2. Shout Lookup Argument
- **Purpose**: "Read-address virtualization" sum-check protocol benefiting from our high-degree linear-time prover
- **Current implementation**: d = 16 (extensible to d = 32)
- **Expected improvement**: Consistent 2.5× improvement for d=16, jumping to 3× for d=32
- **Benchmark**: End-to-end prover time with multithreading enabled

## Reproducing Paper Figures

The benchmarks correspond to the performance measurements in Section 8 of our paper "Speeding Up Sumcheck":

1. **Section 8.1**: Field operations, linear combinations, and polynomial products
   - Table 1: Field operation benchmarks
   - Table 2: Linear combination benchmarks
   - Table 3: High-degree polynomial benchmarks

2. **Section 8.2**: End-to-end evaluation in Jolt
   - Figure 1: Spartan benchmark showing up to an order of magnitude improvement in time when memory starts to become a bottleneck
   - Shout results showing 2.5×-3× improvement

To reproduce:
1. Run all benchmarks with default parameters
2. Extract timing data from criterion output
3. Compare with published results in the paper
4. Note any hardware-specific variations (paper uses M4 MacBook Air)

## Additional Resources

- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/)
- [Our Paper: "Speeding Up Sumcheck"](https://eprint.iacr.org/2023/1217) - Section 8 (Evaluation)
- [Performance Profiling Guide](../README.md#performance-profiling)
- **Hardware Reference**: MacBook Air with 10-core Apple M4 processor and 24GB RAM

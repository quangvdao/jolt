//! Benchmarks comparing the Metal stack-VM `Expr` path against the pre-lowered
//! `KernelIR` path on the same Custom kernels.
//!
//! Run: `cargo bench -p jolt-metal --bench kernel_ir_vs_stack_vm`
//!
//! Two axes:
//!
//! 1. **`pairwise_reduce`**: hot-path throughput. Upload inputs once, dispatch
//!    `pairwise_reduce` many times. Measures steady-state GPU compute.
//!
//! 2. **`compile`**: compile-time cost. Rebuilds the kernel per sample.
//!    `stack_vm` walks the `Expr` AST; `kernel_ir` emits from a flat op list.
//!    For large kernels (D=8+) compile dominates a single-shot call, so any
//!    overhead matters.
//!
//! Three kernels:
//!
//! - **booleanity** (`h² − h`): smallest possible Custom kernel. Highlights
//!    interpolation overhead: IR uses `Fma(diff, t, lo)`, stack-VM uses
//!    incremental `cur += diff`. For degree-2 booleanity (grid `{0, 2}`),
//!    the IR path does `num_openings · (degree − 1) = 1` extra `fr_mul` per
//!    pair vs stack-VM's `t = 2` extra `fr_add`s — both are small relative
//!    to the 2 `fr_mul`s of the kernel body, so expect near parity.
//!
//! - **`γ · o0 · o1`** (degree 3): challenge-bound kernel. IR bakes the
//!    challenge as a constant at compile time; stack-VM does the same. Body
//!    has 2 `fr_mul`s regardless of interpolation cost.
//!
//! - **address kernel** (12 inputs, 6 challenges, degree 2): the real
//!    bytecode_read_raf address-phase kernel from `jolt-zkvm`. 12 openings
//!    + 6 challenges means preamble dominates: 24 loads + 6 challenge lits
//!    + 12 diffs + per-slot 12 Fmas. Exercises register pressure.

#![cfg(target_os = "macos")]
#![allow(unused_results)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_compute::{BindingOrder, ComputeBackend};
use jolt_field::{Field, Fr};
use jolt_ir::{lower_custom_expr, Expr, ExprBuilder, KernelDescriptor, KernelShape};
use jolt_metal::MetalBackend;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn random_fr(rng: &mut StdRng, n: usize) -> Vec<Fr> {
    (0..n).map(|_| Fr::random(rng)).collect()
}

const SMALL: usize = 1 << 14;
const LARGE: usize = 1 << 20;

fn fast_config() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(3))
}

struct KernelSpec {
    name: &'static str,
    expr: Expr,
    num_inputs: usize,
    degree: usize,
    challenges: Vec<Fr>,
}

fn make_booleanity() -> KernelSpec {
    let b = ExprBuilder::new();
    let h = b.opening(0);
    KernelSpec {
        name: "booleanity",
        expr: b.build(h * h - h),
        num_inputs: 1,
        degree: 2,
        challenges: vec![],
    }
}

fn make_gamma_product(rng: &mut StdRng) -> KernelSpec {
    let b = ExprBuilder::new();
    let a = b.opening(0);
    let bv = b.opening(1);
    let gamma = b.challenge(0);
    KernelSpec {
        name: "gamma_o0_o1",
        expr: b.build(gamma * a * bv),
        num_inputs: 2,
        degree: 3,
        challenges: vec![Fr::random(rng)],
    }
}

fn make_address_kernel(rng: &mut StdRng) -> KernelSpec {
    const N_STAGES: usize = 5;
    let num_inputs = 2 * N_STAGES + 2;
    let b = ExprBuilder::new();

    let mut sum = b.challenge(0) * b.opening(0) * b.opening(N_STAGES as u32);
    for s in 1..N_STAGES {
        sum = sum + b.challenge(s as u32) * b.opening(s as u32) * b.opening((N_STAGES + s) as u32);
    }
    let trace_idx = (2 * N_STAGES) as u32;
    let expected_idx = (2 * N_STAGES + 1) as u32;
    sum = sum + b.challenge(N_STAGES as u32) * b.opening(trace_idx) * b.opening(expected_idx);

    KernelSpec {
        name: "address_kernel",
        expr: b.build(sum),
        num_inputs,
        degree: 2,
        challenges: (0..=N_STAGES).map(|_| Fr::random(rng)).collect(),
    }
}

fn bench_pairwise_reduce(c: &mut Criterion) {
    let metal = MetalBackend::new();
    let mut setup_rng = StdRng::seed_from_u64(0xB0000001);
    let specs = [
        make_booleanity(),
        make_gamma_product(&mut setup_rng),
        make_address_kernel(&mut setup_rng),
    ];

    let mut group = c.benchmark_group("pairwise_reduce_custom");

    for spec in &specs {
        let desc = KernelDescriptor {
            shape: KernelShape::Custom {
                expr: spec.expr.clone(),
                num_inputs: spec.num_inputs,
            },
            degree: spec.degree,
            tensor_split: None,
        };

        let stack_vm = metal.compile_kernel_with_challenges::<Fr>(&desc, &spec.challenges);
        let ir = lower_custom_expr(
            &spec.expr,
            spec.num_inputs,
            spec.degree,
            jolt_ir::BindingOrder::LowToHigh,
        );
        let ir_k = metal.compile_kernel_ir::<Fr>(&ir, &spec.challenges);

        for &n in &[SMALL, LARGE] {
            let mut rng = StdRng::seed_from_u64(0xDE01 + n as u64);
            let inputs: Vec<Vec<Fr>> = (0..spec.num_inputs)
                .map(|_| random_fr(&mut rng, n))
                .collect();
            let weights = random_fr(&mut rng, n / 2);

            let mtl_bufs: Vec<_> = inputs.iter().map(|v| metal.upload(v)).collect();
            let mtl_refs: Vec<_> = mtl_bufs.iter().collect();
            let mtl_w = metal.upload(&weights);

            let label = format!("{}/2^{}", spec.name, n.trailing_zeros());
            group.throughput(Throughput::Elements((n / 2) as u64));

            group.bench_with_input(
                BenchmarkId::new(format!("stack_vm/{label}"), n),
                &n,
                |bench, _| {
                    bench.iter(|| {
                        metal.pairwise_reduce(
                            &mtl_refs,
                            &mtl_w,
                            &stack_vm,
                            desc.num_evals(),
                            BindingOrder::LowToHigh,
                        )
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new(format!("kernel_ir/{label}"), n),
                &n,
                |bench, _| {
                    bench.iter(|| {
                        metal.pairwise_reduce(
                            &mtl_refs,
                            &mtl_w,
                            &ir_k,
                            desc.num_evals(),
                            BindingOrder::LowToHigh,
                        )
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_compile(c: &mut Criterion) {
    let metal = MetalBackend::new_fast_compile();
    let mut setup_rng = StdRng::seed_from_u64(0xB0000002);
    let specs = [
        make_booleanity(),
        make_gamma_product(&mut setup_rng),
        make_address_kernel(&mut setup_rng),
    ];

    let mut group = c.benchmark_group("compile_custom");

    for spec in &specs {
        let desc = KernelDescriptor {
            shape: KernelShape::Custom {
                expr: spec.expr.clone(),
                num_inputs: spec.num_inputs,
            },
            degree: spec.degree,
            tensor_split: None,
        };
        let ir = lower_custom_expr(
            &spec.expr,
            spec.num_inputs,
            spec.degree,
            jolt_ir::BindingOrder::LowToHigh,
        );

        group.bench_with_input(
            BenchmarkId::new("stack_vm", spec.name),
            spec.name,
            |bench, _| {
                bench.iter(|| metal.compile_kernel_with_challenges::<Fr>(&desc, &spec.challenges));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("kernel_ir", spec.name),
            spec.name,
            |bench, _| {
                bench.iter(|| metal.compile_kernel_ir::<Fr>(&ir, &spec.challenges));
            },
        );
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = fast_config();
    targets = bench_pairwise_reduce, bench_compile,
}
criterion_main!(benches);

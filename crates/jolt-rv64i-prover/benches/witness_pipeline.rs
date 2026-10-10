//! Trace-to-witness measurements on the executed adapter fixture.
//! Run: `cargo bench -p jolt-rv64i-prover --features test-utils
//! --bench witness_pipeline -- --log-t 20,22 --threads 1,12 --samples 5`.
//! `inventory` lists requests of at least T bytes and fails on recorder overflow.
//! Validation-only is diagnostic: preparation validates again while gathering
//! groups. The production total excludes validation-only. Scatter is lazy in
//! the session and timed separately, with preparation-plus-scatter also reported.
//! Setup, statement admission, warmed pools, and destruction are outside timers.

pub mod support;

use jolt_rv64i_kernels::{router::cycle::RoutersCycleCore, source::ValidatedTrace};
use jolt_rv64i_prover::{
    optimized::{
        routers::router_shapes,
        source::{SharedSource, WitnessColumns, WitnessSource},
    },
    plane::Rv64iWitness,
};
use rayon::ThreadPoolBuilder;
use std::{
    error::Error,
    hint::black_box,
    sync::Arc,
    time::{Duration, Instant},
};
use support::{
    allocator::{AllocationMeasurement, AllocationStats, CountingAllocator},
    runner::{summary, Options},
    witness::TraceFixture,
};

type BenchResult<T> = Result<T, Box<dyn Error>>;
const PHASES: [&str; 5] = ["adapt", "construct", "validate", "prepare", "scatter"];

struct Measurement {
    elapsed: Duration,
    allocations: AllocationStats,
}

#[expect(
    clippy::print_stdout,
    reason = "allocation inventory is benchmark output"
)]
fn measure<T>(
    phase: usize,
    cycles: usize,
    inventory: bool,
    f: impl FnOnce() -> BenchResult<T>,
) -> BenchResult<(T, Measurement)> {
    CountingAllocator::begin(cycles);
    CountingAllocator::phase(phase);
    let allocations = AllocationMeasurement::begin();
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed();
    let allocations = allocations.finish();
    CountingAllocator::stop();
    if CountingAllocator::overflow() != 0 {
        return Err("allocation recorder overflow".into());
    }
    if inventory {
        for (bytes, phase) in CountingAllocator::entries() {
            println!(
                "witness_allocation phase={} requested_bytes={bytes}",
                PHASES[phase]
            );
        }
    }
    Ok((
        result?,
        Measurement {
            elapsed,
            allocations,
        },
    ))
}

#[expect(
    clippy::print_stdout,
    reason = "phase distributions are benchmark output"
)]
fn main() -> BenchResult<()> {
    let options = Options::parse_witness()?;
    println!("witness_note fixture=executed_adapter_program tracing=independent_interpreter validation=standalone_nonadditive prepare=fused_validation_and_groups scatter=lazy_session_plan peak=requested_bytes_above_phase_baseline setup_and_drop=untimed loaded_machine=true");
    for &log_t in &options.log_t {
        if !options
            .threads
            .iter()
            .any(|&threads| options.selected_witness(log_t, threads))
        {
            continue;
        }
        let fixture = TraceFixture::new(log_t)?;
        let cycles = fixture.rows.len();
        for &threads in &options.threads {
            if !options.selected_witness(log_t, threads) {
                continue;
            }
            let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
            let _ = pool.broadcast(|_| black_box(()));
            let mut samples = Vec::with_capacity(options.samples);
            for _ in 0..options.samples {
                let sample = pool
                    .install(|| -> Result<_, String> {
                        let run = || -> BenchResult<_> {
                            let (execution, adapt) =
                                measure(0, cycles, options.inventory, || fixture.adapt())?;
                            let checked = fixture.checked(&execution)?;
                            let layout = checked.layout().clone();
                            let bytecode = Arc::clone(fixture.preprocessing.shared_bytecode());
                            let initial_ram = checked.initial_ram().to_vec();
                            let (witness, construct) =
                                measure(1, cycles, options.inventory, || {
                                    Ok(Rv64iWitness::from_facts(
                                        layout,
                                        bytecode,
                                        &execution.facts,
                                        initial_ram,
                                    )?)
                                })?;
                            drop(execution);
                            let columns = WitnessColumns::new(&witness.layout);
                            let shapes = router_shapes(&columns, &witness.layout, None)?;
                            let selectors = RoutersCycleCore::columns(&shapes);
                            let mut shared = SharedSource::default();
                            let (prepared, prepare) =
                                measure(3, cycles, options.inventory, || {
                                    Ok(shared.prepare(&witness, Some(selectors))?)
                                })?;
                            let (plan, scatter) =
                                measure(4, cycles, options.inventory, || Ok(shared.plan()?))?;
                            let _ = black_box((&witness, &prepared, &plan));
                            let source = Arc::new(WitnessSource::new(&witness)?);
                            let (validated, validate) =
                                measure(2, cycles, options.inventory, || {
                                    Ok(ValidatedTrace::new(source)?)
                                })?;
                            drop(validated);
                            Ok([adapt, construct, validate, prepare, scatter])
                        };
                        run().map_err(|error| error.to_string())
                    })
                    .map_err(|error| -> Box<dyn Error> { error.into() })?;
                samples.push(sample);
            }
            for (phase, name) in PHASES.iter().enumerate() {
                let (ns, min, max) = summary(
                    samples
                        .iter()
                        .map(|s| s[phase].elapsed.as_nanos() as f64 / cycles as f64)
                        .collect(),
                );
                let peak = samples
                    .iter()
                    .map(|s| s[phase].allocations.peak_bytes)
                    .max()
                    .unwrap_or(0);
                let final_bytes = samples
                    .iter()
                    .map(|s| s[phase].allocations.final_bytes)
                    .max()
                    .unwrap_or(0);
                let allocs = samples
                    .iter()
                    .map(|s| s[phase].allocations.allocs)
                    .max()
                    .unwrap_or(0);
                println!("witness_pipeline/{name}/{log_t}/{threads} samples={} ns_per_cycle={ns:.6} ms={:.6} min_ns={min:.6} max_ns={max:.6} peak_bytes={peak} final_bytes={final_bytes} allocs={allocs} loaded_machine=true {}", options.samples, ns * cycles as f64 / 1e6, fixture.mix);
            }
            for (name, phases) in [
                ("preparation", &[3, 4][..]),
                ("production_total", &[0, 1, 3, 4][..]),
            ] {
                let (ns, min, max) = summary(
                    samples
                        .iter()
                        .map(|s| {
                            phases
                                .iter()
                                .map(|&p| s[p].elapsed.as_nanos() as f64)
                                .sum::<f64>()
                                / cycles as f64
                        })
                        .collect(),
                );
                println!("witness_pipeline/{name}/{log_t}/{threads} samples={} ns_per_cycle={ns:.6} ms={:.6} min_ns={min:.6} max_ns={max:.6}", options.samples, ns * cycles as f64 / 1e6);
            }
        }
    }
    Ok(())
}

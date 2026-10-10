//! Run a core with `cargo bench -p jolt-rv64i-kernels --features test-utils
//! --bench example -- --log-t 20,22 --threads 1,12`.
//!
//! The constructor includes prerequisite passes and returns the input claim.
//! Rounds use fixed seeded challenges, finish binds the final challenge, and
//! extraction includes subsequent passes. Source generation and pool creation
//! precede measurement. Times are wall-clock nanoseconds per input cycle;
//! allocator bytes and counts cover the four phases, excluding resident sources.

mod allocator;
pub mod arithmetic;
pub mod word;
pub mod example;

use std::error::Error as StdError;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use jolt_field::{Field, F128};
use jolt_rv64i_kernels::source::CycleSource;
use jolt_rv64i_kernels::synth::{SynthProfile, SyntheticTrace};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rayon::{ThreadPool, ThreadPoolBuilder};
use thiserror::Error;

use self::allocator::{AllocationMeasurement, AllocationStats};

#[derive(Debug, Error)]
pub enum RunnerError {
    #[error("missing value for benchmark option {option}")]
    MissingValue { option: String },
    #[error("invalid value {value} for benchmark option {option}")]
    InvalidValue { option: String, value: String },
    #[error("unknown benchmark argument {argument}")]
    UnknownArgument { argument: String },
    #[error("thread pool construction failed: {message}")]
    ThreadPool { message: String },
    #[error("synthetic trace construction failed: {message}")]
    Trace { message: String },
    #[error("core construction or extraction failed: {message}")]
    Core { message: String },
    #[error("core requires {rounds} rounds, exceeding the seeded challenge capacity {capacity}")]
    ChallengeCount { rounds: usize, capacity: usize },
    #[error("probe case {variant} declares zero measured operations")]
    WorkCount { variant: String },
    #[error(transparent)]
    Sumcheck(#[from] SumcheckError<F128>),
}

struct Options {
    log_t: Vec<usize>,
    threads: Vec<usize>,
    samples: usize,
    units: Vec<String>,
}

impl Options {
    fn parse(probe: bool) -> Result<Self, RunnerError> {
        let mut options = Self {
            log_t: if probe { vec![22] } else { vec![20, 22] },
            threads: vec![1, 12],
            samples: if probe { 5 } else { 1 },
            units: Vec::new(),
        };
        let mut arguments = std::env::args().skip(1);
        while let Some(argument) = arguments.next() {
            if argument == "--bench" {
                continue;
            }
            if !(["--log-t", "--threads", "--samples"].contains(&argument.as_str())
                || probe && argument == "--units")
            {
                return Err(RunnerError::UnknownArgument { argument });
            }
            let value = arguments.next().ok_or_else(|| RunnerError::MissingValue {
                option: argument.clone(),
            })?;
            if argument == "--units" {
                options.units = value.split(',').map(str::to_owned).collect();
                if options.units == ["all"] {
                    options.units.clear();
                }
                continue;
            }
            let mut parsed = Vec::new();
            for part in value.split(',') {
                let entry = part
                    .parse::<usize>()
                    .map_err(|_| RunnerError::InvalidValue {
                        option: argument.clone(),
                        value: value.clone(),
                    })?;
                if (argument != "--log-t" && entry == 0)
                    || (argument == "--log-t" && !(1..=32).contains(&entry))
                    || parsed.contains(&entry)
                {
                    return Err(RunnerError::InvalidValue {
                        option: argument,
                        value,
                    });
                }
                parsed.push(entry);
            }
            match argument.as_str() {
                "--log-t" => options.log_t = parsed,
                "--threads" => options.threads = parsed,
                "--samples" if parsed.len() == 1 => options.samples = parsed[0],
                _ => {
                    return Err(RunnerError::InvalidValue {
                        option: argument,
                        value,
                    });
                }
            }
        }
        Ok(options)
    }
}

struct Summary {
    median: f64,
    min: f64,
    max: f64,
}

impl Summary {
    fn new(mut values: Vec<f64>) -> Self {
        values.sort_by(f64::total_cmp);
        let middle = values.len() / 2;
        let median = if values.len().is_multiple_of(2) {
            values[middle - 1].midpoint(values[middle])
        } else {
            values[middle]
        };
        Self {
            median,
            min: values[0],
            max: values[values.len() - 1],
        }
    }
}

struct Sample {
    times: [f64; 4],
    allocation: AllocationStats,
}

impl Sample {
    fn phase(samples: &[Self], phase: usize, divisor: f64) -> Summary {
        Summary::new(samples.iter().map(|s| s.times[phase] / divisor).collect())
    }

    fn total(samples: &[Self], divisor: f64) -> Summary {
        Summary::new(
            samples
                .iter()
                .map(|s| s.times.iter().sum::<f64>() / divisor)
                .collect(),
        )
    }

    fn allocations(samples: &[Self]) -> (usize, usize, usize) {
        (
            samples
                .iter()
                .map(|s| s.allocation.peak_bytes)
                .max()
                .unwrap_or(0),
            samples
                .iter()
                .map(|s| s.allocation.final_bytes)
                .max()
                .unwrap_or(0),
            samples
                .iter()
                .map(|s| s.allocation.allocs)
                .max()
                .unwrap_or(0),
        )
    }
}

fn warmed_pools(threads: &[usize]) -> Result<Vec<(usize, ThreadPool)>, RunnerError> {
    let mut pools = Vec::with_capacity(threads.len());
    for &threads in threads {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|error| RunnerError::ThreadPool {
                message: error.to_string(),
            })?;
        let _ = pool.broadcast(|_| black_box(()));
        pools.push((threads, pool));
    }
    Ok(pools)
}

/// Measures one constructor and extraction callback over each profile and CLI size.
/// The extraction receives the bound challenge point in low-variable-first order.
#[expect(
    clippy::print_stdout,
    reason = "benchmark records are the runner's output contract"
)]
pub fn run_core<C, E, O>(
    bench: &str,
    profiles: &[SynthProfile],
    construct: impl Fn(Arc<SyntheticTrace>) -> Result<(C, F128), E> + Sync,
    extract: impl Fn(&C, &[F128]) -> Result<O, E> + Sync,
) -> Result<(), RunnerError>
where
    C: ProveRounds<F128> + Send,
    E: StdError,
{
    let options = Options::parse(false)?;
    let pools = warmed_pools(&options.threads)?;
    // Pool drop does not join workers; retain every pool through the last record
    // so worker teardown cannot change a subsequent allocator baseline.
    for (threads, pool) in &pools {
        for &profile in profiles {
            for &log_t in &options.log_t {
                let source = Arc::new(
                    pool.install(|| SyntheticTrace::new(profile, log_t, 1 << 20, 0x5eed))
                        .map_err(|error| RunnerError::Trace {
                            message: error.to_string(),
                        })?,
                );
                let cycles = source.cycles();
                let mut rng = ChaCha20Rng::seed_from_u64(0x726f_756e_6473);
                let challenges: [F128; 256] = std::array::from_fn(|_| F128::random(&mut rng));
                let mut samples = Vec::with_capacity(options.samples);
                for _ in 0..options.samples {
                    let sample = pool.install(|| {
                        let measurement = AllocationMeasurement::begin();
                        let start = Instant::now();
                        let (mut core, mut claim) =
                            construct(Arc::clone(&source)).map_err(|error| RunnerError::Core {
                                message: error.to_string(),
                            })?;
                        let construct_ns = start.elapsed().as_nanos() as f64;
                        let rounds = core.num_rounds();
                        let point =
                            challenges
                                .get(..rounds)
                                .ok_or(RunnerError::ChallengeCount {
                                    rounds,
                                    capacity: challenges.len(),
                                })?;
                        let start = Instant::now();
                        let mut bind = None;
                        for (round, &challenge) in point.iter().enumerate() {
                            let message = core.prove_round(bind, round, claim)?;
                            claim = message.evaluate(challenge);
                            bind = Some(challenge);
                            let _ = black_box(&message);
                        }
                        let rounds_ns = start.elapsed().as_nanos() as f64;
                        let start = Instant::now();
                        if let Some(challenge) = bind {
                            core.finish_rounds(challenge)?;
                        }
                        let finish_ns = start.elapsed().as_nanos() as f64;
                        let start = Instant::now();
                        let values = extract(&core, point).map_err(|error| RunnerError::Core {
                            message: error.to_string(),
                        })?;
                        let _ = black_box(&values);
                        let extract_ns = start.elapsed().as_nanos() as f64;
                        let allocation = measurement.finish();
                        Ok::<_, RunnerError>(Sample {
                            times: [construct_ns, rounds_ns, finish_ns, extract_ns],
                            allocation,
                        })
                    })?;
                    samples.push(sample);
                }
                let divisor = cycles as f64;
                let phases: [Summary; 4] =
                    std::array::from_fn(|phase| Sample::phase(&samples, phase, divisor));
                let total = Sample::total(&samples, divisor);
                let (peak_bytes, final_bytes, allocs) = Sample::allocations(&samples);
                print!(
                    "{bench}/{}/{log_t}/{threads} construct_ns={:.6} rounds_ns={:.6} finish_ns={:.6} extract_ns={:.6} total_ns={:.6} peak_bytes={} final_bytes={} allocs={}",
                    profile.name(),
                    phases[0].median,
                    phases[1].median,
                    phases[2].median,
                    phases[3].median,
                    total.median,
                    peak_bytes,
                    final_bytes,
                    allocs,
                );
                if options.samples > 1 {
                    print!(
                        " samples={} total_min_ns={:.6} total_max_ns={:.6}",
                        options.samples, total.min, total.max
                    );
                    for (phase, summary) in ["construct", "rounds", "finish", "extract"]
                        .iter()
                        .zip(&phases)
                    {
                        print!(
                            " {phase}_min_ns={:.6} {phase}_max_ns={:.6}",
                            summary.min, summary.max
                        );
                    }
                }
                println!();
            }
        }
    }
    Ok(())
}

/// One operation mix and memory layout in the probe's configuration matrix.
/// Empty profiles identify source-independent, fixed-size records.
pub struct ProbeCase {
    pub unit: &'static str,
    pub variant: String,
    pub profiles: &'static [SynthProfile],
    pub minimum_threads: usize,
}

/// A preallocated operation stream. The runner times each pass once per sample.
/// `operations` gives the primary and optional auxiliary operation counts;
/// `finish` is used for separately measured accumulator reductions.
pub trait ProbeKernel: Send {
    fn operations(&self) -> [usize; 2];
    fn run(&mut self) -> F128;
    fn finish(&mut self) -> F128 {
        F128::from_raw(0)
    }
    fn chain_terms(&self) -> Option<usize> {
        None
    }
    fn lookup_layout(&self) -> Option<(usize, usize)> {
        None
    }
}

/// One measured pass, with its full record id.
pub struct ProbeRecord {
    pub id: String,
    pub median: f64,
}

/// Runs unit probes with warmed pools and resident sources excluded from timing.
/// Each sample constructs fresh state under the counting allocator. `ns` is
/// median primary wall time per operation; min/max bound the observed samples.
/// Chain records also print nanoseconds per chain and per term; trace-driven
/// fused-chain totals include their operand preparation.
/// Allocation fields are maxima across samples and include construction; final
/// bytes count the state still resident after its measured passes. Probe defaults
/// are log_t=22, threads=1,12 and samples=5; `--units` selects case groups.
/// Fixed-size records run once per thread count under independent/fixed, without
/// generating a trace. A source is built only when selected cases consume it.
#[expect(
    clippy::print_stdout,
    reason = "unit records are the probe output contract"
)]
pub fn run_probe<C, E>(
    cases: &[ProbeCase],
    construct: impl Fn(&ProbeCase, Option<Arc<SyntheticTrace>>, usize) -> Result<C, E> + Sync,
) -> Result<Vec<ProbeRecord>, RunnerError>
where
    C: ProbeKernel,
    E: StdError,
{
    let options = Options::parse(true)?;
    for unit in &options.units {
        if !cases.iter().any(|case| case.unit == unit) {
            return Err(RunnerError::InvalidValue {
                option: "--units".to_owned(),
                value: unit.clone(),
            });
        }
    }
    let pools = warmed_pools(&options.threads)?;
    let mut records = Vec::new();
    let selected = |case: &&ProbeCase| {
        options.units.is_empty() || options.units.iter().any(|unit| unit == case.unit)
    };
    for (threads, pool) in &pools {
        let mut settings = vec![None];
        for profile in [SynthProfile::Local, SynthProfile::AllRows] {
            for &log_t in &options.log_t {
                settings.push(Some((profile, log_t)));
            }
        }
        for setting in settings {
            let selected_cases: Vec<_> = cases
                .iter()
                .filter(selected)
                .filter(|case| *threads >= case.minimum_threads)
                .filter(|case| {
                    setting.map_or(case.profiles.is_empty(), |(profile, _)| {
                        case.profiles.contains(&profile)
                    })
                })
                .collect();
            if selected_cases.is_empty() {
                continue;
            }
            let source = setting
                .map(|(profile, log_t)| {
                    pool.install(|| SyntheticTrace::new(profile, log_t, 1 << 20, 0x5eed))
                        .map(Arc::new)
                        .map_err(|error| RunnerError::Trace {
                            message: error.to_string(),
                        })
                })
                .transpose()?;
            let profile_name = setting.map_or("independent", |(profile, _)| profile.name());
            let log_t = setting.map_or_else(|| "fixed".to_owned(), |(_, log_t)| log_t.to_string());
            for case in selected_cases {
                let mut samples = Vec::with_capacity(options.samples);
                let mut operations = [0; 2];
                let mut chain_terms = None;
                let mut lookup_layout = None;
                for _ in 0..options.samples {
                    let (sample, (counts, terms, layout)) = pool.install(|| {
                        let measurement = AllocationMeasurement::begin();
                        let start = Instant::now();
                        let mut kernel =
                            construct(case, source.clone(), *threads).map_err(|error| {
                                RunnerError::Core {
                                    message: error.to_string(),
                                }
                            })?;
                        let construct_ns = start.elapsed().as_nanos() as f64;
                        let counts = kernel.operations();
                        let terms = kernel.chain_terms();
                        let layout = kernel.lookup_layout();
                        if counts[0] == 0 {
                            return Err(RunnerError::WorkCount {
                                variant: case.variant.clone(),
                            });
                        }
                        let start = Instant::now();
                        let _ = black_box(kernel.run());
                        let primary_ns = start.elapsed().as_nanos() as f64;
                        let auxiliary_ns = if counts[1] == 0 {
                            0.0
                        } else {
                            let start = Instant::now();
                            let _ = black_box(kernel.finish());
                            start.elapsed().as_nanos() as f64
                        };
                        let allocation = measurement.finish();
                        Ok::<_, RunnerError>((
                            Sample {
                                times: [construct_ns, primary_ns, auxiliary_ns, 0.0],
                                allocation,
                            },
                            (counts, terms, layout),
                        ))
                    })?;
                    samples.push(sample);
                    operations = counts;
                    chain_terms = terms;
                    lookup_layout = layout;
                }
                let primary = Sample::phase(&samples, 1, operations[0] as f64);
                let (peak_bytes, final_bytes, allocs) = Sample::allocations(&samples);
                let id = format!(
                    "probe/{}/{}/{profile_name}/{log_t}/{threads}",
                    case.unit, case.variant
                );
                print!(
                        "{id} ns={:.6} min_ns={:.6} max_ns={:.6} samples={} peak_bytes={} final_bytes={} allocs={}",
                        primary.median,
                        primary.min,
                        primary.max,
                        options.samples,
                        peak_bytes,
                        final_bytes,
                        allocs
                    );
                if let Some((bytes, entries)) = lookup_layout {
                    print!(" allocated_bytes={bytes} addressable_entries={entries}");
                }
                if operations[1] != 0 {
                    let auxiliary = Sample::phase(&samples, 2, operations[1] as f64);
                    print!(
                        " reduce_ns={:.6} reduce_min_ns={:.6} reduce_max_ns={:.6}",
                        auxiliary.median, auxiliary.min, auxiliary.max
                    );
                }
                if let Some(terms) = chain_terms {
                    print!(
                        " chain_ns={:.6} term_ns={:.6}",
                        primary.median,
                        primary.median / terms as f64
                    );
                }
                println!();
                records.push(ProbeRecord {
                    id,
                    median: primary.median,
                });
            }
        }
    }
    Ok(records)
}

/// A packed-machinery pass with a checked operation count and execution result.
pub trait MachineryKernel: Send {
    type Error: StdError;
    fn operations(&self) -> usize;
    fn run(&mut self) -> Result<F128, Self::Error>;
    fn plan_construction_ns(&self) -> Option<f64> {
        None
    }
}

/// Times packed passes under the same warmed pools, samples and allocator as
/// the core runner. Source preparation precedes measurement. Construction is
/// reported separately; the requirement applies only to the measured pass.
#[expect(
    clippy::print_stdout,
    reason = "machinery records are benchmark output"
)]
pub fn run_machinery<C, E, S>(
    cases: &[(&str, Option<f64>)],
    prepare: impl Fn(usize) -> Result<Arc<S>, E> + Sync,
    construct: impl Fn(&str, Arc<S>, usize) -> Result<C, E> + Sync,
) -> Result<(), RunnerError>
where
    C: MachineryKernel<Error = E>,
    E: StdError + Send,
    S: Send + Sync,
{
    let options = Options::parse(true)?;
    let pools = warmed_pools(&options.threads)?;
    for (threads, pool) in &pools {
        for &log_t in &options.log_t {
            let source = pool
                .install(|| prepare(log_t))
                .map_err(|error| RunnerError::Core {
                    message: error.to_string(),
                })?;
            for &(name, requirement) in cases {
                if !options.units.is_empty() && !options.units.iter().any(|unit| unit == name) {
                    continue;
                }
                let mut samples = Vec::with_capacity(options.samples);
                let mut operations = 0;
                let mut plans = Vec::with_capacity(options.samples);
                for _ in 0..options.samples {
                    let (sample, count, plan) =
                        pool.install(|| {
                            let measurement = AllocationMeasurement::begin();
                            let start = Instant::now();
                            let mut kernel = construct(name, Arc::clone(&source), *threads)
                                .map_err(|error| RunnerError::Core {
                                    message: error.to_string(),
                                })?;
                            let construct_ns = start.elapsed().as_nanos() as f64;
                            let count = kernel.operations();
                            let plan = kernel.plan_construction_ns();
                            if count == 0 {
                                return Err(RunnerError::WorkCount {
                                    variant: name.to_owned(),
                                });
                            }
                            let start = Instant::now();
                            let _ = black_box(kernel.run().map_err(|error| RunnerError::Core {
                                message: error.to_string(),
                            })?);
                            let run_ns = start.elapsed().as_nanos() as f64;
                            Ok((
                                Sample {
                                    times: [construct_ns, run_ns, 0.0, 0.0],
                                    allocation: measurement.finish(),
                                },
                                count,
                                plan,
                            ))
                        })?;
                    samples.push(sample);
                    operations = count;
                    if let Some(plan) = plan {
                        plans.push(plan / count as f64);
                    }
                }
                let pass = Sample::phase(&samples, 1, operations as f64);
                let construction = Sample::phase(&samples, 0, (1_usize << log_t) as f64);
                let (peak_bytes, final_bytes, allocs) = Sample::allocations(&samples);
                if let Some(requirement) = requirement {
                    println!(
                        "machinery/{name}/{threads}  {:.6}  requirement {requirement}  {}",
                        pass.median,
                        if pass.median <= requirement {
                            "PASS"
                        } else {
                            "OVER"
                        }
                    );
                } else {
                    println!("machinery/{name}/{threads}  {:.6}", pass.median);
                }
                if !plans.is_empty() {
                    println!(
                        "machinery/{name}_plan/{threads} construct_ns={:.6}",
                        Summary::new(plans).median
                    );
                }
                println!("machinery/{name}_construction/{threads} construct_ns={:.6} pass_min_ns={:.6} pass_max_ns={:.6} samples={} peak_bytes={peak_bytes} final_bytes={final_bytes} allocs={allocs}", construction.median, pass.min, pass.max, options.samples);
            }
        }
    }
    Ok(())
}

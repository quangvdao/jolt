//! Run a core with `cargo bench -p jolt-rv64i-kernels --features test-utils
//! --bench example -- --log-t 20,22 --threads 1,12`.
//!
//! The constructor includes prerequisite passes and returns the input claim.
//! Rounds use fixed seeded challenges, finish binds the final challenge, and
//! extraction includes subsequent passes. Source generation and pool creation
//! precede measurement. Times are wall-clock nanoseconds per input cycle;
//! allocator bytes and counts cover the four phases, excluding resident sources.

pub mod allocator;
pub mod arithmetic;
pub mod example;
pub mod word;

use std::error::Error as StdError;
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

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
        Self::parse_with_defaults(
            probe,
            if probe { vec![22] } else { vec![20, 22] },
            if probe { 5 } else { 1 },
        )
    }

    fn parse_with_defaults(
        probe: bool,
        log_t: Vec<usize>,
        samples: usize,
    ) -> Result<Self, RunnerError> {
        let mut options = Self {
            log_t,
            threads: vec![1, 12],
            samples,
            units: Vec::new(),
        };
        let comparison = !probe
            && std::env::args()
                .any(|argument| ["--compare-monomial", "--variants"].contains(&argument.as_str()));
        if comparison {
            options.threads = vec![1];
            options.samples = 5;
        }
        let mut arguments = std::env::args().skip(1);
        while let Some(argument) = arguments.next() {
            if argument == "--bench" || comparison && argument == "--compare-monomial" {
                continue;
            }
            if !(["--log-t", "--threads", "--samples"].contains(&argument.as_str())
                || probe && argument == "--units"
                || comparison && argument == "--variants")
            {
                return Err(RunnerError::UnknownArgument { argument });
            }
            let value = arguments.next().ok_or_else(|| RunnerError::MissingValue {
                option: argument.clone(),
            })?;
            if argument == "--variants" {
                let variants: Vec<_> = value.split(',').collect();
                if variants.is_empty()
                    || variants.len() > 2
                    || variants
                        .iter()
                        .any(|variant| !["monomial_rounds2", "monomial_rounds3"].contains(variant))
                    || variants.windows(2).any(|pair| pair[0] == pair[1])
                {
                    return Err(RunnerError::InvalidValue {
                        option: argument,
                        value,
                    });
                }
                continue;
            }
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
                "--samples" if parsed.len() == 1 && (!comparison || parsed[0] >= 5) => {
                    options.samples = parsed[0]
                }
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

pub struct Summary {
    pub median: f64,
    pub min: f64,
    pub max: f64,
}

impl Summary {
    pub fn spread(&self) -> f64 {
        self.max - self.min
    }

    pub fn meets(&self, threshold: f64) -> bool {
        self.median <= threshold
    }

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
    times: Vec<f64>,
    allocation: AllocationStats,
}

impl Sample {
    fn phase(samples: &[Self], phase: usize, divisor: f64) -> Summary {
        Summary::new(samples.iter().map(|s| s.times[phase] / divisor).collect())
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

/// Converts measured durations to nanoseconds per input cycle.
#[derive(Clone, Copy)]
pub struct CycleScale(usize);

impl CycleScale {
    /// Normalises durations using the measured source's cycle count.
    pub fn durations<const N: usize>(self, values: [Duration; N]) -> [f64; N] {
        values.map(|time| time.as_nanos() as f64 / self.0 as f64)
    }
}

/// Clock for nested kernel phases. All benchmark clocks are owned by support.
pub struct Clock(Instant);

impl Clock {
    pub fn start() -> Self {
        Self(Instant::now())
    }

    pub fn elapsed(&self) -> Duration {
        self.0.elapsed()
    }
}

/// Durations for the declared phases of one sample, allocated before measurement.
pub struct PhaseTimes {
    times: Vec<Duration>,
}

impl PhaseTimes {
    pub fn set(&mut self, phase: usize, time: Duration) {
        self.times[phase] = time;
    }

    pub fn add(&mut self, phase: usize, time: Duration) {
        self.times[phase] += time;
    }
}

/// A case's record name, phase names, total membership and optional requirement.
/// Supplemental phases must not also enter the total when they overlap a primary.
pub struct Case<V> {
    pub name: String,
    pub variant: V,
    pub phases: Vec<String>,
    pub total: Vec<usize>,
    pub threshold: Option<fn(usize, usize) -> f64>,
}

impl<V> Case<V> {
    pub fn core(name: &str, variant: V, extra: &[&str]) -> Self {
        Self {
            name: name.to_owned(),
            variant,
            phases: ["construct", "rounds", "finish", "extract"]
                .into_iter()
                .chain(extra.iter().copied())
                .map(str::to_owned)
                .collect(),
            total: (0..4).collect(),
            threshold: None,
        }
    }
}

/// Runner-computed distributions, in nanoseconds per cycle, for one case.
pub struct Record {
    pub id: String,
    pub log_t: usize,
    pub threads: usize,
    pub samples: usize,
    pub phases: Vec<Summary>,
    pub total: Summary,
    pub allocation: AllocationStats,
}

impl Record {
    #[expect(clippy::print_stdout, reason = "runner records are benchmark output")]
    pub fn print(&self, id: &str, names: &[String], threshold: Option<f64>) {
        print!("{id}");
        for (name, phase) in names.iter().zip(&self.phases) {
            print!(" {name}_ns={:.6}", phase.median);
        }
        print!(" total_ns={:.6} peak_bytes={} final_bytes={} allocs={} samples={} total_min_ns={:.6} total_max_ns={:.6}", self.total.median, self.allocation.peak_bytes, self.allocation.final_bytes, self.allocation.allocs, self.samples, self.total.min, self.total.max);
        for (name, phase) in names.iter().zip(&self.phases) {
            print!(
                " {name}_min_ns={:.6} {name}_max_ns={:.6}",
                phase.min, phase.max
            );
        }
        if let Some(threshold) = threshold {
            print!(
                " threshold_ns={threshold:.6} meets_threshold={}",
                self.total.meets(threshold)
            );
        }
        println!(" loaded_machine=true");
    }

    #[expect(
        clippy::print_stdout,
        reason = "named phase records are benchmark output"
    )]
    pub fn print_phase(&self, id: &str, phase: usize, model: Option<f64>) {
        let summary = &self.phases[phase];
        print!(
            "{id} ns={:.6} min_ns={:.6} max_ns={:.6} samples={} loaded_machine=true",
            summary.median, summary.min, summary.max, self.samples
        );
        if let Some(model) = model {
            print!(
                " model_ns={model:.6} over_25_percent={}",
                !summary.meets(1.25 * model)
            );
        }
        println!();
    }
}

/// The only case sampling loop. Results and previous fixtures are released
/// between intervals; each allocation baseline sees only its current fixture.
fn collect_samples<R>(
    count: usize,
    samples: usize,
    mut measure: impl FnMut(usize) -> Result<R, RunnerError>,
    mut collect: impl FnMut(usize, R),
) -> Result<(), RunnerError> {
    for _ in 0..samples {
        for index in 0..count {
            collect(index, measure(index)?);
        }
    }
    Ok(())
}

pub fn seeded_challenges() -> [F128; 256] {
    let mut rng = ChaCha20Rng::seed_from_u64(0x726f_756e_6473);
    std::array::from_fn(|_| F128::random(&mut rng))
}

/// Prepares once per source/size and executes interleaved cases on warmed pools.
/// The returned measured state stays alive until allocation counters stop.
/// `prepare` and disposal of its result are outside all sample intervals.
#[expect(
    clippy::print_stdout,
    reason = "source preparation is a separate benchmark record"
)]
pub fn run_cases<P, R, E, V>(
    profiles: &[SynthProfile],
    cases: &[Case<V>],
    prepare: impl Fn(Arc<SyntheticTrace>) -> Result<P, E> + Sync,
    measure: impl Fn(&P, &V, &[F128], &mut PhaseTimes) -> Result<R, E> + Sync,
    inspect: impl Fn(&R, &mut PhaseTimes, CycleScale) + Sync,
    report: impl Fn(&Record, &P, &V) + Sync,
) -> Result<Vec<Record>, RunnerError>
where
    P: Send + Sync,
    V: Sync,
    E: StdError,
{
    let options = Options::parse(false)?;
    let pools = warmed_pools(&options.threads)?;
    let mut records = Vec::new();
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
                let prepared = pool.install(|| {
                    let clock = Clock::start();
                    let fixture = prepare(source).map_err(|error| RunnerError::Core {
                        message: error.to_string(),
                    })?;
                    let duration = clock.elapsed();
                    Ok::<_, RunnerError>((fixture, duration))
                })?;
                if let Some(case) = cases.first() {
                    println!("{}_source_setup/{}/{log_t}/{threads} prepare_ns={:.6} samples=1 loaded_machine=true", case.name, profile.name(), prepared.1.as_nanos() as f64 / cycles as f64);
                }
                let point = seeded_challenges();
                let mut collected: Vec<Vec<Sample>> = cases
                    .iter()
                    .map(|_| Vec::with_capacity(options.samples))
                    .collect();
                collect_samples(
                    cases.len(),
                    options.samples,
                    |index| {
                        let mut times = PhaseTimes {
                            times: vec![Duration::ZERO; cases[index].phases.len()],
                        };
                        pool.install(|| {
                            let measurement = AllocationMeasurement::begin();
                            let state =
                                measure(&prepared.0, &cases[index].variant, &point, &mut times)
                                    .map_err(|error| RunnerError::Core {
                                        message: error.to_string(),
                                    })?;
                            let allocation = measurement.finish();
                            inspect(&state, &mut times, CycleScale(cycles));
                            drop(state);
                            Ok(Sample {
                                times: times
                                    .times
                                    .into_iter()
                                    .map(|time| time.as_nanos() as f64)
                                    .collect(),
                                allocation,
                            })
                        })
                    },
                    |index, sample| collected[index].push(sample),
                )?;
                for (case, samples) in cases.iter().zip(collected) {
                    let divisor = cycles as f64;
                    let phases = (0..case.phases.len())
                        .map(|phase| Sample::phase(&samples, phase, divisor))
                        .collect();
                    let total = Summary::new(
                        samples
                            .iter()
                            .map(|sample| {
                                case.total
                                    .iter()
                                    .map(|&phase| sample.times[phase])
                                    .sum::<f64>()
                                    / divisor
                            })
                            .collect(),
                    );
                    let (peak_bytes, final_bytes, allocs) = Sample::allocations(&samples);
                    let record = Record {
                        id: format!("{}/{}/{log_t}/{threads}", case.name, profile.name()),
                        log_t,
                        threads: *threads,
                        samples: options.samples,
                        phases,
                        total,
                        allocation: AllocationStats {
                            peak_bytes,
                            final_bytes,
                            allocs,
                        },
                    };
                    record.print(
                        &record.id,
                        &case.phases,
                        case.threshold.map(|threshold| threshold(log_t, *threads)),
                    );
                    report(&record, &prepared.0, &case.variant);
                    records.push(record);
                }
            }
        }
    }
    Ok(records)
}

/// Executes canonical low-variable-first rounds and final binding.
pub fn core_rounds<'a, C: ProveRounds<F128>>(
    core: &mut C,
    mut claim: F128,
    challenges: &'a [F128],
) -> Result<CoreRun<'a>, RunnerError> {
    let rounds = core.num_rounds();
    let point = challenges
        .get(..rounds)
        .ok_or(RunnerError::ChallengeCount {
            rounds,
            capacity: challenges.len(),
        })?;
    let clock = Clock::start();
    let mut bind = None;
    for (round, &challenge) in point.iter().enumerate() {
        let message = core.prove_round(bind, round, claim)?;
        claim = message.evaluate(challenge);
        bind = Some(challenge);
        let _ = black_box(&message);
    }
    let rounds = clock.elapsed();
    let clock = Clock::start();
    if let Some(challenge) = bind {
        core.finish_rounds(challenge)?;
    }
    let finish = clock.elapsed();
    Ok(CoreRun {
        challenges: point,
        rounds,
        finish,
    })
}

/// Borrowed seeded point and the disjoint timings of a single core.
pub struct CoreRun<'a> {
    pub challenges: &'a [F128],
    pub rounds: Duration,
    pub finish: Duration,
}

/// Batch challenges and disjoint round/terminal-bind wall times.
pub struct BatchRun {
    pub challenges: Vec<F128>,
    pub rounds: Duration,
    pub finish: Duration,
}

/// Four-phase entry for real batches, including source-scoped preparation.
pub fn run_batch<P, C, O, E>(
    profiles: &[SynthProfile],
    case: Case<()>,
    prepare: impl Fn(Arc<SyntheticTrace>) -> Result<P, E> + Sync,
    construct: impl Fn(&P, &mut PhaseTimes) -> Result<C, E> + Sync,
    prove: impl Fn(&mut C, &mut PhaseTimes) -> Result<BatchRun, E> + Sync,
    extract: impl Fn(&C, &[F128], &mut PhaseTimes) -> Result<O, E> + Sync,
    report: impl Fn(&Record, &P) + Sync,
) -> Result<(), RunnerError>
where
    P: Send + Sync,
    E: StdError,
{
    let _ = run_cases(
        profiles,
        &[case],
        prepare,
        |fixture, (), _, times| {
            let clock = Clock::start();
            let mut core = construct(fixture, times)?;
            times.set(0, clock.elapsed());
            let batch = prove(&mut core, times)?;
            times.set(1, batch.rounds);
            times.set(2, batch.finish);
            let clock = Clock::start();
            let output = extract(&core, &batch.challenges, times)?;
            let _ = black_box(&output);
            times.set(3, clock.elapsed());
            Ok::<_, E>((core, output))
        },
        |_, _, _| {},
        |record, fixture, ()| report(record, fixture),
    )?;
    Ok(())
}

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
    run_core_variants(
        bench,
        profiles,
        &[("", ())],
        |source, ()| construct(source),
        extract,
        |_, _, _| {},
    )
}

pub fn run_core_variants<C, E, O, V: Sync>(
    bench: &str,
    profiles: &[SynthProfile],
    variants: &[(&str, V)],
    construct: impl Fn(Arc<SyntheticTrace>, &V) -> Result<(C, F128), E> + Sync,
    extract: impl Fn(&C, &[F128]) -> Result<O, E> + Sync,
    report: impl Fn(&C, &O, CycleScale) + Sync,
) -> Result<(), RunnerError>
where
    C: ProveRounds<F128> + Send,
    E: StdError,
{
    let cases: Vec<_> = variants
        .iter()
        .map(|(name, variant)| {
            Case::core(
                &if name.is_empty() {
                    bench.to_owned()
                } else {
                    format!("{bench}/{name}")
                },
                variant,
                &[],
            )
        })
        .collect();
    let _ = run_cases(
        profiles,
        &cases,
        Ok::<_, RunnerError>,
        |source, variant, point, times| {
            let clock = Clock::start();
            let (mut core, claim) =
                construct(Arc::clone(source), variant).map_err(|error| RunnerError::Core {
                    message: error.to_string(),
                })?;
            times.set(0, clock.elapsed());
            let batch = core_rounds(&mut core, claim, point)?;
            times.set(1, batch.rounds);
            times.set(2, batch.finish);
            let clock = Clock::start();
            let output = extract(&core, batch.challenges).map_err(|error| RunnerError::Core {
                message: error.to_string(),
            })?;
            let _ = black_box(&output);
            times.set(3, clock.elapsed());
            // Diagnostics execute after the runner closes the allocation interval.
            Ok((core, output))
        },
        |(core, output), _, scale| report(core, output, scale),
        |_, _, _| {},
    )?;
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

impl ProbeCase {
    fn matches(&self, selection: &str) -> bool {
        selection == self.unit
            || selection
                .strip_prefix(self.unit)
                .and_then(|tail| tail.strip_prefix('/'))
                .is_some_and(|prefix| self.variant.starts_with(prefix))
    }
}

/// A preallocated operation stream. The runner times each pass once per sample.
/// `operations` gives the count consumed by the one measured pass.
pub trait ProbeKernel: Send {
    fn operations(&self) -> usize;
    fn run(&mut self) -> F128;
    fn chain_terms(&self) -> Option<usize> {
        None
    }
    fn memory_layout(&self) -> Option<(usize, usize)> {
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
    pub layout: Option<(usize, usize)>,
}

struct ProbeInfo {
    operations: usize,
    chain_terms: Option<usize>,
    lookup_layout: Option<(usize, usize)>,
    memory_layout: Option<(usize, usize)>,
}

/// Runs unit probes with warmed pools and resident sources excluded from timing.
/// Each sample constructs fresh state under the counting allocator. `ns` is
/// median primary wall time per operation; min/max bound the observed samples.
/// Chain records also print nanoseconds per chain and per term; trace-driven
/// fused-chain totals include their operand preparation.
/// Allocation fields are maxima across samples and include construction; final
/// bytes count the state still resident after its measured passes. Probe defaults
/// are log_t=22, threads=1,12 and samples=5; `--units` selects units or
/// unit/variant prefixes. Samples alternate forward and reverse case order.
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
        if !cases.iter().any(|case| case.matches(unit)) {
            return Err(RunnerError::InvalidValue {
                option: "--units".to_owned(),
                value: unit.clone(),
            });
        }
    }
    let pools = warmed_pools(&options.threads)?;
    let mut records = Vec::new();
    let selected = |case: &&ProbeCase| {
        options.units.is_empty() || options.units.iter().any(|unit| case.matches(unit))
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
            let mut collected: Vec<Vec<(Sample, ProbeInfo)>> = selected_cases
                .iter()
                .map(|_| Vec::with_capacity(options.samples))
                .collect();
            collect_samples(
                selected_cases.len(),
                options.samples,
                |index| {
                    let case = selected_cases[index];
                    let sample = pool.install(|| {
                        let measurement = AllocationMeasurement::begin();
                        let start = Instant::now();
                        let mut kernel =
                            construct(case, source.clone(), *threads).map_err(|error| {
                                RunnerError::Core {
                                    message: error.to_string(),
                                }
                            })?;
                        let construct_ns = start.elapsed().as_nanos() as f64;
                        let info = ProbeInfo {
                            operations: kernel.operations(),
                            chain_terms: kernel.chain_terms(),
                            lookup_layout: kernel.lookup_layout(),
                            memory_layout: kernel.memory_layout(),
                        };
                        if info.operations == 0 {
                            return Err(RunnerError::WorkCount {
                                variant: case.variant.clone(),
                            });
                        }
                        let start = Instant::now();
                        let _ = black_box(kernel.run());
                        let primary_ns = start.elapsed().as_nanos() as f64;
                        let allocation = measurement.finish();
                        Ok::<_, RunnerError>((
                            Sample {
                                times: vec![construct_ns, primary_ns, 0.0, 0.0],
                                allocation,
                            },
                            info,
                        ))
                    })?;
                    Ok(sample)
                },
                |index, sample| collected[index].push(sample),
            )?;
            for (case, collected) in selected_cases.iter().zip(collected) {
                let info = &collected[collected.len() - 1].1;
                let operations = info.operations;
                let chain_terms = info.chain_terms;
                let lookup_layout = info.lookup_layout;
                let memory_layout = info.memory_layout;
                let samples: Vec<_> = collected.into_iter().map(|(sample, _)| sample).collect();
                let primary = Sample::phase(&samples, 1, operations as f64);
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
                if let Some((bytes, arrays)) = memory_layout {
                    print!(" bytes_per_array={bytes} arrays={arrays}");
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
                    layout: memory_layout,
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
    /// Resident routing and pass scratch bytes, excluding the shared input source.
    fn memory_bytes(&self) -> Option<(usize, usize)> {
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
    let options = Options::parse_with_defaults(true, vec![20, 22], 5)?;
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
                let mut constructions = Vec::with_capacity(options.samples);
                let mut operations = 0;
                let mut plans = Vec::with_capacity(options.samples);
                let mut memory = None;
                collect_samples(
                    1,
                    options.samples,
                    |_| {
                        let (sample, construction, count, plan, resident) = pool.install(|| {
                            let measurement = AllocationMeasurement::begin();
                            let start = Instant::now();
                            let mut kernel = construct(name, Arc::clone(&source), *threads)
                                .map_err(|error| RunnerError::Core {
                                    message: error.to_string(),
                                })?;
                            let construct_ns = start.elapsed().as_nanos() as f64;
                            let construction_allocation = measurement.finish();
                            let count = kernel.operations();
                            let plan = kernel.plan_construction_ns();
                            let resident = kernel.memory_bytes();
                            if count == 0 {
                                return Err(RunnerError::WorkCount {
                                    variant: name.to_owned(),
                                });
                            }
                            let pass_measurement = AllocationMeasurement::begin();
                            let start = Instant::now();
                            let _ = black_box(kernel.run().map_err(|error| RunnerError::Core {
                                message: error.to_string(),
                            })?);
                            let run_ns = start.elapsed().as_nanos() as f64;
                            Ok((
                                Sample {
                                    times: vec![0.0, run_ns, 0.0, 0.0],
                                    allocation: pass_measurement.finish(),
                                },
                                Sample {
                                    times: vec![construct_ns, 0.0, 0.0, 0.0],
                                    allocation: construction_allocation,
                                },
                                count,
                                plan,
                                resident,
                            ))
                        })?;
                        Ok((sample, construction, count, plan, resident))
                    },
                    |_, (sample, construction, count, plan, resident)| {
                        samples.push(sample);
                        constructions.push(construction);
                        operations = count;
                        memory = resident;
                        if let Some(plan) = plan {
                            plans.push(plan / count as f64);
                        }
                    },
                )?;
                let pass = Sample::phase(&samples, 1, operations as f64);
                let construction = Sample::phase(&constructions, 0, (1_usize << log_t) as f64);
                let (peak_bytes, final_bytes, allocs) = Sample::allocations(&constructions);
                let (pass_peak_bytes, pass_final_bytes, pass_allocs) =
                    Sample::allocations(&samples);
                if let Some(requirement) = requirement {
                    println!(
                        "machinery/{name}/{log_t}/{threads}  {:.6}  requirement {requirement}  {}",
                        pass.median,
                        if pass.median <= requirement {
                            "PASS"
                        } else {
                            "OVER"
                        }
                    );
                } else {
                    println!("machinery/{name}/{log_t}/{threads}  {:.6}", pass.median);
                }
                if !plans.is_empty() {
                    let construction_ns = Summary::new(plans).median;
                    if let Some((plan_bytes, _)) = memory {
                        println!("machinery/{name}_plan/{log_t}/{threads} construct_ns={construction_ns:.6} plan_bytes={plan_bytes}");
                    } else {
                        println!("machinery/{name}_plan/{log_t}/{threads} construct_ns={construction_ns:.6}");
                    }
                }
                println!("machinery/{name}_construction/{log_t}/{threads} construct_ns={:.6} peak_bytes={peak_bytes} final_bytes={final_bytes} allocs={allocs}", construction.median);
                println!("machinery/{name}_pass/{log_t}/{threads} pass_min_ns={:.6} pass_max_ns={:.6} samples={} peak_bytes={pass_peak_bytes} final_bytes={pass_final_bytes} allocs={pass_allocs}", pass.min, pass.max, options.samples);
                if let Some((plan_bytes, scratch_bytes)) = memory {
                    println!("machinery/{name}_memory/{log_t}/{threads} plan_bytes={plan_bytes} scratch_bytes={scratch_bytes}");
                }
            }
        }
    }
    Ok(())
}

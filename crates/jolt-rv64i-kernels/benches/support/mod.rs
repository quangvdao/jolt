//! Run a core with `cargo bench -p jolt-rv64i-kernels --features test-utils
//! --bench example -- --log-t 20,22 --threads 1,12`.
//!
//! The constructor includes prerequisite passes and returns the input claim.
//! Rounds use fixed seeded challenges, finish binds the final challenge, and
//! extraction includes subsequent passes. Source generation and pool creation
//! precede measurement. Times are wall-clock nanoseconds per input cycle;
//! allocator bytes and counts cover the four phases, excluding resident sources.

mod allocator;
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
use rayon::ThreadPoolBuilder;
use thiserror::Error;

use self::allocator::AllocationMeasurement;

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
    #[error(transparent)]
    Sumcheck(#[from] SumcheckError<F128>),
}

struct Options {
    log_t: Vec<usize>,
    threads: Vec<usize>,
}

impl Options {
    fn parse() -> Result<Self, RunnerError> {
        let mut options = Self {
            log_t: vec![20, 22],
            threads: vec![1, 12],
        };
        let mut arguments = std::env::args().skip(1);
        while let Some(argument) = arguments.next() {
            if argument == "--bench" {
                continue;
            }
            if argument != "--log-t" && argument != "--threads" {
                return Err(RunnerError::UnknownArgument { argument });
            }
            let value = arguments.next().ok_or_else(|| RunnerError::MissingValue {
                option: argument.clone(),
            })?;
            let mut parsed = Vec::new();
            for part in value.split(',') {
                let entry = part
                    .parse::<usize>()
                    .map_err(|_| RunnerError::InvalidValue {
                        option: argument.clone(),
                        value: value.clone(),
                    })?;
                if (argument == "--threads" && entry == 0)
                    || (argument == "--log-t" && entry >= usize::BITS as usize)
                    || parsed.contains(&entry)
                {
                    return Err(RunnerError::InvalidValue {
                        option: argument,
                        value,
                    });
                }
                parsed.push(entry);
            }
            if argument == "--log-t" {
                options.log_t = parsed;
            } else {
                options.threads = parsed;
            }
        }
        Ok(options)
    }
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
    let options = Options::parse()?;
    for threads in options.threads {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|error| RunnerError::ThreadPool {
                message: error.to_string(),
            })?;
        let _ = pool.broadcast(|_| black_box(()));
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
                let result = pool.install(|| {
                    let measurement = AllocationMeasurement::begin();
                    let start = Instant::now();
                    let (mut core, mut claim) =
                        construct(Arc::clone(&source)).map_err(|error| RunnerError::Core {
                            message: error.to_string(),
                        })?;
                    let construct_ns = start.elapsed().as_nanos() as f64;
                    let rounds = core.num_rounds();
                    let point = challenges
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
                    Ok::<_, RunnerError>((
                        [construct_ns, rounds_ns, finish_ns, extract_ns],
                        allocation,
                    ))
                })?;
                let ([construct_ns, rounds_ns, finish_ns, extract_ns], allocation) = result;
                let total_ns = construct_ns + rounds_ns + finish_ns + extract_ns;
                let divisor = cycles as f64;
                println!(
                    "{bench}/{}/{log_t}/{threads} construct_ns={:.6} rounds_ns={:.6} finish_ns={:.6} extract_ns={:.6} total_ns={:.6} peak_bytes={} final_bytes={} allocs={}",
                    profile.name(), construct_ns / divisor, rounds_ns / divisor,
                    finish_ns / divisor, extract_ns / divisor, total_ns / divisor,
                    allocation.peak_bytes, allocation.final_bytes, allocation.allocs,
                );
            }
        }
    }
    Ok(())
}

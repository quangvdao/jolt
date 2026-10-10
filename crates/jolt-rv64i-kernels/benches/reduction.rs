//! Compare the digit builder and shared-table reduction on local traces.

pub mod support;

use std::hint::black_box;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use jolt_field::{Field, F128};
use jolt_rv64i_kernels::packed::lift::WordLift;
use jolt_rv64i_kernels::par::CycleChunks;
use jolt_rv64i_kernels::reduction::{
    g_pass_digits, ColumnMap, ReductionCore, ReductionError, ReductionLeg,
};
use jolt_rv64i_kernels::source::{CycleSource, SourceError, ValidatedTrace};
use jolt_rv64i_kernels::synth::{SynthProfile, SyntheticTrace};
use jolt_sumcheck::ProveRounds;
use jolt_utils::unsafe_allocate_zero_vec;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use thiserror::Error;

use support::allocator::{AllocationMeasurement, AllocationStats};
use support::{run_core, RunnerError};

const ZERO: F128 = F128::from_raw(0);
const ONE: F128 = F128::from_raw(1);

#[derive(Debug, Error)]
enum BenchError {
    #[error(transparent)]
    Source(#[from] SourceError),
    #[error(transparent)]
    Reduction(#[from] ReductionError),
    #[error("validated fixture cache lock was poisoned")]
    Cache,
}

struct Fixture {
    source: Arc<SyntheticTrace>,
    validated: ValidatedTrace<SyntheticTrace>,
}

fn map() -> Vec<ColumnMap> {
    let mut map = vec![ColumnMap::Word {
        start: 0,
        trace_word: 5,
    }];
    for column in 0..10 {
        map.push(ColumnMap::Indicators {
            start: 64 + 15 * column,
            column,
        });
    }
    for (start, column) in [(214, 10), (221, 11)] {
        map.push(ColumnMap::Indicators { start, column });
    }
    map.push(ColumnMap::Flags {
        start: 228,
        columns: vec![18, 19, 20],
    });
    map
}

fn weights() -> Vec<Vec<F128>> {
    (0..3)
        .map(|support| {
            (0..256)
                .map(|column| {
                    let active = match support {
                        0 => (64..=228).contains(&column),
                        1 => column < 64 || (139..=230).contains(&column),
                        _ => column < 64,
                    };
                    if active {
                        let scalar = if column < 64 { 9 } else { support as u128 + 7 };
                        F128::from_raw(((column as u128 + 1) << 64) | scalar)
                    } else {
                        ZERO
                    }
                })
                .collect()
        })
        .collect()
}

#[expect(
    clippy::expect_used,
    reason = "SyntheticTrace construction checks the cycle exponent"
)]
fn word_table(source: &SyntheticTrace, weight: &[F128]) -> Vec<F128> {
    let mut values = [ZERO; 64];
    values.copy_from_slice(&weight[..64]);
    let lift = WordLift::new(&values);
    let mut table = unsafe_allocate_zero_vec(source.cycles());
    let chunks = CycleChunks::new(source.cycles().ilog2() as usize, 0)
        .expect("synthetic trace has validated geometry");
    table
        .par_chunks_mut(chunks.chunk_len())
        .enumerate()
        .for_each(|(chunk, output)| {
            for (offset, value) in output.iter_mut().enumerate() {
                *value = lift.lift(source.trace_word(5, chunk * chunks.chunk_len() + offset));
            }
        });
    table
}

fn core_from_tables(
    tables: Vec<Vec<F128>>,
    cycles: usize,
    shared: bool,
) -> Result<(ReductionCore, F128), BenchError> {
    let log_t = cycles.ilog2() as usize;
    let count = if shared { 4 } else { 3 };
    let mut legs = Vec::with_capacity(count);
    let mut claim = ZERO;
    for leg in 0..count {
        let table = if shared { [0, 1, 2, 2][leg] } else { leg };
        // Boolean points give honest input claims by one table read, keeping
        // fixture summation out of the model's measured construction pass.
        let point_leg = if shared { [0, 1, 1, 2][leg] } else { leg };
        let vertex = (0x39a5usize.wrapping_mul(point_leg + 1)) & (cycles - 1);
        let point = (0..log_t)
            .map(|bit| if vertex & (1 << bit) == 0 { ZERO } else { ONE })
            .collect();
        let coefficient = F128::from_raw(if shared {
            [1, 2, 2, 3][leg]
        } else {
            leg as u128 + 1
        });
        let value = tables[table][vertex];
        claim += coefficient * value;
        legs.push(ReductionLeg {
            table,
            point,
            coefficient,
            claim: value,
        });
    }
    Ok((ReductionCore::new(tables, legs)?, claim))
}

fn run_default() -> Result<(), RunnerError> {
    let cache = Mutex::new(None::<Fixture>);
    let map = map();
    let weights = weights();
    run_core(
        "reduction",
        &[SynthProfile::Local],
        |source| {
            let tables = {
                let mut cached = cache.lock().map_err(|_| BenchError::Cache)?;
                if cached
                    .as_ref()
                    .is_none_or(|fixture| !Arc::ptr_eq(&fixture.source, &source))
                {
                    *cached = Some(Fixture {
                        validated: ValidatedTrace::new(Arc::clone(&source))?,
                        source: Arc::clone(&source),
                    });
                }
                let fixture = cached.as_ref().ok_or(BenchError::Cache)?;
                g_pass_digits(&fixture.validated, &map, &weights)?
            };
            core_from_tables(tables, source.cycles(), false)
        },
        |core, _| Ok::<_, BenchError>(core.final_values()?.to_vec()),
    )
}

struct PreparedOptions {
    log_t: Vec<usize>,
    threads: Vec<usize>,
    samples: usize,
}

impl PreparedOptions {
    fn parse() -> Result<Self, RunnerError> {
        let mut options = Self {
            log_t: vec![20, 22],
            threads: vec![1, 12],
            samples: 1,
        };
        let mut arguments = std::env::args().skip(1);
        while let Some(argument) = arguments.next() {
            if argument == "--bench" {
                continue;
            }
            if !["--log-t", "--threads", "--samples"].contains(&argument.as_str()) {
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
                if (argument == "--log-t" && !(1..=32).contains(&entry))
                    || (argument != "--log-t" && entry == 0)
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
                    })
                }
            }
        }
        Ok(options)
    }
}

struct PreparedSample {
    preparation: f64,
    times: [f64; 4],
    allocation: AllocationStats,
}

fn summary(mut values: Vec<f64>) -> [f64; 3] {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    let median = if values.len().is_multiple_of(2) {
        values[middle - 1].midpoint(values[middle])
    } else {
        values[middle]
    };
    [median, values[0], values[values.len() - 1]]
}

#[expect(
    clippy::print_stdout,
    reason = "prepared-fixture records follow the runner's output contract"
)]
fn report_prepared(samples: &[PreparedSample], log_t: usize, threads: usize) {
    let divisor = (1usize << log_t) as f64;
    let phases: [[f64; 3]; 4] = std::array::from_fn(|phase| {
        summary(
            samples
                .iter()
                .map(|sample| sample.times[phase] / divisor)
                .collect(),
        )
    });
    let total = summary(
        samples
            .iter()
            .map(|sample| sample.times.iter().sum::<f64>() / divisor)
            .collect(),
    );
    let preparation = summary(
        samples
            .iter()
            .map(|sample| sample.preparation / divisor)
            .collect(),
    );
    let peak_bytes = samples
        .iter()
        .map(|sample| sample.allocation.peak_bytes)
        .max()
        .unwrap_or(0);
    let final_bytes = samples
        .iter()
        .map(|sample| sample.allocation.final_bytes)
        .max()
        .unwrap_or(0);
    let allocs = samples
        .iter()
        .map(|sample| sample.allocation.allocs)
        .max()
        .unwrap_or(0);
    print!(
        "reduction_shared/local/{log_t}/{threads} construct_ns={:.6} rounds_ns={:.6} finish_ns={:.6} extract_ns={:.6} total_ns={:.6} peak_bytes={peak_bytes} final_bytes={final_bytes} allocs={allocs}",
        phases[0][0], phases[1][0], phases[2][0], phases[3][0], total[0],
    );
    if samples.len() > 1 {
        print!(
            " samples={} total_min_ns={:.6} total_max_ns={:.6}",
            samples.len(),
            total[1],
            total[2]
        );
        for (name, phase) in ["construct", "rounds", "finish", "extract"]
            .iter()
            .zip(phases)
        {
            print!(
                " {name}_min_ns={:.6} {name}_max_ns={:.6}",
                phase[1], phase[2]
            );
        }
    }
    println!(
        " prepared_bytes={} prepare_ns={:.6} prepare_min_ns={:.6} prepare_max_ns={:.6}",
        (1usize << log_t) * std::mem::size_of::<F128>(),
        preparation[0],
        preparation[1],
        preparation[2],
    );
}

fn run_prepared_shared() -> Result<(), RunnerError> {
    let options = PreparedOptions::parse()?;
    let pools = options
        .threads
        .iter()
        .map(|&threads| {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|error| RunnerError::ThreadPool {
                    message: error.to_string(),
                })?;
            let _ = pool.broadcast(|_| black_box(()));
            Ok::<_, RunnerError>((threads, pool))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let map = map();
    let weights = weights();
    let mut pass_weights = weights[..2].to_vec();
    pass_weights[1][..64].fill(ZERO);
    for (threads, pool) in &pools {
        for &log_t in &options.log_t {
            let source = Arc::new(
                pool.install(|| SyntheticTrace::new(SynthProfile::Local, log_t, 1 << 20, 0x5eed))
                    .map_err(|error| RunnerError::Trace {
                        message: error.to_string(),
                    })?,
            );
            let validated =
                ValidatedTrace::new(Arc::clone(&source)).map_err(|error| RunnerError::Core {
                    message: error.to_string(),
                })?;
            let mut rng = ChaCha20Rng::seed_from_u64(0x726f_756e_6473);
            let challenges: [F128; 256] = std::array::from_fn(|_| F128::random(&mut rng));
            let mut samples = Vec::with_capacity(options.samples);
            for _ in 0..options.samples {
                samples.push(pool.install(|| {
                    let start = Instant::now();
                    let prepared = word_table(&source, &weights[2]);
                    let preparation = start.elapsed().as_nanos() as f64;
                    // The prepared table has one owner, transferred into the core.
                    // Its allocation and pass precede the four measured phases.
                    let measurement = AllocationMeasurement::begin();
                    let start = Instant::now();
                    let mut tables =
                        g_pass_digits(&validated, &map, &pass_weights).map_err(|error| {
                            RunnerError::Core {
                                message: error.to_string(),
                            }
                        })?;
                    tables.push(prepared);
                    let (mut core, mut claim) = core_from_tables(tables, source.cycles(), true)
                        .map_err(|error| RunnerError::Core {
                            message: error.to_string(),
                        })?;
                    let construct = start.elapsed().as_nanos() as f64;
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
                    let rounds = start.elapsed().as_nanos() as f64;
                    let start = Instant::now();
                    if let Some(challenge) = bind {
                        core.finish_rounds(challenge)?;
                    }
                    let finish = start.elapsed().as_nanos() as f64;
                    let start = Instant::now();
                    let values = core
                        .final_values()
                        .map_err(|error| RunnerError::Core {
                            message: error.to_string(),
                        })?
                        .to_vec();
                    let _ = black_box(&values);
                    let extract = start.elapsed().as_nanos() as f64;
                    Ok::<_, RunnerError>(PreparedSample {
                        preparation,
                        times: [construct, rounds, finish, extract],
                        allocation: measurement.finish(),
                    })
                })?);
            }
            report_prepared(&samples, log_t, *threads);
        }
    }
    Ok(())
}

fn main() -> Result<(), RunnerError> {
    run_default()?;
    run_prepared_shared()
}

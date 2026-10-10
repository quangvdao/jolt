//! Manual commit/open measurement; run on a host whose load is recorded:
//! `RUSTFLAGS='-C target-cpu=native' cargo run --release -p jolt-rv64i-pcs
//! --features arch,test-utils --example measure -- --threads 1,12 --log-t 22`.
//! Each configuration runs once with warmed workers and seeded rows. The
//! combined allocation interval excludes shared rows and ends with the proof
//! alive; frontend column generation lies inside that interval but outside
//! commit/open timers. This tool has no benchmark acceptance assertions.

#[path = "../../jolt-rv64i-kernels/benches/support/allocator.rs"]
mod allocator;

use allocator::{AllocationMeasurement, CountingAllocator, RAYON_WORKER_ALLOWANCE};
use jolt_field::{Zero, F128};
use jolt_rv64i_pcs::{
    commit::commit_observed,
    measure::{Event, Phase, ReleasePoint},
    open::open_observed,
};
use jolt_rv64i_verifier::{
    commitment::{BitsCommitmentScheme, BitsGeometry, BitsOpening, BitsWire},
    points::{equality_table, PointsError},
    transcript::Rv64iTranscript,
    whir::WhirBits,
};
use jolt_transcript::{Label, Transcript, U64Word};
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use std::error::Error;
use std::hint::black_box;
use std::io::{self, BufWriter, Result as IoResult, Write};
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

const PHASES: [Phase; 12] = [
    Phase::LevelZeroEncode,
    Phase::LevelZeroTree,
    Phase::CommitSample,
    Phase::BridgeTables,
    Phase::BridgeFirstPass,
    Phase::BridgeFirstFold,
    Phase::LaterRounds,
    Phase::InducedWeights,
    Phase::EqualityAndSamples,
    Phase::LaterEncodes,
    Phase::LaterTrees,
    Phase::Queries,
];

struct Observations {
    times: [Duration; 12],
    starts: [Option<Instant>; 12],
    releases: [Option<(ReleasePoint, usize)>; 64],
    count: usize,
    overflow: bool,
    baseline: usize,
}
impl Observations {
    fn new() -> Self {
        Self {
            times: [Duration::ZERO; 12],
            starts: [None; 12],
            releases: [None; 64],
            count: 0,
            overflow: false,
            baseline: CountingAllocator::live_bytes(),
        }
    }
    fn observe(&mut self, event: Event) {
        match event {
            Event::Start(phase) => {
                if let Some(index) = PHASES.iter().position(|candidate| *candidate == phase) {
                    self.starts[index] = Some(Instant::now());
                }
            }
            Event::End(phase) => {
                if let Some(index) = PHASES.iter().position(|candidate| *candidate == phase) {
                    if let Some(start) = self.starts[index].take() {
                        self.times[index] += start.elapsed();
                    }
                }
            }
            Event::Released(point) => {
                if let Some(slot) = self.releases.get_mut(self.count) {
                    *slot = Some((
                        point,
                        CountingAllocator::live_bytes().saturating_sub(self.baseline),
                    ));
                    self.count += 1;
                } else {
                    self.overflow = true;
                }
            }
        }
    }
}

fn seeded_word(index: u64) -> u64 {
    let mut value = index.wrapping_add(0x1280_1920_6400_2026);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn columns(rows: &[[u64; 4]], cycle: &[F128]) -> Result<[F128; 256], PointsError> {
    let weights = equality_table(cycle)?;
    Ok(rows
        .par_iter()
        .zip(weights.par_iter())
        .with_min_len(1024)
        .fold(
            || [F128::zero(); 256],
            |mut values, (row, weight)| {
                for (word_index, word) in row.iter().copied().enumerate() {
                    let mut bits = word;
                    while bits != 0 {
                        let column = 64 * word_index + bits.trailing_zeros() as usize;
                        values[column] += *weight;
                        bits &= bits - 1;
                    }
                }
                values
            },
        )
        .reduce(
            || [F128::zero(); 256],
            |mut left, right| {
                for (value, increment) in left.iter_mut().zip(right) {
                    *value += increment;
                }
                left
            },
        ))
}

fn bind_geometry(transcript: &mut Rv64iTranscript, geometry: BitsGeometry) {
    transcript.append(&Label(b"params"));
    transcript.append(&U64Word(geometry.log_T as u64));
}

fn sample(
    out: &mut impl Write,
    pool: &ThreadPool,
    geometry: BitsGeometry,
    rows: &Arc<[[u64; 4]]>,
) -> Result<(), Box<dyn Error>> {
    let _ = pool.broadcast(|_| black_box([0_u8; 32]));
    let _ = pool.install(|| {
        black_box(
            (0..16_384)
                .into_par_iter()
                .map(seeded_word)
                .reduce(|| 0, u64::wrapping_add),
        )
    });
    let mut prover = Rv64iTranscript::new(b"whir-measure");
    bind_geometry(&mut prover, geometry);
    let mut observations = Observations::new();
    let interval = AllocationMeasurement::begin();
    let start = Instant::now();
    let (commitment, state) = pool.install(|| {
        commit_observed(geometry, rows, &mut prover, &mut |event| {
            observations.observe(event);
        })
    })?;
    let commit_time = start.elapsed();
    let cycle = prover.challenge_vector(geometry.log_T);
    let values = pool.install(|| columns(rows, &cycle))?;
    for value in &values {
        prover.append_labeled(b"opening_claim", value);
    }
    let rho = prover.challenge_vector(8);
    let request = BitsOpening {
        geometry,
        column_point: &rho,
        cycle_point: &cycle,
        columns: &values,
    };
    let start = Instant::now();
    let proof = pool.install(|| {
        open_observed(state, &request, &mut prover, &mut |event| {
            observations.observe(event);
        })
    })?;
    let open_time = start.elapsed();
    let allocation = interval.finish();
    if observations.overflow {
        return Err("release snapshot capacity exceeded".into());
    }
    let mut bytes = Vec::new();
    proof.write(&mut bytes);
    let mut verifier = Rv64iTranscript::new(b"whir-measure");
    bind_geometry(&mut verifier, geometry);
    let start = Instant::now();
    let retained = WhirBits::verify_commit(&(), geometry, &commitment, &mut verifier)?;
    if verifier.challenge_vector(geometry.log_T) != cycle {
        return Err("commit transcript differs between prover and verifier".into());
    }
    for value in &values {
        verifier.append_labeled(b"opening_claim", value);
    }
    if verifier.challenge_vector(8) != rho {
        return Err("opening point differs between prover and verifier".into());
    }
    WhirBits::verify_opening(&(), retained, &request, &proof, &mut verifier)?;
    let verify_time = start.elapsed();
    if verifier.state() != prover.state() {
        return Err("final transcript differs between prover and verifier".into());
    }
    let threads = pool.current_num_threads();
    writeln!(out, "configuration threads={threads} log_t={} rows_bytes={} allocator_baseline_bytes={} warmed_worker_allowance_allocs={} warmed_worker_allowance_bytes={}",
        geometry.log_T, std::mem::size_of_val(rows.as_ref()), observations.baseline,
        RAYON_WORKER_ALLOWANCE.allocs * threads, RAYON_WORKER_ALLOWANCE.bytes * threads)?;
    for (phase, time) in PHASES.into_iter().zip(observations.times) {
        writeln!(out, "phase {phase:?} ms={:.3}", time.as_secs_f64() * 1000.0)?;
    }
    writeln!(out, "commit_ms={:.3} open_ms={:.3} verify_ms={:.3} proof_bytes={} peak_requested_bytes={} peak_requested_mib={:.3} final_requested_bytes={} allocations={}",
        commit_time.as_secs_f64() * 1000.0, open_time.as_secs_f64() * 1000.0,
        verify_time.as_secs_f64() * 1000.0, bytes.len(), allocation.peak_bytes,
        allocation.peak_bytes as f64 / 1_048_576.0, allocation.final_bytes, allocation.allocs)?;
    for (index, level) in proof.levels.iter().enumerate() {
        writeln!(
            out,
            "verifier_level={index} leaves={} sibling_digests={} leaf_bytes={}",
            level.leaves.len(),
            level.digests.len(),
            level.leaves.first().map_or(0, Vec::len)
        )?;
    }
    if geometry.log_T == 22 {
        writeln!(out, "reference model_commit_ms=312.16 model_open_ms=340.43 model_level_zero_encode_ms=161.56 model_level_zero_tree_ms=136.84 model_commit_sample_ms=13.75 model_tables_weights_round_one_ms=130.42 model_later_rounds_ms=46.05 model_induced_weights_ms=6.98 model_equality_samples_ms=2.49 model_later_encodes_ms=65.27 model_later_trees_ms=88.42 model_queries_ms=0.78 expected_proof_bytes=326320.45 maximum_proof_bytes=337592 maximum_peak_requested_mib=604.8")?;
        writeln!(out, "tables_weights_round_one_ms={:.3} commit_ns_per_cycle={:.3} open_ns_per_cycle={:.3} single_thread_threshold_commit_ns=93 single_thread_threshold_open_ns=101 twelve_thread_threshold_commit_ms=85 twelve_thread_threshold_open_ms=83",
            observations.times[3..6].iter().sum::<Duration>().as_secs_f64() * 1000.0,
            commit_time.as_secs_f64() * 1e9 / rows.len() as f64,
            open_time.as_secs_f64() * 1e9 / rows.len() as f64)?;
    }
    for (point, live) in observations.releases.into_iter().flatten() {
        writeln!(
            out,
            "release {point:?} live_requested_bytes={live} live_requested_mib={:.3}",
            live as f64 / 1_048_576.0
        )?;
    }
    Ok(())
}

fn host_load(out: &mut impl Write, when: &str) -> IoResult<()> {
    match Command::new("uptime").output() {
        Ok(result) => writeln!(
            out,
            "host_load_{when} {}",
            String::from_utf8_lossy(&result.stdout).trim()
        ),
        Err(error) => writeln!(out, "host_load_{when} unavailable: {error}"),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut log_t = 22_usize;
    let mut threads = vec![1_usize, 12];
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--log-t" => log_t = args.next().ok_or("missing --log-t value")?.parse()?,
            "--threads" => {
                threads = args
                    .next()
                    .ok_or("missing --threads value")?
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<Vec<_>, _>>()?;
            }
            _ => return Err(format!("unknown argument {argument}").into()),
        }
    }
    if !(1..=32).contains(&log_t) || threads.is_empty() || threads.contains(&0) {
        return Err("log-t must be 1..=32 and thread counts must be positive".into());
    }
    let count = 1_usize
        .checked_shl(u32::try_from(log_t)?)
        .ok_or("row count overflow")?;
    let rows: Arc<[[u64; 4]]> = (0..count)
        .map(|row| std::array::from_fn(|word| seeded_word((4 * row + word) as u64)))
        .collect();
    let mut out = BufWriter::new(io::stdout().lock());
    host_load(&mut out, "before")?;
    out.flush()?;
    for count in threads {
        let pool = ThreadPoolBuilder::new().num_threads(count).build()?;
        sample(&mut out, &pool, BitsGeometry { log_T: log_t }, &rows)?;
        host_load(&mut out, "after")?;
        out.flush()?;
    }
    Ok(())
}

//! Sample rotation, median-of-sums and requested-capacity accounting for WHIR.

use super::allocator::{AllocationMeasurement, CountingAllocator};
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
use std::io::{self, BufWriter, Write};
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
const ROWS: [(&str, f64); 10] = [
    ("level_zero_encode", 161.56),
    ("level_zero_tree", 136.84),
    ("commit_sample", 13.75),
    ("tables_weights_round_one", 130.42),
    ("later_rounds", 46.05),
    ("induced_weights", 6.98),
    ("equality_tables_samples", 2.49),
    ("later_encodes", 65.27),
    ("later_trees", 88.42),
    ("queries_assembly", 0.78),
];

type BenchResult<T> = Result<T, Box<dyn Error>>;

pub struct Options {
    log_t: Vec<usize>,
    threads: Vec<usize>,
    samples: usize,
    warmup: usize,
    proof_samples: usize,
    filter: Option<String>,
}

impl Options {
    pub fn parse() -> BenchResult<Self> {
        let mut result = Self {
            log_t: vec![20, 22],
            threads: vec![1, 12],
            samples: 5,
            warmup: 3,
            proof_samples: 1000,
            filter: None,
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--bench" => {}
                "--log-t" | "--threads" | "--samples" | "--warmup" | "--proof-samples" => {
                    let value = args.next().ok_or("missing option value")?;
                    let values: Vec<usize> =
                        value.split(',').map(str::parse).collect::<Result<_, _>>()?;
                    match arg.as_str() {
                        "--log-t"
                            if !values.is_empty()
                                && values.iter().all(|v| [20, 22].contains(v)) =>
                        {
                            result.log_t = values;
                        }
                        "--threads"
                            if !values.is_empty() && values.iter().all(|v| [1, 12].contains(v)) =>
                        {
                            result.threads = values;
                        }
                        "--samples" if values.len() == 1 && values[0] > 0 => {
                            result.samples = values[0];
                        }
                        "--warmup" if values.len() == 1 => {
                            result.warmup = values[0];
                        }
                        "--proof-samples" if values.len() == 1 => {
                            result.proof_samples = values[0];
                        }
                        _ => return Err(format!("invalid {arg}: {value}").into()),
                    }
                }
                _ if arg.starts_with("bits_whir/") && result.filter.is_none() => {
                    result.filter = Some(arg);
                }
                _ => return Err(format!("unknown argument {arg}").into()),
            }
        }
        if !result.log_t.iter().any(|&log_t| {
            result.threads.iter().any(|&threads| {
                ["commit", "open"]
                    .iter()
                    .any(|name| result.selected(name, log_t, threads))
            })
        }) {
            return Err("id filter matches no cases".into());
        }
        Ok(result)
    }

    fn selected(&self, name: &str, log_t: usize, threads: usize) -> bool {
        self.filter.as_ref().is_none_or(|filter| {
            let id = format!("bits_whir/{name}/{log_t}/{threads}");
            id == *filter || id.starts_with(&format!("{filter}/"))
        })
    }
}

#[derive(Clone, Copy)]
struct Release {
    point: ReleasePoint,
    live: usize,
    peak: usize,
}

struct Observations {
    times: [Duration; 12],
    starts: [Option<Instant>; 12],
    releases: [Option<Release>; 64],
    count: usize,
    invalid: bool,
    baseline: usize,
}

impl Observations {
    fn new() -> Self {
        Self {
            times: [Duration::ZERO; 12],
            starts: [None; 12],
            releases: [None; 64],
            count: 0,
            invalid: false,
            baseline: CountingAllocator::live_bytes(),
        }
    }

    fn observe(&mut self, event: Event) {
        match event {
            Event::Start(phase) => {
                if let Some(index) = PHASES.iter().position(|candidate| *candidate == phase) {
                    self.invalid |= self.starts[index].is_some();
                    CountingAllocator::phase(index);
                    self.starts[index] = Some(Instant::now());
                }
            }
            Event::End(phase) => {
                if let Some(index) = PHASES.iter().position(|candidate| *candidate == phase) {
                    if let Some(start) = self.starts[index].take() {
                        self.times[index] += start.elapsed();
                    } else {
                        self.invalid = true;
                    }
                }
            }
            Event::Released(point) => {
                if let Some(slot) = self.releases.get_mut(self.count) {
                    *slot = Some(Release {
                        point,
                        live: CountingAllocator::live_bytes().saturating_sub(self.baseline),
                        peak: CountingAllocator::peak_bytes().saturating_sub(self.baseline),
                    });
                    self.count += 1;
                } else {
                    self.invalid = true;
                }
            }
        }
    }

    fn phase_table(&self) -> [Duration; 10] {
        [
            self.times[0],
            self.times[1],
            self.times[2],
            self.times[3..6].iter().sum(),
            self.times[6],
            self.times[7],
            self.times[8],
            self.times[9],
            self.times[10],
            self.times[11],
        ]
    }
}

struct Sample {
    observations: Observations,
    commit: Duration,
    open: Duration,
    verify: Duration,
    proof_bytes: usize,
    commitment_bytes: usize,
    peak: usize,
    final_bytes: usize,
    allocs: usize,
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

fn transcript(geometry: BitsGeometry, nonce: u64) -> Rv64iTranscript {
    let mut transcript = Rv64iTranscript::new(b"whir-benchmark");
    transcript.append(&Label(b"params"));
    transcript.append(&U64Word(geometry.log_T as u64));
    transcript.append(&Label(b"benchmark_nonce"));
    transcript.append(&U64Word(nonce));
    transcript
}

fn sample(
    pool: &ThreadPool,
    geometry: BitsGeometry,
    rows: &Arc<[[u64; 4]]>,
    nonce: u64,
) -> BenchResult<Sample> {
    let mut prover = transcript(geometry, nonce);
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
    if observations.invalid || observations.starts.iter().any(Option::is_some) {
        return Err("phase or release recorder capacity/grammar mismatch".into());
    }
    if geometry.log_T == 22 && allocation.peak_bytes as f64 > 604.8 * 1_048_576.0 {
        return Err(format!(
            "peak requested bytes {} exceeds 604.8 MiB",
            allocation.peak_bytes
        )
        .into());
    }
    let mut bytes = Vec::new();
    proof.write(&mut bytes);
    let proof_bytes = bytes.len();
    if geometry.log_T == 22 && proof_bytes > 337_592 {
        return Err(format!("opening size {proof_bytes} exceeds 337592 bytes").into());
    }
    bytes.clear();
    commitment.write(&mut bytes);
    let commitment_bytes = bytes.len();
    let mut verifier = transcript(geometry, nonce);
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
    Ok(Sample {
        observations,
        commit: commit_time,
        open: open_time,
        verify: verify_time,
        proof_bytes,
        commitment_bytes,
        peak: allocation.peak_bytes,
        final_bytes: allocation.final_bytes,
        allocs: allocation.allocs,
    })
}

fn load() -> String {
    match Command::new("uptime").output() {
        Ok(result) => String::from_utf8_lossy(&result.stdout).trim().to_owned(),
        Err(error) => format!("unavailable: {error}"),
    }
}

fn distribution(mut values: Vec<f64>) -> BenchResult<(f64, f64, f64)> {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    let median = if values.len().is_multiple_of(2) {
        values
            .get(middle - 1)
            .ok_or("empty distribution")?
            .midpoint(*values.get(middle).ok_or("empty distribution")?)
    } else {
        *values.get(middle).ok_or("empty distribution")?
    };
    Ok((
        median,
        *values.first().ok_or("empty distribution")?,
        *values.last().ok_or("empty distribution")?,
    ))
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn print_sample(
    out: &mut impl Write,
    geometry: BitsGeometry,
    threads: usize,
    index: usize,
    sample: &Sample,
    before: &str,
    after: &str,
) -> BenchResult<()> {
    writeln!(out, "bits_whir/sample/{}/{threads} sample={index} load_before={before:?} load_after={after:?} commit_ms={:.3} open_ms={:.3} verify_ms={:.3} peak_requested_bytes={} final_requested_bytes={} allocations={} proof_bytes={} commitment_bytes={}", geometry.log_T,
        milliseconds(sample.commit), milliseconds(sample.open), milliseconds(sample.verify), sample.peak, sample.final_bytes, sample.allocs, sample.proof_bytes, sample.commitment_bytes)?;
    for ((label, model), duration) in ROWS.into_iter().zip(sample.observations.phase_table()) {
        writeln!(out, "bits_whir/phase/{}/{threads} sample={index} phase={label} ms={:.3} model_t22_one_thread_ms={model:.2} load_before={before:?} load_after={after:?}", geometry.log_T, milliseconds(duration))?;
    }
    for release in sample.observations.releases.iter().flatten() {
        writeln!(out, "bits_whir/release/{}/{threads} sample={index} point={:?} live_requested_bytes={} peak_requested_bytes={} load_before={before:?} load_after={after:?}", geometry.log_T, release.point, release.live, release.peak)?;
    }
    Ok(())
}

fn print_summary(
    out: &mut impl Write,
    options: &Options,
    geometry: BitsGeometry,
    threads: usize,
    samples: &[Sample],
) -> BenchResult<()> {
    let host_load = load();
    for name in ["commit", "open"] {
        if !options.selected(name, geometry.log_T, threads) {
            continue;
        }
        let (median, min, max) = distribution(
            samples
                .iter()
                .map(|sample| {
                    milliseconds(if name == "commit" {
                        sample.commit
                    } else {
                        sample.open
                    })
                })
                .collect(),
        )?;
        let ns_per_cycle = median * 1e6 / (1_u64 << geometry.log_T) as f64;
        let threshold_ms = (geometry.log_T == 22).then(|| {
            if threads == 1 {
                (if name == "commit" { 93.0 } else { 101.0 }) * (1_u64 << 22) as f64 / 1e6
            } else if name == "commit" {
                85.0
            } else {
                83.0
            }
        });
        let target_ms = (geometry.log_T == 22 && threads == 12).then_some(if name == "commit" {
            60.0
        } else {
            83.0
        });
        let meets_spec = threshold_ms.map(|threshold| median <= threshold);
        let meets_target = target_ms.map(|target| median <= target);
        writeln!(out, "bits_whir/{name}/{}/{threads} samples={} median_ms={median:.3} min_ms={min:.3} max_ms={max:.3} ns_per_cycle={ns_per_cycle:.3} model_t22_one_thread_ms={:.2} spec_threshold_ms={threshold_ms:?} meets_spec_threshold={meets_spec:?} requested_target_ms={target_ms:?} meets_requested_target={meets_target:?} load={host_load:?}", geometry.log_T, samples.len(), if name == "commit" {312.16} else {340.43})?;
    }
    for (phase, (label, model)) in ROWS.into_iter().enumerate() {
        let (median, min, max) = distribution(
            samples
                .iter()
                .map(|sample| milliseconds(sample.observations.phase_table()[phase]))
                .collect(),
        )?;
        writeln!(out, "bits_whir/phase_summary/{}/{threads} phase={label} median_ms={median:.3} min_ms={min:.3} max_ms={max:.3} model_t22_one_thread_ms={model:.2} load={host_load:?}", geometry.log_T)?;
    }
    let (bridge, _, _) = distribution(
        samples
            .iter()
            .map(|sample| {
                milliseconds(
                    sample.observations.phase_table()[3] + sample.observations.phase_table()[4],
                )
            })
            .collect(),
    )?;
    let (verify, _, _) = distribution(
        samples
            .iter()
            .map(|sample| milliseconds(sample.verify))
            .collect(),
    )?;
    writeln!(out, "bits_whir/summary/{}/{threads} bridge_ms={bridge:.3} bridge_model_t22_one_thread_ms=176.48 verify_ms={verify:.3} peak_requested_bytes={} max_final_requested_bytes={} load={host_load:?}", geometry.log_T, samples.iter().map(|sample| sample.peak).max().unwrap_or(0), samples.iter().map(|sample| sample.final_bytes).max().unwrap_or(0))?;
    Ok(())
}

fn proof_distribution(
    out: &mut impl Write,
    geometry: BitsGeometry,
    rows: &Arc<[[u64; 4]]>,
    pool: &ThreadPool,
    count: usize,
) -> BenchResult<()> {
    let before = load();
    if count == 0 {
        writeln!(
            out,
            "bits_whir/proof_size/{} transcripts=0 acceptance_complete=false load={before:?}",
            geometry.log_T
        )?;
        return Ok(());
    }
    let mut sizes = Vec::with_capacity(count);
    let mut verifier_times = Vec::with_capacity(count);
    for nonce in 0..count {
        let result = sample(pool, geometry, rows, 1_000_000 + u64::try_from(nonce)?)?;
        sizes.push(result.proof_bytes);
        verifier_times.push(milliseconds(result.verify));
        if (nonce + 1).is_multiple_of(100) {
            writeln!(
                out,
                "bits_whir/proof_progress/{} verified={} requested={count} load={:?}",
                geometry.log_T,
                nonce + 1,
                load()
            )?;
            out.flush()?;
        }
    }
    sizes.sort_unstable();
    let mean = sizes.iter().map(|&size| size as f64).sum::<f64>() / count as f64;
    let quantile = |percentage: usize| -> BenchResult<usize> {
        let rank = count
            .checked_mul(percentage)
            .ok_or("quantile rank overflow")?
            .div_ceil(100)
            .saturating_sub(1);
        Ok(*sizes.get(rank).ok_or("empty proof distribution")?)
    };
    let (verify, _, _) = distribution(verifier_times)?;
    let within_mean = (mean - 326_320.45).abs() <= 326_320.45 * 0.01;
    let within_max = *sizes.last().ok_or("empty proof distribution")? <= 337_592;
    let after = load();
    writeln!(out, "bits_whir/proof_size/{} threads={} transcripts={count} verified={count} min={} mean={mean:.3} p50={} p90={} p95={} p99={} max={} verifier_median_ms={verify:.3} expected_t22_bytes=326320.45 maximum_t22_bytes=337592 mean_within_one_percent={within_mean} all_within_maximum={within_max} acceptance_complete={} load_before={before:?} load_after={after:?}", geometry.log_T, pool.current_num_threads(), sizes.first().ok_or("empty proof distribution")?, quantile(50)?, quantile(90)?, quantile(95)?, quantile(99)?, sizes.last().ok_or("empty proof distribution")?, count == 1000 && geometry.log_T == 22)?;
    if geometry.log_T == 22 && (!within_mean || !within_max) {
        return Err("proof size distribution violates the Performance acceptance bounds".into());
    }
    Ok(())
}

pub fn run(options: Options) -> BenchResult<()> {
    let mut out = BufWriter::new(io::stdout().lock());
    writeln!(out, "bits_whir_note interval=combined_commit_open rows=shared_once frontend_columns=in_interval_outside_timers domain_setup=inside_phase bridge_setup=inside_phase counters=requested_capacity proof_alive_at_interval_end=true loaded_machine=true")?;
    writeln!(out, "bits_whir/model_correction counted_extra_sample_scaling_products_e=279616 counted_extra_c=3355392 estimated_extra_ms_point1=1.02339456 equality_model_spec_ms=2.49 equality_model_accounted_ms=3.51793344 open_model_accounted_ms=341.45146446 thresholds_unchanged=true")?;
    for &exponent in &options.log_t {
        let selected: Vec<_> = options
            .threads
            .iter()
            .copied()
            .filter(|&threads| {
                ["commit", "open"]
                    .iter()
                    .any(|name| options.selected(name, exponent, threads))
            })
            .collect();
        if selected.is_empty() {
            continue;
        }
        let geometry = BitsGeometry { log_T: exponent };
        let count = 1_usize
            .checked_shl(u32::try_from(exponent)?)
            .ok_or("row count overflow")?;
        let rows: Arc<[[u64; 4]]> = (0..count)
            .map(|row| std::array::from_fn(|word| seeded_word((4 * row + word) as u64)))
            .collect();
        let pools = selected
            .iter()
            .map(|&threads| ThreadPoolBuilder::new().num_threads(threads).build())
            .collect::<Result<Vec<_>, _>>()?;
        writeln!(
            out,
            "bits_whir/setup/{exponent} rows_bytes={} rows_reported_once=true load={:?}",
            std::mem::size_of_val(rows.as_ref()),
            load()
        )?;
        for pool in &pools {
            let _ = pool.broadcast(|_| black_box([0_u8; 32]));
            let _ = pool.install(|| {
                black_box(
                    (0..16_384)
                        .into_par_iter()
                        .map(seeded_word)
                        .reduce(|| 0, u64::wrapping_add),
                )
            });
            for warm in 0..options.warmup {
                let _ = sample(pool, geometry, &rows, u64::try_from(warm)?)?;
            }
        }
        let mut collected: Vec<Vec<Sample>> = pools
            .iter()
            .map(|_| Vec::with_capacity(options.samples))
            .collect();
        for index in 0..options.samples {
            for (slot, pool) in pools.iter().enumerate() {
                let before = load();
                let result = sample(pool, geometry, &rows, 10_000 + u64::try_from(index)?)?;
                let after = load();
                print_sample(
                    &mut out,
                    geometry,
                    pool.current_num_threads(),
                    index,
                    &result,
                    &before,
                    &after,
                )?;
                out.flush()?;
                collected[slot].push(result);
            }
        }
        for (pool, samples) in pools.iter().zip(&collected) {
            print_summary(
                &mut out,
                &options,
                geometry,
                pool.current_num_threads(),
                samples,
            )?;
        }
        let distribution_pool = pools
            .iter()
            .max_by_key(|pool| pool.current_num_threads())
            .ok_or("no selected pool")?;
        out.flush()?;
        proof_distribution(
            &mut out,
            geometry,
            &rows,
            distribution_pool,
            options.proof_samples,
        )?;
        out.flush()?;
    }
    Ok(())
}

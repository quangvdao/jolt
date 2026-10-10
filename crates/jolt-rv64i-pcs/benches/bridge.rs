//! `RUSTFLAGS='-C target-cpu=native' cargo bench -p jolt-rv64i-pcs
//! --bench bridge -- --log-t 22 --threads 1,12 --samples 5`.
//! Cases alternate five/six products on identical seeded rows and warmed pools.
//! Timings include allocation and zero initialization; requested-byte counters
//! are shared with the kernel runner. All records are loaded-host evidence.

#[path = "../../jolt-rv64i-kernels/benches/support/allocator.rs"]
mod allocator;

use allocator::{
    AllocationMeasurement, AllocationStats, CountingAllocator, RAYON_WORKER_ALLOWANCE,
};
use jolt_field::{Accumulator, ExtField, WithAccumulator, Zero, F128, F192, F64};
use jolt_rv64i_pcs::bridge::{BridgeTables, FoldedBridge, PhiTables};
use jolt_rv64i_verifier::whir::error::{try_vec, WhirError, WhirPart};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::error::Error;
use std::hint::black_box;
use std::io::{self, Result as IoResult, Write};
use std::time::{Duration, Instant};

type ProductAccumulator = <F192 as WithAccumulator>::Accumulator;

struct Sample {
    setup: Duration,
    pass: Duration,
    fold: Duration,
    allocation: AllocationStats,
    release: [usize; 3],
}

fn sample(
    rows: &[[u64; 4]],
    point: &[F128],
    alpha: F192,
    challenge: F192,
    composed: bool,
) -> Result<Sample, WhirError> {
    let baseline = CountingAllocator::live_bytes();
    let measurement = AllocationMeasurement::begin();
    let start = Instant::now();
    let tables = BridgeTables::new(black_box(point), black_box(alpha))?;
    let setup = start.elapsed();
    let after_setup = CountingAllocator::live_bytes().saturating_sub(baseline);
    let start = Instant::now();
    let first = if composed {
        tables.first_round_composed(black_box(rows))?
    } else {
        tables.first_round(black_box(rows))?
    };
    let pass = start.elapsed();
    let after_pass = CountingAllocator::live_bytes().saturating_sub(baseline);
    let _ = black_box(first.coefficients());
    let start = Instant::now();
    let folded = first.fold(black_box(rows), black_box(challenge))?;
    let fold = start.elapsed();
    let after_fold = CountingAllocator::live_bytes().saturating_sub(baseline);
    let _ = black_box(&folded);
    let allocation = measurement.finish();
    Ok(Sample {
        setup,
        pass,
        fold,
        allocation,
        release: [after_setup, after_pass, after_fold],
    })
}

fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn summary(mut values: Vec<f64>) -> [f64; 3] {
    values.sort_by(f64::total_cmp);
    [
        values[0],
        values[values.len() / 2],
        values[values.len() - 1],
    ]
}

fn report(
    out: &mut impl Write,
    name: &str,
    times: &[Duration],
    operations: usize,
    divisor: usize,
    threads: usize,
) -> IoResult<()> {
    let [min, median, max] = summary(
        times
            .iter()
            .map(|t| t.as_secs_f64() * 1e9 / operations as f64)
            .collect(),
    );
    writeln!(out, "bridge/{name}/{threads} ns_per_pair={median:.6} ns_per_cycle={median:.6} unit_ns={:.6} min_ns={min:.6} max_ns={max:.6} samples={} loaded_machine=true", median / divisor as f64, times.len())
}

fn units(rows: &[[u64; 4]], inputs: &[F128; 4096], phi: &PhiTables) -> [Duration; 5] {
    let start = Instant::now();
    let lookup = (0..rows.len())
        .into_par_iter()
        .with_min_len(4096)
        .fold(
            || [F192::zero(); 2],
            |mut sum, k| {
                let pair = phi.evaluate(black_box(inputs[k % inputs.len()]));
                sum[0] += pair[0];
                sum[1] += pair[1];
                sum
            },
        )
        .reduce(|| [F192::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]]);
    let _ = black_box(lookup);
    let lookup_time = start.elapsed();
    let start = Instant::now();
    let product: F128 = rows
        .par_iter()
        .map(|row| {
            let a = F128::from_raw(u128::from(row[0]) | (u128::from(row[1]) << 64));
            let b = F128::from_raw(u128::from(row[2]) | (u128::from(row[3]) << 64));
            black_box(a) * black_box(b)
        })
        .sum();
    let _ = black_box(product);
    let h_time = start.elapsed();
    let start = Instant::now();
    let product: F64 = rows
        .par_iter()
        .map(|row| black_box(F64::from_raw(row[0])) * black_box(F64::from_raw(row[1])))
        .sum();
    let _ = black_box(product);
    let k_time = start.elapsed();
    let mut ev = [Duration::ZERO; 2];
    for (composed, time) in ev.iter_mut().enumerate() {
        let start = Instant::now();
        let product = rows
            .par_iter()
            .with_min_len(1024)
            .fold(ProductAccumulator::default, |mut sum, row| {
                let e = F192::from_base_fn(|i| F64::from_raw(row[i]));
                let pair = [F64::from_raw(row[2]), F64::from_raw(row[3])];
                if composed == 0 {
                    sum.fmadd_base_pair(black_box(e), black_box(pair));
                } else {
                    sum.fmadd_base(black_box(e), black_box(pair[0]));
                    sum.fmadd_base(black_box(e.mul_y()), black_box(pair[1]));
                }
                sum
            })
            .reduce(ProductAccumulator::default, |mut a, b| {
                a.merge(b);
                a
            })
            .reduce();
        let _ = black_box(product);
        *time = start.elapsed();
    }
    [lookup_time, h_time, k_time, ev[0], ev[1]]
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut log_t = 22usize;
    let mut threads = vec![1usize, 12];
    let mut samples = 5usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bench" => {}
            "--log-t" => {
                log_t = args.next().ok_or("missing log-t")?.parse()?;
            }
            "--threads" => {
                threads = args
                    .next()
                    .ok_or("missing threads")?
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<_, _>>()?;
            }
            "--samples" => {
                samples = args.next().ok_or("missing samples")?.parse()?;
            }
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    if ![20, 22].contains(&log_t) || samples == 0 || !threads.iter().all(|t| [1, 12].contains(t)) {
        return Err("expected log-t 20/22, threads 1/12 and positive samples".into());
    }
    let cycles = 1usize << log_t;
    let mut seed = 0x6119_8793_aa83_1791;
    let mut rows = try_vec::<[u64; 4]>(WhirPart::Rows, cycles)?;
    rows.resize(cycles, [0; 4]);
    for row in &mut rows {
        *row = std::array::from_fn(|_| next(&mut seed));
    }
    let point: Vec<_> = (0..=log_t)
        .map(|_| F128::from_raw(u128::from(next(&mut seed)) | (u128::from(next(&mut seed)) << 64)))
        .collect();
    let alpha = F192::from_base_fn(|_| F64::from_raw(next(&mut seed)));
    let challenge = F192::from_base_fn(|_| F64::from_raw(next(&mut seed)));
    let inputs = std::array::from_fn(|_| {
        F128::from_raw(u128::from(next(&mut seed)) | (u128::from(next(&mut seed)) << 64))
    });
    let phi = PhiTables::new(alpha, point[0])?;
    let stdout = io::stdout();
    let mut out = stdout.lock();
    writeln!(out, "bridge/inventory log_t={log_t} pairs={cycles} rows_bytes={} paired_table_bytes=196608 split_bytes={} logical_lookup_bytes={} split_read_bytes={} output_bytes={} zero_init_bytes={} model_first_round_ms=130.42188288 model_first_pass_ms=100.99884032 model_point1_c_ns=0.305 model_Lw_ns=0.6 loaded_machine=true", cycles * 32, ((1 << (log_t / 2)) + (1 << (log_t - log_t / 2))) * 16, cycles * 16 * 48, cycles * 32, cycles * 48, cycles * 48)?;
    for threads in threads {
        let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
        pool.install(|| {
            let _ = black_box(rows.par_iter().map(|r| r[0]).reduce(|| 0, |a, b| a ^ b));
        });
        let mut measured: [Vec<Sample>; 2] = std::array::from_fn(|_| Vec::with_capacity(samples));
        let mut unit_times: [Vec<Duration>; 5] =
            std::array::from_fn(|_| Vec::with_capacity(samples));
        let mut dense_times = Vec::with_capacity(samples);
        for sample_index in 0..samples {
            for offset in 0..2 {
                let composed = (sample_index + offset) % 2;
                measured[composed]
                    .push(pool.install(|| sample(&rows, &point, alpha, challenge, composed == 1))?);
            }
            for (times, duration) in unit_times
                .iter_mut()
                .zip(pool.install(|| units(&rows, &inputs, &phi)))
            {
                times.push(duration);
            }
            dense_times.push(pool.install(|| {
                let start = Instant::now();
                for _ in 0..4096 {
                    let dense = FoldedBridge::dense(
                        black_box(&[rows[0], rows[1]]),
                        black_box(&[point[0], point[1]]),
                        black_box(alpha),
                    )?;
                    let _ = black_box(dense);
                }
                Ok::<_, WhirError>(start.elapsed())
            })?);
        }
        for (variant, values) in ["five", "six"].into_iter().zip(&measured) {
            for (phase, select) in [
                ("setup", (|s: &Sample| s.setup) as fn(&Sample) -> Duration),
                ("pass", |s: &Sample| s.pass),
                ("fold", |s: &Sample| s.fold),
                ("first_round", |s: &Sample| s.setup + s.pass + s.fold),
            ] {
                let times: Vec<_> = values.iter().map(select).collect();
                report(
                    &mut out,
                    &format!("{variant}/{phase}"),
                    &times,
                    cycles,
                    1,
                    threads,
                )?;
            }
            let peak = values
                .iter()
                .map(|s| s.allocation.peak_bytes)
                .max()
                .ok_or("empty samples")?;
            let allocs = values
                .iter()
                .map(|s| s.allocation.allocs)
                .max()
                .ok_or("empty samples")?;
            let final_bytes = values
                .iter()
                .map(|s| s.allocation.final_bytes)
                .max()
                .ok_or("empty samples")?;
            let release = values[0].release;
            writeln!(out, "bridge/{variant}/allocation/{threads} peak_bytes={peak} final_bytes={final_bytes} allocs={allocs} rayon_lazy_bytes={} rayon_lazy_allocs={} after_setup_bytes={} after_pass_bytes={} after_fold_bytes={} rows_shared_bytes={} loaded_machine=true", RAYON_WORKER_ALLOWANCE.bytes * threads, RAYON_WORKER_ALLOWANCE.allocs * threads, release[0], release[1], release[2], cycles * 32)?;
        }
        for ((name, divisor), times) in [
            ("Lw", 32),
            ("F128_product", 1),
            ("F64_product_c", 3),
            ("Ev_five", 1),
            ("Ev_six", 1),
        ]
        .into_iter()
        .zip(&unit_times)
        {
            report(&mut out, name, times, cycles, divisor, threads)?;
        }
        let [min, median, max] = summary(
            dense_times
                .iter()
                .map(|t| t.as_secs_f64() * 1e9 / 4096.0)
                .collect(),
        );
        writeln!(out, "bridge/dense_k0_zero/{threads} ns_per_recipe={median:.6} min_ns={min:.6} max_ns={max:.6} symbols=4 loaded_machine=true")?;
        let five = summary(
            measured[0]
                .iter()
                .map(|s| s.pass.as_secs_f64() * 1e9 / cycles as f64)
                .collect(),
        );
        let six = summary(
            measured[1]
                .iter()
                .map(|s| s.pass.as_secs_f64() * 1e9 / cycles as f64)
                .collect(),
        );
        let gain = six[1] - five[1];
        let spread = five[2] - five[0] + six[2] - six[0];
        writeln!(out, "bridge/comparison/{threads} five_gain_ns={gain:.6} combined_spread_ns={spread:.6} gain_exceeds_spread={} loaded_machine=true", gain > spread)?;
    }
    Ok(())
}

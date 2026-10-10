//! Transform measurements at `t=20,22`, with fixtures and warm pools outside timing.
//! `cargo bench -p jolt-rv64i-pcs --bench ntt -- --samples 5` reports CSV rows.
//! Preserve that executable, rebuild after a change, then run the new executable
//! with `--compare <baseline-executable> --log-t 20,22 --threads 1,12 --samples 5`.
//! Each batch runs one sample in each executable, alternating baseline/current
//! order. Comparison CSV reports medians, minima, maxima, and paired ratios.
//! Samples include output allocation, first touch and the call-scoped domain;
//! later levels share one table. `--load` reports uptime outside measurement.
//! `--arithmetic --iterations 8000000 --samples 5` measures fully reduced
//! K multiplication and E-by-K scaling throughput on independent register chains.

use jolt_field::{ExtField, Zero};
use jolt_field::{F192, F64};
use jolt_rv64i_pcs::ntt::Encoder;
use jolt_rv64i_verifier::commitment::BitsGeometry;
use jolt_rv64i_verifier::whir::code::DomainTable;
use jolt_rv64i_verifier::whir::error::{try_vec, WhirError, WhirPart};
use jolt_rv64i_verifier::whir::params::{Level, Schedule};
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use std::collections::BTreeMap;
use std::error::Error;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

struct Words(u64);
impl Words {
    fn gen(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }
}

struct Fixture {
    rows: Vec<[u64; 4]>,
    later: Vec<Vec<F192>>,
    levels: Vec<Level>,
}
struct Sample {
    phases: Vec<Duration>,
    setup: Duration,
    later: Duration,
}
impl Fixture {
    fn new(log_t: usize) -> Result<Self, Box<dyn Error>> {
        let levels = Schedule::new(BitsGeometry { log_T: log_t })?
            .levels()
            .to_vec();
        let mut rng = Words(0x004e_5454);
        let mut rows = try_vec(WhirPart::Rows, 1 << log_t)?;
        rows.extend((0..1 << log_t).map(|_| std::array::from_fn(|_| rng.gen())));
        let mut later = Vec::new();
        for level in levels.iter().skip(1) {
            let len = level.lanes()? * (1 << level.c);
            let mut message = try_vec(WhirPart::FinalValues, len)?;
            message.extend((0..len).map(|_| F192::from_base_fn(|_| F64::from_raw(rng.gen()))));
            later.push(message);
        }
        Ok(Self {
            rows,
            later,
            levels,
        })
    }

    fn sample(&self, pool: &ThreadPool) -> Result<Sample, Box<dyn Error>> {
        pool.install(|| {
            let level0 = self.levels.first().ok_or(WhirError::Shape {
                part: WhirPart::Levels,
                expected: 1,
                actual: 0,
            })?;
            let start = Instant::now();
            let table = DomainTable::new(level0.c, level0.d)?;
            let setup_commit = start.elapsed();
            let code = Encoder::new(&table, level0.c, level0.d, level0.lanes()?)?
                .encode_rows(&self.rows)?;
            let _code = black_box(&code);
            let level0_time = start.elapsed();
            drop(code);
            let start = Instant::now();
            let table = DomainTable::new(level0.c, level0.d)?;
            let setup_open = start.elapsed();
            let mut phases = vec![level0_time];
            for (level, message) in self.levels.iter().skip(1).zip(&self.later) {
                let phase = Instant::now();
                let code = Encoder::new(&table, level.c, level.d, level.lanes()?)?
                    .encode_extension(message)?;
                let _code = black_box(&code);
                phases.push(phase.elapsed());
                drop(code);
            }
            let later = setup_open + phases.iter().skip(1).sum::<Duration>();
            Ok::<_, WhirError>(Sample {
                phases,
                setup: setup_commit + setup_open,
                later,
            })
        })
        .map_err(Into::into)
    }

    // Provisioning overlaps transform traffic, so subtraction is only an estimate.
    fn copy_baseline(&self, pool: &ThreadPool) -> Result<Vec<Duration>, Box<dyn Error>> {
        pool.install(|| {
            let start = Instant::now();
            let level0 = self.levels.first().ok_or(WhirError::Shape {
                part: WhirPart::Levels,
                expected: 1,
                actual: 0,
            })?;
            let len = 2 * level0.lanes()? * (1 << level0.d);
            let mut code = try_vec(WhirPart::Leaves, len)?;
            code.resize(len, F64::zero());
            code.par_chunks_mut(self.rows.len() * 4).for_each(|coset| {
                for (r, out) in self.rows.iter().zip(coset.chunks_exact_mut(4)) {
                    for (word, slot) in r.iter().zip(out) {
                        *slot = F64::from_raw(*word);
                    }
                }
            });
            let _code = black_box(&code);
            let time = start.elapsed();
            drop(code);
            let mut times = vec![time];
            for (level, message) in self.levels.iter().skip(1).zip(&self.later) {
                let start = Instant::now();
                let len = level.lanes()? * (1usize << level.d);
                let mut code = try_vec(WhirPart::Leaves, len)?;
                code.resize(len, F192::zero());
                code.par_chunks_mut(message.len())
                    .for_each(|block| block.copy_from_slice(message));
                let _code = black_box(&code);
                times.push(start.elapsed());
                drop(code);
            }
            Ok::<_, WhirError>(times)
        })
        .map_err(Into::into)
    }
}

struct Options {
    samples: usize,
    sizes: Vec<usize>,
    threads: Vec<usize>,
    compare: Option<PathBuf>,
    load: bool,
    copy: bool,
    arithmetic: bool,
    iterations: usize,
}
impl Options {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut options = Self {
            samples: 5,
            sizes: vec![20, 22],
            threads: vec![1, 12],
            compare: None,
            load: false,
            copy: true,
            arithmetic: false,
            iterations: 8_000_000,
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--bench" => {}
                "--samples" => options.samples = args.next().ok_or("missing samples")?.parse()?,
                "--log-t" => options.sizes = parse_list(&args.next().ok_or("missing log-t")?)?,
                "--threads" => {
                    options.threads = parse_list(&args.next().ok_or("missing threads")?)?;
                }
                "--compare" => {
                    options.compare = Some(args.next().ok_or("missing baseline")?.into());
                }
                "--load" => options.load = true,
                "--no-copy" => options.copy = false,
                "--arithmetic" => options.arithmetic = true,
                "--iterations" => {
                    options.iterations = args.next().ok_or("missing iterations")?.parse()?;
                }
                _ => return Err(format!("unknown argument {arg}").into()),
            }
        }
        if options.samples == 0 || options.iterations == 0 || options.threads.contains(&0) {
            return Err("samples, iterations and threads must be positive".into());
        }
        if options.arithmetic && options.compare.is_some() {
            return Err("arithmetic and executable comparison are separate measurements".into());
        }
        if options.sizes.iter().any(|t| ![20, 22].contains(t)) {
            return Err("log-t must select 20 or 22".into());
        }
        Ok(options)
    }
}

fn parse_list(text: &str) -> Result<Vec<usize>, Box<dyn Error>> {
    Ok(text.split(',').map(str::parse).collect::<Result<_, _>>()?)
}

fn stats(values: &mut [f64]) -> (f64, f64, f64) {
    values.sort_by(f64::total_cmp);
    (
        values[values.len() / 2],
        values[0],
        values[values.len() - 1],
    )
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

#[expect(
    clippy::print_stdout,
    reason = "benchmark measurements are stdout output"
)]
fn report(options: &Options) -> Result<(), Box<dyn Error>> {
    println!("t,threads,level,c,d,butterflies,ms,Gbutterflies/s,ns/butterfly,1t_model_ms,1t_model_x1.25_ms,source_ms,min_ms,max_ms");
    for &log_t in &options.sizes {
        let fixture = Fixture::new(log_t)?;
        for &threads in &options.threads {
            let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
            let _warm = pool.install(|| rayon::broadcast(|_| black_box(0)));
            let _warm = black_box(fixture.sample(&pool)?);
            let mut phases: Vec<Vec<f64>> = fixture.levels.iter().map(|_| Vec::new()).collect();
            let mut setup = Vec::new();
            let mut sums = Vec::new();
            let mut copies = phases.clone();
            for _ in 0..options.samples {
                let sample = fixture.sample(&pool)?;
                for (times, duration) in phases.iter_mut().zip(sample.phases) {
                    times.push(milliseconds(duration));
                }
                setup.push(milliseconds(sample.setup));
                sums.push(milliseconds(sample.later));
                if options.copy {
                    for (times, duration) in copies.iter_mut().zip(fixture.copy_baseline(&pool)?) {
                        times.push(milliseconds(duration));
                    }
                }
            }
            let mut later_count = 0usize;
            for (i, (times, level)) in phases.iter_mut().zip(&fixture.levels).enumerate() {
                let (ms, min, max) = stats(times);
                let count =
                    level.lanes()? * (1usize << level.d) * level.c / 2 * if i == 0 { 2 } else { 1 };
                if i != 0 {
                    later_count += count;
                }
                let price = if i == 0 { 0.535 } else { 1.50 };
                let model = count as f64 * price / 1e6;
                let source = if log_t == 22 && i == 0 {
                    if threads == 1 {
                        "161.58"
                    } else if threads == 12 {
                        "37.71"
                    } else {
                        "unmeasured"
                    }
                } else {
                    "aggregate-only"
                };
                println!("{log_t},{threads},{i},{},{},{count},{ms:.6},{:.6},{:.6},{model:.6},{:.6},{source},{min:.6},{max:.6}",
                    level.c, level.d, count as f64 / ms / 1e6, ms * 1e6 / count as f64, model * 1.25);
            }
            let (ms, min, max) = stats(&mut sums);
            let model = later_count as f64 * 1.50 / 1e6;
            let source = match (log_t, threads) {
                (22, 1) => "65.37",
                (22, 12) => "11.74",
                _ => "unmeasured",
            };
            println!("{log_t},{threads},later-total,-,-,{later_count},{ms:.6},{:.6},{:.6},{model:.6},{:.6},{source},{min:.6},{max:.6}",
                later_count as f64 / ms / 1e6, ms * 1e6 / later_count as f64, model * 1.25);
            println!(
                "# t={log_t},threads={threads},constants_ms={:.6}",
                stats(&mut setup).0
            );
            if options.copy {
                for (i, times) in copies.iter_mut().enumerate() {
                    println!(
                        "# t={log_t},threads={threads},level={i},copy_ms={:.6}",
                        stats(times).0
                    );
                }
            }
        }
    }
    Ok(())
}

fn child_sample(
    executable: &Path,
    log_t: usize,
    threads: usize,
) -> Result<BTreeMap<String, f64>, Box<dyn Error>> {
    let output = Command::new(executable)
        .args([
            "--samples",
            "1",
            "--log-t",
            &log_t.to_string(),
            "--threads",
            &threads.to_string(),
            "--no-copy",
        ])
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "benchmark child failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    let mut rows = BTreeMap::new();
    for line in String::from_utf8(output.stdout)?.lines() {
        let fields: Vec<_> = line.split(',').collect();
        if fields.len() == 15 && fields.first() == Some(&log_t.to_string().as_str()) {
            let _previous = rows.insert(fields[2].to_owned(), fields[6].parse()?);
        }
    }
    if rows.is_empty() {
        return Err("benchmark child emitted no measurement rows".into());
    }
    Ok(rows)
}

#[derive(Default)]
struct Comparison {
    baseline: Vec<f64>,
    current: Vec<f64>,
    ratios: Vec<f64>,
}

#[expect(
    clippy::print_stdout,
    reason = "benchmark comparison CSV is stdout output"
)]
fn compare(options: &Options, baseline: &Path) -> Result<(), Box<dyn Error>> {
    let current = std::env::current_exe()?;
    println!("# paired_ratio=current_ms/baseline_ms; values below 1 favor current");
    println!("t,threads,batch,order,level,baseline_ms,current_ms,paired_ratio");
    for &log_t in &options.sizes {
        for &threads in &options.threads {
            let mut rows: BTreeMap<String, Comparison> = BTreeMap::new();
            for batch in 0..options.samples {
                if options.load {
                    report_load()?;
                }
                let (old, new, order) = if batch % 2 == 0 {
                    (
                        child_sample(baseline, log_t, threads)?,
                        child_sample(&current, log_t, threads)?,
                        "AB",
                    )
                } else {
                    let new = child_sample(&current, log_t, threads)?;
                    (child_sample(baseline, log_t, threads)?, new, "BA")
                };
                if old.keys().ne(new.keys()) {
                    return Err("benchmark child row sets differ".into());
                }
                for (level, baseline_ms) in old {
                    let current_ms = *new.get(&level).ok_or("missing current row")?;
                    let ratio = current_ms / baseline_ms;
                    println!("{log_t},{threads},{batch},{order},{level},{baseline_ms:.6},{current_ms:.6},{ratio:.6}");
                    let entry = rows.entry(level).or_default();
                    entry.baseline.push(baseline_ms);
                    entry.current.push(current_ms);
                    entry.ratios.push(ratio);
                }
            }
            println!("# summary: t,threads,level,baseline_median_ms,current_median_ms,median_ratio,baseline_min_ms,current_min_ms,min_ratio,baseline_max_ms,current_max_ms,max_ratio,paired_median_ratio,paired_min_ratio,paired_max_ratio");
            for (level, mut measurements) in rows {
                let (om, on, ox) = stats(&mut measurements.baseline);
                let (nm, nn, nx) = stats(&mut measurements.current);
                let (rm, rn, rx) = stats(&mut measurements.ratios);
                println!("summary,{log_t},{threads},{level},{om:.6},{nm:.6},{:.6},{on:.6},{nn:.6},{:.6},{ox:.6},{nx:.6},{:.6},{rm:.6},{rn:.6},{rx:.6}", nm/om, nn/on, nx/ox);
            }
        }
    }
    Ok(())
}

#[expect(
    clippy::print_stdout,
    reason = "load snapshots accompany benchmark output"
)]
fn report_load() -> Result<(), Box<dyn Error>> {
    let output = Command::new("uptime").output()?;
    if !output.status.success() {
        return Err("uptime failed".into());
    }
    println!("# load: {}", String::from_utf8(output.stdout)?.trim());
    Ok(())
}

#[inline(never)]
fn reduced_products<const N: usize>(iterations: usize, seed: u64) -> Duration {
    let mut rng = Words(black_box(seed));
    let mut state: [F64; N] = std::array::from_fn(|_| F64::from_raw(rng.gen()));
    let factors: [F64; N] = black_box(std::array::from_fn(|_| F64::from_raw(rng.gen())));
    let start = Instant::now();
    for _ in 0..black_box(iterations) {
        for (value, factor) in state.iter_mut().zip(&factors) {
            *value *= *factor;
        }
    }
    let _result = black_box(state);
    start.elapsed()
}

#[inline(never)]
fn extension_scalings<const N: usize>(iterations: usize, seed: u64) -> Duration {
    let mut rng = Words(black_box(seed));
    let mut state: [F192; N] =
        std::array::from_fn(|_| F192::from_base_fn(|_| F64::from_raw(rng.gen())));
    let factors: [F64; N] = black_box(std::array::from_fn(|_| F64::from_raw(rng.gen())));
    let start = Instant::now();
    for _ in 0..black_box(iterations) {
        for (value, factor) in state.iter_mut().zip(&factors) {
            *value = value.mul_base(*factor);
        }
    }
    let _result = black_box(state);
    start.elapsed()
}

#[expect(
    clippy::print_stdout,
    reason = "arithmetic throughput CSV is benchmark output"
)]
fn report_arithmetic(options: &Options) {
    type ArithmeticCase = (&'static str, usize, fn(usize, u64) -> Duration);
    let cases: [ArithmeticCase; 5] = [
        ("K-reduced-product", 1, reduced_products::<1>),
        ("K-reduced-product", 4, reduced_products::<4>),
        ("K-reduced-product", 8, reduced_products::<8>),
        ("E-by-K-scale", 4, extension_scalings::<4>),
        ("E-by-K-scale", 8, extension_scalings::<8>),
    ];
    println!("# one thread; dependent chains are mutually independent; products include reduction");
    println!("operation,chains,iterations,operations,K-products,median_ms,min_ms,max_ms,median_ns/operation,min_ns/operation,median_ns/K-product,min_ns/K-product");
    for (operation, chains, run) in cases {
        let _warm = black_box(run(options.iterations, 0x6172_6974_686d));
        let mut times: Vec<_> = (0..options.samples)
            .map(|sample| {
                milliseconds(run(
                    options.iterations,
                    black_box(0x6172_6974_686d + sample as u64),
                ))
            })
            .collect();
        let (median, min, max) = stats(&mut times);
        let operations = options.iterations as f64 * chains as f64;
        let k_products = operations
            * if operation == "E-by-K-scale" {
                3.0
            } else {
                1.0
            };
        println!("{operation},{chains},{},{operations:.0},{k_products:.0},{median:.6},{min:.6},{max:.6},{:.6},{:.6},{:.6},{:.6}",
            options.iterations, median * 1e6 / operations, min * 1e6 / operations,
            median * 1e6 / k_products, min * 1e6 / k_products);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse()?;
    if options.load {
        report_load()?;
    }
    if options.arithmetic {
        report_arithmetic(&options);
    } else if let Some(baseline) = &options.compare {
        compare(&options, baseline)?;
    } else {
        report(&options)?;
    }
    if options.load {
        report_load()?;
    }
    Ok(())
}

//! Small driver-level runner: the kernels runner requires SyntheticTrace fixtures
//! and owns its configuration loop, so cannot accept an executed witness or ids.
//! Samples retain its rotation, warmed pools, baseline and median-of-sums rules.

use super::allocator::{AllocationMeasurement, CountingAllocator};
use super::inventory::Inventory;
use super::pipelines::{BenchResult, Fixture, Kernels};
use super::timing::{Phase, Timing};
use super::witness::WitnessFixture;
use jolt_rv64i_prover::optimized::outer::WitnessLanes;
use rayon::ThreadPoolBuilder;
use std::error::Error;
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

const NAMES: [&str; 6] = ["source", "lanes", "outer", "routers", "tail", "session"];

pub struct Options {
    pub log_t: Vec<u8>,
    pub threads: Vec<usize>,
    pub samples: usize,
    pub inventory: bool,
    filter: Option<String>,
}
impl Options {
    pub fn parse() -> BenchResult<Self> {
        let mut result = Self {
            log_t: vec![20, 22],
            threads: vec![1, 12],
            samples: 1,
            inventory: false,
            filter: None,
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--bench" => {}
                "inventory" => result.inventory = true,
                "--log-t" | "--threads" | "--samples" => {
                    let value = args.next().ok_or("missing option value")?;
                    let values: Vec<usize> =
                        value.split(',').map(str::parse).collect::<Result<_, _>>()?;
                    match arg.as_str() {
                        "--log-t" if values.iter().all(|v| [20, 22].contains(v)) => {
                            result.log_t = values.into_iter().map(|v| v as u8).collect();
                        }
                        "--threads" if values.iter().all(|v| [1, 12].contains(v)) => {
                            result.threads = values;
                        }
                        "--samples" if values.len() == 1 && values[0] > 0 => {
                            result.samples = values[0];
                        }
                        _ => return Err(format!("invalid {arg}: {value}").into()),
                    }
                }
                _ if arg.starts_with("adapters/") && result.filter.is_none() => {
                    result.filter = Some(arg);
                }
                _ => return Err(format!("unknown argument {arg}").into()),
            }
        }
        if result.filter.is_some()
            && !NAMES.into_iter().any(|name| {
                result.log_t.iter().any(|&log_t| {
                    result
                        .threads
                        .iter()
                        .any(|&threads| result.selected(name, log_t, threads))
                })
            })
        {
            return Err("id filter matches no cases".into());
        }
        Ok(result)
    }
    fn selected(&self, name: &str, log_t: u8, threads: usize) -> bool {
        self.filter.as_ref().is_none_or(|filter| {
            let id = format!("adapters/{name}/{log_t}/{threads}");
            id == *filter || id.starts_with(&format!("{filter}/"))
        })
    }
}

pub struct Sample {
    pub times: [Duration; Phase::TIMED.len()],
    pub driver: Duration,
    pub total: Duration,
    pub peak: usize,
    pub final_bytes: usize,
    pub allocs: usize,
}

fn threshold(name: &str, log_t: u8, threads: usize) -> f64 {
    let scaling = if threads == 12 { 9.6 } else { 1.0 };
    let router = if threads == 1 {
        if log_t == 22 {
            299.0
        } else {
            377.0
        }
    } else {
        // Performance's zeta is serial; the kernels threshold alone scales.
        (if log_t == 22 { 299.0 } else { 376.0 }) / scaling + 550_000.0 / (1_u64 << log_t) as f64
    };
    match name {
        "source" => 8.0 / scaling,
        "lanes" => 40.0 / scaling,
        "outer" => 294.0 / scaling,
        "routers" => router,
        "tail" => 192.0 / scaling,
        _ => 200.0 / scaling + router,
    }
}

fn summary(mut values: Vec<f64>) -> (f64, f64, f64) {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    let median = if values.len().is_multiple_of(2) {
        values[mid - 1].midpoint(values[mid])
    } else {
        values[mid]
    };
    (median, values[0], values[values.len() - 1])
}

#[expect(
    clippy::print_stdout,
    reason = "phase distributions are benchmark output"
)]
fn print_samples(
    id: &str,
    name: &str,
    log_t: u8,
    threads: usize,
    samples: &[Sample],
    fixture: &WitnessFixture,
) {
    let cycles = fixture.witness.bits.len() as f64;
    let ns = |duration: Duration| duration.as_nanos() as f64 / cycles;
    let (total, min, max) = summary(samples.iter().map(|s| ns(s.total)).collect());
    let peak = samples.iter().map(|s| s.peak).max().unwrap_or(0);
    let final_bytes = samples.iter().map(|s| s.final_bytes).max().unwrap_or(0);
    let allocs = samples.iter().map(|s| s.allocs).max().unwrap_or(0);
    let threshold = threshold(name, log_t, threads);
    let count = samples.len();
    let meets_threshold = total <= threshold;
    let decoded_bytes = fixture.witness.bits.len() * 16;
    let peak_with_decoded_bytes = peak + decoded_bytes;
    print!("{id} samples={count} total_ns={total:.6}");
    print!(" total_min_ns={min:.6} total_max_ns={max:.6}");
    print!(" threshold_ns={threshold:.6} meets_threshold={meets_threshold}");
    print!(" peak_bytes={peak} final_bytes={final_bytes} allocs={allocs}");
    print!(" decoded_bytes={decoded_bytes} peak_with_decoded_bytes={peak_with_decoded_bytes}");
    for phase in Phase::TIMED {
        let label = phase.label();
        let (median, min, max) =
            summary(samples.iter().map(|s| ns(s.times[phase.index()])).collect());
        print!(" {label}_ns={median:.6} {label}_min_ns={min:.6} {label}_max_ns={max:.6}");
    }
    let (driver, _, _) = summary(samples.iter().map(|s| ns(s.driver)).collect());
    println!(" driver_ns={driver:.6} loaded_machine=true {}", fixture.mix);
}

#[expect(
    clippy::print_stdout,
    reason = "setup and inventory notes are benchmark output"
)]
pub fn run(options: Options) -> BenchResult<()> {
    print!("adapters_note phases=kernel_hooks driver_bookkeeping=extract");
    print!(" driver_allocations=driver finish=terminal_bind");
    print!(" extract=validation_and_claims park=dispatch_and_drop");
    println!(" background_drop=feature_selected decoded=resident_apart loaded_machine=true");
    let mut inventory_failed = false;
    // One executed witness per size, shared by both thread configurations.
    for &log_t in &options.log_t {
        if !options.threads.iter().any(|&threads| {
            NAMES
                .iter()
                .any(|name| options.selected(name, log_t, threads))
        }) {
            continue;
        }
        let witness = WitnessFixture::new(log_t)?;
        let _ = witness.checked()?;
        for &threads in &options.threads {
            let names: Vec<_> = NAMES
                .into_iter()
                .filter(|name| options.selected(name, log_t, threads))
                .collect();
            if names.is_empty() {
                continue;
            }
            let start = Instant::now();
            let fixture = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_scoped(
                    |thread| thread.run(),
                    |pool| {
                        let _ = pool.broadcast(|_| black_box(()));
                        pool.install(|| Fixture::new(&witness.witness).map_err(|e| e.to_string()))
                    },
                )?
                .map_err(|e| -> Box<dyn Error> { e.into() })?;
            println!(
                "adapters_setup/{log_t}/{threads} prepare_ns={:.6} loaded_machine=true {}",
                start.elapsed().as_nanos() as f64 / witness.witness.bits.len() as f64,
                witness.mix
            );
            let inventory = Inventory::new(&witness.witness, &fixture.geometry);
            if options.inventory {
                inventory.print_laws(log_t, threads);
            }
            let mut collected: Vec<Vec<Sample>> = names
                .iter()
                .map(|_| Vec::with_capacity(options.samples))
                .collect();
            for _ in 0..options.samples {
                for (index, &name) in names.iter().enumerate() {
                    let sample = ThreadPoolBuilder::new().num_threads(threads).build_scoped(
                        |thread| thread.run(),
                        |pool| {
                            let _ = pool.broadcast(|_| black_box(()));
                            pool.install(|| {
                                let timing = Arc::new(Timing::default());
                                let kernels = Kernels::new(&timing);
                                CountingAllocator::begin(witness.witness.bits.len());
                                CountingAllocator::phase(Phase::WarmPrepare.index());
                                let before = CountingAllocator::live_bytes();
                                let mut session = fixture
                                    .warm_session(&witness.witness, name)
                                    .map_err(|e| e.to_string())?;
                                let resident =
                                    CountingAllocator::live_bytes().saturating_sub(before);
                                let measurement = AllocationMeasurement::begin();
                                CountingAllocator::phase(Phase::Driver.index());
                                let start = Instant::now();
                                let lanes = if name == "lanes" {
                                    Some(timing.measure(Phase::Prepare, || {
                                        WitnessLanes::new(&witness.witness)
                                            .map_err(|e| e.to_string())
                                    })?)
                                } else {
                                    if name == "source" {
                                        timing.measure(Phase::Prepare, || {
                                            fixture
                                                .run(name, &witness.witness, &mut session, &kernels)
                                                .map_err(|e| e.to_string())
                                        })?;
                                    } else {
                                        fixture
                                            .run(name, &witness.witness, &mut session, &kernels)
                                            .map_err(|e| e.to_string())?;
                                    }
                                    None
                                };
                                let total = start.elapsed();
                                let mut times = timing.times();
                                let driver = total.saturating_sub(times.iter().sum());
                                times[Phase::Extract.index()] += driver;
                                let stats = measurement.finish();
                                let _ = black_box(&lanes);
                                // Counters stop with the session and lanes alive; scoped workers
                                // join all queued drops before the next sample's baseline.
                                Ok::<_, String>(Sample {
                                    times,
                                    driver,
                                    total,
                                    peak: stats.peak_bytes + resident,
                                    final_bytes: stats.final_bytes + resident,
                                    allocs: stats.allocs,
                                })
                            })
                        },
                    )?;
                    CountingAllocator::stop();
                    let sample = sample.map_err(|e| -> Box<dyn Error> { e.into() })?;
                    if CountingAllocator::overflow() != 0 {
                        return Err("allocation recorder overflow".into());
                    }
                    if options.inventory {
                        inventory_failed |=
                            inventory.print(&format!("adapters/{name}/{log_t}/{threads}"));
                    }
                    collected[index].push(sample);
                }
            }
            for (&name, samples) in names.iter().zip(&collected) {
                print_samples(
                    &format!("adapters/{name}/{log_t}/{threads}"),
                    name,
                    log_t,
                    threads,
                    samples,
                    &witness,
                );
            }
        }
    }
    if inventory_failed {
        return Err("allocation inventory contains unmatched allocations".into());
    }
    Ok(())
}

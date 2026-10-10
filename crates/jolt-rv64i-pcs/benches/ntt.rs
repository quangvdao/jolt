//! Reference-geometry transform benchmark. Run with target-cpu=native:
//! `cargo bench -p jolt-rv64i-pcs --bench ntt -- --samples 5`.
//! Fixtures and warmed 1/12-thread pools precede timing. Every sample includes
//! output allocation, first touch and the call-scoped domain table. Later levels
//! share one table, matching an opening's lifecycle. Reports median phase times
//! and median-of-sums totals; source measurements were on a loaded host.

use jolt_field::{ExtField, Zero};
use jolt_field::{F192, F64};
use jolt_rv64i_pcs::ntt::Encoder;
use jolt_rv64i_verifier::whir::code::DomainTable;
use jolt_rv64i_verifier::whir::error::{try_vec, WhirError, WhirPart};
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use std::error::Error;
use std::hint::black_box;
use std::time::{Duration, Instant};

const LATER: [(usize, usize); 4] = [(14, 18), (10, 17), (6, 16), (2, 15)];
const COUNTS: [usize; 5] = [301_989_888, 29_360_128, 10_485_760, 3_145_728, 524_288];

struct Words(u64);
impl Words {
    fn seed_from_u64(seed: u64) -> Self {
        Self(seed)
    }
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
    later: [Vec<F192>; 4],
}
impl Fixture {
    fn new() -> Result<Self, Box<dyn Error>> {
        let mut rng = Words::seed_from_u64(0x004e_5454);
        let mut rows = try_vec(WhirPart::Rows, 1 << 22)?;
        rows.extend((0..1 << 22).map(|_| std::array::from_fn(|_| rng.gen())));
        let mut later = std::array::from_fn(|_| Vec::new());
        for (f, &(c, _)) in later.iter_mut().zip(&LATER) {
            *f = try_vec(WhirPart::FinalValues, 16 * (1 << c))?;
            f.extend((0..16 * (1 << c)).map(|_| F192::from_base_fn(|_| F64::from_raw(rng.gen()))));
        }
        Ok(Self { rows, later })
    }

    fn sample(
        &self,
        pool: &ThreadPool,
    ) -> Result<([Duration; 5], Duration, Duration), Box<dyn Error>> {
        pool.install(|| {
            let start = Instant::now();
            let table = DomainTable::new(18, 19)?;
            let setup_commit = start.elapsed();
            let code = Encoder::new(&table, 18, 19, 32)?.encode_rows(&self.rows)?;
            let _code = black_box(&code);
            let level0 = start.elapsed();
            drop(code);
            let start = Instant::now();
            let table = DomainTable::new(18, 19)?;
            let setup_open = start.elapsed();
            let mut durations = [Duration::ZERO; 5];
            durations[0] = level0;
            for (i, ((c, d), message)) in LATER.iter().zip(&self.later).enumerate() {
                let phase = Instant::now();
                let code = Encoder::new(&table, *c, *d, 16)?.encode_extension(message)?;
                let _code = black_box(&code);
                durations[i + 1] = phase.elapsed();
                drop(code);
            }
            Ok::<_, WhirError>((
                durations,
                setup_commit + setup_open,
                setup_open + durations[1..].iter().sum::<Duration>(),
            ))
        })
        .map_err(Into::into)
    }

    // A first-touch/copy baseline separates memory provisioning from arithmetic
    // without instrumentation in the production transform. Subtraction is an
    // estimate, since cache traffic overlaps arithmetic in the full encoding.
    fn copy_baseline(&self, pool: &ThreadPool) -> Result<[Duration; 5], Box<dyn Error>> {
        pool.install(|| {
            let start = Instant::now();
            let mut code = try_vec(WhirPart::Leaves, 1 << 25)?;
            code.resize(1 << 25, F64::zero());
            code.par_chunks_mut(1 << 24).for_each(|coset| {
                for (r, out) in self.rows.iter().zip(coset.chunks_exact_mut(4)) {
                    for (word, slot) in r.iter().zip(out) {
                        *slot = F64::from_raw(*word);
                    }
                }
            });
            let _code = black_box(&code);
            let time = start.elapsed();
            drop(code);
            let mut times = [Duration::ZERO; 5];
            times[0] = time;
            for (i, (&(_, d), message)) in LATER.iter().zip(&self.later).enumerate() {
                let start = Instant::now();
                let len = 16 * (1usize << d);
                let mut code = try_vec(WhirPart::Leaves, len)?;
                code.resize(len, F192::zero());
                code.par_chunks_mut(message.len())
                    .for_each(|block| block.copy_from_slice(message));
                let _code = black_box(&code);
                times[i + 1] = start.elapsed();
                drop(code);
            }
            Ok::<_, WhirError>(times)
        })
        .map_err(Into::into)
    }
}

fn median(values: &mut [Duration]) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

#[expect(
    clippy::print_stdout,
    reason = "benchmark measurements are stdout output"
)]
fn main() -> Result<(), Box<dyn Error>> {
    let mut samples = 5usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bench" => {}
            "--samples" => {
                samples = args.next().ok_or("missing samples")?.parse()?;
                if samples == 0 {
                    return Err("samples must be positive".into());
                }
            }
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    let fixture = Fixture::new()?;
    println!("t=22; seeded random fixtures; samples={samples}; medians; allocation and constants included");
    println!("threads,level,c,d,butterflies,ms,Gbutterflies/s,ns/butterfly,1t_model_ms,1t_model_x1.25_ms,source_ms,min_ms,max_ms");
    for threads in [1, 12] {
        let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
        let _warm = pool.install(|| rayon::broadcast(|_| black_box(0)));
        let _warm = black_box(fixture.sample(&pool)?);
        let mut phases: [Vec<Duration>; 5] = std::array::from_fn(|_| Vec::new());
        let mut setup = Vec::new();
        let mut sums = Vec::new();
        let mut copies: [Vec<Duration>; 5] = std::array::from_fn(|_| Vec::new());
        for _ in 0..samples {
            let (durations, constants, later_sum) = fixture.sample(&pool)?;
            for (times, duration) in phases.iter_mut().zip(durations) {
                times.push(duration);
            }
            setup.push(constants);
            sums.push(later_sum);
            for (times, duration) in copies.iter_mut().zip(fixture.copy_baseline(&pool)?) {
                times.push(duration);
            }
        }
        for (level, times) in phases.iter_mut().enumerate() {
            let duration = median(times);
            let seconds = duration.as_secs_f64();
            let count = COUNTS[level] as f64;
            let model = count * if level == 0 { 0.535 } else { 1.50 } / 1e6;
            let (c, d) = if level == 0 {
                (18, 19)
            } else {
                LATER[level - 1]
            };
            let source = if level == 0 {
                if threads == 1 {
                    "161.58"
                } else {
                    "37.71"
                }
            } else {
                "aggregate-only"
            };
            println!(
                "{threads},{level},{c},{d},{},{:.3},{:.3},{:.3},{model:.3},{:.3},{source},{:.3},{:.3}",
                COUNTS[level],
                seconds * 1e3,
                count / seconds / 1e9,
                seconds * 1e9 / count,
                model * 1.25,
                times[0].as_secs_f64() * 1e3,
                times[times.len() - 1].as_secs_f64() * 1e3
            );
        }
        let seconds = median(&mut sums).as_secs_f64();
        let count = COUNTS[1..].iter().sum::<usize>() as f64;
        let model = count * 1.50 / 1e6;
        println!(
            "{threads},later-total,-,-,{count:.0},{:.3},{:.3},{:.3},{model:.3},{:.3},{:.2},{:.3},{:.3}",
            seconds * 1e3,
            count / seconds / 1e9,
            seconds * 1e9 / count,
            model * 1.25,
            if threads == 1 { 65.37 } else { 11.74 },
            sums[0].as_secs_f64() * 1e3,
            sums[sums.len() - 1].as_secs_f64() * 1e3
        );
        println!(
            "{threads}: constants for both calls {:.3} ms",
            median(&mut setup).as_secs_f64() * 1e3
        );
        for (level, times) in copies.iter_mut().enumerate() {
            println!(
                "{threads}: level-{level} allocation/first-touch/copy baseline {:.3} ms",
                median(times).as_secs_f64() * 1e3
            );
        }
    }
    Ok(())
}

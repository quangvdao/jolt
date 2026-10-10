//! Tree-only phase measurements at t = 22, including tree allocation.
//! Run: `RUSTFLAGS='-C target-cpu=native' cargo bench -p jolt-rv64i-pcs --bench merkle`.
//! Input allocation and warmed Rayon pools precede timing. Optional first
//! argument is the number of repetitions (default five); outputs are medians.

use jolt_rv64i_pcs::merkle::MerkleTree;
use jolt_rv64i_verifier::whir::error::{try_vec, WhirPart};
use rayon::ThreadPoolBuilder;
use std::{error::Error, hint::black_box, io::Write, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let runs = std::env::args()
        .skip(1)
        .find(|arg| arg != "--bench")
        .map_or(Ok(5), |s| s.parse::<usize>())?
        .max(1);
    let mut input = try_vec(WhirPart::Leaves, (1 << 19) * 512)?;
    input.extend((0..(1 << 19) * 512).map(|i: usize| {
        let mixed = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        (mixed ^ (mixed >> 31)) as u8
    }));
    let mut out = std::io::stdout().lock();
    for threads in [1, 12] {
        let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
        drop(pool.install(|| MerkleTree::build(&input, 512))?);
        let mut level0 = try_vec(WhirPart::Digests, runs)?;
        let mut later = try_vec(WhirPart::Digests, runs)?;
        let mut levels = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for samples in &mut levels {
            *samples = try_vec(WhirPart::Digests, runs)?;
        }
        for _ in 0..runs {
            let (first, rest) = pool
                .install(|| -> Result<_, Box<dyn Error + Send + Sync>> {
                    let start = Instant::now();
                    let tree = MerkleTree::build(black_box(&input), 512)?;
                    let first = start.elapsed().as_secs_f64();
                    let _root = black_box(tree.root());
                    drop(tree);
                    let mut rest = [0.0; 4];
                    for (i, depth) in [18, 17, 16, 15].into_iter().enumerate() {
                        let start = Instant::now();
                        let tree = MerkleTree::build(black_box(&input[..(1 << depth) * 384]), 384)?;
                        rest[i] = start.elapsed().as_secs_f64();
                        let _root = black_box(tree.root());
                    }
                    Ok((first, rest))
                })
                .map_err(|e| -> Box<dyn Error> { e })?;
            level0.push(first);
            later.push(rest.iter().sum::<f64>());
            for (samples, duration) in levels.iter_mut().zip(rest) {
                samples.push(duration);
            }
        }
        level0.sort_by(f64::total_cmp);
        later.sort_by(f64::total_cmp);
        let first = level0[runs / 2];
        let rest = later[runs / 2];
        writeln!(out, "threads={threads} runs={runs} level0_ms={:.3} later_ms={:.3} hb0_ns={:.3} hb1_ns={:.3}",
            first * 1000.0, rest * 1000.0, first * 1e9 / 4_718_591.0, rest * 1e9 / 3_440_636.0)?;
        for (depth, samples) in [18, 17, 16, 15].into_iter().zip(&mut levels) {
            samples.sort_by(f64::total_cmp);
            writeln!(
                out,
                "  depth={depth} leaf_bytes=384 ms={:.3}",
                samples[runs / 2] * 1000.0
            )?;
        }
    }
    Ok(())
}

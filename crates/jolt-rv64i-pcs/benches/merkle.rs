//! Tree-phase measurements at t = 22, including tree allocation.
//! Run: `RUSTFLAGS='-C target-cpu=native' cargo bench -p jolt-rv64i-pcs --features arch --bench merkle`.
//! Input allocation and warmed Rayon pools precede timing. Optional first
//! argument is the number of repetitions (default five). Scalar and selected
//! builders alternate order each repetition; outputs are medians.

use jolt_rv64i_pcs::merkle::MerkleTree;
use jolt_rv64i_verifier::whir::{
    error::{checked_product, try_vec, WhirError, WhirPart},
    merkle::{hash_leaf, hash_node, Digest},
};
use rayon::{prelude::*, ThreadPoolBuilder};
use std::{error::Error, hint::black_box, io::Write, time::Instant};

#[derive(Clone, Copy)]
enum Backend {
    Scalar,
    Selected,
}

impl Backend {
    fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Selected => "selected",
        }
    }

    fn measure(self, input: &[u8]) -> Result<[f64; 5], WhirError> {
        let mut times = [0.0; 5];
        for (i, depth) in [19, 18, 17, 16, 15].into_iter().enumerate() {
            let width = if i == 0 { 512 } else { 384 };
            let data = black_box(&input[..(1 << depth) * width]);
            let start = Instant::now();
            times[i] = match self {
                Self::Scalar => {
                    let tree = scalar_tree(data, width)?;
                    let elapsed = start.elapsed().as_secs_f64();
                    let _root = black_box(tree.last());
                    elapsed
                }
                Self::Selected => {
                    let tree = MerkleTree::build(data, width)?;
                    let elapsed = start.elapsed().as_secs_f64();
                    let _root = black_box(tree.root());
                    elapsed
                }
            };
        }
        Ok(times)
    }
}

fn scalar_tree(data: &[u8], leaf_bytes: usize) -> Result<Vec<Digest>, WhirError> {
    let n = data.len() / leaf_bytes;
    let len = checked_product(WhirPart::Digests, &[n, 2])? - 1;
    let mut tree = try_vec(WhirPart::Digests, len)?;
    tree.resize(len, [0; 32]);
    tree[..n]
        .par_chunks_mut(1024)
        .enumerate()
        .for_each(|(group, out)| {
            let start = group * 1024 * leaf_bytes;
            for (input, slot) in data[start..start + out.len() * leaf_bytes]
                .chunks_exact(leaf_bytes)
                .zip(out)
            {
                *slot = hash_leaf(input);
            }
        });
    let mut start = 0;
    let mut width = n;
    while width > 1 {
        let (read, write) = tree.split_at_mut(start + width);
        write[..width / 2]
            .par_chunks_mut(1024)
            .enumerate()
            .for_each(|(group, out)| {
                let first = start + group * 2048;
                for (pair, slot) in read[first..first + out.len() * 2].chunks_exact(2).zip(out) {
                    *slot = hash_node(&pair[0], &pair[1]);
                }
            });
        start += width;
        width /= 2;
    }
    Ok(tree)
}

struct Samples {
    phases: [Vec<f64>; 5],
    later: Vec<f64>,
}
impl Samples {
    fn new(runs: usize) -> Result<Self, WhirError> {
        let mut phases = [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for phase in &mut phases {
            *phase = try_vec(WhirPart::Digests, runs)?;
        }
        Ok(Self {
            phases,
            later: try_vec(WhirPart::Digests, runs)?,
        })
    }
    fn record(&mut self, times: [f64; 5]) {
        self.later.push(times[1..].iter().sum());
        for (phase, time) in self.phases.iter_mut().zip(times) {
            phase.push(time);
        }
    }
    fn report(
        &mut self,
        out: &mut impl Write,
        backend: Backend,
        threads: usize,
        runs: usize,
    ) -> Result<(), Box<dyn Error>> {
        for phase in &mut self.phases {
            phase.sort_by(f64::total_cmp);
        }
        self.later.sort_by(f64::total_cmp);
        let first = self.phases[0][runs / 2];
        let later = self.later[runs / 2];
        writeln!(out, "backend={} threads={threads} runs={runs} level0_ms={:.3} later_ms={:.3} hb0_ns={:.3} hb1_ns={:.3}",
            backend.name(), first*1000.0,later*1000.0,first*1e9/4_718_591.0,later*1e9/3_440_636.0)?;
        for (i, depth) in [18, 17, 16, 15].into_iter().enumerate() {
            writeln!(
                out,
                "  depth={depth} leaf_bytes=384 ms={:.3}",
                self.phases[i + 1][runs / 2] * 1000.0
            )?;
        }
        Ok(())
    }
}

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
    writeln!(
        out,
        "arch_feature={} neon={} avx2={}",
        cfg!(feature = "arch"),
        cfg!(all(
            target_arch = "aarch64",
            target_feature = "neon",
            target_endian = "little"
        )),
        cfg!(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_endian = "little"
        ))
    )?;
    for threads in [1, 12] {
        let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
        for backend in [Backend::Scalar, Backend::Selected] {
            let _warmup = pool.install(|| backend.measure(&input))?;
        }
        let mut scalar = Samples::new(runs)?;
        let mut selected = Samples::new(runs)?;
        for run in 0..runs {
            let order = if run % 2 == 0 {
                [Backend::Scalar, Backend::Selected]
            } else {
                [Backend::Selected, Backend::Scalar]
            };
            for backend in order {
                let times = pool.install(|| backend.measure(&input))?;
                match backend {
                    Backend::Scalar => scalar.record(times),
                    Backend::Selected => selected.record(times),
                }
            }
        }
        scalar.report(&mut out, Backend::Scalar, threads, runs)?;
        selected.report(&mut out, Backend::Selected, threads, runs)?;
    }
    Ok(())
}

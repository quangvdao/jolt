//! Hot arithmetic inputs are prepared before timing. The largest pair block is
//! 32 KiB; the 128-element accumulator bank is 6 KiB on native aarch64. Repeated
//! blocks expose their input reference and one checksum at block boundaries.
//! Reduction's safe control XORs all opaque accumulator lanes without reducing;
//! its three-lane checksum differs from reduction's one-lane checksum, so their
//! signed difference is context, not an isolated reduction price.

use std::hint::black_box;

use jolt_field::{Accumulator, Field, WithAccumulator, F128};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;

type F128Accumulator = <F128 as WithAccumulator>::Accumulator;
const PAIRS: usize = 1024;
const BANK: usize = 128;
const BLOCKS: usize = 4096;

pub enum HotArithmetic {
    Product(Box<[(F128, F128); PAIRS]>),
    Chain {
        pairs: Box<[(F128, F128); PAIRS]>,
        terms: usize,
    },
    Reduce(Box<[F128Accumulator; BANK]>),
    Control(Box<[F128Accumulator; BANK]>),
}

impl HotArithmetic {
    pub fn product() -> Self {
        Self::Product(Self::pairs())
    }
    pub fn chain(terms: usize) -> Self {
        Self::Chain {
            pairs: Self::pairs(),
            terms,
        }
    }
    pub fn reduction(control: bool) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(0x726564756365);
        let bank = Box::new(std::array::from_fn(|_| {
            let mut acc = F128Accumulator::default();
            for _ in 0..5 {
                acc.fmadd(F128::random(&mut rng), F128::random(&mut rng));
            }
            acc
        }));
        if control {
            Self::Control(bank)
        } else {
            Self::Reduce(bank)
        }
    }
    fn pairs() -> Box<[(F128, F128); PAIRS]> {
        let mut rng = ChaCha20Rng::seed_from_u64(0x686f7470616972);
        Box::new(std::array::from_fn(|_| {
            (F128::random(&mut rng), F128::random(&mut rng))
        }))
    }
    pub fn operations(&self) -> usize {
        match self {
            Self::Product(_) => BLOCKS * PAIRS,
            Self::Chain { terms, .. } => BLOCKS * (PAIRS / terms),
            Self::Reduce(_) | Self::Control(_) => BLOCKS * BANK,
        }
    }
    pub fn terms(&self) -> Option<usize> {
        match self {
            Self::Chain { terms, .. } => Some(*terms),
            _ => None,
        }
    }
    pub fn run(&self) -> F128 {
        match self {
            Self::Product(pairs) => (0..BLOCKS)
                .into_par_iter()
                .map(|_| black_box(Self::product_block(black_box(pairs.as_slice()))))
                .reduce(|| F128::from_raw(0), |a, b| a + b),
            Self::Chain { pairs, terms } => (0..BLOCKS)
                .into_par_iter()
                .map(|_| {
                    let pairs = black_box(pairs.as_slice());
                    let mut total = F128::from_raw(0);
                    for chain in pairs.chunks_exact(*terms) {
                        total += Self::chain_block(chain);
                    }
                    black_box(total)
                })
                .reduce(|| F128::from_raw(0), |a, b| a + b),
            Self::Reduce(bank) => (0..BLOCKS)
                .into_par_iter()
                .map(|_| black_box(Self::reduce_block(black_box(bank.as_slice()))))
                .reduce(|| F128::from_raw(0), |a, b| a + b),
            Self::Control(bank) => {
                (0..BLOCKS).into_par_iter().for_each(|_| {
                    let _ = black_box(Self::control_block(black_box(bank.as_slice())));
                });
                F128::from_raw(0)
            }
        }
    }
    #[inline(never)]
    fn product_block(pairs: &[(F128, F128)]) -> F128 {
        pairs
            .iter()
            .fold(F128::from_raw(0), |sum, &(a, b)| sum + a * b)
    }
    // One runtime-length body serves both long chains; the assembly check pins
    // that their incremental cost uses identical term work.
    #[inline(never)]
    fn chain_block(pairs: &[(F128, F128)]) -> F128 {
        let mut acc = F128Accumulator::default();
        for &(a, b) in pairs {
            acc.fmadd(a, b);
        }
        acc.reduce()
    }
    #[inline(never)]
    fn reduce_block(bank: &[F128Accumulator]) -> F128 {
        bank.iter()
            .fold(F128::from_raw(0), |sum, &acc| sum + acc.reduce())
    }
    #[inline(never)]
    fn control_block(bank: &[F128Accumulator]) -> F128Accumulator {
        let mut total = F128Accumulator::default();
        for &acc in bank {
            total.merge(acc);
        }
        total
    }
}

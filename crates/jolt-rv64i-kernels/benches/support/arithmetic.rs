//! Hot arithmetic inputs are prepared before timing. The largest pair block is
//! 32 KiB; the 128-element accumulator bank is 6 KiB on native aarch64. Repeated
//! blocks expose their input reference and one checksum at block boundaries.
//! Reduction's safe control XORs all opaque accumulator lanes without reducing;
//! its three-lane checksum differs from reduction's one-lane checksum, so their
//! signed difference is context, not an isolated reduction price.

use std::hint::black_box;

use jolt_field::{Accumulator, Field, WithAccumulator, F128};
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;

type F128Accumulator = <F128 as WithAccumulator>::Accumulator;
const PAIRS: usize = 1024;
const BANK: usize = 128;
const BLOCKS: usize = 4096;
use super::word::{word_monomial, WordInput};
// Two transforms (18), alignment (2), AND/mask (2), gather (9).
const WORD_OPERATIONS: usize = 31;

pub enum HotArithmetic {
    Product(Box<[(F128, F128); PAIRS]>),
    MulX(Box<[F128; BANK]>),
    Word(Box<[WordInput; BANK]>),
    Chain {
        pairs: Box<[(F128, F128); PAIRS]>,
        terms: usize,
    },
    Reduce(Box<[F128Accumulator; BANK]>),
    Control(Box<[F128Accumulator; BANK]>),
}

impl HotArithmetic {
    pub fn mul_x() -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(0x6d756c78);
        Self::MulX(Box::new(std::array::from_fn(|_| F128::random(&mut rng))))
    }
    pub fn word() -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(0x776f7264);
        Self::Word(Box::new(std::array::from_fn(|_| WordInput {
            a: rng.next_u64(),
            b: rng.next_u64(),
            left: rng.next_u32() & 7,
            right: rng.next_u32() & 7,
        })))
    }
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
            Self::MulX(_) => BLOCKS * BANK,
            Self::Word(_) => BLOCKS * BANK * WORD_OPERATIONS,
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
            Self::MulX(values) => (0..BLOCKS)
                .into_par_iter()
                .map(|_| black_box(Self::mul_x_block(black_box(values.as_slice()))))
                .reduce(|| F128::from_raw(0), |a, b| a + b),
            Self::Word(values) => (0..BLOCKS)
                .into_par_iter()
                .map(|_| {
                    F128::from_raw(u128::from(black_box(Self::word_block(black_box(
                        values.as_slice(),
                    )))))
                })
                .reduce(|| F128::from_raw(0), |a, b| a + b),
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
                let checksum = (0..BLOCKS)
                    .into_par_iter()
                    .map(|_| black_box(Self::control_block(black_box(bank.as_slice()))))
                    .reduce(F128Accumulator::default, |mut a, b| {
                        a.merge(b);
                        a
                    });
                let _ = black_box(checksum);
                F128::from_raw(0)
            }
        }
    }
    #[inline(never)]
    fn mul_x_block(values: &[F128]) -> F128 {
        values.iter().fold(F128::from_raw(0), |sum, value| {
            let raw = value.to_raw();
            sum + F128::from_raw((raw << 1) ^ (0x87 & 0_u128.wrapping_sub(raw >> 127)))
        })
    }
    #[inline(never)]
    fn word_block(values: &[WordInput]) -> u64 {
        values
            .iter()
            .fold(0, |sum, &value| sum ^ word_monomial(value))
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

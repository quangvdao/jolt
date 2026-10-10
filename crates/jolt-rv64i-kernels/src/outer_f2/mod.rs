//! Degree-three sum-check of `sum eq(tau,(y,j)) (A B + C)` over 256
//! low-bit-first rows and a power-of-two cycle domain. Lane bits stay packed
//! until all six position variables have been bound.

mod monomial;
mod window;

use crate::packed::lift::WordLift;
use crate::packed::pool::{PoolError, ScratchPool};
use crate::par::{CycleChunks, ParError};
use crate::round::eq::{eq_table, split_eq};
use crate::round::{coefficients_from_nodes, RoundError};
use crate::source::LaneSource;
use jolt_field::{Accumulator, F128Accumulator, F128};
use jolt_poly::{gruen_mul_linear, gruen_recover_endpoint, GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use monomial::Monomial;
use rayon::prelude::*;
use std::sync::Arc;
use thiserror::Error;
use window::Window;

const ZERO: F128 = F128::from_raw(0);
const ONE: F128 = F128::from_raw(1);
type Sums = [F128; 2];
type Tables = [Vec<F128>; 3];

fn merge<const N: usize>(
    mut left: [F128Accumulator; N],
    right: [F128Accumulator; N],
) -> [F128Accumulator; N] {
    for (left, right) in left.iter_mut().zip(right) {
        left.merge(right);
    }
    left
}

/// Invalid outer dimensions or scratch geometry. No row-satisfaction scan is
/// performed by the constructor; use [`OuterF2Core::check_rows`] diagnostically.
#[derive(Debug, Error)]
pub enum OuterError {
    /// The cycle domain is not a representable nonempty power of two.
    #[error("cycle count {cycles} is not a supported power of two")]
    Cycles { cycles: usize },
    /// The point must have eight row coordinates and one per cycle variable.
    #[error("point length {actual} differs from {expected}")]
    PointLength { expected: usize, actual: usize },
    /// Both schedules require the switch after two through six position rounds.
    #[error("monomial round count {rounds} is outside 2..=6")]
    MonomialRounds { rounds: usize },
    /// A shared round helper rejected its geometry.
    #[error(transparent)]
    Round(#[from] RoundError),
    /// A shared cycle geometry helper rejected its dimensions.
    #[error(transparent)]
    Geometry(#[from] ParError),
    /// Histogram scratch could not be lent or merged.
    #[error(transparent)]
    Scratch(#[from] PoolError),
}

/// Table choices that preserve the defining polynomial and challenge order.
#[derive(Clone, Copy, Debug)]
pub struct OuterF2Options {
    /// Number of position rounds in monomial form, in `2..=6` (default 3).
    pub monomial_rounds: usize,
    /// Use nibble rather than byte lifts in the second round (default false).
    pub nibble_round_2: bool,
    /// Fold group weights into the window lifts (default true).
    pub folded_group_weights: bool,
}

/// Uses three monomial rounds, byte lifts in round two, and folded group weights.
impl Default for OuterF2Options {
    fn default() -> Self {
        Self {
            monomial_rounds: 3,
            nibble_round_2: false,
            folded_group_weights: true,
        }
    }
}

enum State {
    Round(usize),
    LastBind,
    Finished,
    Failed,
}

/// Packed outer core emitting `8 + log_t` degree-three coefficient messages.
/// The source is shared, and row and cycle coordinates are bound low first.
/// Final values are available after a successful `finish_rounds`.
pub struct OuterF2Core<S: LaneSource> {
    source: Arc<S>,
    tau: Vec<F128>,
    options: OuterF2Options,
    log_t: usize,
    chunks: CycleChunks,
    cycle_chunks: Vec<CycleChunks>,
    lo: Vec<F128>,
    hi: Vec<F128>,
    omega: Vec<F128>,
    point: Vec<F128>,
    sigma: F128,
    histogram: [F128; 64],
    groups: [Tables; 2],
    tail: Vec<u8>,
    tail_values: [[F128; 3]; 64],
    cycle_eq: Option<GruenSplitEqPolynomial<F128>>,
    final_values: [F128; 3],
    state: State,
}

impl<S: LaneSource> OuterF2Core<S> {
    /// Validates cycle count, point length and the switch round before any pass.
    /// `tau` is low variable first. `C = A & B` on every lane word and tail
    /// row is required of the caller, not checked, and detected by the verifier
    /// at its final evaluation check when violated. [`Self::check_rows`] is an
    /// optional diagnostic, not a prerequisite to proving.
    pub fn new(source: Arc<S>, tau: &[F128], options: OuterF2Options) -> Result<Self, OuterError> {
        let cycles = source.cycles();
        if !cycles.is_power_of_two() || cycles > isize::MAX as usize / 16 {
            return Err(OuterError::Cycles { cycles });
        }
        let log_t = cycles.ilog2() as usize;
        if tau.len() != 8 + log_t {
            return Err(OuterError::PointLength {
                expected: 8 + log_t,
                actual: tau.len(),
            });
        }
        if !(2..=6).contains(&options.monomial_rounds) {
            return Err(OuterError::MonomialRounds {
                rounds: options.monomial_rounds,
            });
        }
        // An empty cycle point is rejected by the canonical split-equality helper.
        let _ = split_eq(&tau[8..], None)?;
        let chunks = CycleChunks::new(log_t, 0)?;
        let (low, high) = chunks.split_point(&tau[8..])?;
        Ok(Self {
            source,
            tau: tau.to_vec(),
            options,
            log_t,
            chunks,
            cycle_chunks: (1..=log_t)
                .map(|round| CycleChunks::new(log_t, round))
                .collect::<Result<_, _>>()?,
            lo: eq_table(low, None),
            hi: eq_table(high, None),
            omega: eq_table(&tau[6..8], None),
            point: Vec::with_capacity(8),
            sigma: ONE,
            histogram: [ZERO; 64],
            groups: std::array::from_fn(|_| std::array::from_fn(|_| Vec::new())),
            tail: Vec::new(),
            tail_values: [[ZERO; 3]; 64],
            cycle_eq: None,
            final_values: [ZERO; 3],
            state: State::Round(0),
        })
    }

    /// Returns the three defining multilinear extensions at the fully bound
    /// challenge point after `finish_rounds`; before completion returns zeros.
    pub fn final_values(&self) -> [F128; 3] {
        self.final_values
    }

    /// Reports the first cycle whose lane word or six tail bits violate
    /// `C = A & B`. This scan is independent of the constructor and proof.
    pub fn check_rows(source: &S) -> Result<(), usize> {
        for cycle in 0..source.cycles() {
            if source.lanes(cycle).iter().any(|&[a, b, c]| a & b != c) {
                return Err(cycle);
            }
            let tail = source.tail(cycle);
            if (tail & 3) & ((tail >> 2) & 3) != (tail >> 4) & 3 {
                return Err(cycle);
            }
        }
        Ok(())
    }

    fn position<const AT_ONE: bool>(&mut self, k: usize) -> Result<Sums, OuterError> {
        let rho = eq_table(&self.tau[k + 1..6], None);
        let sums = if k < self.options.monomial_rounds {
            let form = Monomial::new(
                &self.point,
                &rho,
                &self.omega,
                k >= 2 || k == 1 && self.options.nibble_round_2,
            );
            let pool = (k == 0 && !AT_ONE)
                .then(|| ScratchPool::new(64))
                .transpose()?;
            macro_rules! pass {
                ($k:literal) => {
                    form.pass::<$k, AT_ONE, S>(
                        &*self.source,
                        self.chunks,
                        &self.lo,
                        &self.hi,
                        pool.as_ref(),
                    )?
                };
            }
            let sums = match k {
                0 => pass!(0),
                1 => pass!(1),
                2 => pass!(2),
                3 => pass!(3),
                4 => pass!(4),
                _ => pass!(5),
            };
            if let Some(pool) = pool {
                self.histogram.copy_from_slice(&pool.merge()?);
            }
            sums
        } else {
            macro_rules! window {
                ($n:literal, $a:literal, $c:literal, $units:literal) => {
                    Window::<$n, $a, $c>::new(
                        &self.point,
                        &rho,
                        &self.omega,
                        self.options.folded_group_weights,
                    )
                    .pass::<$units, AT_ONE, S>(&*self.source, self.chunks, &self.lo, &self.hi)
                };
            }
            match k {
                2 => window!(16, 16, 8, 1),
                3 => window!(256, 8, 4, 1),
                4 => window!(256, 8, 4, 2),
                _ => window!(256, 8, 4, 4),
            }
        };
        let tail = self.tail_position(k, AT_ONE);
        Ok([sums[0] + tail[0], sums[1] + tail[1]])
    }

    fn weighted_source(&self, value: impl Fn(usize) -> F128 + Sync) -> F128 {
        (0..self.chunks.len() / self.chunks.chunk_len())
            .into_par_iter()
            .map(|chunk| {
                let start = chunk * self.chunks.chunk_len();
                let first = start / self.chunks.block_len();
                let blocks = self.chunks.chunk_len() / self.chunks.block_len();
                let mut total = F128Accumulator::default();
                for (block, &hi) in self.hi[first..first + blocks].iter().enumerate() {
                    let mut inner = F128Accumulator::default();
                    for (offset, &lo) in self.lo.iter().enumerate() {
                        inner.fmadd(lo, value(start + block * self.chunks.block_len() + offset));
                    }
                    total.fmadd(hi, inner.reduce());
                }
                total
            })
            .reduce(F128Accumulator::default, |mut a, b| {
                a.merge(b);
                a
            })
            .reduce()
    }

    fn tail_at(&self, byte: usize) -> [F128; 3] {
        std::array::from_fn(|lane| {
            let lo = F128::from_raw(((byte >> (2 * lane)) & 1) as u128);
            let hi = F128::from_raw(((byte >> (2 * lane + 1)) & 1) as u128);
            let mut value = lo + self.point[0] * (lo + hi);
            for &r in &self.point[1..6] {
                value *= ONE + r;
            }
            value
        })
    }

    fn tail_position(&self, k: usize, at_one: bool) -> Sums {
        let rho: F128 = self.tau[k + 1..6].iter().map(|&t| ONE + t).product();
        let mut sums = [F128Accumulator::default(); 2];
        for (byte, &weight) in self.histogram.iter().enumerate() {
            let (values, delta) = if k == 0 {
                let lo: [F128; 3] =
                    std::array::from_fn(|lane| F128::from_raw(((byte >> (2 * lane)) & 1) as u128));
                let hi: [F128; 3] = std::array::from_fn(|lane| {
                    F128::from_raw(((byte >> (2 * lane + 1)) & 1) as u128)
                });
                (if at_one { hi } else { lo }, [lo[0] + hi[0], lo[1] + hi[1]])
            } else {
                let values: [F128; 3] = std::array::from_fn(|lane| {
                    let lo = F128::from_raw(((byte >> (2 * lane)) & 1) as u128);
                    let hi = F128::from_raw(((byte >> (2 * lane + 1)) & 1) as u128);
                    let mut v = lo + self.point[0] * (lo + hi);
                    for &r in &self.point[1..] {
                        v *= ONE + r;
                    }
                    v
                });
                (
                    if at_one { [ZERO; 3] } else { values },
                    [values[0], values[1]],
                )
            };
            sums[0].fmadd(weight, values[0] * values[1] + values[2]);
            sums[1].fmadd(weight, delta[0] * delta[1]);
        }
        sums.map(|sum| sum.reduce() * rho * self.omega[2])
    }

    fn materialise(&mut self) -> Sums {
        let weights: [F128; 64] = std::array::from_fn(|index| {
            self.point
                .iter()
                .enumerate()
                .map(|(bit, &r)| if index >> bit & 1 == 0 { ONE + r } else { r })
                .product()
        });
        let lift = WordLift::new(&weights);
        self.tail_values = std::array::from_fn(|byte| self.tail_at(byte));
        self.groups =
            std::array::from_fn(|_| std::array::from_fn(|_| vec![ZERO; self.chunks.len()]));
        self.tail = vec![0; self.chunks.len()];
        let [group0, group1] = &mut self.groups;
        let [a0, b0, c0] = group0;
        let [a1, b1, c1] = group1;
        let chunk_len = self.chunks.chunk_len();
        let block_len = self.chunks.block_len();
        let lo = &self.lo;
        let hi = &self.hi;
        let source = &self.source;
        let sums = a0
            .par_chunks_mut(chunk_len)
            .zip(b0.par_chunks_mut(chunk_len))
            .zip(c0.par_chunks_mut(chunk_len))
            .zip(a1.par_chunks_mut(chunk_len))
            .zip(b1.par_chunks_mut(chunk_len))
            .zip(c1.par_chunks_mut(chunk_len))
            .zip(self.tail.par_chunks_mut(chunk_len))
            .enumerate()
            .map(|(chunk, ((((((a0, b0), c0), a1), b1), c1), tail))| {
                let mut total = [F128Accumulator::default(); 2];
                let first = chunk * chunk_len / block_len;
                for (block, ((((((a0, b0), c0), a1), b1), c1), tail)) in a0
                    .chunks_exact_mut(block_len)
                    .zip(b0.chunks_exact_mut(block_len))
                    .zip(c0.chunks_exact_mut(block_len))
                    .zip(a1.chunks_exact_mut(block_len))
                    .zip(b1.chunks_exact_mut(block_len))
                    .zip(c1.chunks_exact_mut(block_len))
                    .zip(tail.chunks_exact_mut(block_len))
                    .enumerate()
                {
                    let mut inner = [F128Accumulator::default(); 2];
                    for (offset, (((((((a0, b0), c0), a1), b1), c1), tail), &weight)) in a0
                        .iter_mut()
                        .zip(b0)
                        .zip(c0)
                        .zip(a1)
                        .zip(b1)
                        .zip(c1)
                        .zip(tail)
                        .zip(lo)
                        .enumerate()
                    {
                        let cycle = chunk * chunk_len + block * block_len + offset;
                        let [left, right] = source
                            .lanes(cycle)
                            .map(|lane| lane.map(|word| lift.lift(word)));
                        *a0 = left[0];
                        *b0 = left[1];
                        *c0 = left[2];
                        *a1 = right[0];
                        *b1 = right[1];
                        *c1 = right[2];
                        *tail = source.tail(cycle) & 63;
                        let mut q = F128Accumulator::default();
                        q.fmadd(left[0], left[1]);
                        q.add(left[2]);
                        inner[0].fmadd(weight, q.reduce());
                        inner[1].fmadd(weight, (left[0] + right[0]) * (left[1] + right[1]));
                    }
                    for (total, value) in total.iter_mut().zip(inner) {
                        total.fmadd(hi[first + block], value.reduce());
                    }
                }
                total
            })
            .reduce(|| [F128Accumulator::default(); 2], merge)
            .map(Accumulator::reduce);
        let mut tail = [F128Accumulator::default(); 2];
        for (&weight, values) in self.histogram.iter().zip(self.tail_values) {
            tail[0].fmadd(weight, values[0] * values[1] + values[2]);
            tail[1].fmadd(weight, values[0] * values[1]);
        }
        [
            sums[0] * (ONE + self.tau[7]) + tail[0].reduce() * self.tau[7],
            sums[1] * (ONE + self.tau[7]) + tail[1].reduce() * self.tau[7],
        ]
    }

    fn group_one(&self, round: usize) -> F128 {
        if round == 6 {
            let lanes = self.weighted_source(|cycle| {
                let a = self.groups[1][0][cycle];
                let b = self.groups[1][1][cycle];
                let c = self.groups[1][2][cycle];
                a * b + c
            });
            lanes * (ONE + self.tau[7])
        } else {
            self.weighted_source(|cycle| {
                let [a, b, c] = self.tail_values[usize::from(self.tail[cycle])];
                a * b + c
            })
        }
    }

    fn group_bind(&mut self, r: F128) -> Sums {
        let scale = ONE + r;
        for values in &mut self.tail_values {
            for value in values {
                *value *= scale;
            }
        }
        let [group0, group1] = &mut self.groups;
        let [a, b, c] = group0;
        let [a1, b1, c1] = group1;
        let chunk_len = self.chunks.chunk_len();
        let block_len = self.chunks.block_len();
        let lo = &self.lo;
        let hi = &self.hi;
        let tails = &self.tail_values;
        a.par_chunks_mut(chunk_len)
            .zip(b.par_chunks_mut(chunk_len))
            .zip(c.par_chunks_mut(chunk_len))
            .zip(a1.par_chunks(chunk_len))
            .zip(b1.par_chunks(chunk_len))
            .zip(c1.par_chunks(chunk_len))
            .zip(self.tail.par_chunks(chunk_len))
            .enumerate()
            .map(|(chunk, ((((((a, b), c), a1), b1), c1), tail))| {
                let mut total = [F128Accumulator::default(); 2];
                let first = chunk * chunk_len / block_len;
                for (block, ((((((a, b), c), a1), b1), c1), tail)) in a
                    .chunks_exact_mut(block_len)
                    .zip(b.chunks_exact_mut(block_len))
                    .zip(c.chunks_exact_mut(block_len))
                    .zip(a1.chunks_exact(block_len))
                    .zip(b1.chunks_exact(block_len))
                    .zip(c1.chunks_exact(block_len))
                    .zip(tail.chunks_exact(block_len))
                    .enumerate()
                {
                    let mut inner = [F128Accumulator::default(); 2];
                    for (((((((a, b), c), &a1), &b1), &c1), &tail), &weight) in a
                        .iter_mut()
                        .zip(b)
                        .zip(c)
                        .zip(a1)
                        .zip(b1)
                        .zip(c1)
                        .zip(tail)
                        .zip(lo)
                    {
                        *a += r * (*a + a1);
                        *b += r * (*b + b1);
                        *c += r * (*c + c1);
                        let [ta, tb, _] = tails[usize::from(tail)];
                        inner[0].fmadd(weight, *a * *b + *c);
                        inner[1].fmadd(weight, (*a + ta) * (*b + tb));
                    }
                    for (total, value) in total.iter_mut().zip(inner) {
                        total.fmadd(hi[first + block], value.reduce());
                    }
                }
                total
            })
            .reduce(|| [F128Accumulator::default(); 2], merge)
            .map(Accumulator::reduce)
    }

    fn cycle_first(&mut self, r: F128, at_one: bool) -> Sums {
        let eq = self.cycle_eq.as_ref();
        let Some(eq) = eq else {
            return [ZERO; 2];
        };
        let lo = eq.e_in_current();
        let hi = eq.e_out_current();
        let block = 2 * lo.len();
        let chunk_len = self.cycle_chunks[0].chunk_len() * 2;
        let tails = &self.tail_values;
        let [a, b, c] = &mut self.groups[0];
        a.par_chunks_mut(chunk_len)
            .zip(b.par_chunks_mut(chunk_len))
            .zip(c.par_chunks_mut(chunk_len))
            .zip(self.tail.par_chunks(chunk_len))
            .enumerate()
            .map(|(chunk, (((a, b), c), tail))| {
                let mut total = [F128Accumulator::default(); 2];
                let first = chunk * chunk_len / block;
                for (block_index, (((a, b), c), tail)) in a
                    .chunks_exact_mut(block)
                    .zip(b.chunks_exact_mut(block))
                    .zip(c.chunks_exact_mut(block))
                    .zip(tail.chunks_exact(block))
                    .enumerate()
                {
                    let mut inner = [F128Accumulator::default(); 2];
                    for ((((a, b), c), tail), &weight) in a
                        .chunks_exact_mut(2)
                        .zip(b.chunks_exact_mut(2))
                        .zip(c.chunks_exact_mut(2))
                        .zip(tail.chunks_exact(2))
                        .zip(lo)
                    {
                        if !at_one {
                            for (lane, table) in [&mut *a, &mut *b, &mut *c].into_iter().enumerate()
                            {
                                for (value, &byte) in table.iter_mut().zip(tail) {
                                    *value += r * (*value + tails[usize::from(byte)][lane]);
                                }
                            }
                        }
                        let endpoint = usize::from(at_one);
                        inner[0].fmadd(weight, a[endpoint] * b[endpoint] + c[endpoint]);
                        inner[1].fmadd(weight, (a[0] + a[1]) * (b[0] + b[1]));
                    }
                    for (total, value) in total.iter_mut().zip(inner) {
                        total.fmadd(hi[first + block_index], value.reduce());
                    }
                }
                total
            })
            .reduce(|| [F128Accumulator::default(); 2], merge)
            .map(Accumulator::reduce)
    }

    fn cycle_next(&mut self, round: usize, r: F128, at_one: bool) -> Sums {
        let Some(eq) = &self.cycle_eq else {
            return [ZERO; 2];
        };
        let lo = eq.e_in_current();
        let hi = eq.e_out_current();
        let out_block = 2 * lo.len();
        let remaining = self.chunks.len() >> round;
        let out_chunk = self.cycle_chunks[round].chunk_len() * 2;
        let [input, output] = &mut self.groups;
        let [a, b, c] = input;
        let [oa, ob, oc] = output;
        oa.truncate(remaining);
        ob.truncate(remaining);
        oc.truncate(remaining);
        let sums = oa
            .par_chunks_mut(out_chunk)
            .zip(ob.par_chunks_mut(out_chunk))
            .zip(oc.par_chunks_mut(out_chunk))
            .zip(a.par_chunks(2 * out_chunk))
            .zip(b.par_chunks(2 * out_chunk))
            .zip(c.par_chunks(2 * out_chunk))
            .enumerate()
            .map(|(chunk, (((((oa, ob), oc), a), b), c))| {
                let mut total = [F128Accumulator::default(); 2];
                let first = chunk * out_chunk / out_block;
                for (block_index, (((((oa, ob), oc), a), b), c)) in oa
                    .chunks_exact_mut(out_block)
                    .zip(ob.chunks_exact_mut(out_block))
                    .zip(oc.chunks_exact_mut(out_block))
                    .zip(a.chunks_exact(2 * out_block))
                    .zip(b.chunks_exact(2 * out_block))
                    .zip(c.chunks_exact(2 * out_block))
                    .enumerate()
                {
                    let mut inner = [F128Accumulator::default(); 2];
                    for ((((((oa, ob), oc), a), b), c), &weight) in oa
                        .chunks_exact_mut(2)
                        .zip(ob.chunks_exact_mut(2))
                        .zip(oc.chunks_exact_mut(2))
                        .zip(a.chunks_exact(4))
                        .zip(b.chunks_exact(4))
                        .zip(c.chunks_exact(4))
                        .zip(lo)
                    {
                        for (out, input) in [(&mut *oa, a), (&mut *ob, b), (&mut *oc, c)] {
                            for (out, pair) in out.iter_mut().zip(input.chunks_exact(2)) {
                                *out = pair[0] + r * (pair[0] + pair[1]);
                            }
                        }
                        let endpoint = usize::from(at_one);
                        inner[0].fmadd(weight, oa[endpoint] * ob[endpoint] + oc[endpoint]);
                        inner[1].fmadd(weight, (oa[0] + oa[1]) * (ob[0] + ob[1]));
                    }
                    for (total, value) in total.iter_mut().zip(inner) {
                        total.fmadd(hi[first + block_index], value.reduce());
                    }
                }
                total
            })
            .reduce(|| [F128Accumulator::default(); 2], merge)
            .map(Accumulator::reduce);
        std::mem::swap(input, output);
        sums
    }

    fn cycle_one(&self, round: usize) -> F128 {
        let Some(eq) = &self.cycle_eq else {
            return ZERO;
        };
        let lo = eq.e_in_current();
        let hi = eq.e_out_current();
        let block_len = lo.len() * 2;
        let chunks = self.cycle_chunks[round].chunk_len() * 2;
        let [a, b, c] = &self.groups[0];
        a.par_chunks(chunks)
            .zip(b.par_chunks(chunks))
            .zip(c.par_chunks(chunks))
            .enumerate()
            .map(|(chunk, ((a, b), c))| {
                let mut total = F128Accumulator::default();
                for (block, ((a, b), c)) in a
                    .chunks_exact(block_len)
                    .zip(b.chunks_exact(block_len))
                    .zip(c.chunks_exact(block_len))
                    .enumerate()
                {
                    let mut inner = F128Accumulator::default();
                    for (((a, b), c), &weight) in a
                        .chunks_exact(2)
                        .zip(b.chunks_exact(2))
                        .zip(c.chunks_exact(2))
                        .zip(lo)
                    {
                        inner.fmadd(weight, a[1] * b[1] + c[1]);
                    }
                    total.fmadd(hi[chunk * chunks / block_len + block], inner.reduce());
                }
                total
            })
            .reduce(F128Accumulator::default, |mut a, b| {
                a.merge(b);
                a
            })
            .reduce()
    }
}

impl<S: LaneSource> ProveRounds<F128> for OuterF2Core<S> {
    fn num_rounds(&self) -> usize {
        8 + self.log_t
    }

    fn prove_round(
        &mut self,
        bind: Option<F128>,
        round: usize,
        previous_claim: F128,
    ) -> Result<UnivariatePoly<F128>, SumcheckError<F128>> {
        let State::Round(expected) = self.state else {
            return Err(SumcheckError::WrongNumberOfRounds {
                expected: self.num_rounds(),
                got: round,
            });
        };
        if round != expected {
            return Err(SumcheckError::WrongNumberOfRounds {
                expected,
                got: round,
            });
        }
        if (round == 0) != bind.is_none() {
            return Err(SumcheckError::MissingEvaluationSource {
                kind: "outer previous challenge",
            });
        }
        self.state = State::Failed;
        if let Some(r) = bind {
            if round <= 8 {
                self.sigma *= ONE + self.tau[round - 1] + r;
                self.point.push(r);
            } else if let Some(eq) = &mut self.cycle_eq {
                eq.bind(r);
            }
        }
        let sums = match round {
            0..=5 => self.position::<false>(round).map_err(|_| {
                SumcheckError::MissingEvaluationSource {
                    kind: "outer position scratch",
                }
            })?,
            6 => self.materialise(),
            7 => self.group_bind(bind.unwrap_or(ZERO)),
            8 => {
                self.cycle_eq = Some(split_eq(&self.tau[8..], Some(self.sigma)).map_err(|_| {
                    SumcheckError::MissingEvaluationSource {
                        kind: "outer cycle equality",
                    }
                })?);
                self.cycle_first(bind.unwrap_or(ZERO), false)
            }
            _ => self.cycle_next(round - 8, bind.unwrap_or(ZERO), false),
        };
        let poly = if round < 8 {
            let linear = (
                self.sigma * (ONE + self.tau[round]),
                self.sigma * self.tau[round],
            );
            let endpoint = if linear.1 == ZERO {
                if round < 6 {
                    self.position::<true>(round).map_err(|_| {
                        SumcheckError::MissingEvaluationSource {
                            kind: "outer endpoint scratch",
                        }
                    })?[0]
                } else {
                    self.group_one(round)
                }
            } else {
                ZERO
            };
            let one =
                gruen_recover_endpoint(linear.0 * sums[0], linear.1, previous_claim, || endpoint)
                    .map_err(|actual| SumcheckError::RoundCheckFailed {
                    round,
                    expected: previous_claim,
                    actual,
                })?;
            let coefficients =
                coefficients_from_nodes(2, sums[0], sums[1], one, &[]).map_err(|_| {
                    SumcheckError::MissingEvaluationSource {
                        kind: "outer quadratic",
                    }
                })?;
            gruen_mul_linear(linear, &coefficients)
        } else {
            let Some(eq) = &self.cycle_eq else {
                return Err(SumcheckError::MissingEvaluationSource {
                    kind: "outer cycle equality",
                });
            };
            let one = eq
                .recover_q_one(sums[0], previous_claim, || self.cycle_one(round - 8))
                .map_err(|actual| SumcheckError::RoundCheckFailed {
                    round,
                    expected: previous_claim,
                    actual,
                })?;
            let coefficients =
                coefficients_from_nodes(2, sums[0], sums[1], one, &[]).map_err(|_| {
                    SumcheckError::MissingEvaluationSource {
                        kind: "outer quadratic",
                    }
                })?;
            eq.round_poly_from_q_coeffs(&coefficients)
        };
        self.state = if round + 1 == self.num_rounds() {
            State::LastBind
        } else {
            State::Round(round + 1)
        };
        Ok(poly)
    }

    fn finish_rounds(&mut self, bind: F128) -> Result<(), SumcheckError<F128>> {
        if !matches!(self.state, State::LastBind) {
            return Err(SumcheckError::MissingEvaluationSource {
                kind: "outer final challenge",
            });
        }
        self.final_values = std::array::from_fn(|lane| {
            self.groups[0][lane][0] + bind * (self.groups[0][lane][0] + self.groups[0][lane][1])
        });
        self.groups = std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
        self.tail = Vec::new();
        self.lo = Vec::new();
        self.hi = Vec::new();
        self.cycle_eq = None;
        self.state = State::Finished;
        Ok(())
    }
}

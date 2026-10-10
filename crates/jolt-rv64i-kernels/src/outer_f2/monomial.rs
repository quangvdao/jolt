use super::{merge, OuterError, Sums, ONE, ZERO};
use crate::packed::bits::{gather, moebius};
use crate::packed::lift::{CompactLift, CompactView};
use crate::packed::pool::ScratchPool;
use crate::par::CycleChunks;
use crate::source::LaneSource;
use jolt_field::{Accumulator, F128Accumulator, F128};
use rayon::prelude::*;

struct TermView<'a, const N: usize, const TABLES: usize> {
    pairs: &'a [(usize, usize, usize)],
    x: [CompactView<'a, N, TABLES>; 2],
    y: Option<[CompactView<'a, N, TABLES>; 2]>,
}

struct Term {
    start: usize,
    end: usize,
    x: [CompactLift; 2],
    y: Option<[CompactLift; 2]>,
}

pub(super) struct Monomial {
    terms: Vec<Term>,
    pairs: Vec<(usize, usize, usize)>,
    tables: Vec<F128>,
}

impl Monomial {
    pub(super) fn new(
        point: &[F128],
        rho: &[F128],
        omega: &[F128],
        nibble: bool,
    ) -> Result<Self, OuterError> {
        let k = point.len();
        let count = 3_usize.pow(k as u32);
        let mut pairs = Vec::with_capacity(1 << (2 * k));
        for a in 0..1 << k {
            for b in 0..1 << k {
                let mut exponent = 0;
                let mut power = 1;
                for bit in 0..k {
                    exponent += (((a >> bit) & 1) + ((b >> bit) & 1)) * power;
                    power *= 3;
                }
                pairs.push((exponent, a, b));
            }
        }
        pairs.sort_unstable_by_key(|&(exponent, _, _)| exponent);
        let bits = if nibble { 4 } else { 8 }.min(rho.len());
        let table_size = (1 << bits) * rho.len().div_ceil(bits);
        let mut tables = Vec::with_capacity(2 * (2 * count - (1 << k)) * table_size);
        let mut terms = Vec::with_capacity(count);
        let mut start = 0;
        for exponent in 0..count {
            let mut digits = exponent;
            let mut scalar = ONE;
            let mut reduced = ONE;
            let mut squared = false;
            for &r in point {
                match digits % 3 {
                    1 => {
                        scalar *= r;
                        reduced *= r;
                    }
                    2 => {
                        scalar *= r * r;
                        reduced *= r;
                        squared = true;
                    }
                    _ => {}
                }
                digits /= 3;
            }
            let mut lifts = |scalar| {
                let mut build = |group: usize| {
                    let mut weights = [ZERO; 32];
                    for (out, &weight) in weights.iter_mut().zip(rho) {
                        *out = weight * omega[group] * scalar;
                    }
                    CompactLift::new(&weights[..rho.len()], bits, &mut tables)
                };
                Ok::<_, OuterError>([build(0)?, build(1)?])
            };
            let end = start
                + pairs[start..]
                    .iter()
                    .take_while(|&&(e, _, _)| e == exponent)
                    .count();
            terms.push(Term {
                start,
                end,
                x: lifts(scalar)?,
                y: squared.then(|| lifts(scalar + reduced)).transpose()?,
            });
            start = end;
        }
        Ok(Self {
            terms,
            pairs,
            tables,
        })
    }

    #[inline]
    fn values<const K: usize, const AT_ONE: bool, const N: usize, const TABLES: usize>(
        terms: &[TermView<'_, N, TABLES>],
        lanes: [[u64; 3]; 2],
    ) -> Sums {
        let mut sums = [ZERO; 2];
        for (group, [a, b, _]) in lanes.into_iter().enumerate() {
            // K is dispatched in 0..=5; the checked helpers' error branches fold away.
            let a = moebius(a, K + 1).unwrap_or(0);
            let b = moebius(b, K + 1).unwrap_or(0);
            if K <= 2 {
                let count = 1 << K;
                let x_a: [u64; 4] = std::array::from_fn(|i| a >> (count + i));
                let x_b: [u64; 4] = std::array::from_fn(|i| b >> (count + i));
                let y_a: [u64; 4] = std::array::from_fn(|i| {
                    if AT_ONE {
                        (a >> i) ^ (a >> (count + i))
                    } else {
                        a >> i
                    }
                });
                let y_b: [u64; 4] = std::array::from_fn(|i| {
                    if AT_ONE {
                        (b >> i) ^ (b >> (count + i))
                    } else {
                        b >> i
                    }
                });
                let x = products::<K>(x_a, x_b);
                let y = repeated_products::<K>(y_a, y_b);
                for (term, (&x, &y)) in terms.iter().zip(x.iter().zip(&y)) {
                    sums[1] += term.x[group].lift(gather(x, K + 1).unwrap_or(0));
                    if let Some(lifts) = term.y {
                        sums[0] += lifts[group].lift(gather(y, K + 1).unwrap_or(0));
                    }
                }
                continue;
            }
            for term in terms {
                let mut x = 0;
                let mut y = 0;
                for &(_, left, right) in term.pairs {
                    x ^= (a >> (left + (1 << K))) & (b >> (right + (1 << K)));
                    if term.y.is_some() {
                        let a0 = a >> left;
                        let b0 = b >> right;
                        y ^= if AT_ONE {
                            (a0 ^ (a0 >> (1 << K))) & (b0 ^ (b0 >> (1 << K)))
                        } else {
                            a0 & b0
                        };
                    }
                }
                sums[1] += term.x[group].lift(gather(x, K + 1).unwrap_or(0));
                if let Some(lifts) = &term.y {
                    sums[0] += lifts[group].lift(gather(y, K + 1).unwrap_or(0));
                }
            }
        }
        sums
    }

    pub(super) fn pass<
        const K: usize,
        const AT_ONE: bool,
        const N: usize,
        const TABLES: usize,
        S: LaneSource,
    >(
        &self,
        source: &S,
        chunks: CycleChunks,
        lo: &[F128],
        hi: &[F128],
        histogram: Option<&ScratchPool>,
    ) -> Result<Sums, OuterError> {
        let terms = self
            .terms
            .iter()
            .map(|term| {
                let views = |lifts: [CompactLift; 2]| -> Result<_, OuterError> {
                    Ok([
                        lifts[0].view::<N, TABLES>(&self.tables)?,
                        lifts[1].view::<N, TABLES>(&self.tables)?,
                    ])
                };
                Ok::<_, OuterError>(TermView {
                    pairs: &self.pairs[term.start..term.end],
                    x: views(term.x)?,
                    y: term.y.map(views).transpose()?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let sums = (0..chunks.len() / chunks.chunk_len())
            .into_par_iter()
            .map(|chunk| {
                let mut scratch = histogram.map(ScratchPool::take).transpose()?;
                let mut total = [F128Accumulator::default(); 2];
                let start = chunk * chunks.chunk_len();
                let first_block = start / chunks.block_len();
                let blocks = chunks.chunk_len() / chunks.block_len();
                for (block, &high) in hi[first_block..first_block + blocks].iter().enumerate() {
                    let mut inner = [F128Accumulator::default(); 2];
                    let mut h = [ZERO; 64];
                    for (offset, &low) in lo.iter().enumerate() {
                        let cycle = start + block * chunks.block_len() + offset;
                        let values =
                            Self::values::<K, AT_ONE, N, TABLES>(&terms, source.lanes(cycle));
                        if K != 0 {
                            inner[0].fmadd(low, values[0]);
                        }
                        inner[1].fmadd(low, values[1]);
                        if scratch.is_some() {
                            h[usize::from(source.tail(cycle) & 63)] += low;
                        }
                    }
                    if K != 0 {
                        total[0].fmadd(high, inner[0].reduce());
                    }
                    total[1].fmadd(high, inner[1].reduce());
                    if let Some(scratch) = &mut scratch {
                        for (dest, value) in scratch.iter_mut().zip(h) {
                            *dest += high * value;
                        }
                    }
                }
                Ok::<_, OuterError>(total)
            })
            .try_reduce(
                || [F128Accumulator::default(); 2],
                |left, right| Ok(merge(left, right)),
            )?;
        Ok(sums.map(Accumulator::reduce))
    }
}

// Base-three exponent order for two variables: 00,10,20,01,11,21,02,12,22.
#[inline]
fn products<const K: usize>(a: [u64; 4], b: [u64; 4]) -> [u64; 9] {
    let mut values = [0; 9];
    values[0] = a[0] & b[0];
    if K >= 1 {
        values[1] = (a[1] & b[0]) ^ (a[0] & b[1]);
        values[2] = a[1] & b[1];
    }
    if K >= 2 {
        values[3] = (a[2] & b[0]) ^ (a[0] & b[2]);
        values[4] = (a[3] & b[0]) ^ (a[2] & b[1]) ^ (a[1] & b[2]) ^ (a[0] & b[3]);
        values[5] = (a[3] & b[1]) ^ (a[1] & b[3]);
        values[6] = a[2] & b[2];
        values[7] = (a[3] & b[2]) ^ (a[2] & b[3]);
        values[8] = a[3] & b[3];
    }
    values
}

#[inline]
fn repeated_products<const K: usize>(a: [u64; 4], b: [u64; 4]) -> [u64; 9] {
    let mut values = [0; 9];
    if K >= 1 {
        values[2] = a[1] & b[1];
    }
    if K >= 2 {
        values[5] = (a[3] & b[1]) ^ (a[1] & b[3]);
        values[6] = a[2] & b[2];
        values[7] = (a[3] & b[2]) ^ (a[2] & b[3]);
        values[8] = a[3] & b[3];
    }
    values
}

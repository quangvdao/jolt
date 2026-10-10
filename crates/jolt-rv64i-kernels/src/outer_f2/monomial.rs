use super::{merge, OuterError, Sums, ONE, ZERO};
use crate::packed::bits::{gather, moebius};
use crate::packed::lift::{CompactLift, CompactView, LiftError};
use crate::packed::pool::ScratchPool;
use crate::par::CycleChunks;
use crate::source::LaneSource;
use jolt_field::{Accumulator, F128Accumulator, F128};
use rayon::prelude::*;

#[derive(Clone, Copy)]
struct Exponent {
    digits: [usize; 5],
    pairs: [(usize, usize); 32],
    len: usize,
    red: usize,
    squared: bool,
}

impl Exponent {
    const fn new(k: usize, mut exponent: usize) -> Self {
        let mut result = Self {
            digits: [0; 5],
            pairs: [(0, 0); 32],
            len: 0,
            red: 0,
            squared: false,
        };
        let mut choices = 0;
        let mut bit = 0;
        while bit < k {
            let digit = exponent % 3;
            result.digits[bit] = digit;
            if digit != 0 {
                result.red |= 1 << bit;
            }
            if digit == 1 {
                choices += 1;
            }
            if digit == 2 {
                result.squared = true;
            }
            exponent /= 3;
            bit += 1;
        }
        result.len = 1 << choices;
        let mut choice = 0;
        while choice < result.len {
            let mut left = 0;
            let mut right = 0;
            let mut coordinate = 0;
            let mut bit = 0;
            while bit < k {
                match result.digits[bit] {
                    1 => {
                        let selected = (choice >> coordinate) & 1;
                        left |= selected << bit;
                        right |= (1 ^ selected) << bit;
                        coordinate += 1;
                    }
                    2 => {
                        left |= 1 << bit;
                        right |= 1 << bit;
                    }
                    _ => {}
                }
                bit += 1;
            }
            result.pairs[choice] = (left, right);
            choice += 1;
        }
        result
    }

    const fn squared_slot(k: usize, exponent: usize) -> usize {
        let mut slot = 0;
        let mut e = 0;
        while e < exponent {
            if Self::new(k, e).squared {
                slot += 1;
            }
            e += 1;
        }
        slot
    }

    const fn schedule<const COUNT: usize>(k: usize, squared_only: bool) -> [Self; COUNT] {
        let mut result = [Self::new(0, 0); COUNT];
        let mut i = 0;
        let mut e = 0;
        while i < COUNT {
            let description = Self::new(k, e);
            if !squared_only || description.squared {
                result[i] = description;
                i += 1;
            }
            e += 1;
        }
        result
    }

    #[inline(always)]
    fn product(&self, a: u64, b: u64, shift: usize) -> u64 {
        let mut product = 0;
        for &(left, right) in &self.pairs[..self.len] {
            product ^= (a >> (shift + left)) & (b >> (shift + right));
        }
        product
    }
}

struct Views<'a, const N: usize, const TABLES: usize, const COUNT: usize, const SQUARED: usize> {
    x: [[CompactView<'a, N, TABLES>; 2]; COUNT],
    squared: [[CompactView<'a, N, TABLES>; 2]; SQUARED],
}

pub(super) struct Monomial {
    x: Vec<[CompactLift; 2]>,
    squared: Vec<[CompactLift; 2]>,
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
        let bits = if nibble { 4 } else { 8 }.min(rho.len());
        let table_size = (1 << bits) * rho.len().div_ceil(bits);
        let mut tables = Vec::with_capacity(2 * (2 * count - (1 << k)) * table_size);
        let mut x = Vec::with_capacity(count);
        let mut squared = Vec::with_capacity(count - (1 << k));
        for exponent in 0..count {
            let description = Exponent::new(k, exponent);
            let mut scalar = ONE;
            let mut reduced = ONE;
            for (bit, &r) in point.iter().enumerate() {
                match description.digits[bit] {
                    1 => scalar *= r,
                    2 => scalar *= r * r,
                    _ => {}
                }
                if description.red >> bit & 1 != 0 {
                    reduced *= r;
                }
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
            x.push(lifts(scalar)?);
            if description.squared {
                squared.push(lifts(scalar + reduced)?);
            }
        }
        Ok(Self { x, squared, tables })
    }

    #[inline(always)]
    fn term<
        const K: usize,
        const E: usize,
        const AT_ONE: bool,
        const N: usize,
        const TABLES: usize,
        const COUNT: usize,
        const SQUARED: usize,
    >(
        views: &Views<'_, N, TABLES, COUNT, SQUARED>,
        a: u64,
        b: u64,
        group: usize,
        sums: &mut Sums,
    ) {
        let description = const { Exponent::new(K, E) };
        let x = description.product(a, b, 1 << K);
        sums[1] += views.x[E][group].lift(gather(x, K + 1).unwrap_or(0));
        if const { Exponent::new(K, E).squared } {
            let (a, b) = if AT_ONE {
                (a ^ (a >> (1 << K)), b ^ (b >> (1 << K)))
            } else {
                (a, b)
            };
            let squared_product = description.product(a, b, 0);
            sums[0] += views.squared[const { Exponent::squared_slot(K, E) }][group]
                .lift(gather(squared_product, K + 1).unwrap_or(0));
        }
    }

    #[inline(always)]
    fn values<
        const K: usize,
        const AT_ONE: bool,
        const N: usize,
        const TABLES: usize,
        const COUNT: usize,
        const SQUARED: usize,
    >(
        views: &Views<'_, N, TABLES, COUNT, SQUARED>,
        lanes: [[u64; 3]; 2],
    ) -> Sums {
        let mut sums = [ZERO; 2];
        for (group, [a, b, _]) in lanes.into_iter().enumerate() {
            let a = moebius(a, K + 1).unwrap_or(0);
            let b = moebius(b, K + 1).unwrap_or(0);
            if K <= 2 {
                macro_rules! term {
                    ($e:literal) => {
                        Self::term::<K, $e, AT_ONE, N, TABLES, COUNT, SQUARED>(
                            views, a, b, group, &mut sums,
                        )
                    };
                }
                term!(0);
                if K >= 1 {
                    term!(1);
                    term!(2);
                }
                if K >= 2 {
                    term!(3);
                    term!(4);
                    term!(5);
                    term!(6);
                    term!(7);
                    term!(8);
                }
            } else {
                for (description, lifts) in const { Exponent::schedule::<COUNT>(K, false) }
                    .iter()
                    .zip(&views.x)
                {
                    let x = description.product(a, b, 1 << K);
                    sums[1] += lifts[group].lift(gather(x, K + 1).unwrap_or(0));
                }
                let (a, b) = if AT_ONE {
                    (a ^ (a >> (1 << K)), b ^ (b >> (1 << K)))
                } else {
                    (a, b)
                };
                for (description, lifts) in const { Exponent::schedule::<SQUARED>(K, true) }
                    .iter()
                    .zip(&views.squared)
                {
                    let squared_product = description.product(a, b, 0);
                    sums[0] += lifts[group].lift(gather(squared_product, K + 1).unwrap_or(0));
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
        const COUNT: usize,
        const SQUARED: usize,
        S: LaneSource,
    >(
        &self,
        source: &S,
        chunks: CycleChunks,
        lo: &[F128],
        hi: &[F128],
        histogram: Option<&ScratchPool>,
    ) -> Result<Sums, OuterError> {
        let convert = |lifts: &[[CompactLift; 2]]| {
            lifts
                .iter()
                .map(|pair| {
                    Ok([
                        pair[0].view::<N, TABLES>(&self.tables)?,
                        pair[1].view::<N, TABLES>(&self.tables)?,
                    ])
                })
                .collect::<Result<Vec<_>, LiftError>>()
        };
        let views = Views::<N, TABLES, COUNT, SQUARED> {
            x: convert(&self.x)?
                .try_into()
                .map_err(|_| LiftError::Layout)?,
            squared: convert(&self.squared)?
                .try_into()
                .map_err(|_| LiftError::Layout)?,
        };
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
                        let values = Self::values::<K, AT_ONE, N, TABLES, COUNT, SQUARED>(
                            &views,
                            source.lanes(cycle),
                        );
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

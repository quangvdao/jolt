use super::geometry::WindowGeometry;
use super::{merge, Sums, ONE, ZERO};
use crate::packed::lift::{compact_table, LiftError};
use crate::par::CycleChunks;
use crate::round::eq::eq_table;
use crate::source::LaneSource;
use jolt_field::{Accumulator, F128Accumulator, F128};
use rayon::prelude::*;

pub(super) struct Window<
    const K: usize,
    const N: usize,
    const A: usize,
    const C: usize,
    const UNITS: usize,
> {
    a: Vec<[[F128; N]; A]>,
    c: Vec<[[F128; N]; C]>,
    b: Vec<[F128; N]>,
    omega: [F128; 2],
    folded: bool,
}

impl<const K: usize, const N: usize, const A: usize, const C: usize, const UNITS: usize>
    Window<K, N, A, C, UNITS>
{
    pub(super) fn new(
        point: &[F128],
        rho: &[F128],
        omega: &[F128],
        folded: bool,
    ) -> Result<Self, LiftError> {
        const {
            let geometry = WindowGeometry::new(K);
            assert!(N == geometry.n);
            assert!(A == geometry.a);
            assert!(C == geometry.c);
            assert!(UNITS == geometry.units);
        };
        if point.len() != K || omega.len() < 2 {
            return Err(LiftError::Layout);
        }
        let geometry = WindowGeometry::new(point.len());
        let bits = eq_table(point, None);
        let width = geometry.width;
        let bytes = bits.len() / width;
        if bytes != UNITS || rho.len() != C / UNITS {
            return Err(LiftError::Layout);
        }
        let b = bits
            .chunks_exact(width)
            .map(compact_table)
            .collect::<Result<Vec<_>, _>>()?;
        let mut a = Vec::with_capacity(if folded { 2 } else { 1 });
        let mut c = Vec::with_capacity(a.capacity());
        for &scale in if folded { &omega[..2] } else { &[ONE] } {
            let mut av = [[ZERO; N]; A];
            let mut cv = [[ZERO; N]; C];
            for (byte, table) in av.iter_mut().enumerate() {
                let weight = rho[byte / (2 * bytes)] * scale;
                *table = compact_table(
                    &std::array::from_fn::<_, 8, _>(|bit| {
                        bits[(byte % bytes) * width + bit % width] * weight
                    })[..width],
                )?;
            }
            for (byte, table) in cv.iter_mut().enumerate() {
                let weight = rho[byte / bytes] * scale;
                *table = compact_table(
                    &std::array::from_fn::<_, 8, _>(|bit| {
                        bits[(byte % bytes) * width + bit % width] * weight
                    })[..width],
                )?;
            }
            a.push(av);
            c.push(cv);
        }
        Ok(Self {
            a,
            b,
            c,
            omega: [omega[0], omega[1]],
            folded,
        })
    }

    #[inline]
    fn add<const AT_ONE: bool>(
        &self,
        group: usize,
        lanes: [u64; 3],
        sums: &mut [F128Accumulator; 2],
    ) {
        let [a, b, c] = lanes;
        let width = WindowGeometry::new(K).width;
        let index = |word: u64, unit: usize| ((word >> (width * unit)) & (N as u64 - 1)) as usize;
        let table_group = if self.folded { group } else { 0 };
        for pair in 0..C / UNITS {
            let mut av = [ZERO; 2];
            let mut bv = [ZERO; 2];
            let mut cv = ZERO;
            for byte in 0..UNITS {
                let even = pair * 2 * UNITS + byte;
                let odd = even + UNITS;
                av[0] += self.a[table_group][even][index(a, even)];
                av[1] += self.a[table_group][odd][index(a, odd)];
                bv[0] += self.b[byte][index(b, even)];
                bv[1] += self.b[byte][index(b, odd)];
                cv += self.c[table_group][pair * UNITS + byte]
                    [index(c, if AT_ONE { odd } else { even })];
            }
            let endpoint = usize::from(AT_ONE);
            sums[0].fmadd(av[endpoint], bv[endpoint]);
            sums[0].add(cv);
            sums[1].fmadd(av[0] + av[1], bv[0] + bv[1]);
        }
    }

    pub(super) fn pass<const AT_ONE: bool, S: LaneSource>(
        &self,
        source: &S,
        chunks: CycleChunks,
        lo: &[F128],
        hi: &[F128],
    ) -> Sums {
        let sums = (0..chunks.len() / chunks.chunk_len())
            .into_par_iter()
            .map(|chunk| {
                let start = chunk * chunks.chunk_len();
                let first_block = start / chunks.block_len();
                let blocks = chunks.chunk_len() / chunks.block_len();
                let mut total = [F128Accumulator::default(); 4];
                for (block, &high) in hi[first_block..first_block + blocks].iter().enumerate() {
                    let mut inner = [F128Accumulator::default(); 4];
                    for (offset, &low) in lo.iter().enumerate() {
                        let lanes = source.lanes(start + block * chunks.block_len() + offset);
                        if self.folded {
                            let mut cycle = [F128Accumulator::default(); 2];
                            self.add::<AT_ONE>(0, lanes[0], &mut cycle);
                            self.add::<AT_ONE>(1, lanes[1], &mut cycle);
                            for (sum, value) in inner[..2].iter_mut().zip(cycle) {
                                sum.fmadd(low, value.reduce());
                            }
                        } else {
                            for (group, lane) in lanes.into_iter().enumerate() {
                                let mut cycle = [F128Accumulator::default(); 2];
                                self.add::<AT_ONE>(group, lane, &mut cycle);
                                for (sum, value) in
                                    inner[group * 2..group * 2 + 2].iter_mut().zip(cycle)
                                {
                                    sum.fmadd(low, value.reduce());
                                }
                            }
                        }
                    }
                    for (sum, value) in total.iter_mut().zip(inner) {
                        sum.fmadd(high, value.reduce());
                    }
                }
                total
            })
            .reduce(|| [F128Accumulator::default(); 4], merge)
            .map(Accumulator::reduce);
        if self.folded {
            [sums[0], sums[1]]
        } else {
            [
                sums[0] * self.omega[0] + sums[2] * self.omega[1],
                sums[1] * self.omega[0] + sums[3] * self.omega[1],
            ]
        }
    }
}

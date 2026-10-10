use super::{merge, Sums, ZERO};
use crate::par::CycleChunks;
use crate::round::eq::eq_table;
use crate::source::LaneSource;
use jolt_field::{Accumulator, F128Accumulator, F128};
use rayon::prelude::*;

type ByteTable = [F128; 256];

pub(super) fn byte_table(weights: &[F128]) -> ByteTable {
    let mut table = [ZERO; 256];
    for (bit, &weight) in weights.iter().enumerate() {
        let width = 1 << bit;
        let (lo, hi) = table[..2 * width].split_at_mut(width);
        for (out, &value) in hi.iter_mut().zip(lo.iter()) {
            *out = value + weight;
        }
    }
    table
}

pub(super) struct Window {
    a: Vec<[ByteTable; 8]>,
    c: Vec<[ByteTable; 4]>,
    b: Vec<ByteTable>,
    omega: [F128; 2],
    folded: bool,
}

impl Window {
    pub(super) fn new(point: &[F128], rho: &[F128], omega: &[F128], folded: bool) -> Self {
        let bits = eq_table(point, None);
        let bytes = bits.len() / 8;
        let b: Vec<_> = bits.chunks_exact(8).map(byte_table).collect();
        let mut a = Vec::with_capacity(if folded { 2 } else { 1 });
        let mut c = Vec::with_capacity(a.capacity());
        for &scale in if folded { &omega[..2] } else { &[super::ONE] } {
            a.push(std::array::from_fn(|byte| {
                let weight = rho[byte / (2 * bytes)] * scale;
                byte_table(&std::array::from_fn::<_, 8, _>(|bit| {
                    bits[(byte % bytes) * 8 + bit] * weight
                }))
            }));
            c.push(std::array::from_fn(|byte| {
                let weight = rho[byte / bytes] * scale;
                byte_table(&std::array::from_fn::<_, 8, _>(|bit| {
                    bits[(byte % bytes) * 8 + bit] * weight
                }))
            }));
        }
        Self {
            a,
            b,
            c,
            omega: [omega[0], omega[1]],
            folded,
        }
    }

    #[inline]
    fn add<const BYTES: usize, const AT_ONE: bool>(
        &self,
        group: usize,
        lanes: [u64; 3],
        sums: &mut [F128Accumulator; 2],
    ) {
        let [a, b, c] = lanes.map(u64::to_le_bytes);
        let table_group = if self.folded { group } else { 0 };
        for pair in 0..4 / BYTES {
            let mut av = [ZERO; 2];
            let mut bv = [ZERO; 2];
            let mut cv = ZERO;
            for byte in 0..BYTES {
                let even = pair * 2 * BYTES + byte;
                let odd = even + BYTES;
                av[0] += self.a[table_group][even][usize::from(a[even])];
                av[1] += self.a[table_group][odd][usize::from(a[odd])];
                bv[0] += self.b[byte][usize::from(b[even])];
                bv[1] += self.b[byte][usize::from(b[odd])];
                cv += self.c[table_group][pair * BYTES + byte]
                    [usize::from(c[if AT_ONE { odd } else { even }])];
            }
            let endpoint = usize::from(AT_ONE);
            sums[0].fmadd(av[endpoint], bv[endpoint]);
            sums[0].add(cv);
            sums[1].fmadd(av[0] + av[1], bv[0] + bv[1]);
        }
    }

    pub(super) fn pass<const BYTES: usize, const AT_ONE: bool, S: LaneSource>(
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
                            self.add::<BYTES, AT_ONE>(0, lanes[0], &mut cycle);
                            self.add::<BYTES, AT_ONE>(1, lanes[1], &mut cycle);
                            for (sum, value) in inner[..2].iter_mut().zip(cycle) {
                                sum.fmadd(low, value.reduce());
                            }
                        } else {
                            for (group, lane) in lanes.into_iter().enumerate() {
                                let mut cycle = [F128Accumulator::default(); 2];
                                self.add::<BYTES, AT_ONE>(group, lane, &mut cycle);
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

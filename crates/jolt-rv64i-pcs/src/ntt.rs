// CREDIT: https://github.com/succinctlabs/flock (flock-core), MIT OR Apache-2.0.
// Copyright (c) 2026 Bain Capital Crypto, LP and Ron Rothblum
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Modified from leanVM crates/pcs/src/ntt/additive_ntt_f64.rs,
// crates/pcs/src/whir_ntt_ext.rs and crates/pcs/src/whir_induce.rs at
// revision 48a904208d682848dac0e18ef8b01ebfc40df9ad.
// Notices: crates/jolt-rv64i-pcs/THIRD_PARTY_NOTICES.md.

//! Position-major additive encoding and its transpose, as specified in
//! `specs/rv64i-binary-commitment.md`, sections 2 and 9. Lanes are low variables.

use jolt_field::{ExtField, Ring, Zero};
use jolt_field::{F192, F64};
use jolt_rv64i_verifier::whir::code::DomainTable;
use jolt_rv64i_verifier::whir::error::{checked_product, try_vec, WhirError, WhirPart};
use rayon::prelude::*;

trait CodeSymbol: Ring + Copy + Send + Sync {
    const FUSE_LAYERS: bool;

    fn scale(self, twiddle: F64) -> Self;
}
impl CodeSymbol for F64 {
    const FUSE_LAYERS: bool = true;

    #[inline]
    fn scale(self, twiddle: F64) -> Self {
        self * twiddle
    }
}
impl CodeSymbol for F192 {
    const FUSE_LAYERS: bool = false;

    #[inline]
    fn scale(self, twiddle: F64) -> Self {
        self.mul_base(twiddle)
    }
}

/// A checked encoding shape borrowing the call's domain constants. Construct
/// the table once per commit/open call; subsequent levels borrow that same table.
/// Input and output indices are `position * lanes + lane`.
#[derive(Debug)]
pub struct Encoder<'a> {
    table: &'a DomainTable,
    c: usize,
    lanes: usize,
    message_positions: usize,
    code_positions: usize,
    message_len: usize,
    code_len: usize,
}
impl<'a> Encoder<'a> {
    /// Accepts power-of-two lane counts and a subdomain covered by `table`.
    pub fn new(
        table: &'a DomainTable,
        c: usize,
        d: usize,
        lanes: usize,
    ) -> Result<Self, WhirError> {
        table.validate_dimensions(c, d)?;
        if !lanes.is_power_of_two() {
            return Err(WhirError::Shape {
                part: WhirPart::Leaves,
                expected: lanes.checked_next_power_of_two().unwrap_or(usize::MAX),
                actual: lanes,
            });
        }
        let power = |exponent: usize| {
            1usize
                .checked_shl(exponent as u32)
                .ok_or(WhirError::LengthOverflow {
                    part: WhirPart::Leaves,
                })
        };
        let message_positions = power(c)?;
        let code_positions = power(d)?;
        Ok(Self {
            table,
            c,
            lanes,
            message_positions,
            code_positions,
            message_len: checked_product(WhirPart::FinalValues, &[message_positions, lanes])?,
            code_len: checked_product(WhirPart::Leaves, &[code_positions, lanes])?,
        })
    }

    /// Encodes level 0 coefficientwise in K, directly from the shared row
    /// buffer. Each V lane occupies two consecutive F64 output entries; no
    /// packed-message allocation, byte reinterpretation or transposition occurs.
    pub fn encode_rows(&self, rows: &[[u64; 4]]) -> Result<Vec<F64>, WhirError> {
        let words = checked_product(WhirPart::Rows, &[self.message_len, 2])?;
        if words % 4 != 0 || rows.len() != words / 4 {
            return Err(WhirError::Shape {
                part: WhirPart::Rows,
                expected: words / 4,
                actual: rows.len(),
            });
        }
        let word_lanes = checked_product(WhirPart::Leaves, &[self.lanes, 2])?;
        let len = checked_product(WhirPart::Leaves, &[self.code_len, 2])?;
        let mut out = try_vec(WhirPart::Leaves, len)?;
        out.resize(len, F64::zero());
        self.initialize(&mut out, word_lanes, |index| {
            F64::from_raw(rows[index / 4][index % 4])
        });
        Ok(out)
    }

    /// Encodes later E-valued levels using K twiddles and ascending lanes.
    pub fn encode_extension(&self, message: &[F192]) -> Result<Vec<F192>, WhirError> {
        self.encode(message)
    }

    fn encode<F: CodeSymbol>(&self, message: &[F]) -> Result<Vec<F>, WhirError> {
        if message.len() != self.message_len {
            return Err(WhirError::Shape {
                part: WhirPart::FinalValues,
                expected: self.message_len,
                actual: message.len(),
            });
        }
        let mut out = try_vec(WhirPart::Leaves, self.code_len)?;
        out.resize(self.code_len, F::zero());
        out.par_chunks_mut(self.message_len)
            .for_each(|block| block.copy_from_slice(message));
        self.forward(&mut out, self.lanes, self.c);
        Ok(out)
    }

    fn initialize<F: CodeSymbol>(
        &self,
        data: &mut [F],
        lanes: usize,
        read: impl Fn(usize) -> F + Sync,
    ) {
        let block_len = self.message_positions * lanes;
        data.par_chunks_mut(block_len)
            .enumerate()
            .for_each(|(replica, block)| {
                let base = replica * self.message_positions;
                if self.c < 3 {
                    for (index, value) in block.iter_mut().enumerate() {
                        *value = read(index);
                    }
                    self.local_layers(block, lanes, base, self.c);
                } else {
                    let twiddles = self.radix8_twiddles(self.c - 1, base);
                    let stride = block_len / 8;
                    Self::radix8_tiles(block, true, |offset, mut rows| {
                        for (slab, row) in rows.iter_mut().enumerate() {
                            for (index, value) in row.iter_mut().enumerate() {
                                *value = read(slab * stride + offset + index);
                            }
                        }
                        Self::radix8(&mut rows, &twiddles);
                    });
                }
            });
        self.forward(data, lanes, self.c.saturating_sub(3));
    }

    fn forward<F: CodeSymbol>(&self, data: &mut [F], lanes: usize, remaining: usize) {
        if remaining == 0 {
            return;
        }
        let bytes_per_position = lanes * std::mem::size_of::<F>();
        let cache_positions = ((1 << 20) / bytes_per_position).max(1);
        let cache_log = cache_positions.ilog2() as usize;
        let parallel_log = (self.code_positions / rayon::current_num_threads().next_power_of_two())
            .max(1)
            .ilog2() as usize;
        let local_layers = remaining.min(cache_log).min(parallel_log);
        let mut top_layers = remaining;
        while F::FUSE_LAYERS && top_layers >= local_layers + 3 {
            let run_positions = 1usize << top_layers;
            data.par_chunks_mut(run_positions * lanes)
                .enumerate()
                .for_each(|(run, block)| {
                    let twiddles = self.radix8_twiddles(top_layers - 1, run * run_positions);
                    Self::radix8_tiles(block, true, |_, mut rows| {
                        Self::radix8(&mut rows, &twiddles);
                    });
                });
            top_layers -= 3;
        }
        for l in (local_layers..top_layers).rev() {
            let run_positions = 1usize << (l + 1);
            data.par_chunks_mut(run_positions * lanes)
                .enumerate()
                .for_each(|(run, block)| {
                    let twiddle = self.table.w_hat(l, (run * run_positions) as u32);
                    let (top, bot) = block.split_at_mut(block.len() / 2);
                    top.par_chunks_mut(4096)
                        .zip(bot.par_chunks_mut(4096))
                        .for_each(|(top, bot)| Self::butterfly(top, bot, twiddle));
                });
        }
        let window_positions = 1usize << local_layers;
        data.par_chunks_mut(window_positions * lanes)
            .enumerate()
            .for_each(|(window, data)| {
                let base = window * window_positions;
                self.local_layers(data, lanes, base, local_layers);
            });
    }

    fn local_layers<F: CodeSymbol>(
        &self,
        data: &mut [F],
        lanes: usize,
        base: usize,
        remaining: usize,
    ) {
        if F::FUSE_LAYERS && remaining >= 3 {
            let twiddles = self.radix8_twiddles(remaining - 1, base);
            Self::radix8_tiles(data, false, |_, mut rows| {
                Self::radix8(&mut rows, &twiddles);
            });
            let child_positions = 1usize << (remaining - 3);
            for (child, block) in data.chunks_mut(child_positions * lanes).enumerate() {
                self.local_layers(block, lanes, base + child * child_positions, remaining - 3);
            }
            return;
        }
        for l in (0..remaining).rev() {
            let run_positions = 1usize << (l + 1);
            for (run, block) in data.chunks_mut(run_positions * lanes).enumerate() {
                let twiddle = self.table.w_hat(l, (base + run * run_positions) as u32);
                let (top, bot) = block.split_at_mut(block.len() / 2);
                Self::butterfly(top, bot, twiddle);
            }
        }
    }

    fn radix8_twiddles(&self, l: usize, base: usize) -> [F64; 7] {
        let half = 1usize << l;
        let quarter = half / 2;
        [
            self.table.w_hat(l, base as u32),
            self.table.w_hat(l - 1, base as u32),
            self.table.w_hat(l - 1, (base + half) as u32),
            self.table.w_hat(l - 2, base as u32),
            self.table.w_hat(l - 2, (base + quarter) as u32),
            self.table.w_hat(l - 2, (base + half) as u32),
            self.table.w_hat(l - 2, (base + half + quarter) as u32),
        ]
    }

    fn radix8_tiles<F: CodeSymbol>(
        block: &mut [F],
        parallel: bool,
        visit: impl Fn(usize, [&mut [F]; 8]) + Sync,
    ) {
        let stride = block.len() / 8;
        let (r0, rest) = block.split_at_mut(stride);
        let (r1, rest) = rest.split_at_mut(stride);
        let (r2, rest) = rest.split_at_mut(stride);
        let (r3, rest) = rest.split_at_mut(stride);
        let (r4, rest) = rest.split_at_mut(stride);
        let (r5, rest) = rest.split_at_mut(stride);
        let (r6, r7) = rest.split_at_mut(stride);
        let tile = ((32 << 10) / (8 * std::mem::size_of::<F>())).max(1);
        if parallel {
            r0.par_chunks_mut(tile)
                .zip(r1.par_chunks_mut(tile))
                .zip(r2.par_chunks_mut(tile))
                .zip(r3.par_chunks_mut(tile))
                .zip(r4.par_chunks_mut(tile))
                .zip(r5.par_chunks_mut(tile))
                .zip(r6.par_chunks_mut(tile))
                .zip(r7.par_chunks_mut(tile))
                .enumerate()
                .for_each(|(i, (((((((r0, r1), r2), r3), r4), r5), r6), r7))| {
                    visit(i * tile, [r0, r1, r2, r3, r4, r5, r6, r7]);
                });
        } else {
            for (i, (((((((r0, r1), r2), r3), r4), r5), r6), r7)) in r0
                .chunks_mut(tile)
                .zip(r1.chunks_mut(tile))
                .zip(r2.chunks_mut(tile))
                .zip(r3.chunks_mut(tile))
                .zip(r4.chunks_mut(tile))
                .zip(r5.chunks_mut(tile))
                .zip(r6.chunks_mut(tile))
                .zip(r7.chunks_mut(tile))
                .enumerate()
            {
                visit(i * tile, [r0, r1, r2, r3, r4, r5, r6, r7]);
            }
        }
    }

    #[inline]
    fn radix8<F: CodeSymbol>(rows: &mut [&mut [F]; 8], t: &[F64; 7]) {
        let [r0, r1, r2, r3, r4, r5, r6, r7] = rows;
        Self::butterfly(r0, r4, t[0]);
        Self::butterfly(r1, r5, t[0]);
        Self::butterfly(r2, r6, t[0]);
        Self::butterfly(r3, r7, t[0]);
        Self::butterfly(r0, r2, t[1]);
        Self::butterfly(r1, r3, t[1]);
        Self::butterfly(r4, r6, t[2]);
        Self::butterfly(r5, r7, t[2]);
        Self::butterfly(r0, r1, t[3]);
        Self::butterfly(r2, r3, t[4]);
        Self::butterfly(r4, r5, t[5]);
        Self::butterfly(r6, r7, t[6]);
    }

    #[inline]
    fn butterfly<F: CodeSymbol>(top: &mut [F], bot: &mut [F], twiddle: F64) {
        if F::FUSE_LAYERS && twiddle.is_zero() {
            for (top, bot) in top.iter().zip(bot) {
                *bot += *top;
            }
            return;
        }
        if !F::FUSE_LAYERS {
            for (top, bot) in top.iter_mut().zip(bot) {
                let new_top = *top + bot.scale(twiddle);
                *bot += new_top;
                *top = new_top;
            }
            return;
        }
        let (top_chunks, top_tail) = top.as_chunks_mut::<4>();
        let (bot_chunks, bot_tail) = bot.as_chunks_mut::<4>();
        for ([t0, t1, t2, t3], [b0, b1, b2, b3]) in top_chunks.iter_mut().zip(bot_chunks) {
            let p0 = b0.scale(twiddle);
            let p1 = b1.scale(twiddle);
            let p2 = b2.scale(twiddle);
            let p3 = b3.scale(twiddle);
            let n0 = *t0 + p0;
            let n1 = *t1 + p1;
            let n2 = *t2 + p2;
            let n3 = *t3 + p3;
            *b0 += n0;
            *b1 += n1;
            *b2 += n2;
            *b3 += n3;
            *t0 = n0;
            *t1 = n1;
            *t2 = n2;
            *t3 = n3;
        }
        for (top, bot) in top_tail.iter_mut().zip(bot_tail) {
            let new_top = *top + bot.scale(twiddle);
            *bot += new_top;
            *top = new_top;
        }
    }

    /// Applies Enc^T to one domain-sized E buffer in place, then allocates
    /// only the coefficient output and sums the coset blocks. Zero runs are
    /// skipped. The caller scatters query weights into this single buffer.
    pub fn transpose(&self, data: &mut [F192]) -> Result<Vec<F192>, WhirError> {
        if self.lanes != 1 {
            return Err(WhirError::Shape {
                part: WhirPart::Leaves,
                expected: 1,
                actual: self.lanes,
            });
        }
        if data.len() != self.code_positions {
            return Err(WhirError::Shape {
                part: WhirPart::Leaves,
                expected: self.code_positions,
                actual: data.len(),
            });
        }
        let local_layers = self.c.min(12);
        let window_positions = 1usize << local_layers;
        data.par_chunks_mut(window_positions)
            .enumerate()
            .for_each(|(window, data)| {
                for l in 0..local_layers {
                    let run_positions = 1usize << (l + 1);
                    for (run, block) in data.chunks_mut(run_positions).enumerate() {
                        self.transpose_run(
                            l,
                            window * window_positions + run * run_positions,
                            block,
                        );
                    }
                }
            });
        for l in local_layers..self.c {
            let run_positions = 1usize << (l + 1);
            data.par_chunks_mut(run_positions)
                .enumerate()
                .for_each(|(run, block)| self.transpose_run(l, run * run_positions, block));
        }
        let mut out = try_vec(WhirPart::FinalValues, self.message_positions)?;
        out.resize(self.message_positions, F192::zero());
        out.par_iter_mut().enumerate().for_each(|(w, value)| {
            for block in data.chunks_exact(self.message_positions) {
                *value += block[w];
            }
        });
        Ok(out)
    }

    fn transpose_run(&self, l: usize, base: usize, block: &mut [F192]) {
        if block.iter().all(Zero::is_zero) {
            return;
        }
        let twiddle = self.table.w_hat(l, base as u32);
        let (top, bot) = block.split_at_mut(block.len() / 2);
        for (top, bot) in top.iter_mut().zip(bot) {
            let sum = *top + *bot;
            *bot += sum.mul_base(twiddle);
            *top = sum;
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests assert valid shapes")]
mod tests {
    use super::Encoder;
    use jolt_field::{ExtField, One, Zero};
    use jolt_field::{F192, F64};
    use jolt_rv64i_verifier::whir::code::{encode_by_definition, DomainTable};
    use jolt_rv64i_verifier::whir::error::{WhirError, WhirPart};
    use rayon::ThreadPoolBuilder;

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

    fn element(rng: &mut Words) -> F192 {
        F192::from_base_fn(|_| F64::from_raw(rng.gen()))
    }

    #[test]
    fn encodings_match_direct_polynomial_at_every_position() {
        let mut rng = Words::seed_from_u64(42);
        let table = DomainTable::new(7, 8).unwrap();
        for d in 2..=8 {
            for c in 1..d {
                for lanes in [1, 2, 8, 16, 64] {
                    let encoder = Encoder::new(&table, c, d, lanes).unwrap();
                    let message: Vec<_> =
                        (0..(1 << c) * lanes).map(|_| element(&mut rng)).collect();
                    assert_eq!(
                        encoder.encode_extension(&message).unwrap(),
                        encode_by_definition(&message, c, d, lanes).unwrap(),
                        "E c={c} d={d} lanes={lanes}"
                    );
                    let base: Vec<_> = (0..message.len())
                        .map(|_| F64::from_raw(rng.gen()))
                        .collect();
                    let lifted: Vec<_> = base.iter().copied().map(F192::lift_base).collect();
                    let want = encode_by_definition(&lifted, c, d, lanes).unwrap();
                    let got = encoder.encode(&base).unwrap();
                    assert!(
                        got.iter().zip(want).all(|(&a, b)| F192::lift_base(a) == b),
                        "K c={c} d={d} lanes={lanes}"
                    );
                }
            }
        }
    }

    #[test]
    fn row_encoding_preserves_v_and_word_lane_order() {
        let mut rng = Words::seed_from_u64(81);
        let table = DomainTable::new(7, 8).unwrap();
        for d in 2..=8 {
            for c in 1..d {
                for words_per_position in [2, 32, 64] {
                    let lanes = words_per_position / 2;
                    let rows: Vec<[u64; 4]> = (0..(1 << c) * words_per_position / 4)
                        .map(|_| std::array::from_fn(|_| rng.gen()))
                        .collect();
                    let message: Vec<_> = rows
                        .iter()
                        .flat_map(|row| row.chunks_exact(2))
                        .map(|pair| {
                            F192::from_base_fn(|i| {
                                if i < 2 {
                                    F64::from_raw(pair[i])
                                } else {
                                    F64::zero()
                                }
                            })
                        })
                        .collect();
                    let expected = encode_by_definition(&message, c, d, lanes).unwrap();
                    let output = Encoder::new(&table, c, d, lanes)
                        .unwrap()
                        .encode_rows(&rows)
                        .unwrap();
                    for (pair, expected) in output.chunks_exact(2).zip(expected) {
                        assert_eq!(expected.base_coefficient(0), pair[0]);
                        assert_eq!(expected.base_coefficient(1), pair[1]);
                        assert!(expected.base_coefficient(2).is_zero());
                    }
                }
            }
        }
    }

    #[test]
    fn encoding_linearity_and_lane_fold_commutation() {
        let mut rng = Words::seed_from_u64(190);
        let table = DomainTable::new(5, 7).unwrap();
        let a: Vec<_> = (0..3).map(|_| element(&mut rng)).collect();
        let eq: Vec<_> = (0..8)
            .map(|u| {
                a.iter().enumerate().fold(F192::one(), |p, (l, &a)| {
                    p * if u & (1 << l) == 0 {
                        F192::one() + a
                    } else {
                        a
                    }
                })
            })
            .collect();
        let message: Vec<_> = (0..32 * 8).map(|_| element(&mut rng)).collect();
        let other: Vec<_> = (0..message.len()).map(|_| element(&mut rng)).collect();
        let scale = element(&mut rng);
        let combined: Vec<_> = message
            .iter()
            .zip(&other)
            .map(|(&a, &b)| a + scale * b)
            .collect();
        let encoder = Encoder::new(&table, 5, 7, 8).unwrap();
        let code = encoder.encode_extension(&message).unwrap();
        let code_other = encoder.encode_extension(&other).unwrap();
        let code_combined = encoder.encode_extension(&combined).unwrap();
        assert!(code
            .iter()
            .zip(code_other)
            .zip(code_combined)
            .all(|((&a, b), c)| a + scale * b == c));
        let fold = |lanes: &[F192]| {
            lanes
                .iter()
                .zip(&eq)
                .fold(F192::zero(), |s, (&v, &w)| s + v * w)
        };
        let folded_message: Vec<_> = message.chunks_exact(8).map(fold).collect();
        let folded_code: Vec<_> = code.chunks_exact(8).map(fold).collect();
        let want = Encoder::new(&table, 5, 7, 1)
            .unwrap()
            .encode_extension(&folded_message)
            .unwrap();
        assert_eq!(folded_code, want);
        // The first fold leaves V: its E challenges still commute with the K code.
        let rows: Vec<[u64; 4]> = (0..128)
            .map(|_| std::array::from_fn(|_| rng.gen()))
            .collect();
        let base_code = encoder.encode_rows(&rows).unwrap();
        let pack = |pair: &[F64]| F192::from_base_fn(|i| if i < 2 { pair[i] } else { F64::zero() });
        let v_message: Vec<_> = rows
            .iter()
            .flat_map(|r| r.chunks_exact(2))
            .map(|p| pack(&[F64::from_raw(p[0]), F64::from_raw(p[1])]))
            .collect();
        let folded_message: Vec<_> = v_message.chunks_exact(8).map(fold).collect();
        let folded_code: Vec<_> = base_code
            .chunks_exact(16)
            .map(|row| {
                row.chunks_exact(2)
                    .zip(&eq)
                    .fold(F192::zero(), |s, (p, &w)| s + pack(p) * w)
            })
            .collect();
        assert_eq!(
            folded_code,
            Encoder::new(&table, 5, 7, 1)
                .unwrap()
                .encode_extension(&folded_message)
                .unwrap()
        );
    }

    #[test]
    fn transpose_inner_product_identity_dense_and_sparse() {
        let mut rng = Words::seed_from_u64(509);
        let table = DomainTable::new(7, 8).unwrap();
        for d in 2..=8 {
            for c in 1..d {
                let encoder = Encoder::new(&table, c, d, 1).unwrap();
                let f: Vec<_> = (0..1 << c).map(|_| element(&mut rng)).collect();
                let code = encode_by_definition(&f, c, d, 1).unwrap();
                for sparse in [false, true] {
                    let mut g: Vec<_> = (0..1 << d)
                        .map(|i| {
                            if sparse && i % 7 != 0 {
                                F192::zero()
                            } else {
                                element(&mut rng)
                            }
                        })
                        .collect();
                    let lhs = code
                        .iter()
                        .zip(&g)
                        .fold(F192::zero(), |s, (&f, &g)| s + f * g);
                    let gt = encoder.transpose(&mut g).unwrap();
                    let rhs = f.iter().zip(gt).fold(F192::zero(), |s, (&f, g)| s + f * g);
                    assert_eq!(lhs, rhs, "c={c},d={d},sparse={sparse}");
                }
            }
        }
    }

    #[test]
    fn transpose_across_cache_windows_matches_direct_sparse_weights() {
        let mut rng = Words::seed_from_u64(910);
        let table = DomainTable::new(13, 14).unwrap();
        let positions = [0usize, 73, 8191, 8192, 16383];
        let values: Vec<_> = positions.iter().map(|_| element(&mut rng)).collect();
        // Independently evaluate the subspace polynomials from their roots at
        // the five queried points, then multiply the selected basis factors.
        use jolt_field::Field;
        let factors: Vec<Vec<F64>> = positions
            .iter()
            .map(|&x| {
                (0..13)
                    .map(|l| {
                        let subspace = |x: u64| {
                            (0..1u64 << l).fold(F64::one(), |p, a| p * F64::from_raw(x ^ a))
                        };
                        subspace(x as u64) * subspace(1 << l).inverse().unwrap()
                    })
                    .collect()
            })
            .collect();
        let expected: Vec<_> = (0..1 << 13)
            .map(|w| {
                values
                    .iter()
                    .zip(&factors)
                    .fold(F192::zero(), |sum, (&v, factors)| {
                        let basis = factors
                            .iter()
                            .enumerate()
                            .filter(|(l, _)| w & (1 << l) != 0)
                            .fold(F64::one(), |p, (_, &s)| p * s);
                        sum + v.mul_base(basis)
                    })
            })
            .collect();
        for threads in [1, 12] {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut g = vec![F192::zero(); 1 << 14];
                for (&x, &v) in positions.iter().zip(&values) {
                    g[x] = v;
                }
                assert_eq!(
                    Encoder::new(&table, 13, 14, 1)
                        .unwrap()
                        .transpose(&mut g)
                        .unwrap(),
                    expected
                );
            });
        }
    }

    #[test]
    fn literal_smallest_transform_and_constant_message() {
        let table = DomainTable::new(2, 3).unwrap();
        let base = [F64::from_raw(3), F64::from_raw(5)];
        let out = Encoder::new(&table, 1, 2, 1)
            .unwrap()
            .encode(&base)
            .unwrap();
        assert_eq!(
            out.iter().map(|v| v.to_raw()).collect::<Vec<_>>(),
            [3, 6, 9, 12]
        );
        let message = [F192::lift_base(F64::from_raw(17))];
        let encoder = Encoder::new(&table, 0, 3, 1).unwrap();
        assert_eq!(encoder.encode_extension(&message).unwrap(), [message[0]; 8]);
        let mut g = [message[0]; 8];
        assert_eq!(encoder.transpose(&mut g).unwrap(), [F192::zero()]);
    }

    #[test]
    fn bad_transform_dimensions_are_typed_errors() {
        let table = DomainTable::new(2, 3).unwrap();
        assert!(matches!(
            Encoder::new(&table, 4, 3, 1),
            Err(WhirError::Shape {
                part: WhirPart::Levels,
                ..
            })
        ));
        assert!(matches!(
            Encoder::new(&table, 2, 3, 0),
            Err(WhirError::Shape {
                part: WhirPart::Leaves,
                ..
            })
        ));
        assert!(matches!(
            Encoder::new(&table, 2, 3, 3),
            Err(WhirError::Shape {
                part: WhirPart::Leaves,
                ..
            })
        ));
        assert!(matches!(
            Encoder::new(&table, 2, 3, 1usize << (usize::BITS - 1)),
            Err(WhirError::LengthOverflow { .. })
        ));
        let encoder = Encoder::new(&table, 2, 3, 1).unwrap();
        assert_eq!(
            encoder.encode_extension(&[]),
            Err(WhirError::Shape {
                part: WhirPart::FinalValues,
                expected: 4,
                actual: 0
            })
        );
        assert_eq!(
            encoder.encode_rows(&[]),
            Err(WhirError::Shape {
                part: WhirPart::Rows,
                expected: 2,
                actual: 0
            })
        );
        assert_eq!(
            encoder.transpose(&mut []),
            Err(WhirError::Shape {
                part: WhirPart::Leaves,
                expected: 8,
                actual: 0
            })
        );
        assert_eq!(
            Encoder::new(&table, 2, 3, 2).unwrap().transpose(&mut []),
            Err(WhirError::Shape {
                part: WhirPart::Leaves,
                expected: 1,
                actual: 2
            })
        );
    }

    #[test]
    fn parallel_cache_windows_obey_definition() {
        let mut rng = Words::seed_from_u64(47);
        let table = DomainTable::new(13, 14).unwrap();
        // Unit messages evaluate to their literal basis polynomial; the message
        // size crosses the cache boundary without a quadratic reference encoder.
        for threads in [1, 12] {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let lanes = 64;
                let encoder = Encoder::new(&table, 13, 14, lanes).unwrap();
                let mut f = vec![F192::zero(); (1 << 13) * lanes];
                let coefficient = element(&mut rng);
                let w = 0b1010_1010_1010;
                f[w * lanes + 19] = coefficient;
                let code = encoder.encode_extension(&f).unwrap();
                for (x, row) in code.chunks_exact(lanes).enumerate() {
                    let basis = (0..13)
                        .filter(|l| w & (1 << l) != 0)
                        .fold(F64::one(), |p, l| p * table.w_hat(l, x as u32));
                    assert_eq!(row[19], coefficient.mul_base(basis));
                    assert!(row.iter().enumerate().all(|(u, v)| u == 19 || v.is_zero()));
                }
            });
        }
    }

    #[test]
    fn fused_global_passes_match_sampled_polynomial_values() {
        use jolt_field::Field;

        let c = 17;
        let d = 18;
        let w = (1usize << 13) | (1 << 12) | (1 << 11) | (1 << 3) | 1;
        let table = DomainTable::new(c, d).unwrap();
        let positions = [
            0usize, 1, 9, 127, 128, 1023, 1024, 2047, 2048, 16383, 16384, 65535, 65536, 131_072,
            262_143,
        ];
        let subspace = |l: usize, x: usize| {
            (0..1usize << l).fold(F64::one(), |p, root| p * F64::from_raw((x ^ root) as u64))
        };
        let selected_bits = [0, 3, 11, 12, 13];
        let inverses = selected_bits.map(|l| subspace(l, 1 << l).inverse().unwrap());
        let basis = positions.map(|x| {
            selected_bits
                .iter()
                .zip(&inverses)
                .fold(F64::one(), |p, (&l, &inverse)| p * subspace(l, x) * inverse)
        });
        let coefficient = F192::from_base_fn(|i| F64::from_raw([3, 5, 9][i]));

        for threads in [1, 12] {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let lanes = 16;
                let mut message = vec![F192::zero(); (1 << c) * lanes];
                message[w * lanes + 7] = coefficient;
                let code = Encoder::new(&table, c, d, lanes)
                    .unwrap()
                    .encode_extension(&message)
                    .unwrap();
                for (&x, &basis) in positions.iter().zip(&basis) {
                    let row = &code[x * lanes..(x + 1) * lanes];
                    assert_eq!(row[7], coefficient.mul_base(basis));
                    assert!(row.iter().enumerate().all(|(u, v)| u == 7 || v.is_zero()));
                }
                drop(code);
                drop(message);

                let lanes = 32;
                let word_lanes = 2 * lanes;
                let mut rows = vec![[0u64; 4]; (1 << c) * word_lanes / 4];
                rows[(w * word_lanes + 19) / 4][19 % 4] = 3;
                let code = Encoder::new(&table, c, d, lanes)
                    .unwrap()
                    .encode_rows(&rows)
                    .unwrap();
                for (&x, &basis) in positions.iter().zip(&basis) {
                    let row = &code[x * word_lanes..(x + 1) * word_lanes];
                    assert_eq!(row[19], F64::from_raw(3) * basis);
                    assert!(row.iter().enumerate().all(|(u, v)| u == 19 || v.is_zero()));
                }
            });
        }
    }
}

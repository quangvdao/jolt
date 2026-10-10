//! Streaming bridge weights and the first sumcheck round (§9 of
//! `specs/rv64i-binary-commitment.md`). Rows stay in their four-word representation.

use jolt_field::{Accumulator, ExtField, One, WithAccumulator, Zero, F128, F192, F64};
use jolt_rv64i_verifier::points::equality_table;
use jolt_rv64i_verifier::whir::error::{checked_product, try_vec, WhirError, WhirPart};
use rayon::prelude::*;

type ProductAccumulator = <F192 as WithAccumulator>::Accumulator;

/// Byte tables for `(Phi(e), Phi(r0 * e))`, with both maps in each 48-byte entry.
/// Constructed once per opening, never cached across challenges.
pub struct PhiTables {
    entries: Vec<[F192; 2]>,
}

impl PhiTables {
    pub fn new(alpha: F192, r0: F128) -> Result<Self, WhirError> {
        let powers = Self::powers(alpha);
        let mut columns = [[F192::zero(); 2]; 128];
        let mut rx = r0;
        for (column, power) in columns.iter_mut().zip(powers) {
            *column = [power, Self::from_bits(&powers, rx)];
            rx = rx.mul_x();
        }
        let mut entries = try_vec(WhirPart::CyclePoint, 16 * 256)?;
        entries.resize(16 * 256, [F192::zero(); 2]);
        for (table, basis) in entries.chunks_exact_mut(256).zip(columns.chunks_exact(8)) {
            for byte in 1usize..256 {
                let bit = byte.trailing_zeros() as usize;
                let previous = table[byte & (byte - 1)];
                table[byte] = [previous[0] + basis[bit][0], previous[1] + basis[bit][1]];
            }
        }
        Ok(Self { entries })
    }

    /// Returns the two maps in that order, using 16 paired loads and XORs.
    #[inline]
    pub fn evaluate(&self, e: F128) -> [F192; 2] {
        let mut out = [F192::zero(); 2];
        for (table, byte) in self.entries.chunks_exact(256).zip(e.to_raw().to_le_bytes()) {
            let entry = table[usize::from(byte)];
            out[0] += entry[0];
            out[1] += entry[1];
        }
        out
    }

    fn powers(alpha: F192) -> [F192; 128] {
        let mut powers = [F192::one(); 128];
        let mut power = F192::one();
        for entry in powers.iter_mut().skip(1) {
            power *= alpha;
            *entry = power;
        }
        powers
    }

    fn from_bits(powers: &[F192; 128], e: F128) -> F192 {
        powers
            .iter()
            .enumerate()
            .filter_map(|(bit, power)| ((e.to_raw() >> bit) & 1 != 0).then_some(*power))
            .sum()
    }
}

/// Temporary lookup and split equality tables; consuming the first pass releases
/// all three allocations before a folded message can be allocated.
pub struct BridgeTables {
    phi: PhiTables,
    low: Vec<F128>,
    high: Vec<F128>,
    pairs: usize,
}

impl BridgeTables {
    /// Supports every packed dimension `2..=33`, including empty low halves.
    #[expect(
        non_snake_case,
        reason = "protocol dimensions retain their mathematical names"
    )]
    pub fn new(point: &[F128], alpha: F192) -> Result<Self, WhirError> {
        let log_T = point.len().saturating_sub(1);
        if !(2..=33).contains(&point.len()) {
            return Err(WhirError::UnsupportedGeometry { log_T });
        }
        let (r0, rest) = point
            .split_first()
            .ok_or(WhirError::UnsupportedGeometry { log_T })?;
        let (lo, hi) = rest.split_at(rest.len() / 2);
        let low = equality_table(lo).map_err(|_| WhirError::Allocation {
            part: WhirPart::CyclePoint,
        })?;
        let high = equality_table(hi).map_err(|_| WhirError::Allocation {
            part: WhirPart::CyclePoint,
        })?;
        let pairs = checked_product(WhirPart::Rows, &[low.len(), high.len()])?;
        Ok(Self {
            phi: PhiTables::new(alpha, *r0)?,
            low,
            high,
            pairs,
        })
    }

    /// Reads each row once; writes `W0`, `D` and accumulates `u0`, `u2` together.
    /// Item 8 uses this for the first lane round whenever `k0 >= 1`.
    pub fn first_round(self, rows: &[[u64; 4]]) -> Result<FirstRound, WhirError> {
        self.first_round_impl::<false>(rows)
    }

    /// Measurement counterpart using six base products in the same streaming loop.
    pub fn first_round_composed(self, rows: &[[u64; 4]]) -> Result<FirstRound, WhirError> {
        self.first_round_impl::<true>(rows)
    }

    fn first_round_impl<const COMPOSED: bool>(
        self,
        rows: &[[u64; 4]],
    ) -> Result<FirstRound, WhirError> {
        if rows.len() != self.pairs {
            return Err(WhirError::Shape {
                part: WhirPart::Rows,
                expected: self.pairs,
                actual: rows.len(),
            });
        }
        let mut w0 = try_vec(WhirPart::CyclePoint, self.pairs)?;
        let mut d = try_vec(WhirPart::CyclePoint, self.pairs)?;
        w0.resize(self.pairs, F192::zero());
        d.resize(self.pairs, F192::zero());
        let sums = w0
            .par_iter_mut()
            .zip(d.par_iter_mut())
            .zip(rows.par_iter())
            .enumerate()
            .with_min_len(1024)
            .fold(
                || [ProductAccumulator::default(); 2],
                |mut sums, (k, ((w0, d), row))| {
                    let e = self.low[k % self.low.len()] * self.high[k / self.low.len()];
                    let [delta, w1] = self.phi.evaluate(e);
                    *d = delta;
                    *w0 = delta + w1;
                    let [a0, a1, b0, b1] = row.map(F64::from_raw);
                    if COMPOSED {
                        sums[0].fmadd_base(*w0, a0);
                        sums[0].fmadd_base(w0.mul_y(), a1);
                        sums[1].fmadd_base(delta, a0 + b0);
                        sums[1].fmadd_base(delta.mul_y(), a1 + b1);
                    } else {
                        sums[0].fmadd_base_pair(*w0, [a0, a1]);
                        sums[1].fmadd_base_pair(delta, [a0 + b0, a1 + b1]);
                    }
                    sums
                },
            )
            .reduce(
                || [ProductAccumulator::default(); 2],
                |mut a, b| {
                    for (acc, other) in a.iter_mut().zip(b) {
                        acc.merge(other);
                    }
                    a
                },
            );
        Ok(FirstRound {
            w0,
            d,
            coefficients: sums.map(Accumulator::reduce),
        })
    }
}

/// The two exact-sized weight allocations retained across the first challenge.
/// The coefficients are the constant and quadratic terms; the linear term is
/// recovered from the running claim by the sumcheck caller.
pub struct FirstRound {
    w0: Vec<F192>,
    d: Vec<F192>,
    coefficients: [F192; 2],
}

impl FirstRound {
    pub fn coefficients(&self) -> [F192; 2] {
        self.coefficients
    }
    #[cfg(test)]
    fn weights(&self) -> (&[F192], &[F192]) {
        (&self.w0, &self.d)
    }

    /// Allocates only the folded message, updates `W0` in place and drops `D`.
    /// Item 8 retains these allocations only within level 0; its last lane fold
    /// writes smaller allocations before constructing the next codeword.
    pub fn fold(mut self, rows: &[[u64; 4]], challenge: F192) -> Result<FoldedBridge, WhirError> {
        if rows.len() != self.w0.len() {
            return Err(WhirError::Shape {
                part: WhirPart::Rows,
                expected: self.w0.len(),
                actual: rows.len(),
            });
        }
        let mut message = try_vec(WhirPart::Rows, rows.len())?;
        message.resize(rows.len(), F192::zero());
        message
            .par_iter_mut()
            .zip(self.w0.par_iter_mut())
            .zip(self.d.par_iter())
            .zip(rows.par_iter())
            .for_each(|(((output, weight), delta), row)| {
                let [a0, a1, b0, b1] = row.map(F64::from_raw);
                *output = F192::from_base_fn(|i| match i {
                    0 => a0,
                    1 => a1,
                    _ => F64::zero(),
                }) + challenge.mul_base_pair([a0 + b0, a1 + b1]);
                *weight += challenge * *delta;
            });
        Ok(FoldedBridge {
            message,
            weight: self.w0,
        })
    }
}

/// Message and weight for subsequent adjacent-pair rounds, owned by item 8.
pub struct FoldedBridge {
    pub message: Vec<F192>,
    pub weight: Vec<F192>,
}

impl FoldedBridge {
    /// The `t=1`, `k0=0` recipe: four symbols and four weights from the
    /// definition, without lookup, split, `W0` or `D` allocations.
    pub fn dense(rows: &[[u64; 4]; 2], point: &[F128; 2], alpha: F192) -> Result<Self, WhirError> {
        let powers = PhiTables::powers(alpha);
        let mut message = try_vec(WhirPart::Rows, 4)?;
        let mut weight = try_vec(WhirPart::CyclePoint, 4)?;
        for row in rows {
            for pair in row.chunks_exact(2) {
                message.push(F192::from_base_fn(|i| {
                    if i < 2 {
                        F64::from_raw(pair[i])
                    } else {
                        F64::zero()
                    }
                }));
            }
        }
        for z in 0..4 {
            let eq = point
                .iter()
                .enumerate()
                .map(|(bit, r)| {
                    if z >> bit & 1 == 0 {
                        F128::one() + *r
                    } else {
                        *r
                    }
                })
                .product();
            weight.push(PhiTables::from_bits(&powers, eq));
        }
        Ok(Self { message, weight })
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "tests assert successful bridge construction"
)]
mod tests {
    use super::{BridgeTables, FoldedBridge, PhiTables};
    use jolt_field::{ExtField, One, Zero, F128, F192, F64};
    use jolt_rv64i_verifier::commitment::{BitsGeometry, BitsOpening};
    use jolt_rv64i_verifier::points::eq_index;
    use jolt_rv64i_verifier::whir::bridge::{slice_claims, tau};
    use jolt_rv64i_verifier::whir::error::{WhirError, WhirPart};
    use rayon::ThreadPoolBuilder;

    struct Random(u64);
    impl Random {
        fn word(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn h(&mut self) -> F128 {
            F128::from_raw(u128::from(self.word()) | (u128::from(self.word()) << 64))
        }
        fn e(&mut self) -> F192 {
            F192::from_base_fn(|_| F64::from_raw(self.word()))
        }
    }

    fn phi_definition(e: F128, alpha: F192) -> F192 {
        (0..128).rev().fold(F192::zero(), |sum, bit| {
            sum * alpha
                + if e.to_raw() >> bit & 1 == 0 {
                    F192::zero()
                } else {
                    F192::one()
                }
        })
    }

    fn symbol(row: &[u64; 4], g: usize) -> F192 {
        F192::from_base_fn(|i| {
            if i < 2 {
                F64::from_raw(row[2 * g + i])
            } else {
                F64::zero()
            }
        })
    }

    // General E multiplication and bit-by-bit weights are the independent oracle.
    fn reference_first_round(rows: &[[u64; 4]], point: &[F128], alpha: F192) -> [F192; 2] {
        let mut out = [F192::zero(); 2];
        for (k, row) in rows.iter().enumerate() {
            let w0 = phi_definition(eq_index(point, 2 * k).unwrap(), alpha);
            let w1 = phi_definition(eq_index(point, 2 * k + 1).unwrap(), alpha);
            let p0 = symbol(row, 0);
            let p1 = symbol(row, 1);
            out[0] += p0 * w0;
            out[1] += (p0 + p1) * (w0 + w1);
        }
        out
    }

    #[test]
    fn composed_phi_tables_match_bit_definition() {
        let mut rng = Random(0x8132_9014_1785_efa1);
        for r0 in [F128::zero(), F128::one(), F128::from_raw(2), rng.h()] {
            let alpha = rng.e();
            let tables = PhiTables::new(alpha, r0).unwrap();
            for bit in 0..128 {
                let e = F128::from_raw(1 << bit);
                assert_eq!(
                    tables.evaluate(e),
                    [phi_definition(e, alpha), phi_definition(r0 * e, alpha)]
                );
            }
            for _ in 0..256 {
                let e = rng.h();
                assert_eq!(
                    tables.evaluate(e),
                    [phi_definition(e, alpha), phi_definition(r0 * e, alpha)]
                );
            }
        }
    }

    #[test]
    fn fused_round_and_fold_match_dense_definition_all_small_sizes() {
        let mut rng = Random(0x7869_2017_11e2_8123);
        for t in 1..=11 {
            let rows: Vec<[u64; 4]> = (0..1 << t)
                .map(|_| std::array::from_fn(|_| rng.word()))
                .collect();
            for kind in 0..3 {
                let repeated = rng.h();
                let point: Vec<_> = (0..=t)
                    .map(|bit| match kind {
                        0 => rng.h(),
                        1 => F128::from_raw((bit & 1) as u128),
                        _ => repeated,
                    })
                    .collect();
                let alpha = rng.e();
                let expected = reference_first_round(&rows, &point, alpha);
                let first = BridgeTables::new(&point, alpha)
                    .unwrap()
                    .first_round(&rows)
                    .unwrap();
                assert_eq!(first.coefficients(), expected, "t={t}, coordinates={kind}");
                let (w0, d) = first.weights();
                for k in 0..rows.len() {
                    let a = phi_definition(eq_index(&point, 2 * k).unwrap(), alpha);
                    let b = phi_definition(eq_index(&point, 2 * k + 1).unwrap(), alpha);
                    assert_eq!(w0[k], a);
                    assert_eq!(d[k], a + b);
                }
                let challenge = rng.e();
                let folded = first.fold(&rows, challenge).unwrap();
                for (k, row) in rows.iter().enumerate() {
                    let a = phi_definition(eq_index(&point, 2 * k).unwrap(), alpha);
                    let b = phi_definition(eq_index(&point, 2 * k + 1).unwrap(), alpha);
                    assert_eq!(
                        folded.message[k],
                        symbol(row, 0) + challenge * (symbol(row, 0) + symbol(row, 1))
                    );
                    assert_eq!(folded.weight[k], a + challenge * (a + b));
                }
            }
        }
    }

    #[test]
    fn bridge_claim_matches_bitwise_f128_table_evaluation() {
        let mut rng = Random(0x611b_d571_eca2_9348);
        for t in 1..=10 {
            let rows: Vec<[u64; 4]> = (0..1 << t)
                .map(|_| std::array::from_fn(|_| rng.word()))
                .collect();
            let cycle: Vec<_> = (0..t).map(|_| rng.h()).collect();
            let rho = std::array::from_fn::<_, 8, _>(|_| rng.h());
            let mut columns = [F128::zero(); 256];
            let mut value = F128::zero();
            let column_weights: Vec<_> = (0..256).map(|col| eq_index(&rho, col).unwrap()).collect();
            for (j, row) in rows.iter().enumerate() {
                let cycle_weight = eq_index(&cycle, j).unwrap();
                for col in 0..256 {
                    if row[col / 64] >> (col % 64) & 1 != 0 {
                        columns[col] += cycle_weight;
                        value += cycle_weight * column_weights[col];
                    }
                }
            }
            let request = BitsOpening {
                geometry: BitsGeometry { log_T: t },
                column_point: &rho,
                cycle_point: &cycle,
                columns: &columns,
            };
            let slices = slice_claims(&request).unwrap();
            let reconstructed: F128 = slices
                .iter()
                .enumerate()
                .map(|(b, s)| *s * eq_index(&rho[..7], b).unwrap())
                .sum();
            assert_eq!(reconstructed, value);
            assert_eq!(request.value(), value);
            let point: Vec<_> = std::iter::once(rho[7])
                .chain(cycle.iter().copied())
                .collect();
            let alpha = rng.e();
            let first = BridgeTables::new(&point, alpha)
                .unwrap()
                .first_round(&rows)
                .unwrap();
            let (w0, d) = first.weights();
            let inner: F192 = rows
                .iter()
                .enumerate()
                .map(|(k, row)| symbol(row, 0) * w0[k] + symbol(row, 1) * (w0[k] + d[k]))
                .sum();
            assert_eq!(tau(&slices, alpha), inner);
            if t == 1 {
                let dense =
                    FoldedBridge::dense(&[rows[0], rows[1]], &[point[0], point[1]], alpha).unwrap();
                for z in 0..4 {
                    assert_eq!(dense.message[z], symbol(&rows[z / 2], z % 2));
                    assert_eq!(
                        dense.weight[z],
                        phi_definition(eq_index(&point, z).unwrap(), alpha)
                    );
                }
                assert_eq!(
                    dense
                        .message
                        .iter()
                        .zip(&dense.weight)
                        .map(|(p, w)| *p * *w)
                        .sum::<F192>(),
                    inner
                );
            }
        }
    }

    #[test]
    fn streaming_round_is_deterministic_on_one_and_twelve_threads() {
        let mut rng = Random(0x8793_6123_eca1_8791);
        let rows: Vec<_> = (0..1 << 11)
            .map(|_| std::array::from_fn(|_| rng.word()))
            .collect();
        let point: Vec<_> = (0..12).map(|_| rng.h()).collect();
        let alpha = rng.e();
        let challenge = rng.e();
        let run = |threads| {
            ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let first = BridgeTables::new(&point, alpha)
                        .unwrap()
                        .first_round(&rows)
                        .unwrap();
                    let coefficients = first.coefficients();
                    let folded = first.fold(&rows, challenge).unwrap();
                    (coefficients, folded.message, folded.weight)
                })
        };
        assert_eq!(run(1), run(12));
    }

    #[test]
    fn bridge_dimension_and_row_faults_are_typed() {
        let alpha = F192::one();
        for mu in [0usize, 1, 34] {
            assert!(matches!(
                BridgeTables::new(&vec![F128::one(); mu], alpha),
                Err(WhirError::UnsupportedGeometry { .. })
            ));
        }
        let result = BridgeTables::new(&[F128::zero(); 2], alpha)
            .unwrap()
            .first_round(&[]);
        assert!(matches!(
            result,
            Err(WhirError::Shape {
                part: WhirPart::Rows,
                expected: 2,
                actual: 0
            })
        ));
        let first = BridgeTables::new(&[F128::zero(); 2], alpha)
            .unwrap()
            .first_round(&[[0; 4]; 2])
            .unwrap();
        assert!(matches!(
            first.fold(&[], alpha),
            Err(WhirError::Shape {
                part: WhirPart::Rows,
                expected: 2,
                actual: 0
            })
        ));
        // Large dimensions exercise only split storage, never a 2^t equality allocation.
        for t in 1..=32 {
            let point = vec![F128::one(); t + 1];
            let tables = BridgeTables::new(&point, alpha).unwrap();
            assert_eq!(tables.low.len(), 1 << (t / 2));
            assert_eq!(tables.high.len(), 1 << (t - t / 2));
            assert_eq!(
                tables.low[0],
                if t < 2 { F128::one() } else { F128::zero() }
            );
            assert_eq!(*tables.low.last().unwrap(), F128::one());
            assert_eq!(*tables.high.last().unwrap(), F128::one());
        }
    }
}

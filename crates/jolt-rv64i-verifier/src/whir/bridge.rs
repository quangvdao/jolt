//! The bit transposition and tensor recurrence of the commitment bridge.
//! See §3 of `specs/rv64i-binary-commitment.md`.

use super::error::{WhirError, WhirPart};
use crate::commitment::BitsOpening;
use jolt_field::{ExtField, One, Zero, F128, F192, F64};

/// Derives the 128 partial claims from the request after validating its geometry
/// and all three lengths. The retained-state caller checks geometry equality.
pub fn slice_claims(opening: &BitsOpening<'_>) -> Result<[F128; 128], WhirError> {
    let log_T = opening.geometry.log_T;
    if !(1..=32).contains(&log_T) {
        return Err(WhirError::UnsupportedGeometry { log_T });
    }
    for (part, expected, actual) in [
        (WhirPart::ColumnPoint, 8, opening.column_point.len()),
        (WhirPart::CyclePoint, log_T, opening.cycle_point.len()),
        (WhirPart::Columns, 256, opening.columns.len()),
    ] {
        if actual != expected {
            return Err(WhirError::Shape {
                part,
                expected,
                actual,
            });
        }
    }
    let first = opening
        .column_point
        .last()
        .copied()
        .ok_or(WhirError::Shape {
            part: WhirPart::ColumnPoint,
            expected: 8,
            actual: opening.column_point.len(),
        })?;
    let mut slices = [F128::zero(); 128];
    for ((slice, low), high) in slices
        .iter_mut()
        .zip(opening.columns.iter().take(128))
        .zip(opening.columns.iter().skip(128))
    {
        *slice = *low + first * (*low + *high);
    }
    Ok(slices)
}

/// Transposes the raw bits of the slice claims into the packed-symbol basis and
/// batches the columns by Horner's rule, with exactly 127 products in `F192`.
pub fn tau(slices: &[F128; 128], alpha: F192) -> F192 {
    let transposed = std::array::from_fn(|h| {
        let mut low = 0u64;
        let mut high = 0u64;
        for (b, slice) in slices.iter().enumerate() {
            if (slice.to_raw() >> h) & 1 != 0 {
                if b < 64 {
                    low |= 1u64 << b;
                } else {
                    high |= 1u64 << (b - 64);
                }
            }
        }
        F192::from_base_fn(|coefficient| match coefficient {
            0 => F64::from_raw(low),
            1 => F64::from_raw(high),
            _ => F64::zero(),
        })
    });
    horner(&transposed, alpha)
}

fn horner(coefficients: &[F192; 128], alpha: F192) -> F192 {
    let mut descending = coefficients.iter().rev();
    let leading = descending.next().copied().unwrap_or_default();
    descending.fold(leading, |value, coefficient| value * alpha + *coefficient)
}

/// A value in `F192 ⊗ F128`, initially one, whose updates multiply by
/// `(1 + q_l) ⊗ 1 + 1 ⊗ r_l`. No map between the two fields is assumed.
pub struct BridgeWeight {
    coefficients: [F192; 128],
}

impl Default for BridgeWeight {
    fn default() -> Self {
        Self {
            coefficients: std::array::from_fn(|h| if h == 0 { F192::one() } else { F192::zero() }),
        }
    }
}

impl BridgeWeight {
    /// Starts the tensor product before any coordinate is absorbed.
    pub fn new() -> Self {
        Self::default()
    }

    /// Absorbs one pair of coordinates, in the caller's low-variable-first order.
    /// Every multiplication-matrix column is formed from the preceding one by
    /// multiplication by `x` in `F128`; only its set bits select additions.
    pub fn absorb(&mut self, r_l: F128, q_l: F192) {
        let scale = F192::one() + q_l;
        let mut next = self.coefficients.map(|coefficient| scale * coefficient);
        let mut column = r_l;
        for coefficient in self.coefficients {
            let bits = column.to_raw();
            for (h, destination) in next.iter_mut().enumerate() {
                if (bits >> h) & 1 != 0 {
                    *destination += coefficient;
                }
            }
            column = column.mul_x();
        }
        self.coefficients = next;
    }

    /// Applies the `F192`-linear map `x^h ↦ alpha^h` after the tensor product.
    pub fn finish(self, alpha: F192) -> F192 {
        horner(&self.coefficients, alpha)
    }
}

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    reason = "tests use fixed dimensions and assert validated request results"
)]
mod tests {
    use super::{slice_claims, tau, BridgeWeight};
    use crate::commitment::{BitsGeometry, BitsOpening};
    use crate::whir::error::{WhirError, WhirPart};
    use jolt_field::{ExtField, One, Zero, F128, F192, F64};

    fn packed(low: u64, high: u64) -> F192 {
        F192::from_base_fn(|i| match i {
            0 => F64::from_raw(low),
            1 => F64::from_raw(high),
            _ => F64::zero(),
        })
    }

    fn random_raw(seed: &mut u128) -> u128 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    fn random_e(seed: &mut u128) -> F192 {
        F192::from_base_fn(|_| F64::from_raw(random_raw(seed) as u64))
    }

    fn phi_by_bits(value: F128, alpha: F192) -> F192 {
        let mut power = F192::one();
        let mut result = F192::zero();
        for h in 0..128 {
            if (value.to_raw() >> h) & 1 != 0 {
                result += power;
            }
            power *= alpha;
        }
        result
    }

    fn eq_h(point: &[F128], index: usize) -> F128 {
        point.iter().enumerate().fold(F128::one(), |v, (l, r)| {
            v * if (index >> l) & 1 == 0 {
                F128::one() + *r
            } else {
                *r
            }
        })
    }

    fn eq_e(point: &[F192], index: usize) -> F192 {
        point.iter().enumerate().fold(F192::one(), |v, (l, q)| {
            v * if (index >> l) & 1 == 0 {
                F192::one() + *q
            } else {
                *q
            }
        })
    }

    #[test]
    fn bridge_tau_basis_literals() {
        let alpha = packed(2, 0);
        for (b, expected) in [(3, packed(8, 0)), (67, packed(0, 8))] {
            let mut slices = [F128::zero(); 128];
            slices[b] = F128::one();
            assert_eq!(tau(&slices, alpha), expected);
            slices[b] = F128::from_raw(2);
            let literal = if b == 3 { packed(16, 0) } else { packed(0, 16) };
            assert_eq!(tau(&slices, alpha), literal);
        }
    }

    #[test]
    fn bridge_single_coordinate_literal() {
        let mut bridge = BridgeWeight::new();
        let q = packed(4, 8);
        let alpha = packed(16, 32);
        bridge.absorb(F128::from_raw(2), q);
        assert_eq!(bridge.finish(alpha), packed(21, 40));
    }

    #[test]
    fn bridge_tau_matches_bitwise_table_claims() {
        let mut seed = 0xe7e9_d80a_af50_6ea1_2934_4839_2132_8705;
        for mu in 1..=10 {
            let r: Vec<_> = (0..mu)
                .map(|_| F128::from_raw(random_raw(&mut seed)))
                .collect();
            let alpha = random_e(&mut seed);
            let mut slices = [F128::zero(); 128];
            let mut expected = F192::zero();
            for z in 0..1 << mu {
                let words = [random_raw(&mut seed) as u64, random_raw(&mut seed) as u64];
                let equality = eq_h(&r, z);
                for (b, slice) in slices.iter_mut().enumerate() {
                    if (words[b / 64] >> (b % 64)) & 1 != 0 {
                        *slice += equality;
                    }
                }
                expected += packed(words[0], words[1]) * phi_by_bits(equality, alpha);
            }
            assert_eq!(tau(&slices, alpha), expected);
        }
    }

    #[test]
    fn bridge_recurrence_matches_direct_cube_sum() {
        let mut seed = 0x7dd6_e9c6_3b5d_7b92_5358_4708_1490_0ad7;
        for mu in 0..=10 {
            for kind in 0..3 {
                let repeated = F128::from_raw(random_raw(&mut seed));
                let r: Vec<_> = (0..mu)
                    .map(|l| match kind {
                        0 => F128::from_raw(random_raw(&mut seed)),
                        1 => F128::from_raw((l % 2) as u128),
                        _ => repeated,
                    })
                    .collect();
                let q: Vec<_> = (0..mu)
                    .map(|l| {
                        if kind == 1 {
                            F192::from_base_fn(|i| F64::from_raw(u64::from(i == 0 && l % 2 == 1)))
                        } else {
                            random_e(&mut seed)
                        }
                    })
                    .collect();
                let alpha = random_e(&mut seed);
                let mut bridge = BridgeWeight::new();
                for (r_l, q_l) in r.iter().zip(&q) {
                    bridge.absorb(*r_l, *q_l);
                }
                let expected: F192 = (0..1 << mu)
                    .map(|z| eq_e(&q, z) * phi_by_bits(eq_h(&r, z), alpha))
                    .sum();
                assert_eq!(bridge.finish(alpha), expected, "mu={mu}, kind={kind}");
            }
        }
    }

    #[test]
    fn bridge_recurrence_matches_direct_tensor_multiplication() {
        let mut seed = 0x5191_559e_6407_671b_09ae_0b36_2854_56c3;
        let mut coefficients = [F192::zero(); 128];
        coefficients[0] = F192::one();
        let mut bridge = BridgeWeight::new();
        for _ in 0..10 {
            let r = F128::from_raw(random_raw(&mut seed));
            let q = random_e(&mut seed);
            let mut product = [F192::zero(); 128];
            for (i, coefficient) in coefficients.iter().enumerate() {
                product[i] += *coefficient * (F192::one() + q);
                // Independently reduce each monomial x^(i+j), rather than using
                // successive multiplication-matrix columns.
                for j in 0..128 {
                    if (r.to_raw() >> j) & 1 == 0 {
                        continue;
                    }
                    let mut monomial = 1u128;
                    for _ in 0..i + j {
                        let carry = monomial >> 127;
                        monomial <<= 1;
                        if carry != 0 {
                            monomial ^= 0x87;
                        }
                    }
                    for (h, destination) in product.iter_mut().enumerate() {
                        if (monomial >> h) & 1 != 0 {
                            *destination += *coefficient;
                        }
                    }
                }
            }
            coefficients = product;
            bridge.absorb(r, q);
            assert_eq!(bridge.coefficients, coefficients);
        }
    }

    #[test]
    fn bridge_changed_slice_has_explicit_nonzero_discrepancy_polynomial() {
        let mut slices = [F128::zero(); 128];
        slices[67] = F128::from_raw(1u128 << 127);
        // The discrepancy polynomial is exactly nu_67 * alpha^127, so its
        // degree is 127 and zero is its only root in the opening field.
        assert_eq!(tau(&slices, F192::zero()), F192::zero());
        assert_eq!(tau(&slices, F192::one()), packed(0, 8));
        let mut seed = 0xc0f4_0910_886c_718c_5173_b1d5_5bb7_9c09;
        for _ in 0..128 {
            let alpha = random_e(&mut seed);
            assert_ne!(alpha, F192::zero());
            let mut power = F192::one();
            for _ in 0..127 {
                power *= alpha;
            }
            let discrepancy = packed(0, 8) * power;
            assert_ne!(discrepancy, F192::zero());
            assert_eq!(tau(&slices, alpha), discrepancy);
        }
    }

    #[test]
    fn bridge_slice_claims_reconstruct_opening_value() {
        let mut seed = 0x5778_fba3_75af_f792_f748_54cd_ea43_aa15;
        for t in 1..=10 {
            let point: Vec<_> = (0..8)
                .map(|_| F128::from_raw(random_raw(&mut seed)))
                .collect();
            let cycles = vec![F128::one(); t];
            let columns: Vec<_> = (0..256)
                .map(|_| F128::from_raw(random_raw(&mut seed)))
                .collect();
            let opening = BitsOpening {
                geometry: BitsGeometry { log_T: t },
                column_point: &point,
                cycle_point: &cycles,
                columns: &columns,
            };
            let slices = slice_claims(&opening).unwrap();
            let value: F128 = slices
                .iter()
                .enumerate()
                .map(|(b, s)| eq_h(&point[..7], b) * *s)
                .sum();
            assert_eq!(value, opening.value());
            for b in 0..128 {
                assert_eq!(
                    slices[b],
                    (F128::one() + point[7]) * columns[b] + point[7] * columns[b + 128]
                );
            }
        }
    }

    #[test]
    fn bridge_slice_claims_reject_malformed_requests() {
        let point = [F128::zero(); 8];
        let cycles = [F128::zero(); 1];
        let columns = [F128::zero(); 256];
        for t in [0, 33, usize::MAX] {
            let request = BitsOpening {
                geometry: BitsGeometry { log_T: t },
                column_point: &point,
                cycle_point: &cycles,
                columns: &columns,
            };
            assert_eq!(
                slice_claims(&request),
                Err(WhirError::UnsupportedGeometry { log_T: t })
            );
        }
        for (column_point, cycle_point, columns, part, expected, actual) in [
            (
                &point[..7],
                &cycles[..],
                &columns[..],
                WhirPart::ColumnPoint,
                8,
                7,
            ),
            (
                &point[..],
                &cycles[..0],
                &columns[..],
                WhirPart::CyclePoint,
                1,
                0,
            ),
            (
                &point[..],
                &cycles[..],
                &columns[..255],
                WhirPart::Columns,
                256,
                255,
            ),
        ] {
            let request = BitsOpening {
                geometry: BitsGeometry { log_T: 1 },
                column_point,
                cycle_point,
                columns,
            };
            assert_eq!(
                slice_claims(&request),
                Err(WhirError::Shape {
                    part,
                    expected,
                    actual
                })
            );
        }
    }
}

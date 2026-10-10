//! Concrete reduction of the six committed functionals to one cycle point.

use jolt_claims::SymbolicSumcheck;
use jolt_field::JoltField;
use jolt_rv64i_arith::{Layout, BITS_COLUMNS};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::VerifierError;

use crate::claims::bits_reduction::BitsReductionSymbolic;
pub use crate::claims::bits_reduction::{
    BitsReductionChallenges, BitsReductionInputClaims, BitsReductionOutputClaims,
};
use crate::ids::{BitsReductionDerived, DerivedId};
use crate::points::{self, PointsError};

#[derive(Clone)]
pub struct BitsReduction<F: JoltField> {
    symbolic: BitsReductionSymbolic,
    r_1: Vec<F>,
    r_3: Vec<F>,
    r_5: Vec<F>,
    w: Vec<F>,
    x: Vec<F>,
    weights: [Vec<(usize, F)>; 6],
    pos_zero: [F; 2],
}

impl<F: JoltField> BitsReduction<F> {
    pub fn new(
        layout: &Layout,
        r_1: Vec<F>,
        r_3: Vec<F>,
        r_5: Vec<F>,
        w: Vec<F>,
        x: Vec<F>,
    ) -> Result<Self, PointsError> {
        for point in [&r_3, &r_5] {
            if point.len() != r_1.len() {
                return Err(PointsError::Dimension {
                    expected: r_1.len(),
                    actual: point.len(),
                });
            }
        }
        for (point, expected) in [(&w, 10), (&x, 17)] {
            if point.len() != expected {
                return Err(PointsError::Dimension {
                    expected,
                    actual: point.len(),
                });
            }
        }
        let bit = x.get(..6).ok_or(PointsError::Dimension {
            expected: 6,
            actual: x.len(),
        })?;
        let slot = x.get(6..10).ok_or(PointsError::Dimension {
            expected: 10,
            actual: x.len(),
        })?;
        let w_bit = w.get(..6).ok_or(PointsError::Dimension {
            expected: 6,
            actual: w.len(),
        })?;
        let w_slot = w.get(6..).ok_or(PointsError::Dimension {
            expected: 10,
            actual: w.len(),
        })?;
        let w_low = points::eq_table(w_bit)?;
        let w_high = points::eq_table(w_slot)?;
        let x_low = points::eq_table(bit)?;
        let x_high = points::eq_table(slot)?;
        let split_weight = |low: &[F], high: &[F], index: usize| {
            let low_weight = low.get(index & 63).ok_or(PointsError::Index {
                index,
                variables: 10,
            })?;
            let high_weight = high.get(index >> 6).ok_or(PointsError::Index {
                index,
                variables: 10,
            })?;
            Ok::<F, PointsError>(*low_weight * *high_weight)
        };
        let g = layout
            .ram_ra()
            .first()
            .map(|chunk| usize::from(chunk.start()))
            .ok_or(PointsError::MissingColumn { column: 64 })?;
        let mut weights: [Vec<(usize, F)>; 6] = std::array::from_fn(|_| Vec::new());
        for y in 64..=layout.keys_differ() {
            if let Some(direct) = weights.get_mut(0) {
                direct.push((y, split_weight(&w_low, &w_high, 768 + y)?));
            }
        }
        let inc_slot = x_high.get(8).copied().ok_or(PointsError::Index {
            index: 8,
            variables: 4,
        })?;
        for y in 0..64 {
            let weight = x_low.get(y).copied().ok_or(PointsError::Index {
                index: y,
                variables: 6,
            })?;
            if let Some(variant) = weights.get_mut(1) {
                variant.push((y, inc_slot * weight));
            }
            if let Some(inc) = weights.get_mut(5) {
                inc.push((y, weight));
            }
        }
        for y in g..layout.used_columns() {
            if let Some(variant) = weights.get_mut(1) {
                variant.push((y, split_weight(&x_low, &x_high, 576 + y - g)?));
            }
        }
        let mut pos_zero = [F::zero(); 2];
        for ((descriptor, digit_point), index) in layout
            .pos_ra()
            .into_iter()
            .zip([x.get(6..9), x.get(9..12)])
            .zip(0..2)
        {
            let point = digit_point.ok_or(PointsError::Dimension {
                expected: 12,
                actual: x.len(),
            })?;
            let digit_weights = points::eq_table(point)?;
            let zero = digit_weights.first().copied().ok_or(PointsError::Index {
                index: 0,
                variables: 3,
            })?;
            if let Some(target) = pos_zero.get_mut(index) {
                *target = zero;
            }
            for digit in 1..=descriptor.indicators() {
                if let Some(target) = weights.get_mut(index + 2) {
                    target.push((
                        usize::from(descriptor.start()) + digit - 1,
                        digit_weights
                            .get(digit)
                            .copied()
                            .ok_or(PointsError::Index {
                                index: digit,
                                variables: 3,
                            })?
                            + zero,
                    ));
                }
            }
        }
        if let Some(branch) = weights.get_mut(4) {
            branch.push((layout.should_branch(), F::one()));
        }
        Ok(Self {
            symbolic: BitsReductionSymbolic::new(r_1.len()),
            r_1,
            r_3,
            r_5,
            w,
            x,
            weights,
            pos_zero,
        })
    }

    pub fn r_1(&self) -> &[F] {
        &self.r_1
    }
    pub fn r_3(&self) -> &[F] {
        &self.r_3
    }
    pub fn r_5(&self) -> &[F] {
        &self.r_5
    }
    pub fn weights(&self) -> &[Vec<(usize, F)>; 6] {
        &self.weights
    }
    pub fn pos_zero(&self) -> [F; 2] {
        self.pos_zero
    }

    pub fn input_points(&self) -> BitsReductionInputClaims<Vec<F>> {
        BitsReductionInputClaims {
            direct_columns: self.w.iter().chain(&self.r_1).copied().collect(),
            variant_bits: self.x.iter().take(10).chain(&self.r_3).copied().collect(),
            pos_ra_0: self
                .x
                .iter()
                .skip(6)
                .take(3)
                .chain(&self.r_3)
                .copied()
                .collect(),
            pos_ra_1: self
                .x
                .iter()
                .skip(9)
                .take(3)
                .chain(&self.r_3)
                .copied()
                .collect(),
            should_branch: self.r_3.clone(),
            inc: self.x.iter().take(6).chain(&self.r_5).copied().collect(),
        }
    }

    pub fn column_weight(
        &self,
        column: usize,
        point: &[F],
        challenges: &BitsReductionChallenges<F>,
    ) -> Result<F, PointsError> {
        if column >= BITS_COLUMNS {
            return Err(PointsError::MissingColumn { column });
        }
        let cycle_weights = [
            points::eq(&self.r_1, point)?,
            points::eq(&self.r_3, point)?,
            points::eq(&self.r_5, point)?,
        ];
        let coefficients = [
            challenges.direct_columns,
            challenges.variant_bits,
            challenges.pos_ra_0,
            challenges.pos_ra_1,
            challenges.should_branch,
            challenges.inc,
        ];
        self.weights.iter().zip(coefficients).enumerate().try_fold(
            F::zero(),
            |sum, (index, (weights, coefficient))| {
                let cycle_index = if index == 0 {
                    0
                } else if index == 5 {
                    2
                } else {
                    1
                };
                let cycle = cycle_weights
                    .get(cycle_index)
                    .copied()
                    .ok_or(PointsError::MissingColumn { column })?;
                let weight: F = weights
                    .iter()
                    .filter(|(y, _)| *y == column)
                    .map(|(_, weight)| *weight)
                    .sum();
                Ok(sum + cycle * coefficient * weight)
            },
        )
    }

    fn term_error(error: PointsError) -> VerifierError {
        VerifierError::StageClaimSumcheckFailed {
            stage: "BitsReduction".to_owned(),
            reason: error.to_string(),
        }
    }
}

impl<F: JoltField> ConcreteSumcheck<F> for BitsReduction<F> {
    type Symbolic = BitsReductionSymbolic;
    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }
    fn derive_opening_points(
        &self,
        point: &[F],
        _inputs: &BitsReductionInputClaims<Vec<F>>,
    ) -> Result<BitsReductionOutputClaims<Vec<F>>, VerifierError> {
        if point.len() != self.rounds() {
            return Err(Self::term_error(PointsError::Dimension {
                expected: self.rounds(),
                actual: point.len(),
            }));
        }
        Ok(BitsReductionOutputClaims {
            columns: vec![point.to_vec(); BITS_COLUMNS],
        })
    }
    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "foreign derived ids fail closed as missing claims"
    )]
    fn derive_input_term(
        &self,
        id: &DerivedId,
        _challenges: &BitsReductionChallenges<F>,
    ) -> Result<F, VerifierError> {
        match id {
            DerivedId::BitsReduction(BitsReductionDerived::PosZero(digit)) => self
                .pos_zero
                .get(*digit)
                .copied()
                .ok_or(VerifierError::MissingStageClaimDerived { id: (*id).into() }),
            _ => Err(VerifierError::MissingStageClaimDerived { id: (*id).into() }),
        }
    }
    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "foreign derived ids fail closed as missing claims"
    )]
    fn derive_output_term(
        &self,
        id: &DerivedId,
        _inputs: &BitsReductionInputClaims<Vec<F>>,
        outputs: &BitsReductionOutputClaims<Vec<F>>,
        challenges: &BitsReductionChallenges<F>,
    ) -> Result<F, VerifierError> {
        match id {
            DerivedId::BitsReduction(BitsReductionDerived::ColumnWeight(column)) => {
                let point = outputs
                    .columns
                    .get(*column)
                    .ok_or(VerifierError::MissingStageClaimDerived { id: (*id).into() })?;
                self.column_weight(*column, point, challenges)
                    .map_err(Self::term_error)
            }
            _ => Err(VerifierError::MissingStageClaimDerived { id: (*id).into() }),
        }
    }
}

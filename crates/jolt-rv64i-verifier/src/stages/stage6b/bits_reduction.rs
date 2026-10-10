//! Concrete reduction of the six committed functionals to one cycle point.

use crate::proof::DimensionedRelation;
use std::sync::Arc;

use jolt_claims::{OutputClaims, SumcheckChallenges, SymbolicSumcheck};
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

/// Six committed functionals reduced over low-variable-first cycles to the 256 transmitted bit columns.
/// Its input claims and fixed points come from batches 1, 2, 3a, 3b and 5.
#[derive(Clone)]
pub struct BitsReduction<F: JoltField> {
    symbolic: BitsReductionSymbolic,
    r_1: Vec<F>,
    r_3: Arc<Vec<F>>,
    r_5: Arc<Vec<F>>,
    w: Vec<F>,
    x: Vec<F>,
    weights: [Vec<(usize, F)>; 6],
    pos_zero: [F; 2],
}

impl<F: JoltField> BitsReduction<F> {
    /// Checks equal cycle widths and the ten-column and seventeen-short-slot point widths, returning `PointsError` on mismatch.
    /// The points must be verified upstream outputs; checked inputs establish their common trace-width bound.
    pub fn new(
        layout: &Layout,
        r_1: Vec<F>,
        r_3: Vec<F>,
        r_5: Vec<F>,
        w: Vec<F>,
        x: Vec<F>,
    ) -> Result<Self, PointsError> {
        Self::new_shared(layout, r_1, Arc::new(r_3), Arc::new(r_5), w, x)
    }

    pub(crate) fn new_shared(
        layout: &Layout,
        r_1: Vec<F>,
        r_3: Arc<Vec<F>>,
        r_5: Arc<Vec<F>>,
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
            symbolic: Self::symbolic_for(r_1.len(), layout),
            r_1,
            r_3,
            r_5,
            w,
            x,
            weights,
            pos_zero,
        })
    }

    /// The verified low-variable-first batch-1 cycle point used by the direct-column functional.
    pub fn r_1(&self) -> &[F] {
        &self.r_1
    }
    /// The verified low-variable-first batch-3b cycle point used by the router functionals.
    pub fn r_3(&self) -> &[F] {
        &self.r_3
    }
    /// The verified low-variable-first batch-5 cycle point used by the increment functional.
    pub fn r_5(&self) -> &[F] {
        &self.r_5
    }
    /// Sparse column supports in Direct, Variant, Pos0, Pos1, ShouldBranch and Inc order.
    pub fn weights(&self) -> &[Vec<(usize, F)>; 6] {
        &self.weights
    }
    /// The omitted-zero weights for the two position chunks at batch 3a's fixed short points.
    pub fn pos_zero(&self) -> [F; 2] {
        self.pos_zero
    }

    /// Consumed functional points with fixed column/bit/position coordinates before their upstream cycle coordinates.
    pub fn input_points(&self) -> BitsReductionInputClaims<Vec<F>> {
        BitsReductionInputClaims {
            direct_columns: self.w.iter().chain(&self.r_1).copied().collect(),
            variant_bits: self
                .x
                .iter()
                .take(10)
                .chain(self.r_3.iter())
                .copied()
                .collect(),
            pos_ra_0: self
                .x
                .iter()
                .skip(6)
                .take(3)
                .chain(self.r_3.iter())
                .copied()
                .collect(),
            pos_ra_1: self
                .x
                .iter()
                .skip(9)
                .take(3)
                .chain(self.r_3.iter())
                .copied()
                .collect(),
            should_branch: self.r_3.as_ref().clone(),
            inc: self
                .x
                .iter()
                .take(6)
                .chain(self.r_5.iter())
                .copied()
                .collect(),
        }
    }

    /// The terminal coefficient of one transmitted column at the low-variable-first cycle point.
    /// Returns `PointsError` for a column outside 0–255 or a mismatched cycle width.
    pub fn column_weight(
        &self,
        column: usize,
        point: &[F],
        challenges: &BitsReductionChallenges<F>,
    ) -> Result<F, PointsError> {
        if column >= BITS_COLUMNS {
            return Err(PointsError::MissingColumn { column });
        }
        self.column_weights(point, challenges)?
            .get(column)
            .copied()
            .ok_or(PointsError::MissingColumn { column })
    }

    /// The terminal coefficients of all 256 transmitted columns at a low-variable-first cycle point.
    /// Returns `PointsError` unless its width matches the verified upstream cycle points.
    pub fn column_weights(
        &self,
        point: &[F],
        challenges: &BitsReductionChallenges<F>,
    ) -> Result<[F; BITS_COLUMNS], PointsError> {
        let cycle_1 = points::eq(&self.r_1, point)?;
        let cycle_3 = points::eq(&self.r_3, point)?;
        let cycle_5 = points::eq(&self.r_5, point)?;
        let coefficients = [
            cycle_1 * challenges.direct_columns,
            cycle_3 * challenges.variant_bits,
            cycle_3 * challenges.pos_ra_0,
            cycle_3 * challenges.pos_ra_1,
            cycle_3 * challenges.should_branch,
            cycle_5 * challenges.inc,
        ];
        let mut columns = [F::zero(); BITS_COLUMNS];
        for (weights, coefficient) in self.weights.iter().zip(coefficients) {
            for &(column, weight) in weights {
                *columns
                    .get_mut(column)
                    .ok_or(PointsError::MissingColumn { column })? += coefficient * weight;
            }
        }
        Ok(columns)
    }

    fn term_error(error: PointsError) -> VerifierError {
        VerifierError::StageClaimSumcheckFailed {
            stage: "BitsReduction".to_owned(),
            reason: error.to_string(),
        }
    }
}

impl<F: JoltField> ConcreteSumcheck<F> for BitsReduction<F> {
    fn instance_point_offset(&self, batch_num_vars: usize) -> Result<usize, VerifierError> {
        Self::point_offset(self.rounds(), batch_num_vars)
    }

    type Symbolic = BitsReductionSymbolic;
    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }
    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "foreign derived ids fail closed as missing claims"
    )]
    fn expected_output(
        &self,
        _inputs: &BitsReductionInputClaims<Vec<F>>,
        values: &BitsReductionOutputClaims<F>,
        outputs: &BitsReductionOutputClaims<Vec<F>>,
        challenges: &BitsReductionChallenges<F>,
    ) -> Result<F, VerifierError> {
        let point = outputs.columns.first().ok_or_else(|| {
            Self::term_error(PointsError::Dimension {
                expected: BITS_COLUMNS,
                actual: outputs.columns.len(),
            })
        })?;
        let weights = self
            .column_weights(point, challenges)
            .map_err(Self::term_error)?;
        self.symbolic.output_expression::<F>().try_evaluate(
            |id| {
                values
                    .resolve_output(id)
                    .ok_or(VerifierError::MissingOpeningClaim { id: (*id).into() })
            },
            |id| {
                challenges
                    .resolve_challenge(id)
                    .ok_or(VerifierError::MissingStageClaimChallenge { id: (*id).into() })
            },
            |id| match id {
                DerivedId::BitsReduction(BitsReductionDerived::ColumnWeight(column)) => weights
                    .get(*column)
                    .copied()
                    .ok_or(VerifierError::MissingStageClaimDerived { id: (*id).into() }),
                _ => Err(VerifierError::MissingStageClaimDerived { id: (*id).into() }),
            },
        )
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

impl<F: JoltField> DimensionedRelation<F> for BitsReduction<F> {
    type Dimensions = usize;

    fn symbolic_with(dimensions: Self::Dimensions) -> Self::Symbolic {
        BitsReductionSymbolic::new(dimensions)
    }

    fn dimensions(log_T: usize, _layout: &Layout) -> Self::Dimensions {
        log_T
    }
}

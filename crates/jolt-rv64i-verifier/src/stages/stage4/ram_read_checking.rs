//! Concrete ram read checking with address variables before cycles.
use super::verify::{check_read_points, evaluate_output, term_error};
use crate::claims::ram_read_checking::RamReadCheckingSymbolic;
pub use crate::claims::ram_read_checking::{
    RamReadCheckingChallenges, RamReadCheckingInputClaims, RamReadCheckingOutputClaims,
};
use crate::ids::{DerivedId, ReadCheckingDerived};
use crate::points::{self, PointsError};
use jolt_claims::SymbolicSumcheck;
use jolt_field::JoltField;
use jolt_rv64i_arith::Layout;
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::VerifierError;
use std::sync::Arc;

/// Read checking over low-variable-first RAM addresses followed by cycles, at the shared batch-3a bit point.
/// The read claims and their bit-then-cycle points must come from batch 3b.
#[derive(Clone)]
pub struct RamReadChecking<F: JoltField> {
    symbolic: RamReadCheckingSymbolic,
    r_bit: Arc<Vec<F>>,
    r_3: Arc<Vec<F>>,
    a: usize,
}

impl<F: JoltField> RamReadChecking<F> {
    /// Establishes six bit coordinates and a nonempty cycle point bounded by `LOG_T_MAX`, returning `PointsError` on mismatch.
    /// The checked layout must admit at least five RAM address coordinates, which this constructor checks.
    pub fn new(layout: &Layout, r_bit: Vec<F>, r_3: Vec<F>) -> Result<Self, PointsError> {
        Self::new_shared(layout, Arc::new(r_bit), Arc::new(r_3))
    }

    /// Enforces `new`'s point bounds while sharing coordinates with the other batch-4 members.
    pub(crate) fn new_shared(
        layout: &Layout,
        r_bit: Arc<Vec<F>>,
        r_3: Arc<Vec<F>>,
    ) -> Result<Self, PointsError> {
        check_read_points(&r_bit, &r_3)?;
        let a = layout.log_K_ram();
        if a < 5 {
            return Err(PointsError::Dimension {
                expected: 5,
                actual: a,
            });
        }
        Ok(Self {
            symbolic: RamReadCheckingSymbolic::new((a, r_3.len())),
            r_bit,
            r_3,
            a,
        })
    }
    /// The six low-variable-first bit coordinates verified by batch 3a.
    pub fn r_bit(&self) -> &[F] {
        &self.r_bit
    }
    /// The low-variable-first cycle coordinates verified by batch 3b.
    pub fn r_3(&self) -> &[F] {
        &self.r_3
    }
    /// Consumed read points in six-bit-then-cycle order, low variable first in each part.
    pub fn input_points(&self) -> RamReadCheckingInputClaims<Vec<F>> {
        let point: Vec<_> = self.r_bit.iter().chain(self.r_3.iter()).copied().collect();
        RamReadCheckingInputClaims {
            ram_read_value: point,
        }
    }
}
impl<F: JoltField> ConcreteSumcheck<F> for RamReadChecking<F> {
    type Symbolic = RamReadCheckingSymbolic;
    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }
    fn derive_opening_points(
        &self,
        point: &[F],
        _inputs: &RamReadCheckingInputClaims<Vec<F>>,
    ) -> Result<RamReadCheckingOutputClaims<Vec<F>>, VerifierError> {
        if point.len() != self.rounds() {
            return Err(term_error(PointsError::Dimension {
                expected: self.rounds(),
                actual: point.len(),
            }));
        }
        let address = point.get(..self.a).ok_or_else(|| {
            term_error(PointsError::Dimension {
                expected: self.a,
                actual: point.len(),
            })
        })?;
        let cycle = point.get(self.a..).ok_or_else(|| {
            term_error(PointsError::Dimension {
                expected: self.a,
                actual: point.len(),
            })
        })?;
        Ok(RamReadCheckingOutputClaims {
            ram_ra: point.to_vec(),
            ram_val: address
                .iter()
                .chain(self.r_bit.iter())
                .chain(cycle)
                .copied()
                .collect(),
        })
    }
    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "foreign derived ids fail closed as missing claims"
    )]
    fn derive_output_term(
        &self,
        id: &DerivedId,
        _inputs: &RamReadCheckingInputClaims<Vec<F>>,
        outputs: &RamReadCheckingOutputClaims<Vec<F>>,
        _challenges: &RamReadCheckingChallenges<F>,
    ) -> Result<F, VerifierError> {
        match id {
            DerivedId::RamReadChecking(ReadCheckingDerived::EqCycle) => {
                let point = outputs.ram_ra.get(self.a..).ok_or_else(|| {
                    term_error(PointsError::Dimension {
                        expected: self.rounds(),
                        actual: outputs.ram_ra.len(),
                    })
                })?;
                points::eq(&self.r_3, point).map_err(term_error)
            }
            _ => Err(VerifierError::MissingStageClaimDerived { id: (*id).into() }),
        }
    }
    fn expected_output(
        &self,
        inputs: &RamReadCheckingInputClaims<Vec<F>>,
        values: &RamReadCheckingOutputClaims<F>,
        outputs: &RamReadCheckingOutputClaims<Vec<F>>,
        challenges: &RamReadCheckingChallenges<F>,
    ) -> Result<F, VerifierError> {
        evaluate_output(
            self.symbolic.output_expression(),
            values,
            challenges,
            |id| self.derive_output_term(id, inputs, outputs, challenges),
        )
    }
}

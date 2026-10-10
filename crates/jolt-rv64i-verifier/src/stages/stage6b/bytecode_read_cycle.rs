//! Concrete bytecode chunk-product relation at the terminal cycle point.
use crate::claims::bytecode_read::BytecodeReadCycleSymbolic;
pub use crate::claims::bytecode_read::{
    BytecodeReadCycleInputClaims, BytecodeReadCycleOutputClaims,
};
use crate::ids::{BytecodeCycleDerived, CycleWeight, DerivedId, OpeningId};
use crate::points::{self, PointsError};
use crate::proof::DimensionedRelation;
use jolt_claims::{NoChallenges, SymbolicSumcheck};
use jolt_field::JoltField;
use jolt_rv64i_arith::{Chunk, Layout};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::VerifierError;
use std::collections::BTreeSet;
use std::sync::Arc;

/// Bytecode chunk product reduced over low-variable-first cycles at the address point verified by batch 6a.
/// The five public folds come from that same batch-6a output.
#[derive(Clone)]
pub struct BytecodeReadCycle<F: JoltField> {
    symbolic: BytecodeReadCycleSymbolic,
    chunks: Vec<(Chunk, Vec<F>)>,
    a_bc: Vec<F>,
    h: [F; 5],
    r_3: Arc<Vec<F>>,
    r_4: Arc<Vec<F>>,
    r_5: Arc<Vec<F>>,
}
impl<F: JoltField> BytecodeReadCycle<F> {
    /// Checks the bytecode-address width against the layout and equal widths for batch-3b, batch-4 and batch-5 cycle points.
    /// Returns `PointsError` on mismatch; checked inputs establish the common trace-width bound.
    pub fn new(
        layout: &Layout,
        h: [F; 5],
        a_bc: Vec<F>,
        r_3: Vec<F>,
        r_4: Vec<F>,
        r_5: Vec<F>,
    ) -> Result<Self, PointsError> {
        Self::new_shared(layout, h, a_bc, Arc::new(r_3), Arc::new(r_4), Arc::new(r_5))
    }

    pub(crate) fn new_shared(
        layout: &Layout,
        h: [F; 5],
        a_bc: Vec<F>,
        r_3: Arc<Vec<F>>,
        r_4: Arc<Vec<F>>,
        r_5: Arc<Vec<F>>,
    ) -> Result<Self, PointsError> {
        let expected = layout.log_K_bytecode();
        if a_bc.len() != expected {
            return Err(PointsError::Dimension {
                expected,
                actual: a_bc.len(),
            });
        }
        if r_4.len() != r_5.len() {
            return Err(PointsError::Dimension {
                expected: r_4.len(),
                actual: r_5.len(),
            });
        }
        if r_3.len() != r_4.len() {
            return Err(PointsError::Dimension {
                expected: r_4.len(),
                actual: r_3.len(),
            });
        }
        let mut offset = 0;
        let mut chunks = Vec::new();
        for chunk in layout.bytecode_ra() {
            let end = offset + usize::from(chunk.bits());
            let point = a_bc.get(offset..end).ok_or(PointsError::Dimension {
                expected: end,
                actual: a_bc.len(),
            })?;
            chunks.push((*chunk, point.to_vec()));
            offset = end;
        }
        Ok(Self {
            symbolic: Self::symbolic_for(r_4.len(), layout),
            chunks,
            a_bc,
            h,
            r_3,
            r_4,
            r_5,
        })
    }
    /// The layout's bytecode chunks paired with low-variable-first slices of the verified batch-6a address point.
    pub fn chunks(&self) -> &[(Chunk, Vec<F>)] {
        &self.chunks
    }
    /// The verified low-variable-first batch-4 cycle point.
    pub fn r_4(&self) -> &[F] {
        &self.r_4
    }
    /// The verified low-variable-first batch-5 cycle point.
    pub fn r_5(&self) -> &[F] {
        &self.r_5
    }
    /// The verified low-variable-first batch-3b cycle point.
    pub fn r_3(&self) -> &[F] {
        &self.r_3
    }
    /// The consumed batch-6a address claim's low-variable-first bytecode-index point.
    pub fn input_points(&self) -> BytecodeReadCycleInputClaims<Vec<F>> {
        BytecodeReadCycleInputClaims {
            address_claim: self.a_bc.clone(),
        }
    }
    /// Projects the transmitted 256 columns onto this instance's fixed bytecode-chunk points.
    /// Returns `PointsError` for malformed column or chunk dimensions.
    pub fn project(&self, columns: &[F]) -> Result<BytecodeReadCycleOutputClaims<F>, PointsError> {
        Ok(BytecodeReadCycleOutputClaims {
            chunks: self
                .chunks
                .iter()
                .map(|(c, p)| points::chunk(*c, p, columns))
                .collect::<Result<_, _>>()?,
        })
    }
    /// Preserves a point failure as a typed rejection of the bytecode-cycle relation.
    pub fn term_error(error: PointsError) -> VerifierError {
        VerifierError::StageClaimSumcheckFailed {
            stage: "BytecodeReadCycle".to_owned(),
            reason: error.to_string(),
        }
    }
}
impl<F: JoltField> ConcreteSumcheck<F> for BytecodeReadCycle<F> {
    fn instance_point_offset(&self, batch_num_vars: usize) -> Result<usize, VerifierError> {
        Self::point_offset(self.rounds(), batch_num_vars)
    }

    type Symbolic = BytecodeReadCycleSymbolic;
    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }
    fn wire_output_openings(&self) -> BTreeSet<OpeningId> {
        BTreeSet::new()
    }
    fn derive_opening_points(
        &self,
        point: &[F],
        _inputs: &BytecodeReadCycleInputClaims<Vec<F>>,
    ) -> Result<BytecodeReadCycleOutputClaims<Vec<F>>, VerifierError> {
        if point.len() != self.rounds() {
            return Err(Self::term_error(PointsError::Dimension {
                expected: self.rounds(),
                actual: point.len(),
            }));
        }
        Ok(BytecodeReadCycleOutputClaims {
            chunks: self
                .chunks
                .iter()
                .map(|(_, p)| p.iter().chain(point).copied().collect())
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
        _inputs: &BytecodeReadCycleInputClaims<Vec<F>>,
        outputs: &BytecodeReadCycleOutputClaims<Vec<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let point = outputs
            .chunks
            .first()
            .and_then(|p| p.get(self.chunks.first().map_or(0, |(_, p)| p.len())..))
            .ok_or(VerifierError::MissingStageClaimDerived { id: (*id).into() })?;
        match id {
            DerivedId::BytecodeReadCycle(BytecodeCycleDerived::BytecodeFold(weight)) => {
                let [router, read, val, entry, next] = self.h;
                Ok(match weight {
                    CycleWeight::Router => router,
                    CycleWeight::Read => read,
                    CycleWeight::Val => val,
                    CycleWeight::Entry => entry,
                    CycleWeight::Next => next,
                })
            }
            DerivedId::BytecodeReadCycle(BytecodeCycleDerived::Weight(weight)) => {
                self.weight(*weight, point).map_err(Self::term_error)
            }
            _ => Err(VerifierError::MissingStageClaimDerived { id: (*id).into() }),
        }
    }
}
impl<F: JoltField> BytecodeReadCycle<F> {
    /// The Router, Read, Val, Entry and Next public folds computed at batch 6a's address point.
    pub fn folds(&self) -> [F; 5] {
        self.h
    }
    /// Evaluates the selected public cycle weight at a low-variable-first terminal cycle point.
    /// Returns `PointsError` unless its width matches the earlier cycle point.
    pub fn weight(&self, weight: CycleWeight, point: &[F]) -> Result<F, PointsError> {
        match weight {
            CycleWeight::Router => points::eq(&self.r_3, point),
            CycleWeight::Read => points::eq(&self.r_4, point),
            CycleWeight::Val => points::eq(&self.r_5, point),
            CycleWeight::Entry => {
                if point.len() != self.r_3.len() {
                    return Err(PointsError::Dimension {
                        expected: self.r_3.len(),
                        actual: point.len(),
                    });
                }
                points::eq_index(point, 0)
            }
            CycleWeight::Next => points::next(&self.r_3, point),
        }
    }
}

impl<F: JoltField> DimensionedRelation<F> for BytecodeReadCycle<F> {
    type Dimensions = (usize, usize);

    fn symbolic_with(dimensions: Self::Dimensions) -> Self::Symbolic {
        BytecodeReadCycleSymbolic::new(dimensions)
    }

    fn dimensions(log_T: usize, layout: &Layout) -> Self::Dimensions {
        (log_T, layout.bytecode_ra().len())
    }
}

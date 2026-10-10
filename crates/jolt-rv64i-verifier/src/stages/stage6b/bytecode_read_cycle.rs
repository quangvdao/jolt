//! Concrete bytecode chunk-product relation at the terminal cycle point.
use crate::claims::bytecode_read::BytecodeReadCycleSymbolic;
pub use crate::claims::bytecode_read::{
    BytecodeReadCycleInputClaims, BytecodeReadCycleOutputClaims,
};
use crate::ids::{BytecodeCycleDerived, CycleWeight, DerivedId, OpeningId};
use crate::points::{self, PointsError};
use jolt_claims::{NoChallenges, SymbolicSumcheck};
use jolt_field::JoltField;
use jolt_rv64i_arith::{Chunk, Layout};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::VerifierError;
use std::collections::BTreeSet;

#[derive(Clone)]
pub struct BytecodeReadCycle<F: JoltField> {
    symbolic: BytecodeReadCycleSymbolic,
    chunks: Vec<(Chunk, Vec<F>)>,
    a_bc: Vec<F>,
    h: [F; 5],
    r_3: Vec<F>,
    r_4: Vec<F>,
    r_5: Vec<F>,
}
impl<F: JoltField> BytecodeReadCycle<F> {
    pub fn new(
        layout: &Layout,
        h: [F; 5],
        a_bc: Vec<F>,
        r_3: Vec<F>,
        r_4: Vec<F>,
        r_5: Vec<F>,
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
            symbolic: BytecodeReadCycleSymbolic::new((r_4.len(), chunks.len())),
            chunks,
            a_bc,
            h,
            r_3,
            r_4,
            r_5,
        })
    }
    pub fn chunks(&self) -> &[(Chunk, Vec<F>)] {
        &self.chunks
    }
    pub fn r_4(&self) -> &[F] {
        &self.r_4
    }
    pub fn r_5(&self) -> &[F] {
        &self.r_5
    }
    pub fn r_3(&self) -> &[F] {
        &self.r_3
    }
    pub fn input_points(&self) -> BytecodeReadCycleInputClaims<Vec<F>> {
        BytecodeReadCycleInputClaims {
            address_claim: self.a_bc.clone(),
        }
    }
    pub fn project(&self, columns: &[F]) -> Result<BytecodeReadCycleOutputClaims<F>, PointsError> {
        Ok(BytecodeReadCycleOutputClaims {
            chunks: self
                .chunks
                .iter()
                .map(|(c, p)| points::chunk(*c, p, columns))
                .collect::<Result<_, _>>()?,
        })
    }
    pub fn term_error(error: PointsError) -> VerifierError {
        VerifierError::StageClaimSumcheckFailed {
            stage: "BytecodeReadCycle".to_owned(),
            reason: error.to_string(),
        }
    }
}
impl<F: JoltField> ConcreteSumcheck<F> for BytecodeReadCycle<F> {
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
    pub fn folds(&self) -> [F; 5] {
        self.h
    }
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
                Ok(point.iter().map(|v| F::one() + *v).product())
            }
            CycleWeight::Next => points::next(&self.r_3, point),
        }
    }
}

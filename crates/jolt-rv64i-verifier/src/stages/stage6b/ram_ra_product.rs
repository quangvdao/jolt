//! Concrete RAM chunk-product relation at the terminal cycle point.
use crate::claims::ram_ra_product::RamRaProductSymbolic;
pub use crate::claims::ram_ra_product::{
    RamRaProductChallenges, RamRaProductInputClaims, RamRaProductOutputClaims,
};
use crate::ids::{DerivedId, OpeningId, RamRaProductDerived};
use crate::points::{self, PointsError};
use jolt_claims::SymbolicSumcheck;
use jolt_field::JoltField;
use jolt_rv64i_arith::{Chunk, Layout};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::VerifierError;
use std::collections::BTreeSet;

#[derive(Clone)]
pub struct RamRaProduct<F: JoltField> {
    symbolic: RamRaProductSymbolic,
    chunks: Vec<(Chunk, Vec<F>)>,
    a_ram: Vec<F>,

    r_4: Vec<F>,
    r_5: Vec<F>,
}
impl<F: JoltField> RamRaProduct<F> {
    pub fn new(
        layout: &Layout,
        a_ram: Vec<F>,
        r_4: Vec<F>,
        r_5: Vec<F>,
    ) -> Result<Self, PointsError> {
        let expected = layout.log_K_ram();
        if a_ram.len() != expected {
            return Err(PointsError::Dimension {
                expected,
                actual: a_ram.len(),
            });
        }
        if r_4.len() != r_5.len() {
            return Err(PointsError::Dimension {
                expected: r_4.len(),
                actual: r_5.len(),
            });
        }

        let mut offset = 0;
        let mut chunks = Vec::new();
        for chunk in layout.ram_ra() {
            let end = offset + usize::from(chunk.bits());
            let point = a_ram.get(offset..end).ok_or(PointsError::Dimension {
                expected: end,
                actual: a_ram.len(),
            })?;
            chunks.push((*chunk, point.to_vec()));
            offset = end;
        }
        Ok(Self {
            symbolic: RamRaProductSymbolic::new((r_4.len(), chunks.len())),
            chunks,
            a_ram,
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

    pub fn input_points(&self) -> RamRaProductInputClaims<Vec<F>> {
        RamRaProductInputClaims {
            ram_ra_read: self.a_ram.iter().chain(&self.r_4).copied().collect(),
            ram_ra_val: self.a_ram.iter().chain(&self.r_5).copied().collect(),
        }
    }
    pub fn project(&self, columns: &[F]) -> Result<RamRaProductOutputClaims<F>, PointsError> {
        Ok(RamRaProductOutputClaims {
            chunks: self
                .chunks
                .iter()
                .map(|(c, p)| points::chunk(*c, p, columns))
                .collect::<Result<_, _>>()?,
        })
    }
    pub fn term_error(error: PointsError) -> VerifierError {
        VerifierError::StageClaimSumcheckFailed {
            stage: "RamRaProduct".to_owned(),
            reason: error.to_string(),
        }
    }
}
impl<F: JoltField> ConcreteSumcheck<F> for RamRaProduct<F> {
    type Symbolic = RamRaProductSymbolic;
    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }
    fn wire_output_openings(&self) -> BTreeSet<OpeningId> {
        BTreeSet::new()
    }
    fn derive_opening_points(
        &self,
        point: &[F],
        _inputs: &RamRaProductInputClaims<Vec<F>>,
    ) -> Result<RamRaProductOutputClaims<Vec<F>>, VerifierError> {
        if point.len() != self.rounds() {
            return Err(Self::term_error(PointsError::Dimension {
                expected: self.rounds(),
                actual: point.len(),
            }));
        }
        Ok(RamRaProductOutputClaims {
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
        _inputs: &RamRaProductInputClaims<Vec<F>>,
        outputs: &RamRaProductOutputClaims<Vec<F>>,
        _challenges: &RamRaProductChallenges<F>,
    ) -> Result<F, VerifierError> {
        let point = outputs
            .chunks
            .first()
            .and_then(|p| p.get(self.chunks.first().map_or(0, |(_, p)| p.len())..))
            .ok_or(VerifierError::MissingStageClaimDerived { id: (*id).into() })?;
        match id {
            DerivedId::RamRaProduct(RamRaProductDerived::EqRead) => {
                points::eq(&self.r_4, point).map_err(Self::term_error)
            }
            DerivedId::RamRaProduct(RamRaProductDerived::EqVal) => {
                points::eq(&self.r_5, point).map_err(Self::term_error)
            }
            _ => Err(VerifierError::MissingStageClaimDerived { id: (*id).into() }),
        }
    }
}

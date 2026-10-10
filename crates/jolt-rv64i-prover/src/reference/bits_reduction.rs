//! Dense test oracle for reduction of the six committed linear functionals.
use crate::plane::{Rv64iPlane, Rv64iWitness};
use jolt_field::JoltField;
use jolt_kernels::reference::naive::NaiveSumcheckProver;
use jolt_kernels::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};
use jolt_poly::{BindingOrder, Polynomial};
use jolt_rv64i_arith::BITS_COLUMNS;
use jolt_rv64i_verifier::ids::{
    BitsReductionDerived, CommittedPolynomial, DerivedId, OpeningId, RelationId,
};
use jolt_rv64i_verifier::stages::stage6b::bits_reduction::BitsReduction;
use std::collections::BTreeMap;

#[derive(Default)]
pub struct BitsReductionPrepare;
impl<F: JoltField> PrepareKernel<F, BitsReduction<F>, Rv64iPlane> for BitsReductionPrepare {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &Rv64iWitness,
        inputs: ProverInputs<'_, F, BitsReduction<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BitsReduction<F>>>, KernelError<F>> {
        let mut openings = BTreeMap::new();
        let mut derived = BTreeMap::new();
        for y in 0..BITS_COLUMNS {
            let bits = witness
                .bits
                .iter()
                .map(|r| {
                    F::from_u64(u64::from(
                        r.get(y / 64).is_some_and(|w| (w >> (y % 64)) & 1 != 0),
                    ))
                })
                .collect();
            let _ = openings.insert(
                OpeningId::committed(CommittedPolynomial::Column(y), RelationId::BitsReduction),
                Polynomial::new(bits),
            );
            let weights = (0..witness.bits.len())
                .map(|j| {
                    let point: Vec<_> = (0..inputs.relation.r_1().len())
                        .map(|i| F::from_u64(((j >> i) & 1) as u64))
                        .collect();
                    inputs
                        .relation
                        .column_weight(y, &point, inputs.challenges)
                        .map_err(|error| KernelError::InvalidGeometry {
                            reason: error.to_string(),
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let _ = derived.insert(
                DerivedId::BitsReduction(BitsReductionDerived::ColumnWeight(y)),
                Polynomial::new(weights),
            );
        }
        Ok(Box::new(NaiveSumcheckProver::new(
            &inputs,
            openings,
            derived,
            BindingOrder::LowToHigh,
        )?))
    }
}

//! Dense cycle tables feed the symbolic value-evaluation expressions.

use crate::error::Rv64iProverError;
use crate::plane::{Rv64iPlane, Rv64iWitness};
use crate::reference::views::{self, RegisterSelector};
use jolt_field::{Ring, F128};
use jolt_kernels::{
    reference::naive::NaiveSumcheckProver, KernelError, PrepareKernel, ProofSession, ProverInputs,
    SumcheckKernel,
};
use jolt_poly::{BindingOrder, Polynomial};
use jolt_rv64i_verifier::stages::stage5::val_evaluation::{
    RamValEvaluation, RegistersValEvaluation,
};
use jolt_rv64i_verifier::{
    ids::{
        CommittedPolynomial, DerivedId, OpeningId, RelationId, ValEvaluationDerived,
        VirtualPolynomial,
    },
    points::{self, PointsError},
};
use std::collections::BTreeMap;

fn less_than_table(point: &[F128], cycles: usize) -> Result<Vec<F128>, PointsError> {
    let mut vertex = vec![F128::from_u64(0); point.len()];
    let mut values = Vec::with_capacity(cycles);
    for cycle in 0..cycles {
        for (bit, coordinate) in vertex.iter_mut().enumerate() {
            *coordinate = F128::from_u64(((cycle >> bit) & 1) as u64);
        }
        values.push(points::lt(&vertex, point)?);
    }
    Ok(values)
}

#[derive(Default)]
pub struct RegistersValEvaluationPrepare;
impl PrepareKernel<F128, RegistersValEvaluation<F128>, Rv64iPlane>
    for RegistersValEvaluationPrepare
{
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &Rv64iWitness,
        inputs: ProverInputs<'_, F128, RegistersValEvaluation<F128>>,
    ) -> Result<
        Box<dyn SumcheckKernel<F128, Relation = RegistersValEvaluation<F128>>>,
        KernelError<F128>,
    > {
        let relation = inputs.relation;
        let geometry_error = |error: Rv64iProverError| KernelError::InvalidGeometry {
            reason: error.to_string(),
        };
        let openings = BTreeMap::from([
            (
                OpeningId::virtual_polynomial(
                    VirtualPolynomial::RdWa,
                    RelationId::RegistersValEvaluation,
                ),
                views::register_selector_at(witness, RegisterSelector::Rd, relation.a_reg())
                    .map_err(geometry_error)?,
            ),
            (
                OpeningId::virtual_polynomial(
                    VirtualPolynomial::Store,
                    RelationId::RegistersValEvaluation,
                ),
                views::store(witness).map_err(geometry_error)?,
            ),
            (
                OpeningId::committed(CommittedPolynomial::Inc, RelationId::RegistersValEvaluation),
                views::inc(witness, relation.r_bit()).map_err(geometry_error)?,
            ),
        ]);
        let weights = less_than_table(relation.r_4(), witness.bits.len()).map_err(|error| {
            KernelError::InvalidGeometry {
                reason: error.to_string(),
            }
        })?;
        let derived = BTreeMap::from([(
            DerivedId::RegistersValEvaluation(ValEvaluationDerived::Lt),
            Polynomial::new(weights),
        )]);
        Ok(Box::new(NaiveSumcheckProver::new(
            &inputs,
            openings,
            derived,
            BindingOrder::LowToHigh,
        )?))
    }
}
#[derive(Default)]
pub struct RamValEvaluationPrepare;
impl PrepareKernel<F128, RamValEvaluation<F128>, Rv64iPlane> for RamValEvaluationPrepare {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &Rv64iWitness,
        inputs: ProverInputs<'_, F128, RamValEvaluation<F128>>,
    ) -> Result<Box<dyn SumcheckKernel<F128, Relation = RamValEvaluation<F128>>>, KernelError<F128>>
    {
        let relation = inputs.relation;
        let geometry_error = |error: Rv64iProverError| KernelError::InvalidGeometry {
            reason: error.to_string(),
        };
        let openings = BTreeMap::from([
            (
                OpeningId::virtual_polynomial(
                    VirtualPolynomial::RamRa,
                    RelationId::RamValEvaluation,
                ),
                views::ram_ra_at(witness, relation.a_ram()).map_err(geometry_error)?,
            ),
            (
                OpeningId::virtual_polynomial(
                    VirtualPolynomial::Store,
                    RelationId::RamValEvaluation,
                ),
                views::store(witness).map_err(geometry_error)?,
            ),
            (
                OpeningId::committed(CommittedPolynomial::Inc, RelationId::RamValEvaluation),
                views::inc(witness, relation.r_bit()).map_err(geometry_error)?,
            ),
        ]);
        let weights = less_than_table(relation.r_4(), witness.bits.len()).map_err(|error| {
            KernelError::InvalidGeometry {
                reason: error.to_string(),
            }
        })?;
        let derived = BTreeMap::from([(
            DerivedId::RamValEvaluation(ValEvaluationDerived::Lt),
            Polynomial::new(weights),
        )]);
        Ok(Box::new(NaiveSumcheckProver::new(
            &inputs,
            openings,
            derived,
            BindingOrder::LowToHigh,
        )?))
    }
}

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

/// Prepares dense committed-column and canonical weight tables for low-to-high cycle binding.
/// Invalid weight geometry is reported by the kernel preparation interface.
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
        let mut point = vec![F::zero(); inputs.relation.r_1().len()];
        let mut bit_tables: [Vec<F>; BITS_COLUMNS] =
            std::array::from_fn(|_| Vec::with_capacity(witness.bits.len()));
        let mut weight_tables: [Vec<F>; BITS_COLUMNS] =
            std::array::from_fn(|_| Vec::with_capacity(witness.bits.len()));
        for (cycle, bits) in witness.bits.iter().enumerate() {
            for (bit, coordinate) in point.iter_mut().enumerate() {
                *coordinate = if (cycle >> bit) & 1 != 0 {
                    F::one()
                } else {
                    F::zero()
                };
            }
            let weights = inputs
                .relation
                .column_weights(&point, inputs.challenges)
                .map_err(|error| KernelError::InvalidGeometry {
                    reason: error.to_string(),
                })?;
            for (column, ((bit_table, weight_table), weight)) in bit_tables
                .iter_mut()
                .zip(&mut weight_tables)
                .zip(weights)
                .enumerate()
            {
                bit_table.push(
                    if bits
                        .get(column / 64)
                        .is_some_and(|word| (word >> (column % 64)) & 1 != 0)
                    {
                        F::one()
                    } else {
                        F::zero()
                    },
                );
                weight_table.push(weight);
            }
        }
        for (column, (bits, weights)) in bit_tables.into_iter().zip(weight_tables).enumerate() {
            let _ = openings.insert(
                OpeningId::committed(
                    CommittedPolynomial::Column(column),
                    RelationId::BitsReduction,
                ),
                Polynomial::new(bits),
            );
            let _ = derived.insert(
                DerivedId::BitsReduction(BitsReductionDerived::ColumnWeight(column)),
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

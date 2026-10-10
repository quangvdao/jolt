//! Dense test oracle for the RAM selector's committed digit product.
use crate::plane::{Rv64iPlane, Rv64iWitness};
use crate::reference::views;
use jolt_field::JoltField;
use jolt_kernels::reference::naive::NaiveSumcheckProver;
use jolt_kernels::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};
use jolt_poly::{BindingOrder, Polynomial};
use jolt_rv64i_verifier::ids::{
    DerivedId, OpeningId, RamRaProductDerived, RelationId, VirtualPolynomial,
};
use jolt_rv64i_verifier::points;
use jolt_rv64i_verifier::stages::stage6b::ram_ra_product::RamRaProduct;
use std::collections::BTreeMap;

#[derive(Default)]
pub struct RamRaProductPrepare;
impl<F: JoltField> PrepareKernel<F, RamRaProduct<F>, Rv64iPlane> for RamRaProductPrepare {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &Rv64iWitness,
        inputs: ProverInputs<'_, F, RamRaProduct<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamRaProduct<F>>>, KernelError<F>> {
        let mut openings = BTreeMap::new();
        for (c, (chunk, p)) in inputs.relation.chunks().iter().enumerate() {
            let table = views::chunk_in_field(witness, *chunk, p).map_err(|e| {
                KernelError::InvalidGeometry {
                    reason: e.to_string(),
                }
            })?;
            let _ = openings.insert(
                OpeningId::virtual_polynomial(
                    VirtualPolynomial::RamRaChunk(c),
                    RelationId::RamRaProduct,
                ),
                table,
            );
        }
        let mut derived = BTreeMap::new();
        for (id, p) in [
            (RamRaProductDerived::EqRead, inputs.relation.r_4()),
            (RamRaProductDerived::EqVal, inputs.relation.r_5()),
        ] {
            let table =
                points::equality_table(p).map_err(|error| KernelError::InvalidGeometry {
                    reason: error.to_string(),
                })?;
            let _ = derived.insert(DerivedId::RamRaProduct(id), Polynomial::new(table));
        }
        Ok(Box::new(NaiveSumcheckProver::new(
            &inputs,
            openings,
            derived,
            BindingOrder::LowToHigh,
        )?))
    }
}

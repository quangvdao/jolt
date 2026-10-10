//! Dense test oracle for the RAM selector's committed digit product.
use crate::plane::{Rv64iPlane, Rv64iWitness};
use crate::reference::views;
use jolt_field::{CanonicalEncoding, JoltField, F128};
use jolt_kernels::reference::naive::NaiveSumcheckProver;
use jolt_kernels::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};
use jolt_poly::{BindingOrder, Polynomial};
use jolt_rv64i_verifier::ids::{
    DerivedId, OpeningId, RamRaProductDerived, RelationId, VirtualPolynomial,
};
use jolt_rv64i_verifier::points;
use jolt_rv64i_verifier::stages::stage6b::ram_ra_product::RamRaProduct;
use std::collections::BTreeMap;

pub(crate) fn binary_point<F: JoltField>(point: &[F]) -> Result<Vec<F128>, KernelError<F>> {
    point
        .iter()
        .map(|v| {
            let mut bytes = [0; 16];
            v.to_bytes_le(&mut bytes);
            F128::from_bytes_le_checked(&bytes).ok_or_else(|| KernelError::InvalidGeometry {
                reason: "point is not a binary-field scalar".to_owned(),
            })
        })
        .collect()
}
pub(crate) fn binary_table<F: JoltField>(
    table: Polynomial<F128>,
) -> Result<Polynomial<F>, KernelError<F>> {
    Ok(Polynomial::new(
        table
            .evals()
            .iter()
            .map(|v| {
                F::from_u128_checked(v.to_raw()).ok_or_else(|| KernelError::InvalidGeometry {
                    reason: "table value is not representable by the protocol field".to_owned(),
                })
            })
            .collect::<Result<_, _>>()?,
    ))
}
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
            let table = views::chunk(witness, *chunk, &binary_point(p)?).map_err(|e| {
                KernelError::InvalidGeometry {
                    reason: e.to_string(),
                }
            })?;
            let _ = openings.insert(
                OpeningId::virtual_polynomial(
                    VirtualPolynomial::RamRaChunk(c),
                    RelationId::RamRaProduct,
                ),
                binary_table(table)?,
            );
        }
        let mut derived = BTreeMap::new();
        for (id, p) in [
            (RamRaProductDerived::EqRead, inputs.relation.r_4()),
            (RamRaProductDerived::EqVal, inputs.relation.r_5()),
        ] {
            let table = (0..witness.bits.len())
                .map(|j| {
                    points::eq_index(p, j).map_err(|e| KernelError::InvalidGeometry {
                        reason: e.to_string(),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
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

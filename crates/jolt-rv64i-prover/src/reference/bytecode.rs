//! Dense bytecode address and cycle kernels for batch-local protocol tests.
use crate::plane::{Rv64iPlane, Rv64iWitness};
use crate::reference::views;
use jolt_field::JoltField;
use jolt_kernels::reference::naive::NaiveSumcheckProver;
use jolt_kernels::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};
use jolt_poly::{BindingOrder, Polynomial, UnivariatePoly};
use jolt_rv64i_arith::Bytecode;
use jolt_rv64i_verifier::ids::{
    BytecodeCycleDerived, CycleWeight, DerivedId, OpeningId, RelationId, VirtualPolynomial,
};
use jolt_rv64i_verifier::points::{self, PointsError};
use jolt_rv64i_verifier::stages::stage6a::{
    BytecodeReadAddress, BytecodeReadAddressChallenges, BytecodeReadAddressInputClaims,
    BytecodeReadAddressOutputClaims,
};
use jolt_rv64i_verifier::stages::stage6b::BytecodeReadCycle;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use std::collections::BTreeMap;

#[derive(Default)]
pub struct BytecodeReadAddressPrepare;
#[derive(Default)]
pub struct BytecodeReadCyclePrepare;

impl BytecodeReadAddressPrepare {
    /// Builds the five public H tables using the relation's canonical per-row weights.
    /// Weight preparation and all output allocations are included in this operation.
    pub fn public_tables<F: JoltField>(
        &self,
        bytecode: &Bytecode,
        relation: &BytecodeReadAddress<F>,
        challenges: &BytecodeReadAddressChallenges<F>,
    ) -> Result<[Vec<F>; 5], KernelError<F>> {
        let weights =
            relation
                .public_weights(challenges)
                .map_err(|e| KernelError::InvalidGeometry {
                    reason: e.to_string(),
                })?;
        let count = bytecode.rows().len();
        let mut h: [Vec<F>; 5] = std::array::from_fn(|_| Vec::with_capacity(count));
        for row in bytecode.rows() {
            let values = weights.row(row).map_err(|e| KernelError::InvalidGeometry {
                reason: e.to_string(),
            })?;
            for (table, value) in h.iter_mut().zip(values) {
                table.push(value);
            }
        }
        Ok(h)
    }
}

#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
struct AddressKernel<F: JoltField> {
    pairs: [(Polynomial<F>, Polynomial<F>); 5],
    rounds: usize,
    bound: usize,
}
impl<F: JoltField> AddressKernel<F> {
    fn bind(&mut self, value: F) {
        for (h, r) in &mut self.pairs {
            h.bind_with_order(value, BindingOrder::LowToHigh);
            r.bind_with_order(value, BindingOrder::LowToHigh);
        }
        self.bound += 1;
    }
}
impl<F: JoltField> ProveRounds<F> for AddressKernel<F> {
    fn num_rounds(&self) -> usize {
        self.rounds
    }
    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(value) = bind {
            self.bind(value);
        }
        let mut at_zero = F::zero();
        let mut at_one = F::zero();
        let mut quadratic = F::zero();
        for (h, r) in &self.pairs {
            for (hs, rs) in h.evals().chunks_exact(2).zip(r.evals().chunks_exact(2)) {
                if let ([h0, h1], [r0, r1]) = (hs, rs) {
                    at_zero += *h0 * *r0;
                    at_one += *h1 * *r1;
                    quadratic += (*h1 - *h0) * (*r1 - *r0);
                }
            }
        }
        let actual = at_zero + at_one;
        if actual != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual,
            });
        }
        Ok(UnivariatePoly::new(vec![
            at_zero,
            at_one - at_zero - quadratic,
            quadratic,
        ]))
    }
    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}
impl<F: JoltField> SumcheckKernel<F> for AddressKernel<F> {
    type Relation = BytecodeReadAddress<F>;
    fn output_claims(
        &mut self,
        _inputs: &BytecodeReadAddressInputClaims<F>,
    ) -> Result<BytecodeReadAddressOutputClaims<F>, SumcheckKernelError<F>> {
        if self.bound != self.rounds {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.rounds.saturating_sub(self.bound),
            });
        }
        let address_claim = self.pairs.iter().try_fold(F::zero(), |sum, (h, r)| {
            let h = h
                .evals()
                .first()
                .ok_or(SumcheckKernelError::InvariantViolation {
                    reason: "bound bytecode table is empty",
                })?;
            let r = r
                .evals()
                .first()
                .ok_or(SumcheckKernelError::InvariantViolation {
                    reason: "bound address table is empty",
                })?;
            Ok::<_, SumcheckKernelError<F>>(sum + *h * *r)
        })?;
        Ok(BytecodeReadAddressOutputClaims { address_claim })
    }
}
impl<F: JoltField> PrepareKernel<F, BytecodeReadAddress<F>, Rv64iPlane>
    for BytecodeReadAddressPrepare
{
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &Rv64iWitness,
        inputs: ProverInputs<'_, F, BytecodeReadAddress<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BytecodeReadAddress<F>>>, KernelError<F>> {
        let relation = inputs.relation;
        let count = witness.bytecode.rows().len();
        let h = self.public_tables(&witness.bytecode, relation, inputs.challenges)?;
        let mut r: [Vec<F>; 5] = std::array::from_fn(|_| vec![F::zero(); count]);
        let p = relation.public_points();
        let cycle_weights = [&p.r_3, &p.r_4, &p.r_5].map(|point| {
            points::equality_table(point).map_err(|error| KernelError::InvalidGeometry {
                reason: error.to_string(),
            })
        });
        let [r3, r4, r5] = cycle_weights;
        let [r3, r4, r5] = [r3?, r4?, r5?];
        if [&r3, &r4, &r5]
            .iter()
            .any(|weights| weights.len() != witness.bits.len())
        {
            return Err(KernelError::InvalidGeometry {
                reason: "cycle point does not match the witness length".to_owned(),
            });
        }
        for (j, row) in witness.bits.iter().enumerate() {
            let index = witness.layout.bytecode_index(row) as usize;
            let values = [
                r3.get(j).copied().ok_or(PointsError::Index {
                    index: j,
                    variables: p.r_3.len(),
                }),
                r4.get(j).copied().ok_or(PointsError::Index {
                    index: j,
                    variables: p.r_4.len(),
                }),
                r5.get(j).copied().ok_or(PointsError::Index {
                    index: j,
                    variables: p.r_5.len(),
                }),
                Ok(F::from_u64(u64::from(j == 0))),
                points::shifted_next_weight(&r3, j).ok_or(PointsError::Index {
                    index: j,
                    variables: p.r_3.len(),
                }),
            ];
            for (table, value) in r.iter_mut().zip(values) {
                *table
                    .get_mut(index)
                    .ok_or_else(|| KernelError::InvalidGeometry {
                        reason: "bytecode index exceeds public domain".to_owned(),
                    })? += value.map_err(|e| KernelError::InvalidGeometry {
                    reason: e.to_string(),
                })?;
            }
        }
        let pairs: [(Polynomial<F>, Polynomial<F>); 5] = h
            .into_iter()
            .zip(r)
            .map(|(h, r)| (Polynomial::new(h), Polynomial::new(r)))
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| KernelError::InvalidGeometry {
                reason: "bytecode fold requires five paired tables".to_owned(),
            })?;
        Ok(Box::new(AddressKernel {
            pairs,
            rounds: relation.rounds(),
            bound: 0,
        }))
    }
}
impl<F: JoltField> PrepareKernel<F, BytecodeReadCycle<F>, Rv64iPlane> for BytecodeReadCyclePrepare {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &Rv64iWitness,
        inputs: ProverInputs<'_, F, BytecodeReadCycle<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BytecodeReadCycle<F>>>, KernelError<F>> {
        let mut openings = BTreeMap::new();
        for (c, (chunk, p)) in inputs.relation.chunks().iter().enumerate() {
            let table = views::chunk_in_field(witness, *chunk, p).map_err(|e| {
                KernelError::InvalidGeometry {
                    reason: e.to_string(),
                }
            })?;
            let _ = openings.insert(
                OpeningId::virtual_polynomial(
                    VirtualPolynomial::BytecodeRaChunk(c),
                    RelationId::BytecodeReadCycle,
                ),
                table,
            );
        }
        let router = points::equality_table(inputs.relation.r_3()).map_err(|error| {
            KernelError::InvalidGeometry {
                reason: error.to_string(),
            }
        })?;
        let next = (0..router.len())
            .map(|index| {
                points::shifted_next_weight(&router, index).ok_or(PointsError::Index {
                    index,
                    variables: inputs.relation.r_3().len(),
                })
            })
            .collect::<Result<Vec<_>, _>>();
        let mut entry = vec![F::zero(); witness.bits.len()];
        if let Some(first) = entry.first_mut() {
            *first = F::one();
        }
        let tables = [
            Ok(router),
            points::equality_table(inputs.relation.r_4()),
            points::equality_table(inputs.relation.r_5()),
            Ok(entry),
            next,
        ];
        let mut derived = BTreeMap::new();
        for ((weight, h), table) in [
            CycleWeight::Router,
            CycleWeight::Read,
            CycleWeight::Val,
            CycleWeight::Entry,
            CycleWeight::Next,
        ]
        .into_iter()
        .zip(inputs.relation.folds())
        .zip(tables)
        {
            let _ = derived.insert(
                DerivedId::BytecodeReadCycle(BytecodeCycleDerived::BytecodeFold(weight)),
                Polynomial::new(vec![h; witness.bits.len()]),
            );
            let table = table.map_err(|error| KernelError::InvalidGeometry {
                reason: error.to_string(),
            })?;
            let _ = derived.insert(
                DerivedId::BytecodeReadCycle(BytecodeCycleDerived::Weight(weight)),
                Polynomial::new(table),
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

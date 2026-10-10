//! Tail adapters for the packed RV64I kernels.

use crate::optimized::source::SharedSource;
use crate::plane::{Rv64iPlane, Rv64iWitness};
#[cfg(feature = "allocative")]
use allocative::Allocative;
use jolt_claims::NoChallenges;
use jolt_field::F128;
use jolt_kernels::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};
use jolt_poly::UnivariatePoly;
use jolt_rv64i_arith::{BitsRow, BITS_COLUMNS};
use jolt_rv64i_kernels::chunk_product::{
    combined_weight, ChunkProductCore, ChunkWeight, ChunkWeightTerm,
};
use jolt_rv64i_kernels::column_pass::column_pass;
use jolt_rv64i_kernels::reduction::{g_pass_digits, ReductionCore, ReductionLeg};
use jolt_rv64i_verifier::ids::{BytecodeCycleDerived, CycleWeight, DerivedId, RamRaProductDerived};
use jolt_rv64i_verifier::stages::stage6b::bits_reduction::{
    BitsReductionInputClaims, BitsReductionOutputClaims,
};
use jolt_rv64i_verifier::stages::stage6b::bytecode_read_cycle::{
    BytecodeReadCycleInputClaims, BytecodeReadCycleOutputClaims,
};
use jolt_rv64i_verifier::stages::stage6b::ram_ra_product::{
    RamRaProductChallenges, RamRaProductInputClaims, RamRaProductOutputClaims,
};
use jolt_rv64i_verifier::stages::stage6b::{BitsReduction, BytecodeReadCycle, RamRaProduct};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use std::fmt::Display;
use std::sync::Arc;

fn geometry(error: impl Display) -> KernelError<F128> {
    KernelError::InvalidGeometry {
        reason: error.to_string(),
    }
}

fn chunk_values(core: &ChunkProductCore) -> Result<(F128, Vec<F128>), SumcheckKernelError<F128>> {
    core.final_values()
        .map_err(|_| SumcheckKernelError::InvariantViolation {
            reason: "chunk product values requested before the final bind",
        })
}

/// Consumes the session's bytecode group; a second preparation is rejected.
#[derive(Default)]
pub struct BytecodeReadCyclePrepare;

#[cfg_attr(feature = "allocative", derive(Allocative))]
struct BytecodeKernel {
    #[cfg_attr(feature = "allocative", allocative(skip))]
    core: ChunkProductCore,
}

impl PrepareKernel<F128, BytecodeReadCycle<F128>, Rv64iPlane> for BytecodeReadCyclePrepare {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &Rv64iWitness,
        inputs: ProverInputs<'_, F128, BytecodeReadCycle<F128>>,
    ) -> Result<Box<dyn SumcheckKernel<F128, Relation = BytecodeReadCycle<F128>>>, KernelError<F128>>
    {
        let columns = inputs.relation.chunks().len();
        if !(1..=7).contains(&columns) {
            return Err(geometry(format!(
                "bytecode chunk count {columns}; expected 1..=7"
            )));
        }
        let shared = session.state_or_insert_with(SharedSource::default);
        let _ = shared.prepare(witness, None)?;
        let group = shared.take_bytecode_group()?;
        let log_t = group.cycles().ilog2() as usize;
        let [router, read, val, entry, next] = inputs.relation.folds();
        let terms = [
            ChunkWeightTerm::Eq {
                coefficient: router,
                point: inputs.relation.r_3().to_vec(),
            },
            ChunkWeightTerm::Eq {
                coefficient: read,
                point: inputs.relation.r_4().to_vec(),
            },
            ChunkWeightTerm::Eq {
                coefficient: val,
                point: inputs.relation.r_5().to_vec(),
            },
            ChunkWeightTerm::Eq {
                coefficient: entry,
                point: vec![F128::from_raw(0); log_t],
            },
            ChunkWeightTerm::Next {
                coefficient: next,
                point: inputs.relation.r_3().to_vec(),
            },
        ];
        let weight = ChunkWeight::Dense(combined_weight(log_t, &terms).map_err(geometry)?);
        let points = inputs
            .relation
            .chunks()
            .iter()
            .map(|(_, point)| point.clone())
            .collect();
        let core = ChunkProductCore::new(group, points, weight).map_err(geometry)?;
        Ok(Box::new(BytecodeKernel { core }))
    }
}

impl ProveRounds<F128> for BytecodeKernel {
    fn num_rounds(&self) -> usize {
        self.core.num_rounds()
    }
    fn prove_round(
        &mut self,
        bind: Option<F128>,
        round: usize,
        previous_claim: F128,
    ) -> Result<UnivariatePoly<F128>, SumcheckError<F128>> {
        self.core.prove_round(bind, round, previous_claim)
    }
    fn finish_rounds(&mut self, bind: F128) -> Result<(), SumcheckError<F128>> {
        self.core.finish_rounds(bind)
    }
}

impl SumcheckKernel<F128> for BytecodeKernel {
    type Relation = BytecodeReadCycle<F128>;
    fn output_claims(
        &mut self,
        _inputs: &BytecodeReadCycleInputClaims<F128>,
    ) -> Result<BytecodeReadCycleOutputClaims<F128>, SumcheckKernelError<F128>> {
        let (_, chunks) = chunk_values(&self.core)?;
        Ok(BytecodeReadCycleOutputClaims { chunks })
    }
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &BytecodeReadCycleInputClaims<Vec<F128>>,
        output_points: &BytecodeReadCycleOutputClaims<Vec<F128>>,
        challenges: &NoChallenges<F128>,
    ) -> Result<(), SumcheckKernelError<F128>> {
        let mut expected = F128::from_raw(0);
        for (weight, fold) in [
            CycleWeight::Router,
            CycleWeight::Read,
            CycleWeight::Val,
            CycleWeight::Entry,
            CycleWeight::Next,
        ]
        .into_iter()
        .zip(relation.folds())
        {
            expected += fold
                * relation.derive_output_term(
                    &DerivedId::BytecodeReadCycle(BytecodeCycleDerived::Weight(weight)),
                    input_points,
                    output_points,
                    challenges,
                )?;
        }
        let (actual, _) = chunk_values(&self.core)?;
        if actual != expected {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "bytecode combined weight disagrees with the relation's derived terms",
            });
        }
        Ok(())
    }
}

/// Consumes the session's RAM group; a second preparation is rejected.
#[derive(Default)]
pub struct RamRaProductPrepare;

#[cfg_attr(feature = "allocative", derive(Allocative))]
struct RamKernel {
    #[cfg_attr(feature = "allocative", allocative(skip))]
    core: ChunkProductCore,
}

impl PrepareKernel<F128, RamRaProduct<F128>, Rv64iPlane> for RamRaProductPrepare {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &Rv64iWitness,
        inputs: ProverInputs<'_, F128, RamRaProduct<F128>>,
    ) -> Result<Box<dyn SumcheckKernel<F128, Relation = RamRaProduct<F128>>>, KernelError<F128>>
    {
        let columns = inputs.relation.chunks().len();
        if !(1..=7).contains(&columns) {
            return Err(geometry(format!(
                "RAM chunk count {columns}; expected 1..=7"
            )));
        }
        let shared = session.state_or_insert_with(SharedSource::default);
        let _ = shared.prepare(witness, None)?;
        let group = shared.take_ram_group()?;
        let log_t = group.cycles().ilog2() as usize;
        let terms = [
            ChunkWeightTerm::Eq {
                coefficient: inputs.challenges.read,
                point: inputs.relation.r_4().to_vec(),
            },
            ChunkWeightTerm::Eq {
                coefficient: inputs.challenges.val,
                point: inputs.relation.r_5().to_vec(),
            },
        ];
        let weight = ChunkWeight::Dense(combined_weight(log_t, &terms).map_err(geometry)?);
        let points = inputs
            .relation
            .chunks()
            .iter()
            .map(|(_, point)| point.clone())
            .collect();
        let core = ChunkProductCore::new(group, points, weight).map_err(geometry)?;
        Ok(Box::new(RamKernel { core }))
    }
}

impl ProveRounds<F128> for RamKernel {
    fn num_rounds(&self) -> usize {
        self.core.num_rounds()
    }
    fn prove_round(
        &mut self,
        bind: Option<F128>,
        round: usize,
        previous_claim: F128,
    ) -> Result<UnivariatePoly<F128>, SumcheckError<F128>> {
        self.core.prove_round(bind, round, previous_claim)
    }
    fn finish_rounds(&mut self, bind: F128) -> Result<(), SumcheckError<F128>> {
        self.core.finish_rounds(bind)
    }
}

impl SumcheckKernel<F128> for RamKernel {
    type Relation = RamRaProduct<F128>;
    fn output_claims(
        &mut self,
        _inputs: &RamRaProductInputClaims<F128>,
    ) -> Result<RamRaProductOutputClaims<F128>, SumcheckKernelError<F128>> {
        let (_, chunks) = chunk_values(&self.core)?;
        Ok(RamRaProductOutputClaims { chunks })
    }
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &RamRaProductInputClaims<Vec<F128>>,
        output_points: &RamRaProductOutputClaims<Vec<F128>>,
        challenges: &RamRaProductChallenges<F128>,
    ) -> Result<(), SumcheckKernelError<F128>> {
        let mut expected = F128::from_raw(0);
        for (term, coefficient) in [
            (RamRaProductDerived::EqRead, challenges.read),
            (RamRaProductDerived::EqVal, challenges.val),
        ] {
            expected += coefficient
                * relation.derive_output_term(
                    &DerivedId::RamRaProduct(term),
                    input_points,
                    output_points,
                    challenges,
                )?;
        }
        let (actual, _) = chunk_values(&self.core)?;
        if actual != expected {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "RAM combined weight disagrees with the relation's derived terms",
            });
        }
        Ok(())
    }
}

/// Builds the three grouped reduction tables; returns committed columns at the bound cycle point.
#[derive(Default)]
pub struct BitsReductionPrepare;

#[cfg_attr(feature = "allocative", derive(Allocative))]
struct BitsKernel {
    #[cfg_attr(feature = "allocative", allocative(skip))]
    core: ReductionCore,
    bits: Arc<[BitsRow]>,
    point: Vec<F128>,
    initial_claim: F128,
}

impl PrepareKernel<F128, BitsReduction<F128>, Rv64iPlane> for BitsReductionPrepare {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &Rv64iWitness,
        inputs: ProverInputs<'_, F128, BitsReduction<F128>>,
    ) -> Result<Box<dyn SumcheckKernel<F128, Relation = BitsReduction<F128>>>, KernelError<F128>>
    {
        let shared = session.state_or_insert_with(SharedSource::default);
        let trace = shared.prepare(witness, None)?;
        let c = inputs.challenges;
        let mut weights: [Vec<F128>; 3] =
            std::array::from_fn(|_| vec![F128::from_raw(0); BITS_COLUMNS]);
        for (support, (leg, coefficient)) in inputs.relation.weights().iter().zip([
            (0, c.direct_columns),
            (1, c.variant_bits),
            (1, c.pos_ra_0),
            (1, c.pos_ra_1),
            (1, c.should_branch),
            (2, c.inc),
        ]) {
            for &(column, weight) in support {
                let value = weights[leg].get_mut(column).ok_or_else(|| {
                    geometry(format!("reduction weight column {column} exceeds 255"))
                })?;
                *value += coefficient * weight;
            }
        }
        let tables = g_pass_digits(&trace, trace.source().columns().column_map(), &weights)
            .map_err(geometry)?;
        let [z_0, z_1] = inputs.relation.pos_zero();
        let claims = [
            c.direct_columns * inputs.claims.direct_columns,
            c.variant_bits * inputs.claims.variant_bits
                + c.pos_ra_0 * (inputs.claims.pos_ra_0 + z_0)
                + c.pos_ra_1 * (inputs.claims.pos_ra_1 + z_1)
                + c.should_branch * inputs.claims.should_branch,
            c.inc * inputs.claims.inc,
        ];
        let legs = [
            inputs.relation.r_1(),
            inputs.relation.r_3(),
            inputs.relation.r_5(),
        ]
        .into_iter()
        .zip(claims)
        .enumerate()
        .map(|(table, (point, claim))| ReductionLeg {
            table,
            point: point.to_vec(),
            coefficient: F128::from_raw(1),
            claim,
        })
        .collect();
        let core = ReductionCore::new(tables, legs).map_err(geometry)?;
        let point = Vec::with_capacity(core.num_rounds());
        Ok(Box::new(BitsKernel {
            core,
            bits: Arc::clone(&witness.bits),
            point,
            initial_claim: claims.into_iter().sum(),
        }))
    }
}

impl ProveRounds<F128> for BitsKernel {
    fn num_rounds(&self) -> usize {
        self.core.num_rounds()
    }
    fn prove_round(
        &mut self,
        bind: Option<F128>,
        round: usize,
        previous_claim: F128,
    ) -> Result<UnivariatePoly<F128>, SumcheckError<F128>> {
        if round == 0 && previous_claim != self.initial_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual: self.initial_claim,
            });
        }
        let message = self.core.prove_round(bind, round, previous_claim)?;
        if let Some(bind) = bind {
            self.point.push(bind);
        }
        Ok(message)
    }
    fn finish_rounds(&mut self, bind: F128) -> Result<(), SumcheckError<F128>> {
        self.core.finish_rounds(bind)?;
        self.point.push(bind);
        Ok(())
    }
}

impl SumcheckKernel<F128> for BitsKernel {
    type Relation = BitsReduction<F128>;
    fn output_claims(
        &mut self,
        _inputs: &BitsReductionInputClaims<F128>,
    ) -> Result<BitsReductionOutputClaims<F128>, SumcheckKernelError<F128>> {
        let remaining = self.core.num_rounds() - self.point.len();
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let columns = column_pass(&self.bits, &self.point)
            .map_err(|_| SumcheckKernelError::InvariantViolation {
                reason: "bound reduction point does not match its committed bits",
            })?
            .to_vec();
        Ok(BitsReductionOutputClaims { columns })
    }
}

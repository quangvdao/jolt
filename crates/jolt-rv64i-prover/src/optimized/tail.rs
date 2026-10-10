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
use jolt_rv64i_kernels::chunk_product::{
    combined_weight, ChunkProductCore, ChunkWeight, ChunkWeightTerm,
};
use jolt_rv64i_verifier::ids::{BytecodeCycleDerived, CycleWeight, DerivedId, RamRaProductDerived};
use jolt_rv64i_verifier::stages::stage6b::bytecode_read_cycle::{
    BytecodeReadCycleInputClaims, BytecodeReadCycleOutputClaims,
};
use jolt_rv64i_verifier::stages::stage6b::ram_ra_product::{
    RamRaProductChallenges, RamRaProductInputClaims, RamRaProductOutputClaims,
};
use jolt_rv64i_verifier::stages::stage6b::{BytecodeReadCycle, RamRaProduct};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use std::fmt::Display;

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
            return Err(geometry(format!("bytecode chunk count {columns}; expected 1..=7")));
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
    ) -> Result<Box<dyn SumcheckKernel<F128, Relation = RamRaProduct<F128>>>, KernelError<F128>> {
        let columns = inputs.relation.chunks().len();
        if !(1..=7).contains(&columns) {
            return Err(geometry(format!("RAM chunk count {columns}; expected 1..=7")));
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

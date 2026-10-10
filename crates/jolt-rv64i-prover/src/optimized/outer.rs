//! Packed lanes and the degree-three binary-row outer adapter.

use crate::error::Rv64iProverError;
use crate::plane::{Rv64iPlane, Rv64iWitness, WitnessCycles};
#[cfg(feature = "allocative")]
use allocative::Allocative;
use jolt_field::F128;
use jolt_kernels::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};
use jolt_poly::UnivariatePoly;
use jolt_rv64i_arith::words::F2Words;
use jolt_rv64i_arith::{FormError, RowSystem, Sources};
use jolt_rv64i_kernels::outer_f2::{OuterF2Core, OuterF2Options};
use jolt_rv64i_kernels::par::{CycleChunks, ParError};
use jolt_rv64i_kernels::source::LaneSource;
use jolt_rv64i_verifier::claims::spartan_outer::{
    SpartanOuterF2OutputClaims, SpartanOuterInputClaims,
};
use jolt_rv64i_verifier::stages::stage1::SpartanOuterF2;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use rayon::prelude::*;
use std::fmt::Display;
use std::sync::Arc;
use thiserror::Error;

/// Geometry or per-cycle evaluation rejected by the lanes pass.
#[derive(Debug, Error)]
pub enum LanesError {
    #[error(transparent)]
    Witness(#[from] Rv64iProverError),
    #[error("cycle count {cycles} is not a nonempty power of two")]
    Cycles { cycles: usize },
    #[error(transparent)]
    Geometry(#[from] ParError),
    #[error("cycle {cycle} evaluation failed: {source}")]
    Evaluation { cycle: usize, source: FormError },
}

/// Owned lane triples and six tail bits per cycle. Construction evaluates the
/// decoded witness once, in parallel; no committed row is read or retained.
#[cfg_attr(feature = "allocative", derive(Allocative))]
pub struct WitnessLanes {
    lanes: Vec<[[u64; 3]; 2]>,
    tail: Vec<u8>,
}

impl WitnessLanes {
    /// Writes both lane families and the tail into their final buffers. Errors
    /// identify the lowest offending cycle independently of the Rayon pool.
    pub fn new(witness: &Rv64iWitness) -> Result<Self, LanesError> {
        let cycles = witness.bits.len();
        if !cycles.is_power_of_two() {
            return Err(LanesError::Cycles { cycles });
        }
        if witness.words.len() != cycles {
            return Err(Rv64iProverError::RowCount {
                expected: cycles,
                bits: cycles,
                words: witness.words.len(),
            }
            .into());
        }
        if witness.decoded.len() != cycles {
            return Err(Rv64iProverError::DecodedLength {
                expected: cycles,
                found: witness.decoded.len(),
            }
            .into());
        }
        let chunks = CycleChunks::new(cycles.ilog2() as usize, 0)?;
        let rows = RowSystem::new(&witness.layout);
        let view = witness.cycles();
        let mut lanes = vec![[[0; 3]; 2]; cycles];
        let mut tail = vec![0; cycles];
        let error = lanes
            .par_chunks_mut(chunks.chunk_len())
            .zip(tail.par_chunks_mut(chunks.chunk_len()))
            .enumerate()
            .filter_map(|(chunk, (lanes, tail))| {
                for (offset, (lanes, tail)) in lanes.iter_mut().zip(tail).enumerate() {
                    let cycle = chunk * chunks.chunk_len() + offset;
                    match Self::cycle(&view, &rows, cycle) {
                        Ok((values, byte)) => {
                            *lanes = values;
                            *tail = byte;
                        }
                        Err(error) => return Some((cycle, error)),
                    }
                }
                None
            })
            .min_by_key(|(cycle, _)| *cycle);
        if let Some((_, error)) = error {
            return Err(error);
        }
        Ok(Self { lanes, tail })
    }

    /// Returns one cycle's lane families and packed tail without storing them.
    #[inline]
    pub fn cycle(
        view: &WitnessCycles<'_>,
        rows: &RowSystem,
        cycle: usize,
    ) -> Result<([[u64; 3]; 2], u8), LanesError> {
        let parts = view.parts(cycle)?;
        let sources = Sources::from_parts(parts.fetched, &parts.base, parts.sources);
        let evaluated = F2Words::compute(parts.fetched, &sources, parts.sources.pos)
            .map_err(|source| LanesError::Evaluation { cycle, source })?;
        let (lanes, packed) = rows.f2_values(&evaluated, parts.sources.keys_differ);
        let mut tail = 0;
        for (row, values) in packed.into_iter().enumerate() {
            for (column, value) in values.into_iter().enumerate() {
                tail |= (value.to_raw() as u8) << (2 * column + row);
            }
        }
        Ok((lanes, tail))
    }
}

impl LaneSource for WitnessLanes {
    fn cycles(&self) -> usize {
        self.lanes.len()
    }
    fn lanes(&self, cycle: usize) -> [[u64; 3]; 2] {
        self.lanes.get(cycle).copied().unwrap_or([[0; 3]; 2])
    }
    fn tail(&self, cycle: usize) -> u8 {
        self.tail.get(cycle).copied().unwrap_or(0)
    }
}

/// Prepares `OuterF2Core` with owned lanes and the unchanged `tau` point.
/// Rows are required of the witness and are not checked: a violated row is
/// detected algebraically, with the soundness error of the sum-check, at the
/// driver's final-claim comparison on this tier and at a round check on the
/// reference tier; neither tier rejects it deterministically.
#[derive(Default)]
pub struct SpartanOuterF2Prepare;

#[cfg_attr(feature = "allocative", derive(Allocative))]
struct OuterKernel {
    #[cfg_attr(feature = "allocative", allocative(skip))]
    core: OuterF2Core<WitnessLanes>,
}

fn geometry(error: impl Display) -> KernelError<F128> {
    KernelError::InvalidGeometry {
        reason: error.to_string(),
    }
}

impl PrepareKernel<F128, SpartanOuterF2<F128>, Rv64iPlane> for SpartanOuterF2Prepare {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &Rv64iWitness,
        inputs: ProverInputs<'_, F128, SpartanOuterF2<F128>>,
    ) -> Result<Box<dyn SumcheckKernel<F128, Relation = SpartanOuterF2<F128>>>, KernelError<F128>>
    {
        let lanes = WitnessLanes::new(witness).map_err(geometry)?;
        let core = OuterF2Core::new(
            Arc::new(lanes),
            inputs.relation.tau(),
            OuterF2Options::default(),
        )
        .map_err(geometry)?;
        Ok(Box::new(OuterKernel { core }))
    }
}

impl ProveRounds<F128> for OuterKernel {
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

impl SumcheckKernel<F128> for OuterKernel {
    type Relation = SpartanOuterF2<F128>;
    fn output_claims(
        &mut self,
        _inputs: &SpartanOuterInputClaims<F128>,
    ) -> Result<SpartanOuterF2OutputClaims<F128>, SumcheckKernelError<F128>> {
        let [az, bz, cz] = self.core.final_values();
        Ok(SpartanOuterF2OutputClaims { az, bz, cz })
    }
}

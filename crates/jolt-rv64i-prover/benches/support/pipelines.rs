//! Driver-level pipelines with setup claims summed from the executed witness.

use super::inventory::Geometry;
use super::timing::{timed, Timing};
use std::collections::BTreeMap;
use std::error::Error;
use std::hint::black_box;
use std::sync::Arc;

use jolt_claims::NoChallenges;
use jolt_field::F128;
use jolt_kernels::{PrepareKernel, ProofSession, ProverInputs};
use jolt_prover::driver::StageProver;
use jolt_rv64i_kernels::chunk_product::{combined_weight, ChunkWeightTerm};
use jolt_rv64i_kernels::column_pass::column_pass;
use jolt_rv64i_kernels::round::eq::eq_table;
use jolt_rv64i_kernels::router::cycle::RoutersCycleCore;
use jolt_rv64i_kernels::router::fold::{fold_pass, FoldLayout};
use jolt_rv64i_prover::optimized::{
    outer::{SpartanOuterF2Prepare, WitnessLanes},
    routers::{
        self, RouterCycleBranchPrepare, RouterCycleComparePrepare, RouterCycleMemoryPrepare,
        RouterCycleShiftPrepare, RouterCycleVariantPrepare, RouterShortPrepare,
    },
    source::{SharedSource, WitnessColumns},
    tail::{BitsReductionPrepare, BytecodeReadCyclePrepare, RamRaProductPrepare},
};
use jolt_rv64i_prover::plane::{DigitFields, Rv64iPlane, Rv64iWitness};
use jolt_rv64i_prover::stages::stage3a::{Stage3aKernels, Stage3aSumchecks};
use jolt_rv64i_prover::stages::stage3b::{Stage3bKernels, Stage3bSumchecks};
use jolt_rv64i_prover::stages::stage6b::{Stage6bKernels, Stage6bSumchecks};
use jolt_rv64i_verifier::proof::RouterFoldValues;
use jolt_rv64i_verifier::public::routes::{projected_index, RouteTensors, ROUTERS};
use jolt_rv64i_verifier::stages::stage1::SpartanOuterF2;
use jolt_rv64i_verifier::stages::stage3a::{
    Output as Stage3aOutput, Stage3aSumchecks as VerifierStage3aSumchecks,
};
use jolt_rv64i_verifier::stages::stage3a::{RouterShortInputClaims, Stage3aInputClaims};
use jolt_rv64i_verifier::stages::stage3b::Stage3bSumchecks as VerifierStage3bSumchecks;
use jolt_rv64i_verifier::stages::stage6b::Stage6bSumchecks as VerifierStage6bSumchecks;
use jolt_rv64i_verifier::stages::stage6b::{
    BitsReduction, BitsReductionInputClaims, BytecodeReadCycle, BytecodeReadCycleInputClaims,
    RamRaProduct, RamRaProductInputClaims, Stage6bInputClaims, Stage6bInputPoints,
};
use jolt_rv64i_verifier::stages::{stage3a, stage3b};
use jolt_sumcheck::{
    prove_batch, BatchMember, BatchPrelude, ClearSumcheckRecorder, ProveRounds, SequentialRounds,
};
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_verifier::stages::relations::ConcreteSumcheck;

pub type BenchResult<T> = Result<T, Box<dyn Error>>;

/// The bench fixes challenges before setup so terminal claims can be summed
/// outside measurement. Absorption still runs; this transcript is no proof artifact.
struct SeedTranscript {
    challenges: Blake2bTranscript<F128>,
    absorbed: Blake2bTranscript<F128>,
}
impl Default for SeedTranscript {
    fn default() -> Self {
        Self::new(b"adapter-bench")
    }
}
impl Transcript for SeedTranscript {
    type Challenge = F128;
    fn new(label: &'static [u8]) -> Self {
        Self {
            challenges: Blake2bTranscript::new(label),
            absorbed: Blake2bTranscript::new(label),
        }
    }
    fn append_bytes(&mut self, bytes: &[u8]) {
        self.absorbed.append_bytes(bytes);
    }
    fn challenge(&mut self) -> F128 {
        self.challenges.challenge()
    }
    fn state(&self) -> [u8; 32] {
        self.absorbed.state()
    }
}

pub struct Kernels {
    pub outer: Box<dyn PrepareKernel<F128, SpartanOuterF2<F128>, Rv64iPlane>>,
    pub stage3a: Stage3aKernels<F128>,
    pub stage3b: Stage3bKernels<F128>,
    pub stage6b: Stage6bKernels<F128>,
}
impl Kernels {
    pub fn new(timing: &Arc<Timing>) -> Self {
        Self {
            outer: timed(Box::new(SpartanOuterF2Prepare), timing),
            stage3a: Stage3aKernels {
                router_short: timed(Box::new(RouterShortPrepare), timing),
            },
            stage3b: Stage3bKernels {
                variant: timed(Box::new(RouterCycleVariantPrepare), timing),
                shift: timed(Box::new(RouterCycleShiftPrepare), timing),
                memory: timed(Box::new(RouterCycleMemoryPrepare), timing),
                compare: timed(Box::new(RouterCycleComparePrepare), timing),
                branch: timed(Box::new(RouterCycleBranchPrepare), timing),
            },
            stage6b: Stage6bKernels {
                bytecode_read_cycle: timed(Box::new(BytecodeReadCyclePrepare), timing),
                ram_ra_product: timed(Box::new(RamRaProductPrepare), timing),
                bits_reduction: timed(Box::new(BitsReductionPrepare), timing),
            },
        }
    }
}

pub struct Fixture {
    pub geometry: Geometry,
    short: Stage3aSumchecks<F128>,
    short_claims: Stage3aInputClaims<F128>,
    terminal: Stage6bSumchecks<F128>,
    terminal_claims: Stage6bInputClaims<F128>,
    terminal_points: Stage6bInputPoints<F128>,
    outer: SpartanOuterF2<F128>,
    selectors: Vec<usize>,
    r_1: Vec<F128>,
    x: Vec<F128>,
}
impl Fixture {
    pub fn new(witness: &Rv64iWitness) -> BenchResult<Self> {
        let log_t = witness.bits.len().ilog2() as usize;
        let mut fixed = SeedTranscript::new(b"adapter-setup");
        let r_1 = fixed.challenge_vector(log_t);
        let r_4 = fixed.challenge_vector(log_t);
        let r_5 = fixed.challenge_vector(log_t);
        let w = fixed.challenge_vector(10);
        let a_bc = fixed.challenge_vector(witness.layout.log_K_bytecode());
        let a_ram = fixed.challenge_vector(witness.layout.log_K_ram());
        let folds: [F128; 5] = std::array::from_fn(|_| fixed.challenge());
        let routes = Arc::new(RouteTensors::new(&witness.layout)?);
        let columns = WitnessColumns::new(&witness.layout);
        let shapes = routers::router_shapes(&columns, &witness.layout, Some(&routes))?;
        let selectors = RoutersCycleCore::columns(&shapes);
        let mut shared = SharedSource::default();
        let trace = shared.prepare(witness, Some(selectors.clone()))?;
        let plan = shared.plan()?;
        let mut values = vec![Vec::new(); shapes.len()];
        values[0] = FoldLayout::byte_bucket_values(
            &witness.variant_cycles.map(|count| count as usize),
            FoldLayout::DEFAULT_BYTE_BUCKET_LIMIT,
        );
        let layout = FoldLayout::new(&trace, &shapes, &values)?;
        let geometry = Geometry {
            fold_entries: layout.entries(),
            row_entries: layout.row_entries(),
            fold_lengths: shapes.iter().map(|shape| shape.fold_len()).collect(),
            selector_columns: selectors.len(),
        };
        let folded = fold_pass(&trace, &shapes, &r_1, &plan, &layout, &[])?;
        let output_weights = eq_table(&w, None);
        let mut witness_routed = F128::from_raw(0);
        for ((shape, fold), router) in shapes.iter().zip(&folded.folds).zip(ROUTERS) {
            let mut weights = BTreeMap::new();
            for entry in shape.route() {
                *weights
                    .entry((entry.source, entry.selector))
                    .or_insert(F128::from_raw(0)) += output_weights[entry.output];
            }
            // `slot_map` documents Fold's compact variable order; projection is
            // the verifier's canonical mapping from shared slots into this bank.
            for (index, value) in fold.iter().enumerate() {
                let shared_index = shape
                    .slot_map()
                    .iter()
                    .enumerate()
                    .fold(0, |shared, (position, &(slot, _))| {
                        shared | (((index >> position) & 1) << slot)
                    });
                if let Some(weight) = weights.get(&projected_index(router, shared_index)) {
                    witness_routed += *weight * *value;
                }
            }
        }
        drop(folded);
        drop(plan);
        drop(trace);
        drop(shared);
        let short = Stage3aSumchecks(VerifierStage3aSumchecks::new(
            w.clone(),
            r_1.clone(),
            routes,
        )?);
        let short_claims = Stage3aInputClaims {
            router_short: RouterShortInputClaims { witness_routed },
        };
        let mut scheduled = SeedTranscript::default();
        let short_challenges = short.draw_challenges(&mut scheduled)?;
        let (prelude, _) = short.begin_batch(
            &short_claims,
            &short_challenges,
            &mut ClearSumcheckRecorder::<F128>::new(),
            &mut scheduled,
        )?;
        let x = scheduled.challenge_vector(prelude.max_num_vars);
        let cycle = VerifierStage3bSumchecks::new(&witness.layout, r_1.clone(), x.clone())?;
        let cycle_challenges = cycle.draw_challenges(&mut scheduled)?;
        let zero_folds = RouterFoldValues {
            variant: F128::from_raw(0),
            shift: F128::from_raw(0),
            memory: F128::from_raw(0),
            compare: F128::from_raw(0),
            branch: F128::from_raw(0),
        };
        let (prelude, _) = cycle.begin_batch(
            &stage3b::verify::input_values(&zero_folds),
            &cycle_challenges,
            &mut ClearSumcheckRecorder::<F128>::new(),
            &mut scheduled,
        )?;
        let r_3 = scheduled.challenge_vector(prelude.max_num_vars);
        let bytecode_read_cycle = BytecodeReadCycle::new(
            &witness.layout,
            folds,
            a_bc.clone(),
            r_3.clone(),
            r_4.clone(),
            r_5.clone(),
        )?;
        let ram_ra_product =
            RamRaProduct::new(&witness.layout, a_ram.clone(), r_4.clone(), r_5.clone())?;
        let bits_reduction = BitsReduction::new(
            &witness.layout,
            r_1.clone(),
            r_3.clone(),
            r_5.clone(),
            w,
            x.clone(),
        )?;
        let columns_1 = column_pass(&witness.bits, &r_1)?;
        let columns_3 = column_pass(&witness.bits, &r_3)?;
        let columns_5 = column_pass(&witness.bits, &r_5)?;
        let mut functional = [F128::from_raw(0); 6];
        for ((claim, support), columns) in
            functional.iter_mut().zip(bits_reduction.weights()).zip([
                &columns_1, &columns_3, &columns_3, &columns_3, &columns_3, &columns_5,
            ])
        {
            *claim = support
                .iter()
                .map(|&(column, weight)| columns[column] * weight)
                .sum();
        }
        let [zero_0, zero_1] = bits_reduction.pos_zero();
        let reduction_claims = BitsReductionInputClaims {
            direct_columns: functional[0],
            variant_bits: functional[1],
            pos_ra_0: functional[2] + zero_0,
            pos_ra_1: functional[3] + zero_1,
            should_branch: functional[4],
            inc: functional[5],
        };
        let bytecode_weights = combined_weight(
            log_t,
            &[
                ChunkWeightTerm::Eq {
                    coefficient: folds[0],
                    point: r_3,
                },
                ChunkWeightTerm::Eq {
                    coefficient: folds[1],
                    point: r_4.clone(),
                },
                ChunkWeightTerm::Eq {
                    coefficient: folds[2],
                    point: r_5.clone(),
                },
                ChunkWeightTerm::Eq {
                    coefficient: folds[3],
                    point: vec![F128::from_raw(0); log_t],
                },
                ChunkWeightTerm::Next {
                    coefficient: folds[4],
                    point: bytecode_read_cycle.r_3().to_vec(),
                },
            ],
        )?;
        let bytecode_address = eq_table(&a_bc, None);
        let ram_address = eq_table(&a_ram, None);
        let read_weight = eq_table(&r_4, None);
        let val_weight = eq_table(&r_5, None);
        let fields = DigitFields::new(&witness.layout);
        let mut address_claim = F128::from_raw(0);
        let mut ram_ra_read = F128::from_raw(0);
        let mut ram_ra_val = F128::from_raw(0);
        for (cycle, row) in witness.decoded.iter().enumerate() {
            address_claim += bytecode_weights[cycle]
                * bytecode_address[fields.bytecode_index().read(row) as usize];
            let ram = ram_address[fields.ram_index().read(row) as usize];
            ram_ra_read += read_weight[cycle] * ram;
            ram_ra_val += val_weight[cycle] * ram;
        }
        let terminal_points = Stage6bInputPoints {
            bytecode_read_cycle: bytecode_read_cycle.input_points(),
            ram_ra_product: ram_ra_product.input_points(),
            bits_reduction: bits_reduction.input_points(),
        };
        let terminal = Stage6bSumchecks(VerifierStage6bSumchecks {
            bytecode_read_cycle,
            ram_ra_product,
            bits_reduction,
        });
        let terminal_claims = Stage6bInputClaims {
            bytecode_read_cycle: BytecodeReadCycleInputClaims { address_claim },
            ram_ra_product: RamRaProductInputClaims {
                ram_ra_read,
                ram_ra_val,
            },
            bits_reduction: reduction_claims,
        };
        let outer = SpartanOuterF2::new(log_t, 8, fixed.challenge_vector(log_t + 8))?;
        Ok(Self {
            geometry,
            short,
            short_claims,
            terminal,
            terminal_claims,
            terminal_points,
            outer,
            selectors,
            r_1,
            x,
        })
    }

    pub fn warm_session(&self, witness: &Rv64iWitness, name: &str) -> BenchResult<ProofSession> {
        let mut session = ProofSession::default();
        if matches!(name, "routers" | "tail") {
            let selectors = (name == "routers").then(|| self.selectors.clone());
            let _ = session
                .state_or_insert_with(SharedSource::default)
                .prepare(witness, selectors)?;
        }
        Ok(session)
    }

    pub fn run(
        &self,
        name: &str,
        witness: &Rv64iWitness,
        session: &mut ProofSession,
        kernels: &Kernels,
    ) -> BenchResult<()> {
        if name == "source" {
            let _ = black_box(
                session
                    .state_or_insert_with(SharedSource::default)
                    .prepare(witness, Some(self.selectors.clone()))?,
            );
            return Ok(());
        }
        if name == "lanes" {
            let _ = black_box(WitnessLanes::new(witness)?);
            return Ok(());
        }
        if name == "outer" {
            return self.run_outer(witness, session, kernels);
        }
        let mut transcript = SeedTranscript::default();
        if matches!(name, "routers" | "session") {
            let challenges = self.short.draw_challenges(&mut transcript)?;
            let proved = self.short.prove(
                &kernels.stage3a,
                session,
                &mut SequentialRounds,
                witness,
                &self.short_claims,
                &self.short.input_points(),
                &challenges,
                ClearSumcheckRecorder::<F128>::new(),
                &mut transcript,
            )?;
            let output = Stage3aOutput::new(proved.output_claims, proved.output_points)?;
            if output.x != self.x {
                return Err("short challenge schedule disagrees with setup".into());
            }
            let cycle = Stage3bSumchecks(VerifierStage3bSumchecks::new(
                &witness.layout,
                self.r_1.clone(),
                output.x,
            )?);
            let claims = stage3b::verify::input_values(&stage3a::verify::values(&output.claims));
            let challenges = cycle.draw_challenges(&mut transcript)?;
            let proved = cycle.prove(
                &kernels.stage3b,
                session,
                &mut SequentialRounds,
                witness,
                &claims,
                &cycle.input_points()?,
                &challenges,
                ClearSumcheckRecorder::<F128>::new(),
                &mut transcript,
            )?;
            if proved.output_points.branch.branch != self.terminal.bytecode_read_cycle.r_3() {
                return Err("cycle challenge schedule disagrees with setup".into());
            }
            let _ = black_box(proved);
        }
        if matches!(name, "tail" | "session") {
            let challenges = self.terminal.draw_challenges(&mut transcript)?;
            let _ = black_box(self.terminal.prove(
                &kernels.stage6b,
                session,
                &mut SequentialRounds,
                witness,
                &self.terminal_claims,
                &self.terminal_points,
                &challenges,
                ClearSumcheckRecorder::<F128>::new(),
                &mut transcript,
            )?);
        }
        Ok(())
    }

    fn run_outer(
        &self,
        witness: &Rv64iWitness,
        session: &mut ProofSession,
        kernels: &Kernels,
    ) -> BenchResult<()> {
        let claims = Default::default();
        let points = Default::default();
        let challenges = NoChallenges::default();
        let mut core = kernels.outer.prepare(
            session,
            witness,
            ProverInputs {
                relation: &self.outer,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            },
        )?;
        let prelude = BatchPrelude::try_new(
            vec![BatchMember {
                input_claim: F128::from_raw(0),
                coefficient: F128::from_raw(1),
                rounds: core.num_rounds(),
                offset: 0,
            }],
            core.num_rounds(),
            self.outer.degree(),
        )?;
        let mut transcript = SeedTranscript::new(b"adapter-outer");
        let proved = prove_batch(
            &prelude,
            &mut [&mut *core as &mut dyn ProveRounds<F128>],
            &mut SequentialRounds,
            &mut ClearSumcheckRecorder::<F128>::new(),
            &mut transcript,
        )?;
        let output_points = self
            .outer
            .derive_opening_points(&proved.challenges, &points)?;
        core.validate_derived_tables(&self.outer, &points, &output_points, &challenges)?;
        let output_claims = core.output_claims(&claims)?;
        core.park_residue(session);
        let expected =
            self.outer
                .expected_output(&points, &output_claims, &output_points, &challenges)?;
        if expected != proved.final_claim {
            return Err(format!(
                "outer final claim mismatch: expected {expected:?}, got {:?}",
                proved.final_claim
            )
            .into());
        }
        Ok(())
    }
}

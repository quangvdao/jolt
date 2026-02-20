use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

use ark_bn254::Fr;
#[cfg(feature = "allocative")]
use allocative::size_of_unique_allocated_data;
use jolt_core::{
    field::JoltField,
    poly::opening_proof::ProverOpeningAccumulator,
    poly::unipoly::UniPoly,
    subprotocols::{
        streaming_schedule::{HalfSplitSchedule, LinearOnlySchedule},
        sumcheck::BatchedSumcheck,
        sumcheck_prover::SumcheckInstanceProver, univariate_skip::prove_uniskip_round,
    },
    transcripts::{KeccakTranscript, Transcript},
    utils::profiling::print_current_memory_usage,
    utils::math::Math,
    zkvm::{
        bytecode::BytecodePreprocessing,
        r1cs::constraints::{R1CSConstraint, R1CS_CONSTRAINTS},
        r1cs::key::UniformSpartanKey,
        spartan::{
            outer::{
                OuterRemainingStreamingSumcheck, OuterRemainingStreamingSumcheckMTable,
                OuterSharedState, OuterUniSkipParams, OuterUniSkipProver,
            },
            outer_delayed_reduction::OuterDelayedReductionSumcheckProver,
            outer_naive::OuterBaselineSumcheckProver,
            outer_round_batched::OuterRoundBatchedSumcheckProver,
            outer_split_eq::OuterSplitEqSumcheckProver,
            outer_uni_skip_linear::{
                OuterRemainingSumcheckProverNonStreaming, OuterUniSkipInstanceProver,
            },
        },
    },
};
use tracer::instruction::Cycle;

#[cfg(feature = "host")]
use jolt_core::host;
#[cfg(feature = "host")]
use jolt_inlines_sha2 as _;
#[cfg(not(target_arch = "wasm32"))]
use memory_stats::memory_stats;

type F = Fr;
type ProofTranscript = KeccakTranscript;

const PHASE_IDLE: usize = 0;
const PHASE_UNI_INIT: usize = 1;
const PHASE_UNI_COMPUTE: usize = 2;
const PHASE_UNI_INGEST: usize = 3;
const PHASE_REMAINDER_INIT: usize = 4;
const PHASE_REMAINDER_COMPUTE: usize = 5;
const PHASE_REMAINDER_INGEST: usize = 6;
const PHASE_REMAINDER_FINALIZE: usize = 7;
const PHASE_COUNT: usize = 8;

#[cfg(not(target_arch = "wasm32"))]
fn phase_name(phase: usize) -> &'static str {
    match phase {
        PHASE_IDLE => "idle",
        PHASE_UNI_INIT => "uni_init",
        PHASE_UNI_COMPUTE => "uni_compute_message",
        PHASE_UNI_INGEST => "uni_ingest_challenge",
        PHASE_REMAINDER_INIT => "remainder_init",
        PHASE_REMAINDER_COMPUTE => "remainder_compute_message",
        PHASE_REMAINDER_INGEST => "remainder_ingest_challenge",
        PHASE_REMAINDER_FINALIZE => "remainder_finalize_cache",
        _ => "unknown",
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn print_rss_mb(label: &str) {
    if let Some(mem) = memory_stats() {
        println!(
            "rss_mb_{}={:.2}",
            label,
            mem.physical_mem as f64 / 1_000_000.0
        );
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct PhaseMemSampler {
    current_phase: Arc<AtomicUsize>,
    max_by_phase: Arc<Vec<AtomicU64>>,
    stop_flag: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl PhaseMemSampler {
    fn start() -> Self {
        let current_phase = Arc::new(AtomicUsize::new(PHASE_IDLE));
        let max_by_phase = Arc::new((0..PHASE_COUNT).map(|_| AtomicU64::new(0)).collect::<Vec<_>>());
        let stop_flag = Arc::new(AtomicBool::new(false));

        let phase_clone = Arc::clone(&current_phase);
        let max_clone = Arc::clone(&max_by_phase);
        let stop_clone = Arc::clone(&stop_flag);
        let handle = thread::Builder::new()
            .name("uni-skip-phase-mem-sampler".to_string())
            .spawn(move || {
                while !stop_clone.load(Ordering::Relaxed) {
                    if let Some(mem) = memory_stats() {
                        let phase = phase_clone.load(Ordering::Relaxed);
                        if phase < max_clone.len() {
                            let bytes = mem.physical_mem as u64;
                            let slot = &max_clone[phase];
                            let mut prev = slot.load(Ordering::Relaxed);
                            while bytes > prev
                                && slot
                                    .compare_exchange_weak(
                                        prev,
                                        bytes,
                                        Ordering::Relaxed,
                                        Ordering::Relaxed,
                                    )
                                    .is_err()
                            {
                                prev = slot.load(Ordering::Relaxed);
                            }
                        }
                    }
                    thread::sleep(Duration::from_millis(1));
                }
            })
            .expect("Failed to spawn phase memory sampler");

        Self {
            current_phase,
            max_by_phase,
            stop_flag,
            handle: Some(handle),
        }
    }

    fn set_phase(&self, phase: usize) {
        self.current_phase.store(phase, Ordering::Relaxed);
    }

    fn phase_handle(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.current_phase)
    }

    fn stop(mut self) -> Vec<u64> {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        self.max_by_phase
            .iter()
            .map(|x| x.load(Ordering::Relaxed))
            .collect()
    }
}

#[derive(allocative::Allocative)]
struct PhaseTrackedInstance<I> {
    #[allocative(skip)]
    inner: I,
    #[allocative(skip)]
    phase_handle: Arc<AtomicUsize>,
    compute_phase: usize,
    ingest_phase: usize,
    finalize_phase: usize,
}

impl<I> PhaseTrackedInstance<I> {
    fn new(
        inner: I,
        phase_handle: Arc<AtomicUsize>,
        compute_phase: usize,
        ingest_phase: usize,
        finalize_phase: usize,
    ) -> Self {
        Self {
            inner,
            phase_handle,
            compute_phase,
            ingest_phase,
            finalize_phase,
        }
    }

    #[cfg(feature = "allocative")]
    fn into_inner(self) -> I {
        self.inner
    }
}

impl<F: JoltField, T: Transcript, I: SumcheckInstanceProver<F, T>> SumcheckInstanceProver<F, T>
    for PhaseTrackedInstance<I>
{
    fn degree(&self) -> usize {
        self.inner.degree()
    }

    fn num_rounds(&self) -> usize {
        self.inner.num_rounds()
    }

    fn round_offset(&self, max_num_rounds: usize) -> usize {
        self.inner.round_offset(max_num_rounds)
    }

    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.inner.input_claim(accumulator)
    }

    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        self.phase_handle
            .store(self.compute_phase, Ordering::Relaxed);
        let out = self.inner.compute_message(round, previous_claim);
        self.phase_handle.store(PHASE_IDLE, Ordering::Relaxed);
        out
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        self.phase_handle
            .store(self.ingest_phase, Ordering::Relaxed);
        self.inner.ingest_challenge(r_j, round);
        self.phase_handle.store(PHASE_IDLE, Ordering::Relaxed);
    }

    fn finalize(&mut self) {
        self.phase_handle
            .store(self.finalize_phase, Ordering::Relaxed);
        self.inner.finalize();
        self.phase_handle.store(PHASE_IDLE, Ordering::Relaxed);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        self.phase_handle
            .store(self.finalize_phase, Ordering::Relaxed);
        self.inner
            .cache_openings(accumulator, transcript, sumcheck_challenges);
        self.phase_handle.store(PHASE_IDLE, Ordering::Relaxed);
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        self.inner.update_flamegraph(flamegraph);
    }
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Baseline,
    SplitEq,
    DelayedReduction,
    RoundBatched,
    CurrentLinear,
    UniSkip,
    Streaming,
    StreamingCoeffBasis,
}

impl Mode {
    fn from_env() -> Self {
        let raw = std::env::var("OUTER_MODE")
            .unwrap_or_else(|_| "streaming".to_string())
            .to_lowercase();
        match raw.as_str() {
            "baseline" | "outer-baseline" => Self::Baseline,
            "split-eq" | "spliteq" | "outer-split-eq" => Self::SplitEq,
            "delayed-reduction" | "outer-delayed-reduction" => Self::DelayedReduction,
            "round-batched" | "roundbatched" | "outer-round-batched" => Self::RoundBatched,
            "current" | "outer-current" => Self::CurrentLinear,
            "uniskip" | "outer-uni-skip" => Self::UniSkip,
            "streaming" | "outer-streaming" => Self::Streaming,
            "streaming-coeff-basis" | "streaming-mtable" | "outer-streaming-coeff-basis" => {
                Self::StreamingCoeffBasis
            }
            _ => Self::Streaming,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "outer-baseline",
            Self::SplitEq => "outer-split-eq",
            Self::DelayedReduction => "outer-delayed-reduction",
            Self::RoundBatched => "outer-round-batched",
            Self::CurrentLinear => "outer-current",
            Self::UniSkip => "outer-uni-skip",
            Self::Streaming => "outer-streaming",
            Self::StreamingCoeffBasis => "outer-streaming-coeff-basis",
        }
    }
}

fn setup_for_spartan(
    program_name: &str,
    num_iterations: u32,
) -> (
    Arc<Vec<Cycle>>,
    BytecodePreprocessing,
    usize,
    ProverOpeningAccumulator<F>,
    ProofTranscript,
) {
    let mut program = host::Program::new(program_name);

    let initial_hash_data = [7u8; 32];
    let guest_input_tuple = (initial_hash_data, num_iterations);
    let inputs = postcard::to_stdvec(&guest_input_tuple).unwrap();

    let (bytecode, _memory_init, _) = program.decode();
    let (_lazy_trace, mut trace, _final_memory_state, _io_device) = program.trace(&inputs, &[], &[]);

    let padded_trace_length = (trace.len() + 1).next_power_of_two();
    trace.resize(padded_trace_length, Cycle::NoOp);

    let bytecode_pp = BytecodePreprocessing::preprocess(bytecode);

    let transcript = ProofTranscript::new(b"Jolt transcript");
    let opening_bits = padded_trace_length.log_2();
    let opening_accumulator = ProverOpeningAccumulator::<F>::new(opening_bits);

    (
        Arc::new(trace),
        bytecode_pp,
        padded_trace_length,
        opening_accumulator,
        transcript,
    )
}

#[cfg(feature = "allocative")]
fn bytes_to_mb(bytes: usize) -> f64 {
    bytes as f64 / 1_000_000.0
}

fn main() {
    let mode = Mode::from_env();
    let iters: u32 = std::env::var("SHA2_ITERS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(2048);
    let phase_mem_profile = std::env::var("PHASE_MEM_PROFILE")
        .ok()
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    println!("mode={}", mode.as_str());
    println!("sha2_iters={iters}");

    let (trace, bytecode_pp, padded_trace_length, mut opening_accumulator, mut transcript) =
        setup_for_spartan("sha2-chain-guest", iters);
    #[cfg(not(target_arch = "wasm32"))]
    print_rss_mb("after_setup");
    print_current_memory_usage("outer-binary baseline (after trace setup)");

    match mode {
        Mode::Baseline => {
            let padded_num_constraints = R1CS_CONSTRAINTS.len().next_power_of_two();
            let constraints_vec: Vec<R1CSConstraint> =
                R1CS_CONSTRAINTS.iter().map(|c| c.cons).collect();
            let mut instance = OuterBaselineSumcheckProver::<F>::gen(
                &bytecode_pp,
                Arc::clone(&trace),
                &constraints_vec,
                padded_num_constraints,
                &mut transcript,
            );
            print_current_memory_usage("outer-baseline instance");

            #[cfg(feature = "allocative")]
            {
                let total_mb = bytes_to_mb(size_of_unique_allocated_data(&instance));
                println!("outer_heap_mb={total_mb:.2}");
            }

            let refs: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>> = vec![&mut instance];
            let _ = BatchedSumcheck::prove(refs, &mut opening_accumulator, &mut transcript);

            #[cfg(feature = "allocative")]
            {
                let post_mb = bytes_to_mb(size_of_unique_allocated_data(&instance));
                println!("outer_postprove_heap_mb={post_mb:.2}");
            }
        }
        Mode::SplitEq => {
            let padded_num_constraints = R1CS_CONSTRAINTS.len().next_power_of_two();
            let constraints_vec: Vec<R1CSConstraint> =
                R1CS_CONSTRAINTS.iter().map(|c| c.cons).collect();
            let mut instance = OuterSplitEqSumcheckProver::<F>::gen(
                &bytecode_pp,
                Arc::clone(&trace),
                &constraints_vec,
                padded_num_constraints,
                &mut transcript,
            );
            print_current_memory_usage("outer-split-eq instance");

            #[cfg(feature = "allocative")]
            {
                let total_mb = bytes_to_mb(size_of_unique_allocated_data(&instance));
                println!("outer_heap_mb={total_mb:.2}");
            }

            let refs: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>> = vec![&mut instance];
            let _ = BatchedSumcheck::prove(refs, &mut opening_accumulator, &mut transcript);

            #[cfg(feature = "allocative")]
            {
                let post_mb = bytes_to_mb(size_of_unique_allocated_data(&instance));
                println!("outer_postprove_heap_mb={post_mb:.2}");
            }
        }
        Mode::DelayedReduction => {
            let padded_num_constraints = R1CS_CONSTRAINTS.len().next_power_of_two();
            let constraints_vec: Vec<R1CSConstraint> =
                R1CS_CONSTRAINTS.iter().map(|c| c.cons).collect();
            let mut instance = OuterDelayedReductionSumcheckProver::<F>::gen(
                &bytecode_pp,
                Arc::clone(&trace),
                &constraints_vec,
                padded_num_constraints,
                &mut transcript,
            );
            print_current_memory_usage("outer-delayed-reduction instance");

            #[cfg(feature = "allocative")]
            {
                let total_mb = bytes_to_mb(size_of_unique_allocated_data(&instance));
                println!("outer_heap_mb={total_mb:.2}");
            }

            let refs: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>> = vec![&mut instance];
            let _ = BatchedSumcheck::prove(refs, &mut opening_accumulator, &mut transcript);

            #[cfg(feature = "allocative")]
            {
                let post_mb = bytes_to_mb(size_of_unique_allocated_data(&instance));
                println!("outer_postprove_heap_mb={post_mb:.2}");
            }
        }
        Mode::RoundBatched => {
            let mut instance = OuterRoundBatchedSumcheckProver::<F>::gen::<ProofTranscript>(
                Arc::clone(&trace),
                &bytecode_pp,
                &mut transcript,
            );
            print_current_memory_usage("outer-round-batched instance");

            #[cfg(feature = "allocative")]
            {
                let total_mb = bytes_to_mb(size_of_unique_allocated_data(&instance));
                println!("outer_heap_mb={total_mb:.2}");
            }

            let refs: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>> = vec![&mut instance];
            let _ = BatchedSumcheck::prove(refs, &mut opening_accumulator, &mut transcript);

            #[cfg(feature = "allocative")]
            {
                let post_mb = bytes_to_mb(size_of_unique_allocated_data(&instance));
                println!("outer_postprove_heap_mb={post_mb:.2}");
            }
        }
        Mode::CurrentLinear => {
            let key = UniformSpartanKey::<F>::new(padded_trace_length);
            let uni_skip_params = OuterUniSkipParams::<F>::new(&key, &mut transcript);
            let mut uni_skip = OuterUniSkipProver::<F>::initialize(
                uni_skip_params.clone(),
                trace.as_ref(),
                &bytecode_pp,
            );
            let _ = prove_uniskip_round(&mut uni_skip, &mut opening_accumulator, &mut transcript);

            let num_rounds = uni_skip_params.tau.len() - 1;
            let schedule = LinearOnlySchedule::new(num_rounds);
            let shared = OuterSharedState::<F>::new(
                Arc::clone(&trace),
                &bytecode_pp,
                &uni_skip_params,
                &opening_accumulator,
            );
            let mut instance: OuterRemainingStreamingSumcheck<F, LinearOnlySchedule> =
                OuterRemainingStreamingSumcheck::new(shared, schedule);
            print_current_memory_usage("outer-current instance");

            #[cfg(feature = "allocative")]
            {
                let total_mb = bytes_to_mb(size_of_unique_allocated_data(&instance));
                println!("outer_heap_mb={total_mb:.2}");
            }

            let refs: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>> = vec![&mut instance];
            let _ = BatchedSumcheck::prove(refs, &mut opening_accumulator, &mut transcript);

            #[cfg(feature = "allocative")]
            {
                let post_mb = bytes_to_mb(size_of_unique_allocated_data(&instance));
                println!("outer_postprove_heap_mb={post_mb:.2}");
            }
        }
        Mode::UniSkip => {
            #[cfg(not(target_arch = "wasm32"))]
            let sampler = if phase_mem_profile {
                Some(PhaseMemSampler::start())
            } else {
                None
            };

            #[cfg(not(target_arch = "wasm32"))]
            if let Some(s) = sampler.as_ref() {
                s.set_phase(PHASE_UNI_INIT);
            }
            let key = UniformSpartanKey::<F>::new(padded_trace_length);
            let uni_skip_params = OuterUniSkipParams::<F>::new(&key, &mut transcript);
            let mut uni_skip = OuterUniSkipInstanceProver::<F>::gen(
                trace.as_ref(),
                &bytecode_pp,
                &uni_skip_params.tau,
            );
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(s) = sampler.as_ref() {
                s.set_phase(PHASE_UNI_COMPUTE);
            }
            let uni_input_claim =
                <OuterUniSkipInstanceProver<F> as SumcheckInstanceProver<F, ProofTranscript>>::input_claim(
                    &uni_skip,
                    &opening_accumulator,
                );
            let uni_poly =
                <OuterUniSkipInstanceProver<F> as SumcheckInstanceProver<F, ProofTranscript>>::compute_message(
                    &mut uni_skip,
                    0,
                    uni_input_claim,
                );
            transcript.append_scalars(b"uniskip_poly", &uni_poly.coeffs);
            let r0: <F as JoltField>::Challenge = transcript.challenge_scalar_optimized::<F>();
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(s) = sampler.as_ref() {
                s.set_phase(PHASE_UNI_INGEST);
            }
            <OuterUniSkipInstanceProver<F> as SumcheckInstanceProver<F, ProofTranscript>>::ingest_challenge(
                &mut uni_skip,
                r0,
                0,
            );
            <OuterUniSkipInstanceProver<F> as SumcheckInstanceProver<F, ProofTranscript>>::cache_openings(
                &uni_skip,
                &mut opening_accumulator,
                &mut transcript,
                &[r0],
            );

            #[cfg(not(target_arch = "wasm32"))]
            if let Some(s) = sampler.as_ref() {
                s.set_phase(PHASE_REMAINDER_INIT);
            }
            let mut instance = OuterRemainingSumcheckProverNonStreaming::<F>::gen(
                Arc::clone(&trace),
                &bytecode_pp,
                uni_skip_params,
                &opening_accumulator,
            );
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(s) = sampler.as_ref() {
                s.set_phase(PHASE_IDLE);
            }
            print_current_memory_usage("outer-remainder baseline");

            #[cfg(feature = "allocative")]
            {
                let total_mb = bytes_to_mb(size_of_unique_allocated_data(&instance));
                println!("outer_heap_mb={total_mb:.2}");
                println!("outer_remainder_heap_mb={total_mb:.2}");
            }

            #[cfg(not(target_arch = "wasm32"))]
            if let Some(phase_handle) = sampler.as_ref().map(|s| s.phase_handle()) {
                let mut tracked = PhaseTrackedInstance::new(
                    instance,
                    phase_handle,
                    PHASE_REMAINDER_COMPUTE,
                    PHASE_REMAINDER_INGEST,
                    PHASE_REMAINDER_FINALIZE,
                );
                let refs: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>> =
                    vec![&mut tracked];
                let _ = BatchedSumcheck::prove(refs, &mut opening_accumulator, &mut transcript);
                #[cfg(feature = "allocative")]
                {
                    instance = tracked.into_inner();
                }
                #[cfg(not(feature = "allocative"))]
                {
                    drop(tracked);
                }
            } else {
                let refs: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>> =
                    vec![&mut instance];
                let _ = BatchedSumcheck::prove(refs, &mut opening_accumulator, &mut transcript);
            }
            #[cfg(target_arch = "wasm32")]
            {
                let refs: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>> =
                    vec![&mut instance];
                let _ = BatchedSumcheck::prove(refs, &mut opening_accumulator, &mut transcript);
            }

            #[cfg(feature = "allocative")]
            {
                let post_mb = bytes_to_mb(size_of_unique_allocated_data(&instance));
                println!("outer_postprove_heap_mb={post_mb:.2}");
                println!("outer_remainder_postprove_heap_mb={post_mb:.2}");
            }

            #[cfg(not(target_arch = "wasm32"))]
            if let Some(s) = sampler {
                let phase_max = s.stop();
                let overall = phase_max.iter().copied().max().unwrap_or(0);
                println!(
                    "phase_peak_rss_mb_overall={:.2}",
                    overall as f64 / 1_000_000.0
                );
                for (idx, bytes) in phase_max.iter().enumerate() {
                    if *bytes == 0 {
                        continue;
                    }
                    println!(
                        "phase_peak_rss_mb_{}={:.2}",
                        phase_name(idx),
                        *bytes as f64 / 1_000_000.0
                    );
                }
            }
        }
        Mode::Streaming => {
            let key = UniformSpartanKey::<F>::new(padded_trace_length);
            let uni_skip_params = OuterUniSkipParams::<F>::new(&key, &mut transcript);
            let mut uni_skip = OuterUniSkipProver::<F>::initialize(
                uni_skip_params.clone(),
                trace.as_ref(),
                &bytecode_pp,
            );
            let _ = prove_uniskip_round(&mut uni_skip, &mut opening_accumulator, &mut transcript);

            let num_rounds = uni_skip_params.tau.len() - 1;
            let schedule = HalfSplitSchedule::new(num_rounds, 2);
            let shared = OuterSharedState::<F>::new(
                Arc::clone(&trace),
                &bytecode_pp,
                &uni_skip_params,
                &opening_accumulator,
            );
            #[cfg(feature = "allocative")]
            let shared_mb = bytes_to_mb(size_of_unique_allocated_data(&shared));
            let mut instance: OuterRemainingStreamingSumcheck<F, HalfSplitSchedule> =
                OuterRemainingStreamingSumcheck::new(shared, schedule);
            print_current_memory_usage("outer-remainder baseline");

            #[cfg(feature = "allocative")]
            {
                let total_mb = bytes_to_mb(size_of_unique_allocated_data(&instance));
                let incremental_mb = (total_mb - shared_mb).max(0.0);
                println!("outer_shared_heap_mb={shared_mb:.2}");
                println!("outer_remainder_total_heap_mb={total_mb:.2}");
                println!("outer_remainder_incremental_heap_mb={incremental_mb:.2}");
            }

            let refs: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>> = vec![&mut instance];
            let _ = BatchedSumcheck::prove(refs, &mut opening_accumulator, &mut transcript);

            #[cfg(feature = "allocative")]
            {
                let post_mb = bytes_to_mb(size_of_unique_allocated_data(&instance));
                let post_incremental_mb = (post_mb - shared_mb).max(0.0);
                println!("outer_remainder_postprove_total_heap_mb={post_mb:.2}");
                println!("outer_remainder_postprove_incremental_heap_mb={post_incremental_mb:.2}");
            }
        }
        Mode::StreamingCoeffBasis => {
            let key = UniformSpartanKey::<F>::new(padded_trace_length);
            let uni_skip_params = OuterUniSkipParams::<F>::new(&key, &mut transcript);
            let mut uni_skip = OuterUniSkipProver::<F>::initialize(
                uni_skip_params.clone(),
                trace.as_ref(),
                &bytecode_pp,
            );
            let _ = prove_uniskip_round(&mut uni_skip, &mut opening_accumulator, &mut transcript);

            let num_rounds = uni_skip_params.tau.len() - 1;
            let schedule = HalfSplitSchedule::new(num_rounds, 2);
            let shared = OuterSharedState::<F>::new(
                Arc::clone(&trace),
                &bytecode_pp,
                &uni_skip_params,
                &opening_accumulator,
            );
            #[cfg(feature = "allocative")]
            let shared_mb = bytes_to_mb(size_of_unique_allocated_data(&shared));
            let mut instance: OuterRemainingStreamingSumcheckMTable<F, HalfSplitSchedule> =
                OuterRemainingStreamingSumcheckMTable::new(shared, schedule);
            print_current_memory_usage("outer-remainder baseline");

            #[cfg(feature = "allocative")]
            {
                let total_mb = bytes_to_mb(size_of_unique_allocated_data(&instance));
                let incremental_mb = (total_mb - shared_mb).max(0.0);
                println!("outer_shared_heap_mb={shared_mb:.2}");
                println!("outer_remainder_total_heap_mb={total_mb:.2}");
                println!("outer_remainder_incremental_heap_mb={incremental_mb:.2}");
            }

            let refs: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>> = vec![&mut instance];
            let _ = BatchedSumcheck::prove(refs, &mut opening_accumulator, &mut transcript);

            #[cfg(feature = "allocative")]
            {
                let post_mb = bytes_to_mb(size_of_unique_allocated_data(&instance));
                let post_incremental_mb = (post_mb - shared_mb).max(0.0);
                println!("outer_remainder_postprove_total_heap_mb={post_mb:.2}");
                println!("outer_remainder_postprove_incremental_heap_mb={post_incremental_mb:.2}");
            }
        }
    }
}

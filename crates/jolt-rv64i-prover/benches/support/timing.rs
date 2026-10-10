//! Kernel hooks separate phases without replacing the generated drivers.

use super::allocator::CountingAllocator;
use jolt_field::F128;
use jolt_kernels::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};
use jolt_poly::UnivariatePoly;
use jolt_rv64i_prover::plane::{Rv64iPlane, Rv64iWitness};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const PHASES: [&str; 6] = ["prepare", "rounds", "finish", "extract", "park", "driver"];

#[derive(Default)]
pub struct Timing(Mutex<[Duration; 5]>);
impl Timing {
    #[expect(
        clippy::unwrap_used,
        reason = "a poisoned timing mutex means a benchmark worker panicked"
    )]
    pub fn times(&self) -> [Duration; 5] {
        *self.0.lock().unwrap()
    }

    #[expect(
        clippy::unwrap_used,
        reason = "a poisoned timing mutex means a benchmark worker panicked"
    )]
    pub fn measure<T>(&self, phase: usize, f: impl FnOnce() -> T) -> T {
        CountingAllocator::phase(phase);
        let start = Instant::now();
        let result = f();
        let elapsed = start.elapsed();
        self.0.lock().unwrap()[phase] += elapsed;
        CountingAllocator::phase(5);
        result
    }
}

pub fn timed<R: ConcreteSumcheck<F128> + 'static>(
    inner: Box<dyn PrepareKernel<F128, R, Rv64iPlane>>,
    timing: &Arc<Timing>,
) -> Box<dyn PrepareKernel<F128, R, Rv64iPlane>> {
    Box::new(TimedPrepare {
        inner,
        timing: Arc::clone(timing),
    })
}

struct TimedPrepare<R: ConcreteSumcheck<F128>> {
    inner: Box<dyn PrepareKernel<F128, R, Rv64iPlane>>,
    timing: Arc<Timing>,
}
impl<R: ConcreteSumcheck<F128> + 'static> PrepareKernel<F128, R, Rv64iPlane> for TimedPrepare<R> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &Rv64iWitness,
        inputs: ProverInputs<'_, F128, R>,
    ) -> Result<Box<dyn SumcheckKernel<F128, Relation = R>>, KernelError<F128>> {
        let inner = self
            .timing
            .measure(0, || self.inner.prepare(session, witness, inputs))?;
        Ok(Box::new(TimedKernel {
            inner,
            timing: Arc::clone(&self.timing),
        }))
    }
}
struct TimedKernel<R: ConcreteSumcheck<F128>> {
    inner: Box<dyn SumcheckKernel<F128, Relation = R>>,
    timing: Arc<Timing>,
}
impl<R: ConcreteSumcheck<F128>> ProveRounds<F128> for TimedKernel<R> {
    fn num_rounds(&self) -> usize {
        self.inner.num_rounds()
    }
    fn prove_round(
        &mut self,
        bind: Option<F128>,
        round: usize,
        claim: F128,
    ) -> Result<UnivariatePoly<F128>, SumcheckError<F128>> {
        self.timing
            .measure(1, || self.inner.prove_round(bind, round, claim))
    }
    fn finish_rounds(&mut self, bind: F128) -> Result<(), SumcheckError<F128>> {
        self.timing.measure(2, || self.inner.finish_rounds(bind))
    }
}
impl<R: ConcreteSumcheck<F128> + 'static> SumcheckKernel<F128> for TimedKernel<R> {
    type Relation = R;
    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<F128, R>,
    ) -> Result<SumcheckOutputClaims<F128, R>, SumcheckKernelError<F128>> {
        self.timing.measure(3, || self.inner.output_claims(inputs))
    }
    fn validate_derived_tables(
        &self,
        relation: &R,
        inputs: &SumcheckInputPoints<F128, R>,
        outputs: &SumcheckOutputPoints<F128, R>,
        challenges: &ConcreteSumcheckChallenges<F128, R>,
    ) -> Result<(), SumcheckKernelError<F128>> {
        self.timing.measure(3, || {
            self.inner
                .validate_derived_tables(relation, inputs, outputs, challenges)
        })
    }
    fn park_residue(self: Box<Self>, session: &mut ProofSession) {
        let Self { inner, timing } = *self;
        timing.measure(4, || inner.park_residue(session));
    }
}

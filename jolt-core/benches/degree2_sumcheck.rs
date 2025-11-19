use ark_bn254::Fr;
use ark_ff::UniformRand;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rayon::prelude::*;

use jolt_core::field::JoltField;
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::multilinear_polynomial::BindingOrder;
use jolt_core::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use jolt_core::subprotocols::sumcheck::BatchedSumcheck;
use jolt_core::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use jolt_core::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use jolt_core::transcripts::{Blake2bTranscript, Transcript};

/// Maximum degree of the per-round univariate in this toy sumcheck.
const DEGREE_BOUND: usize = 2;

/// Simple degree-2 product sumcheck over two dense multilinear polynomials p and q.
///
/// We prove that
///   sum_{x in {0,1}^ℓ} p(x) * q(x)
/// equals a given claim, where p and q are DensePolynomial-backed MLEs.
#[derive(Clone)]
struct Degree2ProductSumcheckProver<F: JoltField> {
    p: DensePolynomial<F>,
    q: DensePolynomial<F>,
    claim: F,
}

#[allow(dead_code)]
struct Degree2ProductSumcheckVerifier<F: JoltField> {
    num_rounds: usize,
    /// Claimed sum over the Boolean hypercube: sum_x p(x) * q(x).
    input_claim: F,
    /// Claimed final evaluation at the random point r: p(r) * q(r).
    final_claim: F,
}

/// Specialized helper for the degree-2 case used in this benchmark.
///
/// For a single multilinear variable binding step, the induced univariate in `t`
/// is linear: g(t) with
///   g(0) = eval at bit = 0
///   g(1) = eval at bit = 1
/// We want evaluations at t = 0 and t = 2, and use the identity
///   g(2) = 2 * g(1) - g(0).
fn dense_sumcheck_evals_degree2<F: JoltField>(
    poly: &DensePolynomial<F>,
    index: usize,
    order: BindingOrder,
) -> [F; 2] {
    debug_assert!(index < poly.len() / 2);

    let mut evals = [F::zero(); 2];
    match order {
        BindingOrder::HighToLow => {
            let eval_at_0 = poly[index];
            let eval_at_1 = poly[index + poly.len() / 2];
            let eval_at_2 = eval_at_1 + eval_at_1 - eval_at_0;
            evals[0] = eval_at_0;
            evals[1] = eval_at_2;
        }
        BindingOrder::LowToHigh => {
            let eval_at_0 = poly[2 * index];
            let eval_at_1 = poly[2 * index + 1];
            let eval_at_2 = eval_at_1 + eval_at_1 - eval_at_0;
            evals[0] = eval_at_0;
            evals[1] = eval_at_2;
        }
    };
    evals
}

fn initial_claim<F: JoltField>(p: &DensePolynomial<F>, q: &DensePolynomial<F>) -> F {
    debug_assert_eq!(p.len(), q.len());
    p.evals_ref()
        .iter()
        .zip(q.evals_ref().iter())
        .fold(F::zero(), |acc, (a, b)| acc + (*a * *b))
}

impl<F: JoltField> Degree2ProductSumcheckProver<F> {
    fn new(p: DensePolynomial<F>, q: DensePolynomial<F>) -> Self {
        let claim = initial_claim(&p, &q);
        Self { p, q, claim }
    }

    fn num_rounds(&self) -> usize {
        self.p.get_num_vars()
    }
}

#[allow(dead_code)]
impl<F: JoltField> Degree2ProductSumcheckVerifier<F> {
    /// Construct a verifier instance that is given both the initial sumcheck claim
    /// and the final evaluation claim. It does not recompute either from p or q.
    fn new(num_rounds: usize, input_claim: F, final_claim: F) -> Self {
        Self {
            num_rounds,
            input_claim,
            final_claim,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for Degree2ProductSumcheckProver<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds()
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.claim
    }

    /// Computes evaluations of the round polynomial at {0, 2} (degree-2 sumcheck).
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let half_n = self.p.len() / 2;

        let evals: [F; DEGREE_BOUND] = (0..half_n)
            .into_par_iter()
            .map(|i| {
                let p_evals =
                    dense_sumcheck_evals_degree2::<F>(&self.p, i, BindingOrder::LowToHigh);
                let q_evals =
                    dense_sumcheck_evals_degree2::<F>(&self.q, i, BindingOrder::LowToHigh);
                [
                    p_evals[0] * q_evals[0], // eval at 0
                    p_evals[1] * q_evals[1], // eval at 2
                ]
            })
            .reduce(
                || [F::zero(); DEGREE_BOUND],
                |mut acc, contrib| {
                    for j in 0..DEGREE_BOUND {
                        acc[j] += contrib[j];
                    }
                    acc
                },
            );

        evals.to_vec()
    }

    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        self.p.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.q.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        _accumulator: &mut ProverOpeningAccumulator<F>,
        _transcript: &mut T,
        _sumcheck_challenges: &[F::Challenge],
    ) {
        // No external openings for this toy benchmark.
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for Degree2ProductSumcheckVerifier<F>
{
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn input_claim(&self, _accumulator: &VerifierOpeningAccumulator<F>) -> F {
        self.input_claim
    }

    /// Expected final claim is the externally provided p(r) * q(r) at the bound point r.
    /// This verifier does not recompute it from p or q.
    fn expected_output_claim(
        &self,
        _accumulator: &VerifierOpeningAccumulator<F>,
        _sumcheck_challenges: &[F::Challenge],
    ) -> F {
        self.final_claim
    }

    fn cache_openings(
        &self,
        _accumulator: &mut VerifierOpeningAccumulator<F>,
        _transcript: &mut T,
        _sumcheck_challenges: &[F::Challenge],
    ) {
        // No openings to enqueue in this toy example.
    }
}

fn random_dense_polynomial(num_vars: usize, rng: &mut StdRng) -> DensePolynomial<Fr> {
    let len = 1usize << num_vars;
    let evals: Vec<Fr> = (0..len).map(|_| Fr::rand(rng)).collect();
    DensePolynomial::new(evals)
}

fn run_degree2_sumcheck_once(
    num_vars: usize,
    p_original: &DensePolynomial<Fr>,
    q_original: &DensePolynomial<Fr>,
) {
    // Prover works on its own mutable copies; benchmark only measures prover time.
    let mut prover = Degree2ProductSumcheckProver::new(p_original.clone(), q_original.clone());
    let mut prover_acc = ProverOpeningAccumulator::<Fr>::new(num_vars);
    let mut prover_tr = Blake2bTranscript::new(b"degree2_sumcheck");

    let instances: Vec<&mut dyn SumcheckInstanceProver<Fr, Blake2bTranscript>> =
        vec![&mut prover];
    let _ = BatchedSumcheck::prove(instances, &mut prover_acc, &mut prover_tr);
}

fn degree2_sumcheck_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("degree2_sumcheck");

    // Use a few sizes to showcase scaling; keep reasonably small for quick runs.
    for &num_vars in &[14usize, 16usize, 18usize] {
        let mut rng = StdRng::seed_from_u64(42 + num_vars as u64);
        let p = random_dense_polynomial(num_vars, &mut rng);
        let q = random_dense_polynomial(num_vars, &mut rng);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("vars={num_vars}")),
            &num_vars,
            |b, &n| {
                b.iter(|| {
                    run_degree2_sumcheck_once(n, &p, &q);
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, degree2_sumcheck_bench);
criterion_main!(benches);



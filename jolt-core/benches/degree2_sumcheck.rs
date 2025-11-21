use ark_bn254::Fr;
use ark_ff::UniformRand;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, black_box};
use rayon::prelude::*;

use jolt_core::field::{JoltField, MulTrunc};
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::multilinear_polynomial::BindingOrder;
use jolt_core::poly::opening_proof::ProverOpeningAccumulator;
use jolt_core::poly::unipoly::UniPoly;
use jolt_core::subprotocols::sumcheck::BatchedSumcheck;
use jolt_core::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use jolt_core::transcripts::{Blake2bTranscript, Transcript};

/// Maximum degree of the per-round univariate in this toy sumcheck.
const DEGREE_BOUND: usize = 2;

/// Simple degree-2 product sumcheck over two dense multilinear polynomials p and q.
///
/// We prove that
///   sum_{x in {0,1}^ℓ} p(x) * q(x)
/// equals a given claim, where p and q are DensePolynomial-backed MLEs.
#[derive(Clone, Copy)]
enum MulMode {
    /// Plain field multiplication (each product reduced immediately).
    Plain,
    /// Delayed reduction: accumulate in `F::Unreduced` and reduce once at the end.
    Unreduced,
}

#[derive(Clone)]
struct Degree2ProductSumcheckProver<F: JoltField> {
    p: DensePolynomial<F>,
    q: DensePolynomial<F>,
    claim: F,
    mul_mode: MulMode,
}

/// Specialized helper for the degree-2 case used in this benchmark.
///
/// For a single multilinear variable binding step, the induced univariate in `t`
/// is linear: g(t) with
///   g(0) = eval at bit = 0
///   g(1) = eval at bit = 1
/// We return:
///   - g(0)
///   - g(∞) = g(1) - g(0)  (the slope, i.e. the leading coefficient)
#[inline(always)]
fn dense_sumcheck_evals_degree2<F: JoltField>(
    poly: &DensePolynomial<F>,
    index: usize,
    order: BindingOrder,
) -> [F; 2] {
    debug_assert!(index < poly.len() / 2);

    match order {
        BindingOrder::HighToLow => {
            let eval_at_0 = poly[index];
            let eval_at_1 = poly[index + poly.len() / 2];
            [eval_at_0, eval_at_1 - eval_at_0]
        }
        BindingOrder::LowToHigh => {
            let eval_at_0 = poly[2 * index];
            let eval_at_1 = poly[2 * index + 1];
            [eval_at_0, eval_at_1 - eval_at_0]
        }
    }
}

/// Computes `{g(0), g(∞)}` for the degree-2 helper but keeps every intermediate in
/// `F::Unreduced<4>` so that later batch products can stay one reduction behind.
#[inline(always)]
fn dense_sumcheck_evals_degree2_unreduced<F: JoltField>(
    poly: &DensePolynomial<F>,
    index: usize,
    order: BindingOrder,
) -> [F::Unreduced::<4>; 2] {
    debug_assert!(index < poly.len() / 2);

    let (eval_at_0, eval_at_1) = match order {
        BindingOrder::HighToLow => (poly[index], poly[index + poly.len() / 2]),
        BindingOrder::LowToHigh => (poly[2 * index], poly[2 * index + 1]),
    };

    let eval_at_0_unr = *eval_at_0.as_unreduced_ref();
    let eval_at_1_unr = *eval_at_1.as_unreduced_ref();

    let mut eval_at_inf_unr = eval_at_1_unr;
    eval_at_inf_unr -= eval_at_0_unr;

    [eval_at_0_unr, eval_at_inf_unr]
}

fn initial_claim<F: JoltField>(p: &DensePolynomial<F>, q: &DensePolynomial<F>) -> F {
    debug_assert_eq!(p.len(), q.len());
    p.evals_ref()
        .iter()
        .zip(q.evals_ref().iter())
        .fold(F::zero(), |acc, (a, b)| acc + (*a * *b))
}

impl<F: JoltField> Degree2ProductSumcheckProver<F> {
    fn new(p: DensePolynomial<F>, q: DensePolynomial<F>, mul_mode: MulMode) -> Self {
        let claim = initial_claim(&p, &q);
        Self {
            p,
            q,
            claim,
            mul_mode,
        }
    }

    fn num_rounds(&self) -> usize {
        self.p.get_num_vars()
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

    /// Computes the round polynomial as a univariate `UniPoly` with degree <= 2.
    ///
    /// We first compute its evaluations at {0, 2} in a streaming fashion, then
    /// use the previous round's claim as the hint `g(0) + g(1)` to recover `g(1)`
    /// and interpolate the full polynomial.
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let half_n = self.p.len() / 2;

        // We work with the quadratic round polynomial s(t) induced by binding a single bit.
        // For each group we compute:
        //   s(0)  = p(0) * q(0)
        //   s(∞)  = (p(1) − p(0)) * (q(1) − q(0))   (leading coefficient)
        let evals_0_and_inf: [F; DEGREE_BOUND] = match self.mul_mode {
            MulMode::Plain => {
                // Baseline: regular field multiplication, reduced every time.
                (0..half_n)
                    .into_par_iter()
                    .with_min_len(512)
                    .map(|i| {
                        let p_evals =
                            dense_sumcheck_evals_degree2::<F>(&self.p, i, BindingOrder::LowToHigh);
                        let q_evals =
                            dense_sumcheck_evals_degree2::<F>(&self.q, i, BindingOrder::LowToHigh);
                        [
                            p_evals[0] * q_evals[0], // s(0)
                            p_evals[1] * q_evals[1], // s(∞)
                        ]
                    })
                    .reduce(
                        || [F::zero(); DEGREE_BOUND],
                        |mut acc, contrib| {
                            acc[0] += contrib[0];
                            acc[1] += contrib[1];
                            acc
                        },
                    )
            }
            MulMode::Unreduced => {
                // Optimized: accumulate in unreduced form and Montgomery-reduce once.
                let (sum0_unr, sum2_unr) = (0..half_n)
                    .into_par_iter()
                    .with_min_len(256)
                    .map(|i| {
                        let p_evals = dense_sumcheck_evals_degree2_unreduced::<F>(
                            &self.p,
                            i,
                            BindingOrder::LowToHigh,
                        );
                        let q_evals = dense_sumcheck_evals_degree2_unreduced::<F>(
                            &self.q,
                            i,
                            BindingOrder::LowToHigh,
                        );
                        let prod0_unr = p_evals[0].mul_trunc::<4, 9>(&q_evals[0]);
                        let prod_inf_unr = p_evals[1].mul_trunc::<4, 9>(&q_evals[1]);
                        (prod0_unr, prod_inf_unr)
                    })
                    .reduce(
                        || (Default::default(), Default::default()),
                        |(mut acc0, mut acc2), (d0, d2)| {
                            acc0 += d0;
                            acc2 += d2;
                            (acc0, acc2)
                        },
                    );

                [
                    F::from_montgomery_reduce::<9>(sum0_unr),
                    F::from_montgomery_reduce::<9>(sum2_unr),
                ]
            }
        };

        let s0 = evals_0_and_inf[0];
        let a = evals_0_and_inf[1];
        // previous_claim = s(0) + s(1) = a + b + 2c, with c = s0, a = leading coeff
        let b = previous_claim - a - s0 - s0;
        UniPoly::from_coeff(vec![s0, b, a])
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
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

fn random_dense_polynomial(num_vars: usize, rng: &mut StdRng) -> DensePolynomial<Fr> {
    let len = 1usize << num_vars;
    let evals: Vec<Fr> = (0..len).map(|_| Fr::rand(rng)).collect();
    DensePolynomial::new(evals)
}

fn run_degree2_sumcheck_once(
    num_vars: usize,
    p_original: &DensePolynomial<Fr>,
    q_original: &DensePolynomial<Fr>,
    mul_mode: MulMode,
) {
    // Prover works on its own mutable copies; benchmark only measures prover time.
    let mut prover =
        Degree2ProductSumcheckProver::new(p_original.clone(), q_original.clone(), mul_mode);
    let mut prover_acc = ProverOpeningAccumulator::<Fr>::new(num_vars);
    let mut prover_tr = Blake2bTranscript::new(b"degree2_sumcheck");

    let instances: Vec<&mut dyn SumcheckInstanceProver<Fr, Blake2bTranscript>> =
        vec![&mut prover];
    let _ = BatchedSumcheck::prove(instances, &mut prover_acc, &mut prover_tr);
}

fn degree2_sumcheck_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("degree2_sumcheck");
    // Fewer samples – focus on heavier per-iteration work.
    group.sample_size(10);

    // Use a few sizes to showcase scaling; keep reasonably small for quick runs.
    for &num_vars in &[14usize, 16usize, 18usize, 20usize, 22usize, 24usize, 26usize] {
        let mut rng = StdRng::seed_from_u64(42 + num_vars as u64);
        let p = random_dense_polynomial(num_vars, &mut rng);
        let q = random_dense_polynomial(num_vars, &mut rng);

        // Baseline: plain field multiplication.
        group.bench_with_input(
            BenchmarkId::new("baseline_plain_mul", format!("vars={num_vars}")),
            &num_vars,
            |b, &n| {
                b.iter(|| {
                    run_degree2_sumcheck_once(
                        black_box(n),
                        black_box(&p),
                        black_box(&q),
                        MulMode::Plain,
                    );
                })
            },
        );

        // Optimized multiplication: delayed reduction with mul_unreduced.
        group.bench_with_input(
            BenchmarkId::new("mul_unreduced", format!("vars={num_vars}")),
            &num_vars,
            |b, &n| {
                b.iter(|| {
                    run_degree2_sumcheck_once(
                        black_box(n),
                        black_box(&p),
                        black_box(&q),
                        MulMode::Unreduced,
                    );
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, degree2_sumcheck_bench);
criterion_main!(benches);



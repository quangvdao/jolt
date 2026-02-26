use ark_bn254::Fr;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rayon::prelude::*;

use jolt_core::field::fp128::JoltFp128;
use jolt_core::field::JoltField;
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::multilinear_polynomial::BindingOrder;
use jolt_core::poly::opening_proof::ProverOpeningAccumulator;
use jolt_core::poly::unipoly::UniPoly;
use jolt_core::subprotocols::sumcheck::BatchedSumcheck;
use jolt_core::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use jolt_core::transcripts::{Blake2bTranscript, Transcript};

const DEGREE_BOUND: usize = 2;

#[derive(Clone)]
struct Degree2ProductSumcheckProver<F: JoltField> {
    p: DensePolynomial<F>,
    q: DensePolynomial<F>,
    claim: F,
}

#[inline(always)]
fn sumcheck_evals_degree2<F: JoltField>(
    poly: &DensePolynomial<F>,
    index: usize,
) -> [F; 2] {
    debug_assert!(index < poly.len() / 2);
    let eval_at_0 = poly[2 * index];
    let eval_at_1 = poly[2 * index + 1];
    [eval_at_0, eval_at_1 - eval_at_0]
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
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for Degree2ProductSumcheckProver<F>
{
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.p.get_num_vars()
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.claim
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let half_n = self.p.len() / 2;

        let [s0, a]: [F; DEGREE_BOUND] = (0..half_n)
            .into_par_iter()
            .with_min_len(512)
            .map(|i| {
                let p_evals = sumcheck_evals_degree2::<F>(&self.p, i);
                let q_evals = sumcheck_evals_degree2::<F>(&self.q, i);
                [
                    p_evals[0] * q_evals[0],
                    p_evals[1] * q_evals[1],
                ]
            })
            .reduce(
                || [F::zero(); DEGREE_BOUND],
                |mut acc, contrib| {
                    acc[0] += contrib[0];
                    acc[1] += contrib[1];
                    acc
                },
            );

        // s(t) = s0 + b*t + a*t^2
        // previous_claim = s(0) + s(1) = s0 + (s0 + b + a) = 2*s0 + b + a
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
    }
}

fn random_dense_polynomial<F: JoltField>(num_vars: usize, rng: &mut StdRng) -> DensePolynomial<F> {
    let len = 1usize << num_vars;
    let evals: Vec<F> = (0..len).map(|_| F::random(rng)).collect();
    DensePolynomial::new(evals)
}

fn run_degree2_sumcheck<F: JoltField>(
    num_vars: usize,
    p: &DensePolynomial<F>,
    q: &DensePolynomial<F>,
) {
    let mut prover = Degree2ProductSumcheckProver::new(p.clone(), q.clone());
    let mut prover_acc = ProverOpeningAccumulator::<F>::new(num_vars);
    let mut prover_tr = Blake2bTranscript::new(b"degree2_sumcheck");
    let instances: Vec<&mut dyn SumcheckInstanceProver<F, Blake2bTranscript>> = vec![&mut prover];
    let _ = BatchedSumcheck::prove(instances, &mut prover_acc, &mut prover_tr);
}

fn degree2_sumcheck_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("degree2_sumcheck");
    group.sample_size(10);

    for &num_vars in &[14usize, 16, 18, 20, 22] {
        let mut rng_fr = StdRng::seed_from_u64(42 + num_vars as u64);
        let mut rng_fp = StdRng::seed_from_u64(42 + num_vars as u64);

        let p_fr: DensePolynomial<Fr> = random_dense_polynomial(num_vars, &mut rng_fr);
        let q_fr: DensePolynomial<Fr> = random_dense_polynomial(num_vars, &mut rng_fr);

        let p_fp: DensePolynomial<JoltFp128> = random_dense_polynomial(num_vars, &mut rng_fp);
        let q_fp: DensePolynomial<JoltFp128> = random_dense_polynomial(num_vars, &mut rng_fp);

        group.bench_with_input(
            BenchmarkId::new("BN254", format!("vars={num_vars}")),
            &num_vars,
            |b, &n| {
                b.iter(|| run_degree2_sumcheck(black_box(n), black_box(&p_fr), black_box(&q_fr)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Fp128", format!("vars={num_vars}")),
            &num_vars,
            |b, &n| {
                b.iter(|| run_degree2_sumcheck(black_box(n), black_box(&p_fp), black_box(&q_fp)))
            },
        );
    }

    group.finish();
}

criterion_group!(benches, degree2_sumcheck_bench);
criterion_main!(benches);

use ark_bn254::Fr;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rayon::prelude::*;

use jolt_core::field::fp128::JoltFp128;
use jolt_core::field::JoltField;
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::multilinear_polynomial::BindingOrder;
use jolt_core::poly::opening_proof::ProverOpeningAccumulator;
use jolt_core::poly::split_eq_poly::GruenSplitEqPolynomial;
use jolt_core::poly::unipoly::UniPoly;
use jolt_core::subprotocols::sumcheck::BatchedSumcheck;
use jolt_core::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use jolt_core::transcripts::{Blake2bTranscript, Transcript};

const DEGREE_BOUND: usize = 2;

#[derive(Clone)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct Degree2ProductSumcheckProver<F: JoltField> {
    p: DensePolynomial<F>,
    q: DensePolynomial<F>,
    claim: F,
}

#[inline(always)]
fn sumcheck_evals_degree2<F: JoltField>(poly: &DensePolynomial<F>, index: usize) -> [F; 2] {
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

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for Degree2ProductSumcheckProver<F> {
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
                [p_evals[0] * q_evals[0], p_evals[1] * q_evals[1]]
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

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
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

    for &num_vars in &[14usize, 16, 18, 20, 22, 24] {
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

/// Degree-2 sumcheck with split-eq (Gruen) optimization and unreduced accumulation.
///
/// Computes `sum_x eq(tau, x) * p(x) * q(x)` using `par_fold_out_in_unreduced`,
/// which is the pattern used by the real Jolt sumcheck inner loops (outer.rs,
/// product.rs, mles_product_sum.rs).
#[derive(Clone)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct Degree2EqProductSumcheckProver<F: JoltField> {
    p: DensePolynomial<F>,
    q: DensePolynomial<F>,
    eq: GruenSplitEqPolynomial<F>,
    claim: F,
}

fn initial_claim_eq<F: JoltField>(
    p: &DensePolynomial<F>,
    q: &DensePolynomial<F>,
    eq: &GruenSplitEqPolynomial<F>,
) -> F {
    let vals: [F; 2] = eq.par_fold_out_in_unreduced(&|g| {
        let p0 = p[2 * g];
        let p1 = p[2 * g + 1];
        let q0 = q[2 * g];
        let q1 = q[2 * g + 1];
        [p0 * q0, p1 * q1]
    });
    let r = eq.get_current_w();
    let current_scalar = eq.get_current_scalar();
    ((F::one() - r) * vals[0] + r * vals[1]) * current_scalar
}

impl<F: JoltField> Degree2EqProductSumcheckProver<F> {
    fn new(p: DensePolynomial<F>, q: DensePolynomial<F>, tau: &[F::Challenge]) -> Self {
        let eq = GruenSplitEqPolynomial::new(tau, BindingOrder::LowToHigh);
        let claim = initial_claim_eq(&p, &q, &eq);
        Self { p, q, eq, claim }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for Degree2EqProductSumcheckProver<F>
{
    fn degree(&self) -> usize {
        DEGREE_BOUND + 1
    }

    fn num_rounds(&self) -> usize {
        self.p.get_num_vars()
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.claim
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let current_scalar = self.eq.get_current_scalar();
        let r_round = self.eq.get_current_w();

        // Use split-eq unreduced accumulation: eq * p * q → degree 3
        // Evaluate g(X)/eq(X, r_round) at X=1 and X=∞ using par_fold_out_in_unreduced
        let [eval_at_1, eval_at_inf]: [F; 2] = self.eq.par_fold_out_in_unreduced(&|g| {
            let p0 = self.p[2 * g];
            let p1 = self.p[2 * g + 1];
            let q0 = self.q[2 * g];
            let q1 = self.q[2 * g + 1];
            [p1 * q1, (p1 - p0) * (q1 - q0)]
        });

        let eval_at_1 = eval_at_1 * current_scalar;
        let eval_at_inf = eval_at_inf * current_scalar;

        // Recover eval_at_0 from the claim
        let eq_at_0 = F::one() - r_round;
        let eq_at_1 = r_round;
        let eval_at_0 = (previous_claim - eq_at_1 * eval_at_1) / eq_at_0;

        // Interpolate quotient polynomial q(X) where g(X) = eq(X, r) * q(X)
        // q(X) is degree 2, known at 0, 1, ∞
        // q(0) = eval_at_0, q(1) = eval_at_1, leading coeff = eval_at_inf
        let q_a = eval_at_inf;
        let q_b = eval_at_1 - eval_at_0 - q_a;
        let q_c = eval_at_0;

        // g(X) = eq(X, r) * q(X) = ((1-r) + (2r-1)X) * (q_c + q_b*X + q_a*X^2)
        let eq_const = F::one() - r_round;
        let eq_x = r_round + r_round - F::one();

        let c0 = eq_const * q_c;
        let c1 = eq_x * q_c + eq_const * q_b;
        let c2 = eq_x * q_b + eq_const * q_a;
        let c3 = eq_x * q_a;

        UniPoly::from_coeff(vec![c0, c1, c2, c3])
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.p.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.q.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.eq.bind(r_j);
    }

    fn cache_openings(
        &self,
        _accumulator: &mut ProverOpeningAccumulator<F>,
        _transcript: &mut T,
        _sumcheck_challenges: &[F::Challenge],
    ) {
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

fn run_degree2_eq_sumcheck<F: JoltField>(
    num_vars: usize,
    p: &DensePolynomial<F>,
    q: &DensePolynomial<F>,
    tau: &[F::Challenge],
) {
    let mut prover = Degree2EqProductSumcheckProver::new(p.clone(), q.clone(), tau);
    let mut prover_acc = ProverOpeningAccumulator::<F>::new(num_vars);
    let mut prover_tr = Blake2bTranscript::new(b"degree2_eq_sumcheck");
    let instances: Vec<&mut dyn SumcheckInstanceProver<F, Blake2bTranscript>> = vec![&mut prover];
    let _ = BatchedSumcheck::prove(instances, &mut prover_acc, &mut prover_tr);
}

fn degree2_eq_sumcheck_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("degree2_eq_sumcheck");
    group.sample_size(10);

    for &num_vars in &[14usize, 16, 18, 20, 22, 24] {
        let mut rng_fr = StdRng::seed_from_u64(42 + num_vars as u64);
        let mut rng_fp = StdRng::seed_from_u64(42 + num_vars as u64);

        let p_fr: DensePolynomial<Fr> = random_dense_polynomial(num_vars, &mut rng_fr);
        let q_fr: DensePolynomial<Fr> = random_dense_polynomial(num_vars, &mut rng_fr);
        let tau_fr: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
            .map(|_| <Fr as JoltField>::Challenge::random(&mut rng_fr))
            .collect();

        let p_fp: DensePolynomial<JoltFp128> = random_dense_polynomial(num_vars, &mut rng_fp);
        let q_fp: DensePolynomial<JoltFp128> = random_dense_polynomial(num_vars, &mut rng_fp);
        let tau_fp: Vec<<JoltFp128 as JoltField>::Challenge> = (0..num_vars)
            .map(|_| <JoltFp128 as JoltField>::Challenge::random(&mut rng_fp))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("BN254", format!("vars={num_vars}")),
            &num_vars,
            |b, &n| {
                b.iter(|| {
                    run_degree2_eq_sumcheck(
                        black_box(n),
                        black_box(&p_fr),
                        black_box(&q_fr),
                        black_box(&tau_fr),
                    )
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Fp128", format!("vars={num_vars}")),
            &num_vars,
            |b, &n| {
                b.iter(|| {
                    run_degree2_eq_sumcheck(
                        black_box(n),
                        black_box(&p_fp),
                        black_box(&q_fp),
                        black_box(&tau_fp),
                    )
                })
            },
        );
    }

    group.finish();
}

fn accum_microbench(c: &mut Criterion) {
    use jolt_core::field::fp128::{FoldedAccum4, UnreducedFp128};
    use num_traits::Zero;

    let mut group = c.benchmark_group("accum_mul_add");
    group.sample_size(20);

    for &n in &[1024usize, 4096, 16384] {
        let mut rng = StdRng::seed_from_u64(99);
        let a_vals: Vec<JoltFp128> = (0..n).map(|_| JoltFp128::random(&mut rng)).collect();
        let b_vals: Vec<JoltFp128> = (0..n).map(|_| JoltFp128::random(&mut rng)).collect();

        group.bench_with_input(BenchmarkId::new("UnreducedFp128_5", n), &n, |bench, &_n| {
            bench.iter(|| {
                let mut acc = UnreducedFp128::<5>::zero();
                for i in 0..a_vals.len() {
                    let [r0, r1, r2, r3] = black_box(a_vals[i]).0.mul_wide(black_box(b_vals[i]).0);
                    acc += UnreducedFp128([r0, r1, r2, r3, 0]);
                }
                let result =
                    JoltFp128(hachi_pcs::algebra::Prime128M8M4M1M0::solinas_reduce(&acc.0));
                black_box(result)
            })
        });

        group.bench_with_input(BenchmarkId::new("FoldedAccum4", n), &n, |bench, &_n| {
            bench.iter(|| {
                let mut acc = FoldedAccum4::zero();
                for i in 0..a_vals.len() {
                    acc += FoldedAccum4::from_mul(black_box(a_vals[i]).0, black_box(b_vals[i]).0);
                }
                let result = JoltFp128(hachi_pcs::algebra::Prime128M8M4M1M0::solinas_reduce(
                    &acc.normalize(),
                ));
                black_box(result)
            })
        });
    }

    group.finish();
}

fn bn254_accum_microbench(c: &mut Criterion) {
    use ark_ff::BigInt;
    use jolt_core::field::ark::FoldedBn254Accum;
    use num_traits::Zero;

    let mut group = c.benchmark_group("bn254_accum_mul_add");
    group.sample_size(20);

    for &n in &[1024usize, 4096, 16384] {
        let mut rng = StdRng::seed_from_u64(99);
        let a_vals: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let b_vals: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let a_unreduced: Vec<BigInt<4>> = a_vals.iter().map(|x| x.to_unreduced()).collect();
        let b_unreduced: Vec<BigInt<4>> = b_vals.iter().map(|x| x.to_unreduced()).collect();

        group.bench_with_input(BenchmarkId::new("BigInt9", n), &n, |bench, &_n| {
            bench.iter(|| {
                let mut acc = BigInt::<9>::zero();
                for i in 0..a_unreduced.len() {
                    acc += black_box(a_unreduced[i]).mul_trunc::<4, 9>(&black_box(b_unreduced[i]));
                }
                let result = Fr::from_montgomery_reduce::<9, 5>(acc);
                black_box(result)
            })
        });

        group.bench_with_input(BenchmarkId::new("FoldedBn254Accum", n), &n, |bench, &_n| {
            bench.iter(|| {
                let mut acc = FoldedBn254Accum::zero();
                for i in 0..a_unreduced.len() {
                    acc +=
                        FoldedBn254Accum::from_mul(black_box(a_unreduced[i]), black_box(b_unreduced[i]));
                }
                let result = <Fr as JoltField>::reduce_product_accum(acc);
                black_box(result)
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    degree2_sumcheck_bench,
    degree2_eq_sumcheck_bench,
    accum_microbench,
    bn254_accum_microbench
);
criterion_main!(benches);

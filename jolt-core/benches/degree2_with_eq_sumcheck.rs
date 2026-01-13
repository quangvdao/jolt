use ark_bn254::Fr;
use ark_ff::UniformRand;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use jolt_core::field::JoltField;
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::multilinear_polynomial::BindingOrder;
use jolt_core::poly::opening_proof::ProverOpeningAccumulator;
use jolt_core::poly::split_eq_poly::GruenSplitEqPolynomial;
use jolt_core::poly::unipoly::UniPoly;
use jolt_core::subprotocols::sumcheck::BatchedSumcheck;
use jolt_core::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use jolt_core::transcripts::{Blake2bTranscript, Transcript};

/// Maximum degree of the per-round univariate when we include the eq polynomial.
const DEGREE_BOUND_WITH_EQ: usize = 3;

#[derive(Clone, Copy)]
enum MulMode {
    Plain,
    Unreduced,
}

#[derive(Clone)]
struct Degree2EqProductSumcheckProver<F: JoltField> {
    p: DensePolynomial<F>,
    q: DensePolynomial<F>,
    eq: GruenSplitEqPolynomial<F>,
    claim: F,
    mul_mode: MulMode,
}

impl<F: JoltField> Degree2EqProductSumcheckProver<F> {
    fn new(
        p: DensePolynomial<F>,
        q: DensePolynomial<F>,
        tau: &[F::Challenge],
        mul_mode: MulMode,
    ) -> Self {
        debug_assert_eq!(p.len(), q.len());
        let claim = p.evaluate(tau) * q.evaluate(tau);
        let eq = GruenSplitEqPolynomial::new(tau, BindingOrder::LowToHigh);
        Self {
            p,
            q,
            eq,
            claim,
            mul_mode,
        }
    }

    fn num_rounds(&self) -> usize {
        self.p.get_num_vars()
    }

    #[inline(always)]
    fn per_group_terms(&self, g: usize) -> (F, F) {
        let base = 2 * g;
        let p0 = self.p[base];
        let p1 = self.p[base + 1];
        let q0 = self.q[base];
        let q1 = self.q[base + 1];
        let constant = p0 * q0;
        let quadratic = (p1 - p0) * (q1 - q0);
        (constant, quadratic)
    }

    #[inline(always)]
    fn quadratic_endpoints_plain(&self) -> (F, F) {
        let [t0, t_inf] = self.eq.par_fold_out_in(
            || [F::zero(); 2],
            |inner, g, _x_in, e_in| {
                let (constant, quadratic) = self.per_group_terms(g);
                inner[0] += e_in * constant;
                inner[1] += e_in * quadratic;
            },
            |_x_out, e_out, inner| [e_out * inner[0], e_out * inner[1]],
            |mut a, b| {
                a[0] += b[0];
                a[1] += b[1];
                a
            },
        );
        (t0, t_inf)
    }

    #[inline(always)]
    fn quadratic_endpoints_unreduced(&self) -> (F, F) {
        let [t0, t_inf] = self.eq.par_fold_out_in_unreduced::<9, 2>(&|g| {
            let (constant, quadratic) = self.per_group_terms(g);
            [constant, quadratic]
        });
        (t0, t_inf)
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for Degree2EqProductSumcheckProver<F>
{
    fn degree(&self) -> usize {
        DEGREE_BOUND_WITH_EQ
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds()
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.claim
    }

    // #[inline(always)]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let (q_constant, q_quadratic_coeff) = match self.mul_mode {
            MulMode::Plain => self.quadratic_endpoints_plain(),
            MulMode::Unreduced => self.quadratic_endpoints_unreduced(),
        };

        self.eq
            .gruen_poly_deg_3(q_constant, q_quadratic_coeff, previous_claim)
    }

    // #[inline(always)]
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
}

fn random_dense_polynomial(num_vars: usize, rng: &mut StdRng) -> DensePolynomial<Fr> {
    let len = 1usize << num_vars;
    let evals: Vec<Fr> = (0..len).map(|_| Fr::rand(rng)).collect();
    DensePolynomial::new(evals)
}

fn random_tau(num_vars: usize, rng: &mut StdRng) -> Vec<<Fr as JoltField>::Challenge> {
    (0..num_vars)
        .map(|_| <Fr as JoltField>::Challenge::rand(rng))
        .collect()
}

fn run_degree2_eq_sumcheck_once(
    num_vars: usize,
    p_original: &DensePolynomial<Fr>,
    q_original: &DensePolynomial<Fr>,
    tau: &[<Fr as JoltField>::Challenge],
    mul_mode: MulMode,
) {
    let mut prover =
        Degree2EqProductSumcheckProver::new(p_original.clone(), q_original.clone(), tau, mul_mode);
    let mut prover_acc = ProverOpeningAccumulator::<Fr>::new(num_vars);
    let mut prover_tr = Blake2bTranscript::new(b"degree2_eq_sumcheck");
    let instances: Vec<&mut dyn SumcheckInstanceProver<Fr, Blake2bTranscript>> = vec![&mut prover];
    let _ = BatchedSumcheck::prove(instances, &mut prover_acc, &mut prover_tr);
}

fn degree2_with_eq_sumcheck_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("degree2_eq_sumcheck");
    group.sample_size(10);

    for &num_vars in &[14usize, 16usize, 18usize, 20usize, 22usize] {
        let mut rng = StdRng::seed_from_u64(1234 + num_vars as u64);
        let p = random_dense_polynomial(num_vars, &mut rng);
        let q = random_dense_polynomial(num_vars, &mut rng);
        let tau = random_tau(num_vars, &mut rng);

        group.bench_with_input(
            BenchmarkId::new("baseline_plain_mul", format!("vars={num_vars}")),
            &num_vars,
            |b, &n| {
                b.iter(|| {
                    run_degree2_eq_sumcheck_once(
                        black_box(n),
                        black_box(&p),
                        black_box(&q),
                        black_box(&tau),
                        MulMode::Plain,
                    );
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("mul_unreduced", format!("vars={num_vars}")),
            &num_vars,
            |b, &n| {
                b.iter(|| {
                    run_degree2_eq_sumcheck_once(
                        black_box(n),
                        black_box(&p),
                        black_box(&q),
                        black_box(&tau),
                        MulMode::Unreduced,
                    );
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, degree2_with_eq_sumcheck_bench);
criterion_main!(benches);

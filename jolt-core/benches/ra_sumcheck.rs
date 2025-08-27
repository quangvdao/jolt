use ark_bn254::Fr;
use ark_std::test_rng;
use ark_std::{One, Zero};
use criterion::Criterion;
use jolt_core::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::MultilinearPolynomial,
    },
    subprotocols::{
        large_degree_sumcheck::compute_eq_mle_product_univariate,
        sumcheck::SumcheckInstance,
    },
    transcripts::KeccakTranscript,
};

use jolt_core::zkvm::instruction_lookups::ra_virtual::{RAProverState, RASumCheck};
use jolt_core::field::OptimizedMul;
use jolt_core::utils::math::Math;
use jolt_core::transcripts::Transcript;
use rand_core::RngCore;

fn gen_mles<F: JoltField>(d: usize, t: usize) -> Vec<MultilinearPolynomial<F>> {
    let mut rng = test_rng();
    let mut mles = Vec::with_capacity(d);
    for _ in 0..d {
        let mut vals = Vec::with_capacity(t);
        for _ in 0..t {
            vals.push(F::from_u32(rng.next_u32()));
        }
        mles.push(MultilinearPolynomial::from(vals));
    }
    mles
}

fn naive_coeffs<F: JoltField>(
    mles: &[MultilinearPolynomial<F>],
    round: usize,
    log_t: usize,
    factor: &F,
    e_table: &[Vec<F>],
) -> Vec<F> {
    let d = mles.len();
    let mut total = vec![F::zero(); d + 1];
    for j in 0..(log_t - round - 1).pow2() {
        let j_factor = if round < log_t - 1 {
            factor.mul_1_optimized(e_table[round][j])
        } else {
            *factor
        };

        // Build D linear polynomials (constant, slope)
        let mut acc: Vec<F> = vec![F::one()];
        for (i, mle) in mles.iter().enumerate() {
            let a0 = mle.get_bound_coeff(j);
            let a1 = mle.get_bound_coeff(j + mle.len() / 2) - a0;
            let (c, s) = if i == 0 { (a0 * j_factor, a1 * j_factor) } else { (a0, a1) };

            // acc *= (c + s x)
            let mut next = vec![F::zero(); acc.len() + 1];
            for ii in 0..acc.len() {
                next[ii] += acc[ii] * c;
                next[ii + 1] += acc[ii] * s;
            }
            acc = next;
        }

        for i in 0..=d { total[i] += acc[i]; }
    }

    total
}

fn bench_round_kernel<const D: usize>(c: &mut Criterion, log_t: usize) {
    let t = 1 << log_t;
    let ra = gen_mles::<Fr>(D, t);

    let mut transcript = KeccakTranscript::new(b"rasumcheck");
    let r_cycle: Vec<Fr> = transcript.challenge_vector(log_t);
    let e_table = EqPolynomial::evals_cached_rev(&r_cycle)
        .into_iter()
        .skip(1)
        .rev()
        .skip(1)
        .collect::<Vec<_>>();

    // Optimized kernel via RASumCheck::compute_prover_message
    c.bench_function(&format!("ra_virtual_round_opt_d{}_T{}", D, t), |b| {
        b.iter_with_setup(
            || RASumCheck::<Fr> {
                r_cycle: r_cycle.clone(),
                r_address_chunks: vec![vec![]; D],
                eq_ra_claim: Fr::zero(),
                d: D,
                T: t,
                prover_state: Some(RAProverState {
                    ra_i_polys: ra.clone(),
                    E_table: e_table.clone(),
                    eq_factor: Fr::one(),
                }),
            },
            |mut s| {
                criterion::black_box(s.compute_prover_message(0, Fr::zero()));
            },
        );
    });

    // Naive kernel: same math without Karatsuba
    c.bench_function(&format!("ra_virtual_round_naive_d{}_T{}", D, t), |b| {
        b.iter_with_setup(
            || (ra.clone(), r_cycle.clone(), e_table.clone()),
            |(mles, r, et): (Vec<MultilinearPolynomial<Fr>>, Vec<Fr>, Vec<Vec<Fr>>)| {
                let coeffs = naive_coeffs(&mles, 0, log_t, &Fr::one(), &et);
                let uni = compute_eq_mle_product_univariate(coeffs, 0, &r);
                let _vals: Vec<Fr> = (0..uni.coeffs.len())
                    .filter(|i| *i != 1)
                    .map(|i| uni.evaluate(&Fr::from_u32(i as u32)))
                    .collect();
                criterion::black_box(_vals);
            },
        );
    });
}

fn bench_full_prover<const D: usize>(c: &mut Criterion, log_t: usize) {
    let t = 1 << log_t;
    let ra = gen_mles::<Fr>(D, t);

    let mut transcript = KeccakTranscript::new(b"rasumcheck");
    let r_cycle: Vec<Fr> = transcript.challenge_vector(log_t);
    let e_table = EqPolynomial::evals_cached_rev(&r_cycle)
        .into_iter()
        .skip(1)
        .rev()
        .skip(1)
        .collect::<Vec<_>>();

    c.bench_function(&format!("ra_virtual_full_prover_d{}_T{}", D, t), |b| {
        b.iter_with_setup(
            || RASumCheck::<Fr> {
                r_cycle: r_cycle.clone(),
                r_address_chunks: vec![vec![]; D],
                eq_ra_claim: Fr::zero(),
                d: D,
                T: t,
                prover_state: Some(RAProverState {
                    ra_i_polys: ra.clone(),
                    E_table: e_table.clone(),
                    eq_factor: Fr::one(),
                }),
            },
            |mut s| {
                let rounds = s.num_rounds();
                let mut _claim = s.input_claim();
                for r in 0..rounds {
                    let msg = s.compute_prover_message(r, _claim);
                    let rr = Fr::from_u32((r + 1) as u32);
                    s.bind(rr, r);
                    let _ = criterion::black_box(msg);
                }
            },
        );
    });
}

fn main() {
    let mut c = Criterion::default()
        .configure_from_args()
        .warm_up_time(std::time::Duration::from_secs(5));

    let log_t = 18; // adjust if memory-bound
    bench_round_kernel::<4>(&mut c, log_t);
    bench_round_kernel::<8>(&mut c, log_t);
    bench_round_kernel::<16>(&mut c, log_t);
    bench_full_prover::<8>(&mut c, log_t);

    c.final_summary();
}



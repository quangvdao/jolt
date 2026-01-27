//! Shift sumcheck for Multi-Miller loop packed traces.
//!
//! This prevents a prover from choosing the `*_next` columns arbitrarily.
//! Concretely, for each MultiMillerLoop instance, we enforce that:
//! - `f_next(s,x) = f(s+1,x)` for all `s != last` and all `x`,
//! - `T_next(s) = T(s+1)` component-wise for all `s != last`,
//!
//! where the packed layout is `idx = x * 128 + s` with:
//! - step vars `s ∈ {0,1}^7` (low bits),
//! - element vars `x ∈ {0,1}^4` (high bits),
//!   giving an 11-var MLE overall.
//!
//! We prove a randomized identity using Eq/EqPlusOne over the **step** variables and Eq over the
//! **element** variables, analogous to `ShiftG{1,2}ScalarMul` and `GtShift`.
//!
//! Boundary behavior:
//! - We mask out the last step `s = 2^7-1` on the LHS (so `*_next(127,·)` is unconstrained).
//! - The initial slice `*(0,·)` is also unconstrained by the shift identity (it is a boundary
//!   condition and must be constrained elsewhere if needed).

use crate::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    zkvm::witness::VirtualPolynomial,
};
use rayon::prelude::*;

use crate::zkvm::recursion::gt::shift::{
    eq_lsb_evals, eq_lsb_mle, eq_plus_one_lsb_evals, eq_plus_one_lsb_mle,
};

const NUM_VARS: usize = 11;
const STEP_VARS: usize = 7; // 128 steps
const ELEM_VARS: usize = 4; // 16 Fq12-MLE eval points
const STEP_SIZE: usize = 1 << STEP_VARS;
const ELEM_SIZE: usize = 1 << ELEM_VARS;

#[derive(Clone, Debug)]
struct ShiftPair<F: JoltField> {
    a_poly: MultilinearPolynomial<F>,
    a_next_poly: MultilinearPolynomial<F>,
    a_id: VirtualPolynomial,
    a_next_id: VirtualPolynomial,
}

fn expand_step_7_to_11<F: JoltField>(evals_7: &[F]) -> Vec<F> {
    debug_assert_eq!(evals_7.len(), STEP_SIZE);
    let mut evals_11 = vec![F::zero(); 1 << NUM_VARS];
    for x in 0..ELEM_SIZE {
        let base = x * STEP_SIZE;
        evals_11[base..base + STEP_SIZE].copy_from_slice(evals_7);
    }
    evals_11
}

fn expand_elem_4_to_11<F: JoltField>(evals_4: &[F]) -> Vec<F> {
    debug_assert_eq!(evals_4.len(), ELEM_SIZE);
    let mut evals_11 = vec![F::zero(); 1 << NUM_VARS];
    for x in 0..ELEM_SIZE {
        let base = x * STEP_SIZE;
        let v = evals_4[x];
        for s in 0..STEP_SIZE {
            evals_11[base + s] = v;
        }
    }
    evals_11
}

/// not_last(s) = 1 for all s != 2^7-1, 0 at s=127, replicated across element dimension.
fn not_last_poly_11<F: JoltField>() -> MultilinearPolynomial<F> {
    let mut evals_7 = vec![F::one(); STEP_SIZE];
    evals_7[STEP_SIZE - 1] = F::zero();
    MultilinearPolynomial::from(expand_step_7_to_11(&evals_7))
}

fn not_last_lsb_mle<F: JoltField>(y_step: &[F::Challenge]) -> F {
    // not_last(y) = 1 - ∏_i y_i  (since last index is all-ones)
    debug_assert_eq!(y_step.len(), STEP_VARS);
    let mut prod = F::one();
    for &y_i in y_step {
        let y_i_f: F = y_i.into();
        prod *= y_i_f;
    }
    F::one() - prod
}

#[derive(Clone)]
pub struct ShiftMultiMillerLoopParams {
    pub num_vars: usize,
    pub num_pairs: usize,
    pub sumcheck_id: SumcheckId,
}

impl ShiftMultiMillerLoopParams {
    pub fn new(num_pairs: usize) -> Self {
        Self {
            num_vars: NUM_VARS,
            num_pairs,
            sumcheck_id: SumcheckId::ShiftMultiMillerLoop,
        }
    }
}

pub struct ShiftMultiMillerLoopProver<F: JoltField, T: Transcript> {
    pub params: ShiftMultiMillerLoopParams,
    eq_step_poly: MultilinearPolynomial<F>,
    /// Note: `eq_plus_one_lsb_*` corresponds to Eq(r, s-1) in LSB-first step indexing.
    eq_minus_one_step_poly: MultilinearPolynomial<F>,
    eq_elem_poly: MultilinearPolynomial<F>,
    not_last_poly: MultilinearPolynomial<F>,
    pairs: Vec<ShiftPair<F>>,
    gamma: F,
    round: usize,
    pub _marker: core::marker::PhantomData<T>,
}

impl<F: JoltField, T: Transcript> ShiftMultiMillerLoopProver<F, T> {
    pub fn new(
        params: ShiftMultiMillerLoopParams,
        pairs: Vec<(VirtualPolynomial, Vec<F>, VirtualPolynomial, Vec<F>)>,
        transcript: &mut T,
    ) -> Self {
        // Sample reference points (step + element) and batching gamma.
        let step_ref: Vec<F::Challenge> = (0..STEP_VARS)
            .map(|_| transcript.challenge_scalar_optimized::<F>())
            .collect();
        let elem_ref: Vec<F::Challenge> = (0..ELEM_VARS)
            .map(|_| transcript.challenge_scalar_optimized::<F>())
            .collect();
        let gamma: F = transcript.challenge_scalar_optimized::<F>().into();

        // Build weight polynomials on the full 11-var domain (layout idx = x*128 + s).
        let eq_step_7 = eq_lsb_evals::<F>(&step_ref);
        let eq_minus_one_7 = eq_plus_one_lsb_evals::<F>(&step_ref);
        let eq_elem_4 = eq_lsb_evals::<F>(&elem_ref);

        let eq_step_poly = MultilinearPolynomial::from(expand_step_7_to_11(&eq_step_7));
        let eq_minus_one_step_poly =
            MultilinearPolynomial::from(expand_step_7_to_11(&eq_minus_one_7));
        let eq_elem_poly = MultilinearPolynomial::from(expand_elem_4_to_11(&eq_elem_4));
        let not_last_poly = not_last_poly_11::<F>();

        let pairs = pairs
            .into_iter()
            .map(|(a_id, a, a_next_id, a_next)| ShiftPair {
                a_poly: MultilinearPolynomial::from(a),
                a_next_poly: MultilinearPolynomial::from(a_next),
                a_id,
                a_next_id,
            })
            .collect::<Vec<_>>();

        debug_assert_eq!(params.num_pairs, pairs.len());

        Self {
            params,
            eq_step_poly,
            eq_minus_one_step_poly,
            eq_elem_poly,
            not_last_poly,
            pairs,
            gamma,
            round: 0,
            _marker: core::marker::PhantomData,
        }
    }
}

#[cfg(feature = "allocative")]
impl<F: JoltField, T: Transcript> allocative::Allocative for ShiftMultiMillerLoopProver<F, T> {
    fn visit<'a, 'b: 'a>(&self, _visitor: &'a mut allocative::Visitor<'b>) {}
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for ShiftMultiMillerLoopProver<F, T>
{
    fn degree(&self) -> usize {
        // max product: Eq_step * Eq_elem * not_last * A_next (4 multilinear factors)
        4
    }

    fn num_rounds(&self) -> usize {
        self.params.num_vars
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        // We prove the (batched) shift-consistency sum equals 0.
        F::zero()
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        const DEGREE: usize = 4;

        if self.pairs.is_empty() {
            return UniPoly::from_evals_and_hint(previous_claim, &[F::zero(); DEGREE]);
        }

        let half = self.pairs[0].a_poly.len() / 2;
        let gamma = self.gamma;

        let evals = (0..half)
            .into_par_iter()
            .map(|i| {
                let eqs = self
                    .eq_step_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let eqm1 = self
                    .eq_minus_one_step_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let eqx = self
                    .eq_elem_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let not_last = self
                    .not_last_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                let mut term_evals = [F::zero(); DEGREE];
                let mut gamma_power = F::one();

                for pair in &self.pairs {
                    let a = pair
                        .a_poly
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                    let an = pair
                        .a_next_poly
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                    for t in 0..DEGREE {
                        term_evals[t] += gamma_power
                            * (eqs[t] * eqx[t] * not_last[t] * an[t] - eqm1[t] * eqx[t] * a[t]);
                    }
                    gamma_power *= gamma;
                }

                term_evals
            })
            .reduce(
                || [F::zero(); DEGREE],
                |mut acc, evals| {
                    for (a, e) in acc.iter_mut().zip(evals.iter()) {
                        *a += *e;
                    }
                    acc
                },
            );

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        for pair in &mut self.pairs {
            pair.a_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
            pair.a_next_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        self.eq_step_poly
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.eq_minus_one_step_poly
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.eq_elem_poly
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.not_last_poly
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.round = round + 1;
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = OpeningPoint::<{ BIG_ENDIAN }, F>::new(sumcheck_challenges.to_vec());
        for pair in &self.pairs {
            let a_eval = pair.a_poly.get_bound_coeff(0);
            let a_next_eval = pair.a_next_poly.get_bound_coeff(0);
            accumulator.append_virtual(
                transcript,
                pair.a_id,
                self.params.sumcheck_id,
                opening_point.clone(),
                a_eval,
            );
            accumulator.append_virtual(
                transcript,
                pair.a_next_id,
                self.params.sumcheck_id,
                opening_point.clone(),
                a_next_eval,
            );
        }
    }
}

pub struct ShiftMultiMillerLoopVerifier<F: JoltField> {
    pub params: ShiftMultiMillerLoopParams,
    step_ref: Vec<F::Challenge>,
    elem_ref: Vec<F::Challenge>,
    gamma: F,
    // Same ordering as prover (for gamma batching + opening ids)
    pairs: Vec<(VirtualPolynomial, VirtualPolynomial)>,
}

impl<F: JoltField> ShiftMultiMillerLoopVerifier<F> {
    pub fn new<T: Transcript>(
        params: ShiftMultiMillerLoopParams,
        pairs: Vec<(VirtualPolynomial, VirtualPolynomial)>,
        transcript: &mut T,
    ) -> Self {
        let step_ref: Vec<F::Challenge> = (0..STEP_VARS)
            .map(|_| transcript.challenge_scalar_optimized::<F>())
            .collect();
        let elem_ref: Vec<F::Challenge> = (0..ELEM_VARS)
            .map(|_| transcript.challenge_scalar_optimized::<F>())
            .collect();
        let gamma: F = transcript.challenge_scalar_optimized::<F>().into();

        debug_assert_eq!(params.num_pairs, pairs.len());

        Self {
            params,
            step_ref,
            elem_ref,
            gamma,
            pairs,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for ShiftMultiMillerLoopVerifier<F>
{
    fn degree(&self) -> usize {
        4
    }

    fn num_rounds(&self) -> usize {
        self.params.num_vars
    }

    fn input_claim(&self, _accumulator: &VerifierOpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let y_step = &sumcheck_challenges[..STEP_VARS];
        let y_elem = &sumcheck_challenges[STEP_VARS..];

        let eqs = eq_lsb_mle::<F>(&self.step_ref, y_step);
        let eqm1 = eq_plus_one_lsb_mle::<F>(&self.step_ref, y_step);
        let eqx = eq_lsb_mle::<F>(&self.elem_ref, y_elem);
        let not_last = not_last_lsb_mle::<F>(y_step);

        let mut sum = F::zero();
        let mut gamma_power = F::one();

        for (a_id, a_next_id) in &self.pairs {
            let (_, a_eval) =
                accumulator.get_virtual_polynomial_opening(*a_id, self.params.sumcheck_id);
            let (_, a_next_eval) =
                accumulator.get_virtual_polynomial_opening(*a_next_id, self.params.sumcheck_id);
            sum += gamma_power * (eqs * eqx * not_last * a_next_eval - eqm1 * eqx * a_eval);
            gamma_power *= self.gamma;
        }

        sum
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = OpeningPoint::<{ BIG_ENDIAN }, F>::new(sumcheck_challenges.to_vec());
        for (a_id, a_next_id) in &self.pairs {
            accumulator.append_virtual(
                transcript,
                *a_id,
                self.params.sumcheck_id,
                opening_point.clone(),
            );
            accumulator.append_virtual(
                transcript,
                *a_next_id,
                self.params.sumcheck_id,
                opening_point.clone(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator},
        subprotocols::sumcheck::BatchedSumcheck,
        transcripts::{Blake2bTranscript, Transcript},
    };
    use ark_bn254::{Fq, G1Affine, G2Affine};
    use ark_ec::AffineRepr;

    #[test]
    fn test_shift_multi_miller_loop_honest_trace_verifies() {
        let p = G1Affine::generator();
        let q = G2Affine::generator();
        let steps =
            crate::poly::commitment::dory::witness::multi_miller_loop::MultiMillerLoopSteps::new(
                &[p],
                &[q],
            );
        let pair = 0usize;

        let pairs_prover: Vec<(VirtualPolynomial, Vec<Fq>, VirtualPolynomial, Vec<Fq>)> = vec![
            (
                VirtualPolynomial::multi_miller_loop_f(pair),
                steps.f_packed_mles[pair].clone(),
                VirtualPolynomial::multi_miller_loop_f_next(pair),
                steps.f_next_packed_mles[pair].clone(),
            ),
            (
                VirtualPolynomial::multi_miller_loop_t_x_c0(pair),
                steps.t_x_c0_packed_mles[pair].clone(),
                VirtualPolynomial::multi_miller_loop_t_x_c0_next(pair),
                steps.t_x_c0_next_packed_mles[pair].clone(),
            ),
            (
                VirtualPolynomial::multi_miller_loop_t_x_c1(pair),
                steps.t_x_c1_packed_mles[pair].clone(),
                VirtualPolynomial::multi_miller_loop_t_x_c1_next(pair),
                steps.t_x_c1_next_packed_mles[pair].clone(),
            ),
            (
                VirtualPolynomial::multi_miller_loop_t_y_c0(pair),
                steps.t_y_c0_packed_mles[pair].clone(),
                VirtualPolynomial::multi_miller_loop_t_y_c0_next(pair),
                steps.t_y_c0_next_packed_mles[pair].clone(),
            ),
            (
                VirtualPolynomial::multi_miller_loop_t_y_c1(pair),
                steps.t_y_c1_packed_mles[pair].clone(),
                VirtualPolynomial::multi_miller_loop_t_y_c1_next(pair),
                steps.t_y_c1_next_packed_mles[pair].clone(),
            ),
        ];

        let params = ShiftMultiMillerLoopParams::new(pairs_prover.len());
        let mut prover_transcript: Blake2bTranscript = Transcript::new(b"test_mml_shift");
        let mut prover = ShiftMultiMillerLoopProver::<Fq, Blake2bTranscript>::new(
            params.clone(),
            pairs_prover,
            &mut prover_transcript,
        );

        let mut prover_acc = ProverOpeningAccumulator::<Fq>::new(NUM_VARS);
        let (proof, _r) =
            BatchedSumcheck::prove(vec![&mut prover], &mut prover_acc, &mut prover_transcript);

        // Verifier starts with the prover-provided opening claims (as in RecursionProof.opening_claims).
        let mut verifier_acc = VerifierOpeningAccumulator::<Fq>::new(NUM_VARS);
        verifier_acc.openings = prover_acc.openings.clone();

        let mut verifier_transcript: Blake2bTranscript = Transcript::new(b"test_mml_shift");
        verifier_transcript.compare_to(prover_transcript.clone());

        let pairs_verifier: Vec<(VirtualPolynomial, VirtualPolynomial)> = vec![
            (
                VirtualPolynomial::multi_miller_loop_f(pair),
                VirtualPolynomial::multi_miller_loop_f_next(pair),
            ),
            (
                VirtualPolynomial::multi_miller_loop_t_x_c0(pair),
                VirtualPolynomial::multi_miller_loop_t_x_c0_next(pair),
            ),
            (
                VirtualPolynomial::multi_miller_loop_t_x_c1(pair),
                VirtualPolynomial::multi_miller_loop_t_x_c1_next(pair),
            ),
            (
                VirtualPolynomial::multi_miller_loop_t_y_c0(pair),
                VirtualPolynomial::multi_miller_loop_t_y_c0_next(pair),
            ),
            (
                VirtualPolynomial::multi_miller_loop_t_y_c1(pair),
                VirtualPolynomial::multi_miller_loop_t_y_c1_next(pair),
            ),
        ];
        let verifier = ShiftMultiMillerLoopVerifier::<Fq>::new(
            params,
            pairs_verifier,
            &mut verifier_transcript,
        );

        let _r2 = BatchedSumcheck::verify(
            &proof,
            vec![&verifier],
            &mut verifier_acc,
            &mut verifier_transcript,
        )
        .expect("shift sumcheck should verify for honest trace");
    }

    #[test]
    fn test_shift_multi_miller_loop_detects_corruption() {
        let p = G1Affine::generator();
        let q = G2Affine::generator();
        let steps =
            crate::poly::commitment::dory::witness::multi_miller_loop::MultiMillerLoopSteps::new(
                &[p],
                &[q],
            );
        let pair = 0usize;

        let mut bad_f_next = steps.f_next_packed_mles[pair].clone();
        // Flip one entry (deterministically).
        bad_f_next[0] += Fq::from(1u64);

        let pairs_prover: Vec<(VirtualPolynomial, Vec<Fq>, VirtualPolynomial, Vec<Fq>)> = vec![(
            VirtualPolynomial::multi_miller_loop_f(pair),
            steps.f_packed_mles[pair].clone(),
            VirtualPolynomial::multi_miller_loop_f_next(pair),
            bad_f_next,
        )];

        let params = ShiftMultiMillerLoopParams::new(pairs_prover.len());
        let mut prover_transcript: Blake2bTranscript = Transcript::new(b"test_mml_shift_bad");
        let mut prover = ShiftMultiMillerLoopProver::<Fq, Blake2bTranscript>::new(
            params.clone(),
            pairs_prover,
            &mut prover_transcript,
        );
        let mut prover_acc = ProverOpeningAccumulator::<Fq>::new(NUM_VARS);
        let (proof, _r) =
            BatchedSumcheck::prove(vec![&mut prover], &mut prover_acc, &mut prover_transcript);

        let mut verifier_acc = VerifierOpeningAccumulator::<Fq>::new(NUM_VARS);
        verifier_acc.openings = prover_acc.openings.clone();

        let mut verifier_transcript: Blake2bTranscript = Transcript::new(b"test_mml_shift_bad");
        verifier_transcript.compare_to(prover_transcript.clone());

        let pairs_verifier: Vec<(VirtualPolynomial, VirtualPolynomial)> = vec![(
            VirtualPolynomial::multi_miller_loop_f(pair),
            VirtualPolynomial::multi_miller_loop_f_next(pair),
        )];
        let verifier = ShiftMultiMillerLoopVerifier::<Fq>::new(
            params,
            pairs_verifier,
            &mut verifier_transcript,
        );

        let res = BatchedSumcheck::verify(
            &proof,
            vec![&verifier],
            &mut verifier_acc,
            &mut verifier_transcript,
        );
        assert!(res.is_err(), "corrupted shift relation should not verify");
    }
}

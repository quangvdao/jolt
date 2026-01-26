//! GT wiring (copy/boundary) sumcheck.
//!
//! Proves:
//! \[
//!   0 = \sum_{s,u \in \{0,1\}^{7}\times\{0,1\}^{4}} \mathrm{Eq}(a_{GT}, (s,u)) \cdot
//!       \sum_{e \in E_{GT}} \lambda_e \cdot \Delta_e(s,u)
//! \]
//! where each \(\Delta_e\) is a difference between a GT producer port (GTExp rho, GTMul result)
//! and a GT consumer port (GTMul lhs/rhs) or a boundary constant (GTExp base, joint commitment,
//! pairing RHS).

use ark_bn254::{Fq, Fq12};
use ark_ff::{One, Zero};
use rayon::prelude::*;

use crate::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{OpeningAccumulator, SumcheckId, VerifierOpeningAccumulator},
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    zkvm::{
        proof_serialization::PairingBoundary,
        recursion::{
            constraints::system::ConstraintSystem,
            curve::{Bn254Recursion, RecursionCurve},
            gt::shift::{eq_lsb_evals, eq_lsb_mle},
            verifier::RecursionVerifierInput,
            wiring_plan::{GtConsumer, GtProducer, GtWiringEdge},
        },
        witness::VirtualPolynomial,
    },
};

const NUM_VARS: usize = 11;
const STEP_VARS: usize = 7;
const ELEM_VARS: usize = 4;
const DEGREE: usize = 2;

fn pad_4var_to_11var_replicated(mle_4var: &[Fq]) -> Vec<Fq> {
    debug_assert_eq!(mle_4var.len(), 1 << ELEM_VARS, "expected 4-var MLE (len 16)");
    let mut mle_11var = vec![Fq::zero(); 1 << NUM_VARS];
    // Layout matches packed GT exp: index = x * 128 + s (s in low 7 bits).
    for x in 0..(1 << ELEM_VARS) {
        let v = mle_4var[x];
        for s in 0..(1 << STEP_VARS) {
            mle_11var[x * (1 << STEP_VARS) + s] = v;
        }
    }
    mle_11var
}

fn pad_7var_to_11var_replicated(mle_7var: &[Fq]) -> Vec<Fq> {
    debug_assert_eq!(mle_7var.len(), 1 << STEP_VARS, "expected 7-var MLE (len 128)");
    let mut mle_11var = vec![Fq::zero(); 1 << NUM_VARS];
    // Layout matches packed GT exp: index = x * 128 + s (s in low 7 bits).
    for x in 0..(1 << ELEM_VARS) {
        for s in 0..(1 << STEP_VARS) {
            mle_11var[x * (1 << STEP_VARS) + s] = mle_7var[s];
        }
    }
    mle_11var
}

fn eval_mle_lsb_first(mut evals: Vec<Fq>, r: &[Fq]) -> Fq {
    debug_assert_eq!(evals.len(), 1usize << r.len());
    let mut len = evals.len();
    for r_i in r {
        let half = len / 2;
        let one_minus = Fq::one() - *r_i;
        for j in 0..half {
            let a = evals[2 * j];
            let b = evals[2 * j + 1];
            evals[j] = a * one_minus + b * *r_i;
        }
        len = half;
    }
    evals[0]
}

fn eval_fq12_packed_at(fq12: &Fq12, r_elem: &[Fq]) -> Fq {
    debug_assert_eq!(r_elem.len(), ELEM_VARS);
    eval_mle_lsb_first(Bn254Recursion::fq12_to_mle(fq12), r_elem)
}

#[derive(Clone, Debug)]
pub struct WiringGtProver<T: Transcript> {
    edges: Vec<GtWiringEdge>,
    lambdas: Vec<Fq>,
    /// Eq(u, τ) replicated across step vars (11-var, u-only).
    eq_u_poly: MultilinearPolynomial<Fq>,
    /// Eq(s, 0) replicated across element vars (11-var, s-only). Used for u-only sources.
    eq_s_default: MultilinearPolynomial<Fq>,
    /// Eq(s, step_i) replicated across element vars (11-var, s-only), indexed by GTExp instance.
    ///
    /// Only populated for GTExp instances that appear as `GtProducer::GtExpRho` in `edges`.
    eq_s_by_exp: Vec<Option<MultilinearPolynomial<Fq>>>,

    // Packed GT exp rho polynomials (11-var), indexed by GTExp instance.
    rho_polys: Vec<Option<MultilinearPolynomial<Fq>>>,

    // GT mul polynomials, padded to 11 vars (constant over step vars), indexed by GTMul instance.
    mul_lhs: Vec<Option<MultilinearPolynomial<Fq>>>,
    mul_rhs: Vec<Option<MultilinearPolynomial<Fq>>>,
    mul_result: Vec<Option<MultilinearPolynomial<Fq>>>,

    // GTExp base constants, padded to 11 vars, indexed by GTExp instance.
    exp_base: Vec<Option<MultilinearPolynomial<Fq>>>,

    // Boundary constants, padded to 11 vars.
    joint_commitment: Option<MultilinearPolynomial<Fq>>,
    pairing_rhs: Option<MultilinearPolynomial<Fq>>,

    _marker: std::marker::PhantomData<T>,
}

impl<T: Transcript> WiringGtProver<T> {
    pub fn new(
        cs: &ConstraintSystem,
        edges: Vec<GtWiringEdge>,
        pairing_boundary: &PairingBoundary,
        joint_commitment: Fq12,
        transcript: &mut T,
    ) -> Self {
        // Selector for element variables: Eq(u, τ), where τ is sampled once.
        let tau: Vec<<Fq as JoltField>::Challenge> = transcript.challenge_vector_optimized::<Fq>(ELEM_VARS);
        let eq_u_4 = eq_lsb_evals::<Fq>(&tau);
        let eq_u_poly = MultilinearPolynomial::from(pad_4var_to_11var_replicated(&eq_u_4));

        // Default step selector for u-only values: Eq(s, 0).
        let s0: Vec<<Fq as JoltField>::Challenge> =
            (0..STEP_VARS).map(|_| Fq::zero().into()).collect();
        let eq_s0_7 = eq_lsb_evals::<Fq>(&s0);
        let eq_s_default = MultilinearPolynomial::from(pad_7var_to_11var_replicated(&eq_s0_7));

        // Edge batching coefficients.
        let lambdas: Vec<Fq> = transcript.challenge_vector(edges.len());

        let num_gt_exp = cs.gt_exp_witnesses.len();
        let num_gt_mul = cs.gt_mul_rows.len();

        let mut need_rho = vec![false; num_gt_exp];
        let mut need_eq_s = vec![false; num_gt_exp];
        let mut need_mul_lhs = vec![false; num_gt_mul];
        let mut need_mul_rhs = vec![false; num_gt_mul];
        let mut need_mul_result = vec![false; num_gt_mul];
        let mut need_exp_base = vec![false; num_gt_exp];
        let mut need_joint = false;
        let mut need_pairing_rhs = false;

        for e in &edges {
            match e.src {
                GtProducer::GtExpRho { instance } => {
                    if instance < num_gt_exp {
                        need_rho[instance] = true;
                        need_eq_s[instance] = true;
                    }
                }
                GtProducer::GtMulResult { instance } => {
                    if instance < num_gt_mul {
                        need_mul_result[instance] = true;
                    }
                }
            }
            match e.dst {
                GtConsumer::GtMulLhs { instance } => {
                    if instance < num_gt_mul {
                        need_mul_lhs[instance] = true;
                    }
                }
                GtConsumer::GtMulRhs { instance } => {
                    if instance < num_gt_mul {
                        need_mul_rhs[instance] = true;
                    }
                }
                GtConsumer::GtExpBase { instance } => {
                    if instance < num_gt_exp {
                        need_exp_base[instance] = true;
                    }
                }
                GtConsumer::JointCommitment => need_joint = true,
                GtConsumer::PairingBoundaryRhs => need_pairing_rhs = true,
            }
        }

        let rho_polys = need_rho
            .into_iter()
            .enumerate()
            .map(|(i, need)| {
                need.then(|| MultilinearPolynomial::from(cs.gt_exp_witnesses[i].rho_packed.clone()))
            })
            .collect();

        let mul_lhs = need_mul_lhs
            .into_iter()
            .enumerate()
            .map(|(i, need)| {
                need.then(|| {
                    MultilinearPolynomial::from(pad_4var_to_11var_replicated(&cs.gt_mul_rows[i].lhs))
                })
            })
            .collect();
        let mul_rhs = need_mul_rhs
            .into_iter()
            .enumerate()
            .map(|(i, need)| {
                need.then(|| {
                    MultilinearPolynomial::from(pad_4var_to_11var_replicated(&cs.gt_mul_rows[i].rhs))
                })
            })
            .collect();
        let mul_result = need_mul_result
            .into_iter()
            .enumerate()
            .map(|(i, need)| {
                need.then(|| {
                    MultilinearPolynomial::from(pad_4var_to_11var_replicated(
                        &cs.gt_mul_rows[i].result,
                    ))
                })
            })
            .collect();

        let exp_base = need_exp_base
            .into_iter()
            .enumerate()
            .map(|(i, need)| {
                need.then(|| {
                    let mle_4 = Bn254Recursion::fq12_to_mle(&cs.gt_exp_public_inputs[i].base);
                    MultilinearPolynomial::from(pad_4var_to_11var_replicated(&mle_4))
                })
            })
            .collect();

        let joint_commitment = need_joint.then(|| {
            let mle_4 = Bn254Recursion::fq12_to_mle(&joint_commitment);
            MultilinearPolynomial::from(pad_4var_to_11var_replicated(&mle_4))
        });
        let pairing_rhs = need_pairing_rhs.then(|| {
            let mle_4 = Bn254Recursion::fq12_to_mle(&pairing_boundary.rhs);
            MultilinearPolynomial::from(pad_4var_to_11var_replicated(&mle_4))
        });

        let eq_s_by_exp = need_eq_s
            .into_iter()
            .enumerate()
            .map(|(i, need)| {
                need.then(|| {
                    // Output step index for this GTExp instance (packed trace):
                    // rho[s_out] is the final result, where s_out = num_steps_padded - 1.
                    let s_out = cs.gt_exp_witnesses[i].num_steps.saturating_sub(1);
                    let s_bits: Vec<<Fq as JoltField>::Challenge> = (0..STEP_VARS)
                        .map(|b| {
                            let bit = ((s_out >> b) & 1) == 1;
                            if bit {
                                Fq::one().into()
                            } else {
                                Fq::zero().into()
                            }
                        })
                        .collect();
                    let eq_s_7 = eq_lsb_evals::<Fq>(&s_bits);
                    MultilinearPolynomial::from(pad_7var_to_11var_replicated(&eq_s_7))
                })
            })
            .collect();

        Self {
            edges,
            lambdas,
            eq_u_poly,
            eq_s_default,
            eq_s_by_exp,
            rho_polys,
            mul_lhs,
            mul_rhs,
            mul_result,
            exp_base,
            joint_commitment,
            pairing_rhs,
            _marker: std::marker::PhantomData,
        }
    }

    fn src_poly(&self, src: GtProducer) -> &MultilinearPolynomial<Fq> {
        match src {
            GtProducer::GtExpRho { instance } => self.rho_polys[instance].as_ref().unwrap(),
            GtProducer::GtMulResult { instance } => self.mul_result[instance].as_ref().unwrap(),
        }
    }

    fn dst_poly(&self, dst: GtConsumer) -> &MultilinearPolynomial<Fq> {
        match dst {
            GtConsumer::GtMulLhs { instance } => self.mul_lhs[instance].as_ref().unwrap(),
            GtConsumer::GtMulRhs { instance } => self.mul_rhs[instance].as_ref().unwrap(),
            GtConsumer::GtExpBase { instance } => self.exp_base[instance].as_ref().unwrap(),
            GtConsumer::JointCommitment => self.joint_commitment.as_ref().unwrap(),
            GtConsumer::PairingBoundaryRhs => self.pairing_rhs.as_ref().unwrap(),
        }
    }
}

impl<T: Transcript> SumcheckInstanceProver<Fq, T> for WiringGtProver<T> {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        NUM_VARS
    }

    fn input_claim(&self, _accumulator: &crate::poly::opening_proof::ProverOpeningAccumulator<Fq>) -> Fq {
        Fq::zero()
    }

    fn compute_message(&mut self, _round: usize, previous_claim: Fq) -> UniPoly<Fq> {
        let half = self.eq_u_poly.len() / 2;
        if half == 0 {
            return UniPoly::from_evals_and_hint(previous_claim, &[Fq::zero(); DEGREE]);
        }

        let lambdas = self.lambdas.clone();
        let edges = self.edges.clone();

        let evals = (0..half)
            .into_par_iter()
            .map(|i| {
                let eq_u_evals = self
                    .eq_u_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                let mut delta_evals = [Fq::zero(); DEGREE];
                for (lambda, edge) in lambdas.iter().zip(edges.iter()) {
                    let eq_s_poly = match edge.src {
                        GtProducer::GtExpRho { instance } => {
                            self.eq_s_by_exp[instance].as_ref().unwrap()
                        }
                        GtProducer::GtMulResult { .. } => &self.eq_s_default,
                    };
                    let eq_s_evals =
                        eq_s_poly.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                    let src_evals = self
                        .src_poly(edge.src)
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                    let dst_evals = self
                        .dst_poly(edge.dst)
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                    for t in 0..DEGREE {
                        let weight = eq_u_evals[t] * eq_s_evals[t];
                        delta_evals[t] += *lambda * weight * (src_evals[t] - dst_evals[t]);
                    }
                }
                delta_evals
            })
            .reduce(
                || [Fq::zero(); DEGREE],
                |mut acc, e| {
                    for t in 0..DEGREE {
                        acc[t] += e[t];
                    }
                    acc
                },
            );

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn ingest_challenge(&mut self, r_j: <Fq as JoltField>::Challenge, round: usize) {
        // Bind all active polynomials.
        for poly in self.rho_polys.iter_mut().flatten() {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in self.mul_lhs.iter_mut().flatten() {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in self.mul_rhs.iter_mut().flatten() {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in self.mul_result.iter_mut().flatten() {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in self.exp_base.iter_mut().flatten() {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        if let Some(poly) = self.joint_commitment.as_mut() {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        if let Some(poly) = self.pairing_rhs.as_mut() {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        self.eq_u_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.eq_s_default.bind_parallel(r_j, BindingOrder::LowToHigh);
        for poly in self.eq_s_by_exp.iter_mut().flatten() {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        let _ = round;
    }

    fn cache_openings(
        &self,
        _accumulator: &mut crate::poly::opening_proof::ProverOpeningAccumulator<Fq>,
        _transcript: &mut T,
        _sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        // No-op: this instance only *reads* cached openings from earlier Stage-2 instances.
    }
}

#[derive(Clone, Debug)]
pub struct WiringGtVerifier {
    edges: Vec<GtWiringEdge>,
    lambdas: Vec<Fq>,
    tau: Vec<<Fq as JoltField>::Challenge>,
    gt_exp_out_step: Vec<usize>,
    pairing_boundary: PairingBoundary,
    joint_commitment: Fq12,
    gt_exp_bases: Vec<Fq12>,
}

impl WiringGtVerifier {
    pub fn new<T: Transcript>(input: &RecursionVerifierInput, transcript: &mut T) -> Self {
        // Must mirror prover sampling order: τ (element selector), then λ edge coefficients.
        let tau: Vec<<Fq as JoltField>::Challenge> =
            transcript.challenge_vector_optimized::<Fq>(ELEM_VARS);
        let edges = input.wiring.gt.clone();
        let lambdas = transcript.challenge_vector(edges.len());

        // Output step index per GTExp instance: s_out = min(127, ceil(bits_len/2)).
        let max_s = (1 << STEP_VARS) - 1;
        let gt_exp_out_step = input
            .gt_exp_public_inputs
            .iter()
            .map(|p| {
                let digits_len = (p.scalar_bits.len() + 1) / 2;
                digits_len.min(max_s)
            })
            .collect();
        let gt_exp_bases = input.gt_exp_public_inputs.iter().map(|p| p.base).collect();

        Self {
            edges,
            lambdas,
            tau,
            gt_exp_out_step,
            pairing_boundary: input.pairing_boundary.clone(),
            joint_commitment: input.joint_commitment,
            gt_exp_bases,
        }
    }
}

impl<T: Transcript> SumcheckInstanceVerifier<Fq, T> for WiringGtVerifier {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        NUM_VARS
    }

    fn input_claim(&self, _acc: &VerifierOpeningAccumulator<Fq>) -> Fq {
        Fq::zero()
    }

    fn expected_output_claim(
        &self,
        acc: &VerifierOpeningAccumulator<Fq>,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) -> Fq {
        debug_assert_eq!(sumcheck_challenges.len(), NUM_VARS);
        let r_step = &sumcheck_challenges[..STEP_VARS];
        let r_elem_chal = &sumcheck_challenges[STEP_VARS..];
        let r_elem: Vec<Fq> = r_elem_chal.iter().map(|c| (*c).into()).collect();

        let eq_u = eq_lsb_mle::<Fq>(&self.tau, r_elem_chal);
        let eq_s_default: Fq = r_step
            .iter()
            .map(|c| {
                let r_b: Fq = (*c).into();
                Fq::one() - r_b
            })
            .product();

        let mut delta = Fq::zero();
        for (lambda, edge) in self.lambdas.iter().zip(self.edges.iter()) {
            let (src, eq_s) = match edge.src {
                GtProducer::GtExpRho { instance } => {
                    let s_out = self.gt_exp_out_step[instance];
                    let eq_s_val: Fq = r_step
                        .iter()
                        .enumerate()
                        .map(|(b, c)| {
                            let r_b: Fq = (*c).into();
                            if ((s_out >> b) & 1) == 1 {
                                r_b
                            } else {
                                Fq::one() - r_b
                            }
                        })
                        .product();
                    let src = acc
                        .get_virtual_polynomial_opening(
                            VirtualPolynomial::gt_exp_rho(instance),
                            SumcheckId::GtExpClaimReduction,
                        )
                        .1;
                    (src, eq_s_val)
                }
                GtProducer::GtMulResult { instance } => {
                    let src = acc
                        .get_virtual_polynomial_opening(
                            VirtualPolynomial::gt_mul_result(instance),
                            SumcheckId::GtMul,
                        )
                        .1;
                    (src, eq_s_default)
                }
            };

            let dst = match edge.dst {
                GtConsumer::GtMulLhs { instance } => acc
                    .get_virtual_polynomial_opening(VirtualPolynomial::gt_mul_lhs(instance), SumcheckId::GtMul)
                    .1,
                GtConsumer::GtMulRhs { instance } => acc
                    .get_virtual_polynomial_opening(VirtualPolynomial::gt_mul_rhs(instance), SumcheckId::GtMul)
                    .1,
                GtConsumer::GtExpBase { instance } => {
                    eval_fq12_packed_at(&self.gt_exp_bases[instance], &r_elem)
                }
                GtConsumer::JointCommitment => eval_fq12_packed_at(&self.joint_commitment, &r_elem),
                GtConsumer::PairingBoundaryRhs => {
                    eval_fq12_packed_at(&self.pairing_boundary.rhs, &r_elem)
                }
            };

            let weight = eq_u * eq_s;
            delta += *lambda * weight * (src - dst);
        }

        delta
    }

    fn cache_openings(
        &self,
        _acc: &mut VerifierOpeningAccumulator<Fq>,
        _transcript: &mut T,
        _sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        // No-op: wiring verifier reads openings cached by earlier instances.
    }
}


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
        opening_proof::{OpeningPoint, ProverOpeningAccumulator, BIG_ENDIAN},
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    zkvm::{
        proof_serialization::PairingBoundary,
        recursion::{
            constraints::system::{ConstraintLocator, ConstraintSystem, ConstraintType},
            curve::{Bn254Recursion, RecursionCurve},
            gt::indexing::k_gt,
            gt::shift::{eq_lsb_evals, eq_lsb_mle},
            verifier::RecursionVerifierInput,
            wiring_plan::{GtConsumer, GtProducer, GtWiringEdge},
        },
        witness::VirtualPolynomial,
    },
};

pub(crate) const STEP_VARS: usize = 7;
pub(crate) const ELEM_VARS: usize = 4;
pub(crate) const X_VARS: usize = STEP_VARS + ELEM_VARS;
const DEGREE: usize = 2;

fn pad_4var_to_11var_replicated(mle_4var: &[Fq]) -> Vec<Fq> {
    debug_assert_eq!(
        mle_4var.len(),
        1 << ELEM_VARS,
        "expected 4-var MLE (len 16)"
    );
    let mut mle_11var = vec![Fq::zero(); 1 << X_VARS];
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
    debug_assert_eq!(
        mle_7var.len(),
        1 << STEP_VARS,
        "expected 7-var MLE (len 128)"
    );
    let mut mle_11var = vec![Fq::zero(); 1 << X_VARS];
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
    /// Number of constraint-index variables (k), where num_constraints_padded = 2^k.
    num_c_vars: usize,
    /// Map local GTExp instance -> global constraint index (position in `constraint_types`).
    gt_exp_constraint_idx: Vec<usize>,
    /// Map local GTMul instance -> global constraint index (position in `constraint_types`).
    gt_mul_constraint_idx: Vec<usize>,
    /// Prefix weights for Eq(c, c_i) during the c-binding rounds (length = num_gt_exp).
    gt_exp_c_prefix: Vec<Fq>,
    /// Prefix weights for Eq(c, c_i) during the c-binding rounds (length = num_gt_mul).
    gt_mul_c_prefix: Vec<Fq>,
    /// Precomputed c-round source sums per edge: Σ_{s,u} Eq(u,τ)·Eq(s,s_src)·src(s,u).
    edge_src_sum: Vec<Fq>,
    /// Precomputed c-round destination sums per edge: Σ_{s,u} Eq(u,τ)·Eq(s,s_src)·dst(s,u).
    edge_dst_sum: Vec<Fq>,
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

#[cfg(feature = "allocative")]
impl<T: Transcript> allocative::Allocative for WiringGtProver<T> {
    fn visit<'a, 'b: 'a>(&self, _visitor: &'a mut allocative::Visitor<'b>) {}
}

impl<T: Transcript> WiringGtProver<T> {
    pub fn new(
        cs: &ConstraintSystem,
        edges: Vec<GtWiringEdge>,
        pairing_boundary: &PairingBoundary,
        joint_commitment: Fq12,
        transcript: &mut T,
    ) -> Self {
        // Use GT-local c-domain for end-to-end GT fusion.
        let num_c_vars = k_gt(&cs.constraint_types);

        // Map local instance -> family-local constraint index (embedded in the shared c-domain).
        let mut gt_exp_constraint_idx = vec![usize::MAX; cs.gt_exp_witnesses.len()];
        let mut gt_mul_constraint_idx = vec![usize::MAX; cs.gt_mul_rows.len()];
        let mut c_exp = 0usize;
        let mut c_mul = 0usize;
        for (global_idx, ct) in cs.constraint_types.iter().enumerate() {
            match ct {
                ConstraintType::GtExp => {
                    if let ConstraintLocator::GtExp { local } = cs.locator_by_constraint[global_idx]
                    {
                        gt_exp_constraint_idx[local] = c_exp;
                    }
                    c_exp += 1;
                }
                ConstraintType::GtMul => {
                    if let ConstraintLocator::GtMul { local } = cs.locator_by_constraint[global_idx]
                    {
                        gt_mul_constraint_idx[local] = c_mul;
                    }
                    c_mul += 1;
                }
                _ => {}
            }
        }
        debug_assert!(
            gt_exp_constraint_idx.iter().all(|&v| v != usize::MAX),
            "gt_exp_constraint_idx must be fully populated"
        );
        debug_assert!(
            gt_mul_constraint_idx.iter().all(|&v| v != usize::MAX),
            "gt_mul_constraint_idx must be fully populated"
        );

        // Selector for element variables: Eq(u, τ), where τ is sampled once.
        let tau: Vec<<Fq as JoltField>::Challenge> =
            transcript.challenge_vector_optimized::<Fq>(ELEM_VARS);
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
                GtProducer::GtExpBase { instance } => {
                    if instance < num_gt_exp {
                        need_exp_base[instance] = true;
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

        let rho_polys: Vec<Option<MultilinearPolynomial<Fq>>> = need_rho
            .into_iter()
            .enumerate()
            .map(|(i, need)| {
                need.then(|| MultilinearPolynomial::from(cs.gt_exp_witnesses[i].rho_packed.clone()))
            })
            .collect();

        let mul_lhs: Vec<Option<MultilinearPolynomial<Fq>>> = need_mul_lhs
            .into_iter()
            .enumerate()
            .map(|(i, need)| {
                need.then(|| {
                    MultilinearPolynomial::from(pad_4var_to_11var_replicated(
                        &cs.gt_mul_rows[i].lhs,
                    ))
                })
            })
            .collect();
        let mul_rhs: Vec<Option<MultilinearPolynomial<Fq>>> = need_mul_rhs
            .into_iter()
            .enumerate()
            .map(|(i, need)| {
                need.then(|| {
                    MultilinearPolynomial::from(pad_4var_to_11var_replicated(
                        &cs.gt_mul_rows[i].rhs,
                    ))
                })
            })
            .collect();
        let mul_result: Vec<Option<MultilinearPolynomial<Fq>>> = need_mul_result
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

        let exp_base: Vec<Option<MultilinearPolynomial<Fq>>> = need_exp_base
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

        let eq_s_by_exp: Vec<Option<MultilinearPolynomial<Fq>>> = need_eq_s
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

        // Precompute c-round per-edge sums (integrate out x = (s,u) while c remains unbound).
        //
        // During the first k rounds (c bits), we need message polynomials:
        //   g_j(t) = Σ_{c_{>j}, s, u} P(c_{<j}=r_{<j}, c_j=t, c_{>j}, s, u).
        //
        // Since Σ_{c_{>j}} Eq(c, c0) = 1 (over the boolean hypercube), we can compute g_j(t)
        // by combining per-edge x-sums with the current c-prefix weights.
        let mut edge_src_sum = Vec::with_capacity(edges.len());
        let mut edge_dst_sum = Vec::with_capacity(edges.len());
        for edge in &edges {
            let eq_s_poly = match edge.src {
                GtProducer::GtExpRho { instance } => eq_s_by_exp[instance].as_ref().unwrap(),
                GtProducer::GtMulResult { .. } => &eq_s_default,
                GtProducer::GtExpBase { .. } => &eq_s_default,
            };
            let src_poly = match edge.src {
                GtProducer::GtExpRho { instance } => rho_polys[instance].as_ref().unwrap(),
                GtProducer::GtMulResult { instance } => mul_result[instance].as_ref().unwrap(),
                GtProducer::GtExpBase { instance } => exp_base[instance].as_ref().unwrap(),
            };
            let dst_poly = match edge.dst {
                GtConsumer::GtMulLhs { instance } => mul_lhs[instance].as_ref().unwrap(),
                GtConsumer::GtMulRhs { instance } => mul_rhs[instance].as_ref().unwrap(),
                GtConsumer::GtExpBase { instance } => exp_base[instance].as_ref().unwrap(),
                GtConsumer::JointCommitment => joint_commitment.as_ref().unwrap(),
                GtConsumer::PairingBoundaryRhs => pairing_rhs.as_ref().unwrap(),
            };

            let mut src_sum = Fq::zero();
            let mut dst_sum = Fq::zero();
            for x_idx in 0..(1usize << X_VARS) {
                let eq_u = eq_u_poly.get_bound_coeff(x_idx);
                let eq_s = eq_s_poly.get_bound_coeff(x_idx);
                let w = eq_u * eq_s;
                if w.is_zero() {
                    continue;
                }
                src_sum += w * src_poly.get_bound_coeff(x_idx);
                dst_sum += w * dst_poly.get_bound_coeff(x_idx);
            }
            edge_src_sum.push(src_sum);
            edge_dst_sum.push(dst_sum);
        }

        Self {
            edges,
            lambdas,
            num_c_vars,
            gt_exp_constraint_idx,
            gt_mul_constraint_idx,
            gt_exp_c_prefix: vec![Fq::one(); cs.gt_exp_witnesses.len()],
            gt_mul_c_prefix: vec![Fq::one(); cs.gt_mul_rows.len()],
            edge_src_sum,
            edge_dst_sum,
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
            GtProducer::GtExpBase { instance } => self.exp_base[instance].as_ref().unwrap(),
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
        self.num_c_vars + X_VARS
    }

    fn input_claim(
        &self,
        _accumulator: &crate::poly::opening_proof::ProverOpeningAccumulator<Fq>,
    ) -> Fq {
        Fq::zero()
    }

    fn compute_message(&mut self, round: usize, previous_claim: Fq) -> UniPoly<Fq> {
        // Phase 1: bind c-vars (no bindings on 11-var polynomials).
        if round < self.num_c_vars {
            let bit_pos = round;

            // We must return evaluations at x=0 and x=2 (see `UniPoly::from_evals_and_hint`).
            // Compute g(0) and g(1), then extrapolate g(2) = 2*g(1) - g(0).
            let mut g0 = Fq::zero();
            let mut g1 = Fq::zero();
            for (edge_idx, (lambda, edge)) in self.lambdas.iter().zip(self.edges.iter()).enumerate()
            {
                let src_c = match edge.src {
                    GtProducer::GtExpRho { instance } => self.gt_exp_constraint_idx[instance],
                    GtProducer::GtMulResult { instance } => self.gt_mul_constraint_idx[instance],
                    GtProducer::GtExpBase { instance } => self.gt_exp_constraint_idx[instance],
                };
                // Destination c-index: for global constants, anchor to producer.
                let dst_c = match edge.dst {
                    GtConsumer::GtMulLhs { instance } | GtConsumer::GtMulRhs { instance } => {
                        self.gt_mul_constraint_idx[instance]
                    }
                    GtConsumer::GtExpBase { instance } => self.gt_exp_constraint_idx[instance],
                    GtConsumer::JointCommitment | GtConsumer::PairingBoundaryRhs => src_c,
                };

                let src_prefix = match edge.src {
                    GtProducer::GtExpRho { instance } => self.gt_exp_c_prefix[instance],
                    GtProducer::GtMulResult { instance } => self.gt_mul_c_prefix[instance],
                    GtProducer::GtExpBase { instance } => self.gt_exp_c_prefix[instance],
                };
                let dst_prefix = match edge.dst {
                    GtConsumer::GtMulLhs { instance } | GtConsumer::GtMulRhs { instance } => {
                        self.gt_mul_c_prefix[instance]
                    }
                    GtConsumer::GtExpBase { instance } => self.gt_exp_c_prefix[instance],
                    GtConsumer::JointCommitment | GtConsumer::PairingBoundaryRhs => src_prefix,
                };

                let src_bit = (src_c >> bit_pos) & 1;
                let dst_bit = (dst_c >> bit_pos) & 1;

                // g(0): include endpoints with bit=0, weighted by current prefix.
                if src_bit == 0 {
                    g0 += *lambda * src_prefix * self.edge_src_sum[edge_idx];
                }
                if dst_bit == 0 {
                    g0 -= *lambda * dst_prefix * self.edge_dst_sum[edge_idx];
                }

                // g(1): include endpoints with bit=1.
                if src_bit == 1 {
                    g1 += *lambda * src_prefix * self.edge_src_sum[edge_idx];
                }
                if dst_bit == 1 {
                    g1 -= *lambda * dst_prefix * self.edge_dst_sum[edge_idx];
                }
            }

            debug_assert_eq!(g0 + g1, previous_claim);

            let g2 = g1 + (g1 - g0);
            let evals = [g0, g2];
            return UniPoly::from_evals_and_hint(previous_claim, &evals);
        }

        // Phase 2: bind x-vars (step bits then elem bits), reusing the legacy wiring structure,
        // but with the c-bound weights treated as constants.
        let half = self.eq_u_poly.len() / 2;
        if half == 0 {
            return UniPoly::from_evals_and_hint(previous_claim, &[Fq::zero(); DEGREE]);
        }

        let lambdas = self.lambdas.clone();
        let edges = self.edges.clone();
        let gt_exp_w = self.gt_exp_c_prefix.clone();
        let gt_mul_w = self.gt_mul_c_prefix.clone();

        let evals = (0..half)
            .into_par_iter()
            .map(|i| {
                let eq_u_evals = self
                    .eq_u_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                let mut delta_evals = [Fq::zero(); DEGREE];
                for (lambda, edge) in lambdas.iter().zip(edges.iter()) {
                    let w_src = match edge.src {
                        GtProducer::GtExpRho { instance } => gt_exp_w[instance],
                        GtProducer::GtMulResult { instance } => gt_mul_w[instance],
                        GtProducer::GtExpBase { instance } => gt_exp_w[instance],
                    };
                    let w_dst = match edge.dst {
                        GtConsumer::GtMulLhs { instance } | GtConsumer::GtMulRhs { instance } => {
                            gt_mul_w[instance]
                        }
                        GtConsumer::GtExpBase { instance } => gt_exp_w[instance],
                        // Anchor global constants to the producer (so they share the same c-weight).
                        GtConsumer::JointCommitment | GtConsumer::PairingBoundaryRhs => w_src,
                    };

                    let eq_s_poly = match edge.src {
                        GtProducer::GtExpRho { instance } => {
                            self.eq_s_by_exp[instance].as_ref().unwrap()
                        }
                        GtProducer::GtMulResult { .. } => &self.eq_s_default,
                        GtProducer::GtExpBase { .. } => &self.eq_s_default,
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
                        delta_evals[t] +=
                            *lambda * weight * (w_src * src_evals[t] - w_dst * dst_evals[t]);
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
        // First bind the c-index (update per-instance Eq(c, c_i) prefixes).
        if round < self.num_c_vars {
            let bit_pos = round;
            let r: Fq = r_j.into();
            let one_minus = Fq::one() - r;

            for (i, w) in self.gt_exp_c_prefix.iter_mut().enumerate() {
                let c = self.gt_exp_constraint_idx[i];
                *w *= if ((c >> bit_pos) & 1) == 1 {
                    r
                } else {
                    one_minus
                };
            }
            for (i, w) in self.gt_mul_c_prefix.iter_mut().enumerate() {
                let c = self.gt_mul_constraint_idx[i];
                *w *= if ((c >> bit_pos) & 1) == 1 {
                    r
                } else {
                    one_minus
                };
            }
            return;
        }

        // Then bind x-vars (step bits then elem bits): bind all active polynomials.
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
        self.eq_s_default
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        for poly in self.eq_s_by_exp.iter_mut().flatten() {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<Fq>,
        transcript: &mut T,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        // Emit two auxiliary "fused sums" that can be used by future fused wiring backends:
        // - src_sum = Σ_e λ_e · Eq(r_c, c_src) · Eq_s(e) · src_val(e)
        // - dst_sum = Σ_e λ_e · Eq(r_c, c_dst) · Eq_s(e) · dst_val(e)
        //
        // We intentionally omit Eq(u,τ) here (the verifier multiplies it in).
        let opening_point = OpeningPoint::<BIG_ENDIAN, Fq>::new(sumcheck_challenges.to_vec());

        let mut src_sum = Fq::zero();
        let mut dst_sum = Fq::zero();

        let eq_s_default = self.eq_s_default.get_bound_coeff(0);
        for (lambda, edge) in self.lambdas.iter().zip(self.edges.iter()) {
            // c-weights (already fully bound by the c rounds).
            let w_src = match edge.src {
                GtProducer::GtExpRho { instance } => self.gt_exp_c_prefix[instance],
                GtProducer::GtMulResult { instance } => self.gt_mul_c_prefix[instance],
                GtProducer::GtExpBase { instance } => self.gt_exp_c_prefix[instance],
            };
            let w_dst = match edge.dst {
                GtConsumer::GtMulLhs { instance } | GtConsumer::GtMulRhs { instance } => {
                    self.gt_mul_c_prefix[instance]
                }
                GtConsumer::GtExpBase { instance } => self.gt_exp_c_prefix[instance],
                // Anchor globals to src.
                GtConsumer::JointCommitment | GtConsumer::PairingBoundaryRhs => w_src,
            };

            let eq_s = match edge.src {
                GtProducer::GtExpRho { instance } => self.eq_s_by_exp[instance]
                    .as_ref()
                    .unwrap()
                    .get_bound_coeff(0),
                GtProducer::GtMulResult { .. } => eq_s_default,
                GtProducer::GtExpBase { .. } => eq_s_default,
            };

            let src = self.src_poly(edge.src).get_bound_coeff(0);
            let dst = self.dst_poly(edge.dst).get_bound_coeff(0);

            src_sum += *lambda * w_src * eq_s * src;
            dst_sum += *lambda * w_dst * eq_s * dst;
        }

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::gt_wiring_src_sum(),
            SumcheckId::GtWiring,
            opening_point.clone(),
            src_sum,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::gt_wiring_dst_sum(),
            SumcheckId::GtWiring,
            opening_point,
            dst_sum,
        );
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
    /// Number of constraint-index variables (k), where num_constraints_padded = 2^k.
    num_c_vars: usize,
    /// Map local GTExp instance -> global constraint index (position in `constraint_types`).
    gt_exp_constraint_idx: Vec<usize>,
    /// Map local GTMul instance -> global constraint index (position in `constraint_types`).
    gt_mul_constraint_idx: Vec<usize>,
}

/// Legacy wiring value source: read per-instance port openings from the accumulator.
///
/// This is the *current* wiring backend and is the main coupling point with other Stage-2
/// families: if we later fuse GTExp/GTMul openings, we can swap this backend without touching
/// the wiring polynomial structure.
pub(crate) struct LegacyGtWiringValueSource<'a> {
    acc: &'a VerifierOpeningAccumulator<Fq>,
    r_step: &'a [<Fq as JoltField>::Challenge],
    r_elem: &'a [Fq],
    gt_exp_out_step: &'a [usize],
    gt_exp_bases: &'a [Fq12],
    joint_commitment: &'a Fq12,
    pairing_boundary: &'a PairingBoundary,
}

impl<'a> LegacyGtWiringValueSource<'a> {
    pub(crate) fn new(
        acc: &'a VerifierOpeningAccumulator<Fq>,
        r_step: &'a [<Fq as JoltField>::Challenge],
        r_elem: &'a [Fq],
        gt_exp_out_step: &'a [usize],
        gt_exp_bases: &'a [Fq12],
        joint_commitment: &'a Fq12,
        pairing_boundary: &'a PairingBoundary,
    ) -> Self {
        Self {
            acc,
            r_step,
            r_elem,
            gt_exp_out_step,
            gt_exp_bases,
            joint_commitment,
            pairing_boundary,
        }
    }

    #[inline]
    pub(crate) fn eq_s_for_src(&self, src: GtProducer) -> Fq {
        match src {
            GtProducer::GtExpRho { instance } => {
                let s_out = self.gt_exp_out_step[instance];
                self.r_step
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
                    .product()
            }
            // For u-only values (GTMul), the prover replicates across step vars and the verifier
            // uses Eq(s, 0) as the step selector.
            GtProducer::GtMulResult { .. } | GtProducer::GtExpBase { .. } => self
                .r_step
                .iter()
                .map(|c| {
                    let r_b: Fq = (*c).into();
                    Fq::one() - r_b
                })
                .product(),
        }
    }

    #[inline]
    pub(crate) fn src_at_r(&self, src: GtProducer) -> Fq {
        let enable_gt_fused_end_to_end = std::env::var("JOLT_RECURSION_ENABLE_GT_FUSED_END_TO_END")
            .ok()
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(false);
        match src {
            GtProducer::GtExpRho { instance } => {
                if enable_gt_fused_end_to_end {
                    self.acc
                        .get_virtual_polynomial_opening(
                            VirtualPolynomial::gt_exp_rho_fused(),
                            SumcheckId::GtExpClaimReduction,
                        )
                        .1
                } else {
                    self.acc
                        .get_virtual_polynomial_opening(
                            VirtualPolynomial::gt_exp_rho(instance),
                            SumcheckId::GtExpClaimReduction,
                        )
                        .1
                }
            }
            GtProducer::GtMulResult { instance } => {
                if enable_gt_fused_end_to_end {
                    self.acc
                        .get_virtual_polynomial_opening(
                            VirtualPolynomial::gt_mul_result_fused(),
                            SumcheckId::GtMul,
                        )
                        .1
                } else {
                    self.acc
                        .get_virtual_polynomial_opening(
                            VirtualPolynomial::gt_mul_result(instance),
                            SumcheckId::GtMul,
                        )
                        .1
                }
            }
            GtProducer::GtExpBase { instance } => {
                eval_fq12_packed_at(&self.gt_exp_bases[instance], self.r_elem)
            }
        }
    }

    #[inline]
    pub(crate) fn dst_at_r(&self, dst: GtConsumer) -> Fq {
        let enable_gt_fused_end_to_end = std::env::var("JOLT_RECURSION_ENABLE_GT_FUSED_END_TO_END")
            .ok()
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(false);
        match dst {
            GtConsumer::GtMulLhs { instance } => {
                if enable_gt_fused_end_to_end {
                    self.acc
                        .get_virtual_polynomial_opening(
                            VirtualPolynomial::gt_mul_lhs_fused(),
                            SumcheckId::GtMul,
                        )
                        .1
                } else {
                    self.acc
                        .get_virtual_polynomial_opening(
                            VirtualPolynomial::gt_mul_lhs(instance),
                            SumcheckId::GtMul,
                        )
                        .1
                }
            }
            GtConsumer::GtMulRhs { instance } => {
                if enable_gt_fused_end_to_end {
                    self.acc
                        .get_virtual_polynomial_opening(
                            VirtualPolynomial::gt_mul_rhs_fused(),
                            SumcheckId::GtMul,
                        )
                        .1
                } else {
                    self.acc
                        .get_virtual_polynomial_opening(
                            VirtualPolynomial::gt_mul_rhs(instance),
                            SumcheckId::GtMul,
                        )
                        .1
                }
            }
            GtConsumer::GtExpBase { instance } => {
                eval_fq12_packed_at(&self.gt_exp_bases[instance], self.r_elem)
            }
            GtConsumer::JointCommitment => eval_fq12_packed_at(self.joint_commitment, self.r_elem),
            GtConsumer::PairingBoundaryRhs => {
                eval_fq12_packed_at(&self.pairing_boundary.rhs, self.r_elem)
            }
        }
    }
}

impl WiringGtVerifier {
    pub fn new<T: Transcript>(input: &RecursionVerifierInput, transcript: &mut T) -> Self {
        // Must mirror prover sampling order: τ (element selector), then λ edge coefficients.
        let tau: Vec<<Fq as JoltField>::Challenge> =
            transcript.challenge_vector_optimized::<Fq>(ELEM_VARS);
        let edges = input.wiring.gt.clone();
        let lambdas = transcript.challenge_vector(edges.len());

        // Use GT-local c-domain for end-to-end GT fusion.
        let num_c_vars = k_gt(&input.constraint_types);
        let mut gt_exp_constraint_idx = Vec::with_capacity(input.gt_exp_public_inputs.len());
        let mut gt_mul_constraint_idx = Vec::new();
        let mut c_exp = 0usize;
        let mut c_mul = 0usize;
        for ct in input.constraint_types.iter() {
            match ct {
                ConstraintType::GtExp => {
                    gt_exp_constraint_idx.push(c_exp);
                    c_exp += 1;
                }
                ConstraintType::GtMul => {
                    gt_mul_constraint_idx.push(c_mul);
                    c_mul += 1;
                }
                _ => {}
            }
        }

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
            num_c_vars,
            gt_exp_constraint_idx,
            gt_mul_constraint_idx,
        }
    }

    pub(crate) fn binding(&self) -> super::wiring_binding::GtWiringBinding {
        super::wiring_binding::GtWiringBinding {
            edges: self.edges.clone(),
            lambdas: self.lambdas.clone(),
            tau: self.tau.clone(),
            gt_exp_out_step: self.gt_exp_out_step.clone(),
            gt_exp_bases: self.gt_exp_bases.clone(),
            pairing_boundary: self.pairing_boundary.clone(),
            joint_commitment: self.joint_commitment,
            num_c_vars: self.num_c_vars,
            gt_exp_constraint_idx: self.gt_exp_constraint_idx.clone(),
            gt_mul_constraint_idx: self.gt_mul_constraint_idx.clone(),
        }
    }
}

impl<T: Transcript> SumcheckInstanceVerifier<Fq, T> for WiringGtVerifier {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        self.num_c_vars + X_VARS
    }

    fn input_claim(&self, _acc: &VerifierOpeningAccumulator<Fq>) -> Fq {
        Fq::zero()
    }

    fn expected_output_claim(
        &self,
        acc: &VerifierOpeningAccumulator<Fq>,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) -> Fq {
        debug_assert_eq!(sumcheck_challenges.len(), self.num_c_vars + X_VARS);
        let r_elem_chal = &sumcheck_challenges[self.num_c_vars + STEP_VARS..];

        let eq_u = eq_lsb_mle::<Fq>(&self.tau, r_elem_chal);

        let (_, src_sum) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::gt_wiring_src_sum(),
            SumcheckId::GtWiring,
        );
        let (_, dst_sum) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::gt_wiring_dst_sum(),
            SumcheckId::GtWiring,
        );
        eq_u * (src_sum - dst_sum)
    }

    fn cache_openings(
        &self,
        acc: &mut VerifierOpeningAccumulator<Fq>,
        transcript: &mut T,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        let opening_point = OpeningPoint::<BIG_ENDIAN, Fq>::new(sumcheck_challenges.to_vec());
        // Mirror prover transcript order by appending opening points (and claims) now.
        acc.append_virtual(
            transcript,
            VirtualPolynomial::gt_wiring_src_sum(),
            SumcheckId::GtWiring,
            opening_point.clone(),
        );
        acc.append_virtual(
            transcript,
            VirtualPolynomial::gt_wiring_dst_sum(),
            SumcheckId::GtWiring,
            opening_point,
        );
    }
}

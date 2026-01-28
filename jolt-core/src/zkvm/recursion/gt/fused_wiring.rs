//! Fused GT wiring (copy/boundary) sumcheck (sound, split-k aware).
//!
//! This is the GT wiring backend used when `JOLT_RECURSION_ENABLE_GT_FUSED_END_TO_END=1`.
//! It enforces that GT producer outputs match GT consumer inputs (plus boundary constants)
//! according to the verifier-derived `WiringPlan` edge list.
//!
//! ## Statement (high level)
//! We prove:
//! \[
//!   0 \;=\;
//!   \sum_{(s,u,c)\in\{0,1\}^{7}\times\{0,1\}^{4}\times\{0,1\}^{k}}
//!     \mathrm{Eq}(u,\tau)\cdot
//!     \sum_{e\in E_{GT}} \lambda_e \cdot \mathrm{Eq}_s(e;s)\cdot
//!       \Big(
//!         \mathrm{Eq}(c, \mathrm{embed}(\mathrm{src}(e))) \cdot \mathrm{SrcPort}(e;c,s,u)
//!         \;-\;
//!         \mathrm{Eq}(c, \mathrm{embed}(\mathrm{dst}(e))) \cdot \mathrm{DstPort}(e;c,s,u)
//!       \Big),
//! \]
//! where:
//! - `s` are the packed GTExp step variables (7),
//! - `u` are the GT element variables (4),
//! - `c` is a GT-local **common** constraint-index domain of size `2^k` (`k = k_gt`),
//! - `embed(idx)` places a family-local index into the common domain by shifting left by
//!   `dummy = k - k_family` dummy low bits (so dummy bits are zero).
//! - `Eq_s(e;s)` is:
//!   - `Eq(s, s_out(exp_instance))` if the edge source is `GtExpRho`,
//!   - `Eq(s, 0)` if the edge source is `GtMulResult` (u-only values).
//!
//! Importantly, the *port polynomials* for GTExp/GTmul are treated as **multilinear in `c`**
//! (fused over the instance index). This allows the verifier to consume the already-cached
//! fused openings (e.g. `gt_exp_rho_fused()`), rather than per-instance openings.
//!
//! ## Variable order (Stage 2)
//! This instance uses the same Stage-2 GT ordering as the packed GT gadgets:
//! - first bind `x = (s,u)` (11 rounds, `BindingOrder::LowToHigh` with `s` in the low bits),
//! - then bind the GT-local `c` suffix (k rounds).
//!
//! This matches how `FusedGtExpStage2Openings*` and `FusedGtMul*` interpret the Stage-2
//! challenge vector.
//!
//! ## Soundness note
//! This backend does **not** rely on prover-emitted auxiliary wiring sums (`gt_wiring_src_sum/dst_sum`).
//! It is therefore safe to use in fully fused mode without a separate “binding safety net”.

use ark_bn254::{Fq, Fq12};
use ark_ff::{Field, One, Zero};
use rayon::prelude::*;

use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    zkvm::{
        proof_serialization::PairingBoundary,
        recursion::{
            constraints::system::{index_to_binary, ConstraintSystem},
            curve::{Bn254Recursion, RecursionCurve},
            gt::indexing::{k_exp, k_gt, k_mul},
            gt::shift::{eq_lsb_evals, eq_lsb_mle},
            verifier::RecursionVerifierInput,
            wiring_plan::{GtConsumer, GtProducer, GtWiringEdge},
        },
        witness::VirtualPolynomial,
    },
};

pub(crate) const STEP_VARS: usize = 7;
pub(crate) const ELEM_VARS: usize = 4;
pub(crate) const X_VARS: usize = STEP_VARS + ELEM_VARS; // 11

/// Max degree per variable is 2 (product of multilinears).
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

#[inline]
fn beta(dummy_bits: usize) -> Fq {
    // 2^{-dummy_bits}
    if dummy_bits == 0 {
        return Fq::one();
    }
    let two_inv = ark_ff::Field::inverse(&Fq::from_u64(2)).expect("2 has inverse in field");
    two_inv.pow([dummy_bits as u64])
}

#[derive(Clone, Debug)]
struct CPhaseState {
    /// eq(u,τ) evaluated at the bound `r_u` (scalar).
    eq_u_at_r: Fq,
    /// joint commitment evaluated at `r_u` (scalar).
    joint_at_r: Fq,
    /// pairing rhs evaluated at `r_u` (scalar).
    pairing_rhs_at_r: Fq,
    /// Per-edge eq_s evaluated at bound `r_step` (scalar).
    edge_eq_s_at_r: Vec<Fq>,
    /// Fused GTExp rho values as an MLE over the **common** c domain (k vars), replicated across dummy bits.
    rho_c: MultilinearPolynomial<Fq>,
    /// Fused GTMul ports as MLEs over the **common** c domain (k vars), replicated across dummy bits.
    mul_lhs_c: MultilinearPolynomial<Fq>,
    mul_rhs_c: MultilinearPolynomial<Fq>,
    mul_result_c: MultilinearPolynomial<Fq>,
    /// Selector polynomials Eq(c, idx_embed) as basis vectors in the common c domain.
    /// Key is `(dummy_bits, k_family, idx)` packed into a u64.
    selectors: std::collections::HashMap<u64, MultilinearPolynomial<Fq>>,
    /// Bound GTExp base constants as scalars per GTExp instance (base_i(r_u)).
    exp_base_at_r: Vec<Fq>,
}

impl CPhaseState {
    fn bind(&mut self, r_j: <Fq as JoltField>::Challenge) {
        self.rho_c.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.mul_lhs_c.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.mul_rhs_c.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.mul_result_c
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        for sel in self.selectors.values_mut() {
            sel.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
    }
}

#[derive(Clone, Debug)]
pub struct FusedWiringGtProver<T: Transcript> {
    edges: Vec<GtWiringEdge>,
    lambdas: Vec<Fq>,
    /// Common GT-local c domain size (k = k_gt).
    num_c_vars: usize,
    k_exp: usize,
    k_mul: usize,

    // --- x-phase polynomials (11-var) ---
    /// Eq(u, τ) replicated across step vars (11-var, u-only).
    eq_u_poly: MultilinearPolynomial<Fq>,
    /// Eq(s, 0) replicated across element vars (11-var, s-only). Used for u-only sources.
    eq_s_default: MultilinearPolynomial<Fq>,
    /// Eq(s, step_i) replicated across element vars (11-var, s-only), indexed by GTExp instance.
    eq_s_by_exp: Vec<Option<MultilinearPolynomial<Fq>>>,

    /// Packed GT exp rho polynomials (11-var), indexed by GTExp instance.
    rho_polys: Vec<Option<MultilinearPolynomial<Fq>>>,

    /// GT mul polynomials, padded to 11 vars (constant over step vars), indexed by GTMul instance.
    mul_lhs: Vec<Option<MultilinearPolynomial<Fq>>>,
    mul_rhs: Vec<Option<MultilinearPolynomial<Fq>>>,
    mul_result: Vec<Option<MultilinearPolynomial<Fq>>>,

    /// GTExp base constants, padded to 11 vars, indexed by GTExp instance.
    exp_base: Vec<Option<MultilinearPolynomial<Fq>>>,

    /// Boundary constants, padded to 11 vars.
    joint_commitment: Option<MultilinearPolynomial<Fq>>,
    pairing_rhs: Option<MultilinearPolynomial<Fq>>,

    /// Lazily built when the sumcheck enters the c rounds.
    c_state: Option<CPhaseState>,

    _marker: std::marker::PhantomData<T>,
}

#[cfg(feature = "allocative")]
impl<T: Transcript> allocative::Allocative for FusedWiringGtProver<T> {
    fn visit<'a, 'b: 'a>(&self, _visitor: &'a mut allocative::Visitor<'b>) {}
}

impl<T: Transcript> FusedWiringGtProver<T> {
    pub fn new(
        cs: &ConstraintSystem,
        edges: Vec<GtWiringEdge>,
        pairing_boundary: &PairingBoundary,
        joint_commitment: Fq12,
        transcript: &mut T,
    ) -> Self {
        let num_c_vars = k_gt(&cs.constraint_types);
        let k_exp = k_exp(&cs.constraint_types);
        let k_mul = k_mul(&cs.constraint_types);

        // Must mirror verifier sampling order: τ (element selector), then λ edge coefficients.
        let tau: Vec<<Fq as JoltField>::Challenge> =
            transcript.challenge_vector_optimized::<Fq>(ELEM_VARS);
        let lambdas: Vec<Fq> = transcript.challenge_vector(edges.len());

        let eq_u_4 = eq_lsb_evals::<Fq>(&tau);
        let eq_u_poly = MultilinearPolynomial::from(pad_4var_to_11var_replicated(&eq_u_4));

        // Default step selector for u-only values: Eq(s, 0).
        let s0: Vec<<Fq as JoltField>::Challenge> =
            (0..STEP_VARS).map(|_| Fq::zero().into()).collect();
        let eq_s0_7 = eq_lsb_evals::<Fq>(&s0);
        let eq_s_default = MultilinearPolynomial::from(pad_7var_to_11var_replicated(&eq_s0_7));

        // IMPORTANT: in the fully fused wiring backend, the verifier consumes fused openings
        // like `gt_mul_lhs_fused()` / `gt_exp_rho_fused()`. Those are evaluations of *fully fused*
        // (c-batched) polynomials at the Stage-2 point, which in general depend on **all**
        // instances in the family (even if a particular instance is not referenced by any edge).
        //
        // Therefore, we must materialize the full family tables here (no "need_*" filtering),
        // otherwise we silently change the polynomial being proved and Stage-2 sumcheck
        // verification fails.
        let num_gt_exp = cs.gt_exp_witnesses.len();
        let num_gt_mul = cs.gt_mul_rows.len();
        let need_joint = edges
            .iter()
            .any(|e| matches!(e.dst, GtConsumer::JointCommitment));
        let need_pairing_rhs = edges
            .iter()
            .any(|e| matches!(e.dst, GtConsumer::PairingBoundaryRhs));

        let rho_polys: Vec<Option<MultilinearPolynomial<Fq>>> = (0..num_gt_exp)
            .map(|i| {
                Some(MultilinearPolynomial::from(
                    cs.gt_exp_witnesses[i].rho_packed.clone(),
                ))
            })
            .collect();

        let mul_lhs: Vec<Option<MultilinearPolynomial<Fq>>> = (0..num_gt_mul)
            .map(|i| {
                Some(MultilinearPolynomial::from(pad_4var_to_11var_replicated(
                    &cs.gt_mul_rows[i].lhs,
                )))
            })
            .collect();
        let mul_rhs: Vec<Option<MultilinearPolynomial<Fq>>> = (0..num_gt_mul)
            .map(|i| {
                Some(MultilinearPolynomial::from(pad_4var_to_11var_replicated(
                    &cs.gt_mul_rows[i].rhs,
                )))
            })
            .collect();
        let mul_result: Vec<Option<MultilinearPolynomial<Fq>>> = (0..num_gt_mul)
            .map(|i| {
                Some(MultilinearPolynomial::from(pad_4var_to_11var_replicated(
                    &cs.gt_mul_rows[i].result,
                )))
            })
            .collect();

        let exp_base: Vec<Option<MultilinearPolynomial<Fq>>> = (0..num_gt_exp)
            .map(|i| {
                let mle_4 = Bn254Recursion::fq12_to_mle(&cs.gt_exp_public_inputs[i].base);
                Some(MultilinearPolynomial::from(pad_4var_to_11var_replicated(
                    &mle_4,
                )))
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

        let eq_s_by_exp: Vec<Option<MultilinearPolynomial<Fq>>> = (0..num_gt_exp)
            .map(|i| {
                // Output step index for this GTExp instance (packed trace):
                // rho[s_out] is the final result.
                //
                // We derive `s_out` from public inputs to match the verifier:
                // s_out = min(127, ceil(bits_len/2)).
                let max_s = (1 << STEP_VARS) - 1;
                let digits_len = (cs.gt_exp_public_inputs[i].scalar_bits.len() + 1) / 2;
                let s_out = digits_len.min(max_s);
                let s_bits: Vec<<Fq as JoltField>::Challenge> = (0..STEP_VARS)
                    .map(|b| {
                        let bit = ((s_out >> b) & 1) == 1;
                        if bit { Fq::one() } else { Fq::zero() }.into()
                    })
                    .collect();
                let eq_s_7 = eq_lsb_evals::<Fq>(&s_bits);
                Some(MultilinearPolynomial::from(pad_7var_to_11var_replicated(
                    &eq_s_7,
                )))
            })
            .collect();

        Self {
            edges,
            lambdas,
            num_c_vars,
            k_exp,
            k_mul,
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
            c_state: None,
            _marker: std::marker::PhantomData,
        }
    }

    fn eq_s_poly_for_src(&self, src: GtProducer) -> &MultilinearPolynomial<Fq> {
        match src {
            GtProducer::GtExpRho { instance } => self.eq_s_by_exp[instance].as_ref().unwrap(),
            GtProducer::GtMulResult { .. } | GtProducer::GtExpBase { .. } => &self.eq_s_default,
        }
    }

    fn src_poly_x(&self, src: GtProducer) -> &MultilinearPolynomial<Fq> {
        match src {
            GtProducer::GtExpRho { instance } => self.rho_polys[instance].as_ref().unwrap(),
            GtProducer::GtMulResult { instance } => self.mul_result[instance].as_ref().unwrap(),
            GtProducer::GtExpBase { instance } => self.exp_base[instance].as_ref().unwrap(),
        }
    }

    fn dst_poly_x(&self, dst: GtConsumer) -> &MultilinearPolynomial<Fq> {
        match dst {
            GtConsumer::GtMulLhs { instance } => self.mul_lhs[instance].as_ref().unwrap(),
            GtConsumer::GtMulRhs { instance } => self.mul_rhs[instance].as_ref().unwrap(),
            GtConsumer::GtExpBase { instance } => self.exp_base[instance].as_ref().unwrap(),
            GtConsumer::JointCommitment => self.joint_commitment.as_ref().unwrap(),
            GtConsumer::PairingBoundaryRhs => self.pairing_rhs.as_ref().unwrap(),
        }
    }

    fn ensure_c_state(&mut self) {
        if self.c_state.is_some() {
            return;
        }

        // x has been fully bound (all 11 rounds), so x-polynomials are scalars now.
        let eq_u_at_r = self.eq_u_poly.get_bound_coeff(0);
        let joint_at_r = self
            .joint_commitment
            .as_ref()
            .map(|p| p.get_bound_coeff(0))
            .unwrap_or_else(Fq::zero);
        let pairing_rhs_at_r = self
            .pairing_rhs
            .as_ref()
            .map(|p| p.get_bound_coeff(0))
            .unwrap_or_else(Fq::zero);

        let num_gt_exp = self.rho_polys.len();
        let num_gt_mul = self.mul_result.len();

        let exp_base_at_r: Vec<Fq> = (0..num_gt_exp)
            .map(|i| {
                self.exp_base[i]
                    .as_ref()
                    .map(|p| p.get_bound_coeff(0))
                    .unwrap_or(Fq::zero())
            })
            .collect();

        // Scalars per instance at r_x.
        let rho_at_r: Vec<Fq> = (0..num_gt_exp)
            .map(|i| {
                self.rho_polys[i]
                    .as_ref()
                    .map(|p| p.get_bound_coeff(0))
                    .unwrap_or(Fq::zero())
            })
            .collect();
        let mul_lhs_at_r: Vec<Fq> = (0..num_gt_mul)
            .map(|i| {
                self.mul_lhs[i]
                    .as_ref()
                    .map(|p| p.get_bound_coeff(0))
                    .unwrap_or(Fq::zero())
            })
            .collect();
        let mul_rhs_at_r: Vec<Fq> = (0..num_gt_mul)
            .map(|i| {
                self.mul_rhs[i]
                    .as_ref()
                    .map(|p| p.get_bound_coeff(0))
                    .unwrap_or(Fq::zero())
            })
            .collect();
        let mul_result_at_r: Vec<Fq> = (0..num_gt_mul)
            .map(|i| {
                self.mul_result[i]
                    .as_ref()
                    .map(|p| p.get_bound_coeff(0))
                    .unwrap_or(Fq::zero())
            })
            .collect();

        // Common c-domain tables (replicate across dummy low bits).
        let k = self.num_c_vars;
        let n = 1usize << k;

        let dummy_exp = k.saturating_sub(self.k_exp);
        let dummy_mul = k.saturating_sub(self.k_mul);

        let mut rho_c = vec![Fq::zero(); n];
        for c in 0..n {
            // Split-k convention: the first `dummy_exp` *low* bits are dummy, so the family
            // instance index lives in the remaining high bits. Replicate across dummy bits.
            let family_idx = c >> dummy_exp;
            if family_idx < rho_at_r.len() {
                rho_c[c] = rho_at_r[family_idx];
            }
        }

        let mut mul_lhs_c = vec![Fq::zero(); n];
        let mut mul_rhs_c = vec![Fq::zero(); n];
        let mut mul_result_c = vec![Fq::zero(); n];
        for c in 0..n {
            // Same convention for GTMul family.
            let family_idx = c >> dummy_mul;
            if family_idx < mul_result_at_r.len() {
                mul_lhs_c[c] = mul_lhs_at_r[family_idx];
                mul_rhs_c[c] = mul_rhs_at_r[family_idx];
                mul_result_c[c] = mul_result_at_r[family_idx];
            }
        }

        // Precompute per-edge eq_s(r_step) scalars (x-bound).
        let eq_s_default_at_r = self.eq_s_default.get_bound_coeff(0);
        let edge_eq_s_at_r: Vec<Fq> = self
            .edges
            .iter()
            .map(|e| match e.src {
                GtProducer::GtExpRho { instance } => self.eq_s_by_exp[instance]
                    .as_ref()
                    .unwrap()
                    .get_bound_coeff(0),
                GtProducer::GtMulResult { .. } | GtProducer::GtExpBase { .. } => eq_s_default_at_r,
            })
            .collect();

        // Build selector polynomials that depend only on the family tail bits:
        // sel(c) = 1 if (c_tail == idx), 0 otherwise. This is replicated across the dummy
        // low bits, and requires the Option-B normalization factors (beta) in evaluation.
        let mut selectors: std::collections::HashMap<u64, MultilinearPolynomial<Fq>> =
            std::collections::HashMap::new();
        let mut ensure_selector = |dummy: usize, k_family: usize, idx: usize| {
            let key = ((dummy as u64) << 48) | ((k_family as u64) << 32) | (idx as u64);
            selectors.entry(key).or_insert_with(|| {
                let mut v = vec![Fq::zero(); n];
                for c in 0..n {
                    // Selector depends only on the family index (high bits); it is replicated
                    // across the `dummy` low bits. The `beta(dummy)=2^{-dummy}` factor
                    // normalizes this replication in the sumcheck.
                    let family_idx = c >> dummy;
                    if family_idx == idx {
                        v[c] = Fq::one();
                    }
                }
                MultilinearPolynomial::from(v)
            });
            key
        };

        // Pre-register all selectors used by any edge endpoint.
        for edge in &self.edges {
            match edge.src {
                GtProducer::GtExpRho { instance } => {
                    let _ = ensure_selector(dummy_exp, self.k_exp, instance);
                }
                GtProducer::GtMulResult { instance } => {
                    let _ = ensure_selector(dummy_mul, self.k_mul, instance);
                }
                GtProducer::GtExpBase { instance } => {
                    let _ = ensure_selector(dummy_exp, self.k_exp, instance);
                }
            }
            match edge.dst {
                GtConsumer::GtMulLhs { instance } | GtConsumer::GtMulRhs { instance } => {
                    let _ = ensure_selector(dummy_mul, self.k_mul, instance);
                }
                GtConsumer::GtExpBase { instance } => {
                    let _ = ensure_selector(dummy_exp, self.k_exp, instance);
                }
                GtConsumer::JointCommitment | GtConsumer::PairingBoundaryRhs => {
                    // Anchored to src; no additional selector required here.
                }
            }
        }

        self.c_state = Some(CPhaseState {
            eq_u_at_r,
            joint_at_r,
            pairing_rhs_at_r,
            edge_eq_s_at_r,
            rho_c: MultilinearPolynomial::from(rho_c),
            mul_lhs_c: MultilinearPolynomial::from(mul_lhs_c),
            mul_rhs_c: MultilinearPolynomial::from(mul_rhs_c),
            mul_result_c: MultilinearPolynomial::from(mul_result_c),
            selectors,
            exp_base_at_r,
        });
    }
}

impl<T: Transcript> SumcheckInstanceProver<Fq, T> for FusedWiringGtProver<T> {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        X_VARS + self.num_c_vars
    }

    fn input_claim(&self, _acc: &ProverOpeningAccumulator<Fq>) -> Fq {
        Fq::zero()
    }

    fn compute_message(&mut self, round: usize, previous_claim: Fq) -> UniPoly<Fq> {
        // Phase 1: bind x = (s,u) (11 rounds). c variables are still summed over.
        if round < X_VARS {
            let num_remaining = self.eq_u_poly.get_num_vars();
            debug_assert!(num_remaining > 0);
            let half = 1usize << (num_remaining - 1);

            // Parallelize over half-cube indices.
            let evals = (0..half)
                .into_par_iter()
                .map(|i| {
                    let eq_u_evals = self
                        .eq_u_poly
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                    let mut delta_evals = [Fq::zero(); DEGREE];
                    for (lambda, edge) in self.lambdas.iter().zip(self.edges.iter()) {
                        let eq_s_poly = self.eq_s_poly_for_src(edge.src);
                        let eq_s_evals =
                            eq_s_poly.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                        let src_evals = self
                            .src_poly_x(edge.src)
                            .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                        let dst_evals = self
                            .dst_poly_x(edge.dst)
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
                    |mut acc, arr| {
                        for t in 0..DEGREE {
                            acc[t] += arr[t];
                        }
                        acc
                    },
                );

            UniPoly::from_evals_and_hint(previous_claim, &evals)
        } else {
            // Phase 2: bind c (k rounds), after x has been fully bound.
            self.ensure_c_state();
            let state = self.c_state.as_ref().expect("c_state must be initialized");

            let num_remaining = state.rho_c.get_num_vars();
            debug_assert!(num_remaining > 0);
            let half = 1usize << (num_remaining - 1);

            let dummy_exp = self.num_c_vars.saturating_sub(self.k_exp);
            let dummy_mul = self.num_c_vars.saturating_sub(self.k_mul);
            let beta_exp = beta(dummy_exp);
            let beta_mul = beta(dummy_mul);
            let selector_key = |dummy: usize, k_family: usize, idx: usize| -> u64 {
                ((dummy as u64) << 48) | ((k_family as u64) << 32) | (idx as u64)
            };

            let evals = (0..half)
                .into_par_iter()
                .map(|i| {
                    let rho_e = state
                        .rho_c
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                    let mul_lhs_e = state
                        .mul_lhs_c
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                    let mul_rhs_e = state
                        .mul_rhs_c
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                    let mul_out_e = state
                        .mul_result_c
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                    let mut out = [Fq::zero(); DEGREE];

                    for (edge_idx, edge) in self.edges.iter().enumerate() {
                        let lambda = self.lambdas[edge_idx];
                        let eq_s_at_r = state.edge_eq_s_at_r[edge_idx];
                        let edge_coeff = state.eq_u_at_r * lambda * eq_s_at_r;

                        // Selectors for src/dst endpoints (depend only on the family tail bits).
                        let (sel_src_key, src_port_e, beta_src) = match edge.src {
                            GtProducer::GtExpRho { instance } => (
                                selector_key(dummy_exp, self.k_exp, instance),
                                rho_e,
                                beta_exp,
                            ),
                            GtProducer::GtMulResult { instance } => (
                                selector_key(dummy_mul, self.k_mul, instance),
                                mul_out_e,
                                beta_mul,
                            ),
                            GtProducer::GtExpBase { instance } => (
                                selector_key(dummy_exp, self.k_exp, instance),
                                // Constant source; we ignore `src_port_e` and use `exp_base_at_r` below.
                                rho_e,
                                beta_exp,
                            ),
                        };
                        let sel_src = state
                            .selectors
                            .get(&sel_src_key)
                            .expect("missing selector for src");
                        let sel_src_e =
                            sel_src.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                        let (sel_dst_key, beta_dst) = match edge.dst {
                            GtConsumer::GtMulLhs { instance }
                            | GtConsumer::GtMulRhs { instance } => {
                                (selector_key(dummy_mul, self.k_mul, instance), beta_mul)
                            }
                            GtConsumer::GtExpBase { instance } => {
                                (selector_key(dummy_exp, self.k_exp, instance), beta_exp)
                            }
                            // Anchor globals to src family.
                            GtConsumer::JointCommitment | GtConsumer::PairingBoundaryRhs => {
                                (sel_src_key, beta_src)
                            }
                        };
                        let sel_dst = state
                            .selectors
                            .get(&sel_dst_key)
                            .expect("missing selector for dst");
                        let sel_dst_e =
                            sel_dst.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                        let dst_is_mul_lhs = matches!(edge.dst, GtConsumer::GtMulLhs { .. });
                        let dst_is_mul_rhs = matches!(edge.dst, GtConsumer::GtMulRhs { .. });

                        let src_const = match edge.src {
                            GtProducer::GtExpBase { instance } => Some(state.exp_base_at_r[instance]),
                            _ => None,
                        };
                        for t in 0..DEGREE {
                            let src_val = src_const.unwrap_or(src_port_e[t]);
                            let src_term = sel_src_e[t] * src_val;

                            let dst_term = if dst_is_mul_lhs {
                                sel_dst_e[t] * mul_lhs_e[t]
                            } else if dst_is_mul_rhs {
                                sel_dst_e[t] * mul_rhs_e[t]
                            } else {
                                // Boundary constants.
                                let const_val = match edge.dst {
                                    GtConsumer::GtExpBase { instance } => {
                                        state.exp_base_at_r[instance]
                                    }
                                    GtConsumer::JointCommitment => state.joint_at_r,
                                    GtConsumer::PairingBoundaryRhs => state.pairing_rhs_at_r,
                                    _ => unreachable!("handled above"),
                                };
                                sel_dst_e[t] * const_val
                            };

                            out[t] += edge_coeff * (beta_src * src_term - beta_dst * dst_term);
                        }
                    }

                    out
                })
                .reduce(
                    || [Fq::zero(); DEGREE],
                    |mut acc, arr| {
                        for t in 0..DEGREE {
                            acc[t] += arr[t];
                        }
                        acc
                    },
                );

            UniPoly::from_evals_and_hint(previous_claim, &evals)
        }
    }

    fn ingest_challenge(&mut self, r_j: <Fq as JoltField>::Challenge, round: usize) {
        if round < X_VARS {
            // Bind all x-phase polynomials.
            self.eq_u_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
            self.eq_s_default
                .bind_parallel(r_j, BindingOrder::LowToHigh);
            for poly in self.eq_s_by_exp.iter_mut().flatten() {
                poly.bind_parallel(r_j, BindingOrder::LowToHigh);
            }
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
        } else {
            // Bind the c-phase polynomials (must already be initialized).
            self.ensure_c_state();
            self.c_state
                .as_mut()
                .expect("c_state must be initialized")
                .bind(r_j);
        }
    }

    fn cache_openings(
        &self,
        _accumulator: &mut ProverOpeningAccumulator<Fq>,
        _transcript: &mut T,
        _sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        // No new openings: this instance consumes fused openings cached by earlier GT instances.
    }
}

#[derive(Clone, Debug)]
pub struct FusedWiringGtVerifier {
    edges: Vec<GtWiringEdge>,
    lambdas: Vec<Fq>,
    tau: Vec<<Fq as JoltField>::Challenge>,
    pairing_boundary: PairingBoundary,
    joint_commitment: Fq12,
    gt_exp_bases: Vec<Fq12>,
    gt_exp_out_step: Vec<usize>,
    /// k_common
    num_c_vars: usize,
    k_exp: usize,
    k_mul: usize,
}

impl FusedWiringGtVerifier {
    pub fn new<T: Transcript>(input: &RecursionVerifierInput, transcript: &mut T) -> Self {
        // Must mirror prover sampling order: τ, then λ.
        let tau: Vec<<Fq as JoltField>::Challenge> =
            transcript.challenge_vector_optimized::<Fq>(ELEM_VARS);
        let edges = input.wiring.gt.clone();
        let lambdas = transcript.challenge_vector(edges.len());

        let num_c_vars = k_gt(&input.constraint_types);
        let k_exp = k_exp(&input.constraint_types);
        let k_mul = k_mul(&input.constraint_types);

        let gt_exp_out_step = input
            .gt_exp_public_inputs
            .iter()
            .map(|p| {
                let digits_len = (p.scalar_bits.len() + 1) / 2;
                let max_s = (1 << STEP_VARS) - 1;
                digits_len.min(max_s)
            })
            .collect();
        let gt_exp_bases = input.gt_exp_public_inputs.iter().map(|p| p.base).collect();

        Self {
            edges,
            lambdas,
            tau,
            pairing_boundary: input.pairing_boundary.clone(),
            joint_commitment: input.joint_commitment,
            gt_exp_bases,
            gt_exp_out_step,
            num_c_vars,
            k_exp,
            k_mul,
        }
    }

    #[inline]
    fn eq_s_for_src(&self, r_step: &[<Fq as JoltField>::Challenge], src: GtProducer) -> Fq {
        match src {
            GtProducer::GtExpRho { instance } => {
                let s_out = self.gt_exp_out_step[instance];
                r_step
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
            GtProducer::GtMulResult { .. } | GtProducer::GtExpBase { .. } => r_step
                .iter()
                .map(|c| {
                    let r_b: Fq = (*c).into();
                    Fq::one() - r_b
                })
                .product(),
        }
    }

    #[inline]
    fn eq_c_tail(&self, r_c_tail: &[Fq], idx: usize, k_family: usize) -> Fq {
        let bits = index_to_binary::<Fq>(idx, k_family);
        EqPolynomial::mle(r_c_tail, &bits)
    }
}

impl<T: Transcript> SumcheckInstanceVerifier<Fq, T> for FusedWiringGtVerifier {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        X_VARS + self.num_c_vars
    }

    fn input_claim(&self, _acc: &VerifierOpeningAccumulator<Fq>) -> Fq {
        Fq::zero()
    }

    fn expected_output_claim(
        &self,
        acc: &VerifierOpeningAccumulator<Fq>,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) -> Fq {
        debug_assert_eq!(sumcheck_challenges.len(), X_VARS + self.num_c_vars);

        let r_step = &sumcheck_challenges[..STEP_VARS];
        let r_elem_chal = &sumcheck_challenges[STEP_VARS..STEP_VARS + ELEM_VARS];
        let r_c = &sumcheck_challenges[X_VARS..];
        debug_assert_eq!(r_c.len(), self.num_c_vars);

        let eq_u = eq_lsb_mle::<Fq>(&self.tau, r_elem_chal);

        let dummy_exp = self.num_c_vars.saturating_sub(self.k_exp);
        let dummy_mul = self.num_c_vars.saturating_sub(self.k_mul);
        let beta_exp = beta(dummy_exp);
        let beta_mul = beta(dummy_mul);

        // Fused port openings (already cached by earlier GT instances in Stage 2).
        let (_rho_point, rho_fused) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::gt_exp_rho_fused(),
            SumcheckId::GtExpClaimReduction,
        );
        let (_mul_lhs_point, mul_lhs_fused) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::gt_mul_lhs_fused(),
            SumcheckId::GtMul,
        );
        let (_mul_rhs_point, mul_rhs_fused) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::gt_mul_rhs_fused(),
            SumcheckId::GtMul,
        );
        let (_mul_out_point, mul_out_fused) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::gt_mul_result_fused(),
            SumcheckId::GtMul,
        );

        // Boundary constants at r_elem.
        let r_elem: Vec<Fq> = r_elem_chal.iter().map(|c| (*c).into()).collect();
        let joint_eval = eval_fq12_packed_at(&self.joint_commitment, &r_elem);
        let pairing_rhs_eval = eval_fq12_packed_at(&self.pairing_boundary.rhs, &r_elem);

        let r_c_fq: Vec<Fq> = r_c.iter().map(|c| (*c).into()).collect();
        let r_c_exp_tail = &r_c_fq[dummy_exp..];
        let r_c_mul_tail = &r_c_fq[dummy_mul..];

        let mut sum = Fq::zero();
        for (lambda, edge) in self.lambdas.iter().zip(self.edges.iter()) {
            let eq_s = self.eq_s_for_src(r_step, edge.src);

            let (eq_c_src, src_val, beta_src) = match edge.src {
                GtProducer::GtExpRho { instance } => (
                    self.eq_c_tail(r_c_exp_tail, instance, self.k_exp),
                    rho_fused,
                    beta_exp,
                ),
                GtProducer::GtExpBase { instance } => {
                    let base_eval = eval_fq12_packed_at(&self.gt_exp_bases[instance], &r_elem);
                    (
                        self.eq_c_tail(r_c_exp_tail, instance, self.k_exp),
                        base_eval,
                        beta_exp,
                    )
                }
                GtProducer::GtMulResult { instance } => (
                    self.eq_c_tail(r_c_mul_tail, instance, self.k_mul),
                    mul_out_fused,
                    beta_mul,
                ),
            };

            let (eq_c_dst, dst_val, beta_dst) = match edge.dst {
                GtConsumer::GtMulLhs { instance } => (
                    self.eq_c_tail(r_c_mul_tail, instance, self.k_mul),
                    mul_lhs_fused,
                    beta_mul,
                ),
                GtConsumer::GtMulRhs { instance } => (
                    self.eq_c_tail(r_c_mul_tail, instance, self.k_mul),
                    mul_rhs_fused,
                    beta_mul,
                ),
                GtConsumer::GtExpBase { instance } => {
                    let base_eval = eval_fq12_packed_at(&self.gt_exp_bases[instance], &r_elem);
                    (
                        self.eq_c_tail(r_c_exp_tail, instance, self.k_exp),
                        base_eval,
                        beta_exp,
                    )
                }
                // Anchor globals to src.
                GtConsumer::JointCommitment => (eq_c_src, joint_eval, beta_src),
                GtConsumer::PairingBoundaryRhs => (eq_c_src, pairing_rhs_eval, beta_src),
            };

            sum += *lambda * eq_s * (beta_src * eq_c_src * src_val - beta_dst * eq_c_dst * dst_val);
        }

        eq_u * sum
    }

    fn cache_openings(
        &self,
        _acc: &mut VerifierOpeningAccumulator<Fq>,
        _transcript: &mut T,
        _sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        // No-op.
    }
}

//! GT wiring (copy/boundary) sumcheck
//!
//! This is the GT wiring backend used by recursion.
//! It enforces that GT producer outputs match GT consumer inputs (plus boundary constants)
//! according to the verifier-derived `WiringPlan` edge list.
//!
//! Importantly, the *port polynomials* for GTExp/GTmul are treated as **multilinear in `c`**
//! (over the instance index). This allows the verifier to consume already-cached
//! openings (e.g. `gt_exp_rho()`), rather than per-instance openings.
//!
//! ## Variable order (Stage 2)
//! This instance uses the same Stage-2 GT ordering as the packed GT gadgets:
//! - first bind `x = (s,u)` (11 rounds, `BindingOrder::LowToHigh` with `s` in the low bits),
//! - then bind the GT-local `c` suffix (k rounds).
//!
//! This matches how `GtExpStage2Openings*` and `GtMul*` interpret the Stage-2
//! challenge vector.
//!
//! ## Sumcheck relation
//!
//! **Input claim:** `0`.
//!
//! The polynomial sumchecked by this instance uses the “dummy-low-bits” split-k convention with
//! explicit normalization:
//!
//! ```text
//! Σ_{x=(s,u), c_common} Eq(u, τ) · Σ_{e∈E_GT} λ_e · Eq_s(e; s) · (
//!     β_src(e) · Eq(c_tail^{src(e)}, idx_src(e)) · V_src(e; x)
//!   - β_dst(e) · Eq(c_tail^{dst(e)}, idx_dst(e)) · V_dst(e; x)
//! ) = 0
//! ```
//!
//! where:
//! - `c_common ∈ {0,1}^{k_gt}` is the Stage-2 GT-local common suffix domain,
//! - for a family with `k_f` index bits: `dummy_f = k_gt - k_f`,
//!   `c_tail^f = c_common[dummy_f..]`, and `β_f = 2^{-dummy_f}`,
//! - `V_src/V_dst` are the relevant port polynomials (consumed via cached openings from
//!   `GtExp*`, `GtMul*`, and `GtExpStage2Openings*`), with u-only values padded to 11 vars by
//!   replication over the 7 step bits,
//! - boundary constants are anchored to the opposite endpoint’s selector (GT-style),
//! - `λ_e` are transcript-sampled edge-batching coefficients.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    marker::PhantomData,
};

use ark_bn254::{Fq, Fq12, Fq2, Fq6};
use ark_ff::{Field, One, Zero};
use rayon::prelude::*;

use crate::{
    field::JoltField,
    poly::{
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
            constraints::system::ConstraintSystem,
            curve::{Bn254Recursion, RecursionCurve},
            gt::indexing::{k_exp, k_gt, k_mul},
            gt::types::{eq_lsb_evals, eq_lsb_mle},
            verifier::RecursionVerifierInput,
            wiring_plan::{GtConsumer, GtProducer, GtWiringEdge},
        },
        witness::{GtExpTerm, GtMulTerm, RecursionPoly, VirtualPolynomial},
    },
};

pub(crate) const STEP_VARS: usize = 7;
pub(crate) const ELEM_VARS: usize = 4;
pub(crate) const X_VARS: usize = STEP_VARS + ELEM_VARS; // 11

// Cycle-marker labels must be static strings: the tracer keys markers by the guest string pointer.
const CYCLE_VERIFY_RECURSION_STAGE2_GT_WIRING_TOTAL: &str =
    "verify_recursion_stage2_gt_wiring_total";

/// Max degree per variable is 2 (product of multilinears).
// Degree bound for GT wiring sumcheck:
// - In x-phase we multiply three multilinear factors: eq_u(u), eq_s(s), and (src(x)-dst(x)),
//   giving per-round univariates of degree ≤ 3.
// - In c-phase we multiply at most two multilinear factors (selector * port), degree ≤ 2.
// So the overall bound is 3.
const DEGREE: usize = 3;

fn pad_4var_to_11var_replicated(mle_4var: &[Fq]) -> Vec<Fq> {
    debug_assert_eq!(
        mle_4var.len(),
        1 << ELEM_VARS,
        "expected 4-var MLE (len 16)"
    );
    let mut mle_11var = vec![Fq::zero(); 1 << X_VARS];
    // Layout matches packed x11: index = s * 16 + x (x in low 4 bits).
    for s in 0..(1 << STEP_VARS) {
        let off = s * (1 << ELEM_VARS);
        mle_11var[off..off + (1 << ELEM_VARS)].copy_from_slice(mle_4var);
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
    // Layout matches packed x11: index = s * 16 + x (x in low 4 bits).
    for s in 0..(1 << STEP_VARS) {
        let v = mle_7var[s];
        let off = s * (1 << ELEM_VARS);
        mle_11var[off..off + (1 << ELEM_VARS)].fill(v);
    }
    mle_11var
}

fn elem_weights_4var_lsb_first(r_elem: [Fq; ELEM_VARS]) -> [Fq; 1 << ELEM_VARS] {
    // Returns weights w[idx] = Eq(r_elem, idx) in the same LSB-first indexing convention used by
    // `eq_lsb_evals` / `eval_mle_lsb_first`.
    //
    // For ELEM_VARS=4 this is just 16 weights, and (crucially) can be computed once then reused for
    // all Fq12 evaluations at this `r_elem`.
    let mut w = [Fq::zero(); 1 << ELEM_VARS];
    w[0] = Fq::one();
    let mut len = 1usize;
    for r_i in r_elem {
        let one_minus = Fq::one() - r_i;
        for j in 0..len {
            let prev = w[j];
            w[j] = prev * one_minus;
            w[j + len] = prev * r_i;
        }
        len *= 2;
    }
    debug_assert_eq!(len, 1 << ELEM_VARS);
    w
}

#[inline(always)]
fn eval_fq12_packed_at_with_weights(fq12: &Fq12, elem_weights: &[Fq; 1 << ELEM_VARS]) -> Fq {
    // Evaluate a 4-var MLE at `r_elem` using precomputed Eq weights:
    //   f(r_elem) = Σ_x f(x) * Eq(r_elem, x)
    //
    // This is ~2× fewer field multiplications than folding the MLE per-evaluation, and it avoids
    // recomputing Eq weights for every base input.
    let evals = Bn254Recursion::fq12_to_mle(fq12);
    debug_assert_eq!(evals.len(), 1 << ELEM_VARS);
    let mut out = Fq::zero();
    for (v, w) in evals.iter().zip(elem_weights.iter()) {
        out += *v * *w;
    }
    out
}

#[inline(always)]
fn fq12_coeffs(x: &Fq12) -> [Fq; 12] {
    [
        x.c0.c0.c0, x.c0.c0.c1, x.c0.c1.c0, x.c0.c1.c1, x.c0.c2.c0, x.c0.c2.c1, x.c1.c0.c0,
        x.c1.c0.c1, x.c1.c1.c0, x.c1.c1.c1, x.c1.c2.c0, x.c1.c2.c1,
    ]
}

#[inline(always)]
fn fq12_from_coeffs(c: [Fq; 12]) -> Fq12 {
    let c0 = Fq6::new(
        Fq2::new(c[0], c[1]),
        Fq2::new(c[2], c[3]),
        Fq2::new(c[4], c[5]),
    );
    let c1 = Fq6::new(
        Fq2::new(c[6], c[7]),
        Fq2::new(c[8], c[9]),
        Fq2::new(c[10], c[11]),
    );
    Fq12::new(c0, c1)
}

/// Precompute a linear functional `α` such that for any `x ∈ Fq12`:
/// `eval_fq12_packed_at_with_weights(x, w) == Σ_j α[j] * coeffs(x)[j]`.
///
/// This avoids calling `fq12_to_mle()` per GTExp base input. Instead, we pay 12 basis conversions
/// once per Stage-2 wiring check.
#[inline]
fn fq12_eval_linear_coeffs(elem_weights: &[Fq; 1 << ELEM_VARS]) -> [Fq; 12] {
    let mut alpha = [Fq::zero(); 12];
    for j in 0..12 {
        let mut c = [Fq::zero(); 12];
        c[j] = Fq::one();
        let basis = fq12_from_coeffs(c);
        alpha[j] = eval_fq12_packed_at_with_weights(&basis, elem_weights);
    }
    alpha
}

#[inline(always)]
fn eval_fq12_packed_at_via_linear_coeffs(fq12: &Fq12, alpha: &[Fq; 12]) -> Fq {
    let c = fq12_coeffs(fq12);
    let mut out = Fq::zero();
    for j in 0..12 {
        out += c[j] * alpha[j];
    }
    out
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
    /// pairing miller rhs evaluated at `r_u` (scalar).
    pairing_miller_rhs_at_r: Fq,
    /// Per-edge eq_s evaluated at bound `r_step` (scalar).
    edge_eq_s_at_r: Vec<Fq>,
    /// Scalars per GTExp instance at the bound x-point `r_x` (used for debugging phase split).
    rho_at_r: Vec<Fq>,
    /// Scalars per GTMul instance at the bound x-point `r_x` (used for debugging phase split).
    mul_lhs_at_r: Vec<Fq>,
    mul_rhs_at_r: Vec<Fq>,
    mul_result_at_r: Vec<Fq>,
    /// Scalars per GTExp base port at the bound x-point `r_x` (used for debugging phase split).
    exp_base_port_at_r: Vec<Fq>,
    /// GTExp rho values as an MLE over the **common** c domain (k vars), replicated across dummy bits.
    rho_c: MultilinearPolynomial<Fq>,
    /// GTMul ports as MLEs over the **common** c domain (k vars), replicated across dummy bits.
    mul_lhs_c: MultilinearPolynomial<Fq>,
    mul_rhs_c: MultilinearPolynomial<Fq>,
    mul_result_c: MultilinearPolynomial<Fq>,
    /// GTExp base ports as an MLE over the **common** c domain (k vars), replicated across dummy bits.
    ///
    /// This represents the committed `GtExpTerm::Base` row values evaluated at the bound `r_u`,
    /// stacked over the GTExp family index.
    base_port_c: MultilinearPolynomial<Fq>,
    /// Selector polynomials Eq(c, idx_embed) as basis vectors in the common c domain.
    /// Key is `(dummy_bits, k_family, idx)` packed into a u64.
    selectors: HashMap<u64, MultilinearPolynomial<Fq>>,
    /// Bound GTExp base *inputs* as scalars per GTExp instance (only for `GtProducer::GtExpBase`).
    exp_base_inputs_at_r: Vec<Fq>,
    /// Bound GT-valued AST inputs as scalars, keyed by `ValueId.0` (only for `GtProducer::GtInput`).
    gt_inputs_at_r: HashMap<u32, Fq>,
    /// Bound Multi-Miller loop outputs (f packed) as scalars, keyed by global constraint index.
    miller_out_at_r: HashMap<usize, Fq>,
}

impl CPhaseState {
    fn bind(&mut self, r_j: <Fq as JoltField>::Challenge) {
        self.rho_c.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.mul_lhs_c.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.mul_rhs_c.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.mul_result_c
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.base_port_c.bind_parallel(r_j, BindingOrder::LowToHigh);
        for sel in self.selectors.values_mut() {
            sel.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
    }
}

#[derive(Clone, Debug)]
pub struct WiringGtProver<T: Transcript> {
    edges: Vec<GtWiringEdge>,
    lambdas: Vec<Fq>,
    /// Element selector point τ (4 challenges, LSB-first), used by Eq(u,τ).
    ///
    /// Kept for debugging/validation of the wiring input claim.
    tau: Vec<<Fq as JoltField>::Challenge>,
    /// Output step index for each GTExp instance in the packed GTExp trace.
    ///
    /// Must match the verifier convention: `s_out = min(127, ceil(bits_len/2))`.
    gt_exp_out_step: Vec<usize>,
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
    /// Eq(s, 127) replicated across element vars (11-var, s-only). Used for Miller loop outputs.
    eq_s_miller_out: MultilinearPolynomial<Fq>,

    /// Packed GT exp rho polynomials (11-var), indexed by GTExp instance.
    rho_polys: Vec<Option<MultilinearPolynomial<Fq>>>,

    /// Packed Multi-Miller loop output polynomials `f(s,u)` (11-var), keyed by global constraint idx.
    miller_out: BTreeMap<usize, MultilinearPolynomial<Fq>>,

    /// GT mul polynomials, padded to 11 vars (constant over step vars), indexed by GTMul instance.
    mul_lhs: Vec<Option<MultilinearPolynomial<Fq>>>,
    mul_rhs: Vec<Option<MultilinearPolynomial<Fq>>>,
    mul_result: Vec<Option<MultilinearPolynomial<Fq>>>,

    /// GTExp base *inputs* as boundary constants (only used when the wiring plan references
    /// `GtProducer::GtExpBase { instance }`), padded to 11 vars.
    exp_base_inputs: Vec<Option<MultilinearPolynomial<Fq>>>,

    /// GTExp base *ports* (the committed `GtExpTerm::Base` row), padded to 11 vars.
    ///
    /// This is used for `GtConsumer::GtExpBase { instance }` edges.
    exp_base_port: Vec<Option<MultilinearPolynomial<Fq>>>,

    /// Boundary constants, padded to 11 vars.
    joint_commitment: Option<MultilinearPolynomial<Fq>>,
    pairing_rhs: Option<MultilinearPolynomial<Fq>>,
    pairing_miller_rhs: Option<MultilinearPolynomial<Fq>>,
    /// Boundary GT inputs (u-only constants), keyed by AST `ValueId.0`.
    gt_inputs: BTreeMap<u32, MultilinearPolynomial<Fq>>,

    /// Lazily built when the sumcheck enters the c rounds.
    c_state: Option<CPhaseState>,

    _marker: PhantomData<T>,
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
        gt_inputs: &[(u32, Fq12)],
        transcript: &mut T,
    ) -> Self {
        let num_c_vars = k_gt(&cs.constraint_types);
        let k_exp = k_exp(&cs.constraint_types);
        let k_mul = k_mul(&cs.constraint_types);

        // Must mirror verifier sampling order: τ (element selector), then λ edge coefficients.
        let tau: Vec<<Fq as JoltField>::Challenge> =
            transcript.challenge_vector_optimized::<Fq>(ELEM_VARS);
        let lambdas: Vec<Fq> = transcript.challenge_vector(edges.len());
        if std::env::var_os("JOLT_DEBUG_GT_WIRING_CHALLENGES").is_some() {
            let mut edge_fp: u64 = 0;
            for e in edges.iter() {
                let src_tag: u64 = match e.src {
                    GtProducer::GtExpRho { .. } => 1,
                    GtProducer::GtMulResult { .. } => 2,
                    GtProducer::GtExpBase { .. } => 3,
                    GtProducer::JointCommitment => 4,
                    GtProducer::GtInput { .. } => 5,
                    GtProducer::MultiMillerLoopOut { .. } => 6,
                };
                let src_idx: u64 = match e.src {
                    GtProducer::GtExpRho { instance }
                    | GtProducer::GtMulResult { instance }
                    | GtProducer::GtExpBase { instance }
                    | GtProducer::MultiMillerLoopOut { instance } => instance as u64,
                    GtProducer::GtInput { value_id } => value_id as u64,
                    GtProducer::JointCommitment => 0,
                };
                let dst_tag: u64 = match e.dst {
                    GtConsumer::GtMulLhs { .. } => 11,
                    GtConsumer::GtMulRhs { .. } => 12,
                    GtConsumer::GtExpBase { .. } => 13,
                    GtConsumer::JointCommitment => 14,
                    GtConsumer::PairingBoundaryRhs => 15,
                    GtConsumer::PairingBoundaryMillerRhs => 16,
                };
                let dst_idx: u64 = match e.dst {
                    GtConsumer::GtMulLhs { instance }
                    | GtConsumer::GtMulRhs { instance }
                    | GtConsumer::GtExpBase { instance } => instance as u64,
                    _ => 0,
                };
                let v = (src_tag << 56) ^ (dst_tag << 48) ^ (src_idx << 16) ^ dst_idx;
                edge_fp = edge_fp.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(v);
            }
            tracing::info!(
                tau0 = ?tau.get(0),
                lambda0 = ?lambdas.get(0),
                edges = edges.len(),
                edge0 = ?edges.get(0),
                edge_last = ?edges.last(),
                edge_fp = edge_fp,
                "gt_wiring: prover sampled challenges"
            );
        }

        let eq_u_4 = eq_lsb_evals::<Fq>(&tau);
        let eq_u_poly = MultilinearPolynomial::from(pad_4var_to_11var_replicated(&eq_u_4));

        // Default step selector for u-only values: Eq(s, 0).
        let s0: Vec<<Fq as JoltField>::Challenge> =
            (0..STEP_VARS).map(|_| Fq::zero().into()).collect();
        let eq_s0_7 = eq_lsb_evals::<Fq>(&s0);
        let eq_s_default = MultilinearPolynomial::from(pad_7var_to_11var_replicated(&eq_s0_7));

        // Step selector for Miller loop outputs: Eq(s, 127).
        let s127: Vec<<Fq as JoltField>::Challenge> =
            (0..STEP_VARS).map(|_| Fq::one().into()).collect();
        let eq_s127_7 = eq_lsb_evals::<Fq>(&s127);
        let eq_s_miller_out = MultilinearPolynomial::from(pad_7var_to_11var_replicated(&eq_s127_7));

        // IMPORTANT: the verifier consumes openings like `gt_mul_lhs()` /
        // `gt_exp_rho()`. Those are evaluations of (c-batched) polynomials at the
        // Stage-2 point, which in general depend on **all**
        // instances in the family (even if a particular instance is not referenced by any edge).
        //
        // Therefore, we must materialize the full family tables here (no "need_*" filtering),
        // otherwise we silently change the polynomial being proved and Stage-2 sumcheck
        // verification fails.
        let num_gt_exp = cs.gt_exp_witnesses.len();
        let num_gt_mul = cs.gt_mul_rows.len();
        let need_joint = edges.iter().any(|e| {
            matches!(e.dst, GtConsumer::JointCommitment)
                || matches!(e.src, GtProducer::JointCommitment)
        });
        let need_pairing_rhs = edges
            .iter()
            .any(|e| matches!(e.dst, GtConsumer::PairingBoundaryRhs));
        let need_pairing_miller_rhs = edges
            .iter()
            .any(|e| matches!(e.dst, GtConsumer::PairingBoundaryMillerRhs));

        let mut needed_miller_out: BTreeSet<usize> = BTreeSet::new();
        for e in &edges {
            if let GtProducer::MultiMillerLoopOut { instance } = e.src {
                needed_miller_out.insert(instance);
            }
        }

        // Boundary GT inputs (e.g. proof/setup GT values) keyed by AST `ValueId.0`.
        let gt_inputs_by_id: BTreeMap<u32, Fq12> = gt_inputs.iter().copied().collect();
        let mut needed_gt_inputs: BTreeSet<u32> = BTreeSet::new();
        for e in &edges {
            if let GtProducer::GtInput { value_id } = e.src {
                needed_gt_inputs.insert(value_id);
            }
        }
        let gt_inputs: BTreeMap<u32, MultilinearPolynomial<Fq>> = needed_gt_inputs
            .into_iter()
            .map(|value_id| {
                let fq12 = gt_inputs_by_id
                    .get(&value_id)
                    .unwrap_or_else(|| panic!("missing gt_inputs entry for value_id {value_id}"));
                let mle_4 = Bn254Recursion::fq12_to_mle(fq12);
                let poly = MultilinearPolynomial::from(pad_4var_to_11var_replicated(&mle_4));
                (value_id, poly)
            })
            .collect();

        let rho_polys: Vec<Option<MultilinearPolynomial<Fq>>> = (0..num_gt_exp)
            .map(|i| {
                Some(MultilinearPolynomial::from(
                    cs.gt_exp_witnesses[i].rho_packed.clone(),
                ))
            })
            .collect();

        let miller_out: BTreeMap<usize, MultilinearPolynomial<Fq>> = needed_miller_out
            .into_iter()
            .map(|global_idx| {
                let loc = cs
                    .locator_by_constraint
                    .get(global_idx)
                    .unwrap_or_else(|| panic!("missing locator for constraint index {global_idx}"));
                let local = match *loc {
                    crate::zkvm::recursion::ConstraintLocator::MultiMillerLoop { local } => local,
                    _ => panic!(
                        "constraint {global_idx} is not a MultiMillerLoop (locator: {loc:?})"
                    ),
                };
                let poly = MultilinearPolynomial::from(cs.multi_miller_loop_rows[local].f.clone());
                (global_idx, poly)
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

        // Boundary base inputs (used only when `GtProducer::GtExpBase { instance }` appears).
        let exp_base_inputs: Vec<Option<MultilinearPolynomial<Fq>>> = (0..num_gt_exp)
            .map(|i| {
                let mle_4 = Bn254Recursion::fq12_to_mle(&cs.gt_exp_base_inputs[i]);
                Some(MultilinearPolynomial::from(pad_4var_to_11var_replicated(
                    &mle_4,
                )))
            })
            .collect();

        // Base ports (committed row) are taken from the packed witness base table, which is
        // replicated across step bits. This matches the padding convention used elsewhere.
        let exp_base_port: Vec<Option<MultilinearPolynomial<Fq>>> = (0..num_gt_exp)
            .map(|i| {
                Some(MultilinearPolynomial::from(
                    cs.gt_exp_witnesses[i].base_packed.clone(),
                ))
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
        let pairing_miller_rhs = need_pairing_miller_rhs.then(|| {
            let mle_4 = Bn254Recursion::fq12_to_mle(&pairing_boundary.miller_rhs);
            MultilinearPolynomial::from(pad_4var_to_11var_replicated(&mle_4))
        });

        let gt_exp_out_step: Vec<usize> = (0..num_gt_exp)
            .map(|i| {
                let max_s = (1 << STEP_VARS) - 1;
                let digits_len = cs.gt_exp_public_inputs[i].scalar_bits.len().div_ceil(2);
                digits_len.min(max_s)
            })
            .collect();
        let eq_s_by_exp: Vec<Option<MultilinearPolynomial<Fq>>> = (0..num_gt_exp)
            .map(|i| {
                // Output step index for this GTExp instance (packed trace): rho[s_out] is final.
                let s_out = gt_exp_out_step[i];
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
            tau,
            gt_exp_out_step,
            num_c_vars,
            k_exp,
            k_mul,
            eq_u_poly,
            eq_s_default,
            eq_s_by_exp,
            eq_s_miller_out,
            rho_polys,
            miller_out,
            mul_lhs,
            mul_rhs,
            mul_result,
            exp_base_inputs,
            exp_base_port,
            joint_commitment,
            pairing_rhs,
            pairing_miller_rhs,
            gt_inputs,
            c_state: None,
            _marker: PhantomData,
        }
    }

    fn eq_s_poly_for_src(&self, src: GtProducer) -> &MultilinearPolynomial<Fq> {
        match src {
            GtProducer::GtExpRho { instance } => self.eq_s_by_exp[instance].as_ref().unwrap(),
            GtProducer::MultiMillerLoopOut { .. } => &self.eq_s_miller_out,
            GtProducer::GtMulResult { .. }
            | GtProducer::GtExpBase { .. }
            | GtProducer::JointCommitment
            | GtProducer::GtInput { .. } => &self.eq_s_default,
        }
    }

    fn src_poly_x(&self, src: GtProducer) -> &MultilinearPolynomial<Fq> {
        match src {
            GtProducer::GtExpRho { instance } => self.rho_polys[instance].as_ref().unwrap(),
            GtProducer::GtMulResult { instance } => self.mul_result[instance].as_ref().unwrap(),
            GtProducer::GtExpBase { instance } => self.exp_base_inputs[instance].as_ref().unwrap(),
            GtProducer::JointCommitment => self.joint_commitment.as_ref().unwrap(),
            GtProducer::GtInput { value_id } => self
                .gt_inputs
                .get(&value_id)
                .expect("missing gt_inputs entry for GtProducer::GtInput"),
            GtProducer::MultiMillerLoopOut { instance } => self
                .miller_out
                .get(&instance)
                .unwrap_or_else(|| panic!("missing miller_out poly for constraint {instance}")),
        }
    }

    fn dst_poly_x(&self, dst: GtConsumer) -> &MultilinearPolynomial<Fq> {
        match dst {
            GtConsumer::GtMulLhs { instance } => self.mul_lhs[instance].as_ref().unwrap(),
            GtConsumer::GtMulRhs { instance } => self.mul_rhs[instance].as_ref().unwrap(),
            GtConsumer::GtExpBase { instance } => self.exp_base_port[instance].as_ref().unwrap(),
            GtConsumer::JointCommitment => self.joint_commitment.as_ref().unwrap(),
            GtConsumer::PairingBoundaryRhs => self.pairing_rhs.as_ref().unwrap(),
            GtConsumer::PairingBoundaryMillerRhs => self.pairing_miller_rhs.as_ref().unwrap(),
        }
    }

    fn ensure_c_state(&mut self) {
        if self.c_state.is_some() {
            return;
        }

        // x has been fully bound (all 11 rounds), so x-polynomials are scalars now.
        let eq_u_at_r = self.eq_u_poly.get_bound_coeff(0);

        if std::env::var_os("JOLT_DEBUG_GT_WIRING_EQS").is_some() {
            let eq_s_zero_at_r = self.eq_s_default.get_bound_coeff(0);
            let eq_s_miller_at_r = self.eq_s_miller_out.get_bound_coeff(0);
            let eq_s_exp0_at_r = self
                .eq_s_by_exp
                .get(0)
                .and_then(|x| x.as_ref())
                .map(|p| p.get_bound_coeff(0));
            tracing::info!(
                ?eq_s_zero_at_r,
                ?eq_s_miller_at_r,
                ?eq_s_exp0_at_r,
                "gt_wiring: prover eq_s samples at r_step (from bound polynomials)"
            );
        }
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
        let pairing_miller_rhs_at_r = self
            .pairing_miller_rhs
            .as_ref()
            .map(|p| p.get_bound_coeff(0))
            .unwrap_or_else(Fq::zero);
        let gt_inputs_at_r: HashMap<u32, Fq> = self
            .gt_inputs
            .iter()
            .map(|(id, p)| (*id, p.get_bound_coeff(0)))
            .collect();
        let miller_out_at_r: HashMap<usize, Fq> = self
            .miller_out
            .iter()
            .map(|(idx, p)| (*idx, p.get_bound_coeff(0)))
            .collect();

        let num_gt_exp = self.rho_polys.len();
        let num_gt_mul = self.mul_result.len();

        // Scalar base inputs at r_x (only used when `GtProducer::GtExpBase` is present).
        let exp_base_inputs_at_r: Vec<Fq> = (0..num_gt_exp)
            .map(|i| {
                self.exp_base_inputs[i]
                    .as_ref()
                    .map(|p| p.get_bound_coeff(0))
                    .unwrap_or(Fq::zero())
            })
            .collect();

        // Scalar base ports at r_x (these represent the committed Base row, padded to x11).
        let exp_base_port_at_r: Vec<Fq> = (0..num_gt_exp)
            .map(|i| {
                self.exp_base_port[i]
                    .as_ref()
                    .map(|p| p.get_bound_coeff(0))
                    .unwrap_or(Fq::zero())
            })
            .collect();

        if std::env::var_os("JOLT_DEBUG_GT_WIRING_EXPECTED_SAMPLES").is_some() {
            let gt_input2_at_r = gt_inputs_at_r.get(&2).copied();
            let exp_base_input0_at_r = exp_base_inputs_at_r.get(0).copied();
            tracing::info!(
                ?eq_u_at_r,
                ?joint_at_r,
                ?pairing_rhs_at_r,
                ?pairing_miller_rhs_at_r,
                ?gt_input2_at_r,
                ?exp_base_input0_at_r,
                "gt_wiring: prover boundary/input eval samples at r_u"
            );
        }

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

        // Base ports in the common c domain (replicate across dummy bits).
        let mut base_port_c = vec![Fq::zero(); n];
        for c in 0..n {
            let family_idx = c >> dummy_exp;
            if family_idx < exp_base_port_at_r.len() {
                base_port_c[c] = exp_base_port_at_r[family_idx];
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
                GtProducer::MultiMillerLoopOut { .. } => self.eq_s_miller_out.get_bound_coeff(0),
                GtProducer::GtMulResult { .. }
                | GtProducer::GtExpBase { .. }
                | GtProducer::JointCommitment
                | GtProducer::GtInput { .. } => eq_s_default_at_r,
            })
            .collect();

        // Build selector polynomials that depend only on the family tail bits:
        // sel(c) = 1 if (c_tail == idx), 0 otherwise. This is replicated across the dummy
        // low bits, and requires the Option-B normalization factors (beta) in evaluation.
        let mut selectors: HashMap<u64, MultilinearPolynomial<Fq>> = HashMap::new();
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
                GtProducer::JointCommitment
                | GtProducer::GtInput { .. }
                | GtProducer::MultiMillerLoopOut { .. } => {
                    // Anchor to destination family: ensure the destination selector exists.
                    match edge.dst {
                        GtConsumer::GtMulLhs { instance } | GtConsumer::GtMulRhs { instance } => {
                            let _ = ensure_selector(dummy_mul, self.k_mul, instance);
                        }
                        GtConsumer::GtExpBase { instance } => {
                            let _ = ensure_selector(dummy_exp, self.k_exp, instance);
                        }
                        GtConsumer::JointCommitment
                        | GtConsumer::PairingBoundaryRhs
                        | GtConsumer::PairingBoundaryMillerRhs => {
                            // Should not happen in well-formed plans.
                        }
                    }
                }
            }
            match edge.dst {
                GtConsumer::GtMulLhs { instance } | GtConsumer::GtMulRhs { instance } => {
                    let _ = ensure_selector(dummy_mul, self.k_mul, instance);
                }
                GtConsumer::GtExpBase { instance } => {
                    let _ = ensure_selector(dummy_exp, self.k_exp, instance);
                }
                GtConsumer::JointCommitment
                | GtConsumer::PairingBoundaryRhs
                | GtConsumer::PairingBoundaryMillerRhs => {
                    // Anchored to src; no additional selector required here.
                }
            }
        }

        self.c_state = Some(CPhaseState {
            eq_u_at_r,
            joint_at_r,
            pairing_rhs_at_r,
            pairing_miller_rhs_at_r,
            edge_eq_s_at_r,
            rho_at_r: rho_at_r.clone(),
            mul_lhs_at_r: mul_lhs_at_r.clone(),
            mul_rhs_at_r: mul_rhs_at_r.clone(),
            mul_result_at_r: mul_result_at_r.clone(),
            exp_base_port_at_r: exp_base_port_at_r.clone(),
            rho_c: MultilinearPolynomial::from(rho_c),
            mul_lhs_c: MultilinearPolynomial::from(mul_lhs_c),
            mul_rhs_c: MultilinearPolynomial::from(mul_rhs_c),
            mul_result_c: MultilinearPolynomial::from(mul_result_c),
            base_port_c: MultilinearPolynomial::from(base_port_c),
            selectors,
            exp_base_inputs_at_r,
            gt_inputs_at_r,
            miller_out_at_r,
        });
    }
}

impl<T: Transcript> SumcheckInstanceProver<Fq, T> for WiringGtProver<T> {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        X_VARS + self.num_c_vars
    }

    fn input_claim(&self, _acc: &ProverOpeningAccumulator<Fq>) -> Fq {
        // Debug-only: compute the *true* sum over the Boolean cube for this wiring polynomial.
        //
        // For a well-formed wiring plan and consistent witnesses, this should be 0.
        //
        // This is very useful to distinguish:
        // - a bug in sumcheck message computation, vs
        // - a bug in the wiring constraints/witnesses (ports are not actually equal).
        if std::env::var_os("JOLT_DEBUG_GT_WIRING_TRUE_INPUT_CLAIM").is_some() {
            use crate::zkvm::recursion::gt::types::eq_lsb_evals;
            use crate::zkvm::recursion::wiring_plan::{GtConsumer, GtProducer};

            // Eq weights for evaluating 4-var MLEs at τ.
            let w_u: Vec<Fq> = eq_lsb_evals::<Fq>(&self.tau);
            debug_assert_eq!(w_u.len(), 1 << ELEM_VARS);

            #[inline(always)]
            fn eval_row_at_tau(poly: &MultilinearPolynomial<Fq>, step: usize, w_u: &[Fq]) -> Fq {
                let off = step * (1 << ELEM_VARS);
                let mut out = Fq::zero();
                for x in 0..(1 << ELEM_VARS) {
                    out += poly.get_coeff(off + x) * w_u[x];
                }
                out
            }

            // NOTE: We keep this debug computation simple: it will still be informative even if
            // the wiring is mis-specified (it will surface as a non-zero claim).
            let mut sum = Fq::zero();
            for (lambda, edge) in self.lambdas.iter().zip(self.edges.iter()) {
                let src_eval = match edge.src {
                    GtProducer::GtExpRho { instance } => {
                        let poly = self.rho_polys[instance].as_ref().expect("missing rho poly");
                        let s_out = self.gt_exp_out_step[instance];
                        eval_row_at_tau(poly, s_out, &w_u)
                    }
                    GtProducer::MultiMillerLoopOut { instance } => {
                        let poly = self
                            .miller_out
                            .get(&instance)
                            .expect("missing miller_out poly");
                        eval_row_at_tau(poly, (1 << STEP_VARS) - 1, &w_u)
                    }
                    GtProducer::GtMulResult { instance } => {
                        let poly = self.mul_result[instance].as_ref().expect("missing mul_result");
                        eval_row_at_tau(poly, 0, &w_u)
                    }
                    GtProducer::GtExpBase { instance } => {
                        let poly = self
                            .exp_base_inputs[instance]
                            .as_ref()
                            .expect("missing exp_base_input");
                        eval_row_at_tau(poly, 0, &w_u)
                    }
                    GtProducer::JointCommitment => {
                        let poly = self.joint_commitment.as_ref().expect("missing joint");
                        eval_row_at_tau(poly, 0, &w_u)
                    }
                    GtProducer::GtInput { value_id } => {
                        let poly = self
                            .gt_inputs
                            .get(&value_id)
                            .expect("missing gt_inputs poly");
                        eval_row_at_tau(poly, 0, &w_u)
                    }
                };
                let dst_eval = match edge.dst {
                    GtConsumer::GtMulLhs { instance } => {
                        let poly = self.mul_lhs[instance].as_ref().expect("missing mul_lhs");
                        eval_row_at_tau(poly, 0, &w_u)
                    }
                    GtConsumer::GtMulRhs { instance } => {
                        let poly = self.mul_rhs[instance].as_ref().expect("missing mul_rhs");
                        eval_row_at_tau(poly, 0, &w_u)
                    }
                    GtConsumer::GtExpBase { instance } => {
                        let poly = self
                            .exp_base_port[instance]
                            .as_ref()
                            .expect("missing exp_base_port");
                        eval_row_at_tau(poly, 0, &w_u)
                    }
                    GtConsumer::JointCommitment => {
                        let poly = self.joint_commitment.as_ref().expect("missing joint");
                        eval_row_at_tau(poly, 0, &w_u)
                    }
                    GtConsumer::PairingBoundaryRhs => {
                        let poly = self.pairing_rhs.as_ref().expect("missing pairing_rhs");
                        eval_row_at_tau(poly, 0, &w_u)
                    }
                    GtConsumer::PairingBoundaryMillerRhs => {
                        let poly = self
                            .pairing_miller_rhs
                            .as_ref()
                            .expect("missing pairing_miller_rhs");
                        eval_row_at_tau(poly, 0, &w_u)
                    }
                };
                sum += *lambda * (src_eval - dst_eval);
            }
            tracing::info!(?sum, "gt_wiring: TRUE input claim (debug)");
        }
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

            // Debug aid: check that the claim entering c-phase matches the direct evaluation
            // of Σ_c P(r_x, c), computed from the per-instance endpoint scalars.
            if round == X_VARS && std::env::var_os("JOLT_DEBUG_GT_WIRING_PHASE_SPLIT").is_some() {
                use crate::zkvm::recursion::wiring_plan::{GtConsumer, GtProducer};
                let mut direct = Fq::zero();
                for (edge_idx, edge) in self.edges.iter().enumerate() {
                    let lambda = self.lambdas[edge_idx];
                    let eq_s_at_r = state.edge_eq_s_at_r[edge_idx];
                    let edge_coeff = state.eq_u_at_r * lambda * eq_s_at_r;

                    let src = match edge.src {
                        GtProducer::GtExpRho { instance } => state.rho_at_r[instance],
                        GtProducer::GtMulResult { instance } => state.mul_result_at_r[instance],
                        GtProducer::GtExpBase { instance } => state.exp_base_inputs_at_r[instance],
                        GtProducer::JointCommitment => state.joint_at_r,
                        GtProducer::GtInput { value_id } => *state
                            .gt_inputs_at_r
                            .get(&value_id)
                            .expect("missing gt_inputs_at_r entry for GtProducer::GtInput"),
                        GtProducer::MultiMillerLoopOut { instance } => *state
                            .miller_out_at_r
                            .get(&instance)
                            .expect("missing miller_out_at_r entry for MultiMillerLoopOut"),
                    };

                    let dst = match edge.dst {
                        GtConsumer::GtMulLhs { instance } => state.mul_lhs_at_r[instance],
                        GtConsumer::GtMulRhs { instance } => state.mul_rhs_at_r[instance],
                        GtConsumer::GtExpBase { instance } => state.exp_base_port_at_r[instance],
                        GtConsumer::JointCommitment => state.joint_at_r,
                        GtConsumer::PairingBoundaryRhs => state.pairing_rhs_at_r,
                        GtConsumer::PairingBoundaryMillerRhs => state.pairing_miller_rhs_at_r,
                    };

                    direct += edge_coeff * (src - dst);
                }
                tracing::info!(
                    ?previous_claim,
                    ?direct,
                    "gt_wiring: phase split check (claim entering c-phase)"
                );
            }

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
                    let base_port_e = state
                        .base_port_c
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
                                // Constant source; we ignore `src_port_e` and use `exp_base_inputs_at_r` below.
                                rho_e,
                                beta_exp,
                            ),
                            GtProducer::JointCommitment
                            | GtProducer::GtInput { .. }
                            | GtProducer::MultiMillerLoopOut { .. } => {
                                // Anchor to destination family (this is a global constant source).
                                match edge.dst {
                                    GtConsumer::GtMulLhs { instance }
                                    | GtConsumer::GtMulRhs { instance } => (
                                        selector_key(dummy_mul, self.k_mul, instance),
                                        mul_out_e,
                                        beta_mul,
                                    ),
                                    GtConsumer::GtExpBase { instance } => (
                                        selector_key(dummy_exp, self.k_exp, instance),
                                        rho_e,
                                        beta_exp,
                                    ),
                                    GtConsumer::JointCommitment
                                    | GtConsumer::PairingBoundaryRhs
                                    | GtConsumer::PairingBoundaryMillerRhs => {
                                        // Should not happen in well-formed plans; pick a default.
                                        (
                                            selector_key(dummy_mul, self.k_mul, 0),
                                            mul_out_e,
                                            beta_mul,
                                        )
                                    }
                                }
                            }
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
                            GtConsumer::JointCommitment
                            | GtConsumer::PairingBoundaryRhs
                            | GtConsumer::PairingBoundaryMillerRhs => (sel_src_key, beta_src),
                        };
                        let sel_dst = state
                            .selectors
                            .get(&sel_dst_key)
                            .expect("missing selector for dst");
                        let sel_dst_e =
                            sel_dst.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                        let dst_is_mul_lhs = matches!(edge.dst, GtConsumer::GtMulLhs { .. });
                        let dst_is_mul_rhs = matches!(edge.dst, GtConsumer::GtMulRhs { .. });
                        let dst_is_exp_base = matches!(edge.dst, GtConsumer::GtExpBase { .. });

                        let src_const =
                            match edge.src {
                                GtProducer::GtExpBase { instance } => {
                                    Some(state.exp_base_inputs_at_r[instance])
                                }
                                GtProducer::JointCommitment => Some(state.joint_at_r),
                                GtProducer::GtInput { value_id } => {
                                    Some(*state.gt_inputs_at_r.get(&value_id).expect(
                                        "missing gt_inputs_at_r entry for GtProducer::GtInput",
                                    ))
                                }
                                GtProducer::MultiMillerLoopOut { instance } => {
                                    Some(*state.miller_out_at_r.get(&instance).expect(
                                        "missing miller_out_at_r entry for MultiMillerLoopOut",
                                    ))
                                }
                                _ => None,
                            };
                        for t in 0..DEGREE {
                            let src_val = src_const.unwrap_or(src_port_e[t]);
                            let src_term = sel_src_e[t] * src_val;

                            let dst_term = if dst_is_mul_lhs {
                                sel_dst_e[t] * mul_lhs_e[t]
                            } else if dst_is_mul_rhs {
                                sel_dst_e[t] * mul_rhs_e[t]
                            } else if dst_is_exp_base {
                                sel_dst_e[t] * base_port_e[t]
                            } else {
                                // Boundary constants.
                                let const_val = match edge.dst {
                                    GtConsumer::JointCommitment => state.joint_at_r,
                                    GtConsumer::PairingBoundaryRhs => state.pairing_rhs_at_r,
                                    GtConsumer::PairingBoundaryMillerRhs => {
                                        state.pairing_miller_rhs_at_r
                                    }
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
            self.eq_s_miller_out
                .bind_parallel(r_j, BindingOrder::LowToHigh);
            for poly in self.eq_s_by_exp.iter_mut().flatten() {
                poly.bind_parallel(r_j, BindingOrder::LowToHigh);
            }
            for poly in self.rho_polys.iter_mut().flatten() {
                poly.bind_parallel(r_j, BindingOrder::LowToHigh);
            }
            for poly in self.miller_out.values_mut() {
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
            for poly in self.exp_base_inputs.iter_mut().flatten() {
                poly.bind_parallel(r_j, BindingOrder::LowToHigh);
            }
            for poly in self.exp_base_port.iter_mut().flatten() {
                poly.bind_parallel(r_j, BindingOrder::LowToHigh);
            }
            if let Some(poly) = self.joint_commitment.as_mut() {
                poly.bind_parallel(r_j, BindingOrder::LowToHigh);
            }
            if let Some(poly) = self.pairing_rhs.as_mut() {
                poly.bind_parallel(r_j, BindingOrder::LowToHigh);
            }
            if let Some(poly) = self.pairing_miller_rhs.as_mut() {
                poly.bind_parallel(r_j, BindingOrder::LowToHigh);
            }
            for poly in self.gt_inputs.values_mut() {
                poly.bind_parallel(r_j, BindingOrder::LowToHigh);
            }
        } else {
            // Bind the c-phase polynomials (must already be initialized).
            self.ensure_c_state();
            self.c_state
                .as_mut()
                .expect("c_state must be initialized")
                .bind(r_j);

            // Debug aid: after binding the final c variable, compute the wiring polynomial value
            // at the full random point r = (r_x, r_c) using the *actual* selector polynomials.
            //
            // This lets us distinguish:
            // - selector/indexing mismatches (eq_c vs selector MLE), vs
            // - sumcheck message computation bugs.
            if round + 1 == self.num_rounds()
                && std::env::var_os("JOLT_DEBUG_GT_WIRING_FINAL_EVAL").is_some()
            {
                use crate::zkvm::recursion::wiring_plan::{GtConsumer, GtProducer};
                let state = self.c_state.as_ref().expect("c_state must be initialized");

                let dummy_exp = self.num_c_vars.saturating_sub(self.k_exp);
                let dummy_mul = self.num_c_vars.saturating_sub(self.k_mul);
                let beta_exp = beta(dummy_exp);
                let beta_mul = beta(dummy_mul);
                let selector_key = |dummy: usize, k_family: usize, idx: usize| -> u64 {
                    ((dummy as u64) << 48) | ((k_family as u64) << 32) | (idx as u64)
                };
                let sel_eval = |key: u64| -> Fq {
                    state
                        .selectors
                        .get(&key)
                        .expect("missing selector")
                        .get_bound_coeff(0)
                };

                // Family-oracle evaluations at r_c.
                let rho_val = state.rho_c.get_bound_coeff(0);
                let mul_lhs_val = state.mul_lhs_c.get_bound_coeff(0);
                let mul_rhs_val = state.mul_rhs_c.get_bound_coeff(0);
                let mul_out_val = state.mul_result_c.get_bound_coeff(0);
                let base_port_val = state.base_port_c.get_bound_coeff(0);

                // Quick sanity samples: selector evals for instance 0 (if present).
                let sel_exp0 = state
                    .selectors
                    .get(&selector_key(dummy_exp, self.k_exp, 0))
                    .map(|p| p.get_bound_coeff(0));
                let sel_mul0 = state
                    .selectors
                    .get(&selector_key(dummy_mul, self.k_mul, 0))
                    .map(|p| p.get_bound_coeff(0));
                let sel_exp1 = state
                    .selectors
                    .get(&selector_key(dummy_exp, self.k_exp, 1))
                    .map(|p| p.get_bound_coeff(0));
                let sel_mul1 = state
                    .selectors
                    .get(&selector_key(dummy_mul, self.k_mul, 1))
                    .map(|p| p.get_bound_coeff(0));
                tracing::info!(
                    ?rho_val,
                    ?mul_lhs_val,
                    ?mul_rhs_val,
                    ?mul_out_val,
                    ?base_port_val,
                    ?sel_exp0,
                    ?sel_mul0,
                    ?sel_exp1,
                    ?sel_mul1,
                    "gt_wiring: debug samples at r_c"
                );

                let mut g_at_r = Fq::zero();
                let mut bucket_rho = Fq::zero();
                let mut bucket_mul = Fq::zero();
                let mut bucket_exp_base = Fq::zero();
                let mut bucket_joint = Fq::zero();
                let mut bucket_gt_input = Fq::zero();
                let mut bucket_mml = Fq::zero();
                for (edge_idx, edge) in self.edges.iter().enumerate() {
                    let lambda = self.lambdas[edge_idx];
                    let eq_s_at_r = state.edge_eq_s_at_r[edge_idx];
                    let edge_coeff = state.eq_u_at_r * lambda * eq_s_at_r;

                    // Source selector + beta + value at r.
                    let (sel_src_key, beta_src, src_val) = match edge.src {
                        GtProducer::GtExpRho { instance } => (
                            selector_key(dummy_exp, self.k_exp, instance),
                            beta_exp,
                            rho_val,
                        ),
                        GtProducer::GtMulResult { instance } => (
                            selector_key(dummy_mul, self.k_mul, instance),
                            beta_mul,
                            mul_out_val,
                        ),
                        GtProducer::GtExpBase { instance } => (
                            selector_key(dummy_exp, self.k_exp, instance),
                            beta_exp,
                            state.exp_base_inputs_at_r[instance],
                        ),
                        GtProducer::JointCommitment => {
                            // Anchor to destination family selector.
                            match edge.dst {
                                GtConsumer::GtMulLhs { instance }
                                | GtConsumer::GtMulRhs { instance } => (
                                    selector_key(dummy_mul, self.k_mul, instance),
                                    beta_mul,
                                    state.joint_at_r,
                                ),
                                GtConsumer::GtExpBase { instance } => (
                                    selector_key(dummy_exp, self.k_exp, instance),
                                    beta_exp,
                                    state.joint_at_r,
                                ),
                                GtConsumer::JointCommitment
                                | GtConsumer::PairingBoundaryRhs
                                | GtConsumer::PairingBoundaryMillerRhs => (
                                    selector_key(dummy_mul, self.k_mul, 0),
                                    beta_mul,
                                    state.joint_at_r,
                                ),
                            }
                        }
                        GtProducer::GtInput { value_id } => {
                            let v = *state.gt_inputs_at_r.get(&value_id).expect("missing gt input");
                            match edge.dst {
                                GtConsumer::GtMulLhs { instance }
                                | GtConsumer::GtMulRhs { instance } => (
                                    selector_key(dummy_mul, self.k_mul, instance),
                                    beta_mul,
                                    v,
                                ),
                                GtConsumer::GtExpBase { instance } => (
                                    selector_key(dummy_exp, self.k_exp, instance),
                                    beta_exp,
                                    v,
                                ),
                                GtConsumer::JointCommitment
                                | GtConsumer::PairingBoundaryRhs
                                | GtConsumer::PairingBoundaryMillerRhs => (
                                    selector_key(dummy_mul, self.k_mul, 0),
                                    beta_mul,
                                    v,
                                ),
                            }
                        }
                        GtProducer::MultiMillerLoopOut { instance } => {
                            let v = *state
                                .miller_out_at_r
                                .get(&instance)
                                .expect("missing mml out");
                            match edge.dst {
                                GtConsumer::GtMulLhs { instance }
                                | GtConsumer::GtMulRhs { instance } => (
                                    selector_key(dummy_mul, self.k_mul, instance),
                                    beta_mul,
                                    v,
                                ),
                                GtConsumer::GtExpBase { instance } => (
                                    selector_key(dummy_exp, self.k_exp, instance),
                                    beta_exp,
                                    v,
                                ),
                                GtConsumer::JointCommitment
                                | GtConsumer::PairingBoundaryRhs
                                | GtConsumer::PairingBoundaryMillerRhs => (
                                    selector_key(dummy_mul, self.k_mul, 0),
                                    beta_mul,
                                    v,
                                ),
                            }
                        }
                    };

                    // Destination selector + beta + value at r.
                    let (sel_dst_key, beta_dst, dst_val) = match edge.dst {
                        GtConsumer::GtMulLhs { instance } => (
                            selector_key(dummy_mul, self.k_mul, instance),
                            beta_mul,
                            mul_lhs_val,
                        ),
                        GtConsumer::GtMulRhs { instance } => (
                            selector_key(dummy_mul, self.k_mul, instance),
                            beta_mul,
                            mul_rhs_val,
                        ),
                        GtConsumer::GtExpBase { instance } => (
                            selector_key(dummy_exp, self.k_exp, instance),
                            beta_exp,
                            base_port_val,
                        ),
                        // Anchor boundary constants to src family selector.
                        GtConsumer::JointCommitment => (sel_src_key, beta_src, state.joint_at_r),
                        GtConsumer::PairingBoundaryRhs => {
                            (sel_src_key, beta_src, state.pairing_rhs_at_r)
                        }
                        GtConsumer::PairingBoundaryMillerRhs => (
                            sel_src_key,
                            beta_src,
                            state.pairing_miller_rhs_at_r,
                        ),
                    };

                    let sel_src = sel_eval(sel_src_key);
                    let sel_dst = sel_eval(sel_dst_key);
                    let contrib =
                        edge_coeff * (beta_src * sel_src * src_val - beta_dst * sel_dst * dst_val);
                    g_at_r += contrib;
                    if std::env::var_os("JOLT_DEBUG_GT_WIRING_BUCKETS").is_some() {
                        match edge.src {
                            GtProducer::GtExpRho { .. } => bucket_rho += contrib,
                            GtProducer::GtMulResult { .. } => bucket_mul += contrib,
                            GtProducer::GtExpBase { .. } => bucket_exp_base += contrib,
                            GtProducer::JointCommitment => bucket_joint += contrib,
                            GtProducer::GtInput { .. } => bucket_gt_input += contrib,
                            GtProducer::MultiMillerLoopOut { .. } => bucket_mml += contrib,
                        }
                    }
                }

                tracing::info!(?g_at_r, "gt_wiring: g(r) computed from bound c-polys (debug)");
                if std::env::var_os("JOLT_DEBUG_GT_WIRING_BUCKETS").is_some() {
                    tracing::info!(
                        ?bucket_rho,
                        ?bucket_mul,
                        ?bucket_exp_base,
                        ?bucket_joint,
                        ?bucket_gt_input,
                        ?bucket_mml,
                        "gt_wiring: prover per-src bucket sums (debug)"
                    );
                }
            }
        }
    }

    fn cache_openings(
        &self,
        _accumulator: &mut ProverOpeningAccumulator<Fq>,
        _transcript: &mut T,
        _sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        // No new openings: this instance consumes openings cached by earlier GT instances.
    }
}

#[derive(Clone, Debug)]
pub struct WiringGtVerifier {
    edges: Vec<GtWiringEdge>,
    lambdas: Vec<Fq>,
    tau: Vec<<Fq as JoltField>::Challenge>,
    pairing_boundary: PairingBoundary,
    joint_commitment: Fq12,
    gt_exp_base_inputs: Vec<Option<Fq12>>,
    /// GT-valued AST inputs referenced by the wiring plan, in the order stored in
    /// `RecursionVerifierInput.gt_inputs`.
    gt_input_values: Vec<Fq12>,
    /// Map `value_id -> index into gt_input_values` (dense, fast in the guest).
    ///
    /// `value_id` is an AST node index and is expected to be small (≲ ast.nodes.len()).
    gt_input_index_by_value_id: Vec<Option<usize>>,
    /// Global constraint indices for Multi-Miller loop outputs referenced by the wiring plan.
    miller_instance_ids: Vec<usize>,
    /// Map `constraint_idx -> index into miller_instance_ids` (dense, fast in the guest).
    miller_index_by_constraint: Vec<Option<usize>>,
    gt_exp_out_step: Vec<usize>,
    /// k_common
    num_c_vars: usize,
    k_exp: usize,
    k_mul: usize,
    /// Number of GTMul instances referenced by the wiring plan.
    ///
    /// Cached so `expected_output_claim` does not scan `edges` just to size `eq_c_mul`.
    num_mul_instances: usize,
}

impl WiringGtVerifier {
    pub fn new<T: Transcript>(input: &RecursionVerifierInput, transcript: &mut T) -> Self {
        // Must mirror prover sampling order: τ, then λ.
        let tau: Vec<<Fq as JoltField>::Challenge> =
            transcript.challenge_vector_optimized::<Fq>(ELEM_VARS);
        let edges = input.wiring.gt.clone();
        let lambdas = transcript.challenge_vector(edges.len());
        if std::env::var_os("JOLT_DEBUG_GT_WIRING_CHALLENGES").is_some() {
            let mut edge_fp: u64 = 0;
            for e in edges.iter() {
                let src_tag: u64 = match e.src {
                    GtProducer::GtExpRho { .. } => 1,
                    GtProducer::GtMulResult { .. } => 2,
                    GtProducer::GtExpBase { .. } => 3,
                    GtProducer::JointCommitment => 4,
                    GtProducer::GtInput { .. } => 5,
                    GtProducer::MultiMillerLoopOut { .. } => 6,
                };
                let src_idx: u64 = match e.src {
                    GtProducer::GtExpRho { instance }
                    | GtProducer::GtMulResult { instance }
                    | GtProducer::GtExpBase { instance }
                    | GtProducer::MultiMillerLoopOut { instance } => instance as u64,
                    GtProducer::GtInput { value_id } => value_id as u64,
                    GtProducer::JointCommitment => 0,
                };
                let dst_tag: u64 = match e.dst {
                    GtConsumer::GtMulLhs { .. } => 11,
                    GtConsumer::GtMulRhs { .. } => 12,
                    GtConsumer::GtExpBase { .. } => 13,
                    GtConsumer::JointCommitment => 14,
                    GtConsumer::PairingBoundaryRhs => 15,
                    GtConsumer::PairingBoundaryMillerRhs => 16,
                };
                let dst_idx: u64 = match e.dst {
                    GtConsumer::GtMulLhs { instance }
                    | GtConsumer::GtMulRhs { instance }
                    | GtConsumer::GtExpBase { instance } => instance as u64,
                    _ => 0,
                };
                let v = (src_tag << 56) ^ (dst_tag << 48) ^ (src_idx << 16) ^ dst_idx;
                edge_fp = edge_fp.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(v);
            }
            tracing::info!(
                tau0 = ?tau.get(0),
                lambda0 = ?lambdas.get(0),
                edges = edges.len(),
                edge0 = ?edges.get(0),
                edge_last = ?edges.last(),
                edge_fp = edge_fp,
                "gt_wiring: verifier sampled challenges"
            );
        }

        let num_c_vars = k_gt(&input.constraint_types);
        let k_exp = k_exp(&input.constraint_types);
        let k_mul = k_mul(&input.constraint_types);

        let gt_exp_out_step = input
            .gt_exp_public_inputs
            .iter()
            .map(|p| {
                let digits_len = p.scalar_bits.len().div_ceil(2);
                let max_s = (1 << STEP_VARS) - 1;
                digits_len.min(max_s)
            })
            .collect();
        let gt_exp_base_inputs = input.gt_exp_base_inputs.clone();
        // Build a dense value_id -> index mapping for GT inputs to avoid BTreeMap overhead in the
        // guest hot path (`expected_output_claim`).
        let max_value_id = input
            .gt_inputs
            .iter()
            .map(|(value_id, _)| *value_id as usize)
            .max()
            .unwrap_or(0);
        let mut gt_input_index_by_value_id = if input.gt_inputs.is_empty() {
            Vec::new()
        } else {
            vec![None; max_value_id + 1]
        };
        let mut gt_input_values: Vec<Fq12> = Vec::with_capacity(input.gt_inputs.len());
        for (idx, (value_id, fq12)) in input.gt_inputs.iter().enumerate() {
            let vidx = *value_id as usize;
            if vidx < gt_input_index_by_value_id.len() {
                gt_input_index_by_value_id[vidx] = Some(idx);
            }
            gt_input_values.push(*fq12);
        }

        // Build dense constraint_idx -> index mapping for MultiMillerLoop outputs.
        let mut max_miller = None::<usize>;
        for edge in &edges {
            if let GtProducer::MultiMillerLoopOut { instance } = edge.src {
                max_miller = Some(max_miller.map_or(instance, |m| m.max(instance)));
            }
        }
        let mut miller_instance_ids: Vec<usize> = Vec::new();
        let mut miller_index_by_constraint: Vec<Option<usize>> = match max_miller {
            None => Vec::new(),
            Some(m) => vec![None; m + 1],
        };
        if !miller_index_by_constraint.is_empty() {
            for edge in &edges {
                if let GtProducer::MultiMillerLoopOut { instance } = edge.src {
                    if instance < miller_index_by_constraint.len()
                        && miller_index_by_constraint[instance].is_none()
                    {
                        let idx = miller_instance_ids.len();
                        miller_instance_ids.push(instance);
                        miller_index_by_constraint[instance] = Some(idx);
                    }
                }
            }
        }

        // Cache #mul instances referenced by the edge list (avoid re-scanning each verify).
        let mut max_mul = None::<usize>;
        for edge in &edges {
            match edge.src {
                GtProducer::GtMulResult { instance } => {
                    max_mul = Some(max_mul.map_or(instance, |m| m.max(instance)));
                }
                GtProducer::GtExpRho { .. }
                | GtProducer::GtExpBase { .. }
                | GtProducer::MultiMillerLoopOut { .. }
                | GtProducer::JointCommitment
                | GtProducer::GtInput { .. } => {}
            }
            match edge.dst {
                GtConsumer::GtMulLhs { instance } | GtConsumer::GtMulRhs { instance } => {
                    max_mul = Some(max_mul.map_or(instance, |m| m.max(instance)));
                }
                GtConsumer::GtExpBase { .. }
                | GtConsumer::JointCommitment
                | GtConsumer::PairingBoundaryRhs
                | GtConsumer::PairingBoundaryMillerRhs => {}
            }
        }
        let num_mul_instances = max_mul.map_or(0, |m| m + 1);

        Self {
            edges,
            lambdas,
            tau,
            pairing_boundary: input.pairing_boundary.clone(),
            joint_commitment: input.joint_commitment,
            gt_exp_base_inputs,
            gt_input_values,
            gt_input_index_by_value_id,
            miller_instance_ids,
            miller_index_by_constraint,
            gt_exp_out_step,
            num_c_vars,
            k_exp,
            k_mul,
            num_mul_instances,
        }
    }

    #[inline]
    fn eq_c_tail(&self, r_c_tail: &[Fq], idx: usize, k_family: usize) -> Fq {
        debug_assert_eq!(r_c_tail.len(), k_family);
        let mut out = Fq::one();
        for (i, &r_i) in r_c_tail.iter().enumerate() {
            if ((idx >> i) & 1) == 1 {
                out *= r_i;
            } else {
                out *= Fq::one() - r_i;
            }
        }
        out
    }
}

impl<T: Transcript> SumcheckInstanceVerifier<Fq, T> for WiringGtVerifier {
    fn cycle_tracking_label(&self) -> Option<&'static str> {
        Some(CYCLE_VERIFY_RECURSION_STAGE2_GT_WIRING_TOTAL)
    }

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

        // x11 convention: (x, s) where x are the low 4 bits (elem/u) and s are the high 7 bits (step).
        let r_elem_chal = &sumcheck_challenges[..ELEM_VARS];
        let r_step = &sumcheck_challenges[ELEM_VARS..ELEM_VARS + STEP_VARS];
        let r_c = &sumcheck_challenges[X_VARS..];
        debug_assert_eq!(r_c.len(), self.num_c_vars);

        let eq_u = eq_lsb_mle::<Fq>(&self.tau, r_elem_chal);

        let dummy_exp = self.num_c_vars.saturating_sub(self.k_exp);
        let dummy_mul = self.num_c_vars.saturating_sub(self.k_mul);
        let beta_exp = beta(dummy_exp);
        let beta_mul = beta(dummy_mul);

        // Port openings (already cached by earlier GT instances in Stage 2).
        let rho_val = acc.get_virtual_polynomial_claim(
            VirtualPolynomial::Recursion(RecursionPoly::GtExp {
                term: GtExpTerm::Rho,
            }),
            SumcheckId::GtExpClaimReduction,
        );
        let mul_lhs_val = acc.get_virtual_polynomial_claim(
            VirtualPolynomial::Recursion(RecursionPoly::GtMul {
                term: GtMulTerm::Lhs,
            }),
            SumcheckId::GtMul,
        );
        let mul_rhs_val = acc.get_virtual_polynomial_claim(
            VirtualPolynomial::Recursion(RecursionPoly::GtMul {
                term: GtMulTerm::Rhs,
            }),
            SumcheckId::GtMul,
        );
        let mul_out_val = acc.get_virtual_polynomial_claim(
            VirtualPolynomial::Recursion(RecursionPoly::GtMul {
                term: GtMulTerm::Result,
            }),
            SumcheckId::GtMul,
        );
        let base_port_val = acc.get_virtual_polynomial_claim(
            VirtualPolynomial::Recursion(RecursionPoly::GtExp {
                term: GtExpTerm::Base,
            }),
            SumcheckId::GtExpBaseClaimReduction,
        );

        // Debug aid: print the opening points consumed by GT wiring, to verify that all
        // "producer" claims are being opened at the same effective (x,c) point that the wiring
        // backend assumes.
        //
        // This is particularly important for MultiMillerLoop, whose sumcheck has fewer rounds and
        // is therefore sensitive to batched-sumcheck round alignment.
        if std::env::var_os("JOLT_DEBUG_GT_WIRING_POINTS").is_some() {
            use crate::poly::opening_proof::OpeningAccumulator;
            tracing::info!(
                step_rounds = STEP_VARS,
                elem_rounds = ELEM_VARS,
                c_rounds = self.num_c_vars,
                "gt_wiring: sumcheck_challenges partitioned as (elem, step, c)"
            );
            tracing::info!(?r_step, ?r_elem_chal, "gt_wiring: r_step/r_elem (challenge order)");
            tracing::info!(r_c_len = r_c.len(), "gt_wiring: r_c len");

            let (rho_pt, _rho_claim) = acc.get_virtual_polynomial_opening(
                VirtualPolynomial::Recursion(RecursionPoly::GtExp {
                    term: GtExpTerm::Rho,
                }),
                SumcheckId::GtExpClaimReduction,
            );
            tracing::info!(
                rho_opening_len = rho_pt.r.len(),
                "gt_wiring: rho opening point len"
            );

            let (mul_lhs_pt, _mul_lhs_claim) = acc.get_virtual_polynomial_opening(
                VirtualPolynomial::Recursion(RecursionPoly::GtMul {
                    term: GtMulTerm::Lhs,
                }),
                SumcheckId::GtMul,
            );
            tracing::info!(
                mul_lhs_opening_len = mul_lhs_pt.r.len(),
                "gt_wiring: gt_mul lhs opening point len"
            );

            // If any MML instance is referenced, also print the opening point for its `F` column.
            if let Some(&global_idx) = self.miller_instance_ids.first() {
                let (mml_pt, _mml_claim) = acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::multi_miller_loop_f(global_idx),
                    SumcheckId::MultiMillerLoop,
                );
                tracing::info!(
                    mml_opening_len = mml_pt.r.len(),
                    mml_constraint_idx = global_idx,
                    "gt_wiring: MultiMillerLoop::F opening point len"
                );
                if mml_pt.r.len() >= STEP_VARS + ELEM_VARS {
                    let mml_elem = &mml_pt.r[..ELEM_VARS];
                    let mml_step = &mml_pt.r[ELEM_VARS..ELEM_VARS + STEP_VARS];
                    tracing::info!(?mml_step, ?mml_elem, "gt_wiring: MML opening step/elem");
                }
            }
        }

        // Boundary constants at r_elem.
        let mut r_elem = [Fq::zero(); ELEM_VARS];
        for (i, &c) in r_elem_chal.iter().enumerate() {
            r_elem[i] = c.into();
        }
        let elem_weights = elem_weights_4var_lsb_first(r_elem);
        let joint_eval = eval_fq12_packed_at_with_weights(&self.joint_commitment, &elem_weights);
        let pairing_rhs_eval =
            eval_fq12_packed_at_with_weights(&self.pairing_boundary.rhs, &elem_weights);
        let pairing_miller_rhs_eval =
            eval_fq12_packed_at_with_weights(&self.pairing_boundary.miller_rhs, &elem_weights);

        // Precompute eq(s, s_out) values once per GTExp instance; these are reused across edges.
        //
        // NOTE: `STEP_VARS = 7`, so this is a tiny fixed-size computation; for small `num_exp`
        // it is cheaper than building the full 2^7 table.
        let mut r_step_fq = [Fq::zero(); STEP_VARS];
        for (i, &c) in r_step.iter().enumerate() {
            r_step_fq[i] = c.into();
        }
        let eq_s_zero: Fq = r_step_fq.iter().map(|&r_b| Fq::one() - r_b).product();
        // Eq(s, 127) where 127 = (1<<7)-1 has all step bits set.
        let eq_s_miller: Fq = r_step_fq.iter().product();
        let eq_s_exp: Vec<Fq> = self
            .gt_exp_out_step
            .iter()
            .map(|&s_out| {
                r_step_fq
                    .iter()
                    .enumerate()
                    .map(|(b, &r_b)| {
                        if ((s_out >> b) & 1) == 1 {
                            r_b
                        } else {
                            Fq::one() - r_b
                        }
                    })
                    .product()
            })
            .collect();

        if std::env::var_os("JOLT_DEBUG_GT_WIRING_EQS").is_some() {
            tracing::info!(
                ?eq_s_zero,
                ?eq_s_miller,
                eq_s_exp0 = ?eq_s_exp.get(0).copied(),
                "gt_wiring: verifier eq_s samples at r_step (direct)"
            );
        }

        // Precompute eq(c, idx) once per referenced instance, then accumulate coefficients so we
        // only touch expensive values (Fq12 bases) once.
        let num_exp = self.gt_exp_out_step.len();

        let num_mul = self.num_mul_instances;

        let r_c_fq: Vec<Fq> = r_c.iter().map(|c| (*c).into()).collect();
        let r_c_exp_tail = &r_c_fq[dummy_exp..];
        let r_c_mul_tail = &r_c_fq[dummy_mul..];

        let eq_c_exp: Vec<Fq> = (0..num_exp)
            .map(|i| self.eq_c_tail(r_c_exp_tail, i, self.k_exp))
            .collect();
        let eq_c_mul: Vec<Fq> = (0..num_mul)
            .map(|i| self.eq_c_tail(r_c_mul_tail, i, self.k_mul))
            .collect();

        if std::env::var_os("JOLT_DEBUG_GT_WIRING_EXPECTED_SAMPLES").is_some() {
            let eq_c_exp0 = eq_c_exp.get(0).copied();
            let eq_c_mul0 = eq_c_mul.get(0).copied();
            let eq_c_exp1 = eq_c_exp.get(1).copied();
            let eq_c_mul1 = eq_c_mul.get(1).copied();
            tracing::info!(
                ?rho_val,
                ?mul_lhs_val,
                ?mul_rhs_val,
                ?mul_out_val,
                ?base_port_val,
                ?eq_c_exp0,
                ?eq_c_mul0,
                ?eq_c_exp1,
                ?eq_c_mul1,
                "gt_wiring: verifier samples at r (claims + eq_c[0])"
            );
        }
        // Evaluate the wiring polynomial directly per edge (mirrors prover logic more closely).
        //
        // This avoids subtle bugs in coefficient aggregation, at the cost of O(|edges|) work.
        let alpha = fq12_eval_linear_coeffs(&elem_weights);

        // Precompute evals for boundary GT inputs (value_id -> eval at r_elem).
        let gt_input_eval_by_value_id: Vec<Option<Fq>> = self
            .gt_input_index_by_value_id
            .iter()
            .map(|opt_idx| {
                opt_idx.map(|idx| eval_fq12_packed_at_via_linear_coeffs(&self.gt_input_values[idx], &alpha))
            })
            .collect();

        // Precompute GTExp base input evals (indexed by GTExp instance).
        let mut base_input_eval: Vec<Option<Fq>> = Vec::with_capacity(num_exp);
        for i in 0..num_exp {
            let fq12 = self.gt_exp_base_inputs[i];
            base_input_eval.push(fq12.map(|b| eval_fq12_packed_at_via_linear_coeffs(&b, &alpha)));
        }

        if std::env::var_os("JOLT_DEBUG_GT_WIRING_EXPECTED_SAMPLES").is_some() {
            let gt_input2_eval = gt_input_eval_by_value_id.get(2).copied().flatten();
            let base_input0_eval = base_input_eval.get(0).copied().flatten();
            tracing::info!(
                ?eq_u,
                ?joint_eval,
                ?pairing_rhs_eval,
                ?pairing_miller_rhs_eval,
                ?gt_input2_eval,
                ?base_input0_eval,
                "gt_wiring: verifier boundary/input eval samples at r_u"
            );
        }

        let mut sum = Fq::zero();
        let mut bucket_rho = Fq::zero();
        let mut bucket_mul = Fq::zero();
        let mut bucket_exp_base = Fq::zero();
        let mut bucket_joint = Fq::zero();
        let mut bucket_gt_input = Fq::zero();
        let mut bucket_mml = Fq::zero();
        for (lambda, edge) in self.lambdas.iter().zip(self.edges.iter()) {
            // Eq(s, s_out) depends on the *source* family.
            let eq_s = match edge.src {
                GtProducer::GtExpRho { instance } => eq_s_exp[instance],
                GtProducer::MultiMillerLoopOut { .. } => eq_s_miller,
                GtProducer::GtMulResult { .. }
                | GtProducer::GtExpBase { .. }
                | GtProducer::JointCommitment
                | GtProducer::GtInput { .. } => eq_s_zero,
            };
            let lambda_fq: Fq = (*lambda).into();
            let scale = lambda_fq * eq_s;

            // Source weight and value at r.
            let (src_w, src_v): (Fq, Fq) = match edge.src {
                GtProducer::GtExpRho { instance } => (beta_exp * eq_c_exp[instance], rho_val),
                GtProducer::GtMulResult { instance } => (beta_mul * eq_c_mul[instance], mul_out_val),
                GtProducer::GtExpBase { instance } => (
                    beta_exp * eq_c_exp[instance],
                    base_input_eval[instance]
                        .expect("missing gt_exp_base_inputs entry for GTExp base input producer"),
                ),
                GtProducer::JointCommitment => {
                    // Anchor this global constant source to the destination family selector.
                    let w = match edge.dst {
                        GtConsumer::GtMulLhs { instance } | GtConsumer::GtMulRhs { instance } => {
                            beta_mul * eq_c_mul[instance]
                        }
                        GtConsumer::GtExpBase { instance } => beta_exp * eq_c_exp[instance],
                        GtConsumer::JointCommitment
                        | GtConsumer::PairingBoundaryRhs
                        | GtConsumer::PairingBoundaryMillerRhs => {
                            // Should not happen in well-formed plans; fall back to 1.
                            Fq::one()
                        }
                    };
                    (w, joint_eval)
                }
                GtProducer::GtInput { value_id } => {
                    let w = match edge.dst {
                        GtConsumer::GtMulLhs { instance } | GtConsumer::GtMulRhs { instance } => {
                            beta_mul * eq_c_mul[instance]
                        }
                        GtConsumer::GtExpBase { instance } => beta_exp * eq_c_exp[instance],
                        GtConsumer::JointCommitment
                        | GtConsumer::PairingBoundaryRhs
                        | GtConsumer::PairingBoundaryMillerRhs => {
                            // Should not happen in well-formed plans; fall back to 1.
                            Fq::one()
                        }
                    };
                    let vidx = value_id as usize;
                    let v = gt_input_eval_by_value_id
                        .get(vidx)
                        .and_then(|x| *x)
                        .expect("missing gt_inputs entry for GtProducer::GtInput");
                    (w, v)
                }
                GtProducer::MultiMillerLoopOut { instance } => {
                    // Anchor this proved x-only value to the destination family selector.
                    let w = match edge.dst {
                        GtConsumer::GtMulLhs { instance } | GtConsumer::GtMulRhs { instance } => {
                            beta_mul * eq_c_mul[instance]
                        }
                        GtConsumer::GtExpBase { instance } => beta_exp * eq_c_exp[instance],
                        GtConsumer::JointCommitment
                        | GtConsumer::PairingBoundaryRhs
                        | GtConsumer::PairingBoundaryMillerRhs => {
                            // Should not happen in well-formed plans; fall back to 1.
                            Fq::one()
                        }
                    };
                    let idx = self
                        .miller_index_by_constraint
                        .get(instance)
                        .and_then(|x| *x)
                        .expect("missing miller index for MultiMillerLoopOut");
                    let global_idx = self.miller_instance_ids[idx];
                    let v = acc.get_virtual_polynomial_claim(
                        VirtualPolynomial::multi_miller_loop_f(global_idx),
                        SumcheckId::MultiMillerLoop,
                    );
                    (w, v)
                }
            };

            // Destination weight and value at r.
            let (dst_w, dst_v): (Fq, Fq) = match edge.dst {
                GtConsumer::GtMulLhs { instance } => (beta_mul * eq_c_mul[instance], mul_lhs_val),
                GtConsumer::GtMulRhs { instance } => (beta_mul * eq_c_mul[instance], mul_rhs_val),
                GtConsumer::GtExpBase { instance } => (beta_exp * eq_c_exp[instance], base_port_val),
                // Anchor globals to src.
                GtConsumer::JointCommitment => (src_w, joint_eval),
                GtConsumer::PairingBoundaryRhs => (src_w, pairing_rhs_eval),
                GtConsumer::PairingBoundaryMillerRhs => (src_w, pairing_miller_rhs_eval),
            };

            let contrib = (eq_u * scale) * (src_w * src_v - dst_w * dst_v);
            sum += contrib;
            if std::env::var_os("JOLT_DEBUG_GT_WIRING_BUCKETS").is_some() {
                match edge.src {
                    GtProducer::GtExpRho { .. } => bucket_rho += contrib,
                    GtProducer::GtMulResult { .. } => bucket_mul += contrib,
                    GtProducer::GtExpBase { .. } => bucket_exp_base += contrib,
                    GtProducer::JointCommitment => bucket_joint += contrib,
                    GtProducer::GtInput { .. } => bucket_gt_input += contrib,
                    GtProducer::MultiMillerLoopOut { .. } => bucket_mml += contrib,
                }
            }
        }

        if std::env::var_os("JOLT_DEBUG_GT_WIRING_BUCKETS").is_some() {
            tracing::info!(
                ?bucket_rho,
                ?bucket_mul,
                ?bucket_exp_base,
                ?bucket_joint,
                ?bucket_gt_input,
                ?bucket_mml,
                ?sum,
                "gt_wiring: verifier per-src bucket sums (debug)"
            );
        }

        sum
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

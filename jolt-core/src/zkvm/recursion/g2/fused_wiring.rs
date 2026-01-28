//! Fused G2 wiring (copy/boundary) sumcheck (GT-style split-k aware).
//!
//! This mirrors:
//! - `gt/fused_wiring.rs` (split-k + β(dummy) normalization), and
//! - `g1/fused_wiring.rs` (step-then-c phase split for performance),
//!   but for G2 points over Fq2 (batched into a single Fq scalar using μ powers).
//!
//! Variable order (Stage 2):
//! - Phase 1 (x): bind `s` (8 step vars).
//! - Phase 2 (c): bind `c_common` (k vars), where `k = k_g2 = max(k_smul, k_add)`.
//!
//! Split-k convention (dummy bits):
//! - Dummy bits are the *low* bits of `c_common`. Family-local bits are the suffix.
//! - For a family with `k_family` bits:
//!   - `dummy = k_common - k_family`
//!   - selectors use `beta(dummy) * Eq(c_tail, idx)` where `c_tail = c_common[dummy..]`.

use ark_bn254::{Fq, G2Affine};
use ark_ec::AffineRepr;
use ark_ff::{Field, One, Zero};
use rayon::prelude::*;

use crate::{
    field::JoltField,
    poly::{
        dense_mlpoly::DensePolynomial,
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
            constraints::system::{index_to_binary, ConstraintSystem, G2AddNative},
            g2::indexing::{k_add, k_g2, k_smul},
            gt::types::eq_lsb_evals,
            verifier::RecursionVerifierInput,
            wiring_plan::{G2ValueRef, G2WiringEdge},
            ConstraintType,
        },
        witness::VirtualPolynomial,
    },
};

pub(crate) const STEP_VARS: usize = 8;
const DEGREE: usize = 2;

type G2Point5 = (Fq, Fq, Fq, Fq, Fq); // (x0,x1,y0,y1,ind)

#[inline]
fn g2_const_from_affine(p: &G2Affine) -> G2Point5 {
    if p.is_zero() {
        (Fq::zero(), Fq::zero(), Fq::zero(), Fq::zero(), Fq::one())
    } else {
        (p.x.c0, p.x.c1, p.y.c0, p.y.c1, Fq::zero())
    }
}

#[inline]
fn beta(dummy_bits: usize) -> Fq {
    // 2^{-dummy_bits}
    if dummy_bits == 0 {
        return Fq::one();
    }
    let two_inv = ark_ff::Field::inverse(&Fq::from(2u64)).expect("2 has inverse in field");
    two_inv.pow([dummy_bits as u64])
}

#[derive(Clone, Debug)]
struct CPhaseState {
    /// Eq(s, 255) evaluated at the bound `r_step`.
    eq_step_at_r: Fq,
    /// Batched port polynomials over the **common** c domain (k vars), replicated across dummy bits.
    smul_out_batched_c: MultilinearPolynomial<Fq>,
    add_in_p_batched_c: MultilinearPolynomial<Fq>,
    add_in_q_batched_c: MultilinearPolynomial<Fq>,
    add_out_batched_c: MultilinearPolynomial<Fq>,
    /// Selector polynomials `Eq(c_tail, idx)` replicated across dummy bits.
    selectors: std::collections::HashMap<u64, MultilinearPolynomial<Fq>>,
}

impl CPhaseState {
    fn bind(&mut self, r_j: <Fq as JoltField>::Challenge) {
        for p in [
            &mut self.smul_out_batched_c,
            &mut self.add_in_p_batched_c,
            &mut self.add_in_q_batched_c,
            &mut self.add_out_batched_c,
        ] {
            p.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for sel in self.selectors.values_mut() {
            sel.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
    }
}

#[derive(Clone, Debug)]
pub struct FusedWiringG2Prover<T: Transcript> {
    edges: Vec<G2WiringEdge>,
    lambdas: Vec<Fq>,
    // batching coefficients for (x.c0, x.c1, y.c0, y.c1, ind)
    mu: Fq,
    mu2: Fq,
    mu3: Fq,
    mu4: Fq,

    // common index sizes
    num_c_vars: usize, // k_g2
    k_smul: usize,
    k_add: usize,

    // --- x-phase (step) polynomials ---
    eq_step_poly: MultilinearPolynomial<Fq>,

    // Scalar mul output polys (8-var), indexed by scalar-mul instance.
    xa_next_c0: Vec<Option<MultilinearPolynomial<Fq>>>,
    xa_next_c1: Vec<Option<MultilinearPolynomial<Fq>>>,
    ya_next_c0: Vec<Option<MultilinearPolynomial<Fq>>>,
    ya_next_c1: Vec<Option<MultilinearPolynomial<Fq>>>,
    a_ind: Vec<Option<MultilinearPolynomial<Fq>>>,

    // Add rows (0-var scalars).
    add_rows: Vec<Option<G2AddNative>>,

    // Scalar-mul base constants, indexed by scalar-mul instance.
    smul_base: Vec<Option<G2Point5>>,

    // Pairing boundary constants.
    pairing_p1: G2Point5,
    pairing_p2: G2Point5,
    pairing_p3: G2Point5,

    c_state: Option<CPhaseState>,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Transcript> FusedWiringG2Prover<T> {
    pub fn new(
        cs: &ConstraintSystem,
        edges: Vec<G2WiringEdge>,
        pairing_boundary: &PairingBoundary,
        transcript: &mut T,
    ) -> Self {
        // Selector point a_G2 = 255 = [1,1,...,1] (LSB-first order).
        let a_g2: Vec<<Fq as JoltField>::Challenge> =
            (0..STEP_VARS).map(|_| Fq::one().into()).collect();
        let eq_step_poly = MultilinearPolynomial::from(eq_lsb_evals::<Fq>(&a_g2));

        let mu: Fq = transcript.challenge_scalar();
        let mu2 = mu * mu;
        let mu3 = mu2 * mu;
        let mu4 = mu2 * mu2;
        let lambdas: Vec<Fq> = transcript.challenge_vector(edges.len());

        let num_smul = cs.g2_scalar_mul_rows.len();
        let num_add = cs.g2_add_rows.len();

        let mut need_smul_out = vec![false; num_smul];
        let mut need_smul_base = vec![false; num_smul];
        let mut need_add = vec![false; num_add];

        for e in &edges {
            for v in [e.src, e.dst] {
                match v {
                    G2ValueRef::G2ScalarMulOut { instance } => {
                        if instance < num_smul {
                            need_smul_out[instance] = true;
                        }
                    }
                    G2ValueRef::G2ScalarMulBase { instance } => {
                        if instance < num_smul {
                            need_smul_base[instance] = true;
                        }
                    }
                    G2ValueRef::G2AddOut { instance }
                    | G2ValueRef::G2AddInP { instance }
                    | G2ValueRef::G2AddInQ { instance } => {
                        if instance < num_add {
                            need_add[instance] = true;
                        }
                    }
                    _ => {}
                }
            }
        }

        let xa_next_c0 = need_smul_out
            .iter()
            .enumerate()
            .map(|(i, &need)| {
                need.then(|| {
                    MultilinearPolynomial::from(cs.g2_scalar_mul_rows[i].x_a_next_c0.clone())
                })
            })
            .collect();
        let xa_next_c1 = need_smul_out
            .iter()
            .enumerate()
            .map(|(i, &need)| {
                need.then(|| {
                    MultilinearPolynomial::from(cs.g2_scalar_mul_rows[i].x_a_next_c1.clone())
                })
            })
            .collect();
        let ya_next_c0 = need_smul_out
            .iter()
            .enumerate()
            .map(|(i, &need)| {
                need.then(|| {
                    MultilinearPolynomial::from(cs.g2_scalar_mul_rows[i].y_a_next_c0.clone())
                })
            })
            .collect();
        let ya_next_c1 = need_smul_out
            .iter()
            .enumerate()
            .map(|(i, &need)| {
                need.then(|| {
                    MultilinearPolynomial::from(cs.g2_scalar_mul_rows[i].y_a_next_c1.clone())
                })
            })
            .collect();
        let a_ind = need_smul_out
            .into_iter()
            .enumerate()
            .map(|(i, need)| {
                need.then(|| {
                    MultilinearPolynomial::from(cs.g2_scalar_mul_rows[i].a_indicator.clone())
                })
            })
            .collect();

        let smul_base = need_smul_base
            .into_iter()
            .enumerate()
            .map(|(i, need)| {
                need.then(|| {
                    let (x, y) = cs.g2_scalar_mul_rows[i].base_point;
                    (x.c0, x.c1, y.c0, y.c1, Fq::zero())
                })
            })
            .collect();

        let add_rows = need_add
            .into_iter()
            .enumerate()
            .map(|(i, need)| need.then(|| cs.g2_add_rows[i]))
            .collect();

        let pairing_p1 = g2_const_from_affine(&pairing_boundary.p1_g2);
        let pairing_p2 = g2_const_from_affine(&pairing_boundary.p2_g2);
        let pairing_p3 = g2_const_from_affine(&pairing_boundary.p3_g2);

        let num_c_vars = k_g2(&cs.constraint_types);
        let k_smul = k_smul(&cs.constraint_types);
        let k_add = k_add(&cs.constraint_types);

        Self {
            edges,
            lambdas,
            mu,
            mu2,
            mu3,
            mu4,
            num_c_vars,
            k_smul,
            k_add,
            eq_step_poly,
            xa_next_c0,
            xa_next_c1,
            ya_next_c0,
            ya_next_c1,
            a_ind,
            add_rows,
            smul_base,
            pairing_p1,
            pairing_p2,
            pairing_p3,
            c_state: None,
            _marker: std::marker::PhantomData,
        }
    }

    #[inline]
    fn eval_value_evals_batched(&self, v: G2ValueRef, i: usize) -> [Fq; DEGREE] {
        match v {
            G2ValueRef::G2ScalarMulOut { instance } => {
                let x0 = self.xa_next_c0[instance]
                    .as_ref()
                    .unwrap()
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let x1 = self.xa_next_c1[instance]
                    .as_ref()
                    .unwrap()
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let y0 = self.ya_next_c0[instance]
                    .as_ref()
                    .unwrap()
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let y1 = self.ya_next_c1[instance]
                    .as_ref()
                    .unwrap()
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let ind = self.a_ind[instance]
                    .as_ref()
                    .unwrap()
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let mut out = [Fq::zero(); DEGREE];
                for t in 0..DEGREE {
                    out[t] = x0[t]
                        + self.mu * x1[t]
                        + self.mu2 * y0[t]
                        + self.mu3 * y1[t]
                        + self.mu4 * ind[t];
                }
                out
            }
            G2ValueRef::G2ScalarMulBase { instance } => {
                let (x0, x1, y0, y1, ind) = self.smul_base[instance].unwrap();
                let v = x0 + self.mu * x1 + self.mu2 * y0 + self.mu3 * y1 + self.mu4 * ind;
                [v; DEGREE]
            }
            G2ValueRef::G2AddOut { instance } => {
                let row = self.add_rows[instance].unwrap();
                let v = row.x_r_c0
                    + self.mu * row.x_r_c1
                    + self.mu2 * row.y_r_c0
                    + self.mu3 * row.y_r_c1
                    + self.mu4 * row.ind_r;
                [v; DEGREE]
            }
            G2ValueRef::G2AddInP { instance } => {
                let row = self.add_rows[instance].unwrap();
                let v = row.x_p_c0
                    + self.mu * row.x_p_c1
                    + self.mu2 * row.y_p_c0
                    + self.mu3 * row.y_p_c1
                    + self.mu4 * row.ind_p;
                [v; DEGREE]
            }
            G2ValueRef::G2AddInQ { instance } => {
                let row = self.add_rows[instance].unwrap();
                let v = row.x_q_c0
                    + self.mu * row.x_q_c1
                    + self.mu2 * row.y_q_c0
                    + self.mu3 * row.y_q_c1
                    + self.mu4 * row.ind_q;
                [v; DEGREE]
            }
            G2ValueRef::PairingBoundaryP1 => {
                let (x0, x1, y0, y1, ind) = self.pairing_p1;
                let v = x0 + self.mu * x1 + self.mu2 * y0 + self.mu3 * y1 + self.mu4 * ind;
                [v; DEGREE]
            }
            G2ValueRef::PairingBoundaryP2 => {
                let (x0, x1, y0, y1, ind) = self.pairing_p2;
                let v = x0 + self.mu * x1 + self.mu2 * y0 + self.mu3 * y1 + self.mu4 * ind;
                [v; DEGREE]
            }
            G2ValueRef::PairingBoundaryP3 => {
                let (x0, x1, y0, y1, ind) = self.pairing_p3;
                let v = x0 + self.mu * x1 + self.mu2 * y0 + self.mu3 * y1 + self.mu4 * ind;
                [v; DEGREE]
            }
        }
    }

    fn selector_key(dummy: usize, k_family: usize, idx: usize) -> u64 {
        ((dummy as u64) << 48) | ((k_family as u64) << 32) | (idx as u64)
    }

    fn ensure_c_state(&mut self) {
        if self.c_state.is_some() {
            return;
        }
        debug_assert_eq!(self.eq_step_poly.get_num_vars(), 0);
        let eq_step_at_r = self.eq_step_poly.get_bound_coeff(0);

        let k_common = self.num_c_vars;
        let blocks = 1usize << k_common;

        let dummy_smul = k_common.saturating_sub(self.k_smul);
        let dummy_add = k_common.saturating_sub(self.k_add);

        // Scalar-mul out batched at r_step (per instance), replicated across dummy bits.
        let mut smul_out = vec![Fq::zero(); blocks];
        for c_common in 0..blocks {
            let c_smul = c_common >> dummy_smul;
            if c_smul < self.xa_next_c0.len() {
                if let (Some(x0), Some(x1), Some(y0), Some(y1), Some(ind)) = (
                    &self.xa_next_c0[c_smul],
                    &self.xa_next_c1[c_smul],
                    &self.ya_next_c0[c_smul],
                    &self.ya_next_c1[c_smul],
                    &self.a_ind[c_smul],
                ) {
                    smul_out[c_common] = x0.get_bound_coeff(0)
                        + self.mu * x1.get_bound_coeff(0)
                        + self.mu2 * y0.get_bound_coeff(0)
                        + self.mu3 * y1.get_bound_coeff(0)
                        + self.mu4 * ind.get_bound_coeff(0);
                }
            }
        }

        // Add ports batched, replicated across dummy bits.
        let mut add_p = vec![Fq::zero(); blocks];
        let mut add_q = vec![Fq::zero(); blocks];
        let mut add_r = vec![Fq::zero(); blocks];
        for c_common in 0..blocks {
            let c_add = c_common >> dummy_add;
            if c_add < self.add_rows.len() {
                if let Some(row) = self.add_rows[c_add] {
                    add_p[c_common] = row.x_p_c0
                        + self.mu * row.x_p_c1
                        + self.mu2 * row.y_p_c0
                        + self.mu3 * row.y_p_c1
                        + self.mu4 * row.ind_p;
                    add_q[c_common] = row.x_q_c0
                        + self.mu * row.x_q_c1
                        + self.mu2 * row.y_q_c0
                        + self.mu3 * row.y_q_c1
                        + self.mu4 * row.ind_q;
                    add_r[c_common] = row.x_r_c0
                        + self.mu * row.x_r_c1
                        + self.mu2 * row.y_r_c0
                        + self.mu3 * row.y_r_c1
                        + self.mu4 * row.ind_r;
                }
            }
        }

        let smul_out_batched_c =
            MultilinearPolynomial::LargeScalars(DensePolynomial::new(smul_out));
        let add_in_p_batched_c = MultilinearPolynomial::LargeScalars(DensePolynomial::new(add_p));
        let add_in_q_batched_c = MultilinearPolynomial::LargeScalars(DensePolynomial::new(add_q));
        let add_out_batched_c = MultilinearPolynomial::LargeScalars(DensePolynomial::new(add_r));

        // Build selectors lazily for all referenced indices.
        let mut selectors = std::collections::HashMap::<u64, MultilinearPolynomial<Fq>>::new();
        for e in &self.edges {
            for v in [e.src, e.dst] {
                match v {
                    G2ValueRef::G2ScalarMulOut { instance }
                    | G2ValueRef::G2ScalarMulBase { instance } => {
                        let key = Self::selector_key(dummy_smul, self.k_smul, instance);
                        selectors.entry(key).or_insert_with(|| {
                            let mut evals = vec![Fq::zero(); blocks];
                            for c_common in 0..blocks {
                                if (c_common >> dummy_smul) == instance {
                                    evals[c_common] = Fq::one();
                                }
                            }
                            MultilinearPolynomial::LargeScalars(DensePolynomial::new(evals))
                        });
                    }
                    G2ValueRef::G2AddOut { instance }
                    | G2ValueRef::G2AddInP { instance }
                    | G2ValueRef::G2AddInQ { instance } => {
                        let key = Self::selector_key(dummy_add, self.k_add, instance);
                        selectors.entry(key).or_insert_with(|| {
                            let mut evals = vec![Fq::zero(); blocks];
                            for c_common in 0..blocks {
                                if (c_common >> dummy_add) == instance {
                                    evals[c_common] = Fq::one();
                                }
                            }
                            MultilinearPolynomial::LargeScalars(DensePolynomial::new(evals))
                        });
                    }
                    G2ValueRef::PairingBoundaryP1
                    | G2ValueRef::PairingBoundaryP2
                    | G2ValueRef::PairingBoundaryP3 => {}
                }
            }
        }

        self.c_state = Some(CPhaseState {
            eq_step_at_r,
            smul_out_batched_c,
            add_in_p_batched_c,
            add_in_q_batched_c,
            add_out_batched_c,
            selectors,
        });
    }
}

impl<T: Transcript> SumcheckInstanceProver<Fq, T> for FusedWiringG2Prover<T> {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        STEP_VARS + self.num_c_vars
    }

    fn input_claim(&self, _acc: &ProverOpeningAccumulator<Fq>) -> Fq {
        Fq::zero()
    }

    fn compute_message(&mut self, round: usize, previous_claim: Fq) -> UniPoly<Fq> {
        if round < STEP_VARS {
            // Phase 1: bind step vars. c vars are summed out via the selector/beta convention.
            let half = self.eq_step_poly.len() / 2;
            if half == 0 {
                return UniPoly::from_evals_and_hint(previous_claim, &[Fq::zero(); DEGREE]);
            }

            let lambdas = self.lambdas.clone();
            let edges = self.edges.clone();

            let evals = (0..half)
                .into_par_iter()
                .map(|i| {
                    let eq_evals = self
                        .eq_step_poly
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                    let mut delta_evals = [Fq::zero(); DEGREE];
                    for (lambda, edge) in lambdas.iter().zip(edges.iter()) {
                        let src = self.eval_value_evals_batched(edge.src, i);
                        let dst = self.eval_value_evals_batched(edge.dst, i);
                        for t in 0..DEGREE {
                            delta_evals[t] += *lambda * (src[t] - dst[t]);
                        }
                    }
                    let mut out = [Fq::zero(); DEGREE];
                    for t in 0..DEGREE {
                        out[t] = eq_evals[t] * delta_evals[t];
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
        } else {
            // Phase 2: bind c variables, after step has been fully bound.
            self.ensure_c_state();
            let state = self.c_state.as_ref().expect("c_state must be initialized");
            let num_remaining = state.smul_out_batched_c.get_num_vars();
            let half = 1usize << (num_remaining - 1);

            let dummy_smul = self.num_c_vars.saturating_sub(self.k_smul);
            let dummy_add = self.num_c_vars.saturating_sub(self.k_add);
            let beta_smul = beta(dummy_smul);
            let beta_add = beta(dummy_add);

            let evals = (0..half)
                .into_par_iter()
                .map(|i| {
                    let smul_out_e = state
                        .smul_out_batched_c
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                    let add_p_e = state
                        .add_in_p_batched_c
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                    let add_q_e = state
                        .add_in_q_batched_c
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                    let add_r_e = state
                        .add_out_batched_c
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                    let mut out = [Fq::zero(); DEGREE];

                    for (edge_idx, edge) in self.edges.iter().enumerate() {
                        let lambda = self.lambdas[edge_idx];
                        let coeff = state.eq_step_at_r * lambda;

                        let (beta_src, sel_src_e, src_e) = match edge.src {
                            G2ValueRef::G2ScalarMulOut { instance } => {
                                let key = Self::selector_key(dummy_smul, self.k_smul, instance);
                                let sel = state.selectors.get(&key).expect("missing smul selector");
                                (
                                    beta_smul,
                                    sel.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh),
                                    smul_out_e,
                                )
                            }
                            G2ValueRef::G2ScalarMulBase { instance } => {
                                let key = Self::selector_key(dummy_smul, self.k_smul, instance);
                                let sel = state.selectors.get(&key).expect("missing smul selector");
                                let (x0, x1, y0, y1, ind) = self.smul_base[instance].unwrap();
                                let v = x0
                                    + self.mu * x1
                                    + self.mu2 * y0
                                    + self.mu3 * y1
                                    + self.mu4 * ind;
                                (
                                    beta_smul,
                                    sel.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh),
                                    [v; DEGREE],
                                )
                            }
                            G2ValueRef::G2AddOut { instance } => {
                                let key = Self::selector_key(dummy_add, self.k_add, instance);
                                let sel = state.selectors.get(&key).expect("missing add selector");
                                (
                                    beta_add,
                                    sel.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh),
                                    add_r_e,
                                )
                            }
                            G2ValueRef::G2AddInP { instance } => {
                                let key = Self::selector_key(dummy_add, self.k_add, instance);
                                let sel = state.selectors.get(&key).expect("missing add selector");
                                (
                                    beta_add,
                                    sel.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh),
                                    add_p_e,
                                )
                            }
                            G2ValueRef::G2AddInQ { instance } => {
                                let key = Self::selector_key(dummy_add, self.k_add, instance);
                                let sel = state.selectors.get(&key).expect("missing add selector");
                                (
                                    beta_add,
                                    sel.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh),
                                    add_q_e,
                                )
                            }
                            // Boundary constants are anchored to the *dst* selector (GT-style).
                            G2ValueRef::PairingBoundaryP1
                            | G2ValueRef::PairingBoundaryP2
                            | G2ValueRef::PairingBoundaryP3 => {
                                (Fq::zero(), [Fq::zero(); DEGREE], [Fq::zero(); DEGREE])
                            }
                        };

                        let (beta_dst, sel_dst_e, dst_e) = match edge.dst {
                            G2ValueRef::G2ScalarMulOut { instance } => {
                                let key = Self::selector_key(dummy_smul, self.k_smul, instance);
                                let sel = state.selectors.get(&key).expect("missing smul selector");
                                (
                                    beta_smul,
                                    sel.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh),
                                    smul_out_e,
                                )
                            }
                            G2ValueRef::G2ScalarMulBase { instance } => {
                                let key = Self::selector_key(dummy_smul, self.k_smul, instance);
                                let sel = state.selectors.get(&key).expect("missing smul selector");
                                let (x0, x1, y0, y1, ind) = self.smul_base[instance].unwrap();
                                let v = x0
                                    + self.mu * x1
                                    + self.mu2 * y0
                                    + self.mu3 * y1
                                    + self.mu4 * ind;
                                (
                                    beta_smul,
                                    sel.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh),
                                    [v; DEGREE],
                                )
                            }
                            G2ValueRef::G2AddOut { instance } => {
                                let key = Self::selector_key(dummy_add, self.k_add, instance);
                                let sel = state.selectors.get(&key).expect("missing add selector");
                                (
                                    beta_add,
                                    sel.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh),
                                    add_r_e,
                                )
                            }
                            G2ValueRef::G2AddInP { instance } => {
                                let key = Self::selector_key(dummy_add, self.k_add, instance);
                                let sel = state.selectors.get(&key).expect("missing add selector");
                                (
                                    beta_add,
                                    sel.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh),
                                    add_p_e,
                                )
                            }
                            G2ValueRef::G2AddInQ { instance } => {
                                let key = Self::selector_key(dummy_add, self.k_add, instance);
                                let sel = state.selectors.get(&key).expect("missing add selector");
                                (
                                    beta_add,
                                    sel.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh),
                                    add_q_e,
                                )
                            }
                            G2ValueRef::PairingBoundaryP1
                            | G2ValueRef::PairingBoundaryP2
                            | G2ValueRef::PairingBoundaryP3 => {
                                (Fq::zero(), [Fq::zero(); DEGREE], [Fq::zero(); DEGREE])
                            }
                        };

                        // Handle boundary constants by anchoring to the opposite endpoint selector.
                        let (beta_src, sel_src_e, src_e) = match edge.src {
                            G2ValueRef::PairingBoundaryP1 => {
                                let (x0, x1, y0, y1, ind) = self.pairing_p1;
                                let v = x0
                                    + self.mu * x1
                                    + self.mu2 * y0
                                    + self.mu3 * y1
                                    + self.mu4 * ind;
                                (beta_dst, sel_dst_e, [v; DEGREE])
                            }
                            G2ValueRef::PairingBoundaryP2 => {
                                let (x0, x1, y0, y1, ind) = self.pairing_p2;
                                let v = x0
                                    + self.mu * x1
                                    + self.mu2 * y0
                                    + self.mu3 * y1
                                    + self.mu4 * ind;
                                (beta_dst, sel_dst_e, [v; DEGREE])
                            }
                            G2ValueRef::PairingBoundaryP3 => {
                                let (x0, x1, y0, y1, ind) = self.pairing_p3;
                                let v = x0
                                    + self.mu * x1
                                    + self.mu2 * y0
                                    + self.mu3 * y1
                                    + self.mu4 * ind;
                                (beta_dst, sel_dst_e, [v; DEGREE])
                            }
                            _ => (beta_src, sel_src_e, src_e),
                        };
                        let (beta_dst, sel_dst_e, dst_e) = match edge.dst {
                            G2ValueRef::PairingBoundaryP1 => {
                                let (x0, x1, y0, y1, ind) = self.pairing_p1;
                                let v = x0
                                    + self.mu * x1
                                    + self.mu2 * y0
                                    + self.mu3 * y1
                                    + self.mu4 * ind;
                                (beta_src, sel_src_e, [v; DEGREE])
                            }
                            G2ValueRef::PairingBoundaryP2 => {
                                let (x0, x1, y0, y1, ind) = self.pairing_p2;
                                let v = x0
                                    + self.mu * x1
                                    + self.mu2 * y0
                                    + self.mu3 * y1
                                    + self.mu4 * ind;
                                (beta_src, sel_src_e, [v; DEGREE])
                            }
                            G2ValueRef::PairingBoundaryP3 => {
                                let (x0, x1, y0, y1, ind) = self.pairing_p3;
                                let v = x0
                                    + self.mu * x1
                                    + self.mu2 * y0
                                    + self.mu3 * y1
                                    + self.mu4 * ind;
                                (beta_src, sel_src_e, [v; DEGREE])
                            }
                            _ => (beta_dst, sel_dst_e, dst_e),
                        };

                        for t in 0..DEGREE {
                            out[t] += coeff
                                * (beta_src * sel_src_e[t] * src_e[t]
                                    - beta_dst * sel_dst_e[t] * dst_e[t]);
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
        if round < STEP_VARS {
            for poly in self.xa_next_c0.iter_mut().flatten() {
                poly.bind_parallel(r_j, BindingOrder::LowToHigh);
            }
            for poly in self.xa_next_c1.iter_mut().flatten() {
                poly.bind_parallel(r_j, BindingOrder::LowToHigh);
            }
            for poly in self.ya_next_c0.iter_mut().flatten() {
                poly.bind_parallel(r_j, BindingOrder::LowToHigh);
            }
            for poly in self.ya_next_c1.iter_mut().flatten() {
                poly.bind_parallel(r_j, BindingOrder::LowToHigh);
            }
            for poly in self.a_ind.iter_mut().flatten() {
                poly.bind_parallel(r_j, BindingOrder::LowToHigh);
            }
            self.eq_step_poly
                .bind_parallel(r_j, BindingOrder::LowToHigh);
        } else {
            self.ensure_c_state();
            self.c_state
                .as_mut()
                .expect("c_state must be initialized")
                .bind(r_j);
        }
    }

    fn cache_openings(
        &self,
        _acc: &mut ProverOpeningAccumulator<Fq>,
        _transcript: &mut T,
        _sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        // No-op: this instance only reads cached openings (from fused G2 gadgets).
    }
}

#[derive(Clone, Debug)]
pub struct FusedWiringG2Verifier {
    edges: Vec<G2WiringEdge>,
    lambdas: Vec<Fq>,
    mu: Fq,
    mu2: Fq,
    mu3: Fq,
    mu4: Fq,
    k_common: usize,
    k_smul: usize,
    k_add: usize,
    pairing_boundary: PairingBoundary,
    smul_bases: Vec<(Fq, Fq, Fq, Fq, Fq)>,
}

impl FusedWiringG2Verifier {
    pub fn new<T: Transcript>(input: &RecursionVerifierInput, transcript: &mut T) -> Self {
        let mu: Fq = transcript.challenge_scalar();
        let mu2 = mu * mu;
        let mu3 = mu2 * mu;
        let mu4 = mu2 * mu2;
        let edges = input.wiring.g2.clone();
        let lambdas: Vec<Fq> = transcript.challenge_vector(edges.len());

        // Base points, in local scalar-mul instance order.
        let smul_bases = input
            .constraint_types
            .iter()
            .filter_map(|ct| match ct {
                ConstraintType::G2ScalarMul { base_point } => Some((
                    base_point.0.c0,
                    base_point.0.c1,
                    base_point.1.c0,
                    base_point.1.c1,
                    Fq::zero(),
                )),
                _ => None,
            })
            .collect();

        let k_common = k_g2(&input.constraint_types);
        let k_smul = k_smul(&input.constraint_types);
        let k_add = k_add(&input.constraint_types);

        Self {
            edges,
            lambdas,
            mu,
            mu2,
            mu3,
            mu4,
            k_common,
            k_smul,
            k_add,
            pairing_boundary: input.pairing_boundary.clone(),
            smul_bases,
        }
    }

    #[inline]
    fn eq_c_tail(&self, r_c_tail: &[Fq], idx: usize, k_family: usize) -> Fq {
        let bits = index_to_binary::<Fq>(idx, k_family);
        EqPolynomial::mle(r_c_tail, &bits)
    }
}

impl<T: Transcript> SumcheckInstanceVerifier<Fq, T> for FusedWiringG2Verifier {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        STEP_VARS + self.k_common
    }

    fn input_claim(&self, _acc: &VerifierOpeningAccumulator<Fq>) -> Fq {
        Fq::zero()
    }

    fn expected_output_claim(
        &self,
        acc: &VerifierOpeningAccumulator<Fq>,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) -> Fq {
        debug_assert_eq!(sumcheck_challenges.len(), STEP_VARS + self.k_common);
        let r_step = &sumcheck_challenges[..STEP_VARS];
        let r_c = &sumcheck_challenges[STEP_VARS..];

        // Eq(s,255) at r_step (255 is all-ones).
        let mut eq_step = Fq::one();
        for &c in r_step {
            let r_b: Fq = c.into();
            eq_step *= r_b;
        }

        let dummy_smul = self.k_common.saturating_sub(self.k_smul);
        let dummy_add = self.k_common.saturating_sub(self.k_add);
        let beta_smul = beta(dummy_smul);
        let beta_add = beta(dummy_add);

        let r_c_fq: Vec<Fq> = r_c.iter().map(|c| (*c).into()).collect();
        let r_c_smul_tail = &r_c_fq[dummy_smul..];
        let r_c_add_tail = &r_c_fq[dummy_add..];

        // Fetch fused scalar-mul port openings.
        let get_smul = |vp: VirtualPolynomial| -> Fq {
            acc.get_virtual_polynomial_opening(vp, SumcheckId::G2ScalarMul)
                .1
        };
        let smul_out_fused = get_smul(VirtualPolynomial::g2_scalar_mul_xa_next_c0_fused())
            + self.mu * get_smul(VirtualPolynomial::g2_scalar_mul_xa_next_c1_fused())
            + self.mu2 * get_smul(VirtualPolynomial::g2_scalar_mul_ya_next_c0_fused())
            + self.mu3 * get_smul(VirtualPolynomial::g2_scalar_mul_ya_next_c1_fused())
            + self.mu4 * get_smul(VirtualPolynomial::g2_scalar_mul_a_indicator_fused());

        // Fetch fused add port openings.
        let get_add = |vp: VirtualPolynomial| -> Fq {
            acc.get_virtual_polynomial_opening(vp, SumcheckId::G2Add).1
        };
        let add_p_fused = get_add(VirtualPolynomial::g2_add_xp_c0_fused())
            + self.mu * get_add(VirtualPolynomial::g2_add_xp_c1_fused())
            + self.mu2 * get_add(VirtualPolynomial::g2_add_yp_c0_fused())
            + self.mu3 * get_add(VirtualPolynomial::g2_add_yp_c1_fused())
            + self.mu4 * get_add(VirtualPolynomial::g2_add_p_indicator_fused());
        let add_q_fused = get_add(VirtualPolynomial::g2_add_xq_c0_fused())
            + self.mu * get_add(VirtualPolynomial::g2_add_xq_c1_fused())
            + self.mu2 * get_add(VirtualPolynomial::g2_add_yq_c0_fused())
            + self.mu3 * get_add(VirtualPolynomial::g2_add_yq_c1_fused())
            + self.mu4 * get_add(VirtualPolynomial::g2_add_q_indicator_fused());
        let add_r_fused = get_add(VirtualPolynomial::g2_add_xr_c0_fused())
            + self.mu * get_add(VirtualPolynomial::g2_add_xr_c1_fused())
            + self.mu2 * get_add(VirtualPolynomial::g2_add_yr_c0_fused())
            + self.mu3 * get_add(VirtualPolynomial::g2_add_yr_c1_fused())
            + self.mu4 * get_add(VirtualPolynomial::g2_add_r_indicator_fused());

        let pb1 = g2_const_from_affine(&self.pairing_boundary.p1_g2);
        let pb2 = g2_const_from_affine(&self.pairing_boundary.p2_g2);
        let pb3 = g2_const_from_affine(&self.pairing_boundary.p3_g2);
        let pb_batched = |p: (Fq, Fq, Fq, Fq, Fq)| -> Fq {
            p.0 + self.mu * p.1 + self.mu2 * p.2 + self.mu3 * p.3 + self.mu4 * p.4
        };

        let mut sum = Fq::zero();
        for (lambda, edge) in self.lambdas.iter().zip(self.edges.iter()) {
            let coeff = eq_step * *lambda;

            let endpoint = |v: G2ValueRef| -> (Fq, Fq, Fq) {
                match v {
                    G2ValueRef::G2ScalarMulOut { instance } => (
                        beta_smul,
                        self.eq_c_tail(r_c_smul_tail, instance, self.k_smul),
                        smul_out_fused,
                    ),
                    G2ValueRef::G2ScalarMulBase { instance } => {
                        let (x0, x1, y0, y1, ind) = self.smul_bases[instance];
                        let v = x0 + self.mu * x1 + self.mu2 * y0 + self.mu3 * y1 + self.mu4 * ind;
                        (
                            beta_smul,
                            self.eq_c_tail(r_c_smul_tail, instance, self.k_smul),
                            v,
                        )
                    }
                    G2ValueRef::G2AddOut { instance } => (
                        beta_add,
                        self.eq_c_tail(r_c_add_tail, instance, self.k_add),
                        add_r_fused,
                    ),
                    G2ValueRef::G2AddInP { instance } => (
                        beta_add,
                        self.eq_c_tail(r_c_add_tail, instance, self.k_add),
                        add_p_fused,
                    ),
                    G2ValueRef::G2AddInQ { instance } => (
                        beta_add,
                        self.eq_c_tail(r_c_add_tail, instance, self.k_add),
                        add_q_fused,
                    ),
                    // Boundary constants are anchored to the other endpoint's selector in the caller.
                    G2ValueRef::PairingBoundaryP1 => (Fq::zero(), Fq::zero(), pb_batched(pb1)),
                    G2ValueRef::PairingBoundaryP2 => (Fq::zero(), Fq::zero(), pb_batched(pb2)),
                    G2ValueRef::PairingBoundaryP3 => (Fq::zero(), Fq::zero(), pb_batched(pb3)),
                }
            };

            let (b_src, eqc_src, v_src) = endpoint(edge.src);
            let (b_dst, eqc_dst, v_dst) = endpoint(edge.dst);

            // Anchor boundary constants to the opposite selector/beta (GT-style).
            let (b_src, eqc_src, v_src) = match edge.src {
                G2ValueRef::PairingBoundaryP1
                | G2ValueRef::PairingBoundaryP2
                | G2ValueRef::PairingBoundaryP3 => (b_dst, eqc_dst, v_src),
                _ => (b_src, eqc_src, v_src),
            };
            let (b_dst, eqc_dst, v_dst) = match edge.dst {
                G2ValueRef::PairingBoundaryP1
                | G2ValueRef::PairingBoundaryP2
                | G2ValueRef::PairingBoundaryP3 => (b_src, eqc_src, v_dst),
                _ => (b_dst, eqc_dst, v_dst),
            };

            sum += coeff * (b_src * eqc_src * v_src - b_dst * eqc_dst * v_dst);
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

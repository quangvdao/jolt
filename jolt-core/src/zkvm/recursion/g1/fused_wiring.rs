//! Fused G1 wiring (copy/boundary) sumcheck (GT-style split-k aware).
//!
//! This is the G1 analogue of `gt/fused_wiring.rs`.
//!
//! ## Goal
//! Replace legacy per-instance wiring openings (see `g1/wiring.rs`) with a fully fused wiring
//! backend that consumes:
//! - fused G1ScalarMul port openings (under `SumcheckId::G1ScalarMul`), and
//! - fused G1Add port openings (under `SumcheckId::G1Add`),
//!   while staying compatible with Stage-2 suffix-aligned batching.
//!
//! ## Variable order (Stage 2)
//! - Phase 1 (x): bind `s` (8 step vars).
//! - Phase 2 (c): bind `c_common` (k vars), where `k = k_g1 = max(k_smul, k_add)`.
//!
//! ## Split-k convention (dummy bits)
//! Dummy bits are the *low* bits of `c_common`. Family-local bits are the suffix.
//! For a family with `k_family` bits:
//! - `dummy = k_common - k_family`
//! - selectors use `beta(dummy) * Eq(c_tail, idx)` where `c_tail = c_common[dummy..]`
//!   (matches the GT fused wiring convention).

use ark_bn254::{Fq, G1Affine};
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
            constraints::system::{index_to_binary, ConstraintSystem, G1AddNative},
            g1::indexing::{k_add, k_g1, k_smul},
            gt::types::eq_lsb_evals,
            verifier::RecursionVerifierInput,
            wiring_plan::{G1ValueRef, G1WiringEdge},
            ConstraintType,
        },
        witness::VirtualPolynomial,
    },
};

pub(crate) const STEP_VARS: usize = 8;
const DEGREE: usize = 2;

#[inline]
fn g1_const_from_affine(p: &G1Affine) -> (Fq, Fq, Fq) {
    if p.is_zero() {
        (Fq::zero(), Fq::zero(), Fq::one())
    } else {
        (p.x, p.y, Fq::zero())
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
pub struct FusedWiringG1Prover<T: Transcript> {
    edges: Vec<G1WiringEdge>,
    lambdas: Vec<Fq>,
    mu: Fq,
    mu2: Fq,
    // common index sizes
    num_c_vars: usize, // k_g1
    k_smul: usize,
    k_add: usize,

    // --- x-phase (step) polynomials ---
    eq_step_poly: MultilinearPolynomial<Fq>,

    // Scalar mul output polys (8-var), indexed by scalar-mul instance.
    xa_next: Vec<Option<MultilinearPolynomial<Fq>>>,
    ya_next: Vec<Option<MultilinearPolynomial<Fq>>>,
    a_ind: Vec<Option<MultilinearPolynomial<Fq>>>,

    // Add rows (0-var scalars).
    add_rows: Vec<Option<G1AddNative>>,

    // Scalar-mul base constants (x,y,ind=0), indexed by scalar-mul instance.
    smul_base: Vec<Option<(Fq, Fq, Fq)>>,

    // Pairing boundary constants (x,y,ind).
    pairing_p1: (Fq, Fq, Fq),
    pairing_p2: (Fq, Fq, Fq),
    pairing_p3: (Fq, Fq, Fq),

    c_state: Option<CPhaseState>,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Transcript> FusedWiringG1Prover<T> {
    pub fn new(
        cs: &ConstraintSystem,
        edges: Vec<G1WiringEdge>,
        pairing_boundary: &PairingBoundary,
        transcript: &mut T,
    ) -> Self {
        // Selector point a_G1 = 255 = [1,1,...,1] (LSB-first order).
        let a_g1: Vec<<Fq as JoltField>::Challenge> =
            (0..STEP_VARS).map(|_| Fq::one().into()).collect();
        let eq_step_poly = MultilinearPolynomial::from(eq_lsb_evals::<Fq>(&a_g1));

        let mu: Fq = transcript.challenge_scalar();
        let mu2 = mu * mu;
        let lambdas: Vec<Fq> = transcript.challenge_vector(edges.len());

        let num_smul = cs.g1_scalar_mul_rows.len();
        let num_add = cs.g1_add_rows.len();

        let mut need_smul_out = vec![false; num_smul];
        let mut need_smul_base = vec![false; num_smul];
        let mut need_add = vec![false; num_add];

        for e in &edges {
            for v in [e.src, e.dst] {
                match v {
                    G1ValueRef::G1ScalarMulOut { instance } => {
                        if instance < num_smul {
                            need_smul_out[instance] = true;
                        }
                    }
                    G1ValueRef::G1ScalarMulBase { instance } => {
                        if instance < num_smul {
                            need_smul_base[instance] = true;
                        }
                    }
                    G1ValueRef::G1AddOut { instance }
                    | G1ValueRef::G1AddInP { instance }
                    | G1ValueRef::G1AddInQ { instance } => {
                        if instance < num_add {
                            need_add[instance] = true;
                        }
                    }
                    G1ValueRef::PairingBoundaryP1
                    | G1ValueRef::PairingBoundaryP2
                    | G1ValueRef::PairingBoundaryP3 => {}
                }
            }
        }

        let xa_next = need_smul_out
            .iter()
            .enumerate()
            .map(|(i, &need)| {
                need.then(|| MultilinearPolynomial::from(cs.g1_scalar_mul_rows[i].x_a_next.clone()))
            })
            .collect();
        let ya_next = need_smul_out
            .iter()
            .enumerate()
            .map(|(i, &need)| {
                need.then(|| MultilinearPolynomial::from(cs.g1_scalar_mul_rows[i].y_a_next.clone()))
            })
            .collect();
        let a_ind = need_smul_out
            .into_iter()
            .enumerate()
            .map(|(i, need)| {
                need.then(|| {
                    MultilinearPolynomial::from(cs.g1_scalar_mul_rows[i].a_indicator.clone())
                })
            })
            .collect();

        let smul_base = need_smul_base
            .into_iter()
            .enumerate()
            .map(|(i, need)| {
                need.then(|| {
                    (
                        cs.g1_scalar_mul_rows[i].base_point.0,
                        cs.g1_scalar_mul_rows[i].base_point.1,
                        Fq::zero(),
                    )
                })
            })
            .collect();

        let add_rows = need_add
            .into_iter()
            .enumerate()
            .map(|(i, need)| need.then(|| cs.g1_add_rows[i]))
            .collect();

        let pairing_p1 = g1_const_from_affine(&pairing_boundary.p1_g1);
        let pairing_p2 = g1_const_from_affine(&pairing_boundary.p2_g1);
        let pairing_p3 = g1_const_from_affine(&pairing_boundary.p3_g1);

        let num_c_vars = k_g1(&cs.constraint_types);
        let k_smul = k_smul(&cs.constraint_types);
        let k_add = k_add(&cs.constraint_types);

        Self {
            edges,
            lambdas,
            mu,
            mu2,
            num_c_vars,
            k_smul,
            k_add,
            eq_step_poly,
            xa_next,
            ya_next,
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
    fn eval_value_evals_batched(&self, v: G1ValueRef, i: usize) -> [Fq; DEGREE] {
        match v {
            G1ValueRef::G1ScalarMulOut { instance } => {
                let x = self.xa_next[instance]
                    .as_ref()
                    .unwrap()
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let y = self.ya_next[instance]
                    .as_ref()
                    .unwrap()
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let ind = self.a_ind[instance]
                    .as_ref()
                    .unwrap()
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let mut out = [Fq::zero(); DEGREE];
                for t in 0..DEGREE {
                    out[t] = x[t] + self.mu * y[t] + self.mu2 * ind[t];
                }
                out
            }
            G1ValueRef::G1ScalarMulBase { instance } => {
                let (x, y, ind) = self.smul_base[instance].unwrap();
                let v = x + self.mu * y + self.mu2 * ind;
                [v; DEGREE]
            }
            G1ValueRef::G1AddOut { instance } => {
                let row = self.add_rows[instance].unwrap();
                let v = row.x_r + self.mu * row.y_r + self.mu2 * row.ind_r;
                [v; DEGREE]
            }
            G1ValueRef::G1AddInP { instance } => {
                let row = self.add_rows[instance].unwrap();
                let v = row.x_p + self.mu * row.y_p + self.mu2 * row.ind_p;
                [v; DEGREE]
            }
            G1ValueRef::G1AddInQ { instance } => {
                let row = self.add_rows[instance].unwrap();
                let v = row.x_q + self.mu * row.y_q + self.mu2 * row.ind_q;
                [v; DEGREE]
            }
            G1ValueRef::PairingBoundaryP1 => {
                let (x, y, ind) = self.pairing_p1;
                let v = x + self.mu * y + self.mu2 * ind;
                [v; DEGREE]
            }
            G1ValueRef::PairingBoundaryP2 => {
                let (x, y, ind) = self.pairing_p2;
                let v = x + self.mu * y + self.mu2 * ind;
                [v; DEGREE]
            }
            G1ValueRef::PairingBoundaryP3 => {
                let (x, y, ind) = self.pairing_p3;
                let v = x + self.mu * y + self.mu2 * ind;
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
            if c_smul < self.xa_next.len() {
                if let (Some(xa), Some(ya), Some(ind)) = (
                    &self.xa_next[c_smul],
                    &self.ya_next[c_smul],
                    &self.a_ind[c_smul],
                ) {
                    let x = xa.get_bound_coeff(0);
                    let y = ya.get_bound_coeff(0);
                    let i = ind.get_bound_coeff(0);
                    smul_out[c_common] = x + self.mu * y + self.mu2 * i;
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
                    add_p[c_common] = row.x_p + self.mu * row.y_p + self.mu2 * row.ind_p;
                    add_q[c_common] = row.x_q + self.mu * row.y_q + self.mu2 * row.ind_q;
                    add_r[c_common] = row.x_r + self.mu * row.y_r + self.mu2 * row.ind_r;
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
                    G1ValueRef::G1ScalarMulOut { instance }
                    | G1ValueRef::G1ScalarMulBase { instance } => {
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
                    G1ValueRef::G1AddOut { instance }
                    | G1ValueRef::G1AddInP { instance }
                    | G1ValueRef::G1AddInQ { instance } => {
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
                    G1ValueRef::PairingBoundaryP1
                    | G1ValueRef::PairingBoundaryP2
                    | G1ValueRef::PairingBoundaryP3 => {}
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

impl<T: Transcript> SumcheckInstanceProver<Fq, T> for FusedWiringG1Prover<T> {
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

                        // Pick port eval array + selector polynomial for src.
                        let (beta_src, sel_src_e, src_e) = match edge.src {
                            G1ValueRef::G1ScalarMulOut { instance } => {
                                let key = Self::selector_key(dummy_smul, self.k_smul, instance);
                                let sel = state.selectors.get(&key).expect("missing smul selector");
                                (
                                    beta_smul,
                                    sel.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh),
                                    smul_out_e,
                                )
                            }
                            G1ValueRef::G1ScalarMulBase { instance } => {
                                let key = Self::selector_key(dummy_smul, self.k_smul, instance);
                                let sel = state.selectors.get(&key).expect("missing smul selector");
                                let (x, y, ind) = self.smul_base[instance].unwrap();
                                let v = x + self.mu * y + self.mu2 * ind;
                                (
                                    beta_smul,
                                    sel.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh),
                                    [v; DEGREE],
                                )
                            }
                            G1ValueRef::G1AddOut { instance } => {
                                let key = Self::selector_key(dummy_add, self.k_add, instance);
                                let sel = state.selectors.get(&key).expect("missing add selector");
                                (
                                    beta_add,
                                    sel.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh),
                                    add_r_e,
                                )
                            }
                            G1ValueRef::G1AddInP { instance } => {
                                let key = Self::selector_key(dummy_add, self.k_add, instance);
                                let sel = state.selectors.get(&key).expect("missing add selector");
                                (
                                    beta_add,
                                    sel.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh),
                                    add_p_e,
                                )
                            }
                            G1ValueRef::G1AddInQ { instance } => {
                                let key = Self::selector_key(dummy_add, self.k_add, instance);
                                let sel = state.selectors.get(&key).expect("missing add selector");
                                (
                                    beta_add,
                                    sel.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh),
                                    add_q_e,
                                )
                            }
                            // Boundary constants are anchored to the *dst* selector (GT-style).
                            G1ValueRef::PairingBoundaryP1
                            | G1ValueRef::PairingBoundaryP2
                            | G1ValueRef::PairingBoundaryP3 => {
                                // Defer: handled below after dst selection.
                                (Fq::zero(), [Fq::zero(); DEGREE], [Fq::zero(); DEGREE])
                            }
                        };

                        let (beta_dst, sel_dst_e, dst_e) = match edge.dst {
                            G1ValueRef::G1ScalarMulOut { instance } => {
                                let key = Self::selector_key(dummy_smul, self.k_smul, instance);
                                let sel = state.selectors.get(&key).expect("missing smul selector");
                                (
                                    beta_smul,
                                    sel.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh),
                                    smul_out_e,
                                )
                            }
                            G1ValueRef::G1ScalarMulBase { instance } => {
                                let key = Self::selector_key(dummy_smul, self.k_smul, instance);
                                let sel = state.selectors.get(&key).expect("missing smul selector");
                                let (x, y, ind) = self.smul_base[instance].unwrap();
                                let v = x + self.mu * y + self.mu2 * ind;
                                (
                                    beta_smul,
                                    sel.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh),
                                    [v; DEGREE],
                                )
                            }
                            G1ValueRef::G1AddOut { instance } => {
                                let key = Self::selector_key(dummy_add, self.k_add, instance);
                                let sel = state.selectors.get(&key).expect("missing add selector");
                                (
                                    beta_add,
                                    sel.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh),
                                    add_r_e,
                                )
                            }
                            G1ValueRef::G1AddInP { instance } => {
                                let key = Self::selector_key(dummy_add, self.k_add, instance);
                                let sel = state.selectors.get(&key).expect("missing add selector");
                                (
                                    beta_add,
                                    sel.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh),
                                    add_p_e,
                                )
                            }
                            G1ValueRef::G1AddInQ { instance } => {
                                let key = Self::selector_key(dummy_add, self.k_add, instance);
                                let sel = state.selectors.get(&key).expect("missing add selector");
                                (
                                    beta_add,
                                    sel.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh),
                                    add_q_e,
                                )
                            }
                            G1ValueRef::PairingBoundaryP1
                            | G1ValueRef::PairingBoundaryP2
                            | G1ValueRef::PairingBoundaryP3 => {
                                (Fq::zero(), [Fq::zero(); DEGREE], [Fq::zero(); DEGREE])
                            }
                        };

                        // Handle boundary constants by anchoring to the opposite endpoint selector.
                        let (beta_src, sel_src_e, src_e) = match edge.src {
                            G1ValueRef::PairingBoundaryP1 => {
                                let (x, y, ind) = self.pairing_p1;
                                let v = x + self.mu * y + self.mu2 * ind;
                                (beta_dst, sel_dst_e, [v; DEGREE])
                            }
                            G1ValueRef::PairingBoundaryP2 => {
                                let (x, y, ind) = self.pairing_p2;
                                let v = x + self.mu * y + self.mu2 * ind;
                                (beta_dst, sel_dst_e, [v; DEGREE])
                            }
                            G1ValueRef::PairingBoundaryP3 => {
                                let (x, y, ind) = self.pairing_p3;
                                let v = x + self.mu * y + self.mu2 * ind;
                                (beta_dst, sel_dst_e, [v; DEGREE])
                            }
                            _ => (beta_src, sel_src_e, src_e),
                        };
                        let (beta_dst, sel_dst_e, dst_e) = match edge.dst {
                            G1ValueRef::PairingBoundaryP1 => {
                                let (x, y, ind) = self.pairing_p1;
                                let v = x + self.mu * y + self.mu2 * ind;
                                (beta_src, sel_src_e, [v; DEGREE])
                            }
                            G1ValueRef::PairingBoundaryP2 => {
                                let (x, y, ind) = self.pairing_p2;
                                let v = x + self.mu * y + self.mu2 * ind;
                                (beta_src, sel_src_e, [v; DEGREE])
                            }
                            G1ValueRef::PairingBoundaryP3 => {
                                let (x, y, ind) = self.pairing_p3;
                                let v = x + self.mu * y + self.mu2 * ind;
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
            for poly in self.xa_next.iter_mut().flatten() {
                poly.bind_parallel(r_j, BindingOrder::LowToHigh);
            }
            for poly in self.ya_next.iter_mut().flatten() {
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
        // No-op: this instance only reads cached openings (from fused G1 gadgets).
    }
}

#[derive(Clone, Debug)]
pub struct FusedWiringG1Verifier {
    edges: Vec<G1WiringEdge>,
    lambdas: Vec<Fq>,
    mu: Fq,
    mu2: Fq,
    k_common: usize,
    k_smul: usize,
    k_add: usize,
    pairing_boundary: PairingBoundary,
    smul_bases: Vec<(Fq, Fq, Fq)>,
}

impl FusedWiringG1Verifier {
    pub fn new<T: Transcript>(input: &RecursionVerifierInput, transcript: &mut T) -> Self {
        let mu: Fq = transcript.challenge_scalar();
        let mu2 = mu * mu;
        let edges = input.wiring.g1.clone();
        let lambdas: Vec<Fq> = transcript.challenge_vector(edges.len());

        // Base points, in local scalar-mul instance order.
        let smul_bases = input
            .constraint_types
            .iter()
            .filter_map(|ct| match ct {
                ConstraintType::G1ScalarMul { base_point } => {
                    Some((base_point.0, base_point.1, Fq::zero()))
                }
                _ => None,
            })
            .collect();

        let k_common = k_g1(&input.constraint_types);
        let k_smul = k_smul(&input.constraint_types);
        let k_add = k_add(&input.constraint_types);

        Self {
            edges,
            lambdas,
            mu,
            mu2,
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

impl<T: Transcript> SumcheckInstanceVerifier<Fq, T> for FusedWiringG1Verifier {
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
        let (_, xa_next) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::g1_scalar_mul_xa_next_fused(),
            SumcheckId::G1ScalarMul,
        );
        let (_, ya_next) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::g1_scalar_mul_ya_next_fused(),
            SumcheckId::G1ScalarMul,
        );
        let (_, a_ind) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::g1_scalar_mul_a_indicator_fused(),
            SumcheckId::G1ScalarMul,
        );
        let smul_out_fused = xa_next + self.mu * ya_next + self.mu2 * a_ind;

        // Fetch fused add port openings.
        let get_add = |vp: VirtualPolynomial| -> Fq {
            acc.get_virtual_polynomial_opening(vp, SumcheckId::G1Add).1
        };
        let add_p_fused = get_add(VirtualPolynomial::g1_add_xp_fused())
            + self.mu * get_add(VirtualPolynomial::g1_add_yp_fused())
            + self.mu2 * get_add(VirtualPolynomial::g1_add_p_indicator_fused());
        let add_q_fused = get_add(VirtualPolynomial::g1_add_xq_fused())
            + self.mu * get_add(VirtualPolynomial::g1_add_yq_fused())
            + self.mu2 * get_add(VirtualPolynomial::g1_add_q_indicator_fused());
        let add_r_fused = get_add(VirtualPolynomial::g1_add_xr_fused())
            + self.mu * get_add(VirtualPolynomial::g1_add_yr_fused())
            + self.mu2 * get_add(VirtualPolynomial::g1_add_r_indicator_fused());

        let pb1 = g1_const_from_affine(&self.pairing_boundary.p1_g1);
        let pb2 = g1_const_from_affine(&self.pairing_boundary.p2_g1);
        let pb3 = g1_const_from_affine(&self.pairing_boundary.p3_g1);
        let pb_batched = |p: (Fq, Fq, Fq)| -> Fq { p.0 + self.mu * p.1 + self.mu2 * p.2 };

        let mut sum = Fq::zero();
        for (lambda, edge) in self.lambdas.iter().zip(self.edges.iter()) {
            let coeff = eq_step * *lambda;

            // Helper to produce (beta, eq_c, value) for an endpoint.
            let endpoint = |v: G1ValueRef| -> (Fq, Fq, Fq) {
                match v {
                    G1ValueRef::G1ScalarMulOut { instance } => (
                        beta_smul,
                        self.eq_c_tail(r_c_smul_tail, instance, self.k_smul),
                        smul_out_fused,
                    ),
                    G1ValueRef::G1ScalarMulBase { instance } => {
                        let (x, y, ind) = self.smul_bases[instance];
                        let v = x + self.mu * y + self.mu2 * ind;
                        (
                            beta_smul,
                            self.eq_c_tail(r_c_smul_tail, instance, self.k_smul),
                            v,
                        )
                    }
                    G1ValueRef::G1AddOut { instance } => (
                        beta_add,
                        self.eq_c_tail(r_c_add_tail, instance, self.k_add),
                        add_r_fused,
                    ),
                    G1ValueRef::G1AddInP { instance } => (
                        beta_add,
                        self.eq_c_tail(r_c_add_tail, instance, self.k_add),
                        add_p_fused,
                    ),
                    G1ValueRef::G1AddInQ { instance } => (
                        beta_add,
                        self.eq_c_tail(r_c_add_tail, instance, self.k_add),
                        add_q_fused,
                    ),
                    // Boundary constants are anchored to the other endpoint's selector in the caller.
                    G1ValueRef::PairingBoundaryP1 => (Fq::zero(), Fq::zero(), pb_batched(pb1)),
                    G1ValueRef::PairingBoundaryP2 => (Fq::zero(), Fq::zero(), pb_batched(pb2)),
                    G1ValueRef::PairingBoundaryP3 => (Fq::zero(), Fq::zero(), pb_batched(pb3)),
                }
            };

            let (b_src, eqc_src, v_src) = endpoint(edge.src);
            let (b_dst, eqc_dst, v_dst) = endpoint(edge.dst);

            // Anchor boundary constants to the opposite selector/beta (GT-style).
            let (b_src, eqc_src, v_src) = match edge.src {
                G1ValueRef::PairingBoundaryP1
                | G1ValueRef::PairingBoundaryP2
                | G1ValueRef::PairingBoundaryP3 => (b_dst, eqc_dst, v_src),
                _ => (b_src, eqc_src, v_src),
            };
            let (b_dst, eqc_dst, v_dst) = match edge.dst {
                G1ValueRef::PairingBoundaryP1
                | G1ValueRef::PairingBoundaryP2
                | G1ValueRef::PairingBoundaryP3 => (b_src, eqc_src, v_dst),
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

//! G2 wiring (copy/boundary) sumcheck.
//!
//! Same structure as `g1::wiring`, but coordinates are in \( \mathbb{F}_{q^2} \), so we batch
//! (x.c0, x.c1, y.c0, y.c1, ind) with powers of a random \(\mu\).

use ark_bn254::{Fq, G2Affine};
use ark_ec::AffineRepr;
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
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    zkvm::{
        proof_serialization::PairingBoundary,
        recursion::{
            constraints::system::{ConstraintSystem, G2AddNative},
            gt::shift::{eq_lsb_evals, eq_lsb_mle},
            verifier::RecursionVerifierInput,
            wiring_plan::{G2ValueRef, G2WiringEdge},
            ConstraintType,
        },
        witness::VirtualPolynomial,
    },
};

const NUM_VARS: usize = 8;
const DEGREE: usize = 2;

#[inline]
fn g2_const_from_affine(p: &G2Affine) -> (Fq, Fq, Fq, Fq, Fq) {
    if p.is_zero() {
        (Fq::zero(), Fq::zero(), Fq::zero(), Fq::zero(), Fq::one())
    } else {
        (p.x.c0, p.x.c1, p.y.c0, p.y.c1, Fq::zero())
    }
}

#[derive(Clone, Debug)]
pub struct WiringG2Prover<T: Transcript> {
    edges: Vec<G2WiringEdge>,
    lambdas: Vec<Fq>,
    mu: Fq,
    mu2: Fq,
    mu3: Fq,
    mu4: Fq,
    eq_poly: MultilinearPolynomial<Fq>,
    round: usize,

    // Scalar mul output polys (8-var), indexed by scalar-mul instance.
    xa_next_c0: Vec<Option<MultilinearPolynomial<Fq>>>,
    xa_next_c1: Vec<Option<MultilinearPolynomial<Fq>>>,
    ya_next_c0: Vec<Option<MultilinearPolynomial<Fq>>>,
    ya_next_c1: Vec<Option<MultilinearPolynomial<Fq>>>,
    a_ind: Vec<Option<MultilinearPolynomial<Fq>>>,

    // Add rows (0-var scalars).
    add_rows: Vec<Option<G2AddNative>>,

    // Scalar-mul base constants, indexed by scalar-mul instance.
    smul_base: Vec<Option<(Fq, Fq, Fq, Fq, Fq)>>, // (x0,x1,y0,y1,ind)

    // Pairing boundary constants.
    pairing_p1: (Fq, Fq, Fq, Fq, Fq),
    pairing_p2: (Fq, Fq, Fq, Fq, Fq),
    pairing_p3: (Fq, Fq, Fq, Fq, Fq),

    _marker: std::marker::PhantomData<T>,
}

#[cfg(feature = "allocative")]
impl<T: Transcript> allocative::Allocative for WiringG2Prover<T> {
    fn visit<'a, 'b: 'a>(&self, _visitor: &'a mut allocative::Visitor<'b>) {}
}

impl<T: Transcript> WiringG2Prover<T> {
    pub fn new(
        cs: &ConstraintSystem,
        edges: Vec<G2WiringEdge>,
        pairing_boundary: &PairingBoundary,
        transcript: &mut T,
    ) -> Self {
        let a_g2: Vec<<Fq as JoltField>::Challenge> =
            (0..NUM_VARS).map(|_| Fq::one().into()).collect();
        let eq_poly = MultilinearPolynomial::from(eq_lsb_evals::<Fq>(&a_g2));

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

        Self {
            edges,
            lambdas,
            mu,
            mu2,
            mu3,
            mu4,
            eq_poly,
            round: 0,
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
            _marker: std::marker::PhantomData,
        }
    }

    fn eval_value_evals(
        &self,
        v: G2ValueRef,
        i: usize,
    ) -> (
        [Fq; DEGREE],
        [Fq; DEGREE],
        [Fq; DEGREE],
        [Fq; DEGREE],
        [Fq; DEGREE],
    ) {
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
                (x0, x1, y0, y1, ind)
            }
            G2ValueRef::G2AddOut { instance } => {
                let row = self.add_rows[instance].as_ref().unwrap();
                (
                    [row.x_r_c0; DEGREE],
                    [row.x_r_c1; DEGREE],
                    [row.y_r_c0; DEGREE],
                    [row.y_r_c1; DEGREE],
                    [row.ind_r; DEGREE],
                )
            }
            G2ValueRef::G2AddInP { instance } => {
                let row = self.add_rows[instance].as_ref().unwrap();
                (
                    [row.x_p_c0; DEGREE],
                    [row.x_p_c1; DEGREE],
                    [row.y_p_c0; DEGREE],
                    [row.y_p_c1; DEGREE],
                    [row.ind_p; DEGREE],
                )
            }
            G2ValueRef::G2AddInQ { instance } => {
                let row = self.add_rows[instance].as_ref().unwrap();
                (
                    [row.x_q_c0; DEGREE],
                    [row.x_q_c1; DEGREE],
                    [row.y_q_c0; DEGREE],
                    [row.y_q_c1; DEGREE],
                    [row.ind_q; DEGREE],
                )
            }
            G2ValueRef::G2ScalarMulBase { instance } => {
                let (x0, x1, y0, y1, ind) = self.smul_base[instance].unwrap();
                (
                    [x0; DEGREE],
                    [x1; DEGREE],
                    [y0; DEGREE],
                    [y1; DEGREE],
                    [ind; DEGREE],
                )
            }
            G2ValueRef::PairingBoundaryP1 => {
                let (x0, x1, y0, y1, ind) = self.pairing_p1;
                (
                    [x0; DEGREE],
                    [x1; DEGREE],
                    [y0; DEGREE],
                    [y1; DEGREE],
                    [ind; DEGREE],
                )
            }
            G2ValueRef::PairingBoundaryP2 => {
                let (x0, x1, y0, y1, ind) = self.pairing_p2;
                (
                    [x0; DEGREE],
                    [x1; DEGREE],
                    [y0; DEGREE],
                    [y1; DEGREE],
                    [ind; DEGREE],
                )
            }
            G2ValueRef::PairingBoundaryP3 => {
                let (x0, x1, y0, y1, ind) = self.pairing_p3;
                (
                    [x0; DEGREE],
                    [x1; DEGREE],
                    [y0; DEGREE],
                    [y1; DEGREE],
                    [ind; DEGREE],
                )
            }
        }
    }
}

impl<T: Transcript> SumcheckInstanceProver<Fq, T> for WiringG2Prover<T> {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        NUM_VARS
    }

    fn input_claim(&self, _acc: &crate::poly::opening_proof::ProverOpeningAccumulator<Fq>) -> Fq {
        Fq::zero()
    }

    fn compute_message(&mut self, _round: usize, previous_claim: Fq) -> UniPoly<Fq> {
        let half = self.eq_poly.len() / 2;
        if half == 0 {
            return UniPoly::from_evals_and_hint(previous_claim, &[Fq::zero(); DEGREE]);
        }

        let lambdas = self.lambdas.clone();
        let edges = self.edges.clone();
        let (mu, mu2, mu3, mu4) = (self.mu, self.mu2, self.mu3, self.mu4);

        let evals = (0..half)
            .into_par_iter()
            .map(|i| {
                let eq_evals = self
                    .eq_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                let mut delta_evals = [Fq::zero(); DEGREE];
                for (lambda, edge) in lambdas.iter().zip(edges.iter()) {
                    let (sx0, sx1, sy0, sy1, sind) = self.eval_value_evals(edge.src, i);
                    let (dx0, dx1, dy0, dy1, dind) = self.eval_value_evals(edge.dst, i);
                    for t in 0..DEGREE {
                        let batched = (sx0[t] - dx0[t])
                            + mu * (sx1[t] - dx1[t])
                            + mu2 * (sy0[t] - dy0[t])
                            + mu3 * (sy1[t] - dy1[t])
                            + mu4 * (sind[t] - dind[t]);
                        delta_evals[t] += *lambda * batched;
                    }
                }

                let mut term = [Fq::zero(); DEGREE];
                for t in 0..DEGREE {
                    term[t] = eq_evals[t] * delta_evals[t];
                }
                term
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
        self.eq_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.round = round + 1;
    }

    fn cache_openings(
        &self,
        _acc: &mut crate::poly::opening_proof::ProverOpeningAccumulator<Fq>,
        _transcript: &mut T,
        _r: &[<Fq as JoltField>::Challenge],
    ) {
        // No-op.
    }
}

#[derive(Clone, Debug)]
pub struct WiringG2Verifier {
    edges: Vec<G2WiringEdge>,
    lambdas: Vec<Fq>,
    mu: Fq,
    mu2: Fq,
    mu3: Fq,
    mu4: Fq,
    a_g2: Vec<<Fq as JoltField>::Challenge>,
    pairing_boundary: PairingBoundary,
    smul_bases: Vec<(Fq, Fq, Fq, Fq, Fq)>,
}

impl WiringG2Verifier {
    pub fn new<T: Transcript>(input: &RecursionVerifierInput, transcript: &mut T) -> Self {
        let a_g2: Vec<<Fq as JoltField>::Challenge> =
            (0..NUM_VARS).map(|_| Fq::one().into()).collect();
        let mu: Fq = transcript.challenge_scalar();
        let mu2 = mu * mu;
        let mu3 = mu2 * mu;
        let mu4 = mu2 * mu2;

        let edges = input.wiring.g2.clone();
        let lambdas: Vec<Fq> = transcript.challenge_vector(edges.len());

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

        Self {
            edges,
            lambdas,
            mu,
            mu2,
            mu3,
            mu4,
            a_g2,
            pairing_boundary: input.pairing_boundary.clone(),
            smul_bases,
        }
    }

    fn eval_value_at_r(
        &self,
        v: G2ValueRef,
        acc: &VerifierOpeningAccumulator<Fq>,
        _r: &[<Fq as JoltField>::Challenge],
    ) -> (Fq, Fq, Fq, Fq, Fq) {
        match v {
            G2ValueRef::G2ScalarMulOut { instance } => (
                acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_scalar_mul_xa_next_c0(instance),
                    SumcheckId::G2ScalarMul,
                )
                .1,
                acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_scalar_mul_xa_next_c1(instance),
                    SumcheckId::G2ScalarMul,
                )
                .1,
                acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_scalar_mul_ya_next_c0(instance),
                    SumcheckId::G2ScalarMul,
                )
                .1,
                acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_scalar_mul_ya_next_c1(instance),
                    SumcheckId::G2ScalarMul,
                )
                .1,
                acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_scalar_mul_a_indicator(instance),
                    SumcheckId::G2ScalarMul,
                )
                .1,
            ),
            G2ValueRef::G2AddOut { instance } => (
                acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_xr_c0(instance),
                    SumcheckId::G2Add,
                )
                .1,
                acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_xr_c1(instance),
                    SumcheckId::G2Add,
                )
                .1,
                acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_yr_c0(instance),
                    SumcheckId::G2Add,
                )
                .1,
                acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_yr_c1(instance),
                    SumcheckId::G2Add,
                )
                .1,
                acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_r_indicator(instance),
                    SumcheckId::G2Add,
                )
                .1,
            ),
            G2ValueRef::G2AddInP { instance } => (
                acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_xp_c0(instance),
                    SumcheckId::G2Add,
                )
                .1,
                acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_xp_c1(instance),
                    SumcheckId::G2Add,
                )
                .1,
                acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_yp_c0(instance),
                    SumcheckId::G2Add,
                )
                .1,
                acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_yp_c1(instance),
                    SumcheckId::G2Add,
                )
                .1,
                acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_p_indicator(instance),
                    SumcheckId::G2Add,
                )
                .1,
            ),
            G2ValueRef::G2AddInQ { instance } => (
                acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_xq_c0(instance),
                    SumcheckId::G2Add,
                )
                .1,
                acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_xq_c1(instance),
                    SumcheckId::G2Add,
                )
                .1,
                acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_yq_c0(instance),
                    SumcheckId::G2Add,
                )
                .1,
                acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_yq_c1(instance),
                    SumcheckId::G2Add,
                )
                .1,
                acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_q_indicator(instance),
                    SumcheckId::G2Add,
                )
                .1,
            ),
            G2ValueRef::G2ScalarMulBase { instance } => self.smul_bases[instance],
            G2ValueRef::PairingBoundaryP1 => g2_const_from_affine(&self.pairing_boundary.p1_g2),
            G2ValueRef::PairingBoundaryP2 => g2_const_from_affine(&self.pairing_boundary.p2_g2),
            G2ValueRef::PairingBoundaryP3 => g2_const_from_affine(&self.pairing_boundary.p3_g2),
        }
    }
}

impl<T: Transcript> SumcheckInstanceVerifier<Fq, T> for WiringG2Verifier {
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
        r: &[<Fq as JoltField>::Challenge],
    ) -> Fq {
        debug_assert_eq!(r.len(), NUM_VARS);
        let eq = eq_lsb_mle::<Fq>(&self.a_g2, r);

        let mut delta = Fq::zero();
        for (lambda, edge) in self.lambdas.iter().zip(self.edges.iter()) {
            let (sx0, sx1, sy0, sy1, sind) = self.eval_value_at_r(edge.src, acc, r);
            let (dx0, dx1, dy0, dy1, dind) = self.eval_value_at_r(edge.dst, acc, r);
            let batched = (sx0 - dx0)
                + self.mu * (sx1 - dx1)
                + self.mu2 * (sy0 - dy0)
                + self.mu3 * (sy1 - dy1)
                + self.mu4 * (sind - dind);
            delta += *lambda * batched;
        }

        eq * delta
    }

    fn cache_openings(
        &self,
        _acc: &mut VerifierOpeningAccumulator<Fq>,
        _transcript: &mut T,
        _r: &[<Fq as JoltField>::Challenge],
    ) {
        // No-op.
    }
}

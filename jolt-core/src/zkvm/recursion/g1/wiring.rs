//! G1 wiring (copy/boundary) sumcheck.
//!
//! Proves:
//! \[
//!   0 = \sum_{s \in \{0,1\}^{8}} \mathrm{Eq}(a_{G1}, s) \cdot
//!       \sum_{e \in E_{G1}} \lambda_e \cdot \Delta_e(s)
//! \]
//! where each \(\Delta_e\) batches G1 coordinates via a random \(\mu\):
//! \(\Delta = (x_{src}-x_{dst}) + \mu(y_{src}-y_{dst}) + \mu^2(ind_{src}-ind_{dst})\).

use ark_bn254::{Fq, G1Affine};
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
            constraints::system::{ConstraintSystem, G1AddNative},
            gt::shift::{eq_lsb_evals, eq_lsb_mle},
            verifier::RecursionVerifierInput,
            wiring_plan::{G1ValueRef, G1WiringEdge},
            ConstraintType,
        },
        witness::VirtualPolynomial,
    },
};

const NUM_VARS: usize = 8;
const DEGREE: usize = 2;

#[inline]
fn g1_const_from_affine(p: &G1Affine) -> (Fq, Fq, Fq) {
    if p.is_zero() {
        (Fq::zero(), Fq::zero(), Fq::one())
    } else {
        (p.x, p.y, Fq::zero())
    }
}

#[derive(Clone, Debug)]
pub struct WiringG1Prover<T: Transcript> {
    edges: Vec<G1WiringEdge>,
    lambdas: Vec<Fq>,
    mu: Fq,
    mu2: Fq,
    eq_poly: MultilinearPolynomial<Fq>,
    round: usize,

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

    _marker: std::marker::PhantomData<T>,
}

#[cfg(feature = "allocative")]
impl<T: Transcript> allocative::Allocative for WiringG1Prover<T> {
    fn visit<'a, 'b: 'a>(&self, _visitor: &'a mut allocative::Visitor<'b>) {}
}

impl<T: Transcript> WiringG1Prover<T> {
    pub fn new(
        cs: &ConstraintSystem,
        edges: Vec<G1WiringEdge>,
        pairing_boundary: &PairingBoundary,
        transcript: &mut T,
    ) -> Self {
        // Selector point a_G1 = [1,1,...,1] (LSB-first order).
        let a_g1: Vec<<Fq as JoltField>::Challenge> =
            (0..NUM_VARS).map(|_| Fq::one().into()).collect();
        let eq_poly = MultilinearPolynomial::from(eq_lsb_evals::<Fq>(&a_g1));

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
                    _ => {}
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

        Self {
            edges,
            lambdas,
            mu,
            mu2,
            eq_poly,
            round: 0,
            xa_next,
            ya_next,
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
        v: G1ValueRef,
        i: usize,
    ) -> ([Fq; DEGREE], [Fq; DEGREE], [Fq; DEGREE]) {
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
                (x, y, ind)
            }
            G1ValueRef::G1AddOut { instance } => {
                let row = self.add_rows[instance].as_ref().unwrap();
                ([row.x_r; DEGREE], [row.y_r; DEGREE], [row.ind_r; DEGREE])
            }
            G1ValueRef::G1AddInP { instance } => {
                let row = self.add_rows[instance].as_ref().unwrap();
                ([row.x_p; DEGREE], [row.y_p; DEGREE], [row.ind_p; DEGREE])
            }
            G1ValueRef::G1AddInQ { instance } => {
                let row = self.add_rows[instance].as_ref().unwrap();
                ([row.x_q; DEGREE], [row.y_q; DEGREE], [row.ind_q; DEGREE])
            }
            G1ValueRef::G1ScalarMulBase { instance } => {
                let (x, y, ind) = self.smul_base[instance].unwrap();
                ([x; DEGREE], [y; DEGREE], [ind; DEGREE])
            }
            G1ValueRef::PairingBoundaryP1 => {
                let (x, y, ind) = self.pairing_p1;
                ([x; DEGREE], [y; DEGREE], [ind; DEGREE])
            }
            G1ValueRef::PairingBoundaryP2 => {
                let (x, y, ind) = self.pairing_p2;
                ([x; DEGREE], [y; DEGREE], [ind; DEGREE])
            }
            G1ValueRef::PairingBoundaryP3 => {
                let (x, y, ind) = self.pairing_p3;
                ([x; DEGREE], [y; DEGREE], [ind; DEGREE])
            }
        }
    }
}

impl<T: Transcript> SumcheckInstanceProver<Fq, T> for WiringG1Prover<T> {
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
        let mu = self.mu;
        let mu2 = self.mu2;

        let evals = (0..half)
            .into_par_iter()
            .map(|i| {
                let eq_evals = self
                    .eq_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                let mut delta_evals = [Fq::zero(); DEGREE];
                for (lambda, edge) in lambdas.iter().zip(edges.iter()) {
                    let (sx, sy, sind) = self.eval_value_evals(edge.src, i);
                    let (dx, dy, dind) = self.eval_value_evals(edge.dst, i);
                    for t in 0..DEGREE {
                        let batched =
                            (sx[t] - dx[t]) + mu * (sy[t] - dy[t]) + mu2 * (sind[t] - dind[t]);
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
        for poly in self.xa_next.iter_mut().flatten() {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in self.ya_next.iter_mut().flatten() {
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
        _sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        // No-op: this instance only reads cached openings.
    }
}

#[derive(Clone, Debug)]
pub struct WiringG1Verifier {
    edges: Vec<G1WiringEdge>,
    lambdas: Vec<Fq>,
    mu: Fq,
    mu2: Fq,
    a_g1: Vec<<Fq as JoltField>::Challenge>,
    pairing_boundary: PairingBoundary,
    smul_bases: Vec<(Fq, Fq, Fq)>,
}

impl WiringG1Verifier {
    pub fn new<T: Transcript>(input: &RecursionVerifierInput, transcript: &mut T) -> Self {
        let a_g1: Vec<<Fq as JoltField>::Challenge> =
            (0..NUM_VARS).map(|_| Fq::one().into()).collect();
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

        Self {
            edges,
            lambdas,
            mu,
            mu2,
            a_g1,
            pairing_boundary: input.pairing_boundary.clone(),
            smul_bases,
        }
    }

    fn eval_value_at_r(
        &self,
        v: G1ValueRef,
        acc: &VerifierOpeningAccumulator<Fq>,
        r: &[<Fq as JoltField>::Challenge],
    ) -> (Fq, Fq, Fq) {
        match v {
            G1ValueRef::G1ScalarMulOut { instance } => {
                let x = acc
                    .get_virtual_polynomial_opening(
                        VirtualPolynomial::g1_scalar_mul_xa_next(instance),
                        SumcheckId::G1ScalarMul,
                    )
                    .1;
                let y = acc
                    .get_virtual_polynomial_opening(
                        VirtualPolynomial::g1_scalar_mul_ya_next(instance),
                        SumcheckId::G1ScalarMul,
                    )
                    .1;
                let ind = acc
                    .get_virtual_polynomial_opening(
                        VirtualPolynomial::g1_scalar_mul_a_indicator(instance),
                        SumcheckId::G1ScalarMul,
                    )
                    .1;
                let _ = r; // scalar-mul polys are opened at r by their own sumcheck.
                (x, y, ind)
            }
            G1ValueRef::G1AddOut { instance } => (
                acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::g1_add_xr(instance),
                    SumcheckId::G1Add,
                )
                .1,
                acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::g1_add_yr(instance),
                    SumcheckId::G1Add,
                )
                .1,
                acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::g1_add_r_indicator(instance),
                    SumcheckId::G1Add,
                )
                .1,
            ),
            G1ValueRef::G1AddInP { instance } => (
                acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::g1_add_xp(instance),
                    SumcheckId::G1Add,
                )
                .1,
                acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::g1_add_yp(instance),
                    SumcheckId::G1Add,
                )
                .1,
                acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::g1_add_p_indicator(instance),
                    SumcheckId::G1Add,
                )
                .1,
            ),
            G1ValueRef::G1AddInQ { instance } => (
                acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::g1_add_xq(instance),
                    SumcheckId::G1Add,
                )
                .1,
                acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::g1_add_yq(instance),
                    SumcheckId::G1Add,
                )
                .1,
                acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::g1_add_q_indicator(instance),
                    SumcheckId::G1Add,
                )
                .1,
            ),
            G1ValueRef::G1ScalarMulBase { instance } => self.smul_bases[instance],
            G1ValueRef::PairingBoundaryP1 => g1_const_from_affine(&self.pairing_boundary.p1_g1),
            G1ValueRef::PairingBoundaryP2 => g1_const_from_affine(&self.pairing_boundary.p2_g1),
            G1ValueRef::PairingBoundaryP3 => g1_const_from_affine(&self.pairing_boundary.p3_g1),
        }
    }
}

impl<T: Transcript> SumcheckInstanceVerifier<Fq, T> for WiringG1Verifier {
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
        let eq = eq_lsb_mle::<Fq>(&self.a_g1, r);

        let mut delta = Fq::zero();
        for (lambda, edge) in self.lambdas.iter().zip(self.edges.iter()) {
            let (sx, sy, sind) = self.eval_value_at_r(edge.src, acc, r);
            let (dx, dy, dind) = self.eval_value_at_r(edge.dst, acc, r);
            let batched = (sx - dx) + self.mu * (sy - dy) + self.mu2 * (sind - dind);
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

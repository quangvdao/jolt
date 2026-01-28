//! GT wiring binding check (legacy safety net).
//!
//! This is a zero-polynomial Stage-2 sumcheck instance that enforces that the auxiliary
//! wiring sums emitted by `WiringGtProver` match the *actual* per-edge wiring expression
//! under the verifier's public inputs (pairing boundary, joint commitment, base hints).
//!
//! This module is kept to preserve the legacy wiring test guarantees. In fully fused wiring
//! mode, we intend for the wiring verifier to compute the full expression directly from
//! fused openings and public data, making this redundant.

use crate::{
    field::JoltField,
    poly::{
        opening_proof::{OpeningAccumulator, SumcheckId, VerifierOpeningAccumulator},
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    zkvm::recursion::gt::wiring::{ELEM_VARS, STEP_VARS},
    zkvm::recursion::wiring_plan::{GtConsumer, GtProducer, GtWiringEdge},
    zkvm::witness::VirtualPolynomial,
};

use ark_bn254::{Fq, Fq12};
use ark_ff::{One, Zero};

use crate::zkvm::proof_serialization::PairingBoundary;

#[derive(Clone)]
pub struct GtWiringBinding {
    pub edges: Vec<GtWiringEdge>,
    pub lambdas: Vec<Fq>,
    pub tau: Vec<<Fq as JoltField>::Challenge>,
    pub gt_exp_out_step: Vec<usize>,
    pub gt_exp_bases: Vec<Fq12>,
    pub pairing_boundary: PairingBoundary,
    pub joint_commitment: Fq12,
    pub num_c_vars: usize,
    pub gt_exp_constraint_idx: Vec<usize>,
    pub gt_mul_constraint_idx: Vec<usize>,
}

pub struct GtWiringBindingProver<T: Transcript> {
    num_rounds: usize,
    _marker: core::marker::PhantomData<T>,
}

impl<T: Transcript> GtWiringBindingProver<T> {
    pub fn new(num_rounds: usize) -> Self {
        Self {
            num_rounds,
            _marker: core::marker::PhantomData,
        }
    }
}

#[cfg(feature = "allocative")]
impl<T: Transcript> allocative::Allocative for GtWiringBindingProver<T> {
    fn visit<'a, 'b: 'a>(&self, _visitor: &'a mut allocative::Visitor<'b>) {}
}

impl<T: Transcript> SumcheckInstanceProver<Fq, T> for GtWiringBindingProver<T> {
    fn degree(&self) -> usize {
        1
    }
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }
    fn input_claim(&self, _acc: &crate::poly::opening_proof::ProverOpeningAccumulator<Fq>) -> Fq {
        Fq::zero()
    }
    fn compute_message(&mut self, _round: usize, _previous_claim: Fq) -> UniPoly<Fq> {
        UniPoly::from_coeff(vec![Fq::zero()])
    }
    fn ingest_challenge(&mut self, _r_j: <Fq as JoltField>::Challenge, _round: usize) {}

    fn cache_openings(
        &self,
        _accumulator: &mut crate::poly::opening_proof::ProverOpeningAccumulator<Fq>,
        _transcript: &mut T,
        _sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        // No-op.
    }
}

pub struct GtWiringBindingVerifier {
    inner: GtWiringBinding,
}

impl GtWiringBindingVerifier {
    pub fn new(inner: GtWiringBinding) -> Self {
        Self { inner }
    }

    #[inline]
    fn eq_rc_at(r_c: &[<Fq as JoltField>::Challenge], idx: usize) -> Fq {
        r_c.iter()
            .enumerate()
            .map(|(b, c)| {
                let r_b: Fq = (*c).into();
                if ((idx >> b) & 1) == 1 {
                    r_b
                } else {
                    Fq::one() - r_b
                }
            })
            .product()
    }
}

impl<T: Transcript> SumcheckInstanceVerifier<Fq, T> for GtWiringBindingVerifier {
    fn degree(&self) -> usize {
        1
    }
    fn num_rounds(&self) -> usize {
        self.inner.num_c_vars + STEP_VARS + ELEM_VARS
    }
    fn input_claim(&self, _acc: &VerifierOpeningAccumulator<Fq>) -> Fq {
        Fq::zero()
    }

    fn expected_output_claim(
        &self,
        acc: &VerifierOpeningAccumulator<Fq>,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) -> Fq {
        // If the sums are consistent, this instance is identically zero.
        debug_assert_eq!(
            sumcheck_challenges.len(),
            self.inner.num_c_vars + STEP_VARS + ELEM_VARS
        );

        // Read the prover-emitted wiring sums (these claims are already present in the accumulator).
        let (_pt, src_sum_claim) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::gt_wiring_src_sum(),
            SumcheckId::GtWiring,
        );
        let (_pt, dst_sum_claim) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::gt_wiring_dst_sum(),
            SumcheckId::GtWiring,
        );

        // Compute expected sums from per-edge values under public inputs.
        let r_c = &sumcheck_challenges[..self.inner.num_c_vars];
        let r_step = &sumcheck_challenges[self.inner.num_c_vars..self.inner.num_c_vars + STEP_VARS];
        let r_elem_chal = &sumcheck_challenges[self.inner.num_c_vars + STEP_VARS..];
        let r_elem: Vec<Fq> = r_elem_chal.iter().map(|c| (*c).into()).collect();

        // Build a value source that reads *per-instance* openings (legacy mode).
        let value_source = super::wiring::LegacyGtWiringValueSource::new(
            acc,
            r_step,
            &r_elem,
            &self.inner.gt_exp_out_step,
            &self.inner.gt_exp_bases,
            &self.inner.joint_commitment,
            &self.inner.pairing_boundary,
        );

        let mut src_sum_expected = Fq::zero();
        let mut dst_sum_expected = Fq::zero();

        for (lambda, edge) in self.inner.lambdas.iter().zip(self.inner.edges.iter()) {
            let w_src = match edge.src {
                GtProducer::GtExpRho { instance } => {
                    Self::eq_rc_at(r_c, self.inner.gt_exp_constraint_idx[instance])
                }
                GtProducer::GtMulResult { instance } => {
                    Self::eq_rc_at(r_c, self.inner.gt_mul_constraint_idx[instance])
                }
                GtProducer::GtExpBase { instance } => {
                    Self::eq_rc_at(r_c, self.inner.gt_exp_constraint_idx[instance])
                }
            };

            let w_dst = match edge.dst {
                GtConsumer::GtMulLhs { instance } | GtConsumer::GtMulRhs { instance } => {
                    Self::eq_rc_at(r_c, self.inner.gt_mul_constraint_idx[instance])
                }
                GtConsumer::GtExpBase { instance } => {
                    Self::eq_rc_at(r_c, self.inner.gt_exp_constraint_idx[instance])
                }
                // Anchor globals to src (same convention as wiring prover).
                GtConsumer::JointCommitment | GtConsumer::PairingBoundaryRhs => w_src,
            };

            let eq_s = value_source.eq_s_for_src(edge.src);
            let src = value_source.src_at_r(edge.src);
            let dst = value_source.dst_at_r(edge.dst);

            src_sum_expected += *lambda * w_src * eq_s * src;
            dst_sum_expected += *lambda * w_dst * eq_s * dst;
        }

        // Return 0 iff both sums match.
        (src_sum_claim - src_sum_expected) + (dst_sum_claim - dst_sum_expected)
    }

    fn cache_openings(
        &self,
        _accumulator: &mut VerifierOpeningAccumulator<Fq>,
        _transcript: &mut T,
        _sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        // No-op.
    }
}

//! Cache-only Stage-2 GTExp base openings at the Stage-2 point (u-domain).
//!
//! This is analogous to `stage2_openings.rs` but targets the **native 4-var GT element domain**
//! `(u, c_common)` rather than the packed GTExp domain `(s,u,c_common)`.
//!
//! Its purpose is to make `GtExpTerm::{Base,Base2,Base3}` available as virtual openings under
//! `SumcheckId::GtExpBaseClaimReduction` so that Stage 3 prefix packing can include committed
//! x4 base rows.

use crate::{
    field::JoltField,
    poly::{
        dense_mlpoly::DensePolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    zkvm::recursion::constraints::system::{ConstraintLocator, ConstraintType},
    zkvm::recursion::gt::{
        indexing::{gt_mul_c_tail_range, k_exp, k_gt, num_gt_exp_constraints_padded},
        types::GtExpWitness,
    },
    zkvm::witness::{GtExpTerm, RecursionPoly, VirtualPolynomial},
};

use core::marker::PhantomData;

use allocative::Allocative;
use ark_bn254::Fq;
use ark_ff::Zero;
use jolt_optimizations::get_g_mle;

const U_VARS: usize = 4;
const STEP_STRIDE: usize = 1usize << 7; // 2^STEP_VARS (STEP_VARS = 7)

#[derive(Clone, Debug, Allocative)]
pub struct GtExpBaseStage2OpeningsParams {
    /// Stage-2 GT-local suffix length used for batching (k_common = k_gt).
    pub k_common: usize,
    pub k_exp: usize,
}

impl GtExpBaseStage2OpeningsParams {
    pub fn from_constraint_types(constraint_types: &[ConstraintType]) -> Self {
        let k_common = k_gt(constraint_types);
        let k_exp = k_exp(constraint_types);
        Self { k_common, k_exp }
    }
}

pub struct GtExpBaseStage2OpeningsProver<T: Transcript> {
    params: GtExpBaseStage2OpeningsParams,
    base: MultilinearPolynomial<Fq>,
    base2: MultilinearPolynomial<Fq>,
    base3: MultilinearPolynomial<Fq>,
    base_square_quotient: MultilinearPolynomial<Fq>,
    base_cube_quotient: MultilinearPolynomial<Fq>,
    _marker: PhantomData<T>,
}

#[cfg(feature = "allocative")]
impl<T: Transcript> allocative::Allocative for GtExpBaseStage2OpeningsProver<T> {
    fn visit<'a, 'b: 'a>(&self, _visitor: &'a mut allocative::Visitor<'b>) {}
}

impl<T: Transcript> GtExpBaseStage2OpeningsProver<T> {
    pub fn new(
        constraint_types: &[ConstraintType],
        locator_by_constraint: &[ConstraintLocator],
        gt_exp_witnesses: &[GtExpWitness<Fq>],
        transcript: &mut T,
    ) -> Self {
        let params = GtExpBaseStage2OpeningsParams::from_constraint_types(constraint_types);
        // No transcript sampling here: this instance is cache-only.
        let _ = transcript;
        let row_size = 1usize << U_VARS; // 16
        let num_gt_exp_padded = num_gt_exp_constraints_padded(constraint_types);

        let mut base_uc = vec![Fq::zero(); num_gt_exp_padded * row_size];
        let mut base2_uc = vec![Fq::zero(); num_gt_exp_padded * row_size];
        let mut base3_uc = vec![Fq::zero(); num_gt_exp_padded * row_size];
        let mut q2_uc = vec![Fq::zero(); num_gt_exp_padded * row_size];
        let mut q3_uc = vec![Fq::zero(); num_gt_exp_padded * row_size];

        for global_idx in 0..constraint_types.len() {
            if let ConstraintLocator::GtExp { local } = locator_by_constraint[global_idx] {
                let base11 = &gt_exp_witnesses[local].base_packed;
                let base211 = &gt_exp_witnesses[local].base2_packed;
                let base311 = &gt_exp_witnesses[local].base3_packed;
                debug_assert_eq!(base11.len(), 1usize << 11);
                debug_assert_eq!(base211.len(), 1usize << 11);
                debug_assert_eq!(base311.len(), 1usize << 11);

                let off = local * row_size;
                for u in 0..row_size {
                    // Extract s=0 slice (base is replicated across s).
                    base_uc[off + u] = base11[u * STEP_STRIDE];
                    base2_uc[off + u] = base211[u * STEP_STRIDE];
                    base3_uc[off + u] = base311[u * STEP_STRIDE];
                }
            }
        }

        // Compute pointwise quotient polynomials on the native u-domain:
        // - base(u)^2 - base2(u) = q2(u) * g(u)
        // - base2(u)*base(u) - base3(u) = q3(u) * g(u)
        //
        // These are stacked over c_exp (family-local), same as the base polynomials.
        let g_mle = get_g_mle();
        debug_assert_eq!(g_mle.len(), row_size);
        for c in 0..num_gt_exp_padded {
            let off = c * row_size;
            for u in 0..row_size {
                let g = g_mle[u];
                let b = base_uc[off + u];
                let b2 = base2_uc[off + u];
                let b3 = base3_uc[off + u];
                if g.is_zero() {
                    debug_assert!((b * b - b2).is_zero());
                    debug_assert!((b2 * b - b3).is_zero());
                    q2_uc[off + u] = Fq::zero();
                    q3_uc[off + u] = Fq::zero();
                } else {
                    let inv_g = g.inverse().unwrap();
                    q2_uc[off + u] = (b * b - b2) * inv_g;
                    q3_uc[off + u] = (b2 * b - b3) * inv_g;
                }
            }
        }

        Self {
            params,
            // Store in [u4 low bits, c_exp high bits] order so `c` is a suffix in Stage 2.
            base: MultilinearPolynomial::LargeScalars(DensePolynomial::new(base_uc)),
            base2: MultilinearPolynomial::LargeScalars(DensePolynomial::new(base2_uc)),
            base3: MultilinearPolynomial::LargeScalars(DensePolynomial::new(base3_uc)),
            base_square_quotient: MultilinearPolynomial::LargeScalars(DensePolynomial::new(q2_uc)),
            base_cube_quotient: MultilinearPolynomial::LargeScalars(DensePolynomial::new(q3_uc)),
            _marker: PhantomData,
        }
    }
}

impl<T: Transcript> SumcheckInstanceProver<Fq, T> for GtExpBaseStage2OpeningsProver<T> {
    fn degree(&self) -> usize {
        1
    }
    fn num_rounds(&self) -> usize {
        // Participate with (u4 + k_common) rounds so it is suffix-aligned with other GT Stage-2
        // instances that use the common GT c-domain.
        U_VARS + self.params.k_common
    }
    fn input_claim(&self, _acc: &ProverOpeningAccumulator<Fq>) -> Fq {
        Fq::zero()
    }
    fn compute_message(&mut self, _round: usize, _previous_claim: Fq) -> UniPoly<Fq> {
        UniPoly::from_coeff(vec![Fq::zero()])
    }

    fn ingest_challenge(&mut self, r_j: <Fq as JoltField>::Challenge, round: usize) {
        // Bind all 4 u-bits, and only the tail `k_exp` bits of the c-suffix.
        if round < U_VARS {
            self.base.bind_parallel(r_j, BindingOrder::LowToHigh);
            self.base2.bind_parallel(r_j, BindingOrder::LowToHigh);
            self.base3.bind_parallel(r_j, BindingOrder::LowToHigh);
            self.base_square_quotient
                .bind_parallel(r_j, BindingOrder::LowToHigh);
            self.base_cube_quotient
                .bind_parallel(r_j, BindingOrder::LowToHigh);
            return;
        }
        let c_round = round - U_VARS;
        let dummy = self.params.k_common.saturating_sub(self.params.k_exp);
        if c_round < dummy {
            return;
        }
        self.base.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.base2.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.base3.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.base_square_quotient
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.base_cube_quotient
            .bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<Fq>,
        transcript: &mut T,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        debug_assert_eq!(sumcheck_challenges.len(), U_VARS + self.params.k_common);
        let mut r = Vec::with_capacity(U_VARS + self.params.k_exp);
        r.extend_from_slice(&sumcheck_challenges[..U_VARS]);
        let tail = gt_mul_c_tail_range(self.params.k_common, self.params.k_exp);
        r.extend_from_slice(&sumcheck_challenges[tail]);
        let opening_point = OpeningPoint::<BIG_ENDIAN, Fq>::new(r);

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Recursion(RecursionPoly::GtExp {
                term: GtExpTerm::Base,
            }),
            SumcheckId::GtExpBaseClaimReduction,
            opening_point.clone(),
            self.base.get_bound_coeff(0),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Recursion(RecursionPoly::GtExp {
                term: GtExpTerm::Base2,
            }),
            SumcheckId::GtExpBaseClaimReduction,
            opening_point.clone(),
            self.base2.get_bound_coeff(0),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Recursion(RecursionPoly::GtExp {
                term: GtExpTerm::Base3,
            }),
            SumcheckId::GtExpBaseClaimReduction,
            opening_point.clone(),
            self.base3.get_bound_coeff(0),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Recursion(RecursionPoly::GtExp {
                term: GtExpTerm::BaseSquareQuotient,
            }),
            SumcheckId::GtExpBaseClaimReduction,
            opening_point.clone(),
            self.base_square_quotient.get_bound_coeff(0),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Recursion(RecursionPoly::GtExp {
                term: GtExpTerm::BaseCubeQuotient,
            }),
            SumcheckId::GtExpBaseClaimReduction,
            opening_point,
            self.base_cube_quotient.get_bound_coeff(0),
        );
    }
}

pub struct GtExpBaseStage2OpeningsVerifier {
    params: GtExpBaseStage2OpeningsParams,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for GtExpBaseStage2OpeningsVerifier {
    fn visit<'a, 'b: 'a>(&self, _visitor: &'a mut allocative::Visitor<'b>) {}
}

impl GtExpBaseStage2OpeningsVerifier {
    pub fn new(constraint_types: &[ConstraintType]) -> Self {
        Self {
            params: GtExpBaseStage2OpeningsParams::from_constraint_types(constraint_types),
        }
    }
}

impl<T: Transcript> SumcheckInstanceVerifier<Fq, T> for GtExpBaseStage2OpeningsVerifier {
    fn degree(&self) -> usize {
        1
    }
    fn num_rounds(&self) -> usize {
        U_VARS + self.params.k_common
    }
    fn input_claim(&self, _acc: &VerifierOpeningAccumulator<Fq>) -> Fq {
        Fq::zero()
    }
    fn expected_output_claim(
        &self,
        _acc: &VerifierOpeningAccumulator<Fq>,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) -> Fq {
        debug_assert_eq!(sumcheck_challenges.len(), U_VARS + self.params.k_common);
        // Cache-only: binding of the committed `Base` row is enforced by GT wiring (Stage 2).
        Fq::zero()
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<Fq>,
        transcript: &mut T,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        debug_assert_eq!(sumcheck_challenges.len(), U_VARS + self.params.k_common);
        let mut r = Vec::with_capacity(U_VARS + self.params.k_exp);
        r.extend_from_slice(&sumcheck_challenges[..U_VARS]);
        let tail = gt_mul_c_tail_range(self.params.k_common, self.params.k_exp);
        r.extend_from_slice(&sumcheck_challenges[tail]);
        let opening_point = OpeningPoint::<BIG_ENDIAN, Fq>::new(r);

        for vp in [
            VirtualPolynomial::Recursion(RecursionPoly::GtExp {
                term: GtExpTerm::Base,
            }),
            VirtualPolynomial::Recursion(RecursionPoly::GtExp {
                term: GtExpTerm::Base2,
            }),
            VirtualPolynomial::Recursion(RecursionPoly::GtExp {
                term: GtExpTerm::Base3,
            }),
            VirtualPolynomial::Recursion(RecursionPoly::GtExp {
                term: GtExpTerm::BaseSquareQuotient,
            }),
            VirtualPolynomial::Recursion(RecursionPoly::GtExp {
                term: GtExpTerm::BaseCubeQuotient,
            }),
        ] {
            accumulator.append_virtual(
                transcript,
                vp,
                SumcheckId::GtExpBaseClaimReduction,
                opening_point.clone(),
            );
        }
    }
}

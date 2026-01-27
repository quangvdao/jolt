//! Cache-only Stage-2 GTExp fused openings at the Stage-2 point.
//!
//! End-to-end GT fusion needs `gt_exp_{rho,quotient}_fused()` available at the full Stage-2
//! point `r_stage2 = (r_c_gt, r_x)` so that:\n//! - Stage-3 prefix packing can consume fused GT claims, and\n//! - fused GT shift / wiring can consume fused GT values without per-instance openings.\n+//!\n+//! This is implemented as a no-op (zero) sumcheck instance that:\n+//! - participates in Stage-2 batching only to obtain the Stage-2 point,\n+//! - binds the fused polynomials along the Stage-2 challenges,\n+//! - appends the resulting claims as virtual openings under `SumcheckId::GtExpClaimReduction`.\n+
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
    zkvm::recursion::constraints::config::CONFIG,
    zkvm::recursion::constraints::system::{ConstraintLocator, ConstraintType},
    zkvm::recursion::gt::indexing::{
        gt_exp_c_tail_range, k_exp, k_gt, num_gt_exp_constraints_padded,
    },
    zkvm::witness::VirtualPolynomial,
};

use allocative::Allocative;
use ark_bn254::Fq;
use ark_ff::Zero;

#[derive(Clone, Debug, Allocative)]
pub struct FusedGtExpStage2OpeningsParams {
    pub num_rounds: usize, // k_common + 11
    pub k_common: usize,
    pub k_exp: usize,
}

impl FusedGtExpStage2OpeningsParams {
    pub fn from_constraint_types(constraint_types: &[ConstraintType]) -> Self {
        let k_common = k_gt(constraint_types);
        let k_exp = k_exp(constraint_types);
        Self {
            num_rounds: k_common + CONFIG.packed_vars,
            k_common,
            k_exp,
        }
    }
}

pub struct FusedGtExpStage2OpeningsProver<T: Transcript> {
    params: FusedGtExpStage2OpeningsParams,
    rho: MultilinearPolynomial<Fq>,
    quotient: MultilinearPolynomial<Fq>,
    _marker: core::marker::PhantomData<T>,
}

#[cfg(feature = "allocative")]
impl<T: Transcript> allocative::Allocative for FusedGtExpStage2OpeningsProver<T> {
    fn visit<'a, 'b: 'a>(&self, _visitor: &'a mut allocative::Visitor<'b>) {}
}

impl<T: Transcript> FusedGtExpStage2OpeningsProver<T> {
    pub fn new(
        constraint_types: &[ConstraintType],
        locator_by_constraint: &[ConstraintLocator],
        gt_exp_witnesses: &[crate::zkvm::recursion::gt::exponentiation::GtExpWitness<Fq>],
    ) -> Self {
        let params = FusedGtExpStage2OpeningsParams::from_constraint_types(constraint_types);

        let row_size = 1usize << CONFIG.packed_vars;
        let num_gt_constraints_padded = num_gt_exp_constraints_padded(constraint_types);

        let mut rho_xc = vec![Fq::zero(); num_gt_constraints_padded * row_size];
        let mut quotient_xc = vec![Fq::zero(); num_gt_constraints_padded * row_size];

        for global_idx in 0..constraint_types.len() {
            if let ConstraintLocator::GtExp { local } = locator_by_constraint[global_idx] {
                let rho_src = &gt_exp_witnesses[local].rho_packed;
                let quo_src = &gt_exp_witnesses[local].quotient_packed;
                debug_assert_eq!(rho_src.len(), row_size);
                debug_assert_eq!(quo_src.len(), row_size);
                let off = local * row_size;
                rho_xc[off..off + row_size].copy_from_slice(rho_src);
                quotient_xc[off..off + row_size].copy_from_slice(quo_src);
            }
        }

        Self {
            params,
            // Store in [x11 low bits, c_gt high bits] order so `c_gt` is a suffix in Stage 2.
            rho: MultilinearPolynomial::LargeScalars(DensePolynomial::new(rho_xc)),
            quotient: MultilinearPolynomial::LargeScalars(DensePolynomial::new(quotient_xc)),
            _marker: core::marker::PhantomData,
        }
    }
}

impl<T: Transcript> SumcheckInstanceProver<Fq, T> for FusedGtExpStage2OpeningsProver<T> {
    fn degree(&self) -> usize {
        1
    }
    fn num_rounds(&self) -> usize {
        self.params.num_rounds
    }
    fn input_claim(&self, _acc: &ProverOpeningAccumulator<Fq>) -> Fq {
        Fq::zero()
    }
    fn compute_message(&mut self, _round: usize, _previous_claim: Fq) -> UniPoly<Fq> {
        UniPoly::from_coeff(vec![Fq::zero()])
    }

    fn ingest_challenge(&mut self, r_j: <Fq as JoltField>::Challenge, round: usize) {
        // This instance participates with `num_rounds = 11 + k_common` so its (s,u) challenges
        // align with other GT instances in Stage 2, but the committed fused rows use only `k_exp`.
        //
        // We therefore bind:
        // - all 11 x-bits, and
        // - only the tail `k_exp` bits of the c-suffix (skip the first k_common-k_exp c-bits).
        let x_vars = CONFIG.packed_vars; // 11
        if round < x_vars {
            self.rho.bind_parallel(r_j, BindingOrder::LowToHigh);
            self.quotient.bind_parallel(r_j, BindingOrder::LowToHigh);
            return;
        }
        let c_round = round - x_vars;
        let dummy = self.params.k_common.saturating_sub(self.params.k_exp);
        if c_round < dummy {
            return;
        }
        self.rho.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.quotient.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<Fq>,
        transcript: &mut T,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        // Opening point must match the committed fused row arity: (s,u,c_exp_tail).
        debug_assert_eq!(sumcheck_challenges.len(), self.params.num_rounds);
        let x_vars = CONFIG.packed_vars; // 11
        let mut r = Vec::with_capacity(x_vars + self.params.k_exp);
        r.extend_from_slice(&sumcheck_challenges[..x_vars]);
        let tail = gt_exp_c_tail_range(self.params.k_common, self.params.k_exp);
        r.extend_from_slice(&sumcheck_challenges[tail]);
        let opening_point = OpeningPoint::<BIG_ENDIAN, Fq>::new(r);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::gt_exp_rho_fused(),
            SumcheckId::GtExpClaimReduction,
            opening_point.clone(),
            self.rho.get_bound_coeff(0),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::gt_exp_quotient_fused(),
            SumcheckId::GtExpClaimReduction,
            opening_point,
            self.quotient.get_bound_coeff(0),
        );
    }
}

#[derive(Allocative)]
pub struct FusedGtExpStage2OpeningsVerifier {
    params: FusedGtExpStage2OpeningsParams,
}

impl FusedGtExpStage2OpeningsVerifier {
    pub fn new(constraint_types: &[ConstraintType]) -> Self {
        Self {
            params: FusedGtExpStage2OpeningsParams::from_constraint_types(constraint_types),
        }
    }
}

impl<T: Transcript> SumcheckInstanceVerifier<Fq, T> for FusedGtExpStage2OpeningsVerifier {
    fn degree(&self) -> usize {
        1
    }
    fn num_rounds(&self) -> usize {
        self.params.num_rounds
    }
    fn input_claim(&self, _acc: &VerifierOpeningAccumulator<Fq>) -> Fq {
        Fq::zero()
    }
    fn expected_output_claim(
        &self,
        _acc: &VerifierOpeningAccumulator<Fq>,
        _sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) -> Fq {
        Fq::zero()
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<Fq>,
        transcript: &mut T,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        // Opening point must match the committed fused row arity: (s,u,c_exp_tail).
        debug_assert_eq!(sumcheck_challenges.len(), self.params.num_rounds);
        let x_vars = CONFIG.packed_vars; // 11
        let mut r = Vec::with_capacity(x_vars + self.params.k_exp);
        r.extend_from_slice(&sumcheck_challenges[..x_vars]);
        let tail = gt_exp_c_tail_range(self.params.k_common, self.params.k_exp);
        r.extend_from_slice(&sumcheck_challenges[tail]);
        let opening_point = OpeningPoint::<BIG_ENDIAN, Fq>::new(r);
        for vp in [
            VirtualPolynomial::gt_exp_rho_fused(),
            VirtualPolynomial::gt_exp_quotient_fused(),
        ] {
            accumulator.append_virtual(
                transcript,
                vp,
                SumcheckId::GtExpClaimReduction,
                opening_point.clone(),
            );
        }
    }
}

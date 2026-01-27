//! Cache-only Stage-2 GTExp fused openings at the Stage-2 point.
//!
//! End-to-end GT fusion needs `gt_exp_{rho,quotient}_fused()` available at the full Stage-2
//! point `r_stage2 = (r_c_gt, r_x)` so that:\n//! - Stage-3 prefix packing can consume fused GT claims, and\n//! - fused GT shift / wiring can consume fused GT values without per-instance openings.\n+//!\n+//! This is implemented as a no-op (zero) sumcheck instance that:\n+//! - participates in Stage-2 batching only to obtain the Stage-2 point,\n+//! - binds the fused polynomials along the Stage-2 challenges,\n+//! - appends the resulting claims as virtual openings under `SumcheckId::GtExpClaimReduction`.\n+
use crate::{
    field::JoltField,
    poly::{
        dense_mlpoly::DensePolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN},
        unipoly::UniPoly,
    },
    subprotocols::{sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier},
    transcripts::Transcript,
    zkvm::recursion::constraints::config::CONFIG,
    zkvm::recursion::constraints::system::{ConstraintLocator, ConstraintType},
    zkvm::recursion::gt::indexing::{gt_constraint_indices, k_gt, num_gt_constraints_padded},
    zkvm::witness::VirtualPolynomial,
};

use allocative::Allocative;
use ark_bn254::Fq;
use ark_ff::Zero;

#[inline]
fn transpose_xc_to_cx(input: &[Fq], num_constraints_padded: usize, row_size: usize) -> Vec<Fq> {
    debug_assert_eq!(input.len(), num_constraints_padded * row_size);
    let mut out = vec![Fq::zero(); input.len()];
    for c in 0..num_constraints_padded {
        let row_off = c * row_size;
        for x in 0..row_size {
            out[x * num_constraints_padded + c] = input[row_off + x];
        }
    }
    out
}

#[derive(Clone, Debug, Allocative)]
pub struct FusedGtExpStage2OpeningsParams {
    pub num_rounds: usize, // k_gt + 11
}

impl FusedGtExpStage2OpeningsParams {
    pub fn from_constraint_types(constraint_types: &[ConstraintType]) -> Self {
        let num_c_vars = k_gt(constraint_types);
        Self {
            num_rounds: num_c_vars + CONFIG.packed_vars,
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
        let num_gt_constraints_padded = num_gt_constraints_padded(constraint_types);

        let gt_globals = gt_constraint_indices(constraint_types);

        let mut rho_xc = vec![Fq::zero(); num_gt_constraints_padded * row_size];
        let mut quotient_xc = vec![Fq::zero(); num_gt_constraints_padded * row_size];

        for (c_gt, &global_idx) in gt_globals.iter().enumerate() {
            if let ConstraintLocator::GtExp { local } = locator_by_constraint[global_idx] {
                let rho_src = &gt_exp_witnesses[local].rho_packed;
                let quo_src = &gt_exp_witnesses[local].quotient_packed;
                debug_assert_eq!(rho_src.len(), row_size);
                debug_assert_eq!(quo_src.len(), row_size);
                let off = c_gt * row_size;
                rho_xc[off..off + row_size].copy_from_slice(rho_src);
                quotient_xc[off..off + row_size].copy_from_slice(quo_src);
            }
        }

        let rho_cx = transpose_xc_to_cx(&rho_xc, num_gt_constraints_padded, row_size);
        let quotient_cx = transpose_xc_to_cx(&quotient_xc, num_gt_constraints_padded, row_size);

        Self {
            params,
            rho: MultilinearPolynomial::LargeScalars(DensePolynomial::new(rho_cx)),
            quotient: MultilinearPolynomial::LargeScalars(DensePolynomial::new(quotient_cx)),
            _marker: core::marker::PhantomData,
        }
    }
}

impl<T: Transcript> SumcheckInstanceProver<Fq, T> for FusedGtExpStage2OpeningsProver<T> {
    fn degree(&self) -> usize { 1 }
    fn num_rounds(&self) -> usize { self.params.num_rounds }
    fn input_claim(&self, _acc: &ProverOpeningAccumulator<Fq>) -> Fq { Fq::zero() }
    fn compute_message(&mut self, _round: usize, _previous_claim: Fq) -> UniPoly<Fq> {
        UniPoly::from_coeff(vec![Fq::zero()])
    }

    fn ingest_challenge(&mut self, r_j: <Fq as JoltField>::Challenge, _round: usize) {
        self.rho.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.quotient.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<Fq>,
        transcript: &mut T,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        let opening_point = OpeningPoint::<BIG_ENDIAN, Fq>::new(sumcheck_challenges.to_vec());
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
    fn degree(&self) -> usize { 1 }
    fn num_rounds(&self) -> usize { self.params.num_rounds }
    fn input_claim(&self, _acc: &VerifierOpeningAccumulator<Fq>) -> Fq { Fq::zero() }
    fn expected_output_claim(
        &self,
        _acc: &VerifierOpeningAccumulator<Fq>,
        _sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) -> Fq { Fq::zero() }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<Fq>,
        transcript: &mut T,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        let opening_point = OpeningPoint::<BIG_ENDIAN, Fq>::new(sumcheck_challenges.to_vec());
        for vp in [
            VirtualPolynomial::gt_exp_rho_fused(),
            VirtualPolynomial::gt_exp_quotient_fused(),
        ] {
            accumulator.append_virtual(transcript, vp, SumcheckId::GtExpClaimReduction, opening_point.clone());
        }
    }
}


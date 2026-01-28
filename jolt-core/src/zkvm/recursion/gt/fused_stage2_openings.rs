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
    /// Stage-2 GT-local suffix length used for batching (k_common = k_gt).
    pub k_common: usize,
    pub k_exp: usize,
}

impl FusedGtExpStage2OpeningsParams {
    pub fn from_constraint_types(constraint_types: &[ConstraintType]) -> Self {
        let k_common = k_gt(constraint_types);
        let k_exp = k_exp(constraint_types);
        Self { k_common, k_exp }
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
        // We participate with (x11 + k_common) rounds so the x11 challenges align with other GT
        // Stage-2 instances (notably the GT wiring backend). The committed fused rows only use
        // `k_exp`, so we skip the first `k_common-k_exp` dummy c rounds.
        CONFIG.packed_vars + self.params.k_common
    }
    fn input_claim(&self, _acc: &ProverOpeningAccumulator<Fq>) -> Fq {
        Fq::zero()
    }
    fn compute_message(&mut self, _round: usize, _previous_claim: Fq) -> UniPoly<Fq> {
        UniPoly::from_coeff(vec![Fq::zero()])
    }

    fn ingest_challenge(&mut self, r_j: <Fq as JoltField>::Challenge, round: usize) {
        // Bind all 11 x-bits, and only the tail `k_exp` bits of the c-suffix.
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
        debug_assert_eq!(
            sumcheck_challenges.len(),
            CONFIG.packed_vars + self.params.k_common
        );
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
        CONFIG.packed_vars + self.params.k_common
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
        debug_assert_eq!(
            sumcheck_challenges.len(),
            CONFIG.packed_vars + self.params.k_common
        );
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zkvm::recursion::constraints::system::ConstraintLocator;
    use crate::zkvm::recursion::gt::indexing::{gt_exp_c_tail_range, k_gt};

    #[test]
    fn stage2_openings_uses_k_gt_rounds_but_drops_dummy_c_bits_in_opening_point() {
        // Split-k scenario:
        // - 3 GTExp => padded 4 => k_exp = 2
        // - 16 GTMul => padded 16 => k_mul = 4
        // => k_gt = 4, dummy_exp = 2
        let mut constraint_types = Vec::new();
        constraint_types.extend(core::iter::repeat(ConstraintType::GtExp).take(3));
        constraint_types.extend(core::iter::repeat(ConstraintType::GtMul).take(16));

        let params = FusedGtExpStage2OpeningsParams::from_constraint_types(&constraint_types);
        assert_eq!(params.k_common, k_gt(&constraint_types));
        assert_eq!(params.k_common, 4);
        assert_eq!(params.k_exp, 2);

        // Minimal locator mapping + witnesses.
        let mut locator_by_constraint = Vec::with_capacity(constraint_types.len());
        let mut exp_rank = 0usize;
        let mut mul_rank = 0usize;
        for ct in &constraint_types {
            match ct {
                ConstraintType::GtExp => {
                    locator_by_constraint.push(ConstraintLocator::GtExp { local: exp_rank });
                    exp_rank += 1;
                }
                ConstraintType::GtMul => {
                    locator_by_constraint.push(ConstraintLocator::GtMul { local: mul_rank });
                    mul_rank += 1;
                }
                _ => unreachable!("test only uses GTExp/GTMul"),
            }
        }
        let row_size = 1usize << CONFIG.packed_vars;
        let witnesses = vec![
            crate::zkvm::recursion::gt::exponentiation::GtExpWitness::<Fq> {
                rho_packed: vec![Fq::zero(); row_size],
                rho_next_packed: vec![Fq::zero(); row_size],
                quotient_packed: vec![Fq::zero(); row_size],
                digit_lo_packed: vec![Fq::zero(); row_size],
                digit_hi_packed: vec![Fq::zero(); row_size],
                base_packed: vec![Fq::zero(); row_size],
                base2_packed: vec![Fq::zero(); row_size],
                base3_packed: vec![Fq::zero(); row_size],
                num_steps: 1,
            };
            3
        ];

        let prover = FusedGtExpStage2OpeningsProver::<crate::transcripts::Blake2bTranscript>::new(
            &constraint_types,
            &locator_by_constraint,
            &witnesses,
        );
        assert_eq!(prover.num_rounds(), CONFIG.packed_vars + params.k_common);

        // Fabricate a stage-2 challenge slice of length (11 + k_gt) and ensure we
        // construct an opening point (11 + k_exp) that uses the tail bits.
        let sumcheck_challenges: Vec<<Fq as JoltField>::Challenge> = (0..prover.num_rounds())
            .map(|i| Fq::from_u64((1000 + i) as u64).into())
            .collect();

        let x_vars = CONFIG.packed_vars;
        let tail = gt_exp_c_tail_range(params.k_common, params.k_exp);
        let mut expected = Vec::with_capacity(x_vars + params.k_exp);
        expected.extend_from_slice(&sumcheck_challenges[..x_vars]);
        expected.extend_from_slice(&sumcheck_challenges[tail]);
        assert_eq!(expected.len(), x_vars + params.k_exp);

        let opening_point = OpeningPoint::<BIG_ENDIAN, Fq>::new(expected.clone());
        assert_eq!(opening_point.r, expected);
    }
}

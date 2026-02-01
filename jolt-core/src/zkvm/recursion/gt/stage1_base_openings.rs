//! Cache-only Stage-1 GTExp base openings at the Stage-1 point.
//!
//! Stage 1 runs the packed GTExp sumcheck over `(s,u,c_gt)` (11 + k_gt rounds).
//! The committed base rows live over the native `(u, c_exp)` domain (4 + k_exp vars), so this
//! instance:
//! - participates in Stage 1 with **11 + k_gt rounds** (to share the same batched point),
//! - ignores the 7 step rounds `s` and the dummy c bits (split-k),
//! - binds only the 4 `u` rounds and the last `k_exp` c rounds,
//! - caches `GtExpTerm::{Base,Base2,Base3}` at the normalized point `(u, c_exp_tail)` under
//!   `SumcheckId::GtExpBaseStage1Openings`.

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
    zkvm::recursion::gt::types::GtExpWitness,
    zkvm::witness::{GtExpTerm, RecursionPoly, VirtualPolynomial},
};

use allocative::Allocative;
use ark_bn254::Fq;
use ark_ff::Zero;

const U_VARS: usize = 4;
const STEP_STRIDE: usize = 1usize << 7; // 2^STEP_VARS (STEP_VARS = 7)

#[derive(Clone, Debug, Allocative)]
pub struct GtExpBaseStage1OpeningsParams {
    pub k_common: usize, // k_gt
    pub k_exp: usize,
}

impl GtExpBaseStage1OpeningsParams {
    pub fn from_constraint_types(constraint_types: &[ConstraintType]) -> Self {
        let k_common = k_gt(constraint_types);
        let k_exp = k_exp(constraint_types);
        Self { k_common, k_exp }
    }
}

pub struct GtExpBaseStage1OpeningsProver<T: Transcript> {
    params: GtExpBaseStage1OpeningsParams,
    base: MultilinearPolynomial<Fq>,
    base2: MultilinearPolynomial<Fq>,
    base3: MultilinearPolynomial<Fq>,
    _marker: core::marker::PhantomData<T>,
}

#[cfg(feature = "allocative")]
impl<T: Transcript> allocative::Allocative for GtExpBaseStage1OpeningsProver<T> {
    fn visit<'a, 'b: 'a>(&self, _visitor: &'a mut allocative::Visitor<'b>) {}
}

impl<T: Transcript> GtExpBaseStage1OpeningsProver<T> {
    pub fn new(
        constraint_types: &[ConstraintType],
        locator_by_constraint: &[ConstraintLocator],
        gt_exp_witnesses: &[GtExpWitness<Fq>],
    ) -> Self {
        let params = GtExpBaseStage1OpeningsParams::from_constraint_types(constraint_types);
        let row_size = 1usize << U_VARS; // 16
        let num_gt_exp_padded = num_gt_exp_constraints_padded(constraint_types);

        let mut base_uc = vec![Fq::zero(); num_gt_exp_padded * row_size];
        let mut base2_uc = vec![Fq::zero(); num_gt_exp_padded * row_size];
        let mut base3_uc = vec![Fq::zero(); num_gt_exp_padded * row_size];

        for global_idx in 0..constraint_types.len() {
            if let ConstraintLocator::GtExp { local } = locator_by_constraint[global_idx] {
                let base11 = &gt_exp_witnesses[local].base_packed;
                let base211 = &gt_exp_witnesses[local].base2_packed;
                let base311 = &gt_exp_witnesses[local].base3_packed;
                debug_assert_eq!(base11.len(), 1usize << CONFIG.packed_vars);
                debug_assert_eq!(base211.len(), 1usize << CONFIG.packed_vars);
                debug_assert_eq!(base311.len(), 1usize << CONFIG.packed_vars);

                let off = local * row_size;
                for u in 0..row_size {
                    // Extract s=0 slice (base is replicated across s in the packed witness).
                    base_uc[off + u] = base11[u * STEP_STRIDE];
                    base2_uc[off + u] = base211[u * STEP_STRIDE];
                    base3_uc[off + u] = base311[u * STEP_STRIDE];
                }
            }
        }

        Self {
            params,
            // Store in [u low bits, c_exp high bits] order.
            base: MultilinearPolynomial::LargeScalars(DensePolynomial::new(base_uc)),
            base2: MultilinearPolynomial::LargeScalars(DensePolynomial::new(base2_uc)),
            base3: MultilinearPolynomial::LargeScalars(DensePolynomial::new(base3_uc)),
            _marker: core::marker::PhantomData,
        }
    }
}

impl<T: Transcript> SumcheckInstanceProver<Fq, T> for GtExpBaseStage1OpeningsProver<T> {
    fn degree(&self) -> usize {
        1
    }
    fn num_rounds(&self) -> usize {
        // Stage 1 point length: (s,u,c_gt) = 11 + k_gt.
        CONFIG.packed_vars + self.params.k_common
    }
    fn input_claim(&self, _acc: &ProverOpeningAccumulator<Fq>) -> Fq {
        Fq::zero()
    }
    fn compute_message(&mut self, _round: usize, _previous_claim: Fq) -> UniPoly<Fq> {
        UniPoly::from_coeff(vec![Fq::zero()])
    }

    fn ingest_challenge(&mut self, r_j: <Fq as JoltField>::Challenge, round: usize) {
        // Bind only u rounds and the tail k_exp c rounds.
        let u_start = CONFIG.step_vars; // 7
        let u_end = CONFIG.packed_vars; // 11
        if (u_start..u_end).contains(&round) {
            self.base.bind_parallel(r_j, BindingOrder::LowToHigh);
            self.base2.bind_parallel(r_j, BindingOrder::LowToHigh);
            self.base3.bind_parallel(r_j, BindingOrder::LowToHigh);
            return;
        }
        if round < CONFIG.packed_vars {
            // Step rounds: skip.
            return;
        }
        // c suffix rounds.
        let c_round = round - CONFIG.packed_vars;
        let dummy = self.params.k_common.saturating_sub(self.params.k_exp);
        if c_round < dummy {
            return;
        }
        self.base.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.base2.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.base3.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<Fq>,
        transcript: &mut T,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        debug_assert_eq!(
            sumcheck_challenges.len(),
            CONFIG.packed_vars + self.params.k_common
        );
        // Opening point must match committed row arity: (u, c_exp_tail).
        let mut r = Vec::with_capacity(U_VARS + self.params.k_exp);
        r.extend_from_slice(&sumcheck_challenges[CONFIG.step_vars..CONFIG.packed_vars]);
        let tail = gt_exp_c_tail_range(self.params.k_common, self.params.k_exp);
        r.extend_from_slice(&sumcheck_challenges[tail]);
        let opening_point = OpeningPoint::<BIG_ENDIAN, Fq>::new(r);

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Recursion(RecursionPoly::GtExp {
                term: GtExpTerm::Base,
            }),
            SumcheckId::GtExpBaseStage1Openings,
            opening_point.clone(),
            self.base.get_bound_coeff(0),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Recursion(RecursionPoly::GtExp {
                term: GtExpTerm::Base2,
            }),
            SumcheckId::GtExpBaseStage1Openings,
            opening_point.clone(),
            self.base2.get_bound_coeff(0),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Recursion(RecursionPoly::GtExp {
                term: GtExpTerm::Base3,
            }),
            SumcheckId::GtExpBaseStage1Openings,
            opening_point,
            self.base3.get_bound_coeff(0),
        );
    }
}

#[derive(Allocative)]
pub struct GtExpBaseStage1OpeningsVerifier {
    params: GtExpBaseStage1OpeningsParams,
}

impl GtExpBaseStage1OpeningsVerifier {
    pub fn new(constraint_types: &[ConstraintType]) -> Self {
        Self {
            params: GtExpBaseStage1OpeningsParams::from_constraint_types(constraint_types),
        }
    }
}

impl<T: Transcript> SumcheckInstanceVerifier<Fq, T> for GtExpBaseStage1OpeningsVerifier {
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
        debug_assert_eq!(
            sumcheck_challenges.len(),
            CONFIG.packed_vars + self.params.k_common
        );
        let mut r = Vec::with_capacity(U_VARS + self.params.k_exp);
        r.extend_from_slice(&sumcheck_challenges[CONFIG.step_vars..CONFIG.packed_vars]);
        let tail = gt_exp_c_tail_range(self.params.k_common, self.params.k_exp);
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
        ] {
            accumulator.append_virtual(
                transcript,
                vp,
                SumcheckId::GtExpBaseStage1Openings,
                opening_point.clone(),
            );
        }
    }
}

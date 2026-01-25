//! Fused G1 addition sumcheck (over global constraint index + x variables).
//!
//! This is a transitional implementation used to introduce a global `r_c` prefix
//! (constraint-index challenges) in Stage 2, while keeping `r_x` as the last 11
//! challenges (constraint variables) as usual.
//!
//! Variable order for this sumcheck instance (round order, `BindingOrder::LowToHigh`):
//! - first `k = log2(num_constraints_padded)` rounds bind the global constraint index `c` (LSB first)
//! - last 11 rounds bind the matrix x-variables `x` (LSB first)
//!
//! The witness polynomials for each term are derived from the recursion matrix rows
//! for the corresponding `PolyType::G1Add*` columns, interpreted as fused polynomials
//! `P_term(c, x)`; we gate the constraint by a public indicator `I_g1add(c)` so the
//! fused constraint is 0 on non-G1Add constraints.

use crate::{
    field::JoltField,
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    zkvm::witness::TermEnum,
    zkvm::{
        recursion::{
            constraints::system::{ConstraintType, PolyType},
            g1::addition::G1AddValues,
        },
        witness::{G1AddTerm, VirtualPolynomial},
    },
};

use allocative::Allocative;
use ark_bn254::Fq;
use ark_ff::{One, Zero};
use rayon::prelude::*;

/// Degree bound for the G1Add constraint polynomial (see `g1/addition.rs`).
const DEGREE_BOUND: usize = 6;

/// Transpose a `[x_low, c_high]` table into `[c_low, x_high]`.
///
/// Input layout: `in[c * row_size + x]` (x varies fastest within a constraint row)
/// Output layout: `out[x * num_constraints_padded + c]` (c varies fastest within an x-slice)
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

fn term_to_poly_type(term: G1AddTerm) -> PolyType {
    match term {
        G1AddTerm::XP => PolyType::G1AddXP,
        G1AddTerm::YP => PolyType::G1AddYP,
        G1AddTerm::PIndicator => PolyType::G1AddPIndicator,
        G1AddTerm::XQ => PolyType::G1AddXQ,
        G1AddTerm::YQ => PolyType::G1AddYQ,
        G1AddTerm::QIndicator => PolyType::G1AddQIndicator,
        G1AddTerm::XR => PolyType::G1AddXR,
        G1AddTerm::YR => PolyType::G1AddYR,
        G1AddTerm::RIndicator => PolyType::G1AddRIndicator,
        G1AddTerm::Lambda => PolyType::G1AddLambda,
        G1AddTerm::InvDeltaX => PolyType::G1AddInvDeltaX,
        G1AddTerm::IsDouble => PolyType::G1AddIsDouble,
        G1AddTerm::IsInverse => PolyType::G1AddIsInverse,
    }
}

#[derive(Clone, Allocative)]
pub struct FusedG1AddParams {
    pub num_constraint_index_vars: usize, // k
    pub num_constraint_vars: usize,       // 11
    pub num_constraints: usize,
    pub num_constraints_padded: usize,
}

impl FusedG1AddParams {
    pub fn new(
        num_constraints: usize,
        num_constraints_padded: usize,
        num_constraint_vars: usize,
    ) -> Self {
        debug_assert!(num_constraints_padded.is_power_of_two());
        let num_constraint_index_vars = num_constraints_padded.trailing_zeros() as usize;
        Self {
            num_constraint_index_vars,
            num_constraint_vars,
            num_constraints,
            num_constraints_padded,
        }
    }

    #[inline]
    pub fn num_rounds(&self) -> usize {
        self.num_constraint_index_vars + self.num_constraint_vars
    }
}

impl SumcheckInstanceParams<Fq> for FusedG1AddParams {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds()
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<Fq>) -> Fq {
        // Prove the (Eq-weighted) sum is 0.
        Fq::zero()
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<Fq as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, Fq> {
        // Follow the default recursion convention: treat the raw sumcheck challenges as the opening point.
        OpeningPoint::<BIG_ENDIAN, Fq>::new(challenges.to_vec())
    }
}

#[derive(Allocative)]
pub struct FusedG1AddProver {
    pub params: FusedG1AddParams,
    /// Eq polynomial evaluations for the sampled eq_point (over (c,x) domain).
    eq_poly: MultilinearPolynomial<Fq>,
    /// Public indicator polynomial I_g1add(c), embedded as a fused (c,x) MLE constant in x.
    indicator_poly: MultilinearPolynomial<Fq>,
    /// Fused witness polynomials for each G1Add term, interpreted as P_term(c,x).
    term_polys: Vec<MultilinearPolynomial<Fq>>,
    /// Term batching coefficient (δ) for combining constraint terms.
    term_batch_coeff: Fq,
}

impl FusedG1AddProver {
    pub fn new<T: Transcript>(
        matrix_evals: &[Fq],
        constraint_types: &[ConstraintType],
        params: FusedG1AddParams,
        transcript: &mut T,
    ) -> Self {
        let num_rounds = params.num_rounds();
        let row_size = 1usize << params.num_constraint_vars;

        // Sample eq_point for the fused (c,x) domain.
        let eq_point: Vec<<Fq as JoltField>::Challenge> = (0..num_rounds)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();
        let eq_poly = MultilinearPolynomial::from(EqPolynomial::<Fq>::evals(&eq_point));

        // Sample δ for term batching.
        let term_batch_coeff: Fq = transcript.challenge_scalar_optimized::<Fq>().into();

        // Build indicator table in [x_low, c_high] layout, then transpose to [c_low, x_high].
        let mut ind_xc = vec![Fq::zero(); params.num_constraints_padded * row_size];
        for c in 0..params.num_constraints_padded {
            let is_g1add =
                c < constraint_types.len() && matches!(constraint_types[c], ConstraintType::G1Add);
            if is_g1add {
                let off = c * row_size;
                for x in 0..row_size {
                    ind_xc[off + x] = Fq::one();
                }
            }
        }
        let ind_cx = transpose_xc_to_cx(&ind_xc, params.num_constraints_padded, row_size);
        let indicator_poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(ind_cx));

        // Build fused term polynomials by slicing the matrix by PolyType block, then transposing.
        let block_size = params.num_constraints_padded * row_size;
        let mut term_polys = Vec::with_capacity(G1AddTerm::COUNT);
        for term_idx in 0..G1AddTerm::COUNT {
            let term = G1AddTerm::from_index(term_idx).expect("invalid G1AddTerm index");
            let poly_type = term_to_poly_type(term) as usize;
            let start = poly_type * block_size;
            let end = start + block_size;
            let block_xc = &matrix_evals[start..end];
            let block_cx = transpose_xc_to_cx(block_xc, params.num_constraints_padded, row_size);
            term_polys.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                block_cx,
            )));
        }

        Self {
            params,
            eq_poly,
            indicator_poly,
            term_polys,
            term_batch_coeff,
        }
    }
}

impl<FqT: Transcript> SumcheckInstanceProver<Fq, FqT> for FusedG1AddProver {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fq> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: Fq) -> UniPoly<Fq> {
        let num_remaining = self.eq_poly.get_num_vars();
        debug_assert!(
            num_remaining > 0,
            "fused g1add should have at least one round"
        );
        let half = 1usize << (num_remaining - 1);

        let term_batch_coeff = self.term_batch_coeff;

        let total_evals: [Fq; DEGREE_BOUND] = (0..half)
            .into_par_iter()
            .map(|idx| {
                let eq_evals = self
                    .eq_poly
                    .sumcheck_evals_array::<DEGREE_BOUND>(idx, BindingOrder::LowToHigh);
                let ind_evals = self
                    .indicator_poly
                    .sumcheck_evals_array::<DEGREE_BOUND>(idx, BindingOrder::LowToHigh);

                // Collect per-term eval arrays once per idx.
                let mut poly_evals = vec![[Fq::zero(); DEGREE_BOUND]; self.term_polys.len()];
                for (t, poly) in self.term_polys.iter().enumerate() {
                    poly_evals[t] =
                        poly.sumcheck_evals_array::<DEGREE_BOUND>(idx, BindingOrder::LowToHigh);
                }

                let mut out = [Fq::zero(); DEGREE_BOUND];
                for eval_index in 0..DEGREE_BOUND {
                    let vals = G1AddValues::from_poly_evals(&poly_evals, eval_index);
                    let c_val = vals.eval_constraint(term_batch_coeff);
                    out[eval_index] = eq_evals[eval_index] * ind_evals[eval_index] * c_val;
                }
                out
            })
            .reduce(
                || [Fq::zero(); DEGREE_BOUND],
                |mut acc, arr| {
                    for i in 0..DEGREE_BOUND {
                        acc[i] += arr[i];
                    }
                    acc
                },
            );

        UniPoly::from_evals_and_hint(previous_claim, &total_evals)
    }

    fn ingest_challenge(&mut self, r_j: <Fq as JoltField>::Challenge, _round: usize) {
        self.eq_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.indicator_poly
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        for poly in self.term_polys.iter_mut() {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<Fq>,
        transcript: &mut FqT,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);

        for term_idx in 0..G1AddTerm::COUNT {
            let term = G1AddTerm::from_index(term_idx).expect("invalid G1AddTerm index");
            let claim = self.term_polys[term_idx].get_bound_coeff(0);
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::g1_add_fused(term),
                SumcheckId::G1Add,
                opening_point.clone(),
                claim,
            );
        }
    }
}

#[derive(Allocative)]
pub struct FusedG1AddVerifier {
    params: FusedG1AddParams,
    eq_point: Vec<<Fq as JoltField>::Challenge>,
    term_batch_coeff: Fq,
    /// Global constraint indices where `ConstraintType::G1Add` holds.
    g1add_indices: Vec<usize>,
}

impl FusedG1AddVerifier {
    pub fn new<T: Transcript>(
        params: FusedG1AddParams,
        constraint_types: &[ConstraintType],
        transcript: &mut T,
    ) -> Self {
        let num_rounds = params.num_rounds();
        let eq_point: Vec<<Fq as JoltField>::Challenge> = (0..num_rounds)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();
        let term_batch_coeff: Fq = transcript.challenge_scalar_optimized::<Fq>().into();
        let g1add_indices: Vec<usize> = constraint_types
            .iter()
            .enumerate()
            .filter_map(|(i, ct)| matches!(ct, ConstraintType::G1Add).then_some(i))
            .collect();
        Self {
            params,
            eq_point,
            term_batch_coeff,
            g1add_indices,
        }
    }
}

impl<T: Transcript> SumcheckInstanceVerifier<Fq, T> for FusedG1AddVerifier {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fq> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<Fq>,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) -> Fq {
        // Follow the common convention used by existing constraint-list verifier:
        // reverse challenges to form the evaluation point used by EqPolynomial::mle.
        let eval_point: Vec<Fq> = sumcheck_challenges
            .iter()
            .rev()
            .map(|c| (*c).into())
            .collect();
        let eq_point_f: Vec<Fq> = self.eq_point.iter().map(|c| (*c).into()).collect();
        let eq_eval = EqPolynomial::mle(&eq_point_f, &eval_point);

        // Compute I_g1add(r_c) as Σ_{i in g1add_indices} Eq(r_c, i).
        // We treat the first k sumcheck challenges as the `c` variables in *round order*
        // (LSB-first), so we use `index_to_binary` (little-endian) for the indices.
        let k = self.params.num_constraint_index_vars;
        let r_c: Vec<Fq> = sumcheck_challenges
            .iter()
            .take(k)
            .map(|c| (*c).into())
            .collect();
        let mut ind_eval = Fq::zero();
        for &idx in &self.g1add_indices {
            let bits = crate::zkvm::recursion::constraints::system::index_to_binary::<Fq>(idx, k);
            ind_eval += EqPolynomial::mle(&r_c, &bits);
        }

        // Fetch fused opened claims (one per term).
        let mut claims = Vec::with_capacity(G1AddTerm::COUNT);
        for term_idx in 0..G1AddTerm::COUNT {
            let term = G1AddTerm::from_index(term_idx).expect("invalid G1AddTerm index");
            let (_, claim) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::g1_add_fused(term),
                SumcheckId::G1Add,
            );
            claims.push(claim);
        }

        let vals = G1AddValues::from_claims(&claims);
        let constraint_value = vals.eval_constraint(self.term_batch_coeff);

        eq_eval * ind_eval * constraint_value
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<Fq>,
        transcript: &mut T,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        for term_idx in 0..G1AddTerm::COUNT {
            let term = G1AddTerm::from_index(term_idx).expect("invalid G1AddTerm index");
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::g1_add_fused(term),
                SumcheckId::G1Add,
                opening_point.clone(),
            );
        }
    }
}

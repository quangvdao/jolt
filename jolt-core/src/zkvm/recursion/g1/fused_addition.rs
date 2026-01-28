//! Fused G1 addition sumcheck (family-local, Option B).
//!
//! This is the intended "Option B" fusion style for G1Add:
//! - We fuse the family over a **family-local** constraint index `c_g1add`.
//! - Each committed G1Add term (XP, YP, ..., IsInverse) is treated as an MLE over `c_g1add`.
//! - Padding rows are gated by a public indicator `I_g1add(c)` so the fused constraint is 0
//!   outside the real constraint range.
//!
//! Variable order (round order, `BindingOrder::LowToHigh`):
//! - `k = log2(next_pow2(num_g1add).max(1))` rounds bind the family-local constraint index `c_g1add`
//!   (LSB first).

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
    zkvm::{
        recursion::{
            constraints::system::{index_to_binary, G1AddNative},
            g1::types::G1AddValues,
        },
        witness::{G1AddTerm, TermEnum, VirtualPolynomial},
    },
};

use allocative::Allocative;
use ark_bn254::Fq;
use ark_ff::{One, Zero};
use rayon::prelude::*;

/// Degree bound for the G1Add constraint polynomial (see `g1/addition.rs`).
const DEGREE_BOUND: usize = 6;

#[derive(Clone, Allocative)]
pub struct FusedG1AddParams {
    pub num_constraint_index_vars: usize, // k
    pub num_constraints: usize,
    pub num_constraints_padded: usize,
}

impl FusedG1AddParams {
    pub fn new(num_constraints: usize) -> Self {
        let num_constraints_padded = num_constraints.max(1).next_power_of_two();
        let num_constraint_index_vars = num_constraints_padded.trailing_zeros() as usize;
        Self {
            num_constraint_index_vars,
            num_constraints,
            num_constraints_padded,
        }
    }

    #[inline]
    pub fn num_rounds(&self) -> usize {
        self.num_constraint_index_vars
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
    pub fn new<T: Transcript>(g1_add_rows: &[G1AddNative], transcript: &mut T) -> Self {
        let params = FusedG1AddParams::new(g1_add_rows.len());
        let num_rounds = params.num_rounds();

        // Sample eq_point for the fused c_g1add domain.
        let eq_point: Vec<<Fq as JoltField>::Challenge> = (0..num_rounds)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();
        let eq_poly = MultilinearPolynomial::from(EqPolynomial::<Fq>::evals(&eq_point));

        // Sample δ for term batching.
        let term_batch_coeff: Fq = transcript.challenge_scalar_optimized::<Fq>().into();

        // Family-local indicator I_g1add(c): 1 on real constraints, 0 on padding.
        let mut ind = vec![Fq::zero(); params.num_constraints_padded];
        for c in 0..params.num_constraints {
            ind[c] = Fq::one();
        }
        let indicator_poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(ind));

        // Helper to extract a term scalar from a native G1Add row.
        let term_value = |row: &G1AddNative, term: G1AddTerm| -> Fq {
            match term {
                G1AddTerm::XP => row.x_p,
                G1AddTerm::YP => row.y_p,
                G1AddTerm::PIndicator => row.ind_p,
                G1AddTerm::XQ => row.x_q,
                G1AddTerm::YQ => row.y_q,
                G1AddTerm::QIndicator => row.ind_q,
                G1AddTerm::XR => row.x_r,
                G1AddTerm::YR => row.y_r,
                G1AddTerm::RIndicator => row.ind_r,
                G1AddTerm::Lambda => row.lambda,
                G1AddTerm::InvDeltaX => row.inv_delta_x,
                G1AddTerm::IsDouble => row.is_double,
                G1AddTerm::IsInverse => row.is_inverse,
            }
        };

        // Build fused term polynomials P_term(c) (one per term), padded to 2^k.
        let mut term_polys = Vec::with_capacity(G1AddTerm::COUNT);
        for term_idx in 0..G1AddTerm::COUNT {
            let term = G1AddTerm::from_index(term_idx).expect("invalid G1AddTerm index");
            let mut v = vec![Fq::zero(); params.num_constraints_padded];
            for (c, row) in g1_add_rows.iter().enumerate() {
                v[c] = term_value(row, term);
            }
            term_polys.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(v)));
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
    /// Number of (family-local) G1Add constraints.
    num_constraints: usize,
}

impl FusedG1AddVerifier {
    pub fn new<T: Transcript>(params: FusedG1AddParams, transcript: &mut T) -> Self {
        let num_rounds = params.num_rounds();
        let eq_point: Vec<<Fq as JoltField>::Challenge> = (0..num_rounds)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();
        let term_batch_coeff: Fq = transcript.challenge_scalar_optimized::<Fq>().into();
        Self {
            num_constraints: params.num_constraints,
            params,
            eq_point,
            term_batch_coeff,
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

        // Compute I_g1add(r_c) as Σ_{c < num_constraints} Eq(r_c, c).
        // We treat the first k sumcheck challenges as the `c` variables in *round order*
        // (LSB-first), so we use `index_to_binary` (little-endian) for the indices.
        let k = self.params.num_constraint_index_vars;
        let r_c: Vec<Fq> = sumcheck_challenges
            .iter()
            .take(k)
            .map(|c| (*c).into())
            .collect();
        let mut ind_eval = Fq::zero();
        for c in 0..self.num_constraints {
            let bits = index_to_binary::<Fq>(c, k);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcripts::Blake2bTranscript;

    #[test]
    fn fused_g1add_is_family_local_over_c_only() {
        let rows = vec![
            G1AddNative {
                x_p: Fq::from_u64(1),
                y_p: Fq::zero(),
                ind_p: Fq::zero(),
                x_q: Fq::zero(),
                y_q: Fq::zero(),
                ind_q: Fq::zero(),
                x_r: Fq::zero(),
                y_r: Fq::zero(),
                ind_r: Fq::zero(),
                lambda: Fq::zero(),
                inv_delta_x: Fq::zero(),
                is_double: Fq::zero(),
                is_inverse: Fq::zero(),
            },
            G1AddNative {
                x_p: Fq::from_u64(2),
                y_p: Fq::zero(),
                ind_p: Fq::zero(),
                x_q: Fq::zero(),
                y_q: Fq::zero(),
                ind_q: Fq::zero(),
                x_r: Fq::zero(),
                y_r: Fq::zero(),
                ind_r: Fq::zero(),
                lambda: Fq::zero(),
                inv_delta_x: Fq::zero(),
                is_double: Fq::zero(),
                is_inverse: Fq::zero(),
            },
            G1AddNative {
                x_p: Fq::from_u64(3),
                y_p: Fq::zero(),
                ind_p: Fq::zero(),
                x_q: Fq::zero(),
                y_q: Fq::zero(),
                ind_q: Fq::zero(),
                x_r: Fq::zero(),
                y_r: Fq::zero(),
                ind_r: Fq::zero(),
                lambda: Fq::zero(),
                inv_delta_x: Fq::zero(),
                is_double: Fq::zero(),
                is_inverse: Fq::zero(),
            },
        ];

        let mut transcript = Blake2bTranscript::new(b"test_fused_g1add_family_local");
        let prover = FusedG1AddProver::new(&rows, &mut transcript);
        assert_eq!(prover.params.num_constraints, 3);
        assert_eq!(prover.params.num_constraints_padded, 4);
        assert_eq!(prover.params.num_rounds(), 2);
        assert_eq!(prover.term_polys.len(), G1AddTerm::COUNT);

        // Term polynomials are c-only tables padded to 2^k.
        let xp_idx = G1AddTerm::XP.to_index();
        let MultilinearPolynomial::LargeScalars(xp_poly) = &prover.term_polys[xp_idx] else {
            panic!("expected LargeScalars term poly");
        };
        assert_eq!(xp_poly.Z.len(), prover.params.num_constraints_padded);
        assert_eq!(xp_poly.Z[0], Fq::from_u64(1));
        assert_eq!(xp_poly.Z[1], Fq::from_u64(2));
        assert_eq!(xp_poly.Z[2], Fq::from_u64(3));
        // padding row
        assert_eq!(xp_poly.Z[3], Fq::zero());
    }
}

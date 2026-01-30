//! G2 addition sumcheck
//!
//! Mirrors `g1/addition.rs`, but for G2 points over Fq2 split into (c0,c1) components over Fq.
//!
//! - We batch the family over a **family-local** constraint index `c_g2add`.
//! - Each committed G2Add term is treated as an MLE over `c_g2add`.
//! - Padding rows are gated by a public indicator `I_g2add(c)` so the constraint is 0
//!   outside the real constraint range.
//!
//! Variable order (round order, `BindingOrder::LowToHigh`):
//! - `k = log2(next_pow2(num_g2add).max(1))` rounds bind the family-local constraint index `c_g2add`
//!   (LSB first).
//!
//! ## Sumcheck relation
//!
//! **Input claim:** `0`.
//!
//! Let `r_c ∈ Fq^k` be the verifier-sampled `eq_point`, and let `eq(r_c, c)` denote the multilinear
//! equality polynomial over `c ∈ {0,1}^k` (LSB-first indexing).
//!
//! This sumcheck proves the following identity over `c ∈ {0,1}^k`:
//!
//! ```text
//! Σ_c eq(r_c, c) · I_g2add(c) · C_g2add(c; δ) = 0
//! ```
//!
//! where:
//! - `I_g2add(c) ∈ {0,1}` is the public indicator polynomial (1 on real constraints, 0 on padding),
//! - `δ` is the transcript-sampled term-batching coefficient (`term_batch_coeff`),
//! - `C_g2add(c; δ)` is the (batched) G2 addition constraint polynomial (over Fq, with Fq2 split into
//!   (c0,c1) components), evaluated in `G2AddValues::eval_constraint(...)`.

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
            constraints::system::{eq_lsb_index, G2AddNative},
            g2::types::G2AddValues,
        },
        witness::{G2AddTerm, RecursionPoly, TermEnum, VirtualPolynomial},
    },
};

use allocative::Allocative;
use ark_bn254::Fq;
use ark_ff::{One, Zero};
use rayon::prelude::*;

/// Degree bound for the G2Add constraint polynomial (see `g2/addition.rs`).
const DEGREE_BOUND: usize = 6;

#[derive(Clone, Allocative)]
pub struct G2AddParams {
    pub num_constraint_index_vars: usize, // k
    pub num_constraints: usize,
    pub num_constraints_padded: usize,
}

impl G2AddParams {
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

impl SumcheckInstanceParams<Fq> for G2AddParams {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds()
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<Fq>) -> Fq {
        Fq::zero()
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<Fq as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, Fq> {
        OpeningPoint::<BIG_ENDIAN, Fq>::new(challenges.to_vec())
    }
}

#[derive(Allocative)]
pub struct G2AddProver {
    pub params: G2AddParams,
    eq_poly: MultilinearPolynomial<Fq>,
    indicator_poly: MultilinearPolynomial<Fq>,
    term_polys: Vec<MultilinearPolynomial<Fq>>,
    term_batch_coeff: Fq,
}

impl G2AddProver {
    pub fn new<T: Transcript>(g2_add_rows: &[G2AddNative], transcript: &mut T) -> Self {
        let params = G2AddParams::new(g2_add_rows.len());
        let num_rounds = params.num_rounds();

        let eq_point: Vec<<Fq as JoltField>::Challenge> = (0..num_rounds)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();
        let eq_poly = MultilinearPolynomial::from(EqPolynomial::<Fq>::evals(&eq_point));

        let term_batch_coeff: Fq = transcript.challenge_scalar_optimized::<Fq>().into();

        // Family-local indicator I_g2add(c): 1 on real constraints, 0 on padding.
        let mut ind = vec![Fq::zero(); params.num_constraints_padded];
        for c in 0..params.num_constraints {
            ind[c] = Fq::one();
        }
        let indicator_poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(ind));

        // Helper to extract a term scalar from a native G2Add row.
        let term_value = |row: &G2AddNative, term: G2AddTerm| -> Fq {
            match term {
                G2AddTerm::XPC0 => row.x_p_c0,
                G2AddTerm::XPC1 => row.x_p_c1,
                G2AddTerm::YPC0 => row.y_p_c0,
                G2AddTerm::YPC1 => row.y_p_c1,
                G2AddTerm::PIndicator => row.ind_p,
                G2AddTerm::XQC0 => row.x_q_c0,
                G2AddTerm::XQC1 => row.x_q_c1,
                G2AddTerm::YQC0 => row.y_q_c0,
                G2AddTerm::YQC1 => row.y_q_c1,
                G2AddTerm::QIndicator => row.ind_q,
                G2AddTerm::XRC0 => row.x_r_c0,
                G2AddTerm::XRC1 => row.x_r_c1,
                G2AddTerm::YRC0 => row.y_r_c0,
                G2AddTerm::YRC1 => row.y_r_c1,
                G2AddTerm::RIndicator => row.ind_r,
                G2AddTerm::LambdaC0 => row.lambda_c0,
                G2AddTerm::LambdaC1 => row.lambda_c1,
                G2AddTerm::InvDeltaXC0 => row.inv_delta_x_c0,
                G2AddTerm::InvDeltaXC1 => row.inv_delta_x_c1,
                G2AddTerm::IsDouble => row.is_double,
                G2AddTerm::IsInverse => row.is_inverse,
            }
        };

        let mut term_polys = Vec::with_capacity(G2AddTerm::COUNT);
        for term_idx in 0..G2AddTerm::COUNT {
            let term = G2AddTerm::from_index(term_idx).expect("invalid G2AddTerm index");
            let mut v = vec![Fq::zero(); params.num_constraints_padded];
            for (c, row) in g2_add_rows.iter().enumerate() {
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

impl<FqT: Transcript> SumcheckInstanceProver<Fq, FqT> for G2AddProver {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fq> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: Fq) -> UniPoly<Fq> {
        let num_remaining = self.eq_poly.get_num_vars();
        debug_assert!(num_remaining > 0, "g2add should have at least one round");
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

                let mut poly_evals = vec![[Fq::zero(); DEGREE_BOUND]; self.term_polys.len()];
                for (t, poly) in self.term_polys.iter().enumerate() {
                    poly_evals[t] =
                        poly.sumcheck_evals_array::<DEGREE_BOUND>(idx, BindingOrder::LowToHigh);
                }

                let mut out = [Fq::zero(); DEGREE_BOUND];
                for eval_index in 0..DEGREE_BOUND {
                    let vals = G2AddValues::from_poly_evals(&poly_evals, eval_index);
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
        for term_idx in 0..G2AddTerm::COUNT {
            let term = G2AddTerm::from_index(term_idx).expect("invalid G2AddTerm index");
            let claim = self.term_polys[term_idx].get_bound_coeff(0);
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::Recursion(RecursionPoly::G2Add { term }),
                SumcheckId::G2Add,
                opening_point.clone(),
                claim,
            );
        }
    }
}

#[derive(Allocative)]
pub struct G2AddVerifier {
    params: G2AddParams,
    eq_point: Vec<<Fq as JoltField>::Challenge>,
    term_batch_coeff: Fq,
    num_constraints: usize,
}

impl G2AddVerifier {
    pub fn new<T: Transcript>(params: G2AddParams, transcript: &mut T) -> Self {
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

impl<T: Transcript> SumcheckInstanceVerifier<Fq, T> for G2AddVerifier {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fq> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<Fq>,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) -> Fq {
        let eval_point: Vec<Fq> = sumcheck_challenges
            .iter()
            .rev()
            .map(|c| (*c).into())
            .collect();
        let eq_point_f: Vec<Fq> = self.eq_point.iter().map(|c| (*c).into()).collect();
        let eq_eval = EqPolynomial::mle(&eq_point_f, &eval_point);

        // I_g2add(r_c) = Σ_{c < num_constraints} Eq(r_c, c).
        let k = self.params.num_constraint_index_vars;
        let r_c: Vec<Fq> = sumcheck_challenges
            .iter()
            .take(k)
            .map(|c| (*c).into())
            .collect();
        let mut ind_eval = Fq::zero();
        for c in 0..self.num_constraints {
            ind_eval += eq_lsb_index(&r_c, c);
        }

        // Fetch opened claims (one per term), without heap allocation.
        let vals = G2AddValues {
            x_p_c0: accumulator.get_virtual_polynomial_claim(
                VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                    term: G2AddTerm::XPC0,
                }),
                SumcheckId::G2Add,
            ),
            x_p_c1: accumulator.get_virtual_polynomial_claim(
                VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                    term: G2AddTerm::XPC1,
                }),
                SumcheckId::G2Add,
            ),
            y_p_c0: accumulator.get_virtual_polynomial_claim(
                VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                    term: G2AddTerm::YPC0,
                }),
                SumcheckId::G2Add,
            ),
            y_p_c1: accumulator.get_virtual_polynomial_claim(
                VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                    term: G2AddTerm::YPC1,
                }),
                SumcheckId::G2Add,
            ),
            ind_p: accumulator.get_virtual_polynomial_claim(
                VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                    term: G2AddTerm::PIndicator,
                }),
                SumcheckId::G2Add,
            ),
            x_q_c0: accumulator.get_virtual_polynomial_claim(
                VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                    term: G2AddTerm::XQC0,
                }),
                SumcheckId::G2Add,
            ),
            x_q_c1: accumulator.get_virtual_polynomial_claim(
                VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                    term: G2AddTerm::XQC1,
                }),
                SumcheckId::G2Add,
            ),
            y_q_c0: accumulator.get_virtual_polynomial_claim(
                VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                    term: G2AddTerm::YQC0,
                }),
                SumcheckId::G2Add,
            ),
            y_q_c1: accumulator.get_virtual_polynomial_claim(
                VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                    term: G2AddTerm::YQC1,
                }),
                SumcheckId::G2Add,
            ),
            ind_q: accumulator.get_virtual_polynomial_claim(
                VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                    term: G2AddTerm::QIndicator,
                }),
                SumcheckId::G2Add,
            ),
            x_r_c0: accumulator.get_virtual_polynomial_claim(
                VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                    term: G2AddTerm::XRC0,
                }),
                SumcheckId::G2Add,
            ),
            x_r_c1: accumulator.get_virtual_polynomial_claim(
                VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                    term: G2AddTerm::XRC1,
                }),
                SumcheckId::G2Add,
            ),
            y_r_c0: accumulator.get_virtual_polynomial_claim(
                VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                    term: G2AddTerm::YRC0,
                }),
                SumcheckId::G2Add,
            ),
            y_r_c1: accumulator.get_virtual_polynomial_claim(
                VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                    term: G2AddTerm::YRC1,
                }),
                SumcheckId::G2Add,
            ),
            ind_r: accumulator.get_virtual_polynomial_claim(
                VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                    term: G2AddTerm::RIndicator,
                }),
                SumcheckId::G2Add,
            ),
            lambda_c0: accumulator.get_virtual_polynomial_claim(
                VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                    term: G2AddTerm::LambdaC0,
                }),
                SumcheckId::G2Add,
            ),
            lambda_c1: accumulator.get_virtual_polynomial_claim(
                VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                    term: G2AddTerm::LambdaC1,
                }),
                SumcheckId::G2Add,
            ),
            inv_delta_x_c0: accumulator.get_virtual_polynomial_claim(
                VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                    term: G2AddTerm::InvDeltaXC0,
                }),
                SumcheckId::G2Add,
            ),
            inv_delta_x_c1: accumulator.get_virtual_polynomial_claim(
                VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                    term: G2AddTerm::InvDeltaXC1,
                }),
                SumcheckId::G2Add,
            ),
            is_double: accumulator.get_virtual_polynomial_claim(
                VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                    term: G2AddTerm::IsDouble,
                }),
                SumcheckId::G2Add,
            ),
            is_inverse: accumulator.get_virtual_polynomial_claim(
                VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                    term: G2AddTerm::IsInverse,
                }),
                SumcheckId::G2Add,
            ),
        };
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
        for term_idx in 0..G2AddTerm::COUNT {
            let term = G2AddTerm::from_index(term_idx).expect("invalid G2AddTerm index");
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::Recursion(RecursionPoly::G2Add { term }),
                SumcheckId::G2Add,
                opening_point.clone(),
            );
        }
    }
}

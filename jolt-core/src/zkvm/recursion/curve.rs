//! Curve- and pairing-specific hooks for recursion constraints.
//!
//! The recursion SNARK is *field-generic* in many places, but Stage 1 constraints perform
//! arithmetic over a concrete pairing tower (e.g. \( \mathbb{F}_q \subset \mathbb{F}_{q^2}
//! \subset \mathbb{F}_{q^{12}} \)).
//!
//! This module provides a `RecursionCurve` trait that:
//! - **Centralizes** curve/tower specifics (e.g. g(x) tower polynomial, GT→MLE expansion).
//! - Lets Stage 1 constraint code be generic over `C: RecursionCurve` instead of relying on
//!   `unsafe` transmutes between `F` and `Fq`.

use crate::field::JoltField;
use ark_ff::{Field, PrimeField};
use num_traits::Zero;

use jolt_optimizations::{fq12_to_multilinear_evals, get_g_mle};

/// A trait defining the curve-specific types and operations needed for the recursion SNARK.
///
/// The recursion SNARK operates over a base field `F`, but verifies constraints related to
/// elliptic curve operations (on G2 over Fq2) and pairing operations (on GT over Fq12).
/// This trait abstracts the relationship between the SNARK field `F` and the curve fields.
pub trait RecursionCurve: Send + Sync + 'static {
    /// Base field used *natively* by the recursion SNARK.
    ///
    /// In the current recursion design, this is also the base field \( \mathbb{F}_q \) of the
    /// pairing-friendly curve whose arithmetic we constrain.
    type Fq: JoltField;

    /// The scalar field of the curve (used for public inputs like scalars).
    type Fr: PrimeField;

    /// The quadratic extension field (used for G2 points).
    type Fq2: Field;

    /// The dodecic extension field (used for GT elements).
    type Fq12: Field;

    /// Number of variables in the multilinear-extension (MLE) representations used for:
    /// - the tower reduction polynomial g(x), and
    /// - GT elements expanded into base-field evaluations.
    ///
    /// For BN254, this is 4, yielding 16 evaluations.
    const TOWER_MLE_NUM_VARS: usize = 4;

    /// Number of evaluations in the MLE representation.
    #[inline(always)]
    fn tower_mle_evals_len() -> usize {
        1 << Self::TOWER_MLE_NUM_VARS
    }

    // --- Conversions ---

    /// Construct an `Fq2` element from its (c0, c1) base-field components.
    fn fq2_from_components(c0: Self::Fq, c1: Self::Fq) -> Self::Fq2;

    /// Split an `Fq2` element into its (c0, c1) base-field components.
    fn fq2_to_components(x: &Self::Fq2) -> (Self::Fq, Self::Fq);

    /// Embed a base-field element into `Fq2` as (c0=x, c1=0).
    #[inline(always)]
    fn fq_to_fq2(x: Self::Fq) -> Self::Fq2 {
        Self::fq2_from_components(x, Self::Fq::zero())
    }

    // --- Polynomials / Constants ---

    /// Returns the MLE of the tower reduction polynomial g(x).
    /// This is the MLE of the irreducible polynomial used to construct the extension tower.
    ///
    /// **Expected length:** \(2^{\text{TOWER\_MLE\_NUM\_VARS}}\) (e.g. 16 for BN254).
    fn g_mle() -> Vec<Self::Fq>;

    /// Converts an Fq12 element to its multilinear extension evaluations.
    /// Used for the "packed" GT exponentiation check.
    ///
    /// **Expected length:** \(2^{\text{TOWER\_MLE\_NUM\_VARS}}\) (e.g. 16 for BN254).
    fn fq12_to_mle(gt: &Self::Fq12) -> Vec<Self::Fq>;
}

/// Implementation of RecursionCurve for the BN254 curve.
#[derive(Clone, Copy, Debug, Default)]
pub struct Bn254Recursion;

use ark_bn254::{Fq, Fq12, Fq2, Fr};

impl RecursionCurve for Bn254Recursion {
    type Fq = Fq;
    type Fr = Fr;
    type Fq2 = Fq2;
    type Fq12 = Fq12;

    #[inline(always)]
    fn fq2_from_components(c0: Self::Fq, c1: Self::Fq) -> Self::Fq2 {
        Fq2::new(c0, c1)
    }

    #[inline(always)]
    fn fq2_to_components(x: &Self::Fq2) -> (Self::Fq, Self::Fq) {
        (x.c0, x.c1)
    }

    #[inline(always)]
    fn g_mle() -> Vec<Self::Fq> {
        get_g_mle()
    }

    #[inline(always)]
    fn fq12_to_mle(gt: &Self::Fq12) -> Vec<Self::Fq> {
        fq12_to_multilinear_evals(gt)
    }
}

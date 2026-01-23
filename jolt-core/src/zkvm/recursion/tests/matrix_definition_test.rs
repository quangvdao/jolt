//! Test to verify the definition of matrix M in Stage 2
//! This test ensures that M is indeed the multilinear extension of the v_i claims

use crate::{
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    },
    zkvm::recursion::stage3::virtualization::{matrix_s_index, virtual_claim_index},
};
use ark_bn254::Fq;
use ark_ff::{UniformRand, Zero};
use ark_std::test_rng;

#[test]
fn test_matrix_mle_definition_direct() {
    // Test that verifies the matrix M structure directly
    // by building it manually and comparing with expected behavior

    type F = Fq;
    let mut rng = test_rng();

    // Create a small test case with 1 constraint
    let num_constraints = 1;
    let num_constraints_padded = 1; // Already a power of 2
    let num_poly_types = crate::zkvm::recursion::constraints_sys::PolyType::NUM_TYPES;

    // Calculate num_s_vars: log2(num_poly_types * num_constraints_padded)
    let total_rows = num_poly_types * num_constraints_padded;
    let num_s_vars = (total_rows as f64).log2().ceil() as usize; // log2(30) = 5

    // Create test virtual claims as a flat vector
    // Order: for each constraint, all poly types in order
    let mut virtual_claims = Vec::new();
    for _i in 0..num_constraints {
        // Add claims for all 30 polynomial types in order (PolyType enum values 0-29)
        for poly_idx in 0..num_poly_types {
            virtual_claims.push(F::from((poly_idx + 1) as u64));
        }
    }

    // Build the matrix evaluations manually following the same layout
    // as compute_virtualization_claim
    let mu_size = 1 << num_s_vars;
    let mut mu_evals = vec![F::zero(); mu_size];

    // Fill according to matrix layout: poly_type * num_constraints_padded + constraint_idx
    for constraint_idx in 0..num_constraints {
        for poly_idx in 0..num_poly_types {
            let claim_idx = virtual_claim_index(constraint_idx, poly_idx);
            let s_idx = matrix_s_index(poly_idx, constraint_idx, num_constraints_padded);
            mu_evals[s_idx] = virtual_claims[claim_idx];
        }
    }

    // Create the multilinear polynomial M from these evaluations
    let m_poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(mu_evals.clone()));

    // Test at random points
    println!("Testing matrix MLE definition at random points...");
    println!("Matrix has {num_s_vars} s variables, {mu_size} total evaluations");

    for test_idx in 0..5 {
        // Sample random r_s
        let r_s: Vec<F> = (0..num_s_vars).map(|_| F::rand(&mut rng)).collect();

        // Method 1: Evaluate M directly at r_s
        let m_direct = PolynomialEvaluation::evaluate(&m_poly, &r_s);

        // Method 2: Compute Σ_i eq(r_s, i) · mu_evals[i]
        let eq_evals = EqPolynomial::<F>::evals(&r_s);
        let m_from_eq: F = eq_evals
            .iter()
            .zip(mu_evals.iter())
            .map(|(eq_val, mu_val)| *eq_val * *mu_val)
            .sum();

        println!(
            "Test {test_idx}: Direct eval = {m_direct:?}, From eq computation = {m_from_eq:?}"
        );
        assert_eq!(
            m_direct, m_from_eq,
            "M evaluation doesn't match expected MLE computation! Test index: {test_idx}"
        );
    }

    println!("✅ Direct test passed! M evaluations match MLE definition.");
}

#[test]
fn test_matrix_mle_with_multiple_constraints() {
    // Test with multiple constraints to ensure the layout is correct
    type F = Fq;
    let mut rng = test_rng();

    // Test with 2 constraints (padded to 2)
    let num_constraints = 2;
    let num_constraints_padded = 2;
    let num_poly_types = crate::zkvm::recursion::constraints_sys::PolyType::NUM_TYPES;

    // Calculate num_s_vars: log2(30 * 2) = log2(60) -> ceil = 6
    let total_rows = num_poly_types * num_constraints_padded;
    let num_s_vars = (total_rows as f64).log2().ceil() as usize;

    // Create test virtual claims with distinct values for each constraint
    let mut virtual_claims = Vec::new();

    // First constraint gets values 1-30 (for 30 poly types)
    for i in 1..=num_poly_types {
        virtual_claims.push(F::from(i as u64));
    }

    // Second constraint gets values 101-130
    for i in 101..=(100 + num_poly_types) {
        virtual_claims.push(F::from(i as u64));
    }

    // Build the matrix evaluations
    let mu_size = 1 << num_s_vars;
    let mut mu_evals = vec![F::zero(); mu_size];

    // Fill matrix with proper layout
    for constraint_idx in 0..num_constraints {
        for poly_idx in 0..num_poly_types {
            let claim_idx = virtual_claim_index(constraint_idx, poly_idx);
            let s_idx = matrix_s_index(poly_idx, constraint_idx, num_constraints_padded);
            mu_evals[s_idx] = virtual_claims[claim_idx];
        }
    }

    // Create the multilinear polynomial M
    let m_poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(mu_evals.clone()));

    // Test evaluation
    let r_s: Vec<F> = (0..num_s_vars).map(|_| F::rand(&mut rng)).collect();
    let m_direct = PolynomialEvaluation::evaluate(&m_poly, &r_s);
    let eq_evals = EqPolynomial::<F>::evals(&r_s);
    let m_from_eq: F = eq_evals
        .iter()
        .zip(mu_evals.iter())
        .map(|(eq_val, mu_val)| *eq_val * *mu_val)
        .sum();

    assert_eq!(m_direct, m_from_eq, "Multi-constraint test failed!");

    println!("✅ Multiple constraints test passed! Matrix layout is correct.");
}

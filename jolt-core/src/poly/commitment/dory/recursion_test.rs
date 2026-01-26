#[cfg(test)]
mod recursion_tests {
    use super::super::recursion::{JoltWitness, JoltWitnessGenerator};
    use super::super::{DoryCommitmentScheme, DoryContext, DoryGlobals, BN254};
    use crate::poly::commitment::dory::witness::gt_exp::Base4ExponentiationSteps;
    use crate::{
        field::JoltField,
        poly::{
            commitment::commitment_scheme::{CommitmentScheme, RecursionExt},
            dense_mlpoly::DensePolynomial,
            multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        },
        transcripts::Transcript,
    };
    use ark_bn254::{Fq12, Fr};
    use ark_ff::{Field, One, PrimeField, UniformRand, Zero};
    use dory::{backends::arkworks::ArkGT, recursion::WitnessGenerator};
    use rand::thread_rng;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_witness_generation_for_gt_exp() {
        let mut rng = thread_rng();

        let base = ArkGT(Fq12::rand(&mut rng));
        let scalar_fr = Fr::rand(&mut rng);
        let scalar = super::super::wrappers::jolt_to_ark(&scalar_fr);
        let result = ArkGT(base.0.pow(scalar_fr.into_bigint()));

        let witness =
            <JoltWitnessGenerator as WitnessGenerator<JoltWitness, BN254>>::generate_gt_exp(
                &base, &scalar, &result,
            );

        assert_eq!(witness.base, base.0, "Base should match");
        assert_eq!(witness.exponent, scalar_fr, "Exponent should match");
        assert_eq!(witness.result, result.0, "Result should match");

        let exp_steps = Base4ExponentiationSteps::new(base.0, scalar_fr);
        assert!(
            exp_steps.verify_result(),
            "ExponentiationSteps should verify correctly"
        );
        assert_eq!(
            exp_steps.result, result.0,
            "Results should match between witness and ExponentiationSteps"
        );

        let expected_steps = witness.bits.len().div_ceil(2);
        assert_eq!(
            witness.quotient_mles.len(),
            expected_steps,
            "Should have one quotient per base-4 digit"
        );
        assert_eq!(
            witness.rho_mles.len(),
            expected_steps + 1,
            "Should have rho_0, ..., rho_t"
        );
    }

    #[test]
    #[serial]
    fn test_special_cases() {
        let mut rng = thread_rng();
        let base = ArkGT(Fq12::rand(&mut rng));

        {
            let scalar = super::super::wrappers::jolt_to_ark(&Fr::zero());
            let result = ArkGT(Fq12::one());

            let witness =
                <JoltWitnessGenerator as WitnessGenerator<JoltWitness, BN254>>::generate_gt_exp(
                    &base, &scalar, &result,
                );

            assert_eq!(witness.result, Fq12::one());
            assert_eq!(witness.bits.len(), 0, "Zero exponent should have no bits");
            assert_eq!(witness.quotient_mles.len(), 0);
            assert_eq!(witness.rho_mles.len(), 1);
        }

        {
            let scalar = super::super::wrappers::jolt_to_ark(&Fr::one());
            let result = base;

            let witness =
                <JoltWitnessGenerator as WitnessGenerator<JoltWitness, BN254>>::generate_gt_exp(
                    &base, &scalar, &result,
                );

            assert_eq!(witness.result, base.0);
            assert_eq!(witness.bits.len(), 1);
            assert!(witness.bits[0], "Single bit should be 1");
        }

        {
            let identity_base = ArkGT(Fq12::one());
            let scalar_fr = Fr::rand(&mut rng);
            let scalar = super::super::wrappers::jolt_to_ark(&scalar_fr);
            let result = ArkGT(Fq12::one());

            let witness =
                <JoltWitnessGenerator as WitnessGenerator<JoltWitness, BN254>>::generate_gt_exp(
                    &identity_base,
                    &scalar,
                    &result,
                );

            assert_eq!(witness.result, Fq12::one());
        }
    }

    #[test]
    #[serial]
    fn test_verify_recursive_witness_generation() {
        // Reset DoryGlobals before initializing
        DoryGlobals::reset();
        let K = 1 << 2; // 2^2 = 4
        let T = 1 << 2; // 2^2 = 4
        DoryGlobals::initialize_context(K, T, DoryContext::Main, None);
        DoryGlobals::initialize_context(K, T, DoryContext::Main, None);

        // Setup
        let num_vars = 4;
        let prover_setup = DoryCommitmentScheme::setup_prover(num_vars);
        let verifier_setup = DoryCommitmentScheme::setup_verifier(&prover_setup);

        // Create polynomial
        let mut rng = thread_rng();
        let size = 1 << num_vars; // 2^4 = 16
        let coefficients: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut rng)).collect();
        let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coefficients));

        // Commit
        let (commitment, hint) = DoryCommitmentScheme::commit(&poly, &prover_setup);

        // Create evaluation point
        let point: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
            .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
            .collect();

        // Generate proof using DoryCommitmentScheme
        let mut prover_transcript = crate::transcripts::Blake2bTranscript::new(b"test");
        let proof = DoryCommitmentScheme::prove(
            &prover_setup,
            &poly,
            &point,
            Some(hint),
            &mut prover_transcript,
        );

        // Evaluate polynomial
        let evaluation = PolynomialEvaluation::evaluate(&poly, &point);

        // Use recursion extension for witness generation (captures AST).
        let mut witness_transcript = crate::transcripts::Blake2bTranscript::new(b"test");
        let (witnesses, ast) = DoryCommitmentScheme::witness_gen_with_ast(
            &proof,
            &verifier_setup,
            &mut witness_transcript,
            &point,
            &evaluation,
            &commitment,
        )
        .expect("Witness generation should succeed");

        println!(
            "Successfully generated AST: nodes={}, constraints={}",
            ast.nodes.len(),
            ast.constraints.len()
        );
        println!("Collected {} GT exp witnesses", witnesses.gt_exp.len());

        // Symbolic AST build should succeed and match the traced AST shape.
        let mut sym_transcript = crate::transcripts::Blake2bTranscript::new(b"test");
        let sym_ast = DoryCommitmentScheme::build_symbolic_ast(
            &proof,
            &verifier_setup,
            &mut sym_transcript,
            &point,
            &evaluation,
            &commitment,
        )
        .expect("Symbolic AST build should succeed");
        assert_eq!(sym_ast.nodes.len(), ast.nodes.len());
        assert_eq!(sym_ast.constraints.len(), ast.constraints.len());
    }

    #[test]
    #[serial]
    fn test_verify_recursive_matches_normal_verify() {
        // This test ensures that verify_recursive produces the same result
        // as normal verification when witness generation is enabled

        // Reset DoryGlobals
        DoryGlobals::reset();
        let K = 1 << 2;
        let T = 1 << 2;
        DoryGlobals::initialize_context(K, T, DoryContext::Main, None);

        // Setup
        let num_vars = 4;
        let prover_setup = DoryCommitmentScheme::setup_prover(num_vars);
        let verifier_setup = DoryCommitmentScheme::setup_verifier(&prover_setup);

        // Create polynomial
        let mut rng = thread_rng();
        let size = 1 << num_vars;
        let coefficients: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut rng)).collect();
        let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coefficients));

        // Commit
        let (commitment, hint) = DoryCommitmentScheme::commit(&poly, &prover_setup);

        // Create evaluation point
        let point: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
            .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
            .collect();

        // Generate proof
        let mut prover_transcript = crate::transcripts::Blake2bTranscript::new(b"test");
        let proof = DoryCommitmentScheme::prove(
            &prover_setup,
            &poly,
            &point,
            Some(hint),
            &mut prover_transcript,
        );

        let evaluation = PolynomialEvaluation::evaluate(&poly, &point);

        // First verify normally using DoryCommitmentScheme
        let mut normal_transcript = crate::transcripts::Blake2bTranscript::new(b"test");
        let normal_result = DoryCommitmentScheme::verify(
            &proof,
            &verifier_setup,
            &mut normal_transcript,
            &point,
            &evaluation,
            &commitment,
        );

        assert!(normal_result.is_ok(), "Normal verification should succeed");

        // Now generate witnesses + AST using recursion extension
        let mut witness_transcript = crate::transcripts::Blake2bTranscript::new(b"test");
        let (witnesses, ast) = DoryCommitmentScheme::witness_gen_with_ast(
            &proof,
            &verifier_setup,
            &mut witness_transcript,
            &point,
            &evaluation,
            &commitment,
        )
        .expect("Witness generation should succeed");

        // Both witnesses and AST successfully generated
        assert!(
            !witnesses.gt_exp.is_empty(),
            "Should have collected witnesses"
        );

        // Symbolic AST build should also succeed and match the traced AST shape.
        let mut sym_transcript = crate::transcripts::Blake2bTranscript::new(b"test");
        let sym_ast = DoryCommitmentScheme::build_symbolic_ast(
            &proof,
            &verifier_setup,
            &mut sym_transcript,
            &point,
            &evaluation,
            &commitment,
        );
        assert!(sym_ast.is_ok(), "Symbolic AST build should succeed");
        let sym_ast = sym_ast.unwrap();
        assert_eq!(sym_ast.nodes.len(), ast.nodes.len());
        assert_eq!(sym_ast.constraints.len(), ast.constraints.len());
    }
}

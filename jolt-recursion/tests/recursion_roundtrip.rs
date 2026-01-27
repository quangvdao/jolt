#![cfg(feature = "host")]

use ark_bn254::{Fq, Fr};
use jolt_core::poly::opening_proof::{OpeningId, Openings, PolynomialId};
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
use jolt_core::transcripts::Blake2bTranscript;
use jolt_core::zkvm::program::ProgramPreprocessing;
use jolt_core::zkvm::prover::JoltProverPreprocessing;
use jolt_core::zkvm::verifier::JoltSharedPreprocessing;
use jolt_core::zkvm::{RV64IMACProof, RV64IMACProver};
use jolt_core::zkvm::witness::{RecursionPoly, VirtualPolynomial};
use serial_test::serial;
use std::sync::Arc;

fn enable_gt_fused_end_to_end() -> bool {
    std::env::var("JOLT_RECURSION_ENABLE_GT_FUSED_END_TO_END")
        .ok()
        .map(|v| v != "0" && v.to_lowercase() != "false")
        .unwrap_or(false)
}

fn assert_no_per_instance_gt_openings(openings: &Openings<Fq>) {
    for (opening_id, _) in openings.iter() {
        let OpeningId::Polynomial(PolynomialId::Virtual(vp), _sumcheck) = opening_id else {
            continue;
        };
        let VirtualPolynomial::Recursion(rp) = vp else {
            continue;
        };
        match rp {
            RecursionPoly::GtExp { .. } | RecursionPoly::GtMul { .. } => {
                panic!("found per-instance GT opening in opening_claims: {rp:?}");
            }
            _ => {}
        }
    }
}

#[test]
#[serial]
fn recursion_proof_roundtrip_fibonacci_guest() {
    let mut program = jolt_core::host::Program::new("fibonacci-guest");
    let inputs = postcard::to_stdvec(&9u8).expect("serialize inputs");

    let (instructions, init_memory_state, _) = program.decode();
    let (lazy_trace, trace, final_memory_state, program_io) = program.trace(&inputs, &[], &[]);

    let program = Arc::new(ProgramPreprocessing::preprocess(
        instructions,
        init_memory_state,
    ));
    let shared_preprocessing =
        JoltSharedPreprocessing::new(program.meta(), program_io.memory_layout.clone(), 1 << 16);

    let prover_preprocessing: JoltProverPreprocessing<Fr, DoryCommitmentScheme> =
        JoltProverPreprocessing::new(shared_preprocessing.clone(), Arc::clone(&program));

    let prover = RV64IMACProver::gen_from_trace(
        &prover_preprocessing,
        lazy_trace,
        trace,
        program_io,
        None,
        None,
        final_memory_state,
    );
    let io_device = prover.program_io.clone();
    let (base_proof, _debug_info): (RV64IMACProof, _) = prover.prove();

    let verifier_preprocessing =
        jolt_core::zkvm::verifier::JoltVerifierPreprocessing::from(&prover_preprocessing);

    let recursion_proof = jolt_recursion::prove_recursion::<Blake2bTranscript>(
        &verifier_preprocessing,
        io_device.clone(),
        None,
        &base_proof,
    )
    .expect("prove_recursion should succeed");

    if enable_gt_fused_end_to_end() {
        assert_no_per_instance_gt_openings(&recursion_proof.opening_claims);
    }

    jolt_recursion::verify_recursion::<Blake2bTranscript>(
        &verifier_preprocessing,
        io_device,
        None,
        &base_proof,
        &recursion_proof,
    )
    .expect("verify_recursion should succeed");
}

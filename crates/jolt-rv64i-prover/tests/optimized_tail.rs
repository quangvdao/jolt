//! Batch-6b wire compatibility under every nonempty tail-kernel mask.
#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "fixture failures fail their enclosing test"
)]

#[expect(
    dead_code,
    reason = "shared machine helpers serve the complete protocol corpus"
)]
mod support;

use jolt_field::F128;
use jolt_kernels::ProofSession;
use jolt_rv64i_prover::{
    optimized::{
        source::{SharedSource, WitnessColumns},
        tail::{BitsReductionPrepare, BytecodeReadCyclePrepare, RamRaProductPrepare},
    },
    plane::Rv64iWitness,
    stages::stage6b::{self, Stage6bKernels},
};
use jolt_rv64i_verifier::statement::CheckedInputs;
use jolt_sumcheck::ClearProof;
use support::ReferenceBatches;

fn kernels(mask: u8) -> Stage6bKernels<F128> {
    let mut kernels = Stage6bKernels::default();
    if mask & 1 != 0 {
        kernels.bytecode_read_cycle = Box::new(BytecodeReadCyclePrepare);
    }
    if mask & 2 != 0 {
        kernels.ram_ra_product = Box::new(RamRaProductPrepare);
    }
    if mask & 4 != 0 {
        kernels.bits_reduction = Box::new(BitsReductionPrepare);
    }
    kernels
}

fn assert_distinct(values: &[F128], name: &str) {
    for (index, value) in values.iter().enumerate() {
        if let Some(previous) = values[..index].iter().position(|other| other == value) {
            panic!("{name} values {previous} and {index} coincide");
        }
    }
}

fn assert_separates(reference: &ReferenceBatches) {
    let output = reference
        .stage6b
        .inputs
        .batch
        .expand(&reference.stage6b.proof.values.0)
        .unwrap();
    let chunks: Vec<_> = output
        .bytecode_read_cycle
        .chunks
        .iter()
        .chain(&output.ram_ra_product.chunks)
        .copied()
        .collect();
    assert_distinct(&chunks, "batch 6b chunks");
    let input = &reference.stage6b.inputs.claims.bits_reduction;
    let claims = [
        input.direct_columns,
        input.variant_bits,
        input.pos_ra_0,
        input.pos_ra_1,
        input.should_branch,
        input.inc,
    ];
    for (index, claim) in claims.iter().enumerate() {
        assert_ne!(claim.to_raw(), 0, "BitsReduction input {index} is zero");
    }
    assert_distinct(&claims, "BitsReduction inputs");
}

fn warm_session(witness: &Rv64iWitness) -> ProofSession {
    let columns = WitnessColumns::new(&witness.layout);
    let selectors = vec![
        columns.variant(),
        columns.pos(0).unwrap(),
        columns.pos(1).unwrap(),
        columns.shift_kind(),
        columns.access_kind(),
        columns.key_kind(),
        columns.branch(),
        columns.should_branch(),
    ];
    let mut session = ProofSession::default();
    let shared = session.state_or_insert_with(SharedSource::default);
    let _ = shared.prepare(witness, Some(selectors)).unwrap();
    let _ = shared.plan().unwrap();
    session
}

fn assert_values(found: &[F128], expected: &[F128], name: &str, mask: u8, warm: bool) {
    assert_eq!(
        found.len(),
        expected.len(),
        "batch 6b mask {mask:03b}, warm {warm}, first output {name}[{}] (length)",
        found.len().min(expected.len()),
    );
    for (index, (found, expected)) in found.iter().zip(expected).enumerate() {
        assert_eq!(
            found, expected,
            "batch 6b mask {mask:03b}, warm {warm}, first output {name}[{index}]",
        );
    }
}

#[test]
fn batch_six_b_matches_kept_reference_under_all_masks_and_session_states() {
    let (statement, preprocessing, witness) = support::separating_fixture();
    let reference = support::reference_batches(&statement, &preprocessing, &witness);
    assert_separates(&reference);
    let checked = CheckedInputs::of_statement(
        &preprocessing,
        &statement,
        witness.layout.log_K_ram() as u8,
        witness.final_pc,
    )
    .unwrap();
    let ClearProof::Compressed(expected) = reference.stage6b.proof.rounds.as_clear().unwrap()
    else {
        panic!("reference batch 6b is not compressed clear");
    };
    let expected_claims = reference
        .stage6b
        .inputs
        .batch
        .expand(&reference.stage6b.proof.values.0)
        .unwrap();
    for mask in 1..8 {
        let kernels = kernels(mask);
        for warm in [false, true] {
            let mut session = if warm {
                warm_session(&witness)
            } else {
                ProofSession::default()
            };
            let mut transcript = reference.stage6b.transcript.fork();
            let (proof, output) = stage6b::prove(
                &checked,
                &witness,
                &kernels,
                &mut session,
                &mut transcript,
                &reference.stage1.output,
                &reference.stage2.output,
                &reference.stage3a.output,
                &reference.stage3b.output,
                &reference.stage4.output,
                &reference.stage5.output,
                &reference.stage6a.output,
            )
            .unwrap();
            let ClearProof::Compressed(found) = proof.rounds.as_clear().unwrap() else {
                panic!("batch 6b mask {mask:03b}, warm {warm} is not compressed clear");
            };
            assert_eq!(
                found.round_polynomials.len(),
                expected.round_polynomials.len(),
                "batch 6b mask {mask:03b}, warm {warm}, first differing round {} (round count)",
                found
                    .round_polynomials
                    .len()
                    .min(expected.round_polynomials.len()),
            );
            for (round, (found, expected)) in found
                .round_polynomials
                .iter()
                .zip(&expected.round_polynomials)
                .enumerate()
            {
                assert_eq!(
                    found, expected,
                    "batch 6b mask {mask:03b}, warm {warm}, first differing round {round}",
                );
            }
            let claims = reference
                .stage6b
                .inputs
                .batch
                .expand(&proof.values.0)
                .unwrap();
            assert_values(
                &claims.bytecode_read_cycle.chunks,
                &expected_claims.bytecode_read_cycle.chunks,
                "BytecodeReadCycle.chunks",
                mask,
                warm,
            );
            assert_values(
                &claims.ram_ra_product.chunks,
                &expected_claims.ram_ra_product.chunks,
                "RamRaProduct.chunks",
                mask,
                warm,
            );
            assert_values(
                &proof.values.0,
                &reference.stage6b.proof.values.0,
                "BitsColumns",
                mask,
                warm,
            );
            assert_values(
                &output.point,
                &reference.stage6b.output.point,
                "point",
                mask,
                warm,
            );
        }
    }
}

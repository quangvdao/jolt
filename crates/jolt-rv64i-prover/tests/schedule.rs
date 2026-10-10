//! Generated batch windows and the transcript event schedule of complete proofs.
#![expect(
    clippy::unwrap_used,
    reason = "invalid fixtures fail the enclosing test"
)]
#[expect(
    dead_code,
    reason = "the shared machine fixtures serve several integration tests"
)]
mod support;

use jolt_claims::SymbolicSumcheck;
use jolt_field::F128;
use jolt_rv64i_arith::Layout;
use jolt_rv64i_prover::{
    backend::Rv64iBackend,
    commitment::transparent::TransparentBits,
    prover::{prove, prove_with_transcript, ProverPreprocessing},
};
use jolt_rv64i_verifier::{
    claims::{router_short::RouterShortSymbolic, spartan_inner::SpartanInnerSymbolic},
    commitment::{BitsCommitmentScheme, BitsGeometry},
    proof::geometry,
    public::matrices::RowMatrices,
    stages::{
        stage1, stage1::Stage1Sumchecks, stage2, stage3a, stage3b, stage3b::Stage3bSumchecks,
        stage4, stage4::Stage4Sumchecks, stage5, stage5::Stage5Sumchecks, stage6a,
        stage6a::Stage6aSumchecks, stage6b, stage6b::Stage6bSumchecks,
    },
    statement::CheckedInputs,
    transcript::preamble,
    verifier::verify_with_transcript,
};
use jolt_sumcheck::BatchPrelude;
use jolt_verifier::stages::relations::ConcreteSumcheck;
use std::sync::Arc;
use support::{Event, RecordedTranscript};

macro_rules! member_degrees {
    ($function:ident, batch=$batch:ident,label=$label:literal,aggregates={$($aggregates:tt)*},shape=$shape:ident,
     members=[$({name:$field:ident,relation:$relation:ident,presence:required},)+]) => {
        fn $function(batch:&$batch<F128>)->Vec<usize> { vec![$(batch.$field.degree()),+] }
    };
}
jolt_rv64i_verifier::stage1_sumchecks_members!(member_degrees degrees1,);
jolt_rv64i_verifier::stage3b_sumchecks_members!(member_degrees degrees3b,);
jolt_rv64i_verifier::stage4_sumchecks_members!(member_degrees degrees4,);
jolt_rv64i_verifier::stage5_sumchecks_members!(member_degrees degrees5,);
jolt_rv64i_verifier::stage6a_sumchecks_members!(member_degrees degrees6a,);
jolt_rv64i_verifier::stage6b_sumchecks_members!(member_degrees degrees6b,);

#[test]
fn generated_schedule_matches_both_reference_layouts() {
    let layout = Layout::new(4, 5, 0).unwrap();
    for log_t in [0, usize::MAX] {
        assert!(Stage4Sumchecks::for_geometry(log_t, &layout).is_err());
        assert!(Stage5Sumchecks::for_geometry(log_t, &layout).is_err());
    }
    for (a, rounds, degrees, offsets, elements, total_rounds) in [
        (
            20,
            [30, 10, 17, 22, 42, 22, 20, 22],
            [3, 2, 2, 5, 3, 4, 2, 6],
            vec![
                vec![0, 3],
                vec![0],
                vec![0],
                vec![0; 5],
                vec![15, 0, 0],
                vec![0; 2],
                vec![0],
                vec![0; 3],
            ],
            640,
            185,
        ),
        (
            23,
            [30, 10, 17, 22, 45, 22, 20, 22],
            [3, 2, 2, 5, 3, 4, 2, 7],
            vec![
                vec![0, 3],
                vec![0],
                vec![0],
                vec![0; 5],
                vec![18, 0, 0],
                vec![0; 2],
                vec![0],
                vec![0; 3],
            ],
            671,
            188,
        ),
    ] {
        let schedule = geometry(22, 20, a).unwrap();
        assert_eq!(schedule.each_ref().map(|batch| batch.max_num_vars), rounds);
        assert_eq!(schedule.each_ref().map(|batch| batch.max_degree), degrees);
        assert_eq!(
            schedule
                .iter()
                .map(|batch| batch
                    .members
                    .iter()
                    .map(|member| member.offset)
                    .collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            offsets
        );
        let member_rounds = schedule
            .iter()
            .map(|batch| {
                batch
                    .members
                    .iter()
                    .map(|member| member.rounds)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            member_rounds,
            vec![
                vec![30, 27],
                vec![10],
                vec![17],
                vec![22; 5],
                vec![27, a + 22, a],
                vec![22; 2],
                vec![20],
                vec![22; 3]
            ]
        );
        assert_eq!(
            schedule
                .iter()
                .map(|batch| batch.max_num_vars * batch.max_degree)
                .sum::<usize>(),
            elements
        );
        assert_eq!(
            schedule
                .iter()
                .map(|batch| batch.max_num_vars)
                .sum::<usize>(),
            total_rounds
        );
        let layout = Layout::new(20, a, 0).unwrap();
        let wires = [
            Stage1Sumchecks::for_geometry(22, &layout)
                .unwrap()
                .output_claim_count(),
            SpartanInnerSymbolic::new(())
                .expected_output_openings::<F128>()
                .len(),
            RouterShortSymbolic::new(())
                .expected_output_openings::<F128>()
                .len(),
            Stage3bSumchecks::for_geometry(22, &layout)
                .unwrap()
                .output_claim_count(),
            Stage4Sumchecks::<F128>::for_geometry(22, &layout)
                .unwrap()
                .output_claim_count(),
            Stage5Sumchecks::<F128>::for_geometry(22, &layout)
                .unwrap()
                .output_claim_count(),
            Stage6aSumchecks::<F128>::for_geometry(22, &layout)
                .unwrap()
                .output_claim_count(),
            Stage6bSumchecks::<F128>::for_geometry(22, &layout)
                .unwrap()
                .bits_reduction
                .wire_output_openings()
                .len(),
        ];
        let member_degrees = vec![
            degrees1(&Stage1Sumchecks::for_geometry(22, &layout).unwrap()),
            vec![SpartanInnerSymbolic::new(()).degree()],
            vec![RouterShortSymbolic::new(()).degree()],
            degrees3b(&Stage3bSumchecks::for_geometry(22, &layout).unwrap()),
            degrees4(&Stage4Sumchecks::<F128>::for_geometry(22, &layout).unwrap()),
            degrees5(&Stage5Sumchecks::<F128>::for_geometry(22, &layout).unwrap()),
            degrees6a(&Stage6aSumchecks::<F128>::for_geometry(22, &layout).unwrap()),
            degrees6b(&Stage6bSumchecks::<F128>::for_geometry(22, &layout).unwrap()),
        ];
        assert_eq!(
            member_degrees,
            vec![
                vec![3, 3],
                vec![2],
                vec![2],
                vec![3, 5, 4, 5, 4],
                vec![3; 3],
                vec![4; 2],
                vec![2],
                if a == 20 {
                    vec![6, 6, 2]
                } else {
                    vec![6, 7, 2]
                }
            ]
        );
        assert_eq!(wires, [6, 2, 5, 18, 7, 4, 1, 256]);
        assert_eq!(wires.into_iter().sum::<usize>(), 299);
    }
}

#[test]
fn concrete_member_windows_match_the_envelope_schedule_on_a_counting_loop() {
    let (statement, verifier, witness) = support::counting_loop();
    let preprocessing = ProverPreprocessing {
        verifier,
        scheme: (),
    };
    let proof = prove(
        &preprocessing,
        &statement,
        &witness,
        &Rv64iBackend::reference(),
    )
    .unwrap();
    let checked = CheckedInputs::new(&preprocessing.verifier, &statement, &proof).unwrap();
    let schedule = geometry(
        checked.log_T(),
        checked.log_K_bytecode(),
        checked.log_K_ram(),
    )
    .unwrap();
    let check_batch = |index: usize, concrete: BatchPrelude<F128>| {
        let envelope = &schedule[index];
        assert_eq!(
            (concrete.max_num_vars, concrete.max_degree),
            (envelope.max_num_vars, envelope.max_degree),
            "batch index {index}",
        );
        assert_eq!(
            concrete
                .members
                .iter()
                .map(|member| (member.rounds, member.offset))
                .collect::<Vec<_>>(),
            envelope
                .members
                .iter()
                .map(|member| (member.rounds, member.offset))
                .collect::<Vec<_>>(),
            "member windows of batch index {index}",
        );
    };
    let mut transcript: RecordedTranscript = preamble(&checked);
    let state = TransparentBits::verify_commit(
        &(),
        BitsGeometry {
            log_T: checked.log_T(),
        },
        &proof.bits_commitment,
        &mut transcript,
    )
    .unwrap();
    check_batch(
        0,
        stage1::verify::from_checked(&checked, &mut transcript.fork())
            .unwrap()
            .geometry()
            .unwrap(),
    );
    let s1 = stage1::verify::verify(&checked, &proof.stage1, &mut transcript).unwrap();
    check_batch(
        1,
        stage2::verify::from_upstream(Arc::new(RowMatrices::new(checked.layout())), &s1)
            .unwrap()
            .batch
            .geometry()
            .unwrap(),
    );
    let s2 = stage2::verify::verify(&checked, &proof.stage2, &mut transcript, &s1).unwrap();
    check_batch(
        2,
        stage3a::verify::from_upstream(&checked, &s2)
            .unwrap()
            .batch
            .geometry()
            .unwrap(),
    );
    let s3a = stage3a::verify::verify(&checked, &proof.stage3a, &mut transcript, &s2).unwrap();
    check_batch(
        3,
        stage3b::verify::from_upstream(&checked, &s1, &s3a)
            .unwrap()
            .batch
            .geometry()
            .unwrap(),
    );
    let s3b =
        stage3b::verify::verify(&checked, &proof.stage3b, &mut transcript, &s1, &s3a).unwrap();
    check_batch(
        4,
        stage4::verify::from_upstream(&checked, &mut transcript.fork(), &s3a, &s3b)
            .unwrap()
            .batch
            .geometry()
            .unwrap(),
    );
    let s4 = stage4::verify::verify(&checked, &proof.stage4, &mut transcript, &s3a, &s3b).unwrap();
    check_batch(
        5,
        stage5::verify::from_upstream(&checked, &s3a, &s4)
            .unwrap()
            .batch
            .geometry()
            .unwrap(),
    );
    let s5 = stage5::verify::verify(&checked, &proof.stage5, &mut transcript, &s3a, &s4).unwrap();
    check_batch(
        6,
        stage6a::verify::from_upstream(&checked, &s3a, &s3b, &s4, &s5)
            .unwrap()
            .batch
            .geometry()
            .unwrap(),
    );
    let s6a = stage6a::verify::verify(
        &checked,
        &proof.stage6a,
        &mut transcript,
        &s3a,
        &s3b,
        &s4,
        &s5,
    )
    .unwrap();
    check_batch(
        7,
        stage6b::verify::from_upstream(&checked, &s1, &s2, &s3a, &s3b, &s4, &s5, &s6a)
            .unwrap()
            .batch
            .geometry()
            .unwrap(),
    );
    let _output = stage6b::verify::verify(
        &checked,
        &proof.stage6b,
        &mut transcript,
        &s1,
        &s2,
        &s3a,
        &s3b,
        &s4,
        &s5,
        &s6a,
        state,
        &proof.opening,
    )
    .unwrap();
}

#[test]
#[expect(
    non_snake_case,
    reason = "the extension-row dimension retains its protocol notation"
)]
#[expect(
    clippy::panic,
    reason = "a non-scalar event fails the terminal draw-order fixture"
)]
fn complete_transcript_has_the_prescribed_counts_and_terminal_draw_order() {
    let (statement, verifier, witness) = support::counting_loop();
    let preprocessing = ProverPreprocessing {
        verifier,
        scheme: (),
    };
    let (proof, prover) = prove_with_transcript::<TransparentBits, RecordedTranscript>(
        &preprocessing,
        &statement,
        &witness,
        &Rv64iBackend::reference(),
    )
    .unwrap();
    let verified = verify_with_transcript::<TransparentBits, RecordedTranscript>(
        &preprocessing.verifier,
        &statement,
        &proof,
    )
    .unwrap();
    assert_eq!(prover.events, verified.events);
    assert_eq!(prover.batch_states, verified.batch_states);
    let t = usize::from(statement.log_T);
    let a = witness.layout.log_K_ram();
    let b = witness.layout.log_K_bytecode();
    let m_F = 4;
    assert_eq!(verified.challenge_count(), 104 + m_F + 2 * a + b + 7 * t);
    assert_eq!(verified.label_count(b"sumcheck_claim"), 18);
    assert_eq!(verified.label_count(b"sumcheck_poly"), 35 + 5 * t + a + b);
    assert_eq!(verified.label_count(b"opening_claim"), 299);
    assert_eq!(
        verified.label_count(b"sumcheck_claim")
            + verified.label_count(b"sumcheck_poly")
            + verified.label_count(b"opening_claim"),
        352 + 5 * t + a + b
    );
    let expected_squeezes = [
        18 + m_F + 3 * t,
        17,
        18,
        t + 5,
        2 * a + t + 6,
        t + 4,
        b + 17,
        t + 19,
    ];
    let expected_absorbs = [t + 16, 13, 23, t + 23, a + t + 10, t + 6, b + 2, t + 259];
    let mut start = 38;
    for (batch, &end) in verified.batch_event_ends.iter().enumerate() {
        let end = if batch == 7 {
            verified.events.len()
        } else {
            end
        };
        let events = &verified.events[start..end];
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::Challenge(_) | Event::Scalar(_)))
                .count(),
            expected_squeezes[batch],
            "batch {batch} squeezes"
        );
        let labels=events.iter().filter(|event|matches!(event,Event::Append(bytes) if bytes.len()==32 && [b"sumcheck_claim".as_slice(),b"sumcheck_poly".as_slice(),b"opening_claim".as_slice()].iter().any(|label| bytes.starts_with(label)))).count();
        assert_eq!(
            labels, expected_absorbs[batch],
            "batch {batch} labelled absorbs"
        );
        start = end;
    }
    let after_6a = verified.batch_event_ends[6];
    let draws: Vec<_> = verified.events[after_6a..after_6a + 8]
        .iter()
        .map(|event| {
            if let Event::Scalar(value) = event {
                *value
            } else {
                panic!("batch 6b scalar preceded by an absorption")
            }
        })
        .collect();
    let mut before_6b = verified.inner_at(after_6a);
    let batch = Stage6bSumchecks::<F128>::for_geometry(t, &witness.layout).unwrap();
    let challenges = batch.draw_challenges(&mut before_6b).unwrap();
    let bits = &challenges.bits_reduction;
    let ram = &challenges.ram_ra_product;
    assert_eq!(
        draws,
        vec![
            bits.direct_columns,
            bits.variant_bits,
            bits.pos_ra_0,
            bits.pos_ra_1,
            bits.should_branch,
            bits.inc,
            ram.read,
            ram.val
        ]
    );
    assert!(
        matches!(&verified.events[after_6a+8], Event::Append(bytes) if bytes.starts_with(b"sumcheck_claim"))
    );
}

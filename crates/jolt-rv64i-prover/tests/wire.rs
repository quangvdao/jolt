//! Frozen proof and transcript encodings for the binary-field RV64I boundary.
#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "invalid fixtures fail the enclosing test"
)]
#[expect(
    dead_code,
    reason = "shared machine helpers serve the complete protocol corpus"
)]
mod support;
use blake2::{digest::consts::U32, Blake2b, Digest};
use jolt_field::{CanonicalBytes, F128};
use jolt_rv64i_prover::backend::Rv64iBackend;
use jolt_rv64i_prover::commitment::{transparent::TransparentBits, BitsCommitmentProver};
use jolt_rv64i_prover::prover::{prove, ProverPreprocessing};
use jolt_rv64i_verifier::{
    commitment::{squeeze_bytes, BitsGeometry},
    error::ProofDecodeError,
    proof::Rv64iProof,
    statement::CheckedInputs,
    transcript::{preamble, Rv64iTranscript, PROTOCOL_LABEL},
};
use jolt_sumcheck::{ClearProof, SumcheckProof};
use jolt_transcript::Transcript;

// Each literal is one body of §13, written in order for the counting-loop statement.
const PREAMBLE_BODIES: [&str; 36] = [
    "706172616d730000000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000000000000006",
    "0000000000000000000000000000000000000000000000000000000000000004",
    "0000000000000000000000000000000000000000000000000000000000000005",
    "000000000000000000000000000000000000000000000000000000007fffffe0",
    "73746174656d656e740000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000000080000000",
    "0000000000000000000000000000000000000000000000000000000000000028",
    "0000000000000000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000007fffffe0",
    "000000000000000000000000000000000000000000000000000000007fffffe0",
    "0000000000000000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000007fffffe0",
    "000000000000000000000000000000000000000000000000000000007fffffe0",
    "0000000000000000000000000000000000000000000000000000000000000008",
    "0000000000000000000000000000000000000000000000000000000000000008",
    "000000000000000000000000000000000000000000000000000000007fffffe0",
    "000000000000000000000000000000000000000000000000000000007fffffe8",
    "000000000000000000000000000000000000000000000000000000007fffffe8",
    "000000000000000000000000000000000000000000000000000000007ffffff0",
    "0000000000000000000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000000080000028",
    "0000000000000000000000000000000000000000000000000000000000000000",
    "00000000000000000000000000000000000000000000000000000000800000a8",
    "000000000000000000000000000000000000000000000000000000007ffffff0",
    "000000000000000000000000000000000000000000000000000000007ffffff8",
    "0000000000000000000000000000000000000000000000000000000080000000",
    "696e707574730000000000000000000000000000000000000000000000000002",
    "1234",
    "6f75747075747300000000000000000000000000000000000000000000000000",
    "",
    "0000000000000000000000000000000000000000000000000000000000000000",
    "70726f6772616d00000000000000000000000000000000000000000000000000",
    "a35d446f2fc447f1163a9b79dc2da6400b9ae3c0f2403a19fc9b6d6626ae3b92",
    "66696e616c5f7063000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000000080000020",
];

const PREAMBLE_HEX: &str = "706172616d730000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000600000000000000000000000000000000000000000000000000000000000000040000000000000000000000000000000000000000000000000000000000000005000000000000000000000000000000000000000000000000000000007fffffe073746174656d656e740000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008000000000000000000000000000000000000000000000000000000000000000000000280000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000007fffffe0000000000000000000000000000000000000000000000000000000007fffffe00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000007fffffe0000000000000000000000000000000000000000000000000000000007fffffe000000000000000000000000000000000000000000000000000000000000000080000000000000000000000000000000000000000000000000000000000000008000000000000000000000000000000000000000000000000000000007fffffe0000000000000000000000000000000000000000000000000000000007fffffe8000000000000000000000000000000000000000000000000000000007fffffe8000000000000000000000000000000000000000000000000000000007ffffff000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000080000028000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000800000a8000000000000000000000000000000000000000000000000000000007ffffff0000000000000000000000000000000000000000000000000000000007ffffff80000000000000000000000000000000000000000000000000000000080000000696e70757473000000000000000000000000000000000000000000000000000212346f75747075747300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000070726f6772616d00000000000000000000000000000000000000000000000000a35d446f2fc447f1163a9b79dc2da6400b9ae3c0f2403a19fc9b6d6626ae3b9266696e616c5f70630000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000080000020";

fn bytes(hex: &str) -> Vec<u8> {
    hex.as_bytes()
        .chunks_exact(2)
        .map(|digits| u8::from_str_radix(std::str::from_utf8(digits).unwrap(), 16).unwrap())
        .collect()
}
#[derive(Default)]
struct RecordingTranscript {
    inner: Rv64iTranscript,
    bodies: Vec<Vec<u8>>,
    draws: usize,
}
impl Transcript for RecordingTranscript {
    type Challenge = F128;
    fn new(label: &'static [u8]) -> Self {
        Self {
            inner: Rv64iTranscript::new(label),
            ..Self::default()
        }
    }
    fn append_bytes(&mut self, bytes: &[u8]) {
        self.bodies.push(bytes.to_vec());
        self.inner.append_bytes(bytes);
    }
    fn challenge(&mut self) -> F128 {
        self.draws += 1;
        self.inner.challenge()
    }
    fn state(&self) -> [u8; 32] {
        self.inner.state()
    }
}
#[test]
fn counting_loop_preamble_has_the_frozen_bodies_and_commit_state() {
    let (statement, preprocessing, witness) = support::counting_loop();
    assert_eq!(
        support::counting_loop_facts().last().unwrap().next_pc,
        0x8000_0020
    );
    let checked =
        CheckedInputs::of_statement(&preprocessing, &statement, 5, witness.final_pc).unwrap();
    let mut transcript: RecordingTranscript = preamble(&checked);
    let manual: Vec<_> = PREAMBLE_BODIES.iter().map(|body| bytes(body)).collect();
    assert_eq!(transcript.bodies.len(), 36);
    assert_eq!(transcript.bodies, manual);
    assert_eq!(transcript.bodies.concat(), bytes(PREAMBLE_HEX));
    assert_eq!(transcript.draws, 0);
    let packed_prefix: [[u64; 4]; 12] = [
        [0x0000_0000_8000_0000, 0x0000_0000_0000_0000, 0, 0],
        [0x0000_0000_ffff_fff8, 0x0000_0000_0000_0001, 0, 0],
        [0x0000_0000_0000_0000, 0x0000_0000_0000_0002, 0, 0],
        [0x0000_0000_0000_0003, 0x0000_0000_0000_0004, 0, 0],
        [0x0000_0000_0000_0001, 0x0000_0000_0000_0008, 0, 0],
        [0x0000_0000_0000_0000, 0x0000_6000_8000_0010, 0, 0],
        [0x0000_0000_0000_0003, 0x0000_0000_0000_0008, 0, 0],
        [0x0000_0000_0000_0000, 0x0000_6000_0000_0010, 0, 0],
        [0x0000_0000_0000_0001, 0x0000_0000_0000_0008, 0, 0],
        [0x0000_0000_0000_0000, 0x0000_0000_0000_0010, 0, 0],
        [0x0000_0000_0000_0001, 0x0000_0000_0000_0020, 0, 0],
        [0x0000_0000_0000_0001, 0x0000_0000_0002_0040, 0, 0],
    ];
    assert_eq!(&witness.bits[..12], &packed_prefix);
    assert!(witness.bits[12..].iter().all(|row| *row == [0, 128, 0, 0]));
    let (commitment, _) = TransparentBits::commit(
        &(),
        BitsGeometry { log_T: 6 },
        &witness.bits,
        &mut transcript,
    )
    .unwrap();
    let digest = bytes("8ade10efdf40fcbfbc6ce61ad5bbfea3f31206c05c1e74b118dc0897c2ca0550");
    assert_eq!(commitment.0.as_slice(), digest);
    let mut independent = Rv64iTranscript::new(PROTOCOL_LABEL);
    for body in &manual {
        independent.append_bytes(body);
    }
    independent.append_bytes(&bytes(
        "626974735f636f6d6d69746d656e740000000000000000000000000000000000",
    ));
    independent.append_bytes(&digest);
    assert_eq!(transcript.state(), independent.state());
    assert_eq!(
        transcript.state(),
        [
            0x28, 0x78, 0xa5, 0xfb, 0x94, 0x3b, 0x32, 0xb9, 0x81, 0x0d, 0x5b, 0x1d, 0x8a, 0x6c,
            0x61, 0x3a, 0xaf, 0x7e, 0x23, 0x3b, 0xfa, 0xab, 0xe2, 0xf0, 0x33, 0x5f, 0x19, 0xbd,
            0x3d, 0x04, 0x9a, 0x62
        ]
    );
}

fn seeded_bytes() -> Vec<u8> {
    let mut bytes = vec![0, 5];
    bytes.extend_from_slice(&0x8000_0020_u64.to_le_bytes());
    bytes.extend_from_slice(&32_u64.to_le_bytes());
    bytes.extend_from_slice(&[0xa5; 32]);
    let mut seed = 1_u128;
    for (rounds, degree, values) in [
        (14, 3, 6),
        (10, 2, 2),
        (17, 2, 5),
        (6, 5, 18),
        (11, 3, 7),
        (6, 4, 4),
        (4, 2, 1),
        (6, 3, 256),
    ] {
        for round in 0..rounds {
            for coefficient in 0..degree {
                let value = if round % 2 == 0 && coefficient != 0 {
                    0
                } else {
                    seed
                };
                bytes.extend_from_slice(&value.to_le_bytes());
                seed += 1;
            }
        }
        for _ in 0..values {
            bytes.extend_from_slice(&seed.to_le_bytes());
            seed += 1;
        }
    }
    bytes.extend_from_slice(&2048_u64.to_le_bytes());
    for i in 0..256_u64 {
        bytes.extend_from_slice(&(i * 0x0101).to_le_bytes());
    }
    bytes
}
#[test]
fn proof_envelope_preserves_seeded_fields_and_rejects_noncanonical_envelopes() {
    let bytes = seeded_bytes();
    let proof = Rv64iProof::<TransparentBits>::from_bytes(&bytes, 6, 4).unwrap();
    assert_eq!(proof.to_bytes(), bytes);
    macro_rules! fields {
        ($start:expr; $($value:expr),+ $(,)?) => {
            assert_eq!([$($value),+].as_slice(), (0..[$($value),+].len()).map(|i| F128::from_raw($start + i as u128)).collect::<Vec<_>>());
        };
    }
    let v = &proof.stage1.values;
    fields!(43; v.az_f2, v.bz_f2, v.cz_f2, v.az_f128, v.bz_f128, v.cz_f128);
    let v = &proof.stage2.values;
    fields!(69; v.witness_routed, v.direct_columns);
    let v = &proof.stage3a.values;
    fields!(105; v.variant, v.shift, v.memory, v.compare, v.branch);
    let v = &proof.stage3b.values;
    fields!(140; v.rs1_value, v.rs2_value, v.rd_pre_value, v.imm, v.fall_through_pc, v.pc_plus_imm, v.pc, v.next_pc, v.variant_bits, v.variant, v.shift_kind, v.pos_ra_0, v.pos_ra_1, v.ram_read_value, v.access_kind, v.key_kind, v.branch, v.should_branch);
    let v = &proof.stage4.values;
    fields!(191; v.rs1_ra, v.rs2_ra, v.rd_wa, v.registers_val, v.ram_ra, v.ram_val, v.ram_val_final);
    let v = &proof.stage5.values;
    fields!(222; v.rd_wa, v.store, v.inc, v.ram_ra);
    assert_eq!(proof.stage6a.values.address_claim, F128::from_raw(234));
    assert_eq!(
        proof.stage6b.values.0,
        (253..509).map(F128::from_raw).collect::<Vec<_>>()
    );
    assert_eq!(proof.bits_commitment.0, [0xa5; 32]);
    assert_eq!(
        proof
            .opening
            .0
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>(),
        (0..256_u64).map(|i| i * 0x0101).collect::<Vec<_>>()
    );
    if let SumcheckProof::Clear(ClearProof::Compressed(rounds)) = &proof.stage1.rounds {
        assert_eq!(
            rounds.round_polynomials[0].coeffs_except_linear_term(),
            &[F128::from_raw(1)]
        );
        assert_eq!(
            rounds.round_polynomials[1].coeffs_except_linear_term(),
            &[F128::from_raw(4), F128::from_raw(5), F128::from_raw(6)]
        );
    } else {
        panic!("decoded rounds must be compressed clear messages");
    }
    for end in 0..bytes.len() {
        assert!(Rv64iProof::<TransparentBits>::from_bytes(&bytes[..end], 6, 4).is_err());
    }
    let mut changed = bytes.clone();
    changed.push(0);
    assert!(matches!(
        Rv64iProof::<TransparentBits>::from_bytes(&changed, 6, 4),
        Err(ProofDecodeError::TrailingBytes)
    ));
    changed = bytes.clone();
    changed[0] = 1;
    assert!(matches!(
        Rv64iProof::<TransparentBits>::from_bytes(&changed, 6, 4),
        Err(ProofDecodeError::InvalidVersion)
    ));
    for prefix in [10, bytes.len() - 2056] {
        changed = bytes.clone();
        changed[prefix..prefix + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(
            Rv64iProof::<TransparentBits>::from_bytes(&changed, 6, 4),
            Err(ProofDecodeError::Length)
        ));
    }
    for (t, b, a) in [
        (0, 4, 5),
        (33, 4, 5),
        (6, 0, 5),
        (6, 25, 5),
        (6, 4, 4),
        (6, 4, 62),
    ] {
        changed = bytes.clone();
        changed[1] = a;
        assert!(matches!(
            Rv64iProof::<TransparentBits>::from_bytes(&changed, t, b),
            Err(ProofDecodeError::Dimensions)
        ));
    }
}

#[test]
fn larger_challenge_bytes_use_only_consecutive_scalar_draws() {
    for len in [0_usize, 1, 16, 24, 32, 33] {
        let mut expected = Rv64iTranscript::new(PROTOCOL_LABEL);
        let mut literal = Vec::new();
        for _ in 0..len.div_ceil(16) {
            let mut encoding = [0; 16];
            expected.challenge().to_bytes_le(&mut encoding);
            literal.extend_from_slice(&encoding);
        }
        literal.truncate(len);
        let mut actual = RecordingTranscript::new(PROTOCOL_LABEL);
        let mut output = vec![0; len];
        squeeze_bytes(&mut actual, &mut output);
        assert_eq!(output, literal);
        assert_eq!(actual.draws, len.div_ceil(16));
        assert_eq!(actual.state(), expected.state());
    }
}

#[test]
fn counting_loop_proof_has_the_frozen_encoding_and_rejects_malformed_envelopes() {
    let (statement, verifier, witness) = support::counting_loop();
    let preprocessing = ProverPreprocessing {
        verifier,
        scheme: (),
    };
    let proof = prove::<TransparentBits>(
        &preprocessing,
        &statement,
        &witness,
        &Rv64iBackend::reference(),
    )
    .unwrap();
    let encoded = proof.to_bytes();
    let digest: [u8; 32] = Blake2b::<U32>::digest(&encoded).into();
    assert_eq!(encoded.len(), 10_234);
    assert_eq!(
        digest.as_slice(),
        bytes("b47073578be607e5578cbc512ab46d4ea326c676fbbab24937a583f0d2f8d28e")
    );
    let decoded = Rv64iProof::<TransparentBits>::from_bytes(&encoded, 6, 4).unwrap();
    assert_eq!(decoded.log_K_ram, proof.log_K_ram);
    assert_eq!(decoded.final_pc, proof.final_pc);
    assert_eq!(decoded.bits_commitment, proof.bits_commitment);
    assert_eq!(decoded.opening, proof.opening);
    macro_rules! batch {
        ($name:ident) => {
            assert_eq!(decoded.$name.values, proof.$name.values);
            match (&decoded.$name.rounds, &proof.$name.rounds) {
                (
                    SumcheckProof::Clear(ClearProof::Compressed(a)),
                    SumcheckProof::Clear(ClearProof::Compressed(b)),
                ) => {
                    assert_eq!(a.round_polynomials.len(), b.round_polynomials.len());
                    for (a, b) in a.round_polynomials.iter().zip(&b.round_polynomials) {
                        assert_eq!(a.coeffs_except_linear_term(), b.coeffs_except_linear_term());
                    }
                }
                _ => panic!("the protocol emits compressed clear rounds"),
            }
        };
    }
    batch!(stage1);
    batch!(stage2);
    batch!(stage3a);
    batch!(stage3b);
    batch!(stage4);
    batch!(stage5);
    batch!(stage6a);
    batch!(stage6b);
    assert_eq!(decoded.to_bytes(), encoded);
    for end in 0..encoded.len() {
        assert!(matches!(
            Rv64iProof::<TransparentBits>::from_bytes(&encoded[..end], 6, 4),
            Err(ProofDecodeError::Truncated | ProofDecodeError::Length)
        ));
    }
    let mut changed = encoded.clone();
    changed.push(0);
    assert!(matches!(
        Rv64iProof::<TransparentBits>::from_bytes(&changed, 6, 4),
        Err(ProofDecodeError::TrailingBytes)
    ));
    changed = encoded.clone();
    changed[0] = 1;
    assert!(matches!(
        Rv64iProof::<TransparentBits>::from_bytes(&changed, 6, 4),
        Err(ProofDecodeError::InvalidVersion)
    ));
    for prefix in [10, encoded.len() - 2056] {
        changed = encoded.clone();
        changed[prefix..prefix + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(
            Rv64iProof::<TransparentBits>::from_bytes(&changed, 6, 4),
            Err(ProofDecodeError::Length)
        ));
    }
    for (t, b, a) in [
        (0, 4, 5),
        (33, 4, 5),
        (6, 0, 5),
        (6, 25, 5),
        (6, 4, 4),
        (6, 4, 62),
    ] {
        changed = encoded.clone();
        changed[1] = a;
        assert!(matches!(
            Rv64iProof::<TransparentBits>::from_bytes(&changed, t, b),
            Err(ProofDecodeError::Dimensions)
        ));
    }
}

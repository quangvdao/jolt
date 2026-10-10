//! Packed bit commitments reject altered tables, columns and invalid opening geometry.
#![expect(clippy::unwrap_used, reason = "tests fail on invalid fixtures")]
#[expect(
    dead_code,
    reason = "shared machine helpers serve the complete protocol corpus"
)]
mod support;
use jolt_field::{One, Ring, F128};
use jolt_poly::Polynomial;
use jolt_rv64i_arith::BitsRow;
use jolt_rv64i_prover::commitment::{
    transparent::{TransparentBits, TransparentError, TransparentOpening},
    BitsCommitmentProver,
};
use jolt_rv64i_prover::plane::Rv64iWitness;
use jolt_rv64i_verifier::commitment::{BitsCommitmentScheme, BitsGeometry, BitsOpening, BitsWire};
use jolt_rv64i_verifier::points::to_high_to_low;
use jolt_transcript::{Blake2bTranscript, Transcript};
use std::sync::Arc;
use support::replay::State;
type BinaryTranscript = Blake2bTranscript<F128>;
fn witness() -> Rv64iWitness {
    let (statement, _, source) = support::counting_loop();
    Rv64iWitness::synthetic(
        0x0641_0b17,
        source.layout,
        source.bytecode,
        &statement.device.memory_layout,
        source.initial_ram,
        5,
    )
    .unwrap()
}

#[test]
fn transparent_scheme_rejects_table_column_lengths_and_oversized_geometry() {
    let witness = witness();
    let rho: Vec<_> = (0..8).map(|i| F128::from_raw(0x100 + i)).collect();
    let cycle: Vec<_> = (0..5).map(|i| F128::from_raw(0x200 + i)).collect();
    let columns: Vec<_> = (0..256)
        .map(|y| {
            Polynomial::new(
                witness
                    .bits
                    .iter()
                    .map(|row| F128::from_u64((row[y / 64] >> (y % 64)) & 1))
                    .collect(),
            )
            .evaluate(&to_high_to_low(&cycle))
        })
        .collect();
    let opening_proof = TransparentOpening(Arc::clone(&witness.bits));
    let geometry = BitsGeometry { log_T: 5 };
    let mut commit_transcript = BinaryTranscript::new(b"rv64i-bits-contract");
    let (commitment, _) =
        TransparentBits::commit(&(), geometry, &witness.bits, &mut commit_transcript).unwrap();
    let opening = BitsOpening {
        geometry,
        column_point: &rho,
        cycle_point: &cycle,
        columns: &columns,
    };
    let mut transcript = BinaryTranscript::new(b"rv64i-bytecode-batches");
    let state =
        TransparentBits::verify_commit(&(), geometry, &commitment, &mut transcript).unwrap();
    let mut changed = opening_proof.0.to_vec();
    changed[0][0] ^= 1;
    assert!(TransparentBits::verify_opening(
        &(),
        state,
        &opening,
        &TransparentOpening(changed.into()),
        &mut transcript
    )
    .is_err());
    let state =
        TransparentBits::verify_commit(&(), geometry, &commitment, &mut transcript).unwrap();
    let mut changed = columns.clone();
    changed[0] += F128::one();
    let wrong = BitsOpening {
        columns: &changed,
        ..opening
    };
    assert!(
        TransparentBits::verify_opening(&(), state, &wrong, &opening_proof, &mut transcript)
            .is_err()
    );
    let mut bytes = Vec::new();
    opening_proof.write(&mut bytes);
    for length in 0..bytes.len() {
        assert!(TransparentOpening::read(&bytes[..length], geometry).is_none());
    }
    bytes.push(0);
    assert!(TransparentOpening::read(&bytes, geometry).is_none());
    let oversized = BitsGeometry { log_T: 21 };
    assert!(matches!(
        TransparentBits::commit(&(), oversized, &witness.bits, &mut transcript),
        Err(TransparentError::Dimension { log_T: 21 })
    ));
    assert!(matches!(
        TransparentBits::verify_commit(&(), oversized, &commitment, &mut transcript),
        Err(TransparentError::Dimension { log_T: 21 })
    ));
    assert!(TransparentOpening::read(&[], oversized).is_none());
}

#[test]
fn witness_reads_follow_the_replayed_pre_state_and_reject_noncanonical_rows() {
    use jolt_rv64i_prover::error::Rv64iProverError;
    use std::collections::BTreeMap;
    let witness = &witness();
    let mut registers = [0_u64; 32];
    let mut ram: BTreeMap<_, _> = witness.initial_ram.iter().copied().collect();
    for (j, bits) in witness.bits.iter().enumerate() {
        let row = &witness.bytecode.rows()[witness.layout.bytecode_index(bits) as usize];
        let next = if j + 1 < witness.bits.len() {
            witness.bytecode.rows()[witness.layout.bytecode_index(&witness.bits[j + 1]) as usize].pc
        } else {
            witness.final_pc
        };
        let base = support::replay::base_words(&witness.layout, row, bits, &registers, &ram, next);
        let store = row.variant.unwrap().is_store();
        assert_eq!(
            witness.words[j].base_words(store, witness.layout.inc(bits)),
            base
        );
        if store {
            let index = witness.layout.ram_index(bits);
            *ram.entry(index).or_default() = base.ram_read_value ^ witness.layout.inc(bits);
        } else {
            registers[usize::from(row.rd)] = base.rd_write_value;
        }
    }
    let (_, _, honest) = support::counting_loop();
    let rebuilt = Rv64iWitness::from_facts(
        honest.layout.clone(),
        Arc::clone(&honest.bytecode),
        &support::counting_loop_facts(),
        honest.initial_ram.clone(),
    )
    .unwrap();
    let initial = State {
        pc: honest.bytecode.rows()[honest.layout.bytecode_index(&honest.bits[0]) as usize].pc,
        ram: honest.initial_ram.iter().copied().collect(),
        ..State::default()
    };
    let replay = support::replay::replay(
        &honest.layout,
        &honest.bytecode,
        &honest.bits,
        &initial,
        honest.final_pc,
    )
    .unwrap();
    assert_eq!(rebuilt.final_pc, replay.final_state.pc);
    for (j, bits) in rebuilt.bits.iter().enumerate() {
        let pre = if j == 0 {
            &initial
        } else {
            &replay.cycles[j - 1]
        };
        let row = &rebuilt.bytecode.rows()[rebuilt.layout.bytecode_index(bits) as usize];
        let expected = support::replay::base_words(
            &rebuilt.layout,
            row,
            bits,
            &pre.registers,
            &pre.ram,
            replay.cycles[j].pc,
        );
        assert_eq!(
            rebuilt.words[j].base_words(row.variant.unwrap().is_store(), rebuilt.layout.inc(bits)),
            expected
        );
    }
    let build = |bits: Arc<[BitsRow]>, initial_ram| {
        Rv64iWitness::from_bits(
            witness.layout.clone(),
            Arc::clone(&witness.bytecode),
            bits,
            initial_ram,
            witness.final_pc,
        )
    };
    assert!(matches!(
        build(vec![[0; 4]; 3].into(), vec![]),
        Err(Rv64iProverError::RowCount { rows: 3 })
    ));
    for memory in [
        vec![(0, 0)],
        vec![(1, 1), (0, 2)],
        vec![(0, 1), (0, 2)],
        vec![(1_u64 << witness.layout.log_K_ram(), 1)],
    ] {
        assert!(matches!(
            build(Arc::clone(&witness.bits), memory),
            Err(Rv64iProverError::InitialRam { .. })
        ));
    }
    let mut invalid = witness.bits.to_vec();
    witness
        .layout
        .write_bytecode_index(&mut invalid[0], 9)
        .unwrap();
    assert!(matches!(
        build(invalid.into(), witness.initial_ram.clone()),
        Err(Rv64iProverError::InvalidBytecode { cycle: 0, index: 9 })
    ));
    let mut malformed = witness.bits.to_vec();
    let descriptor = witness.layout.pos_ra()[0];
    let start = usize::from(descriptor.start());
    malformed[0][start / 64] |= 1_u64 << (start % 64);
    let next = start + 1;
    malformed[0][next / 64] |= 1_u64 << (next % 64);
    assert!(matches!(
        build(malformed.into(), witness.initial_ram.clone()),
        Err(Rv64iProverError::MultipleIndicators { .. })
    ));
}

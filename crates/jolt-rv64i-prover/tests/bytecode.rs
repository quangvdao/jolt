//! Batch-local bytecode checking, address products and the authenticated terminal columns.
#![expect(clippy::unwrap_used, reason = "tests fail on invalid fixtures")]

mod support;
use common::constants::RAM_START_ADDRESS;
use jolt_crypto::NoCommitment;
use jolt_field::{One, Ring, Zero, F128};
use jolt_kernels::ProofSession;
use jolt_poly::Polynomial;
use jolt_prover::driver::{Proved, StageProver};
use jolt_rv64i_arith::{BitsRow, BytecodeRow, Chunk, Layout};
use jolt_rv64i_prover::commitment::{
    transparent::{TransparentBits, TransparentCommitment, TransparentError, TransparentOpening},
    BitsCommitmentProver,
};
use jolt_rv64i_prover::plane::Rv64iWitness;
use jolt_rv64i_prover::stages::stage6a::{Stage6aKernels, Stage6aSumchecks};
use jolt_rv64i_prover::stages::stage6b::{Stage6bKernels, Stage6bSumchecks};
use jolt_rv64i_verifier::claims::bits_reduction::BitsReductionInputClaims;
use jolt_rv64i_verifier::claims::bytecode_read::{
    BytecodeReadAddressInputClaims, BytecodeReadCycleInputClaims,
};
use jolt_rv64i_verifier::claims::ram_ra_product::RamRaProductInputClaims;
use jolt_rv64i_verifier::commitment::{BitsCommitmentScheme, BitsGeometry, BitsOpening, BitsWire};
use jolt_rv64i_verifier::points::{eq_index, next, to_high_to_low};
use jolt_rv64i_verifier::preprocessing::VerifierPreprocessing;
use jolt_rv64i_verifier::proof::{BatchProof, BitsColumns, BytecodeAddressValue};
use jolt_rv64i_verifier::public::bytecode::{BytecodeReadPoints, BytecodeWeights};
use jolt_rv64i_verifier::stages::stage6a::bytecode_read::BytecodeReadAddress;
use jolt_rv64i_verifier::stages::stage6a::verify as verify6a;
use jolt_rv64i_verifier::stages::stage6a::{
    Stage6aInputClaims, Stage6aInputPoints, Stage6aSumchecks as VerifierStage6a,
};
use jolt_rv64i_verifier::stages::stage6b::bits_reduction::BitsReduction;
use jolt_rv64i_verifier::stages::stage6b::bytecode_read_cycle::BytecodeReadCycle;
use jolt_rv64i_verifier::stages::stage6b::ram_ra_product::RamRaProduct;
use jolt_rv64i_verifier::stages::stage6b::verify::{self as verify6b, Stage6bPoints};
use jolt_rv64i_verifier::stages::stage6b::{
    Stage6bInputClaims, Stage6bInputPoints, Stage6bSumchecks as VerifierStage6b,
};
use jolt_rv64i_verifier::statement::CheckedInputs;
use jolt_sumcheck::{ClearSumcheckRecorder, SequentialRounds};
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::sync::Arc;

type BinaryTranscript = Blake2bTranscript<F128>;
type AddressProof = Proved<F128, Stage6aSumchecks<F128>, NoCommitment>;
type TerminalProof = Proved<F128, Stage6bSumchecks<F128>, NoCommitment>;

fn bit(row: &BitsRow, y: usize) -> F128 {
    F128::from_u64((row[y / 64] >> (y % 64)) & 1)
}
fn point(rng: &mut StdRng, n: usize) -> Vec<F128> {
    (0..n).map(|_| F128::from_raw(rng.gen())).collect()
}
fn lift(word: u64, p: &[F128]) -> F128 {
    (0..64)
        .map(|i| eq_index(p, i).unwrap() * F128::from_u64((word >> i) & 1))
        .sum()
}
fn evaluate(values: Vec<F128>, p: &[F128]) -> F128 {
    Polynomial::new(values).evaluate(&to_high_to_low(p))
}
fn cube(j: usize, n: usize) -> Vec<F128> {
    (0..n)
        .map(|i| F128::from_u64(((j >> i) & 1) as u64))
        .collect()
}

struct ProvedFixture {
    address: AddressProof,
    batch: Stage6bSumchecks<F128>,
    inputs: Stage6bInputClaims<F128>,
    input_points: Stage6bInputPoints<F128>,
    proof: TerminalProof,
    transcript: BinaryTranscript,
    opening_proof: TransparentOpening,
    rho: Vec<F128>,
}

struct Fixture {
    witness: Rv64iWitness,
    batch: Stage6aSumchecks<F128>,
    inputs: Stage6aInputClaims<F128>,
    input_points: Stage6aInputPoints<F128>,
    points: BytecodeReadPoints<F128>,
    r1: Vec<F128>,
    w: Vec<F128>,
    x: Vec<F128>,
    a_ram: Vec<F128>,
    entry_pc: u64,
}
impl Fixture {
    fn new() -> Self {
        let (statement, _, source) = support::counting_loop();
        let layout = Layout::new(6, 5, source.layout.lowest_address()).unwrap();
        let words = [
            support::asm::auipc(2, 0),
            support::asm::addi(2, 2, -8),
            support::asm::addi(1, 0, 0),
            support::asm::addi(3, 0, 3),
            support::asm::addi(1, 1, 1),
            support::asm::blt(1, 3, -4),
            support::asm::addi(4, 0, 1),
            support::asm::sd(2, 4, 0),
            support::asm::jal(0, 0),
        ];
        let program: Vec<_> = words
            .into_iter()
            .enumerate()
            .map(|(i, word)| (RAM_START_ADDRESS + 4 * i as u64, word))
            .collect();
        let bytecode = Arc::new(support::harness::bytecode(&program, &layout));
        let witness = Rv64iWitness::synthetic(
            0x0641_0b17,
            layout.clone(),
            bytecode,
            &statement.device.memory_layout,
            source.initial_ram.clone(),
            5,
        )
        .unwrap();
        let entry_pc = witness.bytecode.rows()[layout.bytecode_index(&witness.bits[0]) as usize].pc;
        let mut rng = StdRng::seed_from_u64(0x0060_1bca);
        let x = point(&mut rng, 17);
        let points = BytecodeReadPoints {
            r_bit: x[..6].to_vec(),
            q_variant: x[11..17].to_vec(),
            q_shift: x[12..15].to_vec(),
            q_access: x[13..17].to_vec(),
            q_key: x[14..17].to_vec(),
            a_reg: point(&mut rng, 5),
            r_3: point(&mut rng, 5),
            r_4: point(&mut rng, 5),
            r_5: point(&mut rng, 5),
        };
        let values = Self::claims(&witness, &points);
        let relation =
            BytecodeReadAddress::new(6, points.clone(), entry_pc, witness.final_pc).unwrap();
        let input_points = Stage6aInputPoints {
            bytecode_read_address: relation.input_points(),
        };
        Self {
            witness,
            batch: Stage6aSumchecks(VerifierStage6a {
                bytecode_read_address: relation,
            }),
            inputs: Stage6aInputClaims {
                bytecode_read_address: values,
            },
            input_points,
            points,
            r1: point(&mut rng, 5),
            w: point(&mut rng, 10),
            x,
            a_ram: point(&mut rng, 5),
            entry_pc,
        }
    }
    fn claims(
        witness: &Rv64iWitness,
        p: &BytecodeReadPoints<F128>,
    ) -> BytecodeReadAddressInputClaims<F128> {
        let mut values = [F128::zero(); 15];
        for (j, row) in witness.bits.iter().enumerate() {
            let fetched = &witness.bytecode.rows()[witness.layout.bytecode_index(row) as usize];
            let f = functions(fetched, p);
            for i in 0..14 {
                values[i] += eq_index(
                    if i < 9 {
                        &p.r_3
                    } else if i < 12 {
                        &p.r_4
                    } else {
                        &p.r_5
                    },
                    j,
                )
                .unwrap()
                    * f[i];
            }
            values[14] += eq_index(&p.r_3, j).unwrap() * lift(witness.words[j].next_pc, &p.r_bit);
        }
        let [imm, fall_through_pc, pc_plus_imm, pc, variant, shift_kind, access_kind, key_kind, branch, rs1_ra, rs2_ra, rd_wa_read, rd_wa_write, store, next_pc] =
            values;
        BytecodeReadAddressInputClaims {
            imm,
            fall_through_pc,
            pc_plus_imm,
            pc,
            next_pc,
            variant,
            shift_kind,
            access_kind,
            key_kind,
            branch,
            rs1_ra,
            rs2_ra,
            rd_wa_read,
            rd_wa_write,
            store,
        }
    }
    fn transcript(&self) -> (BinaryTranscript, TransparentCommitment) {
        let mut transcript = BinaryTranscript::new(b"rv64i-bytecode-batches");
        let (commitment, _) = TransparentBits::commit(
            &(),
            BitsGeometry { log_T: 5 },
            &self.witness.bits,
            &mut transcript,
        )
        .unwrap();
        (transcript, commitment)
    }
    fn terminal(
        &self,
        a_bc: Vec<F128>,
        h: [F128; 5],
        address_claim: F128,
    ) -> (
        Stage6bSumchecks<F128>,
        Stage6bInputClaims<F128>,
        Stage6bInputPoints<F128>,
    ) {
        let cycle = BytecodeReadCycle::new(
            &self.witness.layout,
            h,
            a_bc,
            self.points.r_3.clone(),
            self.points.r_4.clone(),
            self.points.r_5.clone(),
        )
        .unwrap();
        let ram = RamRaProduct::new(
            &self.witness.layout,
            self.a_ram.clone(),
            self.points.r_4.clone(),
            self.points.r_5.clone(),
        )
        .unwrap();
        let reduction = BitsReduction::new(
            &self.witness.layout,
            self.r1.clone(),
            self.points.r_3.clone(),
            self.points.r_5.clone(),
            self.w.clone(),
            self.x.clone(),
        )
        .unwrap();
        let input_points = Stage6bInputPoints {
            bytecode_read_cycle: cycle.input_points(),
            ram_ra_product: ram.input_points(),
            bits_reduction: reduction.input_points(),
        };
        let mut values = [F128::zero(); 6];
        let mut ram_values = [F128::zero(); 2];
        for (j, row) in self.witness.bits.iter().enumerate() {
            let linear = definitions(&self.witness.layout, &self.w, &self.x, row);
            for u in 0..6 {
                values[u] += eq_index(
                    match u {
                        0 => &self.r1,
                        5 => &self.points.r_5,
                        _ => &self.points.r_3,
                    },
                    j,
                )
                .unwrap()
                    * linear[u];
            }
            for (u, p) in [&self.points.r_4, &self.points.r_5].into_iter().enumerate() {
                ram_values[u] += eq_index(p, j).unwrap()
                    * eq_index(&self.a_ram, self.witness.layout.ram_index(row) as usize).unwrap();
            }
        }
        let [direct_columns, variant_bits, pos_ra_0, pos_ra_1, should_branch, inc] = values;
        (
            Stage6bSumchecks(VerifierStage6b {
                bytecode_read_cycle: cycle,
                ram_ra_product: ram,
                bits_reduction: reduction,
            }),
            Stage6bInputClaims {
                bytecode_read_cycle: BytecodeReadCycleInputClaims { address_claim },
                ram_ra_product: RamRaProductInputClaims {
                    ram_ra_read: ram_values[0],
                    ram_ra_val: ram_values[1],
                },
                bits_reduction: BitsReductionInputClaims {
                    direct_columns,
                    variant_bits,
                    pos_ra_0,
                    pos_ra_1,
                    should_branch,
                    inc,
                },
            },
            input_points,
        )
    }
    fn prove(&self) -> ProvedFixture {
        let (mut transcript, _) = self.transcript();
        let challenges = self.batch.draw_challenges(&mut transcript).unwrap();
        let address = self
            .batch
            .prove(
                &Stage6aKernels::default(),
                &mut ProofSession::default(),
                &mut SequentialRounds,
                &self.witness,
                &self.inputs,
                &self.input_points,
                &challenges,
                ClearSumcheckRecorder::<F128, NoCommitment>::new(),
                &mut transcript,
            )
            .unwrap();
        let a_bc = &address.output_points.bytecode_read_address.address_claim;
        let h = BytecodeWeights::new(&self.points, &challenges.bytecode_read_address)
            .unwrap()
            .evaluate(&self.witness.bytecode, a_bc)
            .unwrap();
        let (batch, inputs, input_points) = self.terminal(
            a_bc.clone(),
            h,
            address.output_claims.bytecode_read_address.address_claim,
        );
        let challenges = batch.draw_challenges(&mut transcript).unwrap();
        let proof = batch
            .prove(
                &Stage6bKernels::default(),
                &mut ProofSession::default(),
                &mut SequentialRounds,
                &self.witness,
                &inputs,
                &input_points,
                &challenges,
                ClearSumcheckRecorder::<F128, NoCommitment>::new(),
                &mut transcript,
            )
            .unwrap();
        let rho = transcript.challenge_vector(8);
        let opening = BitsOpening {
            geometry: BitsGeometry { log_T: 5 },
            column_point: &rho,
            cycle_point: &proof.output_points.bits_reduction.columns[0],
            columns: &proof.output_claims.bits_reduction.columns,
        };
        let mut twin = BinaryTranscript::new(b"rv64i-bytecode-batches");
        let (_, state) =
            TransparentBits::commit(&(), opening.geometry, &self.witness.bits, &mut twin).unwrap();
        let opening_proof = TransparentBits::open(&(), state, &opening, &mut transcript).unwrap();
        ProvedFixture {
            address,
            batch,
            inputs,
            input_points,
            proof,
            transcript,
            opening_proof,
            rho,
        }
    }
}

fn functions(row: &BytecodeRow, p: &BytecodeReadPoints<F128>) -> [F128; 16] {
    let Some(v) = row.variant else {
        return [F128::zero(); 16];
    };
    [
        lift(row.imm, &p.r_bit),
        lift(row.fall_through_pc, &p.r_bit),
        lift(row.pc_plus_imm, &p.r_bit),
        lift(row.pc, &p.r_bit),
        eq_index(&p.q_variant, v.index()).unwrap(),
        v.shift().map_or(F128::zero(), |s| {
            eq_index(&p.q_shift, s.kind.index()).unwrap()
        }),
        v.access()
            .and_then(|a| a.kind)
            .map_or(F128::zero(), |a| eq_index(&p.q_access, a.index()).unwrap()),
        v.key_kind()
            .map_or(F128::zero(), |k| eq_index(&p.q_key, k.index()).unwrap()),
        F128::from_u64(u64::from(v.branch().is_some())),
        eq_index(&p.a_reg, usize::from(row.rs1)).unwrap(),
        eq_index(&p.a_reg, usize::from(row.rs2)).unwrap(),
        eq_index(&p.a_reg, usize::from(row.rd)).unwrap(),
        eq_index(&p.a_reg, usize::from(row.rd)).unwrap(),
        F128::from_u64(u64::from(v.is_store())),
        lift(row.pc, &p.r_bit),
        lift(row.pc, &p.r_bit),
    ]
}
fn cycle_weights(p: &BytecodeReadPoints<F128>, j: usize) -> [F128; 5] {
    [
        eq_index(&p.r_3, j).unwrap(),
        eq_index(&p.r_4, j).unwrap(),
        eq_index(&p.r_5, j).unwrap(),
        F128::from_u64(u64::from(j == 0)),
        next(&p.r_3, &cube(j, p.r_3.len())).unwrap(),
    ]
}
fn chunk_value(desc: Chunk, p: &[F128], row: &BitsRow) -> F128 {
    let zero = eq_index(p, 0).unwrap();
    zero + (1..=desc.indicators())
        .map(|k| (eq_index(p, k).unwrap() + zero) * bit(row, usize::from(desc.start()) + k - 1))
        .sum::<F128>()
}

/// Direct per-cycle evaluations of the six committed functionals.
fn definitions(layout: &Layout, w: &[F128], x: &[F128], row: &BitsRow) -> [F128; 6] {
    let mut values = [F128::zero(); 6];
    for y in 64..=layout.keys_differ() {
        values[0] += eq_index(w, 768 + y).unwrap() * bit(row, y);
    }
    for y in 0..64 {
        let v = eq_index(&x[..6], y).unwrap() * bit(row, y);
        values[1] += eq_index(&x[6..10], 8).unwrap() * v;
        values[5] += v;
    }
    let g = usize::from(layout.ram_ra()[0].start());
    for y in g..layout.used_columns() {
        values[1] += eq_index(&x[..10], 576 + y - g).unwrap() * bit(row, y);
    }
    for (d, ch) in layout.pos_ra().into_iter().enumerate() {
        let p = &x[6 + 3 * d..9 + 3 * d];
        values[2 + d] = chunk_value(ch, p, row);
    }
    values[4] = bit(row, layout.should_branch());
    values
}

#[test]
fn bytecode_batches_match_definitions_and_terminal_opening() {
    let f = Fixture::new();
    let ProvedFixture {
        address,
        batch,
        inputs,
        input_points,
        proof,
        transcript: prover,
        opening_proof,
        rho,
    } = f.prove();
    let (mut transcript, commitment) = f.transcript();
    let mut twin = BinaryTranscript::new(b"rv64i-bytecode-batches");
    let state =
        TransparentBits::verify_commit(&(), BitsGeometry { log_T: 5 }, &commitment, &mut twin)
            .unwrap();
    let ac = f.batch.draw_challenges(&mut transcript).unwrap();
    let c = &ac.bytecode_read_address;
    let coefficients = [
        c.imm,
        c.fall_through_pc,
        c.pc_plus_imm,
        c.pc,
        c.variant,
        c.shift_kind,
        c.access_kind,
        c.key_kind,
        c.branch,
        c.rs1_ra,
        c.rs2_ra,
        c.rd_wa_read,
        c.rd_wa_write,
        c.store,
        c.entry,
        c.next,
    ];
    let mut h: [Vec<F128>; 5] = std::array::from_fn(|_| vec![F128::zero(); 64]);
    let mut r: [Vec<F128>; 5] = std::array::from_fn(|_| vec![F128::zero(); 64]);
    for (k, row) in f.witness.bytecode.rows().iter().enumerate() {
        for (u, value) in functions(row, &f.points).into_iter().enumerate() {
            let group = match u {
                0..=8 => 0,
                9..=11 => 1,
                12..=13 => 2,
                14 => 3,
                _ => 4,
            };
            h[group][k] += coefficients[u] * value;
        }
    }
    for (j, bits) in f.witness.bits.iter().enumerate() {
        let k = f.witness.layout.bytecode_index(bits) as usize;
        for (g, weight) in cycle_weights(&f.points, j).into_iter().enumerate() {
            r[g][k] += weight;
        }
    }
    let sum: F128 = (0..5)
        .flat_map(|g| (0..64).map(move |k| (g, k)))
        .map(|(g, k)| h[g][k] * r[g][k])
        .sum();
    assert_eq!(
        f.batch
            .bytecode_read_address
            .input_claim(&f.inputs.bytecode_read_address, c)
            .unwrap(),
        sum
    );
    let a_bc = &address.output_points.bytecode_read_address.address_claim;
    let h_values: [F128; 5] = std::array::from_fn(|g| evaluate(h[g].clone(), a_bc));
    let terminal: F128 = (0..5)
        .map(|g| h_values[g] * evaluate(r[g].clone(), a_bc))
        .sum();
    assert_eq!(
        terminal,
        address.output_claims.bytecode_read_address.address_claim
    );
    assert_eq!(
        f.batch
            .bytecode_read_address
            .expected_output(
                &f.input_points.bytecode_read_address,
                &address.output_claims.bytecode_read_address,
                &address.output_points.bytecode_read_address,
                c
            )
            .unwrap(),
        terminal
    );
    assert_eq!(
        h_values,
        BytecodeWeights::new(&f.points, c)
            .unwrap()
            .evaluate(&f.witness.bytecode, a_bc)
            .unwrap()
    );
    let _ = f
        .batch
        .verify_clear(
            &f.inputs,
            &f.input_points,
            &ac,
            &address.output_claims,
            &address.recorded.proof,
            &mut transcript,
            0,
        )
        .unwrap();
    f.batch
        .append_output_claims(&mut transcript, &address.output_claims);
    let challenges = batch.draw_challenges(&mut transcript).unwrap();
    let points = batch
        .verify_clear(
            &inputs,
            &input_points,
            &challenges,
            &proof.output_claims,
            &proof.recorded.proof,
            &mut transcript,
            0,
        )
        .unwrap();
    batch.append_output_claims(&mut transcript, &proof.output_claims);
    assert_eq!(rho, transcript.challenge_vector(8));
    let cycle = &points.bits_reduction.columns[0];
    let columns = &proof.output_claims.bits_reduction.columns;
    for (y, value) in columns.iter().enumerate() {
        assert_eq!(
            *value,
            evaluate(
                f.witness.bits.iter().map(|row| bit(row, y)).collect(),
                cycle
            )
        );
    }
    let opening = BitsOpening {
        geometry: BitsGeometry { log_T: 5 },
        column_point: &rho,
        cycle_point: cycle,
        columns,
    };
    TransparentBits::verify_opening(&(), state, &opening, &opening_proof, &mut transcript).unwrap();
    assert_eq!(transcript.state(), prover.state());
    let (mut statement, source_preprocessing, _) = support::counting_loop();
    statement.log_T = 5;
    statement.entry_pc = f.entry_pc;
    let preprocessing = VerifierPreprocessing::<TransparentBits>::new(
        (*f.witness.bytecode).clone(),
        source_preprocessing.image().to_vec(),
        (),
    )
    .unwrap();
    let checked =
        CheckedInputs::of_statement(&preprocessing, &statement, 5, f.witness.final_pc).unwrap();
    let mut stage_transcript = BinaryTranscript::new(b"rv64i-bytecode-batches");
    let state =
        TransparentBits::verify_commit(&(), opening.geometry, &commitment, &mut stage_transcript)
            .unwrap();
    let previous = verify6a::verify(
        &checked,
        f.points.clone(),
        &f.inputs,
        &BatchProof {
            rounds: address.recorded.proof.clone(),
            values: BytecodeAddressValue {
                address_claim: address.output_claims.bytecode_read_address.address_claim,
            },
        },
        &mut stage_transcript,
    )
    .unwrap();
    let final_output = verify6b::verify(
        &checked,
        &BatchProof {
            rounds: proof.recorded.proof.clone(),
            values: BitsColumns(columns.clone()),
        },
        &mut stage_transcript,
        &previous,
        Stage6bPoints {
            r_1: f.r1.clone(),
            r_3: f.points.r_3.clone(),
            r_4: f.points.r_4.clone(),
            r_5: f.points.r_5.clone(),
            w: f.w.clone(),
            x: f.x.clone(),
            a_ram: f.a_ram.clone(),
        },
        &inputs,
        state,
        &opening_proof,
    )
    .unwrap();
    assert_eq!(final_output.point, *cycle);
    assert_eq!(stage_transcript.state(), prover.state());
    let projected = batch.expand(columns).unwrap();
    assert_eq!(
        projected.bytecode_read_cycle,
        proof.output_claims.bytecode_read_cycle
    );
    assert_eq!(projected.ram_ra_product, proof.output_claims.ram_ra_product);
    let full: Vec<_> = rho.iter().chain(cycle).copied().collect();
    assert_eq!(
        opening.value(),
        evaluate(
            f.witness
                .bits
                .iter()
                .flat_map(|r| (0..256).map(move |y| bit(r, y)))
                .collect(),
            &full
        )
    );
    let cycle_weight: Vec<_> = (0..32)
        .map(|j| {
            cycle_weights(&f.points, j)
                .into_iter()
                .zip(h_values)
                .map(|(e, h)| e * h)
                .sum()
        })
        .collect();
    let chunk_tables = |descriptors: &[Chunk], p: &[F128]| {
        let mut offset = 0;
        descriptors
            .iter()
            .map(|desc| {
                let end = offset + usize::from(desc.bits());
                let values = f
                    .witness
                    .bits
                    .iter()
                    .map(|row| chunk_value(*desc, &p[offset..end], row))
                    .collect::<Vec<_>>();
                offset = end;
                values
            })
            .collect::<Vec<_>>()
    };
    let bc_tables = chunk_tables(f.witness.layout.bytecode_ra(), a_bc);
    let bc_sum: F128 = (0..32)
        .map(|j| cycle_weight[j] * bc_tables.iter().map(|v| v[j]).product::<F128>())
        .sum();
    assert_eq!(
        batch
            .bytecode_read_cycle
            .input_claim(&inputs.bytecode_read_cycle, &challenges.bytecode_read_cycle)
            .unwrap(),
        bc_sum
    );
    let bc_terminal = evaluate(cycle_weight, cycle)
        * bc_tables
            .iter()
            .map(|v| evaluate(v.clone(), cycle))
            .product::<F128>();
    assert_eq!(
        batch
            .bytecode_read_cycle
            .expected_output(
                &input_points.bytecode_read_cycle,
                &proof.output_claims.bytecode_read_cycle,
                &points.bytecode_read_cycle,
                &challenges.bytecode_read_cycle
            )
            .unwrap(),
        bc_terminal
    );
    let ram_tables = chunk_tables(f.witness.layout.ram_ra(), &f.a_ram);
    let rc = &challenges.ram_ra_product;
    let ram_weights: Vec<_> = (0..32)
        .map(|j| {
            rc.read * eq_index(&f.points.r_4, j).unwrap()
                + rc.val * eq_index(&f.points.r_5, j).unwrap()
        })
        .collect();
    let ram_sum: F128 = (0..32)
        .map(|j| ram_weights[j] * ram_tables.iter().map(|v| v[j]).product::<F128>())
        .sum();
    assert_eq!(
        batch
            .ram_ra_product
            .input_claim(&inputs.ram_ra_product, rc)
            .unwrap(),
        ram_sum
    );
    let ram_terminal = evaluate(ram_weights, cycle)
        * ram_tables
            .iter()
            .map(|v| evaluate(v.clone(), cycle))
            .product::<F128>();
    assert_eq!(
        batch
            .ram_ra_product
            .expected_output(
                &input_points.ram_ra_product,
                &proof.output_claims.ram_ra_product,
                &points.ram_ra_product,
                rc
            )
            .unwrap(),
        ram_terminal
    );
    let c = &challenges.bits_reduction;
    let coefficients = [
        c.direct_columns,
        c.variant_bits,
        c.pos_ra_0,
        c.pos_ra_1,
        c.should_branch,
        c.inc,
    ];
    let mut reduction_terminal = F128::zero();
    let mut reduction_sum = F128::zero();
    let constants = definitions(&f.witness.layout, &f.w, &f.x, &[0; 4]);
    for y in 0..256 {
        let mut unit = [0; 4];
        unit[y / 64] = 1_u64 << (y % 64);
        let l = definitions(&f.witness.layout, &f.w, &f.x, &unit);
        let weights: Vec<F128> = (0..32)
            .map(|j| {
                (0..6)
                    .map(|u| {
                        coefficients[u]
                            * (l[u] + constants[u])
                            * eq_index(
                                match u {
                                    0 => &f.r1,
                                    5 => &f.points.r_5,
                                    _ => &f.points.r_3,
                                },
                                j,
                            )
                            .unwrap()
                    })
                    .sum()
            })
            .collect();
        reduction_sum += weights
            .iter()
            .zip(f.witness.bits.iter())
            .map(|(w, row)| *w * bit(row, y))
            .sum::<F128>();
        reduction_terminal += evaluate(weights, cycle) * columns[y];
    }
    assert_eq!(
        batch
            .bits_reduction
            .input_claim(&inputs.bits_reduction, c)
            .unwrap(),
        reduction_sum
    );
    assert_eq!(
        batch
            .bits_reduction
            .expected_output(
                &input_points.bits_reduction,
                &proof.output_claims.bits_reduction,
                &points.bits_reduction,
                c
            )
            .unwrap(),
        reduction_terminal
    );
}

#[test]
fn bytecode_batches_reject_each_input_public_pc_and_terminal_column() {
    let f = Fixture::new();
    let ProvedFixture {
        address,
        batch,
        inputs,
        input_points,
        proof,
        ..
    } = f.prove();
    for (entry, final_pc) in [
        (f.entry_pc + 4, f.witness.final_pc),
        (f.entry_pc, f.witness.final_pc + 4),
    ] {
        let altered = Stage6aSumchecks(VerifierStage6a {
            bytecode_read_address: BytecodeReadAddress::new(6, f.points.clone(), entry, final_pc)
                .unwrap(),
        });
        let (mut transcript, _) = f.transcript();
        let challenges = altered.draw_challenges(&mut transcript).unwrap();
        assert!(altered
            .verify_clear(
                &f.inputs,
                &f.input_points,
                &challenges,
                &address.output_claims,
                &address.recorded.proof,
                &mut transcript,
                0
            )
            .is_err());
    }
    for u in 0..15 {
        let (mut transcript, _) = f.transcript();
        let challenges = f.batch.draw_challenges(&mut transcript).unwrap();
        let mut changed = f.inputs.clone();
        let v = &mut changed.bytecode_read_address;
        let cells = [
            &mut v.imm,
            &mut v.fall_through_pc,
            &mut v.pc_plus_imm,
            &mut v.pc,
            &mut v.next_pc,
            &mut v.variant,
            &mut v.shift_kind,
            &mut v.access_kind,
            &mut v.key_kind,
            &mut v.branch,
            &mut v.rs1_ra,
            &mut v.rs2_ra,
            &mut v.rd_wa_read,
            &mut v.rd_wa_write,
            &mut v.store,
        ];
        *cells.into_iter().nth(u).unwrap() += F128::one();
        assert!(f
            .batch
            .verify_clear(
                &changed,
                &f.input_points,
                &challenges,
                &address.output_claims,
                &address.recorded.proof,
                &mut transcript,
                0
            )
            .is_err());
    }
    for y in 0..256 {
        let (mut transcript, _) = f.transcript();
        let challenges = f.batch.draw_challenges(&mut transcript).unwrap();
        let _ = f
            .batch
            .verify_clear(
                &f.inputs,
                &f.input_points,
                &challenges,
                &address.output_claims,
                &address.recorded.proof,
                &mut transcript,
                0,
            )
            .unwrap();
        f.batch
            .append_output_claims(&mut transcript, &address.output_claims);
        let challenges = batch.draw_challenges(&mut transcript).unwrap();
        let mut columns = proof.output_claims.bits_reduction.columns.clone();
        columns[y] += F128::one();
        let outputs = batch.expand(&columns).unwrap();
        let result = batch.verify_clear(
            &inputs,
            &input_points,
            &challenges,
            &outputs,
            &proof.recorded.proof,
            &mut transcript,
            0,
        );
        if result.is_ok() {
            batch.append_output_claims(&mut transcript, &outputs);
            let rho = transcript.challenge_vector(8);
            let opening = BitsOpening {
                geometry: BitsGeometry { log_T: 5 },
                column_point: &rho,
                cycle_point: &proof.output_points.bits_reduction.columns[0],
                columns: &columns,
            };
            let mut twin = BinaryTranscript::new(b"rv64i-bytecode-batches");
            let (_, commitment) = f.transcript();
            let state =
                TransparentBits::verify_commit(&(), opening.geometry, &commitment, &mut twin)
                    .unwrap();
            assert!(TransparentBits::verify_opening(
                &(),
                state,
                &opening,
                &TransparentOpening(Arc::clone(&f.witness.bits)),
                &mut transcript
            )
            .is_err());
        }
    }
    for length in [255, 257] {
        assert!(batch.expand(&vec![F128::zero(); length]).is_err());
    }
    for u in 0..9 {
        let (mut transcript, _) = f.transcript();
        let ac = f.batch.draw_challenges(&mut transcript).unwrap();
        let _ = f
            .batch
            .verify_clear(
                &f.inputs,
                &f.input_points,
                &ac,
                &address.output_claims,
                &address.recorded.proof,
                &mut transcript,
                0,
            )
            .unwrap();
        f.batch
            .append_output_claims(&mut transcript, &address.output_claims);
        let challenges = batch.draw_challenges(&mut transcript).unwrap();
        let mut changed = inputs.clone();
        let cells = [
            &mut changed.bits_reduction.direct_columns,
            &mut changed.bits_reduction.variant_bits,
            &mut changed.bits_reduction.pos_ra_0,
            &mut changed.bits_reduction.pos_ra_1,
            &mut changed.bits_reduction.should_branch,
            &mut changed.bits_reduction.inc,
            &mut changed.ram_ra_product.ram_ra_read,
            &mut changed.ram_ra_product.ram_ra_val,
            &mut changed.bytecode_read_cycle.address_claim,
        ];
        *cells.into_iter().nth(u).unwrap() += F128::one();
        assert!(batch
            .verify_clear(
                &changed,
                &input_points,
                &challenges,
                &proof.output_claims,
                &proof.recorded.proof,
                &mut transcript,
                0
            )
            .is_err());
    }
}

#[test]
fn batch6b_draws_reduction_coefficients_before_ram_coefficients() {
    let f = Fixture::new();
    let (batch, _, _) = f.terminal(vec![F128::zero(); 6], [F128::zero(); 5], F128::zero());
    let mut transcript = BinaryTranscript::new(b"batch6b-draw-order");
    let mut twin = BinaryTranscript::new(b"batch6b-draw-order");
    let challenges = batch.draw_challenges(&mut transcript).unwrap();
    let b = challenges.bits_reduction;
    let actual = [
        b.direct_columns,
        b.variant_bits,
        b.pos_ra_0,
        b.pos_ra_1,
        b.should_branch,
        b.inc,
        challenges.ram_ra_product.read,
        challenges.ram_ra_product.val,
    ];
    let expected: [F128; 8] = std::array::from_fn(|_| twin.challenge_scalar());
    assert_eq!(actual, expected);
    assert_eq!(transcript.state(), twin.state());
}

#[test]
fn transparent_scheme_rejects_table_column_lengths_and_oversized_geometry() {
    let fixture = Fixture::new();
    let ProvedFixture {
        proof,
        opening_proof,
        rho,
        ..
    } = fixture.prove();
    let cycle = &proof.output_points.bits_reduction.columns[0];
    let columns = &proof.output_claims.bits_reduction.columns;
    let geometry = BitsGeometry { log_T: 5 };
    let (_, commitment) = fixture.transcript();
    let opening = BitsOpening {
        geometry,
        column_point: &rho,
        cycle_point: cycle,
        columns,
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
        TransparentBits::commit(&(), oversized, &fixture.witness.bits, &mut transcript),
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
    let fixture = Fixture::new();
    let witness = &fixture.witness;
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
    let initial = support::replay::State {
        pc: honest.bytecode.rows()[honest.layout.bytecode_index(&honest.bits[0]) as usize].pc,
        ram: honest.initial_ram.iter().copied().collect(),
        ..support::replay::State::default()
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

#[test]
fn batch6b_rejects_column_point_drawn_before_column_absorption() {
    let f = Fixture::new();
    let ProvedFixture {
        address,
        batch,
        inputs,
        input_points,
        proof,
        ..
    } = f.prove();
    let (mut early, _) = f.transcript();
    let ac = f.batch.draw_challenges(&mut early).unwrap();
    let _ = f
        .batch
        .verify_clear(
            &f.inputs,
            &f.input_points,
            &ac,
            &address.output_claims,
            &address.recorded.proof,
            &mut early,
            0,
        )
        .unwrap();
    f.batch
        .append_output_claims(&mut early, &address.output_claims);
    let challenges = batch.draw_challenges(&mut early).unwrap();
    let _ = batch
        .verify_clear(
            &inputs,
            &input_points,
            &challenges,
            &proof.output_claims,
            &proof.recorded.proof,
            &mut early,
            0,
        )
        .unwrap();
    let early_rho = early.challenge_vector(8);
    batch.append_output_claims(&mut early, &proof.output_claims);
    let (mut correct, commitment) = f.transcript();
    let ac = f.batch.draw_challenges(&mut correct).unwrap();
    let _ = f
        .batch
        .verify_clear(
            &f.inputs,
            &f.input_points,
            &ac,
            &address.output_claims,
            &address.recorded.proof,
            &mut correct,
            0,
        )
        .unwrap();
    f.batch
        .append_output_claims(&mut correct, &address.output_claims);
    let challenges = batch.draw_challenges(&mut correct).unwrap();
    let _ = batch
        .verify_clear(
            &inputs,
            &input_points,
            &challenges,
            &proof.output_claims,
            &proof.recorded.proof,
            &mut correct,
            0,
        )
        .unwrap();
    batch.append_output_claims(&mut correct, &proof.output_claims);
    let rho = correct.challenge_vector(8);
    assert_ne!(early_rho, rho);
    // A degree-one difference can vanish at a premature opening point.
    let delta: Vec<F128> = (0..256)
        .map(|y| F128::from_u64((y & 1) as u64) + early_rho[0])
        .collect();
    assert_eq!(evaluate(delta.clone(), &early_rho), F128::zero());
    assert_ne!(evaluate(delta.clone(), &rho), F128::zero());
    let columns: Vec<_> = proof
        .output_claims
        .bits_reduction
        .columns
        .iter()
        .zip(delta)
        .map(|(c, d)| *c + d)
        .collect();
    let opening = BitsOpening {
        geometry: BitsGeometry { log_T: 5 },
        column_point: &rho,
        cycle_point: &proof.output_points.bits_reduction.columns[0],
        columns: &columns,
    };
    let mut twin = BinaryTranscript::new(b"rv64i-bytecode-batches");
    let state =
        TransparentBits::verify_commit(&(), opening.geometry, &commitment, &mut twin).unwrap();
    assert!(TransparentBits::verify_opening(
        &(),
        state,
        &opening,
        &TransparentOpening(Arc::clone(&f.witness.bits)),
        &mut correct
    )
    .is_err());
}

#[test]
fn stage6_schedule_matches_reference_layout_literals() {
    for (a, degree) in [(20, 6), (23, 7)] {
        let layout = Layout::new(20, a, 0).unwrap();
        let p = BytecodeReadPoints {
            r_bit: vec![F128::zero(); 6],
            q_variant: vec![F128::zero(); 6],
            q_shift: vec![F128::zero(); 3],
            q_access: vec![F128::zero(); 4],
            q_key: vec![F128::zero(); 3],
            a_reg: vec![F128::zero(); 5],
            r_3: vec![F128::zero(); 22],
            r_4: vec![F128::zero(); 22],
            r_5: vec![F128::zero(); 22],
        };
        let address = BytecodeReadAddress::new(20, p.clone(), 0, 0).unwrap();
        assert_eq!((address.rounds(), address.degree()), (20, 2));
        let cycle = BytecodeReadCycle::new(
            &layout,
            [F128::zero(); 5],
            vec![F128::zero(); 20],
            p.r_3.clone(),
            p.r_4.clone(),
            p.r_5.clone(),
        )
        .unwrap();
        let ram = RamRaProduct::new(&layout, vec![F128::zero(); a], p.r_4, p.r_5).unwrap();
        let reduction = BitsReduction::new(
            &layout,
            vec![F128::zero(); 22],
            vec![F128::zero(); 22],
            vec![F128::zero(); 22],
            vec![F128::zero(); 10],
            vec![F128::zero(); 17],
        )
        .unwrap();
        let batch = VerifierStage6b {
            bytecode_read_cycle: cycle,
            ram_ra_product: ram,
            bits_reduction: reduction,
        };
        assert_eq!(batch.bytecode_read_cycle.rounds(), 22);
        assert_eq!(batch.ram_ra_product.rounds(), 22);
        assert_eq!(batch.bits_reduction.rounds(), 22);
        assert_eq!(batch.bytecode_read_cycle.degree(), 6);
        assert_eq!(batch.ram_ra_product.degree(), degree);
        assert_eq!(batch.bits_reduction.degree(), 2);
        for offset in [
            batch.bytecode_read_cycle.instance_point_offset(22),
            batch.ram_ra_product.instance_point_offset(22),
            batch.bits_reduction.instance_point_offset(22),
        ] {
            assert_eq!(offset.unwrap(), 0);
        }
        let inputs = Stage6bInputClaims {
            bytecode_read_cycle: BytecodeReadCycleInputClaims::default(),
            ram_ra_product: RamRaProductInputClaims::default(),
            bits_reduction: BitsReductionInputClaims::default(),
        };
        let mut transcript = BinaryTranscript::new(b"stage6-schedule");
        let challenges = batch.draw_challenges(&mut transcript).unwrap();
        let (prelude, _) = batch
            .begin_batch(
                &inputs,
                &challenges,
                &mut ClearSumcheckRecorder::<F128, NoCommitment>::new(),
                &mut transcript,
            )
            .unwrap();
        assert_eq!((prelude.max_num_vars, prelude.max_degree), (22, degree));
        assert_eq!(
            prelude
                .members
                .iter()
                .map(|m| (m.rounds, m.offset))
                .collect::<Vec<_>>(),
            [(22, 0), (22, 0), (22, 0)]
        );
        let output = batch.expand(&vec![F128::zero(); 256]).unwrap();
        assert_eq!(batch.opening_values(&output).len(), 256);
    }
}

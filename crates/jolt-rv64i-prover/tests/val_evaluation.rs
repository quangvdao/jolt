//! Stage 5 evaluates replayed pre-state and final-state tables at seeded points.
#![expect(
    clippy::unwrap_used,
    reason = "invalid fixtures fail the enclosing test"
)]

#[expect(
    dead_code,
    reason = "batch-local tests share the full machine corpus helpers"
)]
mod support;
use common::jolt_device::{MemoryConfig, MemoryLayout};
use jolt_claims::NoChallenges;
use jolt_crypto::NoCommitment;
use jolt_field::{One, Ring, Zero, F128};
use jolt_kernels::ProofSession;
use jolt_poly::Polynomial;
use jolt_prover::driver::{Proved, StageProver};
use jolt_rv64i_arith::Layout;
use jolt_rv64i_prover::commitment::transparent::TransparentBits;
use jolt_rv64i_prover::{
    plane::Rv64iWitness,
    stages::stage5::{Stage5Kernels, Stage5Sumchecks},
};
use jolt_rv64i_verifier::claims::val_evaluation::{
    RamValEvaluationInputClaims, RegistersValEvaluationInputClaims,
};
use jolt_rv64i_verifier::stages::stage5::{
    verify, Stage5InputClaims, Stage5InputPoints, Stage5Sumchecks as VerifierStage5Sumchecks,
};
use jolt_rv64i_verifier::{
    points::to_high_to_low,
    preprocessing::VerifierPreprocessing,
    proof::{BatchProof, ReadCheckingValues, ValEvaluationValues},
    public::ram_init,
    statement::{CheckedInputs, Statement},
};
use jolt_sumcheck::{ClearSumcheckRecorder, SequentialRounds};
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::sync::Arc;
use support::replay::{self, State};

type BinaryTranscript = Blake2bTranscript<F128>;
type StageProof = Proved<F128, Stage5Sumchecks<F128>, NoCommitment>;

fn basis(point: &[F128], index: usize) -> F128 {
    point
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if (index >> i) & 1 == 1 {
                *p
            } else {
                F128::one() + *p
            }
        })
        .product()
}
fn evaluate(table: Vec<F128>, point: &[F128]) -> F128 {
    Polynomial::new(table).evaluate(&to_high_to_low(point))
}
fn word(value: u64, point: &[F128]) -> F128 {
    evaluate(
        (0..64).map(|i| F128::from_u64((value >> i) & 1)).collect(),
        point,
    )
}
fn point(rng: &mut StdRng, n: usize) -> Vec<F128> {
    (0..n).map(|_| F128::from_raw(rng.gen())).collect()
}

fn with_stack(m: &MemoryLayout, stack_size: u64) -> MemoryLayout {
    MemoryLayout::try_new(&MemoryConfig {
        max_input_size: m.max_input_size,
        max_output_size: m.max_output_size,
        max_trusted_advice_size: m.max_trusted_advice_size,
        max_untrusted_advice_size: m.max_untrusted_advice_size,
        stack_size,
        heap_size: m.heap_size,
        program_size: Some(m.program_size),
    })
    .unwrap()
}

struct Fixture {
    statement: Statement,
    preprocessing: VerifierPreprocessing<TransparentBits>,
    witness: Rv64iWitness,
    a_ram: Vec<F128>,
    r_bit: Vec<F128>,
    r_4: Vec<F128>,
    stage4: ReadCheckingValues,
    rd: Vec<F128>,
    ra: Vec<F128>,
    store: Vec<F128>,
    inc: Vec<F128>,
}
impl Fixture {
    fn new() -> Self {
        let (mut statement, preprocessing, source) = support::counting_loop();
        statement.log_T = 5;
        statement.device.memory_layout = with_stack(&statement.device.memory_layout, 512);
        let layout = Layout::new(
            source.layout.log_K_bytecode(),
            6,
            source.layout.lowest_address(),
        )
        .unwrap();
        let witness = Rv64iWitness::synthetic(
            0x5_0128,
            layout,
            Arc::clone(&source.bytecode),
            &statement.device.memory_layout,
            source.initial_ram.clone(),
            5,
        )
        .unwrap();
        statement.entry_pc =
            witness.bytecode.rows()[witness.layout.bytecode_index(&witness.bits[0]) as usize].pc;
        let mut rng = StdRng::seed_from_u64(0xa501_5813);
        let a_ram = point(&mut rng, 6);
        let r_bit = point(&mut rng, 6);
        let r_4 = point(&mut rng, 5);
        let a_reg = &a_ram[a_ram.len() - 5..];
        let mut state = State::new(statement.entry_pc);
        for &(index, value) in &witness.initial_ram {
            state.set_ram_word(index, value);
        }
        let mut registers = Vec::new();
        let mut ram = Vec::new();
        let mut rd = Vec::new();
        let mut ra = Vec::new();
        let mut store = Vec::new();
        let mut inc = Vec::new();
        for (j, bits) in witness.bits.iter().enumerate() {
            registers.extend(state.registers.iter().map(|value| word(*value, &r_bit)));
            ram.extend((0..64).map(|k| word(state.ram.get(&k).copied().unwrap_or(0), &r_bit)));
            let row = &witness.bytecode.rows()[witness.layout.bytecode_index(bits) as usize];
            let is_store = row.variant.unwrap().is_store();
            let ram_index = witness.layout.ram_index(bits);
            rd.push(basis(a_reg, usize::from(row.rd)));
            ra.push(basis(&a_ram, ram_index as usize));
            store.push(F128::from_u64(u64::from(is_store)));
            inc.push(word(witness.layout.inc(bits), &r_bit));
            let base = replay::base_words(
                &witness.layout,
                row,
                bits,
                &state.registers,
                &state.ram,
                witness.words[j].next_pc,
            );
            if is_store {
                state.set_ram_word(ram_index, base.ram_read_value ^ witness.layout.inc(bits));
            } else {
                state.registers[usize::from(row.rd)] = base.rd_write_value;
            }
        }
        let final_ram = (0..64)
            .map(|k| word(state.ram.get(&k).copied().unwrap_or(0), &r_bit))
            .collect();
        let register_point: Vec<_> = a_reg.iter().chain(&r_4).copied().collect();
        let ram_point: Vec<_> = a_ram.iter().chain(&r_4).copied().collect();
        let stage4 = ReadCheckingValues {
            rs1_ra: F128::zero(),
            rs2_ra: F128::zero(),
            rd_wa: F128::zero(),
            ram_ra: F128::zero(),
            registers_val: evaluate(registers, &register_point),
            ram_val: evaluate(ram, &ram_point),
            ram_val_final: evaluate(final_ram, &a_ram),
        };
        Self {
            statement,
            preprocessing,
            witness,
            a_ram,
            r_bit,
            r_4,
            stage4,
            rd,
            ra,
            store,
            inc,
        }
    }
    fn checked(&self) -> CheckedInputs<'_, TransparentBits> {
        CheckedInputs::of_statement(
            &self.preprocessing,
            &self.statement,
            6,
            self.witness.final_pc,
        )
        .unwrap()
    }
    fn source(&self) -> Stage5InputPoints<F128> {
        let a_reg = &self.a_ram[self.a_ram.len() - 5..];
        let registers_val = a_reg
            .iter()
            .chain(&self.r_bit)
            .chain(&self.r_4)
            .copied()
            .collect();
        let ram_val = self
            .a_ram
            .iter()
            .chain(&self.r_bit)
            .chain(&self.r_4)
            .copied()
            .collect();
        let ram_val_final = self.a_ram.iter().chain(&self.r_bit).copied().collect();
        Stage5InputPoints {
            registers_val_evaluation: RegistersValEvaluationInputClaims { registers_val },
            ram_val_evaluation: RamValEvaluationInputClaims {
                ram_val,
                ram_val_final,
            },
        }
    }
    fn batch(&self) -> Stage5Sumchecks<F128> {
        Stage5Sumchecks(VerifierStage5Sumchecks::new(&self.checked(), &self.source()).unwrap())
    }
    fn transcript() -> BinaryTranscript {
        BinaryTranscript::new(b"rv64i-stage5-batch")
    }
    fn prove(&self) -> (StageProof, BinaryTranscript) {
        let batch = self.batch();
        let mut transcript = Self::transcript();
        let challenges = batch.draw_challenges(&mut transcript).unwrap();
        let inputs = Stage5InputClaims::from_stage4(&self.stage4);
        let proof = batch
            .prove(
                &Stage5Kernels::default(),
                &mut ProofSession::default(),
                &mut SequentialRounds,
                &self.witness,
                &inputs,
                &batch.input_points(),
                &challenges,
                ClearSumcheckRecorder::<F128, NoCommitment>::new(),
                &mut transcript,
            )
            .unwrap();
        (proof, transcript)
    }
    fn wire(proof: &StageProof) -> BatchProof<ValEvaluationValues> {
        BatchProof {
            rounds: proof.recorded.proof.clone(),
            values: proof.output_claims.clone().into_wire(),
        }
    }
    fn less_table(&self) -> Vec<F128> {
        (0..32)
            .map(|j| (j + 1..32).map(|k| basis(&self.r_4, k)).sum())
            .collect()
    }
}

#[test]
fn value_expressions_equal_replayed_state_and_cube_definitions() {
    let fixture = Fixture::new();
    let batch = fixture.batch();
    let inputs = Stage5InputClaims::from_stage4(&fixture.stage4);
    let input_points: Stage5InputPoints<F128> = batch.input_points();
    let mut transcript = Fixture::transcript();
    let challenges = batch.draw_challenges(&mut transcript).unwrap();
    let lt = fixture.less_table();
    let register_sum: F128 = (0..32)
        .map(|j| lt[j] * fixture.rd[j] * (F128::one() + fixture.store[j]) * fixture.inc[j])
        .sum();
    let ram_sum: F128 = (0..32)
        .map(|j| {
            (challenges.ram_val_evaluation.val * lt[j] + challenges.ram_val_evaluation.final_value)
                * fixture.ra[j]
                * fixture.store[j]
                * fixture.inc[j]
        })
        .sum();
    assert_eq!(
        batch
            .registers_val_evaluation
            .input_claim(&inputs.registers_val_evaluation, &NoChallenges::default())
            .unwrap(),
        register_sum
    );
    assert_eq!(
        batch
            .ram_val_evaluation
            .input_claim(&inputs.ram_val_evaluation, &challenges.ram_val_evaluation)
            .unwrap(),
        ram_sum
    );
    let mut rng = StdRng::seed_from_u64(0x513_ffff);
    let r_5 = point(&mut rng, 5);
    let registers = &batch.registers_val_evaluation;
    let ram = &batch.ram_val_evaluation;
    let register_points = registers
        .derive_opening_points(&r_5, &input_points.registers_val_evaluation)
        .unwrap();
    let ram_points = ram
        .derive_opening_points(&r_5, &input_points.ram_val_evaluation)
        .unwrap();
    let rd = evaluate(fixture.rd.clone(), &r_5);
    let ra = evaluate(fixture.ra.clone(), &r_5);
    let store = evaluate(fixture.store.clone(), &r_5);
    let inc = evaluate(fixture.inc.clone(), &r_5);
    let less = evaluate(lt, &r_5);
    let values = ValEvaluationValues {
        rd_wa: rd,
        ram_ra: ra,
        store,
        inc,
    }
    .expand();
    assert_eq!(
        registers
            .expected_output(
                &input_points.registers_val_evaluation,
                &values.registers_val_evaluation,
                &register_points,
                &NoChallenges::default()
            )
            .unwrap(),
        less * rd * (F128::one() + store) * inc
    );
    assert_eq!(
        ram.expected_output(
            &input_points.ram_val_evaluation,
            &values.ram_val_evaluation,
            &ram_points,
            &challenges.ram_val_evaluation
        )
        .unwrap(),
        (challenges.ram_val_evaluation.val * less + challenges.ram_val_evaluation.final_value)
            * ra
            * store
            * inc
    );
    assert_eq!(register_points.store, r_5);
    assert_eq!(register_points.store, ram_points.store);
    assert_eq!(register_points.inc, ram_points.inc);
    assert_eq!(
        register_points.rd_wa,
        fixture.a_ram[1..]
            .iter()
            .chain(&r_5)
            .copied()
            .collect::<Vec<_>>()
    );
    assert_eq!(
        register_points.inc,
        fixture
            .r_bit
            .iter()
            .chain(&r_5)
            .copied()
            .collect::<Vec<_>>()
    );
    assert_eq!(
        ram_points.ram_ra,
        fixture
            .a_ram
            .iter()
            .chain(&r_5)
            .copied()
            .collect::<Vec<_>>()
    );
    let init: F128 = fixture
        .witness
        .initial_ram
        .iter()
        .map(|&(k, w)| basis(&fixture.a_ram, k as usize) * word(w, &fixture.r_bit))
        .sum();
    assert_eq!(
        ram_init::evaluate(&fixture.checked(), &fixture.a_ram, &fixture.r_bit).unwrap(),
        init
    );
}

#[test]
fn stage5_proves_four_wire_values_and_matches_the_verifier_transcript() {
    let fixture = Fixture::new();
    let (proof, prover_transcript) = fixture.prove();
    assert_eq!(fixture.batch().registers_val_evaluation.rounds(), 5);
    assert_eq!(fixture.batch().ram_val_evaluation.degree(), 4);
    assert_eq!(
        fixture.batch().opening_values(&proof.output_claims).len(),
        4
    );
    let wire = Fixture::wire(&proof);
    let mut transcript = Fixture::transcript();
    let output = verify::verify_inputs(
        &fixture.batch().0,
        &wire,
        &mut transcript,
        &Stage5InputClaims::from_stage4(&fixture.stage4),
        &fixture.source(),
    )
    .unwrap();
    assert_eq!(output.points, proof.output_points);
    assert_eq!(output.claims, proof.output_claims);
    assert_eq!(prover_transcript.state(), transcript.state());
    assert_eq!(
        wire.values.rd_wa,
        evaluate(fixture.rd.clone(), output.r_5())
    );
    assert_eq!(
        wire.values.ram_ra,
        evaluate(fixture.ra.clone(), output.r_5())
    );
    assert_eq!(
        wire.values.store,
        evaluate(fixture.store.clone(), output.r_5())
    );
    assert_eq!(wire.values.inc, evaluate(fixture.inc.clone(), output.r_5()));
}

#[test]
fn stage5_rejects_a_changed_initial_ram_word() {
    let mut fixture = Fixture::new();
    let (proof, _) = fixture.prove();
    let wire = Fixture::wire(&proof);
    let original = ram_init::evaluate(&fixture.checked(), &fixture.a_ram, &fixture.r_bit).unwrap();
    fixture.statement.device.inputs[0] ^= 1;
    let changed = ram_init::evaluate(&fixture.checked(), &fixture.a_ram, &fixture.r_bit).unwrap();
    assert_ne!(original, changed);
    assert_eq!(
        original + changed,
        basis(&fixture.a_ram, 0) * basis(&fixture.r_bit, 0)
    );
    assert!(verify::verify_inputs(
        &fixture.batch().0,
        &wire,
        &mut Fixture::transcript(),
        &Stage5InputClaims::from_stage4(&fixture.stage4),
        &fixture.source()
    )
    .is_err());
}

#[test]
fn stage5_rejects_inconsistent_consumed_points_and_output_aliases() {
    let fixture = Fixture::new();
    let (proof, _) = fixture.prove();
    for cell in 0..3 {
        let mut source = fixture.source();
        let point = match cell {
            0 => &mut source.registers_val_evaluation.registers_val,
            1 => &mut source.ram_val_evaluation.ram_val,
            _ => &mut source.ram_val_evaluation.ram_val_final,
        };
        point[0] += F128::one();
        assert!(VerifierStage5Sumchecks::new(&fixture.checked(), &source).is_err());
    }
    for alias in 0..2 {
        let mut claims = proof.output_claims.clone();
        if alias == 0 {
            claims.ram_val_evaluation.store += F128::one();
        } else {
            claims.ram_val_evaluation.inc += F128::one();
        }
        let batch = fixture.batch();
        let mut transcript = Fixture::transcript();
        let challenges = batch.draw_challenges(&mut transcript).unwrap();
        assert!(batch
            .verify_clear(
                &Stage5InputClaims::from_stage4(&fixture.stage4),
                &fixture.source(),
                &challenges,
                &claims,
                &proof.recorded.proof,
                &mut transcript,
                5
            )
            .is_err());
    }
}

#[test]
fn initial_ram_split_tables_cover_both_address_halves() {
    let (mut statement, original, witness) = support::counting_loop();
    statement.device.memory_layout = with_stack(&statement.device.memory_layout, 16 << 20);
    let mut rng = StdRng::seed_from_u64(0x5000_ffff);
    for a in [5, 6, 20] {
        let k = 1_u64 << a;
        let image = vec![
            (4, 0x0123_4567_89ab_cdef),
            (k / 2, u64::MAX),
            (k - 1, 0xf000_0000_0000_0001),
        ];
        let preprocessing =
            VerifierPreprocessing::<TransparentBits>::new(original.bytecode().clone(), image, ())
                .unwrap();
        let checked =
            CheckedInputs::of_statement(&preprocessing, &statement, a, witness.final_pc).unwrap();
        let address = point(&mut rng, usize::from(a));
        let bit = point(&mut rng, 6);
        let expected: F128 = checked
            .initial_ram()
            .iter()
            .map(|&(index, value)| basis(&address, index as usize) * word(value, &bit))
            .sum();
        assert_eq!(
            ram_init::evaluate(&checked, &address, &bit).unwrap(),
            expected
        );
        assert!(ram_init::evaluate(&checked, &address[..address.len() - 1], &bit).is_err());
        assert!(ram_init::evaluate(&checked, &address, &bit[..5]).is_err());
        let original_initial = checked.initial_ram().to_vec();
        statement.device.outputs = vec![0xff];
        statement.device.panic = true;
        let changed =
            CheckedInputs::of_statement(&preprocessing, &statement, a, witness.final_pc).unwrap();
        assert_eq!(changed.initial_ram(), original_initial);
        assert_eq!(
            ram_init::evaluate(&changed, &address, &bit).unwrap(),
            expected
        );
    }
}

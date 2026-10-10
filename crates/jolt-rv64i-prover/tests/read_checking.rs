//! Batch-local contracts for read checking and the final public RAM interval.
#![expect(clippy::unwrap_used, reason = "invalid test fixtures fail immediately")]

#[expect(
    dead_code,
    reason = "shared machine helpers cover the whole execution corpus"
)]
mod support;
use common::jolt_device::{MemoryConfig, MemoryLayout};
use jolt_claims::NoChallenges;
use jolt_crypto::NoCommitment;
use jolt_field::{Field, One, Ring, Zero, F128};
use jolt_kernels::reference::naive::NaiveSumcheckProver;
use jolt_kernels::ProofSession;
use jolt_kernels::{KernelError, PrepareKernel, ProverInputs, SumcheckKernel};
use jolt_poly::{BindingOrder, Polynomial};
use jolt_program::preprocess::PublicIoMemory;
use jolt_prover::{
    driver::{Proved, StageProver},
    ProverError,
};
use jolt_rv64i_arith::Layout;
use jolt_rv64i_prover::commitment::transparent::TransparentBits;
use jolt_rv64i_prover::plane::Rv64iPlane;
use jolt_rv64i_prover::{
    plane::Rv64iWitness,
    stages::stage4::{Stage4Kernels, Stage4Sumchecks},
};
use jolt_rv64i_verifier::ids::{
    DerivedId, OpeningId, OutputCheckDerived, ReadCheckingDerived, RelationId, VirtualPolynomial,
};
use jolt_rv64i_verifier::stages::stage4::verify as stage4_verify;
use jolt_rv64i_verifier::stages::stage4::{
    ram_output_check::RamOutputCheck, ram_read_checking::RamReadChecking,
    registers_read_checking::RegistersReadChecking,
};
use jolt_rv64i_verifier::{
    claims::ram_output_check::RamOutputCheckOutputClaims,
    claims::ram_read_checking::RamReadCheckingOutputClaims,
    claims::registers_read_checking::RegistersReadCheckingOutputClaims,
    claims::{
        ram_read_checking::RamReadCheckingInputClaims,
        registers_read_checking::RegistersReadCheckingInputClaims,
    },
    points::to_high_to_low,
    proof::BatchProof,
    public::io::{io_mask, val_io},
    stages::stage4::verify::{
        Stage4Challenges, Stage4InputClaims, Stage4InputPoints,
        Stage4Sumchecks as VerifierStage4Sumchecks,
    },
    statement::Statement,
};
use jolt_rv64i_verifier::{preprocessing::VerifierPreprocessing, statement::CheckedInputs};
use jolt_sumcheck::{ClearSumcheckRecorder, SequentialRounds};
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{collections::BTreeMap, sync::Arc};

type BinaryTranscript = Blake2bTranscript<F128>;
type StageProof = Proved<F128, Stage4Sumchecks<F128>, NoCommitment>;

fn point(rng: &mut StdRng, n: usize) -> Vec<F128> {
    (0..n).map(|_| F128::from_raw(rng.gen())).collect()
}
fn basis(point: &[F128], index: usize) -> F128 {
    point
        .iter()
        .enumerate()
        .map(|(i, r)| {
            if (index >> i) & 1 == 1 {
                *r
            } else {
                F128::one() + *r
            }
        })
        .product()
}
fn word(word: u64, bit: &[F128]) -> F128 {
    Polynomial::new((0..64).map(|i| F128::from_u64((word >> i) & 1)).collect())
        .evaluate(&to_high_to_low(bit))
}
fn evaluate(table: &[F128], point: &[F128]) -> F128 {
    Polynomial::new(table.to_vec()).evaluate(&to_high_to_low(point))
}

struct Tables {
    selectors: [Vec<F128>; 3],
    registers: Vec<F128>,
    ram_ra: Vec<F128>,
    ram: Vec<F128>,
    final_ram: Vec<F128>,
    public: Vec<F128>,
    mask: Vec<F128>,
}
impl Tables {
    fn replay(witness: &Rv64iWitness, bit: &[F128], io: &PublicIoMemory) -> Self {
        let count = 1 << witness.layout.log_K_ram();
        let mut registers = [0; 32];
        let mut ram: BTreeMap<_, _> = witness.initial_ram.iter().copied().collect();
        let mut tables = Self {
            selectors: std::array::from_fn(|_| Vec::new()),
            registers: Vec::new(),
            ram_ra: Vec::new(),
            ram: Vec::new(),
            final_ram: Vec::new(),
            public: vec![F128::zero(); count],
            mask: vec![F128::zero(); count],
        };
        for (j, bits) in witness.bits.iter().enumerate() {
            let row = &witness.bytecode.rows()[witness.layout.bytecode_index(bits) as usize];
            let base = support::replay::base_words(
                &witness.layout,
                row,
                bits,
                &registers,
                &ram,
                witness.words[j].next_pc,
            );
            assert_eq!(witness.words[j].rs1_value, base.rs1_value);
            assert_eq!(witness.words[j].rs2_value, base.rs2_value);
            assert_eq!(
                witness.words[j].rd_pre_value,
                registers[usize::from(row.rd)]
            );
            assert_eq!(witness.words[j].ram_read_value, base.ram_read_value);
            for (selector, selected) in tables.selectors.iter_mut().zip([row.rs1, row.rs2, row.rd])
            {
                selector
                    .extend((0..32).map(|k| F128::from_u64(u64::from(k == usize::from(selected)))));
            }
            tables
                .registers
                .extend(registers.iter().map(|w| word(*w, bit)));
            let selected = witness.layout.ram_index(bits);
            for k in 0..count {
                tables
                    .ram_ra
                    .push(F128::from_u64(u64::from(k as u64 == selected)));
                tables
                    .ram
                    .push(word(ram.get(&(k as u64)).copied().unwrap_or(0), bit));
            }
            if row.variant.unwrap().is_store() {
                *ram.entry(selected).or_default() ^= witness.layout.inc(bits);
            } else {
                registers[usize::from(row.rd)] = base.rd_write_value;
            }
        }
        tables.final_ram = (0..count)
            .map(|k| word(ram.get(&(k as u64)).copied().unwrap_or(0), bit))
            .collect();
        for segment in &io.segments {
            for (offset, value) in segment.words.iter().enumerate() {
                tables.public[segment.start_index as usize + offset] = word(*value, bit);
            }
        }
        for k in io.io_mask_start..io.io_mask_end {
            tables.mask[k as usize] = F128::one();
        }
        tables
    }
    fn sums(&self, tau: &[F128], r3: &[F128], c: &Stage4Challenges<F128>) -> [F128; 3] {
        let coefficients = [
            c.registers_read_checking.rs1,
            c.registers_read_checking.rs2,
            c.registers_read_checking.rd,
        ];
        let registers = self
            .registers
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let selector: F128 = self
                    .selectors
                    .iter()
                    .zip(coefficients)
                    .map(|(table, coefficient)| coefficient * table[index])
                    .sum();
                basis(r3, index / 32) * selector * *value
            })
            .sum();
        let count = self.final_ram.len();
        let ram = self
            .ram
            .iter()
            .zip(&self.ram_ra)
            .enumerate()
            .map(|(index, (value, selector))| basis(r3, index / count) * *selector * *value)
            .sum();
        let output = (0..count)
            .map(|k| basis(tau, k) * self.mask[k] * (self.final_ram[k] + self.public[k]))
            .sum();
        [registers, ram, output]
    }
}

struct Fixture {
    statement: Statement,
    preprocessing: VerifierPreprocessing<TransparentBits>,
    witness: Rv64iWitness,
    bit: Vec<F128>,
    r3: Vec<F128>,
}
impl Fixture {
    fn new(t: usize) -> Self {
        let (mut statement, preprocessing, source) = support::counting_loop();
        let m = &statement.device.memory_layout;
        statement.device.memory_layout = MemoryLayout::try_new(&MemoryConfig {
            max_input_size: m.max_input_size,
            max_output_size: m.max_output_size,
            max_trusted_advice_size: 0,
            max_untrusted_advice_size: 0,
            stack_size: 256,
            heap_size: 0,
            program_size: Some(m.program_size),
        })
        .unwrap();
        statement.log_T = t as u8;
        let layout = Layout::new(
            source.layout.log_K_bytecode(),
            6,
            source.layout.lowest_address(),
        )
        .unwrap();
        let witness = Rv64iWitness::synthetic(
            0x3412,
            layout,
            Arc::clone(&source.bytecode),
            &statement.device.memory_layout,
            source.initial_ram.clone(),
            t,
        )
        .unwrap();
        let mut ram: BTreeMap<_, _> = witness.initial_ram.iter().copied().collect();
        for bits in witness.bits.iter() {
            let row = &witness.bytecode.rows()[witness.layout.bytecode_index(bits) as usize];
            if row.variant.unwrap().is_store() {
                *ram.entry(witness.layout.ram_index(bits)).or_default() ^= witness.layout.inc(bits);
            }
        }
        let output = statement
            .device
            .memory_layout
            .remapped_word_address(statement.device.memory_layout.output_start)
            .unwrap();
        statement.device.outputs = ram
            .get(&output)
            .copied()
            .unwrap_or(0)
            .to_le_bytes()
            .to_vec();
        let mut rng = StdRng::seed_from_u64(0x0004_6128);
        Self {
            statement,
            preprocessing,
            witness,
            bit: point(&mut rng, 6),
            r3: point(&mut rng, t),
        }
    }
    fn batch(
        &self,
        statement: &Statement,
    ) -> (
        Stage4Sumchecks<F128>,
        Stage4Challenges<F128>,
        BinaryTranscript,
    ) {
        let mut transcript = BinaryTranscript::new(b"rv64i-stage4-fixture");
        let batch = Stage4Sumchecks(
            VerifierStage4Sumchecks::new(
                &self.witness.layout,
                self.bit.clone(),
                self.r3.clone(),
                PublicIoMemory::new(&statement.device).unwrap(),
                &mut transcript,
            )
            .unwrap(),
        );
        let challenges = batch.draw_challenges(&mut transcript).unwrap();
        (batch, challenges, transcript)
    }
    fn inputs(&self, witness: &Rv64iWitness) -> Stage4InputClaims<F128> {
        let mut values = [F128::zero(); 4];
        for (j, w) in witness.words.iter().enumerate() {
            for (value, raw) in
                values
                    .iter_mut()
                    .zip([w.rs1_value, w.rs2_value, w.rd_pre_value, w.ram_read_value])
            {
                *value += basis(&self.r3, j) * word(raw, &self.bit);
            }
        }
        Stage4InputClaims {
            registers_read_checking: RegistersReadCheckingInputClaims {
                rs1_value: values[0],
                rs2_value: values[1],
                rd_pre_value: values[2],
            },
            ram_read_checking: RamReadCheckingInputClaims {
                ram_read_value: values[3],
            },
            ram_output_check: Default::default(),
        }
    }
    fn input_points(batch: &Stage4Sumchecks<F128>) -> Stage4InputPoints<F128> {
        Stage4InputPoints {
            registers_read_checking: batch.registers_read_checking.input_points(),
            ram_read_checking: batch.ram_read_checking.input_points(),
            ram_output_check: Default::default(),
        }
    }
    fn prove(
        &self,
        witness: &Rv64iWitness,
    ) -> Result<(StageProof, BinaryTranscript), ProverError<F128>> {
        let (batch, challenges, mut transcript) = self.batch(&self.statement);
        let proof = batch.prove(
            &Stage4Kernels::default(),
            &mut ProofSession::default(),
            &mut SequentialRounds,
            witness,
            &self.inputs(witness),
            &Self::input_points(&batch),
            &challenges,
            ClearSumcheckRecorder::<F128, NoCommitment>::new(),
            &mut transcript,
        )?;
        Ok((proof, transcript))
    }
}

#[test]
fn batch4_accepts_replayed_ram_and_shares_address_and_cycle_points() {
    let fixture = Fixture::new(4);
    let (proof, prover_transcript) = fixture.prove(&fixture.witness).unwrap();
    let (batch, challenges, _) = fixture.batch(&fixture.statement);
    let wire = BatchProof {
        rounds: proof.recorded.proof,
        values: proof.output_claims.into_wire(),
    };
    let checked = CheckedInputs::of_statement(
        &fixture.preprocessing,
        &fixture.statement,
        6,
        fixture.witness.final_pc,
    )
    .unwrap();
    let mut transcript = BinaryTranscript::new(b"rv64i-stage4-fixture");
    let replay_batch = VerifierStage4Sumchecks::new(
        checked.layout(),
        fixture.bit.clone(),
        fixture.r3.clone(),
        Arc::clone(checked.shared_io()),
        &mut transcript,
    )
    .unwrap();
    let output = stage4_verify::verify_inputs(
        &replay_batch,
        &wire,
        &mut transcript,
        &fixture.inputs(&fixture.witness),
        &Fixture::input_points(&batch),
    )
    .unwrap();
    let tables = Tables::replay(&fixture.witness, &fixture.bit, batch.ram_output_check.io());
    let reg_point = &output.points.registers_read_checking.rs1_ra;
    let ram_point = &output.points.ram_read_checking.ram_ra;
    assert_eq!(
        wire.values.rs1_ra,
        evaluate(&tables.selectors[0], reg_point)
    );
    assert_eq!(
        wire.values.rs2_ra,
        evaluate(&tables.selectors[1], reg_point)
    );
    assert_eq!(wire.values.rd_wa, evaluate(&tables.selectors[2], reg_point));
    assert_eq!(
        wire.values.registers_val,
        evaluate(&tables.registers, reg_point)
    );
    assert_eq!(wire.values.ram_ra, evaluate(&tables.ram_ra, ram_point));
    assert_eq!(wire.values.ram_val, evaluate(&tables.ram, ram_point));
    assert_eq!(
        wire.values.ram_val_final,
        evaluate(&tables.final_ram, &ram_point[..6])
    );
    assert_eq!(output.claims.into_wire(), wire.values);
    let points = output.points;
    assert_eq!(prover_transcript.state(), transcript.state());
    assert_eq!(
        points.registers_read_checking.rs1_ra,
        points.ram_read_checking.ram_ra[1..]
    );
    assert_eq!(
        &points.ram_output_check.ram_val_final[..6],
        &points.ram_read_checking.ram_ra[..6]
    );
    let mut head_transcript = fixture.batch(&fixture.statement).2;
    let (head, _) = batch
        .begin_batch(
            &fixture.inputs(&fixture.witness),
            &challenges,
            &mut ClearSumcheckRecorder::<F128, NoCommitment>::new(),
            &mut head_transcript,
        )
        .unwrap();
    assert_eq!(head.max_num_vars, 10);
    assert_eq!(head.max_degree, 3);
    assert_eq!(batch.output_claim_count(), 7);
}

#[test]
fn three_read_checking_expressions_equal_direct_cube_definitions() {
    let fixture = Fixture::new(5);
    let (batch, challenges, _) = fixture.batch(&fixture.statement);
    let inputs = fixture.inputs(&fixture.witness);
    let tables = Tables::replay(&fixture.witness, &fixture.bit, batch.ram_output_check.io());
    let sums = tables.sums(batch.ram_output_check.tau(), &fixture.r3, &challenges);
    assert_eq!(
        batch
            .registers_read_checking
            .input_claim(
                &inputs.registers_read_checking,
                &challenges.registers_read_checking
            )
            .unwrap(),
        sums[0]
    );
    assert_eq!(
        batch
            .ram_read_checking
            .input_claim(&inputs.ram_read_checking, &challenges.ram_read_checking)
            .unwrap(),
        sums[1]
    );
    assert_eq!(
        batch
            .ram_output_check
            .input_claim(&inputs.ram_output_check, &challenges.ram_output_check)
            .unwrap(),
        sums[2]
    );
    assert_eq!(sums[2], F128::zero());
    let mut rng = StdRng::seed_from_u64(0x0003_4810_1112);
    let z = point(&mut rng, 11);
    let points = batch
        .derive_opening_points(&z, &Fixture::input_points(&batch))
        .unwrap();
    let reg_point = &z[1..];
    assert_eq!(points.registers_read_checking.rs1_ra, reg_point);
    assert_eq!(points.registers_read_checking.rs2_ra, reg_point);
    assert_eq!(points.registers_read_checking.rd_wa, reg_point);
    assert_eq!(
        points.registers_read_checking.registers_val,
        z[1..6]
            .iter()
            .chain(&fixture.bit)
            .chain(&z[6..])
            .copied()
            .collect::<Vec<_>>()
    );
    assert_eq!(points.ram_read_checking.ram_ra, z);
    assert_eq!(
        points.ram_read_checking.ram_val,
        z[..6]
            .iter()
            .chain(&fixture.bit)
            .chain(&z[6..])
            .copied()
            .collect::<Vec<_>>()
    );
    assert_eq!(
        points.ram_output_check.ram_val_final,
        z[..6]
            .iter()
            .chain(&fixture.bit)
            .copied()
            .collect::<Vec<_>>()
    );
    let reg = &batch.registers_read_checking;
    let reg_values = RegistersReadCheckingOutputClaims {
        rs1_ra: evaluate(&tables.selectors[0], reg_point),
        rs2_ra: evaluate(&tables.selectors[1], reg_point),
        rd_wa: evaluate(&tables.selectors[2], reg_point),
        registers_val: evaluate(&tables.registers, reg_point),
    };
    let weights: Vec<_> = (0..32)
        .flat_map(|j| std::iter::repeat_n(basis(&fixture.r3, j), 32))
        .collect();
    let c = &challenges.registers_read_checking;
    let expected = evaluate(&weights, reg_point)
        * (c.rs1 * reg_values.rs1_ra + c.rs2 * reg_values.rs2_ra + c.rd * reg_values.rd_wa)
        * reg_values.registers_val;
    assert_eq!(
        reg.expected_output(
            &reg.input_points(),
            &reg_values,
            &points.registers_read_checking,
            c
        )
        .unwrap(),
        expected
    );
    let ram_values = RamReadCheckingOutputClaims {
        ram_ra: evaluate(&tables.ram_ra, &z),
        ram_val: evaluate(&tables.ram, &z),
    };
    let weights: Vec<_> = (0..32)
        .flat_map(|j| std::iter::repeat_n(basis(&fixture.r3, j), 64))
        .collect();
    assert_eq!(
        batch
            .ram_read_checking
            .expected_output(
                &batch.ram_read_checking.input_points(),
                &ram_values,
                &points.ram_read_checking,
                &NoChallenges::default()
            )
            .unwrap(),
        evaluate(&weights, &z) * ram_values.ram_ra * ram_values.ram_val
    );
    let output_values = RamOutputCheckOutputClaims {
        ram_val_final: evaluate(&tables.final_ram, &z[..6]),
    };
    let weights: Vec<_> = (0..64)
        .map(|k| basis(batch.ram_output_check.tau(), k))
        .collect();
    assert_eq!(
        batch
            .ram_output_check
            .expected_output(
                &Default::default(),
                &output_values,
                &points.ram_output_check,
                &NoChallenges::default()
            )
            .unwrap(),
        evaluate(&weights, &z[..6])
            * evaluate(&tables.mask, &z[..6])
            * (output_values.ram_val_final + evaluate(&tables.public, &z[..6]))
    );
}

#[test]
fn batch4_rejects_changed_public_output_byte() {
    let fixture = Fixture::new(4);
    let (proof, _) = fixture.prove(&fixture.witness).unwrap();
    let mut statement = fixture.statement.clone();
    statement.device.outputs[0] ^= 1;
    let (batch, challenges, mut transcript) = fixture.batch(&statement);
    let wire = BatchProof {
        rounds: proof.recorded.proof,
        values: proof.output_claims.into_wire(),
    };
    assert!(batch
        .verify(
            &fixture.inputs(&fixture.witness),
            &Fixture::input_points(&batch),
            &challenges,
            &wire,
            &mut transcript
        )
        .is_err());
}

#[test]
fn batch4_rejects_zero_ram_reads_on_cycles_without_access() {
    let fixture = Fixture::new(4);
    assert_ne!(fixture.witness.initial_ram[0].1, 0);
    let mut witness = fixture.witness.clone();
    let words = Arc::make_mut(&mut witness.words);
    let mut changed = 0;
    for (bits, words) in witness.bits.iter().zip(words) {
        let row = &witness.bytecode.rows()[witness.layout.bytecode_index(bits) as usize];
        if row.variant.unwrap().access().is_none() {
            assert_ne!(words.ram_read_value, 0);
            words.ram_read_value = 0;
            changed += 1;
        }
    }
    assert!(changed > 0);
    assert!(fixture.prove(&witness).is_err());
}

#[test]
fn public_io_extensions_equal_full_cube_tables_including_domain_boundary() {
    let fixture = Fixture::new(4);
    let mut io = PublicIoMemory::new(&fixture.statement.device).unwrap();
    let mut rng = StdRng::seed_from_u64(0x0004_0128);
    for a in [5, 6, 7] {
        let address = point(&mut rng, a);
        for (start, end) in [(0, 1 << a), (1, 1 << a), (io.io_mask_start, io.io_mask_end)] {
            io.io_mask_start = start;
            io.io_mask_end = end;
            let mask = (0..1 << a)
                .map(|k| F128::from_u64(u64::from(start <= k && k < end)))
                .collect::<Vec<_>>();
            let mut values = vec![F128::zero(); 1 << a];
            for segment in &io.segments {
                for (offset, value) in segment.words.iter().enumerate() {
                    values[segment.start_index as usize + offset] = word(*value, &fixture.bit);
                }
            }
            assert_eq!(io_mask(&io, &address).unwrap(), evaluate(&mask, &address));
            assert_eq!(
                val_io(&io, &address, &fixture.bit).unwrap(),
                evaluate(&values, &address)
            );
        }
    }
}

struct DensePrepare(Arc<DenseTables>);
struct DenseTables {
    openings: BTreeMap<RelationId, BTreeMap<OpeningId, Polynomial<F128>>>,
    derived: BTreeMap<RelationId, BTreeMap<DerivedId, Polynomial<F128>>>,
}

macro_rules! dense_prepare {
    ($relation:ident) => {
        impl PrepareKernel<F128, $relation<F128>, Rv64iPlane> for DensePrepare {
            fn prepare(
                &self,
                _session: &mut ProofSession,
                _witness: &Rv64iWitness,
                inputs: ProverInputs<'_, F128, $relation<F128>>,
            ) -> Result<Box<dyn SumcheckKernel<F128, Relation = $relation<F128>>>, KernelError<F128>> {
                Ok(Box::new(NaiveSumcheckProver::new(
                    &inputs,
                    self.0.openings[&inputs.relation.id()].clone(),
                    self.0.derived[&inputs.relation.id()].clone(),
                    BindingOrder::LowToHigh,
                )?))
            }
        }
    };
}
dense_prepare!(RegistersReadChecking);
dense_prepare!(RamReadChecking);
dense_prepare!(RamOutputCheck);

#[test]
fn batch4_dense_cubic_members_have_the_specified_zero_extension_scales() {
    let fixture = Fixture::new(3);
    let (batch, challenges, mut transcript) = fixture.batch(&fixture.statement);
    let tau = batch.ram_output_check.tau();
    let mut rng = StdRng::seed_from_u64(0x0406_0303);
    let public = Tables::replay(&fixture.witness, &fixture.bit, batch.ram_output_check.io());
    let mut tables = Tables {
        selectors: std::array::from_fn(|_| point(&mut rng, 32 * 8)),
        registers: point(&mut rng, 32 * 8),
        ram_ra: point(&mut rng, 64 * 8),
        ram: point(&mut rng, 64 * 8),
        final_ram: point(&mut rng, 64),
        public: public.public,
        mask: public.mask,
    };
    let sums = tables.sums(tau, &fixture.r3, &challenges);
    let selected = tables
        .mask
        .iter()
        .position(|value| *value == F128::one())
        .unwrap();
    let weight = basis(tau, selected);
    assert!(!weight.is_zero());
    tables.final_ram[selected] += sums[2] * weight.inverse().unwrap();
    assert_eq!(tables.sums(tau, &fixture.r3, &challenges)[2], F128::zero());
    let rs1 = challenges.registers_read_checking.rs1;
    assert!(!rs1.is_zero());
    let inputs = Stage4InputClaims {
        registers_read_checking: RegistersReadCheckingInputClaims {
            rs1_value: sums[0] * rs1.inverse().unwrap(),
            rs2_value: F128::zero(),
            rd_pre_value: F128::zero(),
        },
        ram_read_checking: RamReadCheckingInputClaims {
            ram_read_value: sums[1],
        },
        ram_output_check: Default::default(),
    };
    let inputs_points = Fixture::input_points(&batch);
    let mut openings = BTreeMap::new();
    let mut derived = BTreeMap::new();
    let register_tables = [
        (VirtualPolynomial::Rs1Ra, &tables.selectors[0]),
        (VirtualPolynomial::Rs2Ra, &tables.selectors[1]),
        (VirtualPolynomial::RdWa, &tables.selectors[2]),
        (VirtualPolynomial::RegistersVal, &tables.registers),
    ]
    .into_iter()
    .map(|(id, table)| {
        (
            OpeningId::virtual_polynomial(id, RelationId::RegistersReadChecking),
            Polynomial::new(table.clone()),
        )
    })
    .collect();
    let _ = openings.insert(RelationId::RegistersReadChecking, register_tables);
    let _ = openings.insert(
        RelationId::RamReadChecking,
        [
            (VirtualPolynomial::RamRa, &tables.ram_ra),
            (VirtualPolynomial::RamVal, &tables.ram),
        ]
        .into_iter()
        .map(|(id, table)| {
            (
                OpeningId::virtual_polynomial(id, RelationId::RamReadChecking),
                Polynomial::new(table.clone()),
            )
        })
        .collect(),
    );
    let _ = openings.insert(
        RelationId::RamOutputCheck,
        [(
            OpeningId::virtual_polynomial(
                VirtualPolynomial::RamValFinal,
                RelationId::RamOutputCheck,
            ),
            Polynomial::new(tables.final_ram.clone()),
        )]
        .into_iter()
        .collect(),
    );
    let reg_weights: Vec<_> = (0..8)
        .flat_map(|j| std::iter::repeat_n(basis(&fixture.r3, j), 32))
        .collect();
    let ram_weights: Vec<_> = (0..8)
        .flat_map(|j| std::iter::repeat_n(basis(&fixture.r3, j), 64))
        .collect();
    let output_weights: Vec<_> = (0..64).map(|k| basis(tau, k)).collect();
    let _ = derived.insert(
        RelationId::RegistersReadChecking,
        [(
            DerivedId::RegistersReadChecking(ReadCheckingDerived::EqCycle),
            Polynomial::new(reg_weights.clone()),
        )]
        .into_iter()
        .collect(),
    );
    let _ = derived.insert(
        RelationId::RamReadChecking,
        [(
            DerivedId::RamReadChecking(ReadCheckingDerived::EqCycle),
            Polynomial::new(ram_weights.clone()),
        )]
        .into_iter()
        .collect(),
    );
    let _ = derived.insert(
        RelationId::RamOutputCheck,
        [
            (
                DerivedId::RamOutputCheck(OutputCheckDerived::EqTau),
                Polynomial::new(output_weights.clone()),
            ),
            (
                DerivedId::RamOutputCheck(OutputCheckDerived::IoMask),
                Polynomial::new(tables.mask.clone()),
            ),
            (
                DerivedId::RamOutputCheck(OutputCheckDerived::ValIo),
                Polynomial::new(tables.public.clone()),
            ),
        ]
        .into_iter()
        .collect(),
    );
    let prepare = Arc::new(DenseTables { openings, derived });
    let kernels = Stage4Kernels {
        registers_read_checking: Box::new(DensePrepare(Arc::clone(&prepare))),
        ram_read_checking: Box::new(DensePrepare(Arc::clone(&prepare))),
        ram_output_check: Box::new(DensePrepare(prepare)),
    };
    let proof = batch
        .prove(
            &kernels,
            &mut ProofSession::default(),
            &mut SequentialRounds,
            &fixture.witness,
            &inputs,
            &inputs_points,
            &challenges,
            ClearSumcheckRecorder::<F128, NoCommitment>::new(),
            &mut transcript,
        )
        .unwrap();
    let z = &proof.output_points.ram_read_checking.ram_ra;
    let c = &challenges.registers_read_checking;
    let reg_point = &z[1..];
    let reg = evaluate(&reg_weights, reg_point)
        * (c.rs1 * evaluate(&tables.selectors[0], reg_point)
            + c.rs2 * evaluate(&tables.selectors[1], reg_point)
            + c.rd * evaluate(&tables.selectors[2], reg_point))
        * evaluate(&tables.registers, reg_point);
    let ram = evaluate(&ram_weights, z) * evaluate(&tables.ram_ra, z) * evaluate(&tables.ram, z);
    let output = evaluate(&output_weights, &z[..6])
        * evaluate(&tables.mask, &z[..6])
        * (evaluate(&tables.final_ram, &z[..6]) + evaluate(&tables.public, &z[..6]));
    for value in [reg, ram, output] {
        assert!(!value.is_zero());
    }
    let (verifier_batch, verifier_challenges, mut twin) = fixture.batch(&fixture.statement);
    let (head, coefficients) = batch
        .begin_batch(
            &inputs,
            &challenges,
            &mut ClearSumcheckRecorder::<F128, NoCommitment>::new(),
            &mut fixture.batch(&fixture.statement).2,
        )
        .unwrap();
    assert_eq!(head.max_num_vars, 9);
    assert_eq!(head.max_degree, 3);
    assert_eq!(
        head.members
            .iter()
            .map(|member| (member.rounds, member.offset))
            .collect::<Vec<_>>(),
        [(8, 1), (9, 0), (6, 0)]
    );
    let tail_scale: F128 = z[6..].iter().map(|r| F128::one() + *r).product();
    let head_scale = F128::one() + z[0];
    assert!(!head_scale.is_zero());
    assert!(!tail_scale.is_zero());
    assert_ne!(head_scale, F128::one());
    assert_ne!(tail_scale, F128::one());
    for coefficient in [
        coefficients.registers_read_checking,
        coefficients.ram_read_checking,
        coefficients.ram_output_check,
    ] {
        assert!(!coefficient.is_zero());
    }
    assert_ne!(
        proof.final_claim,
        coefficients.registers_read_checking * reg
            + coefficients.ram_read_checking * ram
            + coefficients.ram_output_check * output
    );
    assert_eq!(
        proof.final_claim,
        coefficients.registers_read_checking * head_scale * reg
            + coefficients.ram_read_checking * ram
            + coefficients.ram_output_check * tail_scale * output
    );
    let wire = BatchProof {
        rounds: proof.recorded.proof,
        values: proof.output_claims.into_wire(),
    };
    let verified_points = verifier_batch
        .verify(
            &inputs,
            &inputs_points,
            &verifier_challenges,
            &wire,
            &mut twin,
        )
        .unwrap();
    assert_eq!(verified_points.ram_read_checking.ram_ra, *z);
    assert_eq!(transcript.state(), twin.state());
}

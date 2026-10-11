//! Executed, nonterminating witness shared by the adapter measurements.

use self::replay::State;
use common::{
    constants::RAM_START_ADDRESS,
    jolt_device::{JoltDevice, MemoryConfig, MemoryLayout},
};
use jolt_program::{
    execution::{
        RamAccess, RamRead, RamWrite, RegisterRead, RegisterState, RegisterWrite, SourceTraceRow,
    },
    image::decode::decode_instruction,
};
use jolt_riscv::RV64I;
use jolt_rv64i_arith::{Layout, Variant};
use jolt_rv64i_prover::{commitment::transparent::TransparentBits, plane::Rv64iWitness};
use jolt_rv64i_trace::{adapt, Execution};
use jolt_rv64i_verifier::{
    preprocessing::VerifierPreprocessing,
    statement::{CheckedInputs, Statement},
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{
    error::Error,
    fmt::{Display, Formatter, Result as FmtResult},
    sync::Arc,
};

#[expect(
    dead_code,
    reason = "the common assembler serves the entire RV64I corpus"
)]
#[path = "../../../jolt-rv64i-arith/tests/suite/common/asm.rs"]
mod asm;
#[expect(
    dead_code,
    reason = "the shared harness also exposes facts for integration tests"
)]
#[path = "../../../jolt-rv64i-arith/tests/suite/common/harness.rs"]
mod harness;
#[expect(
    dead_code,
    reason = "the common interpreter also exposes fixture inspectors"
)]
#[path = "../../../jolt-rv64i-arith/tests/suite/common/interp.rs"]
mod interp;
#[expect(
    dead_code,
    reason = "only the replay's initial state is needed by this bench"
)]
#[path = "../../../jolt-rv64i-arith/tests/suite/common/replay.rs"]
mod replay;

const LOG_K: usize = 20;
const PROGRAM_ROWS: usize = 1 << LOG_K;
const DATA_BYTES: usize = 1 << 15;
const SEED: [u8; 32] = [0x5b; 32];

#[derive(Default)]
pub struct DynamicMix {
    pub cycles: usize,
    pub shift: usize,
    pub ram_access: usize,
    pub key: usize,
    pub taken_branch: usize,
    pub jalr: usize,
    pub register_write: usize,
    pub visited_rows: usize,
}

impl Display for DynamicMix {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "dynamic_cycles={} dynamic_shift={} dynamic_ram_access={} dynamic_key={} dynamic_taken_branch={} dynamic_jalr={} dynamic_register_write={} visited_rows={}",
            self.cycles,
            self.shift,
            self.ram_access,
            self.key,
            self.taken_branch,
            self.jalr,
            self.register_write,
            self.visited_rows,
        )
    }
}

pub struct WitnessFixture {
    pub witness: Rv64iWitness,
    pub statement: Statement,
    pub preprocessing: VerifierPreprocessing<TransparentBits>,
    pub mix: DynamicMix,
}

impl WitnessFixture {
    pub fn new(log_t: u8) -> Result<Self, Box<dyn Error>> {
        let fixture = TraceFixture::new(log_t)?;
        let execution = fixture.adapt()?;
        let checked = fixture.checked(&execution)?;
        let witness = Rv64iWitness::from_facts(
            checked.layout().clone(),
            Arc::clone(fixture.preprocessing.shared_bytecode()),
            &execution.facts,
            checked.initial_ram().to_vec(),
        )?;
        Ok(Self {
            witness,
            statement: fixture.statement,
            preprocessing: fixture.preprocessing,
            mix: fixture.mix,
        })
    }

    pub fn checked(&self) -> Result<CheckedInputs<'_, TransparentBits>, Box<dyn Error>> {
        Ok(CheckedInputs::of_statement(
            &self.preprocessing,
            &self.statement,
            LOG_K as u8,
            self.witness.final_pc,
        )?)
    }
}

/// Executed architectural transitions retained for untimed pipeline setup.
/// The cut has no exit or padded suffix; initial memory includes the program image.
pub struct TraceFixture {
    pub rows: Vec<SourceTraceRow>,
    pub statement: Statement,
    pub preprocessing: VerifierPreprocessing<TransparentBits>,
    pub mix: DynamicMix,
}

impl TraceFixture {
    pub fn new(log_t: u8) -> Result<Self, Box<dyn Error>> {
        let words = Self::program();
        let program_bytes = (4 * PROGRAM_ROWS) as u64;
        let memory_layout = MemoryLayout::try_new(&MemoryConfig {
            max_input_size: 8,
            max_output_size: 8,
            max_trusted_advice_size: 0,
            max_untrusted_advice_size: 0,
            stack_size: 0,
            heap_size: DATA_BYTES as u64,
            program_size: Some(program_bytes),
        })?;
        let layout = Layout::new(LOG_K, LOG_K, memory_layout.get_lowest_address())?;
        let program: Vec<_> = words
            .iter()
            .enumerate()
            .map(|(index, word)| (RAM_START_ADDRESS + 4 * index as u64, *word))
            .collect();
        let bytecode = Arc::new(harness::bytecode(&program, &layout));
        let image_start = (RAM_START_ADDRESS - layout.lowest_address()) / 8;
        let image = words
            .chunks_exact(2)
            .enumerate()
            .map(|(index, pair)| {
                (
                    image_start + index as u64,
                    u64::from(pair[0]) | (u64::from(pair[1]) << 32),
                )
            })
            .collect();
        let preprocessing = VerifierPreprocessing::new(Arc::clone(&bytecode), image, ())?;
        let statement = Statement {
            log_T: log_t,
            entry_pc: RAM_START_ADDRESS,
            device: JoltDevice {
                memory_layout,
                ..JoltDevice::default()
            },
        };
        let initial_ram = CheckedInputs::of_statement(
            &preprocessing,
            &statement,
            LOG_K as u8,
            RAM_START_ADDRESS,
        )?
        .initial_ram()
        .to_vec();
        let mut initial = State::new(RAM_START_ADDRESS);
        for &(index, value) in &initial_ram {
            initial.set_ram_word(index, value);
        }
        let mut machine = harness::machine(&program, &layout, &initial);
        drop(initial);
        drop(words);
        drop(program);
        let cycles = 1_usize << log_t;
        let mut rows = Vec::with_capacity(cycles);
        let mut visited = vec![false; PROGRAM_ROWS];
        let mut mix = DynamicMix {
            cycles,
            ..DynamicMix::default()
        };
        for _ in 0..cycles {
            let record = machine.step()?;
            let index = bytecode
                .index_of_pc(record.pc)
                .ok_or("interpreter fetched a PC missing from bytecode")?;
            let row = &bytecode.rows()[index];
            let variant = row
                .variant
                .ok_or("interpreter fetched an invalid bytecode row")?;
            mix.shift += usize::from(variant.shift().is_some());
            mix.ram_access += usize::from(record.access.is_some());
            mix.key += usize::from(variant.key_kind().is_some());
            mix.taken_branch +=
                usize::from(variant.branch().is_some() && record.next_pc != record.pc + 4);
            mix.jalr += usize::from(matches!(variant, Variant::JALR | Variant::JALR_X0));
            mix.register_write += usize::from(row.rd != 0 && !variant.is_store());
            if !visited[index] {
                mix.visited_rows += 1;
                visited[index] = true;
            }
            let instruction = decode_instruction(record.word, record.pc, false, RV64I)?;
            let operands = instruction.row().operands;
            let registers = RegisterState {
                rs1: operands.rs1.map(|register| RegisterRead {
                    register,
                    value: record.rs1_value,
                }),
                rs2: operands.rs2.map(|register| RegisterRead {
                    register,
                    value: record.rs2_value,
                }),
                rd: operands.rd.map(|register| RegisterWrite {
                    register,
                    pre_value: record.rd_pre_value,
                    post_value: record.rd_post_value,
                }),
            };
            let ram = record.access.map_or(RamAccess::NoOp, |access| {
                let address = layout.lowest_address() + 8 * access.word_index;
                if access.is_store {
                    RamAccess::Write(RamWrite {
                        address,
                        pre_value: access.word_before,
                        post_value: access.word_after,
                    })
                } else {
                    RamAccess::Read(RamRead {
                        address,
                        value: access.word_before,
                    })
                }
            });
            rows.push(SourceTraceRow::new(
                u32::try_from(index)?,
                record.pc,
                record.next_pc,
                registers,
                ram,
            ));
        }
        drop(machine);
        drop(visited);
        Ok(Self {
            rows,
            statement,
            preprocessing,
            mix,
        })
    }

    pub fn adapt(&self) -> Result<Execution, Box<dyn Error>> {
        Ok(adapt(
            self.preprocessing.bytecode(),
            self.preprocessing.image(),
            &self.statement.device.memory_layout,
            self.statement.entry_pc,
            &self.rows,
        )?)
    }

    pub fn checked(
        &self,
        execution: &Execution,
    ) -> Result<CheckedInputs<'_, TransparentBits>, Box<dyn Error>> {
        Ok(CheckedInputs::of_statement(
            &self.preprocessing,
            &self.statement,
            execution.log_K_ram,
            execution.final_pc(),
        )?)
    }

    fn program() -> Vec<u32> {
        let mut rng = StdRng::from_seed(SEED);
        let mut words = Vec::with_capacity(PROGRAM_ROWS);
        words.push(asm::auipc(31, 0));
        // Sixteen immutable address registers cover 4096 data words after the image.
        // Each centre admits its entire 256-word block with a signed 12-bit offset.
        for register in 15..31 {
            let target = 4 * PROGRAM_ROWS + usize::from(register - 15) * 2048 + 1024;
            let delta = target as i32 - (4 * words.len()) as i32;
            let high = (delta + 2048) >> 12;
            words.push(asm::auipc(register, high));
            words.push(asm::addi(register, register, delta - (high << 12)));
        }
        while words.len() < PROGRAM_ROWS - 1 {
            let rd = rng.gen_range(1..15);
            let rs1 = rng.gen_range(0..15);
            let rs2 = rng.gen_range(0..15);
            let immediate = rng.gen_range(-2048..2048);
            let choice = rng.gen_range(0..100);
            let word = match choice {
                0..=34 => {
                    let address_word: usize = if rng.gen_ratio(9, 10) {
                        rng.gen_range(0..64)
                    } else {
                        rng.gen_range(0..4096)
                    };
                    let address_register = 15 + (address_word / 256) as u8;
                    let offset = (8 * (address_word % 256)) as i32 - 1024;
                    if choice < 25 {
                        match rng.gen_range(0..7) {
                            0 => asm::lb(rd, address_register, offset),
                            1 => asm::lbu(rd, address_register, offset),
                            2 => asm::lh(rd, address_register, offset),
                            3 => asm::lhu(rd, address_register, offset),
                            4 => asm::lw(rd, address_register, offset),
                            5 => asm::lwu(rd, address_register, offset),
                            _ => asm::ld(rd, address_register, offset),
                        }
                    } else {
                        match rng.gen_range(0..4) {
                            0 => asm::sb(address_register, rs2, offset),
                            1 => asm::sh(address_register, rs2, offset),
                            2 => asm::sw(address_register, rs2, offset),
                            _ => asm::sd(address_register, rs2, offset),
                        }
                    }
                }
                35..=39 => match rng.gen_range(0..6) {
                    0 => asm::sll(rd, rs1, rs2),
                    1 => asm::srl(rd, rs1, rs2),
                    2 => asm::sra(rd, rs1, rs2),
                    3 => asm::slli(rd, rs1, rng.gen_range(0..64)),
                    4 => asm::srli(rd, rs1, rng.gen_range(0..64)),
                    _ => asm::srai(rd, rs1, rng.gen_range(0..64)),
                },
                40..=47 => match rng.gen_range(0..4) {
                    0 => asm::slt(rd, rs1, rs2),
                    1 => asm::sltu(rd, rs1, rs2),
                    2 => asm::slti(rd, rs1, immediate),
                    _ => asm::sltiu(rd, rs1, immediate),
                },
                48..=59 => {
                    // Equal operands fix half the branches taken, regardless of data.
                    let offset = if words.len() == PROGRAM_ROWS - 2 {
                        4
                    } else {
                        8
                    };
                    if choice < 54 {
                        asm::beq(rs1, rs1, offset)
                    } else {
                        asm::bne(rs1, rs1, offset)
                    }
                }
                60..=62 => asm::addi(0, rs1, immediate),
                _ => match rng.gen_range(0..10) {
                    0 => asm::add(rd, rs1, rs2),
                    1 => asm::sub(rd, rs1, rs2),
                    2 => asm::xor(rd, rs1, rs2),
                    3 => asm::and(rd, rs1, rs2),
                    4 => asm::or(rd, rs1, rs2),
                    5 => asm::addi(rd, rs1, immediate),
                    6 => asm::xori(rd, rs1, immediate),
                    7 => asm::andi(rd, rs1, immediate),
                    8 => asm::ori(rd, rs1, immediate),
                    _ => asm::addiw(rd, rs1, immediate),
                },
            };
            words.push(word);
        }
        words.push(asm::jalr(0, 31, 4));
        words
    }
}

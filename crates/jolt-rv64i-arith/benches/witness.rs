//! Frozen RV64I corpus; run with `RUSTFLAGS="-C target-cpu=native" cargo bench
//! -p jolt-rv64i-arith --bench witness`. Figures are medians after one warm-up.

#![expect(
    clippy::unwrap_used,
    reason = "benchmark setup and frozen-corpus assertions fail on invalid input"
)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use jolt_program::image::decode::decode_instruction;
use jolt_riscv::RV64IMAC_JOLT;
use jolt_rv64i_arith::{
    BaseWords, BitsBuilder, BitsRow, Bytecode, CycleFacts, Layout, RowSystem, WitnessRow,
};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

#[path = "../tests/suite/common/interp.rs"]
#[expect(
    dead_code,
    reason = "the benchmark uses only the oracle execution entry points"
)]
mod interp;

const LOWEST_ADDRESS: u64 = 0x7fff_0000;
const ENTRY_PC: u64 = 0x8000_0000;
const SEED: u64 = 0x9e37_79b9_7f4a_7c15;
const WORDS: [u32; 32] = [
    0x00d0_9293,
    0x0050_c0b3,
    0x0070_d293,
    0x0050_c0b3,
    0x0110_9293,
    0x0050_c0b3,
    0x7f80_f313,
    0x0023_0333,
    0x0003_3383,
    0x0013_8433,
    0x0083_3023,
    0x0043_2483,
    0x4014_853b,
    0x00a3_2023,
    0x0033_4583,
    0x00b3_03a3,
    0x0023_1603,
    0x0083_b6b3,
    0x00a4_a733,
    0x00e6_8263,
    0x0016_4263,
    0x0013_f263,
    0x00b0_97b3,
    0x40c7_d833,
    0x0050_d89b,
    0x0107_e933,
    0x0119_79b3,
    0xfff9_8a1b,
    0xabcd_eab7,
    0x0000_0b17,
    0x008b_0be7,
    0xf85f_f06f,
];
const PARALLEL_CHUNK: usize = 1 << 14;
const VALUES_BATCH: usize = 128;

fn facts(bytecode: &Bytecode, program: &[(u64, u32)], cycles: usize) -> Vec<CycleFacts> {
    let mut machine = interp::Machine::new(program, ENTRY_PC, LOWEST_ADDRESS, 20, 256).unwrap();
    machine.set_register(1, SEED).unwrap();
    machine.set_register(2, LOWEST_ADDRESS).unwrap();
    let mut facts = Vec::with_capacity(cycles);
    for _ in 0..cycles {
        let record = machine.step().unwrap();
        let index = bytecode.index_of_pc(record.pc).unwrap();
        let (ram_word_index, ram_pre_value, ram_post_value) =
            record.access.map_or((0, 0, 0), |access| {
                (access.word_index, access.word_before, access.word_after)
            });
        facts.push(CycleFacts {
            bytecode_index: index as u32,
            rs1_value: record.rs1_value,
            rs2_value: record.rs2_value,
            rd_pre_value: record.rd_pre_value,
            rd_post_value: record.rd_post_value,
            ram_word_index,
            ram_pre_value,
            ram_post_value,
            next_pc: record.next_pc,
        });
    }
    machine.run(0).unwrap();
    facts
}

fn checksum(bits: &[BitsRow]) -> u64 {
    bits.as_flattened()
        .iter()
        .fold(0xcbf2_9ce4_8422_2325, |hash, word| {
            (hash ^ word).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

fn median(mut pass: impl FnMut() -> Duration) -> Duration {
    let _ = pass();
    let mut times = [pass(), pass(), pass()];
    times.sort_unstable();
    let [_, middle, _] = times;
    middle
}

fn fill_pass(
    builder: &BitsBuilder<'_>,
    facts: &[CycleFacts],
    bits: &mut [BitsRow],
    expected: u64,
) -> Duration {
    let start = Instant::now();
    builder
        .fill(black_box(facts), black_box(&mut *bits))
        .unwrap();
    let elapsed = start.elapsed();
    assert_eq!(black_box(checksum(bits)), expected);
    elapsed
}

fn parallel_pass(
    pool: &ThreadPool,
    builder: &BitsBuilder<'_>,
    facts: &[CycleFacts],
    bits: &mut [BitsRow],
    expected: u64,
) -> Duration {
    let start = Instant::now();
    pool.install(|| {
        facts
            .par_chunks(PARALLEL_CHUNK)
            .zip(bits.par_chunks_mut(PARALLEL_CHUNK))
            .for_each(|(facts, bits)| {
                builder.fill(black_box(facts), black_box(bits)).unwrap();
            });
    });
    let elapsed = start.elapsed();
    assert_eq!(black_box(checksum(bits)), expected);
    elapsed
}

fn compute_pass(
    layout: &Layout,
    bytecode: &Bytecode,
    facts: &[CycleFacts],
    bits: &[BitsRow],
) -> Duration {
    let start = Instant::now();
    for (facts, bits) in facts.iter().zip(bits) {
        let row = bytecode.rows().get(facts.bytecode_index as usize).unwrap();
        let base = BaseWords::from_facts(facts);
        let witness = WitnessRow::compute(
            black_box(layout),
            black_box(row),
            black_box(&base),
            black_box(bits),
        );
        let _ = black_box(witness);
    }
    start.elapsed()
}

fn values_pass(
    layout: &Layout,
    bytecode: &Bytecode,
    system: &RowSystem,
    facts: &[CycleFacts],
    bits: &[BitsRow],
) -> Duration {
    let mut batch = [WitnessRow([0; 16]); VALUES_BATCH];
    let mut elapsed = Duration::ZERO;
    let mut checksum = 0_u128;
    for (facts, bits) in facts.chunks(VALUES_BATCH).zip(bits.chunks(VALUES_BATCH)) {
        for ((target, facts), bits) in batch.iter_mut().zip(facts).zip(bits) {
            let row = bytecode.rows().get(facts.bytecode_index as usize).unwrap();
            *target = WitnessRow::compute(layout, row, &BaseWords::from_facts(facts), bits);
        }
        let start = Instant::now();
        for witness in batch.iter().take(facts.len()) {
            for row in system.lane_rows() {
                for value in row.values(black_box(witness)) {
                    checksum ^= u128::from(value);
                }
            }
            for row in system.packed_rows() {
                for value in row.values(black_box(witness)) {
                    checksum ^= value.to_raw();
                }
            }
        }
        elapsed += start.elapsed();
    }
    let _ = black_box(checksum);
    elapsed
}

#[expect(
    clippy::print_stdout,
    reason = "the benchmark's contract is eight timing figures"
)]
fn report(name: &str, threads: usize, log_cycles: u32, duration: Duration) {
    let ns_per_cycle = duration.as_secs_f64() * 1e9 / f64::from(1_u32 << log_cycles);
    println!("witness {name} log_T={log_cycles} threads={threads} ns_per_cycle={ns_per_cycle:.3}");
}

fn main() {
    let layout = Layout::new(20, 20, LOWEST_ADDRESS).unwrap();
    let program = WORDS
        .into_iter()
        .enumerate()
        .map(|(index, word)| (ENTRY_PC + 4 * index as u64, word))
        .collect::<Vec<_>>();
    let instructions = program
        .iter()
        .map(|&(pc, word)| decode_instruction(word, pc, false, RV64IMAC_JOLT).unwrap())
        .collect::<Vec<_>>();
    let bytecode = Bytecode::preprocess(&instructions, &layout).unwrap();
    let builder = BitsBuilder::new(&layout, &bytecode).unwrap();
    let system = RowSystem::new(&layout);
    let pool = ThreadPoolBuilder::new().num_threads(12).build().unwrap();
    let all_facts = facts(&bytecode, &program, 1 << 22);
    let mut all_bits = vec![[0; 4]; all_facts.len()];
    builder.fill(&all_facts, &mut all_bits).unwrap();
    for (log_cycles, expected) in [(20, 0xfc7f_3b40_737c_010a), (22, 0x1b85_e435_a7e6_acab)] {
        assert_eq!(
            checksum(all_bits.get(..1 << log_cycles).unwrap()),
            expected,
            "frozen corpus at 2^{log_cycles} cycles"
        );
    }
    for (log_cycles, expected) in [(20, 0xfc7f_3b40_737c_010a), (22, 0x1b85_e435_a7e6_acab)] {
        let facts = all_facts.get(..1 << log_cycles).unwrap();
        let bits = all_bits.get_mut(..1 << log_cycles).unwrap();
        let fill = median(|| fill_pass(&builder, facts, bits, expected));
        let parallel = median(|| parallel_pass(&pool, &builder, facts, bits, expected));
        let compute = median(|| compute_pass(&layout, &bytecode, facts, bits));
        let values = median(|| values_pass(&layout, &bytecode, &system, facts, bits));
        report("fill", 1, log_cycles, fill);
        report("fill", 12, log_cycles, parallel);
        report("compute", 1, log_cycles, compute);
        report("values", 1, log_cycles, values);
    }
}

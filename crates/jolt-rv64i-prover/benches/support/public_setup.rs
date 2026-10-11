//! Public setup cases include allocation and table preparation in their timers.

use super::pipelines::BenchResult;
use super::runner::{measure_witness, summary, Options};
use common::{
    constants::RAM_START_ADDRESS,
    jolt_device::{JoltDevice, MemoryConfig, MemoryLayout},
};
use jolt_field::F128;
use jolt_program::image::decode::decode_instruction;
use jolt_riscv::RV64I;
use jolt_rv64i_arith::{Bytecode, Layout};
use jolt_rv64i_prover::commitment::transparent::TransparentBits;
use jolt_rv64i_prover::plane::Rv64iWitness;
use jolt_rv64i_prover::stages::stage5::evaluate_initial_ram;
use jolt_rv64i_verifier::{
    preprocessing::VerifierPreprocessing,
    statement::{CheckedInputs, Statement},
};
use rayon::ThreadPoolBuilder;
use std::error::Error;
use std::hint::black_box;
use std::sync::Arc;

const LOG_K: usize = 20;

struct PublicFixture {
    statement: Statement,
    preprocessing: VerifierPreprocessing<TransparentBits>,
    a_ram: Vec<F128>,
    r_bit: Vec<F128>,
}
impl PublicFixture {
    fn new(log_t: u8, maximal: bool) -> BenchResult<Self> {
        let memory_layout = MemoryLayout::try_new(&MemoryConfig {
            max_input_size: 8,
            max_output_size: 8,
            max_trusted_advice_size: 0,
            max_untrusted_advice_size: 0,
            stack_size: 0,
            heap_size: 0,
            program_size: Some(8 << LOG_K),
        })?;
        let layout = Layout::new(1, LOG_K, memory_layout.get_lowest_address())?;
        let instruction = decode_instruction(0x0000_006f, RAM_START_ADDRESS, false, RV64I)?;
        let bytecode = Arc::new(Bytecode::preprocess(&[instruction], &layout)?);
        let statement = Statement {
            log_T: log_t,
            entry_pc: RAM_START_ADDRESS,
            device: JoltDevice {
                memory_layout,
                ..JoltDevice::default()
            },
        };
        let empty =
            VerifierPreprocessing::<TransparentBits>::new(Arc::clone(&bytecode), vec![], ())?;
        let checked =
            CheckedInputs::of_statement(&empty, &statement, LOG_K as u8, RAM_START_ADDRESS)?;
        let start = u64::try_from(checked.io().io_mask_end)?;
        let end = 1_u64 << LOG_K;
        let count = if maximal { end - start } else { 64 };
        let image = (0..count)
            .map(|offset| {
                let index = if maximal {
                    start + offset
                } else {
                    start + offset * (end - start) / count
                };
                (index, index.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
            })
            .collect();
        Ok(Self {
            statement,
            preprocessing: VerifierPreprocessing::new(bytecode, image, ())?,
            a_ram: (0..LOG_K)
                .map(|i| F128::from_raw((i as u128 + 1) * 0x1234_5678_9abc_def1))
                .collect(),
            r_bit: (0..6)
                .map(|i| F128::from_raw((i + 1) * 0x9876_5432_10fe_dcba))
                .collect(),
        })
    }
}

pub fn run(options: &Options) -> BenchResult<()> {
    for &log_t in &options.log_t {
        for (name, maximal, initial) in [
            ("init_eval_sparse", false, false),
            ("init_eval_maximal", true, false),
            ("initial_state_sparse", false, true),
            ("initial_state_maximal", true, true),
        ] {
            if !options
                .threads
                .iter()
                .any(|&threads| options.selected_public(name, log_t, threads))
            {
                continue;
            }
            let fixture = PublicFixture::new(log_t, maximal)?;
            let checked = CheckedInputs::of_statement(
                &fixture.preprocessing,
                &fixture.statement,
                LOG_K as u8,
                RAM_START_ADDRESS,
            )?;
            if initial {
                run_case(options, name, log_t, checked.initial_ram().len(), || {
                    Ok(Rv64iWitness::initial_state(
                        checked.layout(),
                        checked.initial_ram(),
                    )?)
                })?;
            } else {
                run_case(options, name, log_t, checked.initial_ram().len(), || {
                    Ok(evaluate_initial_ram(
                        &checked,
                        &fixture.a_ram,
                        &fixture.r_bit,
                    )?)
                })?;
            }
        }
    }
    Ok(())
}

#[expect(
    clippy::print_stdout,
    reason = "phase distributions are benchmark output"
)]
fn run_case<T: Send>(
    options: &Options,
    name: &str,
    log_t: u8,
    ram_words: usize,
    evaluate: impl Fn() -> BenchResult<T> + Sync,
) -> BenchResult<()> {
    for &threads in &options.threads {
        if !options.selected_public(name, log_t, threads) {
            continue;
        }
        let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
        let _ = pool.broadcast(|_| black_box(()));
        let mut samples = Vec::with_capacity(options.samples);
        for _ in 0..options.samples {
            let (value, sample) = pool
                .install(|| {
                    measure_witness(name, 1 << log_t, options.inventory, &evaluate)
                        .map_err(|error| error.to_string())
                })
                .map_err(|error| -> Box<dyn Error> { error.into() })?;
            drop(black_box(value));
            samples.push(sample);
        }
        let cycles = (1_u64 << log_t) as f64;
        let (ns, min, max) = summary(
            samples
                .iter()
                .map(|s| s.elapsed.as_nanos() as f64 / cycles)
                .collect(),
        );
        let peak = samples
            .iter()
            .map(|s| s.allocations.peak_bytes)
            .max()
            .unwrap_or(0);
        println!("witness_pipeline/{name}/{log_t}/{threads} samples={} ns_per_cycle={ns:.6} ms={:.6} min_ns={min:.6} min_ms={:.6} max_ns={max:.6} peak_bytes={peak} ram_words={ram_words} domain_words={} loaded_machine=true", options.samples, ns * cycles / 1e6, min * cycles / 1e6, 1 << LOG_K);
    }
    Ok(())
}

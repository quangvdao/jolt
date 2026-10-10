//! Memory views for the experiment in RV64I hash-based Jolt over binary fields.
//! Run with `RUSTFLAGS='-C target-cpu=native' cargo bench -p jolt-rv64i-kernels
//! --features test-utils --bench memory_machinery -- --log-t 20,22 --threads 1,12
//! --samples 5`. Source preparation is outside the measured views. The models
//! price a word for `fold_words` and a cycle for the other views; they are not
//! independent performance requirements.

pub mod support;

use jolt_field::F128;
use jolt_rv64i_kernels::memory::{
    address_column, fold_words, row_weights, MemoryError, MemoryTrace, RowWeight,
};
use jolt_rv64i_kernels::packed::lift::WordLift;
use jolt_rv64i_kernels::packed::scatter::{ScatterError, ScatterPlan};
use jolt_rv64i_kernels::round::eq::eq_table;
use jolt_rv64i_kernels::source::{CycleSource, PrepareRequest, SourceError, ValidatedTrace};
use jolt_rv64i_kernels::synth::{SynthError, SynthProfile, SyntheticTrace};
use std::hint::black_box;
use std::mem::size_of;
use std::sync::Arc;
use support::allocator::CountingAllocator;
use support::{run_machinery, Clock, MachineryKernel, RunnerError};
use thiserror::Error;

#[derive(Debug, Error)]
enum BenchError {
    #[error(transparent)]
    Memory(#[from] MemoryError),
    #[error(transparent)]
    Scatter(#[from] ScatterError),
    #[error(transparent)]
    Source(#[from] SourceError),
    #[error(transparent)]
    Synth(#[from] SynthError),
    #[error("unknown memory machinery case {name}")]
    Case { name: String },
    #[error("prepared source omitted its {group} group")]
    Group { group: &'static str },
    #[error("word lift received {length} weights instead of 64")]
    LiftWeights { length: usize },
}

struct Inputs {
    trace: Arc<ValidatedTrace<SyntheticTrace>>,
    memory: MemoryTrace,
    words: Vec<u64>,
    lift: WordLift,
    fold_point: Vec<F128>,
    address_point: Vec<F128>,
    cycle_point: Vec<F128>,
}

impl Inputs {
    fn new(log_t: usize) -> Result<Arc<Self>, BenchError> {
        let source = Arc::new(SyntheticTrace::new(
            SynthProfile::AllRows,
            log_t,
            1 << 20,
            71,
        )?);
        let (ram, registers, store) = SyntheticTrace::memory_columns();
        let (trace, groups) = ValidatedTrace::prepare(
            Arc::clone(&source),
            PrepareRequest {
                present: vec![ram.to_vec(), registers.to_vec()],
                optional: vec![vec![store]],
            },
        )?;
        let mut present = groups.present.into_iter();
        let mut optional = groups.optional.into_iter();
        let ram = present.next().ok_or(BenchError::Group { group: "RAM" })?;
        let registers = present
            .next()
            .ok_or(BenchError::Group { group: "registers" })?;
        let store = optional
            .next()
            .ok_or(BenchError::Group { group: "store" })?;
        let memory = MemoryTrace::new(ram, registers, store)?;
        let weights = eq_table(&point(6, 0x913), None);
        let weights: [F128; 64] =
            weights
                .try_into()
                .map_err(|weights: Vec<F128>| BenchError::LiftWeights {
                    length: weights.len(),
                })?;
        Ok(Arc::new(Self {
            words: (0..source.cycles())
                .map(|cycle| source.trace_word(5, cycle))
                .collect(),
            address_point: point(memory.address_bits(), 0x419),
            memory,
            trace: Arc::new(trace),
            lift: WordLift::new(&weights),
            fold_point: point(log_t.min(12), 0x217),
            cycle_point: point(log_t, 0x613),
        }))
    }
}

fn point(length: usize, seed: u128) -> Vec<F128> {
    (0..length)
        .map(|bit| F128::from_raw(seed + bit as u128 * 37))
        .collect()
}

enum View {
    FoldWords,
    AddressColumn,
    RowWeights {
        plan: Box<ScatterPlan<SyntheticTrace>>,
        construct_ns: f64,
        bytes: usize,
    },
}

struct MemoryMachinery {
    inputs: Arc<Inputs>,
    view: View,
    output: Vec<F128>,
}

impl MemoryMachinery {
    fn new(name: &str, inputs: Arc<Inputs>) -> Result<Self, BenchError> {
        let view = match name {
            "fold_words" => View::FoldWords,
            "address_column" => View::AddressColumn,
            "row_weights" => {
                let before = CountingAllocator::live_bytes();
                let start = Clock::start();
                let plan = Box::new(ScatterPlan::new(Arc::clone(&inputs.trace))?);
                let construct_ns = start.elapsed().as_nanos() as f64;
                let bytes = CountingAllocator::live_bytes() - before;
                View::RowWeights {
                    plan,
                    construct_ns,
                    bytes,
                }
            }
            _ => {
                return Err(BenchError::Case {
                    name: name.to_owned(),
                })
            }
        };
        Ok(Self {
            inputs,
            view,
            output: Vec::new(),
        })
    }
}

impl MachineryKernel for MemoryMachinery {
    type Error = BenchError;

    fn operations(&self) -> usize {
        match self.view {
            View::FoldWords => self.inputs.words.len(),
            _ => self.inputs.memory.cycles(),
        }
    }

    fn plan_construction_ns(&self) -> Option<f64> {
        match self.view {
            View::RowWeights { construct_ns, .. } => Some(construct_ns),
            _ => None,
        }
    }

    #[expect(
        clippy::print_stdout,
        reason = "output capacity belongs to the machinery record"
    )]
    fn memory_bytes(&self) -> Option<(usize, usize)> {
        let output_elements = match &self.view {
            View::FoldWords => self.inputs.words.len() >> self.inputs.fold_point.len(),
            View::AddressColumn => self.inputs.memory.cycles(),
            View::RowWeights { plan, .. } => plan.bytecode_rows(),
        };
        println!(
            "memory_machinery/output output_bytes={}",
            output_elements * size_of::<F128>()
        );
        match &self.view {
            View::RowWeights { plan, bytes, .. } => {
                Some((*bytes, plan.cycles() * size_of::<F128>()))
            }
            _ => Some((0, 0)),
        }
    }

    fn run(&mut self) -> Result<F128, BenchError> {
        let inputs = black_box(self.inputs.as_ref());
        self.output = match &self.view {
            View::FoldWords => fold_words(&inputs.words, &inputs.lift, &inputs.fold_point)?,
            View::AddressColumn => address_column(&inputs.memory, &inputs.address_point)?,
            View::RowWeights { plan, .. } => row_weights(plan, RowWeight::Eq(&inputs.cycle_point))?,
        };
        let _ = black_box(&self.output);
        Ok(self.output.first().copied().unwrap_or(F128::from_raw(0)))
    }
}

#[expect(
    clippy::print_stdout,
    reason = "models and machine conditions are benchmark output"
)]
fn main() -> Result<(), RunnerError> {
    println!("memory_machinery/models fold_words_ns_per_word=4.3 address_column_ns_per_cycle=3.1 row_weights_ns_per_cycle=3.2 requirements=none loaded_machine=true");
    run_machinery(
        &[
            ("fold_words", None),
            ("address_column", None),
            ("row_weights", None),
        ],
        Inputs::new,
        |name, inputs, _threads| MemoryMachinery::new(name, inputs),
    )
}

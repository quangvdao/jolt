use std::path::Path;

use common::jolt_device::MemoryLayout;
use jolt_program::{
    execution::{OwnedTrace, SourceTraceRow},
    image::DecodeMode,
};
use jolt_rv64i_trace::{adapt, preprocess, Execution, Program};

use super::{
    criterion_output_dir, read_adapter_measurements_at,
    source_trace_gen::{SourceTraceGenObjective, SourceTraceGenSetup},
};
use crate::objective::{Objective, OptimizationObjective, PerformanceObjective};

pub const RV64I_TRACE_ADAPT: OptimizationObjective = OptimizationObjective::Performance(
    PerformanceObjective::Rv64iTraceAdapt(Rv64iTraceAdaptObjective),
);

/// Pool labels shared by the benchmark and its result reader.
pub const THREAD_POOLS: [&str; 2] = ["one_thread", "default_pool"];

/// Criterion medians normalized by the benchmark's recorded geometry.
/// The peak counts live allocator-requested bytes above the pretraced baseline;
/// it includes layout and Rayon allocations, and excludes allocator overhead and RSS.
#[derive(Debug)]
pub struct Rv64iTraceAdaptMeasurement {
    pub program: &'static str,
    pub pool: &'static str,
    pub ns_per_padded_cycle: f64,
    pub ns_per_executed_row: f64,
    pub facts_buffer_bytes: u64,
    pub peak_incremental_allocated_bytes: u64,
}

/// One pretraced workload. The benchmark keeps only one workload's rows live.
pub struct Rv64iTraceAdaptSetup {
    pub label: &'static str,
    pub program: Program,
    pub memory_layout: MemoryLayout,
    pub rows: OwnedTrace<SourceTraceRow>,
}

/// Measures conversion from RV64I execution rows to padded cycle facts.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Rv64iTraceAdaptObjective;

impl Rv64iTraceAdaptObjective {
    /// Reads each workload and pool's medians, facts-buffer size and measured peak.
    pub fn read_measurements(
        &self,
        work_dir: &Path,
        baseline: &str,
    ) -> Option<Vec<Rv64iTraceAdaptMeasurement>> {
        read_adapter_measurements_at(&criterion_output_dir(work_dir), baseline)
    }

    /// Decodes and traces the existing source-trace workload before timing.
    #[expect(
        clippy::expect_used,
        reason = "hand-assembled benchmark fixtures must decode and trace"
    )]
    pub fn prepare(&self, mut source: SourceTraceGenSetup) -> Rv64iTraceAdaptSetup {
        let program = preprocess(
            source.program.elf_bytes(),
            source.inputs.memory_config,
            DecodeMode::Strict,
        )
        .expect("RV64I adapter benchmark decode failed");
        source.inputs.memory_config = program.memory_config;
        let memory_layout = MemoryLayout::try_new(&program.memory_config)
            .expect("RV64I adapter benchmark memory layout failed");
        let rows = SourceTraceGenObjective.run_source(&source);
        assert_eq!(rows.rows().len(), source.row_count);
        Rv64iTraceAdaptSetup {
            label: source.label,
            program,
            memory_layout,
            rows,
        }
    }

    /// Converts the pretraced rows; allocation, conversion and padding are included.
    #[expect(
        clippy::expect_used,
        reason = "the checked benchmark fixtures must adapt"
    )]
    pub fn run_adapt(&self, setup: &Rv64iTraceAdaptSetup) -> Execution {
        adapt(
            &setup.program.bytecode,
            &setup.program.image,
            &setup.memory_layout,
            setup.program.entry_pc,
            setup.rows.rows(),
        )
        .expect("RV64I adapter benchmark conversion failed")
    }
}

impl Objective for Rv64iTraceAdaptObjective {
    type Setup = [SourceTraceGenSetup; 3];

    fn name(&self) -> &str {
        "rv64i_trace_adapt"
    }

    fn description(&self) -> String {
        "RV64I trace adaptation over ALU, memory and call-frame workloads; one-thread medians normalized per padded cycle".to_owned()
    }

    fn setup(&self) -> Self::Setup {
        SourceTraceGenObjective.setup()
    }

    fn run(&self, setup: Self::Setup) {
        for source in setup {
            let prepared = self.prepare(source);
            drop(std::hint::black_box(self.run_adapt(&prepared)));
        }
    }

    fn units(&self) -> Option<&str> {
        Some("ns/cycle")
    }
}

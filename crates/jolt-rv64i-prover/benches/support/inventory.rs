//! Exact capacity laws from the two specs' Memory tables. Equal sizes can have
//! several owners; the inventory lists every candidate, never a guessed owner.

use super::allocator::{AllocationRecorder, CountingAllocator};
use super::timing::PHASES;
use jolt_rv64i_prover::plane::Rv64iWitness;

pub struct Geometry {
    pub fold_entries: usize,
    pub row_entries: usize,
    pub fold_lengths: Vec<usize>,
    pub selector_columns: usize,
}

struct MemoryRow {
    object: &'static str,
    law: &'static str,
    spec: &'static str,
    sizes: Vec<usize>,
}

pub struct Inventory {
    rows: Vec<MemoryRow>,
}
impl Inventory {
    pub fn new(witness: &Rv64iWitness, geometry: &Geometry) -> Self {
        let cycles = witness.bits.len();
        let bytecode = witness.bytecode.rows().len();
        let layout = &witness.layout;
        let rows = vec![
            MemoryRow {
                object: "lanes",
                law: "48*T",
                spec: "adapters:155",
                sizes: vec![48 * cycles],
            },
            MemoryRow {
                object: "lane_or_core_tail",
                law: "T",
                spec: "adapters:155;kernels:306",
                sizes: vec![cycles],
            },
            MemoryRow {
                object: "prepared_row_cache",
                law: "2*K",
                spec: "adapters:156",
                sizes: vec![2 * bytecode],
            },
            MemoryRow {
                object: "chunk_bytes",
                law: "d_b*T|d_a*T",
                spec: "adapters:157",
                sizes: vec![
                    layout.bytecode_ra().len() * cycles,
                    layout.ram_ra().len() * cycles,
                ],
            },
            MemoryRow {
                object: "selector_bytes",
                law: "distinct_selector_columns*T",
                spec: "adapters:157;kernels:315",
                sizes: vec![geometry.selector_columns * cycles],
            },
            MemoryRow {
                object: "scatter_plan_arrays",
                law: "2*T (two arrays)",
                spec: "adapters:158;kernels:325",
                sizes: vec![2 * cycles],
            },
            MemoryRow {
                object: "fold_worker_buckets",
                law: "16*FoldLayout.entries()",
                spec: "kernels:308",
                sizes: vec![16 * geometry.fold_entries],
            },
            MemoryRow {
                object: "fold_row_buckets",
                law: "16*FoldLayout.row_entries()",
                spec: "kernels:308",
                sizes: vec![16 * geometry.row_entries],
            },
            MemoryRow {
                object: "fold_or_short_weight",
                law: "16*RouterShape.fold_len()",
                spec: "kernels:311",
                sizes: geometry.fold_lengths.iter().map(|n| 16 * n).collect(),
            },
            MemoryRow {
                object: "short_second_buffers",
                law: "8*RouterShape.fold_len()",
                spec: "kernels:331",
                sizes: geometry.fold_lengths.iter().map(|n| 8 * n).collect(),
            },
            MemoryRow {
                object: "row_field_tables",
                law: "16*K",
                spec: "kernels:310,312",
                sizes: vec![16 * bytecode],
            },
            MemoryRow {
                object: "cycle_field_tables",
                law: "16*T",
                spec: "kernels:306,309,313,314,316,318",
                sizes: vec![16 * cycles],
            },
            MemoryRow {
                object: "cycle_second_buffers",
                law: "8*T",
                spec: "kernels:314,316,318",
                sizes: vec![8 * cycles],
            },
            MemoryRow {
                object: "materialised_selector_or_chunk",
                law: "16*(T/16)=T at fourth bind",
                spec: "kernels:315,317",
                sizes: vec![cycles],
            },
        ];
        Self { rows }
    }

    #[expect(
        clippy::print_stdout,
        reason = "capacity laws are the inventory output contract"
    )]
    pub fn print_laws(&self, log_t: u8, threads: usize) {
        for row in &self.rows {
            println!("inventory_law/{log_t}/{threads} object={} law={:?} exact_sizes={:?} spec={} loaded_machine=true",row.object,row.law,row.sizes,row.spec);
        }
    }

    /// Prints every event after scoped workers have joined. Any unexplained
    /// event or overflow fails acceptance independently of timing thresholds.
    #[expect(
        clippy::print_stdout,
        reason = "allocation inventory is benchmark output"
    )]
    pub fn print(&self, id: &str) -> bool {
        let mut unmatched = 0;
        let mut entries = 0;
        for (index, (size, phase)) in CountingAllocator::entries().enumerate() {
            entries += 1;
            let matches: Vec<_> = self
                .rows
                .iter()
                .filter(|row| row.sizes.contains(&size))
                .map(|row| row.object)
                .collect();
            let phase = PHASES.get(phase).copied().unwrap_or("warm_prepare");
            if matches.is_empty() {
                unmatched += 1;
            }
            println!("inventory/{id} entry={index} bytes={size} phase={phase} matches={} loaded_machine=true",if matches.is_empty() {"unmatched".to_owned()} else {matches.join("|")});
        }
        let overflow = CountingAllocator::overflow();
        println!("inventory/{id} entries={entries} unmatched={unmatched} overflow={overflow} loaded_machine=true");
        unmatched != 0 || overflow != 0
    }
}

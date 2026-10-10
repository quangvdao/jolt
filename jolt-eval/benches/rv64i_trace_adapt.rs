use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::mem::size_of;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, Criterion, SamplingMode, Throughput};
use jolt_eval::objective::performance::{
    criterion_output_dir,
    rv64i_trace_adapt::{Rv64iTraceAdaptObjective, THREAD_POOLS},
    source_trace_gen::SourceTraceGenObjective,
};
use jolt_eval::Objective as _;
use jolt_rv64i_arith::CycleFacts;
use rayon::ThreadPoolBuilder;
use serde_json::{json, Map};

struct AllocationCounter {
    live: AtomicUsize,
    peak: AtomicUsize,
}

impl AllocationCounter {
    fn record_allocation(&self, bytes: usize) {
        let live = self.live.fetch_add(bytes, Ordering::SeqCst) + bytes;
        self.peak.fetch_max(live, Ordering::SeqCst);
    }

    fn begin_measurement(&self) -> usize {
        let baseline = self.live.load(Ordering::SeqCst);
        self.peak.store(baseline, Ordering::SeqCst);
        baseline
    }

    fn peak_since(&self, baseline: usize) -> usize {
        self.peak.load(Ordering::SeqCst).saturating_sub(baseline)
    }
}

// SAFETY: Every allocation operation delegates unchanged pointers, layouts and
// sizes to System; bookkeeping never reads or writes the allocated memory.
unsafe impl GlobalAlloc for AllocationCounter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: GlobalAlloc's caller supplies a valid layout for System.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            self.record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: GlobalAlloc's caller supplies a valid layout for System.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            self.record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: The caller supplies a live allocation and its original layout.
        unsafe { System.dealloc(pointer, layout) };
        self.live.fetch_sub(layout.size(), Ordering::SeqCst);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: The caller supplies a live allocation, its layout and a valid
        // new size, as required by System's GlobalAlloc implementation.
        let replacement = unsafe { System.realloc(pointer, layout, new_size) };
        if !replacement.is_null() {
            if new_size >= layout.size() {
                self.record_allocation(new_size - layout.size());
            } else {
                self.live
                    .fetch_sub(layout.size() - new_size, Ordering::SeqCst);
            }
        }
        replacement
    }
}

#[global_allocator]
static ALLOCATOR: AllocationCounter = AllocationCounter {
    live: AtomicUsize::new(0),
    peak: AtomicUsize::new(0),
};

#[expect(
    clippy::unwrap_used,
    clippy::print_stderr,
    reason = "benchmark setup and artifact failures abort the measurement; results are printed for the runner"
)]
fn bench(c: &mut Criterion) {
    std::env::remove_var("TRACER_PARALLEL");
    std::env::remove_var("JOLT_BACKTRACE");
    let pools = [
        (
            THREAD_POOLS[0],
            ThreadPoolBuilder::new().num_threads(1).build().unwrap(),
        ),
        (THREAD_POOLS[1], ThreadPoolBuilder::new().build().unwrap()),
    ];
    let objective = Rv64iTraceAdaptObjective;
    for source in SourceTraceGenObjective.setup() {
        let setup = objective.prepare(source);
        let executed = setup.rows.rows().len();
        let mut padded = 0;
        let mut facts_bytes = 0;
        let mut peaks = Map::new();
        for (label, pool) in &pools {
            pool.broadcast(|_| {});
            let baseline = ALLOCATOR.begin_measurement();
            let execution = pool.install(|| objective.run_adapt(&setup));
            let peak_bytes = ALLOCATOR.peak_since(baseline);
            assert_eq!(execution.facts.capacity(), execution.facts.len());
            padded = execution.facts.len();
            facts_bytes = padded * size_of::<CycleFacts>();
            assert!(peak_bytes >= facts_bytes);
            drop(execution);
            peaks.insert((*label).to_owned(), json!(peak_bytes));
            eprintln!("{}/{label}: executed_rows={executed} padded_cycles={padded} facts_buffer_bytes={facts_bytes} peak_incremental_allocated_bytes={peak_bytes}", setup.label);
        }
        let group_name = format!("{}/{}", objective.name(), setup.label);
        let directory = criterion_output_dir(Path::new(".")).join(group_name.replace('/', "_"));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(
            directory.join("counts.json"),
            json!({
                "executed_rows": executed, "padded_cycles": padded,
                "facts_buffer_bytes": facts_bytes,
                "trace_sized_adapter_buffers": 1,
                "peak_incremental_allocated_bytes": peaks,
                "peak_allocation_metric": "live allocator-requested bytes above pretraced baseline; allocator overhead and RSS excluded",
                "default_pool_threads": pools[1].1.current_num_threads(),
            })
            .to_string(),
        )
        .unwrap();
        let mut group = c.benchmark_group(group_name);
        group.sample_size(10);
        group.sampling_mode(SamplingMode::Flat);
        group.warm_up_time(Duration::from_secs(3));
        group.measurement_time(Duration::from_secs(10));
        group.throughput(Throughput::Elements(padded as u64));
        for (label, pool) in &pools {
            group.bench_function(*label, |b| {
                b.iter_custom(|iterations| {
                    let mut elapsed = Duration::ZERO;
                    for _ in 0..iterations {
                        let (duration, execution) = pool.install(|| {
                            let start = Instant::now();
                            let execution = objective.run_adapt(black_box(&setup));
                            (start.elapsed(), execution)
                        });
                        elapsed += duration;
                        drop(black_box(execution));
                    }
                    elapsed
                });
            });
        }
        group.finish();
    }
    if let Some(report) = objective.read_measurements(Path::new("."), "new") {
        for measurement in report {
            eprintln!("{}/{}: median_ns_per_padded_cycle={:.3} median_ns_per_executed_row={:.3} exact_facts_buffer_bytes={} peak_incremental_allocated_bytes={} (allocator overhead/RSS excluded)",
                measurement.program, measurement.pool, measurement.ns_per_padded_cycle,
                measurement.ns_per_executed_row, measurement.facts_buffer_bytes,
                measurement.peak_incremental_allocated_bytes);
        }
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().output_directory(&criterion_output_dir(Path::new(".")));
    targets = bench
}
criterion_main!(benches);

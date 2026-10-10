#![cfg(feature = "test-utils")]

#[path = "../benches/support/allocator.rs"]
mod allocator;

#[expect(
    clippy::unwrap_used,
    reason = "test setup and checked pass failures must fail the test"
)]
mod tests {
    use super::allocator::{AllocationMeasurement, CountingAllocator, RAYON_WORKER_ALLOWANCE};
    use jolt_field::F128;
    use jolt_rv64i_kernels::packed::buckets::NibbleBuckets;
    use jolt_rv64i_kernels::packed::pool::ScratchPool;
    use jolt_rv64i_kernels::packed::scatter::ScatterPlan;
    use jolt_rv64i_kernels::source::ValidatedTrace;
    use jolt_rv64i_kernels::synth::{SynthProfile, SyntheticTrace};
    use rayon::prelude::*;
    use rayon::ThreadPoolBuilder;
    use std::sync::Arc;

    #[test]
    fn prepared_scatter_and_bucket_passes_allocate_nothing() {
        let workers = ThreadPoolBuilder::new().num_threads(12).build().unwrap();
        let runtime_allocs = RAYON_WORKER_ALLOWANCE.allocs * workers.current_num_threads();
        let runtime_bytes = RAYON_WORKER_ALLOWANCE.bytes * workers.current_num_threads();
        workers.install(|| {
            let source = Arc::new(SyntheticTrace::new(SynthProfile::AllRows, 16, 256, 81).unwrap());
            let plan = ScatterPlan::new(Arc::new(ValidatedTrace::new(source).unwrap())).unwrap();
            let mut weights = vec![F128::from_raw(0); plan.cycles()];
            let mut output = vec![F128::from_raw(0); plan.bytecode_rows()];
            plan.scatter_into(
                |cycle| F128::from_raw(cycle as u128 + 1),
                &mut weights,
                &mut output,
            )
            .unwrap();
            output.fill(F128::from_raw(0));
            let before = CountingAllocator::live_bytes();
            let measurement = AllocationMeasurement::begin();
            plan.scatter_into(
                |cycle| F128::from_raw(cycle as u128 + 1),
                &mut weights,
                &mut output,
            )
            .unwrap();
            let stats = measurement.finish();
            assert!(stats.allocs <= runtime_allocs);
            assert!(stats.peak_bytes <= runtime_bytes);
            assert!(stats.final_bytes <= runtime_bytes);
            assert!((before..=before + runtime_bytes).contains(&CountingAllocator::live_bytes()));

            let pool = ScratchPool::new(16 * 16).unwrap();
            let guards: Vec<_> = (0..12).map(|_| pool.take().unwrap()).collect();
            drop(guards);
            let words: Vec<_> = (0..65536_u64)
                .map(|word| word.wrapping_mul(0x9e37_79b9_7f4a_7c15))
                .collect();
            let pass = || {
                words.par_chunks(1024).for_each(|words| {
                    let mut guard = pool.take().unwrap();
                    let mut buckets = NibbleBuckets::new(&mut guard).unwrap();
                    for &word in words {
                        for (positions, byte) in buckets
                            .positions_mut()
                            .as_chunks_mut::<2>()
                            .0
                            .iter_mut()
                            .zip(word.to_le_bytes())
                        {
                            positions[0][usize::from(byte & 15)] += F128::from_raw(1);
                            positions[1][usize::from(byte >> 4)] += F128::from_raw(1);
                        }
                    }
                });
            };
            pass();
            pool.zero().unwrap();
            let measurement = AllocationMeasurement::begin();
            pass();
            let stats = measurement.finish();
            assert!(stats.allocs <= runtime_allocs);
            assert!(stats.peak_bytes <= runtime_bytes);
            assert!(stats.final_bytes <= runtime_bytes);
        });
    }
}

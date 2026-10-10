#![cfg(feature = "test-utils")]

#[path = "../benches/support/scatter.rs"]
mod scatter;

#[expect(clippy::unwrap_used, reason = "test setup failures must fail the test")]
mod tests {
    use super::scatter::PartitionedScatter;
    use jolt_field::F128;
    use jolt_rv64i_kernels::source::CycleSource;
    use jolt_rv64i_kernels::synth::SyntheticTrace;
    use std::sync::Arc;
    const ROWS: usize = 1 << 20;
    use jolt_rv64i_kernels::synth::SynthProfile;
    use rayon::ThreadPoolBuilder;

    #[test]
    fn pair_emission_and_application_equal_direct_cycle_order_updates() {
        for log_t in [1, 13] {
            for profile in [SynthProfile::Local, SynthProfile::AllRows] {
                let source = Arc::new(SyntheticTrace::new(profile, log_t, ROWS, 41).unwrap());
                let mut expected = vec![F128::from_raw(0); ROWS];
                for cycle in 0..source.cycles() {
                    expected[source.bytecode_index(cycle)] += F128::from_raw(
                        u128::from(source.trace_word(0, cycle))
                            | (u128::from(source.trace_word(1, cycle)) << 64),
                    );
                }
                for threads in [1, 3] {
                    let pool = ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .unwrap();
                    let mut scatter = PartitionedScatter::new(Arc::clone(&source)).unwrap();
                    assert_eq!(scatter.cycles(), 1 << log_t);
                    let _ = pool.install(|| scatter.run());
                    assert_eq!(scatter.output, expected);
                }
            }
        }
    }
}

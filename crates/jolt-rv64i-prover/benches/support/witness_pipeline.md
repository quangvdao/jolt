# Witness pipeline measurements

Run the benchmark with the workspace toolchain and a shared target directory:

```sh
RUSTFLAGS='-C target-cpu=native' cargo bench -p jolt-rv64i-prover \
  --features test-utils --bench witness_pipeline -- \
  --log-t 20,22 --threads 1,12 --samples 5 inventory
```

The existing runner parses size, pool, samples and inventory options. The case
filter is `witness_pipeline/pipeline/<log_t>/<threads>`. The existing counting
allocator records requested bytes, including Arc headers, rather than resident
pages. Each phase's peak excludes storage already resident at its start;
`inventory` prints requests of at least T bytes and fails on recorder overflow.

The fixture executes the seeded adapter program, with 2^20 bytecode rows and
2^20 RAM words, for exactly T cycles using the independent interpreter. Its
records become architectural trace rows during untimed setup. Adaptation calls
`jolt_rv64i_trace::adapt`; construction calls `Rv64iWitness::from_facts`;
preparation calls the production `SharedSource::prepare`, followed by the lazy
`SharedSource::plan` in its own scatter phase. Statement admission, initial RAM
copy, source metadata for validation-only, canonical router selectors, warmed
pools and destruction are untimed. No operation is stubbed and all typed errors
propagate to a failing benchmark process.

Validation-only calls `ValidatedTrace::new` after production preparation. It is
a diagnostic alternative to the fused validation/gather walk in preparation;
its time must not be added to the production total. Preparation-plus-scatter
and production totals are medians of per-sample sums, rather than sums of phase
medians. The production total covers adaptation, construction, source
preparation and lazy scatter construction. It does not include the lanes,
commitment, sum-checks, ELF preprocessing or tracing. The cut execution has no
termination, output check or padded stall suffix. Both sizes are executed;
neither uses `Rv64iWitness::synthetic`.

## Baseline

Apple M4 Max, 16 logical/physical cores, 64 GiB RAM; native target flags;
five samples per case. This is loaded-machine evidence, not a budget gate.
Load averages (1/5/15 minutes) were 33.96/22.21/16.53 before the run and
32.76/22.16/16.54 afterwards. Phase medians and maximum incremental peaks:

| log T | Threads | Phase | ns/cycle | ms | Peak MiB |
|---|---|---|---|---|---|
| 20 | 1 | adapt | 5.376 | 5.637 | 72.000 |
| 20 | 1 | construct | 50.902 | 53.375 | 96.000 |
| 20 | 1 | validate | 8.773 | 9.199 | 2.000 |
| 20 | 1 | prepare | 13.770 | 14.439 | 20.034 |
| 20 | 1 | scatter | 3.649 | 3.826 | 4.750 |
| 20 | 12 | adapt | 1.416 | 1.485 | 72.000 |
| 20 | 12 | construct | 37.183 | 38.990 | 96.000 |
| 20 | 12 | validate | 2.032 | 2.131 | 2.000 |
| 20 | 12 | prepare | 2.860 | 2.999 | 20.034 |
| 20 | 12 | scatter | 0.772 | 0.810 | 4.750 |
| 22 | 1 | adapt | 4.312 | 18.085 | 288.000 |
| 22 | 1 | construct | 35.003 | 146.813 | 360.000 |
| 22 | 1 | validate | 7.416 | 31.103 | 2.000 |
| 22 | 1 | prepare | 12.034 | 50.474 | 74.105 |
| 22 | 1 | scatter | 3.547 | 14.879 | 19.000 |
| 22 | 12 | adapt | 0.927 | 3.889 | 288.000 |
| 22 | 12 | construct | 33.713 | 141.403 | 360.000 |
| 22 | 12 | validate | 1.063 | 4.458 | 2.000 |
| 22 | 12 | prepare | 1.626 | 6.820 | 74.105 |
| 22 | 12 | scatter | 0.475 | 1.991 | 19.000 |

At 2^22 on 12 threads, the production total was 154.191 ms (36.762 ns/cycle),
of which construction was 141.403 ms. This constructor figure includes its
allocation/zero-fill and canonical initial RAM, not just its replay loop.
At that size the inventory's large requests were: adaptation 72T; construction
8 MiB final RAM plus 32T, 40T and 16T buffers (the latter three each have a
16-byte Arc header); preparation 5T, 5T and 8T groups; scatter two 2T buffers.
The 2 MiB temporary row cache is below the T-byte inventory cutoff at log T 22.
No recorder overflow occurred.

## Acceptance coverage

| Contract | Check |
|---|---|
| Requested sizes/pools, named timings, untimed fixture, requested-byte peaks | The benchmark command above; phase and inventory output, including overflow rejection |
| Executed fixture remains compatible with the adapters bench | `adapters/source/20/12 --samples 1 inventory` smoke run |
| Fact replay and nonaccess RAM reads | `facts_match_the_arithmetisation_replay_and_keep_nonaccess_ram_reads` |
| Earliest typed failure and register/RAM ordering | `fact_differences_name_the_first_cycle_and_register_field`; `load_facts_reject_ram_pre_values_and_register_reads_before_row_generation` |
| Absent operands and RAM allocation failure | `absent_operands_and_nonaccess_ram_facts_do_not_replace_replayed_reads`; `constructors_return_a_typed_error_for_unallocatable_ram` |
| Decoded rows and counts against committed ground truth | `decoded_rows_match_committed_rows_in_both_constructors` |
| Validated digits, byte groups, shared ownership | `source_matches_committed_columns_words_and_weighted_sum`; `shared_source_enforces_group_ownership_and_shared_lifetimes` |
| Scatter against direct algebraic sums | `scatter_equals_cycle_order_summation_in_small_and_large_domains`; `scatter_fused_chunk_emission_equals_cycle_summation` |
| Whole-phase performance retention | Alternating saved release executables, same fixture and native flags; results below |

Baseline checks: prover all-target clippy with test-utils and denied warnings;
prover nextest 75/75; formatting and style invariant checks. No new permanent
old-versus-new tests or diagnostic test hooks were introduced.

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
| Parallel chunk counts and earliest generation fault, with earlier replay faults taking precedence | `parallel_fact_generation_keeps_counts_and_first_fault_across_chunks`, using literal JAL and LD instructions on 1/12-thread pools |
| Absent operands and RAM allocation failure | `absent_operands_and_nonaccess_ram_facts_do_not_replace_replayed_reads`; `constructors_return_a_typed_error_for_unallocatable_ram` |
| Decoded rows and counts against committed ground truth | `decoded_rows_match_committed_rows_in_both_constructors` |
| Validated digits, byte groups, shared ownership | `source_matches_committed_columns_words_and_weighted_sum`; `shared_source_enforces_group_ownership_and_shared_lifetimes` |
| Scatter against direct algebraic sums | `scatter_equals_cycle_order_summation_in_small_and_large_domains`; `scatter_fused_chunk_emission_equals_cycle_summation` |
| Whole-phase performance retention | Alternating saved release executables, same fixture and native flags; results below |

Baseline checks: prover all-target clippy with test-utils and denied warnings;
prover nextest 75/75; formatting and style invariant checks. No new permanent
old-versus-new tests or diagnostic test hooks were introduced.

## Constructor retention

The retained change generates committed bits, decoded rows and variant counts
in parallel using the canonical `BitsBuilder` and `DigitFields`, then runs the
state replay in order. It reduces generation faults to the earliest cycle but
reports that fault only after the replay checks that cycle's pre-state. It adds
no trace-sized intermediate. One-thread pools keep generation fused with replay.
The private replay input selects these cases at compile time; keeping replay
out of line removed a measured regression of the fused case.

Three alternating baseline/candidate pairs used saved release executables with
identical native flags and seven samples per case per block. The table reports
the median of the three block medians of the whole constructor, including
allocation, canonical RAM and state replay. Load averages at the six block
starts were 16.32/18.25/17.20, 15.97/18.14/17.17, 17.85/18.46/17.29,
17.54/18.38/17.27, 16.86/18.23/17.22 and 16.47/18.12/17.19; afterwards
16.19/18.04/17.16. All are loaded-machine measurements.

| log T | Threads | Before ns/cycle | Before ms | After ns/cycle | After ms | Reduction |
|---|---|---|---|---|---|---|
| 20 | 1 | 32.049 | 33.606 | 30.930 | 32.432 | 3.49% |
| 20 | 12 | 31.999 | 33.554 | 18.260 | 19.147 | 42.94% |
| 22 | 1 | 32.334 | 135.619 | 30.173 | 126.554 | 6.68% |
| 22 | 12 | 31.571 | 132.417 | 17.617 | 73.889 | 44.20% |

The constructor peak remains 96 MiB at log T 20 and 360 MiB at log T 22,
plus at most 336 bytes of Rayon bookkeeping in these blocks. Earlier candidate
builds regressed the single-thread phase and were revised before retention.

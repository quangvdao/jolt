# Witness pipeline measurements

Run the benchmark with the workspace toolchain and a shared target directory:

```sh
RUSTFLAGS='-C target-cpu=native' cargo bench -p jolt-rv64i-prover \
  --features test-utils --bench witness_pipeline -- \
  --log-t 20,22 --threads 1,12 --samples 5 inventory
```

The existing runner owns configuration, sampling, phase timing, allocation
intervals and summaries, and parses size, pool, samples and inventory options. The case
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
| Session rejection and decoded dimensions | `prepare_rejects_taken_members_and_missing_selectors`; `both_registries_reject_mismatched_decoded_length` |
| Protocol bytes and statement-failure behavior stay intact | `mixed_registry_proofs_match_reference`; `optimized_corpus_proofs_match_reference`; `optimized_rejects_the_same_public_statement_failures`; `counting_loop_proof_has_the_frozen_encoding_and_rejects_malformed_envelopes` |
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

## Scatter retention and final phases

`ScatterPlan::new` now reads each source index once, storing lossless `u32`
indices in a reusable chunk-local buffer for placement. The validated row bound
is 2^24; the cache has only 4,096 entries at the measured sizes. Three alternating
constructor/scatter-candidate pairs used eleven samples per configuration per
block. Retention uses the whole preparation-plus-scatter phase, not the index
loop alone. Block-start loads were 33.60/22.16/18.75, 36.39/23.16/19.15,
35.83/23.49/19.31, 34.81/23.48/19.33, 34.15/23.73/19.47 and
36.46/24.39/19.73; afterwards 33.42/24.13/19.69. The following are medians
of three block medians. Contention changes the unchanged preparation phase too;
these results establish loaded-machine improvement, not a quiet-machine gate.

| log T | Threads | Before ns/cycle | Before ms | After ns/cycle | After ms | Reduction |
|---|---|---|---|---|---|---|
| 20 | 1 | 18.342 | 19.233 | 16.142 | 16.926 | 11.99% |
| 20 | 12 | 5.249 | 5.504 | 3.814 | 3.999 | 27.35% |
| 22 | 1 | 15.169 | 63.623 | 13.982 | 58.643 | 7.83% |
| 22 | 12 | 4.850 | 20.344 | 3.567 | 14.960 | 26.46% |

Retained candidate phase medians, shown as ns/cycle / ms, from those three blocks:

| Phase | 20 / 1 thread | 20 / 12 threads | 22 / 1 thread | 22 / 12 threads |
|---|---|---|---|---|
| adapt | 4.181 / 4.384 | 1.264 / 1.325 | 4.007 / 16.807 | 1.197 / 5.021 |
| construct | 36.381 / 38.148 | 18.052 / 18.929 | 30.028 / 125.945 | 17.832 / 74.795 |
| validate (diagnostic) | 8.246 / 8.646 | 2.011 / 2.109 | 6.654 / 27.910 | 1.901 / 7.975 |
| prepare | 12.979 / 13.609 | 3.020 / 3.167 | 11.370 / 47.691 | 2.737 / 11.478 |
| scatter | 2.892 / 3.032 | 0.741 / 0.777 | 2.658 / 11.149 | 0.775 / 3.251 |
| preparation + scatter | 16.142 / 16.926 | 3.814 / 3.999 | 13.982 / 58.643 | 3.567 / 14.960 |
| production total | 57.304 / 60.088 | 23.165 / 24.290 | 48.366 / 202.863 | 22.783 / 95.558 |

Maximum incremental requested-byte peaks across these candidate blocks:

| Phase | log T 20 | log T 22 |
|---|---|---|
| adapt | 75,497,800 | 301,990,216 |
| construct | 100,663,344 | 377,487,456 |
| validate | 2,097,456 | 2,097,456 |
| prepare | 21,007,392 | 77,704,224 |
| scatter | 5,160,960 | 20,119,552 |

The scatter peak adds up to twelve 16 KiB caches above the original peak. No
trace-sized allocation is added. Standalone validation is excluded from the
production total, which is 95.558 ms at 22/12 in these blocks. That measures
only this path, not whether the whole prover meets the 673 ms target.

## Remaining costs and specification corrections

- `src/plane.rs`, `Rv64iWitness::initial_state`: canonical initial-RAM checks,
  dense RAM zero-fill and population remain ordered. `prepare_with_ram` also
  initializes the final 32T, 40T and 16T shared buffers before filling them.
- `src/plane.rs`, `Rv64iWitness::replay`: register/RAM pre-state checks and XOR
  updates depend on earlier cycles. Parallel row generation leaves this ordered
  walk, and adds one facts walk without adding another owner of trace storage.
- `src/optimized/source.rs`, `SharedSource::prepare`, through kernel
  `ValidatedTrace::prepare`: digit validation and group generation remain one
  parallel walk. They check the source boundary and write the required 18T
  group bytes. Standalone validation measures those checks separately but
  cannot replace the fused production preparation.
- `src/packed/scatter.rs`, `ScatterPlan::new` in the kernels crate: source indices
  are now read once; counts, prefix sums and placement remain required to build
  the cached permutation and descriptors.

The specifications are unchanged. In `specs/rv64i-binary-prover-adapters.md`,
replace the following stale statements:

1. In invariant 5, “The decoded rows are written by the replay the witness
   already makes”: “Decoded rows are written with committed-row generation
   from facts, in parallel on multi-thread pools; committed-row constructors
   write them during replay. No adapter decodes committed rows.”
2. In Decoded rows, “it is the one walk of the cycles a witness constructor
   makes”: “Single-thread facts construction and committed-row construction
   fuse packing and replay. Multi-thread facts construction generates and packs
   in parallel, then checks state in an ordered replay, deferring generation
   faults to preserve the first-fault contract.”
3. In Performance, “the constructor is serial as a whole and this spec does
   not change that”: “State replay and initialization remain serial; facts-row
   generation and packing are parallel. Measure the whole constructor instead
   of treating its decoded-row write as the constructor's cost.”
4. In Source and shared state, “`ScatterPlan::new` keeps its two reads of the
   bytecode index per cycle”: “`ScatterPlan::new` reads each index once and
   retains it in bounded chunk-local scratch for placement.”
5. The existing adapters-fixture statement that facts are dropped before any
   measurement needs the qualification “in `adapters`; `witness_pipeline`
   retains architectural trace rows, and facts through construction, so it
   measures adaptation and construction themselves.”
6. The 918 ms budget paragraph conflicts with the task's 673 ms target:
   “The end-to-end target is 673 ms at 2^22 cycles on 12 threads; allocate costs
   using whole-phase measurements and do not double-count constructor work.”
The corresponding kernels-spec budget paragraph also still states 918 ms.

## Final runner integration probe

The timing loop was moved into the existing support runner without changing
phase boundaries or ordering. A five-sample inventory run of that final entry
on both sizes and pools started at load 26.08/26.72/21.87 and ended at
25.19/26.52/21.83. Its medians, ns/cycle / ms, were:

| Phase | 20 / 1 thread | 20 / 12 threads | 22 / 1 thread | 22 / 12 threads |
|---|---|---|---|---|
| adapt | 4.217 / 4.422 | 2.090 / 2.191 | 6.764 / 28.370 | 1.332 / 5.585 |
| construct | 37.629 / 39.457 | 25.227 / 26.452 | 37.348 / 156.648 | 22.728 / 95.330 |
| validate (diagnostic) | 8.823 / 9.252 | 3.648 / 3.825 | 8.399 / 35.229 | 2.227 / 9.341 |
| prepare | 13.350 / 13.998 | 5.267 / 5.523 | 12.605 / 52.869 | 3.472 / 14.562 |
| scatter | 3.597 / 3.772 | 1.195 / 1.253 | 3.377 / 14.164 | 0.996 / 4.176 |
| preparation + scatter | 17.540 / 18.393 | 6.409 / 6.720 | 16.893 / 70.854 | 4.643 / 19.472 |
| production total | 59.003 / 61.870 | 33.177 / 34.789 | 61.005 / 255.872 | 28.909 / 121.251 |

The run recorded 220 large-allocation entries without overflow. Scatter's
maximum peak at log T 20 was 5,177,344 bytes (twelve simultaneous 16 KiB
caches); log T 22's scatter maximum remained 20,119,552 bytes. The cache size
per Rayon job is bounded by chunk length, not T.

An `adapters/source` inventory smoke on all four configurations followed,
ending at load 24.14/26.28/21.77. All four inventories had zero unmatched
allocations and zero overflow. Its existing source thresholds were exceeded
on this loaded machine; the smoke checks compatibility and allocation laws,
and does not establish a quiet-machine performance gate.

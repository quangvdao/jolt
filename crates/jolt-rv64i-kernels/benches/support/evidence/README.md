# Runner-port evidence

The seven benches now share source preparation, sampling, per-phase and per-sample-total statistics, allocation snapshots, warmed pool lifetime and comparison decisions. No library source changed.

## Comparison method

[The full table](comparison.md) contains every emitted timing field and all six per-invocation medians, including profiles, variants and phase records. [Raw outputs](raw/) preserve allocation maxima, model values and thresholds. [The session log](session.txt) records invocation start timestamps and load averages.

Base native binaries were saved from `a398d5f0a` before the ports. The base chunk reporter printed minima for its split and gather records; only its post-measurement reporter was instrumented to emit medians instead. No timed or allocation work changed in that baseline binary. The other six base binaries were unchanged. Keeping the seven small executables avoided cloning a target directory on the nearly full disk; temporary executables and build logs were removed after the comparisons.

After native builds, one uninterrupted session executed each bench in order before, after, before, after, before, after. Every invocation used `--log-t 20 --threads 1 --samples 3`. The old auxiliary router/chunk comparison drivers nevertheless forced five samples; the new shared cases honor three. This sampling correction is disclosed rather than presenting those auxiliary runs as equal sample counts. The machine was loaded throughout; this is not quiet-machine calibration.

Summary columns are the median of the three reported per-invocation medians, not the median of nine unavailable raw samples. Brackets cover all emitted min/max bounds, or the range of emitted medians when the old reporter omitted bounds. Fold's old repeated subphase lines are reduced to a median within each invocation. New records and retired fields use `—`.

Recreate the table from the archived outputs, from the worktree root:

```sh
python3 crates/jolt-rv64i-kernels/benches/support/evidence/summarize.py
```

The parser flags a change when the median difference exceeds the sum of the two observed spreads. All measured totals remain within those combined spreads. The two flagged fields are accounted for by interval corrections: tail's raw finish now measures actual batch finishes, and outer monomial-rounds4 extraction no longer inserts benchmark telemetry. Loaded router and fold observations sometimes exceed their unchanged requirements; passing the smoke commands does not certify quiet-machine performance thresholds.

| Primary record (ns/cycle) | Before (loaded) | After (loaded) |
|---|---:|---:|
| tail/local | 136.858145 | 157.940228 |
| routers/default/all_rows | 441.943289 | 452.361900 |
| reduction/local | 23.824653 | 24.187525 |
| chunk_product/uniform_digits | 42.197785 | 42.973994 |
| column_pass/local | 19.730250 | 19.855618 |
| outer_f2/local | 118.058206 | 121.510585 |
| fold/default/all_rows, construction + pass | 225.553592 | 228.491625 |

Fold's fixed requirement applies to its pass phase, not the construction-plus-pass total above. Exact record names and ranges are in the full table.

## Interval corrections and preserved boundaries

- Tail, routers, chunk and default reduction now prepare validation, claims and reusable plans once per source outside all sample clocks/counters. Fixture-cache locking/replacement and disposal no longer contaminate construction or allocation intervals. Tail still constructs its timed g pass, weights and cores afresh each sample.
- Tail now enters the actual batch: its raw rounds/finish fields use disjoint real batch durations. The one-round entry adapter, dummy polynomial, unrelated challenge evaluation and empty finish were removed. Actual batch member work is unchanged.
- Column and outer extraction no longer allocate/insert benchmark telemetry through local locks/maps/vectors. The runner collects it after counters stop.
- Pass-only column and fold report zero rounds and finish rather than timing empty adapters. No pass work was removed.
- Router's local total now uses the canonical full callback boundaries rather than summing local partial callback timers. The runner computes every total from each sample's nominated sum.
- Auxiliary router, chunk and outer comparisons now use the same nominated phase sums and interleaved session as their main cases; their separate timing loops/warmups and forced sample counts were removed. Kernel work is preserved.
- The chunk standalone diagnostic retains its five-column `lo_hi_all`, checksum reductions and first four binds, including fourth-bind materialisation. Tail adds new, separate intervals for both five-column groups on its own digit distribution. Compact-source and lazy-family construction remain outside that diagnostic clock; lazy-family construction is included in its new allocation record. These records are not additive shares of fused rounds.
- Shared reduction's freshly owned word table remains per-sample, outside primary phase clocks but inside allocation accounting. Fold's complete constructor and ordinary outer tau generation retain their original timing boundaries.

## Verification and surface

Formatting, both all-target Clippy modes, style invariants, the seven-file Instant guard and whitespace checks passed. `cargo nextest` passed 139 tests with zero skipped; [its log](nextest.txt) is retained. The required native one-sample smoke loop passed all seven benches. An independent reviewer checked allocation lifetimes, interval boundaries, threshold ownership and all 803 table rows.

[Public support inventory](surface.md) lists each public item. Cumulative deleted lines relative to `a398d5f0a`: tail 202, routers 271, chunk_product 527, reduction 312, column_pass 93, outer_f2 400, fold 70; reduction's canonical-map test cleanup deletes 27 lines. These are deletion counts, not net file shrinkage.

## Requests

No library change is needed. Outer's existing banner contains legacy threshold fields 268/28 while the Performance specification requires 254/26. Those old fields were retained, explicit `spec_threshold` fields were added, and runner comparisons use 254/26. Reconcile or remove the legacy banner fields in a later authorized specification cleanup. No specification-named record was renamed.

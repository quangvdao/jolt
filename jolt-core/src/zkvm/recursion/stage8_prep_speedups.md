# Stage 8 Prep Speedups (Recursion Verifier)

This document tracks investigation + implementation progress for speeding up **Stage 8 prep**
(`verify_recursion_stage8_prep_total`) in the recursion verifier.

## Goal

Reduce the cycle cost of Stage 8 prep in the zkVM verifier path (cycle-tracked), without changing
soundness or transcript semantics.

## Where “Stage 8 prep” lives

- Cycle scope: `verify_recursion_stage8_prep_total`
- Entrypoint: `jolt-core/src/zkvm/recursion/api.rs` (`verify_recursion()`)
  - Stage 8 prep region: around `start_cycle_tracking("verify_recursion_stage8_prep_total")`

## Baseline (from cycle tracking)

Stage 8 prep shows up as a significant chunk of total recursion verification cycles.
Primary suspected hotspots (pre-change):

- `RecursionExt::build_symbolic_ast` (Dory symbolic verification, transcript hashing, AST build)
- `derive_plan_with_hints` (multiple AST passes + sorts + wiring-plan derivation)

## Key invariants / safety constraints

- **Transcript ordering**:
  - AST build must use the *pre-opening-proof* transcript state.
  - Main transcript must be advanced to the post-Stage8 state via `replay_opening_proof_transcript`.
- **Deterministic ordering**:
  - Constraint instance ordering must match the prover/extractor ordering.
  - Wiring instance indices must align with `constraint_types` construction.
- **Soundness**:
  - No caching of Fiat–Shamir challenges across runs.
  - Do not reorder transcript mutations.

Relevant regression tests:

- Transcript fork invariant:
  - `jolt-core/src/zkvm/recursion/tests/pipeline_invariants_test.rs`
- Wiring/boundary tampering rejection:
  - `jolt-core/src/zkvm/recursion/tests/wiring_integration_test.rs`

## Plan (incremental, with correctness checks)

### Phase A: Remove defensive sorts (keep debug asserts)

Hypothesis: OpIds are already monotone in encounter order, so per-type sorting is redundant.

- Remove sorts of per-type OpId vectors in:
  - `derive_plan_with_hints` (Dory instance plan)
  - wiring plan derivation (op-id collection)
- Add debug-only monotonicity assertions per op-type.

### Phase B: Make wiring-plan derivation O(n) with O(1) lookups

Replace:
- sorting + `BTreeMap<OpId, usize>` indexing

With:
- single pass over `ast.nodes` in creation/topological order
- ValueId-indexed `Vec<Option<usize>>` mapping each node output to its per-type instance index
- immediate wiring edge emission with producer lookup via those arrays

This should eliminate:
- `collect_op_ids()` sorts
- `index_map()` BTreeMap construction
- repeated log-time map lookups inside hot loops

### Phase C: Validate + measure

- Run targeted tests
- (Optional) add finer-grained cycle markers to confirm wins

## Progress log

### 2026-02-02

- **Scoping confirmed**:
  - Stage 8 prep scope is `verify_recursion_stage8_prep_total` in
    `jolt-core/src/zkvm/recursion/api.rs` (approx. lines 275–390).
- **Dory ordering basis confirmed**:
  - `OpId` structure: `(round, op_type, index)` in `dory-pcs` (`src/recursion/witness.rs`).
  - `OpIdBuilder` is monotone per op type, resets counters on round transitions
    (`dory-pcs/src/recursion/collector.rs`).
  - `AstGraph.nodes` is creation/topological order with `ValueId(i)` == node index
    (`dory-pcs/src/recursion/ast/core.rs`).

- **Implemented: remove defensive sorts + add debug monotonicity checks**
  - `jolt-core/src/poly/commitment/dory/instance_plan.rs`
    - Removed per-type sorts; added debug-only checks that `OpId` is monotone per family.
  - `jolt-core/src/zkvm/recursion/wiring_plan.rs`
    - Removed op-id list sorts; kept debug-only checks.

- **Implemented: remove BTreeMaps from wiring plan**
  - `jolt-core/src/zkvm/recursion/wiring_plan.rs`
    - Replaced `BTreeMap<OpId, usize>` lookups with `Vec<Option<usize>>` indexed by `ValueId`.
    - Preserved the “GT input uses GTExpBase if present anywhere” behavior via a precomputed
      `gt_exp_base_instance_by_value` table.

- **Targeted tests passed (with `--features recursion`)**
  - Transcript fork invariant:
    - `zkvm::recursion::tests::pipeline_invariants_test::test_stage8_transcript_fork_matches_prover_after_witness_gen`
  - Wiring tamper rejection:
    - `zkvm::recursion::tests::wiring_integration_test::wiring_rejects_tampered_pairing_boundary_rhs`
    - `zkvm::recursion::tests::wiring_integration_test::wiring_rejects_tampered_pairing_boundary_points`
    - `zkvm::recursion::tests::wiring_integration_test::wiring_rejects_tampered_gt_exp_base_input`

- **Cycle tracking run (fibonacci, scale 24, committed, address-major, recursion, embedded, disk)**
  - Command (per `spec.md`):
    - `RUST_LOG=info cargo run --release -p recursion -- trace --example fibonacci --workdir output/fib_committed_addrmajor_recursion_scale24_cycletrack --embed --committed --layout address-major --cycle-tracking --disk`
  - Totals (virtual cycles):
    - (baseline with Stage-8 sub-spans; before wiring/plan speedups)
      - `verify_recursion_total`: **242.948M**
      - `verify_recursion_stage8_prep_total`: **15.922M**
      - `verify_recursion_stage8_prep_plan_total`: **5.271M**

    - (after wiring/plan speedups: no OpId sorts + array-index wiring plan)
      - `verify_recursion_total`: **242.183M**
      - `verify_recursion_base_stages_1_to_7_total`: **22.672M**
      - `verify_recursion_stage8_prep_total`: **15.132M**
      - `jolt_recursion_stage1`: **4.159M**
      - `jolt_recursion_stage2`: **24.165M**
      - `jolt_recursion_stage3`: **0.449M**
      - `jolt_recursion_pcs_opening`: **149.202M**
      - `jolt_external_pairing_check`: **17.199M**
      - residual “other overhead” (by subtraction): **~9.207M**

    - (after also skipping Stage8 transcript replay in release by reusing symbolic transcript)
      - `verify_recursion_total`: **239.170M**
      - `verify_recursion_base_stages_1_to_7_total`: **22.666M**
      - `verify_recursion_stage8_prep_total`: **12.027M**
      - `verify_recursion_snark_verify_total`: **184.193M**
      - `jolt_external_pairing_check`: **17.199M**

  - Stage 8 prep breakdown (new sub-spans, virtual cycles):
    - after wiring/plan speedups:
      - `verify_recursion_stage8_prep_commitments_map_total`: **0.156M**
      - `verify_recursion_stage8_prep_rlc_total`: **0.114M**
      - `verify_recursion_stage8_prep_symbolic_ast_total`: **6.587M**
      - `verify_recursion_stage8_prep_replay_total`: **3.104M**
      - `verify_recursion_stage8_prep_plan_total`: **4.479M**

    - after skipping replay (release):
      - `verify_recursion_stage8_prep_commitments_map_total`: **0.156M**
      - `verify_recursion_stage8_prep_rlc_total`: **0.114M**
      - `verify_recursion_stage8_prep_symbolic_ast_total`: **6.586M**
      - `verify_recursion_stage8_prep_plan_total`: **4.479M**
      - (note: `verify_recursion_stage8_prep_replay_total` is not executed in release)


# Spec: `jolt-openings` PCS API Cutover

| Field       | Value        |
|-------------|--------------|
| Author(s)   | @quangvdao   |
| Created     | 2026-05-12   |
| Status      | proposed     |
| PR          | [#1521](https://github.com/a16z/jolt/pull/1521) |

## Summary

Jolt currently has two polynomial commitment scheme APIs in the workspace.
The production zkVM path still uses the monolithic in-core trait in `jolt-core/src/poly/commitment/commitment_scheme.rs`, while `crates/jolt-openings` and `crates/jolt-dory` already sketch an extracted crate boundary that is not wired into `jolt-core`.

This spec proposes a main-target API refactor that ports PR [#1467](https://github.com/a16z/jolt/pull/1467) directly onto current `main`.
The PR makes `jolt-openings` the canonical backend-neutral opening API, splits verifier and prover PCS surfaces, makes fused batched openings the primary API, moves Dory onto the extracted trait family, and cuts `jolt-core` over to `PCS::BatchProof` without introducing Akita or changing the Jolt protocol.

The implementation should be a mechanical transplant of #1467's hard-earned design, not a greenfield rewrite.
Adaptation is only for current `main` drift, especially current Dory hardening, current Dory ZK evaluation commitments, Stage 8's streaming RLC optimization, and BlindFold wiring.

## Intent

### Goal

Make `crates/jolt-openings` the canonical polynomial-opening abstraction used by `jolt-core`, with a verifier-first trait hierarchy and fused `prove_batch` / `verify_batch` API that can support Dory today and future non-Dory schemes such as Akita without forcing them into Dory-shaped single-opening hooks.

### Source of Truth

PR [#1467](https://github.com/a16z/jolt/pull/1467), branch `quang/pcs-prover-verifier-split`, is the source of truth for the abstract PCS API.
Current `main` is the source of truth for concrete Dory correctness, Dory proof hardening, current proof serialization context, Stage 8 behavior, and BlindFold behavior.

Port directly from #1467:

1. `crates/jolt-openings/src/schemes.rs`: verifier/prover split and extension traits.
2. `crates/jolt-openings/src/homomorphic.rs`: homomorphic fused batch helpers.
3. `crates/jolt-openings/src/claims.rs`: `ProverClaim` and verifier-only `OpeningClaim`.
4. `crates/jolt-openings/src/lib.rs`: public exports and crate-level API documentation.
5. `crates/jolt-openings/src/mock.rs`: mock PCS implementation under the split trait family.
6. `crates/jolt-dory/src/scheme.rs`: split trait implementation structure for Dory.
7. Relevant tests and benches for the new `jolt-openings` API.

Preserve from current `main`:

1. Current `crates/jolt-dory` wrapper types, transcript bridge, streaming support, and bounded proof deserialization.
2. Current Dory ZK commitment fixes.
3. Current `jolt-core` Stage 8 claim construction, advice handling, and Dory layout behavior.
4. Current BlindFold opening proof data flow and ZK verification behavior.
5. Current `muldiv` behavior in standard and ZK modes.

### Invariants

1. `jolt-openings` remains backend-neutral and must not depend on `jolt-core`, `jolt-dory`, `dory`, arkworks, `common`, `tracer`, `jolt-sdk`, Akita, or Hachi.
2. `jolt-openings` depends only on reusable leaf crates plus generic dependencies: `jolt-field`, `jolt-poly`, `jolt-transcript`, `jolt-crypto`, `serde`, `thiserror`, and `tracing`.
3. The base verifier trait does not expose prover-only associated types such as `ProverSetup`, `Polynomial`, `OpeningHint`, or `SetupParams`.
4. The base PCS traits expose fused batched openings through `prove_batch` and `verify_batch`.
5. Single-claim `open` and `verify` are not on the base PCS trait.
   They live only on homomorphic extension traits as primitives for `homomorphic_prove_batch` and `homomorphic_verify_batch`.
6. Homomorphic batch proving and verification have byte-identical Fiat-Shamir behavior between prover and verifier.
   They absorb the same claim count, the same evaluations, and draw the same per-point RLC challenges in the same order.
7. `prove_batch` returns both `PCS::BatchProof` and the per-group joint evaluations needed by later transcript binding.
   Batch verification does not silently perform post-opening transcript binding.
8. `OpeningClaim` is generic over `PCS: CommitmentSchemeVerifier`, not over a raw commitment type.
   Verifier-only code can name opening claims without importing prover-only PCS types.
9. `jolt-core` keeps protocol-specific opening bookkeeping.
   `OpeningId`, `PolynomialId`, `SumcheckId`, `OpeningPoint`, `ProverOpeningAccumulator`, and `VerifierOpeningAccumulator` do not move into `jolt-openings`.
10. Dory layout, Dory matrix embedding policy, Stage 8 claim ordering, and BlindFold constraints do not move into `jolt-openings`.
11. Dory's current transparent and ZK proofs remain verifier-compatible with current `main`.
12. `JoltProof` stores the opening proof as `PCS::BatchProof`, not `PCS::Proof`.
13. Standard and ZK `muldiv` end-to-end proofs continue to pass.
14. The implementation introduces no Akita dependency and no compatibility shim for old PCS trait names.
15. `cargo tree -d` must not show duplicate resolved versions of `jolt-field`, `jolt-transcript`, `jolt-crypto`, or `jolt-openings`.

No new `jolt-eval` invariant is required for this spec.
The relevant invariants are proof acceptance, transcript parity, and prover/verifier consistency, which are covered by focused crate tests and `jolt-core` end-to-end tests.

### Non-Goals

1. Integrating Akita or Hachi.
2. Adding lattice parameters, lattice proof types, or Akita-specific Stage 8 branches.
3. Redesigning the whole proof format beyond replacing the opening proof field with `PCS::BatchProof`.
4. Moving Stage 8 sumcheck wiring, opening IDs, accumulator types, or BlindFold constraints into `jolt-openings`.
5. Replacing the current Jolt prover pipeline with the verifier/compiler runtime from `refactor/crates`.
6. Porting unrelated #1467 hygiene changes such as debug-test clippy suppressions when they are not needed on current `main`.
7. Adding a new batch-ZK opening abstraction beyond the #1467 ZK split.
8. Preserving the old internal `CommitmentScheme` trait as a compatibility layer.
9. Changing guest execution, bytecode expansion, memory checking, instruction lookups, or sumcheck protocol semantics.
10. Changing Dory's transcript labels or proof verification behavior except where required to preserve current `main` behavior under the new trait API.

## Evaluation

### Acceptance Criteria

- [ ] `crates/jolt-openings/src/schemes.rs` defines `CommitmentSchemeVerifier`, `CommitmentScheme`, `AdditivelyHomomorphicVerifier`, `AdditivelyHomomorphic`, `ZkOpeningSchemeVerifier`, `ZkOpeningScheme`, and `StreamingCommitment` with the #1467 role split.
- [ ] `CommitmentSchemeVerifier` contains `Field`, `VerifierSetup`, `Proof`, `BatchProof`, `VerifierSetupParams`, `verifier_setup`, `verify_batch`, and `bind_opening_inputs`.
- [ ] `CommitmentScheme` extends `CommitmentSchemeVerifier` and contains `ProverSetup`, `Polynomial`, `OpeningHint`, `SetupParams`, `setup`, `project_verifier_setup`, `commit`, and `prove_batch`.
- [ ] `open` and `verify` are removed from the base PCS trait and exist only on homomorphic extension traits.
- [ ] `crates/jolt-openings/src/homomorphic.rs` contains #1467's `homomorphic_prove_batch`, `homomorphic_verify_batch`, `rlc_combine`, and `rlc_combine_scalars`.
- [ ] `homomorphic_prove_batch` and `homomorphic_verify_batch` group claims by opening point and use the same transcript schedule.
- [ ] `crates/jolt-openings/src/claims.rs` exposes `ProverClaim<F>` and `OpeningClaim<F, PCS: CommitmentSchemeVerifier<Field = F>>`.
- [ ] The old standalone `reduce_prover` / `reduce_verifier` production API is removed or demoted so production callers use `prove_batch` / `verify_batch`.
- [ ] `crates/jolt-openings/src/mock.rs` implements the split traits and has tests covering single-claim, multi-claim, shared-point, distinct-point, and tampered-evaluation cases.
- [ ] `crates/jolt-dory` implements the split trait family while preserving current `main` wrapper types, bounded deserialization, transcript bridge, streaming support, and ZK behavior.
- [ ] `DoryScheme::BatchProof = Vec<DoryProof>` for the homomorphic Dory implementation.
- [ ] `DoryScheme::prove_batch` delegates to `homomorphic_prove_batch`.
- [ ] `DoryScheme::verify_batch` delegates to `homomorphic_verify_batch`.
- [ ] `jolt-core` depends on `jolt-openings` and `jolt-dory` instead of using the internal PCS trait as the canonical interface.
- [ ] `jolt-core` call sites use `PCS::Output` for commitments, matching `jolt_crypto::Commitment`, rather than `PCS::Commitment`.
- [ ] `JoltProof` stores `joint_opening_proof: PCS::BatchProof` or an equivalently named `PCS::BatchProof` field.
- [ ] Stage 8 prover calls `PCS::prove_batch`.
- [ ] Stage 8 verifier calls `PCS::verify_batch`.
- [ ] Stage 8 keeps the same dense increment scaling, RA polynomial ordering, advice Lagrange scaling, `opening_ids`, `constraint_coeffs`, and `joint_claim` semantics as current `main`.
- [ ] ZK mode still extracts and binds the Dory evaluation commitment needed by BlindFold.
- [ ] `cargo nextest run -p jolt-openings --features test-utils --cargo-quiet` passes.
- [ ] `cargo nextest run -p jolt-dory --cargo-quiet` passes.
- [ ] `cargo nextest run -p jolt-core muldiv --cargo-quiet --features host` passes.
- [ ] `cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk` passes.
- [ ] `cargo clippy --all --features host -q --all-targets -- -D warnings` passes.
- [ ] `cargo clippy --all --features host,zk -q --all-targets -- -D warnings` passes.
- [ ] `cargo fmt -q` produces no diff.

### Testing Strategy

Focused `jolt-openings` tests should validate the API in isolation:

1. A mock PCS can commit, prove, and verify one opening.
2. Multiple claims at the same point reduce to one proof.
3. Multiple claims at distinct points produce one proof per point group.
4. Mixed shared and distinct points verify.
5. A tampered evaluation is rejected.
6. `rlc_combine` and `rlc_combine_scalars` agree with direct polynomial evaluation.
7. Prover and verifier transcripts remain in sync for the batch helper path.

Focused `jolt-dory` tests should validate the real PCS implementation:

1. Commit, open, and verify round trips still pass.
2. Homomorphic combination verifies against direct polynomial combination.
3. `prove_batch` / `verify_batch` pass for one claim and multiple claims.
4. Streaming commitment still matches direct commitment.
5. Dory proof deserialization still rejects oversized or malformed proof round counts.
6. ZK opening behavior still produces and verifies the expected hiding commitment.

`jolt-core` tests should validate the real zkVM integration:

1. `muldiv` passes with `--features host`.
2. `muldiv` passes with `--features host,zk`.
3. Advice examples continue to pass in standard mode because they exercise non-ZK opening claims.
4. Any existing Stage 8 or Dory layout tests continue to pass.

Feature and dependency checks:

1. `cargo tree -d -p jolt-core` should not show duplicate Jolt leaf crates.
2. `cargo tree -p jolt-openings` should show no dependency on `jolt-core`, `jolt-dory`, `dory`, `tracer`, or Akita.
3. `cargo tree -p jolt-dory` should show `jolt-dory` depending on `jolt-openings`, not the other way around.

### Performance

The refactor should not regress prover performance.
The Dory Stage 8 path is performance-sensitive because current `main` builds a streaming RLC polynomial directly from the trace rather than regenerating witness polynomials.

Performance requirements:

1. Preserve the current Stage 8 streaming RLC optimization unless an equivalent or faster path is implemented.
2. Avoid materializing all committed polynomials in Stage 8 merely to fit the new API.
3. Keep Dory row-commitment hint combination in the hot path when it avoids recomputing row commitments.
4. Do not introduce dynamic dispatch in the prover hot path.
5. Keep parallelism in Dory commitment and hint combination.
6. Preserve current `#[inline]` and low-level optimized group operation behavior where code moves between crates.

Concrete checks:

1. `cargo nextest run -p jolt-core muldiv --cargo-quiet --features host` should not show obvious runtime blowups.
2. Existing Dory benches should compile.
3. If a reviewer requests performance data, run the existing Dory and opening reduction benches before and after the cutover.

## Design

### Architecture

After the refactor, the dependency direction is:

```text
jolt-field
jolt-poly
jolt-transcript
jolt-crypto
    |
    v
jolt-openings
    |
    +--> jolt-dory
    |
    +--> future Akita adapter
    |
    v
jolt-core
```

`jolt-openings` owns the abstract PCS API and generic homomorphic batch helper.
`jolt-dory` owns Dory-specific commitments, proofs, setup types, transcript adaptation, row commitments, streaming commitments, and Dory proof hardening.
`jolt-core` owns the Jolt protocol's opening IDs, accumulators, Stage 8 claim assembly, Dory layout selection, proof object, and BlindFold wiring.

The trait hierarchy is:

```text
CommitmentSchemeVerifier
  - Field
  - VerifierSetup
  - Proof
  - BatchProof
  - VerifierSetupParams
  - verifier_setup
  - verify_batch
  - bind_opening_inputs

CommitmentScheme: CommitmentSchemeVerifier
  - ProverSetup
  - Polynomial
  - OpeningHint
  - SetupParams
  - setup
  - project_verifier_setup
  - commit
  - prove_batch

AdditivelyHomomorphicVerifier: CommitmentSchemeVerifier
  - combine
  - verify

AdditivelyHomomorphic: AdditivelyHomomorphicVerifier + CommitmentScheme
  - combine_hints
  - open

ZkOpeningSchemeVerifier: CommitmentSchemeVerifier
  - HidingCommitment
  - verify_zk

ZkOpeningScheme: ZkOpeningSchemeVerifier + CommitmentScheme
  - Blind
  - open_zk

StreamingCommitment: CommitmentScheme
  - begin
  - feed
  - finish
```

This hierarchy is a role split, not a lifecycle split.
Verifier-only code can bound on `CommitmentSchemeVerifier` without naming prover-only data.
Prover code gets the verifier surface because `CommitmentScheme` extends `CommitmentSchemeVerifier`.
Homomorphic single-opening operations are extension primitives rather than required PCS basics.

### Homomorphic Batched Opening Protocol

The default homomorphic batch helper implements the standard group-by-point RLC protocol.

Prover:

1. Receive `Vec<ProverClaim<F>>` and matching `Vec<PCS::OpeningHint>`.
2. Append the claim count under `rlc_claims`.
3. Append all claimed evaluations.
4. Group claims by opening point.
5. For each group, draw `rho`.
6. RLC-combine polynomial evaluation tables with powers of `rho`.
7. RLC-combine scalar evaluations with powers of `rho`.
8. RLC-combine opening hints with powers of `rho`.
9. Call `PCS::open` on the combined polynomial and point.
10. Return `PCS::BatchProof` and per-group joint evaluations.

Verifier:

1. Receive `Vec<OpeningClaim<F, PCS>>` and `PCS::BatchProof`.
2. Append the same claim count under `rlc_claims`.
3. Append all claimed evaluations in the same order.
4. Group claims by opening point.
5. For each group, draw the same `rho`.
6. RLC-combine commitments with powers of `rho`.
7. RLC-combine scalar evaluations with powers of `rho`.
8. Call `PCS::verify` on the combined commitment and point.

The helper is generic over homomorphic schemes.
Non-homomorphic schemes are not required to implement `open`, `verify`, `combine`, or `combine_hints`; they can implement fused `prove_batch` and `verify_batch` directly.

### Dory Implementation

`DoryScheme` should follow #1467's impl split while preserving current `main` correctness work.

`CommitmentSchemeVerifier for DoryScheme`:

1. `Field = Fr`.
2. `VerifierSetup = DoryVerifierSetup`.
3. `Proof = DoryProof`.
4. `BatchProof = Vec<DoryProof>`.
5. `VerifierSetupParams = usize`.
6. `verifier_setup(max_num_vars)` derives the deterministic verifier setup.
7. `verify_batch` delegates to `homomorphic_verify_batch`.
8. `bind_opening_inputs` preserves Dory's transcript binding semantics.

`CommitmentScheme for DoryScheme`:

1. `ProverSetup = DoryProverSetup`.
2. `Polynomial = jolt_poly::Polynomial<Fr>`.
3. `OpeningHint = DoryHint`.
4. `SetupParams = usize`.
5. `setup(max_num_vars)` returns prover and verifier setup.
6. `project_verifier_setup(&prover_setup)` projects prover setup down to verifier setup.
7. `commit` commits through the current Dory row commitment path.
8. `prove_batch` delegates to `homomorphic_prove_batch`.

`AdditivelyHomomorphicVerifier for DoryScheme`:

1. `combine` linearly combines Dory commitments.
2. `verify` verifies one Dory opening proof.

`AdditivelyHomomorphic for DoryScheme`:

1. `combine_hints` linearly combines Dory row-commitment hints.
2. `open` proves one Dory opening.

`ZkOpeningSchemeVerifier` and `ZkOpeningScheme for DoryScheme`:

1. Preserve current Dory `y_com` behavior.
2. Preserve current `y_blinding` behavior.
3. Preserve BlindFold compatibility.

### `jolt-core` Integration

`jolt-core` should be cut over without changing protocol semantics.

The high-level migration is:

1. Add `jolt-openings` and `jolt-dory` as dependencies.
2. Replace imports of the internal PCS trait with `jolt_openings` traits.
3. Replace `PCS::Commitment` associated type usage with `PCS::Output`.
4. Replace `PCS::Proof` proof storage with `PCS::BatchProof`.
5. Replace Stage 8's direct `PCS::prove` call with `PCS::prove_batch`.
6. Replace Stage 8's direct `PCS::verify` call with `PCS::verify_batch`.
7. Preserve Stage 8's claim construction and ZK constraint coefficient logic.

Stage 8 is the main adaptation point.
Current `main` builds a single Dory-shaped joint polynomial using `DoryOpeningState::build_streaming_rlc`.
That optimization should remain unless the new batch API can express the same work without extra materialization.

The acceptable first cutover is:

1. Keep current Stage 8's streaming RLC construction.
2. Wrap the resulting joint polynomial, unified opening point, and joint claim as the input to `PCS::prove_batch`.
3. Treat Dory's batch proof as a one-group `Vec<DoryProof>`.
4. On the verifier, build the corresponding one-group `OpeningClaim` and call `PCS::verify_batch`.

This preserves current Dory behavior while changing the public PCS boundary to #1467's fused batch API.
Future Akita work can implement a different `prove_batch` / `verify_batch` body without changing `jolt-core`'s trait definitions.

### Proof Serialization

Current `JoltProof` contains:

```rust
pub joint_opening_proof: PCS::Proof
```

The refactor changes it to:

```rust
pub joint_opening_proof: PCS::BatchProof
```

Renaming the field is optional.
The behaviorally important change is that the proof object is scheme-defined batch proof storage.

`Claims<F>` for non-ZK mode should remain unless a strictly mechanical replacement falls out of the trait cutover.
`dory_layout` should remain in this PR unless a small PCS config type already exists and can replace it without broad proof-format redesign.

### ZK and BlindFold

Current `main` ZK mode is not just a PCS verify call.
It also binds a Dory evaluation commitment into the transcript and passes opening proof data into BlindFold.

The cutover must preserve:

1. `y_com` extraction or equivalent `HidingCommitment` extraction.
2. `y_blinding` or equivalent `Blind` extraction.
3. BlindFold `OpeningProofData`.
4. ZK opening input binding.
5. Pedersen generator derivation used by BlindFold.

If the #1467 ZK split is not sufficient to express all BlindFold-specific Dory needs, add a narrow extension trait rather than widening the base `CommitmentSchemeVerifier`.
The extension trait should describe the capability directly, such as deriving Pedersen generators or exposing a hiding commitment, and should not mention BlindFold from inside `jolt-openings`.

### Alternatives Considered

1. **Greenfield `jolt-openings` implementation.**
   Rejected because #1467 already encodes design decisions learned from prior verifier and PCS refactors.
   Reimplementing from memory risks missing important details such as `VerifierSetupParams`, `project_verifier_setup`, `OpeningClaim`'s verifier-only bound, and separation of batch verification from opening-input binding.

2. **Keep current `jolt-openings` reduce API and only wire it into `jolt-core`.**
   Rejected because it keeps batching as an external orchestration step and still forces future non-homomorphic schemes into the wrong abstraction.
   #1467 correctly makes fused batching the core API.

3. **Keep `open` and `verify` on the base PCS trait.**
   Rejected because Hachi/Akita-style schemes may have native fused batch openings without a meaningful single-opening primitive.
   Homomorphic Dory can still expose single-opening primitives through extension traits.

4. **Move Jolt opening accumulators into `jolt-openings`.**
   Rejected because accumulators are Jolt protocol bookkeeping, not PCS abstraction.
   They depend on `CommittedPolynomial`, `VirtualPolynomial`, `SumcheckId`, advice kinds, and Stage 8 ordering.

5. **Bundle Akita into this PR.**
   Rejected because this PR should make the reusable PCS boundary reviewable on its own.
   Akita should build on the new boundary after it lands.

## Documentation

No Jolt book changes are required for the spec-only PR.

The implementation PR should update crate-level docs in:

1. `crates/jolt-openings/src/lib.rs`
2. `crates/jolt-openings/src/schemes.rs`
3. `crates/jolt-openings/src/homomorphic.rs`
4. `crates/jolt-dory/src/lib.rs`

The implementation PR should not add user-facing Jolt book documentation unless reviewers request it.
This is an internal API boundary refactor with no intended SDK or guest-facing behavior change.

## Execution

Implementation should proceed mechanically:

1. Create a fresh branch from current `main`.
2. Use `git show layerzero/quang/pcs-prover-verifier-split:<path>` to copy #1467 source files into the current branch.
3. Port `crates/jolt-openings` files first and make `jolt-openings` compile.
4. Port `crates/jolt-dory` trait impl structure next and merge current `main` Dory fixes into that structure.
5. Add `jolt-openings` and `jolt-dory` dependencies to `jolt-core`.
6. Replace internal PCS trait imports with `jolt_openings` imports.
7. Convert `PCS::Commitment` call sites to `PCS::Output`.
8. Convert `JoltProof` opening proof storage to `PCS::BatchProof`.
9. Update Stage 8 prover to produce a batch proof through `PCS::prove_batch`.
10. Update Stage 8 verifier to verify through `PCS::verify_batch`.
11. Reconcile ZK and BlindFold extraction through the split ZK traits or a narrow extension trait.
12. Remove or quarantine the old internal PCS trait after call sites stop depending on it.
13. Run focused crate tests.
14. Run `muldiv` in standard and ZK mode.
15. Run clippy in standard and ZK mode.

Recommended commit structure for the implementation PR:

1. `refactor(openings): port verifier PCS split`
2. `refactor(dory): implement split openings traits`
3. `refactor(core): use batched opening proofs`
4. `test(openings): cover fused batch helpers`

This spec PR should land first so implementation review can reference the intended boundary.

## References

- PR [#1467](https://github.com/a16z/jolt/pull/1467): `refactor(openings): split PCS traits into verifier/prover halves and fuse batched openings`.
- Branch `layerzero/quang/pcs-prover-verifier-split`.
- `crates/jolt-openings/src/schemes.rs` on PR #1467.
- `crates/jolt-openings/src/homomorphic.rs` on PR #1467.
- `crates/jolt-openings/src/claims.rs` on PR #1467.
- `crates/jolt-dory/src/scheme.rs` on PR #1467.
- `crates/jolt-openings` on current `main`.
- `crates/jolt-dory` on current `main`.
- `jolt-core/src/poly/commitment/commitment_scheme.rs`.
- `jolt-core/src/poly/opening_proof.rs`.
- `jolt-core/src/zkvm/prover.rs`.
- `jolt-core/src/zkvm/verifier.rs`.
- `jolt-core/src/zkvm/proof_serialization.rs`.

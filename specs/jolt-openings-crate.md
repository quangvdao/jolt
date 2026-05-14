# Spec: `jolt-openings` PCS API Cutover

| Field       | Value        |
|-------------|--------------|
| Author(s)   | @quangvdao   |
| Created     | 2026-05-12   |
| Status      | active       |
| PR          | [#1521](https://github.com/a16z/jolt/pull/1521) |

## Summary

Jolt previously had two polynomial commitment scheme APIs in the workspace.
The production zkVM path had a monolithic in-core trait in `jolt-core/src/poly/commitment/commitment_scheme.rs`, while `crates/jolt-openings` and `crates/jolt-dory` defined the extracted crate boundary.
This PR's target is the full `jolt-core` PCS type-family migration: `jolt-core` compiles against the canonical `jolt-openings` traits directly, and the old in-core PCS trait family is removed rather than left as a compatibility layer.

This spec describes the active main-target API refactor that ports PR [#1467](https://github.com/a16z/jolt/pull/1467) onto current `main`, with review-driven adjustments recorded here.
The target makes `jolt-openings` the canonical backend-neutral opening API, splits verifier and prover PCS surfaces, exposes source-backed commitment and opening entry points, moves Dory onto the extracted trait family, and cuts `jolt-core` over to `PCS::BatchProof` / `PCS::Output` without introducing Akita or changing the Jolt protocol.
Any bridge traits introduced earlier in the branch were implementation scaffolding only; they are not part of the merge target and are removed by the direct type-family migration.

The branch has now completed the full `jolt-core` PCS type-family cutover and introduced source-backed commitment and opening entry points.
An interim version used explicit shaped commitment extension traits so Dory/Jolt could pass a protocol-selected matrix shape without putting that shape on the base PCS trait.
Those traits have been removed.
The merge-target API moves only backend-neutral traversal information onto `CommitmentSource` / `BatchCommitmentSource`: a source may advertise the natural commitment chunk length it can stream efficiently, while Dory privately interprets that chunk length as its internal matrix split.
The opening-side cutover follows the same boundary: Stage 8 provides raw opening terms and a source batch, while the PCS owns fusion, transcript challenges, proof construction, and the returned output relation.

This distinction is important.
`jolt-openings` should not learn Dory's partition vocabulary, and a source should not have a backend-specific associated `Partition` type.
The source API describes what can be traversed without materialization.
The backend API decides how that traversal maps to its commitment algorithm.
Dory's `sigma`, `nu`, row-commitment aggregation, opening-point coordinate conversion, and proof output choices remain Dory/Jolt implementation details.

The implementation should be a mechanical transplant of #1467's hard-earned design except where this spec explicitly diverges, not a greenfield rewrite.
Adaptation is only for current `main` drift, especially current Dory hardening, current Dory ZK evaluation commitments, Stage 8's streaming RLC optimization, and BlindFold wiring.

## Intent

### Goal

Make `crates/jolt-openings` the canonical polynomial-opening abstraction used by `jolt-core`, with a verifier-first trait hierarchy and batch-opening APIs that Stage 8 can call with raw opening claims plus an opening source batch.
Stage 8 should no longer build a Dory-shaped joint RLC polynomial before entering the PCS.
Instead, the PCS should produce the batch proof and return a backend-neutral description of the output relation it chose.
This can support Dory today and future non-Dory schemes such as Akita while keeping native batch proving on the primary production path.
This includes replacing the old `jolt-core` PCS associated types, proof storage, setup plumbing, Stage 8 accumulators, ZK hooks, mock PCS, HyperKZG integration points, and public prover/verifier generic bounds with the `jolt-openings` type family.

### Source of Truth

PR [#1467](https://github.com/a16z/jolt/pull/1467), branch `quang/pcs-prover-verifier-split`, is the starting point for the abstract PCS API.
This spec is the source of truth where it differs from #1467.
Current `main` is the source of truth for concrete Dory correctness, Dory proof hardening, current proof serialization context, Stage 8 behavior, and BlindFold behavior.

Port or adapt from #1467:

1. `crates/jolt-openings/src/schemes.rs`: verifier/prover split and extension traits.
2. `crates/jolt-openings/src/sources.rs`: backend-neutral commitment source and batch-source traits.
3. `crates/jolt-openings/src/homomorphic.rs`: homomorphic batch helpers.
4. `crates/jolt-openings/src/claims.rs`: `ProverClaim` and verifier-only `OpeningClaim`.
5. `crates/jolt-openings/src/lib.rs`: public exports and crate-level API documentation.
6. `crates/jolt-openings/src/mock.rs`: mock PCS implementation under the split trait family.
7. `crates/jolt-dory/src/scheme.rs`: split trait implementation structure for Dory.
8. Relevant tests and benches for the new `jolt-openings` API.

Preserve from current `main`:

1. Current `crates/jolt-dory` wrapper types, transcript bridge, batch-source streaming behavior, and bounded proof deserialization.
2. Current Dory ZK commitment fixes.
3. Current `jolt-core` Stage 8 claim construction, advice handling, and Dory layout behavior.
4. Current BlindFold opening proof data flow and ZK verification behavior.
5. Current `muldiv` behavior in standard and ZK modes.

### Invariants

1. `jolt-openings` remains backend-neutral and must not depend on `jolt-core`, `jolt-dory`, `dory`, arkworks, `common`, `tracer`, `jolt-sdk`, Akita, or Hachi.
2. `jolt-openings` depends only on reusable leaf crates plus generic dependencies: `jolt-field`, `jolt-poly`, `jolt-transcript`, `jolt-crypto`, `serde`, `thiserror`, and `tracing`.
3. The base verifier trait does not expose prover-only associated types such as `ProverSetup`, `Polynomial`, `OpeningHint`, or `SetupParams`.
4. The base verifier trait does not require verifier setup to be derivable from public parameters.
   Verifier-only code receives `PCS::VerifierSetup`.
   Schemes whose verifier setup can be constructed from public parameters implement the separate `PublicVerifierSetup` extension trait.
5. The base PCS traits expose single-claim openings through `open` and `verify`, ordinary batched openings through `prove_batch` and `verify_batch`, source-backed batched openings through the batch-opening-source API, and batched commitment through `commit_batch`.
   A singleton ordinary batch is a raw single opening wrapped in `PCS::BatchProof`; it does not draw a homomorphic batch challenge.
6. Single-claim `open` and `verify` are semantic PCS operations, not homomorphic-only operations.
   Native batched schemes may implement them with their own singleton proof path, but verifier code should be able to verify one opening through the base verifier trait.
7. `commit_batch` must preserve current CycleMajor Dory streaming behavior: one padded trace scan, per-row work for all committed polynomials, small-scalar MSM for dense increment rows, and one-hot grouped additions for RA rows.
8. Homomorphic batch proving and verification have byte-identical Fiat-Shamir behavior between prover and verifier.
   They absorb the same claim count, the same evaluations, and draw the same per-point RLC challenges in the same order.
9. Source-backed batch opening returns both `PCS::BatchProof` and a public description of the output relation created by the PCS.
   In transparent mode the output can be a public scalar.
   In ZK mode the output can be a hiding commitment plus prover-only witness data such as the hidden scalar and blinding.
10. The PCS owns the batch-opening transcript challenge schedule.
    `jolt-core` must not sample Stage 8 fusion challenges and then ask the PCS to verify a pre-fused singleton opening.
    The relation returned by the PCS is the source of truth for BlindFold coefficients.
11. Stage 8 raw opening terms carry two distinct scalars when needed: the raw accumulator evaluation and an `eval_scale` describing how that raw value embeds into the PCS output relation.
    Dory's current invariant must be preserved: source, hint, and commitment fusion use the PCS batch coefficient, while BlindFold/evaluation coefficients use `batch_coefficient * eval_scale`.
12. `OpeningClaim` is generic over `PCS: CommitmentSchemeVerifier`, not over a raw commitment type.
   Verifier-only code can name opening claims without importing prover-only PCS types.
13. `jolt-core` keeps protocol-specific opening bookkeeping.
   `OpeningId`, `PolynomialId`, `SumcheckId`, `OpeningPoint`, `ProverOpeningAccumulator`, and `VerifierOpeningAccumulator` do not move into `jolt-openings`.
14. Dory layout, Dory matrix embedding policy, Stage 8 claim ordering, and BlindFold constraints do not move into `jolt-openings`.
    Stage 8 claim collection remains in `jolt-core`; Stage 8 fusion and proof-output construction move behind the PCS batch-opening API.
15. `jolt-openings` must not expose Dory's `sigma`/`nu` split, Dory matrix shape, or a generic associated partition type whose real values are Dory-specific.
    It may expose backend-neutral traversal facts, such as a natural commitment chunk length, that a concrete backend can interpret privately.
16. Opening hints for schemes whose opening algorithm depends on a commitment traversal must carry enough backend-owned information to replay the same traversal at opening time.
    For Dory, the hint records the selected chunk length and row commitments, and Dory derives its private opening shape from that hint instead of recomputing a balanced split from the opening point.
17. The merge-target API does not require `jolt-core` to call shaped PCS extensions.
18. Dory's current transparent and ZK proofs remain verifier-compatible with current `main`.
19. `JoltProof` stores the opening proof as `PCS::BatchProof`, not `PCS::Proof`.
20. Standard and ZK `muldiv` end-to-end proofs continue to pass.
21. The implementation introduces no Akita dependency.
22. The final PR state contains no in-core PCS compatibility trait family.
    `jolt-core/src/poly/commitment/commitment_scheme.rs` should no longer define `CommitmentScheme`, `SourceBatchCommitmentScheme`, `BatchOpeningScheme`, or `ZkOpeningSupport` as wrappers around `jolt-openings`.
    Direct users should import the canonical `jolt-openings` traits or a narrow backend-owned extension trait.
23. `cargo tree -d` must not show duplicate resolved versions of `jolt-field`, `jolt-transcript`, `jolt-crypto`, or `jolt-openings`.

No new `jolt-eval` invariant is required for this spec.
The relevant invariants are proof acceptance, transcript parity, and prover/verifier consistency, which are covered by focused crate tests and `jolt-core` end-to-end tests.

### Non-Goals

1. Integrating Akita or Hachi.
2. Adding lattice parameters, lattice proof types, or Akita-specific Stage 8 branches.
3. Redesigning the whole proof format beyond replacing PCS-owned associated fields with the canonical `jolt-openings` types.
4. Moving Stage 8 sumcheck wiring, opening IDs, accumulator types, or BlindFold constraints into `jolt-openings`.
5. Replacing the current Jolt prover pipeline with the verifier/compiler runtime from `refactor/crates`.
6. Porting unrelated #1467 hygiene changes such as debug-test clippy suppressions when they are not needed on current `main`.
7. Moving BlindFold itself into `jolt-openings`.
8. Preserving the old internal `CommitmentScheme` trait as a permanent or merge-ready compatibility layer.
9. Changing guest execution, bytecode expansion, memory checking, instruction lookups, or sumcheck protocol semantics.
10. Changing Dory's transcript labels or proof verification behavior except where required to preserve current `main` behavior under the new trait API.

## Evaluation

### Acceptance Criteria

- [x] `crates/jolt-openings/src/schemes.rs` defines `CommitmentSchemeVerifier`, `PublicVerifierSetup`, `CommitmentScheme`, `AdditivelyHomomorphicVerifier`, `AdditivelyHomomorphic`, `ZkOpeningSchemeVerifier`, and `ZkOpeningScheme` with the role split.
- [x] `StreamingCommitment` is not part of the canonical `jolt-openings` API.
- [x] `crates/jolt-openings/src/sources.rs` defines `SourceId`, `SourceRow`, `CommitmentSource`, and `BatchCommitmentSource`.
- [x] `crates/jolt-openings/src/claims.rs` defines raw batch-opening claim types parameterized by claim id and source id.
- [x] `crates/jolt-openings/src/sources.rs` defines a source-backed batch-opening trait that can expose committed sources, opening hints, and efficient linear row folding without materializing every source.
- [x] `crates/jolt-openings/src/schemes.rs` defines source-backed transparent and ZK batch-opening methods that return proof plus output-relation metadata.
- [x] `CommitmentSchemeVerifier` contains `Field`, `VerifierSetup`, `Proof`, `BatchProof`, `verify`, `verify_batch`, and `bind_opening_inputs`.
- [x] `PublicVerifierSetup` contains `PublicParams` and `verifier_setup` for schemes whose verifier setup is derivable without prover setup.
- [x] `CommitmentScheme` extends `CommitmentSchemeVerifier` and contains `ProverSetup`, `OpeningHint`, `SetupParams`, `setup`, `project_verifier_setup`, `commit`, `commit_batch`, `open`, and `prove_batch`.
- [x] `commit_batch` has a default implementation that commits one source at a time, and Dory overrides it for batch-source row streaming.
- [x] Homomorphic extension traits contain only the additive-combination operations needed by the default homomorphic batch helper.
- [x] `crates/jolt-openings/src/homomorphic.rs` contains #1467's `homomorphic_prove_batch`, `homomorphic_verify_batch`, `rlc_combine`, and `rlc_combine_scalars`.
- [x] `homomorphic_prove_batch` and `homomorphic_verify_batch` group claims by opening point and use the same transcript schedule.
- [x] `crates/jolt-openings/src/claims.rs` exposes `ProverClaim<F, P>` and `OpeningClaim<F, PCS: CommitmentSchemeVerifier<Field = F>>`.
- [x] The old standalone `reduce_prover` / `reduce_verifier` production API is not part of the new production surface; ordinary Dory batch opening uses `prove_batch` / `verify_batch`, and Stage 8 uses source-backed batch opening.
- [x] `crates/jolt-openings/src/mock.rs` implements the split traits and has tests covering single-claim, multi-claim, shared-point, distinct-point, and tampered-evaluation cases.
- [x] `crates/jolt-dory` implements the split trait family while preserving current `main` wrapper types, bounded deserialization, transcript bridge, batch-source streaming support, and ZK behavior.
- [x] `crates/jolt-dory` does not expose merge-target public APIs for borrowed-ark setup, Dory-native proof entrypoints, or standalone begin/feed/finish streaming commitments; those strategies live behind canonical `jolt-openings` trait methods.
- [x] Dory `commit_batch` preserves CycleMajor trace commitment shape: same polynomial order, same row length, same row-commitment ordering, same `DoryHint` row commitments, and same transcript-visible commitments as current `main`.
- [x] `DoryScheme::BatchProof = Vec<DoryProof>` for the homomorphic Dory implementation.
- [x] Ordinary `DoryScheme::prove_batch` and `DoryScheme::verify_batch` can still use the generic homomorphic helper for non-Stage-8 callers.
- [x] Dory's source-backed batch-opening implementation owns Stage 8 fusion, samples the same challenge schedule, combines hints and commitments internally, and returns the output relation.
- [x] `jolt-core` depends on `jolt-openings` and `jolt-dory`.
- [x] `jolt-core` imports the canonical `jolt_openings` PCS traits and extension traits directly instead of the old in-core PCS trait family.
- [x] `jolt-core` call sites use `PCS::Output` for commitments, matching `jolt_crypto::Commitment`, rather than the old `PCS::Commitment`.
- [x] `jolt-core` call sites use `PCS::OpeningHint`, `PCS::Proof`, `PCS::BatchProof`, `PCS::ProverSetup`, and `PCS::VerifierSetup` from `jolt-openings` directly.
- [x] `JoltProof` stores `joint_opening_proof` as canonical `PCS::BatchProof`.
- [x] Prover preprocessing and verifier preprocessing store canonical `jolt-openings` setup, commitment, and opening-hint associated types.
- [x] Public prover/verifier generic bounds and SDK-facing generated functions name the canonical `jolt-openings` PCS traits.
- [x] Stage 8 prover calls canonical source-backed `jolt-openings` batch opening with raw opening terms and a Stage 8 opening source batch, not with one already-combined streaming joint claim.
- [x] Stage 8 verifier calls the matching canonical source-backed batch verifier with raw verifier terms and commitments, not with one precomputed joint commitment.
- [x] Stage 8 keeps the same dense increment scaling, RA polynomial ordering, advice Lagrange scaling, `opening_ids`, `constraint_coeffs`, and `joint_claim` semantics as current `main`, with those coefficients returned by the PCS output relation.
- [x] ZK mode receives the Dory evaluation commitment and prover-only blinding through the PCS batch-opening result needed by BlindFold.
- [x] ZK evaluation commitment output, evaluation blinding witness, and Pedersen generator derivation are expressed through `jolt-openings` results / backend-owned extension traits rather than in-core `ZkOpeningSupport`.
- [x] `DoryCommitmentScheme`'s old wrapper types are removed; Dory layout glue remains protocol-owned in `jolt-core`.
- [x] The in-core `CommitmentScheme`, `SourceBatchCommitmentScheme`, `BatchOpeningScheme`, and `ZkOpeningSupport` bridges are removed.
- [x] No production `jolt-core` code calls Dory-specific borrowed-ark bridge APIs to enter the PCS; it enters through canonical `jolt-openings` methods.
- [x] HyperKZG and the mock PCS compile against the same `jolt-openings` trait family used by Dory, or are explicitly removed from `jolt-core` production generic bounds if they are no longer supported there.
- [x] `cargo nextest run -p jolt-openings --features test-utils --cargo-quiet` passes.
- [x] `cargo nextest run -p jolt-dory --cargo-quiet` passes.
- [x] `cargo nextest run -p jolt-core muldiv --cargo-quiet --features host` passes.
- [x] `cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk` passes.
- [x] `cargo clippy --all --features host -q --all-targets -- -D warnings` passes.
- [x] `cargo clippy --all --features host,zk -q --all-targets -- -D warnings` passes.
- [x] `cargo fmt -q` produces no diff.

Source-traversal cleanup acceptance criteria:

- [x] `ShapedCommitmentScheme`, `ShapedZkOpeningScheme`, `commit_with_shape`, and `commit_zk_with_shape` are removed from the public merge-target API.
- [x] `CommitmentSource` exposes `natural_chunk_len` instead of accepting Dory's `sigma` as the generic source traversal parameter.
- [x] `BatchCommitmentSource` exposes the same batch-level traversal hint for a selected source-id set, so CycleMajor can preserve one trace scan for all committed sources.
- [x] Dory derives its private `sigma` from the hint's recorded chunk length and its private `nu` from the hint's row-commitment count.
- [x] Dory commitment hints record the selected chunk length, and `open` / `open_zk` use the hint to replay the commitment traversal rather than recomputing a balanced split from the opening point length.
- [x] `jolt-core` call sites enter through `PCS::commit`, `PCS::commit_zk`, `PCS::commit_batch`, or `PCS::commit_batch_zk`; no production `jolt-core`, SDK, or transpiler call site calls a shaped PCS extension.
- [x] `PolynomialCommitmentSource` and `CycleMajorTraceBatch` no longer read Dory globals in order to satisfy a generic source API; Jolt layout choices such as AddressMajor striding and one-hot flat-index order are passed into the source adapter as source configuration.

### Testing Strategy

Focused `jolt-openings` tests should validate the API in isolation:

1. A mock PCS can commit, prove, and verify one opening.
2. Multiple claims at the same point reduce to one proof.
3. Multiple claims at distinct points produce one proof per point group.
4. Mixed shared and distinct points verify.
5. A tampered evaluation is rejected.
6. `rlc_combine` and `rlc_combine_scalars` agree with direct polynomial evaluation.
7. Prover and verifier transcripts remain in sync for the batch helper path.
8. Source-backed batch opening returns a proof and a linear output relation whose coefficients match the PCS-sampled challenge schedule.
9. ZK source-backed batch opening returns hidden outputs in public data and keeps output values/blinds prover-only.

Focused `jolt-dory` tests should validate the real PCS implementation:

1. Commit, open, and verify round trips still pass.
2. Homomorphic combination verifies against direct polynomial combination.
3. `prove_batch` / `verify_batch` pass for one claim and multiple claims.
4. Source-batch streamed Dory commitment still matches direct commitment.
5. Dory proof deserialization still rejects oversized or malformed proof round counts.
6. ZK opening behavior still produces and verifies the expected hiding commitment.
7. A source committed with a non-default natural chunk length opens using the chunk length recorded in `DoryHint`.
8. CycleMajor batch commitment produces byte-identical commitments before and after replacing shaped traits with source traversal hints.
9. Stage 8 source-backed Dory opening produces the same proof acceptance behavior and BlindFold coefficients as the pre-cutover streaming RLC path.

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
There are two separate streaming paths to keep straight.
Witness commitment streaming is the current CycleMajor commitment path that builds Dory row commitments while scanning the padded trace.
That path should move behind `commit_batch` and `commit_batch_zk`.
Stage 8 opening streaming is the current Dory RLC path that builds a joint polynomial directly from the trace rather than regenerating witness polynomials.
That path should keep the same algorithmic shape, but the joint source should be constructed inside the PCS source-backed batch-opening implementation instead of in `jolt-core`.

Performance requirements:

1. Preserve the current Stage 8 streaming RLC optimization while moving fusion behind the PCS batch-opening boundary.
2. Avoid materializing all committed polynomials in Stage 8 merely to fit the new API.
3. Keep Dory row-commitment hint combination in the hot path when it avoids recomputing row commitments.
4. Do not introduce dynamic dispatch in the prover hot path.
5. Keep parallelism in Dory commitment, opening-source folding, and hint combination.
6. Preserve the current distinction between source/hint/commitment coefficients and evaluation/BlindFold coefficients.
7. Preserve current `#[inline]` and low-level optimized group operation behavior where code moves between crates.

Concrete checks:

1. `cargo nextest run -p jolt-core muldiv --cargo-quiet --features host` should not show obvious runtime blowups.
2. Existing Dory benches should compile.
3. Run the existing Criterion benchmark path for Dory/opening-heavy workloads before and after the full `jolt-core` type-family cutover.
   The PR should include the measured comparison against `main` or a clear explanation of any unavoidable noise.
4. Treat any material prover regression in Stage 8 or CycleMajor commitment as a blocker unless the implementation explains and justifies a deliberate algorithmic tradeoff.
5. Treat the source-traversal cleanup as performance-sensitive: replacing `sigma` with a natural chunk length must preserve the same chunk size, row order, row encodings, Dory row commitments, and parallel schedule for existing CycleMajor paths.

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

`jolt-openings` owns the abstract PCS API, source traits, batch-opening claim/result vocabulary, and generic ordinary homomorphic batch helper.
`jolt-dory` owns Dory-specific commitments, proofs, setup types, transcript adaptation, row commitments, batch-source commitment adapters, source-backed opening fusion, and Dory proof hardening.
`jolt-core` owns the Jolt protocol's opening IDs, accumulators, Stage 8 raw claim assembly, proof object, source adapters, and BlindFold wiring.

The trait hierarchy is:

```text
CommitmentSchemeVerifier
  - Field
  - VerifierSetup
  - Proof
  - BatchProof
  - verify
  - verify_batch
  - verify_batch_opening
  - bind_opening_inputs

PublicVerifierSetup: CommitmentSchemeVerifier
  - PublicParams
  - verifier_setup

CommitmentScheme: CommitmentSchemeVerifier
  - ProverSetup
  - OpeningHint
  - SetupParams
  - setup
  - project_verifier_setup
  - commit
  - commit_batch
  - open
  - prove_batch
  - prove_batch_opening

AdditivelyHomomorphicVerifier: CommitmentSchemeVerifier
  - combine

AdditivelyHomomorphic: AdditivelyHomomorphicVerifier + CommitmentScheme
  - combine_hints

ZkOpeningSchemeVerifier: CommitmentSchemeVerifier
  - HidingCommitment
  - verify_zk
  - verify_batch_zk
  - verify_batch_opening_zk

ZkOpeningScheme: ZkOpeningSchemeVerifier + CommitmentScheme
  - Blind
  - commit_zk
  - commit_batch_zk
  - open_zk
  - prove_batch_zk
  - prove_batch_opening_zk
```

This hierarchy is a role split, not a lifecycle split.
Verifier-only code can bound on `CommitmentSchemeVerifier` without naming prover-only data.
Prover code gets the verifier surface because `CommitmentScheme` extends `CommitmentSchemeVerifier`.
Verifier setup construction is not part of the base verifier trait because not every scheme can derive verifier setup from public parameters alone.
Transparent schemes such as Dory can implement `PublicVerifierSetup`; structured-reference-string schemes such as HyperKZG receive verifier setup generated by setup and do not need an identity-style verifier setup constructor.
Single-opening `open` and `verify` are required PCS basics.
Ordinary `prove_batch` / `verify_batch` remain useful for generic homomorphic batching over already-available sources.
Stage 8 should use the source-backed batch-opening methods because the PCS, not `jolt-core`, should own the fusion challenge and output relation.
Homomorphic extension traits only add linear combination primitives.

### Source-Oriented Commitment API

The PCS should consume polynomial sources directly.
Streaming is not a standalone commitment-scheme trait, but it is also not quarantined entirely inside source implementations.
The source abstraction describes what data is available and which traversal shapes can expose it without materializing all committed polynomials.
The PCS implementation still owns the commitment algorithm: parallel scheduling, row MSM strategy, one-hot grouping, tier-2 aggregation, hint construction, and transparent-vs-ZK finishing.
The source abstraction should describe traversal in backend-neutral terms.
In particular, it should expose a natural commitment chunk length, not Dory's `sigma` and not a scheme-specific partition enum.
For current Dory/Jolt paths, `chunk_len` is the number of columns in each streamed row.
Dory privately derives `sigma = log2(chunk_len)` and derives `nu` from its backend-owned row-commitment hint, but no `jolt-openings` trait method should name those variables.
If a future backend has a different partition concept, it can ignore this hint, derive its own plan from source facts, or add a backend-owned extension trait only if the generic source facts are insufficient.

`jolt-openings` should expose source traits with two layers.
The core semantic object is `CommitmentSource`; `SourceRow` is only a traversal view for commitment implementations that can exploit row structure:

```rust
/// Stable identifier for a committed source inside a batch commitment source.
///
/// In the Dory/Jolt trace path this can be a logical committed polynomial id
/// such as `InstructionRa(0)` or `RamInc`. In a packed PCS path this can instead
/// identify a packed witness group. The id names what the PCS commits to; it
/// does not have to be one logical Jolt polynomial.
pub trait SourceId: Copy + Eq + Ord + Send + Sync + 'static {}

/// A compact coordinate into a one-hot domain.
///
/// The value is the hot basis-vector index `k` in `e_k`. The surrounding
/// `OneHotRow` carries the domain size, so this type only stores the coordinate.
/// Current Jolt one-hot chunks have at most `2^8` entries and the extraction
/// helpers already return `u8`, so this intentionally avoids `usize`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct OneHotIndex(u8);

impl OneHotIndex {
    /// Creates a one-hot coordinate when `index < 2^log_domain_size`.
    ///
    /// Returns `None` if the caller gives a coordinate outside the row domain.
    pub fn new(index: u8, log_domain_size: u8) -> Option<Self> {
        ((index as usize) < (1usize << log_domain_size)).then_some(Self(index))
    }

    /// Returns the coordinate as an array/vector index.
    pub fn get(self) -> usize {
        self.0 as usize
    }
}

/// A row of one-hot entries, one entry per trace column in the current chunk.
///
/// `log_domain_size` says that every hot coordinate lives in a one-hot domain
/// of size `2^log_domain_size`. For current Jolt this is `4` or `8`.
/// The `entries` field records whether each trace column has a required hot
/// coordinate or may be zero, depending on the row source. Dory consumes this
/// as its streaming one-hot chunk shape: one row commitment per hot coordinate,
/// with trace columns contributing to the row for their hot coordinate.
pub struct OneHotRow<'a> {
    pub log_domain_size: u8,
    pub entries: OneHotEntries<'a>,
}

/// Per-column one-hot data for a `OneHotRow`.
///
/// This enum avoids forcing all one-hot rows through `Option`.
/// `InstructionRa` and `BytecodeRa` have one hot coordinate for every trace
/// column. `RamRa` can have no committed RAM address for a column after address
/// remapping, so it needs the zero-or-one representation.
pub enum OneHotEntries<'a> {
    /// Every trace column contributes exactly one one-hot basis vector.
    ///
    /// Entry `indices[col] = k` means column `col` contributes `e_k`.
    OnePerColumn(&'a [OneHotIndex]),

    /// Each trace column contributes either zero or one one-hot basis vector.
    ///
    /// Entry `indices[col] = Some(k)` means column `col` contributes `e_k`.
    /// Entry `indices[col] = None` means column `col` contributes the zero vector.
    MaybeZero(&'a [Option<OneHotIndex>]),
}

impl OneHotEntries<'_> {
    /// Number of trace columns represented by this one-hot row.
    pub fn len(&self) -> usize {
        match self {
            Self::OnePerColumn(indices) => indices.len(),
            Self::MaybeZero(indices) => indices.len(),
        }
    }
}

/// A borrowed row view of a polynomial source.
///
/// This is a traversal hint, not the core polynomial abstraction. Backends that
/// can exploit row structure, such as Dory, consume these rows directly.
/// Backends that do not care about the encoding may interpret the row as field
/// evaluations and use the default source traversal.
pub enum SourceRow<'a, F> {
    /// A dense row of field evaluations.
    FieldElements(&'a [F]),

    /// A dense row whose entries occupy evenly spaced columns.
    ///
    /// `column_stride` is the distance between consecutive occupied columns.
    /// Values with stride `4`, for example, occupy columns `0, 4, 8, ...`.
    /// This captures layout-induced sparse rows without making the source know
    /// a concrete commitment backend.
    StridedFieldElements {
        values: &'a [F],
        column_stride: usize,
    },

    /// A dense row of signed integers embedded canonically into `F`.
    ///
    /// This preserves the current Dory small-scalar MSM path for increment
    /// polynomials without first materializing field elements.
    I128(&'a [i128]),

    /// A strided signed-integer row embedded canonically into `F`.
    StridedI128 {
        values: &'a [i128],
        column_stride: usize,
    },

    /// A dense row of unsigned 64-bit integers embedded canonically into `F`.
    ///
    /// This preserves compact materialized advice and benchmark paths without
    /// first materializing field elements.
    U64(&'a [u64]),

    /// A strided unsigned-integer row embedded canonically into `F`.
    StridedU64 {
        values: &'a [u64],
        column_stride: usize,
    },

    /// A streaming one-hot chunk whose entries are one-hot vectors over a small
    /// domain.
    ///
    /// This preserves the current Dory grouped-addition path for RA polynomials.
    /// Backends that do not exploit this shape can expand it explicitly in
    /// hot-coordinate-major order.
    OneHot(OneHotRow<'a>),
}

/// A single polynomial-like object that a PCS can commit to and open.
///
/// The source owns the semantic operations: evaluate at a point, traverse rows,
/// and fold rows for opening-time vector/matrix products. It may be materialized
/// or lazy; for example, it can be backed by the execution trace.
pub trait CommitmentSource<F>: Send + Sync {
    /// Number of multilinear variables in the source.
    fn num_vars(&self) -> usize;

    /// Evaluates the source at a multilinear point.
    fn evaluate(&self, point: &[F]) -> F;

    /// Preferred row length for commitment traversal, when the source has one.
    ///
    /// The value is a source traversal fact, not a commitment-scheme partition.
    /// For Jolt's current Dory-backed sources it is the row length that
    /// preserves existing trace streaming behavior. A backend may ignore it,
    /// clamp it, or use its own default when the source returns `None`.
    fn natural_chunk_len(&self) -> Option<usize> {
        None
    }

    /// Visits row-shaped chunks of the source using `chunk_len` columns.
    ///
    /// Implementations should call `visit(row_index, row)` once for each row.
    /// The borrowed row only has to remain valid for the duration of the visit
    /// call, which lets trace-backed sources allocate temporary row buffers and
    /// avoid ownership wrappers such as `Cow`.
    fn for_each_row<V>(&self, chunk_len: usize, visit: V)
    where
        V: for<'row> FnMut(usize, SourceRow<'row, F>);

    /// Maps row-shaped chunks into owned backend results.
    ///
    /// The default is a sequential traversal through `for_each_row`.
    /// Materialized sources can override this to parallelize over borrowed row
    /// chunks without copying rows into an owned staging buffer.
    fn map_rows<R, V>(&self, chunk_len: usize, visit: V) -> Vec<R>
    where
        R: Send,
        V: for<'row> Fn(usize, SourceRow<'row, F>) -> R + Send + Sync;

    /// Folds rows against the left-side weights used by opening algorithms.
    ///
    /// Dory uses this shape for its vector/matrix product path; other schemes
    /// can implement it by materializing or by using their own source layout.
    fn fold_rows(&self, left: &[F], chunk_len: usize) -> Vec<F>;
}

/// A batch of committed sources that can share one traversal.
///
/// This is the no-regression hook for current CycleMajor Dory commitment. It
/// lets Jolt scan the padded trace once and produce row work for many committed
/// sources in source-id order. The default PCS implementation can ignore this
/// hook and commit sources one at a time through `source(id)`.
pub trait BatchCommitmentSource<F>: Send + Sync {
    type Id: SourceId;

    /// Borrowed single-source adapter for a source in this batch.
    type Source<'a>: CommitmentSource<F> + 'a
    where
        Self: 'a;

    /// All source ids this batch can expose, in the natural protocol order.
    fn source_ids(&self) -> &[Self::Id];

    /// Number of multilinear variables in the selected source.
    fn num_vars(&self, id: Self::Id) -> usize;

    /// Preferred shared row length for committing the selected sources.
    ///
    /// Returning a shared chunk length is how CycleMajor preserves today's
    /// one-pass trace scan. The value is still backend-neutral: Dory interprets
    /// it as a row width, while a different backend may use it only as a cache
    /// tiling hint or ignore it completely.
    fn natural_chunk_len(&self, ids: &[Self::Id]) -> Option<usize> {
        None
    }

    /// Returns a single-source view for backends that do not use batch traversal.
    fn source(&self, id: Self::Id) -> Self::Source<'_>;

    /// Maps a row visitor over many sources while sharing the source traversal.
    ///
    /// For Jolt's trace-backed source this should preserve the current loop
    /// shape: one padded trace scan, parallel work over trace rows, and inner
    /// parallel work over the requested source ids. The returned vector is
    /// row-major: `output[row_index][id_index]`.
    fn map_rows<R, V>(
        &self,
        chunk_len: usize,
        ids: &[Self::Id],
        visit: V,
    ) -> Vec<Vec<R>>
    where
        R: Send,
        V: for<'row> Fn(Self::Id, SourceRow<'row, F>) -> R + Send + Sync;
}
```

The generic semantics are:

1. `FieldElements` is the canonical dense row of field evaluations.
2. `StridedFieldElements` is the same dense row view, but its values occupy evenly spaced columns and all skipped columns are zero.
3. `I128` and `U64` are dense rows of small scalars embedded canonically into `F`.
4. `StridedI128` and `StridedU64` are the small-scalar counterparts of `StridedFieldElements`.
5. `OneHot` is a row of one-hot vector entries.
   `log_domain_size` means each column entry lives in `{0, ..., 2^log_domain_size - 1}`.
   `OneHotEntries::OnePerColumn(indices)` means every column contributes one basis vector.
   `OneHotEntries::MaybeZero(indices)` means `Some(k)` contributes `e_k`, and `None` contributes the zero vector.
   Its dense expansion for this row shape is hot-coordinate-major inside the chunk: entry `(hot_index, column)` maps to `hot_index * num_columns + column`.

Only `CommitmentSource` and `BatchCommitmentSource` are core API concepts.
`I128`, `U64`, strided rows, and `OneHot` are optional row encodings.
They are included to preserve current Jolt/Dory performance without forcing `jolt-core` to call Dory-specific APIs:

1. `I128` maps exactly to the previous Dory dense-row chunk path for `RdInc` and `RamInc`.
2. `U64` preserves compact materialized advice and benchmark paths.
3. Strided rows preserve AddressMajor dense-polynomial embedding, where dense values sit in every `K`th column of the Dory matrix.
4. `OneHot` maps exactly to the previous Dory one-hot chunk path for `InstructionRa`, `BytecodeRa`, and `RamRa`.
5. A backend that does not care about these encodings can immediately materialize or interpret them as field rows.

`OneHotIndex` is intentionally not `usize`.
Current Jolt chunks have `log_k_chunk` equal to `4` or `8`, and the chunk extraction helpers already return `u8`.
The row encoding should require `log_domain_size <= 8`, so `u8` covers every valid hot coordinate.
Current code uses `Option<usize>` for all one-hot rows, but semantically only `RamRa` needs optional columns:

1. `InstructionRa` always has one lookup index per cycle, so it can use `OneHotEntries::OnePerColumn`.
2. `BytecodeRa` always has one bytecode PC chunk per cycle, so it can use `OneHotEntries::OnePerColumn`.
3. `RamRa` can map a cycle to no committed RAM address after `remap_address`, so it uses `OneHotEntries::MaybeZero`.

This is also why the source trait should not have an associated `Partition` type.
The source knows how to expose rows and evaluate the committed object; it does not know every backend's preferred partition language.
If `CommitmentSource` had `type Partition`, then either Jolt would have to choose a Dory-shaped partition for every source, or `jolt-openings` would need a universal enum that tries to anticipate Dory, Akita, HyperKZG, and future GPU-specific layouts.
Both choices put backend strategy into the wrong layer.
A natural chunk length is deliberately weaker: it is just the row width the source can stream efficiently.
Dory can turn that into its matrix split; another backend can ignore it or reinterpret it as a tiling hint.

This is the Dory-dependent part of the source API.
The alternative is to move these encodings behind a Dory-only trait, but then stable Rust cannot make the generic `PCS::commit_batch(&batch, ...)` dispatch to the optimized Dory implementation only for batches that implement that Dory-only trait.
Without specialization or downcasting, Jolt would have to call a Dory-specific method directly, which defeats the cutover goal.

The current branch uses this source-traversal shape directly.
Source traversal receives a backend-neutral `chunk_len`, Jolt-specific layout choices are carried by source-adapter configuration, and Dory's `sigma`/`nu` interpretation is private to `jolt-dory`.

`BatchCommitmentSource::map_rows` is the no-regression traversal hook.
It lets Jolt's trace-backed source scan the padded trace once, parallelize over trace rows, and run the caller's row-processing closure for every requested committed polynomial in source-id order.
Calling `CommitmentSource::for_each_row` independently for every committed polynomial would be simpler but would rescan the trace per polynomial, so it is not acceptable for the zkVM hot path.

The hook does not make the source responsible for the commitment algorithm.
Dory's `commit_batch` and `commit_batch_zk` choose how to consume the row views, how much work to parallelize at the row/source/aggregation layers, and how to build `DoryHint`.
The default `commit_batch` can ignore shared traversal and commit one `CommitmentSource` at a time.
Future CPU, GPU, or lattice backends can make different scheduling choices behind the same PCS method boundary.

The PCS trait should commit sources, not materialized polynomials:

```rust
fn commit<S: CommitmentSource<Self::Field> + ?Sized>(
    source: &S,
    setup: &Self::ProverSetup,
) -> (Self::Output, Self::OpeningHint);

fn commit_batch<B: BatchCommitmentSource<Self::Field>>(
    batch: &B,
    ids: &[B::Id],
    setup: &Self::ProverSetup,
) -> Vec<(Self::Output, Self::OpeningHint)> {
    ids.iter()
        .map(|&id| {
            let source = batch.source(id);
            Self::commit(&source, setup)
        })
        .collect()
}
```

`ZkOpeningScheme` should mirror this shape:

```rust
fn commit_zk<S: CommitmentSource<Self::Field> + ?Sized>(
    source: &S,
    setup: &Self::ProverSetup,
) -> (Self::Output, Self::OpeningHint);

fn commit_batch_zk<B: BatchCommitmentSource<Self::Field>>(
    batch: &B,
    ids: &[B::Id],
    setup: &Self::ProverSetup,
) -> Vec<(Self::Output, Self::OpeningHint)> {
    ids.iter()
        .map(|&id| {
            let source = batch.source(id);
            Self::commit_zk(&source, setup)
        })
        .collect()
}
```

Dory overrides both `commit_batch` and `commit_batch_zk`.
The implementation is the current streaming algorithm moved behind the PCS boundary:

1. Choose `chunk_len` from `batch.natural_chunk_len(ids)` when present, otherwise from Dory's default balanced layout.
2. Privately derive Dory's commitment-time row width from `chunk_len`; the opening-time split is later replayed from `DoryHint`.
3. Call `batch.map_rows(chunk_len, ids, |id, row| commit_row(row, setup))` over padded trace rows.
4. For each row and each requested source id, compute a Dory row commitment from `SourceRow`.
5. Receive row-major row commitments in the same source-id order as `ids`.
6. Transpose to per-source row-commitment vectors exactly as current `generate_and_commit_witness_polynomials` does.
7. Aggregate each per-source row-commitment vector with Dory tier 2 in transparent or ZK mode.
8. Return the same `(DoryCommitment, DoryHint)` shape used by `open` / `open_zk`, with `DoryHint` recording the selected `chunk_len`.

The Dory row helper is private to `jolt-dory`:

```rust
fn commit_row(row: SourceRow<'_, Fr>, setup: &DoryProverSetup) -> Vec<ArkG1> {
    match row {
        SourceRow::FieldElements(values) => commit_field_row(values, setup),
        SourceRow::StridedFieldElements { values, column_stride } => {
            commit_field_row_at_stride(values, column_stride, setup)
        }
        SourceRow::I128(values) => commit_small_scalar_row(values, setup),
        SourceRow::StridedI128 { values, column_stride } => {
            commit_small_scalar_row_at_stride(values, column_stride, setup)
        }
        SourceRow::U64(values) => commit_u64_row(values, setup),
        SourceRow::StridedU64 { values, column_stride } => {
            commit_u64_row_at_stride(values, column_stride, setup)
        }
        SourceRow::OneHot(row) => commit_onehot_row(row, setup),
    }
}
```

The one-hot Dory helper is the previous `process_chunk_onehot` behavior with only the row representation changed:

```rust
fn commit_onehot_row(row: OneHotRow<'_>, setup: &DoryProverSetup) -> Vec<ArkG1> {
    let k = 1usize << row.log_domain_size;

    let row_len = row.entries.len();
    let g1_bases = setup.g1_vec[..row_len]
        .iter()
        .map(|g| g.0.into_affine())
        .collect::<Vec<G1Affine>>();

    let mut columns_by_hot_index = vec![Vec::new(); k];
    match row.entries {
        OneHotEntries::OnePerColumn(indices) => {
            debug_assert_eq!(indices.len(), row_len);
            for (column, hot_index) in indices.iter().enumerate() {
                columns_by_hot_index[hot_index.get()].push(column);
            }
        }
        OneHotEntries::MaybeZero(indices) => {
            debug_assert_eq!(indices.len(), row_len);
            for (column, hot_index) in indices.iter().enumerate() {
                if let Some(hot_index) = hot_index {
                    columns_by_hot_index[hot_index.get()].push(column);
                }
            }
        }
    }

    let sums = batch_g1_additions_multi_affine(&g1_bases, &columns_by_hot_index);

    let mut row_commitments = vec![ArkG1(G1Projective::zero()); k];
    for (hot_index, sum) in sums.into_iter().enumerate() {
        if !columns_by_hot_index[hot_index].is_empty() {
            row_commitments[hot_index] = ArkG1(G1Projective::from(sum));
        }
    }
    row_commitments
}
```

This is the same division of labor as the old `process_chunk` / `process_chunk_onehot` path, but without exposing Dory tier-1 chunks as a public PCS trait.
The row-processing closure is generic, not trait-object based, so the Jolt trace-batch hot path can remain statically dispatched.
Dory decides the parallel schedule inside its `commit_batch` implementation while using the concrete batch source for shared data access.
For non-Dory schemes, the default `commit_batch` is correct and simple.
They can add an optimized override only if their backend benefits from batch-row source traversal.

### Source-Backed Batch Opening API

The opening-side API should mirror the commitment-side source split, but it must abstract one more thing: the PCS output.
For Stage 8, the useful boundary is not "Jolt hands the PCS one already-combined polynomial."
The useful boundary is "Jolt hands the PCS raw claims against committed sources, and the PCS returns the proof plus the relation between raw claims and whatever output it opened."

The generic claim vocabulary should distinguish:

1. The `claim_id`, which names the protocol value that later systems such as BlindFold care about.
   In Jolt this is `OpeningId`.
2. The `source_id`, which names the committed source that the PCS opens.
   In Dory/Jolt this is usually `CommittedPolynomial`; in a packed scheme it could name a packed witness group.
3. The raw evaluation, before any Stage 8 embedding factor.
4. The evaluation scale, which says how the raw value contributes to the PCS output relation.
   This is not necessarily the same scalar used to combine sources, hints, or commitments.

```rust
/// Public verifier knowledge about a claimed evaluation.
///
/// Transparent opening verification receives the scalar value.
/// ZK opening verification receives no scalar value; the PCS output is instead
/// represented by a hiding commitment returned in `BatchOpeningPublic`.
pub enum OpeningValue<F> {
    Public(F),
    Hidden,
}

/// Opening point in both protocol and backend coordinates.
///
/// `public` is the semantic point used for transcript binding and for the
/// protocol's opening claim. `proof` is the coordinate order the PCS backend
/// uses internally for the proof. Most schemes set these equal. Dory/Jolt sets
/// `proof` to the current layout-coordinate opening point.
pub struct BatchOpeningPoint<F> {
    pub public: Vec<F>,
    pub proof: Vec<F>,
}

/// Prover-side raw opening term for source-backed batch opening.
///
/// `eval` is the raw accumulator value for `claim_id`.
/// `eval_scale` embeds that raw value into the PCS output relation.
/// For current Dory Stage 8, source/hint/commitment fusion uses only the PCS
/// challenge coefficient, while the returned BlindFold relation uses
/// `challenge_coefficient * eval_scale`.
pub struct ProverBatchOpeningTerm<F, ClaimId, SourceId> {
    pub claim_id: ClaimId,
    pub source_id: SourceId,
    pub point: BatchOpeningPoint<F>,
    pub eval: F,
    pub eval_scale: F,
}

/// Verifier-side raw opening term for source-backed batch opening.
pub struct VerifierBatchOpeningTerm<F, PCS, ClaimId, SourceId>
where
    PCS: CommitmentSchemeVerifier<Field = F>,
{
    pub claim_id: ClaimId,
    pub source_id: SourceId,
    pub commitment: PCS::Output,
    pub point: BatchOpeningPoint<F>,
    pub eval: OpeningValue<F>,
    pub eval_scale: F,
}

/// A linear coefficient applied to a committed source.
pub struct LinearSourceTerm<F, SourceId> {
    pub source_id: SourceId,
    pub coefficient: F,
}
```

The source batch used for opening is deliberately different from `BatchCommitmentSource`.
Commitment asks how to traverse many sources to produce commitments.
Opening asks how to recover one source, replay its opening hint, and optionally fold a linear combination of sources without materializing them all.

```rust
/// A batch of already-committed sources available for opening.
///
/// Opening hints are backend-owned, so the trait is parameterized by the hint
/// type instead of by a concrete PCS. This keeps source plumbing neutral while
/// still allowing Dory to preserve its row-commitment hint path.
pub trait BatchOpeningSource<F, OpeningHint>: Send + Sync
where
    F: Field,
{
    type Id: SourceId;

    type Source<'a>: CommitmentSource<F> + 'a
    where
        Self: 'a;

    /// Returns the source identified by `id`.
    fn source(&self, id: Self::Id) -> Self::Source<'_>;

    /// Borrows the opening hint produced when this source was committed.
    ///
    /// This is intentionally borrowed rather than returned by value: Dory hints
    /// contain row commitments, and source-backed batch opening must be able to
    /// combine those hints without cloning the row-commitment vectors in Stage 8.
    fn opening_hint(&self, id: Self::Id) -> &OpeningHint;

    /// Folds a linear combination of sources against the left-side weights used
    /// by opening algorithms.
    ///
    /// The default implementation may fold each source separately and add the
    /// results. Dory's Stage 8 source batch should override this to preserve the
    /// current one-pass streaming RLC path.
    fn fold_linear_rows(
        &self,
        terms: &[LinearSourceTerm<F, Self::Id>],
        left: &[F],
        chunk_len: usize,
    ) -> Vec<F>;
}
```

The PCS batch-opening result should describe the opened output without requiring callers to know whether the backend used one Dory RLC, several native batch proofs, a packed opening, or a future nonlinear proof relation.

```rust
/// The value opened by a batch-opening proof.
pub enum BatchOutputValue<F, HidingCommitment> {
    Public(F),
    Hidden(HidingCommitment),
}

/// One PCS output created during batch opening.
pub struct OpenedBatchOutput<F, HidingCommitment> {
    /// Public semantic point used for protocol binding.
    pub point: Vec<F>,
    pub value: BatchOutputValue<F, HidingCommitment>,
}

/// Relation between raw opening claims and an opened PCS output.
pub struct BatchOutputRelation<F, ClaimId> {
    pub output_index: usize,
    pub expression: BatchOutputExpression<F, ClaimId>,
}

/// Current Dory/Jolt uses `Linear`.
/// `SumOfProducts` leaves room for a future PCS whose public output relation is
/// not a single linear combination of raw openings.
pub enum BatchOutputExpression<F, ClaimId> {
    Linear(Vec<(ClaimId, F)>),
    SumOfProducts(Vec<Vec<(ClaimId, F)>>),
}

/// Public data returned by both prover and verifier batch-opening paths.
pub struct BatchOpeningPublic<F, ClaimId, HidingCommitment> {
    pub outputs: Vec<OpenedBatchOutput<F, HidingCommitment>>,
    pub relations: Vec<BatchOutputRelation<F, ClaimId>>,
}

/// Prover-only data for hidden outputs.
pub struct ZkBatchOpeningWitness<F, Blind> {
    pub output_values: Vec<F>,
    pub output_blinds: Vec<Blind>,
}

pub struct BatchOpeningProveResult<Proof, F, ClaimId> {
    pub proof: Proof,
    pub public: BatchOpeningPublic<F, ClaimId, ()>,
}

pub struct BatchOpeningVerifyResult<F, ClaimId> {
    pub public: BatchOpeningPublic<F, ClaimId, ()>,
}

pub struct ZkBatchOpeningProveResult<Proof, F, ClaimId, HidingCommitment, Blind> {
    pub proof: Proof,
    pub public: BatchOpeningPublic<F, ClaimId, HidingCommitment>,
    pub witness: ZkBatchOpeningWitness<F, Blind>,
}

pub struct ZkBatchOpeningVerifyResult<F, ClaimId, HidingCommitment> {
    pub public: BatchOpeningPublic<F, ClaimId, HidingCommitment>,
}
```

The corresponding trait methods should live on the PCS traits rather than in `jolt-core`:

```rust
fn prove_batch_opening<B, ClaimId>(
    source_batch: &B,
    claims: &[ProverBatchOpeningTerm<Self::Field, ClaimId, B::Id>],
    setup: &Self::ProverSetup,
    transcript: &mut impl Transcript<Challenge = Self::Field>,
) -> BatchOpeningProveResult<Self::BatchProof, Self::Field, ClaimId>
where
    Self: Sized,
    B: BatchOpeningSource<Self::Field, Self::OpeningHint>,
    ClaimId: Copy + Eq + Ord + Send + Sync + 'static;

fn verify_batch_opening<ClaimId, SrcId>(
    claims: &[VerifierBatchOpeningTerm<Self::Field, Self, ClaimId, SrcId>],
    proof: &Self::BatchProof,
    setup: &Self::VerifierSetup,
    transcript: &mut impl Transcript<Challenge = Self::Field>,
) -> Result<BatchOpeningVerifyResult<Self::Field, ClaimId>, OpeningsError>
where
    Self: Sized,
    ClaimId: Copy + Eq + Ord + Send + Sync + 'static,
    SrcId: SourceId;
```

The ZK extension mirrors this shape:

```rust
fn prove_batch_opening_zk<B, ClaimId>(
    source_batch: &B,
    claims: &[ProverBatchOpeningTerm<Self::Field, ClaimId, B::Id>],
    setup: &Self::ProverSetup,
    transcript: &mut impl Transcript<Challenge = Self::Field>,
) -> ZkBatchOpeningProveResult<
    Self::BatchProof,
    Self::Field,
    ClaimId,
    Self::HidingCommitment,
    Self::Blind,
>
where
    Self: Sized,
    B: BatchOpeningSource<Self::Field, Self::OpeningHint>,
    ClaimId: Copy + Eq + Ord + Send + Sync + 'static;

fn verify_batch_opening_zk<ClaimId, SrcId>(
    claims: &[VerifierBatchOpeningTerm<Self::Field, Self, ClaimId, SrcId>],
    proof: &Self::BatchProof,
    setup: &Self::VerifierSetup,
    transcript: &mut impl Transcript<Challenge = Self::Field>,
) -> Result<
    ZkBatchOpeningVerifyResult<Self::Field, ClaimId, Self::HidingCommitment>,
    OpeningsError,
>
where
    Self: Sized,
    ClaimId: Copy + Eq + Ord + Send + Sync + 'static,
    SrcId: SourceId;
```

For current Dory Stage 8, the returned public data has one output and one linear relation.
In transparent mode:

```text
outputs[0] = Public(joint_claim)
relations[0] = Linear([(opening_id_i, gamma_i * eval_scale_i)])
```

In ZK mode:

```text
outputs[0] = Hidden(y_com)
relations[0] = Linear([(opening_id_i, gamma_i * eval_scale_i)])
witness.output_values[0] = joint_claim
witness.output_blinds[0] = y_blinding
```

This gives BlindFold exactly the same information it receives today, but from the PCS result rather than from `jolt-core` reimplementing Dory's fusion logic.

### Compatibility with Bolt's Compute Boundary

The source API should line up with the compiler/backend split being developed on the `refactor/crates` branch.
Bolt represents commitment work as compute-level obligations such as `compute.oracle_dense_trace`, `compute.oracle_one_hot_chunk`, `compute.oracle_family_append`, and `compute.pcs_commit_batch`.
Those operations describe oracle data, oracle families, and batch commitment obligations without requiring protocol code to know whether the prover will materialize each oracle, stream trace rows, use a sparse one-hot path, or eventually target a non-CPU backend.

This PR should provide the Rust PCS boundary that Bolt can lower into:

1. Bolt's oracle buffers map naturally to `CommitmentSource` values.
2. Bolt's oracle families and `compute.pcs_commit_batch` map naturally to `BatchCommitmentSource` plus `PCS::commit_batch`.
3. Bolt's opening obligations map naturally to raw batch-opening terms plus `BatchOpeningSource`.
4. Bolt's current provider-style override for `commit_batch` corresponds to a PCS/backend override in this API, and the same split should apply to batch openings.
5. Protocol code should request commitments and openings over sources; backend code should choose the execution strategy.

This means the abstraction should remain strategy-oblivious above the PCS boundary.
It should not bake Dory's row-major streaming schedule into `jolt-openings`, but it should leave enough structure for Dory, Bolt-generated CPU code, and future GPU or lattice backends to preserve their own optimized commitment paths.

The Akita/Hachi packed path suggests one important constraint on this abstraction: a source id names a committed source, not necessarily one logical Jolt polynomial.
For Dory, the natural source ids are the individual `CommittedPolynomial` variants, and `commit_batch` returns one commitment per logical polynomial.
For a packed lattice scheme, the natural source id can instead be a packed witness group whose internal source is the lazy matrix `(cycle, logical_poly) -> hot_index`.
That packed source commits once, and the Jolt/Akita adapter is responsible for translating logical polynomial opening claims into packed-source opening points with the right selector coordinates.

This keeps `jolt-openings` neutral about commitment granularity:

1. Dory can use `BatchCommitmentSource` to preserve its current one-trace-scan / many-Dory-commitments behavior.
2. A packed scheme can use one `CommitmentSource` for the whole packed witness group and return a single commitment for that group.
3. The protocol layer owns logical-polynomial-to-source claim routing because that routing depends on Jolt's witness IDs, packed layout, and opening-point construction.
4. The PCS layer only sees committed sources and opening claims against those sources.

`jolt-core` should replace `CommittedPolynomial::stream_witness_and_commit_rows` with a trace-backed batch commitment source.
The current implementation has this shape as `CycleMajorTraceBatch`: row generation now lives in the source adapter, and the prover passes the same batch source directly to canonical `PCS::commit_batch` / `PCS::commit_batch_zk` without an in-core source-batch bridge.

```rust
struct CycleMajorTraceBatch<'a, I> {
    trace: LazyTraceIterator,
    padded_len: usize,
    preprocessing: &'a JoltSharedPreprocessing,
    one_hot_params: &'a OneHotParams,
    ids: Vec<CommittedPolynomial>,
    row_len: usize,
}

impl BatchCommitmentSource<Fr> for CycleMajorTraceBatch<'_, LazyTraceIterator> {
    type Id = CommittedPolynomial;
    type Source<'a> = CycleMajorTraceSource<'a, LazyTraceIterator> where Self: 'a;

    fn source_ids(&self) -> &[Self::Id] {
        &self.ids
    }

    fn source(&self, id: Self::Id) -> Self::Source<'_> {
        JoltTracePolynomialSource { batch: self, id }
    }

    fn natural_chunk_len(&self, ids: &[Self::Id]) -> Option<usize> {
        (!ids.is_empty()).then_some(self.row_len)
    }

    fn map_rows<R, V>(&self, chunk_len: usize, ids: &[Self::Id], visit: V) -> Vec<Vec<R>>
    where
        R: Send,
        V: for<'row> Fn(Self::Id, SourceRow<'row, Fr>) -> R + Send + Sync,
    {
        assert_eq!(chunk_len, self.row_len);

        let rows = self.trace
            .clone()
            .pad_using(self.padded_len, |_| Cycle::NoOp)
            .iter_chunks(chunk_len)
            .enumerate()
            .par_bridge()
            .map(|(row_index, cycles)| {
                let row = ids
                    .iter()
                    .map(|&id| self.visit_row(id, &cycles, &visit))
                    .collect::<Vec<_>>();
                (row_index, row)
            })
            .collect::<Vec<_>>();

        reorder_by_row_index(rows)
    }
}
```

The borrowed-row helper has this shape:

```rust
impl CycleMajorTraceBatch<'_, LazyTraceIterator> {
    fn visit_row<R, V>(&self, id: CommittedPolynomial, cycles: &[Cycle], visit: &V) -> R
    where
        V: for<'row> Fn(CommittedPolynomial, SourceRow<'row, Fr>) -> R,
    {
        match id {
            CommittedPolynomial::RdInc => {
                let row: Vec<i128> = cycles
                    .iter()
                    .map(|cycle| {
                        let (_, pre_value, post_value) = cycle.rd_write().unwrap_or_default();
                        post_value as i128 - pre_value as i128
                    })
                    .collect();
                visit(id, SourceRow::I128(&row))
            }
            CommittedPolynomial::InstructionRa(idx) => {
                let row: Vec<OneHotIndex> = cycles
                    .iter()
                    .map(|cycle| {
                        let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                        let k = self.one_hot_params.lookup_index_chunk(lookup_index, idx);
                        one_hot_index(k, self.one_hot_params.log_k_chunk as u8)
                    })
                    .collect();
                visit(
                    id,
                    SourceRow::OneHot(OneHotRow {
                        log_domain_size: self.one_hot_params.log_k_chunk as u8,
                        entries: OneHotEntries::OnePerColumn(&row),
                    }),
                )
            }
            CommittedPolynomial::RamInc => { /* same i128 row shape */ }
            CommittedPolynomial::BytecodeRa(idx) => { /* same OnePerColumn row shape */ }
            CommittedPolynomial::RamRa(idx) => { /* same MaybeZero row shape */ }
            CommittedPolynomial::TrustedAdvice | CommittedPolynomial::UntrustedAdvice => {
                panic!("advice polynomials are outside the CycleMajor trace batch")
            }
        }
    }
}
```

`visit_row` contains exactly the row generation logic that used to live in `CommittedPolynomial::stream_witness_and_commit_rows`, but it invokes the row visitor instead of calling an old chunk-level PCS method.
For example, `RdInc` and `RamInc` build the same temporary `Vec<i128>` as today and call `visit(id, SourceRow::I128(&row))`.
`InstructionRa` and `BytecodeRa` build a temporary `Vec<OneHotIndex>` and use `OneHotEntries::OnePerColumn`.
`RamRa` builds a temporary `Vec<Option<OneHotIndex>>` and uses `OneHotEntries::MaybeZero`.
Materialized dense sources can call `visit(row_index, SourceRow::FieldElements(existing_slice))` directly.
For layout-shaped dense rows, a materialized source can instead emit a strided
row such as `SourceRow::StridedI128 { values, column_stride }`.
This is needed for Jolt's AddressMajor Dory layout: dense trace polynomials are
embedded in evenly spaced columns of the main matrix, while advice contexts
remain contiguous in their smaller preprocessing-only matrices.
No `Cow` is needed because the row view only has to live for the duration of the `visit` call.

The prover call-site change is narrow: the old row-generation helper disappears and the CycleMajor branch passes `CycleMajorTraceBatch` to canonical source-batch commit entry points.
The call resolves to `jolt-openings::CommitmentScheme` / `ZkOpeningScheme` directly, with Dory providing the optimized batch-source override behind the canonical trait.

```rust
let row_len = layout.commitment_chunk_len();
let batch = CycleMajorTraceBatch::new(
    self.lazy_trace.clone(),
    preprocessing,
    one_hot_params,
    ids.clone(),
    T,
    row_len,
);

#[cfg(feature = "zk")]
let commitments_and_hints = PCS::commit_batch_zk(&batch, &ids, &setup);
#[cfg(not(feature = "zk"))]
let commitments_and_hints = PCS::commit_batch(&batch, &ids, &setup);
```

After the full old/new PCS trait cutover, the same call shape should remain but resolve to the canonical `jolt-openings` traits:

```rust
let ids = CommittedPolynomial::all_for_config(one_hot_params);
let batch = CycleMajorTraceBatch::new(
    self.lazy_trace.clone(),
    T,
    preprocessing,
    one_hot_params,
    ids.clone(),
    row_len,
);

let commitments_and_hints = PCS::commit_batch(&batch, &ids, &setup);
```

For ZK mode the last line becomes:

```rust
let commitments_and_hints = PCS::commit_batch_zk(&batch, &ids, &setup);
```

The loop body, row order, and row encodings are unchanged; they move from the prover into `CycleMajorTraceBatch::map_rows` plus the canonical Dory source-batch implementation.
That is why this can preserve current streaming behavior exactly while moving toward `commit_batch` / `commit_batch_zk` as the public commitment boundary.

The closure does not imply dynamic dispatch.
`map_rows` is generic over `V`, so Rust monomorphizes the concrete closure at the call site, the same way it monomorphizes `Iterator::map` or Rayon closures.
The higher-ranked bound `for<'row> Fn(... SourceRow<'row, F>) -> R` says only that the closure must accept a row borrowed for any short lifetime; it does not create a trait object or heap allocation.
The temporary row vector is allocated in `visit_row`, borrowed into `visit`, consumed immediately by Dory's row MSM or one-hot addition helper, and then dropped.
This corresponds to the old path, where the same temporary row vector was allocated and passed immediately to the chunk-level Dory commitment helper.

### Ordinary Homomorphic Batched Opening Helper

The default homomorphic batch helper implements the standard group-by-point RLC protocol.
It is still useful for schemes and tests that already have concrete per-claim sources and want an ordinary homomorphic batch proof.
It is not the Stage 8 boundary after this spec's opening-source cutover, because Stage 8 must preserve streaming fusion inside the PCS.

Singleton batches are special.
When `prove_batch` / `verify_batch` receive exactly one claim, they call `open` / `verify` directly and wrap or unwrap the one proof in `PCS::BatchProof`.
They do not absorb `rlc_claims` and do not draw a batch challenge.
This remains the correct behavior for a real singleton opening.
It should not be used as a loophole for Stage 8 to pre-fuse many raw claims into one Dory-shaped source before entering the PCS layer.

Prover:

1. Receive `Vec<ProverClaim<F>>` and matching `Vec<PCS::OpeningHint>`.
2. If there is one claim, call `PCS::open` on that claim and return a one-proof `PCS::BatchProof`.
3. Otherwise append the claim count under `rlc_claims`.
4. Append all claimed evaluations.
5. Group claims by opening point.
6. For each group, draw `rho`.
7. RLC-combine polynomial evaluation tables with powers of `rho`.
8. RLC-combine scalar evaluations with powers of `rho`.
9. RLC-combine opening hints with powers of `rho`.
10. Call `PCS::open` on the combined polynomial and point.
11. Return `PCS::BatchProof`.

Verifier:

1. Receive `Vec<OpeningClaim<F, PCS>>` and `PCS::BatchProof`.
2. If there is one claim, require one proof and call `PCS::verify` on that claim.
3. Otherwise append the same claim count under `rlc_claims`.
4. Append all claimed evaluations in the same order.
5. Group claims by opening point.
6. For each group, draw the same `rho`.
7. RLC-combine commitments with powers of `rho`.
8. RLC-combine scalar evaluations with powers of `rho`.
9. Call `PCS::verify` on the combined commitment and point.

The helper is generic over homomorphic schemes.
It emits one `PCS::Proof` per opening-point group after RLC combination, and the scheme's `PCS::BatchProof` stores those per-group proofs.
Non-homomorphic schemes are not required to implement `combine` or `combine_hints`; they can implement `prove_batch` and `verify_batch` directly while satisfying base `open` and `verify` as singleton special cases.
Source-backed batch opening is a separate higher-level PCS entry point.
It may internally use this helper, a native batch protocol, one proof per claim, or a scheme-specific streaming fusion path.

### Dory Implementation

`DoryScheme` should follow #1467's impl split while preserving current `main` correctness work.

`CommitmentSchemeVerifier for DoryScheme`:

1. `Field = Fr`.
2. `VerifierSetup = DoryVerifierSetup`.
3. `Proof = DoryProof`.
4. `BatchProof = Vec<DoryProof>`.
   For ordinary homomorphic batch opening, each element is the single Dory proof for one opening-point group after the helper RLC-combines that group's claims.
   For Stage 8 source-backed batch opening, the current Dory implementation returns a one-element vector because Dory proves one fused source/output.
5. `verify` verifies one Dory opening proof.
6. `verify_batch` delegates to `homomorphic_verify_batch`.
7. `bind_opening_inputs` preserves Dory's transcript binding semantics.

`PublicVerifierSetup for DoryScheme`:

1. `PublicParams = usize`.
2. `verifier_setup(max_num_vars)` derives the deterministic verifier setup.

`CommitmentScheme for DoryScheme`:

1. `ProverSetup = DoryProverSetup`.
2. `OpeningHint = DoryHint`.
3. `SetupParams = usize`.
4. `setup(max_num_vars)` returns prover and verifier setup.
5. `project_verifier_setup(&prover_setup)` projects prover setup down to verifier setup.
6. `commit` commits through the current Dory row commitment path.
7. `commit_batch` overrides the default with batch-source row streaming.
8. `open` proves one Dory opening.
9. Ordinary `prove_batch` can delegate to `homomorphic_prove_batch`.
10. Source-backed `prove_batch_opening` owns Stage 8 fusion and does not route through the materializing multi-claim helper.

Dory no longer implements a shaped commitment extension.
Source-selected traversal is enough: `CommitmentSource::natural_chunk_len` / `BatchCommitmentSource::natural_chunk_len` provide the row width, and Dory stores the selected chunk length in `DoryHint` so opening uses the same traversal as commitment.

`AdditivelyHomomorphicVerifier for DoryScheme`:

1. `combine` linearly combines Dory commitments.

`AdditivelyHomomorphic for DoryScheme`:

1. `combine_hints` linearly combines Dory row-commitment hints.

`ZkOpeningSchemeVerifier` and `ZkOpeningScheme for DoryScheme`:

1. Preserve current Dory `y_com` behavior.
2. Preserve current `y_blinding` behavior.
3. Preserve BlindFold compatibility.
4. Source-backed ZK batch opening returns `Hidden(y_com)` in `BatchOpeningPublic` and returns `joint_claim` / `y_blinding` in the prover-only witness.

ZK commitments use the same source-selected traversal.
`commit_zk` and `commit_batch_zk` choose the same chunk length as transparent commitment and store it in the same backend-owned hint shape.

The awkwardness being removed is precise:

1. Jolt sometimes chooses a layout outside the backend and passes that layout into a source adapter as a natural chunk length or strided-row configuration.
2. The source API names only `chunk_len`, not Dory's `sigma`.
3. Dory opening has both row-commitment data and the row width used to produce those row commitments.
4. Commitment and opening therefore share an explicit hint contract instead of relying on a balanced split recomputed from `point.len()`.

The cleaned-up Dory flow is:

1. Commitment asks the source or batch source for `natural_chunk_len`.
2. Dory validates that the chosen chunk length is a power of two and usable for the source traversal.
3. Dory derives `sigma` from `chunk_len` and derives `nu` from the row-commitment count in `DoryHint`.
4. Dory commits rows and stores `chunk_len` in `DoryHint`.
5. Opening reads `chunk_len` from `DoryHint`, derives the same private split, folds rows with that chunk length, and produces the single Dory proof.

The Stage 8 source-backed Dory flow is:

1. Receive raw `ProverBatchOpeningTerm`s in Jolt's Stage 8 order.
2. In transparent mode, absorb the scaled evaluations under the same labels and in the same order as current Stage 8.
   In ZK mode, do not absorb secret evaluations.
3. Draw the batch challenge powers inside Dory's source-backed batch-opening method.
4. Aggregate source coefficients by `source_id` using `gamma_i`, not `gamma_i * eval_scale_i`.
5. Combine borrowed Dory row-commitment hints with those unscaled source coefficients, without cloning hint row-commitment vectors.
6. Use the term's `BatchOpeningPoint::proof` coordinate for the Dory proof while retaining `BatchOpeningPoint::public` for protocol binding/output metadata.
7. Build or fold the fused source through `BatchOpeningSource::fold_linear_rows` so the existing streaming RLC path is preserved.
8. Compute the PCS output relation coefficients as `gamma_i * eval_scale_i` in the original claim order.
9. Prove one Dory opening of the fused source and return a one-element `Vec<DoryProof>`.
10. In transparent mode, return `Public(joint_claim)`.
   In ZK mode, return `Hidden(y_com)` plus prover-only `joint_claim` and `y_blinding`.

### `jolt-core` Integration

`jolt-core` should be cut over without changing protocol semantics.

#### Cutover Boundary

`jolt-core` now enters the PCS through the canonical `jolt-openings` trait family and the concrete `jolt_dory::DoryScheme`.
The old in-core `CommitmentScheme`, `SourceBatchCommitmentScheme`, `BatchOpeningScheme`, `ZkOpeningSupport`, `DoryCommitmentScheme`, in-core mock PCS, and in-core HyperKZG/KZG implementations are not part of the merge target.
Backend implementations live in their own crates (`crates/jolt-dory`, `crates/jolt-openings`, and `crates/jolt-hyperkzg`) and `jolt-core` depends only on the canonical associated types it actually needs.

Several boundaries intentionally remain outside `jolt-openings`:

1. `JoltProof`, prover preprocessing, verifier preprocessing, and SDK generated APIs continue to own the zkVM proof format and serialization surface.
   The PCS-owned fields are canonical associated types such as `PCS::Output`, `PCS::OpeningHint`, `PCS::ProverSetup`, `PCS::VerifierSetup`, and `PCS::BatchProof`.
2. Stage 8 owns Jolt-specific claim accumulation, `OpeningId` ordering, and BlindFold consumption of the returned output relation.
   Stage 8 does not own PCS batch fusion, joint commitment construction, or joint proof-output construction.
3. ZK mode consumes the hidden-output data returned by the source-backed batch-opening API, plus backend-owned extension traits for Pedersen generator derivation.
   This keeps BlindFold compatible without putting Dory-specific `y_com` semantics on the base PCS trait.
4. Dory layout globals remain protocol-owned in `jolt-core`.
   The layout affects Jolt's polynomial indexing, opening points, and streaming witness sources, so it should not be hidden inside a backend-neutral PCS API.
   The generic source traits should receive the resulting traversal and point-conversion choices as source configuration, not reach back into Dory globals themselves.

#### Final Cutover Shape

The high-level migration is:

1. `jolt-core` depends on `jolt-openings` and `jolt-dory`.
2. Imports of the internal PCS trait are replaced with `jolt_openings` traits.
3. `PCS::Commitment` associated type usage is replaced with `PCS::Output`.
4. Source-batch commitment call sites enter through canonical `PCS::commit_batch` / `PCS::commit_batch_zk`.
5. Dory tier-1 and tier-2 row aggregation are private to the concrete backend.
6. Proof storage uses `PCS::BatchProof`.
7. Stage 8 proving calls source-backed `PCS::prove_batch_opening` / `PCS::prove_batch_opening_zk`.
8. Stage 8 verification calls source-backed `PCS::verify_batch_opening` / `PCS::verify_batch_opening_zk`.
9. Stage 8's raw claim construction is preserved, while fusion coefficients and output relations are returned by the PCS.
10. The old in-core PCS bridge files are deleted.

Stage 8 is the main adaptation point for openings, not for witness commitment.
The old commitment-time streaming trait should disappear from the public PCS API, because `commit_batch` and `commit_batch_zk` take over that boundary.
Pre-cutover `main` had a separate Stage 8 optimization where `jolt-core`
built a single joint streaming RLC polynomial directly from the trace and
existing hints.
The implementation preserves that algorithmic behavior, but the construction
now lives behind Dory's source-backed batch-opening implementation rather than
as a pre-PCS joint-claim step in `jolt-core`.

The cutover should be:

1. Keep Stage 8's collection of raw accumulator openings and `OpeningId` order in `jolt-core`.
2. Stop multiplying advice and dense-increment claims into a pre-scaled `polynomial_claims` vector.
   Instead, pass raw `eval` plus `eval_scale` on each batch-opening term.
3. Build a `Stage8OpeningSourceBatch` that can expose trace-backed committed sources, advice sources, opening hints, and the current streaming row-folding behavior.
4. Let the PCS sample the fusion challenge, combine sources/hints/commitments, and produce the opened output.
5. Use the returned `BatchOutputRelation` to populate non-ZK claim data and ZK BlindFold data.

The final Stage 8 code calls canonical source-backed batch-opening methods, stores canonical `PCS::BatchProof`, and consumes the returned public/prover-only output data.
Future Akita work can implement a different source-backed batch-opening body without changing `jolt-core`'s Stage 8 raw-claim collection.

The prover call should have this shape:

```rust
let opening_point = self.stage8_opening_point();
let opening_batch = Stage8OpeningSourceBatch::<F, PCS>::new(
    self.one_hot_params.clone(),
    TraceSource::Materialized(Arc::clone(&self.trace)),
    Arc::new(RLCStreamingData {
        bytecode: Arc::clone(&self.preprocessing.shared.bytecode),
        memory_layout: self.preprocessing.shared.memory_layout.clone(),
    }),
    opening_proof_hints,
    self.take_stage8_advice_polys(),
);

let terms: Vec<ProverBatchOpeningTerm<F, OpeningId, CommittedPolynomial>> =
    self.stage8_raw_opening_terms(&opening_point);

#[cfg(feature = "zk")]
let opening_result = PCS::prove_batch_opening_zk(
    terms,
    &opening_batch,
    &self.preprocessing.generators,
    &mut self.transcript,
);

#[cfg(not(feature = "zk"))]
let opening_result = PCS::prove_batch_opening(
    terms,
    &opening_batch,
    &self.preprocessing.generators,
    &mut self.transcript,
);

let proof = opening_result.proof;
```

Each Stage 8 term uses the same public/proof point pair:

```rust
let point = BatchOpeningPoint {
    public: opening_point.r.iter().map(|x| (*x).into()).collect(),
    proof: DoryGlobals::reorder_opening_point_for_layout(&opening_point.r)
        .iter()
        .map(|x| (*x).into())
        .collect(),
};
```

The pair is generic: future schemes whose proof coordinates match protocol coordinates set `proof = public`.
This keeps the point-coordinate distinction explicit without putting Dory's `sigma`/`nu` or row layout vocabulary into `jolt-openings`.

The Stage 8 source batch should concentrate the current streaming RLC behavior:

```rust
struct Stage8OpeningSourceBatch<F, PCS>
where
    PCS: CommitmentScheme<Field = F>,
{
    one_hot_params: OneHotParams,
    trace_source: TraceSource,
    streaming_data: Arc<RLCStreamingData>,
    opening_hints: HashMap<CommittedPolynomial, PCS::OpeningHint>,
    advice_polys: Mutex<Option<HashMap<CommittedPolynomial, MultilinearPolynomial<F>>>>,
    cached_rlc: Mutex<Option<Stage8CachedRlc<F>>>,
}

impl<F, PCS> BatchOpeningSource<F, PCS::OpeningHint> for Stage8OpeningSourceBatch<F, PCS>
where
    F: JoltField + Field,
    for<'challenge> &'challenge F::Challenge: Into<F>,
    PCS: CommitmentScheme<Field = F>,
{
    type Id = CommittedPolynomial;
    type Source<'a> = Stage8OpeningSource<'a, F, PCS> where Self: 'a;

    fn source(&self, id: Self::Id) -> Self::Source<'_> {
        self.source_for_id(id)
    }

    fn opening_hint(&self, id: Self::Id) -> &PCS::OpeningHint {
        &self.opening_hints[&id]
    }

    fn fold_linear_rows(
        &self,
        terms: &[LinearSourceTerm<F, Self::Id>],
        left: &[F],
        chunk_len: usize,
    ) -> Vec<F> {
        let cached = self.cached_stage8_streaming_rlc(terms, chunk_len);
        let sigma = chunk_len.trailing_zeros() as usize;
        jolt_poly::MultilinearPoly::fold_rows(cached, left, sigma)
    }
}
```

`Stage8OpeningSource` can be an enum or adapter over trace-backed committed polynomials and materialized advice polynomials.
It is a `jolt-core` source adapter, not a new `jolt-openings` concept.

The verifier call should mirror the raw terms, with commitments attached:

```rust
let terms: Vec<VerifierBatchOpeningTerm<F, PCS, OpeningId, CommittedPolynomial>> =
    self.stage8_verifier_opening_terms(&opening_point)?;

let opening_public = PCS::verify_batch_opening(
    terms,
    &self.proof.joint_opening_proof,
    &self.preprocessing.generators,
    &mut self.transcript,
)?;

let stage8_relation = opening_public.single_linear_relation()?;
let BatchOutputExpression::Linear(linear_terms) = &stage8_relation.expression;
let (opening_ids, constraint_coeffs): (Vec<_>, Vec<_>) =
    linear_terms.iter().copied().unzip();

Ok(Stage8VerifyData {
    opening_ids,
    constraint_coeffs,
    eval_commitment: None,
})
```

In ZK mode the verifier uses the same shape with `verify_batch_opening_zk`; the returned public data contains `Hidden(y_com)` instead of `Public(joint_claim)`.

### Proof Serialization

Pre-cutover `JoltProof` contained:

```rust
pub joint_opening_proof: PCS::Proof
```

The full cutover stores:

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
After the source-backed batch-opening cutover, that information should come from the PCS batch-opening result.

The cutover must preserve:

1. `Hidden(y_com)` or equivalent `HidingCommitment` output in `BatchOpeningPublic`.
2. `y_blinding` or equivalent `Blind` in the prover-only `ZkBatchOpeningWitness`.
3. `joint_claim` in the prover-only `ZkBatchOpeningWitness`.
4. BlindFold `OpeningProofData` populated from the returned linear relation and witness.
5. ZK opening input binding performed by the PCS exactly once, or represented in the returned public output with a single clear call site.
6. Pedersen generator derivation used by BlindFold.

The returned ZK public relation is the bridge between PCS and BlindFold:

```rust
let relation = opening_result
    .public
    .single_linear_relation()
    .ok_or(ProofVerifyError::InvalidOpeningProof)?;

let output_index = relation.output_index;
let eval_commitment = *opening_result.public.outputs[output_index]
    .value
    .as_hidden()
    .ok_or(ProofVerifyError::InvalidOpeningProof)?;
let BatchOutputExpression::Linear(linear_terms) = &relation.expression;
let (opening_ids, constraint_coeffs): (Vec<_>, Vec<_>) =
    linear_terms.iter().copied().unzip();

self.blindfold_accumulator.set_opening_proof_data(OpeningProofData {
    opening_ids,
    constraint_coeffs,
    joint_claim: opening_result.witness.output_values[output_index],
    y_blinding: opening_result.witness.output_blinds[output_index],
    eval_commitment,
});
```

Verifier-side ZK Stage 8 consumes the same public relation but does not know `joint_claim` or `y_blinding`:

```rust
let opening_public = PCS::verify_batch_opening_zk(
    terms,
    &self.proof.joint_opening_proof,
    &self.preprocessing.generators,
    &mut self.transcript,
)?;

let relation = opening_public.single_linear_relation()?;

let output_index = relation.output_index;
let eval_commitment = *opening_public.outputs[output_index]
    .value
    .as_hidden()
    .ok_or(ProofVerifyError::InvalidOpeningProof)?;
let BatchOutputExpression::Linear(linear_terms) = &relation.expression;
let (opening_ids, constraint_coeffs): (Vec<_>, Vec<_>) =
    linear_terms.iter().copied().unzip();

Ok(Stage8VerifyData {
    opening_ids,
    constraint_coeffs,
    eval_commitment: Some(eval_commitment),
})
```

If the ZK split is not sufficient to express all BlindFold-adjacent needs, add a narrow extension trait rather than widening the base `CommitmentSchemeVerifier`.
The extension trait should describe the capability directly, such as deriving Pedersen generators.
It should not mention BlindFold from inside `jolt-openings`.

### Alternatives Considered

1. **Greenfield `jolt-openings` implementation.**
   Rejected because #1467 already encodes design decisions learned from prior verifier and PCS refactors.
   Reimplementing from memory risks missing important details such as `project_verifier_setup`, `OpeningClaim`'s verifier-only bound, and separation of batch verification from opening-input binding.

2. **Keep current `jolt-openings` reduce API and only wire it into `jolt-core`.**
   Rejected because it keeps batching as an external orchestration step and still forces future non-homomorphic schemes into the wrong abstraction.
   #1467 correctly makes batch proving part of the PCS API.

3. **Put single-claim `open` and `verify` only on homomorphic extension traits.**
   Rejected because single-claim openings are still a natural PCS semantic, and verifier-only code benefits from a base `verify` method.
   Hachi/Akita-style schemes can keep native batching as the hot path while implementing singleton opening as the one-claim special case.

4. **Move Jolt opening accumulators into `jolt-openings`.**
   Rejected because accumulators are Jolt protocol bookkeeping, not PCS abstraction.
   They depend on `CommittedPolynomial`, `VirtualPolynomial`, `SumcheckId`, advice kinds, and Stage 8 ordering.

5. **Bundle Akita into this PR.**
   Rejected because this PR should make the reusable PCS boundary reviewable on its own.
   Akita should build on the new boundary after it lands.

6. **Let Stage 8 pre-fuse raw claims into one singleton batch opening.**
   Rejected because it preserves the current Dory-shaped ownership boundary.
   It makes `jolt-core` sample PCS fusion challenges, build a joint commitment/source, and manually derive BlindFold coefficients.
   That is exactly the coupling the new source-backed batch-opening API is meant to remove.

## Documentation

No Jolt book changes are required for this internal API refactor unless reviewers request them.

The implementation PR should update crate-level docs in:

1. `crates/jolt-openings/src/lib.rs`
2. `crates/jolt-openings/src/schemes.rs`
3. `crates/jolt-openings/src/sources.rs`
4. `crates/jolt-openings/src/claims.rs`
5. `crates/jolt-openings/src/homomorphic.rs`
6. `crates/jolt-dory/src/lib.rs`

The PR should not add user-facing Jolt book documentation unless reviewers request it.
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
9. Add source-backed batch-opening claim/result/source traits to `jolt-openings`.
10. Implement Dory source-backed batch opening while preserving the current Stage 8 streaming RLC and hint-combination behavior.
11. Update Stage 8 prover to produce a batch proof through `PCS::prove_batch_opening` / `PCS::prove_batch_opening_zk`.
12. Update Stage 8 verifier to verify through `PCS::verify_batch_opening` / `PCS::verify_batch_opening_zk`.
13. Reconcile ZK and BlindFold extraction through the returned batch-opening output relation and any narrow generator-derivation extension trait.
14. Remove the old internal PCS trait family after call sites stop depending on it.
15. Delete bridge-only Dory APIs, wrappers, and conversions that become unused after direct `jolt-openings` integration.
16. Run focused crate tests.
17. Run `muldiv` in standard and ZK mode.
18. Run clippy in standard and ZK mode.
19. Run the existing Criterion benchmark path against `main` and the PR branch for Dory/opening-heavy workloads.

Recommended commit structure for the implementation PR:

1. `refactor(openings): port verifier PCS split`
2. `refactor(dory): implement split openings traits`
3. `refactor(core): migrate PCS type family`
4. `refactor(core): remove in-core PCS bridges`
5. `refactor(openings): add source-backed batch openings`
6. `refactor(dory): own stage eight opening fusion`
7. `refactor(core): pass stage eight raw openings to PCS`
8. `perf(dory): preserve opening and commitment hot paths`
9. `test(openings): cover batch-opening output relations`

The spec should stay in the implementation PR so reviewers can check code against the intended boundary as the cutover proceeds.

## References

- PR [#1467](https://github.com/a16z/jolt/pull/1467): `refactor(openings): split PCS traits into verifier/prover halves and fuse batched openings`.
- Branch `layerzero/quang/pcs-prover-verifier-split`.
- `crates/jolt-openings/src/schemes.rs` on PR #1467.
- `crates/jolt-openings/src/homomorphic.rs` on PR #1467.
- `crates/jolt-openings/src/claims.rs` on PR #1467.
- `crates/jolt-dory/src/scheme.rs` on PR #1467.
- `crates/jolt-openings` on current `main`.
- `crates/jolt-dory` on current `main`.
- Removed pre-cutover file: `jolt-core/src/poly/commitment/commitment_scheme.rs`.
- `jolt-core/src/poly/opening_proof.rs`.
- `jolt-core/src/zkvm/prover.rs`.
- `jolt-core/src/zkvm/verifier.rs`.
- `jolt-core/src/zkvm/proof_serialization.rs`.

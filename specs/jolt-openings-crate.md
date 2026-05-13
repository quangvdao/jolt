# Spec: `jolt-openings` PCS API Cutover

| Field       | Value        |
|-------------|--------------|
| Author(s)   | @quangvdao   |
| Created     | 2026-05-12   |
| Status      | proposed     |
| PR          | [#1521](https://github.com/a16z/jolt/pull/1521) |

## Summary

Jolt currently has two polynomial commitment scheme APIs in the workspace.
The production zkVM path still uses the monolithic in-core trait in `jolt-core/src/poly/commitment/commitment_scheme.rs`, while `crates/jolt-openings` and `crates/jolt-dory` define the extracted crate boundary.
The current branch has started wiring that boundary into `jolt-core` for source-batch commitments, but the in-core trait family is still the public PCS surface used by `JoltProof`, prover preprocessing, verifier preprocessing, Stage 8, and BlindFold setup.

This spec proposes a main-target API refactor that ports PR [#1467](https://github.com/a16z/jolt/pull/1467) onto current `main`, with review-driven adjustments recorded here.
The final target makes `jolt-openings` the canonical backend-neutral opening API, splits verifier and prover PCS surfaces, makes fused batched openings the primary API, moves Dory onto the extracted trait family, and cuts `jolt-core` over to `PCS::BatchProof` without introducing Akita or changing the Jolt protocol.
Until `jolt-core`'s proof, setup, hint, transcript, and field surfaces are migrated, this PR may use narrow compatibility traits that delegate to `jolt-dory` internally.

The implementation should be a mechanical transplant of #1467's hard-earned design except where this spec explicitly diverges, not a greenfield rewrite.
Adaptation is only for current `main` drift, especially current Dory hardening, current Dory ZK evaluation commitments, Stage 8's streaming RLC optimization, and BlindFold wiring.

## Intent

### Goal

Make `crates/jolt-openings` the canonical polynomial-opening abstraction used by `jolt-core`, with a verifier-first trait hierarchy and fused `prove_batch` / `verify_batch` API that can support Dory today and future non-Dory schemes such as Akita while keeping native batch proving on the primary production path.

### Source of Truth

PR [#1467](https://github.com/a16z/jolt/pull/1467), branch `quang/pcs-prover-verifier-split`, is the starting point for the abstract PCS API.
This spec is the source of truth where it differs from #1467.
Current `main` is the source of truth for concrete Dory correctness, Dory proof hardening, current proof serialization context, Stage 8 behavior, and BlindFold behavior.

Port or adapt from #1467:

1. `crates/jolt-openings/src/schemes.rs`: verifier/prover split and extension traits.
2. `crates/jolt-openings/src/sources.rs`: backend-neutral commitment source and batch-source traits.
3. `crates/jolt-openings/src/homomorphic.rs`: homomorphic fused batch helpers.
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
5. The base PCS traits expose single-claim openings through `open` and `verify`, fused batched openings through `prove_batch` and `verify_batch`, and batched commitment through `commit_batch`.
6. Single-claim `open` and `verify` are semantic PCS operations, not homomorphic-only operations.
   Native batched schemes may implement them as singleton wrappers around their fused batch path, but verifier code should be able to verify one opening through the base verifier trait.
7. `commit_batch` must preserve current CycleMajor Dory streaming behavior: one padded trace scan, per-row work for all committed polynomials, small-scalar MSM for dense increment rows, and one-hot grouped additions for RA rows.
8. Homomorphic batch proving and verification have byte-identical Fiat-Shamir behavior between prover and verifier.
   They absorb the same claim count, the same evaluations, and draw the same per-point RLC challenges in the same order.
9. `prove_batch` returns `PCS::BatchProof`.
   `jolt-core` remains responsible for the per-group joint evaluations, hiding commitments, and post-opening transcript binding needed by Stage 8 and BlindFold.
   Batch verification does not silently perform post-opening transcript binding.
10. `OpeningClaim` is generic over `PCS: CommitmentSchemeVerifier`, not over a raw commitment type.
   Verifier-only code can name opening claims without importing prover-only PCS types.
11. `jolt-core` keeps protocol-specific opening bookkeeping.
   `OpeningId`, `PolynomialId`, `SumcheckId`, `OpeningPoint`, `ProverOpeningAccumulator`, and `VerifierOpeningAccumulator` do not move into `jolt-openings`.
12. Dory layout, Dory matrix embedding policy, Stage 8 claim ordering, and BlindFold constraints do not move into `jolt-openings`.
13. Dory's current transparent and ZK proofs remain verifier-compatible with current `main`.
14. `JoltProof` stores the opening proof as `PCS::BatchProof`, not `PCS::Proof`.
15. Standard and ZK `muldiv` end-to-end proofs continue to pass.
16. The implementation introduces no Akita dependency.
17. Temporary in-core compatibility traits are allowed only as staging boundaries while `jolt-core` still stores old setup, commitment, hint, proof, field, transcript, and serialization types.
    They must delegate to the canonical `jolt-openings` / `jolt-dory` operations where the type boundary already permits it, and they must be removed in the final old/new trait-family cutover.
18. `cargo tree -d` must not show duplicate resolved versions of `jolt-field`, `jolt-transcript`, `jolt-crypto`, or `jolt-openings`.

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
8. Preserving the old internal `CommitmentScheme` trait as a permanent compatibility layer after the final old/new trait-family cutover.
9. Changing guest execution, bytecode expansion, memory checking, instruction lookups, or sumcheck protocol semantics.
10. Changing Dory's transcript labels or proof verification behavior except where required to preserve current `main` behavior under the new trait API.

## Evaluation

### Acceptance Criteria

- [ ] `crates/jolt-openings/src/schemes.rs` defines `CommitmentSchemeVerifier`, `PublicVerifierSetup`, `CommitmentScheme`, `AdditivelyHomomorphicVerifier`, `AdditivelyHomomorphic`, `ZkOpeningSchemeVerifier`, and `ZkOpeningScheme` with the role split.
- [ ] `StreamingCommitment` is not part of the canonical `jolt-openings` API.
- [ ] `crates/jolt-openings/src/sources.rs` defines `SourceId`, `SourceRow`, `CommitmentSource`, and `BatchCommitmentSource`.
- [ ] `CommitmentSchemeVerifier` contains `Field`, `VerifierSetup`, `Proof`, `BatchProof`, `verify`, `verify_batch`, and `bind_opening_inputs`.
- [ ] `PublicVerifierSetup` contains `PublicParams` and `verifier_setup` for schemes whose verifier setup is derivable without prover setup.
- [ ] `CommitmentScheme` extends `CommitmentSchemeVerifier` and contains `ProverSetup`, `Polynomial`, `OpeningHint`, `SetupParams`, `setup`, `project_verifier_setup`, `commit`, `commit_batch`, `open`, and `prove_batch`.
- [ ] `commit_batch` has a default implementation that commits one source at a time, and Dory overrides it for batch-source row streaming.
- [ ] Homomorphic extension traits contain only the additive-combination operations needed by the default homomorphic batch helper.
- [ ] `crates/jolt-openings/src/homomorphic.rs` contains #1467's `homomorphic_prove_batch`, `homomorphic_verify_batch`, `rlc_combine`, and `rlc_combine_scalars`.
- [ ] `homomorphic_prove_batch` and `homomorphic_verify_batch` group claims by opening point and use the same transcript schedule.
- [ ] `crates/jolt-openings/src/claims.rs` exposes `ProverClaim<F>` and `OpeningClaim<F, PCS: CommitmentSchemeVerifier<Field = F>>`.
- [ ] The old standalone `reduce_prover` / `reduce_verifier` production API is removed or demoted so production callers use `prove_batch` / `verify_batch`.
- [ ] `crates/jolt-openings/src/mock.rs` implements the split traits and has tests covering single-claim, multi-claim, shared-point, distinct-point, and tampered-evaluation cases.
- [ ] `crates/jolt-dory` implements the split trait family while preserving current `main` wrapper types, bounded deserialization, transcript bridge, batch-source streaming support, and ZK behavior.
- [ ] Dory `commit_batch` preserves CycleMajor trace commitment shape: same polynomial order, same row length, same row-commitment ordering, same `DoryHint` row commitments, and same transcript-visible commitments as current `main`.
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
4. Source-batch streamed Dory commitment still matches direct commitment.
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
There are two separate streaming paths to keep straight.
Witness commitment streaming is the current CycleMajor commitment path that builds Dory row commitments while scanning the padded trace.
That path should move behind `commit_batch` and `commit_batch_zk`.
Stage 8 opening streaming is the current Dory RLC path that builds a joint polynomial directly from the trace rather than regenerating witness polynomials.
That path should keep the same algorithmic shape while entering the PCS through `prove_batch` and `verify_batch`.

Performance requirements:

1. Preserve the current Stage 8 streaming RLC optimization unless an equivalent or faster source-based path is implemented.
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
`jolt-dory` owns Dory-specific commitments, proofs, setup types, transcript adaptation, row commitments, batch-source commitment adapters, and Dory proof hardening.
`jolt-core` owns the Jolt protocol's opening IDs, accumulators, Stage 8 claim assembly, Dory layout selection, proof object, source adapters, and BlindFold wiring.

The trait hierarchy is:

```text
CommitmentSchemeVerifier
  - Field
  - VerifierSetup
  - Proof
  - BatchProof
  - verify
  - verify_batch
  - bind_opening_inputs

PublicVerifierSetup: CommitmentSchemeVerifier
  - PublicParams
  - verifier_setup

CommitmentScheme: CommitmentSchemeVerifier
  - ProverSetup
  - Polynomial
  - OpeningHint
  - SetupParams
  - setup
  - project_verifier_setup
  - commit
  - commit_batch
  - open
  - prove_batch

AdditivelyHomomorphicVerifier: CommitmentSchemeVerifier
  - combine

AdditivelyHomomorphic: AdditivelyHomomorphicVerifier + CommitmentScheme
  - combine_hints

ZkOpeningSchemeVerifier: CommitmentSchemeVerifier
  - HidingCommitment
  - verify_zk

ZkOpeningScheme: ZkOpeningSchemeVerifier + CommitmentScheme
  - Blind
  - commit_zk
  - commit_batch_zk
  - open_zk
```

This hierarchy is a role split, not a lifecycle split.
Verifier-only code can bound on `CommitmentSchemeVerifier` without naming prover-only data.
Prover code gets the verifier surface because `CommitmentScheme` extends `CommitmentSchemeVerifier`.
Verifier setup construction is not part of the base verifier trait because not every scheme can derive verifier setup from public parameters alone.
Transparent schemes such as Dory can implement `PublicVerifierSetup`; structured-reference-string schemes such as HyperKZG receive verifier setup generated by setup and do not need an identity-style verifier setup constructor.
Single-opening `open` and `verify` are required PCS basics; fused batching remains the primary production API.
For schemes with no specialized singleton protocol, the singleton methods may wrap the one-claim batch path.
Homomorphic extension traits only add linear combination primitives.

### Source-Oriented Commitment API

The PCS should consume polynomial sources directly.
Streaming is not a standalone commitment-scheme trait, but it is also not quarantined entirely inside source implementations.
The source abstraction describes what data is available and which traversal shapes can expose it without materializing all committed polynomials.
The PCS implementation still owns the commitment algorithm: parallel scheduling, row MSM strategy, one-hot grouping, tier-2 aggregation, hint construction, and transparent-vs-ZK finishing.

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

/// A borrowed row view of a polynomial source.
///
/// This is a traversal hint, not the core polynomial abstraction. Backends that
/// can exploit row structure, such as Dory, consume these rows directly.
/// Backends that do not care about the encoding may interpret the row as field
/// evaluations and use the default source traversal.
pub enum SourceRow<'a, F> {
    /// A dense row of field evaluations.
    FieldElements(&'a [F]),

    /// A dense row of signed integers embedded canonically into `F`.
    ///
    /// This preserves the current Dory small-scalar MSM path for increment
    /// polynomials without first materializing field elements.
    I128(&'a [i128]),

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

    /// Visits row-shaped chunks of the source using `sigma` column variables.
    ///
    /// Implementations should call `visit(row_index, row)` once for each row.
    /// The borrowed row only has to remain valid for the duration of the visit
    /// call, which lets trace-backed sources allocate temporary row buffers and
    /// avoid ownership wrappers such as `Cow`.
    fn for_each_row<V>(&self, sigma: usize, visit: V)
    where
        V: for<'row> FnMut(usize, SourceRow<'row, F>);

    /// Folds rows against the left-side weights used by opening algorithms.
    ///
    /// Dory uses this shape for its vector/matrix product path; other schemes
    /// can implement it by materializing or by using their own source layout.
    fn fold_rows(&self, left: &[F], sigma: usize) -> Vec<F>;
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
        sigma: usize,
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
2. `I128` is a dense row of signed integers embedded canonically into `F`.
3. `OneHot` is a row of one-hot vector entries.
   `log_domain_size` means each column entry lives in `{0, ..., 2^log_domain_size - 1}`.
   `OneHotEntries::OnePerColumn(indices)` means every column contributes one basis vector.
   `OneHotEntries::MaybeZero(indices)` means `Some(k)` contributes `e_k`, and `None` contributes the zero vector.
   Its dense expansion for this row shape is hot-coordinate-major inside the chunk: entry `(hot_index, column)` maps to `hot_index * num_columns + column`.

Only `CommitmentSource` and `BatchCommitmentSource` are core API concepts.
`I128` and `OneHot` are optional row encodings.
They are included to preserve current Jolt/Dory performance without forcing `jolt-core` to call Dory-specific APIs:

1. `I128` maps exactly to current Dory `PCS::process_chunk` for `RdInc` and `RamInc`.
2. `OneHot` maps exactly to current Dory `PCS::process_chunk_onehot` for `InstructionRa`, `BytecodeRa`, and `RamRa`.
3. A backend that does not care about these encodings can immediately materialize or interpret them as field rows.

`OneHotIndex` is intentionally not `usize`.
Current Jolt chunks have `log_k_chunk` equal to `4` or `8`, and the chunk extraction helpers already return `u8`.
The row encoding should require `log_domain_size <= 8`, so `u8` covers every valid hot coordinate.
Current code uses `Option<usize>` for all one-hot rows, but semantically only `RamRa` needs optional columns:

1. `InstructionRa` always has one lookup index per cycle, so it can use `OneHotEntries::OnePerColumn`.
2. `BytecodeRa` always has one bytecode PC chunk per cycle, so it can use `OneHotEntries::OnePerColumn`.
3. `RamRa` can map a cycle to no committed RAM address after `remap_address`, so it uses `OneHotEntries::MaybeZero`.

This is the Dory-dependent part of the source API.
The alternative is to move these encodings behind a Dory-only trait, but then stable Rust cannot make the generic `PCS::commit_batch(&batch, ...)` dispatch to the optimized Dory implementation only for batches that implement that Dory-only trait.
Without specialization or downcasting, Jolt would have to call a Dory-specific method directly, which defeats the cutover goal.

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

1. Call `batch.map_rows(sigma, ids, |id, row| commit_row(row, setup))` over padded trace rows.
2. For each row and each requested source id, compute a Dory row commitment from `SourceRow`.
3. Receive row-major row commitments in the same source-id order as `ids`.
4. Transpose to per-source row-commitment vectors exactly as current `generate_and_commit_witness_polynomials` does.
5. Aggregate each per-source row-commitment vector with Dory tier 2 in transparent or ZK mode.
6. Return the same `(DoryCommitment, DoryHint)` shape used by `open` / `open_zk`.

The Dory row helper is private to `jolt-dory`:

```rust
fn commit_row(row: SourceRow<'_, Fr>, setup: &DoryProverSetup) -> Vec<ArkG1> {
    match row {
        SourceRow::I128(values) => commit_small_scalar_row(values, setup),
        SourceRow::OneHot(row) => commit_onehot_row(row, setup),
        SourceRow::FieldElements(values) => commit_field_row(values, setup),
    }
}
```

The one-hot Dory helper is the current `process_chunk_onehot` with only the row representation changed:

```rust
fn commit_onehot_row(row: OneHotRow<'_>, setup: &DoryProverSetup) -> Vec<ArkG1> {
    let k = 1usize << row.log_domain_size;

    let row_len = DoryGlobals::get_num_columns();
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

This is the same division of labor as current `process_chunk` / `process_chunk_onehot`, but without exposing Dory tier-1 chunks as a public PCS trait.
The row-processing closure is generic, not trait-object based, so the Jolt trace-batch hot path can remain statically dispatched.
Dory decides the parallel schedule inside its `commit_batch` implementation while using the concrete batch source for shared data access.
For non-Dory schemes, the default `commit_batch` is correct and simple.
They can add an optimized override only if their backend benefits from batch-row source traversal.

### Compatibility with Bolt's Compute Boundary

The source API should line up with the compiler/backend split being developed on the `refactor/crates` branch.
Bolt represents commitment work as compute-level obligations such as `compute.oracle_dense_trace`, `compute.oracle_one_hot_chunk`, `compute.oracle_family_append`, and `compute.pcs_commit_batch`.
Those operations describe oracle data, oracle families, and batch commitment obligations without requiring protocol code to know whether the prover will materialize each oracle, stream trace rows, use a sparse one-hot path, or eventually target a non-CPU backend.

This PR should provide the Rust PCS boundary that Bolt can lower into:

1. Bolt's oracle buffers map naturally to `CommitmentSource` values.
2. Bolt's oracle families and `compute.pcs_commit_batch` map naturally to `BatchCommitmentSource` plus `PCS::commit_batch`.
3. Bolt's current provider-style override for `commit_batch` corresponds to a PCS/backend override in this API.
4. Protocol code should request commitments over sources; backend code should choose the execution strategy.

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
The first implementation slice has this shape as `CycleMajorTraceBatch`: row generation now lives in the source adapter, and the existing in-core `StreamingCommitmentScheme` call site consumes those source rows through a small bridge while the larger old/new PCS trait cutover is still in progress.
The final state is for the prover to pass the same batch source directly to `PCS::commit_batch` / `PCS::commit_batch_zk`.

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

    fn map_rows<R, V>(&self, sigma: usize, ids: &[Self::Id], visit: V) -> Vec<Vec<R>>
    where
        R: Send,
        V: for<'row> Fn(Self::Id, SourceRow<'row, Fr>) -> R + Send + Sync,
    {
        let row_len = 1usize << sigma;
        assert_eq!(row_len, self.row_len);

        let rows = self.trace
            .clone()
            .pad_using(self.padded_len, |_| Cycle::NoOp)
            .iter_chunks(row_len)
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

`visit_row` contains exactly the row generation logic that used to live in `CommittedPolynomial::stream_witness_and_commit_rows`, but it invokes the row visitor instead of calling `PCS::process_chunk` or `PCS::process_chunk_onehot`.
For example, `RdInc` and `RamInc` build the same temporary `Vec<i128>` as today and call `visit(id, SourceRow::I128(&row))`.
`InstructionRa` and `BytecodeRa` build a temporary `Vec<OneHotIndex>` and use `OneHotEntries::OnePerColumn`.
`RamRa` builds a temporary `Vec<Option<OneHotIndex>>` and uses `OneHotEntries::MaybeZero`.
Materialized dense sources can call `visit(row_index, SourceRow::FieldElements(existing_slice))` directly.
No `Cow` is needed because the row view only has to live for the duration of the `visit` call.

The prover call-site change is narrow: the old row-generation helper disappears and the CycleMajor branch passes `CycleMajorTraceBatch` to source-batch commit entry points.
While `jolt-core` is still on the in-core PCS trait family, those entry points are temporarily exposed through a `SourceBatchCommitmentScheme` compatibility trait.
That trait is deliberately separate from `StreamingCommitmentScheme`: source-batch support is a PCS capability, not a property every chunk-streaming helper should advertise.
For Dory, the compatibility implementation delegates to `jolt-dory::DoryScheme`'s canonical `jolt_openings::CommitmentScheme::commit_batch` and `ZkOpeningScheme::commit_batch_zk`, then converts the commitment and hint back into the old `jolt-core` wrapper types.
After the full trait-family cutover the same calls should resolve to `jolt-openings::CommitmentScheme` / `ZkOpeningScheme` directly.

```rust
let row_len = DoryGlobals::get_num_columns();
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
This corresponds to the current path, where the same temporary row vector is allocated and passed immediately to `PCS::process_chunk` or `PCS::process_chunk_onehot`.

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
It emits one `PCS::Proof` per opening-point group after RLC combination, and the scheme's `PCS::BatchProof` stores those per-group proofs.
Non-homomorphic schemes are not required to implement `combine` or `combine_hints`; they can implement fused `prove_batch` and `verify_batch` directly while satisfying base `open` and `verify` as singleton special cases.

### Dory Implementation

`DoryScheme` should follow #1467's impl split while preserving current `main` correctness work.

`CommitmentSchemeVerifier for DoryScheme`:

1. `Field = Fr`.
2. `VerifierSetup = DoryVerifierSetup`.
3. `Proof = DoryProof`.
4. `BatchProof = Vec<DoryProof>`.
   Each element is the single Dory proof for one opening-point group after the homomorphic helper RLC-combines that group's claims.
   The current Stage 8 cutover creates one such group, so the proof vector has length one there.
5. `verify` verifies one Dory opening proof.
6. `verify_batch` delegates to `homomorphic_verify_batch`.
7. `bind_opening_inputs` preserves Dory's transcript binding semantics.

`PublicVerifierSetup for DoryScheme`:

1. `PublicParams = usize`.
2. `verifier_setup(max_num_vars)` derives the deterministic verifier setup.

`CommitmentScheme for DoryScheme`:

1. `ProverSetup = DoryProverSetup`.
2. `Polynomial = jolt_poly::Polynomial<Fr>`.
3. `OpeningHint = DoryHint`.
4. `SetupParams = usize`.
5. `setup(max_num_vars)` returns prover and verifier setup.
6. `project_verifier_setup(&prover_setup)` projects prover setup down to verifier setup.
7. `commit` commits through the current Dory row commitment path.
8. `commit_batch` overrides the default with batch-source row streaming.
9. `open` proves one Dory opening.
10. `prove_batch` delegates to `homomorphic_prove_batch`.

`AdditivelyHomomorphicVerifier for DoryScheme`:

1. `combine` linearly combines Dory commitments.

`AdditivelyHomomorphic for DoryScheme`:

1. `combine_hints` linearly combines Dory row-commitment hints.

`ZkOpeningSchemeVerifier` and `ZkOpeningScheme for DoryScheme`:

1. Preserve current Dory `y_com` behavior.
2. Preserve current `y_blinding` behavior.
3. Preserve BlindFold compatibility.

### `jolt-core` Integration

`jolt-core` should be cut over without changing protocol semantics.

#### Current Wiring

The current branch is not a full `jolt-core` cutover to `jolt-openings` / `jolt-dory`.
It has two deliberate bridge points:

1. `SourceBatchCommitmentScheme` lets `jolt-core` commit trace/advice source batches while still returning old in-core commitment and hint associated types.
   The Dory implementation delegates to `jolt_dory::DoryScheme::{commit_batch, commit_batch_zk}` and then converts the result back into the old `DoryCommitmentScheme` wrapper types.
2. `BatchOpeningScheme` lets Stage 8 store and verify a batch-shaped proof while still using the old in-core `JoltProof`, transcript, setup, and BlindFold plumbing.
   The Dory prover side delegates the single combined Stage 8 opening to `jolt_dory::DoryScheme::{open_source_with_shape, open_zk_source_with_shape}` and converts the resulting proof back into a one-element `Vec<ArkDoryProof>`.
   The verifier side still delegates to the old in-core Dory verifier until the verifier setup, commitment, transcript, and proof wrapper boundary is migrated.

These bridges exist because `jolt-core` still has several old surfaces that are intentionally outside `jolt-openings`:

1. `JoltProof`, prover preprocessing, verifier preprocessing, and the SDK serialization path derive or require `ark_serialize::{CanonicalSerialize, CanonicalDeserialize}`.
   `jolt-openings` traits use serde-owned proof/setup types, and `jolt-dory::{DoryProof, DoryCommitment, DoryVerifierSetup}` currently expose serde wrappers around the Dory internals.
2. `jolt-core` is still generic over `ark_bn254::Fr: JoltField` and challenge types such as `F::Challenge`.
   `jolt-openings` / `jolt-dory` are built over the extracted `jolt_field::Fr` newtype and `jolt_transcript` traits.
3. Stage 8 still owns Jolt-specific claim accumulation, `OpeningId` ordering, RLC polynomial construction, layout-sensitive opening point reordering, and BlindFold constraints.
   Those are protocol concerns and should not move into `jolt-openings`.
4. ZK mode needs Dory evaluation commitments, evaluation blindings, Pedersen generator derivation, and BlindFold opening proof data.
   `jolt-dory` exposes most of these capabilities, but `jolt-core` still consumes them through its old `ZkEvalCommitment<C>` and `PedersenGenerators<C>` surface.
5. Dory layout globals are still used by `jolt-core` polynomial adapters, opening point construction, proof serialization, and tests.
   Full cutover should either keep that layout state explicitly in `jolt-core` or replace it with a small protocol-owned layout/config object; it should not hide the layout inside the backend-neutral PCS API.

Because of these differences, replacing `DoryCommitmentScheme` with `jolt_dory::DoryScheme` in `jolt-core` is not a local import change.
It is a cross-cutting migration of associated types, serialization contracts, field/challenge conversions, transcript adapters, and ZK extension hooks.
The current bridge is therefore acceptable as an implementation staging point, but not as the final architecture.

#### Final Cutover

The high-level migration is:

1. Add `jolt-openings` and `jolt-dory` as dependencies.
2. Replace imports of the internal PCS trait with `jolt_openings` traits.
3. Replace `PCS::Commitment` associated type usage with `PCS::Output`.
4. Keep any pre-cutover source-batch bridge separate from the old base `StreamingCommitmentScheme`; only PCS backends that really support the source row shapes should implement it.
5. Remove the remaining pre-cutover source-batch bridge once `jolt-core` proof, hint, setup, and transcript types are on the new trait family.
6. Remove public exposure of `process_chunk`, `process_chunk_onehot`, and `aggregate_chunks` once no in-core caller needs them.
7. Replace `PCS::Proof` proof storage with `PCS::BatchProof`.
8. Replace Stage 8's direct `PCS::prove` call with `PCS::prove_batch`.
9. Replace Stage 8's direct `PCS::verify` call with `PCS::verify_batch`.
10. Preserve Stage 8's claim construction and ZK constraint coefficient logic.

The final cutover additionally requires:

1. Decide whether `JoltProof` and preprocessing continue using ark canonical serialization with adapter impls for `jolt-dory` types, or move the PCS-associated proof/setup fields onto a serde-based boundary.
   This must preserve the SDK proof save/load behavior and bounded Dory proof deserialization.
2. Introduce explicit conversions or a broader field migration between `ark_bn254::Fr` / `JoltField` and `jolt_field::Fr` / `jolt_field::Field`.
   This should be done without dense rematerialization in prover hot paths.
3. Reconcile `crate::transcripts::Transcript` with `jolt_transcript::Transcript` so Dory transcript binding remains byte-identical in transparent and ZK mode.
4. Replace `ZkEvalCommitment<C>` with one or more narrow `jolt-openings` extension traits that expose exactly the needed ZK capabilities: hiding evaluation commitment extraction, evaluation blinding extraction, and Pedersen generator derivation.
5. Decide the ownership boundary for `DoryGlobals` / `DoryLayout`.
   The layout affects Jolt's polynomial indexing and opening points, so it should remain protocol-owned even if the Dory backend consumes a layout/config value.
6. Remove `CommitmentScheme`, `SourceBatchCommitmentScheme`, `BatchOpeningScheme`, `StreamingCommitmentScheme`, and `ZkEvalCommitment` from the in-core PCS surface once all call sites compile against `jolt-openings`.

Stage 8 is the main adaptation point for openings, not for witness commitment.
The old commitment-time streaming trait should disappear from the public PCS API, because `commit_batch` and `commit_batch_zk` take over that boundary.
Current `main` also has a separate Stage 8 optimization: `DoryOpeningState::build_streaming_rlc` builds a single joint RLC polynomial directly from the trace and existing hints.
That optimization is still the right first implementation as long as it is exposed through the new batch-opening API.

The first cutover should be:

1. Keep the current Stage 8 joint RLC construction and hint-combination algorithm.
2. Represent the resulting joint polynomial, unified opening point, and joint claim as the single opening-point group passed to `PCS::prove_batch`.
3. Let Dory's `prove_batch` delegate to the homomorphic helper, which returns a one-element `Vec<DoryProof>` for this one group.
4. On the verifier, build the corresponding one-group `OpeningClaim` and call `PCS::verify_batch`.

In the final cutover this means the public API is new, but the performance-critical Stage 8 work is not rederived during the cutover.
The same source abstraction can later make the joint RLC polynomial less Dory-shaped, but the first PR should not require moving Jolt's opening accumulator, claim ordering, or BlindFold constraint logic into `jolt-openings`.
Future Akita work can implement a different `prove_batch` / `verify_batch` body without changing `jolt-core`'s trait definitions.

While `jolt-core` is still on the in-core PCS trait family, this first cutover is represented by a `BatchOpeningScheme` compatibility trait.
It keeps the existing Stage 8 RLC polynomial construction and exposes the result as a single-group `PCS::BatchedProof`.
For the old Dory wrapper this is a one-element `Vec<ArkDoryProof>`, matching the canonical `jolt-dory` shape.
ZK mode extracts the evaluation commitment from that single-group batch proof before binding the same transcript input and BlindFold opening data as before.
This bridge should disappear once `JoltProof`, preprocessing, transcripts, fields, and ZK extension hooks are on the canonical `jolt-openings` surfaces.

### Proof Serialization

Current `JoltProof` contains:

```rust
pub joint_opening_proof: PCS::Proof
```

The refactor changes it to:

```rust
pub joint_opening_proof: PCS::BatchProof
```

In the in-core compatibility layer this is spelled `PCS::BatchedProof`; the canonical `jolt-openings` trait spells the same concept `PCS::BatchProof`.

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
   Reimplementing from memory risks missing important details such as `project_verifier_setup`, `OpeningClaim`'s verifier-only bound, and separation of batch verification from opening-input binding.

2. **Keep current `jolt-openings` reduce API and only wire it into `jolt-core`.**
   Rejected because it keeps batching as an external orchestration step and still forces future non-homomorphic schemes into the wrong abstraction.
   #1467 correctly makes fused batching the core API.

3. **Put single-claim `open` and `verify` only on homomorphic extension traits.**
   Rejected because single-claim openings are still a natural PCS semantic, and verifier-only code benefits from a base `verify` method.
   Hachi/Akita-style schemes can keep native fused batching as the hot path while implementing singleton opening as the one-claim special case.

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

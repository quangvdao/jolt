# No-Hints Dory Recursion Migration Plan (Jolt)

Last updated: 2026-01-26 (verified against repo at `563a34053`)  
Owner: Jolt recursion + Dory integration

This document is a **single source of truth** for migrating Jolt’s Dory recursion integration to the
intended “no hint map” model:

- **Verifier does not receive any intermediate Dory values (“hints”).**
- Verifier flow becomes: **replay Stage 8 Fiat–Shamir → build Dory AST symbolically → verify recursion SNARK proof obligations → do external pairing check**.
- Prover flow becomes: **generate Dory witnesses (+ AST) once → build recursion constraint system (including wiring + boundary) → prove recursion SNARK**.

The plan is written so another agent can implement end-to-end without needing additional context.

---

## 0. Current repo state (as of this doc)

### 0.0 Local working tree snapshot (for handoff / re-sync)

This plan is intentionally kept **local/uncommitted** in this worktree.

As of 2026-01-26, the active worktree state is:

- branch: `no-hints-dory-migration`
- working tree: **not clean**
  - modified (tracked): 14 files (includes `Cargo.toml`, `Cargo.lock`, and several `jolt-core` recursion/proof files)
  - untracked: this plan + analysis/tooling files + `instance_plan.rs`

Because the working tree drifts quickly, **always** re-check current state before implementing anything:

```bash
git status -sb
git diff --stat
git diff
```

### 0.1 Dory dependency status

✅ **Done in this worktree**: Jolt is already on the no-hints, AST-driven Dory API (no `HintMap`,
no hint-based verification, `TraceContext::{for_symbolic, for_witness_gen_with_ast}`).

### 0.2 Jolt recursion payload and verifier flow (no metadata / no hints)

✅ **Done in this worktree**:

- `RecursionExt` is hint-free and AST-driven (`witness_gen_with_ast`, `build_symbolic_ast`, `replay_opening_proof_transcript`).
- `RecursionPayload` contains **no PCS hints** and **no recursion metadata**.
- Recursion-mode verification is:
  - Stage 8 Fiat–Shamir replay
  - symbolic AST build
  - AST-derived instance-plan construction
  - recursion SNARK verification
  - external pairing boundary check

### 0.3 Recursion prover pipeline (witnesses + AST + pairing boundary)

✅ **Done in this worktree**:

- `RecursionProver::witness_generation` uses `PCS::witness_gen_with_ast(...)` and produces:
  - recursion witnesses
  - the Dory `AstGraph`
  - a derived `PairingBoundary` (for external pairing check binding)

### 0.4 Stage 8 prover snapshot infrastructure already exists (good)

Stage 8 returns `(PCS::Proof, DoryOpeningSnapshot)`:

- `jolt-core/src/zkvm/prover.rs`:
  - `prove_stage8` header + docstring: lines `1787–1793`.

This snapshot is the right vehicle for the “no-hints” recursion prover path.

---

## 1. Target semantics (non-negotiable)

### 1.1 No intermediate “hint map” values in the top-level proof

**Delete** the concept of “PCS verification hint” that is shipped to verifier:

- remove `stage9_pcs_hint` from `RecursionPayload`
- remove `type Hint` from `RecursionExt`
- remove `PCS::verify_with_hint(...)` path(s)

### 1.2 Verifier must not trust prover-supplied AST/constraint list

The recursion verifier must derive the recursion “instance plan” **deterministically** from public data:

- Dory opening proof (Stage 8)
- public verifier setup (Dory verifier setup)
- public opening point + claimed evaluation (Stage 8 joint claim)
- Fiat–Shamir transcript-derived scalars

### 1.3 Recursion SNARK must cover Dory verification obligations

The recursion SNARK must prove (at minimum):

- **All expensive ops** used by Dory verification (GTExp/GTMul/G1ScalarMul/G2ScalarMul/G1Add/G2Add; plus optional MultiMillerLoop if enabled).
- **Wiring/copy constraints** induced by Dory’s AST (prevents “bag of correct ops” attack).
- **Boundary outputs** needed for the outside verifier’s final pairing check (3-way multi-pairing), per `spec.md`.

---

## 2. Implementation phases (do in order)

### Phase A — Update Dory dependency to the no-hints upstream (required)

**Goal**: Move Jolt from Dory `6dd2c56…` to the upstream commit where:

- `HintMap` is deleted
- hint-based verification is deleted
- `TraceContext` has two modes: `for_witness_gen*` and `for_symbolic`
- internal AST evaluator (`TaskExecutor`) is removed

Upstream comparison summary (for reference):  
`LayerZero-Research/dory` `lz-recursion` has commits like:
- “remove hint-based verification, add symbolic mode”
- “remove parallel AST evaluator”

**Concrete target today (branch head):**
- `lz-recursion` at `692d9487fa25b4c5d15f5ca3609d3e7456792950`

Key upstream API (as of `692d9487…`, from `dory/src/recursion/context.rs`):
- `TraceContext::for_symbolic()`
- `TraceContext::for_witness_gen_with_ast()`
- `TraceContext::finalize_with_ast() -> (Option<WitnessCollection>, Option<AstGraph>)`
- `TraceContext::take_ast() -> Option<AstGraph>`

#### A.1 Pin Dory to a specific rev (recommended)

In `Cargo.toml` (workspace deps), change the Dory dependency from `branch = "lz-recursion"` to a pinned `rev = "<sha>"`.

Rationale: recursion proof semantics are consensus-sensitive; relying on a moving branch is dangerous.

#### A.2 Update `Cargo.lock`

This step **will update the lockfile**. Do it in the same PR as the corresponding code changes.

#### A.3 Expected compile fallout in Jolt

You should expect errors in:

- `jolt-core/src/poly/commitment/dory/recursion.rs`:
  - `HintMap`, `HintResult`, `TraceContext::for_hints`, `TaskExecutor`, `OperationEvaluator`, `DoryInputProviderWithCommitment` will be missing.
- Any call sites of `verify_with_hint` / `type Hint`.

**Do not patch around this**—we are removing the hint-based design entirely.

---

### Phase B — Remove the “hint-based verification” surface in Jolt (API + serialization)

This is the core mechanical refactor.

#### B.1 Refactor `RecursionExt` trait to remove hints

File: `jolt-core/src/poly/commitment/commitment_scheme.rs`

Current:
- `RecursionExt` defines `type Hint` and `verify_with_hint`:
  - lines `159–188`.

Change to (proposed):

```rust
pub trait RecursionExt<F: JoltField>: CommitmentScheme<Field = F> {
    type Witness;
    type Ast; // commitment-scheme specific (for Dory: AstGraph<BN254>)
    type CombineHint;

    fn witness_gen_with_ast<ProofTranscript: Transcript>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        point: &[<F as JoltField>::Challenge],
        evaluation: &F,
        commitment: &Self::Commitment,
    ) -> Result<(Self::Witness, Self::Ast), ProofVerifyError>;

    fn build_symbolic_ast<ProofTranscript: Transcript>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        point: &[<F as JoltField>::Challenge],
        evaluation: &F,
        commitment: &Self::Commitment,
    ) -> Result<Self::Ast, ProofVerifyError>;

    fn replay_opening_proof_transcript<ProofTranscript: Transcript>(
        proof: &Self::Proof,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError>;

    // Combine witness API stays, but see Phase E for parallelization split.
    fn generate_combine_witness<C: Borrow<Self::Commitment>>(
        commitments: &[C],
        coeffs: &[F],
    ) -> (GTCombineWitness, Self::CombineHint);
    fn combine_with_hint(hint: &Self::CombineHint) -> Self::Commitment;
    fn combine_hint_to_fq12(hint: &Self::CombineHint) -> ark_bn254::Fq12;
    fn combine_with_hint_fq12(hint: &ark_bn254::Fq12) -> Self::Commitment;
}
```

Notes:
- `build_symbolic_ast` is required so verifiers can reconstruct topology without hints.
- `type Ast` keeps the trait generic (Dory is not the only possible PCS).

Also remove now-unnecessary bounds like:
- `jolt-core/src/zkvm/prover.rs` currently has `where <PCS as RecursionExt<F>>::Hint: Send + Sync + 'static`:
  - see `jolt-core/src/zkvm/prover.rs` lines `193–198`.

#### B.2 Remove `stage9_pcs_hint` and `stage10_recursion_metadata` from proof serialization

File: `jolt-core/src/zkvm/proof_serialization.rs`

Current payload:
- `RecursionPayload` lines `63–77` includes:
  - `stage9_pcs_hint`
  - `stage10_recursion_metadata`

Replace with a **minimal recursion payload**:

```rust
pub struct RecursionPayload<F: JoltField, PCS: RecursionExt<F>, FS: Transcript> {
    pub stage8_combine_hint: Option<Fq12>, // optional; see Phase E
    pub recursion_proof: RecursionProof<Fq, FS, Hyrax<1, GrumpkinProjective>>,
    pub pairing_boundary: PairingBoundary, // new, see Phase D
}
```

Where `PairingBoundary` contains the 3 pairing input pairs + rhs (or enough to compute rhs).

Also update:
- `GuestSerialize`/`GuestDeserialize` for `RecursionPayload`
- Any code that constructs or consumes the old fields.

Current producer-side construction sites to update/remove:
- `jolt-core/src/zkvm/prover.rs` still writes both fields when assembling the top-level proof:
  - grep hits show `stage9_pcs_hint` and `stage10_recursion_metadata` are written around lines `750–764`.

#### B.3 Delete any “PCS verify with hint” codepaths

File: `jolt-core/src/zkvm/verifier.rs`

Remove:
- `verify_stage8_with_pcs_hint` and `_from_snapshot` (these currently call `PCS::verify_with_hint` in older designs).

After Phase B, Stage 8 has exactly two modes:
- **Normal verification**: run Stage 8 PCS verification natively (already exists as `verify_stage8`).
- **Recursion verification**: `verify_stage8_with_recursion` which does FS replay then recursion proof verification.

---

### Phase C — Rework Dory recursion bridge to “no hints” (after Dory update)

File: `jolt-core/src/poly/commitment/dory/recursion.rs`

Current code still implements a hint-producing 4-phase scheme:
- `impl RecursionExt<Fr> for DoryCommitmentScheme`:
  - `type Hint = JoltHintMap` at line `1552`
  - `witness_gen` does “build AST → TaskExecutor → HintMap → expand deferred witnesses” at lines `1567–1636`
  - `verify_with_hint` uses `TraceContext::for_hints` at lines `1639–1684`

After updating Dory, replace this with:

#### C.1 Delete `JoltHintMap` entirely

Delete:
- struct `JoltHintMap` and all ark/guest serialization for it (currently starts at line `23`).

#### C.2 Implement `witness_gen_with_ast` using Dory’s native witness-gen tracing

Pseudo:

```rust
let ctx = Rc::new(TraceContext::<JoltWitness, BN254, JoltWitnessGenerator>::for_witness_gen_with_ast());
verify_recursive::<_, BN254, JoltG1Routines, JoltG2Routines, _, _, _>(
    *commitment, ark_evaluation, &ark_point, proof, setup.clone().into(), &mut dory_transcript, ctx.clone()
)?;
let (witnesses_opt, ast_opt) = Rc::try_unwrap(ctx).ok().unwrap().finalize_with_ast();
let witnesses = witnesses_opt.ok_or(...)?
let ast = ast_opt.ok_or(...)?
Ok((witnesses, ast))
```

#### C.3 Implement `build_symbolic_ast` via Dory symbolic mode

Pseudo:

```rust
let ctx = Rc::new(TraceContext::<JoltWitness, BN254, JoltWitnessGenerator>::for_symbolic());
verify_recursive(..., ctx.clone())?;
let ast = ctx.take_ast().ok_or(...)?
Ok(ast)
```

**Important**: Symbolic mode must assign OpIds deterministically in the same order as witness-gen mode.

#### C.4 Keep `replay_opening_proof_transcript` (already correct)

This already exists and is used by recursion verification to synchronize transcript:
- `replay_opening_proof_transcript` lines `1555–1565`.

---

### Phase D — Make recursion verifier independent of prover metadata (AST → instance plan)

This phase eliminates `stage10_recursion_metadata`.

#### D.1 Introduce a deterministic “instance plan” derivation module

Add file (suggested):
- `jolt-core/src/zkvm/recursion/instance_plan.rs`

Expose:

```rust
pub struct RecursionInstancePlan {
    pub constraint_types: Vec<ConstraintType>,
    pub gt_exp_public_inputs: Vec<GtExpPublicInputs>,
    pub g1_scalar_mul_public_inputs: Vec<G1ScalarMulPublicInputs>,
    pub g2_scalar_mul_public_inputs: Vec<G2ScalarMulPublicInputs>,
    pub dense_num_vars: usize, // derived from constraint_types via PrefixPackingLayout
}

pub fn derive_plan_from_dory_ast(
    ast: &dory::recursion::ast::AstGraph<BN254>,
    proof: &ArkDoryProof,
    setup: &ArkworksVerifierSetup,
) -> Result<RecursionInstancePlan, ProofVerifyError>;
```

#### D.1.1 Use Dory’s `InputSource` and `ScalarValue` (do not invent new naming)

From upstream Dory (`dory/src/recursion/ast/core.rs` at `692d9487…`):
- `InputSource` is one of:
  - `Setup { name: &'static str, index: Option<usize> }`
  - `Proof { name: &'static str }`
  - `ProofRound { round: usize, msg: RoundMsg::{First|Second}, name: &'static str }`
- `ScalarValue<F>` contains:
  - `value: F`
  - `name: Option<&'static str>` (debug only)

Plan derivation must use:
- `scalar.value` (ignore `scalar.name` except for diagnostics)
- `InputSource` to resolve concrete setup/proof elements

#### D.2 Canonical ordering (consensus-critical)

Both prover and verifier MUST use the same ordering for constraint instances.

Define the canonical order as:
- sort by `OpId` (round, op_type, index) for traced ops, and
- stable AST node order for non-traced ops *only if they exist* (ideally every op we prove is traced).

Enforce:
- every AST node corresponding to a proved constraint has `op_id: Some(OpId)`
- if not, fail fast (debug assert + error).

#### D.3 Base-point extraction rule (no-hints requirement)

`ConstraintType::G1ScalarMul { base_point }` and `ConstraintType::G2ScalarMul { base_point }`
must be derivable from **public inputs**.

Implementation rule:
- the scalar-mul input `point: ValueId` must resolve to an `AstOp::Input { source: InputSource::{Setup|Proof|ProofRound} }`
- fetch the actual group element from `(setup, proof)` using that `InputSource`

If you encounter scalar mul where the input point is not an Input:
- **either** (a) treat that point as a witness port and wire it (protocol change),
- **or** (b) prohibit it (preferred if Dory never does this).

Start with (b) + `debug_assert!` on prover + verifier.

#### D.4 GTExp public inputs extraction rule

`GtExpPublicInputs { base: Fq12, scalar_bits: Vec<bool> }` must be derivable from:
- base resolved from `InputSource` as above, and
- scalar read from the AST node’s embedded `ScalarValue` (or derived transcript scalar if needed).

If base is not an Input:
- same policy as scalar mul: fail fast unless we redesign GTExp gadget inputs.

Implementation detail (current Jolt gadgets):
- `GtExpPublicInputs::scalar_bits` is **MSB-first**:
  - see `jolt-core/src/zkvm/recursion/gt/exponentiation.rs` lines `66–71`.
- For scalar mul, `G1ScalarMulPublicInputs` stores the scalar directly:
  - see `jolt-core/src/zkvm/recursion/g1/scalar_multiplication.rs` lines `45–72`.

#### D.5 Integrate into verifier

File: `jolt-core/src/zkvm/verifier.rs`

In `verify_stage8_with_recursion`:
- remove use of `payload.stage10_recursion_metadata`
- build symbolic AST using `PCS::build_symbolic_ast` (new trait method)
- derive `RecursionInstancePlan`
- build `RecursionVerifierInput` from the plan
- run `RecursionVerifier::verify(...)`

Also delete/update any tests that still call hint-based verification:
- `jolt-core/src/zkvm/recursion/constraints/system.rs` has a test hook calling `DoryCommitmentScheme::verify_with_hint` (grep hit around line `3705`).
- `jolt-core/src/poly/commitment/dory/recursion_test.rs` calls `verify_with_hint` (grep hits around lines `178` and `267`).

---

### Phase G — Implement AST-derived wiring/boundary constraints (Stage 2 TODO)

This is the **soundness-critical** part that makes “no hints” meaningful.

Status in code today:
- `jolt-core/src/zkvm/recursion/prover.rs` has `// TODO: Add wiring/boundary constraints sumcheck (AST-driven).` at line `1191`.
- `jolt-core/src/zkvm/recursion/verifier.rs` has `// TODO: wiring/boundary constraints.` at line `510`.
- `jolt-core/src/zkvm/recursion/spec.md` has Stage 2 wiring/boundary section starting at line `1027`.

This repo previously referenced `wiring_sumcheck_design.md` / `wiring_{gt,g1,g2}.md`; if those docs are missing, treat **this section + `spec.md`** as the authoritative design.

#### G.1 Inputs and outputs

**Inputs (public / verifier-derivable):**
- `AstGraph<BN254>` reconstructed by verifier in symbolic mode (Phase D)

**Inputs (witness/prover-only):**
- The per-op witnesses already used to build the recursion constraint system (GTExp/GTMul/G1/G2 scalar mul/add)

**Output:**
- A Stage-2 check that enforces, for every AST edge, that the produced value equals the consumed value (“copy constraints”).

#### G.2 Canonical edge list from AST (no prover discretion)

Build edges from `AstGraph.nodes`:
- For every node `dst` with operation `op`, for each `input: ValueId` in `op.input_ids()`:
  - If `input` comes from an `AstOp::Input { .. }`, it’s a boundary input, not a copy constraint (skip).
  - Otherwise, add an edge `(src_value_id = input, dst_node = dst.out, dst_slot = slot_index)` keyed by the **typed port** it represents.

Canonical ordering:
- Sort edges by `(dst_node ValueId, slot_index)` (or any other deterministic key).
- Do **not** sort by witness map iteration order.

#### G.3 Port model (what exactly is “equal”)

We wire typed values:
- **GT element**: represented as a 4-var MLE over “element vars” (16 evals).
- **G1 element**: represented by affine coordinates (x,y) + indicator encoding already used by existing witness polynomials.
- **G2 element**: represented by affine coordinates (x_c0,x_c1,y_c0,y_c1) + indicator encoding.

For step-indexed/packed operations, the wired port is an **endpoint**:
- **Packed GT exp**: output port is last-step slice of `rho(s,x)` (see `spec.md` lines `1059–1074`).
- **G1/G2 scalar mul**: output port is last-step accumulator (see `spec.md` lines `1065–1079`).

#### G.4 Where to get each port value from existing witnesses

This plan intentionally reuses existing witness polynomials and does not add new “port polynomials” unless necessary.

- **GTMul instance**
  - input ports: `VirtualPolynomial::gt_mul_lhs(i)`, `VirtualPolynomial::gt_mul_rhs(i)`
  - output port: `VirtualPolynomial::gt_mul_result(i)`

- **GTExp instance**
  - output port: last-step slice of `VirtualPolynomial::gt_exp_rho(i)` (packed `rho(s,x)`).
  - (If GTExp base ever needs wiring, treat it as an InputSource-resolved boundary input; fail fast if it is not.)

- **G1ScalarMul instance**
  - output port: last-step slice of `VirtualPolynomial::g1_scalar_mul_xa_next(i)`, `..._ya_next(i)`, plus infinity indicator `..._a_indicator(i)`
  - layout note: scalar-mul traces are 8-var padded into the 11-var ambient domain; enforce pad selector `Eq(0,pad)` as in `G1ScalarMulPublicInputs::evaluate_bit_mle` (`g1/scalar_multiplication.rs` lines `74–94`).

- **G1Add instance**
  - input ports: `VirtualPolynomial::{g1_add_xp, g1_add_yp, g1_add_p_indicator}` etc.
  - output port: `VirtualPolynomial::{g1_add_xr, g1_add_yr, g1_add_r_indicator}` etc.

G2 is analogous using the `g2_add_*` / `g2_scalar_mul_*` virtual polynomials.

#### G.5 How to enforce endpoint ports (selector constraints)

For any step-indexed trace column `A(step, ...)` with step vars `s`:

- Let `EqLast(s) = Eq(last, s)` (multilinear selector for the last step).
- Let `PadSel` be the pad selector if the trace is embedded via zero padding.

To enforce `A(last, ...) == B(...)` (where `B` is the consumer input port lifted to the same ambient vars):

\[
\sum_{s,x} EqLast(s)\cdot Eq(\tau, x)\cdot PadSel \cdot (A(s,x) - B(x)) = 0
\]

Operationally in code, this becomes a **sumcheck instance** whose constraint polynomial is linear in `(A - B)` and multiplied by selectors.

Variable alignment rule (must match existing recursion codepaths):
- **GT wiring domain**: 11 vars split as `(step_vars = 7, element_vars = 4)` in the same order used by packed GT exp (`gt/exponentiation.rs`).
  - Any 4-var GT element port (e.g., GTMul lhs/rhs/result) is interpreted as depending only on the **element_vars** (the last 4 vars in this 11-var domain).
- **G1/G2 wiring domain**: 11 vars split as `(step_vars = 8, pad_vars = 3)` in the same order used by scalar-mul traces (`g1/scalar_multiplication.rs` and `g2/scalar_multiplication.rs`).
  - Add ports are constant in `step_vars` and gated by `Eq(0,pad)` when represented in the ambient domain.

#### G.6 Wiring batching (avoid per-edge blowup)

Do not create one sumcheck instance per edge.

Instead, do **three** wiring sumchecks (one per type: GT, G1, G2), each proving a single batched identity:

\[
0 \stackrel{!}{=} \sum_{e \in \text{edges(type)}} \lambda_e \cdot \langle \mu, \Delta_e(r_x) \rangle
\]

Where:
- `λ_e` are Fiat–Shamir challenges sampled once (deterministic order: edges list order).
- `⟨μ, ·⟩` is coordinate batching:
  - for GT: **16 MLE-evals** (consistent with `fq12_to_multilinear_evals` usage elsewhere), batch with powers of `μ`
  - for G1: 3 coords/flags
  - for G2: 5 coords/flags

Implementation hint:
- Sample `μ_type` per type and use powers `μ^0, μ^1, ...` to aggregate coordinates into a single field equation.

#### G.7 Where to plug this into the pipeline

Add a new prover/verifier module pair, suggested:
- `jolt-core/src/zkvm/recursion/wiring.rs`

Add a Stage-2 sumcheck instance (or instances) at:
- `RecursionProver::prove_stage2_constraints` right at the TODO (line `1191`)
- `RecursionVerifier::verify_stage2_constraints` right at the TODO (line `510`)

The wiring prover/verifier must:
- consume the **public** edge list derived from AST (Phase D)
- use the Stage-2 transcript to sample `λ_e`/`μ_type` in deterministic order

---

### Phase E — Prover boundary refactor (fusion) + parallelism hooks

This aligns with the current `state-handoff.md` (fusion plan).

#### E.1 Move Stage 9 witness generation behind `RecursionProver`

✅ **Done in this worktree**:

- Stage 9 witness generation happens inside `RecursionProver::witness_generation` via
  `PCS::witness_gen_with_ast(...)` (no hints).

After Phase B/C:
- `RecursionProver::witness_generation` should call `PCS::witness_gen_with_ast(...)`
  and receive `(WitnessCollection, AstGraph)` with **no hint**.

Then:
- build constraint system using witness collection, combine witness, and AST-derived ordering.

#### E.2 Introduce `RecursionInput` to cleanly pass Stage 8 data

✅ **Done in this worktree**:

- Added `RecursionInput` (Stage 8 opening proof + snapshot + verifier setup + commitments map)
  and updated the Stage8→recursion call site to pass it.

#### E.3 Parallelize “combine witness gen” with Dory witness collection (optional but recommended)

Today `RecursionProver::witness_generation` does:
1) `PCS::generate_combine_witness` (expensive)
2) `PCS::witness_gen` (expensive, sequential)

Refactor combine API into two steps:

```rust
fn combine_commitments_value_only(commitments, coeffs) -> Commitment; // fast
fn generate_combine_witness(commitments, coeffs) -> GTCombineWitness; // heavy
```

✅ **Done in this worktree**:

- We compute `joint_commitment` value-only (needed for Dory verification) via `combine_commitments`.
- In **host/prover** builds, combine-witness generation is overlapped with `PCS::witness_gen_with_ast`.
- In **verifier-only** builds, the pipeline remains sequential.

---

### Phase F — Add “external pairing boundary outputs” and the verifier-side check

✅ **Done in this worktree** (minimal binding; see F.4 note).

#### F.1 Add boundary outputs to recursion proof payload

Add a new struct (suggested location: `jolt-core/src/zkvm/recursion/mod.rs` or `proof_serialization.rs`):

```rust
pub struct PairingBoundary {
    pub p1_g1: ark_bn254::G1Affine,
    pub p1_g2: ark_bn254::G2Affine,
    pub p2_g1: ark_bn254::G1Affine,
    pub p2_g2: ark_bn254::G2Affine,
    pub p3_g1: ark_bn254::G1Affine,
    pub p3_g2: ark_bn254::G2Affine,
    pub rhs: ark_bn254::Fq12, // or ArkGT
}
```

This must be guest-serializable (use existing guest serde conventions).

#### F.2 Prover: compute pairing boundary from Dory verification state

We must compute the exact three pairs as described in `spec.md` Section `1.4` (see `spec.md` lines `161–171`).

**Preferred extraction strategy (minimal ambiguity):**

1. In the symbolic AST (Phase D), identify the final pairing constraint:
   - Dory AST has `constraints: Vec<AstConstraint>`.
   - The final check is expected to appear as `AstConstraint::AssertEq { lhs, rhs, what }`.

2. Find the unique `AstOp::MultiPairing` node feeding into that constraint:
   - If neither side of the equality is a `MultiPairing`, fail fast (the “3-way check” assumption changed upstream).
   - Assert `g1s.len() == 3` and `g2s.len() == 3`.

3. Define the boundary “slots” as `(g1s[i], g2s[i])` for `i=0..2`.

4. In witness-gen-with-ast mode (Phase C), compute the **concrete values** of those six inputs and serialize them in `PairingBoundary`.

5. Bind these values inside the recursion SNARK (required; see below).

**Do not** introduce new prover-supplied “hints” for internal nodes; these must be derivable or witnessed by recursion SNARK.

#### F.3 Verifier: after recursion proof verifies, do the 3-way multi-pairing check

In recursion mode, `verify_stage8_with_recursion` must:
- compute `multi_pair(p1,p2,p3)` (native)
- check equals `rhs`

This is the external boundary check that replaces in-SNARK pairing.

#### F.4 (Required) Bind boundary outputs inside the recursion SNARK

Without binding, the prover could send arbitrary `(p1,p2,p3,rhs)` values that satisfy the external pairing check but do not match the Dory verification computation being proven.

Binding approach (fits current architecture):
- Treat boundary points as **public outputs** of the recursion SNARK:
  - serialize them in `RecursionPayload` (`PairingBoundary`)
  - add recursion constraints that force these public outputs to equal the correct internal values.

Concrete plan:
- Extend the recursion constraint system with 3 “boundary constraints” that:
  - take the boundary `ValueId`s (from the AST’s `MultiPairing { g1s, g2s }`) and
  - enforce their coordinates match the serialized boundary points.
- Coordinate representation should reuse existing affine coordinate conventions already in the recursion matrix:
  - for G1: `(x: Fq, y: Fq)` plus the same infinity/indicator encoding used in `G1Add`/scalar-mul traces (do not invent a new one).
  - for G2: `(x: Fq2, y: Fq2)` likewise.

Status note (important):

- This worktree implements **minimal binding**: prover includes a `PairingBoundary` in the payload,
  the verifier re-derives the expected boundary from the symbolic AST and checks equality, then runs the external pairing check.
- Full “SNARK-internal binding” of the boundary (as a public output constrained inside the recursion SNARK)
  is intentionally deferred until wiring/copy constraints exist (Phase G), because mapping AST `ValueId`s
  to unique witness ports becomes well-defined only once wiring is enforced.

---

## 3. Acceptance criteria (definition of done)

### DOD-1 No hint map / no hint verification
- No `HintMap`, `JoltHintMap`, `TraceContext::for_hints`, or `verify_with_hint` in the codebase.
- `RecursionExt` has no `type Hint` and no `verify_with_hint`.
- `RecursionPayload` contains no `stage9_pcs_hint` and no `stage10_recursion_metadata`.

### DOD-2 Verifier derives recursion plan from symbolic AST
- `verify_stage8_with_recursion` constructs `RecursionVerifierInput` without reading any prover-supplied recursion metadata.

### DOD-3 Boundary check is implemented
- Recursion payload includes pairing boundary outputs.
- Verifier performs the external 3-way multi-pairing check.

### DOD-4 Tests
At minimum:
- `cargo check -p jolt-core`
- `cargo test -p jolt-core` (or a scoped suite if CI is too heavy)
- any existing recursion e2e test(s) you rely on for correctness

---

## 4. Implementation checklist (tick as you go)

### Phase A (Dory update)
- [x] Pin `dory-pcs` to a rev that includes symbolic mode + no hints
- [x] Update `Cargo.lock` accordingly
- [x] Fix compilation fallout in Dory bridge

### Phase B (Jolt API + serialization)
- [x] Refactor `RecursionExt` (remove `Hint`, add AST methods)
- [x] Remove `stage9_pcs_hint` from `RecursionPayload`
- [x] Remove `stage10_recursion_metadata` from `RecursionPayload`
- [x] Delete any hint-verification paths in verifier

### Phase C (Dory bridge)
- [x] Delete `JoltHintMap` and all hint serialization
- [x] Implement `witness_gen_with_ast` via Dory witness-gen tracing
- [x] Implement `build_symbolic_ast` via Dory symbolic mode

### Phase D (Verifier plan derivation)
- [x] Add `instance_plan.rs`
- [x] Implement `derive_plan_from_dory_ast` with canonical ordering
- [x] Update `verify_stage8_with_recursion` to use symbolic AST + derived plan

### Phase E (Fusion + perf)
- [x] Move Stage 9 witness generation behind `RecursionProver`
- [x] Add `RecursionInput` and refactor Stage 8/9 boundary
- [x] (Optional) overlap combine witness-gen with witness collection on host

### Phase F (Pairing boundary)
- [x] Add `PairingBoundary` to recursion payload
- [x] Prover computes pairing boundary outputs
- [x] Verifier checks 3-way multi-pairing equals rhs

### Phase G (Wiring / boundary sumcheck)
- [ ] Not implemented in this worktree (explicitly out-of-scope for this plan execution)

---

## 5. Key file map (where to change what)

- **Trait & PCS API**
  - `jolt-core/src/poly/commitment/commitment_scheme.rs` (RecursionExt)

- **Dory integration**
  - `jolt-core/src/poly/commitment/dory/recursion.rs` (remove hint map; add symbolic + witness-gen-with-ast)

- **Recursion prover**
  - `jolt-core/src/zkvm/recursion/prover.rs` (witness_generation flow; include AST; fusion)

- **Recursion verifier**
  - `jolt-core/src/zkvm/verifier.rs` (`verify_stage8_with_recursion` must derive plan from AST)

- **Proof serialization**
  - `jolt-core/src/zkvm/proof_serialization.rs` (RecursionPayload structure + guest serde)

- **Spec reference**
  - `jolt-core/src/zkvm/recursion/spec.md` (architecture + pairing boundary + wiring TODOs)

---

## 6. Notes on future work (not required for no-hints, but aligned)

There is an untracked design doc on “fused virtual polynomials”:
- `jolt-core/src/zkvm/recursion/fused_virtual_polynomials_porting_plan.md`

This is **performance work** (proof size & verifier cost) and should be scheduled *after* the no-hints migration is correct.

